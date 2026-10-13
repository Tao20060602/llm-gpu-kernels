#!/usr/bin/env python3
"""Numerical smoke checks and small diagnostic timings for M25 BF16 MMA."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import traceback

import torch

from llm_gpu_kernels.gemm_mma import prepare_bf16_mma


SEED = 425
STAGES = (1, 2, 3, 4)
NUMERIC_CASES = (
    (0, 17, 16),
    (1, 1, 16),
    (17, 65, 48),
    (32, 64, 32),
    (33, 63, 64),
    (193, 768, 2048),
    (256, 256, 256),
)
BENCH_CASES = (
    (17, 768, 2048),
    (193, 768, 2048),
    (1024, 768, 2048),
    (1024, 1024, 1024),
)
TOLERANCE = 0.008
WARMUP_CALLS = 5
ROUNDS = 3
SAMPLES_PER_ARM = 5
CALLS_PER_SAMPLE = 10


def shape_record(shape: tuple[int, int, int]) -> dict[str, int]:
    m, n, k = shape
    return {"M": m, "N": n, "K": k}


def make_inputs(shape: tuple[int, int, int], generator: torch.Generator):
    m, n, k = shape
    a = torch.randn((m, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    return a, w


def numerical_metrics(out: torch.Tensor, reference: torch.Tensor) -> dict:
    actual = out.float()
    finite = bool(torch.isfinite(out).all().item()) and bool(torch.isfinite(reference).all().item())
    if out.numel() == 0:
        return {
            "finite": finite,
            "relative_rms": 0.0,
            "max_abs": 0.0,
            "max_abs_reference": 0.0,
            "max_abs_over_max_abs_reference": 0.0,
            "passed": finite,
        }

    delta = actual - reference
    error_rms = float(torch.mean(delta.square()).sqrt().item())
    reference_rms = float(torch.mean(reference.square()).sqrt().item())
    max_abs = float(delta.abs().max().item())
    max_abs_reference = float(reference.abs().max().item())
    if reference_rms == 0.0:
        relative_rms = 0.0 if error_rms == 0.0 else None
    else:
        relative_rms = error_rms / reference_rms
    if max_abs_reference == 0.0:
        max_abs_ratio = 0.0 if max_abs == 0.0 else None
    else:
        max_abs_ratio = max_abs / max_abs_reference

    passed = (
        finite
        and relative_rms is not None
        and max_abs_ratio is not None
        and relative_rms <= TOLERANCE
        and max_abs_ratio <= TOLERANCE
    )
    return {
        "finite": finite,
        "relative_rms": relative_rms,
        "max_abs": max_abs,
        "max_abs_reference": max_abs_reference,
        "max_abs_over_max_abs_reference": max_abs_ratio,
        "passed": passed,
    }


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_numerical_checks(result: dict, output_path: Path, generator: torch.Generator) -> bool:
    numeric = result["numerics"]
    for shape in NUMERIC_CASES:
        m, n, _ = shape
        a, w = make_inputs(shape, generator)
        reference = torch.mm(a.float(), w.float().T)
        torch.cuda.synchronize()
        for stages in STAGES:
            entry = {**shape_record(shape), "stages": stages}
            try:
                out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                plan = prepare_bf16_mma(a, w, out=out, stages=stages)
                plan.run()
                torch.cuda.synchronize()
                entry.update(numerical_metrics(out, reference))
            except Exception as exc:  # Preserve the first failure before stopping.
                entry.update({
                    "finite": False,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            numeric["results"].append(entry)
            write_result(output_path, result)
            print(
                f"NUM M={m} N={n} K={shape[2]} s={stages} "
                f"rms={entry.get('relative_rms')} "
                f"max/ref={entry.get('max_abs_over_max_abs_reference')} "
                f"{'PASS' if entry['passed'] else 'FAIL'}",
                flush=True,
            )
            if not entry["passed"]:
                numeric["status"] = "failed"
                numeric["failure"] = {k: entry[k] for k in entry if k in {
                    "M", "N", "K", "stages", "finite", "relative_rms", "max_abs",
                    "max_abs_reference", "max_abs_over_max_abs_reference", "error", "passed",
                }}
                result["status"] = "numerical_failure"
                write_result(output_path, result)
                return False
    numeric["status"] = "passed"
    return True


def event_time_api(call, stream: torch.cuda.Stream) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(CALLS_PER_SAMPLE):
            call()
        end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / CALLS_PER_SAMPLE


def build_graph(call, stream: torch.cuda.Stream) -> torch.cuda.CUDAGraph:
    torch.cuda.synchronize()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(WARMUP_CALLS):
            call()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(CALLS_PER_SAMPLE):
            call()
    with torch.cuda.stream(stream):
        for _ in range(WARMUP_CALLS):
            graph.replay()
    stream.synchronize()
    return graph


def event_time_graph(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        graph.replay()
        end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / CALLS_PER_SAMPLE


def validate_bench_outputs(outputs, reference, shape, phase, result, output_path):
    torch.cuda.synchronize()
    checked = {}
    for arm, out in outputs.items():
        metrics = numerical_metrics(out, reference)
        checked[arm] = metrics
        if not metrics["passed"]:
            failure = {**shape_record(shape), "phase": phase, "arm": arm, **metrics}
            result["status"] = "benchmark_numerical_failure"
            result["failure"] = failure
            write_result(output_path, result)
            print(
                f"BENCH CHECK FAIL M={shape[0]} N={shape[1]} K={shape[2]} "
                f"phase={phase} arm={arm} rms={metrics['relative_rms']} "
                f"max/ref={metrics['max_abs_over_max_abs_reference']}",
                flush=True,
            )
            return None
    print(f"CHECK M={shape[0]} N={shape[1]} K={shape[2]} phase={phase} PASS", flush=True)
    return checked


def run_benchmarks(result: dict, output_path: Path, generator: torch.Generator) -> bool:
    benchmark = result["benchmark"]
    stream = torch.cuda.Stream()
    for shape in BENCH_CASES:
        m, n, k = shape
        a, w = make_inputs(shape, generator)
        wt = w.T
        reference = torch.mm(a.float(), wt.float())
        baseline_out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        calls = {"torch_bf16": lambda: torch.mm(a, wt, out=baseline_out)}
        outputs = {"torch_bf16": baseline_out}
        plans = []
        for stages in STAGES:
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            plan = prepare_bf16_mma(a, w, out=out, stages=stages)
            plans.append(plan)
            calls[f"stage_{stages}"] = plan.run
            outputs[f"stage_{stages}"] = out

        # Warm every API arm and synchronize before any timed samples.
        stream.wait_stream(torch.cuda.current_stream())
        for call in calls.values():
            with torch.cuda.stream(stream):
                for _ in range(WARMUP_CALLS):
                    call()
            stream.synchronize()
        pre_api = validate_bench_outputs(
            outputs, reference, shape, "before_timing_api", result, output_path
        )
        if pre_api is None:
            return False

        graph_streams = {arm: torch.cuda.Stream() for arm in calls}
        graphs = {
            arm: build_graph(call, graph_streams[arm])
            for arm, call in calls.items()
        }
        pre_graph = validate_bench_outputs(
            outputs, reference, shape, "before_timing_graph", result, output_path
        )
        if pre_graph is None:
            return False

        case_record = {
            **shape_record(shape),
            "rounds": ROUNDS,
            "samples_per_arm_per_round": SAMPLES_PER_ARM,
            "calls_per_api_sample": CALLS_PER_SAMPLE,
            "calls_per_graph_replay": CALLS_PER_SAMPLE,
            "stage_order_by_round": [
                list(STAGES) if i % 2 == 0 else list(reversed(STAGES))
                for i in range(ROUNDS)
            ],
            "validation": {
                "before_timing_api": pre_api,
                "before_timing_graph": pre_graph,
                "after_timing_api": None,
                "after_timing_graph": None,
            },
        }
        for mode in ("api", "graph"):
            case_record[mode] = {
                "samples_us_per_call": {arm: [] for arm in calls},
                "median_us_per_call": {},
            }
        benchmark["cases"].append(case_record)
        write_result(output_path, result)

        for round_index in range(ROUNDS):
            stage_order = list(STAGES) if round_index % 2 == 0 else list(reversed(STAGES))
            ordered_arms = [f"stage_{s}" for s in stage_order]
            if round_index % 2 == 0:
                ordered_arms.insert(0, "torch_bf16")
            else:
                ordered_arms.append("torch_bf16")
            mode_order = ("api", "graph") if round_index % 2 == 0 else ("graph", "api")
            for mode in mode_order:
                for arm in ordered_arms:
                    arm_samples = []
                    for _ in range(SAMPLES_PER_ARM):
                        if mode == "api":
                            elapsed = event_time_api(calls[arm], stream)
                        else:
                            elapsed = event_time_graph(graphs[arm], graph_streams[arm])
                        arm_samples.append(elapsed)
                    case_record[mode]["samples_us_per_call"][arm].append(arm_samples)
                checked = validate_bench_outputs(
                    outputs, reference, shape, f"after_timing_{mode}_round_{round_index + 1}",
                    result, output_path,
                )
                if checked is None:
                    return False
                case_record["validation"][f"after_timing_{mode}"] = checked
                write_result(output_path, result)

        for mode in ("api", "graph"):
            samples = case_record[mode]["samples_us_per_call"]
            case_record[mode]["median_us_per_call"] = {
                arm: statistics.median(
                    sample for round_samples in rounds for sample in round_samples
                )
                for arm, rounds in samples.items()
            }
        write_result(output_path, result)
        print(
            f"BENCH M={m} N={n} K={k} "
            + " ".join(
                f"{mode}=" + "/".join(
                    f"{case_record[mode]['median_us_per_call'][arm]:.2f}"
                    for arm in ["torch_bf16", *(f"stage_{s}" for s in STAGES)]
                )
                for mode in ("api", "graph")
            ),
            flush=True,
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or repo / "artifacts" / "m25-smoke" / f"m25_mma_check_{timestamp}.json"

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    bf16_reduction_attr = "allow_bf16_reduced_precision_reduction"
    if hasattr(torch.backends.cuda.matmul, bf16_reduction_attr):
        setattr(torch.backends.cuda.matmul, bf16_reduction_attr, False)
        bf16_reduced_precision_reduction = getattr(torch.backends.cuda.matmul, bf16_reduction_attr)
    else:
        bf16_reduced_precision_reduction = None
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    result = {
        "status": "running",
        "seed": SEED,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "tf32": False,
            "allow_bf16_reduced_precision_reduction": bf16_reduced_precision_reduction,
        },
        "numerics": {
            "reference": "torch.mm(A.float(), W.float().T), FP32 oracle, unrounded",
            "comparison": "BF16 output converted to FP32; no elementwise relative error",
            "relative_rms_limit": TOLERANCE,
            "max_abs_over_max_abs_reference_limit": TOLERANCE,
            "results": [],
        },
        "benchmark": {
            "timing_unit": "us/call",
            "baseline": "torch.mm(A_bf16, W_bf16.T, out=BF16_output)",
            "warmup_calls": WARMUP_CALLS,
            "rounds": ROUNDS,
            "samples_per_arm_per_round": SAMPLES_PER_ARM,
            "calls_per_api_sample": CALLS_PER_SAMPLE,
            "calls_per_graph_replay": CALLS_PER_SAMPLE,
            "cases": [],
        },
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if not run_numerical_checks(result, output_path, generator):
            return 2
        result["status"] = "numerics_passed_benchmark_running"
        write_result(output_path, result)
        if not run_benchmarks(result, output_path, generator):
            if result["status"] == "numerics_passed_benchmark_running":
                result["status"] = "benchmark_failed"
            write_result(output_path, result)
            return 3
        result["status"] = "passed"
        write_result(output_path, result)
        print(f"RESULT {output_path}", flush=True)
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["failure"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
        write_result(output_path, result)
        print(f"ABORT {type(exc).__name__}: {exc}\nRESULT {output_path}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
