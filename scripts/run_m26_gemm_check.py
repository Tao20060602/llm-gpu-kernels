#!/usr/bin/env python3
"""M26 tiled BF16 GEMM numerical checks and CUDA Graph diagnostic benchmark."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import traceback

import torch

from llm_gpu_kernels.gemm_mma import prepare_bf16_mma
from llm_gpu_kernels.gemm_mma_tiled import prepare_tiled_bf16_mma


SEED = 426
TILES = ((64, 64), (64, 128), (128, 128))
LAYOUTS = ("padding", "swizzle")
NUMERIC_STAGES = (1, 2, 3, 4)
BENCH_STAGES = (2, 3)
NUMERIC_CASES = (
    (0, 17, 16),
    (1, 1, 16),
    (65, 129, 48),
    (129, 127, 64),
    (256, 256, 256),
)
BENCH_CASES = ((1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096))
TOLERANCE = 0.008
WARMUP_CALLS = 3
GRAPH_WARMUP_REPLAYS = 3
ROUNDS = 3
SAMPLES_PER_ARM_PER_ROUND = 3
CALLS_PER_GRAPH_REPLAY = 5
BF16_REDUCTION_ATTR = "allow_bf16_reduced_precision_reduction"


def shape_record(shape: tuple[int, int, int]) -> dict[str, int]:
    m, n, k = shape
    return {"M": m, "N": n, "K": k}


def make_inputs(shape: tuple[int, int, int], generator: torch.Generator):
    m, n, k = shape
    a = torch.randn((m, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    return a, w


def numerical_metrics(out: torch.Tensor, reference: torch.Tensor) -> dict:
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

    delta = out.float() - reference
    error_rms = float(torch.mean(delta.square()).sqrt().item())
    reference_rms = float(torch.mean(reference.square()).sqrt().item())
    max_abs = float(delta.abs().max().item())
    max_abs_reference = float(reference.abs().max().item())
    relative_rms = 0.0 if error_rms == 0.0 else error_rms / reference_rms if reference_rms else None
    max_abs_ratio = 0.0 if max_abs == 0.0 else max_abs / max_abs_reference if max_abs_reference else None
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


def tiled_configs(stages: tuple[int, ...]):
    for tile, stage, layout in itertools.product(TILES, stages, LAYOUTS):
        yield {"tile": list(tile), "stages": stage, "layout": layout}


def config_name(config: dict) -> str:
    tile = "x".join(str(part) for part in config["tile"])
    return f"tiled_{tile}_s{config['stages']}_{config['layout']}"


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def input_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def source_hashes(repo: Path) -> dict[str, str]:
    sources = (
        Path(__file__).resolve(),
        repo / "src" / "llm_gpu_kernels" / "gemm_mma.py",
        repo / "src" / "llm_gpu_kernels" / "gemm_mma_tiled.py",
        repo / "src" / "llm_gpu_kernels" / "cuda_bf16_mma_tiled.cu",
    )
    found = {}
    for source in sources:
        if source.is_file():
            found[str(source.relative_to(repo))] = hashlib.sha256(source.read_bytes()).hexdigest()
    return found


def check_outputs(outputs: dict[str, torch.Tensor], reference: torch.Tensor, shape,
                  phase: str, result: dict, output_path: Path):
    torch.cuda.synchronize()
    checked = {}
    for arm, out in outputs.items():
        metrics = numerical_metrics(out, reference)
        checked[arm] = metrics
        if not metrics["passed"]:
            result["status"] = "benchmark_numerical_failure"
            result["failure"] = {**shape_record(shape), "phase": phase, "arm": arm, **metrics}
            write_result(output_path, result)
            print(
                f"CHECK FAIL {phase} M={shape[0]} arm={arm} "
                f"rms={metrics['relative_rms']} max/ref={metrics['max_abs_over_max_abs_reference']}",
                flush=True,
            )
            return None
    return checked


def run_numerical_checks(result: dict, output_path: Path, generator: torch.Generator) -> bool:
    numeric = result["numerics"]
    configs = list(tiled_configs(NUMERIC_STAGES))
    for shape in NUMERIC_CASES:
        m, n, _ = shape
        a, w = make_inputs(shape, generator)
        reference = torch.mm(a.float(), w.float().T)
        torch.cuda.synchronize()
        for config in configs:
            entry = {**shape_record(shape), **config}
            entry["arm"] = config_name(config)
            try:
                out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                plan = prepare_tiled_bf16_mma(
                    a, w, out=out, tile=tuple(config["tile"]),
                    stages=config["stages"], layout=config["layout"],
                )
                plan.run()
                torch.cuda.synchronize()
                entry.update(numerical_metrics(out, reference))
            except Exception as exc:
                entry.update({
                    "finite": False,
                    "passed": False,
                    "failure_type": "extension_build_or_prepare_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                })
            numeric["results"].append(entry)
            write_result(output_path, result)
            print(
                f"NUM M={m} N={n} K={shape[2]} tile={config['tile']} "
                f"s={config['stages']} {config['layout']} "
                f"rms={entry.get('relative_rms')} "
                f"max/ref={entry.get('max_abs_over_max_abs_reference')} "
                f"{'PASS' if entry['passed'] else 'FAIL'}",
                flush=True,
            )
            if not entry["passed"]:
                numeric["status"] = "failed"
                result["status"] = (
                    "numerical_execution_failure"
                    if entry.get("failure_type")
                    else "numerical_failure"
                )
                result["failure"] = entry
                write_result(output_path, result)
                return False
    numeric["status"] = "passed"
    return True


@contextmanager
def bf16_reduction_setting(value: bool):
    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, BF16_REDUCTION_ATTR):
        raise RuntimeError(f"torch is missing cuda.matmul.{BF16_REDUCTION_ATTR}")
    previous = getattr(matmul, BF16_REDUCTION_ATTR)
    setattr(matmul, BF16_REDUCTION_ATTR, value)
    try:
        yield previous
    finally:
        setattr(matmul, BF16_REDUCTION_ATTR, previous)
        if getattr(matmul, BF16_REDUCTION_ATTR) != previous:
            raise RuntimeError("failed to restore allow_bf16_reduced_precision_reduction")


def torch_mm_call(a: torch.Tensor, wt: torch.Tensor, out: torch.Tensor, reduction: bool):
    def call():
        with bf16_reduction_setting(reduction):
            torch.mm(a, wt, out=out)
    return call


def build_graph(call, stream: torch.cuda.Stream,
                reduction_setting: bool | None = None) -> torch.cuda.CUDAGraph:
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(WARMUP_CALLS):
            call()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(CALLS_PER_GRAPH_REPLAY):
            call()
    with torch.cuda.stream(stream):
        for _ in range(GRAPH_WARMUP_REPLAYS):
            if reduction_setting is None:
                graph.replay()
            else:
                with bf16_reduction_setting(reduction_setting):
                    graph.replay()
    stream.synchronize()
    return graph


def event_time_graph(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream,
                     reduction_setting: bool | None = None) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        if reduction_setting is None:
            graph.replay()
        else:
            with bf16_reduction_setting(reduction_setting):
                graph.replay()
        end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / CALLS_PER_GRAPH_REPLAY


def make_benchmark_arms(a, w, shape):
    m, n, _ = shape
    wt = w.T
    calls = {}
    outputs = {}
    owners = []
    configs = list(tiled_configs(BENCH_STAGES))
    for config in configs:
        name = config_name(config)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        plan = prepare_tiled_bf16_mma(
            a, w, out=out, tile=tuple(config["tile"]),
            stages=config["stages"], layout=config["layout"],
        )
        calls[name] = plan.run
        outputs[name] = out
        owners.append(plan)

    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    m25 = prepare_bf16_mma(a, w, out=out, stages=2)
    calls["m25_stage_2"] = m25.run
    outputs["m25_stage_2"] = out
    owners.append(m25)

    baseline_settings = {"torch_strict_bf16_reduction": False,
                         "torch_default_bf16_reduction": True}
    for name, reduction in baseline_settings.items():
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        calls[name] = torch_mm_call(a, wt, out, reduction)
        outputs[name] = out
    return calls, outputs, owners, configs


def run_benchmarks(result: dict, output_path: Path, generator: torch.Generator) -> bool:
    benchmark = result["benchmark"]
    for shape in BENCH_CASES:
        m, n, k = shape
        a, w = make_inputs(shape, generator)
        reference = torch.mm(a.float(), w.float().T)
        torch.cuda.synchronize()
        calls, outputs, owners, candidate_configs = make_benchmark_arms(a, w, shape)
        streams = {arm: torch.cuda.Stream() for arm in calls}
        reduction_settings = {
            "torch_strict_bf16_reduction": False,
            "torch_default_bf16_reduction": True,
        }
        graphs = {
            arm: build_graph(call, streams[arm], reduction_settings.get(arm))
            for arm, call in calls.items()
        }

        pre_graph = check_outputs(outputs, reference, shape, "before_graph_timing", result, output_path)
        if pre_graph is None:
            return False

        arm_order = list(calls)
        result_case = {
            **shape_record(shape),
            "input_sha256": {"A": input_sha256(a), "W": input_sha256(w)},
            "arms": {},
            "rounds": ROUNDS,
            "samples_per_arm_per_round": SAMPLES_PER_ARM_PER_ROUND,
            "calls_per_graph_replay": CALLS_PER_GRAPH_REPLAY,
            "graph_warmup_replays": GRAPH_WARMUP_REPLAYS,
            "arm_order_by_round": [arm_order if i % 2 == 0 else list(reversed(arm_order))
                                    for i in range(ROUNDS)],
            "validation": {"before_graph_timing": pre_graph,
                           "after_graph_timing": None,
                           "after_graph_timing_by_round": []},
            "graph": {"samples_us_per_call": {arm: [] for arm in calls},
                      "median_us_per_call": {}},
        }
        # Keep the record explicit for each arm, including the exact Torch flag.
        arm_records = {}
        for config in candidate_configs:
            arm_records[config_name(config)] = {"kind": "tiled_mma", **config}
        arm_records["m25_stage_2"] = {"kind": "m25_bf16_mma", "stages": 2}
        arm_records["torch_strict_bf16_reduction"] = {
            "kind": "torch_mm", "allow_bf16_reduced_precision_reduction": False,
            "flag_set_explicitly_during_call_and_capture": True,
            "flag_restored_after_call": True,
        }
        arm_records["torch_default_bf16_reduction"] = {
            "kind": "torch_mm", "allow_bf16_reduced_precision_reduction": True,
            "flag_set_explicitly_during_call_and_capture": True,
            "flag_restored_after_call": True,
        }
        result_case["arms"] = arm_records
        benchmark["cases"].append(result_case)
        write_result(output_path, result)

        for round_index, round_arms in enumerate(result_case["arm_order_by_round"]):
            for arm in round_arms:
                samples = []
                for _ in range(SAMPLES_PER_ARM_PER_ROUND):
                    samples.append(event_time_graph(
                        graphs[arm], streams[arm], reduction_settings.get(arm)
                    ))
                result_case["graph"]["samples_us_per_call"][arm].append(samples)
            checked = check_outputs(outputs, reference, shape,
                                    f"after_graph_round_{round_index + 1}", result, output_path)
            if checked is None:
                return False
            result_case["validation"]["after_graph_timing_by_round"].append(
                {"round": round_index + 1, "arms": checked}
            )
            result_case["validation"]["after_graph_timing"] = checked
            write_result(output_path, result)

        samples = result_case["graph"]["samples_us_per_call"]
        result_case["graph"]["median_us_per_call"] = {
            arm: statistics.median(sample for round_samples in rounds for sample in round_samples)
            for arm, rounds in samples.items()
        }
        write_result(output_path, result)
        medians = result_case["graph"]["median_us_per_call"]
        fastest = min((name for name in medians if name.startswith("tiled_")), key=medians.get)
        print(
            f"BENCH {m}x{n}x{k} fastest={fastest}:{medians[fastest]:.2f}us "
            f"M25={medians['m25_stage_2']:.2f}us "
            f"torch_strict={medians['torch_strict_bf16_reduction']:.2f}us "
            f"torch_default={medians['torch_default_bf16_reduction']:.2f}us",
            flush=True,
        )
        del graphs, streams, calls, outputs, owners, reference, a, w
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or repo / "artifacts" / "m26-diagnostic" / f"m26_gemm_check_{timestamp}.json"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not hasattr(torch.backends.cuda.matmul, BF16_REDUCTION_ATTR):
        raise RuntimeError(f"torch is missing cuda.matmul.{BF16_REDUCTION_ATTR}")
    setattr(torch.backends.cuda.matmul, BF16_REDUCTION_ATTR, False)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    result = {
        "status": "running",
        "seed": SEED,
        "source_sha256": source_hashes(repo),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_home": os.environ.get("CUDA_HOME"),
            "cxx": os.environ.get("CXX"),
            "max_jobs": os.environ.get("MAX_JOBS"),
            "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "tf32": False,
            "float32_matmul_precision": "highest",
            "allow_bf16_reduced_precision_reduction_outside_baseline_calls": False,
        },
        "numerics": {
            "reference": "torch.mm(A.float(), W.float().T), FP32 oracle, TF32 disabled",
            "comparison": "BF16 output converted to FP32; no elementwise relative error",
            "relative_rms_limit": TOLERANCE,
            "max_abs_over_max_abs_reference_limit": TOLERANCE,
            "candidate_configs_per_shape": 24,
            "results": [],
        },
        "benchmark": {
            "timing_mode": "CUDA Graph only",
            "timing_unit": "us/call",
            "graph_replays_per_sample": 1,
            "calls_per_graph_replay": CALLS_PER_GRAPH_REPLAY,
            "warmup_calls_per_arm_before_capture": WARMUP_CALLS,
            "graph_warmup_replays_per_arm": GRAPH_WARMUP_REPLAYS,
            "rounds": ROUNDS,
            "samples_per_arm_per_round": SAMPLES_PER_ARM_PER_ROUND,
            "arm_order": "forward, reverse, forward",
            "tile_configs_per_shape": 12,
            "candidate_comparison": "12 tiled candidates, M25 stages=2, Torch strict BF16 reduction off, Torch reduced-precision reduction on",
            "cases": [],
        },
    }

    try:
        write_result(output_path, result)
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
