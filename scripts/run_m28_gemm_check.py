#!/usr/bin/env python3
"""M28 warp-specialized BF16 MMA numerical and CUDA Graph diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import traceback

import torch

from llm_gpu_kernels.gemm_mma_scheduled import prepare_scheduled_bf16_mma
from llm_gpu_kernels.gemm_mma_tiled import prepare_tiled_bf16_mma
from llm_gpu_kernels.gemm_mma_warp_specialized import prepare_warp_specialized_bf16_mma
from run_m26_gemm_check import (
    BF16_REDUCTION_ATTR,
    CALLS_PER_GRAPH_REPLAY,
    GRAPH_WARMUP_REPLAYS,
    ROUNDS,
    SAMPLES_PER_ARM_PER_ROUND,
    TOLERANCE,
    WARMUP_CALLS,
    build_graph,
    event_time_graph,
    numerical_metrics,
    shape_record,
    torch_mm_call,
)


SEED = 428
PRODUCER_WARPS = (1, 2)
STAGES = (2, 3)
NUMERIC_CASES = (
    (0, 17, 16),
    (1, 1, 16),
    (65, 129, 48),
    (129, 127, 64),
    (17, 33, 80),
    (33, 65, 112),
    (256, 256, 256),
)
BENCH_CASES = (
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
)


def configs():
    for producer_warps, stages in itertools.product(PRODUCER_WARPS, STAGES):
        yield {"producer_warps": producer_warps, "stages": stages}


def candidate_name(config: dict) -> str:
    return f"m28_ws_p{config['producer_warps']}_s{config['stages']}"


def matched_name(stages: int) -> str:
    return f"m27_group2_s{stages}_swizzle"


def m26_name(stages: int) -> str:
    return f"m26_64x128_s{stages}_swizzle"


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_inputs(shape: tuple[int, int, int], generator: torch.Generator):
    m, n, k = shape
    a = torch.randn((m, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    return a, w


def run_numerical_checks(result: dict, output_path: Path,
                         generator: torch.Generator) -> bool:
    numeric = result["numerics"]
    for shape in NUMERIC_CASES:
        m, n, k = shape
        a, w = make_inputs(shape, generator)
        reference = torch.mm(a.float(), w.float().T)
        torch.cuda.synchronize()

        matched_outputs = {}
        matched_plans = []
        for stages in STAGES:
            arm = matched_name(stages)
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            try:
                plan = prepare_scheduled_bf16_mma(
                    a, w, out=out, group=2, stages=stages, layout="swizzle",
                )
                plan.run()
                torch.cuda.synchronize()
            except Exception as exc:
                result["status"] = "matched_m27_build_or_execution_failure"
                result["failure"] = {
                    **shape_record(shape), "phase": "matched_m27_reference",
                    "arm": arm, "failure_type": "build_or_prepare_or_execution_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                }
                numeric["status"] = "matched_m27_reference_failure"
                write_result(output_path, result)
                print(f"REFERENCE FAIL M={m} N={n} K={k} {arm}: {exc}", flush=True)
                return False
            matched_outputs[stages] = out
            matched_plans.append(plan)

        for config in configs():
            entry = {**shape_record(shape), **config}
            entry["cta_k_tile_iterations"] = (k + 31) // 32
            entry["arm"] = candidate_name(config)
            entry["matched_m27_arm"] = matched_name(config["stages"])
            try:
                out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                plan = prepare_warp_specialized_bf16_mma(
                    a, w, out=out, producer_warps=config["producer_warps"],
                    stages=config["stages"],
                )
            except Exception as exc:
                entry.update({
                    "finite": False, "passed": False,
                    "failure_type": "extension_build_or_prepare_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                })
                numeric["results"].append(entry)
                numeric["status"] = "build_or_prepare_failure"
                result["status"] = "build_or_prepare_failure"
                result["failure"] = entry
                write_result(output_path, result)
                print(f"BUILD/PREPARE FAIL {entry['arm']}: {exc}", flush=True)
                return False

            try:
                plan.run()
                torch.cuda.synchronize()
            except Exception as exc:
                entry.update({
                    "finite": False, "passed": False,
                    "failure_type": "kernel_execution_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                })
                numeric["results"].append(entry)
                numeric["status"] = "execution_failure"
                result["status"] = "numerical_execution_failure"
                result["failure"] = entry
                write_result(output_path, result)
                print(f"EXECUTION FAIL {entry['arm']}: {exc}", flush=True)
                return False

            entry.update(numerical_metrics(out, reference))
            entry["bitwise_equal_to_matched_m27"] = bool(
                torch.equal(out, matched_outputs[config["stages"]])
            )
            if not entry["bitwise_equal_to_matched_m27"]:
                entry["matched_m27_numerical_metrics"] = numerical_metrics(
                    matched_outputs[config["stages"]], reference
                )
            numeric["results"].append(entry)
            write_result(output_path, result)
            print(
                f"NUM M={m} N={n} K={k} producer={config['producer_warps']} "
                f"stages={config['stages']} rms={entry['relative_rms']} "
                f"max/ref={entry['max_abs_over_max_abs_reference']} "
                f"m27_bitwise={entry['bitwise_equal_to_matched_m27']} "
                f"{'PASS' if entry['passed'] else 'FAIL'}",
                flush=True,
            )
            if not entry["passed"]:
                numeric["status"] = "failed"
                result["status"] = "numerical_failure"
                result["failure"] = entry
                write_result(output_path, result)
                return False
            del plan, out

        del matched_plans, matched_outputs, reference, a, w
    numeric["status"] = "passed"
    return True


def make_benchmark_arms(a, w, shape):
    m, n, _ = shape
    wt = w.T
    calls, outputs, owners = {}, {}, []

    for config in configs():
        arm = candidate_name(config)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        plan = prepare_warp_specialized_bf16_mma(
            a, w, out=out, producer_warps=config["producer_warps"],
            stages=config["stages"],
        )
        calls[arm], outputs[arm] = plan.run, out
        owners.append(plan)

    for stages in STAGES:
        arm = matched_name(stages)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        plan = prepare_scheduled_bf16_mma(
            a, w, out=out, group=2, stages=stages, layout="swizzle",
        )
        calls[arm], outputs[arm] = plan.run, out
        owners.append(plan)

    for stages in STAGES:
        arm = m26_name(stages)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        plan = prepare_tiled_bf16_mma(
            a, w, out=out, tile=(64, 128), stages=stages, layout="swizzle",
        )
        calls[arm], outputs[arm] = plan.run, out
        owners.append(plan)

    baseline_settings = {
        "torch_strict_bf16_reduction": False,
        "torch_default_bf16_reduction": True,
    }
    for arm, reduction in baseline_settings.items():
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        calls[arm] = torch_mm_call(a, wt, out, reduction)
        outputs[arm] = out
    return calls, outputs, owners, baseline_settings


def check_outputs(outputs: dict[str, torch.Tensor], reference: torch.Tensor,
                  shape, phase: str, result: dict, output_path: Path):
    torch.cuda.synchronize()
    checked = {}
    for arm, out in outputs.items():
        metrics = numerical_metrics(out, reference)
        checked[arm] = metrics
        if not metrics["passed"]:
            result["status"] = "benchmark_numerical_failure"
            result["failure"] = {
                **shape_record(shape), "phase": phase, "arm": arm, **metrics,
            }
            write_result(output_path, result)
            print(
                f"CHECK FAIL {phase} M={shape[0]} arm={arm} "
                f"rms={metrics['relative_rms']} "
                f"max/ref={metrics['max_abs_over_max_abs_reference']}",
                flush=True,
            )
            return None
    return checked


def run_benchmarks(result: dict, output_path: Path,
                   generator: torch.Generator) -> bool:
    benchmark = result["benchmark"]
    for shape in BENCH_CASES:
        m, n, k = shape
        a, w = make_inputs(shape, generator)
        reference = torch.mm(a.float(), w.float().T)
        torch.cuda.synchronize()
        try:
            calls, outputs, owners, baseline_settings = make_benchmark_arms(a, w, shape)
            streams = {arm: torch.cuda.Stream() for arm in calls}
            graphs = {
                arm: build_graph(call, streams[arm], baseline_settings.get(arm))
                for arm, call in calls.items()
            }
        except Exception as exc:
            result["status"] = "benchmark_build_or_capture_failure"
            result["failure"] = {
                **shape_record(shape), "phase": "benchmark_setup",
                "failure_type": "build_prepare_or_graph_capture_error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
            write_result(output_path, result)
            print(f"BENCH SETUP FAIL M={m}: {exc}", flush=True)
            return False

        pre_graph = check_outputs(
            outputs, reference, shape, "before_graph_timing", result, output_path
        )
        if pre_graph is None:
            return False

        arm_order = list(calls)
        result_case = {
            **shape_record(shape),
            "arms": {},
            "rounds": ROUNDS,
            "samples_per_arm_per_round": SAMPLES_PER_ARM_PER_ROUND,
            "calls_per_graph_replay": CALLS_PER_GRAPH_REPLAY,
            "graph_warmup_replays": GRAPH_WARMUP_REPLAYS,
            "arm_order_by_round": [
                arm_order if i % 2 == 0 else list(reversed(arm_order))
                for i in range(ROUNDS)
            ],
            "validation": {
                "before_graph_timing": pre_graph,
                "after_graph_timing": None,
                "after_graph_timing_by_round": [],
            },
            "graph": {
                "samples_us_per_call": {arm: [] for arm in calls},
                "median_us_per_call": {},
            },
        }
        for config in configs():
            result_case["arms"][candidate_name(config)] = {
                "kind": "m28_warp_specialized_mma",
                "cta_tile": [64, 128, 32],
                "producer_warps": config["producer_warps"],
                "consumer_warps": 4,
                "total_warps": config["producer_warps"] + 4,
                "stages": config["stages"],
                "shared_layout": "swizzle",
                "fragment_group": 2,
            }
        for stages in STAGES:
            result_case["arms"][matched_name(stages)] = {
                "kind": "m27_scheduled_mma", "group": 2,
                "stages": stages, "shared_layout": "swizzle",
                "cta_tile": [64, 128, 32], "total_warps": 4,
            }
            result_case["arms"][m26_name(stages)] = {
                "kind": "m26_tiled_mma", "tile": [64, 128],
                "stages": stages, "layout": "swizzle",
            }
        result_case["arms"]["torch_strict_bf16_reduction"] = {
            "kind": "torch_mm", "allow_bf16_reduced_precision_reduction": False,
            "flag_set_explicitly_during_call_and_capture": True,
            "flag_restored_after_call": True,
        }
        result_case["arms"]["torch_default_bf16_reduction"] = {
            "kind": "torch_mm", "allow_bf16_reduced_precision_reduction": True,
            "flag_set_explicitly_during_call_and_capture": True,
            "flag_restored_after_call": True,
        }
        benchmark["cases"].append(result_case)
        write_result(output_path, result)

        for round_index, round_arms in enumerate(result_case["arm_order_by_round"]):
            for arm in round_arms:
                result_case["graph"]["samples_us_per_call"][arm].append([
                    event_time_graph(
                        graphs[arm], streams[arm], baseline_settings.get(arm)
                    )
                    for _ in range(SAMPLES_PER_ARM_PER_ROUND)
                ])
            checked = check_outputs(
                outputs, reference, shape, f"after_graph_round_{round_index + 1}",
                result, output_path,
            )
            if checked is None:
                return False
            result_case["validation"]["after_graph_timing_by_round"].append(
                {"round": round_index + 1, "arms": checked}
            )
            result_case["validation"]["after_graph_timing"] = checked
            write_result(output_path, result)

        sample_sets = result_case["graph"]["samples_us_per_call"]
        result_case["graph"]["median_us_per_call"] = {
            arm: statistics.median(sample for round_samples in rounds for sample in round_samples)
            for arm, rounds in sample_sets.items()
        }
        write_result(output_path, result)
        print(f"BENCH {m}x{n}x{k}", flush=True)
        for arm, us in sorted(result_case["graph"]["median_us_per_call"].items(),
                              key=lambda item: item[1]):
            print(f"  {arm}: {us:.2f} us/call", flush=True)
        del graphs, streams, calls, outputs, owners, reference, a, w
        torch.cuda.empty_cache()
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or (
        repo / "artifacts" / "m28-diagnostic" / f"m28_gemm_check_{timestamp}.json"
    )

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
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_home": os.environ.get("CUDA_HOME"),
            "cc": os.environ.get("CC"),
            "cxx": os.environ.get("CXX"),
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
            "cta_k_tile": 32,
            "ring_iteration_counts": sorted({(shape[2] + 31) // 32 for shape in NUMERIC_CASES}),
            "candidate_configs_per_shape": len(tuple(configs())),
            "cases": [shape_record(shape) for shape in NUMERIC_CASES],
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
            "arms_per_shape": 10,
            "candidate_comparison": (
                "four M28 producer/stage candidates; M27 group2 swizzle stages2/3; "
                "M26 tile64x128 swizzle stages2/3; Torch strict/default BF16 reduction"
            ),
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
