#!/usr/bin/env python3
"""M30 K-depth BF16 MMA numerical and CUDA Graph diagnostic."""

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

from llm_gpu_kernels.gemm_mma_kdepth import prepare_kdepth_bf16_mma
from llm_gpu_kernels.gemm_mma_register_pipeline import (
    prepare_register_pipeline_bf16_mma,
)
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
    make_inputs,
    numerical_metrics,
    shape_record,
    torch_mm_call,
)


SEED = 430
CTA_M = 64
CTA_N = 128
ELEMENT_BYTES = 2
BK_VALUES = (32, 64)
STAGES = (2, 3)
NUMERIC_CASES = (
    (0, 17, 16),
    (1, 1, 16),
    (65, 129, 48),
    (17, 33, 80),
    (33, 65, 112),
    (33, 65, 144),
    (65, 129, 208),
    (256, 256, 256),
)
BENCH_CASES = (
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
)


def candidate_configs():
    for bk, stages in itertools.product(BK_VALUES, STAGES):
        yield {"bk": bk, "stages": stages}


def candidate_name(config: dict) -> str:
    return f"m30_bk{config['bk']}_s{config['stages']}"


def matched_name(stages: int) -> str:
    return f"m29_g2_mode1_s{stages}"


def shared_bytes(config: dict) -> int:
    return (CTA_M + CTA_N) * config["bk"] * ELEMENT_BYTES * config["stages"]


def k_iterations(k: int, bk: int) -> int:
    return (k + bk - 1) // bk


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def record_failure(result: dict, output_path: Path, status: str,
                   failure: dict, message: str) -> bool:
    result["status"] = status
    result["failure"] = failure
    write_result(output_path, result)
    print(f"{message}\nRESULT {output_path}", flush=True)
    return False


def make_candidate_plan(a, w, out, config):
    return prepare_kdepth_bf16_mma(
        a, w, out=out, bk=config["bk"], stages=config["stages"],
    )


def make_matched_m29_plan(a, w, out, stages: int):
    return prepare_register_pipeline_bf16_mma(
        a, w, out=out, group=2, mode=1, stages=stages,
    )


def run_numerical_checks(result: dict, output_path: Path,
                         generator: torch.Generator) -> bool:
    numeric = result["numerics"]
    bitwise_mismatches = []

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
            entry = {
                **shape_record(shape), "arm": arm, "group": 2,
                "mode": 1, "stages": stages, "layout": "swizzle",
                "cta_k_iterations": k_iterations(k, 32),
            }
            try:
                plan = make_matched_m29_plan(a, w, out, stages)
                plan.run()
                torch.cuda.synchronize()
            except Exception as exc:
                entry.update({
                    "passed": False,
                    "failure_type": "matched_m29_prepare_or_execution_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                })
                numeric["matched_m29_results"].append(entry)
                numeric["status"] = "matched_m29_reference_failure"
                return record_failure(
                    result, output_path, "matched_m29_reference_failure",
                    {**entry, "phase": "matched_m29_reference"},
                    f"M29 REFERENCE FAIL M={m} N={n} K={k} {arm}: {exc}",
                )

            metrics = numerical_metrics(out, reference)
            entry.update(metrics)
            numeric["matched_m29_results"].append(entry)
            write_result(output_path, result)
            print(
                f"M29 REF M={m} N={n} K={k} stages={stages} "
                f"rms={metrics['relative_rms']} "
                f"max/ref={metrics['max_abs_over_max_abs_reference']} "
                f"{'PASS' if metrics['passed'] else 'FAIL'}",
                flush=True,
            )
            if not metrics["passed"]:
                numeric["status"] = "matched_m29_reference_numerical_failure"
                return record_failure(
                    result, output_path,
                    "matched_m29_reference_numerical_failure",
                    {**entry, "phase": "matched_m29_reference"},
                    f"M29 REFERENCE NUMERICAL FAIL M={m} N={n} K={k} {arm}",
                )
            matched_outputs[stages] = out
            matched_plans.append(plan)

        for config in candidate_configs():
            bk = config["bk"]
            stages = config["stages"]
            entry = {
                **shape_record(shape), **config,
                "cta_tile": [CTA_M, CTA_N, bk],
                "group": 2,
                "mode": 1,
                "shared_layout": "swizzle",
                "dynamic_shared_bytes": shared_bytes(config),
                "cta_k_iterations": k_iterations(k, bk),
                "arm": candidate_name(config),
                "matched_m29_arm": matched_name(stages),
            }
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            try:
                plan = make_candidate_plan(a, w, out, config)
            except Exception as exc:
                entry.update({
                    "finite": False,
                    "fp32_passed": False,
                    "bitwise_equal_to_matched_m29": False,
                    "passed": False,
                    "failure_type": "m30_prepare_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                })
                numeric["results"].append(entry)
                numeric["status"] = "m30_prepare_failure"
                return record_failure(
                    result, output_path, "m30_prepare_failure", entry,
                    f"M30 PREPARE FAIL {entry['arm']}: {exc}",
                )

            try:
                plan.run()
                torch.cuda.synchronize()
            except Exception as exc:
                entry.update({
                    "finite": False,
                    "fp32_passed": False,
                    "bitwise_equal_to_matched_m29": False,
                    "passed": False,
                    "failure_type": "m30_kernel_execution_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                })
                numeric["results"].append(entry)
                numeric["status"] = "m30_execution_failure"
                return record_failure(
                    result, output_path, "m30_execution_failure", entry,
                    f"M30 EXECUTION FAIL {entry['arm']}: {exc}",
                )

            metrics = numerical_metrics(out, reference)
            matched = matched_outputs[stages]
            bitwise_equal = bool(torch.equal(out, matched))
            entry.update(metrics)
            entry["fp32_passed"] = metrics["passed"]
            entry["bitwise_equal_to_matched_m29"] = bitwise_equal
            entry["passed"] = metrics["passed"] and bitwise_equal
            numeric["results"].append(entry)
            write_result(output_path, result)
            print(
                f"NUM M={m} N={n} K={k} BK={bk} stages={stages} "
                f"shared={entry['dynamic_shared_bytes']}B "
                f"iterations={entry['cta_k_iterations']} "
                f"rms={metrics['relative_rms']} "
                f"max/ref={metrics['max_abs_over_max_abs_reference']} "
                f"m29_bitwise={bitwise_equal} "
                f"{'PASS' if entry['passed'] else 'MISMATCH' if metrics['passed'] else 'FAIL'}",
                flush=True,
            )
            if not metrics["passed"]:
                numeric["status"] = "m30_numerical_failure"
                return record_failure(
                    result, output_path, "m30_numerical_failure", entry,
                    f"M30 FP32 NUMERICAL FAIL {entry['arm']}",
                )
            if not bitwise_equal:
                mismatch = {
                    **shape_record(shape), "arm": entry["arm"],
                    "matched_m29_arm": entry["matched_m29_arm"],
                }
                bitwise_mismatches.append(mismatch)
                entry["passed"] = False
                entry["failure_type"] = "matched_m29_bitwise_mismatch"
                entry["matched_m29_numerical_metrics"] = numerical_metrics(
                    matched, reference
                )
                numeric["bitwise_mismatches"] = bitwise_mismatches
                numeric["status"] = "m30_m29_bitwise_mismatch"
                write_result(output_path, result)
                return record_failure(
                    result, output_path, "m30_m29_bitwise_mismatch", entry,
                    f"M30/M29 BITWISE MISMATCH {entry['arm']}; "
                    "stopping before remaining numerics and benchmarks",
                )
            del plan, out

        del matched_plans, matched_outputs, reference, a, w

    numeric["bitwise_mismatches"] = bitwise_mismatches
    numeric["status"] = (
        "fp32_passed_with_bitwise_mismatches"
        if bitwise_mismatches else "passed"
    )
    return True


def make_benchmark_arms(a, w, shape):
    m, n, _ = shape
    wt = w.T
    calls, outputs, owners = {}, {}, []

    for config in candidate_configs():
        arm = candidate_name(config)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        plan = make_candidate_plan(a, w, out, config)
        calls[arm], outputs[arm] = plan.run, out
        owners.append(plan)

    for stages in STAGES:
        arm = matched_name(stages)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        plan = make_matched_m29_plan(a, w, out, stages)
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
                f"CHECK FAIL {phase} M={shape[0]} N={shape[1]} K={shape[2]} "
                f"arm={arm} rms={metrics['relative_rms']} "
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
            calls, outputs, owners, baseline_settings = make_benchmark_arms(
                a, w, shape
            )
            streams = {arm: torch.cuda.Stream() for arm in calls}
            graphs = {
                arm: build_graph(call, streams[arm], baseline_settings.get(arm))
                for arm, call in calls.items()
            }
        except Exception as exc:
            failure = {
                **shape_record(shape), "phase": "benchmark_setup",
                "failure_type": "prepare_or_graph_capture_error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
            benchmark["status"] = "prepare_or_graph_capture_failure"
            return record_failure(
                result, output_path,
                "benchmark_prepare_or_graph_capture_failure", failure,
                f"BENCH SETUP FAIL M={m} N={n} K={k}: {exc}",
            )

        pre_timing = check_outputs(
            outputs, reference, shape, "before_graph_timing", result, output_path
        )
        if pre_timing is None:
            benchmark["status"] = "pre_timing_numerical_failure"
            write_result(output_path, result)
            return False

        arm_order = list(calls)
        result_case = {
            **shape_record(shape),
            "rounds": ROUNDS,
            "samples_per_arm_per_round": SAMPLES_PER_ARM_PER_ROUND,
            "calls_per_graph_replay": CALLS_PER_GRAPH_REPLAY,
            "warmup_calls_per_arm_before_capture": WARMUP_CALLS,
            "graph_warmup_replays_per_arm": GRAPH_WARMUP_REPLAYS,
            "arm_order_by_round": [
                arm_order if i % 2 == 0 else list(reversed(arm_order))
                for i in range(ROUNDS)
            ],
            "validation": {
                "before_graph_timing": pre_timing,
                "after_graph_timing": None,
            },
            "arms": {},
            "graph": {
                "samples_us_per_call": {arm: [] for arm in calls},
                "median_us_per_call": {},
            },
        }
        for config in candidate_configs():
            result_case["arms"][candidate_name(config)] = {
                "kind": "m30_kdepth_mma",
                "cta_tile": [CTA_M, CTA_N, config["bk"]],
                "group": 2,
                "mode": 1,
                "mode_name": "b_prefetch",
                "bk": config["bk"],
                "stages": config["stages"],
                "shared_layout": "swizzle",
                "dynamic_shared_bytes": shared_bytes(config),
                "cta_k_iterations": k_iterations(k, config["bk"]),
                "warps": 4,
            }
        for stages in STAGES:
            result_case["arms"][matched_name(stages)] = {
                "kind": "m29_register_pipeline_mma",
                "cta_tile": [CTA_M, CTA_N, 32],
                "group": 2,
                "mode": 1,
                "mode_name": "b_prefetch",
                "stages": stages,
                "shared_layout": "swizzle",
                "cta_k_iterations": k_iterations(k, 32),
                "warps": 4,
            }
        result_case["arms"]["torch_strict_bf16_reduction"] = {
            "kind": "torch_mm",
            "allow_bf16_reduced_precision_reduction": False,
            "flag_set_explicitly_during_call_capture_and_replay": True,
            "flag_restored_after_call": True,
        }
        result_case["arms"]["torch_default_bf16_reduction"] = {
            "kind": "torch_mm",
            "allow_bf16_reduced_precision_reduction": True,
            "flag_set_explicitly_during_call_capture_and_replay": True,
            "flag_restored_after_call": True,
        }
        benchmark["cases"].append(result_case)
        write_result(output_path, result)

        for round_index, round_arms in enumerate(result_case["arm_order_by_round"]):
            for arm in round_arms:
                samples = [
                    event_time_graph(
                        graphs[arm], streams[arm], baseline_settings.get(arm)
                    )
                    for _ in range(SAMPLES_PER_ARM_PER_ROUND)
                ]
                result_case["graph"]["samples_us_per_call"][arm].append(samples)
            print(
                f"TIMED {m}x{n}x{k} round={round_index + 1}/"
                f"{ROUNDS} order={'forward' if round_index % 2 == 0 else 'reverse'}",
                flush=True,
            )

        after_timing = check_outputs(
            outputs, reference, shape, "after_graph_timing", result, output_path
        )
        if after_timing is None:
            benchmark["status"] = "post_timing_numerical_failure"
            write_result(output_path, result)
            return False
        result_case["validation"]["after_graph_timing"] = after_timing
        sample_sets = result_case["graph"]["samples_us_per_call"]
        result_case["graph"]["median_us_per_call"] = {
            arm: statistics.median(
                sample for round_samples in rounds for sample in round_samples
            )
            for arm, rounds in sample_sets.items()
        }
        write_result(output_path, result)
        print(f"BENCH {m}x{n}x{k} ({len(calls)} arms)", flush=True)
        for arm, us in sorted(
            result_case["graph"]["median_us_per_call"].items(),
            key=lambda item: item[1],
        ):
            print(f"  {arm}: {us:.2f} us/call", flush=True)
        del graphs, streams, calls, outputs, owners, reference, a, w
        torch.cuda.empty_cache()
    benchmark["status"] = "passed"
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or (
        repo / "artifacts" / "m30-diagnostic" /
        f"m30_gemm_check_{timestamp}.json"
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
            "comparison": "BF16 output vs FP32 oracle and matched M29 output",
            "relative_rms_limit": TOLERANCE,
            "max_abs_over_max_abs_reference_limit": TOLERANCE,
            "cta_m": CTA_M,
            "cta_n": CTA_N,
            "groups": 2,
            "mode": 1,
            "mode_name": "b_prefetch",
            "bk_values": list(BK_VALUES),
            "stages": list(STAGES),
            "candidate_configs_per_shape": len(tuple(candidate_configs())),
            "cases": [shape_record(shape) for shape in NUMERIC_CASES],
            "matched_m29_results": [],
            "results": [],
            "bitwise_mismatches": [],
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
            "arms_per_shape": 8,
            "candidate_comparison": (
                "4 M30 BK32/64 stage2/3 configs; M29 group2/mode1 stage2/3; "
                "Torch strict/default BF16 reduction"
            ),
            "cases": [],
        },
    }

    try:
        write_result(output_path, result)
        if not run_numerical_checks(result, output_path, generator):
            return 2
        result["status"] = (
            "numerics_passed_with_bitwise_mismatches_benchmark_running"
            if result["numerics"]["bitwise_mismatches"]
            else "numerics_passed_benchmark_running"
        )
        write_result(output_path, result)
        if not run_benchmarks(result, output_path, generator):
            if result["status"].endswith("benchmark_running"):
                result["status"] = "benchmark_failed"
            write_result(output_path, result)
            return 3
        result["status"] = (
            "passed_with_bitwise_mismatches"
            if result["numerics"]["bitwise_mismatches"] else "passed"
        )
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
