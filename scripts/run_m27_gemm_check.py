#!/usr/bin/env python3
"""M27 scheduled BF16 MMA numerical check and CUDA Graph benchmark."""

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

from llm_gpu_kernels.gemm_mma import prepare_bf16_mma
from llm_gpu_kernels.gemm_mma_scheduled import prepare_scheduled_bf16_mma
from llm_gpu_kernels.gemm_mma_tiled import prepare_tiled_bf16_mma
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


SEED = 427
GROUPS = (1, 2, 4, 8)
STAGES = (2, 3)
LAYOUTS = ("padding", "swizzle")
NUMERIC_CASES = (
    (0, 17, 16),
    (1, 1, 16),
    (65, 129, 48),
    (129, 127, 64),
    (256, 256, 256),
)
BENCH_CASES = (
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
)


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def config_name(config: dict) -> str:
    return f"m27_group{config['group']}_s{config['stages']}_{config['layout']}"


def m26_name(stages: int, layout: str) -> str:
    return f"m26_64x128_s{stages}_{layout}"


def candidate_configs():
    for group, stages, layout in itertools.product(GROUPS, STAGES, LAYOUTS):
        yield {"group": group, "stages": stages, "layout": layout}


def matched_m26_configs():
    for stages, layout in itertools.product(STAGES, LAYOUTS):
        yield {"stages": stages, "layout": layout}


def make_inputs(shape: tuple[int, int, int], generator: torch.Generator):
    m, n, k = shape
    a = torch.randn((m, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    return a, w


def run_numerical_checks(result: dict, output_path: Path,
                         generator: torch.Generator) -> bool:
    numeric = result["numerics"]
    configs = list(candidate_configs())
    for shape in NUMERIC_CASES:
        m, n, k = shape
        a, w = make_inputs(shape, generator)
        reference = torch.mm(a.float(), w.float().T)
        torch.cuda.synchronize()

        matched_outputs = {}
        matched_plans = []
        for config in matched_m26_configs():
            arm = m26_name(config["stages"], config["layout"])
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            try:
                plan = prepare_tiled_bf16_mma(
                    a, w, out=out, tile=(64, 128),
                    stages=config["stages"], layout=config["layout"],
                )
                plan.run()
                torch.cuda.synchronize()
            except Exception as exc:
                numeric["status"] = "reference_build_or_execution_failure"
                result["status"] = "reference_build_or_execution_failure"
                result["failure"] = {
                    **shape_record(shape), "phase": "matched_m26_reference",
                    "arm": arm, "failure_type": "build_or_prepare_or_execution_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                }
                write_result(output_path, result)
                print(f"REFERENCE FAIL M={m} N={n} K={k} {arm}: {exc}", flush=True)
                return False
            matched_outputs[(config["stages"], config["layout"])] = out
            matched_plans.append(plan)

        for config in configs:
            entry = {**shape_record(shape), **config}
            entry["arm"] = config_name(config)
            entry["matched_m26_arm"] = m26_name(config["stages"], config["layout"])
            try:
                out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                plan = prepare_scheduled_bf16_mma(
                    a, w, out=out, group=config["group"],
                    stages=config["stages"], layout=config["layout"],
                )
            except Exception as exc:
                entry.update({
                    "finite": False,
                    "passed": False,
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
                    "finite": False,
                    "passed": False,
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
            matched = matched_outputs[(config["stages"], config["layout"])]
            entry["bitwise_equal_to_matched_m26"] = bool(torch.equal(out, matched))
            if not entry["bitwise_equal_to_matched_m26"]:
                entry["matched_m26_numerical_metrics"] = numerical_metrics(matched, reference)
            numeric["results"].append(entry)
            write_result(output_path, result)
            print(
                f"NUM M={m} N={n} K={k} group={config['group']} "
                f"s={config['stages']} {config['layout']} "
                f"rms={entry['relative_rms']} "
                f"max/ref={entry['max_abs_over_max_abs_reference']} "
                f"m26_bitwise={entry['bitwise_equal_to_matched_m26']} "
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
    calls = {}
    outputs = {}
    owners = []
    configs = list(candidate_configs())

    for config in configs:
        arm = config_name(config)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        plan = prepare_scheduled_bf16_mma(
            a, w, out=out, group=config["group"],
            stages=config["stages"], layout=config["layout"],
        )
        calls[arm] = plan.run
        outputs[arm] = out
        owners.append(plan)

    m26_configs = list(matched_m26_configs())
    for config in m26_configs:
        arm = m26_name(config["stages"], config["layout"])
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        plan = prepare_tiled_bf16_mma(
            a, w, out=out, tile=(64, 128),
            stages=config["stages"], layout=config["layout"],
        )
        calls[arm] = plan.run
        outputs[arm] = out
        owners.append(plan)

    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    m25 = prepare_bf16_mma(a, w, out=out, stages=2)
    calls["m25_stage_2"] = m25.run
    outputs["m25_stage_2"] = out
    owners.append(m25)

    baseline_settings = {
        "torch_strict_bf16_reduction": False,
        "torch_default_bf16_reduction": True,
    }
    for arm, reduction in baseline_settings.items():
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        calls[arm] = torch_mm_call(a, wt, out, reduction)
        outputs[arm] = out
    return calls, outputs, owners, configs, m26_configs, baseline_settings


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
        calls, outputs, owners, configs, m26_configs, baseline_settings = (
            make_benchmark_arms(a, w, shape)
        )
        streams = {arm: torch.cuda.Stream() for arm in calls}
        reduction_settings = {**baseline_settings}
        graphs = {
            arm: build_graph(call, streams[arm], reduction_settings.get(arm))
            for arm, call in calls.items()
        }

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

        arm_records = {}
        for config in configs:
            arm_records[config_name(config)] = {"kind": "m27_scheduled_mma", **config}
        for config in m26_configs:
            arm_records[m26_name(config["stages"], config["layout"])] = {
                "kind": "m26_tiled_mma", "tile": [64, 128], **config,
            }
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

        samples = result_case["graph"]["samples_us_per_call"]
        result_case["graph"]["median_us_per_call"] = {
            arm: statistics.median(sample for round_samples in rounds for sample in round_samples)
            for arm, rounds in samples.items()
        }
        write_result(output_path, result)
        medians = result_case["graph"]["median_us_per_call"]
        fastest = min((name for name in medians if name.startswith("m27_")), key=medians.get)
        print(
            f"BENCH {m}x{n}x{k} fastest={fastest}:{medians[fastest]:.2f}us "
            f"M26_matched_min={min(medians[m26_name(c['stages'], c['layout'])] for c in m26_configs):.2f}us "
            f"M25={medians['m25_stage_2']:.2f}us "
            f"torch_strict={medians['torch_strict_bf16_reduction']:.2f}us "
            f"torch_default={medians['torch_default_bf16_reduction']:.2f}us",
            flush=True,
        )
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
        repo / "artifacts" / "m27-diagnostic" / f"m27_gemm_check_{timestamp}.json"
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
            "candidate_configs_per_shape": len(tuple(candidate_configs())),
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
            "scheduled_configs_per_shape": len(tuple(candidate_configs())),
            "matched_m26_configs_per_shape": len(tuple(matched_m26_configs())),
            "total_arms_per_shape": len(tuple(candidate_configs()))
            + len(tuple(matched_m26_configs())) + 3,
            "candidate_comparison": (
                "16 scheduled group/stage/layout candidates, four M26 matched 64x128 "
                "stage/layout candidates, M25 stage=2, Torch strict BF16 reduction off, "
                "Torch reduced-precision reduction on"
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
