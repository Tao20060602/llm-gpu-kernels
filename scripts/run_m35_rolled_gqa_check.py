#!/usr/bin/env python3
"""M35 rolled-segment CUDA GQA numerical and fixed-arm graph diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_m34_attention_check as m34
from llm_gpu_kernels.attention_baseline_benchmark import (
    _flat_tensors,
    _make_custom_inputs,
    _make_inputs,
    _runtime_info,
    _telemetry,
    _tensor_hash,
)
from llm_gpu_kernels.attention_segmented_benchmark import (
    _check,
    _chunked_fp32_reference,
)
from llm_gpu_kernels.attention_cuda import prepare_cuda_gqa_attention
from llm_gpu_kernels.attention_cuda_rolled import prepare_cuda_gqa_rolled
from llm_gpu_kernels.attention_raw_boundary import RawPackAttention


RESULT_PATH = ROOT / "results/m35-rolled-gqa-diagnostic.json"
ARTIFACT_ROOT = ROOT / "artifacts/m35-diagnostic"
REPORT_PATH = ROOT / "docs/M35_ROLLED_GQA_REPORT.md"
SEED = 3505
LIMITS = {"relative_rms": 0.008, "normalized_max": 0.008}
NUMERIC_CASES = m34.NUMERIC_CASES
PERFORMANCE_TOKENS = m34.PERFORMANCE_TOKENS
PERFORMANCE_HISTORY = m34.PERFORMANCE_HISTORY
CUDA_CONFIGS = m34.CUDA_CONFIGS
GRAPH_CALLS = m34.GRAPH_CALLS
MEASUREMENT_ORDERS = m34.MEASUREMENT_ORDERS
SAMPLES_PER_ARM_PER_ROUND = m34.SAMPLES_PER_ARM_PER_ROUND
ROLLED_COMPILE_RESOURCES = {
    (1, 0): {"registers_per_thread": 149, "spill_bytes": 0,
             "spill_stores": 0, "spill_loads": 0, "stack_bytes": 0,
             "ldl_instructions": 0, "stl_instructions": 0},
    (1, 1): {"registers_per_thread": 150, "spill_bytes": 0,
             "spill_stores": 0, "spill_loads": 0, "stack_bytes": 0,
             "ldl_instructions": 0, "stl_instructions": 0},
    (2, 0): {"registers_per_thread": 127, "spill_bytes": 0,
             "spill_stores": 0, "spill_loads": 0, "stack_bytes": 0,
             "ldl_instructions": 0, "stl_instructions": 0},
    (2, 1): {"registers_per_thread": 128, "spill_bytes": 0,
             "spill_stores": 0, "spill_loads": 0, "stack_bytes": 0,
             "ldl_instructions": 0, "stl_instructions": 0},
}
M34_REFERENCE_ARM = "m34_s1_p1"
FA2_ARM = "reusable_pack_fa2"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _source_paths() -> list[str]:
    paths = [
        "AGENTS.md",
        "docs/CONTRACT.md",
        "scripts/run_m34_attention_check.py",
        "scripts/run_m35_rolled_gqa_check.py",
        "src/llm_gpu_kernels/attention_baseline.py",
        "src/llm_gpu_kernels/attention_baseline_benchmark.py",
        "src/llm_gpu_kernels/attention_segmented.py",
        "src/llm_gpu_kernels/attention_segmented_benchmark.py",
        "src/llm_gpu_kernels/attention_cuda.py",
        "src/llm_gpu_kernels/attention_cuda_rolled.py",
        "src/llm_gpu_kernels/attention_raw_boundary.py",
        "src/llm_gpu_kernels/cuda_gqa_attention.cu",
        "src/llm_gpu_kernels/cuda_gqa_attention_rolled.cu",
    ]
    return paths


def _source_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in _source_paths()
        if (ROOT / name).is_file()
    }


def _rolled_resource_record(config: dict) -> dict:
    return {
        **config,
        "cta_m": 64,
        "key_tile": 32,
        "warps": 4,
        **m34._shared_memory_budget(config["stages"]),
        **ROLLED_COMPILE_RESOURCES[(config["stages"], config["p_mode"])],
    }


def _config_name(prefix: str, config: dict) -> str:
    return f"{prefix}_s{config['stages']}_p{config['p_mode']}"


def _make_inputs_for_case(index: int, tokens: int, lengths: tuple[int, ...]):
    q, segments = _make_custom_inputs(tokens, lengths, seed=SEED + index)
    if tokens in (17, 96):
        q, segments = m34._with_padded_current_v(q, segments)
    return q, segments


def _run_numerical_checks(record: dict, save) -> None:
    for case_index, (case, tokens, lengths) in enumerate(NUMERIC_CASES):
        q, segments = _make_inputs_for_case(case_index, tokens, lengths)
        oracle = _chunked_fp32_reference(q, segments)
        stride = list(segments[-1][1].stride())
        for config_index, config in enumerate(CUDA_CONFIGS):
            name = _config_name("rolled", config)
            out_mode = "provided" if (case_index + config_index) % 2 else "allocated"
            out = torch.empty_like(q) if out_mode == "provided" else None
            row = {
                "case": case,
                "T": tokens,
                "segment_lengths": list(lengths),
                "current_v_padded": not segments[-1][1].is_contiguous(),
                "current_v_stride": stride,
                "config": config,
                "out_mode": out_mode,
                "status": "running",
            }
            record["numerical_cases"].append(row)
            save()
            try:
                rolled_plan = prepare_cuda_gqa_rolled(
                    q, segments, out=out,
                    stages=config["stages"], p_mode=config["p_mode"],
                )
                rolled_output = rolled_plan.run()
                reference_plan = prepare_cuda_gqa_attention(
                    q, segments,
                    stages=config["stages"], p_mode=config["p_mode"],
                )
                reference_output = reference_plan.run()
                torch.cuda.synchronize(q.device)
                if tokens == 0:
                    empty_shape = (0, 32, 128)
                    if (tuple(rolled_output.shape) != empty_shape
                            or tuple(reference_output.shape) != empty_shape
                            or tuple(oracle.shape) != empty_shape):
                        raise RuntimeError("T0 output/reference/oracle shape mismatch")
                    rolled_finite = bool(torch.isfinite(rolled_output).all())
                    reference_finite = bool(torch.isfinite(reference_output).all())
                    if not rolled_finite or not reference_finite:
                        raise RuntimeError("T0 output/reference finite check failed")
                    rolled_metric = {
                        "relative_rms": 0.0, "normalized_max": 0.0, "max_abs": 0.0,
                        "finite": rolled_finite, "empty": True, "passed": True,
                    }
                    reference_metric = {
                        "relative_rms": 0.0, "normalized_max": 0.0, "max_abs": 0.0,
                        "finite": reference_finite, "empty": True, "passed": True,
                    }
                else:
                    rolled_metric = _check(rolled_output, oracle, f"{case} {name}")
                    reference_metric = _check(
                        reference_output, oracle,
                        f"{case} {_config_name('m34', config)} reference",
                    )
                row.update({
                    "vs_chunked_fp32": rolled_metric,
                    "m34_same_config_vs_chunked_fp32": reference_metric,
                    "bitwise_equal_to_m34_same_config": bool(
                        torch.equal(rolled_output, reference_output)
                    ),
                    "resources": _rolled_resource_record(config),
                    "passed": True,
                    "status": "passed",
                })
            except Exception as exc:
                row.update({
                    "status": "failed",
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=12),
                })
                save()
                raise
            save()


def _performance_case(tokens: int, case_index: int, record: dict, save) -> dict:
    progress = record["performance_cases"][-1]
    q, segments = _make_inputs(tokens, PERFORMANCE_HISTORY[0], seed=SEED + 1000)
    q, segments = m34._with_padded_current_v(q, segments)
    actual_history = tuple(k.shape[0] for k, _ in segments[:3])
    if actual_history != PERFORMANCE_HISTORY:
        raise RuntimeError(f"unexpected history lengths: {actual_history}")
    input_hash = _tensor_hash(*_flat_tensors(q, segments))
    oracle = _chunked_fp32_reference(q, segments)
    names = []
    functions = {}
    plans = {}
    conversion = {}

    for config in CUDA_CONFIGS:
        name = _config_name("rolled", config)
        plan = prepare_cuda_gqa_rolled(
            q, segments, stages=config["stages"], p_mode=config["p_mode"],
        )
        names.append(name)
        functions[name] = plan.run
        plans[name] = plan
        conversion[name] = {
            "converted_value_segments": [],
            "conversion_logical_bytes_per_call": 0,
            "conversion_read_write_traffic_bytes_per_call": 0,
            "contiguous_staging_bytes": 0,
        }

    m34_config = {"stages": 1, "p_mode": 1, "p_mode_name": "direct_register"}
    m34_plan = prepare_cuda_gqa_attention(q, segments, stages=1, p_mode=1)
    names.append(M34_REFERENCE_ARM)
    functions[M34_REFERENCE_ARM] = m34_plan.run
    plans[M34_REFERENCE_ARM] = m34_plan
    conversion[M34_REFERENCE_ARM] = {
        "converted_value_segments": [],
        "conversion_logical_bytes_per_call": 0,
        "conversion_read_write_traffic_bytes_per_call": 0,
        "contiguous_staging_bytes": 0,
    }

    fa2_plan = RawPackAttention(q, segments, reusable=True)
    fa2_function = fa2_plan.run
    packed_bytes = fa2_plan.packed_workspace_bytes
    fa2_conversion = {
        "packing_mode": "reusable_raw_layout_pack_each_call",
        "converted_value_segments": [],
        "conversion_logical_bytes_per_call": packed_bytes,
        "conversion_read_write_traffic_bytes_per_call": packed_bytes * 2,
        "contiguous_staging_bytes": 0,
        "reusable_packed_kv_staging_bytes": packed_bytes,
    }
    names.append(FA2_ARM)
    functions[FA2_ARM] = fa2_function
    plans[FA2_ARM] = fa2_plan
    conversion[FA2_ARM] = fa2_conversion

    progress.update({
        "phase": "prepared",
        "arm_order": names,
        "conversion": conversion,
        "input_sha256": input_hash,
        "current_v_stride": list(segments[-1][1].stride()),
    })
    save()

    before = {}
    before_outputs = {}
    for name in names:
        before_outputs[name] = functions[name]()
        before[name] = _check(before_outputs[name], oracle, f"{name} before timing")
        progress["before_timing_vs_fp32"] = dict(before)
        save()
    bitwise_before = bool(torch.equal(
        before_outputs["rolled_s1_p1"], before_outputs[M34_REFERENCE_ARM]
    ))

    captures = {}
    graph_before = {}
    for name in names:
        captures[name] = m34._make_graph(functions[name], q.device)
        graph_before[name] = _check(captures[name][1], oracle, f"{name} graph before timing")
        progress["graph_before_timing_vs_fp32"] = dict(graph_before)
        save()
    bitwise_graph_before = bool(torch.equal(
        captures["rolled_s1_p1"][1], captures[M34_REFERENCE_ARM][1]
    ))

    progress["phase"] = "timing"
    timing = m34._measure_graphs(captures, q.device, progress, save)

    after = {}
    after_outputs = {}
    graph_after = {}
    for name in names:
        after_outputs[name] = functions[name]()
        after[name] = _check(after_outputs[name], oracle, f"{name} after timing")
        captures[name][0].replay()
        graph_after[name] = _check(captures[name][1], oracle, f"{name} graph after timing")
        progress["after_timing_vs_fp32"] = dict(after)
        progress["graph_after_timing_vs_fp32"] = dict(graph_after)
        progress["phase"] = "post_timing_validation"
        save()
    bitwise_after = bool(torch.equal(
        after_outputs["rolled_s1_p1"], after_outputs[M34_REFERENCE_ARM]
    ))
    bitwise_graph_after = bool(torch.equal(
        captures["rolled_s1_p1"][1], captures[M34_REFERENCE_ARM][1]
    ))

    if input_hash != _tensor_hash(*_flat_tensors(q, segments)):
        raise RuntimeError(f"performance inputs changed for T={tokens}")

    resources = {
        _config_name("rolled", config): _rolled_resource_record(config)
        for config in CUDA_CONFIGS
    }
    resources[M34_REFERENCE_ARM] = m34._cuda_resource_record(m34_config)
    for config in CUDA_CONFIGS:
        name = _config_name("rolled", config)
        same_config_m34 = m34._cuda_resource_record(config)
        resources[name]["registers_delta_vs_m34_same_config"] = (
            resources[name]["registers_per_thread"]
            - same_config_m34["registers_per_thread"]
        )
        resources[name]["shared_memory_delta_vs_m34_same_config_bytes"] = (
            resources[name]["shared_memory_bytes"]
            - same_config_m34["shared_memory_bytes"]
        )
    resources[FA2_ARM] = {
        "packed_kv_workspace_bytes": fa2_plan.packed_workspace_bytes,
        "owned_output_bytes": q.numel() * q.element_size(),
        "shared_memory_bytes": None,
    }
    shared_baseline = resources["rolled_s1_p0"]["shared_memory_bytes"]
    shared_differences = {
        name: resource["shared_memory_bytes"] - shared_baseline
        for name, resource in resources.items()
        if name.startswith("rolled_")
    }
    progress.update({"resources": resources, "shared_memory_differences_vs_s1_p0": shared_differences})
    save()

    return {
        "case_index": case_index,
        "T": tokens,
        "history_lengths": list(PERFORMANCE_HISTORY),
        "segment_lengths": [k.shape[0] for k, _ in segments],
        "current_v_stride": list(segments[-1][1].stride()),
        "seed": SEED + 1000,
        "input_sha256": input_hash,
        "input_hash_preserved": True,
        "arm_order": names,
        "conversion": conversion,
        "before_timing_vs_fp32": before,
        "graph_before_timing_vs_fp32": graph_before,
        "timing": timing,
        "after_timing_vs_fp32": after,
        "graph_after_timing_vs_fp32": graph_after,
        "bitwise_equal_to_m34_s1_p1": {
            "api_before_timing": bitwise_before,
            "graph_before_timing": bitwise_graph_before,
            "api_after_timing": bitwise_after,
            "graph_after_timing": bitwise_graph_after,
            "required_for_correctness": False,
        },
        "resources": resources,
        "workspace_bytes": {
            **{name: plans[name].owners[-1].numel() * plans[name].owners[-1].element_size()
               for name in names if name != FA2_ARM},
            FA2_ARM: fa2_plan.packed_workspace_bytes,
        },
        "workspace_scope": (
            "rolled and M34 values are owned BF16 output bytes; FA2 is reusable packed K/V bytes "
            "and excludes its output; contiguous V staging is reported separately"
        ),
        "shared_memory_differences_vs_s1_p0": shared_differences,
        "status": "passed",
        "passed": True,
    }


def _resource_table() -> list[dict]:
    baseline = m34._shared_memory_budget(1)["shared_memory_bytes"]
    rows = []
    for config in CUDA_CONFIGS:
        row = _rolled_resource_record(config)
        same_config_m34 = m34._cuda_resource_record(config)
        row["shared_memory_delta_vs_s1_p0_bytes"] = row["shared_memory_bytes"] - baseline
        row["registers_delta_vs_m34_same_config"] = (
            row["registers_per_thread"] - same_config_m34["registers_per_thread"]
        )
        row["shared_memory_delta_vs_m34_same_config_bytes"] = (
            row["shared_memory_bytes"] - same_config_m34["shared_memory_bytes"]
        )
        rows.append(row)
    return rows


def _summary_report(record: dict) -> str:
    lines = [
        "# M35 rolled GQA comparison",
        "",
        f"Status: **{record.get('status', 'unknown')}**",
        "",
        "## Scope and protocol",
        "",
        "Fixed four-segment BF16 GQA semantics use Q[T,32,128], four K/V[L,8,128] pairs, "
        "fully visible history, causal current keys, GQA head mapping q_head//4, and FP32 "
        "softmax state with BF16 probabilities/output. The eight functional shapes match M34. "
        "All four rolled configurations use stages 1/2 × P mode 0/1 and the same declared "
        "CTA/shared-memory budget as M34.",
        "",
        "The fixed timing workloads are T32/T96/T256 with history lengths (2048,64,512) "
        "and current-V token stride 6144. Six arms are the four rolled variants, M34 S1/P1, "
        "and reusable raw-layout pack+FA2. Each graph replay contains four forwards; three "
        "forward/reverse/forward rounds use three samples per arm per round. FA2 copies the "
        "four segments into reusable packed K/V buffers on each invocation, so packing is "
        "included in its timed path; it reads padded current V directly without a separate "
        "contiguous-V staging copy. M34's earlier FA2 result used an additional contiguous-V "
        "conversion and is not a speed-comparable raw-layout baseline for this run.",
        "",
        "FP32 relative RMS and normalized max limits are 0.008. Bitwise comparisons with M34 "
        "are recorded for information and do not determine correctness.",
        "",
        "## Functional results",
        "",
        "| Case | T | Segment lengths | Config | Output mode | Padded current V | RMS | Normalized max | Bitwise equal to same-config M34 | Status |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in record.get("numerical_cases", []):
        metric = row.get("vs_chunked_fp32", {})
        config = row.get("config", {})
        lines.append(
            f"| {row.get('case')} | {row.get('T')} | {row.get('segment_lengths')} | "
            f"S{config.get('stages')} / P{config.get('p_mode')} ({config.get('p_mode_name')}) | "
            f"{row.get('out_mode')} | {row.get('current_v_padded')} | "
            f"{metric.get('relative_rms')} | {metric.get('normalized_max')} | "
            f"{row.get('bitwise_equal_to_m34_same_config')} | {row.get('status')} |"
        )
    lines += [
        "",
        "## Graph timing",
        "",
        "Milliseconds per complete forward; raw samples are retained in the JSON result.",
        "",
        "| T | Arm | p50 | p95 |",
        "|---:|---|---:|---:|",
    ]
    for case in record.get("performance_cases", []):
        for arm, summary in case.get("timing", {}).get("summary", {}).items():
            lines.append(
                f"| {case.get('T')} | {arm} | {summary.get('event_p50_ms_per_forward')} | "
                f"{summary.get('event_p95_ms_per_forward')} |"
            )
    lines += [
        "",
        "## M34 bitwise comparison",
        "",
        "The timing arm M34 S1/P1 is compared bitwise with rolled S1/P1 before and after "
        "timing. Functional cases compare every rolled configuration with the matching M34 "
        "stage/P mode. A mismatch is reported without failing the independent FP32 checks.",
        "",
        "| T | API before | Graph before | API after | Graph after |",
        "|---:|---:|---:|---:|---:|",
    ]
    for case in record.get("performance_cases", []):
        equal = case.get("bitwise_equal_to_m34_s1_p1", {})
        lines.append(
            f"| {case.get('T')} | {equal.get('api_before_timing')} | "
            f"{equal.get('graph_before_timing')} | {equal.get('api_after_timing')} | "
            f"{equal.get('graph_after_timing')} |"
        )
    lines += [
        "",
        "## Shared-memory and compiler resources",
        "",
        "The fixed allocation is `(64*128 + 2*32*128*stages + 64*32) * 2` bytes; P mode "
        "does not change its size.",
        "",
        "| Stages | P mode | Shared memory (B) | Delta vs S1/P0 (B) | Delta vs matching M34 (B) | Registers/thread | Register delta vs M34 | Spill bytes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in record.get("rolled_resource_table", []):
        lines.append(
            f"| {row.get('stages')} | P{row.get('p_mode')} ({row.get('p_mode_name')}) | "
            f"{row.get('shared_memory_bytes')} | {row.get('shared_memory_delta_vs_s1_p0_bytes')} | "
            f"{row.get('shared_memory_delta_vs_m34_same_config_bytes')} | "
            f"{row.get('registers_per_thread')} | {row.get('registers_delta_vs_m34_same_config')} | "
            f"{row.get('spill_bytes')} |"
        )
    lines += [
        "",
        "Per-case output/workspace sizes, conversion bytes, source hashes, and runtime versions "
        "are recorded in the JSON. Static resource figures do not measure occupancy or runtime utilization.",
        "",
        "## Artifacts",
        "",
        f"Diagnostic JSON: `{RESULT_PATH.relative_to(ROOT)}`.",
        f"Raw event/artifact directory: `{ARTIFACT_ROOT.relative_to(ROOT)}`.",
        "CUDA extension build output: `artifacts/m35-build/`.",
        "",
    ]
    if record.get("failure"):
        lines += ["## Failure", "", f"`{record['failure'].get('error')}`", ""]
    return "\n".join(lines)


def run() -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    source_hashes = _source_hashes()
    record = {
        "schema": 1,
        "run_id": "m35-rolled-gqa-diagnostic",
        "status": "running",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "limits": LIMITS,
        "configs": list(CUDA_CONFIGS),
        "rolled_resource_table": _resource_table(),
        "protocol": {
            "functional_shapes": 8,
            "rolled_configurations": 4,
            "expected_numerical_results": 32,
            "performance_tokens": list(PERFORMANCE_TOKENS),
            "performance_history_lengths": list(PERFORMANCE_HISTORY),
            "performance_arms": 6,
            "graph_calls_per_replay": GRAPH_CALLS,
            "round_order": list(MEASUREMENT_ORDERS),
            "samples_per_arm_per_round": SAMPLES_PER_ARM_PER_ROUND,
            "timed_unit": "complete forward, milliseconds",
        },
        "source_hashes": source_hashes,
        "numerical_cases": [],
        "performance_cases": [],
        "telemetry_before": None,
    }

    def save() -> None:
        _write_json(RESULT_PATH, record)
        (ARTIFACT_ROOT / "run_manifest.json").write_text(
            json.dumps({
                "run_id": record["run_id"],
                "status": record["status"],
                "source_hashes": record["source_hashes"],
                "rolled_resource_table": record["rolled_resource_table"],
            }, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        REPORT_PATH.write_text(_summary_report(record), encoding="utf-8")

    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the active project environment")
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
        if (gpu.major, gpu.minor) != (8, 6):
            raise RuntimeError(f"M35 is scoped to SM86, got SM{gpu.major}{gpu.minor}")
        record["runtime"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_path": torch.__file__,
            "cuda_runtime": torch.version.cuda,
            "triton": triton.__version__,
            "gpu": gpu.name,
            "sm": [gpu.major, gpu.minor],
            "fa2": _runtime_info(),
            "tf32": False,
            "bf16_reduced_precision_reduction": False,
        }
        record["telemetry_before"] = _telemetry()
        save()

        _run_numerical_checks(record, save)
        for case_index, tokens in enumerate(PERFORMANCE_TOKENS):
            row = {"T": tokens, "status": "running"}
            record["performance_cases"].append(row)
            save()
            row.update(_performance_case(tokens, case_index, record, save))
            save()

        after_hashes = _source_hashes()
        if source_hashes != after_hashes:
            raise RuntimeError("source files changed during M35 diagnostic")
        record["source_hashes_after"] = after_hashes
        record["status"] = "passed"
        record["passed"] = True
    except Exception as exc:
        record["status"] = "failed"
        record["passed"] = False
        record["failure"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=16),
        }
        for collection in (record["numerical_cases"], record["performance_cases"]):
            if collection and collection[-1].get("status") == "running":
                collection[-1]["status"] = "failed"
                collection[-1]["passed"] = False
                collection[-1]["failure"] = record["failure"]
        raise
    finally:
        record["telemetry_after"] = _telemetry()
        record["source_hashes_after"] = _source_hashes()
        save()
        print(f"M35 result: {RESULT_PATH.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    run()
