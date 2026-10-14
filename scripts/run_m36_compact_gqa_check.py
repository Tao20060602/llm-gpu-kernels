#!/usr/bin/env python3
"""M37 compact-v2 CUDA GQA numerical and fixed-arm graph diagnostic."""

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
from llm_gpu_kernels.attention_cuda_compact import (
    load_cuda_gqa_compact,
    prepare_cuda_gqa_compact,
)
from llm_gpu_kernels.attention_raw_boundary import RawPackAttention


RESULT_PATH = ROOT / "results/m37-compact-v2-gqa-diagnostic.json"
ARTIFACT_ROOT = ROOT / "artifacts/m37-compact-v2-diagnostic"
REPORT_PATH = ROOT / "docs/M37_COMPACT_V2_GQA_REPORT.md"
SEED = 3607
LIMITS = {"relative_rms": 0.008, "normalized_max": 0.008}
NUMERIC_CASES = m34.NUMERIC_CASES
PERFORMANCE_TOKENS = m34.PERFORMANCE_TOKENS
PERFORMANCE_HISTORY = m34.PERFORMANCE_HISTORY
GRAPH_CALLS = m34.GRAPH_CALLS
MEASUREMENT_ORDERS = m34.MEASUREMENT_ORDERS
SAMPLES_PER_ARM_PER_ROUND = m34.SAMPLES_PER_ARM_PER_ROUND
COMPACT_CONFIGS = tuple(
    # p_mode records the fixed implementation fact; it is not a compact API choice.
    {"stages": stages, "p_mode": 1, "p_mode_name": "direct_register"}
    for stages in (1, 2)
)
M34_REFERENCE_ARMS = {1: "m34_s1_p1", 2: "m34_s2_p1"}
FA2_ARM = "reusable_raw_pack_fa2"

# Register/local-memory and SASS counts are from the M37 v2 build evidence in
# artifacts/m37-build. Dynamic shared bytes are derived by _compact_resource_record.
COMPACT_COMPILE_RESOURCES = {
    1: {
        "registers_per_thread": 221,
        "spill_bytes": 0,
        "spill_stores": 0,
        "spill_loads": 0,
        "stack_bytes": 0,
        "ldl_instructions": 0,
        "stl_instructions": 0,
        "hmma_bf16_f32_instructions": 256,
        "ldsm_instructions": 288,
    },
    2: {
        "registers_per_thread": 231,
        "spill_bytes": 0,
        "spill_stores": 0,
        "spill_loads": 0,
        "stack_bytes": 0,
        "ldl_instructions": 0,
        "stl_instructions": 0,
        "hmma_bf16_f32_instructions": 256,
        "ldsm_instructions": 288,
    },
}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _source_paths() -> list[str]:
    paths = [
        "AGENTS.md",
        "docs/CONTRACT.md",
        "scripts/run_m34_attention_check.py",
        "scripts/run_m36_compact_gqa_check.py",
        "src/llm_gpu_kernels/attention_baseline.py",
        "src/llm_gpu_kernels/attention_baseline_benchmark.py",
        "src/llm_gpu_kernels/attention_segmented.py",
        "src/llm_gpu_kernels/attention_segmented_benchmark.py",
        "src/llm_gpu_kernels/attention_cuda.py",
        "src/llm_gpu_kernels/attention_cuda_compact.py",
        "src/llm_gpu_kernels/attention_raw_boundary.py",
    ]
    paths.extend(
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "src/llm_gpu_kernels").glob("*.cu"))
        if path.is_file()
    )
    return list(dict.fromkeys(paths))


def _source_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in _source_paths()
        if (ROOT / name).is_file()
    }


def _compact_resource_record(config: dict) -> dict:
    stages = config["stages"]
    qk_bytes = 64 * 128 * 2
    kv_pipeline_bytes = 2 * 32 * 128 * stages * 2
    return {
        **config,
        "cta_m": 64,
        "key_tile": 32,
        "warps": 4,
        "qk_tile_bytes": qk_bytes,
        "kv_pipeline_bytes": kv_pipeline_bytes,
        "probability_scratch_bytes": 0,
        "shared_memory_bytes": qk_bytes + kv_pipeline_bytes,
        **COMPACT_COMPILE_RESOURCES[stages],
    }


def _config_name(prefix: str, stages: int) -> str:
    return f"{prefix}_s{stages}_p1"


def _make_numeric_inputs(index: int, tokens: int, lengths: tuple[int, ...]):
    q, segments = _make_custom_inputs(tokens, lengths, seed=SEED + index)
    if tokens in (17, 96):
        q, segments = m34._with_padded_current_v(q, segments)
    return q, segments


def _check_empty(actual, reference, oracle, label: str) -> dict:
    expected_shape = (0, 32, 128)
    if (tuple(actual.shape) != expected_shape
            or tuple(reference.shape) != expected_shape
            or tuple(oracle.shape) != expected_shape):
        raise RuntimeError(f"{label}: output/reference/oracle empty shape mismatch")
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(reference).all()):
        raise RuntimeError(f"{label}: output/reference empty finite check failed")
    return {
        "relative_rms": 0.0,
        "normalized_max": 0.0,
        "max_abs": 0.0,
        "finite": True,
        "empty": True,
        "passed": True,
    }


def _run_numerical_checks(record: dict, save) -> None:
    for case_index, (case, tokens, lengths) in enumerate(NUMERIC_CASES):
        q, segments = _make_numeric_inputs(case_index, tokens, lengths)
        oracle = _chunked_fp32_reference(q, segments)
        stride = list(segments[-1][1].stride())
        for config_index, config in enumerate(COMPACT_CONFIGS):
            stages = config["stages"]
            name = _config_name("compact", stages)
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
                compact_plan = prepare_cuda_gqa_compact(
                    q, segments, out=out, stages=stages,
                )
                compact_output = compact_plan.run()
                reference_plan = prepare_cuda_gqa_attention(
                    q, segments, stages=stages, p_mode=1,
                )
                reference_output = reference_plan.run()
                torch.cuda.synchronize(q.device)
                if tokens == 0:
                    compact_metric = _check_empty(compact_output, reference_output, oracle, case)
                    reference_metric = {
                        "relative_rms": 0.0,
                        "normalized_max": 0.0,
                        "max_abs": 0.0,
                        "finite": True,
                        "empty": True,
                        "passed": True,
                    }
                else:
                    compact_metric = _check(compact_output, oracle, f"{case} {name}")
                    reference_metric = _check(
                        reference_output,
                        oracle,
                        f"{case} {_config_name('m34', stages)} reference",
                    )
                row.update({
                    "vs_chunked_fp32": compact_metric,
                    "m34_same_stage_p1_vs_chunked_fp32": reference_metric,
                    "bitwise_equal_to_m34_same_stage_p1": bool(
                        torch.equal(compact_output, reference_output)
                    ),
                    "resources": _compact_resource_record(config),
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


def _performance_case(tokens: int, record: dict, save) -> dict:
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

    for config in COMPACT_CONFIGS:
        stages = config["stages"]
        name = _config_name("compact", stages)
        plan = prepare_cuda_gqa_compact(
            q, segments, stages=stages,
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

    for stages in (1, 2):
        name = M34_REFERENCE_ARMS[stages]
        plan = prepare_cuda_gqa_attention(q, segments, stages=stages, p_mode=1)
        names.append(name)
        functions[name] = plan.run
        plans[name] = plan
        conversion[name] = {
            "converted_value_segments": [],
            "conversion_logical_bytes_per_call": 0,
            "conversion_read_write_traffic_bytes_per_call": 0,
            "contiguous_staging_bytes": 0,
        }

    fa2_plan = RawPackAttention(q, segments, reusable=True)
    packed_bytes = fa2_plan.packed_workspace_bytes
    names.append(FA2_ARM)
    functions[FA2_ARM] = fa2_plan.run
    plans[FA2_ARM] = fa2_plan
    conversion[FA2_ARM] = {
        "packing_mode": "reusable_raw_layout_pack_each_call",
        "converted_value_segments": [],
        "conversion_logical_bytes_per_call": packed_bytes,
        "conversion_read_write_traffic_bytes_per_call": packed_bytes * 2,
        "contiguous_staging_bytes": 0,
        "reusable_packed_kv_staging_bytes": packed_bytes,
    }

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

    captures = {}
    graph_before = {}
    for name in names:
        captures[name] = m34._make_graph(functions[name], q.device)
        graph_before[name] = _check(captures[name][1], oracle, f"{name} graph before timing")
        progress["graph_before_timing_vs_fp32"] = dict(graph_before)
        save()

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

    bitwise = {}
    for stages in (1, 2):
        compact_name = _config_name("compact", stages)
        m34_name = M34_REFERENCE_ARMS[stages]
        bitwise[str(stages)] = {
            "api_before_timing": bool(torch.equal(
                before_outputs[compact_name], before_outputs[m34_name]
            )),
            "graph_before_timing": bool(torch.equal(captures[compact_name][1], captures[m34_name][1])),
            "api_after_timing": bool(torch.equal(
                after_outputs[compact_name], after_outputs[m34_name]
            )),
            "graph_after_timing": bool(torch.equal(captures[compact_name][1], captures[m34_name][1])),
            "required_for_correctness": False,
        }

    if input_hash != _tensor_hash(*_flat_tensors(q, segments)):
        raise RuntimeError(f"performance inputs changed for T={tokens}")

    compact_resources = {
        _config_name("compact", config["stages"]): _compact_resource_record(config)
        for config in COMPACT_CONFIGS
    }
    for stages in (1, 2):
        compact_name = _config_name("compact", stages)
        m34_resource = m34._cuda_resource_record({
            "stages": stages,
            "p_mode": 1,
            "p_mode_name": "direct_register",
        })
        compact_resources[compact_name]["shared_memory_delta_vs_m34_bytes"] = (
            compact_resources[compact_name]["shared_memory_bytes"]
            - m34_resource["shared_memory_bytes"]
        )
        compact_resources[compact_name]["registers_delta_vs_m34"] = (
            None if compact_resources[compact_name]["registers_per_thread"] is None
            else compact_resources[compact_name]["registers_per_thread"]
            - m34_resource["registers_per_thread"]
        )
    resources = dict(compact_resources)
    for stages in (1, 2):
        resources[M34_REFERENCE_ARMS[stages]] = m34._cuda_resource_record({
            "stages": stages,
            "p_mode": 1,
            "p_mode_name": "direct_register",
        })
    resources[FA2_ARM] = {
        "packed_kv_workspace_bytes": fa2_plan.packed_workspace_bytes,
        "owned_output_bytes": q.numel() * q.element_size(),
        "shared_memory_bytes": None,
    }
    shared_differences = {
        _config_name("compact", stages): (
            compact_resources[_config_name("compact", stages)]["shared_memory_bytes"]
            - resources[M34_REFERENCE_ARMS[stages]]["shared_memory_bytes"]
        )
        for stages in (1, 2)
    }
    progress.update({
        "resources": resources,
        "shared_memory_differences_vs_m34_same_stage_p1": shared_differences,
    })
    save()

    return {
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
        "bitwise_equal_to_m34_same_stage_p1": bitwise,
        "resources": resources,
        "workspace_bytes": {
            **{
                name: plans[name].owners[-1].numel() * plans[name].owners[-1].element_size()
                for name in names if name != FA2_ARM
            },
            FA2_ARM: fa2_plan.packed_workspace_bytes,
        },
        "workspace_scope": (
            "compact and M34 values are owned BF16 output bytes; FA2 is reusable packed K/V bytes "
            "and excludes its output; no contiguous V staging is used"
        ),
        "shared_memory_differences_vs_m34_same_stage_p1": shared_differences,
        "status": "passed",
        "passed": True,
    }


def _resource_table() -> list[dict]:
    compact_s1_shared = _compact_resource_record(COMPACT_CONFIGS[0])["shared_memory_bytes"]
    rows = []
    for config in COMPACT_CONFIGS:
        row = _compact_resource_record(config)
        stages = config["stages"]
        m34_resource = m34._cuda_resource_record({
            "stages": stages,
            "p_mode": 1,
            "p_mode_name": "direct_register",
        })
        row["shared_memory_delta_vs_m34_bytes"] = (
            row["shared_memory_bytes"] - m34_resource["shared_memory_bytes"]
        )
        row["shared_memory_delta_vs_compact_s1_bytes"] = (
            row["shared_memory_bytes"] - compact_s1_shared
        )
        row["registers_delta_vs_m34"] = (
            None if row["registers_per_thread"] is None
            else row["registers_per_thread"] - m34_resource["registers_per_thread"]
        )
        rows.append(row)
    return rows


def _summary_report(record: dict) -> str:
    lines = [
        "# M37 compact-v2 GQA comparison",
        "",
        f"Status: **{record.get('status', 'unknown')}**",
        "",
        "## Scope and protocol",
        "",
        "Fixed BF16 GQA semantics use Q[T,32,128], four K/V[L,8,128] pairs, fully visible "
        "history, causal current keys, and the same FP32 softmax/BF16 probability and output "
        "arithmetic as M34. M37 compact v2 fixes direct-register P and supports stages 1/2. Its shared "
        "layout omits M34's unused 4 KiB probability scratch.",
        "",
        "The eight functional shapes match M34; each of the two compact configurations is "
        "checked against the independent chunked FP32 oracle, and bitwise equality against "
        "M34 with the same stage count is recorded as a non-gating comparison.",
        "",
        "Timing uses T32/T96/T256 with history lengths (2048,64,512), current-V token stride "
        "6144, and five arms: compact v2 S1/S2, M34 S1/P1 and S2/P1, and reusable raw-layout "
        "pack+FA2. Each graph replay contains four forwards; three forward/reverse/forward "
        "rounds use three samples per arm per round. FA2 packs the four source segments into "
        "reusable K/V buffers on each invocation, so packing is timed; it reads padded current "
        "V directly without separate contiguous-V staging.",
        "",
        "FP32 relative RMS and normalized max limits are 0.008.",
        "",
        "## Functional results",
        "",
        "| Case | T | Segment lengths | Stages | Output mode | Padded current V | RMS | Normalized max | Bitwise equal to same-stage M34 | Status |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in record.get("numerical_cases", []):
        metric = row.get("vs_chunked_fp32", {})
        config = row.get("config", {})
        lines.append(
            f"| {row.get('case')} | {row.get('T')} | {row.get('segment_lengths')} | "
            f"{config.get('stages')} | {row.get('out_mode')} | {row.get('current_v_padded')} | "
            f"{metric.get('relative_rms')} | {metric.get('normalized_max')} | "
            f"{row.get('bitwise_equal_to_m34_same_stage_p1')} | {row.get('status')} |"
        )
    lines += [
        "",
        "## Graph timing",
        "",
        "Milliseconds per complete forward; all raw samples are retained in the JSON.",
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
        "## Compact resources and M34 comparison",
        "",
        "Compact v2 shared memory is `64*128*2 + 2*32*128*stages*2`; the omitted probability "
        "scratch is 4096 bytes. M34 uses the same Q/KV allocation plus that scratch.",
        "",
        "| Stages | Shared bytes | Delta vs M34 (B) | Delta vs compact S1 (B) | Registers/thread | Delta regs vs M34 | Spill bytes | Stack bytes | LDL/STL | HMMA/LDSM |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in record.get("compact_resource_table", []):
        lines.append(
            f"| {row.get('stages')} | {row.get('shared_memory_bytes')} | "
            f"{row.get('shared_memory_delta_vs_m34_bytes')} | "
            f"{row.get('shared_memory_delta_vs_compact_s1_bytes')} | "
            f"{row.get('registers_per_thread')} | {row.get('registers_delta_vs_m34')} | "
            f"{row.get('spill_bytes')} | {row.get('stack_bytes')} | "
            f"{row.get('ldl_instructions')}/{row.get('stl_instructions')} | "
            f"{row.get('hmma_bf16_f32_instructions')}/{row.get('ldsm_instructions')} |"
        )
    lines += [
        "",
        "Per-case arm resources, workspace sizes, raw-pack bytes, source hashes, and runtime "
        "versions are recorded in the diagnostic JSON. Static resources do not measure runtime occupancy.",
        "",
        "## Artifacts",
        "",
        f"Diagnostic JSON: `{RESULT_PATH.relative_to(ROOT)}`.",
        f"Raw diagnostic directory: `{ARTIFACT_ROOT.relative_to(ROOT)}`.",
        "CUDA extension build output: `artifacts/m36-build-v2/`.",
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
        "run_id": "m37-compact-v2-gqa-diagnostic",
        "status": "running",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "limits": LIMITS,
        "configs": list(COMPACT_CONFIGS),
        "compact_resource_table": _resource_table(),
        "protocol": {
            "functional_shapes": 8,
            "compact_configurations": 2,
            "expected_numerical_results": 16,
            "performance_tokens": list(PERFORMANCE_TOKENS),
            "performance_history_lengths": list(PERFORMANCE_HISTORY),
            "performance_arms": 5,
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
                "compact_resource_table": record["compact_resource_table"],
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
            raise RuntimeError(f"compact v2 is scoped to SM86, got SM{gpu.major}{gpu.minor}")
        record["compact_extension_module"] = load_cuda_gqa_compact().__name__
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
        for tokens in PERFORMANCE_TOKENS:
            row = {"T": tokens, "status": "running"}
            record["performance_cases"].append(row)
            save()
            row.update(_performance_case(tokens, record, save))
            save()

        after_hashes = _source_hashes()
        if source_hashes != after_hashes:
            raise RuntimeError("source files changed during M37 compact-v2 diagnostic")
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
        print(f"M37 result: {RESULT_PATH.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    run()
