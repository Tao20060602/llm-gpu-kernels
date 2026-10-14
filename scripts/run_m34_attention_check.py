#!/usr/bin/env python3
"""M34 fixed CUDA GQA numerical and diagnostic timing run.

This script intentionally contains only the predeclared 32 numerical checks
and three six-arm graph measurements.  It is not a tuning or search harness.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

import numpy as np
import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_gpu_kernels.attention_baseline import prepare_segmented_attention
from llm_gpu_kernels.attention_baseline_benchmark import (
    _flat_tensors,
    _make_custom_inputs,
    _make_inputs,
    _runtime_info,
    _telemetry,
    _tensor_hash,
)
from llm_gpu_kernels.attention_nonempty import prepare_nonempty_kernel
from llm_gpu_kernels.attention_segmented import AttentionConfig
from llm_gpu_kernels.attention_segmented_benchmark import (
    _check,
    _chunked_fp32_reference,
    _resources as _triton_resources,
)
from llm_gpu_kernels.benchmark import compile_artifacts
from llm_gpu_kernels.attention_cuda import prepare_cuda_gqa_attention


RESULT_PATH = ROOT / "results/m34-cuda-gqa-diagnostic.json"
ARTIFACT_ROOT = ROOT / "artifacts/m34-diagnostic"
SEED = 3409
LIMITS = {"relative_rms": 0.008, "normalized_max": 0.008}
NUMERIC_CASES = (
    ("T0_history_empty", 0, (0, 0, 0, 0)),
    ("T1_history_empty", 1, (0, 0, 0, 1)),
    ("T9_history_17_0_3", 9, (17, 0, 3, 9)),
    ("T17_history_33_1_0", 17, (33, 1, 0, 17)),
    ("T64_history_64_16_32", 64, (64, 16, 32, 64)),
    ("T65_history_31_32_17", 65, (31, 32, 17, 65)),
    ("T96_history_128_64_32", 96, (128, 64, 32, 96)),
    ("T129_history_64_0_33", 129, (64, 0, 33, 129)),
)
PERFORMANCE_TOKENS = (32, 96, 256)
PERFORMANCE_HISTORY = (2048, 64, 512)
CUDA_CONFIGS = tuple(
    {"stages": stages, "p_mode": p_mode,
     "p_mode_name": "shared" if p_mode == 0 else "direct_register"}
    for stages in (1, 2)
    for p_mode in (0, 1)
)
CUDA_COMPILE_RESOURCES = {
    (1, 0): {"registers_per_thread": 235, "spill_bytes": 0,
             "spill_stores": 0, "spill_loads": 0, "stack_bytes": 0, "barriers": 1},
    (1, 1): {"registers_per_thread": 223, "spill_bytes": 0,
             "spill_stores": 0, "spill_loads": 0, "stack_bytes": 0, "barriers": 1},
    (2, 0): {"registers_per_thread": 237, "spill_bytes": 0,
             "spill_stores": 0, "spill_loads": 0, "stack_bytes": 0, "barriers": 1},
    (2, 1): {"registers_per_thread": 230, "spill_bytes": 0,
             "spill_stores": 0, "spill_loads": 0, "stack_bytes": 0, "barriers": 1},
}
GRAPH_CALLS = 4
MEASUREMENT_ORDERS = ("forward", "reverse", "forward")
SAMPLES_PER_ARM_PER_ROUND = 3
TRITON_ARM = "m15_triton_g4_s2"
FA2_ARM = "reusable_pack_fa2"
CUDA_ARM_PREFIX = "cuda"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _source_paths() -> list[str]:
    paths = [
        "AGENTS.md",
        "docs/CONTRACT.md",
        "scripts/run_m34_attention_check.py",
        "src/llm_gpu_kernels/attention_baseline.py",
        "src/llm_gpu_kernels/attention_baseline_benchmark.py",
        "src/llm_gpu_kernels/attention_nonempty.py",
        "src/llm_gpu_kernels/attention_segmented.py",
        "src/llm_gpu_kernels/attention_segmented_benchmark.py",
        "src/llm_gpu_kernels/benchmark.py",
    ]
    for pattern in ("attention_cuda*", "cuda_gqa_attention*"):
        paths.extend(
            path.relative_to(ROOT).as_posix()
            for path in sorted((ROOT / "src/llm_gpu_kernels").glob(pattern))
            if path.is_file()
        )
    return list(dict.fromkeys(paths))


def _source_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in _source_paths()
        if (ROOT / name).is_file()
    }


def _shared_memory_budget(stages: int) -> dict:
    """M34's fixed CTA allocation, including the retained probability scratch."""
    qk_bytes = 64 * 128 * 2
    kv_pipeline_bytes = 2 * 32 * 128 * stages * 2
    probability_scratch_bytes = 64 * 32 * 2
    return {
        "qk_tile_bytes": qk_bytes,
        "kv_pipeline_bytes": kv_pipeline_bytes,
        "probability_scratch_bytes": probability_scratch_bytes,
        "shared_memory_bytes": qk_bytes + kv_pipeline_bytes + probability_scratch_bytes,
    }


def _cuda_resource_record(config: dict) -> dict:
    budget = _shared_memory_budget(config["stages"])
    return {
        **config,
        "cta_m": 64,
        "key_tile": 32,
        "warps": 4,
        **budget,
        **CUDA_COMPILE_RESOURCES[(config["stages"], config["p_mode"])],
    }


def _padded_value(value: torch.Tensor, token_stride: int = 6144) -> torch.Tensor:
    if value.shape[0] == 0:
        return value
    if token_stride < value.shape[1] * value.shape[2] or token_stride % 8:
        raise ValueError("padded V token stride must fit the row and be divisible by 8")
    result = torch.empty_strided(
        tuple(value.shape),
        (token_stride, value.shape[2], 1),
        dtype=value.dtype,
        device=value.device,
    )
    result.copy_(value)
    return result


def _with_padded_current_v(q, segments):
    copied = list(segments)
    key, value = copied[-1]
    padded = _padded_value(value)
    copied[-1] = (key, padded)
    return q, tuple(copied)


def _make_numeric_inputs(index: int, tokens: int, lengths: tuple[int, ...]):
    q, segments = _make_custom_inputs(tokens, lengths, seed=SEED + index)
    if tokens in (17, 96):
        q, segments = _with_padded_current_v(q, segments)
    return q, segments


def _prepare_legacy_segments(q, segments):
    """Preallocate contiguous V staging and return its full per-call copy."""
    prepared = []
    copies = []
    for key, value in segments:
        if value.is_contiguous():
            prepared.append((key, value))
            continue
        contiguous = torch.empty(tuple(value.shape), dtype=value.dtype, device=value.device)
        contiguous.copy_(value)
        prepared.append((key, contiguous))
        copies.append((contiguous, value))

    def convert():
        for destination, source in copies:
            destination.copy_(source)

    conversion_logical_bytes = sum(source.numel() * source.element_size() for _, source in copies)
    return tuple(prepared), convert, {
        "converted_value_segments": [i for i, (_, value) in enumerate(segments) if not value.is_contiguous()],
        "conversion_logical_bytes_per_call": conversion_logical_bytes,
        "conversion_read_write_traffic_bytes_per_call": conversion_logical_bytes * 2,
        "contiguous_staging_bytes": sum(destination.numel() * destination.element_size()
                                        for destination, _ in copies),
    }


def _make_graph(function, device: torch.device):
    current = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(current)
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            function()
    current.wait_stream(capture_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        output = None
        for _ in range(GRAPH_CALLS):
            output = function()
    current.wait_stream(capture_stream)
    graph.replay()
    torch.cuda.synchronize(device)
    return graph, output


def _measure_graphs(captures: dict[str, tuple], device: torch.device, progress, save) -> dict:
    names = list(captures)
    raw = {name: [] for name in names}
    rounds = []
    summary = {name: {"event_p50_ms_per_forward": None,
                      "event_p95_ms_per_forward": None} for name in names}
    progress["timing"] = {
        "rounds": rounds,
        "summary": summary,
        "round_order": list(MEASUREMENT_ORDERS),
        "samples_per_arm_per_round": SAMPLES_PER_ARM_PER_ROUND,
        "calls_per_graph_replay": GRAPH_CALLS,
        "units": "milliseconds per complete attention forward",
    }
    for direction in MEASUREMENT_ORDERS:
        order = names if direction == "forward" else list(reversed(names))
        round_row = {
            "direction": direction,
            "order": order,
            "samples_ms_per_forward": {},
        }
        for name in order:
            graph = captures[name][0]
            samples = []
            for _ in range(SAMPLES_PER_ARM_PER_ROUND):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / GRAPH_CALLS)
                raw[name].append(samples[-1])
                round_row["samples_ms_per_forward"][name] = list(samples)
                summary[name] = {
                    "event_p50_ms_per_forward": statistics.median(raw[name]),
                    "event_p95_ms_per_forward": float(np.percentile(raw[name], 95)),
                }
                progress["timing"]["rounds"] = [*rounds, round_row]
                progress["timing"]["summary"] = dict(summary)
                save()
            round_row["samples_ms_per_forward"][name] = samples
        rounds.append(round_row)
        progress["timing"]["rounds"] = list(rounds)
        save()
    progress["timing"]["rounds_count"] = len(MEASUREMENT_ORDERS)
    return progress["timing"]


def _make_cuda_arm(q, segments, config: dict):
    plan = prepare_cuda_gqa_attention(
        q,
        segments,
        stages=config["stages"],
        p_mode=config["p_mode"],
    )
    return plan.run, plan


def _run_numerical_checks(record: dict, save) -> None:
    rows = record["numerical_cases"]
    for case_index, (name, tokens, lengths) in enumerate(NUMERIC_CASES):
        q, segments = _make_numeric_inputs(case_index, tokens, lengths)
        oracle = _chunked_fp32_reference(q, segments)
        padded_current_v = not segments[-1][1].is_contiguous()
        stride = list(segments[-1][1].stride())
        for config_index, config in enumerate(CUDA_CONFIGS):
            output_mode = "provided" if (case_index + config_index) % 2 else "allocated"
            out = torch.empty_like(q) if output_mode == "provided" else None
            row = {
                "case": name,
                "T": tokens,
                "segment_lengths": list(lengths),
                "config": config,
                "out_mode": output_mode,
                "current_v_padded": padded_current_v,
                "current_v_stride": stride,
                "status": "running",
            }
            rows.append(row)
            save()
            try:
                plan = prepare_cuda_gqa_attention(
                    q,
                    segments,
                    out=out,
                    stages=config["stages"],
                    p_mode=config["p_mode"],
                )
                actual = plan.run()
                torch.cuda.synchronize(q.device)
                if tokens == 0:
                    expected_shape = (0, 32, 128)
                    if tuple(actual.shape) != expected_shape or tuple(oracle.shape) != expected_shape:
                        raise RuntimeError(
                            f"empty output shape mismatch: {tuple(actual.shape)} / {tuple(oracle.shape)}"
                        )
                    finite = bool(torch.isfinite(actual).all())
                    if not finite:
                        raise RuntimeError("empty output finite check failed")
                    row["vs_chunked_fp32"] = {
                        "relative_rms": 0.0,
                        "normalized_max": 0.0,
                        "max_abs": 0.0,
                        "finite": True,
                        "empty": True,
                        "passed": True,
                    }
                else:
                    row["vs_chunked_fp32"] = _check(actual, oracle, f"{name} {config}")
                row["output_shape"] = list(actual.shape)
                row["output_dtype"] = str(actual.dtype)
                row["resources"] = _cuda_resource_record(config)
                row["status"] = "passed"
                row["passed"] = True
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
    q, segments = _with_padded_current_v(q, segments)
    input_hash = _tensor_hash(*_flat_tensors(q, segments))
    oracle = _chunked_fp32_reference(q, segments)
    names = []
    functions = {}
    plans = {}
    arm_inputs = {}

    for config in CUDA_CONFIGS:
        name = f"{CUDA_ARM_PREFIX}_s{config['stages']}_p{config['p_mode']}"
        function, plan = _make_cuda_arm(q, segments, config)
        names.append(name)
        functions[name] = function
        plans[name] = plan
        arm_inputs[name] = {"converted_value_segments": [],
                            "conversion_logical_bytes_per_call": 0,
                            "conversion_read_write_traffic_bytes_per_call": 0,
                            "contiguous_staging_bytes": 0}

    legacy_segments, legacy_convert, legacy_conversion = _prepare_legacy_segments(q, segments)
    legacy_config = AttentionConfig(head_group=4, token_tile=32, key_tile=64, warps=4, stages=2)
    triton_plan = prepare_nonempty_kernel(q, legacy_segments, config=legacy_config)
    triton_function = lambda: (legacy_convert(), triton_plan.run())[1]
    names.append(TRITON_ARM)
    functions[TRITON_ARM] = triton_function
    plans[TRITON_ARM] = triton_plan
    arm_inputs[TRITON_ARM] = legacy_conversion

    fa2_segments, fa2_convert, fa2_conversion = _prepare_legacy_segments(q, segments)
    fa2_plan = prepare_segmented_attention(q, fa2_segments)
    fa2_function = lambda: (fa2_convert(), fa2_plan.run())[1]
    names.append(FA2_ARM)
    functions[FA2_ARM] = fa2_function
    plans[FA2_ARM] = fa2_plan
    arm_inputs[FA2_ARM] = fa2_conversion

    progress.update({
        "phase": "prepared",
        "arm_order": names,
        "conversion": arm_inputs,
        "input_sha256": input_hash,
        "current_v_stride": list(segments[-1][1].stride()),
    })
    save()

    # All six arms must agree with the independent chunked FP32 oracle before
    # timing. Preparation/compilation is complete before graph capture.
    before = {}
    for name in names:
        before[name] = _check(functions[name](), oracle, f"{name} before timing")
        progress["before_timing_vs_fp32"] = dict(before)
        save()
    captures = {}
    graph_before = {}
    for name in names:
        captures[name] = _make_graph(functions[name], q.device)
        graph_before[name] = _check(captures[name][1], oracle, f"{name} graph before timing")
        progress["graph_before_timing_vs_fp32"] = dict(graph_before)
        save()
    progress["phase"] = "timing"
    timing = _measure_graphs(captures, q.device, progress, save)

    # Post-timing API and Graph validations catch stale outputs or overwritten
    # input buffers while keeping each arm's verification adjacent to its run.
    after = {}
    graph_after = {}
    for name in names:
        after[name] = _check(functions[name](), oracle, f"{name} after timing")
        captures[name][0].replay()
        graph_after[name] = _check(captures[name][1], oracle, f"{name} graph after timing")
        progress["after_timing_vs_fp32"] = dict(after)
        progress["graph_after_timing_vs_fp32"] = dict(graph_after)
        progress["phase"] = "post_timing_validation"
        save()

    if input_hash != _tensor_hash(*_flat_tensors(q, segments)):
        raise RuntimeError(f"performance inputs changed for T={tokens}")

    compile_evidence = {}
    compile_evidence[TRITON_ARM] = compile_artifacts(
        triton_plan.kernel, f"m15-triton-g4-s2-T{tokens}", "m34-diagnostic"
    )
    if compile_evidence[TRITON_ARM] is not None:
        compile_evidence[TRITON_ARM]["spills"] = triton_plan.kernel.n_spills

    resources = {}
    for config in CUDA_CONFIGS:
        name = f"{CUDA_ARM_PREFIX}_s{config['stages']}_p{config['p_mode']}"
        resources[name] = _cuda_resource_record(config)
    resources[TRITON_ARM] = _triton_resources(triton_plan)
    resources[FA2_ARM] = {
        "packed_kv_workspace_bytes": fa2_plan.packed_workspace_bytes,
        "owned_output_bytes": q.numel() * q.element_size(),
        "shared_memory_bytes": None,
    }
    progress["resources"] = resources
    progress["compile_artifacts"] = compile_evidence
    progress["workspace_scope"] = (
        "CUDA and Triton values are owned BF16 output bytes; FA2 is reusable packed K/V bytes "
        "and excludes its output; contiguous V staging is reported separately"
    )
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
        "conversion": arm_inputs,
        "before_timing_vs_fp32": before,
        "graph_before_timing_vs_fp32": graph_before,
        "timing": timing,
        "after_timing_vs_fp32": after,
        "graph_after_timing_vs_fp32": graph_after,
        "resources": resources,
        "compile_artifacts": compile_evidence,
        "workspace_bytes": {
            **{name: plans[name].owners[-1].numel() * plans[name].owners[-1].element_size()
               for name in names if name.startswith(CUDA_ARM_PREFIX)},
            TRITON_ARM: triton_plan.workspace_bytes,
            FA2_ARM: fa2_plan.packed_workspace_bytes,
        },
        "workspace_scope": (
            "CUDA and Triton values are owned BF16 output bytes; FA2 is reusable packed K/V bytes "
            "and excludes its output; contiguous V staging is reported separately"
        ),
        "shared_memory_differences_vs_s1_p0": {
            name: resources[name].get("shared_memory_bytes")
            - resources["cuda_s1_p0"].get("shared_memory_bytes")
            for name in resources
            if name.startswith(CUDA_ARM_PREFIX)
        },
        "status": "passed",
        "passed": True,
    }


def _summary_report(record: dict) -> str:
    lines = [
        "# M34 CUDA GQA diagnostic",
        "",
        f"Status: **{record.get('status', 'unknown')}**",
        "",
        "## Scope and protocol",
        "",
        "Fixed four-segment GQA inputs use Q[T,32,128], four K/V[L,8,128] pairs, "
        "history visible to every query, and current keys visible only when j <= query_i. "
        "CUDA variants are stages 1/2 crossed with shared/direct-register P modes. "
        "Every correctness row uses the independent chunked FP32 oracle and the 0.008 "
        "relative RMS / normalized max limits.",
        "",
        "The benchmark uses T=32/96/256 with history lengths (2048,64,512), padded "
        "current-V token stride 6144, and six arms: four CUDA variants, M15 Triton "
        "G=4 / token tile 32 / key tile 64 / 4 warps / 2 stages, and reusable pack+FA2. "
        "Each graph replay contains four forwards. Timing order is forward/reverse/forward "
        "with three samples per arm per round; compilation and preparation are excluded. "
        "Legacy arms copy padded V into preallocated contiguous staging on every invocation, "
        "so that conversion is included in their measured path.",
        "",
        "## Numerical cases",
        "",
        "| Case | T | Segment lengths | Config | Output mode | Padded current V | RMS | Normalized max | Finite | Status |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in record.get("numerical_cases", []):
        metric = row.get("vs_chunked_fp32", {})
        config = row.get("config", {})
        lines.append(
            f"| {row.get('case')} | {row.get('T')} | {row.get('segment_lengths')} | "
            f"S{config.get('stages')} / P{config.get('p_mode')} ({config.get('p_mode_name')}) | "
            f"{row.get('out_mode')} | "
            f"{row.get('current_v_padded')} | {metric.get('relative_rms')} | "
            f"{metric.get('normalized_max')} | {metric.get('finite')} | {row.get('status')} |"
        )
    lines += [
        "",
        "## Graph timing",
        "",
        "Milliseconds per complete forward; samples are retained in the fixed JSON result.",
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
        "## CUDA resources",
        "",
        "The native CTA is fixed at M=64, key tile 32, and four warps. The shared-memory "
        "budget is `(64*128 + 2*32*128*stages + 64*32) * 2` bytes. P mode does not "
        "change the retained scratch allocation.",
        "",
        "| Stages | P mode | Shared memory (B) | Delta vs S1/P0 (B) | Registers/thread | Spill bytes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    resource_rows = record.get("cuda_resource_table", [])
    for row in resource_rows:
        lines.append(
            f"| {row.get('stages')} | P{row.get('p_mode')} ({row.get('p_mode_name')}) | "
            f"{row.get('shared_memory_bytes')} | "
            f"{row.get('shared_memory_delta_vs_s1_p0_bytes')} | "
            f"{row.get('registers_per_thread')} | {row.get('spill_bytes')} |"
        )
    lines += [
        "",
        "Per-case arm resources and workspace sizes are also included in the JSON. Static "
        "shared-memory and compiler resource figures do not measure runtime occupancy or utilization.",
        "",
        "## Artifacts and provenance",
        "",
        f"Fixed diagnostic JSON: `{RESULT_PATH.relative_to(ROOT)}`.",
        f"Raw compiled artifacts: `{ARTIFACT_ROOT.relative_to(ROOT)}`.",
        "CUDA extension build and ptxas resource evidence: `artifacts/m34-build/`.",
        "Source hashes and runtime versions are recorded in the JSON.",
        "",
    ]
    if record.get("failure"):
        failure = record["failure"]
        lines += ["## Failure", "", f"`{failure.get('error', failure)}`", ""]
    return "\n".join(lines)


def _refresh_resource_deltas(record: dict) -> None:
    baseline = _shared_memory_budget(1)["shared_memory_bytes"]
    record["cuda_resource_table"] = []
    for config in CUDA_CONFIGS:
        row = _cuda_resource_record(config)
        row["shared_memory_delta_vs_s1_p0_bytes"] = row["shared_memory_bytes"] - baseline
        record["cuda_resource_table"].append(row)


def run() -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    source_hashes = _source_hashes()
    record = {
        "schema": 1,
        "run_id": "m34-diagnostic",
        "status": "running",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "limits": LIMITS,
        "configs": list(CUDA_CONFIGS),
        "cuda_resource_table": [],
        "protocol": {
            "numerical_cases": 8,
            "cuda_configurations": 4,
            "expected_numerical_results": 32,
            "performance_tokens": list(PERFORMANCE_TOKENS),
            "performance_history_lengths": list(PERFORMANCE_HISTORY),
            "arms": 6,
            "graph_calls_per_replay": GRAPH_CALLS,
            "round_order": list(MEASUREMENT_ORDERS),
            "samples_per_arm_per_round": SAMPLES_PER_ARM_PER_ROUND,
            "timed_unit": "complete forward, milliseconds",
        },
        "scope": "GPU-resident four-segment GQA attention only; no RoPE, model, H2D, routing, or serving",
        "source_hashes": source_hashes,
        "numerical_cases": [],
        "performance_cases": [],
        "telemetry_before": None,
    }
    _refresh_resource_deltas(record)

    def save() -> None:
        _write_json(RESULT_PATH, record)
        (ARTIFACT_ROOT / "run_manifest.json").write_text(
            json.dumps({
                "run_id": record["run_id"],
                "status": record["status"],
                "source_hashes": record["source_hashes"],
                "cuda_resource_table": record["cuda_resource_table"],
            }, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (ROOT / "docs/M34_CUDA_GQA_REPORT.md").write_text(_summary_report(record), encoding="utf-8")

    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the active project environment")
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
        if (gpu.major, gpu.minor) != (8, 6):
            raise RuntimeError(f"M34 is scoped to SM86, got SM{gpu.major}{gpu.minor}")
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
            raise RuntimeError("source files changed during M34 diagnostic")
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
        print(f"M34 result: {RESULT_PATH.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    run()
