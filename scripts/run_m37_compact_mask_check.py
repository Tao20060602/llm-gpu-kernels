#!/usr/bin/env python3
"""Bounded M37 compact-attention mask and five-arm GPU diagnostic."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_gpu_kernels.attention_cuda_compact import prepare_cuda_gqa_compact
from llm_gpu_kernels.attention_baseline_benchmark import _runtime_info as _fa2_runtime_info
from llm_gpu_kernels.attention_raw_boundary import RawPackAttention


RESULT_PATH = ROOT / "results/m37-compact-mask-diagnostic.json"
REPORT_PATH = ROOT / "docs/M37_COMPACT_MASK_REPORT.md"
ARTIFACT_ROOT = ROOT / "artifacts/m37-diagnostic"
V1_BINARY = ROOT / "artifacts/m37-control/llm_gpu_kernels_m36_compact_gqa_sm86_v1.so"
V1_MODULE_NAME = "llm_gpu_kernels_m36_compact_gqa_sm86_v1"
ROOT_DECISION_MARKER = "<!-- M37_ROOT_DECISION: preserved by the diagnostic runner -->"
SEED = 3701
LIMITS = {"relative_rms": 0.008, "normalized_max": 0.008}
NUMERIC_CASES = (
    ("T0_empty", 0, (0, 0, 0, 0)),
    ("T1_empty_history", 1, (0, 0, 0, 1)),
    ("T9_partial_history", 9, (17, 0, 3, 9)),
    ("T17_key_tail", 17, (33, 1, 0, 17)),
    ("T64_complete_query_and_history_tiles", 64, (64, 16, 32, 64)),
    ("T65_query_tail_and_key_tails", 65, (31, 32, 17, 65)),
    ("T96_complete_history_tiles_and_query_tail", 96, (128, 64, 32, 96)),
    ("T129_query_tail_empty_middle_and_key_tail", 129, (64, 0, 33, 129)),
)
PERFORMANCE_TOKENS = (32, 96, 256)
PERFORMANCE_HISTORY = (2048, 64, 512)
GRAPH_CALLS = 4
MEASUREMENT_ORDERS = ("forward", "reverse", "forward")
SAMPLES_PER_ARM_PER_ROUND = 3
NEW_PREFIX = "new_v2"
OLD_PREFIX = "old_v1"
FA2_ARM = "reusable_raw_pack_fa2"
ARM_NAMES = tuple(
    [f"{prefix}_s{stages}" for prefix in (NEW_PREFIX, OLD_PREFIX) for stages in (1, 2)]
    + [FA2_ARM]
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_v1_control():
    """Import the preserved v1 binary directly for measurement only."""
    if not V1_BINARY.is_file():
        raise FileNotFoundError(f"preserved M36 v1 control binary is missing: {V1_BINARY}")
    spec = importlib.util.spec_from_file_location(V1_MODULE_NAME, V1_BINARY)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load preserved M36 v1 control binary: {V1_BINARY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[V1_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _make_old_v1_run(module, q, segments, stages: int):
    output = torch.empty_like(q)
    keys = [key for key, _ in segments]
    values = [value for _, value in segments]
    # The historical six-argument native constructor retains its M36 ABI.
    native = module.CompactCudaGqaPlan(q, keys, values, output, stages, 1)

    def run():
        native.run()
        return output

    return run


def _random_tensor(shape: tuple[int, ...], generator, device: torch.device) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    )


def _padded_value(value: torch.Tensor, token_stride: int = 6144) -> torch.Tensor:
    if value.shape[0] == 0:
        return value
    padded = torch.empty_strided(
        tuple(value.shape),
        (token_stride, value.shape[2], 1),
        dtype=value.dtype,
        device=value.device,
    )
    padded.copy_(value)
    return padded


def _make_inputs(tokens: int, lengths: tuple[int, int, int, int], *, seed: int, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = _random_tensor((tokens, 32, 128), generator, device)
    segments = []
    for length in lengths:
        key = _random_tensor((length, 8, 128), generator, device)
        value = _random_tensor((length, 8, 128), generator, device)
        segments.append((key, value))
    return q, tuple(segments)


def _sdpa_reference(q, segments):
    """Independent FP32 attention through public PyTorch SDPA math."""
    if q.shape[0] == 0:
        return torch.empty_like(q, dtype=torch.float32)

    history = sum(key.shape[0] for key, _ in segments[:3])
    key = torch.cat(tuple(key.float() for key, _ in segments), dim=0)
    value = torch.cat(tuple(value.float() for _, value in segments), dim=0)
    key = key.repeat_interleave(4, dim=1).permute(1, 0, 2).unsqueeze(0)
    value = value.repeat_interleave(4, dim=1).permute(1, 0, 2).unsqueeze(0)
    query = q.float().permute(1, 0, 2).unsqueeze(0)

    query_positions = torch.arange(q.shape[0], device=q.device)
    current_positions = torch.arange(q.shape[0], device=q.device)
    allowed = torch.ones(
        (q.shape[0], key.shape[-2]), dtype=torch.bool, device=q.device
    )
    allowed[:, history:] = current_positions[None, :] <= query_positions[:, None]
    with sdpa_kernel(SDPBackend.MATH):
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(128.0),
        )
    return output.squeeze(0).permute(1, 0, 2)


def _metrics(actual, expected) -> dict:
    if tuple(actual.shape) != tuple(expected.shape):
        return {
            "shape_match": False,
            "relative_rms": None,
            "normalized_max": None,
            "max_abs": None,
            "finite": False,
            "passed": False,
        }
    if actual.numel() == 0:
        finite = bool(torch.isfinite(actual).all())
        return {
            "shape_match": True,
            "relative_rms": 0.0,
            "normalized_max": 0.0,
            "max_abs": 0.0,
            "finite": finite,
            "empty": True,
            "passed": finite,
        }

    actual_f = actual.float()
    expected_f = expected.float()
    delta = actual_f - expected_f
    relative_rms = (
        delta.square().mean().sqrt()
        / expected_f.square().mean().sqrt().clamp_min(1e-12)
    ).item()
    normalized_max = (
        delta.abs().max() / expected_f.abs().max().clamp_min(1e-12)
    ).item()
    finite = bool(torch.isfinite(actual_f).all() and torch.isfinite(expected_f).all())
    return {
        "shape_match": True,
        "relative_rms": relative_rms,
        "normalized_max": normalized_max,
        "max_abs": delta.abs().max().item(),
        "finite": finite,
        "passed": finite
        and relative_rms <= LIMITS["relative_rms"]
        and normalized_max <= LIMITS["normalized_max"],
    }


def _check(actual, expected, label: str) -> dict:
    result = _metrics(actual, expected)
    if not result["passed"]:
        raise RuntimeError(f"{label}: numerical check failed: {result}")
    return result


def _run_numerical_checks(record: dict, save, v1_module, device: torch.device) -> None:
    for case_index, (name, tokens, lengths) in enumerate(NUMERIC_CASES):
        q, segments = _make_inputs(tokens, lengths, seed=SEED + case_index, device=device)
        if tokens in (17, 96):
            segments = tuple(segments[:-1]) + (
                (segments[-1][0], _padded_value(segments[-1][1])),
            )
        expected = _sdpa_reference(q, segments)
        row = {
            "case": name,
            "T": tokens,
            "segment_lengths": list(lengths),
            "current_v_stride": list(segments[-1][1].stride()),
            "stages": {},
            "status": "running",
        }
        record["numerical_cases"].append(row)
        save()

        for stages in (1, 2):
            new_output = torch.empty_like(q) if (case_index + stages) % 2 else None
            new_plan = prepare_cuda_gqa_compact(
                q, segments, out=new_output, stages=stages
            )
            old_run = _make_old_v1_run(v1_module, q, segments, stages)
            new_actual = new_plan.run()
            old_actual = old_run()
            torch.cuda.synchronize(device)
            new_metrics = _check(new_actual, expected, f"{name} new v2 S{stages}")
            old_metrics = _check(old_actual, expected, f"{name} old v1 S{stages}")
            bitwise = bool(torch.equal(new_actual, old_actual))
            stage_row = {
                "new_v2_vs_fp32_sdpa_math": new_metrics,
                "old_v1_vs_fp32_sdpa_math": old_metrics,
                "bitwise_equal_to_explicit_old_v1": bitwise,
            }
            row["stages"][str(stages)] = stage_row
            save()
            if not bitwise:
                raise RuntimeError(
                    f"{name} stage {stages}: new compact output differs from explicit v1 control"
                )
        row["status"] = "passed"
        row["passed"] = True
        save()


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


def _measure_graphs(captures: dict[str, tuple], device: torch.device) -> dict:
    names = list(captures)
    raw = {name: [] for name in names}
    rounds = []
    for direction in MEASUREMENT_ORDERS:
        order = names if direction == "forward" else list(reversed(names))
        round_row = {"direction": direction, "order": order, "samples_ms_per_forward": {}}
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
            round_row["samples_ms_per_forward"][name] = samples
        rounds.append(round_row)

    summary = {
        name: {
            "event_p50_ms_per_forward": statistics.median(samples),
            "event_p95_ms_per_forward": float(np.percentile(samples, 95)),
        }
        for name, samples in raw.items()
    }
    return {
        "rounds": rounds,
        "summary": summary,
        "round_order": list(MEASUREMENT_ORDERS),
        "samples_per_arm_per_round": SAMPLES_PER_ARM_PER_ROUND,
        "calls_per_graph_replay": GRAPH_CALLS,
        "units": "milliseconds per complete attention forward",
    }


def _performance_case(tokens: int, record: dict, save, v1_module, device) -> dict:
    row = {"T": tokens, "status": "running"}
    record["performance_cases"].append(row)
    save()

    lengths = (*PERFORMANCE_HISTORY, tokens)
    q, segments = _make_inputs(tokens, lengths, seed=SEED + 1000, device=device)
    segments = tuple(segments[:-1]) + (
        (segments[-1][0], _padded_value(segments[-1][1])),
    )
    expected_before = _sdpa_reference(q, segments)

    functions = {}
    new_plans = {}
    old_plans = {}
    for stages in (1, 2):
        new_name = f"{NEW_PREFIX}_s{stages}"
        new_plans[new_name] = prepare_cuda_gqa_compact(q, segments, stages=stages)
        functions[new_name] = new_plans[new_name].run
    for stages in (1, 2):
        old_name = f"{OLD_PREFIX}_s{stages}"
        old_plans[old_name] = _make_old_v1_run(v1_module, q, segments, stages)
        functions[old_name] = old_plans[old_name]

    fa2_plan = RawPackAttention(q, segments, reusable=True)
    functions[FA2_ARM] = fa2_plan.run

    before_outputs = {}
    before_metrics = {}
    for name, function in functions.items():
        before_outputs[name] = function().clone()
        before_metrics[name] = _check(
            before_outputs[name], expected_before, f"T{tokens} {name} before timing"
        )
    torch.cuda.synchronize(device)

    captures = {name: _make_graph(function, device) for name, function in functions.items()}
    graph_outputs_before = {name: output.clone() for name, (_, output) in captures.items()}
    graph_metrics_before = {
        name: _check(output, expected_before, f"T{tokens} {name} graph before timing")
        for name, output in graph_outputs_before.items()
    }
    bitwise_before = {
        str(stages): bool(
            torch.equal(
                before_outputs[f"{NEW_PREFIX}_s{stages}"],
                before_outputs[f"{OLD_PREFIX}_s{stages}"],
            )
        )
        for stages in (1, 2)
    }
    graph_bitwise_before = {
        str(stages): bool(
            torch.equal(
                graph_outputs_before[f"{NEW_PREFIX}_s{stages}"],
                graph_outputs_before[f"{OLD_PREFIX}_s{stages}"],
            )
        )
        for stages in (1, 2)
    }

    timing = _measure_graphs(captures, device)
    expected_after = _sdpa_reference(q, segments)
    after_outputs = {}
    after_metrics = {}
    graph_outputs_after = {}
    graph_metrics_after = {}
    for name, function in functions.items():
        after_outputs[name] = function().clone()
        after_metrics[name] = _check(
            after_outputs[name], expected_after, f"T{tokens} {name} after timing"
        )
        graph = captures[name][0]
        graph.replay()
        graph_outputs_after[name] = captures[name][1].clone()
        graph_metrics_after[name] = _check(
            graph_outputs_after[name], expected_after, f"T{tokens} {name} graph after timing"
        )
    torch.cuda.synchronize(device)

    bitwise_after = {
        str(stages): bool(
            torch.equal(
                after_outputs[f"{NEW_PREFIX}_s{stages}"],
                after_outputs[f"{OLD_PREFIX}_s{stages}"],
            )
        )
        for stages in (1, 2)
    }
    graph_bitwise_after = {
        str(stages): bool(
            torch.equal(
                graph_outputs_after[f"{NEW_PREFIX}_s{stages}"],
                graph_outputs_after[f"{OLD_PREFIX}_s{stages}"],
            )
        )
        for stages in (1, 2)
    }

    row.update({
        "history_lengths": list(PERFORMANCE_HISTORY),
        "segment_lengths": list(lengths),
        "current_v_stride": list(segments[-1][1].stride()),
        "arm_order": list(functions),
        "before_timing_vs_fp32_sdpa_math": before_metrics,
        "graph_before_timing_vs_fp32_sdpa_math": graph_metrics_before,
        "timing": timing,
        "after_timing_vs_fp32_sdpa_math": after_metrics,
        "graph_after_timing_vs_fp32_sdpa_math": graph_metrics_after,
        "bitwise_equal_to_explicit_old_v1_before_timing": bitwise_before,
        "graph_bitwise_equal_to_explicit_old_v1_before_timing": graph_bitwise_before,
        "bitwise_equal_to_explicit_old_v1_after_timing": bitwise_after,
        "graph_bitwise_equal_to_explicit_old_v1_after_timing": graph_bitwise_after,
        "fa2_packed_kv_workspace_bytes": fa2_plan.packed_workspace_bytes,
        "workspace_bytes": {
            **{
                name: q.numel() * q.element_size()
                for name in (*new_plans, *old_plans)
            },
            FA2_ARM: fa2_plan.packed_workspace_bytes,
        },
    })
    # Preserve every case's full raw samples in the primary result JSON.
    save()
    for check_group in (
        before_metrics,
        graph_metrics_before,
        after_metrics,
        graph_metrics_after,
    ):
        if not all(metric["passed"] for metric in check_group.values()):
            raise RuntimeError(f"T{tokens}: post-timing numerical verification failed")
    if not all((*bitwise_before.values(), *graph_bitwise_before.values(),
                *bitwise_after.values(), *graph_bitwise_after.values())):
        raise RuntimeError(f"T{tokens}: new compact stage differs bitwise from explicit v1 control")
    row["status"] = "passed"
    row["passed"] = True
    save()
    return row


def _runtime(device: torch.device) -> dict:
    gpu = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_path": torch.__file__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "flash_attn": _fa2_runtime_info(),
        "gpu": gpu.name,
        "sm": [gpu.major, gpu.minor],
        "device_index": device.index,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }


def _summary_report(record: dict) -> str:
    lines = [
        "# M37 compact full-history mask diagnostic",
        "",
        f"Status: **{record.get('status', 'unknown')}**",
        "",
        "## Scope and protocol",
        "",
        "The public v2 `prepare_cuda_gqa_compact` entry is compared with the explicitly loaded, "
        "preserved M36 v1 native binary and reusable raw-pack plus public FA2. The v1 binary is "
        "a measurement control only. Both CUDA stages are compared against public PyTorch SDPA "
        "math in FP32; the v2 and v1 compact outputs must also match bitwise for each stage.",
        "",
        "Eight predeclared boundary shapes run at both stages. The timing cases use T=32/96/256, "
        "history lengths (2048,64,512), and padded current-V token stride 6144. Each of five "
        "arms is captured with four forwards per graph replay, then measured in forward/reverse/forward "
        "order with three samples per arm per round. Preparation is outside timing. Every arm "
        "is checked against an independent FP32 oracle before and after timing, including graph output. "
        "Inputs are seeded synthetic BF16 tensors generated by PyTorch, not captured model activations.",
        "",
        "## Numerical cases",
        "",
        "| Case | T | Segment lengths | S1 RMS / max norm | S2 RMS / max norm | v1 bitwise S1 / S2 |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in record.get("numerical_cases", []):
        s1 = row.get("stages", {}).get("1", {})
        s2 = row.get("stages", {}).get("2", {})
        m1 = s1.get("new_v2_vs_fp32_sdpa_math", {})
        m2 = s2.get("new_v2_vs_fp32_sdpa_math", {})
        lines.append(
            f"| {row.get('case')} | {row.get('T')} | {row.get('segment_lengths')} | "
            f"{m1.get('relative_rms')} / {m1.get('normalized_max')} | "
            f"{m2.get('relative_rms')} / {m2.get('normalized_max')} | "
            f"{s1.get('bitwise_equal_to_explicit_old_v1')} / "
            f"{s2.get('bitwise_equal_to_explicit_old_v1')} |"
        )
    lines += [
        "",
        "## Graph timing p50 / p95",
        "",
        "| T | New S1 | New S2 | Old v1 S1 | Old v1 S2 | Reusable pack + FA2 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in record.get("performance_cases", []):
        summary = row.get("timing", {}).get("summary", {})
        values = []
        for name in ARM_NAMES:
            item = summary.get(name, {})
            p50, p95 = item.get("event_p50_ms_per_forward"), item.get("event_p95_ms_per_forward")
            values.append("n/a" if p50 is None else f"{p50:.6f} / {p95:.6f}")
        lines.append(f"| {row.get('T')} | " + " | ".join(values) + " |")
    lines += [
        "",
        "The complete event samples, per-arm before/after oracle errors, and bitwise controls are "
        "in the diagnostic JSON. Results are a fixed GPU operator diagnostic, not a serving or "
        "model-level performance claim.",
        "",
        f"Diagnostic JSON: `{RESULT_PATH.relative_to(ROOT)}`.",
        f"Raw run manifest: `{(ARTIFACT_ROOT / 'run_manifest.json').relative_to(ROOT)}`.",
        f"Preserved control binary: `{V1_BINARY.relative_to(ROOT)}`.",
        "",
    ]
    if record.get("failure"):
        lines += ["## Failure", "", f"`{record['failure'].get('error')}`", ""]
    return "\n".join(lines)


def _report_with_root_decision(record: dict) -> str:
    generated = _summary_report(record).rstrip()
    if REPORT_PATH.is_file():
        existing = REPORT_PATH.read_text(encoding="utf-8")
        marker_index = existing.find(ROOT_DECISION_MARKER)
        root_section = (
            existing[marker_index:].rstrip()
            if marker_index >= 0
            else ROOT_DECISION_MARKER
            + "\n\n## Root decision\n\n_To be completed after review of the GPU diagnostic._"
        )
    else:
        root_section = (
            ROOT_DECISION_MARKER
            + "\n\n## Root decision\n\n_To be completed after review of the GPU diagnostic._"
        )
    return f"{generated}\n\n{root_section}\n"


def run() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": 1,
        "run_id": "m37-compact-mask-diagnostic",
        "status": "running",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "limits": LIMITS,
        "protocol": {
            "numerical_cases": len(NUMERIC_CASES),
            "stages": [1, 2],
            "expected_numerical_results": len(NUMERIC_CASES) * 2,
            "performance_tokens": list(PERFORMANCE_TOKENS),
            "performance_history_lengths": list(PERFORMANCE_HISTORY),
            "arms": list(ARM_NAMES),
            "graph_calls_per_replay": GRAPH_CALLS,
            "round_order": list(MEASUREMENT_ORDERS),
            "samples_per_arm_per_round": SAMPLES_PER_ARM_PER_ROUND,
            "timed_unit": "milliseconds per complete attention forward",
            "reference": "FP32 torch.nn.functional.scaled_dot_product_attention with SDPBackend.MATH and explicit current-key mask",
            "historical_control": "explicit v1 extension import and six-argument native constructor; never used by the product loader",
        },
        "numerical_cases": [],
        "performance_cases": [],
    }

    def save() -> None:
        _write_json(RESULT_PATH, record)
        _write_json(
            ARTIFACT_ROOT / "raw-timing-samples.json",
            {
                "run_id": record["run_id"],
                "performance_cases": [
                    {
                        "T": row.get("T"),
                        "arm_order": row.get("arm_order", []),
                        "timing": row.get("timing"),
                    }
                    for row in record["performance_cases"]
                ],
            },
        )
        _write_json(
            ARTIFACT_ROOT / "run_manifest.json",
            {
                "run_id": record["run_id"],
                "status": record["status"],
                "protocol": record["protocol"],
                "numerical_case_count": len(record["numerical_cases"]),
                "performance_case_count": len(record["performance_cases"]),
            },
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(_report_with_root_decision(record), encoding="utf-8")

    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the active project environment")
        device = torch.device("cuda", torch.cuda.current_device())
        capability = torch.cuda.get_device_capability(device)
        if capability != (8, 6):
            raise RuntimeError(f"M37 is scoped to SM86, got SM{capability[0]}{capability[1]}")
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        record["runtime"] = _runtime(device)
        v1_module = _load_v1_control()
        save()

        _run_numerical_checks(record, save, v1_module, device)
        for tokens in PERFORMANCE_TOKENS:
            _performance_case(tokens, record, save, v1_module, device)

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
        raise
    finally:
        save()
        print(f"M37 result: {RESULT_PATH.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    run()
