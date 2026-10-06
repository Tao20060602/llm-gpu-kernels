"""M8 GPU-resident segmented-KV GQA cost baseline; official FA2 only."""
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback

import numpy as np
import torch

from .attention_baseline import prepare_segmented_attention, segmented_attention_reference


ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_PATH = ROOT / "results/provenance/flash-attention-2.8.3.post1-runtime.json"
LIMITS = {"relative_rms": 0.008, "normalized_max": 0.008}
SEED = 819
TOKENS = (64, 256, 1024, 4096)
SELECTED = (0, 512, 2048)
SINK = 64
RECENT = 512
API_ROUNDS = 4
API_SAMPLES = 8
API_CALLS = 2
GRAPH_FORWARDS = 8


def _command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as exc:
        return {"returncode": None, "error": f"{type(exc).__name__}: {exc}"}


def _source_hashes():
    files = (
        ROOT / "docs/M8_GQA_PREFILL_BASELINE_PLAN.md",
        ROOT / "src/llm_gpu_kernels/attention_baseline.py",
        ROOT / "src/llm_gpu_kernels/attention_baseline_benchmark.py",
    )
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _tensor_hash(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _error(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"output shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
    if actual.numel() == 0:
        finite = bool(torch.isfinite(actual).all())
        return {"relative_rms": 0.0, "normalized_max": 0.0, "max_abs": 0.0,
                "finite": finite, "empty": True, "passed": finite}
    actual_f, expected_f = actual.float(), expected.float()
    delta = actual_f - expected_f
    metrics = {
        "relative_rms": (delta.square().mean().sqrt() / expected_f.square().mean().sqrt().clamp_min(1e-12)).item(),
        "normalized_max": (delta.abs().max() / expected_f.abs().max().clamp_min(1e-12)).item(),
        "max_abs": delta.abs().max().item(),
        "finite": bool(torch.isfinite(actual_f).all()),
    }
    metrics["passed"] = metrics["finite"] and all(metrics[key] <= value for key, value in LIMITS.items())
    return metrics


def _flat_tensors(q, segments):
    return (q, *(tensor for pair in segments for tensor in pair))


def _make_inputs(tokens, selected, *, seed=SEED, device="cuda"):
    device = torch.device(device)
    def random(shape, offset):
        generator = torch.Generator(device="cpu").manual_seed(seed + tokens * 17 + offset)
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)

    q = random((tokens, 32, 128), 0)
    lengths = (selected, SINK, RECENT, tokens)
    segments = tuple(
        (random((length, 8, 128), 100 + index * 31), random((length, 8, 128), 101 + index * 31))
        for index, length in enumerate(lengths)
    )
    return q, segments


def _make_custom_inputs(tokens, lengths, *, seed=SEED, extreme=False, head_map=False, device="cuda"):
    device = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(seed + tokens * 17 + sum(lengths))
    q = torch.randn((tokens, 32, 128), generator=generator).to(device=device, dtype=torch.bfloat16)
    segments = []
    for index, length in enumerate(lengths):
        k = torch.randn((length, 8, 128), generator=generator).to(device=device, dtype=torch.bfloat16)
        v = torch.randn((length, 8, 128), generator=generator).to(device=device, dtype=torch.bfloat16)
        segments.append((k, v))
    if extreme and tokens:
        q.mul_(64)
        for k, _ in segments:
            k.mul_(64)
    if head_map:
        q.zero_()
        for _, v in segments:
            values = torch.arange(8, device=device, dtype=torch.float32)[None, :, None] * 8.0
            values = values + torch.arange(128, device=device, dtype=torch.float32)[None, None, :] * 0.125
            v.copy_(values.expand(v.shape[0], -1, -1).to(torch.bfloat16))
        if tokens:
            for k, _ in segments:
                k.zero_()
    return q, tuple(segments)


def _telemetry():
    return _command([
        "nvidia-smi", "--query-gpu=driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used",
        "--format=csv,noheader",
    ])


def _runtime_info():
    provenance = json.loads(PROVENANCE_PATH.read_text())
    import flash_attn
    expected_root = Path(provenance["runtime_path"]).resolve()
    module_path = Path(flash_attn.__file__).resolve()
    if not module_path.is_relative_to(expected_root):
        raise RuntimeError(f"FA2 imported from {module_path}, outside pinned overlay {expected_root}")
    actual_version = importlib.metadata.version("flash_attn")
    if actual_version != provenance["version"]:
        raise RuntimeError(f"expected FA2 {provenance['version']}, got {actual_version}")
    requested_root = os.environ.get("FLASH_ATTN_OPERATOR_RUNTIME")
    if requested_root and Path(requested_root).resolve() != expected_root:
        raise RuntimeError("FLASH_ATTN_OPERATOR_RUNTIME differs from the pinned provenance path")
    if not provenance.get("all_upstream_package_files_match_wheel") or provenance.get("official_files_verified_count") != 86:
        raise RuntimeError("pinned FA2 provenance does not report the expected official wheel verification")
    return {
        "provenance": provenance,
        "provenance_sha256": hashlib.sha256(PROVENANCE_PATH.read_bytes()).hexdigest(),
        "imported_module_path": str(module_path),
        "actual_distribution_version": actual_version,
        "runtime_path": str(expected_root),
    }


def _graph(function, device, repeats=GRAPH_FORWARDS):
    current = torch.cuda.current_stream(device)
    warm_stream = torch.cuda.Stream(device=device)
    warm_stream.wait_stream(current)
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            function()
    current.wait_stream(warm_stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=warm_stream):
        for _ in range(repeats):
            output = function()
    current.wait_stream(warm_stream)
    graph.replay()
    torch.cuda.synchronize(device)
    return graph, output


def _measure(functions, *, divisor=1):
    names = list(functions)
    orders = [names, list(reversed(names)), list(reversed(names)), names]
    for function in functions.values():
        for _ in range(5):
            function()
    torch.cuda.synchronize()
    event_samples = {name: [] for name in names}
    wall_samples = {name: [] for name in names}
    rounds = []
    for order in orders:
        row = {"order": order, "samples_ms_per_forward": {}, "wall_ms_per_forward": {}}
        for name in order:
            values, walls = [], []
            for _ in range(API_SAMPLES):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                began = time.perf_counter()
                start.record()
                for _ in range(API_CALLS):
                    functions[name]()
                end.record()
                end.synchronize()
                walls.append((time.perf_counter() - began) * 1000 / API_CALLS / divisor)
                values.append(start.elapsed_time(end) / API_CALLS / divisor)
            event_samples[name].extend(values)
            wall_samples[name].extend(walls)
            row["samples_ms_per_forward"][name] = values
            row["wall_ms_per_forward"][name] = walls
        rounds.append(row)
    summary = {
        name: {
            "event_p50_ms_per_forward": statistics.median(values),
            "event_p95_ms_per_forward": float(np.percentile(values, 95)),
            "wall_p50_ms_per_forward": statistics.median(wall_samples[name]),
            "wall_p95_ms_per_forward": float(np.percentile(wall_samples[name], 95)),
        }
        for name, values in event_samples.items()
    }
    return {
        "rounds": rounds,
        "summary": summary,
        "rounds_count": API_ROUNDS,
        "samples_per_round": API_SAMPLES,
        "calls_per_sample": API_CALLS,
        "forwards_per_graph": divisor if divisor > 1 else None,
        "wall_scope": "includes Python dispatch, event creation, and end synchronization; not isolated host launch latency",
    }


def _validate_small_case(name, q, segments, *, head_map=False):
    plan = prepare_segmented_attention(q, segments)
    full = plan.run()
    prepacked = plan.attend_prepacked()
    oracle = segmented_attention_reference(q, segments)
    full_error = _error(full, oracle)
    prepacked_error = _error(prepacked, oracle)
    pair_equal = bool(torch.equal(full, prepacked))
    mapping_error = None
    if head_map and q.shape[0]:
        all_v = torch.cat(tuple(v.float() for _, v in segments), dim=0)
        expected = all_v[0, torch.arange(32, device=q.device) // 4]
        mapping_error = (full.float() - expected.unsqueeze(0)).abs().max().item()
        mapping_passed = mapping_error <= 0.008
    else:
        mapping_passed = True
    return {
        "name": name,
        "query_length": q.shape[0],
        "segment_lengths": [k.shape[0] for k, _ in segments],
        "full_chain_vs_fp32": full_error,
        "prepacked_fa2_vs_fp32": prepacked_error,
        "full_chain_equals_prepacked": pair_equal,
        "gqa_mapping_max_abs": mapping_error,
        "gqa_mapping_passed": mapping_passed,
        "passed": full_error["passed"] and prepacked_error["passed"] and pair_equal and mapping_passed,
    }


def _graph_update_validation():
    tokens = 9
    q, segments = _make_custom_inputs(tokens, (3, 4, 5, tokens), seed=SEED + 4000)
    plan = prepare_segmented_attention(q, segments)
    original_hash = _tensor_hash(*_flat_tensors(q, segments))
    original_copies = tuple(tensor.clone() for tensor in _flat_tensors(q, segments))
    full_graph, full_graph_output = _graph(plan.run, q.device)
    plan.pack()
    prepacked_graph, prepacked_graph_output = _graph(plan.attend_prepacked, q.device)
    updated_generator = torch.Generator(device="cpu").manual_seed(SEED + 4999)
    updated = tuple(
        torch.randn(tensor.shape, generator=updated_generator, dtype=torch.float32).to(device=q.device, dtype=torch.bfloat16)
        for tensor in _flat_tensors(q, segments)
    )
    update_stream = torch.cuda.Stream(device=q.device)
    current = torch.cuda.current_stream(q.device)
    update_stream.wait_stream(current)
    with torch.cuda.stream(update_stream):
        for tensor, value in zip(_flat_tensors(q, segments), updated):
            tensor.copy_(value)
    current.wait_stream(update_stream)
    full_graph.replay()
    prepacked_graph.replay()
    torch.cuda.synchronize(q.device)
    updated_hash = _tensor_hash(*_flat_tensors(q, segments))
    updated_oracle = segmented_attention_reference(q, segments)
    updated_errors = {
        "full_chain": _error(full_graph_output, updated_oracle),
        "prepacked_fa2": _error(prepacked_graph_output, updated_oracle),
    }
    restore_stream = torch.cuda.Stream(device=q.device)
    current = torch.cuda.current_stream(q.device)
    restore_stream.wait_stream(current)
    with torch.cuda.stream(restore_stream):
        for tensor, value in zip(_flat_tensors(q, segments), original_copies):
            tensor.copy_(value)
    current.wait_stream(restore_stream)
    full_graph.replay()
    prepacked_graph.replay()
    torch.cuda.synchronize(q.device)
    restored_hash = _tensor_hash(*_flat_tensors(q, segments))
    restored_oracle = segmented_attention_reference(q, segments)
    restored_errors = {
        "full_chain": _error(full_graph_output, restored_oracle),
        "prepacked_fa2": _error(prepacked_graph_output, restored_oracle),
    }
    passed = (
        all(item["passed"] for item in updated_errors.values())
        and all(item["passed"] for item in restored_errors.values())
        and original_hash == restored_hash
        and updated_hash != original_hash
        and torch.equal(full_graph_output, prepacked_graph_output)
    )
    return {
        "name": "fixed_address_graph_update_nondefault_stream",
        "query_length": tokens,
        "segment_lengths": [k.shape[0] for k, _ in segments],
        "graph_forwards_per_replay": GRAPH_FORWARDS,
        "updated_input_hash": updated_hash,
        "restored_input_hash": restored_hash,
        "original_input_hash": original_hash,
        "updated_output_errors": updated_errors,
        "restored_output_errors": restored_errors,
        "passed": passed,
    }


def _functional_validation():
    cases = []
    q, segments = _make_custom_inputs(7, (0, 0, 0, 7), seed=SEED + 1)
    cases.append(_validate_small_case("empty_history_and_current_tail", q, segments))
    q, segments = _make_custom_inputs(17, (5, 4, 7, 17), seed=SEED + 2)
    cases.append(_validate_small_case("non_aligned_segment_tails", q, segments))
    q, segments = _make_custom_inputs(5, (2, 3, 4, 5), seed=SEED + 3, extreme=True)
    cases.append(_validate_small_case("extreme_logits", q, segments))
    q, segments = _make_custom_inputs(6, (3, 2, 5, 6), seed=SEED + 4, head_map=True)
    cases.append(_validate_small_case("gqa_head_mapping", q, segments, head_map=True))
    q, segments = _make_custom_inputs(0, (3, 4, 5, 0), seed=SEED + 5)
    cases.append(_validate_small_case("zero_query_length_skips_attention", q, segments))
    q, segments = _make_custom_inputs(65, (33, 17, 41, 65), seed=SEED + 6)
    cases.append(_validate_small_case("query_tile_tail_65_and_history_91", q, segments))
    graph_update = _graph_update_validation()
    return {"cases": cases, "graph_update": graph_update,
            "passed": all(case["passed"] for case in cases) and graph_update["passed"]}


def _run_case(tokens, selected):
    telemetry_before = _telemetry()
    q, segments = _make_inputs(tokens, selected)
    inputs_before = _tensor_hash(*_flat_tensors(q, segments))
    plan = prepare_segmented_attention(q, segments)
    full = plan.run()
    prepacked = plan.attend_prepacked()
    exact_initial_match = bool(torch.equal(full, prepacked))
    del full, prepacked

    full_graph, full_graph_output = _graph(plan.run, q.device)
    plan.pack()
    prepacked_graph, prepacked_graph_output = _graph(plan.attend_prepacked, q.device)
    graph_match = bool(torch.equal(full_graph_output, prepacked_graph_output))
    if not exact_initial_match or not graph_match:
        raise RuntimeError("full-chain and prepacked FA2 outputs differ on fixed inputs")

    api = _measure({"full_chain_pack_plus_fa2": plan.run, "prepacked_fa2": plan.attend_prepacked})
    graph_api = _measure({"full_chain_pack_plus_fa2_graph": full_graph.replay,
                          "prepacked_fa2_graph": prepacked_graph.replay}, divisor=GRAPH_FORWARDS)
    plan.pack()
    full_graph.replay()
    prepacked_graph.replay()
    torch.cuda.synchronize(q.device)
    post_timing_hash = _tensor_hash(*_flat_tensors(q, segments))
    post_timing_match = bool(torch.equal(full_graph_output, prepacked_graph_output))
    if post_timing_hash != inputs_before or not post_timing_match:
        raise RuntimeError("timing changed inputs or full/prepacked graph outputs")

    pack_api = _measure({"reusable_buffer_pack_only": plan.pack})
    peak_allocated = torch.cuda.max_memory_allocated(q.device)
    peak_reserved = torch.cuda.max_memory_reserved(q.device)
    result = {
        "T": tokens,
        "selected_history": selected,
        "sink": SINK,
        "previous_recent": RECENT,
        "current": tokens,
        "segment_lengths": [k.shape[0] for k, _ in segments],
        "packed_kv_length": sum(k.shape[0] for k, _ in segments),
        "seed": SEED,
        "input_sha256_before": inputs_before,
        "input_sha256_after": post_timing_hash,
        "full_chain_matches_prepacked_fa2": exact_initial_match,
        "graph_chain_matches_prepacked_fa2": graph_match and post_timing_match,
        "api_timing": api,
        "graph_timing": graph_api,
        "pack_only_diagnostic": pack_api,
        "pack_workspace_bytes": plan.packed_workspace_bytes,
        "telemetry_before": telemetry_before,
        "telemetry_after": _telemetry(),
        "memory": {
            "allocated_after_prepare_and_graphs_bytes": torch.cuda.memory_allocated(q.device),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "scope": "process allocator totals include inputs, outputs, graph pools and retained runtime state; not operator-only workspace",
        },
        "passed": exact_initial_match and graph_match and post_timing_hash == inputs_before and post_timing_match,
    }
    del full_graph, prepacked_graph, full_graph_output, prepacked_graph_output, plan, q, segments
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the M8 attention baseline")
    if not torch.__version__.startswith("2.7.1+cu128"):
        raise RuntimeError(f"M8 expects torch 2.7.1+cu128, got {torch.__version__}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"M8 expects CUDA runtime 12.8, got {torch.version.cuda}")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    runtime = _runtime_info()
    device = torch.cuda.current_device()
    gpu = torch.cuda.get_device_properties(device)
    if (gpu.major, gpu.minor) != (8, 6):
        raise RuntimeError(f"M8 was scoped to SM86, got SM{gpu.major}{gpu.minor}")
    run_id = "m8-gqa-baseline-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(time.time_ns() % 10**9)
    result_path = ROOT / "results" / f"{run_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": 1,
        "run_id": run_id,
        "status": "running",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "GPU-resident four-segment BF16 GQA attention cost baseline; no NanoKV, model, router, H2D, or custom attention kernel",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": gpu.name,
            "sm": [gpu.major, gpu.minor],
            "device_index": device,
            "fa2": runtime,
        },
        "source_hashes": _source_hashes(),
        "provenance_sha256": runtime["provenance_sha256"],
        "limits": LIMITS,
        "protocol": {
            "Q": "[T,32,128] BF16 contiguous",
            "segments": ["selected history", "sink", "previous recent", "current"],
            "K_V": "each [L,8,128] BF16 contiguous; current length equals T",
            "scale": 1.0 / math.sqrt(128),
            "mask": "all first-three-segment history visible; current key j visible iff j <= query i; FA2 public causal bottom-right alignment",
            "api_measurement": {"balanced_orders": ["AB", "BA", "BA", "AB"], "rounds": API_ROUNDS,
                                 "samples_per_round": API_SAMPLES, "calls_per_sample": API_CALLS},
            "graph_measurement": {"forwards_per_graph": GRAPH_FORWARDS, "rounds": API_ROUNDS,
                                  "samples_per_round": API_SAMPLES, "replay_calls_per_sample": API_CALLS},
            "timed_arms": {
                "full_chain_pack_plus_fa2": "torch.cat into reusable packed K/V buffers, then official public flash_attn_func",
                "prepacked_fa2": "official public flash_attn_func only, after identical K/V were packed outside timing",
                "pack_only_diagnostic": "reusable-buffer torch.cat only; separately reported and never subtracted from full-chain time",
            },
            "graph_update": "small T=9 fixed-address graph updates Q and every K/V on a non-default stream, checks oracle output, then restores and hashes all inputs",
            "no_output_allocation_claim": "FA2 public API allocates its output; eager API timing includes that allocation behavior, while graph pool behavior is reported separately",
        },
        "seed": SEED,
        "telemetry_before": _telemetry(),
        "functional_validation": [],
        "graph_update_validation": None,
        "cases": [],
    }
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        functional = _functional_validation()
        record["functional_validation"] = functional["cases"]
        record["graph_update_validation"] = functional["graph_update"]
        if not functional["passed"]:
            raise RuntimeError("M8 functional attention validation failed")
        for tokens in TOKENS:
            for selected in SELECTED:
                print(f"M8 GQA baseline T={tokens} selected={selected}", flush=True)
                torch.cuda.reset_peak_memory_stats(device)
                entry = {"T": tokens, "selected_history": selected, "status": "running", "passed": False}
                record["cases"].append(entry)
                result_path.write_text(json.dumps(record, indent=2) + "\n")
                try:
                    entry.update(_run_case(tokens, selected))
                    entry["status"] = "passed"
                except Exception as exc:
                    entry["status"] = "failed"
                    entry["error"] = repr(exc)
                    entry["traceback"] = traceback.format_exc()
                    raise
                result_path.write_text(json.dumps(record, indent=2) + "\n")
        record["telemetry_after"] = _telemetry()
        record["status"] = "passed" if all(case["passed"] for case in record["cases"]) else "failed"
    except Exception as exc:
        record["status"] = "failed"
        record["failure"] = f"{type(exc).__name__}: {exc}"
        record["telemetry_after"] = _telemetry()
        raise
    finally:
        result_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"M8 result: {result_path.relative_to(ROOT)}", flush=True)
    return result_path


if __name__ == "__main__":
    run()
