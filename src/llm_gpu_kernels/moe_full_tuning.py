"""Training-only full-forward tile search for the ranked fixed-route MoE path.

This experiment uses the local configured launcher, a full FP32 oracle, and
CUDA Graph A/B measurements. It does not read route counts back to the host and
does not inspect or tune against the M5 holdout matrix.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
import platform
import time
import traceback
from pathlib import Path

import torch
import triton

from .baseline_benchmark import graphs
from .benchmark import LIMITS, ROOT, command, error, tensor_hash
from .gemm import GemmConfig
from .moe_benchmark import make_inputs, telemetry
from .moe_configured import prepare_configured_moe
from .moe_full_policy import (
    BASELINE_CONFIG,
    BUCKET_NAMES,
    EXPECTED_DOMAIN,
    EXPECTED_THRESHOLDS,
    POLICY_SCHEMA,
    POLICY_STATUS,
    TRAINING_RECORD_HASH_SCOPE,
    canonical_training_record_sha256,
)
from .optimization_benchmark import measure
from .reference import moe_reference


TRAINING_SEED = 409
TRAINING_TOKENS = (64, 512, 4096)
ROUTE_PATTERNS = ("balanced", "hot16")
TOPK = 8
GRAPH_REPEATS = 8
SCAN_SAMPLES = 6
SCAN_CALLS_PER_SAMPLE = 2
RECHECK_SAMPLES = 8
RECHECK_CALLS_PER_SAMPLE = 4
ROUNDS = 4

# Stable order is also the deterministic tie-break for equal training scores.
CONFIGS = (
    GemmConfig(),
    GemmConfig(block_m=16, block_n=64, block_k=32, warps=4, stages=3),
    GemmConfig(block_m=64, block_n=64, block_k=32, warps=4, stages=3),
    GemmConfig(block_m=16, block_n=32, block_k=64, warps=4, stages=3),
    GemmConfig(block_m=32, block_n=32, block_k=32, warps=4, stages=3),
    GemmConfig(block_m=32, block_n=128, block_k=32, warps=4, stages=3),
    GemmConfig(block_m=32, block_n=256, block_k=32, warps=4, stages=3),
    GemmConfig(block_m=32, block_n=64, block_k=64, warps=4, stages=3),
    GemmConfig(block_m=32, block_n=128, block_k=64, warps=4, stages=3),
    GemmConfig(block_m=64, block_n=128, block_k=32, warps=4, stages=3),
    GemmConfig(block_m=32, block_n=64, block_k=32, warps=8, stages=3),
    GemmConfig(block_m=32, block_n=64, block_k=32, warps=4, stages=2),
)

# Deliberately explicit: provenance covers code and protocol files that affect
# this runner, not unrelated files discovered by a future recursive glob.
SOURCE_PATHS = (
    "docs/M7_FULL_MOE_TUNING_PLAN.md",
    "src/llm_gpu_kernels/__init__.py",
    "src/llm_gpu_kernels/benchmark.py",
    "src/llm_gpu_kernels/baseline_benchmark.py",
    "src/llm_gpu_kernels/gemm.py",
    "src/llm_gpu_kernels/moe.py",
    "src/llm_gpu_kernels/moe_benchmark.py",
    "src/llm_gpu_kernels/moe_configured.py",
    "src/llm_gpu_kernels/moe_fused.py",
    "src/llm_gpu_kernels/moe_full_policy.py",
    "src/llm_gpu_kernels/moe_full_tuning.py",
    "src/llm_gpu_kernels/moe_prepared.py",
    "src/llm_gpu_kernels/moe_ranked.py",
    "src/llm_gpu_kernels/optimization_benchmark.py",
    "src/llm_gpu_kernels/reference.py",
)


class _BaselineFailure(RuntimeError):
    """A baseline correctness failure invalidates the training run."""

    def __init__(self, message, metrics=None):
        super().__init__(message)
        self.metrics = metrics


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    return {
        relative: _sha256_file(ROOT / relative)
        for relative in SOURCE_PATHS
    }


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _save(path, record):
    path.write_bytes(_json_bytes(record))


def _bucket(T, E=EXPECTED_DOMAIN["experts"], topk=EXPECTED_DOMAIN["topk"]):
    """Shape-only route density estimate; never examines IDs/counts."""
    mu = int(T) * int(topk) / int(E)
    if mu <= EXPECTED_THRESHOLDS[0]:
        return BUCKET_NAMES[0], mu
    if mu <= EXPECTED_THRESHOLDS[1]:
        return BUCKET_NAMES[1], mu
    return BUCKET_NAMES[2], mu


def _make_training_inputs(tokens, route_pattern, device):
    experts = EXPECTED_DOMAIN["experts"]
    hidden = EXPECTED_DOMAIN["hidden"]
    intermediate = EXPECTED_DOMAIN["intermediate"]

    # Generate X, weights, and route weights identically for both patterns at
    # a given T. The hot helper generates the same seed-409 top-k routes; the
    # balanced case then replaces only the IDs with the specified exact cycle.
    buffers = list(make_inputs(
        tokens=tokens,
        experts=experts,
        topk=TOPK,
        hidden=hidden,
        intermediate=intermediate,
        id_dtype=torch.int32,
        pattern="hot",
        seed=TRAINING_SEED,
        device=device,
    ))
    if route_pattern == "balanced":
        balanced_ids = (
            torch.arange(tokens * TOPK, dtype=torch.int64)
            .remainder(experts)
            .reshape(tokens, TOPK)
            .to(device=device, dtype=buffers[3].dtype)
        )
        buffers[3].copy_(balanced_ids)
    elif route_pattern != "hot16":
        raise ValueError(f"unknown training route pattern: {route_pattern}")
    return tuple(buffers)


def _kernel_resources(kernel):
    if kernel is None:
        return None
    metadata = getattr(kernel, "metadata", None)
    registers = getattr(kernel, "n_regs", None)
    spills = getattr(kernel, "n_spills", None)
    shared = getattr(metadata, "shared", None)
    return {
        "registers": None if registers is None else int(registers),
        "spills": None if spills is None else int(spills),
        "shared_memory_bytes": None if shared is None else int(shared),
    }


def _plan_metadata(prepared):
    base = prepared.base
    return {
        "launch_count": len(prepared.launches),
        "workspace_bytes": int(prepared.workspace_bytes),
        "base_workspace_bytes": int(base.workspace_bytes),
        "rank_workspace_bytes": int(prepared.ranks.numel() * prepared.ranks.element_size()),
        "gemm_resources": {
            "gate_up_swiglu": _kernel_resources(base.kernels.get("gate_up_swiglu")),
            "down": _kernel_resources(base.kernels.get("down")),
        },
    }


def _metrics(output, oracle):
    return error(output, oracle)


def _measure_full_forward_pair(baseline, baseline_graph, candidate, oracle,
                               graph_repeats, samples, calls_per_sample):
    """Correctness-check, capture, pair-measure, and post-check one candidate."""
    device = baseline.base.x.device
    result = {"correctness": {}, "timing": None}
    candidate_graph = None
    try:
        candidate.run()
        torch.cuda.synchronize(device)
        baseline_eager = _metrics(baseline.base.output, oracle)
        candidate_eager = _metrics(candidate.base.output, oracle)
        result["correctness"]["premeasurement_eager"] = {
            "baseline": baseline_eager,
            "candidate": candidate_eager,
        }
        if not baseline_eager["passed"]:
            raise _BaselineFailure("baseline failed premeasurement oracle", baseline_eager)
        if not candidate_eager["passed"]:
            result["status"] = "candidate_numerical_failure_pre"
            result["valid_for_selection"] = False
            return result

        candidate_graph = graphs({"candidate": candidate.run}, graph_repeats)["candidate"]
        baseline_graph.replay()
        candidate_graph.replay()
        torch.cuda.synchronize(device)
        baseline_graph_pre = _metrics(baseline.base.output, oracle)
        candidate_graph_pre = _metrics(candidate.base.output, oracle)
        result["correctness"]["premeasurement_graph"] = {
            "baseline": baseline_graph_pre,
            "candidate": candidate_graph_pre,
        }
        if not baseline_graph_pre["passed"]:
            raise _BaselineFailure("baseline failed premeasurement Graph oracle", baseline_graph_pre)
        if not candidate_graph_pre["passed"]:
            result["status"] = "candidate_numerical_failure_graph_pre"
            result["valid_for_selection"] = False
            return result

        flops_per_forward = (
            6 * baseline.base.x.shape[0] * baseline.base.ids.shape[1]
            * baseline.base.x.shape[1] * baseline.base.down.shape[2]
        )
        timing = measure(
            {
                "baseline": baseline_graph.replay,
                "candidate": candidate_graph.replay,
            },
            flops_per_forward,
            divisor=graph_repeats,
            samples=samples,
            repeats=calls_per_sample,
        )
        # optimization_benchmark.measure historically labels this GEMM; M7
        # uses its exact raw values but records the full-forward unit explicitly.
        for row in timing["rounds"]:
            row["samples_ms_per_forward"] = row.pop("samples_ms_per_gemm")
            row["wall_ms_per_forward"] = row.pop("wall_ms_per_gemm")
        timing["sample_unit"] = "milliseconds per complete fixed-route forward"
        timing["graph_repeats_per_replay"] = graph_repeats
        timing["calls_per_sample"] = calls_per_sample
        timing["balanced_order"] = "AB, BA, BA, AB"
        result["timing"] = timing
        baseline_p50 = timing["summary"]["baseline"]["p50_ms"]
        candidate_p50 = timing["summary"]["candidate"]["p50_ms"]
        result["p50_ratio_to_baseline"] = candidate_p50 / baseline_p50

        baseline_graph.replay()
        candidate_graph.replay()
        torch.cuda.synchronize(device)
        baseline_post = _metrics(baseline.base.output, oracle)
        candidate_post = _metrics(candidate.base.output, oracle)
        result["correctness"]["postmeasurement_graph"] = {
            "baseline": baseline_post,
            "candidate": candidate_post,
        }
        if not baseline_post["passed"]:
            raise _BaselineFailure("baseline failed postmeasurement Graph oracle", baseline_post)
        if not candidate_post["passed"]:
            result["status"] = "candidate_numerical_failure_post"
            result["valid_for_selection"] = False
            return result

        result["status"] = "passed"
        result["valid_for_selection"] = True
        return result
    finally:
        # Drop the temporary graph pool before the next tile plan is prepared.
        if candidate_graph is not None:
            del candidate_graph
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def _candidate_entry(case_id, tokens, route_pattern, config_index, config):
    bucket, mu = _bucket(tokens)
    return {
        "case_id": case_id,
        "T": tokens,
        "route_pattern": route_pattern,
        "mu_shape_estimate": mu,
        "bucket": bucket,
        "config_index": config_index,
        "config": asdict(config),
        "status": "running",
        "valid_for_selection": False,
    }


def _blocked_entries(case_record, reason):
    for index, config in enumerate(CONFIGS):
        row = _candidate_entry(
            case_record["case_id"], case_record["T"],
            case_record["route_pattern"], index, config,
        )
        row["status"] = "blocked_baseline_failure"
        row["valid_for_selection"] = False
        row["error"] = reason
        case_record["candidates"].append(row)


def _runtime_record(device):
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": properties.name,
        "sm": [properties.major, properties.minor],
        "memory_bytes": int(properties.total_memory),
        "device_index": int(device.index),
    }


def _validate_runtime(device):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for M7 full-MoE tuning")
    if torch.__version__ != "2.7.1+cu128":
        raise RuntimeError(f"M7 expects torch 2.7.1+cu128, got {torch.__version__}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"M7 expects CUDA runtime 12.8, got {torch.version.cuda}")
    if triton.__version__ != "3.3.1":
        raise RuntimeError(f"M7 expects Triton 3.3.1, got {triton.__version__}")
    properties = torch.cuda.get_device_properties(device)
    if properties.major != 8 or properties.minor != 6:
        raise RuntimeError(f"M7 is scoped to SM86, got SM{properties.major}{properties.minor}")


def _case_hashes(buffers):
    return {
        "inputs_sha256": tensor_hash(*buffers),
        "routes_sha256": tensor_hash(buffers[3], buffers[4]),
    }


def _candidate_lookup(record):
    lookup = {}
    for case in record["cases"]:
        for candidate in case.get("candidates", ()):
            lookup[(case["case_id"], candidate["config_index"])] = candidate
    return lookup


def _rank_bucket_candidates(record, bucket_name):
    candidates = _candidate_lookup(record)
    tokens_for_bucket = [
        tokens for tokens in TRAINING_TOKENS if _bucket(tokens)[0] == bucket_name
    ]
    case_ids = [
        f"T{tokens}_{route}"
        for tokens in tokens_for_bucket
        for route in ROUTE_PATTERNS
    ]
    rows = []
    for index, config in enumerate(CONFIGS):
        chosen_rows = [candidates.get((case_id, index)) for case_id in case_ids]
        ratios = [
            row.get("p50_ratio_to_baseline")
            for row in chosen_rows
            if row is not None and row.get("valid_for_selection")
        ]
        valid_global = all(
            row is not None and row.get("valid_for_selection")
            for row in candidates.values()
            if row.get("config_index") == index
        )
        # There must be exactly one completed row for all six training cases.
        all_cases = [
            row for row in candidates.values()
            if row.get("config_index") == index
        ]
        valid_global = valid_global and len(all_cases) == len(TRAINING_TOKENS) * len(ROUTE_PATTERNS)
        score = math.exp(sum(math.log(value) for value in ratios) / len(ratios)) if ratios else None
        maximum = max(ratios) if ratios else None
        eligible = bool(
            valid_global and len(ratios) == len(case_ids)
            and score is not None and score <= 0.95
            and maximum is not None and maximum <= 1.05
        )
        rows.append({
            "config_index": index,
            "config": asdict(config),
            "case_ids": case_ids,
            "ratios": ratios,
            "geomean_ratio": score,
            "max_case_ratio": maximum,
            "valid_across_all_training_cases": valid_global,
            "eligible": eligible,
        })
    eligible_rows = [row for row in rows if row["eligible"]]
    eligible_rows.sort(key=lambda row: (row["geomean_ratio"], row["config_index"]))
    selected_index = eligible_rows[0]["config_index"] if eligible_rows else 0
    reason = "best_eligible_training_geomean" if eligible_rows else "no_candidate_met_training_criteria"
    return {
        "bucket": bucket_name,
        "training_case_ids": case_ids,
        "candidate_scores": rows,
        "initial_selected_index": selected_index,
        "initial_selection_reason": reason,
    }


def _new_scan_result(run_id, run_path, device):
    return {
        "schema": 1,
        "run_id": run_id,
        "source_run": run_id,
        "status": "running",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_hashes": _source_hashes(),
        "plan_sha256": _sha256_file(ROOT / "docs/M7_FULL_MOE_TUNING_PLAN.md"),
        "runtime": _runtime_record(device),
        "limits": dict(LIMITS),
        "training_protocol": {
            "seed": TRAINING_SEED,
            "tokens": list(TRAINING_TOKENS),
            "routes": list(ROUTE_PATTERNS),
            "domain": dict(EXPECTED_DOMAIN),
            "balanced_ids": "arange(T * topk) % E, reshaped to [T, topk]",
            "hot16_ids": "moe_benchmark.make_inputs(pattern='hot') with E=128; its fixed helper selects 16 experts",
            "same_inputs_across_routes": "same seed per T; X, gate_up, down, and routing weights match; only IDs differ",
        },
        "configs": [
            {"index": index, **asdict(config)}
            for index, config in enumerate(CONFIGS)
        ],
        "scope": "complete ranked fixed-route BF16 MoE forward; shared GU/down config; no router or serving stack",
        "selection_rule": {
            "shape_only_mu": "T * topk / E; no GPU counts readback or route-value selection",
            "thresholds_mu": list(EXPECTED_THRESHOLDS),
            "training_gain_geomean_ratio_max": 0.95,
            "training_worst_case_ratio_max": 1.05,
            "recheck_failure_action": "fallback to baseline; do not try the next-ranked config",
        },
        "timing_protocol": {
            "mode": "paired complete-forward CUDA Graph",
            "graph_forwards_per_replay": GRAPH_REPEATS,
            "scan_samples_per_round": SCAN_SAMPLES,
            "scan_calls_per_sample": SCAN_CALLS_PER_SAMPLE,
            "recheck_samples_per_round": RECHECK_SAMPLES,
            "recheck_calls_per_sample": RECHECK_CALLS_PER_SAMPLE,
            "balanced_round_orders": ["AB", "BA", "BA", "AB"],
            "rounds": ROUNDS,
            "setup_and_graph_capture_timed": False,
            "reported_unit": "milliseconds per complete forward, event time divided by calls and graph forward count",
            "theoretical_flops_per_forward": "6 * T * topk * H * I; derived throughput is not a hardware counter",
        },
        "holdout_policy": "M5 seeds 73/991 and T193/1024/2049 are not read or used for selection; run them only after freeze",
        "result_path": str(run_path.relative_to(ROOT)),
        "telemetry_before": telemetry(),
        "cases": [],
        "selection": [],
        "rechecks": [],
    }


def _run_scan_case(record, run_path, tokens, route_pattern, device):
    case_id = f"T{tokens}_{route_pattern}"
    bucket_name, mu = _bucket(tokens)
    case_record = {
        "case_id": case_id,
        "T": tokens,
        "E": EXPECTED_DOMAIN["experts"],
        "topk": TOPK,
        "H": EXPECTED_DOMAIN["hidden"],
        "I": EXPECTED_DOMAIN["intermediate"],
        "route_pattern": route_pattern,
        "mu_shape_estimate": mu,
        "bucket": bucket_name,
        "status": "running",
        "telemetry_before": telemetry(),
        "candidates": [],
    }
    record["cases"].append(case_record)
    _save(run_path, record)

    buffers = None
    oracle = None
    baseline = None
    baseline_graph = None
    baseline_graphs = None
    baseline_metrics = None
    case_failed_baseline = False
    try:
        buffers = _make_training_inputs(tokens, route_pattern, device)
        hashes = _case_hashes(buffers)
        case_record.update(hashes)
        torch.cuda.reset_peak_memory_stats(device)
        oracle = moe_reference(*buffers)
        baseline = prepare_configured_moe(*buffers, BASELINE_CONFIG)
        baseline.run()
        torch.cuda.synchronize(device)
        baseline_metrics = _metrics(baseline.base.output, oracle)
        case_record["baseline_premeasurement_eager"] = baseline_metrics
        if not baseline_metrics["passed"]:
            raise _BaselineFailure("baseline failed training oracle", baseline_metrics)

        baseline_graphs = graphs({"baseline": baseline.run}, GRAPH_REPEATS)
        baseline_graph = baseline_graphs["baseline"]
        baseline_graph.replay()
        torch.cuda.synchronize(device)
        baseline_graph_metrics = _metrics(baseline.base.output, oracle)
        case_record["baseline_premeasurement_graph"] = baseline_graph_metrics
        case_record["baseline_resources"] = _plan_metadata(baseline)
        if not baseline_graph_metrics["passed"]:
            raise _BaselineFailure("baseline failed Graph oracle", baseline_graph_metrics)
        case_record["status"] = "baseline_passed"
        _save(run_path, record)
    except Exception as exc:
        case_failed_baseline = True
        case_record["status"] = "baseline_failure"
        case_record["baseline_error"] = repr(exc)
        case_record["baseline_traceback"] = traceback.format_exc()
        if isinstance(exc, _BaselineFailure) and exc.metrics is not None:
            case_record["baseline_failure_metrics"] = exc.metrics
        _blocked_entries(case_record, repr(exc))
        case_record["telemetry_after"] = telemetry()
        case_record["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        _save(run_path, record)
        del baseline_graphs, baseline_graph, baseline, oracle, buffers
        gc.collect()
        torch.cuda.empty_cache()
        return True

    for config_index, config in enumerate(CONFIGS):
        entry = _candidate_entry(case_id, tokens, route_pattern, config_index, config)
        entry["case_input_sha256"] = case_record["inputs_sha256"]
        entry["resources"] = None
        entry["workspace"] = None
        entry["launch_count"] = None
        case_record["candidates"].append(entry)
        _save(run_path, record)
        candidate = None
        try:
            torch.cuda.reset_peak_memory_stats(device)
            candidate = prepare_configured_moe(*buffers, config)
            metadata = _plan_metadata(candidate)
            entry["resources"] = metadata["gemm_resources"]
            entry["workspace"] = {
                "total_bytes": metadata["workspace_bytes"],
                "base_bytes": metadata["base_workspace_bytes"],
                "rank_bytes": metadata["rank_workspace_bytes"],
            }
            entry["launch_count"] = metadata["launch_count"]
            measured = _measure_full_forward_pair(
                baseline,
                baseline_graph,
                candidate,
                oracle,
                GRAPH_REPEATS,
                SCAN_SAMPLES,
                SCAN_CALLS_PER_SAMPLE,
            )
            entry.update(measured)
            entry["pair_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            if measured.get("status") == "passed":
                entry["passed"] = True
            elif measured.get("status", "").startswith("candidate_numerical_failure"):
                entry["passed"] = False
                entry["failure_kind"] = "numerical"
            elif measured.get("status") == "baseline_failure":
                case_failed_baseline = True
                entry["passed"] = False
                entry["failure_kind"] = "baseline_numerical"
            else:
                entry["passed"] = False
            case_record["peak_allocated_bytes"] = max(
                case_record.get("peak_allocated_bytes", 0),
                entry["pair_peak_allocated_bytes"],
            )
        except _BaselineFailure as exc:
            case_failed_baseline = True
            entry["status"] = "baseline_failure"
            entry["valid_for_selection"] = False
            entry["passed"] = False
            entry["failure_kind"] = "baseline_numerical"
            entry["error"] = repr(exc)
            entry["traceback"] = traceback.format_exc()
            if exc.metrics is not None:
                entry["baseline_failure_metrics"] = exc.metrics
        except Exception as exc:
            entry["status"] = "candidate_exception"
            entry["valid_for_selection"] = False
            entry["passed"] = False
            message = (type(exc).__name__ + " " + str(exc)).lower()
            entry["failure_kind"] = (
                "resource_or_compile"
                if "resource" in message or "out of memory" in message or "oom" in message
                else "compile_or_runtime"
            )
            entry["error"] = repr(exc)
            entry["traceback"] = traceback.format_exc()
        finally:
            if candidate is not None:
                del candidate
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            _save(run_path, record)

    try:
        final_hashes = _case_hashes(buffers)
        case_record["post_scan_input_sha256"] = final_hashes["inputs_sha256"]
        case_record["post_scan_routes_sha256"] = final_hashes["routes_sha256"]
        case_record["inputs_unchanged"] = final_hashes == hashes
        if not case_record["inputs_unchanged"]:
            for entry in case_record["candidates"]:
                entry["valid_for_selection"] = False
                entry["failure_kind"] = "input_mutation"
                entry["status"] = "input_mutation"
                entry["passed"] = False
            case_record["status"] = "input_mutation"
        elif case_failed_baseline:
            case_record["status"] = "baseline_failure"
        else:
            case_record["status"] = "passed"
    except Exception as exc:
        case_record["status"] = "input_hash_failure"
        case_record["input_hash_error"] = repr(exc)
        case_failed_baseline = True
    case_record["telemetry_after"] = telemetry()
    case_record["peak_allocated_bytes"] = max(
        int(case_record.get("peak_allocated_bytes", 0)),
        int(torch.cuda.max_memory_allocated(device)),
    )
    _save(run_path, record)

    if baseline_graphs is not None:
        del baseline_graphs
    if baseline_graph is not None:
        del baseline_graph
    if baseline is not None:
        del baseline
    if oracle is not None:
        del oracle
    if buffers is not None:
        del buffers
    gc.collect()
    torch.cuda.empty_cache()
    return case_failed_baseline or case_record.get("status") != "passed"


def _recheck_bucket(record, run_path, bucket_decision, device):
    bucket_name = bucket_decision["bucket"]
    initial_index = bucket_decision["initial_selected_index"]
    initial_config = CONFIGS[initial_index]
    tokens_for_bucket = [
        tokens for tokens in TRAINING_TOKENS if _bucket(tokens)[0] == bucket_name
    ]
    recheck_rows = []
    fatal_baseline_failure = False

    for tokens in tokens_for_bucket:
        for route_pattern in ROUTE_PATTERNS:
            case_id = f"T{tokens}_{route_pattern}"
            entry = {
                "bucket": bucket_name,
                "case_id": case_id,
                "T": tokens,
                "route_pattern": route_pattern,
                "config_index": initial_index,
                "config": asdict(initial_config),
                "status": "running",
            }
            record["rechecks"].append(entry)
            recheck_rows.append(entry)
            _save(run_path, record)
            buffers = None
            oracle = None
            baseline = None
            selected = None
            baseline_graph = None
            baseline_graphs = None
            candidate_graphs = None
            candidate_graph = None
            try:
                torch.cuda.reset_peak_memory_stats(device)
                buffers = _make_training_inputs(tokens, route_pattern, device)
                hashes = _case_hashes(buffers)
                entry["inputs_sha256"] = hashes["inputs_sha256"]
                scan_case = next(
                    case for case in record["cases"] if case["case_id"] == case_id
                )
                entry["matches_scan_inputs"] = (
                    hashes["inputs_sha256"] == scan_case.get("inputs_sha256")
                    and hashes["routes_sha256"] == scan_case.get("routes_sha256")
                )
                if not entry["matches_scan_inputs"]:
                    raise RuntimeError("recheck input hashes do not match the training scan")

                oracle = moe_reference(*buffers)
                baseline = prepare_configured_moe(*buffers, BASELINE_CONFIG)
                selected = prepare_configured_moe(*buffers, initial_config)
                baseline.run()
                selected.run()
                torch.cuda.synchronize(device)
                baseline_pre = _metrics(baseline.base.output, oracle)
                selected_pre = _metrics(selected.base.output, oracle)
                entry["correctness_premeasurement_eager"] = {
                    "baseline": baseline_pre,
                    "selected": selected_pre,
                }
                if not baseline_pre["passed"]:
                    raise _BaselineFailure("baseline failed recheck eager oracle", baseline_pre)
                if not selected_pre["passed"]:
                    entry["status"] = "selected_numerical_failure_pre"
                    entry["passed"] = False
                    continue

                baseline_graphs = graphs({"baseline": baseline.run}, GRAPH_REPEATS)
                baseline_graph = baseline_graphs["baseline"]
                candidate_graphs = graphs({"selected": selected.run}, GRAPH_REPEATS)
                candidate_graph = candidate_graphs["selected"]
                baseline_graph.replay()
                candidate_graph.replay()
                torch.cuda.synchronize(device)
                baseline_graph_pre = _metrics(baseline.base.output, oracle)
                selected_graph_pre = _metrics(selected.base.output, oracle)
                entry["correctness_premeasurement_graph"] = {
                    "baseline": baseline_graph_pre,
                    "selected": selected_graph_pre,
                }
                if not baseline_graph_pre["passed"]:
                    raise _BaselineFailure("baseline failed recheck Graph oracle", baseline_graph_pre)
                if not selected_graph_pre["passed"]:
                    entry["status"] = "selected_numerical_failure_graph_pre"
                    entry["passed"] = False
                    continue

                measured = measure(
                    {
                        "baseline": baseline_graph.replay,
                        "selected": candidate_graph.replay,
                    },
                    6 * tokens * TOPK * EXPECTED_DOMAIN["hidden"] * EXPECTED_DOMAIN["intermediate"],
                    divisor=GRAPH_REPEATS,
                    samples=RECHECK_SAMPLES,
                    repeats=RECHECK_CALLS_PER_SAMPLE,
                )
                for row in measured["rounds"]:
                    row["samples_ms_per_forward"] = row.pop("samples_ms_per_gemm")
                    row["wall_ms_per_forward"] = row.pop("wall_ms_per_gemm")
                measured["sample_unit"] = "milliseconds per complete fixed-route forward"
                measured["graph_repeats_per_replay"] = GRAPH_REPEATS
                measured["calls_per_sample"] = RECHECK_CALLS_PER_SAMPLE
                measured["balanced_order"] = "AB, BA, BA, AB"
                entry["timing"] = measured
                entry["p50_ratio_to_baseline"] = (
                    measured["summary"]["selected"]["p50_ms"]
                    / measured["summary"]["baseline"]["p50_ms"]
                )

                baseline_graph.replay()
                candidate_graph.replay()
                torch.cuda.synchronize(device)
                baseline_post = _metrics(baseline.base.output, oracle)
                selected_post = _metrics(selected.base.output, oracle)
                entry["correctness_postmeasurement_graph"] = {
                    "baseline": baseline_post,
                    "selected": selected_post,
                }
                if not baseline_post["passed"]:
                    raise _BaselineFailure("baseline failed recheck post oracle", baseline_post)
                if not selected_post["passed"]:
                    entry["status"] = "selected_numerical_failure_post"
                    entry["passed"] = False
                    continue
                entry["status"] = "passed"
                entry["passed"] = True
                entry["resources"] = _plan_metadata(selected)
                entry["baseline_resources"] = _plan_metadata(baseline)
            except _BaselineFailure as exc:
                entry["status"] = "baseline_failure"
                entry["passed"] = False
                entry["error"] = repr(exc)
                entry["traceback"] = traceback.format_exc()
                if exc.metrics is not None:
                    entry["baseline_failure_metrics"] = exc.metrics
                fatal_baseline_failure = True
            except Exception as exc:
                entry["status"] = "recheck_exception"
                entry["passed"] = False
                entry["error"] = repr(exc)
                entry["traceback"] = traceback.format_exc()
            finally:
                entry["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
                _save(run_path, record)
                if candidate_graphs is not None:
                    del candidate_graphs
                if candidate_graph is not None:
                    del candidate_graph
                if baseline_graphs is not None:
                    del baseline_graphs
                if baseline_graph is not None:
                    del baseline_graph
                if selected is not None:
                    del selected
                if baseline is not None:
                    del baseline
                if oracle is not None:
                    del oracle
                if buffers is not None:
                    del buffers
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

    candidate_ratios = [
        row["p50_ratio_to_baseline"]
        for row in recheck_rows
        if row.get("passed") and "p50_ratio_to_baseline" in row
    ]
    score = (
        math.exp(sum(math.log(ratio) for ratio in candidate_ratios) / len(candidate_ratios))
        if candidate_ratios else None
    )
    maximum = max(candidate_ratios) if candidate_ratios else None
    complete_correctness = len(recheck_rows) == len(ROUTE_PATTERNS) and all(
        row.get("passed") for row in recheck_rows
    )
    decision = {
        **bucket_decision,
        "initial_config": asdict(initial_config),
        "recheck_case_ids": [row["case_id"] for row in recheck_rows],
        "recheck_geomean_ratio": score,
        "recheck_max_case_ratio": maximum,
        "recheck_all_passed": complete_correctness,
        "recheck_baseline_failure": fatal_baseline_failure,
    }

    if initial_index == 0:
        final_index = 0
        decision["final_selection_reason"] = "baseline_selected_by_scan_or_fallback"
    elif (complete_correctness and score is not None and score <= 0.95
          and maximum is not None and maximum <= 1.05 and not fatal_baseline_failure):
        final_index = initial_index
        decision["final_selection_reason"] = "selected_config_passed_longer_recheck"
    else:
        final_index = 0
        decision["final_selection_reason"] = "recheck_failed_fallback_to_baseline_no_next_candidate"

    decision["final_config_index"] = final_index
    decision["final_config"] = asdict(CONFIGS[final_index])
    return decision, fatal_baseline_failure


def _policy_payload(record, run_path, decisions):
    record_hash = canonical_training_record_sha256(record)
    return {
        "schema": POLICY_SCHEMA,
        "status": POLICY_STATUS,
        "source_run": record["run_id"],
        "training_record_path": str(run_path.relative_to(ROOT)),
        "training_record_sha256": record_hash,
        "training_record_hash_scope": TRAINING_RECORD_HASH_SCOPE,
        "training_protocol": {
            "seed": TRAINING_SEED,
            "tokens": list(TRAINING_TOKENS),
            "routes": list(ROUTE_PATTERNS),
        },
        "domain": dict(EXPECTED_DOMAIN),
        "thresholds_mu": list(EXPECTED_THRESHOLDS),
        "configs": {
            bucket_name: decisions[bucket_name]["final_config"]
            for bucket_name in BUCKET_NAMES
        },
        "fallback": asdict(BASELINE_CONFIG),
        "buckets": {
            bucket_name: {
                "config_index": decisions[bucket_name]["final_config_index"],
                "config": decisions[bucket_name]["final_config"],
                "training_geomean_ratio": next(
                    (row["geomean_ratio"] for row in decisions[bucket_name]["candidate_scores"]
                     if row["config_index"] == decisions[bucket_name]["final_config_index"]),
                    None,
                ),
                "recheck_geomean_ratio": decisions[bucket_name]["recheck_geomean_ratio"],
                "selection_reason": decisions[bucket_name]["final_selection_reason"],
            }
            for bucket_name in BUCKET_NAMES
        },
        "search_config_count": len(CONFIGS),
        "source_hashes": record["source_hashes"],
        "policy_scope": "shape-only rule for exact E/H/I/topk domain; no active route-count inspection",
    }


def run():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for M7 full-MoE tuning")
    device = torch.device("cuda", torch.cuda.current_device())
    _validate_runtime(device)

    run_id = (
        "m7-full-moe-tuning-"
        + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-"
        + str(time.time_ns() % 10**9)
    )
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    run_path = results_dir / (run_id + ".json")
    policy_path = results_dir / "m7-policy.json"
    if policy_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen policy: {policy_path}")
    if run_path.exists():
        raise FileExistsError(f"refusing to overwrite training record: {run_path}")

    record = _new_scan_result(run_id, run_path, device)
    _save(run_path, record)
    baseline_failed = False
    successful_freeze = False
    try:
        if len(CONFIGS) != 12:
            raise RuntimeError("M7 requires the fixed 12-configuration search set")
        if len(SOURCE_PATHS) != len(set(SOURCE_PATHS)):
            raise RuntimeError("M7 explicit provenance source list contains duplicates")

        for tokens in TRAINING_TOKENS:
            for route_pattern in ROUTE_PATTERNS:
                print(f"M7 tile scan T={tokens} route={route_pattern}", flush=True)
                baseline_failed = _run_scan_case(
                    record, run_path, tokens, route_pattern, device
                ) or baseline_failed

        expected_rows = len(TRAINING_TOKENS) * len(ROUTE_PATTERNS) * len(CONFIGS)
        actual_rows = sum(len(case.get("candidates", ())) for case in record["cases"])
        record["candidate_result_count"] = actual_rows
        if actual_rows != expected_rows:
            raise RuntimeError(f"incomplete M7 scan: expected {expected_rows} rows, got {actual_rows}")
        if baseline_failed:
            raise RuntimeError("one or more M7 training cases failed; no policy was frozen")

        decisions = {}
        for bucket_name in BUCKET_NAMES:
            decisions[bucket_name] = _rank_bucket_candidates(record, bucket_name)
        record["selection"] = [decisions[name] for name in BUCKET_NAMES]
        _save(run_path, record)

        for bucket_name in BUCKET_NAMES:
            print(f"M7 longer recheck bucket={bucket_name}", flush=True)
            decisions[bucket_name], recheck_baseline_failed = _recheck_bucket(
                record, run_path, decisions[bucket_name], device
            )
            baseline_failed = recheck_baseline_failed or baseline_failed
            # Keep the visible decision synchronized with the final freeze.
            record["selection"] = [decisions[name] for name in BUCKET_NAMES
                                   if name in decisions]
            _save(run_path, record)
        if baseline_failed:
            raise RuntimeError("baseline failed during M7 recheck; no policy was frozen")

        record["status"] = "passed"
        record["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record["telemetry_after"] = telemetry()
        record["policy_path"] = str(policy_path.relative_to(ROOT))
        # `policy_sha256` is added only after writing the policy; its back
        # reference is omitted by the policy's documented canonical record hash.
        payload = _policy_payload(record, run_path, decisions)
        policy_bytes = _json_bytes(payload)
        policy_path.write_bytes(policy_bytes)
        policy_hash = hashlib.sha256(policy_bytes).hexdigest()
        record["policy_sha256"] = policy_hash
        _save(run_path, record)
        if canonical_training_record_sha256(record) != payload["training_record_sha256"]:
            raise RuntimeError("final training record does not match its canonical policy hash")
        successful_freeze = True
        print(f"Saved {run_path} and frozen policy {policy_path}", flush=True)
        return run_path
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
        raise
    finally:
        if not successful_freeze:
            record["telemetry_after"] = telemetry()
            _save(run_path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
