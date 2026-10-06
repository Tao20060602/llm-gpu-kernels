"""Frozen M7 full-MoE policy holdout.

This driver compares the fixed M6 ranked-dispatch launch sequence, an explicit
configuration selected by the frozen M7 policy, and the unmodified vLLM public
operator over the M5 full-forward matrix. It performs no tuning or configuration
search. Run it only after the M7 training result and policy have been frozen.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch
import triton

from .baselines.vllm_moe import prepare_vllm_moe
from .benchmark import LIMITS, ROOT, tensor_hash
from .gemm import GemmConfig
from .mature_moe_benchmark import (
    GRAPH_REPEATS,
    _capture_graph,
    _copy_mutable_values,
    _make_mutable_update,
    _mutable_hash,
    _oracle_metrics,
    _performance_specs,
    _runtime_metadata,
    _spec_metadata,
    _json_value,
)
from .moe_benchmark import _functional_specs, make_inputs, telemetry, validate_dispatch
from .moe_configured import prepare_configured_moe
from .moe_full_policy import FrozenMoEPolicy
from .moe_ranked import prepare_ranked_moe_launcher
from .ranked_moe_benchmark import _validate_ranked_dispatch
from .optimization_benchmark import measure
from .reference import moe_reference


POLICY_PATH = ROOT / "results/m7-policy.json"
VLLM_PROVENANCE_PATH = ROOT / "results/provenance/vllm-0.10.0-runtime.json"
TRAINING_HASH_SCOPE = "canonical_json_without_top_level_policy_sha256"
API_SAMPLES = 6
API_REPEATS = 2
BOUNDARY_TOKENS = 31
BOUNDARY_EXPERTS = 3
BOUNDARY_TOPK = 2
BOUNDARY_HIDDEN = 131
BOUNDARY_INTERMEDIATE = 67
HOLDOUT_SUPPORT_SOURCES = (
    "docs/M5_EXPERIMENT_PLAN.md",
    "src/llm_gpu_kernels/baselines/vllm_moe.py",
    "src/llm_gpu_kernels/benchmark.py",
    "src/llm_gpu_kernels/gemm.py",
    "src/llm_gpu_kernels/mature_moe_benchmark.py",
    "src/llm_gpu_kernels/moe_benchmark.py",
    "src/llm_gpu_kernels/moe_configured.py",
    "src/llm_gpu_kernels/moe_full_holdout.py",
    "src/llm_gpu_kernels/moe_ranked.py",
    "src/llm_gpu_kernels/optimization_benchmark.py",
    "src/llm_gpu_kernels/ranked_moe_benchmark.py",
    "src/llm_gpu_kernels/reference.py",
)


def _canonical_training_hash(training_record):
    without_policy_hash = dict(training_record)
    without_policy_hash.pop("policy_sha256", None)
    encoded = json.dumps(
        without_policy_hash, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contained_path(relative_path):
    path = (ROOT / relative_path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact path escapes repository root: {relative_path}") from exc
    return path


def _record_source_hashes(policy_sources):
    """Record only explicit holdout helpers plus the policy-pinned sources."""
    paths = set(policy_sources) | set(HOLDOUT_SUPPORT_SOURCES)
    return {
        relative: hashlib.sha256(_contained_path(relative).read_bytes()).hexdigest()
        for relative in sorted(paths)
    }


def _validate_frozen_artifacts():
    """Check the frozen policy, its training record, and only its pinned sources."""
    policy_bytes = POLICY_PATH.read_bytes()
    policy_data = json.loads(policy_bytes)
    policy = FrozenMoEPolicy.load(POLICY_PATH)
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    if policy.sha256 != policy_sha256:
        raise RuntimeError("policy loader SHA256 differs from the policy file bytes")
    if policy_data.get("schema") != 1 or policy_data.get("status") != "frozen_before_holdout":
        raise RuntimeError("M7 policy is not marked frozen_before_holdout")

    expected_domain = {
        "experts": 128, "hidden": 2048, "intermediate": 768, "topk": 8,
    }
    domain = policy_data.get("domain", {})
    if any(domain.get(key) != value for key, value in expected_domain.items()):
        raise RuntimeError(f"unexpected frozen M7 policy domain: {domain!r}")
    if tuple(policy_data.get("thresholds_mu", ())) != (8.0, 64.0):
        raise RuntimeError("frozen M7 thresholds must remain [8, 64]")

    record_path_value = policy_data.get("training_record_path")
    if not isinstance(record_path_value, str) or not record_path_value:
        raise RuntimeError("frozen policy is missing training_record_path")
    training_path = _contained_path(record_path_value)
    training_data = json.loads(training_path.read_text(encoding="utf-8"))
    if training_data.get("status") != "passed":
        raise RuntimeError("M7 training record status is not passed")
    if training_data.get("policy_sha256") != policy_sha256:
        raise RuntimeError("training record is not bound to this frozen policy SHA256")
    source_run = policy_data.get("source_run")
    if (not source_run or training_data.get("run_id") != source_run
            or training_data.get("source_run", source_run) != source_run):
        raise RuntimeError("M7 policy source_run does not match its training record")
    if policy_data.get("training_record_hash_scope") != TRAINING_HASH_SCOPE:
        raise RuntimeError("unsupported M7 training-record hash scope")
    expected_training_hash = policy_data.get("training_record_sha256")
    if not expected_training_hash or _canonical_training_hash(training_data) != expected_training_hash:
        raise RuntimeError("M7 training record canonical SHA256 mismatch")

    protocol = training_data.get("training_protocol", {})
    patterns = protocol.get("routes")
    if (protocol.get("seed") != 409
            or list(protocol.get("tokens", ())) != [64, 512, 4096]
            or set(patterns or ()) != {"balanced", "hot16"}):
        raise RuntimeError("M7 record does not prove the frozen seed-409 training matrix")

    policy_sources = policy_data.get("source_hashes")
    training_sources = training_data.get("source_hashes")
    if not isinstance(policy_sources, dict) or not policy_sources:
        raise RuntimeError("M7 policy must pin the runtime source hashes")
    if training_sources != policy_sources:
        raise RuntimeError("policy and training record source hash maps differ")
    changed_sources = []
    for relative_path, expected_hash in policy_sources.items():
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise RuntimeError("invalid M7 source hash entry")
        source_path = _contained_path(relative_path)
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            changed_sources.append(relative_path)
    if changed_sources:
        raise RuntimeError("M7 pinned runtime sources changed: " + ", ".join(changed_sources))
    verification = policy.verify_provenance(ROOT)
    if not verification.get("passed"):
        raise RuntimeError("frozen policy provenance verification failed")

    return policy, {
        "path": str(POLICY_PATH.relative_to(ROOT)),
        "sha256": policy_sha256,
        "status": policy_data["status"],
        "domain": domain,
        "thresholds_mu": policy_data["thresholds_mu"],
        "configs": policy_data.get("configs"),
        "fallback": policy_data.get("fallback"),
        "source_run": source_run,
        "training_record_path": record_path_value,
        "training_record_sha256": expected_training_hash,
        "training_record_hash_scope": TRAINING_HASH_SCOPE,
        "training_status": training_data["status"],
        "training_protocol": protocol,
        "runtime_source_hashes": policy_sources,
        "verified_runtime_sources": verification["verified_sources"],
    }


def _ranked_state(prepared, buffers):
    """Read status/dispatch metadata only in a correctness diagnostic."""
    prepared.check_inputs()
    status = int(prepared.base.status.item())
    dispatch = _validate_ranked_dispatch(prepared, buffers[0], buffers[3])
    return {
        "status": status,
        "check_inputs_passed": status == 0,
        "ranked_dispatch": dispatch,
        "passed": status == 0 and dispatch.get("passed", False),
    }


def _validate_ranked_backends(backends, buffers):
    return {name: _ranked_state(backends[name], buffers)
            for name in ("ranked_default", "ranked_selected")}


def _status_and_dispatch_passed(checks):
    return all(item.get("passed", False) for item in checks.values())


def _vllm_runtime(provenance):
    module = sys.modules.get("vllm")
    if module is None:
        raise RuntimeError("vLLM public operator did not import the vLLM package")
    module_path = Path(module.__file__).resolve()
    expected_root = Path(provenance["runtime_path"]).resolve()
    try:
        module_path.relative_to(expected_root)
        matches = True
    except ValueError:
        matches = False
    runtime = {
        "version": getattr(module, "__version__", None),
        "path": str(module_path),
        "expected_runtime_path": str(expected_root),
        "matches_provenance_runtime": matches,
    }
    if not matches:
        raise RuntimeError("loaded vLLM is outside the documented isolated runtime")
    if runtime["version"] != "0.10.0":
        raise RuntimeError(f"expected official vLLM 0.10.0, found {runtime['version']}")
    return runtime


def _ranked_outputs(backends):
    return {
        "ranked_default": backends["ranked_default"].base.output,
        "ranked_selected": backends["ranked_selected"].base.output,
    }


def _m4_functional_case(spec, policy, device):
    buffers = list(make_inputs(
        tokens=spec["T"], experts=spec["E"], topk=spec["topk"], hidden=spec["H"],
        intermediate=spec["I"], id_dtype=spec["id_dtype"], pattern=spec["pattern"],
        seed=spec["seed"], device=device,
    ))
    if spec.get("signed_weights"):
        buffers[4].sub_(0.5)
    entry = _spec_metadata(spec, buffers)
    config, bucket = policy.select(spec["T"], spec["E"], spec["topk"], spec["H"], spec["I"])
    if not isinstance(config, GemmConfig):
        raise TypeError("FrozenMoEPolicy.select must return a GemmConfig")
    prepared = prepare_configured_moe(*buffers, config)
    output = prepared.run()
    prepared.check_inputs()
    torch.cuda.synchronize(device)
    oracle = moe_reference(*buffers)
    correctness = _oracle_metrics({"configured_ranked": output}, oracle)
    state = _ranked_state(prepared, buffers)
    entry.update({
        "policy_bucket": bucket,
        "selected_config": asdict(config),
        "launch_count": len(prepared.launches),
        "workspace_bytes": int(prepared.workspace_bytes),
        "correctness_full_fp32_reference": correctness,
        "status_and_dispatch": state,
        "passed": all(item["passed"] for item in correctness.values()) and state["passed"],
    })

    if spec.get("graph_updates") and spec["T"]:
        original_values = (buffers[0].clone(), buffers[3].clone(), buffers[4].clone())
        original_hash = _mutable_hash(buffers)
        graph = _capture_graph(prepared.run, 2, device)
        updates = []
        try:
            for update_index in range(2):
                seed = spec["seed"] + 401 + update_index
                _copy_mutable_values(
                    buffers, _make_mutable_update(buffers, seed, pattern="hot"),
                )
                graph.replay()
                torch.cuda.synchronize(device)
                update_oracle = moe_reference(*buffers)
                update_correctness = _oracle_metrics(
                    {"configured_ranked": prepared.base.output}, update_oracle,
                )
                update_state = _ranked_state(prepared, buffers)
                updates.append({
                    "update_index": update_index,
                    "seed": seed,
                    "mutable_input_sha256": _mutable_hash(buffers),
                    "correctness_full_fp32_reference": update_correctness,
                    "status_and_dispatch": update_state,
                    "passed": (all(item["passed"] for item in update_correctness.values())
                               and update_state["passed"]),
                })
        finally:
            _copy_mutable_values(buffers, original_values)
            torch.cuda.synchronize(device)
        restored_hash = _mutable_hash(buffers)
        graph.replay()
        torch.cuda.synchronize(device)
        restored_oracle = moe_reference(*buffers)
        restored_correctness = _oracle_metrics(
            {"configured_ranked": prepared.base.output}, restored_oracle,
        )
        entry["graph_updates"] = {
            "original_mutable_sha256": original_hash,
            "updates": updates,
            "restored_mutable_sha256": restored_hash,
            "restored_matches_original": restored_hash == original_hash,
            "restored_correctness_full_fp32_reference": restored_correctness,
            "passed": (restored_hash == original_hash
                       and all(update["passed"] for update in updates)
                       and all(item["passed"] for item in restored_correctness.values())),
        }
        entry["passed"] = entry["passed"] and entry["graph_updates"]["passed"]
        del graph
    return entry


def _selected_config_boundary(config, selected_buckets, device):
    spec = {
        "name": "selected_config_boundary_duplicate_tail",
        "T": BOUNDARY_TOKENS,
        "E": BOUNDARY_EXPERTS,
        "topk": BOUNDARY_TOPK,
        "H": BOUNDARY_HIDDEN,
        "I": BOUNDARY_INTERMEDIATE,
        "id_dtype": torch.int32,
        "pattern": "duplicate",
        "seed": 401,
    }
    buffers = list(make_inputs(
        tokens=spec["T"], experts=spec["E"], topk=spec["topk"], hidden=spec["H"],
        intermediate=spec["I"], id_dtype=spec["id_dtype"], pattern=spec["pattern"],
        seed=spec["seed"], device=device,
    ))
    entry = {
        **_spec_metadata(spec, buffers),
        "selected_config": asdict(config),
        "source_buckets": selected_buckets,
        "launch_count_expected": 7,
        "telemetry_before": telemetry(),
    }
    original_values = (buffers[0].clone(), buffers[3].clone(), buffers[4].clone())
    original_hash = _mutable_hash(buffers)
    prepared = prepare_configured_moe(*buffers, config)
    output = prepared.run()
    prepared.check_inputs()
    torch.cuda.synchronize(device)
    oracle = moe_reference(*buffers)
    correctness = _oracle_metrics({"selected_configured": output}, oracle)
    state = _ranked_state(prepared, buffers)
    if len(prepared.launches) != entry["launch_count_expected"]:
        raise RuntimeError("configured ranked path did not retain seven launches")

    graph = _capture_graph(prepared.run, GRAPH_REPEATS, device)
    updates = []
    try:
        for update_index in range(2):
            seed = spec["seed"] + 601 + update_index
            _copy_mutable_values(
                buffers, _make_mutable_update(buffers, seed, pattern="hot"),
            )
            graph.replay()
            torch.cuda.synchronize(device)
            update_oracle = moe_reference(*buffers)
            update_correctness = _oracle_metrics(
                {"selected_configured": prepared.base.output}, update_oracle,
            )
            update_state = _ranked_state(prepared, buffers)
            updates.append({
                "update_index": update_index,
                "seed": seed,
                "mutable_input_sha256": _mutable_hash(buffers),
                "correctness_full_fp32_reference": update_correctness,
                "status_and_dispatch": update_state,
                "passed": (all(item["passed"] for item in update_correctness.values())
                           and update_state["passed"]),
            })

        stream_values = _make_mutable_update(
            buffers, spec["seed"] + 603, pattern="uniform_random",
        )
        current = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            _copy_mutable_values(buffers, stream_values)
            prepared.run()
        current.wait_stream(side)
        torch.cuda.synchronize(device)
        stream_oracle = moe_reference(*buffers)
        stream_correctness = _oracle_metrics(
            {"selected_configured": prepared.base.output}, stream_oracle,
        )
        stream_state = _ranked_state(prepared, buffers)
        entry["nondefault_stream"] = {
            "stream_handle": int(side.cuda_stream),
            "mutable_input_sha256": _mutable_hash(buffers),
            "correctness_full_fp32_reference": stream_correctness,
            "status_and_dispatch": stream_state,
            "passed": (all(item["passed"] for item in stream_correctness.values())
                       and stream_state["passed"]),
        }
    finally:
        _copy_mutable_values(buffers, original_values)
        torch.cuda.synchronize(device)

    restored_hash = _mutable_hash(buffers)
    graph.replay()
    torch.cuda.synchronize(device)
    restored_oracle = moe_reference(*buffers)
    restored_correctness = _oracle_metrics(
        {"selected_configured": prepared.base.output}, restored_oracle,
    )
    entry.update({
        "correctness_full_fp32_reference": correctness,
        "initial_status_and_dispatch": state,
        "graph_repeats": GRAPH_REPEATS,
        "graph_updates": updates,
        "restored_mutable_sha256": restored_hash,
        "restored_matches_original": restored_hash == original_hash,
        "restored_correctness_full_fp32_reference": restored_correctness,
        "workspace_bytes": int(prepared.workspace_bytes),
        "telemetry_after": telemetry(),
        "passed": (
            all(item["passed"] for item in correctness.values())
            and state["passed"]
            and all(item["passed"] for item in updates)
            and entry["nondefault_stream"]["passed"]
            and restored_hash == original_hash
            and all(item["passed"] for item in restored_correctness.values())
        ),
    })
    del graph, prepared, buffers, original_values
    return entry


def _functional_validation(policy, device, record, save):
    specs = _functional_specs()
    if len(specs) != 9:
        raise RuntimeError(f"expected the nine frozen M4 functional specs, found {len(specs)}")
    failed = False
    for spec in specs:
        print("M7 functional", spec["name"], flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        entry = {"name": spec["name"], "telemetry_before": telemetry(), "passed": False}
        record["functional_cases"].append(entry)
        save()
        try:
            entry.update(_m4_functional_case(spec, policy, device))
        except Exception as exc:
            entry["passed"] = False
            entry["error"] = repr(exc)
            entry["traceback"] = traceback.format_exc()
        entry["telemetry_after"] = telemetry()
        entry["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        failed = failed or not entry["passed"]
        save()
    record["functional_validation_passed"] = not failed
    if failed:
        raise RuntimeError("one or more M7 configured M4 functional cases failed")


def _boundary_validation(policy, device, record, save):
    unique = {}
    for spec in _performance_specs():
        config, bucket = policy.select(spec["T"], spec["E"], spec["topk"], spec["H"], spec["I"])
        unique.setdefault(tuple(asdict(config).values()), {"config": config, "buckets": []})
        if bucket not in unique[tuple(asdict(config).values())]["buckets"]:
            unique[tuple(asdict(config).values())]["buckets"].append(bucket)
    if not unique:
        raise RuntimeError("M7 policy selected no configurations for the holdout domain")
    for selected in unique.values():
        torch.cuda.reset_peak_memory_stats(device)
        entry = {"name": "selected_config_boundary_duplicate_tail", "passed": False}
        record["selected_config_boundaries"].append(entry)
        save()
        try:
            completed = _selected_config_boundary(
                selected["config"], selected["buckets"], device,
            )
            entry.update(completed)
        except Exception as exc:
            entry["passed"] = False
            entry["error"] = repr(exc)
            entry["traceback"] = traceback.format_exc()
            entry["telemetry_after"] = telemetry()
            entry["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            save()
            raise
        entry["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        save()
        if not entry["passed"]:
            raise RuntimeError("selected configuration boundary validation failed")


def _timing_record(functions, flops, *, divisor=1):
    timing = measure(
        functions, flops, divisor=divisor, samples=API_SAMPLES,
        repeats=API_REPEATS,
    )
    for round_item in timing["rounds"]:
        round_item["samples_ms_per_forward"] = round_item.pop("samples_ms_per_gemm")
        round_item["wall_ms_per_forward"] = round_item.pop("wall_ms_per_gemm")
    timing["sample_unit"] = "milliseconds per complete forward"
    timing["round_order"] = "three forward cyclic and three reverse cyclic backend orders"
    return timing


def _run_performance_case(spec, policy, device, provenance, record, save):
    torch.cuda.reset_peak_memory_stats(device)
    entry = {"name": spec["name"], "telemetry_before": telemetry(), "passed": False}
    record["performance_cases"].append(entry)
    save()

    buffers = list(make_inputs(
        tokens=spec["T"], experts=spec["E"], topk=spec["topk"], hidden=spec["H"],
        intermediate=spec["I"], id_dtype=spec["id_dtype"], pattern=spec["pattern"],
        seed=spec["seed"], device=device,
    ))
    entry.update(_spec_metadata(spec, buffers))
    original_mutable = (buffers[0].clone(), buffers[3].clone(), buffers[4].clone())
    original_mutable_hash = _mutable_hash(buffers)
    config, bucket = policy.select(spec["T"], spec["E"], spec["topk"], spec["H"], spec["I"])
    if not isinstance(config, GemmConfig):
        raise TypeError("FrozenMoEPolicy.select must return a GemmConfig")

    backends = {
        "ranked_default": prepare_ranked_moe_launcher(*buffers),
        "ranked_selected": prepare_configured_moe(*buffers, config),
        "vllm_public": prepare_vllm_moe(*buffers),
    }
    if (len(backends["ranked_default"].launches) != 7
            or len(backends["ranked_selected"].launches) != 7):
        raise RuntimeError("M7 performance backends must both use the ranked seven-launch path")
    vllm_runtime = _vllm_runtime(provenance)
    entry.update({
        "policy_bucket": bucket,
        "selected_config": asdict(config),
        "default_config": asdict(backends["ranked_default"].base.config),
        "selected_is_default_config": config == backends["ranked_default"].base.config,
        "ranked_default_launches": len(backends["ranked_default"].launches),
        "ranked_selected_launches": len(backends["ranked_selected"].launches),
        "ranked_launch_contract_passed": True,
        "ranked_default_workspace_bytes": int(backends["ranked_default"].workspace_bytes),
        "ranked_selected_workspace_bytes": int(backends["ranked_selected"].workspace_bytes),
        "vllm_configuration": _json_value(backends["vllm_public"].configuration()),
        "vllm_runtime": vllm_runtime,
    })
    functions = {name: backend.run for name, backend in backends.items()}

    eager_outputs = {name: function() for name, function in functions.items()}
    torch.cuda.synchronize(device)
    oracle = moe_reference(*buffers)
    api_correctness = _oracle_metrics(eager_outputs, oracle)
    initial_states = _validate_ranked_backends(backends, buffers)
    entry["api_correctness_full_fp32_reference"] = api_correctness
    entry["api_status_dispatch_validation"] = initial_states
    if (not all(metric["passed"] for metric in api_correctness.values())
            or not _status_and_dispatch_passed(initial_states)):
        raise RuntimeError(f"M7 API correctness failed: {spec['name']}")

    graph_outputs = {}
    graphs = {}
    for name, function in functions.items():
        graphs[name] = _capture_graph(function, GRAPH_REPEATS, device)
        if name == "vllm_public":
            graph_outputs[name] = backends[name].output
        else:
            graph_outputs[name] = backends[name].base.output

    graph_updates = []
    for update_index in range(2):
        seed = spec["seed"] + 701 + update_index
        _copy_mutable_values(
            buffers, _make_mutable_update(buffers, seed, pattern="hot"),
        )
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize(device)
        update_oracle = moe_reference(*buffers)
        update_correctness = _oracle_metrics(graph_outputs, update_oracle)
        update_states = _validate_ranked_backends(backends, buffers)
        graph_updates.append({
            "update_index": update_index,
            "seed": seed,
            "mutable_input_sha256": _mutable_hash(buffers),
            "graph_correctness_full_fp32_reference": update_correctness,
            "status_dispatch_validation": update_states,
            "passed": (all(metric["passed"] for metric in update_correctness.values())
                       and _status_and_dispatch_passed(update_states)),
        })

    _copy_mutable_values(buffers, original_mutable)
    torch.cuda.synchronize(device)
    restored_mutable_hash = _mutable_hash(buffers)
    restored_input_hash = tensor_hash(buffers[0], buffers[1], buffers[2], buffers[4])
    restored_routing_hash = tensor_hash(buffers[3])
    entry["graph_input_updates"] = {
        "original_mutable_sha256": original_mutable_hash,
        "updates": graph_updates,
        "restored_mutable_sha256": restored_mutable_hash,
        "restored_input_sha256": restored_input_hash,
        "restored_routing_sha256": restored_routing_hash,
        "matches_original_hashes": (
            restored_mutable_hash == original_mutable_hash
            and restored_input_hash == entry["input_sha256"]
            and restored_routing_hash == entry["routing_sha256"]
        ),
    }
    entry["graph_input_updates"]["passed"] = (
        entry["graph_input_updates"]["matches_original_hashes"]
        and all(item["passed"] for item in graph_updates)
    )
    if not entry["graph_input_updates"]["passed"]:
        raise RuntimeError(f"M7 graph input update/restore failed: {spec['name']}")

    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize(device)
    restored_oracle = moe_reference(*buffers)
    graph_correctness = _oracle_metrics(graph_outputs, restored_oracle)
    graph_states = _validate_ranked_backends(backends, buffers)
    entry["graph_correctness_full_fp32_reference"] = graph_correctness
    entry["graph_status_dispatch_validation"] = graph_states
    if (not all(metric["passed"] for metric in graph_correctness.values())
            or not _status_and_dispatch_passed(graph_states)):
        raise RuntimeError(f"M7 graph correctness failed: {spec['name']}")

    bitwise = None
    if entry["selected_is_default_config"]:
        bitwise = torch.equal(
            backends["ranked_default"].base.output,
            backends["ranked_selected"].base.output,
        )
    entry["default_vs_selected_bitwise_diagnostic"] = (
        {"available": True, "torch_equal": bool(bitwise)}
        if bitwise is not None else {"available": False, "reason": "selected config differs from default"}
    )

    flops = int(entry["flops_gemm_only_per_forward"])
    entry["api"] = _timing_record(functions, flops)
    entry["graph"] = _timing_record(
        {name: graph.replay for name, graph in graphs.items()}, flops,
        divisor=GRAPH_REPEATS,
    )
    entry["graph_repeats_per_replay"] = GRAPH_REPEATS
    entry["graph_sample_divisor"] = GRAPH_REPEATS

    post_api_outputs = {name: function() for name, function in functions.items()}
    torch.cuda.synchronize(device)
    post_oracle = moe_reference(*buffers)
    post_api = _oracle_metrics(post_api_outputs, post_oracle)
    post_api_states = _validate_ranked_backends(backends, buffers)
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize(device)
    post_graph_oracle = moe_reference(*buffers)
    post_graph = _oracle_metrics(graph_outputs, post_graph_oracle)
    post_graph_states = _validate_ranked_backends(backends, buffers)
    entry["post_timing_api_correctness_full_fp32_reference"] = post_api
    entry["post_timing_api_status_dispatch_validation"] = post_api_states
    entry["post_timing_graph_correctness_full_fp32_reference"] = post_graph
    entry["post_timing_graph_status_dispatch_validation"] = post_graph_states
    if (not all(metric["passed"] for metric in post_api.values())
            or not all(metric["passed"] for metric in post_graph.values())
            or not _status_and_dispatch_passed(post_api_states)
            or not _status_and_dispatch_passed(post_graph_states)):
        raise RuntimeError(f"M7 post-timing correctness failed: {spec['name']}")

    entry["post_timing_mutable_sha256"] = _mutable_hash(buffers)
    if entry["post_timing_mutable_sha256"] != original_mutable_hash:
        raise RuntimeError(f"M7 timing mutated user inputs: {spec['name']}")
    entry["telemetry_after"] = telemetry()
    entry["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    entry["peak_memory_scope"] = (
        "includes input tensors, both ranked workspaces, oracle temporaries, and all captured graph pools"
    )
    entry["passed"] = True
    save()
    del graphs, graph_outputs, functions, backends, eager_outputs, oracle, buffers, original_mutable
    return entry


def run():
    run_id = (
        "m7-full-moe-holdout-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-" + str(time.time_ns() % 10**9)
    )
    destination = ROOT / "results" / (run_id + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    driver_path = Path(__file__).resolve()
    record = {
        "schema": 1,
        "run_id": run_id,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
        "experiment": "M7 frozen-config full MoE holdout",
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "holdout_driver_sha256": hashlib.sha256(driver_path.read_bytes()).hexdigest(),
        "limits": LIMITS,
        "api_samples_per_round": API_SAMPLES,
        "api_repeats_per_sample": API_REPEATS,
        "cyclic_rounds": "three forward cyclic and three reverse cyclic backend orders",
        "graph_repeats": GRAPH_REPEATS,
        "graph_divisor": GRAPH_REPEATS,
        "timing_scope": {
            "api": "complete fixed-route forward; setup, compile, oracle, dispatch readback and correctness excluded",
            "graph": "complete forward CUDA Graph replay normalized by eight forwards; setup/capture excluded",
            "backends": ["original ranked 7-launch", "frozen-policy configured ranked 7-launch", "official vLLM 0.10.0 public API"],
            "flops": "6 * T * topk * H * I per complete forward; this is a GEMM-only operation count",
        },
        "scope": "fixed-route synthetic full MoE operator; independent of NanoKV, model weights/quality, routing model, and serving",
        "correctness": "each implementation checked against full FP32 oracle at relative_rms and normalized_max <= 0.008; no bitwise requirement across configurations",
        "telemetry_before": telemetry(),
        "functional_cases": [],
        "selected_config_boundaries": [],
        "performance_cases": [],
    }

    def save():
        destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        policy, policy_info = _validate_frozen_artifacts()
        record["policy"] = policy_info
        record["source_hashes"] = _record_source_hashes(
            policy_info["runtime_source_hashes"],
        )
        provenance = json.loads(VLLM_PROVENANCE_PATH.read_text(encoding="utf-8"))
        record["libraries"] = _runtime_metadata(provenance)
        save()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the M7 full-MoE holdout")
        if triton.__version__ != "3.3.1":
            raise RuntimeError(f"M7 fixed launchers require Triton 3.3.1, got {triton.__version__}")
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cudnn.allow_tf32 = False

        device = torch.device("cuda", torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device)
        record["runtime"] = {
            "python": {"version": platform.python_version(), "executable": sys.executable},
            "numpy": {"version": np.__version__, "path": str(Path(np.__file__).resolve())},
            "torch": {"version": torch.__version__, "path": str(Path(torch.__file__).resolve()),
                      "cuda_runtime": torch.version.cuda},
            "triton": {"version": triton.__version__, "path": str(Path(triton.__file__).resolve())},
            "gpu": props.name,
            "sm": [props.major, props.minor],
            "memory_bytes": props.total_memory,
            "device_index": device.index,
            "device_count": torch.cuda.device_count(),
            "pythonpath": os.environ.get("PYTHONPATH"),
        }
        save()

        _functional_validation(policy, device, record, save)
        _boundary_validation(policy, device, record, save)

        specs = _performance_specs()
        if len(specs) != 15:
            raise RuntimeError(f"expected exactly the M5 15-case matrix, found {len(specs)}")
        record["performance_spec_names"] = [spec["name"] for spec in specs]
        save()
        for spec in specs:
            print("M7 performance", spec["name"], flush=True)
            try:
                _run_performance_case(spec, policy, device, provenance, record, save)
            except Exception as exc:
                entry = record["performance_cases"][-1]
                entry["passed"] = False
                entry["error"] = repr(exc)
                entry["traceback"] = traceback.format_exc()
                entry.setdefault("telemetry_after", telemetry())
                entry.setdefault("peak_allocated_bytes", int(torch.cuda.max_memory_allocated(device)))
                save()
                raise
        if hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() != policy.sha256:
            raise RuntimeError("frozen M7 policy file changed during holdout")
        verification = policy.verify_provenance(ROOT)
        if not verification.get("passed"):
            raise RuntimeError("frozen M7 provenance changed during holdout")
        record["policy_provenance_after_holdout"] = verification
        record["status"] = "passed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
        raise
    finally:
        record["telemetry_after"] = telemetry()
        save()
    print("Saved", destination, flush=True)
    return destination


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()


if __name__ == "__main__":
    main()
