"""M6 fixed experiment for cached fused and ranked-dispatch MoE forwards.

The preregistered matrix and interpretation limits are in
``docs/M6_RANKED_DISPATCH_PLAN.md``. This driver does not call vLLM. It reuses
the mature M5 driver's read-only case/update helpers and the shared M4 timing
helper; the compared runtime paths are the cached fused and ranked launchers.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import torch
import triton

from .benchmark import LIMITS, ROOT
from .mature_moe_benchmark import (
    _capture_graph,
    _copy_mutable_values,
    _make_mutable_update,
    _mutable_hash,
    _oracle_metrics,
    _spec_metadata,
)
from .moe_benchmark import (
    _functional_specs,
    make_inputs,
    telemetry,
    validate_dispatch,
)
from .moe_prepared import prepare_moe_launcher
from .moe_ranked import prepare_ranked_moe_launcher
from .optimization_benchmark import measure
from .reference import moe_reference


API_SAMPLES = 8
API_REPEATS = 2
GRAPH_REPEATS = 8
FUNCTIONAL_DUPLICATE_REPEATS = 3


def _source_hashes():
    """Hash source and contract files that can affect preparation or checks."""
    paths = (
        ROOT / "src/llm_gpu_kernels/ranked_moe_benchmark.py",
        ROOT / "src/llm_gpu_kernels/benchmark.py",
        ROOT / "src/llm_gpu_kernels/gemm.py",
        ROOT / "src/llm_gpu_kernels/reference.py",
        ROOT / "src/llm_gpu_kernels/moe.py",
        ROOT / "src/llm_gpu_kernels/moe_fused.py",
        ROOT / "src/llm_gpu_kernels/moe_prepared.py",
        ROOT / "src/llm_gpu_kernels/moe_ranked.py",
        ROOT / "src/llm_gpu_kernels/moe_benchmark.py",
        ROOT / "src/llm_gpu_kernels/mature_moe_benchmark.py",
        ROOT / "src/llm_gpu_kernels/optimization_benchmark.py",
        ROOT / "src/llm_gpu_kernels/baselines/vllm_moe.py",
        ROOT / "docs/M6_RANKED_DISPATCH_PLAN.md",
        ROOT / "docs/CONTRACT.md",
        ROOT / "docs/M4_FUSION_EXPERIMENT_PLAN.md",
        ROOT / "docs/M4_MOE_REPORT.md",
        ROOT / "docs/M5_EXPERIMENT_PLAN.md",
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }


def _functional_specs_fixed():
    """Return the nine frozen M4 cases without introducing a second matrix."""
    specs = _functional_specs()
    if len(specs) != 9:
        raise RuntimeError(f"expected all 9 M4 functional cases, found {len(specs)}")
    return specs


def _performance_specs():
    specs = []
    for tokens in (1, 17, 193):
        specs.append({
            "name": f"small_E16_T{tokens}_random_seed73",
            "T": tokens, "E": 16, "topk": 4, "H": 128, "I": 64,
            "id_dtype": torch.int32, "pattern": "uniform_random", "seed": 73,
        })
    for seed in (73, 991):
        for tokens in (1, 17, 193, 1024, 2049):
            specs.append({
                "name": f"qwen3_30b_a3b_T{tokens}_random_seed{seed}",
                "T": tokens, "E": 128, "topk": 8, "H": 2048, "I": 768,
                "id_dtype": torch.int32, "pattern": "uniform_random", "seed": seed,
            })
    for tokens in (193, 1024):
        specs.append({
            "name": f"qwen3_30b_a3b_T{tokens}_hot16_seed73",
            "T": tokens, "E": 128, "topk": 8, "H": 2048, "I": 768,
            "id_dtype": torch.int32, "pattern": "hot", "seed": 73,
        })
    if len(specs) != 15:
        raise RuntimeError(f"expected 15 preregistered performance cases, found {len(specs)}")
    return specs


def _prepare_pair(buffers):
    """Prepare both entry points over the identical user tensor objects."""
    cached = prepare_moe_launcher(*buffers, fused=True)
    ranked = prepare_ranked_moe_launcher(*buffers)
    return {"cached_fused": cached, "ranked": ranked}


def _prepared_output(prepared):
    return prepared.base.output


def _validate_ranked_dispatch(ranked, x, ids):
    """Check ranked indices, map destinations, segments and copied activations."""
    base = ranked.base
    base_validation = validate_dispatch(base, x, ids)
    flat_ids = ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist()
    ranks = ranked.ranks.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist()
    mapping = base.mapping.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist()
    experts = int(base.counts.numel())
    counts = [0] * experts
    ranks_by_expert = [[] for _ in range(experts)]
    offsets = [0] * experts
    for expert in flat_ids:
        counts[expert] += 1
    cursor = 0
    for expert, count in enumerate(counts):
        offsets[expert] = cursor
        cursor += count

    rank_to_mapping = (
        len(ranks) == len(flat_ids) == len(mapping)
        and all(mapping[q] == offsets[expert] + ranks[q]
                for q, expert in enumerate(flat_ids))
    )
    ranks_in_range = True
    for expert, rank in zip(flat_ids, ranks):
        ranks_by_expert[expert].append(rank)
        ranks_in_range = ranks_in_range and 0 <= rank < counts[expert]
    ranks_unique_contiguous = all(
        sorted(expert_ranks) == list(range(count))
        for expert_ranks, count in zip(ranks_by_expert, counts)
    )
    checks = {
        "base_dispatch_and_grouped_input": base_validation,
        "rank_shape_matches_routes": len(ranks) == len(flat_ids),
        "rank_values_in_expert_ranges": ranks_in_range,
        "rank_values_unique_and_contiguous_per_expert": ranks_unique_contiguous,
        "map_equals_offset_plus_rank": rank_to_mapping,
    }
    checks["passed"] = (
        base_validation.get("passed", False)
        and checks["rank_shape_matches_routes"]
        and ranks_in_range
        and ranks_unique_contiguous
        and rank_to_mapping
    )
    checks["expected_counts"] = counts
    checks["rank_values_by_expert"] = ranks_by_expert
    return checks


def _launch_workspace_contract(prepared, tokens, route_count):
    cached = prepared["cached_fused"]
    ranked = prepared["ranked"]
    cached_launches = len(cached.launches)
    ranked_launches = len(ranked.launches)
    cached_workspace = int(cached.base.workspace_bytes)
    ranked_workspace = int(ranked.workspace_bytes)
    rank_workspace = int(ranked.ranks.numel() * ranked.ranks.element_size())
    expected = {
        "cached_launches": 0 if tokens == 0 else 8,
        "ranked_launches": 0 if tokens == 0 else 7,
        "workspace_delta_bytes": 4 * route_count,
    }
    actual = {
        "cached_launches": cached_launches,
        "ranked_launches": ranked_launches,
        "cached_workspace_bytes": cached_workspace,
        "ranked_workspace_bytes": ranked_workspace,
        "rank_workspace_bytes": rank_workspace,
        "workspace_delta_bytes": ranked_workspace - cached_workspace,
    }
    actual["passed"] = (
        cached_launches == expected["cached_launches"]
        and ranked_launches == expected["ranked_launches"]
        and actual["workspace_delta_bytes"] == expected["workspace_delta_bytes"]
        and rank_workspace == expected["workspace_delta_bytes"]
    )
    actual["expected"] = expected
    return actual


def _compare_current_outputs(prepared, buffers, device):
    """Compare outputs currently held by prepared objects to the FP32 oracle."""
    torch.cuda.synchronize(device)
    statuses = {}
    for name, instance in prepared.items():
        instance.check_inputs()
        statuses[name] = int(instance.base.status.item())
    oracle = moe_reference(*buffers)
    outputs = {name: _prepared_output(instance) for name, instance in prepared.items()}
    correctness = _oracle_metrics(outputs, oracle)
    bitwise = {
        "left": "cached_fused",
        "right": "ranked",
        "torch_equal": bool(torch.equal(outputs["cached_fused"], outputs["ranked"])),
    }
    bitwise["passed"] = bitwise["torch_equal"]
    dispatch = {
        "cached_fused": validate_dispatch(
            prepared["cached_fused"].base, buffers[0], buffers[3],
        ),
        "ranked": _validate_ranked_dispatch(
            prepared["ranked"], buffers[0], buffers[3],
        ),
    }
    passed = (
        all(item["passed"] for item in correctness.values())
        and bitwise["passed"]
        and all(item.get("passed", False) for item in dispatch.values())
        and all(status == 0 for status in statuses.values())
    )
    return {
        "correctness_full_fp32_reference": correctness,
        "cached_fused_vs_ranked_bitwise": bitwise,
        "dispatch_validation": dispatch,
        "status_flags": statuses,
        "passed": passed,
    }


def _run_pair(prepared, device):
    for instance in prepared.values():
        instance.run()
    torch.cuda.synchronize(device)


def _validate_functional_graph_updates(buffers, spec, prepared, device,
                                       original_mutable, original_hash):
    graphs = {
        name: _capture_graph(instance.run, 2, device)
        for name, instance in prepared.items()
    }
    updates = []
    try:
        for update_index in range(2):
            seed = spec["seed"] + 401 + update_index
            values = _make_mutable_update(buffers, seed, pattern="hot")
            _copy_mutable_values(buffers, values)
            updated_hash = _mutable_hash(buffers)
            for graph in graphs.values():
                graph.replay()
            check = _compare_current_outputs(prepared, buffers, device)
            updates.append({
                "update_index": update_index,
                "seed": seed,
                "mutable_input_sha256": updated_hash,
                **check,
            })
    finally:
        _copy_mutable_values(buffers, original_mutable)
        torch.cuda.synchronize(device)

    restored_hash = _mutable_hash(buffers)
    for graph in graphs.values():
        graph.replay()
    restored = _compare_current_outputs(prepared, buffers, device)
    del graphs
    return {
        "updates": updates,
        "original_mutable_sha256": original_hash,
        "restored_mutable_sha256": restored_hash,
        "restored_hash_matches": restored_hash == original_hash,
        "restored_graph_result": restored,
        "passed": (
            restored_hash == original_hash
            and restored["passed"]
            and all(item["passed"] for item in updates)
        ),
    }


def _validate_nondefault_stream(buffers, spec, prepared, device,
                                original_mutable, original_hash):
    values = _make_mutable_update(
        buffers, spec["seed"] + 503, pattern="uniform_random",
    )
    current = torch.cuda.current_stream(device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        _copy_mutable_values(buffers, values)
        for instance in prepared.values():
            instance.run()
    current.wait_stream(side)
    torch.cuda.synchronize(device)
    mutable_hash = _mutable_hash(buffers)
    check = _compare_current_outputs(prepared, buffers, device)

    _copy_mutable_values(buffers, original_mutable)
    torch.cuda.synchronize(device)
    restored_hash = _mutable_hash(buffers)
    return {
        "stream_handle": int(side.cuda_stream),
        "mutable_input_sha256": mutable_hash,
        "original_mutable_sha256": original_hash,
        "restored_mutable_sha256": restored_hash,
        "restored_hash_matches": restored_hash == original_hash,
        **check,
        "passed": check["passed"] and restored_hash == original_hash,
    }


def _run_functional_case(spec, device):
    buffers = list(make_inputs(
        tokens=spec["T"], experts=spec["E"], topk=spec["topk"], hidden=spec["H"],
        intermediate=spec["I"], id_dtype=spec["id_dtype"], pattern=spec["pattern"],
        seed=spec["seed"], device=device,
    ))
    if spec.get("signed_weights"):
        buffers[4].sub_(0.5)

    entry = _spec_metadata(spec, buffers)
    original_mutable = (
        buffers[0].clone(), buffers[3].clone(), buffers[4].clone(),
    )
    original_hash = _mutable_hash(buffers)
    prepared = _prepare_pair(buffers)
    _run_pair(prepared, device)
    initial = _compare_current_outputs(prepared, buffers, device)
    launch_workspace = _launch_workspace_contract(
        prepared, spec["T"], spec["T"] * spec["topk"],
    )

    entry.update({
        "initial_validation": initial,
        "launch_workspace_contract": launch_workspace,
        "duplicate_repeats": [],
        "graph_updates": None,
        "nondefault_stream": None,
    })
    if spec["pattern"] == "duplicate":
        for repeat_index in range(FUNCTIONAL_DUPLICATE_REPEATS):
            _run_pair(prepared, device)
            entry["duplicate_repeats"].append({
                "repeat_index": repeat_index,
                **_compare_current_outputs(prepared, buffers, device),
            })

    graph_passed = True
    stream_passed = True
    if spec.get("graph_updates"):
        entry["graph_updates"] = _validate_functional_graph_updates(
            buffers, spec, prepared, device, original_mutable, original_hash,
        )
        graph_passed = entry["graph_updates"]["passed"]
        entry["nondefault_stream"] = _validate_nondefault_stream(
            buffers, spec, prepared, device, original_mutable, original_hash,
        )
        stream_passed = entry["nondefault_stream"]["passed"]

    entry["passed"] = (
        initial["passed"]
        and launch_workspace["passed"]
        and all(item["passed"] for item in entry["duplicate_repeats"])
        and graph_passed
        and stream_passed
    )
    return entry


def _validate_bad_runtime_inputs(device):
    """Both entry points must set the same status and raise in debug checks."""
    cases = []
    for name, bad_kind, expected_status, seed in (
        ("out_of_range_id", "id", 1, 101),
        ("nonfinite_routing_weight", "weight", 2, 103),
    ):
        buffers = list(make_inputs(
            tokens=1, experts=4, topk=1, hidden=17, intermediate=9,
            id_dtype=torch.int32, pattern="uniform_random", seed=seed, device=device,
        ))
        if bad_kind == "id":
            buffers[3][0, 0] = 4
        else:
            buffers[4][0, 0] = float("nan")

        prepared = _prepare_pair(buffers)
        _run_pair(prepared, device)
        result = {"name": name, "expected_status": expected_status, "implementations": {}}
        for implementation, instance in prepared.items():
            try:
                instance.check_inputs()
            except ValueError as exc:
                raised = True
                reason = str(exc)
            else:
                raised = False
                reason = "check_inputs did not raise"
            status = int(instance.base.status.item())
            result["implementations"][implementation] = {
                "raised_value_error": raised,
                "actual_status": status,
                "reason": reason,
                "passed": raised and status == expected_status,
            }
        result["passed"] = all(
            item["passed"] for item in result["implementations"].values()
        )
        cases.append(result)
    return cases


def _timing_record(functions, flops, *, samples, repeats, divisor=1):
    timing = measure(
        functions, flops, divisor=divisor, samples=samples, repeats=repeats,
    )
    for round_item in timing["rounds"]:
        round_item["samples_ms_per_forward"] = round_item.pop("samples_ms_per_gemm")
        round_item["wall_ms_per_forward"] = round_item.pop("wall_ms_per_gemm")
    timing["sample_unit"] = "milliseconds per complete forward"
    timing["round_order"] = "AB / BA / BA / AB from optimization_benchmark.measure"
    timing["raw_samples_note"] = (
        "CUDA event samples and wall samples are retained for every implementation "
        "in every round; wall includes event synchronization and host gaps."
    )
    return timing


def _run_performance_case(spec, device, *, samples, repeats, graph_repeats,
                         record, save):
    torch.cuda.reset_peak_memory_stats(device)
    entry = {
        "name": spec["name"],
        "telemetry_before": telemetry(),
        "passed": False,
    }
    record["performance_cases"].append(entry)
    save()

    buffers = list(make_inputs(
        tokens=spec["T"], experts=spec["E"], topk=spec["topk"], hidden=spec["H"],
        intermediate=spec["I"], id_dtype=spec["id_dtype"], pattern=spec["pattern"],
        seed=spec["seed"], device=device,
    ))
    entry.update(_spec_metadata(spec, buffers))
    original_mutable = (
        buffers[0].clone(), buffers[3].clone(), buffers[4].clone(),
    )
    original_mutable_hash = _mutable_hash(buffers)
    prepared = _prepare_pair(buffers)
    entry["launch_workspace_contract"] = _launch_workspace_contract(
        prepared, spec["T"], spec["T"] * spec["topk"],
    )
    if not entry["launch_workspace_contract"]["passed"]:
        raise RuntimeError(f"launch/workspace contract failed: {spec['name']}")

    entry["pre_timing_api_validation"] = None
    _run_pair(prepared, device)
    entry["pre_timing_api_validation"] = _compare_current_outputs(
        prepared, buffers, device,
    )
    if not entry["pre_timing_api_validation"]["passed"]:
        raise RuntimeError(f"pre-timing API correctness failed: {spec['name']}")

    graphs = {
        name: _capture_graph(instance.run, graph_repeats, device)
        for name, instance in prepared.items()
    }
    graph_validation = _compare_current_outputs(prepared, buffers, device)
    entry["pre_timing_graph_validation"] = graph_validation
    if not graph_validation["passed"]:
        raise RuntimeError(f"pre-timing graph correctness failed: {spec['name']}")

    restored_before_timing = _mutable_hash(buffers)
    entry["common_input_restore"] = {
        "original_mutable_sha256": original_mutable_hash,
        "restored_mutable_sha256": restored_before_timing,
        "matches_original": restored_before_timing == original_mutable_hash,
    }
    if restored_before_timing != original_mutable_hash:
        raise RuntimeError(f"input values changed before timing: {spec['name']}")

    functions = {name: instance.run for name, instance in prepared.items()}
    flops = int(entry["flops_gemm_only_per_forward"])
    entry["api"] = _timing_record(
        functions, flops, samples=samples, repeats=repeats,
    )
    entry["graph"] = _timing_record(
        {name: graph.replay for name, graph in graphs.items()}, flops,
        samples=samples, repeats=repeats, divisor=graph_repeats,
    )
    entry["graph_repeats_per_replay"] = graph_repeats
    entry["graph_sample_divisor"] = graph_repeats

    after_timing_hash = _mutable_hash(buffers)
    entry["common_input_restore"].update({
        "post_timing_mutable_sha256": after_timing_hash,
        "post_timing_matches_original": after_timing_hash == original_mutable_hash,
    })
    if after_timing_hash != original_mutable_hash:
        raise RuntimeError(f"input values changed during timing: {spec['name']}")

    _run_pair(prepared, device)
    entry["post_timing_api_validation"] = _compare_current_outputs(
        prepared, buffers, device,
    )
    for graph in graphs.values():
        graph.replay()
    entry["post_timing_graph_validation"] = _compare_current_outputs(
        prepared, buffers, device,
    )
    if not (entry["post_timing_api_validation"]["passed"]
            and entry["post_timing_graph_validation"]["passed"]):
        raise RuntimeError(f"post-timing correctness failed: {spec['name']}")

    entry["telemetry_after"] = telemetry()
    entry["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    entry["peak_memory_scope"] = (
        "includes input tensors, both prepared workspaces, graph pools, and correctness oracle"
    )
    entry["passed"] = True
    save()
    del graphs, functions, prepared, buffers, original_mutable
    return entry


def run(samples=API_SAMPLES, repeats=API_REPEATS, graph_repeats=GRAPH_REPEATS):
    if samples < 1 or repeats < 1 or graph_repeats < 1:
        raise ValueError("samples, repeats, and graph_repeats must be positive")

    run_id = (
        "m6-ranked-dispatch-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-" + str(time.time_ns() % 10**9)
    )
    destination = ROOT / "results" / (run_id + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    plan_path = ROOT / "docs/M6_RANKED_DISPATCH_PLAN.md"
    record = {
        "schema": 1,
        "run_id": run_id,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
        "experiment": "M6 cached fused versus ranked dispatch",
        "experiment_plan_path": str(plan_path.relative_to(ROOT)),
        "experiment_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "source_hashes": _source_hashes(),
        "limits": LIMITS,
        "functional_cases": [],
        "invalid_runtime_input_checks": [],
        "performance_cases": [],
        "api_samples_per_round": samples,
        "api_repeats_per_sample": repeats,
        "balanced_timing_rounds": "AB / BA / BA / AB",
        "graph_repeats": graph_repeats,
        "timing_scope": {
            "api": "complete cached-launcher run, CUDA events over Python launches, host gaps included",
            "graph": "complete forward graph replay normalized by graph_repeats; graph submit gaps may remain",
            "setup": "prepare, compile, warmup, correctness oracle, dispatch readback excluded",
        },
        "comparison": "same fixed-route inputs; FP32 oracle at 0.008/0.008 and cached_fused/ranked exact equality",
        "telemetry_before": telemetry(),
    }

    def save():
        destination.write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the M6 ranked-dispatch experiment")
        if triton.__version__ != "3.3.1":
            raise RuntimeError(
                f"M6 cached launchers require Triton 3.3.1, got {triton.__version__}"
            )
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cudnn.allow_tf32 = False

        device = torch.device("cuda", torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device)
        record["runtime"] = {
            "python": {"version": platform.python_version(), "executable": sys.executable},
            "torch": {"version": torch.__version__, "cuda_runtime": torch.version.cuda},
            "triton": {"version": triton.__version__},
            "gpu": properties.name,
            "sm": [properties.major, properties.minor],
            "memory_bytes": properties.total_memory,
            "device_index": device.index,
            "device_count": torch.cuda.device_count(),
        }
        record["tf32"] = False
        record["bf16_reduced_precision_reduction"] = False
        save()

        functional_failed = False
        for spec in _functional_specs_fixed():
            print("M6 functional", spec["name"], flush=True)
            entry = {
                "name": spec["name"],
                "telemetry_before": telemetry(),
                "passed": False,
            }
            record["functional_cases"].append(entry)
            save()
            try:
                completed = _run_functional_case(spec, device)
                entry.update(completed)
                entry["telemetry_after"] = telemetry()
                functional_failed = functional_failed or not entry["passed"]
            except Exception as exc:
                entry["passed"] = False
                entry["error"] = repr(exc)
                entry["traceback"] = traceback.format_exc()
                entry["telemetry_after"] = telemetry()
                functional_failed = True
            save()

        record["invalid_runtime_input_checks"] = _validate_bad_runtime_inputs(device)
        record["invalid_runtime_inputs_passed"] = all(
            case["passed"] for case in record["invalid_runtime_input_checks"]
        )
        functional_failed = functional_failed or not record["invalid_runtime_inputs_passed"]
        record["functional_validation_passed"] = not functional_failed
        save()
        if functional_failed:
            raise RuntimeError(
                "M6 functional validation failed; performance matrix was not started"
            )

        for spec in _performance_specs():
            print("M6 performance", spec["name"], flush=True)
            try:
                _run_performance_case(
                    spec, device, samples=samples, repeats=repeats,
                    graph_repeats=graph_repeats, record=record, save=save,
                )
            except Exception as exc:
                entry = record["performance_cases"][-1]
                entry["passed"] = False
                entry["error"] = repr(exc)
                entry["traceback"] = traceback.format_exc()
                entry.setdefault("telemetry_after", telemetry())
                entry.setdefault("peak_allocated_bytes", int(torch.cuda.max_memory_allocated(device)))
                save()
                raise

        record["status"] = "passed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
        raise
    finally:
        try:
            record["telemetry_after"] = telemetry()
        except Exception as exc:
            record["telemetry_after_error"] = repr(exc)
        save()
    print("Saved", destination, flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=API_SAMPLES)
    parser.add_argument("--repeats", type=int, default=API_REPEATS)
    parser.add_argument("--graph-repeats", type=int, default=GRAPH_REPEATS)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1 or args.graph_repeats < 1:
        parser.error("samples, repeats, and graph-repeats must be positive")
    run(args.samples, args.repeats, args.graph_repeats)


if __name__ == "__main__":
    main()
