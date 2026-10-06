"""M4 fixed-route MoE correctness and full-forward timing driver.

This is a functional baseline for the in-repository PreparedMoE implementation.
The expert-loop FP32 reference is used for correctness only; it is not timed or
presented as a mature MoE backend comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .benchmark import ROOT, LIMITS, command, compile_artifacts, error, measure_pair, tensor_hash
from .reference import moe_reference
from .moe import prepare_moe


SEED = 20261006
CONFIG = {"block_m": 32, "block_n": 64, "block_k": 32, "warps": 4, "stages": 3}
TIMING_SAMPLES = 5
TIMING_REPEATS = 1
GRAPH_REPEATS = 8


def telemetry():
    return command([
        "nvidia-smi", "--query-gpu=driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem",
        "--format=csv,noheader",
    ])


def source_hashes():
    paths = sorted((ROOT / "src").rglob("*.py"))
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def make_ids(tokens, experts, topk, pattern, generator):
    """Generate deterministic fixed routes on CPU before copying to the GPU."""
    if topk < 1 or topk > experts:
        raise ValueError("topk must be in [1, experts]")
    if tokens == 0:
        return torch.empty((0, topk), dtype=torch.int64)
    if pattern == "duplicate":
        base = torch.arange(tokens, dtype=torch.int64)[:, None] % experts
        slots = torch.arange(topk, dtype=torch.int64)[None, :]
        # Each token has repeated IDs, while different tokens exercise different experts.
        return (base + torch.div(slots, 2, rounding_mode="floor")) % experts
    if pattern == "all_hot":
        return torch.zeros((tokens, topk), dtype=torch.int64)
    if pattern == "hot":
        hot_count = min(experts, max(topk, experts // 8))
        hot_experts = torch.randperm(experts, generator=generator)[:hot_count]
        choices = torch.rand((tokens, hot_count), generator=generator)
        return hot_experts[choices.topk(topk, dim=1).indices]
    if pattern in ("random", "uniform_random"):
        scores = torch.rand((tokens, experts), generator=generator)
        return scores.topk(topk, dim=1).indices
    raise ValueError(f"unknown route pattern: {pattern}")


def make_inputs(*, tokens, experts, topk, hidden, intermediate, id_dtype,
                pattern, seed, device):
    if id_dtype not in (torch.int32, torch.int64):
        raise ValueError("expert ID dtype must be int32 or int64")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((tokens, hidden), generator=generator).to(device=device, dtype=torch.bfloat16)
    gate_up = torch.randn((experts, 2 * intermediate, hidden), generator=generator)
    gate_up = gate_up.to(device=device, dtype=torch.bfloat16)
    down = torch.randn((experts, hidden, intermediate), generator=generator)
    down = down.to(device=device, dtype=torch.bfloat16)
    ids = make_ids(tokens, experts, topk, pattern, generator).to(device=device, dtype=id_dtype)
    # Finite, non-normalized routing coefficients: MoE reference semantics do not
    # apply softmax or renormalize them.
    routing_weights = torch.rand((tokens, topk), generator=generator, dtype=torch.float32)
    routing_weights = routing_weights.to(device=device)
    return x, gate_up, down, ids, routing_weights


def staged_bf16_oracle(x, gate_up, down, expert_ids, routing_weights):
    """Independent PyTorch oracle with the M4 BF16 stage boundaries.

    BF16 gate/up projection -> FP32 SwiGLU -> BF16 activation -> BF16 down
    projection -> FP32 route weighting/reduction -> BF16 output.
    """
    tokens, hidden = x.shape
    topk = expert_ids.shape[1]
    intermediate = down.shape[2]
    if tokens == 0:
        return torch.empty((0, hidden), dtype=torch.bfloat16, device=x.device)

    # Work expert by expert so the Qwen-sized validation never materializes
    # route copies of all [E,2I,H] and [E,H,I] weights (which would be many GB).
    route_values = torch.zeros((tokens, topk, hidden), dtype=torch.bfloat16, device=x.device)
    for expert in range(gate_up.shape[0]):
        token_rows, slots = torch.where(expert_ids == expert)
        if token_rows.numel() == 0:
            continue
        projection = x.index_select(0, token_rows) @ gate_up[expert].T
        gate = projection[:, :intermediate].float()
        up = projection[:, intermediate:].float()
        activation = (F.silu(gate) * up).to(torch.bfloat16)
        down_result = activation @ down[expert].T
        route_values[token_rows, slots] = down_result
    weighted = route_values.float() * routing_weights.unsqueeze(-1).float()
    return weighted.sum(dim=1).to(torch.bfloat16)


def safe_error(actual, oracle):
    """Use the repository thresholds while handling an empty T dimension."""
    if actual.shape != oracle.shape:
        raise ValueError(f"output shape mismatch: {tuple(actual.shape)} != {tuple(oracle.shape)}")
    if actual.numel() == 0:
        return {"passed": True, "empty": True, "finite": True,
                "relative_rms": 0.0, "normalized_max": 0.0, "max_abs": 0.0}
    return error(actual, oracle)


def compare_output(prepared, x, gate_up, down, ids, routing_weights):
    output = prepared.run()
    torch.cuda.synchronize(x.device)
    reference = moe_reference(x, gate_up, down, ids, routing_weights)
    staged = staged_bf16_oracle(x, gate_up, down, ids, routing_weights)
    metrics = {
        "fp32_reference": safe_error(output, reference),
        "staged_bf16_oracle": safe_error(output, staged),
        "output_dtype": str(output.dtype),
        "output_shape": list(output.shape),
    }
    metrics["passed"] = all(metrics[name]["passed"] for name in ("fp32_reference", "staged_bf16_oracle"))
    return output, metrics


def validate_dispatch(prepared, x, ids):
    """Read back routing metadata only in this correctness/diagnostic path."""
    tokens, topk = ids.shape
    experts = prepared.counts.numel()
    flat_ids = ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    expected_counts = torch.bincount(flat_ids, minlength=experts).tolist()
    expected_offsets = []
    cursor = 0
    for count in expected_counts:
        expected_offsets.append(cursor)
        cursor += count

    actual_counts = prepared.counts.detach().to(device="cpu", dtype=torch.int64).tolist()
    actual_offsets = prepared.offsets.detach().to(device="cpu", dtype=torch.int64).tolist()
    mapping = prepared.mapping.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    rows = tokens * topk
    checks = {
        "counts_match_cpu_histogram": actual_counts == expected_counts,
        "offsets_match_exclusive_prefix": actual_offsets == expected_offsets,
        "mapping_shape": mapping.numel() == rows,
        "mapping_is_permutation": sorted(mapping.tolist()) == list(range(rows)),
        "grouped_input_shape": tuple(prepared.grouped_input.shape) == (rows, x.shape[1]),
    }
    if checks["mapping_shape"] and checks["offsets_match_exclusive_prefix"]:
        for flat_route, grouped_row in enumerate(mapping.tolist()):
            expert = int(flat_ids[flat_route])
            begin = expected_offsets[expert]
            if not begin <= grouped_row < begin + expected_counts[expert]:
                checks["mapping_stays_in_expert_segment"] = False
                break
        else:
            checks["mapping_stays_in_expert_segment"] = True
    else:
        checks["mapping_stays_in_expert_segment"] = False

    if checks["mapping_shape"] and checks["grouped_input_shape"]:
        if rows:
            token_rows = torch.arange(tokens, device=x.device).repeat_interleave(topk)
            mapped_rows = prepared.mapping.to(device=x.device, dtype=torch.int64)
            gathered = prepared.grouped_input.index_select(0, mapped_rows)
            checks["inverse_gather_matches_original_tokens"] = torch.equal(gathered, x.index_select(0, token_rows))
        else:
            checks["inverse_gather_matches_original_tokens"] = True
    else:
        checks["inverse_gather_matches_original_tokens"] = False

    checks["passed"] = all(checks.values())
    checks["expected_counts"] = expected_counts
    checks["actual_counts"] = actual_counts
    checks["expected_offsets"] = expected_offsets
    checks["actual_offsets"] = actual_offsets
    return checks


def validate_buffer_aliasing(prepared, user_buffers):
    """Check the public output and prepared workspace own separate storage."""
    names = (
        "output", "counts", "offsets", "cursors", "tasks", "mapping",
        "grouped_input", "projected", "activated", "down_output", "status",
    )
    workspace = [(name, getattr(prepared, name)) for name in names]
    input_storage = {tensor.untyped_storage()._cdata for tensor in user_buffers}
    workspace_storage = [tensor.untyped_storage()._cdata for _, tensor in workspace]
    return {
        "workspace_distinct_from_inputs": all(storage not in input_storage for storage in workspace_storage),
        "workspace_storages_distinct": len(set(workspace_storage)) == len(workspace_storage),
        "output_distinct_from_inputs": prepared.output.untyped_storage()._cdata not in input_storage,
    }


def validate_bad_runtime_inputs(device):
    """Exercise the explicit synchronized debug validator, outside hot timings."""
    cases = []
    for name, bad_kind, expected_status in (
        ("out_of_range_id", "id", 1),
        ("nonfinite_routing_weight", "weight", 2),
    ):
        buffers = list(make_inputs(
            tokens=1, experts=4, topk=1, hidden=17, intermediate=9,
            id_dtype=torch.int32, pattern="random", seed=101 if bad_kind == "id" else 103,
            device=device,
        ))
        if bad_kind == "id":
            buffers[3][0, 0] = 4
        else:
            buffers[4][0, 0] = float("nan")
        prepared = prepare_moe(*buffers)
        try:
            prepared.check_inputs()
        except ValueError as exc:
            status = int(prepared.status.item())
            passed = status == expected_status
            reason = str(exc)
        else:
            status = int(prepared.status.item())
            passed = False
            reason = "check_inputs did not raise"
        cases.append({"name": name, "expected_status": expected_status,
                      "actual_status": status, "reason": reason, "passed": passed})
        if not passed:
            break
    return cases


def capture_graph(function, graph_repeats):
    # Inputs, outputs, and workspace are already allocated by prepare_moe. Warm
    # and capture on the caller's current stream to match the single-stream M4 contract.
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_repeats):
            function()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def measure_stage_graph(graph, stage_name, *, samples, repeats, graph_repeats):
    """Measure repeated graph calls for one stage; results are per stage call."""
    measured = measure_pair(
        {stage_name: graph.replay}, 0, samples, repeats, rounds_count=3,
    )
    for row in measured["rounds"]:
        replay_samples = row["samples_ms"].pop(stage_name)
        row["samples_ms_per_graph_replay"] = {stage_name: replay_samples}
        row["samples_ms_per_stage_call"] = {
            stage_name: [value / graph_repeats for value in replay_samples]
        }
    measured["summary_per_stage_call"] = {
        name: {
            "p50_ms": summary["p50_ms"] / graph_repeats,
            "p95_ms": summary["p95_ms"] / graph_repeats,
            "graph_repeats": graph_repeats,
        }
        for name, summary in measured["summary"].items()
    }
    for summary in measured["summary"].values():
        summary.pop("effective_tflops", None)
    measured["samples_per_graph_replay"] = graph_repeats
    measured["stage"] = stage_name
    measured["scope"] = "isolated stage graph; do not sum these values as a strict full-forward latency"
    return measured


def replace_input_values(buffers, replacement):
    for current, new in zip(buffers, replacement):
        current.copy_(new)


def check_graph_updates(prepared, graph, buffers, *, spec, seed, pattern, repetitions=3):
    """Update values at the original addresses, replay, then compare with oracles."""
    names = ("x", "gate_up", "down", "ids", "routing_weights")
    checks = []
    for update_index in range(repetitions):
        replacement = make_inputs(
            tokens=spec["T"], experts=spec["E"], topk=spec["topk"],
            hidden=spec["H"], intermediate=spec["I"], id_dtype=buffers[3].dtype,
            pattern=pattern, seed=seed + update_index + 1, device=buffers[0].device,
        )
        replace_input_values(buffers, replacement)
        graph.replay()
        torch.cuda.synchronize(buffers[0].device)
        fp32 = moe_reference(*buffers)
        staged = staged_bf16_oracle(*buffers)
        output = prepared.output
        item = {
            "update": update_index,
            "seed": seed + update_index + 1,
            "input_sha256": tensor_hash(*buffers),
            "correctness": {
                "fp32_reference": safe_error(output, fp32),
                "staged_bf16_oracle": safe_error(output, staged),
            },
            "dispatch": validate_dispatch(prepared, buffers[0], buffers[3]),
        }
        item["passed"] = (
            all(metric["passed"] for metric in item["correctness"].values())
            and item["dispatch"]["passed"]
        )
        checks.append(item)
    return checks


def compiled_stage_kernel(kernels, stage):
    """Find the compiled Triton kernel for a stage in a small API-tolerant way."""
    if not isinstance(kernels, dict):
        return None
    aliases = {
        "gate_up": ("gate_up", "gateup", "grouped_gate_up", "moe_gate_up"),
        "down": ("down", "grouped_down", "moe_down"),
    }[stage]
    for alias in aliases:
        for key, value in kernels.items():
            if str(key).lower() == alias and hasattr(value, "asm"):
                return value
    for key, value in kernels.items():
        key_text = str(key).lower().replace("-", "_")
        if stage == "gate_up" and "gate" in key_text and "up" in key_text and hasattr(value, "asm"):
            return value
        if stage == "down" and "down" in key_text and hasattr(value, "asm"):
            return value
    return None


def collect_codegen(prepared, case_name, run_id):
    kernels = getattr(prepared, "kernels", {})
    result = {}
    for stage in ("gate_up", "down"):
        kernel = compiled_stage_kernel(kernels, stage)
        result[stage] = compile_artifacts(kernel, f"{case_name}_{stage}", run_id) if kernel is not None else None
    result["kernel_keys"] = [str(key) for key in kernels] if isinstance(kernels, dict) else []
    result["missing_stages"] = [stage for stage in ("gate_up", "down") if result[stage] is None]
    return result


def _functional_specs():
    return [
        {"name": "empty_T", "T": 0, "E": 8, "topk": 4, "H": 33, "I": 17,
         "id_dtype": torch.int32, "pattern": "random", "seed": 11},
        {"name": "random_topk1_tail_i32", "T": 7, "E": 8, "topk": 1, "H": 31, "I": 17,
         "id_dtype": torch.int32, "pattern": "random", "seed": 13},
        {"name": "random_topk2_tail_i64", "T": 17, "E": 8, "topk": 2, "H": 33, "I": 19,
         "id_dtype": torch.int64, "pattern": "random", "seed": 17},
        {"name": "duplicate_topk4", "T": 5, "E": 8, "topk": 4, "H": 35, "I": 23,
         "id_dtype": torch.int32, "pattern": "duplicate", "seed": 19},
        {"name": "all_hot_topk8", "T": 9, "E": 16, "topk": 8, "H": 37, "I": 25,
         "id_dtype": torch.int64, "pattern": "all_hot", "seed": 23},
        {"name": "random_topk4_graph_update", "T": 7, "E": 8, "topk": 4, "H": 33, "I": 21,
         "id_dtype": torch.int64, "pattern": "random", "seed": 29, "graph_updates": True},
        {"name": "random_topk8_tail", "T": 17, "E": 16, "topk": 8, "H": 39, "I": 27,
         "id_dtype": torch.int32, "pattern": "random", "seed": 31},
        {"name": "nonpower_experts_signed_weights", "T": 19, "E": 3, "topk": 2, "H": 67, "I": 33,
         "id_dtype": torch.int32, "pattern": "duplicate", "seed": 37, "signed_weights": True},
        {"name": "single_expert_M_tail", "T": 33, "E": 1, "topk": 1, "H": 65, "I": 31,
         "id_dtype": torch.int64, "pattern": "all_hot", "seed": 41},
    ]


def _performance_specs():
    specs = []
    for family, experts, hidden, intermediate, topk in (
        ("small", 16, 128, 64, 4),
        ("qwen3_30b_a3b_shape", 128, 2048, 768, 8),
    ):
        for tokens in (17, 193):
            for pattern, seed in (("uniform_random", 73), ("uniform_random", 991), ("hot", 73)):
                tag = "random" if pattern == "uniform_random" else "hot"
                if pattern == "uniform_random" and seed == 991:
                    tag = "random_seed991"
                specs.append({
                    "name": f"{family}_T{tokens}_{tag}", "T": tokens, "E": experts,
                    "topk": topk, "H": hidden, "I": intermediate, "id_dtype": torch.int32,
                    "pattern": pattern, "seed": seed,
                })
    return specs


def _prepare_case(spec, device):
    buffers = make_inputs(
        tokens=spec["T"], experts=spec["E"], topk=spec["topk"], hidden=spec["H"],
        intermediate=spec["I"], id_dtype=spec["id_dtype"], pattern=spec["pattern"],
        seed=spec["seed"], device=device,
    )
    if spec.get("signed_weights"):
        buffers[4].sub_(0.5)
    prepared = prepare_moe(*buffers)
    return buffers, prepared


def _case_metadata(spec, buffers):
    x, gate_up, down, ids, routing_weights = buffers
    return {
        "name": spec["name"], "T": spec["T"], "E": spec["E"], "topk": spec["topk"],
        "H": spec["H"], "I": spec["I"], "id_dtype": str(ids.dtype),
        "route_pattern": spec["pattern"], "seed": spec["seed"],
        "routing_sha256": tensor_hash(ids), "input_sha256": tensor_hash(x, gate_up, down, routing_weights),
        "flops_gemm_only": 6 * spec["T"] * spec["topk"] * spec["H"] * spec["I"],
        "scope": "complete fixed-route forward: reset/count/prefix/scatter/gate-up/SwiGLU/down/combine; prepare and oracle excluded",
    }


def run(samples=TIMING_SAMPLES, repeats=TIMING_REPEATS, graph_repeats=GRAPH_REPEATS):
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(SEED)

    run_id = "m4-moe-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(time.time_ns() % 10**9)
    results = ROOT / "results" / (run_id + ".json")
    results.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": 1, "run_id": run_id, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running", "seed": SEED, "fixed_tile_config": CONFIG,
        "limits": LIMITS, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "source_hashes": source_hashes(), "samples": samples, "repeats": repeats,
        "graph_repeats": graph_repeats,
        "scope": "M4 functional GPU-route forward baseline; no router, softmax, communication, serving, backward, or mature-backend comparison",
        "baseline_note": "moe_reference is correctness-only; no Python expert loop is timed",
        "timing": {
            "eager": "CUDA events around PreparedMoE.run; complete forward includes reset/count/prefix/scatter/gate-up/SwiGLU/down/combine; prepare/oracle excluded",
            "graph": "CUDA Graph captures repeated complete fixed-buffer forwards; event time normalized per graph replay and per forward",
            "limitations": "Python launch gaps can contribute to eager event spans; static shapes and addresses are required for graph replay",
        },
        "telemetry_before": telemetry(), "functional_cases": [], "performance_cases": [],
    }

    def save():
        results.write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the M4 MoE driver")
        device = torch.device("cuda", torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device)
        record["gpu"] = properties.name
        record["sm"] = [properties.major, properties.minor]
        record["memory_bytes"] = properties.total_memory
        save()

        for spec in _functional_specs():
            print("Validating", spec["name"], flush=True)
            buffers, prepared = _prepare_case(spec, device)
            output, correctness = compare_output(prepared, *buffers)
            dispatch = validate_dispatch(prepared, buffers[0], buffers[3])
            entry = {
                **_case_metadata(spec, buffers), "correctness": correctness,
                "dispatch_validation": dispatch,
                "buffer_aliasing": validate_buffer_aliasing(prepared, buffers),
            }
            if spec.get("graph_updates"):
                graph = capture_graph(prepared.run, graph_repeats=2)
                entry["graph_updates"] = check_graph_updates(
                    prepared, graph, buffers, spec=spec, seed=spec["seed"], pattern="hot", repetitions=3,
                )
            else:
                entry["graph_updates"] = []
            entry["passed"] = (
                correctness["passed"] and dispatch["passed"]
                and all(entry["buffer_aliasing"].values())
                and all(item["passed"] for item in entry["graph_updates"])
            )
            record["functional_cases"].append(entry)
            save()
            if not entry["passed"]:
                raise RuntimeError(f"M4 functional validation failed: {spec['name']}")
            del output, buffers, prepared

        record["debug_input_validation"] = validate_bad_runtime_inputs(device)
        save()
        if not all(item["passed"] for item in record["debug_input_validation"]):
            raise RuntimeError("M4 explicit invalid-input validation failed")

        for spec in _performance_specs():
            print("Timing", spec["name"], flush=True)
            torch.cuda.reset_peak_memory_stats(device)
            buffers, prepared = _prepare_case(spec, device)
            output, correctness = compare_output(prepared, *buffers)
            dispatch = validate_dispatch(prepared, buffers[0], buffers[3])
            entry = {
                **_case_metadata(spec, buffers), "correctness": correctness,
                "dispatch_validation": dispatch,
                "buffer_aliasing": validate_buffer_aliasing(prepared, buffers),
                "telemetry_before": telemetry(),
                "codegen": collect_codegen(prepared, spec["name"], run_id),
            }
            if (not correctness["passed"] or not dispatch["passed"]
                    or not all(entry["buffer_aliasing"].values())):
                record["performance_cases"].append(entry)
                save()
                raise RuntimeError(f"M4 performance case correctness failed: {spec['name']}")
            if entry["codegen"]["missing_stages"]:
                record["performance_cases"].append(entry)
                save()
                raise RuntimeError(f"missing M4 GEMM codegen artifacts: {spec['name']}")

            # This is intentionally a single-backend timing call: the result is
            # a complete-forward functional baseline, not an expert-loop speedup.
            functions = {"prepared_moe": prepared.run}
            flops = entry["flops_gemm_only"]
            entry["eager"] = measure_pair(functions, flops, samples, repeats, rounds_count=3)

            graph = capture_graph(prepared.run, graph_repeats)
            entry["graph_updates"] = check_graph_updates(
                prepared, graph, buffers, spec=spec, seed=spec["seed"],
                pattern="hot" if spec["pattern"] == "uniform_random" else "uniform_random",
                repetitions=3,
            )
            if not all(item["passed"] for item in entry["graph_updates"]):
                record["performance_cases"].append(entry)
                save()
                raise RuntimeError(f"M4 graph update validation failed: {spec['name']}")

            # Restore the exact original input values after the dynamic replay
            # check so eager and graph timings, hashes, and later stage graphs
            # describe the same case inputs.
            original = make_inputs(
                tokens=spec["T"], experts=spec["E"], topk=spec["topk"],
                hidden=spec["H"], intermediate=spec["I"], id_dtype=spec["id_dtype"],
                pattern=spec["pattern"], seed=spec["seed"], device=device,
            )
            replace_input_values(buffers, original)
            del original
            graph.replay()
            torch.cuda.synchronize(device)
            restored_input_sha = tensor_hash(buffers[0], buffers[1], buffers[2], buffers[4])
            restored_route_sha = tensor_hash(buffers[3])
            entry["graph_rebaseline"] = {
                "input_sha256": restored_input_sha,
                "routing_sha256": restored_route_sha,
                "matches_case_hashes": (
                    restored_input_sha == entry["input_sha256"]
                    and restored_route_sha == entry["routing_sha256"]
                ),
            }
            if not entry["graph_rebaseline"]["matches_case_hashes"]:
                record["performance_cases"].append(entry)
                save()
                raise RuntimeError(f"failed to restore original graph inputs: {spec['name']}")

            graph_timing = measure_pair(
                {"prepared_moe_graph": graph.replay}, flops * graph_repeats,
                samples, repeats, rounds_count=3,
            )
            for row in graph_timing["rounds"]:
                replay_samples = row["samples_ms"].pop("prepared_moe_graph")
                row["samples_ms_per_graph_replay"] = {"prepared_moe_graph": replay_samples}
                row["samples_ms_per_forward"] = {
                    "prepared_moe_graph": [value / graph_repeats for value in replay_samples]
                }
            graph_timing["summary_per_forward"] = {
                name: {
                    **summary,
                    "p50_ms": summary["p50_ms"] / graph_repeats,
                    "p95_ms": summary["p95_ms"] / graph_repeats,
                    "graph_repeats": graph_repeats,
                }
                for name, summary in graph_timing["summary"].items()
            }
            graph_timing["samples_per_graph_replay"] = graph_repeats
            entry["graph"] = graph_timing

            # Validate the output produced by the timed graph after the final
            # graph replay, using the then-current fixed-address input values.
            reference = moe_reference(*buffers)
            staged = staged_bf16_oracle(*buffers)
            entry["graph_correctness"] = {
                "fp32_reference": safe_error(prepared.output, reference),
                "staged_bf16_oracle": safe_error(prepared.output, staged),
            }
            entry["graph_correctness"]["passed"] = all(
                metric["passed"] for metric in entry["graph_correctness"].values()
            )

            # Stage-only graph timings are diagnostic. Dispatch may rewrite
            # metadata and later stages replay against the resulting workspace;
            # they are not a decomposition that sums exactly to full-forward time.
            stage_methods = {
                "dispatch": prepared.dispatch,
                "gate_up": prepared.project,
                "swiglu": prepared.activate,
                "down": prepared.down_project,
                "combine": prepared.combine,
            }
            entry["graph_stage_timings"] = {}
            for stage_name, method in stage_methods.items():
                stage_graph = capture_graph(method, graph_repeats)
                entry["graph_stage_timings"][stage_name] = measure_stage_graph(
                    stage_graph, stage_name, samples=samples, repeats=repeats,
                    graph_repeats=graph_repeats,
                )
                del stage_graph

            # Stage captures intentionally modify intermediate buffers; restore
            # the complete route, then recheck output and dispatch invariants.
            output = prepared.run()
            torch.cuda.synchronize(device)
            final_reference = moe_reference(*buffers)
            final_staged = staged_bf16_oracle(*buffers)
            entry["post_stage_graph_correctness"] = {
                "fp32_reference": safe_error(output, final_reference),
                "staged_bf16_oracle": safe_error(output, final_staged),
                "dispatch": validate_dispatch(prepared, buffers[0], buffers[3]),
            }
            entry["post_stage_graph_correctness"]["passed"] = (
                entry["post_stage_graph_correctness"]["fp32_reference"]["passed"]
                and entry["post_stage_graph_correctness"]["staged_bf16_oracle"]["passed"]
                and entry["post_stage_graph_correctness"]["dispatch"]["passed"]
            )
            entry["telemetry_after"] = telemetry()
            entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            entry["passed"] = (
                entry["graph_correctness"]["passed"]
                and entry["post_stage_graph_correctness"]["passed"]
            )
            record["performance_cases"].append(entry)
            save()
            if not entry["passed"]:
                raise RuntimeError(f"M4 graph output correctness failed: {spec['name']}")
            print(json.dumps({"eager": entry["eager"]["summary"],
                              "graph_per_forward": entry["graph"]["summary_per_forward"]}), flush=True)
            del graph, functions, output, buffers, prepared, reference, staged
            del final_reference, final_staged, stage_methods, method

        record["status"] = "passed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        record["telemetry_after"] = telemetry()
        save()
    print("Saved", results, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=TIMING_SAMPLES)
    parser.add_argument("--repeats", type=int, default=TIMING_REPEATS)
    parser.add_argument("--graph-repeats", type=int, default=GRAPH_REPEATS)
    args = parser.parse_args()
    if args.samples < 2 or args.repeats < 1 or args.graph_repeats < 1:
        parser.error("samples >= 2 and repeats/graph-repeats >= 1 required")
    run(args.samples, args.repeats, args.graph_repeats)
