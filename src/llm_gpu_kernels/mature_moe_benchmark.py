"""M5 full-forward comparison: Python M4, cached launch, and official vLLM.

Run this module from the repository root after sourcing scripts/activate-vllm.sh.
The fixed case matrix and timing protocol are defined in docs/M5_EXPERIMENT_PLAN.md.
This driver keeps the expert-loop implementation correctness-only; it is never
used as a performance backend.
"""
import argparse
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

from .benchmark import LIMITS, ROOT, tensor_hash
from .baselines.vllm_moe import prepare_vllm_moe
from .moe_benchmark import (
    _functional_specs,
    make_ids,
    make_inputs,
    safe_error,
    telemetry,
    validate_dispatch,
)
from .moe_fused import prepare_fused_moe
from .moe_prepared import prepare_moe_launcher
from .optimization_benchmark import measure
from .reference import moe_reference


GRAPH_REPEATS = 8
API_SAMPLES = 6
API_REPEATS = 2
VLLM_PROVENANCE = ROOT / "results/provenance/vllm-0.10.0-runtime.json"


def _source_hashes():
    paths = [*sorted((ROOT / "src").rglob("*.py")),
             ROOT / "docs/M5_EXPERIMENT_PLAN.md", ROOT / "scripts/activate-vllm.sh"]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _json_value(value):
    """Convert backend config values into stable JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return repr(value)


def _spec_metadata(spec, buffers):
    x, gate_up, down, ids, routing_weights = buffers
    return {
        "name": spec["name"],
        "T": spec["T"],
        "E": spec["E"],
        "topk": spec["topk"],
        "H": spec["H"],
        "I": spec["I"],
        "id_dtype": str(ids.dtype),
        "route_pattern": spec["pattern"],
        "seed": spec["seed"],
        "signed_weights": bool(spec.get("signed_weights", False)),
        "input_sha256": tensor_hash(x, gate_up, down, routing_weights),
        "routing_sha256": tensor_hash(ids),
        "flops_gemm_only_per_forward": 6 * spec["T"] * spec["topk"] * spec["H"] * spec["I"],
    }


def _make_mutable_update(buffers, seed, pattern="hot"):
    """Build only the small mutable values; avoid cloning large expert weights."""
    x, _, _, ids, _ = buffers
    tokens, hidden = x.shape
    experts = buffers[1].shape[0]
    topk = ids.shape[1]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    updated_x = torch.randn((tokens, hidden), generator=generator, dtype=torch.float32)
    updated_ids = make_ids(tokens, experts, topk, pattern, generator).to(dtype=ids.dtype)
    updated_weights = torch.rand((tokens, topk), generator=generator, dtype=torch.float32)
    # Signed finite coefficients stress that runtime values are not specialized.
    updated_weights = updated_weights.mul_(2.0).sub_(1.0)
    return (
        updated_x.to(device=x.device, dtype=x.dtype),
        updated_ids.to(device=ids.device),
        updated_weights.to(device=buffers[4].device),
    )


def _copy_mutable_values(buffers, values):
    buffers[0].copy_(values[0])
    buffers[3].copy_(values[1])
    buffers[4].copy_(values[2])


def _mutable_hash(buffers):
    return tensor_hash(buffers[0], buffers[3], buffers[4])


def _capture_graph(function, graph_repeats, device):
    # All setup and warmup are outside the timed graph replay calls. Capture on
    # the caller's current stream, matching the M4 fixed-stream contract.
    for _ in range(3):
        function()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_repeats):
            function()
    graph.replay()
    torch.cuda.synchronize(device)
    return graph


def _oracle_metrics(outputs, oracle):
    return {name: safe_error(output, oracle) for name, output in outputs.items()}


def _cached_bitwise(outputs, left="cached_fused", right="cached_unfused"):
    if left not in outputs or right not in outputs:
        return {"passed": False, "available": False}
    equal = torch.equal(outputs[left], outputs[right])
    return {"passed": bool(equal), "available": True, "torch_equal": bool(equal)}


def _validate_cached_status(prepared):
    prepared.check_inputs()
    return True


def _functional_graph_updates(buffers, spec, cached_fused, cached_unfused, device):
    """Exercise graph replay after in-place X/IDs/route-weight updates."""
    graph_repeats = 2
    functions = {
        "cached_fused": cached_fused.run,
        "cached_unfused": cached_unfused.run,
    }
    graphs = {
        name: _capture_graph(function, graph_repeats, device)
        for name, function in functions.items()
    }
    checks = []
    for update_index in range(2):
        seed = spec["seed"] + 401 + update_index
        values = _make_mutable_update(buffers, seed, pattern="hot")
        _copy_mutable_values(buffers, values)
        update_hash = _mutable_hash(buffers)
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize(device)

        oracle = moe_reference(*buffers)
        outputs = {
            "cached_fused": cached_fused.base.output,
            "cached_unfused": cached_unfused.base.output,
        }
        correctness = _oracle_metrics(outputs, oracle)
        dispatch = {
            "cached_fused": validate_dispatch(cached_fused.base, buffers[0], buffers[3]),
            "cached_unfused": validate_dispatch(cached_unfused.base, buffers[0], buffers[3]),
        }
        item = {
            "update_index": update_index,
            "seed": seed,
            "mutable_input_sha256": update_hash,
            "correctness": correctness,
            "dispatch_validation": dispatch,
            "cached_bitwise": _cached_bitwise(outputs),
        }
        item["passed"] = (
            all(value["passed"] for value in correctness.values())
            and all(value["passed"] for value in dispatch.values())
            and item["cached_bitwise"]["passed"]
        )
        checks.append(item)
    return graphs, checks


def _validate_nondefault_stream(buffers, spec, cached_fused, cached_unfused, device):
    values = _make_mutable_update(buffers, spec["seed"] + 503, pattern="uniform_random")
    current = torch.cuda.current_stream(device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        _copy_mutable_values(buffers, values)
        cached_fused.run()
        cached_unfused.run()
    current.wait_stream(side)
    torch.cuda.synchronize(device)

    oracle = moe_reference(*buffers)
    outputs = {
        "cached_fused": cached_fused.base.output,
        "cached_unfused": cached_unfused.base.output,
    }
    correctness = _oracle_metrics(outputs, oracle)
    dispatch = {
        "cached_fused": validate_dispatch(cached_fused.base, buffers[0], buffers[3]),
        "cached_unfused": validate_dispatch(cached_unfused.base, buffers[0], buffers[3]),
    }
    return {
        "stream_handle": int(side.cuda_stream),
        "mutable_input_sha256": _mutable_hash(buffers),
        "correctness": correctness,
        "dispatch_validation": dispatch,
        "cached_bitwise": _cached_bitwise(outputs),
        "passed": (
            all(value["passed"] for value in correctness.values())
            and all(value["passed"] for value in dispatch.values())
            and _cached_bitwise(outputs)["passed"]
        ),
    }


def _functional_case(spec, device):
    buffers = list(make_inputs(
        tokens=spec["T"], experts=spec["E"], topk=spec["topk"], hidden=spec["H"],
        intermediate=spec["I"], id_dtype=spec["id_dtype"], pattern=spec["pattern"],
        seed=spec["seed"], device=device,
    ))
    if spec.get("signed_weights"):
        buffers[4].sub_(0.5)

    entry = _spec_metadata(spec, buffers)
    prepared = {}
    preparation_errors = {}
    prepare_functions = {
        "python_fused": lambda: prepare_fused_moe(*buffers),
        "cached_fused": lambda: prepare_moe_launcher(*buffers, fused=True),
        "cached_unfused": lambda: prepare_moe_launcher(*buffers, fused=False),
        "vllm": lambda: prepare_vllm_moe(*buffers),
    }
    for name, prepare in prepare_functions.items():
        try:
            prepared[name] = prepare()
        except Exception as exc:
            preparation_errors[name] = repr(exc)

    outputs = {}
    execution_errors = {}
    for name, instance in prepared.items():
        try:
            outputs[name] = instance.run()
            if name.startswith("cached_"):
                _validate_cached_status(instance)
            elif name == "python_fused":
                instance.check_inputs()
        except Exception as exc:
            execution_errors[name] = repr(exc)

    torch.cuda.synchronize(device)
    oracle = moe_reference(*buffers)
    correctness = _oracle_metrics(outputs, oracle)
    dispatch = {}
    for name in ("python_fused", "cached_fused", "cached_unfused"):
        if name in prepared:
            base = prepared[name].base
            try:
                dispatch[name] = validate_dispatch(base, buffers[0], buffers[3])
            except Exception as exc:
                dispatch[name] = {"passed": False, "error": repr(exc)}

    cached_bitwise = _cached_bitwise(outputs)
    entry.update({
        "preparation_errors": preparation_errors,
        "execution_errors": execution_errors,
        "correctness_full_fp32_reference": correctness,
        "dispatch_validation": dispatch,
        "cached_bitwise": cached_bitwise,
        "graph_updates": [],
        "nondefault_stream": None,
    })

    graph_passed = True
    stream_passed = True
    if spec.get("graph_updates") and all(
        name in prepared for name in ("cached_fused", "cached_unfused")
    ):
        graphs, graph_checks = _functional_graph_updates(
            buffers, spec, prepared["cached_fused"], prepared["cached_unfused"], device,
        )
        entry["graph_updates"] = graph_checks
        graph_passed = all(item["passed"] for item in graph_checks)
        entry["nondefault_stream"] = _validate_nondefault_stream(
            buffers, spec, prepared["cached_fused"], prepared["cached_unfused"], device,
        )
        stream_passed = entry["nondefault_stream"]["passed"]
        del graphs

    base_dispatch_passed = all(item.get("passed", False) for item in dispatch.values())
    entry["passed"] = (
        not preparation_errors and not execution_errors
        and all(value["passed"] for value in correctness.values())
        and base_dispatch_passed and cached_bitwise["passed"]
        and graph_passed and stream_passed
    )
    return entry, buffers, prepared


def _performance_specs():
    specs = []
    # Qwen3 30B-A3B shape: five T values for both frozen random-route seeds.
    for seed in (73, 991):
        for tokens in (1, 17, 193, 1024, 2049):
            specs.append({
                "name": f"qwen3_30b_a3b_T{tokens}_random_seed{seed}",
                "T": tokens, "E": 128, "topk": 8, "H": 2048, "I": 768,
                "id_dtype": torch.int32, "pattern": "uniform_random", "seed": seed,
            })
    # Fixed 16-expert hot set, generated by M4's deterministic hot router.
    for tokens in (193, 1024):
        specs.append({
            "name": f"qwen3_30b_a3b_T{tokens}_hot16_seed73",
            "T": tokens, "E": 128, "topk": 8, "H": 2048, "I": 768,
            "id_dtype": torch.int32, "pattern": "hot", "seed": 73,
        })
    for tokens in (1, 17, 193):
        specs.append({
            "name": f"small_E16_T{tokens}_random_seed73",
            "T": tokens, "E": 16, "topk": 4, "H": 128, "I": 64,
            "id_dtype": torch.int32, "pattern": "uniform_random", "seed": 73,
        })
    return specs


def _prepare_performance_backends(buffers):
    # All three receive the same five tensor objects and values. Preparations
    # and warmup runs occur before any measured sample.
    python_fused = prepare_fused_moe(*buffers)
    cached_fused = prepare_moe_launcher(*buffers, fused=True)
    vllm = prepare_vllm_moe(*buffers)
    return {
        "m4_fused_python": python_fused,
        "cached_fused": cached_fused,
        "vllm_public": vllm,
    }


def _validate_backend_outputs(outputs, oracle):
    return _oracle_metrics(outputs, oracle)


def _runtime_metadata(provenance):
    metadata = {
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "numpy": {"version": np.__version__, "path": str(Path(np.__file__).resolve())},
        "torch": {"version": torch.__version__, "path": str(Path(torch.__file__).resolve()),
                  "cuda_runtime": torch.version.cuda},
        "triton": {"version": triton.__version__, "path": str(Path(triton.__file__).resolve())},
        "environment": {
            "pythonpath": os.environ.get("PYTHONPATH"),
            "vllm_fused_moe_chunk_size": os.environ.get("VLLM_FUSED_MOE_CHUNK_SIZE"),
        },
        "vllm_provenance_path": str(VLLM_PROVENANCE.relative_to(ROOT)),
        "vllm_provenance": provenance,
    }
    return metadata


def _run_performance_case(spec, device, samples, repeats, graph_repeats, record, save):
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
    initial_mutable_hash = _mutable_hash(buffers)

    backends = _prepare_performance_backends(buffers)
    vllm = backends["vllm_public"]
    entry["vllm_configuration"] = _json_value(vllm.configuration())

    functions = {name: prepared.run for name, prepared in backends.items()}

    # Capture all three complete forwards. The vLLM public API allocates a new
    # output each call, so retain the exact tensor produced in each captured
    # graph's pool for correctness validation after replay.
    graphs = {}
    graph_outputs = {}
    for name, function in functions.items():
        graphs[name] = _capture_graph(function, graph_repeats, device)
        if name == "vllm_public":
            graph_outputs[name] = vllm.output
        elif name == "m4_fused_python":
            graph_outputs[name] = backends[name].base.output
        else:
            graph_outputs[name] = backends[name].base.output

    # Update only small mutable inputs while graph addresses stay fixed. Both
    # cached implementations and the vLLM graph are checked against the same
    # values; then restore exactly before either API or graph timing begins.
    updated_inputs = []
    for update_index in range(2):
        seed = spec["seed"] + 701 + update_index
        values = _make_mutable_update(buffers, seed, pattern="hot")
        _copy_mutable_values(buffers, values)
        updated_hash = _mutable_hash(buffers)
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize(device)
        oracle = moe_reference(*buffers)
        update_outputs = {
            "m4_fused_python": backends["m4_fused_python"].base.output,
            "cached_fused": backends["cached_fused"].base.output,
            "vllm_public": graph_outputs["vllm_public"],
        }
        correctness = _oracle_metrics(update_outputs, oracle)
        updated_inputs.append({
            "update_index": update_index,
            "seed": seed,
            "mutable_input_sha256": updated_hash,
            "graph_correctness_full_fp32_reference": correctness,
            "passed": all(value["passed"] for value in correctness.values()),
        })

    _copy_mutable_values(buffers, original_mutable)
    torch.cuda.synchronize(device)
    restored_mutable_hash = _mutable_hash(buffers)
    entry["graph_input_updates"] = {
        "original_mutable_sha256": initial_mutable_hash,
        "updates": updated_inputs,
        "restored_mutable_sha256": restored_mutable_hash,
        "restored_input_sha256": tensor_hash(buffers[0], buffers[1], buffers[2], buffers[4]),
        "restored_routing_sha256": tensor_hash(buffers[3]),
    }
    entry["graph_input_updates"]["matches_original_hashes"] = (
        restored_mutable_hash == initial_mutable_hash
        and entry["graph_input_updates"]["restored_input_sha256"] == entry["input_sha256"]
        and entry["graph_input_updates"]["restored_routing_sha256"] == entry["routing_sha256"]
    )
    entry["graph_input_updates"]["passed"] = (
        entry["graph_input_updates"]["matches_original_hashes"]
        and all(item["passed"] for item in updated_inputs)
    )
    if not entry["graph_input_updates"]["passed"]:
        raise RuntimeError(f"graph update validation failed: {spec['name']}")

    # Ensure the retained graph outputs now hold outputs for the restored
    # original case values. Check all three against the same full FP32 oracle.
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize(device)
    oracle = moe_reference(*buffers)
    graph_correctness = _validate_backend_outputs(graph_outputs, oracle)
    entry["graph_correctness_full_fp32_reference"] = graph_correctness
    if not all(value["passed"] for value in graph_correctness.values()):
        raise RuntimeError(f"graph correctness failed: {spec['name']}")

    eager_outputs = {name: function() for name, function in functions.items()}
    torch.cuda.synchronize(device)
    oracle = moe_reference(*buffers)
    api_correctness = _validate_backend_outputs(eager_outputs, oracle)
    entry["api_correctness_full_fp32_reference"] = api_correctness
    entry["dispatch_validation"] = {
        "m4_fused_python": validate_dispatch(
            backends["m4_fused_python"].base, buffers[0], buffers[3],
        ),
        "cached_fused": validate_dispatch(
            backends["cached_fused"].base, buffers[0], buffers[3],
        ),
    }
    backends["m4_fused_python"].check_inputs()
    backends["cached_fused"].check_inputs()
    if (not all(value["passed"] for value in api_correctness.values())
            or not all(value["passed"] for value in entry["dispatch_validation"].values())):
        raise RuntimeError(f"API correctness failed: {spec['name']}")

    flops = entry["flops_gemm_only_per_forward"]
    entry["api"] = measure(functions, flops, samples=samples, repeats=repeats)
    entry["graph"] = measure(
        {name: graph.replay for name, graph in graphs.items()}, flops,
        divisor=graph_repeats, samples=samples, repeats=repeats,
    )
    for timing in (entry["api"], entry["graph"]):
        for round_item in timing["rounds"]:
            # optimization_benchmark.measure is GEMM-oriented; retain its raw
            # keys and add the full-forward units used by this M5 experiment.
            round_item["samples_ms_per_forward"] = round_item["samples_ms_per_gemm"]
            round_item["wall_ms_per_forward"] = round_item["wall_ms_per_gemm"]
        timing["sample_unit"] = "milliseconds per complete forward"
    entry["graph_samples_per_replay"] = graph_repeats
    entry["graph_summary_unit"] = "per complete forward; raw samples are retained per graph replay normalized by divisor"

    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize(device)
    entry["post_timing_graph_correctness_full_fp32_reference"] = _validate_backend_outputs(
        graph_outputs, oracle,
    )
    if not all(value["passed"] for value in entry["post_timing_graph_correctness_full_fp32_reference"].values()):
        raise RuntimeError(f"post-timing graph correctness failed: {spec['name']}")

    entry["telemetry_after"] = telemetry()
    entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    entry["passed"] = True
    save()
    del graphs, graph_outputs, functions, backends, eager_outputs, oracle, buffers, original_mutable
    return entry


def run(samples=API_SAMPLES, repeats=API_REPEATS, graph_repeats=GRAPH_REPEATS):
    if samples < 1 or repeats < 1 or graph_repeats < 1:
        raise ValueError("samples, repeats, and graph_repeats must be positive")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False

    provenance = json.loads(VLLM_PROVENANCE.read_text())
    plan_hash = hashlib.sha256((ROOT / "docs/M5_EXPERIMENT_PLAN.md").read_bytes()).hexdigest()
    run_id = "m5-mature-moe-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(time.time_ns() % 10**9)
    destination = ROOT / "results" / (run_id + ".json")
    record = {
        "schema": 1,
        "run_id": run_id,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
        "m5_experiment_plan_sha256": plan_hash,
        "source_hashes": _source_hashes(),
        "libraries": _runtime_metadata(provenance),
        "limits": LIMITS,
        "api_samples_per_round": samples,
        "api_repeats_per_sample": repeats,
        "cyclic_rounds": "three forward cyclic and three reverse cyclic orders",
        "graph_repeats": graph_repeats,
        "graph_divisor": graph_repeats,
        "timing": {
            "api": "optimization_benchmark.measure CUDA events; three backends in six forward/reverse cyclic rounds; includes public wrapper costs and host gaps",
            "graph": "one graph per backend; each graph captures repeated complete forwards and is timed with divisor equal to graph_repeats; capture/setup excluded",
            "flops": "6 * T * topk * H * I per complete forward; reports effective TFLOPS from this count",
            "correctness": "all functional/API/graph outputs against full FP32 oracle at relative_rms and normalized_max <= 0.008; cached fused/unfused exact bitwise comparison",
        },
        "scope": "fixed-route complete-forward comparison: M4 Python fused, M4 cached fused launcher, and unmodified official vLLM 0.10.0 public API",
        "baseline_policy": "vLLM output and route-weight rounding follow the upstream public API; no Python expert-loop performance backend",
        "telemetry_before": telemetry(),
        "functional_cases": [],
        "performance_cases": [],
    }

    def save():
        destination.write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the M5 MoE benchmark")
        if triton.__version__ != "3.3.1":
            raise RuntimeError(f"M5 cached launcher requires Triton 3.3.1, got {triton.__version__}")
        device = torch.device("cuda", torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device)
        record["gpu"] = properties.name
        record["sm"] = [properties.major, properties.minor]
        record["memory_bytes"] = properties.total_memory
        record["libraries"]["torch"]["device_count"] = torch.cuda.device_count()
        save()

        functional_failed = False
        for spec in _functional_specs():
            print("M5 functional", spec["name"], flush=True)
            torch.cuda.reset_peak_memory_stats(device)
            entry = {
                "name": spec["name"],
                "telemetry_before": telemetry(),
                "passed": False,
            }
            record["functional_cases"].append(entry)
            save()
            try:
                completed, buffers, prepared = _functional_case(spec, device)
                entry.update(completed)
                entry["telemetry_after"] = telemetry()
                entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                functional_failed = functional_failed or not entry["passed"]
                del prepared, buffers
            except Exception as exc:
                entry["passed"] = False
                entry["error"] = repr(exc)
                entry["traceback"] = traceback.format_exc()
                entry["telemetry_after"] = telemetry()
                entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                functional_failed = True
            save()
        vllm_module = sys.modules.get("vllm")
        if vllm_module is None:
            record["libraries"]["vllm"] = {"imported": False}
            functional_failed = True
        else:
            vllm_path = Path(vllm_module.__file__).resolve()
            expected_root = Path(provenance["runtime_path"]).resolve()
            try:
                vllm_path.relative_to(expected_root)
                runtime_matches = True
            except ValueError:
                runtime_matches = False
            record["libraries"]["vllm"] = {
                "version": getattr(vllm_module, "__version__", None),
                "path": str(vllm_path),
                "expected_runtime_path": str(expected_root),
                "matches_provenance_runtime": runtime_matches,
            }
            if not runtime_matches:
                functional_failed = True
                record["vllm_runtime_path_error"] = (
                    "loaded vLLM module does not come from the documented isolated runtime"
                )
        record["functional_validation_passed"] = not functional_failed
        save()
        if functional_failed:
            raise RuntimeError("one or more M5 functional cases failed; performance matrix not started")

        for spec in _performance_specs():
            print("M5 performance", spec["name"], flush=True)
            try:
                _run_performance_case(spec, device, samples, repeats, graph_repeats, record, save)
            except Exception as exc:
                entry = record["performance_cases"][-1]
                entry["passed"] = False
                entry["error"] = repr(exc)
                entry["traceback"] = traceback.format_exc()
                entry.setdefault("telemetry_after", telemetry())
                entry.setdefault("peak_allocated_bytes", torch.cuda.max_memory_allocated(device))
                save()
                raise

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
