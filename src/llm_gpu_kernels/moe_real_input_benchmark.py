"""M20 isolated checkpoint layer-0 replay; frozen M7 policy, no retuning."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import traceback

import torch
import triton

from .baselines.vllm_moe import prepare_vllm_moe
from .benchmark import ROOT, LIMITS, tensor_hash, compile_artifacts
from .mature_moe_benchmark import _capture_graph, _json_value
from .moe_benchmark import safe_error, telemetry
from .moe_configured import prepare_configured_moe
from .moe_full_holdout import _ranked_state, _vllm_runtime
from .moe_full_policy import FrozenMoEPolicy
from .moe_ranked import prepare_ranked_moe_launcher
from .moe_real_trace import validate_real_moe_trace, sha256_file
from .optimization_benchmark import measure
from .reference import moe_reference


def owner_check():
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    for line in rows.splitlines():
        if line.strip() and (not line.strip().isdigit() or int(line) != os.getpid()):
            raise RuntimeError("another GPU process or unavailable process information: "+line)


def source_hashes(policy):
    paths = set(dict(policy.source_hashes)) | {
        "src/llm_gpu_kernels/moe_real_input_benchmark.py", "src/llm_gpu_kernels/moe_real_trace.py",
        "src/llm_gpu_kernels/moe_full_holdout.py", "src/llm_gpu_kernels/mature_moe_benchmark.py",
        "src/llm_gpu_kernels/ranked_moe_benchmark.py", "src/llm_gpu_kernels/moe_benchmark.py",
        "src/llm_gpu_kernels/optimization_benchmark.py", "src/llm_gpu_kernels/reference.py",
        "src/llm_gpu_kernels/baselines/vllm_moe.py", "docs/M20_REAL_MOE_INPUT_PLAN.md",
        "docs/M20_REAL_MOE_TRACE_CONTRACT.md", "results/provenance/vllm-0.10.0-runtime.json",
        "scripts/fetch_qwen3_moe_layer0.py", "scripts/capture_qwen3_moe_layer0.py"}
    return {name: sha256_file(ROOT/name) for name in sorted(paths)}


def dispatch_check(backends, buffers):
    result = {name: _ranked_state(backends[name], buffers) for name in ("ranked_default", "ranked_selected")}
    if not all(row["passed"] for row in result.values()):
        raise RuntimeError("ranked dispatch/status check failed")
    # Keep the counts and validation booleans; per-route maps are already validated.
    for row in result.values():
        row["ranked_dispatch"].pop("rank_values_by_expert", None)
    return result


def check_outputs(outputs, oracle):
    return {name: safe_error(output, oracle) for name, output in outputs.items()}


def artifact_hashes(capture_root, capture):
    paths = {capture_root/"capture-summary.json"}
    for spec in capture["cases"]:
        manifest_path = (capture_root/spec["manifest"]).resolve()
        data = json.loads(manifest_path.read_text())
        paths.add(manifest_path)
        for ref in (data["trace"], *data["weights"].values()):
            paths.add((manifest_path.parent/ref["path"]).resolve())
    return {str(p.resolve()): sha256_file(p) for p in sorted(paths)}


def run_case(case, sibling, gpu_weights, policy, row, artifact_run_id):
    tensors = case.tensors
    x = tensors["x"].cuda().contiguous()
    ids = tensors["expert_ids"].cuda().contiguous()
    rw = tensors["route_weights"].cuda().contiguous()
    gate_up, down = gpu_weights["gate_up"], gpu_weights["down"]
    buffers = [x, gate_up, down, ids, rw]
    config, bucket = policy.select(x.shape[0], 128, 8, 2048, 768)
    row.update(tokens=x.shape[0], input_sha256=tensor_hash(x, ids, rw), policy_bucket=bucket,
               selected_config=asdict(config), telemetry_before=telemetry())
    counts = torch.bincount(tensors["expert_ids"].long().flatten(), minlength=128)
    row["route_counts"] = counts.tolist()
    row["routing_distribution"] = dict(active_experts=int((counts > 0).sum()),
                                       maximum_routes_per_expert=int(counts.max()),
                                       mean_routes_per_expert=float(counts.float().mean()))
    backends = {"ranked_default": prepare_ranked_moe_launcher(*buffers),
                "ranked_selected": prepare_configured_moe(*buffers, config),
                "vllm_public": prepare_vllm_moe(*buffers)}
    if any(len(backends[name].launches) != 7 for name in ("ranked_default", "ranked_selected")):
        raise RuntimeError("expected complete seven-launch ranked path")
    row["vllm_configuration"] = _json_value(backends["vllm_public"].configuration())
    proof = json.loads((ROOT/"results/provenance/vllm-0.10.0-runtime.json").read_text())
    row["vllm_runtime"] = _vllm_runtime(proof)
    functions = {name: backend.run for name, backend in backends.items()}
    outputs = {name: function() for name, function in functions.items()}
    torch.cuda.synchronize()
    oracle = moe_reference(*buffers)
    row["initial_vs_independent_fp32"] = check_outputs(outputs, oracle)
    row["initial_dispatch"] = dispatch_check(backends, buffers)
    # A failed metric is preserved before stopping. No threshold or scale adjustment.
    if not all(m["passed"] for m in row["initial_vs_independent_fp32"].values()):
        raise RuntimeError("checkpoint forward failed fixed FP32 accuracy limits")
    graphs = {name: _capture_graph(function, 8, x.device) for name, function in functions.items()}
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    graph_outputs = {"ranked_default": backends["ranked_default"].base.output,
                     "ranked_selected": backends["ranked_selected"].base.output,
                     "vllm_public": backends["vllm_public"].output}
    row["graph_initial_vs_fp32"] = check_outputs(graph_outputs, oracle)
    other = sibling.tensors
    x.copy_(other["x"]); ids.copy_(other["expert_ids"]); rw.copy_(other["route_weights"])
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    row["graph_rebound_to_other_real_prompt_vs_fp32"] = check_outputs(graph_outputs, moe_reference(*buffers))
    row["graph_rebound_dispatch"] = dispatch_check(backends, buffers)
    x.copy_(tensors["x"]); ids.copy_(tensors["expert_ids"]); rw.copy_(tensors["route_weights"])
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    row["restored_input_sha256"] = tensor_hash(x, ids, rw)
    if row["restored_input_sha256"] != row["input_sha256"]:
        raise RuntimeError("mutable buffers were not restored")
    row["graph_restored_vs_fp32"] = check_outputs(graph_outputs, oracle)
    for key in ("graph_initial_vs_fp32", "graph_rebound_to_other_real_prompt_vs_fp32", "graph_restored_vs_fp32"):
        if not all(m["passed"] for m in row[key].values()):
            raise RuntimeError("real-input graph update failed fixed accuracy limits")
    flops = 6*x.shape[0]*8*2048*768
    row["api"] = measure(functions, flops, samples=6, repeats=2)
    row["graph"] = measure({name: graph.replay for name, graph in graphs.items()},
                           flops, divisor=8, samples=6, repeats=2)
    row["post_api_vs_fp32"] = check_outputs({n: f() for n, f in functions.items()}, oracle)
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    row["post_graph_vs_fp32"] = check_outputs(graph_outputs, oracle)
    row["post_dispatch"] = dispatch_check(backends, buffers)
    row["graph_launches_per_forward"] = {name: len(backends[name].launches) for name in ("ranked_default", "ranked_selected")}
    row["workspace_bytes"] = {name: backends[name].workspace_bytes for name in ("ranked_default", "ranked_selected")}
    row["compile_artifacts"] = {}
    for name in ("ranked_default", "ranked_selected"):
        row["compile_artifacts"][name] = {}
        for kernel_name in ("gate_up_swiglu", "down"):
            # The original fused preparer keeps its dot kernel separately from
            # base.kernels. Read the actual launch path for both preparers.
            kernel = next(launch.kernel for launch in backends[name].launches
                          if launch.name == kernel_name)
            artifact = compile_artifacts(kernel, f"{row['name']}/{name}/{kernel_name}", artifact_run_id)
            artifact["spills"] = getattr(kernel, "n_spills", None)
            row["compile_artifacts"][name][kernel_name] = artifact
    row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    row["telemetry_after"] = telemetry()
    row["status"] = "passed" if all(m["passed"] for key in ("post_api_vs_fp32", "post_graph_vs_fp32")
                                   for m in row[key].values()) else "failed"
    if row["status"] != "passed":
        raise RuntimeError("post-timing real-input accuracy failed")


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON, existing output refused")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("preserve the existing output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    capture_path = args.capture_root/"capture-summary.json"
    record = dict(schema=1, status="running", experiment="M20 isolated checkpoint layer0 real MoE replay",
                  cases=[], limits=LIMITS,
                  scope="real single-layer GPU-resident fixed-route full forward, not full model or serving",
                  timing_excludes="checkpoint download/loading, text encoding, layer0 attention/router, FP32 oracle, dispatch readback, prepare/compile, telemetry",
                  runtime=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                               triton=triton.__version__), graph_forwards_per_replay=8,
                  selection="frozen M7 shape policy, unchanged; no tuning on real-input cases")
    def save():
        args.output.write_text(json.dumps(record, indent=2)+"\n")
    save()
    try:
        capture = json.loads(capture_path.read_text())
        expected = {(prompt, tokens) for prompt in ("archive", "code") for tokens in (17,193,1024,2049)}
        actual = [(s["prompt"], s["tokens"]) for s in capture["cases"]]
        if capture["status"] != "passed" or len(actual) != 8 or set(actual) != expected:
            raise RuntimeError("complete two-prompt by four-token-count real layer0 capture required")
        if len({s["name"] for s in capture["cases"]}) != 8:
            raise RuntimeError("case names must be unique")
        policy = FrozenMoEPolicy.load(ROOT/"results/m7-policy.json")
        sources = source_hashes(policy)
        before_artifacts = artifact_hashes(args.capture_root, capture)
        record.update(source_hashes=sources, policy_sha256=policy.sha256,
                      policy_provenance_before=policy.verify_provenance(ROOT),
                      capture_summary_sha256=sha256_file(capture_path), artifact_hashes_before=before_artifacts)
        save()
        owner_check()
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if torch.cuda.get_device_capability() != (8, 6):
            raise RuntimeError("M20 requires SM86")
        # Safe-load once; subsequent traces reuse the same real expert tensors.
        first = validate_real_moe_trace(args.capture_root/capture["cases"][0]["manifest"], load_weights=True)
        weight_refs = first.manifest["weights"]
        gpu_weights = {name: tensor.cuda().contiguous() for name, tensor in first.weights.items()}
        for name, shape in (("gate_up", (128,1536,2048)), ("down", (128,2048,768))):
            value = gpu_weights[name]
            if value.dtype != torch.bfloat16 or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise RuntimeError("invalid real expert weights: "+name)
        del first
        for spec in capture["cases"]:
            owner_check()
            row = dict(name=spec["name"], status="running", trace_manifest=spec["manifest"])
            record["cases"].append(row); save()
            torch.cuda.reset_peak_memory_stats()
            try:
                case = validate_real_moe_trace(args.capture_root/spec["manifest"], load_weights=False)
                sibling_spec = next(s for s in capture["cases"] if s["tokens"] == spec["tokens"] and s["prompt"] != spec["prompt"])
                sibling = validate_real_moe_trace(args.capture_root/sibling_spec["manifest"], load_weights=False)
                if case.manifest["weights"] != sibling.manifest["weights"] or case.manifest["weights"] != weight_refs:
                    raise RuntimeError("real prompt traces use different expert weights")
                if case.tensors["x"].shape[0] != spec["tokens"] or sibling.tensors["x"].shape[0] != spec["tokens"]:
                    raise RuntimeError("capture case token count mismatch")
                row["trace_manifest_sha256"] = sha256_file(args.capture_root/spec["manifest"])
                run_case(case, sibling, gpu_weights, policy, row, args.output.stem)
            except Exception as exc:
                row.update(status="failed", failure=str(exc)); raise
            finally:
                save()
            print("real MoE", row["name"], row["status"], flush=True)
            del case, sibling
            gc.collect(); torch.cuda.empty_cache()
        after = source_hashes(policy)
        after_artifacts = artifact_hashes(args.capture_root, capture)
        if after != sources or before_artifacts != after_artifacts:
            raise RuntimeError("source or capture changed during replay")
        record.update(status="passed", source_hashes_after=after,
                      artifact_hashes_after=after_artifacts,
                      policy_provenance_after=policy.verify_provenance(ROOT))
    except Exception as exc:
        record.update(status="failed", failure=str(exc), traceback=traceback.format_exc()); raise
    finally:
        save(); print("real-input record", args.output, flush=True)


if __name__ == "__main__":
    run()
