"""M2: fixed CUTLASS baseline vs untuned Triton, pregrouped GEMM only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import re
import time
import torch
from .benchmark import ROOT, LIMITS, command, error, tensor_hash, measure_pair, compile_artifacts
from .gemm import GemmConfig, GroupedPlan, grouped_gemm
from .reference import grouped_reference
from .baselines.cutlass_grouped import prepare_cutlass_grouped_gemm


def cutlass_artifacts(directory):
    from .baselines.cutlass_grouped import _load_extension
    import triton
    library = Path(_load_extension().__file__)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(library, directory / library.name)
    ninja = library.parent / "build.ninja"
    shutil.copyfile(ninja, directory / "build.ninja")
    tool = Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
    dump = command([str(tool), "--dump-sass", str(library)])
    (directory / "cutlass.sass").write_text(dump["stdout"] + "\n" + dump["stderr"])
    return {"library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "build_ninja": ninja.read_text(), "sass_disassembly_returncode": dump["returncode"],
            "sass_hmma_instructions": len(re.findall(r"\bHMMA\.", dump["stdout"])),
            "artifact_path": str(directory.relative_to(ROOT))}


def routing(tokens, experts, topk, pattern):
    """Auditable synthetic unique routes; no model router or quality claim."""
    if topk > experts:
        raise ValueError("topk must fit expert count")
    span = experts if pattern == "balanced" else max(topk, experts // 16)
    ids = (torch.arange(tokens)[:, None] * topk + torch.arange(topk)[None, :]) % span
    order = torch.argsort(ids.flatten(), stable=True)
    counts = torch.bincount(ids.flatten(), minlength=experts).tolist()
    token_order = torch.div(order, topk, rounding_mode="floor")
    return ids, token_order, counts


def graphs(functions, repeats):
    captured = {}
    for name, function in functions.items():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                function()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                function()
        graph.replay()
        torch.cuda.synchronize()
        captured[name] = graph
    return captured


def run(samples=8, repeats=4, graph_repeats=8):
    import os
    torch.set_num_threads(8)
    torch.manual_seed(20261006)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cfg = GemmConfig()
    gpu = torch.cuda.get_device_properties(0)
    run_id = "m2-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(time.time_ns() % 10**9)
    artifact_root = ROOT / "artifacts" / run_id
    artifact_root.mkdir(parents=True)
    result = ROOT / "results" / (run_id + ".json")
    sources = [*sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "src").rglob("*.cu"))]
    record = {"schema": 1, "run_id": run_id, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "status": "running", "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "gpu": gpu.name, "sm": [gpu.major, gpu.minor], "limits": LIMITS, "seed": 20261006,
              "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "cutlass_commit": command(["git", "-C", os.environ["CUTLASS_PATH"], "rev-parse", "HEAD"]),
              "cutlass_git_status": command(["git", "-C", os.environ["CUTLASS_PATH"], "status", "--short"]),
              "nvcc": command([str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"), "--version"]),
              "cxx": command([os.environ["CXX"], "--version"]),
              "triton_config": cfg.__dict__, "samples": samples, "repeats": repeats,
              "rounds_count": 4,
              "cutlass_config": {"cta": [64, 128, 32], "warp": [32, 64, 32], "instruction": [16, 8, 16],
                                 "stages": 3, "alignment": 8, "schedule": "DeviceOnly", "accumulator": "FP32"},
              "graph_repeats": graph_repeats,
              "scope": "Pregrouped GEMM only; excludes route generation, dispatch, plan/workspace/output allocation, activation/down/combine",
              "timing_modes": {"api": "CUDA events over Python calls; includes host submission gaps and backend-specific wrapper checks",
                               "graph": "CUDA Graph captures repeated fixed-buffer GEMM calls; time divided by number of GEMMs; graph.replay submission gaps may remain"},
              "toolchain_provenance": {name: json.loads((ROOT / "results/provenance" / filename).read_text())
                                       for name, filename in (("cuda", "cuda-toolchain-source.json"),
                                                              ("gcc", "gcc14-source.json"),
                                                              ("glibc_patch", "cuda-glibc-compat.json"))},
              "model_shape_provenance": json.loads((ROOT / "results/provenance/qwen3-30b-a3b-source.json").read_text()),
              "routing_provenance": "balanced modular unique topk IDs; skew concentrates routes on max(topk,E//16) experts; synthetic not observed model routing",
              "telemetry_before": command(["nvidia-smi", "--query-gpu=driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"]),
              "validation": [], "cases": []}
    def save():
        result.write_text(json.dumps(record, indent=2) + "\n")
    save()
    try:
        print("Building/loading fixed CUTLASS extension and validating ...", flush=True)
        for counts, n, k in [((0, 1, 31, 33), 128, 64), ((127, 0, 2, 0), 136, 72),
                              ((32, 32, 32, 32), 256, 128), ((0, 0, 0, 0), 128, 64)]:
            a = torch.randn(sum(counts), k, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(len(counts), n, k, device="cuda", dtype=torch.bfloat16)
            plan = prepare_cutlass_grouped_gemm(a, w, counts)
            actual = plan.run()
            custom_out = torch.empty_like(actual)
            triton_plan = GroupedPlan.build(counts, a.device)
            grouped_gemm(a, w, triton_plan, custom_out)
            torch.cuda.synchronize()
            if sum(counts):
                oracle = grouped_reference(a.float(), w.float(), counts)
                metrics = {"cutlass": error(actual, oracle), "triton": error(custom_out, oracle)}
            else:
                metrics = {backend: {"passed": output.shape == (0, n), "empty": True}
                           for backend, output in (("cutlass", actual), ("triton", custom_out))}
            record["validation"].append({"counts": counts, "N": n, "K": k, "correctness": metrics})
            if not all(value["passed"] for value in metrics.values()):
                raise RuntimeError("CUTLASS boundary numerical check failed")
            save()
        record["cutlass_codegen"] = cutlass_artifacts(artifact_root / "cutlass")
        if record["cutlass_codegen"]["sass_disassembly_returncode"] != 0:
            raise RuntimeError("CUTLASS SASS disassembly failed")
        # Inspect broader backend layout contract explicitly, never silently pad/fallback.
        a = torch.randn(3, 67, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2, 131, 67, device="cuda", dtype=torch.bfloat16)
        try:
            prepare_cutlass_grouped_gemm(a, w, (1, 2))
        except (ValueError, RuntimeError) as exc:
            record["validation"].append({"kind": "alignment_rejection", "passed": True, "reason": str(exc)})
        else:
            raise RuntimeError("expected baseline alignment contract rejection")
        # Contiguous tensors can still have a misaligned storage offset.
        a = torch.randn(3 * 64 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(3, 64)
        w = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
        try:
            prepare_cutlass_grouped_gemm(a, w, (1, 2))
        except (ValueError, RuntimeError) as exc:
            record["validation"].append({"kind": "pointer_alignment_rejection", "passed": True, "reason": str(exc)})
        else:
            raise RuntimeError("expected misaligned pointer rejection")
        shape = record["model_shape_provenance"]["shape"]
        hidden, n_experts, intermediate, model_topk = [shape[k] for k in ("hidden_size", "num_experts", "moe_intermediate_size", "num_experts_per_tok")]
        cases = [("small_decode", 1, 8, 2, 128, 256, "balanced"),
                 ("small_balanced", 64, 8, 2, 128, 256, "balanced"),
                 ("small_skewed", 64, 8, 2, 128, 256, "skewed"),
                 ("medium_top4", 128, 16, 4, 256, 512, "balanced")]
        for tokens, pattern in [(1, "balanced"), (64, "balanced"), (64, "skewed"),
                                (512, "balanced"), (512, "skewed"), (4096, "balanced")]:
            cases.append((f"qwen_T{tokens}_{pattern}", tokens, n_experts, model_topk, hidden, 2 * intermediate, pattern))
        for name, tokens, experts, topk, k, n, pattern in cases:
            print(f"Running {name} ...", flush=True)
            torch.cuda.reset_peak_memory_stats()
            ids, token_order, counts = routing(tokens, experts, topk, pattern)
            x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
            a = x[token_order.to(device="cuda")].contiguous()
            w = torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16)
            triton_out = torch.empty(a.shape[0], n, device="cuda", dtype=torch.bfloat16)
            cutlass_out = torch.empty_like(triton_out)
            torch.cuda.synchronize()
            start = time.perf_counter()
            triton_plan = GroupedPlan.build(counts, a.device, cfg.block_m)
            torch.cuda.synchronize()
            triton_prepare_ms = 1000 * (time.perf_counter() - start)
            start = time.perf_counter()
            cutlass_plan = prepare_cutlass_grouped_gemm(a, w, counts, cutlass_out)
            torch.cuda.synchronize()
            cutlass_prepare_ms = 1000 * (time.perf_counter() - start)
            custom = lambda: grouped_gemm(a, w, triton_plan, triton_out, cfg)
            baseline = cutlass_plan.run
            _, kernel = grouped_gemm(a, w, triton_plan, triton_out, cfg, True)
            baseline()
            oracle = grouped_reference(a.float(), w.float(), counts)
            metrics = {"triton": error(triton_out, oracle), "cutlass": error(cutlass_out, oracle)}
            entry = {"name": name, "T": tokens, "E": experts, "topk": topk, "N": n, "K": k,
                     "rows": a.shape[0], "counts": counts, "routing_sha256": tensor_hash(ids),
                     "input_sha256": tensor_hash(a, w), "correctness": metrics,
                     "cutlass_workspace_bytes": cutlass_plan.workspace_bytes,
                     "active_experts": cutlass_plan.active_experts,
                     "cold_once_prepare_ms": {"triton": triton_prepare_ms, "cutlass": cutlass_prepare_ms},
                     "flops": 2 * a.shape[0] * n * k,
                     "triton_codegen": compile_artifacts(kernel, name, run_id)}
            record["cases"].append(entry)
            if not all(m["passed"] for m in metrics.values()):
                raise RuntimeError(f"numerical check failed: {name}")
            del oracle
            functions = {"triton": custom, "cutlass": baseline}
            entry["api"] = measure_pair(functions, entry["flops"], samples, repeats, rounds_count=4)
            try:
                captured = graphs(functions, graph_repeats)
                graph_functions = {name: graph.replay for name, graph in captured.items()}
                entry["graph"] = measure_pair(graph_functions, entry["flops"] * graph_repeats, samples, repeats, rounds_count=4)
                for backend, value in entry["graph"]["summary"].items():
                    value["p50_ms"] /= graph_repeats
                    value["p95_ms"] /= graph_repeats
                for row in entry["graph"]["rounds"]:
                    row["samples_ms_per_gemm"] = {backend: [v / graph_repeats for v in values]
                                                  for backend, values in row["samples_ms"].items()}
                    row["samples_ms_per_graph_replay"] = row.pop("samples_ms")
                # Validate the graph replay output, not just eager setup output.
                oracle = grouped_reference(a.float(), w.float(), counts)
                entry["graph_correctness"] = {"triton": error(triton_out, oracle), "cutlass": error(cutlass_out, oracle)}
                if not all(m["passed"] for m in entry["graph_correctness"].values()):
                    raise RuntimeError("graph replay numerical check failed")
                del graph_functions, captured, oracle
            except Exception as exc:
                entry["graph_failure"] = repr(exc)
                save()
                raise
            entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            print(json.dumps({mode: entry[mode]["summary"] for mode in ("api", "graph")}), flush=True)
            save()
            del functions, custom, baseline, x, a, w, triton_out, cutlass_out, triton_plan, cutlass_plan
        record["status"] = "passed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        record["telemetry_after"] = command(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"])
        save()
    print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--graph-repeats", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 2 or args.repeats < 1 or args.graph_repeats < 1:
        parser.error("samples >= 2 and repeats/graph-repeats >= 1 required")
    run(args.samples, args.repeats, args.graph_repeats)
