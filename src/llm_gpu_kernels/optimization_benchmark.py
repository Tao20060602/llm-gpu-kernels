"""M3 holdout: apply a previously frozen policy without tuning on these inputs."""
from dataclasses import asdict
import hashlib
import json
import statistics
import time
import numpy as np
import torch
import triton
from .benchmark import ROOT, LIMITS, command, error, tensor_hash, compile_artifacts
from .baseline_benchmark import graphs
from .gemm import GemmConfig, GroupedPlan, grouped_gemm
from .prepared import prepare_grouped_gemm
from .selection import FrozenPolicy
from .reference import grouped_reference
from .baselines.cutlass_grouped import prepare_cutlass_grouped_gemm


def telemetry():
    return command(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw", "--format=csv,noheader"])


def random_routes(tokens, experts, topk, seed, hotspot=False):
    generator = torch.Generator().manual_seed(seed)
    # Uniform random distinct IDs per token. Hotspot uses 16 of 128 experts,
    # unlike the training set's fixed eight-expert modular route pattern.
    span = min(experts, max(topk, 16)) if hotspot else experts
    hot_experts = torch.randperm(experts, generator=generator)[:span]
    scores = torch.rand(tokens, span, generator=generator)
    ids = hot_experts[torch.argsort(scores, dim=1)[:, :topk]]
    order = torch.argsort(ids.flatten(), stable=True)
    counts = torch.bincount(ids.flatten(), minlength=experts).tolist()
    return ids, torch.div(order, topk, rounding_mode="floor"), counts


def measure(functions, flops, divisor=1, samples=6, repeats=2):
    """Forward/reverse cyclic orders put every backend in each position twice."""
    names = list(functions)
    orders = []
    for base in (names, list(reversed(names))):
        orders.extend(base[i:] + base[:i] for i in range(len(base)))
    for function in functions.values():
        for _ in range(5):
            function()
    torch.cuda.synchronize()
    raw = {name: [] for name in names}
    wall = {name: [] for name in names}
    rounds = []
    for order in orders:
        row = {"order": order, "samples_ms_per_gemm": {}, "wall_ms_per_gemm": {}}
        for name in order:
            values, walls = [], []
            for _ in range(samples):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter()
                start.record()
                for _ in range(repeats):
                    functions[name]()
                end.record()
                end.synchronize()
                walls.append((time.perf_counter() - begin) * 1000 / repeats / divisor)
                values.append(start.elapsed_time(end) / repeats / divisor)
            raw[name].extend(values)
            wall[name].extend(walls)
            row["samples_ms_per_gemm"][name] = values
            row["wall_ms_per_gemm"][name] = walls
        rounds.append(row)
    summary = {name: {"p50_ms": statistics.median(values), "p95_ms": float(np.percentile(values, 95)),
                      "effective_tflops": flops / statistics.median(values) / 1e9,
                      "wall_p50_ms": statistics.median(wall[name])}
               for name, values in raw.items()}
    return {"rounds": rounds, "summary": summary, "samples_per_round": samples, "repeats": repeats,
            "divisor": divisor, "wall_scope": "includes event recording and end synchronization; not isolated host-launch latency"}


def validate_configs(policy):
    configurations = set(policy.configs + (policy.fallback,))
    validation = []
    for config in sorted(configurations, key=lambda c: tuple(asdict(c).values())):
        for counts in ((0, 1, 31, 33), (127, 0, 2, 0), (32, 32, 32, 32), (0, 0, 0, 0)):
            a = torch.randn(sum(counts), 67, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(4, 131, 67, device="cuda", dtype=torch.bfloat16)
            plan = prepare_grouped_gemm(a, w, GroupedPlan.build(counts, a.device, config.block_m), config=config)
            if sum(counts) == 0:
                plan.run()
                validation.append({"config": asdict(config), "counts": counts, "empty": True,
                                   "passed": plan.kernel is None and tuple(plan.output.shape) == (0, 131)})
                continue
            metrics = {"eager": error(plan.run(), grouped_reference(a.float(), w.float(), counts))}
            # A value update on a different stream must be seen by the cached launcher.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                a.mul_(0.5)
                plan.run()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            oracle = grouped_reference(a.float(), w.float(), counts)
            metrics["update_nondefault_stream"] = error(plan.output, oracle)
            captured = graphs({"prepared": plan.run}, 3)
            captured["prepared"].replay()
            torch.cuda.synchronize()
            metrics["graph"] = error(plan.output, oracle)
            item = {"config": asdict(config), "counts": counts, "N": 131, "K": 67, "metrics": metrics,
                    "passed": all(v["passed"] for v in metrics.values())}
            validation.append(item)
            del captured, plan, a, w, oracle
    if not all(v["passed"] for v in validation):
        raise RuntimeError("selected configuration boundary validation failed")
    return validation


def run():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    policy_path = ROOT / "results/m3-policy.json"
    policy = FrozenPolicy.load(policy_path)
    scan = json.loads((ROOT / "results/m3-scan-20261006T093119Z.json").read_text())
    if scan["status"] != "passed" or scan["policy_sha256"] != policy.sha256:
        raise RuntimeError("training policy provenance mismatch")
    for path, digest in scan["source_hashes"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("training source changed before holdout: " + path)
    run_id = "m3-holdout-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = ROOT / "results" / (run_id + ".json")
    record = {"schema": 1, "run_id": run_id, "status": "running", "policy_sha256": policy.sha256,
              "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "limits": LIMITS,
              "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / "src").rglob("*.py"))},
              "telemetry_before": telemetry(), "cases": [], "validation": [],
              "scope": "hot fixed-buffer pregrouped GEMM; no routing/dispatch/activation/down/combine or prepare costs",
              "timing": "CUDA events; API includes host gaps; graph has eight GEMMs/replay; forward/reverse cyclic backend orders",
              "routing": "random distinct topk per token; hotspot uniformly samples 16 fixed random experts; seeds73/991"}
    def save():
        destination.write_text(json.dumps(record, indent=2) + "\n")
    save()
    try:
        torch.manual_seed(738991)
        record["validation"] = validate_configs(policy)
        save()
        cases = [(t, 128, 8, 1536, 2048, seed, False) for seed in (73, 991) for t in (17, 193, 1024, 2049)]
        cases += [(t, 128, 8, 1536, 2048, 73, True) for t in (193, 1024)]
        cases += [(t, 128, 8, 2048, 768, 991, False) for t in (193, 1024)]
        cases += [(t, 8, 2, 256, 128, 73, False) for t in (1, 64)]
        for tokens, experts, topk, n, k, seed, hot in cases:
            name = f"T{tokens}_E{experts}_N{n}_K{k}_seed{seed}_{'hot' if hot else 'random'}"
            print("Holdout", name, flush=True)
            torch.manual_seed(seed)
            torch.cuda.reset_peak_memory_stats()
            ids, order, counts = random_routes(tokens, experts, topk, seed, hot)
            x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
            a = x[order.to("cuda")].contiguous()
            w = torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16)
            config, category = policy.select(counts, n, k)
            original = prepare_grouped_gemm(a, w, GroupedPlan.build(counts, a.device))
            selected = prepare_grouped_gemm(a, w, GroupedPlan.build(counts, a.device, config.block_m), config=config)
            cutlass = prepare_cutlass_grouped_gemm(a, w, counts)
            public_out = torch.empty_like(original.output)
            functions = {"public_original": lambda: grouped_gemm(a, w, original.plan, public_out),
                         "prepared_original": original.run, "prepared_selected": selected.run, "cutlass": cutlass.run}
            for function in functions.values():
                function()
            torch.cuda.synchronize()
            oracle = grouped_reference(a.float(), w.float(), counts)
            outputs = {"public_original": public_out, "prepared_original": original.output,
                       "prepared_selected": selected.output, "cutlass": cutlass.output}
            correctness = {backend: error(output, oracle) for backend, output in outputs.items()}
            entry = {"name": name, "T": tokens, "E": experts, "topk": topk, "N": n, "K": k, "seed": seed,
                     "counts": counts, "active_experts": sum(c > 0 for c in counts), "bucket": category,
                     "selected_config": asdict(config), "input_sha256": tensor_hash(a, w),
                     "routing_sha256": tensor_hash(ids), "correctness": correctness, "telemetry_before": telemetry(),
                     "codegen": compile_artifacts(selected.kernel, name, run_id), "spills": selected.kernel.n_spills}
            record["cases"].append(entry)
            if not all(m["passed"] for m in correctness.values()):
                raise RuntimeError("holdout eager numerical failure")
            del oracle
            flops = 2 * a.shape[0] * n * k
            entry["api"] = measure(functions, flops)
            captured = graphs(functions, 8)
            entry["graph"] = measure({backend: graph.replay for backend, graph in captured.items()}, flops, divisor=8)
            oracle = grouped_reference(a.float(), w.float(), counts)
            entry["graph_correctness"] = {backend: error(output, oracle) for backend, output in outputs.items()}
            if not all(m["passed"] for m in entry["graph_correctness"].values()):
                raise RuntimeError("holdout graph numerical failure")
            entry["telemetry_after"] = telemetry()
            entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            print(json.dumps({mode: entry[mode]["summary"] for mode in ("api", "graph")}), flush=True)
            save()
            del captured, functions, outputs, original, selected, cutlass, x, a, w, public_out, oracle
        if hashlib.sha256(policy_path.read_bytes()).hexdigest() != policy.sha256:
            raise RuntimeError("policy changed during holdout")
        record["status"] = "passed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        record["telemetry_after"] = telemetry()
        save()
    print("Saved", destination, flush=True)


if __name__ == "__main__":
    run()
