"""Fixed full-forward A/B for local gate/up+SwiGLU fusion; no parameter scan."""
import hashlib
import json
import time
import torch
from .benchmark import ROOT, error, tensor_hash, measure_pair, compile_artifacts
from .baseline_benchmark import graphs
from .moe import prepare_moe
from .moe_fused import prepare_fused_moe
from .moe_benchmark import (_functional_specs, _prepare_case, make_inputs, safe_error,
                            staged_bf16_oracle, telemetry, source_hashes, validate_dispatch)
from .reference import moe_reference


def metrics(baseline, fused, buffers):
    reference = moe_reference(*buffers)
    staged = staged_bf16_oracle(*buffers)
    values = {"baseline_fp32": safe_error(baseline.output, reference),
              "fused_fp32": safe_error(fused.base.output, reference),
              "fused_staged": safe_error(fused.base.output, staged),
              "fused_unfused": safe_error(fused.base.output, baseline.output)}
    return {"errors": values, "passed": all(m["passed"] for m in values.values()),
            "fused_unfused_bitwise_equal": torch.equal(baseline.output, fused.base.output)}


def normalize_graph(timing, count):
    for summary in timing["summary"].values():
        summary["p50_ms"] /= count
        summary["p95_ms"] /= count
    for row in timing["rounds"]:
        raw = row.pop("samples_ms")
        row["samples_ms_per_graph_replay"] = raw
        row["samples_ms_per_forward"] = {k: [v / count for v in values] for k, values in raw.items()}
    return timing


def run():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    run_id = "m4-fusion-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = ROOT / "results" / (run_id + ".json")
    record = {"schema": 1, "run_id": run_id, "status": "running", "source_hashes": source_hashes(),
              "plan_sha256": hashlib.sha256((ROOT / "docs/M4_FUSION_EXPERIMENT_PLAN.md").read_bytes()).hexdigest(),
              "scope": "complete GPU fixed-route forward; local epilogue fusion only, fixed tile; allocated workspace unchanged",
              "timing": "4 AB/BA rounds, 6 samples, 2 calls/sample; graph8 forwards/replay; hot fixed buffers",
              "telemetry_before": telemetry(), "validation": [], "cases": []}
    def save():
        destination.write_text(json.dumps(record, indent=2) + "\n")
    save()
    try:
        for spec in _functional_specs():
            print("Validate fusion", spec["name"], flush=True)
            buffers, baseline = _prepare_case(spec, torch.device("cuda", 0))
            fused = prepare_fused_moe(*buffers)
            baseline.run(); fused.run(); torch.cuda.synchronize()
            item = {"name": spec["name"], "eager": metrics(baseline, fused, buffers),
                    "dispatch": validate_dispatch(fused.base, buffers[0], buffers[3])}
            if spec["T"]:
                captured = graphs({"baseline": baseline.run, "fused": fused.run}, 2)
                buffers[0].mul_(0.5)
                buffers[1].mul_(-1)
                buffers[2].mul_(0.25)
                buffers[3].add_(1).remainder_(spec["E"])
                buffers[4].mul_(-0.5)
                for graph in captured.values():
                    graph.replay()
                torch.cuda.synchronize()
                fused.check_inputs(); baseline.check_inputs()
                item["graph_updated_values"] = metrics(baseline, fused, buffers)
                item["graph_dispatch"] = validate_dispatch(fused.base, buffers[0], buffers[3])
                del captured
            item["passed"] = all(value["passed"] for value in item.values() if isinstance(value, dict))
            record["validation"].append(item)
            save()
            if not item["passed"]:
                raise RuntimeError("fusion boundary validation failed")
            del baseline, fused, buffers
        specs = [(t, 16, 4, 128, 64, "random") for t in (17, 193)]
        specs += [(t, 128, 8, 2048, 768, pattern) for t in (17, 193, 1024) for pattern in ("random", "hot")]
        for t, e, topk, h, intermediate, pattern in specs:
            name = f"T{t}_E{e}_H{h}_I{intermediate}_{pattern}"
            print("Fusion A/B", name, flush=True)
            buffers = make_inputs(tokens=t, experts=e, topk=topk, hidden=h, intermediate=intermediate,
                                  id_dtype=torch.int32, pattern=pattern, seed=73, device=torch.device("cuda", 0))
            baseline = prepare_moe(*buffers)
            fused = prepare_fused_moe(*buffers)
            baseline.run(); fused.run(); torch.cuda.synchronize()
            baseline.check_inputs(); fused.check_inputs()
            entry = {"name": name, "T": t, "E": e, "topk": topk, "H": h, "I": intermediate,
                     "pattern": pattern, "seed": 73, "input_sha256": tensor_hash(*buffers),
                     "correctness": metrics(baseline, fused, buffers), "telemetry_before": telemetry(),
                     "workspace_bytes": {"baseline": baseline.workspace_bytes, "fused": fused.base.workspace_bytes},
                     "codegen": compile_artifacts(fused.kernel, name, run_id),
                     "spills": fused.kernel.n_spills}
            record["cases"].append(entry)
            if not entry["correctness"]["passed"]:
                raise RuntimeError("fusion eager numerical failure")
            functions = {"baseline": baseline.run, "fused": fused.run}
            flops = 6 * t * topk * h * intermediate
            entry["api"] = measure_pair(functions, flops, 6, 2, rounds_count=4)
            captured = graphs(functions, 8)
            timing = measure_pair({k: graph.replay for k, graph in captured.items()}, flops * 8, 6, 2, rounds_count=4)
            entry["graph"] = normalize_graph(timing, 8)
            entry["graph_correctness"] = metrics(baseline, fused, buffers)
            entry["telemetry_after"] = telemetry()
            if not entry["graph_correctness"]["passed"]:
                raise RuntimeError("fusion graph numerical failure")
            print(json.dumps(entry["graph"]["summary"]), flush=True)
            save()
            del captured, functions, baseline, fused, buffers
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
