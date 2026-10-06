"""M3 preregistered bounded parameter scan; no holdout-driven selection."""
from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import torch
import triton
from triton.runtime.errors import OutOfResources
from triton.compiler.errors import CompilationError
from .benchmark import ROOT, error, tensor_hash, command, measure_pair, compile_artifacts
from .baseline_benchmark import routing, graphs
from .gemm import GemmConfig, GroupedPlan
from .prepared import prepare_grouped_gemm
from .reference import grouped_reference


TRAIN = [(64, "balanced"), (64, "skewed"), (512, "balanced"), (512, "skewed"), (4096, "balanced")]


def candidates():
    tiles = [(16,64,32),(16,128,32),(16,256,32),(32,64,32),(32,128,32),(32,256,32),
             (64,64,32),(64,128,32),(64,256,32),(32,128,64),(64,128,64),(64,256,64)]
    configs = [GemmConfig(m,n,k,w,s) for m,n,k in tiles for w,s in ((4,2),(4,3),(8,3))]
    configs += [GemmConfig(m,n,k,4,4) for m,n,k in ((32,128,32),(64,128,32),(64,128,64),(64,256,64))]
    return configs


def key(config):
    return "-".join(str(v) for v in asdict(config).values())


def bucket(counts):
    active = sum(c > 0 for c in counts)
    density = sum(counts) / active if active else 0
    return "small" if density <= 8 else "medium" if density <= 64 else "large"


def graph_time(function, samples=3):
    # Single graph replay per event window: no multi-replay Python loop in sweep.
    for _ in range(3): function()
    torch.cuda.synchronize()
    values=[]
    for _ in range(samples):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();function();end.record();end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def fast_pair(left,right,graph_repeats=4):
    rows=[]
    for order in (("original","candidate"),("candidate","original")):
        row={"order":order,"samples_ms_per_gemm":{}}
        for name in order:
            values=graph_time(left if name=="original" else right)
            row["samples_ms_per_gemm"][name]=[v/graph_repeats for v in values]
        rows.append(row)
    medians={name:statistics.median([v for row in rows for v in row["samples_ms_per_gemm"][name]])
             for name in ("original","candidate")}
    return {"rounds":rows,"p50_ms":medians,"ratio":medians["candidate"]/medians["original"]}


def make_case(tokens,pattern,seed=20261006):
    torch.manual_seed(seed)
    ids,order,counts=routing(tokens,128,8,pattern)
    x=torch.randn(tokens,2048,device="cuda",dtype=torch.bfloat16)
    a=x[order.to("cuda")].contiguous()
    w=torch.randn(128,1536,2048,device="cuda",dtype=torch.bfloat16)
    oracle=grouped_reference(a.float(),w.float(),counts)
    return a,w,counts,oracle,ids


def run_scan():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    run_id="m3-scan-"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())
    destination=ROOT/"results"/(run_id+".json")
    configs=candidates()
    record={"schema":1,"run_id":run_id,"status":"running","kind":"training_only",
            "seed":20261006,"configs":[asdict(c) for c in configs],"cases":[],"recheck":[],
            "source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((ROOT/"src").rglob("*.py"))},
            "plan_sha256":hashlib.sha256((ROOT/"docs/M3_EXPERIMENT_PLAN.md").read_bytes()).hexdigest()}
    def save(): destination.write_text(json.dumps(record,indent=2)+"\n")
    save()
    try:
        for tokens,pattern in TRAIN:
            print(f"Training T{tokens}/{pattern}",flush=True)
            a,w,counts,oracle,ids=make_case(tokens,pattern)
            original=prepare_grouped_gemm(a,w,GroupedPlan.build(counts,a.device),config=GemmConfig())
            captured=graphs({"original":original.run},4)
            case={"T":tokens,"pattern":pattern,"counts":counts,"bucket":bucket(counts),
                  "input_sha256":tensor_hash(a,w),"routing_sha256":tensor_hash(ids),"candidates":[],
                  "telemetry_before":command(["nvidia-smi","--query-gpu=temperature.gpu,clocks.sm,power.draw","--format=csv,noheader"])}
            record["cases"].append(case)
            for index,config in enumerate(configs):
                entry={"key":key(config),"config":asdict(config),"status":"running"}
                case["candidates"].append(entry)
                try:
                    plan=GroupedPlan.build(counts,a.device,config.block_m)
                    selected=prepare_grouped_gemm(a,w,plan,config=config)
                    selected.run();torch.cuda.synchronize()
                    metrics=error(selected.output,oracle)
                    entry["correctness"]=metrics
                    if not metrics["passed"]:
                        entry["status"]="numerical_failure"
                        save();continue
                    candidate_graph=graphs({"candidate":selected.run},4)
                    entry["timing"]=fast_pair(captured["original"].replay,candidate_graph["candidate"].replay)
                    entry["resources"]={"registers":selected.kernel.n_regs,"spills":selected.kernel.n_spills,
                                        "shared_memory_bytes":selected.kernel.metadata.shared}
                    entry["status"]="passed"
                    del candidate_graph,selected,plan
                except (OutOfResources,CompilationError) as exc:
                    entry["status"]="unsupported_config";entry["error"]=str(exc)
                save()
                if (index+1)%10==0:
                    print(f"  {index+1}/{len(configs)} configurations",flush=True)
            case["telemetry_after"]=command(["nvidia-smi","--query-gpu=temperature.gpu,clocks.sm,power.draw","--format=csv,noheader"])
            del captured,original,a,w,oracle
        decisions={}
        for category in ("small","medium","large"):
            cases=[c for c in record["cases"] if c["bucket"]==category]
            scores=[]
            for config in configs:
                rows=[next(c for c in case["candidates"] if c["key"]==key(config)) for case in cases]
                if all(row["status"]=="passed" for row in rows):
                    score=math.exp(sum(math.log(row["timing"]["ratio"]) for row in rows)/len(rows))
                    scores.append({"config":asdict(config),"score":score})
            scores.sort(key=lambda row:row["score"])
            winner=scores[0] if scores and scores[0]["score"]<=0.95 else {"config":asdict(GemmConfig()),"score":1.0}
            decisions[category]={"scan_winner":winner,"rankings":scores,"selected":winner["config"]}
        record["decisions"]=decisions
        save()
        # Only training is revisited here. Holdout is in a separate executable.
        for tokens,pattern in TRAIN:
            a,w,counts,oracle,ids=make_case(tokens,pattern)
            category=bucket(counts)
            config=GemmConfig(**decisions[category]["selected"])
            original=prepare_grouped_gemm(a,w,GroupedPlan.build(counts,a.device))
            selected=prepare_grouped_gemm(a,w,GroupedPlan.build(counts,a.device,config.block_m),config=config)
            captured=graphs({"original":original.run,"selected":selected.run},8)
            timing=measure_pair({name:g.replay for name,g in captured.items()},2*a.shape[0]*1536*2048*8,6,2,rounds_count=4)
            ratio=timing["summary"]["selected"]["p50_ms"]/timing["summary"]["original"]["p50_ms"]
            metrics=error(selected.output,oracle)
            record["recheck"].append({"T":tokens,"pattern":pattern,"bucket":category,"config":asdict(config),
                                      "ratio":ratio,"timing_ms_per_graph":timing,"correctness":metrics})
            if not metrics["passed"]: raise RuntimeError("recheck numerical failure")
            save()
            del captured,original,selected,a,w,oracle
        for category in decisions:
            rows=[r for r in record["recheck"] if r["bucket"]==category]
            score=math.exp(sum(math.log(r["ratio"]) for r in rows)/len(rows))
            decisions[category]["recheck_score"]=score
            if score>0.95:decisions[category]["selected"]=asdict(GemmConfig())
        policy={"schema":1,"source_run":run_id,"domain":{"N":1536,"K":2048},
                "density_thresholds":[8,64],"configs":{name:row["selected"] for name,row in decisions.items()},
                "fallback":asdict(GemmConfig()),"status":"frozen_before_holdout"}
        policy_path=ROOT/"results/m3-policy.json"
        if policy_path.exists():raise RuntimeError("policy already exists; preserve it and choose an explicit new run")
        policy_path.write_text(json.dumps(policy,indent=2)+"\n")
        record["policy_sha256"]=hashlib.sha256(policy_path.read_bytes()).hexdigest()
        record["status"]="passed"
    except Exception as exc:
        record["status"]="failed";record["error"]=repr(exc);raise
    finally:save()
    print("Saved",destination,"and frozen policy",policy_path,flush=True)


if __name__=="__main__":run_scan()
