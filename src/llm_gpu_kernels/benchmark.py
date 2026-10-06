"""Bootstrap validation and CUDA-event diagnostics (not a mature MoE comparison)."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time
import numpy as np
import torch
import triton
from .gemm import GemmConfig, GroupedPlan, dense_gemm, grouped_gemm
from .reference import grouped_reference, moe_reference


ROOT = Path(__file__).resolve().parents[2]
LIMITS = {"relative_rms": 0.008, "normalized_max": 0.008}


def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def error(actual, oracle):
    if actual.shape != oracle.shape:
        raise ValueError(f"output shape mismatch: {tuple(actual.shape)} != {tuple(oracle.shape)}")
    actual, oracle = actual.float(), oracle.float()
    delta = actual - oracle
    relative_rms = (delta.square().mean().sqrt() / oracle.square().mean().sqrt().clamp_min(1e-12)).item()
    normalized_max = (delta.abs().max() / oracle.abs().max().clamp_min(1e-12)).item()
    metrics = {"relative_rms": relative_rms, "normalized_max": normalized_max,
               "max_abs": delta.abs().max().item(), "finite": bool(torch.isfinite(actual).all())}
    metrics["passed"] = metrics["finite"] and all(metrics[k] <= v for k, v in LIMITS.items())
    return metrics


def tensor_hash(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def compile_artifacts(kernel, name, run_id):
    if kernel is None:
        return None
    directory = ROOT / "artifacts" / run_id / name
    directory.mkdir(parents=True, exist_ok=True)
    ptx = kernel.asm["ptx"]
    cubin = kernel.asm["cubin"]
    (directory / "kernel.ptx").write_text(ptx)
    (directory / "kernel.cubin").write_bytes(cubin)
    binary = Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
    sass = command([str(binary), "--dump-sass", str(directory / "kernel.cubin")])
    (directory / "kernel.sass").write_text(sass["stdout"] + "\n" + sass["stderr"])
    return {"path": str(directory.relative_to(ROOT)), "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
            "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
            "ptx_mma_instructions": len(re.findall(r"\bmma\.sync", ptx)),
            "sass_hmma_instructions": len(re.findall(r"\bHMMA\.", sass["stdout"])),
            "sass_disassembly_returncode": sass["returncode"], "registers": kernel.n_regs,
            "shared_memory_bytes": kernel.metadata.shared}


def measure_pair(functions, flops, samples, repeats, rounds_count=3):
    for function in functions.values():
        for _ in range(10):
            function()
    torch.cuda.synchronize()
    raw = {name: [] for name in functions}
    rounds = []
    for round_index in range(rounds_count):
        order = list(functions)
        if round_index % 2:
            order.reverse()
        row = {"order": order, "samples_ms": {}}
        for name in order:
            values = []
            for _ in range(samples):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):
                    functions[name]()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end) / repeats)
            raw[name].extend(values)
            row["samples_ms"][name] = values
        rounds.append(row)
    summaries = {name: {"p50_ms": statistics.median(values),
                        "p95_ms": float(np.percentile(values, 95)),
                        "effective_tflops": flops / statistics.median(values) / 1e9}
                 for name, values in raw.items()}
    return {"rounds": rounds, "summary": summaries}


def run(samples=12, repeats=4):
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261006)
    cfg = GemmConfig()
    gpu = torch.cuda.get_device_properties(0)
    run_id = "bootstrap-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(time.time_ns() % 10**9)
    record = {"schema": 1, "run_id": run_id, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "status": "running", "python": platform.python_version(), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "triton": triton.__version__,
              "gpu": gpu.name, "sm": [gpu.major, gpu.minor], "memory_bytes": gpu.total_memory,
              "config": asdict(cfg), "seed": 20261006, "limits": LIMITS, "samples": samples,
              "repeats": repeats, "tf32": False, "bf16_reduced_precision_reduction": False,
              "timing_scope": "CUDA events over Python launches, hot repeated inputs, preallocated output; no graph; host scheduling gaps included",
              "grouped_exclusions": "host counts/metadata creation, routing, dispatch, activation, down GEMM and combine",
              "baseline": "dense torch.mm/cuBLAS; grouped Python expert-loop torch.mm diagnostic only",
              "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / "src").rglob("*.py"))},
              "git": command(["git", "-C", str(ROOT), "status", "--short", "--branch"]),
              "telemetry_before": command(["nvidia-smi", "--query-gpu=driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"]),
              "ncu": command(["bash", "-c", "command -v ncu"]),
              "validation": [], "benchmarks": []}
    results = ROOT / "results" / (run_id + ".json")
    results.parent.mkdir(exist_ok=True)
    def save():
        results.write_text(json.dumps(record, indent=2) + "\n")
    save()
    try:
        # Predeclared tail/empty/skewed cases: FP32 products of BF16 operands.
        for m, n, k in [(1, 17, 33), (31, 65, 63), (65, 127, 129), (128, 128, 128)]:
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
            metrics = error(dense_gemm(a, b), a.float() @ b.float())
            record["validation"].append({"kind": "dense", "shape": [m, n, k], **metrics})
            if not metrics["passed"]:
                raise RuntimeError("dense numerical check failed")
        for counts in [(0, 1, 31, 33), (127, 0, 2, 0), (32, 32, 32, 32), (0, 0, 0, 0)]:
            a = torch.randn(sum(counts), 67, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(len(counts), 131, 67, device="cuda", dtype=torch.bfloat16)
            plan = GroupedPlan.build(counts, a.device)
            out = grouped_gemm(a, w, plan)
            if sum(counts):
                oracle = grouped_reference(a.float(), w.float(), counts)
                metrics = error(out, oracle)
            else:
                metrics = {"passed": tuple(out.shape) == (0, 131), "empty_no_launch": True}
            record["validation"].append({"kind": "grouped", "counts": counts, "N": 131, "K": 67, **metrics})
            if not metrics["passed"]:
                raise RuntimeError("grouped numerical check failed")
        # Independent per-route oracle confirms combine semantics, including duplicates.
        x = torch.randn(3, 8, device="cuda", dtype=torch.bfloat16)
        gu = torch.randn(3, 10, 8, device="cuda", dtype=torch.bfloat16)
        down = torch.randn(3, 8, 5, device="cuda", dtype=torch.bfloat16)
        ids = torch.tensor([[0, 0], [2, 1], [1, 2]], device="cuda")
        rw = torch.tensor([[0.3, 0.7], [0.1, 0.9], [0.8, 0.2]], device="cuda")
        oracle = torch.zeros_like(x, dtype=torch.float32)
        for token in range(3):
            for slot in range(2):
                e = int(ids[token, slot])
                g, u = (gu[e].float() @ x[token].float()).chunk(2)
                oracle[token] += rw[token, slot] * (down[e].float() @ (torch.nn.functional.silu(g) * u))
        metrics = error(moe_reference(x, gu, down, ids, rw), oracle)
        record["validation"].append({"kind": "moe_reference_combine", **metrics})
        if not metrics["passed"]:
            raise RuntimeError("MoE semantic oracle failed")
        save()
        cases = [("dense_512", None, 512, 512, 512),
                 ("dense_1024", None, 1024, 1024, 1024),
                 ("grouped_balanced", (32,) * 8, 256, 512, 256),
                 ("grouped_skewed", (1, 0, 2, 0, 64, 1, 0, 188), 256, 512, 256),
                 ("qwen_shape_grouped", (4,) * 128, 512, 1536, 2048)]
        for name, counts, m, n, k in cases:
            print(f"Running {name} ...", flush=True)
            torch.cuda.reset_peak_memory_stats()
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = torch.randn((k, n) if counts is None else (len(counts), n, k), device="cuda", dtype=torch.bfloat16)
            out, ref = [torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
            if counts is None:
                custom = lambda: dense_gemm(a, w, out, cfg)
                baseline = lambda: torch.mm(a, w, out=ref)
                _, kernel = dense_gemm(a, w, out, cfg, True)
                oracle = a.float() @ w.float()
            else:
                plan = GroupedPlan.build(counts, a.device, cfg.block_m)
                custom = lambda: grouped_gemm(a, w, plan, out, cfg)
                baseline = lambda: grouped_reference(a, w, counts, ref)
                _, kernel = grouped_gemm(a, w, plan, out, cfg, True)
                oracle = grouped_reference(a.float(), w.float(), counts)
            baseline()
            metrics = error(out, oracle)
            baseline_metrics = error(ref, oracle)
            entry = {"name": name, "shape": [m, n, k], "counts": counts,
                     "flops": 2 * m * n * k, "input_sha256": tensor_hash(a, w),
                     "correctness": metrics, "baseline_correctness": baseline_metrics,
                     "codegen": compile_artifacts(kernel, name, run_id)}
            record["benchmarks"].append(entry)
            if not metrics["passed"] or not baseline_metrics["passed"]:
                raise RuntimeError(f"numerical check failed: {name}")
            entry["timing"] = measure_pair({"triton": custom, "torch": baseline}, entry["flops"], samples, repeats)
            entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            print(json.dumps(entry["timing"]["summary"]), flush=True)
            save()
            del custom, baseline, a, w, out, ref, oracle
        record["status"] = "passed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        record["telemetry_after"] = command(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"])
        save()
    print(f"Saved {results}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if args.samples < 2 or args.repeats < 1:
        parser.error("samples >= 2 and repeats >= 1 required")
    run(args.samples, args.repeats)
