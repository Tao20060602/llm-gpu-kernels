"""M24 W4 physical-layout correctness audit; no performance claim."""

import argparse
import gc
import json
from pathlib import Path
import time
import traceback

import torch

from .benchmark import ROOT, LIMITS, tensor_hash
from .mature_moe_benchmark import _capture_graph
from .moe_benchmark import safe_error
from .moe_full_holdout import _ranked_state
from .moe_full_policy import FrozenMoEPolicy
from .moe_profile import write_record
from .moe_real_trace import sha256_file, validate_real_moe_trace
from .moe_weight_layout import _digest, repack_gate_up, restore_v1_layout
from .moe_weight_quant import PackedGateUp, dequantize_gate_up_reference
from .moe_w4_layout_cuda import _load_w4_layout_gate_up, prepare_w4_layout_moe
from .reference import moe_reference


M24_SOURCE_NAMES = (
    "docs/M24_W4_LAYOUT_PLAN.md",
    "src/llm_gpu_kernels/cuda_w4_layout_gate_up.cu",
    "src/llm_gpu_kernels/moe_w4_layout_cuda.py",
    "src/llm_gpu_kernels/moe_weight_layout.py",
    "src/llm_gpu_kernels/moe_w4_layout_audit.py",
    "src/llm_gpu_kernels/moe_w4_cuda.py",
    "src/llm_gpu_kernels/moe_cuda.py",
    "src/llm_gpu_kernels/moe.py",
    "src/llm_gpu_kernels/moe_ranked.py",
    "src/llm_gpu_kernels/moe_prepared.py",
    "src/llm_gpu_kernels/gemm.py",
    "src/llm_gpu_kernels/moe_weight_quant.py",
    "src/llm_gpu_kernels/moe_full_policy.py",
    "src/llm_gpu_kernels/moe_real_trace.py",
    "src/llm_gpu_kernels/moe_full_holdout.py",
    "src/llm_gpu_kernels/reference.py",
    "src/llm_gpu_kernels/moe_benchmark.py",
    "src/llm_gpu_kernels/mature_moe_benchmark.py",
    "src/llm_gpu_kernels/moe_profile.py",
    "src/llm_gpu_kernels/benchmark.py",
)
EXPECTED_EXTENSION = "llm_gpu_kernels_m24_w4_layout_sm86_v1.so"
EXPECTED_LAUNCHES = (
    "reset", "rank_histogram", "prefix", "rank_dispatch",
    "gate_up_swiglu", "down", "combine",
)
EXPECTED_STATES = (
    "original-api", "original-graph", "duplicate-expert0-graph",
    "duplicate-expert127-graph", "nondefault-stream", "restored-api",
    "restored-graph",
)
MIB = 1024 * 1024


def _source_hashes(policy):
    """Hash explicit M24 dependencies and the frozen policy's verified sources."""
    policy.verify_provenance(ROOT)
    policy_names = tuple(sorted(name for name, _ in policy.source_hashes))
    all_names = tuple(sorted(set(M24_SOURCE_NAMES) | set(policy_names)))
    hashes = {name: sha256_file(ROOT / name) for name in all_names}
    return hashes, policy_names


def _capture_artifact_hashes(capture_root, capture_file, manifest_path, manifest):
    paths = [
        capture_file,
        manifest_path,
        *((manifest_path.parent / ref["path"]).resolve()
          for ref in (manifest["trace"], *manifest["weights"].values())),
    ]
    unique_paths = sorted(set(paths))
    return {
        str(path.relative_to(capture_root)): sha256_file(path)
        for path in unique_paths
    }, unique_paths


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--pack-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.capture_root = args.capture_root.expanduser().resolve(strict=True)
    args.packed = args.packed.expanduser().resolve(strict=True)
    args.pack_record = args.pack_record.expanduser().resolve(strict=True)
    args.output = args.output.expanduser().resolve()
    partial_output = args.output.with_suffix(args.output.suffix + ".partial")
    if args.output.exists() or partial_output.exists():
        raise RuntimeError("refusing overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    record = dict(
        schema=1,
        status="running",
        experiment="M24 W4A16 physical-layout full-forward correctness",
        limits=LIMITS,
        cases=[],
        interpretation=(
            "layout-only bit-preserving conversion with the original v1 dequantized-BF16 "
            "oracle; not model task quality or performance"
        ),
    )
    write_record(args.output, record)
    try:
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if torch.cuda.get_device_capability() != (8, 6):
            raise RuntimeError("SM86 required")

        pack_record_bytes = args.pack_record.read_bytes()
        pack_record = json.loads(pack_record_bytes)
        if pack_record.get("status") != "passed":
            raise RuntimeError("offline format preparation did not pass")
        record["packed_sha256_before"] = sha256_file(args.packed)
        if record["packed_sha256_before"] != pack_record["artifact"]["sha256"]:
            raise RuntimeError("packed artifact differs from audited preparation")
        quantizer_path = ROOT / "src/llm_gpu_kernels/moe_weight_quant.py"
        expected_quantizer_sha = pack_record["format_source"]["quantizer_sha256_before_after"]["after"]
        if sha256_file(quantizer_path) != expected_quantizer_sha:
            raise RuntimeError("packing reference source differs from preparation")
        record["pack_record_sha256_before"] = sha256_file(args.pack_record)

        data = torch.load(args.packed, map_location="cpu", weights_only=True)
        packed_cpu = PackedGateUp(data["words"], data["scales"], data["metadata"])
        if (_digest(packed_cpu.words) != packed_cpu.metadata["words_sha256"]
                or _digest(packed_cpu.scales) != packed_cpu.metadata["scales_sha256"]):
            raise RuntimeError("packed tensor hash mismatch")
        record["format_metadata"] = packed_cpu.metadata

        repack_started = time.perf_counter()
        repacked_cpu = repack_gate_up(packed_cpu)
        cpu_repack_seconds = time.perf_counter() - repack_started
        restore_started = time.perf_counter()
        restored_cpu = restore_v1_layout(repacked_cpu)
        cpu_restore_seconds = time.perf_counter() - restore_started
        words_equal = torch.equal(restored_cpu.words, packed_cpu.words)
        scales_equal = torch.equal(restored_cpu.scales, packed_cpu.scales)
        canonical_hashes = {
            "words_sha256": packed_cpu.metadata["words_sha256"],
            "scales_sha256": packed_cpu.metadata["scales_sha256"],
        }
        physical_cpu_hashes = {
            "words_sha256": _digest(repacked_cpu.words),
            "scales_sha256": _digest(repacked_cpu.scales),
        }
        ledger = repacked_cpu.metadata["byte_ledger"]
        roundtrip_passed = (
            words_equal and scales_equal
            and canonical_hashes["words_sha256"] == repacked_cpu.metadata["canonical_words_sha256"]
            and canonical_hashes["scales_sha256"] == repacked_cpu.metadata["canonical_scales_sha256"]
            and physical_cpu_hashes["words_sha256"] == repacked_cpu.metadata["physical_words_sha256"]
            and physical_cpu_hashes["scales_sha256"] == repacked_cpu.metadata["physical_scales_sha256"]
        )
        record["layout_roundtrip"] = {
            "passed": bool(roundtrip_passed),
            "words_torch_equal": bool(words_equal),
            "scales_torch_equal": bool(scales_equal),
            "cpu_repack_seconds": cpu_repack_seconds,
            "cpu_repack_timing_scope": (
                "repack_gate_up including v1 input and v2 output digests; excludes restore and H2D"
            ),
            "cpu_restore_seconds": cpu_restore_seconds,
            "canonical_v1_sha256": canonical_hashes,
            "physical_v2_cpu_sha256": physical_cpu_hashes,
            "byte_ledger": ledger,
            "memory_accounting": {
                "canonical_v1_resident_bytes": ledger["canonical_packed_total_bytes"],
                "additional_repacked_bytes": ledger["additional_repacked_tensor_bytes"],
                "additional_roundtrip_restore_bytes": ledger["canonical_packed_total_bytes"],
                "peak_canonical_repacked_restored_bytes": (
                    ledger["canonical_packed_total_bytes"]
                    + ledger["physical_packed_total_bytes"]
                    + ledger["canonical_packed_total_bytes"]
                ),
                "unit": "bytes",
            },
            "oracle_source": "original_v1_cpu_packed_words_and_scales",
        }
        write_record(args.output, record)
        if not roundtrip_passed:
            raise RuntimeError("M24 layout failed complete v1 bit-pattern round trip")
        if ledger["canonical_packed_total_bytes"] != 198 * MIB:
            raise RuntimeError("unexpected canonical packed+scale byte count")
        del restored_cpu
        gc.collect()

        # Build the correctness oracle from the original v1 CPU buffers, not
        # from the restored copy. No canonical v1 packed buffers are retained on GPU.
        quantized_weight_cpu = dequantize_gate_up_reference(packed_cpu)
        if not torch.isfinite(quantized_weight_cpu).all().item():
            raise RuntimeError("nonfinite original-v1 quantized reference")
        quantized_weight = quantized_weight_cpu.cuda().contiguous()
        del quantized_weight_cpu
        repacked_gpu = repacked_cpu.to("cuda")
        del repacked_cpu
        gc.collect()
        physical_gpu_hashes = {
            "words_sha256": _digest(repacked_gpu.words),
            "scales_sha256": _digest(repacked_gpu.scales),
        }
        record["layout_roundtrip"]["physical_v2_gpu_sha256_after_transfer"] = physical_gpu_hashes
        record["layout_roundtrip"]["canonical_v1_expected_sha256"] = canonical_hashes
        record["layout_roundtrip"]["canonical_v1_gpu_buffers_retained"] = False
        if physical_gpu_hashes != physical_cpu_hashes:
            raise RuntimeError("physical M24 digests changed during H2D transfer")

        policy = FrozenMoEPolicy.load(ROOT / "results/m7-policy.json")
        source_hashes_before, policy_source_names = _source_hashes(policy)
        record["source_hash_scope"] = {
            "explicit_m24_sources": list(M24_SOURCE_NAMES),
            "additional_m7_policy_verified_sources": list(policy_source_names),
        }
        record["source_hashes_before"] = source_hashes_before
        record["m7_policy_sha256_before"] = sha256_file(ROOT / "results/m7-policy.json")
        policy_record_path = (ROOT / policy.training_record_path).resolve(strict=True)
        record["m7_training_record_sha256_before"] = sha256_file(policy_record_path)
        write_record(args.output, record)

        module = _load_w4_layout_gate_up()
        binary_path = Path(module.__file__).resolve(strict=True)
        if binary_path.name != EXPECTED_EXTENSION:
            raise RuntimeError(f"unexpected M24 extension basename: {binary_path.name}")
        record["binary"] = {
            "name": binary_path.name,
            "resolved_path": str(binary_path),
            "sha256_before": sha256_file(binary_path),
        }

        capture_file = args.capture_root / "capture-summary.json"
        capture = json.loads(capture_file.read_text())
        specifications = sorted(
            (spec for spec in capture["cases"] if spec["tokens"] in (17, 193, 1024)),
            key=lambda spec: (spec["prompt"], spec["tokens"]),
        )
        if capture["status"] != "passed" or len(specifications) != 6:
            raise RuntimeError("expected six passed real inputs")

        for spec in specifications:
            manifest_path = (args.capture_root / spec["manifest"]).resolve(strict=True)
            manifest = json.loads(manifest_path.read_text())
            artifacts_before, artifact_paths = _capture_artifact_hashes(
                args.capture_root, capture_file, manifest_path, manifest
            )
            if artifacts_before.get("weights.pt") != pack_record["input"]["manifest_weight_sha256"]:
                raise RuntimeError("source checkpoint differs from format preparation")
            case = validate_real_moe_trace(manifest_path, load_weights=True)
            if _digest(case.weights["gate_up"]) != packed_cpu.metadata["source_sha256"]:
                raise RuntimeError("packed weights do not match pinned BF16 source")

            x = case.tensors["x"].cuda().contiguous()
            ids = case.tensors["expert_ids"].cuda().contiguous()
            routing_weights = case.tensors["route_weights"].cuda().contiguous()
            down = case.weights["down"].cuda().contiguous()
            original_gate = case.weights["gate_up"].cuda().contiguous()
            originals = (x.clone(), ids.clone(), routing_weights.clone())
            inputs = (x, quantized_weight, down, ids, routing_weights)
            input_sha = tensor_hash(x, ids, routing_weights)
            config, bucket = policy.select(x.shape[0], 128, 8, 2048, 768)
            row = dict(
                case=f"{spec['prompt']}-T{spec['tokens']}",
                selected_bucket=bucket,
                input_sha256_before=input_sha,
                artifact_hashes_before=artifacts_before,
                audits=[],
                status="running",
            )
            record["cases"].append(row)
            write_record(args.output, record)

            backend = prepare_w4_layout_moe(
                x, repacked_gpu, down, ids, routing_weights, down_config=config
            )
            if tuple(launch.name for launch in backend.launches) != EXPECTED_LAUNCHES:
                raise RuntimeError("unexpected M24 seven-launch DAG")

            def check(label, replay=None):
                reference = moe_reference(*inputs)
                (backend.run if replay is None else replay.replay)()
                torch.cuda.synchronize()
                numerical = safe_error(backend.base.output, reference)
                dispatch = _ranked_state(backend, inputs)
                dispatch["ranked_dispatch"].pop("rank_values_by_expert", None)
                row["audits"].append(dict(state=label, numerical=numerical, dispatch=dispatch))
                write_record(args.output, record)
                if not numerical["passed"] or not dispatch["passed"]:
                    raise RuntimeError(f"{row['case']} / {label} failed")
                if label == "original-api":
                    bf16_reference = moe_reference(x, original_gate, down, ids, routing_weights)
                    row["format_forward_change"] = safe_error(reference, bf16_reference)
                    row["cuda_vs_original_bf16"] = safe_error(backend.base.output, bf16_reference)
                    row["format_forward_change"].pop("passed", None)
                    row["cuda_vs_original_bf16"].pop("passed", None)
                    row["format_forward_change_interpretation"] = (
                        "proxy differences, not implementation pass/fail or task scores"
                    )

            check("original-api")
            graph = _capture_graph(backend.run, 8, x.device)
            check("original-graph", graph)
            x.copy_(originals[0] * -0.75)
            ids.fill_(0)
            routing_weights.copy_(originals[2] * 0.5)
            check("duplicate-expert0-graph", graph)
            ids.fill_(127)
            check("duplicate-expert127-graph", graph)

            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                x.copy_(originals[0] * 0.5)
                ids.copy_(originals[1])
                routing_weights.copy_(originals[2] * -0.5)
                backend.run()
            torch.cuda.synchronize()
            reference = moe_reference(*inputs)
            numerical = safe_error(backend.base.output, reference)
            dispatch = _ranked_state(backend, inputs)
            dispatch["ranked_dispatch"].pop("rank_values_by_expert", None)
            row["audits"].append(
                dict(state="nondefault-stream", numerical=numerical, dispatch=dispatch)
            )
            write_record(args.output, record)
            if not numerical["passed"] or not dispatch["passed"]:
                raise RuntimeError("nondefault-stream gate failed")

            for target, original in zip((x, ids, routing_weights), originals):
                target.copy_(original)
            check("restored-api")
            check("restored-graph", graph)
            row["input_sha256_after"] = tensor_hash(x, ids, routing_weights)
            row["artifact_hashes_after"] = {
                str(path.relative_to(args.capture_root)): sha256_file(path)
                for path in artifact_paths
            }
            if row["input_sha256_after"] != input_sha or row["artifact_hashes_after"] != artifacts_before:
                raise RuntimeError("input/capture changed")
            if tuple(item["state"] for item in row["audits"]) != EXPECTED_STATES:
                raise RuntimeError("M24 case did not record the exact seven required states")
            row["status"] = "passed"

            if record.get("zero_tokens") is None:
                empty = prepare_w4_layout_moe(
                    x[:0], repacked_gpu, down, ids[:0], routing_weights[:0]
                )
                output = empty.run()
                if (output.shape != (0, 2048) or output.dtype != torch.bfloat16
                        or empty.launches):
                    raise RuntimeError("T0 contract failed")
                record["zero_tokens"] = {
                    "passed": True,
                    "launch_count": 0,
                    "dtype": "bfloat16",
                    "shape": [0, 2048],
                }
                del empty, output
            write_record(args.output, record)
            del graph, backend, inputs, originals, case, reference
            del x, ids, routing_weights, down, original_gate
            gc.collect()

        source_hashes_after, policy_source_names_after = _source_hashes(policy)
        record["source_hashes_after"] = source_hashes_after
        record["binary"]["sha256_after"] = sha256_file(binary_path)
        record["packed_sha256_after"] = sha256_file(args.packed)
        record["pack_record_sha256_after"] = sha256_file(args.pack_record)
        record["m7_policy_sha256_after"] = sha256_file(ROOT / "results/m7-policy.json")
        record["m7_training_record_sha256_after"] = sha256_file(policy_record_path)
        canonical_cpu_after = {
            "words_sha256": _digest(packed_cpu.words),
            "scales_sha256": _digest(packed_cpu.scales),
        }
        physical_gpu_final = {
            "words_sha256": _digest(repacked_gpu.words),
            "scales_sha256": _digest(repacked_gpu.scales),
        }
        record["layout_digest_after"] = {
            "canonical_v1_cpu_sha256": canonical_cpu_after,
            "physical_v2_gpu_sha256": physical_gpu_final,
            "canonical_v1_gpu_buffers_retained": False,
        }
        record["source_hash_scope"]["additional_m7_policy_verified_sources"] = list(
            policy_source_names_after
        )
        policy.verify_provenance(ROOT)
        if (record["source_hashes_before"] != record["source_hashes_after"]
                or record["binary"]["sha256_before"] != record["binary"]["sha256_after"]
                or record["packed_sha256_before"] != record["packed_sha256_after"]
                or record["pack_record_sha256_before"] != record["pack_record_sha256_after"]
                or record["m7_policy_sha256_before"] != record["m7_policy_sha256_after"]
                or record["m7_training_record_sha256_before"] != record["m7_training_record_sha256_after"]
                or canonical_cpu_after != canonical_hashes
                or physical_gpu_final != physical_cpu_hashes
                or len(policy_source_names_after) != len(policy_source_names)):
            raise RuntimeError("source/binary/packed/layout inputs changed")
        record["status"] = "passed"
    except Exception as exc:
        record.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_record(args.output, record)


if __name__ == "__main__":
    run()
