"""M24 one-case mechanism benchmark for the bit-preserving W4 layout candidate."""

import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
import triton

from . import moe_cuda as bf16_cuda
from . import moe_w4_cuda as w4_cuda
from . import moe_w4_layout_cuda as layout_cuda
from .baselines.vllm_moe import prepare_vllm_moe
from .benchmark import ROOT, LIMITS, tensor_hash
from .mature_moe_benchmark import _capture_graph, VLLM_PROVENANCE
from .moe_benchmark import safe_error, telemetry
from .moe_cuda import prepare_cuda_moe
from .moe_full_holdout import _ranked_state
from .moe_full_policy import FrozenMoEPolicy
from .moe_profile import write_record
from .moe_real_input_benchmark import owner_check
from .moe_real_trace import sha256_file, validate_real_moe_trace
from .moe_weight_layout import RepackedGateUp, _digest as layout_digest, repack_gate_up
from .moe_weight_quant import (
    PackedGateUp,
    _digest as packed_digest,
    dequantize_gate_up_reference,
)
from .moe_w4_benchmark import CASES, require_audit
from .moe_w4_cuda import prepare_w4_moe
from .moe_w4_layout_cuda import prepare_w4_layout_moe
from .moe_w4_repeat import _load_native_binary, _resolve_audited_binary
from .optimization_benchmark import measure
from .reference import moe_reference


LAYOUT_BINARY_NAME = "llm_gpu_kernels_m24_w4_layout_sm86_v1.so"
LAYOUT_BINARY_MODULE = Path(LAYOUT_BINARY_NAME).stem
LAYOUT_SOURCE_PATHS = (
    "src/llm_gpu_kernels/cuda_w4_layout_gate_up.cu",
    "src/llm_gpu_kernels/moe_w4_layout_cuda.py",
    "src/llm_gpu_kernels/moe_weight_layout.py",
)
LAYOUT_AUDIT_STATES = (
    "original-api",
    "original-graph",
    "duplicate-expert0-graph",
    "duplicate-expert127-graph",
    "nondefault-stream",
    "restored-api",
    "restored-graph",
)
LAYOUT_REPACKED_DIGEST_FIELDS = (
    "canonical_words_sha256",
    "canonical_scales_sha256",
    "physical_words_sha256",
    "physical_scales_sha256",
)
FULL_ARMS = ("quant_vllm", "quant_cuda_sync", "w4_sync", "w4_layout")
STAGE_ARMS = ("quant_cuda_sync", "w4_sync", "w4_layout")


def _layout_audit_digests(audit):
    """Read the audit's explicit M24 round-trip digest fields."""
    roundtrip = audit.get("layout_roundtrip")
    if not isinstance(roundtrip, dict):
        raise RuntimeError("layout audit is missing layout_roundtrip")
    canonical = roundtrip.get("canonical_v1_sha256")
    physical = roundtrip.get("physical_v2_cpu_sha256")
    if not isinstance(canonical, dict) or not isinstance(physical, dict):
        raise RuntimeError("layout audit is missing canonical-v1 or physical-v2 CPU digests")
    values = {
        "canonical_words_sha256": canonical.get("words_sha256"),
        "canonical_scales_sha256": canonical.get("scales_sha256"),
        "physical_words_sha256": physical.get("words_sha256"),
        "physical_scales_sha256": physical.get("scales_sha256"),
    }
    physical_gpu = roundtrip.get("physical_v2_gpu_sha256_after_transfer")
    canonical_expected = roundtrip.get("canonical_v1_expected_sha256")
    if physical_gpu != physical or canonical_expected != canonical:
        raise RuntimeError("layout audit CPU/GPU or canonical expected digests disagree")
    if roundtrip.get("canonical_v1_gpu_buffers_retained") is not False:
        raise RuntimeError("layout audit must release canonical v1 GPU buffers")
    for name, value in values.items():
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise RuntimeError(f"layout audit contains an invalid {name}")
    return values


def require_layout_audit(path):
    audit = json.loads(path.read_text())
    if not isinstance(audit, dict) or audit.get("status") != "passed":
        raise RuntimeError("layout correctness audit must have status='passed'")

    before = audit.get("source_hashes_before")
    after = audit.get("source_hashes_after")
    if not isinstance(before, dict) or not before or not isinstance(after, dict) or not after:
        raise RuntimeError("layout audit must include source hashes before and after")
    if before != after:
        raise RuntimeError("layout audit source hashes changed between before and after")
    for name, expected in after.items():
        if (not isinstance(name, str) or not isinstance(expected, str)
                or sha256_file(ROOT / name) != expected):
            raise RuntimeError("layout audit source hash does not match live source: " + name)
    for name in LAYOUT_SOURCE_PATHS:
        if name not in after:
            raise RuntimeError("layout audit does not cover required source: " + name)

    binary = audit.get("binary")
    if not isinstance(binary, dict) or binary.get("name") != LAYOUT_BINARY_NAME:
        raise RuntimeError("layout audit must name the fixed M24 extension binary")
    digest_before = binary.get("sha256_before")
    digest_after = binary.get("sha256_after")
    if (not isinstance(digest_before, str) or len(digest_before) != 64
            or any(char not in "0123456789abcdef" for char in digest_before)
            or digest_before != digest_after):
        raise RuntimeError("layout audit binary must have identical before/after SHA-256")
    roundtrip = audit.get("layout_roundtrip", {})
    if (roundtrip.get("passed") is not True
            or roundtrip.get("words_torch_equal") is not True
            or roundtrip.get("scales_torch_equal") is not True):
        raise RuntimeError("layout audit bit-pattern round-trip/equality did not pass")
    _layout_audit_digests(audit)

    cases = audit.get("cases")
    if not isinstance(cases, list) or len(cases) != len(CASES):
        raise RuntimeError("layout audit must contain exactly the six M22 cases")
    case_names = [row.get("case") for row in cases if isinstance(row, dict)]
    if len(case_names) != len(CASES) or len(set(case_names)) != len(CASES) or set(case_names) != set(CASES):
        raise RuntimeError("layout audit case set does not match the six pinned M22 inputs")
    for row in cases:
        if row.get("status") != "passed":
            raise RuntimeError("layout audit case did not pass: " + str(row.get("case")))
        audits = row.get("audits")
        if not isinstance(audits, list) or tuple(item.get("state") for item in audits) != LAYOUT_AUDIT_STATES:
            raise RuntimeError("layout audit case is missing the exact seven required states")
        for item in audits:
            numerical = item.get("numerical", {})
            dispatch = item.get("dispatch", {})
            if numerical.get("passed") is not True or dispatch.get("passed") is not True:
                raise RuntimeError(
                    f"layout audit state failed for {row['case']}: {item.get('state')}"
                )
    zero_tokens = audit.get("zero_tokens")
    if (not isinstance(zero_tokens, dict) or zero_tokens.get("passed") is not True
            or zero_tokens.get("launch_count") != 0
            or zero_tokens.get("dtype") != "bfloat16"
            or zero_tokens.get("shape") != [0, 2048]):
        raise RuntimeError("layout audit T0 contract did not pass")

    for prefix in ("packed_sha256", "pack_record_sha256", "m7_policy_sha256", "m7_training_record_sha256"):
        digest_before = audit.get(prefix + "_before")
        digest_after = audit.get(prefix + "_after")
        if (not isinstance(digest_before, str) or len(digest_before) != 64
                or any(char not in "0123456789abcdef" for char in digest_before)
                or digest_before != digest_after):
            raise RuntimeError(f"layout audit {prefix} changed before/after")
    return audit


def resolve_layout_binary(audit, root):
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"layout extension root is not a directory: {root}")
    binary = audit["binary"]
    expected = root / LAYOUT_BINARY_NAME[:-3] / LAYOUT_BINARY_NAME
    if not expected.is_file():
        raise RuntimeError(f"audited layout binary is absent from the explicit extension root: {expected}")
    resolved = expected.resolve(strict=True)
    digest = sha256_file(resolved)
    if digest != binary["sha256_after"]:
        raise RuntimeError(f"layout binary differs from audit SHA-256: {resolved}")
    return resolved, LAYOUT_BINARY_MODULE, digest


def _verify_layout_vs_v1(record, label):
    comparison = record.get("layout_vs_v1_w4")
    if not isinstance(comparison, dict):
        raise RuntimeError(f"{label} is missing the layout-vs-v1 W4 comparison")
    if not comparison.get("bitwise_equal"):
        raise RuntimeError(f"layout output differs bitwise from v1 W4 at {label}; stopping further timing")
    numerical = comparison.get("numerical", {})
    if not numerical.get("passed"):
        raise RuntimeError(f"layout-vs-v1 W4 numerical comparison failed at {label}")


def _load_layout_audited_binary(path, module_name):
    return _load_native_binary(path, module_name, "layout")


def _native_binary_record(module, path, sha_before):
    actual = Path(module.__file__).resolve(strict=True)
    digest_after = sha256_file(actual)
    if actual != path or digest_after != sha_before:
        raise RuntimeError(f"native binary changed or loaded from an unexpected path: {actual}")
    return {
        "name": actual.name,
        "resolved_path": str(actual),
        "sha256_before_load": sha_before,
        "sha256_after_load": digest_after,
    }


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--pack-record", type=Path, required=True)
    parser.add_argument("--w4-audit", type=Path, required=True)
    parser.add_argument("--bf16-audit", type=Path, required=True)
    parser.add_argument("--layout-audit", type=Path, required=True)
    parser.add_argument("--bf16-extension-root", type=Path, required=True)
    parser.add_argument("--w4-extension-root", type=Path, required=True)
    parser.add_argument("--layout-extension-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    a.capture_root = a.capture_root.resolve()
    if a.output.exists() or a.output.with_suffix(a.output.suffix + ".partial").exists():
        raise RuntimeError("refusing overwrite")
    a.output.parent.mkdir(parents=True, exist_ok=True)

    record = dict(
        schema=1,
        status="running",
        experiment="M24 W4 bit-preserving packed-layout mechanism benchmark",
        case=a.case,
        limits=LIMITS,
        full_graph_repeats=8,
        full_graph_arms=list(FULL_ARMS),
        stage_graph_repeats=8,
        stage_arms=list(STAGE_ARMS),
        sampling={
            "full_forward_rounds": 8,
            "full_forward_order": "four-arm forward/reverse cyclic rotations",
            "stage_rounds": 6,
            "stage_order": "three-arm forward/reverse cyclic rotations",
            "samples_per_round": 6,
            "repeats_per_sample": 2,
            "graph_replay_forwards": 8,
        },
        interpretation=(
            "layout-only mechanism comparison; all W4 arms use the same quantized values, "
            "BF16 down and seven-launch forward; no task-quality claim"
        ),
        native_loading_scope=(
            "the three audited native binaries are loaded directly before preparation; "
            "the worker-facing loader caches are pre-populated, so native extension JIT is not invoked"
        ),
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        runtime=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            triton=triton.__version__,
        ),
    )
    write_record(a.output, record)

    try:
        owner_check()
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if torch.cuda.get_device_capability() != (8, 6):
            raise RuntimeError("SM86 required")

        w4_audit = require_audit(a.w4_audit)
        bf16_audit = require_audit(a.bf16_audit)
        layout_audit = require_layout_audit(a.layout_audit)
        layout_audit_digests = _layout_audit_digests(layout_audit)
        policy = FrozenMoEPolicy.load(ROOT / "results/m7-policy.json")
        policy.verify_provenance(ROOT)
        policy_record_path = (ROOT / policy.training_record_path).resolve(strict=True)
        if layout_audit["m7_policy_sha256_after"] != sha256_file(ROOT / "results/m7-policy.json"):
            raise RuntimeError("layout audit M7 policy SHA does not match the live policy")
        if layout_audit["m7_training_record_sha256_after"] != sha256_file(policy_record_path):
            raise RuntimeError("layout audit M7 training-record SHA does not match the live record")
        audited_records = (a.w4_audit, a.bf16_audit, a.layout_audit, a.pack_record)
        record["record_hashes_before"] = {
            str(path): sha256_file(path) for path in audited_records
        }

        sources = (
            set(w4_audit["source_hashes_after"])
            | set(bf16_audit["source_hashes_after"])
            | set(layout_audit["source_hashes_after"])
            | {
                "src/llm_gpu_kernels/moe_w4_layout_benchmark.py",
                "src/llm_gpu_kernels/moe_w4_benchmark.py",
                "src/llm_gpu_kernels/moe_w4_repeat.py",
                "src/llm_gpu_kernels/moe_w4_cuda.py",
                "src/llm_gpu_kernels/moe_cuda.py",
                "src/llm_gpu_kernels/moe_weight_layout.py",
                "src/llm_gpu_kernels/baselines/vllm_moe.py",
                "src/llm_gpu_kernels/optimization_benchmark.py",
                "src/llm_gpu_kernels/moe_real_input_benchmark.py",
                "docs/M22_W4_PERFORMANCE_PLAN.md",
                "docs/M24_W4_LAYOUT_PLAN.md",
                "results/provenance/vllm-0.10.0-runtime.json",
            }
        )
        record["source_hashes_before"] = {
            name: sha256_file(ROOT / name) for name in sorted(sources)
        }

        if w4_cuda._EXTENSION is not None or bf16_cuda._EXTENSION is not None or layout_cuda._EXTENSION is not None:
            raise RuntimeError("refusing a process with any pre-populated native extension cache")

        binary_paths = {}
        binary_sha_before_load = {}
        loaded_modules = {}
        binary_specs = (
            ("w4", w4_audit, a.w4_extension_root),
            ("bf16", bf16_audit, a.bf16_extension_root),
        )
        for label, audit, root in binary_specs:
            path, module_name, digest = _resolve_audited_binary(audit, root, label)
            binary_paths[label] = path
            binary_sha_before_load[label] = digest
            loaded_modules[label] = _load_native_binary(path, module_name, label)
            if digest != audit["binary"]["sha256_after"]:
                raise RuntimeError(f"{label} binary resolver returned an unexpected digest")

        layout_path, layout_module_name, layout_digest_before = resolve_layout_binary(
            layout_audit, a.layout_extension_root
        )
        binary_paths["layout"] = layout_path
        binary_sha_before_load["layout"] = layout_digest_before
        loaded_modules["layout"] = _load_layout_audited_binary(
            layout_path, layout_module_name
        )
        w4_cuda._EXTENSION = loaded_modules["w4"]
        bf16_cuda._EXTENSION = loaded_modules["bf16"]
        layout_cuda._EXTENSION = loaded_modules["layout"]
        record["binaries"] = {
            label: _native_binary_record(
                loaded_modules[label], path, binary_sha_before_load[label]
            )
            for label, path in binary_paths.items()
        }
        for label, audit in (("w4", w4_audit), ("bf16", bf16_audit), ("layout", layout_audit)):
            if record["binaries"][label]["sha256_after_load"] != audit["binary"]["sha256_after"]:
                raise RuntimeError(f"loaded {label} binary does not match its audit")
        if record["binaries"]["layout"]["sha256_before_load"] != layout_digest_before:
            raise RuntimeError("layout binary digest changed between resolution and load")

        pack_record = json.loads(a.pack_record.read_text())
        record["packed_sha256_before"] = sha256_file(a.packed)
        if (
            pack_record.get("status") != "passed"
            or record["packed_sha256_before"] != pack_record.get("artifact", {}).get("sha256")
            or record["packed_sha256_before"] != w4_audit.get("packed_sha256_after")
            or record["packed_sha256_before"] != layout_audit["packed_sha256_before"]
        ):
            raise RuntimeError("packed preparation/audit mismatch")
        if sha256_file(a.pack_record) != layout_audit["pack_record_sha256_before"]:
            raise RuntimeError("layout audit pack-record SHA does not match the live record")

        data = torch.load(a.packed, map_location="cpu", weights_only=True)
        packed_cpu = PackedGateUp(data["words"], data["scales"], data["metadata"])
        if (
            packed_digest(packed_cpu.words) != packed_cpu.metadata["words_sha256"]
            or packed_digest(packed_cpu.scales) != packed_cpu.metadata["scales_sha256"]
        ):
            raise RuntimeError("canonical v1 packed payload digest mismatch")

        repack_started = time.perf_counter()
        repacked_cpu = repack_gate_up(packed_cpu)
        repack_elapsed_ms = (time.perf_counter() - repack_started) * 1000.0
        if not isinstance(repacked_cpu, RepackedGateUp):
            raise RuntimeError("CPU layout conversion did not return RepackedGateUp")
        actual_layout_digests = {
            name: repacked_cpu.metadata[name] for name in LAYOUT_REPACKED_DIGEST_FIELDS
        }
        if actual_layout_digests != layout_audit_digests:
            raise RuntimeError("CPU repackage canonical/physical digests differ from the layout audit")
        canonical_metadata = repacked_cpu.metadata.get("canonical_v1_metadata", {})
        if (
            canonical_metadata.get("words_sha256") != packed_cpu.metadata.get("words_sha256")
            or canonical_metadata.get("scales_sha256") != packed_cpu.metadata.get("scales_sha256")
        ):
            raise RuntimeError("layout conversion does not preserve the canonical v1 packed digests")
        record["layout_repack"] = {
            "cpu_elapsed_ms": repack_elapsed_ms,
            "timing_scope": "CPU-only bit-preserving permutation; excluded from all API and Graph measurements",
            "audit_digest_match": True,
            "canonical_v1_metadata": packed_cpu.metadata,
            "repacked_metadata": repacked_cpu.metadata,
            "canonical_and_physical_digests": actual_layout_digests,
            "byte_ledger": repacked_cpu.metadata["byte_ledger"],
        }

        packed_v1 = packed_cpu.to("cuda")
        packed_layout = repacked_cpu.to("cuda")
        record["packed_gpu_digests_before"] = {
            "v1_words_sha256": packed_digest(packed_v1.words),
            "v1_scales_sha256": packed_digest(packed_v1.scales),
            "layout_physical_words_sha256": layout_digest(packed_layout.words),
            "layout_physical_scales_sha256": layout_digest(packed_layout.scales),
        }
        if (
            record["packed_gpu_digests_before"]["v1_words_sha256"]
            != packed_cpu.metadata["words_sha256"]
            or record["packed_gpu_digests_before"]["v1_scales_sha256"]
            != packed_cpu.metadata["scales_sha256"]
            or record["packed_gpu_digests_before"]["layout_physical_words_sha256"]
            != repacked_cpu.metadata["physical_words_sha256"]
            or record["packed_gpu_digests_before"]["layout_physical_scales_sha256"]
            != repacked_cpu.metadata["physical_scales_sha256"]
        ):
            raise RuntimeError("GPU copies of the packed layouts differ from their CPU digests")
        quant_gate = dequantize_gate_up_reference(packed_v1)

        capture_file = a.capture_root / "capture-summary.json"
        capture = json.loads(capture_file.read_text())
        if capture.get("status") != "passed":
            raise RuntimeError("capture failed")
        spec = next(
            item for item in capture["cases"]
            if f"{item['prompt']}-T{item['tokens']}" == a.case
        )
        manifest_path = (a.capture_root / spec["manifest"]).resolve()
        case_manifest = json.loads(manifest_path.read_text())
        artifacts = [
            capture_file,
            manifest_path,
            *((manifest_path.parent / ref["path"]).resolve()
              for ref in (case_manifest["trace"], *case_manifest["weights"].values())),
        ]
        record["artifact_hashes_before"] = {
            str(path.relative_to(a.capture_root)): sha256_file(path)
            for path in sorted(set(artifacts))
        }
        if record["artifact_hashes_before"].get("weights.pt") != pack_record["input"]["manifest_weight_sha256"]:
            raise RuntimeError("source checkpoint mismatch")
        case = validate_real_moe_trace(manifest_path, load_weights=True)
        if packed_digest(case.weights["gate_up"]) != packed_cpu.metadata["source_sha256"]:
            raise RuntimeError("BF16 source weight mismatch")

        x, ids, routing_weights = (
            case.tensors[name].cuda().contiguous()
            for name in ("x", "expert_ids", "route_weights")
        )
        gate, down = (
            case.weights[name].cuda().contiguous()
            for name in ("gate_up", "down")
        )
        quant_inputs = (x, quant_gate, down, ids, routing_weights)
        original_inputs = (x, gate, down, ids, routing_weights)

        config, bucket = policy.select(x.shape[0], 128, 8, 2048, 768)
        record.update(
            input_sha256=tensor_hash(x, ids, routing_weights),
            selected_bucket=bucket,
            frozen_config=asdict(config),
            w4_config=dict(BM=32, BN=64, BK=32, warps=4, group_size=128),
            layout_config=dict(
                physical_words_shape=list(repacked_cpu.words.shape),
                physical_scales_shape=list(repacked_cpu.scales.shape),
                layout=repacked_cpu.metadata["layout"],
                arithmetic="same signed W4G128 decode, BF16 WMMA and SwiGLU order as v1",
            ),
            telemetry_before=telemetry(),
        )

        backends = {
            "quant_vllm": prepare_vllm_moe(*quant_inputs),
            "quant_cuda_sync": prepare_cuda_moe(
                *quant_inputs, variant="sync", down_config=config
            ),
            "w4_sync": prepare_w4_moe(
                x, packed_v1, down, ids, routing_weights, down_config=config
            ),
            "w4_layout": prepare_w4_layout_moe(
                x, packed_layout, down, ids, routing_weights, down_config=config
            ),
        }
        if backends["w4_sync"].base.packed is not packed_v1:
            raise RuntimeError("v1 W4 workspace does not own only its canonical pack")
        if backends["w4_layout"].base.packed is not packed_layout:
            raise RuntimeError("layout workspace does not own only its repacked layout")

        provenance = json.loads(VLLM_PROVENANCE.read_text())
        vllm_path = Path(sys.modules["vllm"].__file__).resolve()
        if not vllm_path.is_relative_to(Path(provenance["runtime_path"]).resolve()):
            raise RuntimeError("vLLM loaded from wrong runtime")
        record["vllm_provenance"] = {
            "record_path": str(VLLM_PROVENANCE),
            "record_sha256": sha256_file(VLLM_PROVENANCE),
            "runtime_path": provenance["runtime_path"],
            "loaded_module_path": str(vllm_path),
        }
        record["vllm_configuration"] = backends["quant_vllm"].configuration()

        quant_oracle = moe_reference(*quant_inputs)
        original_oracle = moe_reference(*original_inputs)
        format_proxy = safe_error(quant_oracle, original_oracle)
        format_proxy.pop("passed", None)
        record["quantized_layer_output_proxy"] = {
            **format_proxy,
            "interpretation": "layer-output format proxy only; not model task quality",
        }
        functions = {name: backend.run for name, backend in backends.items()}

        def audit_outputs(field):
            result = {}
            record[field] = result
            observed = {}
            failures = []
            for name, backend in backends.items():
                output = backend.run()
                torch.cuda.synchronize()
                result[name] = {"numerical": safe_error(output, quant_oracle)}
                if not result[name]["numerical"]["passed"]:
                    failures.append("numerical gate failed: " + name)
                if name != "quant_vllm":
                    state = _ranked_state(backend, quant_inputs)
                    state["ranked_dispatch"].pop("rank_values_by_expert", None)
                    result[name]["dispatch"] = state
                    if not state["passed"]:
                        failures.append("dispatch gate failed: " + name)
                if name in ("w4_sync", "w4_layout"):
                    observed[name] = output
            if "w4_sync" in observed and "w4_layout" in observed:
                result["layout_vs_v1_w4"] = {
                    "bitwise_equal": torch.equal(observed["w4_sync"], observed["w4_layout"]),
                    "numerical": safe_error(observed["w4_layout"], observed["w4_sync"]),
                }
            if failures:
                raise RuntimeError(f"{field} failed: " + "; ".join(failures))
            return result

        audit_before = audit_outputs("audit_before")
        _verify_layout_vs_v1(audit_before, "audit_before")

        outputs = {
            name: backend.output if name == "quant_vllm" else backend.base.output
            for name, backend in backends.items()
        }
        graphs = {
            name: _capture_graph(function, 8, x.device)
            for name, function in functions.items()
        }

        def graph_checks(field):
            result = {}
            record[field] = result
            for name, graph in graphs.items():
                graph.replay()
                torch.cuda.synchronize()
                result[name] = safe_error(outputs[name], quant_oracle)
            result["layout_vs_v1_w4"] = {
                "bitwise_equal": torch.equal(outputs["w4_sync"], outputs["w4_layout"]),
                "numerical": safe_error(outputs["w4_layout"], outputs["w4_sync"]),
            }
            if not all(result[name]["passed"] for name in FULL_ARMS):
                raise RuntimeError(f"Graph numerical gate failed at {field}")
            _verify_layout_vs_v1(result, field)
            return result

        graph_checks("graph_numerics_before_timing")
        flops = 6 * ids.numel() * 2048 * 768
        record["full_api"] = measure(
            functions, flops, samples=6, repeats=2
        )
        record["full_graph"] = measure(
            {name: graph.replay for name, graph in graphs.items()},
            flops,
            divisor=8,
            samples=6,
            repeats=2,
        )
        graph_checks("graph_numerics_after_full_timing")

        stages = {
            name: backends[name].launches[4].run
            for name in STAGE_ARMS
        }
        stage_graphs = {
            name: _capture_graph(function, 8, x.device)
            for name, function in stages.items()
        }
        stage_flops = 4 * ids.numel() * 2048 * 768
        record["stage_api"] = measure(
            stages, stage_flops, samples=6, repeats=2
        )
        record["stage_graph"] = measure(
            {name: graph.replay for name, graph in stage_graphs.items()},
            stage_flops,
            divisor=8,
            samples=6,
            repeats=2,
        )
        record["stage_scope"] = (
            "three fixed-metadata gate/up+SwiGLU arms; no routing, down, combine, "
            "or official vLLM stage"
        )

        audit_after = audit_outputs("audit_after")
        _verify_layout_vs_v1(audit_after, "audit_after")
        record["input_sha256_after"] = tensor_hash(x, ids, routing_weights)
        record["source_hashes_after"] = {
            name: sha256_file(ROOT / name) for name in sorted(sources)
        }
        record["artifact_hashes_after"] = {
            str(path.relative_to(a.capture_root)): sha256_file(path)
            for path in sorted(set(artifacts))
        }
        record["record_hashes_after"] = {
            str(path): sha256_file(path) for path in audited_records
        }
        record["packed_sha256_after"] = sha256_file(a.packed)
        record["packed_gpu_digests_after"] = {
            "v1_words_sha256": packed_digest(packed_v1.words),
            "v1_scales_sha256": packed_digest(packed_v1.scales),
            "layout_physical_words_sha256": layout_digest(packed_layout.words),
            "layout_physical_scales_sha256": layout_digest(packed_layout.scales),
        }
        for label, path in binary_paths.items():
            record["binaries"][label]["sha256_after_run"] = sha256_file(path)
        record["telemetry_after"] = telemetry()
        policy.verify_provenance(ROOT)

        if (
            record["source_hashes_before"] != record["source_hashes_after"]
            or record["artifact_hashes_before"] != record["artifact_hashes_after"]
            or record["record_hashes_before"] != record["record_hashes_after"]
            or record["input_sha256"] != record["input_sha256_after"]
            or record["packed_sha256_before"] != record["packed_sha256_after"]
            or record["packed_gpu_digests_before"] != record["packed_gpu_digests_after"]
            or any(
                info["sha256_before_load"] != info["sha256_after_run"]
                for info in record["binaries"].values()
            )
            or record["packed_gpu_digests_after"]["v1_words_sha256"]
            != packed_cpu.metadata["words_sha256"]
            or record["packed_gpu_digests_after"]["v1_scales_sha256"]
            != packed_cpu.metadata["scales_sha256"]
            or record["packed_gpu_digests_after"]["layout_physical_words_sha256"]
            != repacked_cpu.metadata["physical_words_sha256"]
            or record["packed_gpu_digests_after"]["layout_physical_scales_sha256"]
            != repacked_cpu.metadata["physical_scales_sha256"]
        ):
            raise RuntimeError("source, input, packed payload, record, or binary changed")
        record["status"] = "passed"
    except Exception as exc:
        record.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_record(a.output, record)


if __name__ == "__main__":
    run()
