#!/usr/bin/env python3
"""Validate and summarize the fixed six-case M24 W4 layout run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "run_manifest.json"
EXPERIMENT = "M24 W4 bit-preserving packed-layout mechanism benchmark"
CASES = (
    "archive-T17",
    "archive-T193",
    "archive-T1024",
    "code-T17",
    "code-T193",
    "code-T1024",
)
SMALL_CASES = ("archive-T17", "archive-T193", "code-T17", "code-T193")
LARGE_CASES = ("archive-T1024", "code-T1024")
FULL_ARMS = ("quant_vllm", "quant_cuda_sync", "w4_sync", "w4_layout")
STAGE_ARMS = ("quant_cuda_sync", "w4_sync", "w4_layout")
PROJECT_ARMS = frozenset(("quant_cuda_sync", "w4_sync", "w4_layout"))
SECTIONS = {
    "full_api": (FULL_ARMS, 8, 1),
    "full_graph": (FULL_ARMS, 8, 8),
    "stage_api": (STAGE_ARMS, 6, 1),
    "stage_graph": (STAGE_ARMS, 6, 8),
}
FULL_REFERENCES = ("w4_sync", "quant_cuda_sync", "quant_vllm")
STAGE_REFERENCES = ("w4_sync", "quant_cuda_sync")
ERROR_FIELDS = ("relative_rms", "normalized_max", "max_abs")
LIMIT = 0.008
EXPECTED_SAMPLING = {
    "full_forward_rounds": 8,
    "full_forward_order": "four-arm forward/reverse cyclic rotations",
    "stage_rounds": 6,
    "stage_order": "three-arm forward/reverse cyclic rotations",
    "samples_per_round": 6,
    "repeats_per_sample": 2,
    "graph_replay_forwards": 8,
}
LAYOUT_DIGEST_FIELDS = (
    "canonical_words_sha256",
    "canonical_scales_sha256",
    "physical_words_sha256",
    "physical_scales_sha256",
)
HASH_CHARS = frozenset("0123456789abcdef")


class SummaryError(RuntimeError):
    """An input is incomplete, inconsistent, or outside the frozen schema."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fail(where: str, message: str) -> None:
    raise SummaryError(f"{where}: {message}")


def object_at(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(where, "expected an object")
    return value


def finite_number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(where, "expected a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "finite and positive" if positive else "finite"
        fail(where, f"expected a {qualifier} number")
    return result


def require_int(value: Any, expected: int, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        fail(where, f"expected {expected}")


def require_hash(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HASH_CHARS for character in value)
    ):
        fail(where, "expected a lowercase SHA-256")
    return value


def safe_relative_path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        fail(where, "expected a safe relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        fail(where, "expected a safe relative path")
    return value


def safe_hash_map(value: Any, where: str) -> dict[str, str]:
    mapping = object_at(value, where)
    if not mapping:
        fail(where, "expected a nonempty hash map")
    result: dict[str, str] = {}
    for raw_path, raw_digest in mapping.items():
        path = safe_relative_path(raw_path, f"{where}/path")
        result[path] = require_hash(raw_digest, f"{where}/{path}")
    return result


def repo_relative(path: Path, where: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError) as exc:
        fail(where, f"path is absent or outside the repository ({exc})")


def expected_orders(arms: tuple[str, ...]) -> list[list[str]]:
    result: list[list[str]] = []
    for base in (list(arms), list(reversed(arms))):
        result.extend(base[index:] + base[:index] for index in range(len(base)))
    return result


def percentile_linear(values: list[float], percentile: float) -> float:
    """Match numpy.percentile's default method='linear' used by the worker."""
    ordered = sorted(values)
    if not ordered:
        fail("samples", "cannot calculate a percentile from an empty sample set")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def require_stat_match(actual: float, calculated: float, where: str) -> None:
    if not math.isclose(actual, calculated, rel_tol=1e-12, abs_tol=1e-12):
        fail(where, f"stored value {actual} differs from raw-sample value {calculated}")


def validate_numeric_check(value: Any, where: str) -> dict[str, Any]:
    check = object_at(value, where)
    if check.get("finite") is not True or check.get("passed") is not True:
        fail(where, "finite/oracle pass flags are not true")
    metrics: dict[str, float] = {}
    for field in ERROR_FIELDS:
        metric = finite_number(check.get(field), f"{where}/{field}")
        if metric < 0:
            fail(f"{where}/{field}", "error metric must be nonnegative")
        metrics[field] = metric
    if metrics["relative_rms"] > LIMIT:
        fail(f"{where}/relative_rms", f"exceeds {LIMIT}")
    if metrics["normalized_max"] > LIMIT:
        fail(f"{where}/normalized_max", f"exceeds {LIMIT}")
    return {"finite": True, "passed": True, **metrics}


def validate_dispatch(value: Any, where: str) -> dict[str, Any]:
    dispatch = object_at(value, where)
    if dispatch.get("passed") is not True:
        fail(f"{where}/passed", "ranked dispatch did not pass")
    if dispatch.get("check_inputs_passed") is not True:
        fail(f"{where}/check_inputs_passed", "input validation did not pass")
    status = dispatch.get("status")
    if isinstance(status, bool) or status != 0:
        fail(f"{where}/status", "expected zero")
    ranked = object_at(dispatch.get("ranked_dispatch"), f"{where}/ranked_dispatch")
    if ranked.get("passed") is not True:
        fail(f"{where}/ranked_dispatch/passed", "ranked routing check did not pass")
    return {
        "status": 0,
        "check_inputs_passed": True,
        "ranked_dispatch_passed": True,
        "passed": True,
    }


def validate_audit(value: Any, where: str, *, dispatch_required: bool) -> dict[str, Any]:
    audit = object_at(value, where)
    expected = set(FULL_ARMS) | {"layout_vs_v1_w4"}
    if set(audit) != expected:
        fail(where, f"expected exactly the four full-forward arms and layout comparison")
    compact: dict[str, Any] = {}
    for arm in FULL_ARMS:
        arm_where = f"{where}/{arm}"
        result = object_at(audit[arm], arm_where)
        numerical_record = result.get("numerical") if dispatch_required else result
        numerical = validate_numeric_check(
            numerical_record,
            f"{arm_where}/numerical" if dispatch_required else arm_where,
        )
        dispatch = None
        if arm in PROJECT_ARMS and dispatch_required:
            dispatch = validate_dispatch(result.get("dispatch"), f"{arm_where}/dispatch")
        elif "dispatch" in result:
            fail(f"{arm_where}/dispatch", "dispatch data is not expected in this numerical check")
        compact[arm] = {"numerical": numerical, "dispatch": dispatch}

    comparison = object_at(audit["layout_vs_v1_w4"], f"{where}/layout_vs_v1_w4")
    if comparison.get("bitwise_equal") is not True:
        fail(f"{where}/layout_vs_v1_w4/bitwise_equal", "layout and v1 W4 outputs differ")
    compact["layout_vs_v1_w4"] = {
        "bitwise_equal": True,
        "numerical": validate_numeric_check(
            comparison.get("numerical"), f"{where}/layout_vs_v1_w4/numerical"
        ),
    }
    return compact


def validate_measurement(
    value: Any,
    where: str,
    arms: tuple[str, ...],
    round_count: int,
    divisor: int,
) -> dict[str, Any]:
    measurement = object_at(value, where)
    rounds = measurement.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != round_count:
        fail(f"{where}/rounds", f"expected exactly {round_count} rounds")
    require_int(measurement.get("samples_per_round"), 6, f"{where}/samples_per_round")
    require_int(measurement.get("repeats"), 2, f"{where}/repeats")
    require_int(measurement.get("divisor"), divisor, f"{where}/divisor")
    wall_scope = measurement.get("wall_scope")
    if not isinstance(wall_scope, str) or not wall_scope:
        fail(f"{where}/wall_scope", "missing wall-clock timing scope")

    orders = expected_orders(arms)
    gpu_samples = {arm: [] for arm in arms}
    wall_samples = {arm: [] for arm in arms}
    retained_rounds: list[dict[str, Any]] = []
    for index, raw_round in enumerate(rounds):
        round_where = f"{where}/rounds/{index}"
        row = object_at(raw_round, round_where)
        if row.get("order") != orders[index]:
            fail(f"{round_where}/order", "does not match the fixed cyclic forward/reverse order")
        samples = object_at(row.get("samples_ms_per_gemm"), f"{round_where}/samples_ms_per_gemm")
        wall = object_at(row.get("wall_ms_per_gemm"), f"{round_where}/wall_ms_per_gemm")
        if set(samples) != set(arms) or set(wall) != set(arms):
            fail(round_where, "timing arm set differs from the frozen schema")
        retained_samples: dict[str, list[float]] = {}
        retained_wall: dict[str, list[float]] = {}
        for arm in arms:
            arm_where = f"{round_where}/{arm}"
            sample_values = samples[arm]
            wall_values = wall[arm]
            if not isinstance(sample_values, list) or len(sample_values) != 6:
                fail(f"{arm_where}/samples_ms_per_gemm", "expected six samples")
            if not isinstance(wall_values, list) or len(wall_values) != 6:
                fail(f"{arm_where}/wall_ms_per_gemm", "expected six samples")
            validated_samples = [
                finite_number(number, f"{arm_where}/samples_ms_per_gemm/{sample_index}", positive=True)
                for sample_index, number in enumerate(sample_values)
            ]
            validated_wall = [
                finite_number(number, f"{arm_where}/wall_ms_per_gemm/{sample_index}", positive=True)
                for sample_index, number in enumerate(wall_values)
            ]
            gpu_samples[arm].extend(validated_samples)
            wall_samples[arm].extend(validated_wall)
            retained_samples[arm] = validated_samples
            retained_wall[arm] = validated_wall
        retained_rounds.append({
            "order": list(orders[index]),
            "samples_ms_per_gemm": retained_samples,
            "wall_ms_per_gemm": retained_wall,
        })

    reported = object_at(measurement.get("summary"), f"{where}/summary")
    if set(reported) != set(arms):
        fail(f"{where}/summary", "timing arm set differs from the frozen schema")
    compact_summary: dict[str, dict[str, float]] = {}
    for arm in arms:
        arm_where = f"{where}/summary/{arm}"
        stats = object_at(reported[arm], arm_where)
        calculated = {
            "p50_ms": float(statistics.median(gpu_samples[arm])),
            "p95_ms": percentile_linear(gpu_samples[arm], 95.0),
            "wall_p50_ms": float(statistics.median(wall_samples[arm])),
        }
        for field, expected in calculated.items():
            stored = finite_number(stats.get(field), f"{arm_where}/{field}", positive=True)
            require_stat_match(stored, expected, f"{arm_where}/{field}")
        # Retain the worker's derived throughput for transparency; it is never
        # used as an acceptance criterion or in candidate/reference ratios.
        tflops = finite_number(
            stats.get("effective_tflops"), f"{arm_where}/effective_tflops", positive=True
        )
        compact_summary[arm] = {**calculated, "effective_tflops_reported": tflops}
    return {
        "round_count": round_count,
        "samples_per_round": 6,
        "repeats_per_sample": 2,
        "divisor": divisor,
        "wall_scope": wall_scope,
        "arms": compact_summary,
        "rounds": retained_rounds,
    }


def logical_record_hashes(value: Any, where: str) -> dict[str, dict[str, str]]:
    mapping = object_at(value, where)
    if len(mapping) != 4:
        fail(where, "expected W4, BF16, layout audits and pack-record identity")
    result: dict[str, dict[str, str]] = {}
    for raw_path, raw_digest in mapping.items():
        if not isinstance(raw_path, str) or not raw_path:
            fail(f"{where}/path", "expected a nonempty record path")
        basename = raw_path.replace("\\", "/").rsplit("/", 1)[-1]
        if not basename or basename in (".", ".."):
            fail(f"{where}/path", "record path has no safe basename")
        lowered = raw_path.lower()
        if "pack" in lowered or "quant-format" in lowered or "format-preparation" in lowered:
            role = "pack_record"
        elif "layout" in lowered:
            role = "layout_audit"
        elif "w4" in lowered:
            role = "w4_audit"
        elif "bf16" in lowered or "cuda-audit" in lowered or "cuda_audit" in lowered:
            role = "bf16_audit"
        else:
            fail(f"{where}/{basename}", "cannot identify audit-record role")
        if role in result:
            fail(where, f"duplicate {role} identity")
        result[role] = {"basename": basename, "sha256": require_hash(raw_digest, f"{where}/{basename}")}
    expected_roles = {"w4_audit", "bf16_audit", "layout_audit", "pack_record"}
    if set(result) != expected_roles:
        fail(where, "audit and pack-record identities are incomplete")
    return result


def validate_binaries(value: Any, where: str) -> dict[str, dict[str, str]]:
    binaries = object_at(value, where)
    expected = {"w4", "bf16", "layout"}
    if set(binaries) != expected:
        fail(where, "expected exactly the W4, BF16, and layout binaries")
    compact: dict[str, dict[str, str]] = {}
    for role in ("w4", "bf16", "layout"):
        item_where = f"{where}/{role}"
        item = object_at(binaries[role], item_where)
        name = item.get("name")
        if not isinstance(name, str) or not name or Path(name).name != name:
            fail(f"{item_where}/name", "expected a binary basename")
        resolved = item.get("resolved_path")
        if not isinstance(resolved, str) or not Path(resolved).is_absolute():
            fail(f"{item_where}/resolved_path", "expected an absolute worker-recorded path")
        if Path(resolved).name != name:
            fail(f"{item_where}/resolved_path", "basename differs from binary name")
        before = require_hash(item.get("sha256_before_load"), f"{item_where}/sha256_before_load")
        after_load = require_hash(item.get("sha256_after_load"), f"{item_where}/sha256_after_load")
        after_run = require_hash(item.get("sha256_after_run"), f"{item_where}/sha256_after_run")
        if before != after_load or before != after_run:
            fail(item_where, "binary SHA changed before/after load or run")
        compact[role] = {"basename": name, "sha256": before}
    return compact


def read_worker(path: Path, case: str, expected_sha256: str) -> tuple[dict[str, Any], str]:
    where = f"{case}/{path.name}"
    try:
        data = path.read_bytes()
    except OSError as exc:
        fail(where, f"cannot read worker JSON ({exc})")
    actual_sha256 = sha256_bytes(data)
    if actual_sha256 != expected_sha256:
        fail(where, "worker JSON SHA differs from run_manifest.json")
    try:
        record = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(where, f"invalid JSON ({exc})")
    return object_at(record, where), actual_sha256


def validate_worker(
    record: dict[str, Any],
    case: str,
    source_path: str,
    record_sha256: str,
) -> dict[str, Any]:
    where = case
    if record.get("schema") != 1:
        fail(f"{where}/schema", "expected schema 1")
    if record.get("experiment") != EXPERIMENT:
        fail(f"{where}/experiment", "does not match the frozen M24 experiment")
    if record.get("status") != "passed":
        fail(f"{where}/status", "worker did not pass")
    if record.get("case") != case:
        fail(f"{where}/case", "does not match manifest case")
    if not isinstance(record.get("head"), str) or not record["head"]:
        fail(f"{where}/head", "missing source revision")

    if record.get("full_graph_arms") != list(FULL_ARMS):
        fail(f"{where}/full_graph_arms", "arm order differs from the frozen schema")
    if record.get("stage_arms") != list(STAGE_ARMS):
        fail(f"{where}/stage_arms", "arm order differs from the frozen schema")
    require_int(record.get("full_graph_repeats"), 8, f"{where}/full_graph_repeats")
    require_int(record.get("stage_graph_repeats"), 8, f"{where}/stage_graph_repeats")
    if record.get("sampling") != EXPECTED_SAMPLING:
        fail(f"{where}/sampling", "sampling contract differs from the frozen schema")

    runtime_raw = object_at(record.get("runtime"), f"{where}/runtime")
    runtime: dict[str, str] = {}
    for field in ("python", "torch", "cuda", "triton"):
        value = runtime_raw.get(field)
        if not isinstance(value, str) or not value:
            fail(f"{where}/runtime/{field}", "expected a version string")
        runtime[field] = value

    source_before = safe_hash_map(record.get("source_hashes_before"), f"{where}/source_hashes_before")
    source_after = safe_hash_map(record.get("source_hashes_after"), f"{where}/source_hashes_after")
    if source_before != source_after:
        fail(f"{where}/source_hashes_after", "source hashes changed during worker run")
    worker_source = "src/llm_gpu_kernels/moe_w4_layout_benchmark.py"
    if worker_source not in source_before:
        fail(f"{where}/source_hashes_before/{worker_source}", "worker source identity is missing")
    for relative, digest in source_before.items():
        source_file = (ROOT / relative).resolve()
        try:
            source_file.relative_to(ROOT.resolve())
            actual_digest = sha256_bytes(source_file.read_bytes())
        except (OSError, ValueError) as exc:
            fail(f"{where}/source_hashes_before/{relative}", f"source is absent or outside repo ({exc})")
        if actual_digest != digest:
            fail(f"{where}/source_hashes_before/{relative}", "recorded SHA differs from current source")

    record_before = logical_record_hashes(record.get("record_hashes_before"), f"{where}/record_hashes_before")
    record_after = logical_record_hashes(record.get("record_hashes_after"), f"{where}/record_hashes_after")
    if record_before != record_after:
        fail(f"{where}/record_hashes_after", "audit or pack-record identity changed during worker run")

    artifact_before = safe_hash_map(record.get("artifact_hashes_before"), f"{where}/artifact_hashes_before")
    artifact_after = safe_hash_map(record.get("artifact_hashes_after"), f"{where}/artifact_hashes_after")
    if artifact_before != artifact_after:
        fail(f"{where}/artifact_hashes_after", "capture input artifacts changed during worker run")
    input_before = require_hash(record.get("input_sha256"), f"{where}/input_sha256")
    input_after = require_hash(record.get("input_sha256_after"), f"{where}/input_sha256_after")
    if input_before != input_after:
        fail(f"{where}/input_sha256_after", "input tensor digest changed during worker run")

    packed_before = require_hash(record.get("packed_sha256_before"), f"{where}/packed_sha256_before")
    packed_after = require_hash(record.get("packed_sha256_after"), f"{where}/packed_sha256_after")
    if packed_before != packed_after:
        fail(f"{where}/packed_sha256_after", "packed artifact changed during worker run")
    gpu_digests_before = safe_hash_map(
        record.get("packed_gpu_digests_before"), f"{where}/packed_gpu_digests_before"
    )
    gpu_digests_after = safe_hash_map(
        record.get("packed_gpu_digests_after"), f"{where}/packed_gpu_digests_after"
    )
    expected_gpu_digest_fields = {
        "v1_words_sha256",
        "v1_scales_sha256",
        "layout_physical_words_sha256",
        "layout_physical_scales_sha256",
    }
    if set(gpu_digests_before) != expected_gpu_digest_fields:
        fail(f"{where}/packed_gpu_digests_before", "packed GPU digest fields differ from schema")
    if gpu_digests_before != gpu_digests_after:
        fail(f"{where}/packed_gpu_digests_after", "packed GPU digests changed during worker run")

    binaries = validate_binaries(record.get("binaries"), f"{where}/binaries")
    repack = object_at(record.get("layout_repack"), f"{where}/layout_repack")
    if repack.get("audit_digest_match") is not True:
        fail(f"{where}/layout_repack/audit_digest_match", "CPU repack digest did not match audit")
    repack_ms = finite_number(
        repack.get("cpu_elapsed_ms"), f"{where}/layout_repack/cpu_elapsed_ms"
    )
    if repack_ms < 0:
        fail(f"{where}/layout_repack/cpu_elapsed_ms", "expected a nonnegative duration")
    repack_digests = safe_hash_map(
        repack.get("canonical_and_physical_digests"),
        f"{where}/layout_repack/canonical_and_physical_digests",
    )
    if set(repack_digests) != set(LAYOUT_DIGEST_FIELDS):
        fail(f"{where}/layout_repack/canonical_and_physical_digests", "digest fields differ from schema")
    repacked_metadata = object_at(repack.get("repacked_metadata"), f"{where}/layout_repack/repacked_metadata")
    for field, digest in repack_digests.items():
        if repacked_metadata.get(field) != digest:
            fail(f"{where}/layout_repack/repacked_metadata/{field}", "differs from recorded repack digest")

    correctness = {
        "audit_before": validate_audit(
            record.get("audit_before"), f"{where}/audit_before", dispatch_required=True
        ),
        "graph_numerics_before_timing": validate_audit(
            record.get("graph_numerics_before_timing"),
            f"{where}/graph_numerics_before_timing",
            dispatch_required=False,
        ),
        "graph_numerics_after_full_timing": validate_audit(
            record.get("graph_numerics_after_full_timing"),
            f"{where}/graph_numerics_after_full_timing",
            dispatch_required=False,
        ),
        "audit_after": validate_audit(
            record.get("audit_after"), f"{where}/audit_after", dispatch_required=True
        ),
    }

    format_proxy = object_at(
        record.get("quantized_layer_output_proxy"), f"{where}/quantized_layer_output_proxy"
    )
    if format_proxy.get("finite") is not True:
        fail(f"{where}/quantized_layer_output_proxy/finite", "expected finite format comparison")
    compact_proxy: dict[str, float] = {}
    for field in ERROR_FIELDS:
        compact_proxy[field] = finite_number(
            format_proxy.get(field), f"{where}/quantized_layer_output_proxy/{field}"
        )
        if compact_proxy[field] < 0:
            fail(f"{where}/quantized_layer_output_proxy/{field}", "expected a nonnegative metric")

    timings: dict[str, Any] = {}
    for section, (arms, rounds, divisor) in SECTIONS.items():
        timings[section] = validate_measurement(
            record.get(section), f"{where}/{section}", arms, rounds, divisor
        )

    vllm = object_at(record.get("vllm_provenance"), f"{where}/vllm_provenance")
    vllm_path = vllm.get("record_path")
    if not isinstance(vllm_path, str) or not vllm_path:
        fail(f"{where}/vllm_provenance/record_path", "missing provenance record path")
    vllm_sha = require_hash(vllm.get("record_sha256"), f"{where}/vllm_provenance/record_sha256")
    vllm_identity = {
        "record_basename": vllm_path.replace("\\", "/").rsplit("/", 1)[-1],
        "record_sha256": vllm_sha,
    }
    if not vllm_identity["record_basename"]:
        fail(f"{where}/vllm_provenance/record_path", "provenance path has no basename")

    selected_config = record.get("frozen_config")
    if not isinstance(selected_config, dict):
        fail(f"{where}/frozen_config", "missing selected M7 config")
    bucket = record.get("selected_bucket")
    if not isinstance(bucket, str) or not bucket:
        fail(f"{where}/selected_bucket", "missing selected M7 bucket")

    return {
        "case": case,
        "worker": {"path": source_path, "sha256": record_sha256, "status": "passed"},
        "head": record["head"],
        "runtime": runtime,
        "vllm_provenance": vllm_identity,
        "source_hashes": source_before,
        "record_hashes": record_before,
        "artifact_hashes_capture_relative": artifact_before,
        "input_sha256": input_before,
        "packed_identity": {
            "file_sha256": packed_before,
            "gpu_digests": gpu_digests_before,
            "layout_repack_digests": repack_digests,
        },
        "binaries": binaries,
        "repack_cpu": {
            "cpu_elapsed_ms": repack_ms,
            "timing_scope": repack.get("timing_scope"),
            "byte_ledger": repack.get("byte_ledger"),
        },
        "native_loading_scope": record.get("native_loading_scope"),
        "interpretation": record.get("interpretation"),
        "stage_scope": record.get("stage_scope"),
        "selected_bucket": bucket,
        "frozen_config": selected_config,
        "layout_config": record.get("layout_config"),
        "quantized_layer_output_proxy": {
            "finite": True,
            **compact_proxy,
            "quality_gate_applied": False,
        },
        "correctness": correctness,
        "timings": timings,
        "telemetry": {
            "before": record.get("telemetry_before"),
            "after": record.get("telemetry_after"),
        },
    }


def resolve_manifest_output(raw_path: Any, input_dir: Path, case: str) -> Path:
    where = f"run_manifest.json/workers/{case}/output"
    if not isinstance(raw_path, str) or not raw_path:
        fail(where, "missing worker output path")
    candidate_path = Path(raw_path)
    if candidate_path.is_absolute():
        candidates = [candidate_path.resolve()]
    else:
        candidates = [(ROOT / candidate_path).resolve(), (input_dir / candidate_path).resolve()]
    inside: list[Path] = []
    for candidate in candidates:
        try:
            candidate.relative_to(input_dir.resolve())
            if candidate not in inside:
                inside.append(candidate)
        except ValueError:
            continue
    if len(inside) != 1:
        fail(where, "output path does not resolve uniquely inside the selected input directory")
    path = inside[0]
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        fail(where, "worker output must be inside the repository")
    if path.stem != case or path.suffix != ".json":
        fail(where, f"expected a {case}.json worker result")
    return path


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        fail("aggregate", "geometric mean requires finite positive ratios")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def validate_manifest_identities(
    manifest: dict[str, Any],
    source_hashes: dict[str, str],
    record_hashes: dict[str, dict[str, str]],
    binaries: dict[str, dict[str, str]],
    packed_identity: dict[str, Any],
    vllm_provenance: dict[str, str],
) -> dict[str, dict[str, str]]:
    where = MANIFEST_NAME
    if manifest.get("worker_is_frozen") is not True:
        fail(f"{where}/worker_is_frozen", "run manifest does not identify a frozen worker")
    worker_source = "src/llm_gpu_kernels/moe_w4_layout_benchmark.py"
    manifest_worker_sha = require_hash(
        manifest.get("worker_source_sha256"), f"{where}/worker_source_sha256"
    )
    if source_hashes.get(worker_source) != manifest_worker_sha:
        fail(f"{where}/worker_source_sha256", "differs from the worker source hash map")
    manifest_sources = safe_hash_map(
        manifest.get("source_hashes_after"), f"{where}/source_hashes_after"
    )
    expected_manifest_extras = {
        "scripts/activate-vllm.sh",
        "scripts/activate.sh",
    }
    if not set(source_hashes).issubset(manifest_sources):
        missing = sorted(set(source_hashes) - set(manifest_sources))
        fail(f"{where}/source_hashes_after", f"missing worker source entries: {', '.join(missing)}")
    manifest_extras = set(manifest_sources) - set(source_hashes)
    if manifest_extras != expected_manifest_extras:
        unexpected = sorted(manifest_extras ^ expected_manifest_extras)
        fail(f"{where}/source_hashes_after", f"unexpected source-set difference: {', '.join(unexpected)}")
    for relative, digest in source_hashes.items():
        if manifest_sources.get(relative) != digest:
            fail(f"{where}/source_hashes_after/{relative}", "differs from worker source identity")
    for relative, digest in manifest_sources.items():
        source_file = (ROOT / relative).resolve()
        try:
            source_file.relative_to(ROOT.resolve())
            actual = sha256_bytes(source_file.read_bytes())
        except (OSError, ValueError) as exc:
            fail(f"{where}/source_hashes_after/{relative}", f"source is absent or outside repo ({exc})")
        if actual != digest:
            fail(f"{where}/source_hashes_after/{relative}", "recorded SHA differs from current source")

    inputs = object_at(manifest.get("inputs"), f"{where}/inputs")
    input_roles = ("w4_audit", "bf16_audit", "layout_audit", "pack_record")
    binary_for_audit = {"w4_audit": "w4", "bf16_audit": "bf16", "layout_audit": "layout"}
    sanitized: dict[str, dict[str, str]] = {}
    for role in input_roles:
        input_where = f"{where}/inputs/{role}"
        item = object_at(inputs.get(role), input_where)
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            fail(f"{input_where}/path", "missing manifest input path")
        basename = raw_path.replace("\\", "/").rsplit("/", 1)[-1]
        if basename != record_hashes[role]["basename"]:
            fail(f"{input_where}/path", "basename differs from worker record identity")
        digest = require_hash(item.get("sha256"), f"{input_where}/sha256")
        if digest != record_hashes[role]["sha256"]:
            fail(f"{input_where}/sha256", "differs from worker before/after record hashes")
        sanitized[role] = {"basename": basename, "sha256": digest}

        if role in binary_for_audit:
            binary_role = binary_for_audit[role]
            audited = object_at(item.get("binary_audit"), f"{input_where}/binary_audit")
            if audited.get("name") != binaries[binary_role]["basename"]:
                fail(f"{input_where}/binary_audit/name", "differs from loaded binary basename")
            before = require_hash(
                audited.get("sha256_before"), f"{input_where}/binary_audit/sha256_before"
            )
            after = require_hash(
                audited.get("sha256_after"), f"{input_where}/binary_audit/sha256_after"
            )
            if before != after or after != binaries[binary_role]["sha256"]:
                fail(f"{input_where}/binary_audit", "audit and loaded binary SHA identities differ")

    packed_input = object_at(inputs.get("packed"), f"{where}/inputs/packed")
    packed_path = packed_input.get("path")
    if not isinstance(packed_path, str) or not packed_path:
        fail(f"{where}/inputs/packed/path", "missing packed artifact path")
    packed_basename = packed_path.replace("\\", "/").rsplit("/", 1)[-1]
    packed_sha = require_hash(packed_input.get("sha256"), f"{where}/inputs/packed/sha256")
    if packed_sha != packed_identity["file_sha256"]:
        fail(f"{where}/inputs/packed/sha256", "differs from worker packed before/after SHA")
    sanitized["packed"] = {"basename": packed_basename, "sha256": packed_sha}

    vllm_input = object_at(inputs.get("vllm_provenance"), f"{where}/inputs/vllm_provenance")
    vllm_path = vllm_input.get("path")
    if not isinstance(vllm_path, str) or not vllm_path:
        fail(f"{where}/inputs/vllm_provenance/path", "missing provenance path")
    vllm_basename = vllm_path.replace("\\", "/").rsplit("/", 1)[-1]
    vllm_sha = require_hash(vllm_input.get("sha256"), f"{where}/inputs/vllm_provenance/sha256")
    if vllm_basename != vllm_provenance["record_basename"] or vllm_sha != vllm_provenance["record_sha256"]:
        fail(f"{where}/inputs/vllm_provenance", "differs from worker vLLM provenance identity")
    sanitized["vllm_provenance"] = {"basename": vllm_basename, "sha256": vllm_sha}

    manifest_binaries = object_at(
        manifest.get("binary_hashes_before"), f"{where}/binary_hashes_before"
    )
    for role in ("w4", "bf16", "layout"):
        item = object_at(manifest_binaries.get(role), f"{where}/binary_hashes_before/{role}")
        if item.get("audit_name") != binaries[role]["basename"]:
            fail(f"{where}/binary_hashes_before/{role}/audit_name", "differs from worker binary basename")
        before = require_hash(
            item.get("sha256_before_run"), f"{where}/binary_hashes_before/{role}/sha256_before_run"
        )
        audited = require_hash(
            item.get("sha256_expected_from_audit"),
            f"{where}/binary_hashes_before/{role}/sha256_expected_from_audit",
        )
        if before != audited or before != binaries[role]["sha256"]:
            fail(f"{where}/binary_hashes_before/{role}", "manifest and worker binary SHAs differ")
    return sanitized


def aggregate_comparisons(case_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scopes = {
        "all_6": CASES,
        "small_4": SMALL_CASES,
        "large_2": LARGE_CASES,
    }
    result: dict[str, Any] = {}
    for section in ("full_api", "full_graph", "stage_api", "stage_graph"):
        references = FULL_REFERENCES if section.startswith("full_") else STAGE_REFERENCES
        section_result: dict[str, Any] = {}
        for reference in references:
            reference_result: dict[str, Any] = {}
            for scope_name, scope_cases in scopes.items():
                ratios: dict[str, float] = {}
                for case in scope_cases:
                    arms = case_results[case]["timings"][section]["arms"]
                    candidate_ms = arms["w4_layout"]["p50_ms"]
                    reference_ms = arms[reference]["p50_ms"]
                    ratios[case] = candidate_ms / reference_ms
                gm = geometric_mean(list(ratios.values()))
                reference_result[scope_name] = {
                    "count": len(scope_cases),
                    "ratio_definition": "w4_layout candidate p50 / reference p50; >1 is a regression",
                    "case_ratios": ratios,
                    "geometric_mean_ratio": gm,
                    "regressions": [
                        {"case": case, "candidate_over_reference": ratio}
                        for case, ratio in ratios.items()
                        if ratio > 1.0
                    ],
                }
            section_result[reference] = reference_result
        result[section] = section_result
    return result


def aggregate(input_dir_arg: str | Path) -> dict[str, Any]:
    input_dir = Path(input_dir_arg)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    try:
        input_dir = input_dir.resolve(strict=True)
        input_dir.relative_to(ROOT.resolve())
    except (OSError, ValueError) as exc:
        fail("--input-dir", f"must be an existing directory inside the repository ({exc})")
    if not input_dir.is_dir():
        fail("--input-dir", "must name a directory")
    manifest_path = input_dir / MANIFEST_NAME
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        fail(str(manifest_path), f"cannot read run manifest ({exc})")
    try:
        manifest = object_at(json.loads(manifest_bytes), MANIFEST_NAME)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(MANIFEST_NAME, f"invalid JSON ({exc})")
    if manifest.get("experiment") != EXPERIMENT:
        fail(f"{MANIFEST_NAME}/experiment", "does not match the M24 layout experiment")
    if manifest.get("status") not in ("passed", "complete", "completed"):
        fail(f"{MANIFEST_NAME}/status", "run manifest is not complete and passed")
    workers = manifest.get("workers")
    if not isinstance(workers, list) or len(workers) != len(CASES):
        fail(f"{MANIFEST_NAME}/workers", "expected exactly six listed workers")
    if "completed_count" in manifest:
        require_int(manifest["completed_count"], len(CASES), f"{MANIFEST_NAME}/completed_count")

    by_case: dict[str, dict[str, Any]] = {}
    for index, raw_worker in enumerate(workers):
        entry = object_at(raw_worker, f"{MANIFEST_NAME}/workers/{index}")
        case = entry.get("case")
        if case not in CASES:
            fail(f"{MANIFEST_NAME}/workers/{index}/case", "unknown case")
        if case in by_case:
            fail(f"{MANIFEST_NAME}/workers/{index}/case", f"duplicate {case}")
        if entry.get("result_status") != "passed":
            fail(f"{MANIFEST_NAME}/workers/{case}/result_status", "worker result did not pass")
        expected_sha = require_hash(
            entry.get("result_sha256"), f"{MANIFEST_NAME}/workers/{case}/result_sha256"
        )
        worker_path = resolve_manifest_output(entry.get("output"), input_dir, case)
        relative_worker_path = repo_relative(worker_path, f"{case}/worker path")
        record, actual_sha = read_worker(worker_path, case, expected_sha)
        by_case[case] = validate_worker(record, case, relative_worker_path, actual_sha)
    if set(by_case) != set(CASES):
        missing = sorted(set(CASES) - set(by_case))
        fail(f"{MANIFEST_NAME}/workers", f"missing fixed cases: {', '.join(missing)}")

    common_runtime = by_case[CASES[0]]["runtime"]
    common_sources = by_case[CASES[0]]["source_hashes"]
    common_records = by_case[CASES[0]]["record_hashes"]
    common_binaries = by_case[CASES[0]]["binaries"]
    common_packed = by_case[CASES[0]]["packed_identity"]
    common_head = by_case[CASES[0]]["head"]
    for case in CASES[1:]:
        checks = (
            ("runtime", common_runtime),
            ("source_hashes", common_sources),
            ("record_hashes", common_records),
            ("binaries", common_binaries),
            ("packed_identity", common_packed),
            ("head", common_head),
        )
        for field, expected in checks:
            if by_case[case][field] != expected:
                fail(f"{case}/{field}", "identity differs across the six-case run")

    manifest_inputs = validate_manifest_identities(
        manifest,
        common_sources,
        common_records,
        common_binaries,
        common_packed,
        by_case[CASES[0]]["vllm_provenance"],
    )
    manifest_sources = safe_hash_map(
        manifest.get("source_hashes_after"), f"{MANIFEST_NAME}/source_hashes_after"
    )

    manifest_relative = repo_relative(manifest_path, MANIFEST_NAME)
    case_results = [by_case[case] for case in CASES]
    keyed_results = {item["case"]: item for item in case_results}
    return {
        "schema": 1,
        "experiment": EXPERIMENT,
        "status": "passed_validation",
        "input_manifest": {
            "path": manifest_relative,
            "sha256": sha256_bytes(manifest_bytes),
        },
        "scope": {
            "fixed_cases": list(CASES),
            "small_cases": list(SMALL_CASES),
            "large_cases": list(LARGE_CASES),
            "timing_source": "raw worker rounds; p50, p95 and wall p50 recalculated and checked",
            "p95_aggregation": "per case only; no pooled or geometric-mean p95",
            "ratio_interpretation": "candidate/reference > 1 is slower; all regressions are retained",
            "stage_comparisons": "reported separately from full-forward comparisons; never subtracted",
            "holdout_scope": "M22 inputs reused for this layout mechanism comparison; not a new holdout",
            "repeatability_scope": "one six-case M24 run; not a multi-batch layout replication",
            "memory_scope": (
                "each prepared candidate and v1 W4 workspace owns one packed payload; all benchmark arms "
                "are resident together, so this run is not a single-path VRAM measurement"
            ),
            "repack_scope": "CPU-only repack duration listed per case and excluded from timed API/Graph arms",
            "quality_scope": (
                "quantized-layer output proxy is descriptive only; no full-model quality, production, "
                "or Marlin superiority claim"
            ),
        },
        "run_identity": {
            "head": common_head,
            "runtime": common_runtime,
            "vllm_provenance": by_case[CASES[0]]["vllm_provenance"],
            "binary_sha256": common_binaries,
            "source_sha256": common_sources,
            "manifest_source_sha256": manifest_sources,
            "audit_and_pack_records": common_records,
            "packed_identity": common_packed,
            "manifest_input_identities": manifest_inputs,
        },
        "cases": case_results,
        "repack_cpu_by_case": {
            case: by_case[case]["repack_cpu"] for case in CASES
        },
        "comparisons": aggregate_comparisons(keyed_results),
    }


def resolve_output_path(raw_path: str) -> tuple[Path, str]:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        fail("--output", f"must be inside the repository ({exc})")
    if not resolved.parent.is_dir():
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved, relative


def write_exclusive(path: Path, data: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        fail(str(path), "refusing to overwrite existing output")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        fail(str(path), f"failed writing output ({exc})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="directory containing run_manifest.json and its six worker outputs")
    parser.add_argument("--output", required=True, help="new repository-relative summary JSON path")
    args = parser.parse_args(argv)
    try:
        summary = aggregate(args.input_dir)
        output_path, output_relative = resolve_output_path(args.output)
        summary["output_path"] = output_relative
        encoded = (json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        write_exclusive(output_path, encoded)
    except SummaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"created {output_relative} bytes={len(encoded)} sha256={sha256_bytes(encoded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
