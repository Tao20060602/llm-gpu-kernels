#!/usr/bin/env python3
"""Validate and summarize the three fixed M23 W4 measurement batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADAPTER_SOURCE_PATH = "src/llm_gpu_kernels/moe_w4_repeat.py"
WORKER_SOURCE_PATH = "src/llm_gpu_kernels/moe_w4_benchmark.py"
NATIVE_LOAD_MODE = "verified_native_binary_no_jit"
NATIVE_LOAD_SCOPE = "startup native binary loading only; worker Triton preparation is outside this scope"
BATCHES = (1, 2, 3)
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
CASE_BUCKETS = {
    "archive-T17": "small",
    "archive-T193": "medium",
    "archive-T1024": "medium",
    "code-T17": "small",
    "code-T193": "medium",
    "code-T1024": "medium",
}
BUCKET_CONFIGS = {
    "small": {"block_m": 32, "block_n": 64, "block_k": 32, "warps": 4, "stages": 3},
    "medium": {"block_m": 32, "block_n": 256, "block_k": 32, "warps": 4, "stages": 3},
    "large": {"block_m": 64, "block_n": 128, "block_k": 32, "warps": 4, "stages": 3},
}
QUANT_ARMS = ("quant_vllm", "quant_frozen", "quant_cuda_sync", "w4_sync")
FULL_ARMS = (*QUANT_ARMS, "bf16_vllm", "bf16_frozen")
STAGE_ARMS = ("quant_frozen", "quant_cuda_sync", "w4_sync")
PROJECT_ARMS = frozenset(("quant_frozen", "quant_cuda_sync", "w4_sync", "bf16_frozen"))
FULL_SECTIONS = ("full_api", "full_graph")
STAGE_SECTIONS = ("stage_api", "stage_graph")
SECTIONS = {
    "full_api": (FULL_ARMS, 1, 12),
    "full_graph": (FULL_ARMS, 8, 12),
    "stage_api": (STAGE_ARMS, 1, 6),
    "stage_graph": (STAGE_ARMS, 8, 6),
}
SAME_QUANT_REFERENCES = ("quant_vllm", "quant_frozen", "quant_cuda_sync")
LOSSY_REFERENCES = ("bf16_vllm", "bf16_frozen")
STAGE_REFERENCES = ("quant_frozen", "quant_cuda_sync")
MEASURE_FIELDS = ("p50_ms", "p95_ms", "effective_tflops", "wall_p50_ms")
ERROR_FIELDS = ("relative_rms", "normalized_max", "max_abs")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
HEAD_RE = re.compile(r"^[0-9a-f]{40,64}$")
LIMIT = 0.008


class SummaryError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite_number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{where}: expected a number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        expected = "finite positive" if positive else "finite"
        raise SummaryError(f"{where}: expected a {expected} number")
    return number


def require_int(value: Any, expected: int, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise SummaryError(f"{where}: expected {expected}")


def safe_relative_key(key: Any, where: str) -> str:
    if not isinstance(key, str) or not key or "\\" in key:
        raise SummaryError(f"{where}: expected a portable relative path")
    posix = pathlib.PurePosixPath(key)
    windows = pathlib.PureWindowsPath(key)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in ("", ".", "..") for part in key.split("/"))
    ):
        raise SummaryError(f"{where}: expected a safe repository-relative path")
    return key


def safe_hash_map(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise SummaryError(f"{where}: expected a nonempty hash map")
    result: dict[str, str] = {}
    for raw_key, digest in value.items():
        key = safe_relative_key(raw_key, f"{where}/path")
        if not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
            raise SummaryError(f"{where}/{key}: invalid SHA-256")
        result[key] = digest
    return result


def logical_record_hashes(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) != 3:
        raise SummaryError(f"{where}: expected W4 audit, BF16 audit, and pack record hashes")
    result: dict[str, str] = {}
    for raw_path, digest in value.items():
        path = safe_relative_key(raw_path, f"{where}/path")
        if not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
            raise SummaryError(f"{where}/{path}: invalid SHA-256")
        normalized = path.lower()
        if "pack" in normalized or "quant-format" in normalized or "format-preparation" in normalized:
            role = "pack_record"
        elif "w4" in normalized:
            role = "w4_audit"
        elif "bf16" in normalized or "cuda-audit" in normalized or "cuda_audit" in normalized:
            role = "bf16_audit"
        else:
            raise SummaryError(f"{where}/{path}: cannot identify audit-record role")
        if role in result:
            raise SummaryError(f"{where}: duplicate {role} record")
        result[role] = digest
    if set(result) != {"w4_audit", "bf16_audit", "pack_record"}:
        raise SummaryError(f"{where}: incomplete logical record hashes")
    return result


def check_pass_flags(value: Any, where: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_where = f"{where}/{key}"
            if key in ("passed", "finite", "check_inputs_passed"):
                if child is not True:
                    raise SummaryError(f"{child_where}: expected true")
            check_pass_flags(child, child_where)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_pass_flags(child, f"{where}/{index}")


def compact_error(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SummaryError(f"{where}: missing numerical check")
    check_pass_flags(value, where)
    if value.get("finite") is not True or value.get("passed") is not True:
        raise SummaryError(f"{where}: numerical check did not pass")
    result: dict[str, Any] = {"finite": True, "passed": True}
    for field in ERROR_FIELDS:
        if field not in value:
            raise SummaryError(f"{where}/{field}: missing numerical metric")
        number = finite_number(value[field], f"{where}/{field}")
        if number < 0:
            raise SummaryError(f"{where}/{field}: expected a nonnegative error metric")
        result[field] = number
    if result["relative_rms"] > LIMIT:
        raise SummaryError(f"{where}/relative_rms: exceeds {LIMIT}")
    if result["normalized_max"] > LIMIT:
        raise SummaryError(f"{where}/normalized_max: exceeds {LIMIT}")
    return result


def compact_format_change(value: Any, where: str) -> dict[str, Any]:
    """Keep format-loss measurements without treating them as a quality gate."""
    if not isinstance(value, dict):
        raise SummaryError(f"{where}: missing format-forward comparison")
    if value.get("finite") is not True:
        raise SummaryError(f"{where}/finite: expected true")
    result: dict[str, Any] = {"finite": True}
    for field in ERROR_FIELDS:
        if field not in value:
            raise SummaryError(f"{where}/{field}: missing format-loss metric")
        number = finite_number(value[field], f"{where}/{field}")
        if number < 0:
            raise SummaryError(f"{where}/{field}: expected a nonnegative metric")
        result[field] = number
    if "passed" in value:
        if not isinstance(value["passed"], bool):
            raise SummaryError(f"{where}/passed: expected a boolean")
        result["passed"] = value["passed"]
    return result


def compact_dispatch(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SummaryError(f"{where}: missing ranked dispatch check")
    check_pass_flags(value, where)
    if value.get("passed") is not True or value.get("check_inputs_passed") is not True:
        raise SummaryError(f"{where}: dispatch check did not pass")
    status = value.get("status")
    if isinstance(status, bool) or status != 0:
        raise SummaryError(f"{where}/status: expected zero")
    ranked = value.get("ranked_dispatch")
    if not isinstance(ranked, dict) or ranked.get("passed") is not True:
        raise SummaryError(f"{where}/ranked_dispatch: ranked dispatch did not pass")
    flags = {key: child for key, child in ranked.items() if isinstance(child, bool)}
    return {
        "status": 0,
        "check_inputs_passed": True,
        "passed": True,
        "ranked_dispatch_flags": flags,
    }


def compact_audits(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(FULL_ARMS):
        raise SummaryError(f"{where}: expected exactly the six full-forward arms")
    result: dict[str, Any] = {}
    for arm in FULL_ARMS:
        item = value[arm]
        arm_where = f"{where}/{arm}"
        if not isinstance(item, dict):
            raise SummaryError(f"{arm_where}: invalid arm audit")
        numerical = compact_error(item.get("numerical"), f"{arm_where}/numerical")
        if arm in PROJECT_ARMS:
            dispatch = compact_dispatch(item.get("dispatch"), f"{arm_where}/dispatch")
        else:
            if "dispatch" in item:
                raise SummaryError(f"{arm_where}/dispatch: unexpected vLLM dispatch record")
            dispatch = None
        result[arm] = {"numerical": numerical, "dispatch": dispatch}
    return result


def expected_orders(arms: tuple[str, ...]) -> list[list[str]]:
    orders: list[list[str]] = []
    for base in (list(arms), list(reversed(arms))):
        orders.extend(base[index:] + base[:index] for index in range(len(base)))
    return orders


def percentile_linear(values: list[float], percentile: float) -> float:
    """Match numpy.percentile's default method="linear" used by the worker."""
    ordered = sorted(values)
    if not ordered:
        raise SummaryError("samples: cannot calculate a percentile from no values")
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def require_stat_match(actual: float, expected: float, where: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise SummaryError(f"{where}: stored summary differs from raw samples (stored={actual}, calculated={expected})")


def validate_measurement(
    value: Any,
    where: str,
    arms: tuple[str, ...],
    divisor: int,
    round_count: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SummaryError(f"{where}: missing measurement")
    summary = value.get("summary")
    if not isinstance(summary, dict) or set(summary) != set(arms):
        raise SummaryError(f"{where}/summary: arm set differs")
    rounds = value.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != round_count:
        raise SummaryError(f"{where}/rounds: expected {round_count} timing rounds")
    require_int(value.get("samples_per_round"), 6, f"{where}/samples_per_round")
    require_int(value.get("repeats"), 2, f"{where}/repeats")
    require_int(value.get("divisor"), divisor, f"{where}/divisor")

    reported_summary: dict[str, dict[str, float]] = {}
    for arm in arms:
        stats = summary.get(arm)
        if not isinstance(stats, dict):
            raise SummaryError(f"{where}/summary/{arm}: invalid arm summary")
        reported_summary[arm] = {
            field: finite_number(stats.get(field), f"{where}/summary/{arm}/{field}", positive=True)
            for field in MEASURE_FIELDS
        }

    wanted_orders = expected_orders(arms)
    clean_rounds: list[dict[str, Any]] = []
    for index, round_record in enumerate(rounds):
        round_where = f"{where}/rounds/{index}"
        if not isinstance(round_record, dict) or round_record.get("order") != wanted_orders[index]:
            raise SummaryError(f"{round_where}/order: arm order differs")
        cleaned: dict[str, Any] = {"order": wanted_orders[index]}
        for dimension in ("samples_ms_per_gemm", "wall_ms_per_gemm"):
            values_by_arm = round_record.get(dimension)
            if not isinstance(values_by_arm, dict) or set(values_by_arm) != set(arms):
                raise SummaryError(f"{round_where}/{dimension}: arm set differs")
            clean_values: dict[str, list[float]] = {}
            for arm in arms:
                samples = values_by_arm[arm]
                sample_where = f"{round_where}/{dimension}/{arm}"
                if not isinstance(samples, list) or len(samples) != 6:
                    raise SummaryError(f"{sample_where}: expected all 6 samples")
                clean_values[arm] = [
                    finite_number(sample, f"{sample_where}/{sample_index}", positive=True)
                    for sample_index, sample in enumerate(samples)
                ]
            cleaned[dimension] = clean_values
        clean_rounds.append(cleaned)

    calculated_summary: dict[str, dict[str, float]] = {}
    for arm in arms:
        event_samples = [sample for row in clean_rounds for sample in row["samples_ms_per_gemm"][arm]]
        wall_samples = [sample for row in clean_rounds for sample in row["wall_ms_per_gemm"][arm]]
        p50 = statistics.median(event_samples)
        p95 = percentile_linear(event_samples, 95)
        wall_p50 = statistics.median(wall_samples)
        for field, calculated in (("p50_ms", p50), ("p95_ms", p95), ("wall_p50_ms", wall_p50)):
            require_stat_match(
                reported_summary[arm][field], calculated, f"{where}/summary/{arm}/{field}"
            )
        calculated_summary[arm] = {
            "p50_ms": p50,
            "p95_ms": p95,
            # Keep the reported value; it is not recomputed or used for ratios.
            "effective_tflops": reported_summary[arm]["effective_tflops"],
            "wall_p50_ms": wall_p50,
        }

    result: dict[str, Any] = {
        "summary": calculated_summary,
        "rounds": clean_rounds,
        "samples_per_round": 6,
        "repeats": 2,
        "divisor": divisor,
    }
    if isinstance(value.get("wall_scope"), str):
        result["wall_scope"] = value["wall_scope"]
    return result


def validate_shape(item: dict[str, Any], case: str, where: str) -> dict[str, Any]:
    bucket = item.get("selected_bucket")
    expected_bucket = CASE_BUCKETS[case]
    if bucket != expected_bucket:
        raise SummaryError(f"{where}/selected_bucket: expected {expected_bucket} for {case}")
    frozen = item.get("frozen_config")
    expected_frozen = BUCKET_CONFIGS[expected_bucket]
    if frozen != expected_frozen:
        raise SummaryError(f"{where}/frozen_config: differs from the pinned {expected_bucket} M7 configuration")
    w4 = item.get("w4_config")
    expected_w4 = {"BM": 32, "BN": 64, "BK": 32, "warps": 4, "group_size": 128}
    if w4 != expected_w4:
        raise SummaryError(f"{where}/w4_config: differs from the pinned configuration")
    limits = item.get("limits")
    if not isinstance(limits, dict):
        raise SummaryError(f"{where}/limits: missing numerical limits")
    if finite_number(limits.get("relative_rms"), f"{where}/limits/relative_rms", positive=True) != LIMIT:
        raise SummaryError(f"{where}/limits/relative_rms: differs from {LIMIT}")
    if finite_number(limits.get("normalized_max"), f"{where}/limits/normalized_max", positive=True) != LIMIT:
        raise SummaryError(f"{where}/limits/normalized_max: differs from {LIMIT}")
    require_int(item.get("graph_repeats"), 8, f"{where}/graph_repeats")

    metadata = item.get("format_metadata")
    if not isinstance(metadata, dict):
        raise SummaryError(f"{where}/format_metadata: missing packed format metadata")
    require_int(metadata.get("format_version"), 1, f"{where}/format_metadata/format_version")
    if metadata.get("logical_shape") != [128, 1536, 2048]:
        raise SummaryError(f"{where}/format_metadata/logical_shape: differs from the pinned shape")
    require_int(metadata.get("group_size"), 128, f"{where}/format_metadata/group_size")
    pinned_format_fields = {
        "encoding": "q+8, q in [-7,7], low nibble is earliest K",
        "word_storage": "int32 storage of uint32 bit pattern",
        "scale_dtype": "bfloat16",
        "activation_dtype": "bfloat16",
        "accumulator_dtype": "float32",
    }
    for field, expected in pinned_format_fields.items():
        if metadata.get(field) != expected:
            raise SummaryError(f"{where}/format_metadata/{field}: differs from the pinned format")
    for field in ("source_sha256", "words_sha256", "scales_sha256"):
        digest = metadata.get(field)
        if not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
            raise SummaryError(f"{where}/format_metadata/{field}: invalid SHA-256")
    for field in ("source_weight_bytes", "packed_weight_bytes", "scale_bytes"):
        value = metadata.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SummaryError(f"{where}/format_metadata/{field}: expected a positive integer")

    if item.get("quantized_arms") != list(QUANT_ARMS):
        raise SummaryError(f"{where}/quantized_arms: differs from the pinned arm list")
    if item.get("original_bf16_arms") != ["bf16_vllm", "bf16_frozen"]:
        raise SummaryError(f"{where}/original_bf16_arms: differs from the pinned arm list")
    if not isinstance(item.get("stage_scope"), str):
        raise SummaryError(f"{where}/stage_scope: missing scope description")

    return {
        "selected_bucket": item["selected_bucket"],
        "frozen_config": dict(frozen),
        "w4_config": dict(w4),
        "graph_repeats": 8,
        "limits": {"relative_rms": LIMIT, "normalized_max": LIMIT},
        "format": {
            key: metadata[key]
            for key in (
                "format_version", "logical_shape", "group_size", "encoding", "word_storage",
                "scale_dtype", "activation_dtype", "accumulator_dtype", "source_sha256",
                "words_sha256", "scales_sha256", "source_weight_bytes", "packed_weight_bytes", "scale_bytes",
            )
            if key in metadata
        },
        "quantized_arms": list(QUANT_ARMS),
        "original_bf16_arms": ["bf16_vllm", "bf16_frozen"],
        "stage_scope": item["stage_scope"],
    }


def validate_binaries(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"w4", "bf16"}:
        raise SummaryError(f"{where}: expected W4 and BF16 binary provenance")
    result: dict[str, str] = {}
    for role in ("w4", "bf16"):
        entry = value[role]
        if not isinstance(entry, dict):
            raise SummaryError(f"{where}/{role}: invalid binary provenance")
        before = entry.get("sha256_before")
        after = entry.get("sha256_after")
        if not isinstance(before, str) or not HASH_RE.fullmatch(before) or before != after:
            raise SummaryError(f"{where}/{role}: binary SHA missing or changed within worker")
        if not isinstance(entry.get("name"), str) or not entry["name"]:
            raise SummaryError(f"{where}/{role}/name: missing binary name")
        result[role] = before
    return result


def read_worker(path: pathlib.Path, batch: int, case: str) -> dict[str, Any]:
    where = f"batch{batch}/{case}"
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(ROOT).as_posix()
        safe_relative_key(relative, f"{where}/worker_path")
        raw = resolved.read_bytes()
    except (OSError, ValueError) as exc:
        raise SummaryError(f"{where}/worker_path: cannot read a repo-local worker JSON at {path}") from exc
    try:
        item = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"{where}/{relative}: invalid JSON") from exc
    if not isinstance(item, dict):
        raise SummaryError(f"{where}/{relative}: top level must be an object")
    return {"path": relative, "sha256": sha256(raw), "bytes": len(raw), "item": item}


def read_receipt(path: pathlib.Path, batch: int, case: str) -> dict[str, Any]:
    where = f"batch{batch}/{case}/startup_receipt"
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(ROOT).as_posix()
        safe_relative_key(relative, f"{where}/path")
        raw = resolved.read_bytes()
    except (OSError, ValueError) as exc:
        raise SummaryError(f"{where}/path: cannot read the required repo-local receipt at {path}") from exc
    try:
        item = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"{where}/{relative}: invalid JSON") from exc
    if not isinstance(item, dict):
        raise SummaryError(f"{where}/{relative}: top level must be an object")
    return {"path": relative, "sha256": sha256(raw), "item": item}


def validate_startup_receipt(
    worker: dict[str, Any],
    receipt: dict[str, Any],
    batch: int,
    case: str,
    current_adapter_sha256: str,
) -> dict[str, Any]:
    where = f"batch{batch}/{case}/startup_receipt"
    item = receipt["item"]
    require_int(item.get("schema"), 1, f"{where}/schema")
    adapter = item.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("name") != "moe_w4_repeat" or adapter.get("version") != "1":
        raise SummaryError(f"{where}/adapter: expected moe_w4_repeat version 1")

    adapter_sha = item.get("adapter_source_sha256")
    if not isinstance(adapter_sha, str) or not HASH_RE.fullmatch(adapter_sha):
        raise SummaryError(f"{where}/adapter_source_sha256: invalid SHA-256")
    if adapter_sha != current_adapter_sha256:
        raise SummaryError(f"{where}/adapter_source_sha256: differs from current adapter source")

    source_hashes = worker["item"].get("source_hashes_before")
    worker_source_sha = item.get("worker_source_sha256")
    if not isinstance(source_hashes, dict) or WORKER_SOURCE_PATH not in source_hashes:
        raise SummaryError(f"batch{batch}/{case}/source_hashes_before/{WORKER_SOURCE_PATH}: missing worker source hash")
    if worker_source_sha != source_hashes[WORKER_SOURCE_PATH]:
        raise SummaryError(f"{where}/worker_source_sha256: differs from worker.source_hashes_before")

    receipt_audits = item.get("audit_record_sha256")
    if not isinstance(receipt_audits, dict) or set(receipt_audits) != {"w4", "bf16"}:
        raise SummaryError(f"{where}/audit_record_sha256: expected W4 and BF16 audit hashes")
    for role, digest in receipt_audits.items():
        if not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
            raise SummaryError(f"{where}/audit_record_sha256/{role}: invalid SHA-256")
    worker_audits = logical_record_hashes(
        worker["item"].get("record_hashes_before"), f"batch{batch}/{case}/record_hashes_before"
    )
    expected_audits = {"w4": worker_audits["w4_audit"], "bf16": worker_audits["bf16_audit"]}
    if receipt_audits != expected_audits:
        raise SummaryError(f"{where}/audit_record_sha256: differs from worker audit record hashes")

    receipt_binaries = item.get("binaries")
    worker_binaries = worker["item"].get("binaries")
    if not isinstance(receipt_binaries, dict) or set(receipt_binaries) != {"w4", "bf16"}:
        raise SummaryError(f"{where}/binaries: expected W4 and BF16 native binaries")
    if not isinstance(worker_binaries, dict) or set(worker_binaries) != {"w4", "bf16"}:
        raise SummaryError(f"batch{batch}/{case}/binaries: missing worker binary records")
    binary_summary: dict[str, dict[str, str]] = {}
    for role in ("w4", "bf16"):
        entry = receipt_binaries[role]
        worker_entry = worker_binaries[role]
        binary_where = f"{where}/binaries/{role}"
        if not isinstance(entry, dict) or not isinstance(worker_entry, dict):
            raise SummaryError(f"{binary_where}: invalid native binary record")
        filename = entry.get("filename")
        worker_filename = worker_entry.get("name")
        if not isinstance(filename, str) or pathlib.PurePath(filename).name != filename or filename != worker_filename:
            raise SummaryError(f"{binary_where}/filename: differs from worker binary basename")
        module_name = entry.get("module_name")
        if not isinstance(module_name, str) or module_name != pathlib.Path(filename).stem:
            raise SummaryError(f"{binary_where}/module_name: differs from native module basename")
        resolved_path = entry.get("resolved_path")
        if not isinstance(resolved_path, str) or not pathlib.Path(resolved_path).is_absolute():
            raise SummaryError(f"{binary_where}/resolved_path: expected a local absolute path")
        if pathlib.Path(resolved_path).name != filename:
            raise SummaryError(f"{binary_where}/resolved_path: basename differs from filename")
        before = entry.get("sha256_before_load")
        after = entry.get("sha256_after_load")
        if not isinstance(before, str) or not HASH_RE.fullmatch(before) or before != after:
            raise SummaryError(f"{binary_where}: native binary SHA changed during load")
        if before != worker_entry.get("sha256_before") or after != worker_entry.get("sha256_after"):
            raise SummaryError(f"{binary_where}: startup binary SHA differs from worker binary hashes")
        binary_summary[role] = {"filename": filename, "sha256": before}

    runtime = worker["item"].get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("torch"), str):
        raise SummaryError(f"batch{batch}/{case}/runtime/torch: missing worker runtime version")
    if item.get("torch_version") != runtime["torch"]:
        raise SummaryError(f"{where}/torch_version: differs from worker.runtime.torch")

    if item.get("load_mode") != NATIVE_LOAD_MODE:
        raise SummaryError(f"{where}/load_mode: expected {NATIVE_LOAD_MODE}")
    if item.get("native_cpp_extension_jit_invoked") is not False:
        raise SummaryError(f"{where}/native_cpp_extension_jit_invoked: expected false")
    if item.get("load_scope") != NATIVE_LOAD_SCOPE:
        raise SummaryError(f"{where}/load_scope: differs from the reviewed startup-only scope")

    expected_output_path = item.get("expected_output_path")
    if not isinstance(expected_output_path, str) or not pathlib.Path(expected_output_path).is_absolute():
        raise SummaryError(f"{where}/expected_output_path: expected an absolute worker output path")
    actual_worker_path = (ROOT / worker["path"]).resolve()
    if pathlib.Path(expected_output_path).resolve() != actual_worker_path:
        raise SummaryError(f"{where}/expected_output_path: differs from the actual worker JSON path")

    return {
        "startup_verified": True,
        "receipt_path": receipt["path"],
        "receipt_sha256": receipt["sha256"],
        "adapter_source_sha256": adapter_sha,
        "binaries": binary_summary,
        "load_mode": NATIVE_LOAD_MODE,
        "load_scope": NATIVE_LOAD_SCOPE,
    }


def validate(
    worker: dict[str, Any],
    batch: int,
    case: str,
    receipt: dict[str, Any] | None = None,
    current_adapter_sha256: str | None = None,
) -> dict[str, Any]:
    where = f"batch{batch}/{case}"
    item = worker["item"]
    require_int(item.get("schema"), 1, f"{where}/schema")
    if item.get("status") != "passed":
        raise SummaryError(f"{where}/status: worker did not pass")
    require_int(item.get("batch"), batch, f"{where}/batch")
    if item.get("case") != case:
        raise SummaryError(f"{where}/case: worker identity differs")

    source_before = safe_hash_map(item.get("source_hashes_before"), f"{where}/source_hashes_before")
    source_after = safe_hash_map(item.get("source_hashes_after"), f"{where}/source_hashes_after")
    if source_before != source_after:
        raise SummaryError(f"{where}/source_hashes: changed during worker")
    artifact_before = safe_hash_map(item.get("artifact_hashes_before"), f"{where}/artifact_hashes_before")
    artifact_after = safe_hash_map(item.get("artifact_hashes_after"), f"{where}/artifact_hashes_after")
    if artifact_before != artifact_after:
        raise SummaryError(f"{where}/artifact_hashes: changed during worker")
    for trace_path in (f"traces/{case}.json", f"traces/{case}.pt"):
        if trace_path not in artifact_before:
            raise SummaryError(f"{where}/artifact_hashes_before/{trace_path}: missing trace identity")

    input_before = item.get("input_sha256")
    input_after = item.get("input_sha256_after")
    if not isinstance(input_before, str) or not HASH_RE.fullmatch(input_before) or input_before != input_after:
        raise SummaryError(f"{where}/input_sha256: missing or changed within worker")
    packed_before = item.get("packed_sha256_before")
    packed_after = item.get("packed_sha256_after")
    if not isinstance(packed_before, str) or not HASH_RE.fullmatch(packed_before) or packed_before != packed_after:
        raise SummaryError(f"{where}/packed_sha256: missing or changed within worker")

    packed_gpu = item.get("packed_gpu_digest_after")
    if not isinstance(packed_gpu, dict) or set(packed_gpu) != {"words", "scales"}:
        raise SummaryError(f"{where}/packed_gpu_digest_after: expected words and scales hashes")
    if any(not isinstance(packed_gpu[key], str) or not HASH_RE.fullmatch(packed_gpu[key]) for key in packed_gpu):
        raise SummaryError(f"{where}/packed_gpu_digest_after: invalid SHA-256")
    record_before = logical_record_hashes(item.get("record_hashes_before"), f"{where}/record_hashes_before")
    record_after = logical_record_hashes(item.get("record_hashes_after"), f"{where}/record_hashes_after")
    if record_before != record_after:
        raise SummaryError(f"{where}/record_hashes: changed during worker")
    binary_hashes = validate_binaries(item.get("binaries"), f"{where}/binaries")

    metadata = item.get("format_metadata")
    if isinstance(metadata, dict) and (
        metadata.get("words_sha256") != packed_gpu["words"]
        or metadata.get("scales_sha256") != packed_gpu["scales"]
    ):
        raise SummaryError(f"{where}/format_metadata: packed hashes differ from GPU digest")
    head = item.get("head")
    if not isinstance(head, str) or not HEAD_RE.fullmatch(head):
        raise SummaryError(f"{where}/head: invalid commit hash")

    shape = validate_shape(item, case, where)
    measurements = {
        section: validate_measurement(item.get(section), f"{where}/{section}", arms, divisor, rounds)
        for section, (arms, divisor, rounds) in SECTIONS.items()
    }
    audits_before = compact_audits(item.get("audit_before"), f"{where}/audit_before")
    audits_after = compact_audits(item.get("audit_after"), f"{where}/audit_after")
    graph_values = item.get("graph_numerics")
    if not isinstance(graph_values, dict) or set(graph_values) != set(FULL_ARMS):
        raise SummaryError(f"{where}/graph_numerics: expected exactly the six full-forward arms")
    graph_numerics = {
        arm: compact_error(graph_values[arm], f"{where}/graph_numerics/{arm}")
        for arm in FULL_ARMS
    }
    format_change = compact_format_change(item.get("format_forward_change"), f"{where}/format_forward_change")

    if batch == 1:
        if receipt is not None:
            raise SummaryError(f"{where}/startup_receipt: first batch must use the legacy load path")
        startup = {
            "startup_verified": False,
            "load_mode": "legacy_jit_preparation",
            "load_scope": "legacy cpp_extension.load preparation; preparation is excluded from timing",
        }
    else:
        if receipt is None:
            raise SummaryError(f"{where}/startup_receipt: required for repeated batches")
        if current_adapter_sha256 is None:
            raise SummaryError(f"{where}/startup_receipt: current adapter source SHA is unavailable")
        startup = validate_startup_receipt(worker, receipt, batch, case, current_adapter_sha256)

    trace_hashes = {key: digest for key, digest in artifact_before.items() if key.startswith("traces/")}
    identity = {
        "input_sha256": input_before,
        "trace_hashes": trace_hashes,
        "artifact_hashes": artifact_before,
        "packed_sha256": packed_before,
        "packed_gpu_digest": dict(packed_gpu),
        "source_hashes": source_before,
        "binary_sha256": binary_hashes,
        "audit_record_sha256": record_before,
        "shape": shape,
    }
    return {
        "batch": batch,
        "case": case,
        "status": "passed",
        "shape": shape,
        "format_forward_change_vs_original_bf16": format_change,
        "audit_before": audits_before,
        "audit_after": audits_after,
        "graph_numerics": graph_numerics,
        "measurements": measurements,
        "startup": startup,
        "identity": identity,
        "provenance": {
            "worker_json_path": worker["path"],
            "worker_json_sha256": worker["sha256"],
            "worker_json_bytes": worker["bytes"],
            "head": head,
            "source_hashes": source_before,
            "artifact_hashes": artifact_before,
            "trace_hashes": trace_hashes,
            "input_sha256_before": input_before,
            "input_sha256_after": input_after,
            "packed_sha256_before": packed_before,
            "packed_sha256_after": packed_after,
            "packed_gpu_digest_after": dict(packed_gpu),
            "binary_sha256": binary_hashes,
            "audit_record_sha256": record_before,
        },
    }


def ensure_consistent(records: dict[tuple[int, str], dict[str, Any]]) -> None:
    first = records[(1, CASES[0])]
    global_fields = ("source_hashes", "binary_sha256", "packed_sha256", "packed_gpu_digest", "audit_record_sha256")
    for (batch, case), record in records.items():
        for field in global_fields:
            if record["identity"][field] != first["identity"][field]:
                raise SummaryError(f"batch{batch}/{case}/identity/{field}: differs from the frozen batch identity")
    for case in CASES:
        reference = records[(1, case)]["identity"]
        for batch in (2, 3):
            current = records[(batch, case)]["identity"]
            for field in ("input_sha256", "trace_hashes", "artifact_hashes", "packed_sha256", "packed_gpu_digest", "source_hashes", "binary_sha256", "audit_record_sha256", "shape"):
                if current[field] != reference[field]:
                    raise SummaryError(f"batch{batch}/{case}/identity/{field}: differs from batch1/{case}")


def geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise SummaryError("comparison: geometric mean requires finite positive ratios")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def compare_group(
    records: dict[tuple[int, str], dict[str, Any]],
    keys: list[tuple[int, str]],
    section: str,
    candidate: str,
    reference: str,
) -> dict[str, Any]:
    ratios: dict[str, float] = {}
    for batch, case in keys:
        entry = records[(batch, case)]
        candidate_time = entry["measurements"][section]["summary"][candidate]["p50_ms"]
        reference_time = entry["measurements"][section]["summary"][reference]["p50_ms"]
        key = f"batch{batch}/{case}" if len({batch for batch, _ in keys}) > 1 else case
        ratios[key] = candidate_time / reference_time
    ratio_values = list(ratios.values())
    ratio_geomean = geometric_mean(ratio_values)
    worst_case = max(ratios, key=ratios.__getitem__)
    regressions = [
        {"case": case, "time_ratio": ratio, "regression_fraction": ratio - 1.0}
        for case, ratio in ratios.items()
        if ratio > 1.0
    ]
    return {
        "candidate_arm": candidate,
        "reference_arm": reference,
        "equal_weight_case_count": len(ratios),
        "case_time_ratios": ratios,
        "case_speedups": {case: 1.0 / ratio for case, ratio in ratios.items()},
        "time_ratio_geomean": ratio_geomean,
        "speedup_geomean": geometric_mean([1.0 / ratio for ratio in ratio_values]),
        "time_reduction_fraction_from_geomean_ratio": 1.0 - ratio_geomean,
        "worst_case": worst_case,
        "worst_case_time_ratio": ratios[worst_case],
        "worst_case_regression_fraction": ratios[worst_case] - 1.0,
        "regression_cases": regressions,
    }


def comparison_set(
    records: dict[tuple[int, str], dict[str, Any]], keys: list[tuple[int, str]]
) -> dict[str, Any]:
    full: dict[str, Any] = {}
    for section in FULL_SECTIONS:
        full[section] = {
            "same_quantization": {
                reference: {
                    "arms": {
                        candidate: compare_group(records, keys, section, candidate, reference)
                        for candidate in QUANT_ARMS
                    }
                }
                for reference in SAME_QUANT_REFERENCES
            },
            "separate_original_bf16_lossy_tradeoff": {
                reference: {
                    "arms": {
                        candidate: compare_group(records, keys, section, candidate, reference)
                        for candidate in QUANT_ARMS
                    }
                }
                for reference in LOSSY_REFERENCES
            },
        }
    stage = {
        section: {
            reference: {
                "arms": {
                    candidate: compare_group(records, keys, section, candidate, reference)
                    for candidate in STAGE_ARMS
                }
            }
            for reference in STAGE_REFERENCES
        }
        for section in STAGE_SECTIONS
    }
    return {"full_forward_comparisons": full, "stage_comparisons_reported_separately": stage}


def aggregate(records: dict[tuple[int, str], dict[str, Any]]) -> dict[str, Any]:
    ensure_consistent(records)
    repeated_startups = [records[(batch, case)]["startup"] for batch in (2, 3) for case in CASES]
    if len(repeated_startups) != 12 or any(not entry["startup_verified"] for entry in repeated_startups):
        raise SummaryError("startup_receipts: all 12 repeat receipts must validate before startup can be marked verified")
    adapter_hashes = {entry["adapter_source_sha256"] for entry in repeated_startups}
    if len(adapter_hashes) != 1:
        raise SummaryError("startup_receipts/adapter_source_sha256: differs across the 12 repeat receipts")
    adapter_sha256 = next(iter(adapter_hashes))
    worker_results: list[dict[str, Any]] = []
    for batch in BATCHES:
        for case in CASES:
            entry = records[(batch, case)]
            case_keys = [(batch, case)]
            worker_results.append({
                **entry,
                "case_comparisons": comparison_set(records, case_keys),
            })

    per_batch: dict[str, Any] = {}
    for batch in BATCHES:
        for label, cases in (("all_six", CASES), ("small_four", SMALL_CASES), ("large_two", LARGE_CASES)):
            keys = [(batch, case) for case in cases]
            per_batch.setdefault(f"batch{batch}", {})[label] = {
                "cases": list(cases),
                "case_count": len(cases),
                **comparison_set(records, keys),
            }

    all_batches: dict[str, Any] = {}
    for label, cases in (("all_18", CASES), ("small_12", SMALL_CASES), ("large_6", LARGE_CASES)):
        keys = [(batch, case) for batch in BATCHES for case in cases]
        all_batches[label] = {
            "case_count": len(keys),
            "case_weights": "equal weight per case across the three batches",
            **comparison_set(records, keys),
        }

    reproduction: dict[str, Any] = {
        "strict_rule": "Each batch's small-four full_graph time-ratio geometric mean must be strictly below 1.0.",
        "same_quantization": {},
        "separate_original_bf16_lossy_tradeoff": {},
    }
    for report_key, refs_key, references in (
        ("same_quantization", "same_quantization", SAME_QUANT_REFERENCES),
        ("separate_original_bf16_lossy_tradeoff", "separate_original_bf16_lossy_tradeoff", LOSSY_REFERENCES),
    ):
        for reference in references:
            ratios = {
                f"batch{batch}": per_batch[f"batch{batch}"]["small_four"]["full_forward_comparisons"]["full_graph"][refs_key][reference]["arms"]["w4_sync"]["time_ratio_geomean"]
                for batch in BATCHES
            }
            reproduced = all(ratio < 1.0 for ratio in ratios.values())
            reproduction[report_key][reference] = {
                "status": "reproduced_all_batches" if reproduced else "not_reproduced_all_batches",
                "all_batches_strictly_below_one": reproduced,
                "per_batch_small_graph_time_ratio_geomean": ratios,
            }

    return {
        "schema": 1,
        "status": "complete",
        "experiment": "M23 W4A16 three-batch cross-process repeat",
        "sampling_scope": {
            "batch_count": 3,
            "batches": list(BATCHES),
            "cases_per_batch": list(CASES),
            "worker_count": 18,
            "one_process_per_case": True,
            "full_api_rounds": 12,
            "full_graph_rounds": 12,
            "stage_api_rounds": 6,
            "stage_graph_rounds": 6,
            "samples_per_round": 6,
            "repeats_per_sample": 2,
            "graph_divisor": 8,
        },
        "arms": {
            "full_api_and_graph": list(FULL_ARMS),
            "same_quantization_references": list(SAME_QUANT_REFERENCES),
            "separate_original_bf16_lossy_references": list(LOSSY_REFERENCES),
            "stage_api_and_graph": list(STAGE_ARMS),
        },
        "validation": {
            "all_18_workers_passed": True,
            "all_timing_rounds_and_samples_complete": True,
            "reported_p50_p95_and_wall_p50_recalculated_from_samples": True,
            "effective_tflops_not_recomputed_or_used_for_ratios": True,
            "per_case_m7_bucket_and_frozen_config_verified": True,
            "all_audit_and_graph_numeric_checks_passed_with_0_008_limits": True,
            "ranked_project_dispatch_checks_passed": True,
            "within_worker_input_packed_binary_and_provenance_hashes_stable": True,
            "cross_batch_per_case_input_trace_packed_source_binary_audit_and_shape_identity_stable": True,
            "source_hashes_compared_directly": True,
            "all_12_repeat_startup_receipts_verified": True,
            "adapter_source_sha256_matches_current_file_and_all_receipts": True,
        },
        "startup_verification": {
            "repeat_receipt_count": 12,
            "status": "verified_all_repeat_receipts",
            "adapter_source_sha256": adapter_sha256,
            "batch1_load_mode": "legacy_jit_preparation",
            "batches2_3_load_mode": NATIVE_LOAD_MODE,
            "scope": NATIVE_LOAD_SCOPE,
        },
        "load_prepare_difference": {
            "batch1": "The original worker used legacy cpp_extension.load preparation.",
            "batches2_3": "The adapter directly loaded the audited native shared libraries before starting the frozen worker.",
            "timing": "Both load/preparation phases occur before the measured rounds and are excluded from the timing samples.",
            "no_jit_scope": "verified_native_binary_no_jit describes only startup native C++ extension loading; it does not claim that Triton preparation in the worker compiles nothing.",
        },
        "aggregation_policy": {
            "full_comparison_candidate_arms": list(QUANT_ARMS),
            "stage_comparison_candidate_arms": list(STAGE_ARMS),
            "ratio": "candidate p50_ms divided by reference p50_ms",
            "case_weights": "equal weight; three-batch aggregate has 18/12/6 cases for all/small/large scopes",
            "geometric_mean": "geometric mean of per-case time ratios",
            "worst_case": "largest casewise time ratio; every case ratio and every regression is retained",
            "p95": "retained for every arm and case; not pooled or aggregated",
            "stage": "reported independently; never subtracted from full-forward timing",
            "winner_selected": False,
        },
        "worker_results": worker_results,
        "per_batch_scopes": per_batch,
        "three_batch_equal_weight_scopes": all_batches,
        "small_graph_reproduction": reproduction,
        "interpretation_limits": [
            "No statistical confidence intervals are provided.",
            "This repeat is not a holdout claim.",
            "No production deployment, full-model quality, or Marlin superiority claim is made.",
            "format_forward_change_vs_original_bf16 reports format loss and is not judged against the 0.008 kernel error limits.",
            "The no_jit label applies only to startup native C++ extension loading; Triton preparation may still compile kernels outside this scope.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-batch-dir", required=True, type=pathlib.Path)
    parser.add_argument("--repeat-dir", required=True, type=pathlib.Path,
                        help="directory containing the fixed batch2-*.json and batch3-*.json workers")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output if args.output.is_absolute() else pathlib.Path.cwd() / args.output
    output_path = output_path.resolve()
    if output_path.exists():
        raise SummaryError(f"output {output_path}: refusing to overwrite existing file")

    adapter_source = ROOT / ADAPTER_SOURCE_PATH
    try:
        current_adapter_sha256 = sha256(adapter_source.read_bytes())
    except OSError as exc:
        raise SummaryError(f"adapter source {ADAPTER_SOURCE_PATH}: cannot read current source") from exc

    first_dir = args.first_batch_dir.resolve()
    repeat_dir = args.repeat_dir.resolve()
    expected: list[tuple[pathlib.Path, int, str]] = []
    for case in CASES:
        expected.append((first_dir / f"batch1-{case}.json", 1, case))
    for batch in (2, 3):
        for case in CASES:
            expected.append((repeat_dir / f"batch{batch}-{case}.json", batch, case))

    records: dict[tuple[int, str], dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for path, batch, case in expected:
        worker = read_worker(path, batch, case)
        canonical_path = str(path.resolve())
        if canonical_path in seen_paths:
            raise SummaryError(f"batch{batch}/{case}/worker_path: duplicate worker record")
        seen_paths.add(canonical_path)
        key = (batch, case)
        if key in records:
            raise SummaryError(f"batch{batch}/{case}: duplicate worker identity")
        receipt = None
        if batch in (2, 3):
            receipt_path = path.with_name(f"{path.stem}.startup.json")
            receipt = read_receipt(receipt_path, batch, case)
        records[key] = validate(worker, batch, case, receipt, current_adapter_sha256)

    if len(records) != 18 or set(records) != {(batch, case) for batch in BATCHES for case in CASES}:
        raise SummaryError("worker allowlist: expected exactly the 18 fixed batch/case records")
    report = aggregate(records)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise SummaryError(f"output {output_path}: refusing to overwrite existing file") from exc
    except OSError as exc:
        raise SummaryError(f"output {output_path}: cannot create summary: {exc}") from exc
    output_bytes = encoded.encode("utf-8")
    print(f"created {output_path} bytes={len(output_bytes)} sha256={sha256(output_bytes)}")
    print("workers=18 batches=3 cases=6; samples and provenance validated")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SummaryError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
