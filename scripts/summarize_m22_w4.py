#!/usr/bin/env python3
"""Create a CPU-only, allowlisted summary of the six M22 W4 single-batch workers.

Only six fixed JSON reports are read. Runtime metadata, telemetry, backend
configuration dumps, logs, and local paths from record-hash keys are omitted.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import sys
from collections import Counter
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "artifacts/m22-w4-perf-20261007-v2"
OUTPUT = ROOT / "results/m22-w4-performance-summary.json"
BATCH = 1
CASES = (
    "archive-T17",
    "archive-T193",
    "archive-T1024",
    "code-T17",
    "code-T193",
    "code-T1024",
)
QUANT_ARMS = ("quant_vllm", "quant_frozen", "quant_cuda_sync", "w4_sync")
FULL_ARMS = (*QUANT_ARMS, "bf16_vllm", "bf16_frozen")
STAGE_ARMS = ("quant_frozen", "quant_cuda_sync", "w4_sync")
SECTIONS = {
    "full_api": (FULL_ARMS, 1),
    "full_graph": (FULL_ARMS, 8),
    "stage_api": (STAGE_ARMS, 1),
    "stage_graph": (STAGE_ARMS, 8),
}
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
HEAD_RE = re.compile(r"^[0-9a-f]{40,64}$")
MEASURE_SUMMARY_FIELDS = ("p50_ms", "p95_ms", "effective_tflops", "wall_p50_ms")
ERROR_FIELDS = ("finite", "relative_rms", "normalized_max", "max_abs")


class SummaryError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite_number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{where}: expected a number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "finite positive" if positive else "finite"
        raise SummaryError(f"{where}: expected a {qualifier} number")
    return number


def check_flags(value: Any, where: str, counts: Counter[str]) -> None:
    """Require recorded numerical and dispatch gates to be true."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_where = f"{where}/{key}"
            if key in ("passed", "finite", "check_inputs_passed"):
                if not isinstance(child, bool):
                    raise SummaryError(f"{child_where}: expected a boolean")
                counts[key] += 1
                if not child:
                    raise SummaryError(f"{child_where}: check failed")
            check_flags(child, child_where, counts)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_flags(child, f"{where}/{index}", counts)


def safe_hash_map(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise SummaryError(f"{where}: expected a nonempty hash map")
    for key, digest in value.items():
        if pathlib.PurePosixPath(str(key)).is_absolute() or "\\" in str(key):
            raise SummaryError(f"{where}: absolute or nonportable path key")
        if not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
            raise SummaryError(f"{where}/{key}: invalid SHA-256")
    return value


def safe_record_hashes(value: Any, where: str) -> dict[str, str]:
    """Replace CLI path keys with stable logical names before writing output."""
    if not isinstance(value, dict) or len(value) != 3:
        raise SummaryError(f"{where}: expected three audit/pack record hashes")
    result: dict[str, str] = {}
    for path_key, digest in value.items():
        if not isinstance(path_key, str) or not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
            raise SummaryError(f"{where}: invalid record hash entry")
        normalized = path_key.replace("\\", "/").lower()
        if "pack" in normalized or "quant-format" in normalized or "format-preparation" in normalized:
            role = "pack_record"
        elif "w4" in normalized:
            role = "w4_audit"
        elif "bf16" in normalized or "cuda-audit" in normalized or "cuda_audit" in normalized:
            role = "bf16_audit"
        else:
            raise SummaryError(f"{where}: record role could not be classified")
        if role in result:
            raise SummaryError(f"{where}: duplicate logical record role")
        result[role] = digest
    if set(result) != {"w4_audit", "bf16_audit", "pack_record"}:
        raise SummaryError(f"{where}: expected W4 audit, BF16 audit, and pack record")
    return result


def expected_orders(arms: tuple[str, ...]) -> list[list[str]]:
    orders: list[list[str]] = []
    for base in (list(arms), list(reversed(arms))):
        orders.extend(base[index:] + base[:index] for index in range(len(base)))
    return orders


def validate_measurement(
    record: Any,
    relative_path: str,
    section: str,
    arms: tuple[str, ...],
    divisor: int,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise SummaryError(f"{relative_path}/{section}: missing measurement")
    summary = record.get("summary")
    if not isinstance(summary, dict) or set(summary) != set(arms):
        raise SummaryError(f"{relative_path}/{section}: arm set differs")
    rounds = record.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 2 * len(arms):
        raise SummaryError(f"{relative_path}/{section}: unexpected round count")
    if record.get("samples_per_round") != 6 or record.get("repeats") != 2:
        raise SummaryError(f"{relative_path}/{section}: sample/repeat contract differs")
    if record.get("divisor") != divisor:
        raise SummaryError(f"{relative_path}/{section}: divisor differs")

    clean_summary: dict[str, dict[str, float]] = {}
    for arm in arms:
        stats = summary.get(arm)
        if not isinstance(stats, dict):
            raise SummaryError(f"{relative_path}/{section}/{arm}: invalid summary")
        clean_summary[arm] = {
            metric: finite_number(stats.get(metric), f"{relative_path}/{section}/{arm}/{metric}", positive=True)
            for metric in MEASURE_SUMMARY_FIELDS
        }

    wanted_orders = expected_orders(arms)
    clean_rounds: list[dict[str, Any]] = []
    for index, round_record in enumerate(rounds):
        if not isinstance(round_record, dict) or round_record.get("order") != wanted_orders[index]:
            raise SummaryError(f"{relative_path}/{section}/round{index}: arm order differs")
        cleaned: dict[str, Any] = {"order": list(wanted_orders[index])}
        for dimension in ("samples_ms_per_gemm", "wall_ms_per_gemm"):
            values_by_arm = round_record.get(dimension)
            if not isinstance(values_by_arm, dict) or set(values_by_arm) != set(arms):
                raise SummaryError(f"{relative_path}/{section}/round{index}/{dimension}: arm set differs")
            cleaned_values: dict[str, list[float]] = {}
            for arm in arms:
                values = values_by_arm[arm]
                if not isinstance(values, list) or len(values) != 6:
                    raise SummaryError(f"{relative_path}/{section}/round{index}/{dimension}/{arm}: sample count differs")
                cleaned_values[arm] = [
                    finite_number(v, f"{relative_path}/{section}/round{index}/{dimension}/{arm}/{i}", positive=True)
                    for i, v in enumerate(values)
                ]
            cleaned[dimension] = cleaned_values
        clean_rounds.append(cleaned)

    return {
        "summary": clean_summary,
        "rounds": clean_rounds,
        "samples_per_round": 6,
        "repeats": 2,
        "divisor": divisor,
    }


def compact_error(value: Any, where: str, counts: Counter[str], *, require_passed: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SummaryError(f"{where}: missing numerical result")
    check_flags(value, where, counts)
    if require_passed and value.get("passed") is not True:
        raise SummaryError(f"{where}: numerical check did not pass")
    result: dict[str, Any] = {}
    for key in ERROR_FIELDS:
        if key not in value:
            raise SummaryError(f"{where}: missing {key}")
        if key == "finite":
            if value[key] is not True:
                raise SummaryError(f"{where}/finite: expected true")
            result[key] = True
        else:
            result[key] = finite_number(value[key], f"{where}/{key}")
    if "passed" in value:
        result["passed"] = value["passed"]
    return result


def compact_dispatch(value: Any, where: str, counts: Counter[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SummaryError(f"{where}: missing dispatch result")
    check_flags(value, where, counts)
    if value.get("passed") is not True or value.get("check_inputs_passed") is not True:
        raise SummaryError(f"{where}: dispatch check did not pass")
    if isinstance(value.get("status"), bool) or value.get("status") != 0:
        raise SummaryError(f"{where}/status: expected zero")
    ranked = value.get("ranked_dispatch")
    if not isinstance(ranked, dict) or ranked.get("passed") is not True:
        raise SummaryError(f"{where}/ranked_dispatch: check did not pass")
    # Keep boolean assertions only; omit detailed route arrays from the summary.
    ranked_flags = {key: child for key, child in ranked.items() if isinstance(child, bool)}
    return {
        "status": 0,
        "check_inputs_passed": True,
        "passed": True,
        "ranked_dispatch_flags": ranked_flags,
    }


def compact_audits(value: Any, where: str, counts: Counter[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(FULL_ARMS):
        raise SummaryError(f"{where}: full arm set differs")
    result: dict[str, Any] = {}
    for arm in FULL_ARMS:
        item = value[arm]
        if not isinstance(item, dict):
            raise SummaryError(f"{where}/{arm}: invalid audit")
        numerical = compact_error(item.get("numerical"), f"{where}/{arm}/numerical", counts, require_passed=True)
        if arm.endswith("_vllm"):
            if "dispatch" in item:
                raise SummaryError(f"{where}/{arm}: unexpected project dispatch record")
            result[arm] = {"numerical": numerical, "dispatch": None}
        else:
            result[arm] = {
                "numerical": numerical,
                "dispatch": compact_dispatch(item.get("dispatch"), f"{where}/{arm}/dispatch", counts),
            }
    return result


def validate_binary_set(item: dict[str, Any], where: str) -> dict[str, dict[str, str]]:
    binaries = item.get("binaries")
    if not isinstance(binaries, dict) or set(binaries) != {"w4", "bf16"}:
        raise SummaryError(f"{where}: expected W4 and BF16 binary provenance")
    clean: dict[str, dict[str, str]] = {}
    for role in ("w4", "bf16"):
        info = binaries[role]
        if not isinstance(info, dict):
            raise SummaryError(f"{where}/{role}: invalid binary provenance")
        before, after = info.get("sha256_before"), info.get("sha256_after")
        if not isinstance(before, str) or not HASH_RE.fullmatch(before) or before != after:
            raise SummaryError(f"{where}/{role}: binary hash missing or changed")
        clean[role] = {"sha256_before": before, "sha256_after": after}
    return clean


def geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise SummaryError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def comparison(values: dict[str, tuple[float, float]]) -> dict[str, Any]:
    ratios = {case: measured / reference for case, (measured, reference) in values.items()}
    speedups = {case: 1.0 / ratio for case, ratio in ratios.items()}
    worst_case = max(ratios, key=ratios.__getitem__)
    worst_ratio = ratios[worst_case]
    ratio_geomean = geometric_mean(list(ratios.values()))
    return {
        "equal_weight_case_count": len(ratios),
        "case_time_ratios": ratios,
        "case_speedups": speedups,
        "time_ratio_geomean": ratio_geomean,
        "time_reduction_fraction_from_geomean_ratio": 1.0 - ratio_geomean,
        "time_reduction_percent_from_geomean_ratio": 100.0 * (1.0 - ratio_geomean),
        "speedup_geomean": geometric_mean(list(speedups.values())),
        "worst_case": worst_case,
        "worst_case_time_ratio": worst_ratio,
        "worst_case_regression_fraction": worst_ratio - 1.0,
    }


def compare_scope(
    records: dict[str, dict[str, Any]],
    section: str,
    references: tuple[str, ...],
    measured_arms: tuple[str, ...],
    cases: tuple[str, ...] = CASES,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for reference in references:
        arms: dict[str, Any] = {}
        for arm in measured_arms:
            pairs = {
                case: (
                    records[case]["measurements"][section]["summary"][arm]["p50_ms"],
                    records[case]["measurements"][section]["summary"][reference]["p50_ms"],
                )
                for case in cases
            }
            arms[arm] = comparison(pairs)
        result[reference] = {"arms": arms}
    return result


def classify_input_record_hashes(value: dict[str, str], where: str) -> dict[str, str]:
    if set(value) != {"w4_audit", "bf16_audit", "pack_record"}:
        raise SummaryError(f"{where}: missing logical record hashes")
    return value


def main() -> int:
    if OUTPUT.exists():
        raise SummaryError("refusing to overwrite results/m22-w4-performance-summary.json")
    expected_paths = [INPUT_DIR / f"batch{BATCH}-{case}.json" for case in CASES]
    if len(expected_paths) != 6 or any(not path.is_file() for path in expected_paths):
        raise SummaryError("the exact six-file single-batch allowlist is incomplete")

    records: dict[str, dict[str, Any]] = {}
    source_maps: set[str] = set()
    head_hashes: set[str] = set()
    binary_sets: set[str] = set()
    packed_hashes: set[str] = set()
    packed_gpu_digests: set[str] = set()
    input_record_hash_sets: set[str] = set()
    check_counts: Counter[str] = Counter()

    for path in expected_paths:
        relative_path = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        try:
            item = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SummaryError(f"{relative_path}: invalid JSON") from exc
        if not isinstance(item, dict):
            raise SummaryError(f"{relative_path}: top level must be an object")
        if item.get("schema") != 1 or item.get("status") != "passed":
            raise SummaryError(f"{relative_path}: unsupported schema or worker failed")
        if item.get("batch") != BATCH or item.get("case") != path.stem.removeprefix(f"batch{BATCH}-"):
            raise SummaryError(f"{relative_path}: worker identity differs")
        case = item["case"]
        if case not in CASES or case in records:
            raise SummaryError(f"{relative_path}: unexpected or duplicate case")

        source_before = safe_hash_map(item.get("source_hashes_before"), f"{relative_path}/source_hashes_before")
        source_after = safe_hash_map(item.get("source_hashes_after"), f"{relative_path}/source_hashes_after")
        artifact_before = safe_hash_map(item.get("artifact_hashes_before"), f"{relative_path}/artifact_hashes_before")
        artifact_after = safe_hash_map(item.get("artifact_hashes_after"), f"{relative_path}/artifact_hashes_after")
        if source_before != source_after or artifact_before != artifact_after:
            raise SummaryError(f"{relative_path}: source/artifact hashes changed")

        input_before = item.get("input_sha256")
        input_after = item.get("input_sha256_after")
        if not isinstance(input_before, str) or not HASH_RE.fullmatch(input_before) or input_before != input_after:
            raise SummaryError(f"{relative_path}: input hash missing or changed")

        record_hashes_before = classify_input_record_hashes(
            safe_record_hashes(item.get("record_hashes_before"), f"{relative_path}/record_hashes_before"),
            f"{relative_path}/record_hashes_before",
        )
        record_hashes_after = classify_input_record_hashes(
            safe_record_hashes(item.get("record_hashes_after"), f"{relative_path}/record_hashes_after"),
            f"{relative_path}/record_hashes_after",
        )
        if record_hashes_before != record_hashes_after:
            raise SummaryError(f"{relative_path}: audit/pack record hashes changed")
        input_record_hash_sets.add(json.dumps(record_hashes_before, sort_keys=True, separators=(",", ":")))

        binaries = validate_binary_set(item, relative_path)
        packed_before, packed_after = item.get("packed_sha256_before"), item.get("packed_sha256_after")
        if not isinstance(packed_before, str) or not HASH_RE.fullmatch(packed_before) or packed_before != packed_after:
            raise SummaryError(f"{relative_path}: packed weight hash missing or changed")
        gpu_digest = item.get("packed_gpu_digest_after")
        if not isinstance(gpu_digest, dict) or set(gpu_digest) != {"words", "scales"}:
            raise SummaryError(f"{relative_path}: packed GPU digest fields differ")
        if any(not isinstance(gpu_digest[key], str) or not HASH_RE.fullmatch(gpu_digest[key]) for key in ("words", "scales")):
            raise SummaryError(f"{relative_path}: invalid packed GPU digest")
        packed_gpu_digests.add(json.dumps(gpu_digest, sort_keys=True, separators=(",", ":")))

        head = item.get("head")
        if not isinstance(head, str) or not HEAD_RE.fullmatch(head):
            raise SummaryError(f"{relative_path}: invalid HEAD hash")

        measurements = {
            section: validate_measurement(
                item.get(section), relative_path, section, arms, divisor)
            for section, (arms, divisor) in SECTIONS.items()
        }
        audits_before = compact_audits(item.get("audit_before"), f"{relative_path}/audit_before", check_counts)
        audits_after = compact_audits(item.get("audit_after"), f"{relative_path}/audit_after", check_counts)
        graph_values = item.get("graph_numerics")
        if not isinstance(graph_values, dict) or set(graph_values) != set(FULL_ARMS):
            raise SummaryError(f"{relative_path}/graph_numerics: full arm set differs")
        graph_numerics = {
            arm: compact_error(graph_values[arm], f"{relative_path}/graph_numerics/{arm}", check_counts, require_passed=True)
            for arm in FULL_ARMS
        }

        format_change = compact_error(
            item.get("format_forward_change"), f"{relative_path}/format_forward_change", check_counts,
            require_passed=False,
        )
        source_maps.add(json.dumps(source_before, sort_keys=True, separators=(",", ":")))
        head_hashes.add(head)
        binary_sets.add(json.dumps(binaries, sort_keys=True, separators=(",", ":")))
        packed_hashes.add(packed_before)

        records[case] = {
            "batch": BATCH,
            "case": case,
            "status": "passed",
            "selected_bucket": item.get("selected_bucket"),
            "frozen_config": item.get("frozen_config"),
            "w4_config": item.get("w4_config"),
            "graph_repeats": item.get("graph_repeats"),
            "limits": item.get("limits"),
            "stage_scope": item.get("stage_scope"),
            "format_forward_change_vs_original_bf16": format_change,
            "audit_before": audits_before,
            "audit_after": audits_after,
            "graph_numerics": graph_numerics,
            "measurements": measurements,
            "provenance": {
                "worker_path": relative_path,
                "worker_bytes": len(raw),
                "worker_sha256": sha256(raw),
                "head": head,
                "input_sha256_before": input_before,
                "input_sha256_after": input_after,
                "source_hashes_before": source_before,
                "source_hashes_after": source_after,
                "artifact_hashes_before": artifact_before,
                "artifact_hashes_after": artifact_after,
                "record_hashes_before": record_hashes_before,
                "record_hashes_after": record_hashes_after,
                "binaries": binaries,
                "packed_sha256_before": packed_before,
                "packed_sha256_after": packed_after,
                "packed_gpu_digest_after": gpu_digest,
            },
        }

    if set(records) != set(CASES):
        raise SummaryError("the six expected case identities are incomplete")
    if (len(source_maps) != 1 or len(head_hashes) != 1 or len(binary_sets) != 1
            or len(packed_hashes) != 1 or len(packed_gpu_digests) != 1
            or len(input_record_hash_sets) != 1):
        raise SummaryError("source, HEAD, binary, packed, or audit-record provenance differs across cases")

    full_same_quant = {
        section: compare_scope(records, section, ("quant_vllm", "quant_frozen", "quant_cuda_sync"), QUANT_ARMS)
        for section in ("full_api", "full_graph")
    }
    lossy_tradeoff = {
        "accuracy_interpretation": "Original BF16 arms are a separate lossy-format latency tradeoff, not a same-quantization comparison.",
        "sections": {
            section: compare_scope(records, section, ("bf16_vllm", "bf16_frozen"), QUANT_ARMS)
            for section in ("full_api", "full_graph")
        },
    }
    stage_comparisons = {
        section: compare_scope(records, section, ("quant_frozen", "quant_cuda_sync"), STAGE_ARMS)
        for section in ("stage_api", "stage_graph")
    }

    def descriptive_scope(cases: tuple[str, ...]) -> dict[str, Any]:
        return {
            "cases": list(cases),
            "case_count": len(cases),
            "same_quantization_full_comparisons": {
                section: compare_scope(
                    records, section, ("quant_vllm", "quant_frozen", "quant_cuda_sync"), QUANT_ARMS, cases)
                for section in ("full_api", "full_graph")
            },
            "separate_lossy_tradeoff": {
                "accuracy_interpretation": "Original BF16 arms are a separate lossy-format latency tradeoff, not a same-quantization comparison.",
                "sections": {
                    section: compare_scope(records, section, ("bf16_vllm", "bf16_frozen"), QUANT_ARMS, cases)
                    for section in ("full_api", "full_graph")
                },
            },
            "stage_comparisons": {
                section: compare_scope(
                    records, section, ("quant_frozen", "quant_cuda_sync"), STAGE_ARMS, cases)
                for section in ("stage_api", "stage_graph")
            },
        }

    post_result_scope_descriptions = {
        "small_token_scope": {
            **descriptive_scope(("archive-T17", "archive-T193", "code-T17", "code-T193")),
            "scope_note": "Descriptive post-result grouping of T17 and T193 cases; not a predeclared holdout policy or winner selection.",
        },
        "T1024_scope": {
            **descriptive_scope(("archive-T1024", "code-T1024")),
            "scope_note": "Descriptive separate reporting of both T1024 cases; both remain included in the six-case overall comparisons.",
        },
    }

    report = {
        "schema": 1,
        "status": "complete",
        "experiment": "M22 W4A16 single-batch performance capture",
        "sampling_scope": {
            "batch_count": 1,
            "batch": BATCH,
            "case_count": 6,
            "single_process_per_case": True,
            "cross_batch_stability_claimed": False,
        },
        "cases": list(CASES),
        "arms": {
            "full_api_and_graph": list(FULL_ARMS),
            "same_quantization_comparisons": list(QUANT_ARMS),
            "separate_lossy_tradeoff_arms": ["bf16_vllm", "bf16_frozen"],
            "stage_api_and_graph": list(STAGE_ARMS),
        },
        "aggregation_policy": {
            "time_metric": "original worker p50_ms",
            "case_weights": "equal weight across the six pinned cases",
            "comparison_ratio": "measured arm p50 divided by named reference arm p50",
            "speedup": "reciprocal of time ratio",
            "time_reduction": "1 minus the equal-weight geometric mean time ratio; distinct from the geometric mean of per-case speedups",
            "worst_regression": "largest casewise time ratio minus one",
            "p95": "preserved per case; not pooled or aggregated",
            "winner_selected": False,
        },
        "validation": {
            "all_six_workers_passed": True,
            "all_numeric_dispatch_and_graph_checks_passed": True,
            "source_artifact_record_input_binary_and_packed_hashes_stable": True,
            "source_hashes_and_HEAD_stable_across_cases": True,
            "binary_and_packed_hashes_stable_across_cases": True,
            "check_flag_counts": dict(check_counts),
        },
        "case_results": [records[case] for case in CASES],
        "same_quantization_full_comparisons": full_same_quant,
        "separate_lossy_tradeoff": lossy_tradeoff,
        "stage_comparisons": stage_comparisons,
        "post_result_scope_descriptions": post_result_scope_descriptions,
        "interpretation_limit": "Single-batch diagnostic measurements only; do not generalize stability or select a performance winner.",
    }

    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with OUTPUT.open("x", encoding="utf-8") as output_file:
            output_file.write(encoded)
    except FileExistsError as exc:
        raise SummaryError("refusing to overwrite results/m22-w4-performance-summary.json") from exc
    output_bytes = encoded.encode("utf-8")
    print(f"created results/m22-w4-performance-summary.json bytes={len(output_bytes)} sha256={sha256(output_bytes)}")
    print(f"records={len(records)} cases={len(CASES)} batch={BATCH} checks={dict(check_counts)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SummaryError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
