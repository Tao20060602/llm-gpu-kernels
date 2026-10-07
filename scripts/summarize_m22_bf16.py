#!/usr/bin/env python3
"""Build a bounded CPU summary from the 18 M22 BF16 batch reports.

Only the fixed batch JSON allowlist is read. Runtime/environment and telemetry
fields are deliberately excluded from the summary.
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
INPUT_DIR = ROOT / "artifacts/m22-bf16-perf-20261007-v1"
OUTPUT = ROOT / "results/m22-bf16-performance-summary.json"
BATCHES = (1, 2, 3)
CASES = (
    "archive-T17",
    "archive-T193",
    "archive-T1024",
    "code-T17",
    "code-T193",
    "code-T1024",
)
FULL_ARMS = ("vllm", "default", "frozen", "sync", "async", "async_skip")
STAGE_ARMS = ("default", "frozen", "sync", "async", "async_skip")
SECTIONS = {
    "full_api": (FULL_ARMS, 12),
    "full_graph": (FULL_ARMS, 12),
    "stage_api": (STAGE_ARMS, 10),
    "stage_graph": (STAGE_ARMS, 10),
}
REFERENCES = ("frozen", "vllm")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
HEAD_RE = re.compile(r"^[0-9a-f]{40,64}$")


class SummaryError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite_positive(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{where}: expected a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise SummaryError(f"{where}: expected a finite positive number")
    return number


def validate_pass_fields(value: Any, where: str, counts: Counter[str]) -> None:
    """Require every recorded numerical/dispatch pass flag to be true."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_where = f"{where}/{key}"
            if key in ("passed", "finite", "check_inputs_passed"):
                if not isinstance(child, bool):
                    raise SummaryError(f"{child_where}: expected a boolean")
                counts[key] += 1
                if not child:
                    raise SummaryError(f"{child_where}: recorded check failed")
            validate_pass_fields(child, child_where, counts)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_pass_fields(child, f"{where}/{index}", counts)


def validate_dispatch(audits: dict[str, Any], where: str) -> None:
    if set(audits) != set(FULL_ARMS):
        raise SummaryError(f"{where}: numerical arm set differs")
    for arm in FULL_ARMS:
        record = audits[arm]
        if not isinstance(record, dict) or "numerical" not in record:
            raise SummaryError(f"{where}/{arm}: numerical result is missing")
        dispatch = record.get("dispatch")
        if arm == "vllm":
            if dispatch is not None:
                raise SummaryError(f"{where}/vllm: unexpected project dispatch audit")
            continue
        if not isinstance(dispatch, dict):
            raise SummaryError(f"{where}/{arm}: dispatch result is missing")
        if dispatch.get("status") != 0:
            raise SummaryError(f"{where}/{arm}: dispatch status is not zero")
        if dispatch.get("check_inputs_passed") is not True:
            raise SummaryError(f"{where}/{arm}: dispatch input check failed")
        ranked = dispatch.get("ranked_dispatch")
        if not isinstance(ranked, dict) or ranked.get("passed") is not True:
            raise SummaryError(f"{where}/{arm}: ranked dispatch check failed")


def validate_relative_hash_map(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise SummaryError(f"{where}: expected a nonempty hash map")
    for key, digest in value.items():
        if pathlib.PurePosixPath(str(key)).is_absolute() or "\\" in str(key):
            raise SummaryError(f"{where}: absolute or nonportable path key")
        if not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
            raise SummaryError(f"{where}/{key}: invalid sha256")
    return value


def validate_benchmark(
    record: dict[str, Any], path: str, section: str, arms: tuple[str, ...],
    expected_rounds: int,
) -> None:
    if not isinstance(record, dict):
        raise SummaryError(f"{path}/{section}: expected an object")
    summary = record.get("summary")
    if not isinstance(summary, dict) or set(summary) != set(arms):
        raise SummaryError(f"{path}/{section}: summary arm set differs")
    rounds = record.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != expected_rounds:
        raise SummaryError(f"{path}/{section}: expected {expected_rounds} rounds")
    samples_per_round = record.get("samples_per_round")
    if not isinstance(samples_per_round, int) or samples_per_round <= 0:
        raise SummaryError(f"{path}/{section}: invalid samples_per_round")

    for arm, stats in summary.items():
        if not isinstance(stats, dict):
            raise SummaryError(f"{path}/{section}/{arm}: invalid summary record")
        for metric, value in stats.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise SummaryError(f"{path}/{section}/{arm}/{metric}: nonfinite value")
        finite_positive(stats.get("p50_ms"), f"{path}/{section}/{arm}/p50_ms")
        finite_positive(stats.get("p95_ms"), f"{path}/{section}/{arm}/p95_ms")

    for round_index, round_record in enumerate(rounds):
        if not isinstance(round_record, dict):
            raise SummaryError(f"{path}/{section}/round{round_index}: invalid round")
        if set(round_record.get("order", [])) != set(arms):
            raise SummaryError(f"{path}/{section}/round{round_index}: arm order differs")
        for dimension in ("samples_ms_per_gemm", "wall_ms_per_gemm"):
            values_by_arm = round_record.get(dimension)
            if not isinstance(values_by_arm, dict) or set(values_by_arm) != set(arms):
                raise SummaryError(
                    f"{path}/{section}/round{round_index}/{dimension}: arm set differs")
            for arm, values in values_by_arm.items():
                if not isinstance(values, list) or len(values) != samples_per_round:
                    raise SummaryError(
                        f"{path}/{section}/round{round_index}/{dimension}/{arm}: sample count differs")
                for sample_index, value in enumerate(values):
                    finite_positive(
                        value,
                        f"{path}/{section}/round{round_index}/{dimension}/{arm}/{sample_index}",
                    )


def geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise SummaryError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def comparison(
    values_by_case: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Summarize (measured p50, reference p50) pairs with equal case weights."""
    ratios = {
        case: measured / reference
        for case, (measured, reference) in values_by_case.items()
    }
    speedups = {case: 1.0 / ratio for case, ratio in ratios.items()}
    worst_case = max(ratios, key=ratios.__getitem__)
    worst_ratio = ratios[worst_case]
    return {
        "equal_weight_case_count": len(ratios),
        "case_time_ratios": ratios,
        "case_speedups": speedups,
        "time_ratio_geomean": geometric_mean(list(ratios.values())),
        "speedup_geomean": geometric_mean(list(speedups.values())),
        "worst_case": worst_case,
        "worst_case_time_ratio": worst_ratio,
        "worst_case_regression_fraction": worst_ratio - 1.0,
    }


def main() -> int:
    if OUTPUT.exists():
        raise SummaryError(f"refusing to overwrite {OUTPUT.relative_to(ROOT)}")

    expected_paths = [
        INPUT_DIR / f"batch{batch}-{case}.json"
        for batch in BATCHES
        for case in CASES
    ]
    if len(expected_paths) != 18 or any(not path.is_file() for path in expected_paths):
        raise SummaryError("the exact 18-file batch allowlist is incomplete")

    records: dict[tuple[int, str], dict[str, Any]] = {}
    case_inputs: dict[str, set[str]] = {case: set() for case in CASES}
    case_artifacts: dict[str, set[str]] = {case: set() for case in CASES}
    source_maps: set[str] = set()
    binary_hashes: set[str] = set()
    heads: set[str] = set()
    all_check_counts: Counter[str] = Counter()

    for path in expected_paths:
        relative_path = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        try:
            item = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SummaryError(f"{relative_path}: invalid JSON") from exc
        batch_match = re.fullmatch(r"batch([1-3])-(.+)\.json", path.name)
        if batch_match is None:
            raise SummaryError(f"{relative_path}: unexpected filename")
        batch = int(batch_match.group(1))
        case = batch_match.group(2)
        if item.get("status") != "passed" or item.get("batch") != batch or item.get("case") != case:
            raise SummaryError(f"{relative_path}: status or identity mismatch")
        if item.get("schema") != 1:
            raise SummaryError(f"{relative_path}: unsupported schema")

        source_before = validate_relative_hash_map(
            item.get("source_hashes_before"), f"{relative_path}/source_hashes_before")
        source_after = validate_relative_hash_map(
            item.get("source_hashes_after"), f"{relative_path}/source_hashes_after")
        artifact_before = validate_relative_hash_map(
            item.get("artifact_hashes_before"), f"{relative_path}/artifact_hashes_before")
        artifact_after = validate_relative_hash_map(
            item.get("artifact_hashes_after"), f"{relative_path}/artifact_hashes_after")
        if source_before != source_after:
            raise SummaryError(f"{relative_path}: source hashes changed")
        if artifact_before != artifact_after:
            raise SummaryError(f"{relative_path}: artifact hashes changed")
        if item.get("input_sha256") != item.get("input_sha256_after"):
            raise SummaryError(f"{relative_path}: input hash changed")
        if not HASH_RE.fullmatch(str(item.get("input_sha256", ""))):
            raise SummaryError(f"{relative_path}: invalid input hash")
        if not HASH_RE.fullmatch(str(item.get("audit_record_sha256", ""))):
            raise SummaryError(f"{relative_path}: invalid audit record hash")
        binary = item.get("binary")
        if not isinstance(binary, dict):
            raise SummaryError(f"{relative_path}: binary provenance is missing")
        binary_before = binary.get("sha256_before")
        binary_after = binary.get("sha256_after")
        if not HASH_RE.fullmatch(str(binary_before or "")) or binary_before != binary_after:
            raise SummaryError(f"{relative_path}: binary hash missing or changed")
        head = item.get("head")
        if not isinstance(head, str) or not HEAD_RE.fullmatch(head):
            raise SummaryError(f"{relative_path}: head is not a commit hash")

        for section, (arms, expected_rounds) in SECTIONS.items():
            validate_benchmark(item.get(section), relative_path, section, arms, expected_rounds)
        for check_name in ("audit_before", "audit_after"):
            validate_dispatch(item.get(check_name, {}), f"{relative_path}/{check_name}")
            validate_pass_fields(item[check_name], f"{relative_path}/{check_name}", all_check_counts)
        graph_numerics = item.get("graph_numerics")
        if not isinstance(graph_numerics, dict) or set(graph_numerics) != set(FULL_ARMS):
            raise SummaryError(f"{relative_path}/graph_numerics: arm set differs")
        validate_pass_fields(graph_numerics, f"{relative_path}/graph_numerics", all_check_counts)

        source_maps.add(json.dumps(source_before, sort_keys=True, separators=(",", ":")))
        binary_hashes.add(binary_before)
        heads.add(head)
        case_inputs[case].add(item["input_sha256"])
        case_artifacts[case].add(json.dumps(artifact_before, sort_keys=True, separators=(",", ":")))

        measurements = {
            section: item[section]
            for section in SECTIONS
        }
        case_batch_p50_p95: dict[str, dict[str, dict[str, float]]] = {}
        for section, (arms, _) in SECTIONS.items():
            case_batch_p50_p95[section] = {
                arm: {
                    "p50_ms": float(item[section]["summary"][arm]["p50_ms"]),
                    "p95_ms": float(item[section]["summary"][arm]["p95_ms"]),
                }
                for arm in arms
            }

        records[(batch, case)] = {
            "batch": batch,
            "case": case,
            "status": item["status"],
            "selected_bucket": item.get("selected_bucket"),
            "cases_are_previously_measured_m20_inputs": item.get(
                "cases_are_previously_measured_m20_inputs"),
            "frozen_config": item.get("frozen_config"),
            "cuda_gate_up_config": item.get("cuda_gate_up_config"),
            "graph_repeats": item.get("graph_repeats"),
            "stage_scope": item.get("stage_scope"),
            "limits": item.get("limits"),
            "audit_before": item["audit_before"],
            "audit_after": item["audit_after"],
            "graph_numerics": graph_numerics,
            "case_batch_p50_p95": case_batch_p50_p95,
            "measurements": measurements,
            "provenance": {
                "report_path": relative_path,
                "report_bytes": len(raw),
                "report_sha256": sha256(raw),
                "head": head,
                "input_sha256_before": item["input_sha256"],
                "input_sha256_after": item["input_sha256_after"],
                "binary_sha256_before": binary_before,
                "binary_sha256_after": binary_after,
                "audit_record_sha256": item["audit_record_sha256"],
                "source_hashes_before": source_before,
                "source_hashes_after": source_after,
                "artifact_hashes_before": artifact_before,
                "artifact_hashes_after": artifact_after,
            },
        }

    if len(records) != 18:
        raise SummaryError(f"expected 18 unique records, got {len(records)}")
    if len(source_maps) != 1 or len(binary_hashes) != 1 or len(heads) != 1:
        raise SummaryError("source, binary, or HEAD provenance differs across batches")
    if any(len(hashes) != 1 for hashes in case_inputs.values()):
        raise SummaryError("a case input hash differs across its three batches")
    if any(len(hashes) != 1 for hashes in case_artifacts.values()):
        raise SummaryError("a case artifact hash set differs across its three batches")

    # Equal-weight comparisons for each individual batch, using six cases.
    by_batch: dict[str, Any] = {}
    for batch in BATCHES:
        scope_results: dict[str, Any] = {}
        for section, (arms, _) in SECTIONS.items():
            refs: dict[str, Any] = {}
            for reference in REFERENCES:
                if reference not in arms:
                    refs[reference] = {
                        "available": False,
                        "reason": f"{reference} arm is not present in {section}",
                    }
                    continue
                method_results: dict[str, Any] = {}
                for arm in arms:
                    pairs = {
                        case: (
                            records[(batch, case)]["case_batch_p50_p95"][section][arm]["p50_ms"],
                            records[(batch, case)]["case_batch_p50_p95"][section][reference]["p50_ms"],
                        )
                        for case in CASES
                    }
                    method_results[arm] = comparison(pairs)
                refs[reference] = {"available": True, "arms": method_results}
            scope_results[section] = refs
        by_batch[f"batch{batch}"] = scope_results

    # First summarize each case across three batch p50s, then aggregate cases.
    per_case_three_batch: dict[str, Any] = {}
    for case in CASES:
        sections: dict[str, Any] = {}
        for section, (arms, _) in SECTIONS.items():
            sections[section] = {
                arm: {
                    "batch_p50_ms": [
                        records[(batch, case)]["case_batch_p50_p95"][section][arm]["p50_ms"]
                        for batch in BATCHES
                    ],
                    "p50_ms_geomean_three_batches": geometric_mean([
                        records[(batch, case)]["case_batch_p50_p95"][section][arm]["p50_ms"]
                        for batch in BATCHES
                    ]),
                }
                for arm in arms
            }
        per_case_three_batch[case] = sections

    overall: dict[str, Any] = {}
    for section, (arms, _) in SECTIONS.items():
        refs: dict[str, Any] = {}
        for reference in REFERENCES:
            if reference not in arms:
                refs[reference] = {
                    "available": False,
                    "reason": f"{reference} arm is not present in {section}",
                }
                continue
            method_results: dict[str, Any] = {}
            for arm in arms:
                pairs = {
                    case: (
                        per_case_three_batch[case][section][arm][
                            "p50_ms_geomean_three_batches"],
                        per_case_three_batch[case][section][reference][
                            "p50_ms_geomean_three_batches"],
                    )
                    for case in CASES
                }
                method_results[arm] = comparison(pairs)
            refs[reference] = {"available": True, "arms": method_results}
        overall[section] = refs

    report = {
        "schema": 1,
        "status": "complete",
        "experiment": "M22 BF16 repeated performance capture",
        "input_count": len(records),
        "batches": list(BATCHES),
        "cases": list(CASES),
        "arms": {"full_api_and_graph": list(FULL_ARMS), "stage_api_and_graph": list(STAGE_ARMS)},
        "aggregation_policy": {
            "time_metric": "p50_ms from each original benchmark summary",
            "batch_comparisons": "equal-weight geometric mean of six casewise p50 time ratios and reciprocal speedups",
            "overall_comparisons": "for each case, geometric mean of its three batch p50s first; then equal-weight aggregation over six cases",
            "worst_regression": "largest casewise time ratio minus one relative to the named reference",
            "p95": "reported per case and batch from the original summary; not pooled across batches",
            "winner_selected": False,
        },
        "validation": {
            "all_18_status_passed": True,
            "all_numeric_and_dispatch_before_after_passed": True,
            "all_graph_numerics_passed": True,
            "source_hashes_stable_before_after_and_across_records": True,
            "artifact_hashes_stable_before_after_and_per_case": True,
            "input_hashes_stable_before_after_and_per_case": True,
            "binary_hash_stable_before_after_and_across_records": True,
            "head_stable_across_records": True,
            "check_flag_counts": dict(all_check_counts),
        },
        "case_batch_results": [records[(batch, case)] for batch in BATCHES for case in CASES],
        "batch_equal_weight_comparisons": by_batch,
        "per_case_three_batch_p50_geomeans": per_case_three_batch,
        "overall_equal_weight_comparisons": overall,
        "interpretation_limit": "Descriptive measurements only; this summary selects no arm and does not declare a performance winner.",
    }

    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with OUTPUT.open("x", encoding="utf-8") as output_file:
            output_file.write(encoded)
    except FileExistsError as exc:
        raise SummaryError(f"refusing to overwrite {OUTPUT.relative_to(ROOT)}") from exc
    output_bytes = encoded.encode("utf-8")
    print(f"created {OUTPUT.relative_to(ROOT)} bytes={len(output_bytes)} sha256={sha256(output_bytes)}")
    print(f"records={len(records)} cases={len(CASES)} batches={len(BATCHES)} checks={dict(all_check_counts)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SummaryError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
