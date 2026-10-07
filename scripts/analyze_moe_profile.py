#!/usr/bin/env python3
"""Summarize a narrowly scoped MoE Nsight profile without exporting raw labels.

This helper uses only Python's standard library and opens the Nsight SQLite
file read-only. It does not import CUDA, PyTorch, Triton, or the operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import sys
from typing import Any


STAGES = (
    "reset",
    "rank_histogram",
    "prefix",
    "rank_dispatch",
    "gate_up_swiglu",
    "down",
    "combine",
)
REPEATS = 8
OUTER_LABEL = "m21.moe.roi"
THREAD_ID_BITS = 24
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")

REQUIRED_COLUMNS = {
    "NVTX_EVENTS": ("start", "end", "globalTid", "text", "textId"),
    "StringIds": ("id", "value"),
    "CUPTI_ACTIVITY_KIND_RUNTIME": (
        "start", "end", "globalTid", "correlationId", "nameId",
    ),
    "CUPTI_ACTIVITY_KIND_KERNEL": (
        "start", "end", "correlationId", "globalPid",
    ),
}


class ProfileError(ValueError):
    """A safe, actionable diagnostic for malformed profile evidence."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileError(message)


def finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} is not finite")
    return result


def integer(value: Any, label: str, *, nullable: bool = False) -> int | None:
    if nullable and value is None:
        return None
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} is not an integer")
    return value


def percentile(values: list[float], fraction: float) -> float:
    """Linear interpolation on the (n - 1) scale, including for p95."""
    require(bool(values), "cannot summarize an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_ms(nanoseconds: list[float]) -> dict[str, float | int]:
    milliseconds = [value / 1_000_000 for value in nanoseconds]
    return {
        "count": len(milliseconds),
        "p50": float(statistics.median(milliseconds)),
        "p95": percentile(milliseconds, 0.95),
        "sum": sum(milliseconds),
    }


def validate_worker(path: Path) -> dict[str, str]:
    try:
        worker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProfileError("worker JSON cannot be read or parsed") from exc
    require(isinstance(worker, dict), "worker JSON root is not an object")
    require(worker.get("schema") == 1 and not isinstance(worker.get("schema"), bool),
            "worker schema is not 1")
    require(worker.get("status") == "passed", "worker status is not passed")
    require(worker.get("phase") == "profile", "worker phase is not profile")
    for key in ("case", "arm"):
        value = worker.get(key)
        require(isinstance(value, str) and SAFE_ID.fullmatch(value) is not None,
                f"worker {key} is not a safe identifier")
    repeats = worker.get("profile_repeats")
    require(isinstance(repeats, int) and not isinstance(repeats, bool) and repeats == REPEATS,
            "worker profile_repeats is not 8")
    require(worker.get("stages") == list(STAGES), "worker stage list does not match the fixed profile")

    for stem in ("source_hashes", "artifact_hashes"):
        before = worker.get(f"{stem}_before")
        after = worker.get(f"{stem}_after")
        require(isinstance(before, dict) and bool(before),
                f"worker {stem}_before is empty or invalid")
        require(isinstance(after, dict) and before == after,
                f"worker {stem} changed or is invalid")
        require(all(isinstance(name, str) and name
                    and isinstance(digest, str) and SHA256.fullmatch(digest) is not None
                    for name, digest in before.items()),
                f"worker {stem} entries are invalid")
    input_sha = worker.get("input_sha256_after")
    require(isinstance(input_sha, str) and SHA256.fullmatch(input_sha) is not None,
            "worker input_sha256_after is invalid")
    # Do not copy or export any other worker fields; they may contain environment data.
    return {"case": worker["case"], "arm": worker["arm"]}


def validate_schema(connection: sqlite3.Connection) -> None:
    for table, required in REQUIRED_COLUMNS.items():
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        found = {row["name"] for row in rows}
        for column in required:
            require(column in found, f"required Nsight schema column is missing: {table}.{column}")


def load_nvtx_ranges(connection: sqlite3.Connection) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    labels = [OUTER_LABEL, *(f"m21.stage.{stage}" for stage in STAGES)]
    placeholders = ",".join("?" for _ in labels)
    query = f"""SELECT n.start, n.end, n.globalTid,
                       coalesce(n.text, s.value) AS label
                FROM NVTX_EVENTS AS n
                LEFT JOIN StringIds AS s ON s.id = n.textId
                WHERE coalesce(n.text, s.value) IN ({placeholders})"""
    rows = [dict(row) for row in connection.execute(query, labels)]
    outer_rows = [row for row in rows if row["label"] == OUTER_LABEL]
    require(len(outer_rows) == 1, "expected exactly one outer m21.moe.roi range")
    outer = outer_rows[0]
    outer_start = finite_number(outer["start"], "outer NVTX start")
    outer_end = finite_number(outer["end"], "outer NVTX end")
    require(outer_end > outer_start, "outer NVTX range has invalid duration")
    outer_tid = integer(outer["globalTid"], "outer NVTX thread")
    outer["start"] = outer_start
    outer["end"] = outer_end
    outer["globalTid"] = outer_tid

    stage_ranges: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    for row in rows:
        if row is outer:
            continue
        # Returned labels can only be one of the fixed labels in the SQL whitelist.
        label = row["label"]
        stage = label.removeprefix("m21.stage.")
        require(stage in stage_ranges, "unexpected whitelisted NVTX label")
        start = finite_number(row["start"], f"{stage} NVTX start")
        end = finite_number(row["end"], f"{stage} NVTX end")
        tid = integer(row["globalTid"], f"{stage} NVTX thread")
        require(end > start, f"{stage} NVTX range has invalid duration")
        require(outer_start <= start and end <= outer_end,
                f"{stage} NVTX range is outside the outer profile window")
        require(tid == outer_tid,
                f"{stage} NVTX range is not on the outer profile thread")
        row.update(start=start, end=end, globalTid=tid)
        stage_ranges[stage].append(row)

    for stage, ranges in stage_ranges.items():
        require(len(ranges) == REPEATS, f"{stage} NVTX coverage is not exactly 8")
        ranges.sort(key=lambda row: row["start"])

    chronological = sorted(
        (row for ranges in stage_ranges.values() for row in ranges),
        key=lambda row: (row["start"], row["end"]),
    )
    for previous, current in zip(chronological, chronological[1:]):
        require(current["start"] >= previous["end"],
                "stage NVTX ranges overlap")
    observed_order = [row["label"].removeprefix("m21.stage.") for row in chronological]
    expected_order = list(STAGES) * REPEATS
    require(observed_order == expected_order,
            "stage NVTX ranges do not follow the fixed repeated stage order")
    return outer, stage_ranges


def load_cuda_apis(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    # The StringIds join is constrained in SQL to CUDA runtime/driver API prefixes.
    # The API label is used only by SQLite's filter and is never returned or exported.
    query = """SELECT r.start, r.end, r.globalTid, r.correlationId
               FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
               JOIN StringIds AS s ON s.id = r.nameId
               WHERE s.value GLOB 'cuda*' OR s.value GLOB 'cu*'"""
    result = []
    process_mask = ~((1 << THREAD_ID_BITS) - 1)
    for raw in connection.execute(query):
        start = finite_number(raw["start"], "CUDA API start")
        end = finite_number(raw["end"], "CUDA API end")
        require(end >= start, "CUDA API interval has invalid duration")
        tid = integer(raw["globalTid"], "CUDA API thread")
        require(tid >= 0, "CUDA API thread is negative")
        correlation = integer(raw["correlationId"], "CUDA API correlation ID", nullable=True)
        result.append({
            "start": start,
            "end": end,
            "globalTid": tid,
            "correlationId": correlation,
            "globalPid": tid & process_mask,
        })
    return result


def load_kernels(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    # Kernel names/shortName and StringIds are deliberately not selected.
    result = []
    for raw in connection.execute(
        """SELECT start, end, correlationId, globalPid
           FROM CUPTI_ACTIVITY_KIND_KERNEL"""
    ):
        start = finite_number(raw["start"], "GPU kernel start")
        end = finite_number(raw["end"], "GPU kernel end")
        require(end > start, "GPU kernel interval has invalid duration")
        correlation = integer(raw["correlationId"], "GPU kernel correlation ID", nullable=True)
        pid = integer(raw["globalPid"], "GPU kernel process ID")
        require(pid >= 0, "GPU kernel process ID is negative")
        result.append({
            "start": start,
            "end": end,
            "correlationId": correlation,
            "globalPid": pid,
        })
    return result


def analyze_sqlite(path: Path, diagnostics: dict[str, Any]) -> dict[str, Any]:
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        validate_schema(connection)
        diagnostics["required_schema_checked"] = True
        outer, stage_ranges = load_nvtx_ranges(connection)
        diagnostics["outer_nvtx_matches"] = 1
        diagnostics["stage_nvtx_range_counts"] = {
            stage: len(stage_ranges[stage]) for stage in STAGES
        }
        apis = load_cuda_apis(connection)
        kernels = load_kernels(connection)
    finally:
        connection.close()

    outer_start, outer_end = outer["start"], outer["end"]
    invocations: list[dict[str, Any]] = []
    invocation_ids_by_api: dict[int, list[int]] = {}
    api_counts: dict[str, int] = {}
    for stage in STAGES:
        stage_invocations = []
        for ordinal, annotation in enumerate(stage_ranges[stage]):
            calls = [
                api for api in apis
                if api["globalTid"] == annotation["globalTid"]
                and annotation["start"] <= api["start"]
                and api["end"] <= annotation["end"]
            ]
            invocation_id = len(invocations)
            invocation = {
                "stage": stage,
                "ordinal": ordinal,
                "start": annotation["start"],
                "end": annotation["end"],
                "calls": calls,
                "kernels": [],
            }
            invocations.append(invocation)
            stage_invocations.append(invocation)
            for api in calls:
                invocation_ids_by_api.setdefault(id(api), []).append(invocation_id)
        api_counts[stage] = sum(len(invocation["calls"]) for invocation in stage_invocations)
    diagnostics["stage_cuda_api_call_counts"] = api_counts

    # Build process-qualified correlation candidates. Names and raw IDs are not emitted.
    apis_by_key: dict[tuple[int, int], list[dict[str, Any]]] = {}
    apis_by_correlation: dict[int, list[dict[str, Any]]] = {}
    for api in apis:
        correlation = api["correlationId"]
        if correlation is None:
            continue
        key = (correlation, api["globalPid"])
        apis_by_key.setdefault(key, []).append(api)
        apis_by_correlation.setdefault(correlation, []).append(api)

    kernels_in_outer = [
        kernel for kernel in kernels
        if kernel["start"] < outer_end and kernel["end"] > outer_start
    ]
    diagnostics["gpu_kernels_overlapping_outer"] = len(kernels_in_outer)
    stage_linked_kernel_counts = {stage: 0 for stage in STAGES}

    for kernel in kernels:
        correlation = kernel["correlationId"]
        if correlation is None:
            candidates = []
        else:
            candidates = apis_by_key.get((correlation, kernel["globalPid"]), [])
        api = None
        if len(candidates) == 1:
            api = candidates[0]
        elif len(candidates) > 1:
            require(not (kernel["start"] < outer_end and kernel["end"] > outer_start),
                    "ambiguous process-qualified CUDA API correlation in outer window")
            continue

        if api is None:
            if correlation is not None and apis_by_correlation.get(correlation):
                other_pids = {row["globalPid"] for row in apis_by_correlation[correlation]}
                if kernel["globalPid"] not in other_pids and (
                    kernel["start"] < outer_end and kernel["end"] > outer_start
                ):
                    raise ProfileError("GPU kernel process mask does not match its CUDA API")
            if kernel["start"] < outer_end and kernel["end"] > outer_start:
                raise ProfileError("unlinked extra GPU kernel overlaps outer profile window")
            continue

        memberships = invocation_ids_by_api.get(id(api), [])
        if memberships:
            require(len(memberships) == 1,
                    "CUDA API is contained by multiple stage ranges on its thread")
            invocation = invocations[memberships[0]]
            require(kernel["start"] < outer_end and kernel["end"] > outer_start,
                    "stage-linked GPU kernel is outside the outer profile window")
            invocation["kernels"].append(kernel)
            stage_linked_kernel_counts[invocation["stage"]] += 1
        elif kernel["start"] < outer_end and kernel["end"] > outer_start:
            raise ProfileError("extra GPU kernel in outer profile window is outside all stage launches")

    diagnostics["stage_linked_kernel_counts"] = stage_linked_kernel_counts
    summaries: dict[str, Any] = {}
    for stage in STAGES:
        stage_invocations = [row for row in invocations if row["stage"] == stage]
        linked_counts = [len(row["kernels"]) for row in stage_invocations]
        require(len(stage_invocations) == REPEATS,
                f"{stage} invocation count is not exactly 8")
        require(all(count == 1 for count in linked_counts),
                f"{stage} does not have exactly one linked GPU kernel per invocation")
        cpu_samples = [row["end"] - row["start"] for row in stage_invocations]
        gpu_samples = [row["kernels"][0]["end"] - row["kernels"][0]["start"]
                       for row in stage_invocations]
        summaries[stage] = {
            "cpu_inclusive_ms": summarize_ms(cpu_samples),
            "linked_gpu_duration_ms": summarize_ms(gpu_samples),
            "linked_cuda_api_call_count": api_counts[stage],
        }

    require(len(kernels_in_outer) == REPEATS * len(STAGES),
            "outer profile window does not contain exactly 56 GPU kernels")
    require(sum(stage_linked_kernel_counts.values()) == REPEATS * len(STAGES),
            "linked GPU kernel coverage is not exactly 56")
    return {
        "step_cpu_ms": (outer_end - outer_start) / 1_000_000,
        "profile_repeats": REPEATS,
        "stage_order": list(STAGES),
        "stages": summaries,
        "gpu_kernel_count_in_outer": len(kernels_in_outer),
        "linked_gpu_kernel_count": sum(stage_linked_kernel_counts.values()),
        "api_name_filter": "CUDA runtime/driver API names beginning cuda or cu; names not exported",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True, help="Nsight SQLite capture")
    parser.add_argument("--worker", type=Path, required=True, help="profile worker JSON")
    parser.add_argument("--output", type=Path, required=True, help="new JSON report path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists():
        print("profile report output already exists; refusing to overwrite", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "schema": 1,
        "status": "failed",
        "analysis_mode": "CPU-only read-only Nsight SQLite analysis; no CUDA workload",
        "worker_provenance_audited": False,
        "parser_hashes": {
            "before_sha256": None,
            "after_sha256": None,
            "unchanged": False,
        },
        "diagnostics": {
            "required_schema_checked": False,
            "outer_nvtx_matches": 0,
            "stage_nvtx_range_counts": {},
            "stage_cuda_api_call_counts": {},
            "stage_linked_kernel_counts": {},
            "gpu_kernels_overlapping_outer": None,
        },
        "caveats": [
            "CPU labels are launch-submission ranges, not GPU execution intervals.",
            "GPU stage sums are correlated kernel durations from this same-stream capture only; they are not unprofiled performance measurements.",
            "Stage and GPU durations are not independent wall-time components and should not be summed as an exclusive critical path.",
            "Raw StringIds, kernel names, process environment metadata, and input paths are not exported.",
        ],
    }
    input_paths = {"sqlite": args.sqlite, "worker": args.worker}
    before: dict[str, str | None] = {key: None for key in input_paths}
    after: dict[str, str | None] = {key: None for key in input_paths}
    failure: str | None = None

    try:
        parser_before = sha256_file(Path(__file__).resolve())
    except OSError:
        parser_before = None
        failure = "parser source cannot be hashed"

    for label, path in input_paths.items():
        try:
            before[label] = sha256_file(path)
        except OSError:
            failure = f"{label} input cannot be read"

    if failure is None:
        try:
            identity = validate_worker(args.worker)
            report["worker_provenance_audited"] = True
            report.update(phase="profile", case=identity["case"], arm=identity["arm"],
                          profile_repeats=REPEATS)
            report["worker_stage_order"] = list(STAGES)
            report.update(analyze_sqlite(args.sqlite, report["diagnostics"]))
            report["status"] = "passed"
        except ProfileError as exc:
            failure = str(exc)
        except sqlite3.Error:
            failure = "SQLite read failed"
        except OSError:
            failure = "input file read failed"
        except Exception as exc:  # Keep diagnostics useful without exposing exception payloads.
            failure = f"analysis failed ({type(exc).__name__})"

    for label, path in input_paths.items():
        try:
            after[label] = sha256_file(path)
        except OSError:
            if failure is None:
                failure = f"{label} input could not be rechecked"

    try:
        parser_after = sha256_file(Path(__file__).resolve())
    except OSError:
        parser_after = None
        if failure is None:
            failure = "parser source could not be rechecked"
    report["parser_hashes"] = {
        "before_sha256": parser_before,
        "after_sha256": parser_after,
        "unchanged": parser_before is not None and parser_before == parser_after,
    }
    if parser_before is None or parser_after is None or parser_before != parser_after:
        failure = failure or "parser source SHA256 changed or could not be verified"

    unchanged = True
    for label in input_paths:
        if before[label] is None or after[label] is None or before[label] != after[label]:
            unchanged = False
    if not unchanged:
        failure = failure or "input SHA256 changed or could not be verified during analysis"
    if failure is not None:
        report["status"] = "failed"
        report["failure"] = failure
    report["input_hashes"] = {
        label: {
            "before_sha256": before[label],
            "after_sha256": after[label],
            "unchanged": before[label] is not None and before[label] == after[label],
        }
        for label in input_paths
    }

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        print("profile report output already exists; refusing to overwrite", file=sys.stderr)
        return 2
    except OSError:
        print("profile report could not be written", file=sys.stderr)
        return 2

    if report["status"] != "passed":
        print("profile analysis failed; safe diagnostics were written", file=sys.stderr)
        return 1
    print("profile analysis passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
