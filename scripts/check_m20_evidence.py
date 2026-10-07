#!/usr/bin/env python3
"""Audit M20 real-MoE benchmark records using only Python's standard library.

This tool reads saved JSON and compiled artifacts. It does not import the
operator package, PyTorch, Triton, CUDA, or model code, and never runs a GPU.
Non-passed benchmark records are preserved as excluded evidence and are not
included in passed-run aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROMPTS = ("archive", "code")
EXPECTED_TOKENS = (17, 193, 1024, 2049)
EXPECTED_BACKENDS = ("ranked_default", "ranked_selected", "vllm_public")
EXPECTED_RANKED = ("ranked_default", "ranked_selected")
EXPECTED_MODES = ("api", "graph")
EXPECTED_METRIC_GROUPS = (
    "initial_vs_independent_fp32",
    "graph_initial_vs_fp32",
    "graph_rebound_to_other_real_prompt_vs_fp32",
    "graph_restored_vs_fp32",
    "post_api_vs_fp32",
    "post_graph_vs_fp32",
)
EXPECTED_DISPATCH_GROUPS = ("initial_dispatch", "graph_rebound_dispatch", "post_dispatch")
LIMITS = {"relative_rms": 0.008, "normalized_max": 0.008}
M7_POLICY_PATH = "results/m7-policy.json"
M7_POLICY_STATUS = "frozen_before_holdout"
M7_HASH_SCOPE = "canonical_json_without_top_level_policy_sha256"
M20_EXPERIMENT = "M20 isolated checkpoint layer0 real MoE replay"
M20_SELECTION = "frozen M7 shape policy, unchanged; no tuning on real-input cases"
SHAPE = {"experts": 128, "topk": 8, "hidden": 2048, "intermediate": 768}


class AuditError(ValueError):
    """Raised when a saved evidence record fails a required audit."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {label}: {path}: {exc}") from exc
    require(isinstance(value, dict), f"{label} root must be a JSON object: {path}")
    return value


def repo_file(relative: str, label: str) -> Path:
    require(isinstance(relative, str) and relative, f"{label} path must be a non-empty string")
    path = (ROOT / relative).resolve()
    require(path != ROOT and ROOT in path.parents, f"{label} path escapes repository: {relative}")
    require(path.is_file(), f"{label} file is missing: {relative}")
    return path


def repo_artifact_path(relative: str, label: str) -> Path:
    require(isinstance(relative, str) and relative, f"{label} path must be a non-empty string")
    path = (ROOT / relative).resolve()
    require(path != ROOT and ROOT in path.parents, f"{label} path escapes repository: {relative}")
    return path


def safe_external_file(root: Path, value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    require(path != root and root in path.parents, f"{label} escapes capture root: {value}")
    require(path.is_file(), f"{label} file is missing: {path}")
    return path


def finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    require(number > 0 if positive else number >= 0, f"{label} is outside its valid range")
    return number


def close(actual: float, expected: float, label: str) -> None:
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15),
            f"{label} mismatch: recorded={actual:.17g}, recomputed={expected:.17g}")


def p95_linear(values: list[float]) -> float:
    """Match numpy.percentile(values, 95)'s default linear interpolation."""
    require(bool(values), "cannot compute percentile for empty samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    low = math.floor(position)
    high = math.ceil(position)
    fraction = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def _canonical_training_sha(record: dict[str, Any]) -> str:
    normalized = dict(record)
    normalized.pop("policy_sha256", None)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_frozen_policy(record: dict[str, Any]) -> dict[str, Any]:
    policy_path = ROOT / M7_POLICY_PATH
    policy_bytes = policy_path.read_bytes()
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    policy = load_json(policy_path, "frozen M7 policy")
    require(policy.get("schema") == 1 and policy.get("status") == M7_POLICY_STATUS,
            "M7 policy is not frozen_before_holdout")
    require(record.get("policy_sha256") == policy_sha,
            "benchmark policy_sha256 does not match the current M7 policy file")
    require(record.get("selection") == M20_SELECTION,
            "benchmark record does not declare the unchanged frozen M7 policy")
    require(policy.get("domain") == {
        "experts": SHAPE["experts"], "hidden": SHAPE["hidden"],
        "intermediate": SHAPE["intermediate"], "topk": SHAPE["topk"],
    }, "frozen M7 policy domain differs from the M20 layer-0 shape")
    require(policy.get("thresholds_mu") == [8.0, 64.0],
            "frozen M7 policy thresholds differ from the pinned policy contract")
    configs = policy.get("configs")
    require(isinstance(configs, dict) and set(configs) == {"small", "medium", "large"},
            "frozen M7 policy must contain small, medium, and large configurations")
    for name, bucket in policy.get("buckets", {}).items():
        require(name in configs and bucket.get("config") == configs[name],
                f"M7 policy bucket {name!r} differs from its frozen config")

    source_hashes = policy.get("source_hashes")
    require(isinstance(source_hashes, dict) and bool(source_hashes),
            "frozen M7 policy has no source hashes")
    benchmark_sources = record.get("source_hashes")
    require(isinstance(benchmark_sources, dict),
            "benchmark record has no source_hashes map")
    verified_sources: list[str] = []
    for relative, expected in sorted(source_hashes.items()):
        require(is_sha256(expected), f"invalid M7 source SHA256: {relative}")
        require(benchmark_sources.get(relative) == expected,
                f"benchmark source map disagrees with frozen M7 source: {relative}")
        actual = sha256_file(repo_file(relative, "M7 policy source"))
        require(actual == expected, f"frozen M7 source changed: {relative}")
        verified_sources.append(relative)

    training_path = policy.get("training_record_path")
    training_file = repo_file(training_path, "M7 training record")
    training = load_json(training_file, "M7 training record")
    require(training.get("status") == "passed", "M7 training record is not passed")
    require(training.get("run_id") == policy.get("source_run")
            and training.get("source_run") == policy.get("source_run"),
            "M7 training record run identity differs from frozen policy")
    require(training.get("policy_sha256") == policy_sha,
            "M7 training record does not cite the frozen policy file hash")
    require(training.get("source_hashes") == source_hashes,
            "M7 training-record source hashes differ from the frozen policy")
    require(policy.get("training_record_hash_scope") == M7_HASH_SCOPE,
            "unsupported frozen-policy training-record hash scope")
    training_sha = _canonical_training_sha(training)
    require(training_sha == policy.get("training_record_sha256"),
            "M7 canonical training-record hash does not match frozen policy")

    expected_sources = sorted(source_hashes)
    provenance = {}
    for key in ("policy_provenance_before", "policy_provenance_after"):
        value = record.get(key)
        require(isinstance(value, dict) and value.get("passed") is True,
                f"{key} did not pass")
        require(value.get("source_run") == policy.get("source_run"),
                f"{key} references a different M7 training run")
        require(value.get("training_record_path") == training_path,
                f"{key} references a different M7 training record")
        require(value.get("verified_sources") == expected_sources,
                f"{key} verified-source list differs from the frozen policy")
        provenance[key] = value
    require(provenance["policy_provenance_before"] == provenance["policy_provenance_after"],
            "M7 policy provenance changed during benchmark")
    return {
        "path": M7_POLICY_PATH,
        "sha256": policy_sha,
        "status": policy["status"],
        "source_run": policy["source_run"],
        "training_record_path": training_path,
        "training_record_sha256": training_sha,
        "verified_source_count": len(verified_sources),
        "configs": configs,
        "thresholds_mu": policy["thresholds_mu"],
    }


def audit_record_sources(record: dict[str, Any]) -> dict[str, str]:
    before = record.get("source_hashes")
    after = record.get("source_hashes_after")
    require(isinstance(before, dict) and bool(before), "benchmark record has empty source_hashes")
    require(isinstance(after, dict), "passed benchmark record lacks source_hashes_after")
    require(before == after, "benchmark source hashes changed during the run")
    verified: dict[str, str] = {}
    for relative, expected in sorted(before.items()):
        require(is_sha256(expected), f"invalid benchmark source SHA256: {relative}")
        actual = sha256_file(repo_file(relative, "benchmark source"))
        require(actual == expected, f"benchmark source changed after the run: {relative}")
        verified[relative] = actual
    return verified


def expected_case_rows(cases: Any, label: str) -> dict[str, dict[str, Any]]:
    require(isinstance(cases, list) and len(cases) == 8,
            f"{label} must contain exactly eight cases")
    expected = {(prompt, tokens) for prompt in EXPECTED_PROMPTS for tokens in EXPECTED_TOKENS}
    actual: dict[tuple[str, int], dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    pattern = re.compile(r"^(archive|code)-T(17|193|1024|2049)$")
    for row in cases:
        require(isinstance(row, dict), f"{label} case entries must be objects")
        name = row.get("name")
        match = pattern.fullmatch(name) if isinstance(name, str) else None
        require(match is not None, f"{label} has an unexpected case name: {name!r}")
        prompt, token_text = match.groups()
        tokens = int(token_text)
        require(row.get("tokens") == tokens, f"{label} case token count disagrees with its name: {name}")
        key = (prompt, tokens)
        require(key not in actual, f"{label} contains duplicate case: {name}")
        actual[key] = row
        by_name[name] = row
    require(set(actual) == expected, f"{label} does not cover archive/code × 17/193/1024/2049")
    require(len(by_name) == 8, f"{label} case names must be unique")
    return by_name


def audit_metric_groups(case: dict[str, Any], case_name: str) -> dict[str, Any]:
    total = 0
    maximum_rms = 0.0
    maximum_normalized = 0.0
    group_details: dict[str, Any] = {}
    for group_name in EXPECTED_METRIC_GROUPS:
        group = case.get(group_name)
        require(isinstance(group, dict) and set(group) == set(EXPECTED_BACKENDS),
                f"{case_name}.{group_name} must contain all three backends")
        group_details[group_name] = {}
        for backend in EXPECTED_BACKENDS:
            metric = group[backend]
            require(isinstance(metric, dict), f"{case_name}.{group_name}.{backend} must be an object")
            require(metric.get("finite") is True and metric.get("passed") is True,
                    f"{case_name}.{group_name}.{backend} failed finite/numeric checks")
            rms = finite_number(metric.get("relative_rms"), f"{case_name}.{group_name}.{backend}.relative_rms")
            normalized = finite_number(
                metric.get("normalized_max"),
                f"{case_name}.{group_name}.{backend}.normalized_max",
            )
            max_abs = finite_number(metric.get("max_abs"), f"{case_name}.{group_name}.{backend}.max_abs")
            require(rms <= LIMITS["relative_rms"],
                    f"{case_name}.{group_name}.{backend} relative_rms exceeds 0.008")
            require(normalized <= LIMITS["normalized_max"],
                    f"{case_name}.{group_name}.{backend} normalized_max exceeds 0.008")
            maximum_rms = max(maximum_rms, rms)
            maximum_normalized = max(maximum_normalized, normalized)
            group_details[group_name][backend] = {
                "relative_rms": rms,
                "normalized_max": normalized,
                "max_abs": max_abs,
                "finite": True,
                "passed": True,
            }
            total += 1
    require(total == 18, f"{case_name} must contain 18 required numeric metrics")
    return {
        "count": total,
        "maximum_relative_rms": maximum_rms,
        "maximum_normalized_max": maximum_normalized,
        "groups": group_details,
    }


def audit_dispatch(case: dict[str, Any], case_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_name in EXPECTED_DISPATCH_GROUPS:
        group = case.get(group_name)
        require(isinstance(group, dict) and set(group) == set(EXPECTED_RANKED),
                f"{case_name}.{group_name} must contain both ranked dispatch paths")
        result[group_name] = {}
        for backend in EXPECTED_RANKED:
            row = group[backend]
            require(isinstance(row, dict), f"{case_name}.{group_name}.{backend} must be an object")
            require(type(row.get("status")) is int and row["status"] == 0,
                    f"{case_name}.{group_name}.{backend} device status was not zero")
            require(row.get("check_inputs_passed") is True and row.get("passed") is True,
                    f"{case_name}.{group_name}.{backend} dispatch/status check failed")
            ranked = row.get("ranked_dispatch")
            require(isinstance(ranked, dict) and ranked.get("passed") is True,
                    f"{case_name}.{group_name}.{backend} ranked mapping audit failed")
            result[group_name][backend] = {
                "status": row["status"],
                "check_inputs_passed": True,
                "ranked_dispatch_passed": True,
                "passed": True,
            }
    return result


def _expected_measure_orders() -> list[list[str]]:
    orders: list[list[str]] = []
    for base in (list(EXPECTED_BACKENDS), list(reversed(EXPECTED_BACKENDS))):
        for offset in range(len(base)):
            orders.append(base[offset:] + base[:offset])
    return orders


def audit_timing_mode(case: dict[str, Any], case_name: str, mode: str) -> dict[str, Any]:
    measurement = case.get(mode)
    require(isinstance(measurement, dict), f"{case_name}.{mode} timing is missing")
    expected_divisor = 1 if mode == "api" else 8
    require(measurement.get("divisor") == expected_divisor,
            f"{case_name}.{mode} divisor must be {expected_divisor}")
    require(measurement.get("repeats") == 2 and measurement.get("samples_per_round") == 6,
            f"{case_name}.{mode} timing protocol must use 2 repeats and 6 samples per round")
    rounds = measurement.get("rounds")
    require(isinstance(rounds, list) and len(rounds) == 6,
            f"{case_name}.{mode} timing must have six balanced cyclic rounds")
    expected_orders = _expected_measure_orders()
    collected = {backend: [] for backend in EXPECTED_BACKENDS}
    collected_wall = {backend: [] for backend in EXPECTED_BACKENDS}
    for index, row in enumerate(rounds):
        require(isinstance(row, dict), f"{case_name}.{mode} round {index} must be an object")
        require(row.get("order") == expected_orders[index],
                f"{case_name}.{mode} round {index} violates the forward/reverse cyclic order")
        samples = row.get("samples_ms_per_gemm")
        walls = row.get("wall_ms_per_gemm")
        require(isinstance(samples, dict) and set(samples) == set(EXPECTED_BACKENDS),
                f"{case_name}.{mode} round {index} is missing raw event samples")
        require(isinstance(walls, dict) and set(walls) == set(EXPECTED_BACKENDS),
                f"{case_name}.{mode} round {index} is missing raw wall samples")
        for backend in EXPECTED_BACKENDS:
            values = samples[backend]
            wall_values = walls[backend]
            require(isinstance(values, list) and len(values) == 6,
                    f"{case_name}.{mode}.{backend} round {index} must have six event samples")
            require(isinstance(wall_values, list) and len(wall_values) == 6,
                    f"{case_name}.{mode}.{backend} round {index} must have six wall samples")
            collected[backend].extend(
                finite_number(value, f"{case_name}.{mode}.{backend} event sample", positive=True)
                for value in values
            )
            collected_wall[backend].extend(
                finite_number(value, f"{case_name}.{mode}.{backend} wall sample", positive=True)
                for value in wall_values
            )
    require(all(len(values) == 36 for values in collected.values()),
            f"{case_name}.{mode} must preserve 36 event samples per backend")
    require(all(len(values) == 36 for values in collected_wall.values()),
            f"{case_name}.{mode} must preserve 36 wall samples per backend")

    summary = measurement.get("summary")
    require(isinstance(summary, dict) and set(summary) == set(EXPECTED_BACKENDS),
            f"{case_name}.{mode} summary must contain all three backends")
    tokens = case["tokens"]
    flops = 6 * tokens * SHAPE["topk"] * SHAPE["hidden"] * SHAPE["intermediate"]
    recomputed: dict[str, dict[str, float]] = {}
    for backend in EXPECTED_BACKENDS:
        event_values = collected[backend]
        wall_values = collected_wall[backend]
        p50 = median(event_values)
        p95 = p95_linear(event_values)
        wall_p50 = median(wall_values)
        effective_tflops = flops / p50 / 1e9
        recorded = summary[backend]
        require(isinstance(recorded, dict), f"{case_name}.{mode}.{backend} summary must be an object")
        close(finite_number(recorded.get("p50_ms"), "recorded p50", positive=True),
              p50, f"{case_name}.{mode}.{backend}.p50_ms")
        close(finite_number(recorded.get("p95_ms"), "recorded p95", positive=True),
              p95, f"{case_name}.{mode}.{backend}.p95_ms")
        close(finite_number(recorded.get("wall_p50_ms"), "recorded wall p50", positive=True),
              wall_p50, f"{case_name}.{mode}.{backend}.wall_p50_ms")
        close(finite_number(recorded.get("effective_tflops"), "recorded effective TFLOPS", positive=True),
              effective_tflops, f"{case_name}.{mode}.{backend}.effective_tflops")
        recomputed[backend] = {
            "p50_ms": p50,
            "p95_ms": p95,
            "wall_p50_ms": wall_p50,
            "effective_tflops": effective_tflops,
            "event_samples": len(event_values),
            "wall_samples": len(wall_values),
        }
    return {
        "divisor": expected_divisor,
        "repeats": 2,
        "samples_per_round": 6,
        "rounds": 6,
        "raw_samples_per_backend": 36,
        "summary_recomputed": recomputed,
    }


def _audit_compiled_artifacts(case: dict[str, Any], case_name: str) -> list[dict[str, Any]]:
    artifacts = case.get("compile_artifacts")
    require(isinstance(artifacts, dict) and set(artifacts) == set(EXPECTED_RANKED),
            f"{case_name} must record compiled artifacts for both ranked backends")
    rows: list[dict[str, Any]] = []
    for backend in EXPECTED_RANKED:
        kernels = artifacts[backend]
        require(isinstance(kernels, dict) and set(kernels) == {"gate_up_swiglu", "down"},
                f"{case_name}.{backend} must record gate/up and down kernel metadata")
        for kernel_name in ("gate_up_swiglu", "down"):
            item = kernels[kernel_name]
            require(isinstance(item, dict), f"{case_name}.{backend}.{kernel_name} artifact must be an object")
            required = (
                "path", "ptx_sha256", "cubin_sha256", "ptx_mma_instructions",
                "sass_hmma_instructions", "sass_disassembly_returncode",
                "registers", "shared_memory_bytes", "spills",
            )
            for field in required:
                require(field in item, f"{case_name}.{backend}.{kernel_name} lacks {field}")
            require(isinstance(item["path"], str) and item["path"],
                    f"{case_name}.{backend}.{kernel_name} has no artifact path")
            require(is_sha256(item["ptx_sha256"]) and is_sha256(item["cubin_sha256"]),
                    f"{case_name}.{backend}.{kernel_name} has invalid PTX/CUBIN SHA256")
            for field in ("ptx_mma_instructions", "sass_hmma_instructions", "registers",
                          "shared_memory_bytes", "sass_disassembly_returncode"):
                require(type(item[field]) is int and item[field] >= 0,
                        f"{case_name}.{backend}.{kernel_name}.{field} must be a nonnegative integer")
            spills = item["spills"]
            require(spills is None or (type(spills) is int and spills >= 0),
                    f"{case_name}.{backend}.{kernel_name}.spills must be a nonnegative integer or null")

            artifact_dir = repo_artifact_path(item["path"], "compiled artifact directory")
            ptx_path = artifact_dir / "kernel.ptx"
            cubin_path = artifact_dir / "kernel.cubin"
            sass_path = artifact_dir / "kernel.sass"
            artifact_files = (ptx_path, cubin_path, sass_path)
            present = tuple(path.is_file() for path in artifact_files)
            require(not any(path.exists() for path in artifact_files) or all(present),
                    f"{case_name}.{backend}.{kernel_name} compiled files are incomplete")
            if not any(present):
                rows.append({
                    "backend": backend,
                    "kernel": kernel_name,
                    "artifact_path": item["path"],
                    "verification": "recorded_only",
                    "ptx_sha256": item["ptx_sha256"],
                    "cubin_sha256": item["cubin_sha256"],
                    "ptx_mma_instructions": item["ptx_mma_instructions"],
                    "sass_hmma_instructions": item["sass_hmma_instructions"],
                    "sass_disassembly_returncode": item["sass_disassembly_returncode"],
                    "sass_current_sha256": None,
                    "sass_expected_sha256": None,
                    "registers": item["registers"],
                    "shared_memory_bytes": item["shared_memory_bytes"],
                    "spills": spills,
                })
                continue
            ptx_bytes = ptx_path.read_bytes()
            cubin_sha = sha256_file(cubin_path)
            require(hashlib.sha256(ptx_bytes).hexdigest() == item["ptx_sha256"],
                    f"{case_name}.{backend}.{kernel_name} PTX file hash mismatch")
            require(cubin_sha == item["cubin_sha256"],
                    f"{case_name}.{backend}.{kernel_name} CUBIN file hash mismatch")
            ptx_mma = len(re.findall(rb"\bmma\.sync", ptx_bytes))
            require(ptx_mma == item["ptx_mma_instructions"],
                    f"{case_name}.{backend}.{kernel_name} PTX MMA instruction count mismatch")
            sass_bytes = sass_path.read_bytes()
            if item["sass_disassembly_returncode"] == 0:
                sass_hmma = len(re.findall(rb"\bHMMA\.", sass_bytes))
                require(sass_hmma == item["sass_hmma_instructions"],
                        f"{case_name}.{backend}.{kernel_name} SASS HMMA count mismatch")
            rows.append({
                "backend": backend,
                "kernel": kernel_name,
                "artifact_path": item["path"],
                "verification": "verified",
                "ptx_sha256": item["ptx_sha256"],
                "cubin_sha256": item["cubin_sha256"],
                "ptx_mma_instructions": item["ptx_mma_instructions"],
                "sass_hmma_instructions": item["sass_hmma_instructions"],
                "sass_disassembly_returncode": item["sass_disassembly_returncode"],
                "sass_current_sha256": hashlib.sha256(sass_bytes).hexdigest(),
                "sass_expected_sha256": None,
                "registers": item["registers"],
                "shared_memory_bytes": item["shared_memory_bytes"],
                "spills": spills,
            })
    return rows


def policy_selection(policy: dict[str, Any], tokens: int) -> tuple[str, dict[str, Any]]:
    mu = tokens * SHAPE["topk"] / SHAPE["experts"]
    bucket = "small" if mu <= 8.0 else "medium" if mu <= 64.0 else "large"
    return bucket, policy["configs"][bucket]


def audit_route_counts(case: dict[str, Any], case_name: str) -> dict[str, Any]:
    counts = case.get("route_counts")
    require(isinstance(counts, list) and len(counts) == SHAPE["experts"],
            f"{case_name} must record 128 expert route counts")
    require(all(type(value) is int and value >= 0 for value in counts),
            f"{case_name} route counts must be nonnegative integers")
    expected_routes = case["tokens"] * SHAPE["topk"]
    require(sum(counts) == expected_routes,
            f"{case_name} route counts do not sum to T*topk")
    distribution = case.get("routing_distribution")
    require(isinstance(distribution, dict), f"{case_name} routing distribution is missing")
    active = sum(value > 0 for value in counts)
    maximum = max(counts, default=0)
    mean = expected_routes / SHAPE["experts"]
    require(distribution.get("active_experts") == active
            and distribution.get("maximum_routes_per_expert") == maximum,
            f"{case_name} routing distribution disagrees with route counts")
    close(finite_number(distribution.get("mean_routes_per_expert"),
                        f"{case_name}.mean_routes_per_expert"),
          mean, f"{case_name}.mean_routes_per_expert")
    require(is_sha256(case.get("input_sha256")), f"{case_name} input hash is invalid")
    require(case.get("restored_input_sha256") == case.get("input_sha256"),
            f"{case_name} mutable input buffers were not restored")
    require(is_sha256(case.get("trace_manifest_sha256")),
            f"{case_name} trace manifest hash is invalid")
    return {
        "active_experts": active,
        "maximum_routes_per_expert": maximum,
        "mean_routes_per_expert": mean,
        "total_routes": expected_routes,
    }


def _ratio_summary(values: list[float]) -> dict[str, Any]:
    require(bool(values), "cannot summarize an empty ratio list")
    require(all(math.isfinite(value) and value > 0 for value in values),
            "ratios must be finite and positive")
    geometric_mean = math.exp(sum(math.log(value) for value in values) / len(values))
    return {
        "n": len(values),
        "geometric_mean_ratio": geometric_mean,
        "median_ratio": median(values),
        "minimum_ratio": min(values),
        "maximum_ratio": max(values),
        "interpretation": "selected/reference; below 1 means selected was faster",
    }


def ratio_families() -> tuple[str, ...]:
    return tuple(
        f"{mode}.{stat}.{comparison}"
        for mode in EXPECTED_MODES
        for stat in ("p50_ms", "p95_ms")
        for comparison in ("selected_vs_ranked_default", "selected_vs_vllm_public")
    )


def case_ratios(timing: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for mode in EXPECTED_MODES:
        values = timing[mode]["summary_recomputed"]
        for stat in ("p50_ms", "p95_ms"):
            numerator = values["ranked_selected"][stat]
            for reference, key in (
                ("ranked_default", "selected_vs_ranked_default"),
                ("vllm_public", "selected_vs_vllm_public"),
            ):
                result[f"{mode}.{stat}.{key}"] = numerator / values[reference][stat]
    return result


def _audit_case(case: dict[str, Any], name: str, policy: dict[str, Any]) -> dict[str, Any]:
    match = re.fullmatch(r"(archive|code)-T(17|193|1024|2049)", name)
    require(match is not None, f"unexpected M20 case name: {name}")
    prompt, token_text = match.groups()
    tokens = int(token_text)
    require(case.get("name") == name and case.get("tokens") == tokens,
            f"{name} identity/token count mismatch")
    bucket, selected = policy_selection(policy, tokens)
    require(case.get("policy_bucket") == bucket and case.get("selected_config") == selected,
            f"{name} selected configuration differs from frozen M7 policy")
    require(case.get("graph_launches_per_forward") == {
        "ranked_default": 7, "ranked_selected": 7,
    }, f"{name} must retain seven launches for both ranked paths")
    metric_report = audit_metric_groups(case, name)
    require(metric_report["count"] == 18, f"{name} numeric metric count mismatch")
    dispatch_report = audit_dispatch(case, name)
    route_report = audit_route_counts(case, name)
    timing = {
        mode: audit_timing_mode(case, name, mode)
        for mode in EXPECTED_MODES
    }
    compile_rows = _audit_compiled_artifacts(case, name)
    ratios = case_ratios(timing)
    return {
        "name": name,
        "prompt": prompt,
        "tokens": tokens,
        "shape": {**SHAPE, "tokens": tokens},
        "policy_bucket": bucket,
        "selected_config": selected,
        "timing": timing,
        "ratios": ratios,
        "numeric_metrics": {
            "count": metric_report["count"],
            "maximum_relative_rms": metric_report["maximum_relative_rms"],
            "maximum_normalized_max": metric_report["maximum_normalized_max"],
        },
        "dispatch": dispatch_report,
        "launches_per_forward": dict(case["graph_launches_per_forward"]),
        "routing": route_report,
        "compile_artifacts": compile_rows,
    }


def _resolve_manifest_artifact(root: Path, base: Path, value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve()
    require(resolved != root and root in resolved.parents,
            f"{label} resolves outside capture root: {value}")
    require(resolved.is_file(), f"{label} is missing: {resolved}")
    return resolved


def audit_capture(record: dict[str, Any], capture_root: Path) -> dict[str, Any]:
    root = capture_root.expanduser().resolve(strict=True)
    require(root.is_dir(), f"capture root is not a directory: {root}")
    summary_path = safe_external_file(root, "capture-summary.json", "capture summary")
    capture = load_json(summary_path, "external capture summary")
    require(capture.get("status") == "passed",
            "external capture summary is not passed")
    require(capture.get("model", {}).get("repo_id") == "Qwen/Qwen3-30B-A3B"
            and capture.get("model", {}).get("revision") == "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
            "external capture summary is not the pinned Qwen3-30B-A3B layer-0 capture")
    capture_cases = expected_case_rows(capture.get("cases"), "external capture")
    require(capture.get("source_hashes_before") == capture.get("source_hashes_after"),
            "external capture sources changed during capture")
    require(record.get("capture_summary_sha256") == sha256_file(summary_path),
            "benchmark capture-summary SHA256 does not match the selected capture root")

    hash_cache: dict[Path, str] = {}

    def external_sha(path: Path) -> str:
        if path not in hash_cache:
            hash_cache[path] = sha256_file(path)
        return hash_cache[path]

    declared_artifacts = capture.get("artifact_hashes")
    require(isinstance(declared_artifacts, dict) and bool(declared_artifacts),
            "external capture summary has no artifact hashes")
    actual_artifacts: dict[str, dict[str, Any]] = {}
    for relative, details in declared_artifacts.items():
        path = safe_external_file(root, relative, "capture artifact")
        require(isinstance(details, dict) and is_sha256(details.get("sha256")),
                f"invalid capture artifact hash record: {relative}")
        actual_sha = external_sha(path)
        require(actual_sha == details["sha256"],
                f"external capture artifact SHA256 mismatch: {relative}")
        require(details.get("size_bytes") == path.stat().st_size,
                f"external capture artifact size mismatch: {relative}")
        actual_artifacts[path.relative_to(root).as_posix()] = {
            "sha256": actual_sha, "size_bytes": path.stat().st_size,
        }

    expected_relative = {"weights.pt"}
    weights_by_path: dict[Path, str] = {}
    for name, case in capture_cases.items():
        manifest_path = safe_external_file(root, case["manifest"], f"{name} manifest")
        expected_relative.add(manifest_path.relative_to(root).as_posix())
        manifest = load_json(manifest_path, f"{name} trace manifest")
        require(manifest.get("status") == "passed",
                f"{name} trace manifest is not passed")
        model = manifest.get("model", {})
        require(model.get("repo_id") == "Qwen/Qwen3-30B-A3B"
                and model.get("revision") == "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
                and model.get("num_experts") == 128
                and model.get("hidden_size") == 2048
                and model.get("intermediate_size") == 768
                and model.get("top_k") == 8,
                f"{name} trace manifest has an unexpected model contract")
        capture_meta = manifest.get("capture", {})
        require(capture_meta.get("prompt") == case["prompt"]
                and capture_meta.get("tokens") == case["tokens"],
                f"{name} trace manifest capture identity differs from summary")
        trace = manifest.get("trace", {})
        trace_path = _resolve_manifest_artifact(
            root, manifest_path.parent, trace.get("path"), f"{name} trace tensor bundle"
        )
        require(is_sha256(trace.get("sha256")) and external_sha(trace_path) == trace["sha256"],
                f"{name} trace tensor bundle hash mismatch")
        trace_tensors = trace.get("tensors", {})
        expected_trace_shapes = {
            "x": ([case["tokens"], 2048], "bfloat16"),
            "expert_ids": ([case["tokens"], 8], None),
            "route_weights": ([case["tokens"], 8], "float32"),
        }
        require(set(trace_tensors) == set(expected_trace_shapes),
                f"{name} trace manifest has an unexpected tensor list")
        for tensor_name, (shape, dtype) in expected_trace_shapes.items():
            tensor_record = trace_tensors[tensor_name]
            require(tensor_record.get("shape") == shape and is_sha256(tensor_record.get("sha256")),
                    f"{name} trace metadata is invalid for {tensor_name}")
            if dtype is not None:
                require(tensor_record.get("dtype") == dtype,
                        f"{name} trace dtype is invalid for {tensor_name}")
            elif tensor_record.get("dtype") not in ("int32", "int64"):
                raise AuditError(f"{name} expert_ids dtype is not an integer type")
        expected_relative.add(trace_path.relative_to(root).as_posix())

        weights = manifest.get("weights", {})
        require(set(weights) == {"gate_up", "down"},
                f"{name} manifest must reference gate_up and down weights")
        require(weights["gate_up"].get("path") == weights["down"].get("path")
                and weights["gate_up"].get("sha256") == weights["down"].get("sha256"),
                f"{name} gate_up/down do not share one weights bundle")
        for weight_name, shape in (
            ("gate_up", [128, 1536, 2048]),
            ("down", [128, 2048, 768]),
        ):
            ref = weights[weight_name]
            require(ref.get("shape") == shape and ref.get("dtype") == "bfloat16"
                    and is_sha256(ref.get("sha256"))
                    and is_sha256(ref.get("source_manifest_sha256")),
                    f"{name} {weight_name} reference violates the fixed weight contract")
        weight_path = _resolve_manifest_artifact(
            root, manifest_path.parent, weights["gate_up"].get("path"),
            f"{name} shared weights bundle",
        )
        weight_sha = external_sha(weight_path)
        require(weight_sha == weights["gate_up"]["sha256"],
                f"{name} shared weights bundle hash mismatch")
        expected_relative.add(weight_path.relative_to(root).as_posix())
        weights_by_path[weight_path] = weight_sha
        actual_artifacts[manifest_path.relative_to(root).as_posix()] = {
            "sha256": external_sha(manifest_path), "size_bytes": manifest_path.stat().st_size,
        }

    require(set(actual_artifacts) == expected_relative,
            "external capture artifact list does not exactly match weights plus eight manifests/traces")
    capture_before = record.get("artifact_hashes_before")
    capture_after = record.get("artifact_hashes_after")
    require(isinstance(capture_before, dict) and isinstance(capture_after, dict)
            and capture_before == capture_after,
            "benchmark external artifact hashes changed during timing")
    expected_absolute = {
        str(summary_path): external_sha(summary_path),
    }
    for relative in expected_relative:
        path = safe_external_file(root, relative, "expected capture artifact")
        expected_absolute[str(path)] = external_sha(path)
    require(capture_before == expected_absolute,
            "benchmark artifact_hashes do not match the selected capture root")
    return {
        "status": "verified",
        "capture_root": str(root),
        "capture_summary_sha256": external_sha(summary_path),
        "artifact_count_including_summary": len(expected_absolute),
        "trace_case_count": len(capture_cases),
        "weights_bundle_count": len(weights_by_path),
        "artifact_hashes_verified": True,
    }


def audit_passed_record(path: Path, record: dict[str, Any], capture_root: Path | None) -> dict[str, Any]:
    require(record.get("schema") == 1, "unsupported M20 benchmark record schema")
    require(record.get("status") == "passed", "benchmark record status is not passed")
    require(record.get("experiment") == M20_EXPERIMENT,
            "record does not identify the M20 layer-0 replay experiment")
    require(record.get("scope") == "real single-layer GPU-resident fixed-route full forward, not full model or serving",
            "record scope does not match the fixed M20 benchmark contract")
    require(record.get("limits") == LIMITS, "record numerical limits differ from fixed 0.008 limits")
    require(record.get("graph_forwards_per_replay") == 8,
            "Graph replay divisor must correspond to eight forwards")

    source_hashes = audit_record_sources(record)
    policy_info = audit_frozen_policy(record)
    policy = load_json(ROOT / M7_POLICY_PATH, "frozen M7 policy")
    cases_by_name = expected_case_rows(record.get("cases"), f"passed record {path.name}")

    per_case: list[dict[str, Any]] = []
    all_compile: list[dict[str, Any]] = []
    metric_count = 0
    max_rms = 0.0
    max_normalized = 0.0
    for name in (f"{prompt}-T{tokens}" for prompt in EXPECTED_PROMPTS for tokens in EXPECTED_TOKENS):
        case = cases_by_name[name]
        require(case.get("status") == "passed", f"{name} case status is not passed")
        row = _audit_case(case, name, policy)
        metric_count += row["numeric_metrics"]["count"]
        max_rms = max(max_rms, row["numeric_metrics"]["maximum_relative_rms"])
        max_normalized = max(max_normalized, row["numeric_metrics"]["maximum_normalized_max"])
        all_compile.extend(row["compile_artifacts"])
        per_case.append(row)
    require(metric_count == 144, f"passed record must contain 144 passing numeric metrics, found {metric_count}")
    compile_verifications = {row["verification"] for row in all_compile}
    compiled_artifact_verification = (
        "verified" if compile_verifications == {"verified"}
        else "recorded_only" if compile_verifications == {"recorded_only"}
        else "partially_verified"
    )

    artifact_map = record.get("artifact_hashes_before")
    artifact_map_after = record.get("artifact_hashes_after")
    require(isinstance(artifact_map, dict) and bool(artifact_map)
            and artifact_map == artifact_map_after,
            "benchmark capture artifact hashes are missing or changed during timing")
    for artifact_path, digest in artifact_map.items():
        require(isinstance(artifact_path, str) and Path(artifact_path).is_absolute(),
                "benchmark capture artifact paths must be absolute")
        require(is_sha256(digest), f"benchmark capture artifact hash is invalid: {artifact_path}")
    summary_sha = record.get("capture_summary_sha256")
    require(is_sha256(summary_sha), "benchmark capture summary SHA256 is invalid")
    if capture_root is None:
        capture_report = {
            "status": "recorded_only",
            "verified": False,
            "reason": "no --capture-root supplied; external capture files were not opened or rehashed",
            "capture_summary_sha256_recorded": summary_sha,
            "artifact_count_recorded": len(artifact_map),
        }
    else:
        capture_report = audit_capture(record, capture_root)

    ratios = {row["name"]: row["ratios"] for row in per_case}
    return {
        "record": str(path),
        "record_sha256": sha256_file(path),
        "status": "passed",
        "included_in_passed_aggregate": True,
        "policy": policy_info,
        "sources": {
            "status": "verified_current_and_unchanged",
            "source_count": len(source_hashes),
            "source_hashes_before_after_equal": True,
        },
        "capture_artifacts": capture_report,
        "case_count": len(per_case),
        "numeric_metric_count": metric_count,
        "maximum_relative_rms": max_rms,
        "maximum_normalized_max": max_normalized,
        "per_case": per_case,
        "per_case_ratios": ratios,
        "compile_artifacts": all_compile,
        "compiled_artifact_verification": compiled_artifact_verification,
        "ratio_families": list(ratio_families()),
    }


def build_aggregates(valid_runs: list[dict[str, Any]]) -> dict[str, Any]:
    ratio_values = {family: [] for family in ratio_families()}
    by_shape_values: dict[str, dict[str, list[float]]] = {}
    by_prompt_shape_values: dict[str, dict[str, list[float]]] = {}
    compile_rows: list[dict[str, Any]] = []
    timing_count = 0
    for run in valid_runs:
        for case in run["per_case"]:
            timing_count += 1
            shape_key = (
                f"T{case['tokens']}_E{SHAPE['experts']}_topk{SHAPE['topk']}"
                f"_H{SHAPE['hidden']}_I{SHAPE['intermediate']}"
            )
            prompt_shape_key = f"{case['prompt']}-T{case['tokens']}"
            by_shape_values.setdefault(shape_key, {family: [] for family in ratio_families()})
            by_prompt_shape_values.setdefault(
                prompt_shape_key, {family: [] for family in ratio_families()}
            )
            for family, value in case["ratios"].items():
                ratio_values[family].append(value)
                by_shape_values[shape_key][family].append(value)
                by_prompt_shape_values[prompt_shape_key][family].append(value)
            compile_rows.extend(case["compile_artifacts"])

    require(bool(valid_runs), "no fully audited passed benchmark record is available to aggregate")
    resource_values: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in compile_rows:
        resource_values.setdefault(row["backend"], {}).setdefault(row["kernel"], []).append(row)
    resource_summary: dict[str, Any] = {}
    for backend, kernels in resource_values.items():
        resource_summary[backend] = {}
        for kernel, rows in kernels.items():
            spill_values = [row["spills"] for row in rows if row["spills"] is not None]
            resource_summary[backend][kernel] = {
                "records": len(rows),
                "verified_artifact_records": sum(row["verification"] == "verified" for row in rows),
                "recorded_only_artifact_records": sum(
                    row["verification"] == "recorded_only" for row in rows
                ),
                "ptx_mma_instruction_counts": [row["ptx_mma_instructions"] for row in rows],
                "sass_hmma_instruction_counts": [row["sass_hmma_instructions"] for row in rows],
                "register_counts": [row["registers"] for row in rows],
                "shared_memory_bytes": [row["shared_memory_bytes"] for row in rows],
                "spills_recorded": len(spill_values),
                "spill_values": [row["spills"] for row in rows],
                "maximum_spills": max(spill_values) if spill_values else None,
                "sass_hash_recorded_by_benchmark": False,
            }

    return {
        "basis": "all and only fully audited benchmark records with top-level status passed",
        "passed_run_count": len(valid_runs),
        "case_measurement_count": timing_count,
        "shape": dict(SHAPE),
        "ratios_all_cases": {
            family: _ratio_summary(values) for family, values in ratio_values.items()
        },
        "ratios_by_shape": {
            key: {family: _ratio_summary(values) for family, values in families.items()}
            for key, families in sorted(by_shape_values.items())
        },
        "ratios_by_prompt_and_shape": {
            key: {family: _ratio_summary(values) for family, values in families.items()}
            for key, families in sorted(by_prompt_shape_values.items())
        },
        "compiled_kernel_resources": resource_summary,
        "compiled_kernel_resource_record_count": len(compile_rows),
        "interpretation": (
            "ratios are selected/default or selected/vLLM; below 1 favors the frozen selected config. "
            "API and Graph are separate timing modes."
        ),
    }


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        raise AuditError(f"refusing to overwrite existing audit output: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record", type=Path, action="append", required=True,
        help="M20 benchmark JSON to audit; repeat for each run",
    )
    parser.add_argument(
        "--capture-root", type=Path,
        help="optional external capture directory for independent trace/artifact rehashing",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="new audit JSON; existing files are never overwritten",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        print(f"refusing to overwrite existing audit output: {output_path}", file=sys.stderr)
        return 2
    capture_root = args.capture_root.expanduser().resolve() if args.capture_root else None
    report: dict[str, Any] = {
        "schema": 1,
        "audit": "M20 real Qwen3-30B-A3B layer-0 operator evidence",
        "implementation": "Python standard library only; no GPU/runtime/model imports",
        "status": "running",
        "capture_artifact_verification": (
            "verified" if capture_root is not None else "recorded_only"
        ),
        "capture_root": str(capture_root) if capture_root is not None else None,
        "fixed_contract": {
            "model": "Qwen/Qwen3-30B-A3B",
            "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
            "prompt_token_matrix": [
                {"prompt": prompt, "tokens": tokens}
                for prompt in EXPECTED_PROMPTS for tokens in EXPECTED_TOKENS
            ],
            "numeric_limits": LIMITS,
            "metrics_per_passed_run": 144,
            "raw_samples_per_backend_per_mode": 36,
            "api_divisor": 1,
            "graph_forwards_and_divisor": 8,
            "ranked_launches_per_forward": 7,
        },
        "records": [],
        "excluded_nonpassed_records": [],
        "errors": [],
    }
    valid_runs: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for supplied in args.record:
        path = supplied.expanduser()
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if path in seen_paths:
            report["errors"].append(f"duplicate --record input: {path}")
            continue
        seen_paths.add(path)
        try:
            record = load_json(path, "M20 benchmark record")
            record_status = record.get("status")
            if record_status != "passed":
                excluded = {
                    "record": str(path),
                    "record_sha256": sha256_file(path),
                    "record_status": record_status,
                    "included_in_passed_aggregate": False,
                    "reason": "non-passed run records are preserved but excluded from passed aggregates",
                    "failure": record.get("failure", record.get("error")),
                    "case_statuses": [
                        {"name": item.get("name"), "status": item.get("status")}
                        for item in record.get("cases", [])
                        if isinstance(item, dict)
                    ],
                }
                report["excluded_nonpassed_records"].append(excluded)
                report["records"].append(excluded)
                continue
            audit = audit_passed_record(path, record, capture_root)
            report["records"].append(audit)
            valid_runs.append(audit)
        except Exception as exc:
            report["errors"].append({
                "record": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            })
            report["records"].append({
                "record": str(path),
                "record_sha256": sha256_file(path) if path.is_file() else None,
                "record_status": "passed_unverified",
                "included_in_passed_aggregate": False,
                "audit_error": f"{type(exc).__name__}: {exc}",
            })

    report["passed_run_count"] = len(valid_runs)
    report["excluded_nonpassed_run_count"] = len(report["excluded_nonpassed_records"])
    if valid_runs:
        try:
            report["aggregates"] = build_aggregates(valid_runs)
        except Exception as exc:
            report["errors"].append({"aggregate_error": f"{type(exc).__name__}: {exc}"})
    else:
        report["errors"].append("no fully audited passed M20 run; no passed aggregate was produced")

    report["status"] = "passed" if valid_runs and not report["errors"] else "failed"
    try:
        write_new_json(output_path, report)
    except Exception as exc:
        print(f"could not write audit JSON: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": report["status"],
        "output": str(output_path),
        "passed_run_count": report["passed_run_count"],
        "excluded_nonpassed_run_count": report["excluded_nonpassed_run_count"],
        "errors": report["errors"],
    }, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
