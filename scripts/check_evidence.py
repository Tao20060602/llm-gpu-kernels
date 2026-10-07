#!/usr/bin/env python3
"""Read-only audit of committed evidence. Standard library; no CUDA imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
LIMIT = 0.008


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Evidence:
    def __init__(self, root):
        self.root = root.resolve()
        self.files = {}

    def path(self, name):
        path = (self.root / name).resolve()
        require(path.is_relative_to(self.root), f"path outside checkout: {name}")
        return path

    def sha(self, name, expected=None):
        payload = self.path(name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected is not None:
            require(digest == expected, f"hash mismatch: {name}")
        self.files[name] = {"sha256": digest, "bytes": len(payload)}
        return digest

    def read(self, name, expected=None):
        self.sha(name, expected)
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def sources(self, record):
        hashes = record["source_hashes"]
        require(bool(hashes), "empty source hash map")
        if "source_hashes_after" in record:
            require(hashes == record["source_hashes_after"], "source changed during measurement")
        for name, digest in hashes.items():
            self.sha(name, digest)
        return len(hashes)

    def policy(self, name):
        data = self.read(name)
        require(data["status"] == "frozen_before_holdout", f"policy not frozen: {name}")
        scope = data.get("training_hash_scope", data.get("training_record_hash_scope"))
        require(scope == "canonical_json_without_top_level_policy_sha256", "unsupported training hash scope")
        self.sources(data)
        training = self.read(data["training_record_path"])
        require(training["status"] == "passed", "training record not passed")
        require(training["policy_sha256"] == self.sha(name), "training policy hash mismatch")
        copied = dict(training)
        copied.pop("policy_sha256", None)
        payload = json.dumps(copied, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        require(hashlib.sha256(payload).hexdigest() == data["training_record_sha256"],
                "canonical training hash mismatch")
        return data


def metrics(value):
    found = []
    if isinstance(value, dict):
        if {"relative_rms", "normalized_max", "finite", "passed"} <= value.keys():
            require(value["finite"] is True and value["passed"] is True, "failed numeric metric")
            for key in ("relative_rms", "normalized_max"):
                require(math.isfinite(value[key]) and 0 <= value[key] <= LIMIT,
                        f"numeric metric outside fixed limit: {key}")
            found.append(value)
        for child in value.values():
            found.extend(metrics(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(metrics(child))
    return found


def metric_summary(rows):
    require(bool(rows), "no numeric metrics found")
    return {"count": len(rows), "maximum_relative_rms": max(r["relative_rms"] for r in rows),
            "maximum_normalized_max": max(r["normalized_max"] for r in rows)}


def check_operators(e):
    e.policy("results/m7-policy.json")
    e.policy("results/m16-resource-policy.json")
    m7 = e.read("results/m7-full-moe-holdout-20261006T104838Z-727559245.json")
    require(m7["status"] == "passed", "M7 holdout failed")
    e.sources(m7)
    e.sha(m7["policy"]["path"], m7["policy"]["sha256"])
    e.sha("src/llm_gpu_kernels/moe_full_holdout.py", m7["holdout_driver_sha256"])
    compile_record = e.read("results/m7-compile-evidence-20261006T105911Z.json")
    require(compile_record["status"] == "passed", "M7 compile evidence failed")
    require(compile_record["policy_sha256"] == e.sha("results/m7-policy.json"), "M7 compile policy mismatch")
    m16 = e.read("results/m16-summary.json")
    for name, digest in m16["source_hashes"].items():
        record = e.read(name, digest)
        require(record["status"] == "passed", f"M16 record failed: {name}")
        e.sources(record)
    m17 = e.read("results/m17-summary.json")
    replay = e.read(m17["derived_from"], m17["result_sha256"])
    require(replay["status"] == "passed", "M17 replay failed")
    e.sources(replay)
    m17_metrics = metric_summary(metrics(replay))
    require(m17_metrics["count"] == m17["metric_count"], "M17 metric count mismatch")
    m18 = e.read("results/m18-summary.json")
    require(m18["status"] == "passed", "M18 summary not passed")
    main = e.read(m18["record_path"], m18["record_sha256"])
    audit = e.read(m18["audit_path"], m18["audit_sha256"])
    for record in (main, audit):
        require(record["status"] == "passed", "M18 record failed")
        e.sources(record)
        require(record["policy_sha256"] == e.sha("results/m16-resource-policy.json"), "M18 policy mismatch")
    m18_metrics = metric_summary(metrics(main) + metrics(audit))
    require(m18_metrics["count"] == m18["metric_count"], "M18 metric count mismatch")
    require(m18_metrics["maximum_relative_rms"] == m18["max_relative_rms"] and
            m18_metrics["maximum_normalized_max"] == m18["max_normalized_max"], "M18 metric maxima mismatch")
    direct = [c for c in main["cases"] if c["mode"] == "direct"]
    api_ratios = [c["api"]["summary"]["cached_dynamic"]["event_p50_ms_per_forward"] /
                  c["api"]["summary"]["strong_pack_fa2"]["event_p50_ms_per_forward"] for c in direct]
    require(len(api_ratios) == m18["ratios"]["direct_api_vs_strong_pack_fa2"]["cases"], "M18 direct case count mismatch")
    gm = math.exp(sum(map(math.log, api_ratios)) / len(api_ratios))
    require(math.isclose(gm, m18["ratios"]["direct_api_vs_strong_pack_fa2"]["geomean_ratio"], rel_tol=1e-12),
            "M18 API ratio mismatch")
    return {"m7_metrics": metric_summary(metrics(m7)), "m17_metrics": m17_metrics,
            "m18_metrics": m18_metrics, "m18_direct_api_ratio_recomputed": gm}


def check_native(operators, native):
    prefix = "benchmarks/results/operator_bridge/m19-native/"
    summary = native.read(prefix + "summary.json")
    operators.sha("results/m19-native-integration-summary.json", native.sha(prefix + "summary.json"))
    records = {Path(row["tracked"]).name: native.read(row["tracked"], row["sha256"])
               for row in summary["tracked_artifacts"]}
    suite = records["suite.json"]
    require(suite["status"] == "passed", "M19 suite not passed")
    native.sources(suite)
    require(suite["summary"] == summary["paired_measurement"], "M19 copied summary mismatch")
    manifest = "benchmarks/results/operator_bridge/real-model-capture-manifest.json"
    native.sha(manifest, suite["manifest_sha256"])
    native.sha(prefix + "operator-audit-recovered.json", suite["audit_sha256"])
    all_metrics, states = [], 0
    for name in ("operator-audit-recovered.json", "reuse-audit-recovered.json"):
        record = records[name]
        require(record["status"] == "passed" and record["phase"] == "audit", "M19 audit failed")
        native.sources(record)
        require(record["manifest_sha256"] == suite["manifest_sha256"], "audit manifest mismatch")
        all_metrics.extend(metrics(record))
        for request in [record["warmup"], *record["requests"]]:
            for layer in request["layer_audit"]:
                require(layer["state_checked"] is True, "state audit failed")
                states += 1
    numeric = metric_summary(all_metrics)
    require(numeric["count"] == summary["audited_numeric_metrics"] and
            states == summary["audited_later_layer_states"], "M19 audit counts mismatch")
    require(numeric["maximum_relative_rms"] == summary["maximum_relative_rms"] and
            numeric["maximum_normalized_max"] == summary["maximum_normalized_max"], "M19 numeric maxima mismatch")
    require(len(suite["jobs"]) == 9, "M19 needs nine recorded workers")
    for job in suite["jobs"]:
        name = f"group{job['group']}-{job['backend']}.json"
        native.sha(prefix + name, job["result_sha256"])
        worker = records[name]
        require(worker["status"] == "passed" and worker["phase"] == "perf" and
                worker["backend"] == job["backend"], "worker identity/status mismatch")
        native.sources(worker)
        require(worker["source_hashes"] == records["operator-audit-recovered.json"]["source_hashes"],
                "worker source differs from audit")
    fields = {"operator_vs_reuse_prefill": ("operator", "flash_reuse", "prefill_wall_ms"),
              "operator_vs_reuse_tail": ("operator", "flash_reuse", "tail_chunk_ms"),
              "operator_vs_flash_prefill": ("operator", "flash", "prefill_wall_ms"),
              "reuse_vs_flash_prefill": ("flash_reuse", "flash", "prefill_wall_ms")}
    ratios = {key: [] for key in fields}
    ordered = sets = 0
    for group in range(3):
        arms = {backend: records[f"group{group}-{backend}.json"] for backend in ("flash", "flash_reuse", "operator")}
        for prompt in ("archive", "code"):
            rows = {b: next(r for r in d["requests"] if r["name"] == prompt) for b, d in arms.items()}
            require(rows["flash"]["generated_ids"] == rows["flash_reuse"]["generated_ids"] ==
                    rows["operator"]["generated_ids"], "greedy IDs differ")
            require(rows["flash"]["states"] == rows["flash_reuse"]["states"], "FA2 arm states differ")
            for key, (num, den, timing) in fields.items():
                ratios[key].append(rows[num][timing] / rows[den][timing])
            actual, ref = rows["operator"]["states"], rows["flash_reuse"]["states"]
            require(len(actual) == len(ref), "chunk count differs")
            for actual_chunk, ref_chunk in zip(actual, ref):
                require(len(actual_chunk) == len(ref_chunk), "layer count differs")
                for a, r in zip(actual_chunk, ref_chunk):
                    require(all(a[k] == r[k] for k in ("layer_id", "valid_len", "prefill_len", "blocks", "protected")),
                            "logical states differ")
                    if r["selected_ids"] is not None:
                        ordered += a["selected_ids"] != r["selected_ids"]
                        sets += set(a["selected_ids"]) != set(r["selected_ids"])
    for key, values in ratios.items():
        recomputed = {"geomean": math.exp(sum(map(math.log, values)) / len(values)),
                      "minimum": min(values), "maximum": max(values), "n": len(values)}
        recorded = suite["summary"]["ratios"][key]
        require(recomputed["n"] == recorded["n"] and
                all(math.isclose(recomputed[k], recorded[k], rel_tol=1e-12)
                    for k in ("geomean", "minimum", "maximum")), f"M19 ratio mismatch: {key}")
    require(ordered == summary["paired_measurement"]["ordered_selection_differences"] and
            sets == summary["paired_measurement"]["selection_set_differences"], "selection difference counts mismatch")
    require(summary["promotion_accepted"] is False and summary["default_backend"] == "flash",
            "unexpected promotion claim")
    model = suite["model_audit"]
    require(model["status"] == "passed" and len(model["files"]) == summary["model_files_verified_before_suite"]
            and all(row["matches_manifest"] is True for row in model["files"]), "historical model audit mismatch")
    return {"numeric": numeric, "state_checks": states, "workers": len(suite["jobs"]),
            "ordered_selection_differences_recomputed": ordered, "set_differences_recomputed": sets,
            "ratios_recomputed": suite["summary"]["ratios"],
            "model_files_historically_verified": len(suite["model_audit"]["files"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nanokv-root", type=Path, help="optional native checkout for M19 cross-repository audit")
    args = parser.parse_args()
    report = {"schema": 1, "scope": "committed evidence integrity; no GPU/runtime/model-file validation",
              "model_weights_and_capture_tensors_required": False, "fixed_numeric_limit": LIMIT}
    operator = Evidence(ROOT)
    operator.sha("scripts/check_evidence.py")
    native = Evidence(args.nanokv_root) if args.nanokv_root else None
    try:
        report["operators"] = check_operators(operator)
        if native:
            report["native"] = check_native(operator, native)
        report["status"] = "passed_combined" if native else "passed_operator_only"
        report["native_checked"] = native is not None
    except (OSError, ValueError, KeyError, TypeError, StopIteration, ZeroDivisionError) as exc:
        report.update(status="failed", failure=str(exc))
    report["operator_files"] = operator.files
    if native:
        report["native_files"] = native.files
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
