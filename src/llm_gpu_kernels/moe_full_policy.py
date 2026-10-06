"""Load the frozen, shape-only M7 full-MoE tile policy.

This policy never reads route IDs or expert counts. Its selection key is only
the prepared tensor shape, so it remains safe for fixed-address Graph replay.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .gemm import GemmConfig


POLICY_SCHEMA = 1
POLICY_STATUS = "frozen_before_holdout"
BUCKET_NAMES = ("small", "medium", "large")
TRAINING_RECORD_HASH_SCOPE = (
    "canonical_json_without_top_level_policy_sha256"
)
EXPECTED_DOMAIN = {
    "experts": 128,
    "hidden": 2048,
    "intermediate": 768,
    "topk": 8,
}
EXPECTED_THRESHOLDS = (8.0, 64.0)
BASELINE_CONFIG = GemmConfig()


def canonical_training_record_sha256(record: dict) -> str:
    """Hash a training record while excluding its policy-hash back-reference.

    The policy binds the final training report, while the report also records
    the policy file hash. Excluding only that top-level field avoids a circular
    hash without excluding the scan results, provenance, or frozen decisions.
    """
    normalized = dict(record)
    normalized.pop("policy_sha256", None)
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FrozenMoEPolicy:
    """M7 policy with its exact file hash and limited supported domain."""

    sha256: str
    thresholds: tuple[float, float]
    domain: tuple[int, int, int, int]
    configs: tuple[GemmConfig, GemmConfig, GemmConfig]
    fallback: GemmConfig
    source_run: str
    training_record_path: str
    training_record_sha256: str
    source_hashes: tuple[tuple[str, str], ...]

    @classmethod
    def load(cls, path):
        content = Path(path).read_bytes()
        data = json.loads(content)
        if data.get("schema") != POLICY_SCHEMA or data.get("status") != POLICY_STATUS:
            raise ValueError("expected a frozen M7 full-MoE policy")

        domain_data = data.get("domain")
        if domain_data != EXPECTED_DOMAIN:
            raise ValueError(f"unsupported M7 policy domain: {domain_data!r}")

        thresholds = tuple(float(value) for value in data.get("thresholds_mu", ()))
        if thresholds != EXPECTED_THRESHOLDS:
            raise ValueError(f"unsupported M7 density thresholds: {thresholds!r}")

        config_data = data.get("configs")
        if not isinstance(config_data, dict) or set(config_data) != set(BUCKET_NAMES):
            raise ValueError("policy must define small, medium, and large configs")
        configs = tuple(GemmConfig(**config_data[name]) for name in BUCKET_NAMES)
        fallback_data = data.get("fallback")
        fallback = GemmConfig(**fallback_data)

        bucket_data = data.get("buckets")
        if not isinstance(bucket_data, dict) or set(bucket_data) != set(BUCKET_NAMES):
            raise ValueError("policy must define small, medium, and large bucket metadata")
        for name, config in zip(BUCKET_NAMES, configs):
            recorded = bucket_data[name].get("config")
            if recorded != config_data[name]:
                raise ValueError(f"bucket {name!r} config does not match policy configs")

        source_hashes_data = data.get("source_hashes")
        if not isinstance(source_hashes_data, dict) or not source_hashes_data:
            raise ValueError("policy is missing explicit source hashes")
        source_hashes = tuple(sorted(
            (str(relative), str(digest))
            for relative, digest in source_hashes_data.items()
        ))
        for relative, digest in source_hashes:
            if len(digest) != 64:
                raise ValueError(f"invalid source digest for {relative!r}")

        source_run = data.get("source_run")
        record_path = data.get("training_record_path")
        record_sha256 = data.get("training_record_sha256")
        if not source_run or not record_path or not record_sha256:
            raise ValueError("policy is missing training-run provenance")
        if data.get("training_record_hash_scope") != TRAINING_RECORD_HASH_SCOPE:
            raise ValueError("unsupported training-record hash scope")
        if len(str(record_sha256)) != 64:
            raise ValueError("invalid training-record digest")

        protocol = data.get("training_protocol", {})
        if (protocol.get("seed") != 409
                or protocol.get("tokens") != [64, 512, 4096]
                or protocol.get("routes") != ["balanced", "hot16"]):
            raise ValueError("policy training protocol does not match the M7 design")

        return cls(
            sha256=hashlib.sha256(content).hexdigest(),
            thresholds=(thresholds[0], thresholds[1]),
            domain=(domain_data["experts"], domain_data["hidden"],
                    domain_data["intermediate"], domain_data["topk"]),
            configs=(configs[0], configs[1], configs[2]),
            fallback=fallback,
            source_run=str(source_run),
            training_record_path=str(record_path),
            training_record_sha256=str(record_sha256),
            source_hashes=source_hashes,
        )

    def select(self, T, E, topk, H, I):
        """Return ``(GemmConfig, bucket)`` using dimensions only.

        Inputs outside M7's exact E/H/I/top-k domain receive the documented
        baseline fallback. T may be zero; its density estimate selects small.
        """
        tokens = int(T)
        experts = int(E)
        routed_topk = int(topk)
        hidden = int(H)
        intermediate = int(I)
        if tokens < 0:
            raise ValueError("T must be nonnegative")
        expected_e, expected_h, expected_i, expected_topk = self.domain
        if (experts, hidden, intermediate, routed_topk) != (
                expected_e, expected_h, expected_i, expected_topk):
            return self.fallback, "shape_fallback"

        mu = tokens * routed_topk / experts
        index = 0 if mu <= self.thresholds[0] else 1 if mu <= self.thresholds[1] else 2
        return self.configs[index], BUCKET_NAMES[index]

    def verify_provenance(self, repository_root):
        """Verify the frozen report and every explicitly listed source file."""
        root = Path(repository_root).resolve()
        record_path = (root / self.training_record_path).resolve()
        if root not in record_path.parents:
            raise ValueError("training record path escapes repository root")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "passed":
            raise ValueError("M7 training record is not passed")
        if record.get("run_id") != self.source_run:
            raise ValueError("M7 training record run_id does not match policy")
        if record.get("source_run") != self.source_run:
            raise ValueError("M7 training record source_run does not match policy")
        if record.get("policy_sha256") != self.sha256:
            raise ValueError("M7 training record policy hash does not match policy")
        if record.get("source_hashes") != dict(self.source_hashes):
            raise ValueError("M7 source hashes differ between policy and training record")
        if canonical_training_record_sha256(record) != self.training_record_sha256:
            raise ValueError("M7 training record canonical hash does not match policy")

        protocol = record.get("training_protocol", {})
        if (protocol.get("seed") != 409
                or protocol.get("tokens") != [64, 512, 4096]
                or protocol.get("routes") != ["balanced", "hot16"]):
            raise ValueError("M7 training record protocol does not match frozen design")

        verified = []
        for relative, expected in self.source_hashes:
            source = (root / relative).resolve()
            if root not in source.parents:
                raise ValueError(f"source path escapes repository root: {relative}")
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"M7 source changed after freeze: {relative}")
            verified.append(relative)
        return {
            "passed": True,
            "source_run": self.source_run,
            "training_record_path": self.training_record_path,
            "verified_sources": verified,
        }
