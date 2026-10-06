"""Load the M3 frozen, shape-limited policy; never search during execution."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from .gemm import GemmConfig


@dataclass(frozen=True)
class FrozenPolicy:
    sha256: str
    domain: tuple
    thresholds: tuple
    configs: tuple
    fallback: GemmConfig

    @classmethod
    def load(cls, path):
        content = Path(path).read_bytes()
        data = json.loads(content)
        if data.get("schema") != 1 or data.get("status") != "frozen_before_holdout":
            raise ValueError("expected a frozen M3 policy")
        thresholds = tuple(data["density_thresholds"])
        if len(thresholds) != 2 or not 0 < thresholds[0] < thresholds[1]:
            raise ValueError("invalid density thresholds")
        return cls(hashlib.sha256(content).hexdigest(), (data["domain"]["N"], data["domain"]["K"]),
                   thresholds, tuple(GemmConfig(**data["configs"][name]) for name in ("small", "medium", "large")),
                   GemmConfig(**data["fallback"]))

    def select(self, counts, n, k):
        if (n, k) != self.domain:
            return self.fallback, "shape_fallback"
        active = sum(c > 0 for c in counts)
        density = sum(counts) / active if active else 0
        index = 0 if density <= self.thresholds[0] else 1 if density <= self.thresholds[1] else 2
        return self.configs[index], ("small", "medium", "large")[index]
