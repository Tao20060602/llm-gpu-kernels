#!/usr/bin/env python3
"""Launch exactly four fixed-shape GEMMs for an external Nsight Compute run."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPE = (4096, 4096)


def _binary_for_arm(arm: str) -> Path:
    if arm == "m25":
        return (REPO_ROOT / "artifacts/m25-build/torch_extensions"
                / "llm_gpu_kernels_m25_bf16_mma_sm86_v1"
                / "llm_gpu_kernels_m25_bf16_mma_sm86_v1.so")
    return REPO_ROOT / "artifacts/m26-build/llm_gpu_kernels_m26_bf16_mma_tiled_sm86_v1.so"


def _load_binary(binary: Path):
    """Load an already-built PyTorch extension without invoking a build path."""
    if not binary.is_file():
        raise FileNotFoundError(f"compiled extension is missing: {binary}")
    spec = importlib.util.spec_from_file_location(binary.stem, binary)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for compiled extension: {binary}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        required=True,
        choices=("m25", "m26_padding", "m26_swizzle"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    torch.manual_seed(426)
    torch.cuda.manual_seed_all(426)

    binary = _binary_for_arm(args.arm)
    extension = _load_binary(binary)
    a = torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(SHAPE, device="cuda", dtype=torch.bfloat16)

    if args.arm == "m25":
        plan = extension.Bf16MmaPlan(a, weight, out, 2)
    else:
        swizzle = args.arm == "m26_swizzle"
        plan = extension.TiledBf16MmaPlan(a, weight, out, 1, 2, swizzle)

    for _ in range(4):
        plan.run()
        torch.cuda.synchronize(device=0)

    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    result = {
        "arm": args.arm,
        "shape": {"M": 4096, "N": 4096, "K": 4096},
        "binary_sha256": digest,
        "runs": 4,
        "profile_only": True,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
