#!/usr/bin/env python3
"""Launch three warmups and one NVTX-scoped M29 GEMM profile call."""

import argparse
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPE = (4096, 4096)
M27_BINARY = (
    REPO_ROOT
    / "artifacts/m27-build/llm_gpu_kernels_m27_bf16_mma_scheduled_sm86_v1.so"
)


def _load_m27(binary: Path):
    """Load the precompiled M27 extension without invoking a build path."""
    if not binary.is_file():
        raise FileNotFoundError(f"compiled M27 extension is missing: {binary}")
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
        choices=("torch_default", "torch_strict", "m27"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    torch.manual_seed(429)
    torch.cuda.manual_seed_all(429)

    a = torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(SHAPE, device="cuda", dtype=torch.bfloat16)
    allow_reduced_precision_reduction = None

    if args.arm == "m27":
        extension = _load_m27(M27_BINARY)
        plan = extension.ScheduledBf16MmaPlan(a, weight, out, 2, 2, True)

        def call() -> None:
            plan.run()

    else:
        allow_reduced_precision_reduction = args.arm == "torch_default"
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            allow_reduced_precision_reduction
        )

        def call() -> None:
            torch.mm(a, weight.T, out=out)

    for _ in range(3):
        call()
        torch.cuda.synchronize(device=0)

    torch.cuda.nvtx.range_push("profile_gemm")
    try:
        call()
        torch.cuda.synchronize(device=0)
    finally:
        torch.cuda.nvtx.range_pop()

    result = {
        "arm": args.arm,
        "baseline": args.arm.startswith("torch_"),
        "allow_bf16_reduced_precision_reduction": allow_reduced_precision_reduction,
        "accumulator": "native_fp32" if args.arm == "m27" else None,
        "profile_only": True,
        "runs": 4,
        "profiled_run": 4,
        "shape": {"M": 4096, "N": 4096, "K": 4096},
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
