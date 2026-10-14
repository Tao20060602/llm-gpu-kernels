#!/usr/bin/env python3
"""Warm up one prebuilt CUDA GQA arm, then profile one synchronized call."""

import argparse
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
M34_BINARY = (
    REPO_ROOT
    / "artifacts/m34-build/llm_gpu_kernels_m34_cuda_gqa_sm86_v1.so"
)
M36_BINARY = (
    REPO_ROOT
    / "artifacts/m36-build/llm_gpu_kernels_m36_compact_gqa_sm86_v1.so"
)
SEED = 436
HISTORY_LENGTHS = (2048, 64, 512)
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
CURRENT_V_TOKEN_STRIDE = 6144
QUERY_TILE = 64
KEY_TILE = 32


def _load_extension(binary: Path):
    """Load an existing pybind extension directly, without a JIT build path."""
    if not binary.is_file():
        raise FileNotFoundError(f"prebuilt CUDA extension is missing: {binary}")
    spec = importlib.util.spec_from_file_location(binary.stem, binary)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for extension: {binary}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(torch, tokens: int):
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    q = torch.randn((tokens, QUERY_HEADS, HEAD_DIM), device=device, dtype=dtype)
    segments = []
    for index, length in enumerate((*HISTORY_LENGTHS, tokens)):
        key = torch.randn((length, KV_HEADS, HEAD_DIM), device=device, dtype=dtype)
        if index == len(HISTORY_LENGTHS):
            value = torch.empty_strided(
                (length, KV_HEADS, HEAD_DIM),
                (CURRENT_V_TOKEN_STRIDE, HEAD_DIM, 1),
                device=device,
                dtype=dtype,
            )
            value.normal_()
        else:
            value = torch.randn((length, KV_HEADS, HEAD_DIM), device=device, dtype=dtype)
        segments.append((key, value))
    output = torch.empty_like(q)
    return q, segments, output


def _shared_declared_bytes(arm: str, stages: int) -> int:
    q_elements = QUERY_TILE * HEAD_DIM
    kv_elements = 2 * KEY_TILE * HEAD_DIM
    if arm.startswith("m34"):
        p_elements = QUERY_TILE * KEY_TILE
    else:
        p_elements = 0
    return (q_elements + stages * kv_elements + p_elements) * 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", required=True, choices=("m34_s2", "m36_s2", "m34_s1")
    )
    parser.add_argument("--tokens", type=int, choices=(32, 96, 256), default=256)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    q, segments, output = _inputs(torch, args.tokens)
    keys = [key for key, _ in segments]
    values = [value for _, value in segments]

    stages = 1 if args.arm == "m34_s1" else 2
    if args.arm == "m36_s2":
        extension = _load_extension(M36_BINARY)
        plan = extension.CompactCudaGqaPlan(q, keys, values, output, stages, 1)
    else:
        extension = _load_extension(M34_BINARY)
        plan = extension.CudaGqaAttentionPlan(q, keys, values, output, stages, 1)

    for _ in range(3):
        plan.run()
        torch.cuda.synchronize(device=0)

    torch.cuda.nvtx.range_push("profile_gqa")
    try:
        plan.run()
        torch.cuda.synchronize(device=0)
    finally:
        torch.cuda.nvtx.range_pop()

    shared_bytes = _shared_declared_bytes(args.arm, stages)
    result = {
        "arm": args.arm,
        "tokens": args.tokens,
        "seed": SEED,
        "history_lengths": list(HISTORY_LENGTHS),
        "current_v_token_stride": CURRENT_V_TOKEN_STRIDE,
        "stages": stages,
        "p_mode": 1,
        "shared_declared_bytes": shared_bytes,
        "shared_declared_bits": shared_bytes * 8,
        "profile_only": True,
        "warmups": 3,
        "profiled_runs": 1,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(output_path)


if __name__ == "__main__":
    main()
