"""Prepared wrapper for the experimental tiled SM86 BF16 GEMM candidate."""

import os
from pathlib import Path
import subprocess
import threading

import torch

from .gemm_mma import (
    PreparedBf16Mma,
    _storage_identity,
    _validate_bf16_cuda_tensor,
)


_EXTENSION = None
_LOCK = threading.Lock()
_MODULE_NAME = "llm_gpu_kernels_m26_bf16_mma_tiled_sm86_v1"
_TILE_IDS = {(64, 64): 0, (64, 128): 1, (128, 128): 2}


def load_tiled_bf16_mma():
    """Load the separate CUDA 12.8 / SM86 M26 extension on first use."""
    global _EXTENSION
    with _LOCK:
        if _EXTENSION is None:
            value = os.environ.get("CUDA_HOME")
            if not value:
                raise RuntimeError("explicit CUDA 12.8 toolkit path required in CUDA_HOME")
            home = Path(value).resolve()
            nvcc = home / "bin/nvcc"
            if not nvcc.is_file():
                raise RuntimeError(f"CUDA_HOME has no nvcc executable: {nvcc}")
            version = subprocess.check_output([str(nvcc), "--version"], text=True)
            if "release 12.8," not in version:
                raise RuntimeError("tiled BF16 MMA extension requires the verified CUDA 12.8 toolkit")
            if torch.version.cuda != "12.8":
                raise RuntimeError(
                    f"PyTorch CUDA runtime must be 12.8; got {torch.version.cuda!r}"
                )

            from torch.utils import cpp_extension

            if (cpp_extension.CUDA_HOME is None
                    or Path(cpp_extension.CUDA_HOME).resolve() != home):
                raise RuntimeError("cpp_extension CUDA_HOME differs from explicit CUDA_HOME")
            build_directory = Path(__file__).resolve().parents[2] / "artifacts" / "m26-build"
            build_directory.mkdir(parents=True, exist_ok=True)
            _EXTENSION = cpp_extension.load(
                name=_MODULE_NAME,
                sources=[str(Path(__file__).with_name("cuda_bf16_mma_tiled.cu"))],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=[
                    "-O3", "-std=c++17", "-lineinfo",
                    "-gencode=arch=compute_86,code=sm_86",
                    "--ptxas-options=-v",
                ],
                build_directory=str(build_directory),
                with_cuda=True,
                verbose=True,
            )
        return _EXTENSION


def prepare_tiled_bf16_mma(
    a,
    weight,
    out=None,
    *,
    tile=(64, 64),
    stages=2,
    layout="padding",
):
    """Prepare experimental ``A[M,K] @ weight[N,K].T`` into BF16 ``out``.

    The supported CTA tiles are (64, 64), (64, 128), and (128, 128); stages
    are 1 through 4, and layout is ``"padding"`` or ``"swizzle"``. These M26
    tile/layout candidates passed the limited M26 diagnostic checks; no
    configuration is claimed to be generally optimal. See the M26 report.

    The returned plan binds fixed shapes and storage addresses. Buffer values
    may be updated in place between runs, but the buffers must not be resized
    or replaced. Callers establish cross-stream dependencies and avoid
    overlapping writes to the same output.
    """
    if (type(tile) is not tuple or len(tile) != 2
            or any(type(size) is not int for size in tile)):
        raise ValueError("tile must be one of (64, 64), (64, 128), or (128, 128)")
    try:
        tile_id = _TILE_IDS[tile]
    except KeyError as exc:
        raise ValueError("tile must be one of (64, 64), (64, 128), or (128, 128)") from exc
    if type(stages) is not int or stages not in (1, 2, 3, 4):
        raise ValueError("stages must be an integer from 1 through 4")
    if layout not in ("padding", "swizzle"):
        raise ValueError("layout must be 'padding' or 'swizzle'")
    swizzle = layout == "swizzle"

    _validate_bf16_cuda_tensor(a, "a")
    _validate_bf16_cuda_tensor(weight, "weight", a.device)
    if a.ndim != 2 or weight.ndim != 2:
        raise ValueError("a and weight must be rank-2 matrices")

    m, k = a.shape
    n, weight_k = weight.shape
    if k != weight_k:
        raise ValueError("a and weight must have the same K dimension")
    if k < 16 or k % 16:
        raise ValueError("K must be at least 16 and divisible by 16")
    if n <= 0:
        raise ValueError("N must be positive")

    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    _validate_bf16_cuda_tensor(out, "out", a.device)
    if out.ndim != 2 or tuple(out.shape) != (m, n):
        raise ValueError(f"out must have shape ({m}, {n})")
    if _storage_identity(out) in (_storage_identity(a), _storage_identity(weight)):
        raise ValueError("out must not share storage with a or weight")
    if torch.cuda.get_device_capability(a.device) != (8, 6):
        raise RuntimeError("tiled BF16 MMA requires an SM86 CUDA device")

    with torch.cuda.device(a.device):
        module = load_tiled_bf16_mma()
        native = module.TiledBf16MmaPlan(a, weight, out, tile_id, stages, swizzle)
    return PreparedBf16Mma(owners=(a, weight, out), native=native)
