"""Prepared wrapper for the SM86 BF16 cache-policy GEMM experiment."""

import os
from pathlib import Path
import re
import shutil
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
_MODULE_NAME = "llm_gpu_kernels_m33_bf16_mma_cache_sm86_v1"
_CACHE_MODES = (0, 1, 2, 3)
_GCC14_VERSION = re.compile(r"^14(?:\.|$)")


def _verified_gcc14(variable):
    """Require the compiler selected by the build environment to be GCC 14."""
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"explicit GCC 14 compiler path required in {variable}")
    compiler = shutil.which(value)
    if compiler is None:
        raise RuntimeError(f"{variable} does not name an executable compiler: {value}")
    try:
        version = subprocess.check_output(
            [compiler, "-dumpfullversion", "-dumpversion"], text=True
        ).strip().splitlines()[0]
        banner = subprocess.check_output([compiler, "--version"], text=True).lower()
    except (OSError, subprocess.CalledProcessError, IndexError) as exc:
        raise RuntimeError(
            f"cannot verify the compiler selected by {variable}: {value}"
        ) from exc
    if not _GCC14_VERSION.match(version) or not any(
        marker in banner for marker in ("gcc", "g++", "gnu")
    ):
        raise RuntimeError(
            f"{variable} must select GNU GCC 14; got {value!r} ({version})"
        )
    return str(Path(compiler).resolve())


def load_cache_policy_bf16_mma():
    """Load the CUDA 12.8 / GCC 14 / SM86 M33 extension on first use."""
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
                raise RuntimeError(
                    "cache-policy BF16 MMA requires the verified CUDA 12.8 toolkit"
                )
            if torch.version.cuda != "12.8":
                raise RuntimeError(
                    f"PyTorch CUDA runtime must be 12.8; got {torch.version.cuda!r}"
                )

            # Match scripts/activate-cutlass.sh and reject a compiler that
            # could silently select system GCC 15 for this extension.
            _verified_gcc14("CC")
            _verified_gcc14("CXX")

            from torch.utils import cpp_extension

            if (cpp_extension.CUDA_HOME is None
                    or Path(cpp_extension.CUDA_HOME).resolve() != home):
                raise RuntimeError("cpp_extension CUDA_HOME differs from explicit CUDA_HOME")
            build_directory = Path(__file__).resolve().parents[2] / "artifacts" / "m33-build"
            build_directory.mkdir(parents=True, exist_ok=True)
            source = Path(__file__).with_name("cuda_bf16_mma_cache_policy.cu")
            _EXTENSION = cpp_extension.load(
                name=_MODULE_NAME,
                sources=[str(source)],
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


def prepare_cache_policy_bf16_mma(a, w, out=None, *, cache_mode=0):
    """Prepare ``A[M,K] @ W[N,K].T`` into a bound BF16 ``out[M,N]``.

    M33 fixes the M32 group-16 output traversal, CTA64x128x32, two stages,
    and group-2 B-register prefetch. ``cache_mode`` selects ``ca``, ``cg``,
    ``ca`` with an L2 128-byte hint, or ``cg`` with that hint. Buffers remain
    bound and may only be updated in place; callers establish cross-stream
    dependencies and avoid overlapping writes to the same output.
    """
    if type(cache_mode) is not int or cache_mode not in _CACHE_MODES:
        raise ValueError("cache_mode must be 0, 1, 2, or 3")

    _validate_bf16_cuda_tensor(a, "a")
    _validate_bf16_cuda_tensor(w, "weight", a.device)
    if a.ndim != 2 or w.ndim != 2:
        raise ValueError("a and weight must be rank-2 matrices")

    m, k = a.shape
    n, weight_k = w.shape
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
    if _storage_identity(out) in (_storage_identity(a), _storage_identity(w)):
        raise ValueError("out must not share storage with a or weight")
    if torch.cuda.get_device_capability(a.device) != (8, 6):
        raise RuntimeError("cache-policy BF16 MMA requires an SM86 CUDA device")

    with torch.cuda.device(a.device):
        module = load_cache_policy_bf16_mma()
        native = module.CachePolicyBf16MmaPlan(a, w, out, cache_mode)
    return PreparedBf16Mma(owners=(a, w, out), native=native)
