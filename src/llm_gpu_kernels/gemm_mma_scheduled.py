"""Prepared wrapper for the scheduled SM86 BF16 Tensor Core GEMM candidate."""

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
_MODULE_NAME = "llm_gpu_kernels_m27_bf16_mma_scheduled_sm86_v1"
_GROUPS = (1, 2, 4, 8)
_STAGES = (2, 3)
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
        raise RuntimeError(f"cannot verify the compiler selected by {variable}: {value}") from exc
    if not _GCC14_VERSION.match(version) or not any(
            marker in banner for marker in ("gcc", "g++", "gnu")):
        raise RuntimeError(f"{variable} must select GNU GCC 14; got {value!r} ({version})")
    return str(Path(compiler).resolve())


def load_scheduled_bf16_mma():
    """Load the independent CUDA 12.8 / GCC 14 / SM86 M27 extension."""
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
                    "scheduled BF16 MMA extension requires the verified CUDA 12.8 toolkit"
                )
            if torch.version.cuda != "12.8":
                raise RuntimeError(
                    f"PyTorch CUDA runtime must be 12.8; got {torch.version.cuda!r}"
                )

            # Match scripts/activate-cutlass.sh and reject an ambient compiler
            # that could silently change the extension's build environment.
            _verified_gcc14("CC")
            _verified_gcc14("CXX")

            from torch.utils import cpp_extension

            if (cpp_extension.CUDA_HOME is None
                    or Path(cpp_extension.CUDA_HOME).resolve() != home):
                raise RuntimeError("cpp_extension CUDA_HOME differs from explicit CUDA_HOME")
            build_directory = Path(__file__).resolve().parents[2] / "artifacts" / "m27-build"
            build_directory.mkdir(parents=True, exist_ok=True)

            source = Path(__file__).with_name("cuda_bf16_mma_scheduled.cu")

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


def prepare_scheduled_bf16_mma(
    a,
    w,
    out=None,
    *,
    group=2,
    stages=2,
    layout="swizzle",
):
    """Prepare ``A[M,K] @ weight[N,K].T`` into BF16 ``out[M,N]``.

    M27 uses a fixed 64x128x32 CTA with four warps. ``group`` selects how many
    N8 B fragments are loaded before the two M16-by-group MMA updates; valid
    groups are 1, 2, 4, and 8. Stages 2 and 3 and the padding/swizzle shared
    layouts are supported. Group 8 retains the M26 reference instruction order.

    The returned plan binds fixed shapes and storage addresses. Buffer values
    may be updated in place, but buffers must not be resized or replaced.
    Callers establish cross-stream dependencies and avoid overlapping writes
    to the same output.
    """
    if type(group) is not int or group not in _GROUPS:
        raise ValueError("group must be one of 1, 2, 4, or 8")
    if type(stages) is not int or stages not in _STAGES:
        raise ValueError("stages must be 2 or 3")
    if layout not in ("padding", "swizzle"):
        raise ValueError("layout must be 'padding' or 'swizzle'")
    swizzle = layout == "swizzle"

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
        raise RuntimeError("scheduled BF16 MMA requires an SM86 CUDA device")

    with torch.cuda.device(a.device):
        module = load_scheduled_bf16_mma()
        native = module.ScheduledBf16MmaPlan(a, w, out, group, stages, swizzle)
    return PreparedBf16Mma(owners=(a, w, out), native=native)
