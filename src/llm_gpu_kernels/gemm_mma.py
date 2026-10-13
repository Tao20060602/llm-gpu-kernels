"""Prepared wrapper for the SM86 BF16 Tensor Core GEMM experiment."""

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import threading

import torch


_EXTENSION = None
_LOCK = threading.Lock()
_MODULE_NAME = "llm_gpu_kernels_m25_bf16_mma_sm86_v1"


def load_bf16_mma():
    """Load the CUDA 12.8 / SM86 extension on first use."""
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
                raise RuntimeError("BF16 MMA extension requires the verified CUDA 12.8 toolkit")
            if torch.version.cuda != "12.8":
                raise RuntimeError(
                    f"PyTorch CUDA runtime must be 12.8; got {torch.version.cuda!r}"
                )

            from torch.utils import cpp_extension

            if (cpp_extension.CUDA_HOME is None
                    or Path(cpp_extension.CUDA_HOME).resolve() != home):
                raise RuntimeError("cpp_extension CUDA_HOME differs from explicit CUDA_HOME")
            _EXTENSION = cpp_extension.load(
                name=_MODULE_NAME,
                sources=[str(Path(__file__).with_name("cuda_bf16_mma.cu"))],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=[
                    "-O3", "-std=c++17", "-lineinfo",
                    "-gencode=arch=compute_86,code=sm_86",
                    "--ptxas-options=-v",
                ],
                with_cuda=True,
                verbose=True,
            )
        return _EXTENSION


def _validate_bf16_cuda_tensor(tensor, name, device=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must have dtype torch.bfloat16")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if device is not None and tensor.device != device:
        raise ValueError("A, weight, and out must be on the same CUDA device")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _storage_identity(tensor):
    # Storage identity catches aliasing through distinct views, including views
    # whose logical data pointers do not match.
    return tensor.untyped_storage()._cdata


@dataclass(frozen=True)
class PreparedBf16Mma:
    """A plan bound to fixed input/output shapes and storage addresses.

    Keep the bound tensors alive and do not resize or replace them. Their
    contents may be updated in place between runs.
    """

    owners: tuple
    native: object

    def run(self):
        self.native.run()
        return self.owners[2]


def prepare_bf16_mma(a, weight, out=None, stages=2):
    """Prepare ``A[M,K] @ weight[N,K].T`` into a BF16 ``out[M,N]`` buffer.

    The native plan validates implementation-specific dimension and launch
    bounds. Calls to ``run`` use the current CUDA stream; callers establish
    cross-stream dependencies and avoid overlapping writes to the same output.
    """
    if type(stages) is not int or stages not in (1, 2, 3, 4):
        raise ValueError("stages must be an integer from 1 through 4")
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
        raise RuntimeError("BF16 MMA requires an SM86 CUDA device")

    with torch.cuda.device(a.device):
        module = load_bf16_mma()
        native = module.Bf16MmaPlan(a, weight, out, stages)
    return PreparedBf16Mma(owners=(a, weight, out), native=native)
