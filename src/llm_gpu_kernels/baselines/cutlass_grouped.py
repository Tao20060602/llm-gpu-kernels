"""Prepared CUTLASS 3.8.0 grouped GEMM baseline for SM86.

Build requirements are intentionally explicit: set ``CUDA_HOME`` to the CUDA
12.8 toolchain and ``CUTLASS_PATH`` to an unmodified CUTLASS v3.8.0 checkout
before calling :func:`prepare_cutlass_grouped_gemm`. The compiled extension is
cached by PyTorch's C++ extension loader.

The prepared plan binds A, W, and output tensors. Host counts, active expert
problems, pointer/stride tables, and CUTLASS workspace are prepared before
``plan.run()``. Repeated calls to ``run`` submit the cached grouped operator on
the current PyTorch stream and return the bound output tensor.
"""

from __future__ import annotations

import operator
import os
from pathlib import Path
import re
import threading
from typing import Iterable

import torch


_EXTENSION = None
_EXTENSION_LOCK = threading.Lock()


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set explicitly before building the CUTLASS wrapper")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"{name} does not name an existing directory: {path}")
    return path


def _check_cutlass_380(root: Path) -> Path:
    include = root / "include"
    version_header = include / "cutlass" / "version.h"
    if not version_header.is_file() or not (include / "cutlass" / "cutlass.h").is_file():
        raise RuntimeError(f"CUTLASS_PATH must be a CUTLASS source checkout: {root}")
    source = version_header.read_text(encoding="utf-8")
    for macro, expected in (("CUTLASS_MAJOR", 3), ("CUTLASS_MINOR", 8), ("CUTLASS_PATCH", 0)):
        match = re.search(rf"^\s*#\s*define\s+{macro}\s+(\d+)\s*$", source, re.MULTILINE)
        if match is None or int(match.group(1)) != expected:
            raise RuntimeError(f"CUTLASS_PATH must contain CUTLASS 3.8.0 headers: {version_header}")
    return include


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION

        cuda_home = _required_path("CUDA_HOME")
        nvcc = cuda_home / "bin" / "nvcc"
        if not nvcc.is_file():
            raise RuntimeError(f"CUDA_HOME must contain bin/nvcc: {cuda_home}")
        cutlass_include = _check_cutlass_380(_required_path("CUTLASS_PATH"))

        # Import only after validating CUDA_HOME so cpp_extension cannot silently
        # select a CUDA installation discovered from PATH or another environment.
        from torch.utils import cpp_extension

        detected_home = cpp_extension.CUDA_HOME
        if detected_home is None or Path(detected_home).expanduser().resolve() != cuda_home:
            raise RuntimeError(
                "torch.utils.cpp_extension selected CUDA_HOME="
                f"{detected_home!r}, but the wrapper requires the explicit environment path "
                f"{str(cuda_home)!r}; set CUDA_HOME before importing this module"
            )

        source = Path(__file__).with_name("cutlass_grouped.cu")
        _EXTENSION = cpp_extension.load(
            name="llm_gpu_kernels_cutlass_grouped_sm86_v380",
            sources=[str(source)],
            extra_include_paths=[str(cutlass_include)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "-gencode=arch=compute_86,code=sm_86",
            ],
            with_cuda=True,
            verbose=os.environ.get("CUTLASS_VERBOSE_BUILD", "0") == "1",
        )
        return _EXTENSION


def _counts_tuple(counts: Iterable[int]) -> tuple[int, ...]:
    if isinstance(counts, torch.Tensor):
        if counts.device.type != "cpu":
            raise ValueError("counts must be host-known; CUDA count tensors are not read back")
        counts = counts.tolist()
    try:
        result = tuple(operator.index(count) for count in counts)
    except (TypeError, ValueError) as exc:
        raise TypeError("counts must be an iterable of host integers") from exc
    if not result or any(count < 0 for count in result):
        raise ValueError("counts must be a nonempty sequence of nonnegative host integers")
    return result


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    # Comparing StorageImpl identities also handles zero-element tensors, whose
    # data_ptr values may both be null despite owning different storages.
    return left.untyped_storage()._cdata == right.untyped_storage()._cdata


class CutlassGroupedPlan:
    """Object-bound, reusable CUTLASS grouped GEMM plan."""

    def __init__(self, output: torch.Tensor, native_plan) -> None:
        self._output = output
        self._native_plan = native_plan

    @property
    def output(self) -> torch.Tensor:
        return self._output

    @property
    def out(self) -> torch.Tensor:
        return self._output

    @property
    def active_experts(self) -> tuple[int, ...]:
        return tuple(self._native_plan.active_experts)

    @property
    def device_index(self) -> int:
        return self._native_plan.device_index

    @property
    def workspace_bytes(self) -> int:
        return self._native_plan.workspace_bytes

    def run(self) -> torch.Tensor:
        """Submit the cached operator on the current stream and return output."""
        self._native_plan.run()
        return self._output


def prepare_cutlass_grouped_gemm(
    a: torch.Tensor,
    weights: torch.Tensor,
    counts: Iterable[int],
    out: torch.Tensor | None = None,
) -> CutlassGroupedPlan:
    """Prepare a BF16 ragged grouped GEMM with CUTLASS 3.8.0.

    ``a`` has shape ``[sum(counts), K]``. ``weights`` has shape ``[E, N, K]``;
    each expert slice is interpreted as a column-major ``[K, N]`` B operand.
    Counts are host-known. Their M values remain ragged and are not padded.

    This small baseline fixes the kernel at an SM80 TensorOp CTA tile of
    ``64x128x32``, warp tile ``32x64x32``, instruction ``16x8x16``, and
    alignment 8. It is built specifically for the repository's SM86 device 0.
    Call ``plan.run()`` repeatedly to execute without rebuilding metadata.
    """
    if not isinstance(a, torch.Tensor) or not isinstance(weights, torch.Tensor):
        raise TypeError("a and weights must be torch.Tensor objects")
    if a.ndim != 2 or weights.ndim != 3:
        raise ValueError("expected A[sumM,K] and W[E,N,K]")
    if a.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16:
        raise ValueError("A and weights must be BF16")
    if not a.is_cuda or not weights.is_cuda:
        raise ValueError("A and weights must be CUDA tensors")
    if not a.is_contiguous() or not weights.is_contiguous():
        raise ValueError("A and weights must be contiguous")
    if a.device != weights.device:
        raise ValueError("A and weights must be on the same CUDA device")

    host_counts = _counts_tuple(counts)
    experts, n, k = weights.shape
    if len(host_counts) != experts or experts <= 0:
        raise ValueError("counts must have one entry for each nonempty expert dimension")
    if sum(host_counts) != a.shape[0] or a.shape[1] != k:
        raise ValueError("sum(counts) must equal A.rows and A.K must equal W.K")
    if n <= 0 or k <= 0:
        raise ValueError("N and K must be positive")
    if n % 8 or k % 8:
        raise ValueError("the align8 CUTLASS configuration requires N and K divisible by 8")

    if out is None:
        out = torch.empty((a.shape[0], n), device=a.device, dtype=torch.bfloat16)
    elif not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if out.ndim != 2 or tuple(out.shape) != (a.shape[0], n):
        raise ValueError("out must have shape [A.rows, W.N]")
    if out.dtype != torch.bfloat16 or not out.is_cuda or not out.is_contiguous():
        raise ValueError("out must be a contiguous CUDA BF16 tensor")
    if out.device != a.device:
        raise ValueError("A, weights, and out must be on the same CUDA device")
    if _shares_storage(out, a) or _shares_storage(out, weights):
        raise ValueError("out must not share storage with A or weights")
    if any(tensor.data_ptr() % 16 for tensor in (a, weights, out)):
        raise ValueError("the align8 BF16 configuration requires 16-byte aligned tensor pointers")

    # Fail early with a useful message; the native layer checks the same device
    # capability before constructing CUTLASS arguments.
    if a.device.index != 0 or torch.cuda.get_device_capability(a.device) != (8, 6):
        raise ValueError("the CUTLASS grouped baseline requires CUDA device 0 with SM86")

    native_plan = _load_extension().prepare(a, weights, out, list(host_counts))
    return CutlassGroupedPlan(out, native_plan)


__all__ = ["CutlassGroupedPlan", "prepare_cutlass_grouped_gemm"]
