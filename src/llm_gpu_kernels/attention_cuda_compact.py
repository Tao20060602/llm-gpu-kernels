"""Prepared Python wrapper for the M36 compact-SMEM CUDA GQA forward kernel."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import threading

import torch

from .gemm_mma import PreparedBf16Mma, _storage_identity


_EXTENSION = None
_LOCK = threading.Lock()
_MODULE_NAME = "llm_gpu_kernels_m36_compact_gqa_sm86_v2"
_STAGES = (1, 2)
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


def load_cuda_gqa_compact():
    """Load the CUDA 12.8 / GCC 14 / SM86 compact v2 extension on first use."""
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
                raise RuntimeError("CUDA GQA extension requires the verified CUDA 12.8 toolkit")
            if torch.version.cuda != "12.8":
                raise RuntimeError(
                    f"PyTorch CUDA runtime must be 12.8; got {torch.version.cuda!r}"
                )

            _verified_gcc14("CC")
            _verified_gcc14("CXX")

            from torch.utils import cpp_extension

            if (cpp_extension.CUDA_HOME is None
                    or Path(cpp_extension.CUDA_HOME).resolve() != home):
                raise RuntimeError("cpp_extension CUDA_HOME differs from explicit CUDA_HOME")
            build_directory = Path(__file__).resolve().parents[2] / "artifacts" / "m36-build-v2"
            build_directory.mkdir(parents=True, exist_ok=True)
            source = Path(__file__).with_name("cuda_gqa_attention_compact.cu")
            if not source.is_file():
                raise FileNotFoundError(f"M36 CUDA source is missing: {source}")
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


def _validate_attention_input(tensor, name, *, device, heads):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (heads, 128):
        raise ValueError(f"{name} must have shape [L, {heads}, 128]")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must have dtype torch.bfloat16")
    if tensor.device.type != "cuda" or tensor.device != device:
        raise ValueError(f"{name} must be on q's CUDA device")
    if tuple(tensor.stride()[1:]) != (128, 1):
        raise ValueError(f"{name} must have inner strides (128, 1)")
    if tensor.stride(0) < heads * 128 or tensor.stride(0) % 8:
        raise ValueError(f"{name} token stride must be at least {heads * 128} and divisible by 8")
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} base address must be 16-byte aligned")


def _validate_output(out, q):
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if tuple(out.shape) != tuple(q.shape) or out.ndim != 3:
        raise ValueError(f"out must have shape {tuple(q.shape)}")
    if out.dtype != torch.bfloat16:
        raise TypeError("out must have dtype torch.bfloat16")
    if out.device != q.device:
        raise ValueError("out must be on q's CUDA device")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")


class PreparedCudaGqaCompact(PreparedBf16Mma):
    """M25 owner/lifetime holder with the M36 output at the end of owners."""

    def run(self):
        self.native.run()
        return self.owners[-1]


def prepare_cuda_gqa_compact(q, segments, out=None, *, stages=1):
    """Prepare the fixed-address compact four-segment BF16 GQA forward on SM86.

    ``segments`` is ordered as selected, sink, recent, and current. Input values
    may change in place; shapes, strides, devices, and storage must stay fixed.
    Callers establish cross-stream dependencies and avoid concurrent writes to
    one output.
    """
    if type(stages) is not int or stages not in _STAGES:
        raise ValueError("stages must be 1 or 2")
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or tuple(q.shape[1:]) != (32, 128):
        raise ValueError("q must have shape [T, 32, 128]")
    if q.device.type != "cuda" or q.dtype != torch.bfloat16:
        raise ValueError("q must be a CUDA BF16 tensor")
    _validate_attention_input(q, "q", device=q.device, heads=32)
    try:
        fixed_segments = tuple(segments)
    except TypeError as exc:
        raise TypeError("segments must be an iterable of four K/V pairs") from exc
    if len(fixed_segments) != 4:
        raise ValueError("exactly four K/V segments are required")

    keys, values = [], []
    for index, pair in enumerate(fixed_segments):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"segment {index} must be a (key, value) pair")
        key, value = pair
        _validate_attention_input(key, f"segments[{index}].key", device=q.device, heads=8)
        _validate_attention_input(value, f"segments[{index}].value", device=q.device, heads=8)
        if key.shape != value.shape:
            raise ValueError(f"segment {index} key and value shapes must match")
        keys.append(key)
        values.append(value)
    if keys[-1].shape[0] != q.shape[0]:
        raise ValueError("current K/V segment length must equal T")

    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _validate_output(out, q)
    inputs = (q, *(tensor for pair in fixed_segments for tensor in pair))
    output_storage = _storage_identity(out)
    if any(output_storage == _storage_identity(tensor) for tensor in inputs):
        raise ValueError("out must not share storage with any input")
    if torch.cuda.get_device_capability(q.device) != (8, 6):
        raise RuntimeError("CUDA GQA attention requires an SM86 CUDA device")

    with torch.cuda.device(q.device):
        module = load_cuda_gqa_compact()
        native = module.CompactCudaGqaPlan(
            q, keys, values, out, stages
        )
    return PreparedCudaGqaCompact(owners=(*inputs, out), native=native)
