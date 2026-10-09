"""M24 synchronous W4 gate/up layout candidate with the M22 seven-launch path."""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading

import torch
import triton

from .gemm import GemmConfig
from .moe import _combine, _dynamic_grouped, _prefix, _reset
from .moe_cuda import CudaGateUpLaunch
from .moe_prepared import _launch
from .moe_ranked import PreparedRankedMoE, _rank_dispatch, _rank_histogram
from .moe_weight_layout import (
    LAYOUT,
    PHYSICAL_SCALES_SHAPE,
    PHYSICAL_WORDS_SHAPE,
    RepackedGateUp,
)
from .moe_w4_cuda import W4Workspace


_EXTENSION = None
_LOCK = threading.Lock()
_GCC_VERSION_RE = re.compile(r"^14(?:\.|$)")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class W4LayoutWorkspace(W4Workspace):
    """M22 workspace ownership with the packed field narrowed to M24 v2."""

    packed: RepackedGateUp


def _verified_gcc_from_env(variable: str) -> str:
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
    if not _GCC_VERSION_RE.match(version) or not any(
            marker in banner for marker in ("gcc", "g++", "gnu")):
        raise RuntimeError(f"{variable} must select GNU GCC 14; got {value!r} ({version})")
    return str(Path(compiler).resolve())


def _load_w4_layout_gate_up():
    """Load/compile the separate M24 extension only when prepare requests it."""
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
                raise RuntimeError("M24 layout extension requires the verified CUDA 12.8 toolkit")
            if torch.version.cuda != "12.8":
                raise RuntimeError(f"PyTorch CUDA runtime must be 12.8; got {torch.version.cuda!r}")

            # Compiler locations come only from the operator's environment.
            # They are checked here, not selected or rewritten by this module.
            _verified_gcc_from_env("CC")
            _verified_gcc_from_env("CXX")

            from torch.utils import cpp_extension
            if (cpp_extension.CUDA_HOME is None
                    or Path(cpp_extension.CUDA_HOME).resolve() != home):
                raise RuntimeError("cpp_extension CUDA_HOME differs from explicit CUDA_HOME")
            _EXTENSION = cpp_extension.load(
                name="llm_gpu_kernels_m24_w4_layout_sm86_v1",
                sources=[str(Path(__file__).with_name("cuda_w4_layout_gate_up.cu"))],
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


def _validate_repacked_schema(packed: RepackedGateUp) -> None:
    if not isinstance(packed, RepackedGateUp):
        raise TypeError("RepackedGateUp format_version=2 is required; v1 PackedGateUp is not accepted")
    if not isinstance(packed.words, torch.Tensor) or not isinstance(packed.scales, torch.Tensor):
        raise TypeError("M24 words and scales must be torch tensors")
    metadata = packed.metadata
    if (not isinstance(metadata, dict)
            or metadata.get("format_version") != 2
            or metadata.get("layout") != LAYOUT
            or metadata.get("group_size") != 128
            or metadata.get("logical_shape") != [128, 1536, 2048]
            or metadata.get("canonical_words_shape") != [128, 1536, 256]
            or metadata.get("canonical_scales_shape") != [128, 1536, 16]
            or metadata.get("physical_words_shape") != list(PHYSICAL_WORDS_SHAPE)
            or metadata.get("physical_scales_shape") != list(PHYSICAL_SCALES_SHAPE)):
        raise ValueError("incomplete or unsupported M24 physical-layout metadata")

    canonical = metadata.get("canonical_v1_metadata")
    if (not isinstance(canonical, dict)
            or canonical.get("format_version") != 1
            or canonical.get("group_size") != 128
            or canonical.get("logical_shape") != [128, 1536, 2048]
            or canonical.get("words_sha256") != metadata.get("canonical_words_sha256")
            or canonical.get("scales_sha256") != metadata.get("canonical_scales_sha256")):
        raise ValueError("M24 metadata is missing its matching canonical v1 snapshot")
    for key in (
        "canonical_words_sha256", "canonical_scales_sha256",
        "physical_words_sha256", "physical_scales_sha256",
    ):
        value = metadata.get(key)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"M24 metadata has an invalid {key}")

    ledger = metadata.get("byte_ledger")
    physical_words_bytes = packed.words.numel() * packed.words.element_size()
    physical_scales_bytes = packed.scales.numel() * packed.scales.element_size()
    packed_bytes = physical_words_bytes + physical_scales_bytes
    if (not isinstance(ledger, dict)
            or ledger.get("canonical_words_bytes") != 192 * 1024 * 1024
            or ledger.get("canonical_scales_bytes") != 6 * 1024 * 1024
            or ledger.get("canonical_packed_total_bytes") != 198 * 1024 * 1024
            or ledger.get("physical_words_bytes") != physical_words_bytes
            or ledger.get("physical_scales_bytes") != physical_scales_bytes
            or ledger.get("physical_packed_total_bytes") != packed_bytes
            or ledger.get("additional_repacked_tensor_bytes") != packed_bytes
            or ledger.get("physical_packed_total_bytes") != ledger.get("canonical_packed_total_bytes")):
        raise ValueError("M24 byte ledger must preserve the 198 MiB canonical packed+scale size")


def prepare_w4_layout_moe(x, repacked, down, ids, routing_weights, *, down_config=None):
    """Prepare the v2 packed layout for the existing fixed-domain seven launches.

    Repacked weights are immutable after their offline layout audit. Tensor
    addresses and shapes stay fixed; x, IDs and routing values can change in
    place. Callers establish cross-stream dependencies and do not overlap runs.
    This path retains the new repacked object only; it never prepares or keeps
    the original v1 word/scales tensors.
    """
    config = GemmConfig() if down_config is None else down_config
    if (not isinstance(config, GemmConfig)
            or config.block_m != 32 or config.block_k != 32
            or config.block_n not in (64, 128, 256)
            or config.warps != 4 or config.stages != 3):
        raise ValueError("only declared BM32/BK32/W4/S3 down configurations")
    _validate_repacked_schema(repacked)
    if (x.ndim != 2 or x.shape[1] != 2048
            or tuple(down.shape) != (128, 2048, 768)
            or x.dtype != torch.bfloat16 or down.dtype != torch.bfloat16):
        raise ValueError("BF16 E128/H2048/I768 domain required")
    if (ids.ndim != 2 or tuple(ids.shape) != (x.shape[0], 8)
            or ids.dtype not in (torch.int32, torch.int64)
            or routing_weights.shape != ids.shape
            or routing_weights.dtype != torch.float32):
        raise ValueError("IDs[T,8] integer and FP32 routing weights required")
    words, scales = repacked.words, repacked.scales
    if (tuple(words.shape) != PHYSICAL_WORDS_SHAPE
            or tuple(scales.shape) != PHYSICAL_SCALES_SHAPE
            or words.dtype != torch.int32 or scales.dtype != torch.bfloat16):
        raise ValueError("M24 physical packed shape/dtype mismatch")
    owners = (x, words, scales, down, ids, routing_weights)
    if any(t.device.type != "cuda" or t.device != x.device or not t.is_contiguous()
           for t in owners):
        raise ValueError("same-device contiguous CUDA inputs required")
    if ids.numel() > 2**31 - 1 - 128 * 32:
        raise ValueError("int32 route bound exceeded")
    if any(t.data_ptr() % 16 for t in (x, words)):
        raise ValueError("16-byte input alignment required")
    if torch.cuda.get_device_capability(x.device) != (8, 6):
        raise RuntimeError("SM86 required")

    device, tokens, rows = x.device, x.shape[0], ids.numel()
    bf = lambda shape: torch.empty(shape, device=device, dtype=torch.bfloat16)
    ix = lambda shape: torch.empty(shape, device=device, dtype=torch.int32)
    base = W4LayoutWorkspace(
        x, down, ids, routing_weights, repacked, bf((tokens, 2048)),
        ix((128,)), ix((128,)), ix((128,)), ix((129,)), ix((rows,)),
        bf((rows, 2048)), bf((rows, 768)), bf((rows, 2048)),
        torch.zeros((1,), device=device, dtype=torch.int32), config,
    )
    ranks = ix((rows,))
    if tokens == 0:
        return PreparedRankedMoE(base, ranks, ())

    bound = triton.cdiv(rows, 32) + 127
    histogram_grid = (triton.cdiv(rows, 256),)
    dispatch_grid = (triton.cdiv(rows, 16), triton.cdiv(2048, 128))
    down_grid = (bound, triton.cdiv(2048, config.block_n))
    combine_grid = (tokens, triton.cdiv(2048, 256))
    module = _load_w4_layout_gate_up()
    native = module.W4LayoutGateUpPlan(
        base.grouped_input, words, scales, base.activated,
        base.offsets, base.counts, base.tasks,
    )
    gate_launch = CudaGateUpLaunch(native, (bound, 12))
    with torch.cuda.device(device):
        reset = _reset[(1,)](base.counts, base.status, 128, 128)
        hist = _rank_histogram[histogram_grid](
            ids, routing_weights, base.counts, base.status, ranks,
            rows, 128, 256,
        )
        prefix = _prefix[(1,)](
            base.counts, base.offsets, base.cursors, base.tasks, 128, 32, 128,
        )
        dispatch = _rank_dispatch[dispatch_grid](
            x, ids, ranks, base.offsets, base.mapping, base.grouped_input,
            rows, 2048, 8, 128, 16, 128,
        )
        gate_launch.run()
        down_kernel = _dynamic_grouped[down_grid](
            base.activated, down, base.down_output, base.offsets, base.counts,
            base.tasks, 128, 2048, 768, 128, 32, config.block_n, 32,
            num_warps=4, num_stages=3,
        )
        combine = _combine[combine_grid](
            base.down_output, base.mapping, routing_weights, base.output,
            tokens, 2048, 8, 256,
        )
    base.kernels.update(reset=reset, rank_histogram=hist, prefix=prefix,
                        rank_dispatch=dispatch, down=down_kernel, combine=combine)
    pointer = lambda tensor: tensor.data_ptr()
    launches = (
        _launch("reset", reset, (1,), (pointer(base.counts), pointer(base.status), 128, 128)),
        _launch("rank_histogram", hist, histogram_grid,
                (pointer(ids), pointer(routing_weights), pointer(base.counts),
                 pointer(base.status), pointer(ranks), rows, 128, 256)),
        _launch("prefix", prefix, (1,),
                (pointer(base.counts), pointer(base.offsets), pointer(base.cursors),
                 pointer(base.tasks), 128, 32, 128)),
        _launch("rank_dispatch", dispatch, dispatch_grid,
                (pointer(x), pointer(ids), pointer(ranks), pointer(base.offsets),
                 pointer(base.mapping), pointer(base.grouped_input),
                 rows, 2048, 8, 128, 16, 128)),
        gate_launch,
        _launch("down", down_kernel, down_grid,
                (pointer(base.activated), pointer(down), pointer(base.down_output),
                 pointer(base.offsets), pointer(base.counts), pointer(base.tasks),
                 128, 2048, 768, 128, 32, config.block_n, 32)),
        _launch("combine", combine, combine_grid,
                (pointer(base.down_output), pointer(base.mapping),
                 pointer(routing_weights), pointer(base.output),
                 tokens, 2048, 8, 256)),
    )
    return PreparedRankedMoE(base, ranks, launches)
