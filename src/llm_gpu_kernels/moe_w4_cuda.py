"""M22 fixed-route W4 gate/up with owned GPU metadata and BF16 down.

Preparation accepts immutable, already validated PackedGateUp words/scales.
It owns no dense gate/up weight or projected scratch tensor. The hot path
unpacks inside the CUDA gate/up tile and retains the seven-launch DAG.
"""
from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import threading

import torch
import triton

from .gemm import GemmConfig
from .moe import _reset, _prefix, _dynamic_grouped, _combine
from .moe_cuda import CudaGateUpLaunch
from .moe_prepared import _launch
from .moe_ranked import _rank_histogram, _rank_dispatch, PreparedRankedMoE
from .moe_weight_quant import PackedGateUp

_EXTENSION = None
_LOCK = threading.Lock()


def load_w4_gate_up():
    global _EXTENSION
    with _LOCK:
        if _EXTENSION is None:
            value = os.environ.get('CUDA_HOME')
            if not value:
                raise RuntimeError('explicit CUDA 12.8 toolkit required')
            home = Path(value).resolve()
            if not (home / 'bin/nvcc').is_file():
                raise RuntimeError('CUDA_HOME has no nvcc')
            version = subprocess.check_output([str(home / 'bin/nvcc'), '--version'], text=True)
            if 'release 12.8,' not in version:
                raise RuntimeError('requires verified CUDA 12.8 compiler')
            from torch.utils import cpp_extension
            if cpp_extension.CUDA_HOME is None or Path(cpp_extension.CUDA_HOME).resolve() != home:
                raise RuntimeError('cpp_extension CUDA_HOME mismatch')
            _EXTENSION = cpp_extension.load(
                name='llm_gpu_kernels_m22_w4_gate_up_sm86_v1',
                sources=[str(Path(__file__).with_name('cuda_w4_gate_up.cu'))],
                extra_cflags=['-O3', '-std=c++17'],
                extra_cuda_cflags=['-O3', '-std=c++17', '-lineinfo',
                                  '-gencode=arch=compute_86,code=sm_86', '--ptxas-options=-v'],
                with_cuda=True, verbose=True)
        return _EXTENSION


@dataclass
class W4Workspace:
    x: torch.Tensor
    down: torch.Tensor
    ids: torch.Tensor
    routing_weights: torch.Tensor
    packed: PackedGateUp
    output: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor
    cursors: torch.Tensor
    tasks: torch.Tensor
    mapping: torch.Tensor
    grouped_input: torch.Tensor
    activated: torch.Tensor
    down_output: torch.Tensor
    status: torch.Tensor
    config: GemmConfig
    kernels: dict = field(default_factory=dict)

    @property
    def workspace_bytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.counts, self.offsets, self.cursors, self.tasks, self.mapping,
                    self.grouped_input, self.activated, self.down_output, self.status))

    def check_inputs(self):
        flag = int(self.status.item())
        if flag:
            raise ValueError(f'invalid runtime route inputs: device status={flag}')


def prepare_w4_moe(x, packed, down, ids, routing_weights, *, down_config=None):
    """Prepare outside timing; no dense weight reconstruction or counts readback.

    Packed words/scales must remain immutable after their offline format audit.
    Tensor shapes/storage are fixed; values/IDs/route weights may change in place.
    Caller establishes cross-stream dependencies; workspace runs cannot overlap.
    """
    config = GemmConfig() if down_config is None else down_config
    if (not isinstance(config, GemmConfig) or config.block_m != 32 or config.block_k != 32
            or config.block_n not in (64, 128, 256) or config.warps != 4 or config.stages != 3):
        raise ValueError('only declared BM32/BK32/W4/S3 down configurations')
    if not isinstance(packed, PackedGateUp) or packed.metadata.get('format_version') != 1 or packed.metadata.get('group_size') != 128:
        raise ValueError('validated W4G128 format version 1 required')
    if (x.ndim != 2 or x.shape[1] != 2048 or tuple(down.shape) != (128, 2048, 768)
            or x.dtype != torch.bfloat16 or down.dtype != torch.bfloat16):
        raise ValueError('BF16 E128/H2048/I768 domain required')
    if (ids.ndim != 2 or tuple(ids.shape) != (x.shape[0], 8) or ids.dtype not in (torch.int32, torch.int64)
            or routing_weights.shape != ids.shape or routing_weights.dtype != torch.float32):
        raise ValueError('IDs[T,8] integer and FP32 routing weights required')
    words, scales = packed.words, packed.scales
    if (tuple(words.shape) != (128, 1536, 256) or words.dtype != torch.int32
            or tuple(scales.shape) != (128, 1536, 16) or scales.dtype != torch.bfloat16):
        raise ValueError('packed shape/dtype mismatch')
    owners = (x, words, scales, down, ids, routing_weights)
    if any(t.device.type != 'cuda' or t.device != x.device or not t.is_contiguous() for t in owners):
        raise ValueError('same-device contiguous CUDA inputs required')
    if ids.numel() > 2**31 - 1 - 128 * 32:
        raise ValueError('int32 route bound exceeded')
    if any(t.data_ptr() % 16 for t in (x, words)):
        raise ValueError('16-byte input alignment required')
    if torch.cuda.get_device_capability(x.device) != (8, 6):
        raise RuntimeError('SM86 required')
    device, tokens, rows = x.device, x.shape[0], ids.numel()
    bf = lambda shape: torch.empty(shape, device=device, dtype=torch.bfloat16)
    ix = lambda shape: torch.empty(shape, device=device, dtype=torch.int32)
    base = W4Workspace(x, down, ids, routing_weights, packed, bf((tokens, 2048)),
                       ix((128,)), ix((128,)), ix((128,)), ix((129,)), ix((rows,)),
                       bf((rows, 2048)), bf((rows, 768)), bf((rows, 2048)),
                       torch.zeros((1,), device=device, dtype=torch.int32), config)
    ranks = ix((rows,))
    if tokens == 0:
        return PreparedRankedMoE(base, ranks, ())
    bound = triton.cdiv(rows, 32) + 127
    histogram_grid = (triton.cdiv(rows, 256),)
    dispatch_grid = (triton.cdiv(rows, 16), triton.cdiv(2048, 128))
    down_grid = (bound, triton.cdiv(2048, config.block_n))
    combine_grid = (tokens, triton.cdiv(2048, 256))
    module = load_w4_gate_up()
    native = module.W4GateUpPlan(base.grouped_input, words, scales, base.activated,
                                 base.offsets, base.counts, base.tasks)
    gate_launch = CudaGateUpLaunch(native, (bound, 12))
    with torch.cuda.device(device):
        reset = _reset[(1,)](base.counts, base.status, 128, 128)
        hist = _rank_histogram[histogram_grid](ids, routing_weights, base.counts, base.status, ranks, rows, 128, 256)
        prefix = _prefix[(1,)](base.counts, base.offsets, base.cursors, base.tasks, 128, 32, 128)
        dispatch = _rank_dispatch[dispatch_grid](x, ids, ranks, base.offsets, base.mapping,
                                                base.grouped_input, rows, 2048, 8, 128, 16, 128)
        gate_launch.run()
        down_kernel = _dynamic_grouped[down_grid](
            base.activated, down, base.down_output, base.offsets, base.counts, base.tasks,
            128, 2048, 768, 128, 32, config.block_n, 32, num_warps=4, num_stages=3)
        combine = _combine[combine_grid](base.down_output, base.mapping, routing_weights,
                                         base.output, tokens, 2048, 8, 256)
    base.kernels.update(reset=reset, rank_histogram=hist, prefix=prefix,
                        rank_dispatch=dispatch, down=down_kernel, combine=combine)
    pointer = lambda t: t.data_ptr()
    launches = (
        _launch('reset', reset, (1,), (pointer(base.counts), pointer(base.status), 128, 128)),
        _launch('rank_histogram', hist, histogram_grid,
                (pointer(ids), pointer(routing_weights), pointer(base.counts), pointer(base.status), pointer(ranks), rows, 128, 256)),
        _launch('prefix', prefix, (1,), (pointer(base.counts), pointer(base.offsets), pointer(base.cursors), pointer(base.tasks), 128, 32, 128)),
        _launch('rank_dispatch', dispatch, dispatch_grid,
                (pointer(x), pointer(ids), pointer(ranks), pointer(base.offsets), pointer(base.mapping), pointer(base.grouped_input), rows, 2048, 8, 128, 16, 128)),
        gate_launch,
        _launch('down', down_kernel, down_grid,
                (pointer(base.activated), pointer(down), pointer(base.down_output), pointer(base.offsets), pointer(base.counts), pointer(base.tasks),
                 128, 2048, 768, 128, 32, config.block_n, 32)),
        _launch('combine', combine, combine_grid,
                (pointer(base.down_output), pointer(base.mapping), pointer(routing_weights), pointer(base.output), tokens, 2048, 8, 256)))
    return PreparedRankedMoE(base, ranks, launches)
