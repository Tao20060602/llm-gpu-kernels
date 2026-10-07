"""M22 BF16 hand-written CUDA foundation; quantization is not implemented here."""
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import threading

import torch
import triton

from .gemm import GemmConfig
from .moe_configured import prepare_configured_moe
from .moe_ranked import PreparedRankedMoE

_EXTENSION = None
_LOCK = threading.Lock()
VARIANTS = {'sync': 0, 'async': 1, 'async_skip': 2}


def load_cuda_gate_up():
    global _EXTENSION
    with _LOCK:
        if _EXTENSION is None:
            home = Path(os.environ.get('CUDA_HOME', '')).resolve()
            if not os.environ.get('CUDA_HOME') or not (home / 'bin/nvcc').is_file():
                raise RuntimeError('explicit CUDA 12.8 toolkit required before import/build')
            version = subprocess.check_output([str(home / 'bin/nvcc'), '--version'], text=True)
            if 'release 12.8,' not in version:
                raise RuntimeError('M22 requires the verified CUDA 12.8 compiler')
            from torch.utils import cpp_extension
            if cpp_extension.CUDA_HOME is None or Path(cpp_extension.CUDA_HOME).resolve() != home:
                raise RuntimeError('cpp_extension CUDA_HOME differs from explicit toolkit')
            _EXTENSION = cpp_extension.load(
                name='llm_gpu_kernels_m22_cuda_gate_up_sm86_v1',
                sources=[str(Path(__file__).with_name('cuda_gate_up.cu'))],
                extra_cflags=['-O3', '-std=c++17'],
                extra_cuda_cflags=['-O3', '-std=c++17', '-lineinfo',
                                  '-gencode=arch=compute_86,code=sm_86',
                                  '--ptxas-options=-v'],
                with_cuda=True, verbose=True)
        return _EXTENSION


@dataclass(frozen=True)
class CudaGateUpLaunch:
    native: object
    grid: tuple
    name: str = 'gate_up_swiglu'

    def run(self):
        self.native.run()


def prepare_cuda_moe(x, gate_up, down, ids, routing_weights, *, variant='sync',
                     down_config=None):
    """Same BM32 ranked metadata; replace only gate/up. No hot-path readback."""
    if variant not in VARIANTS:
        raise ValueError('unsupported CUDA variant')
    config = GemmConfig() if down_config is None else down_config
    if config.block_m != 32:
        raise ValueError('CUDA gate/up requires BM32 prefix/down metadata')
    if tuple(gate_up.shape) != (128, 1536, 2048) or tuple(down.shape) != (128, 2048, 768):
        raise ValueError('CUDA gate/up only supports E128/H2048/I768')
    original = prepare_configured_moe(x, gate_up, down, ids, routing_weights, config)
    if x.shape[0] == 0:
        return original
    base = original.base
    module = load_cuda_gate_up()
    native = module.GateUpPlan(base.grouped_input, gate_up, base.activated,
                              base.offsets, base.counts, base.tasks, VARIANTS[variant])
    launch = CudaGateUpLaunch(native, (triton.cdiv(ids.numel(), 32) + 127, 12))
    launches = (*original.launches[:4], launch, *original.launches[5:])
    return PreparedRankedMoE(base=base, ranks=original.ranks, launches=launches)
