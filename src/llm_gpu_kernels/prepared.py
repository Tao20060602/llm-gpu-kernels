"""Version-pinned prepared launch for fixed buffers and host-known group counts.

Triton 3.3.1 CompiledKernel.__getitem__ still takes all Python arguments,
including constexpr placeholders. Its C launcher omits constexprs only from GPU
kernel parameters, not Python argument parsing. This distinction is intentional.
"""
from dataclasses import dataclass
import torch
import triton
from .gemm import GemmConfig, grouped_gemm


@dataclass(frozen=True)
class PreparedGroupedGemm:
    """Buffers/metadata must not be resized, replaced or mutated after prepare.

    Input *values* may change. Keep metadata immutable and establish cross-stream
    data dependencies explicitly. No concurrent writes to the bound output.
    """
    a: torch.Tensor
    weights: torch.Tensor
    output: torch.Tensor
    plan: object
    config: GemmConfig
    kernel: object
    grid: tuple
    _runner: object
    _args: tuple

    def run(self):
        if self._runner is not None:
            # Compiled runner obtains the current stream on every invocation.
            # Fast normal case avoids re-entering the same CUDA device context.
            if torch.cuda.current_device() == self.a.device.index:
                self._runner(*self._args)
            else:
                with torch.cuda.device(self.a.device):
                    self._runner(*self._args)
        return self.output


def prepare_grouped_gemm(a, weights, plan, out=None, config=GemmConfig()):
    if triton.__version__ != "3.3.1":
        raise RuntimeError("prepared launcher is validated only with Triton 3.3.1")
    out, kernel = grouped_gemm(a, weights, plan, out, config, return_kernel=True)
    grid = (plan.experts.numel(), triton.cdiv(weights.shape[1], config.block_n), 1)
    runner, args = None, ()
    if kernel is not None:
        # Retain all tensor owners above while caching validated device pointers.
        # This avoids repeated Python tensor.data_ptr / driver pointer lookups.
        pointers = (a, weights, out, plan.offsets, plan.device_counts, plan.experts, plan.starts)
        args = tuple(t.data_ptr() for t in pointers) + (weights.shape[1], weights.shape[2],
                                                       config.block_m, config.block_n, config.block_k)
        if len(kernel.src.signature) != len(args):
            raise RuntimeError("unexpected Triton grouped launcher signature")
        runner = kernel[grid]
    return PreparedGroupedGemm(a, weights, out, plan, config, kernel, grid, runner, args)
