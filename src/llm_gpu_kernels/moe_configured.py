"""Explicit tile configurations over the ranked fixed-route MoE path.

Preparation recompiles prefix/GEMM launchers for a shared tile configuration.
It does not select configurations from GPU counts or change the M4 rounding
contract. The original M4/M5/M6 entry points stay fixed for reproducibility.
"""
import torch
import triton

from .gemm import GemmConfig
from .moe import _prefix
from .moe_fused import _dynamic_grouped_gate_up_swiglu
from .moe_prepared import _launch
from .moe_ranked import PreparedRankedMoE, prepare_ranked_moe_launcher


def prepare_configured_moe(x, gate_up, down, ids, routing_weights, config):
    """Bind one explicit configuration; compile and allocate outside timing."""
    if not isinstance(config, GemmConfig):
        raise TypeError("config must be GemmConfig")
    original = prepare_ranked_moe_launcher(x, gate_up, down, ids, routing_weights)
    base = original.base
    base.config = config
    if x.shape[0] == 0:
        return original
    experts, hidden, intermediate = down.shape
    routed_rows = ids.numel()
    be = triton.next_power_of_2(experts)
    task_bound = triton.cdiv(routed_rows, config.block_m) + experts - 1
    fused_grid = (task_bound, triton.cdiv(intermediate, config.block_n))

    with torch.cuda.device(x.device):
        original.launches[0].run()
        original.launches[1].run()
        prefix_kernel = _prefix[(1,)](
            base.counts, base.offsets, base.cursors, base.tasks,
            experts, config.block_m, be,
        )
        original.launches[3].run()
        fused_kernel = _dynamic_grouped_gate_up_swiglu[fused_grid](
            base.grouped_input, gate_up, base.activated,
            base.offsets, base.counts, base.tasks,
            experts, intermediate, hidden, be,
            config.block_m, config.block_n, config.block_k,
            num_warps=config.warps, num_stages=config.stages,
        )
        base.down_project()
        original.launches[-1].run()
    base.kernels["prefix"] = prefix_kernel
    base.kernels["gate_up_swiglu"] = fused_kernel
    prefix = _launch("prefix", prefix_kernel, (1,), (
        base.counts.data_ptr(), base.offsets.data_ptr(), base.cursors.data_ptr(),
        base.tasks.data_ptr(), experts, config.block_m, be,
    ))
    fused = _launch("gate_up_swiglu", fused_kernel, fused_grid, (
        base.grouped_input.data_ptr(), gate_up.data_ptr(), base.activated.data_ptr(),
        base.offsets.data_ptr(), base.counts.data_ptr(), base.tasks.data_ptr(),
        experts, intermediate, hidden, be,
        config.block_m, config.block_n, config.block_k,
    ))
    down_launch = _launch("down", base.kernels["down"],
                         (task_bound, triton.cdiv(hidden, config.block_n)), (
        base.activated.data_ptr(), down.data_ptr(), base.down_output.data_ptr(),
        base.offsets.data_ptr(), base.counts.data_ptr(), base.tasks.data_ptr(),
        experts, hidden, intermediate, be,
        config.block_m, config.block_n, config.block_k,
    ))
    launches = (*original.launches[:2], prefix, original.launches[3],
                fused, down_launch, original.launches[-1])
    return PreparedRankedMoE(base=base, ranks=original.ranks, launches=launches)
