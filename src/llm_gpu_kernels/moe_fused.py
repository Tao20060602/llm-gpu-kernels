"""Local gate/up GEMM epilogue fusion for the fixed-route MoE baseline.

The fused CTA computes gate and up projections, rounds each projection to
BF16 exactly where the unfused path stores its gate/up buffer, applies SwiGLU,
and writes the existing BF16 activation buffer. The base still owns its
projected buffer; this saves intermediate traffic, not allocated workspace.
"""
from dataclasses import dataclass, field

import torch
import triton
import triton.language as tl

from .gemm import GemmConfig
from .moe import PreparedMoE, prepare_moe


@triton.jit
def _dynamic_grouped_gate_up_swiglu(
    A, W, ACT, OFFSETS, COUNTS, TASKS,
    E: tl.constexpr, I: tl.constexpr, H: tl.constexpr, BE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Grouped gate/up GEMMs with a numerically matched SwiGLU epilogue."""
    task = tl.program_id(0)
    total_tasks = tl.load(TASKS + E)
    if task < total_tasks:
        experts = tl.arange(0, BE)
        task_ends = tl.load(TASKS + experts + 1, experts < E, other=2147483647)
        expert = tl.sum(((experts < E) & (task >= task_ends)).to(tl.int32), 0)
        expert_task_start = tl.load(TASKS + expert)
        local_start = (task - expert_task_start) * BM
        row_base = tl.load(OFFSETS + expert)
        row_count = tl.load(COUNTS + expert)

        local_rows = local_start + tl.arange(0, BM)
        rows = row_base.to(tl.int64) + local_rows.to(tl.int64)
        cols = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
        ks = tl.arange(0, BK).to(tl.int64)

        gate_acc = tl.full((BM, BN), 0, tl.float32)
        up_acc = tl.full((BM, BN), 0, tl.float32)
        expert_base = expert.to(tl.int64) * (2 * I * H)
        for k_start in range(tl.cdiv(H, BK)):
            k = k_start * BK + ks
            a = tl.load(
                A + rows[:, None] * H + k[None, :],
                (local_rows[:, None] < row_count) & (k[None, :] < H),
                other=0,
            )
            gate_w = tl.load(
                W + expert_base + cols[None, :] * H + k[:, None],
                (cols[None, :] < I) & (k[:, None] < H),
                other=0,
            )
            up_w = tl.load(
                W + expert_base + (I + cols[None, :]) * H + k[:, None],
                (cols[None, :] < I) & (k[:, None] < H),
                other=0,
            )
            gate_acc = tl.dot(a, gate_w, gate_acc)
            up_acc = tl.dot(a, up_w, up_acc)

        # Match the baseline stores and reloads: BF16 round each projection,
        # then promote those rounded values for the FP32 SwiGLU calculation.
        gate = gate_acc.to(tl.bfloat16).to(tl.float32)
        up = up_acc.to(tl.bfloat16).to(tl.float32)
        activation = (gate / (1.0 + tl.exp(-gate))) * up
        tl.store(
            ACT + rows[:, None] * I + cols[None, :],
            activation,
            (local_rows[:, None] < row_count) & (cols[None, :] < I),
        )


@dataclass
class PreparedFusedMoE:
    """Fused local gate/up+SwiGLU path over an owned PreparedMoE workspace."""

    base: PreparedMoE
    kernel: object = field(default=None, init=False)

    def run(self):
        if self.base.x.shape[0] == 0:
            return self.base.output

        base = self.base
        config = base.config
        tokens, hidden = base.x.shape
        experts, _, intermediate = base.down.shape
        topk = base.ids.shape[1]
        routed_rows = tokens * topk
        task_bound = triton.cdiv(routed_rows, config.block_m) + experts - 1

        # Use the same device and current stream as the base workspace. No
        # tensor allocation, metadata readback, or host-dependent route choice
        # occurs in this run method.
        with torch.cuda.device(base.x.device):
            base.dispatch()
            self.kernel = _dynamic_grouped_gate_up_swiglu[
                (task_bound, triton.cdiv(intermediate, config.block_n))
            ](
                base.grouped_input,
                base.gate_up,
                base.activated,
                base.offsets,
                base.counts,
                base.tasks,
                experts,
                intermediate,
                hidden,
                triton.next_power_of_2(experts),
                config.block_m,
                config.block_n,
                config.block_k,
                num_warps=config.warps,
                num_stages=config.stages,
            )
            base.down_project()
            base.combine()
        return base.output

    def check_inputs(self):
        """Delegate explicit synchronizing runtime-value checks to the base."""
        return self.base.check_inputs()


def prepare_fused_moe(x, gate_up, down, ids, routing_weights):
    """Prepare fixed buffers and metadata for gate/up+SwiGLU fusion.

    The base prepare validates tensor metadata and owns all input/workspace
    buffers. It may execute the unfused baseline once for compilation/warmup;
    subsequent ``run`` calls use the fused kernel.
    """
    base = prepare_moe(x, gate_up, down, ids, routing_weights)
    expected = GemmConfig()
    actual = base.config
    if (actual.block_m, actual.block_n, actual.block_k, actual.warps, actual.stages) != (
        expected.block_m, expected.block_n, expected.block_k, expected.warps, expected.stages
    ):
        raise RuntimeError("fused MoE supports only the fixed 32/64/32/4/3 tile configuration")
    return PreparedFusedMoE(base)
