"""Ranked-route dispatch that combines M4 counting and position assignment.

The histogram's atomic increment returns each route's expert-local row rank.
After the existing prefix kernel computes expert offsets, one scatter kernel
uses those ranks to populate both the global route map and grouped activations.
The M4 GEMM, SwiGLU, down projection, and combine kernels remain unchanged.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .moe import PreparedMoE
from .moe_prepared import _launch, prepare_moe_launcher


@triton.jit
def _rank_histogram(IDS, RW, COUNTS, STATUS, RANKS, R: tl.constexpr,
                    E: tl.constexpr, B: tl.constexpr):
    q = tl.program_id(0) * B + tl.arange(0, B)
    q64 = q.to(tl.int64)
    ids = tl.load(IDS + q64, q < R, other=0)
    rw = tl.load(RW + q64, q < R, other=0)

    valid_id = (ids >= 0) & (ids < E)
    finite_weight = (rw == rw) & (tl.abs(rw) < float("inf"))
    invalid_id = tl.sum(((q < R) & ~valid_id).to(tl.int32), 0) > 0
    invalid_weight = tl.sum(((q < R) & ~finite_weight).to(tl.int32), 0) > 0
    status = invalid_id.to(tl.int32) | (invalid_weight.to(tl.int32) * 2)
    tl.atomic_or(STATUS, status)

    safe_id = tl.where(valid_id, ids, 0).to(tl.int64)
    local_rank = tl.atomic_add(COUNTS + safe_id, 1,
                               (q < R) & valid_id, sem="relaxed")
    tl.store(RANKS + q64,
             tl.where(valid_id, local_rank, -1).to(tl.int32), q < R)


@triton.jit
def _rank_dispatch(X, IDS, RANKS, OFFSETS, MAP, A, R: tl.constexpr,
                   H: tl.constexpr, TOPK: tl.constexpr, E: tl.constexpr,
                   BR: tl.constexpr, BH: tl.constexpr):
    q = tl.program_id(0) * BR + tl.arange(0, BR)
    q64 = q.to(tl.int64)
    h = (tl.program_id(1) * BH + tl.arange(0, BH)).to(tl.int64)

    ids = tl.load(IDS + q64, q < R, other=0)
    ranks = tl.load(RANKS + q64, q < R, other=-1)
    valid_id = (ids >= 0) & (ids < E)
    safe_id = tl.where(valid_id, ids, 0).to(tl.int64)
    offsets = tl.load(OFFSETS + safe_id, q < R, other=0)
    pos = offsets + ranks
    valid = (q < R) & valid_id & (ranks >= 0)

    # One column tile owns each MAP entry. Other column tiles use RANKS and
    # OFFSETS directly, so MAP is never read while another CTA writes it.
    if tl.program_id(1) == 0:
        tl.store(MAP + q64, tl.where(valid, pos, -1).to(tl.int32), q < R)

    token = (q // TOPK).to(tl.int64)
    values = tl.load(X + token[:, None] * H + h[None, :],
                     valid[:, None] & (h[None, :] < H), other=0)
    tl.store(A + pos.to(tl.int64)[:, None] * H + h[None, :], values,
             valid[:, None] & (h[None, :] < H))


@dataclass(frozen=True)
class PreparedRankedMoE:
    """Prepared 7-launch M4 sequence with per-route local-rank workspace."""

    base: PreparedMoE
    ranks: torch.Tensor
    launches: tuple

    @property
    def workspace_bytes(self) -> int:
        rank_bytes = self.ranks.numel() * self.ranks.element_size()
        return self.base.workspace_bytes + rank_bytes

    def run(self):
        if self.base.x.shape[0] == 0:
            return self.base.output

        if torch.cuda.current_device() == self.base.x.device.index:
            for launch in self.launches:
                launch.run()
        else:
            with torch.cuda.device(self.base.x.device):
                for launch in self.launches:
                    launch.run()
        return self.base.output

    def check_inputs(self):
        """Run the base workspace's explicit synchronizing input check."""
        return self.base.check_inputs()


def prepare_ranked_moe_launcher(x, gate_up, down, ids, routing_weights):
    """Prepare the ranked dispatch variant over the unchanged fused M4 path.

    This calls the existing M4 preparer to validate buffers and obtain its
    compiled prefix, fused GEMM/activation, down, and combine runners. The
    ranked dispatch adds one int32 value per route, replaces the old cursor
    scatter with returned ranks, and keeps all existing M4 workspaces alive.
    """
    original = prepare_moe_launcher(x, gate_up, down, ids, routing_weights,
                                    fused=True)
    base = original.base
    ranks = torch.empty((base.ids.numel(), ), device=base.ids.device,
                        dtype=torch.int32)

    if base.x.shape[0] == 0:
        return PreparedRankedMoE(base=base, ranks=ranks, launches=())

    if len(original.launches) != 8:
        raise RuntimeError("unexpected fused M4 launch sequence")

    _, hidden = base.x.shape
    experts = base.down.shape[0]
    topk = base.ids.shape[1]
    routed_rows = base.ids.numel()
    br, bh = 16, 128

    reset_launch = original.launches[0]
    prefix_launch = original.launches[2]
    fused_compute_launches = original.launches[5:]

    def initialize_ranked_path():
        reset_launch.run()
        rank_histogram_kernel = _rank_histogram[
            (triton.cdiv(routed_rows, 256), )
        ](base.ids, base.routing_weights, base.counts, base.status, ranks,
          routed_rows, experts, 256)
        prefix_launch.run()
        rank_dispatch_kernel = _rank_dispatch[
            (triton.cdiv(routed_rows, br), triton.cdiv(hidden, bh))
        ](base.x, base.ids, ranks, base.offsets, base.mapping,
          base.grouped_input, routed_rows, hidden, topk, experts, br, bh)
        for launch in fused_compute_launches:
            launch.run()
        return rank_histogram_kernel, rank_dispatch_kernel

    if torch.cuda.current_device() == base.x.device.index:
        rank_histogram_kernel, rank_dispatch_kernel = initialize_ranked_path()
    else:
        with torch.cuda.device(base.x.device):
            rank_histogram_kernel, rank_dispatch_kernel = initialize_ranked_path()

    histogram_grid = (triton.cdiv(routed_rows, 256), )
    dispatch_grid = (triton.cdiv(routed_rows, br), triton.cdiv(hidden, bh))
    histogram_args = (base.ids.data_ptr(), base.routing_weights.data_ptr(),
                      base.counts.data_ptr(), base.status.data_ptr(),
                      ranks.data_ptr(), routed_rows, experts, 256)
    dispatch_args = (base.x.data_ptr(), base.ids.data_ptr(), ranks.data_ptr(),
                     base.offsets.data_ptr(), base.mapping.data_ptr(),
                     base.grouped_input.data_ptr(), routed_rows, hidden, topk,
                     experts, br, bh)

    launches = (
        reset_launch,
        _launch("rank_histogram", rank_histogram_kernel, histogram_grid,
                histogram_args),
        prefix_launch,
        _launch("rank_dispatch", rank_dispatch_kernel, dispatch_grid,
                dispatch_args),
        *fused_compute_launches,
    )
    return PreparedRankedMoE(base=base, ranks=ranks, launches=launches)
