"""Unified logical-KV traversal for the fixed four-segment BF16 GQA case.

This candidate keeps the same online-softmax arithmetic as M9 but replaces its
four inlined segment loops with one logical-key loop and per-lane pointer
selection. It neither concatenates KV nor expands grouped-query heads.
"""
from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl

from .attention_segmented import (
    AttentionConfig,
    HEAD_DIM,
    KV_HEADS,
    QUERY_HEADS,
    SEGMENT_COUNT,
    _validate,
)


LOG2E = 1.4426950408889634
SOFTMAX_SCALE_LOG2 = (1.0 / math.sqrt(HEAD_DIM)) * LOG2E
SUPPORTED_TOKEN_TILES = (16, 32)
SUPPORTED_HEAD_GROUPS = (1, 2, 4)
SUPPORTED_WARPS = (4, 8)
KEY_TILE = 64
PIPELINE_STAGES = 2


@triton.jit
def _unified_attention_fwd(
    Q,
    K0,
    V0,
    K1,
    V1,
    K2,
    V2,
    K3,
    V3,
    O,
    T: tl.constexpr,
    L0: tl.constexpr,
    L1: tl.constexpr,
    L2: tl.constexpr,
    L3: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TOKEN_TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUERY_HEAD_COUNT: tl.constexpr,
    KV_HEAD_COUNT: tl.constexpr,
    HEAD_DIM_SIZE: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
):
    """One CTA computes a token tile and a contiguous group of Q heads."""
    token_start = tl.program_id(0) * TOKEN_TILE_SIZE
    q_group = tl.program_id(1)
    row = tl.arange(0, BLOCK_M)
    d = tl.arange(0, HEAD_DIM_SIZE)

    # Token-major flattening: row = token_lane * G + group_lane.
    q_position = token_start + row // GROUP_SIZE
    group_lane = row % GROUP_SIZE
    q_head_base = q_group * GROUP_SIZE
    q_head = q_head_base + group_lane
    q_valid = q_position < T
    kv_head = q_head_base // (QUERY_HEAD_COUNT // KV_HEAD_COUNT)

    q_offset = (
        q_position.to(tl.int64) * QUERY_HEAD_COUNT
        + q_head.to(tl.int64)
    ) * HEAD_DIM_SIZE
    q = tl.load(
        Q + q_offset[:, None] + d[None, :].to(tl.int64),
        q_valid[:, None] & (d[None, :] < HEAD_DIM_SIZE),
        other=0,
    )

    history_length = L0 + L1 + L2
    total_length = history_length + L3
    # All history is consumed. The final segment is causal and only needs keys
    # through the largest query position owned by this CTA.
    stop = history_length + tl.minimum(L3, token_start + TOKEN_TILE_SIZE)
    block_count = tl.cdiv(stop, BLOCK_N)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM_SIZE), tl.float32)
    offs_n = tl.arange(0, BLOCK_N)

    l01 = L0 + L1
    l012 = history_length
    for block_index in range(block_count):
        logical_key = block_index * BLOCK_N + offs_n
        key_valid = logical_key < total_length

        # Map each logical key lane to exactly one physical segment. Local
        # positions are nonnegative for all lanes; invalid tail lanes remain
        # masked at load. Pointer selection avoids packing or copying KV.
        in0 = logical_key < L0
        in1 = (logical_key >= L0) & (logical_key < l01)
        in2 = (logical_key >= l01) & (logical_key < l012)
        key_pointer = tl.where(in0, K0, tl.where(in1, K1, tl.where(in2, K2, K3)))
        value_pointer = tl.where(in0, V0, tl.where(in1, V1, tl.where(in2, V2, V3)))
        local_key = tl.where(
            in0,
            logical_key,
            tl.where(in1, logical_key - L0, tl.where(in2, logical_key - l01, logical_key - l012)),
        )
        key_base = (
            local_key.to(tl.int64) * KV_HEAD_COUNT + kv_head.to(tl.int64)
        ) * HEAD_DIM_SIZE
        k = tl.load(
            key_pointer[None, :]
            + key_base[None, :]
            + d[:, None].to(tl.int64),
            key_valid[None, :] & (d[:, None] < HEAD_DIM_SIZE),
            other=0,
        )
        v = tl.load(
            value_pointer[:, None]
            + key_base[:, None]
            + d[None, :].to(tl.int64),
            key_valid[:, None] & (d[None, :] < HEAD_DIM_SIZE),
            other=0,
        )

        visible_history = logical_key < history_length
        visible_current = (
            (logical_key >= history_length)
            & ((logical_key - history_length)[None, :] <= q_position[:, None])
        )
        visible = visible_history[None, :] | visible_current
        valid = q_valid[:, None] & key_valid[None, :] & visible

        scores = tl.dot(q, k) * SCALE_LOG2
        masked_scores = tl.where(valid, scores, float("-inf"))
        block_max = tl.max(masked_scores, axis=1)
        m_candidate = tl.maximum(m_i, block_max)
        block_active = tl.sum(valid.to(tl.int32), axis=1) > 0

        # Empty causal tiles and padded Q rows leave the shared online state
        # unchanged; this prevents the -inf - -inf NaN case.
        m_safe = tl.where(
            block_active,
            m_candidate,
            tl.where(l_i > 0.0, m_i, 0.0),
        )
        p = tl.where(valid, tl.math.exp2(scores - m_safe[:, None]), 0.0)
        alpha = tl.where(block_active, tl.math.exp2(m_i - m_safe), 1.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        m_i = tl.where(block_active, m_candidate, m_i)

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    output = (acc / safe_l[:, None]).to(tl.bfloat16)
    out_offset = (
        q_position.to(tl.int64) * QUERY_HEAD_COUNT
        + q_head.to(tl.int64)
    ) * HEAD_DIM_SIZE
    tl.store(
        O + out_offset[:, None] + d[None, :].to(tl.int64),
        output,
        q_valid[:, None] & (d[None, :] < HEAD_DIM_SIZE),
    )


@dataclass(frozen=True)
class PreparedUnifiedAttention:
    """Owns fixed inputs, BF16 output and one compiled current-stream runner."""

    q: torch.Tensor
    segments: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    output: torch.Tensor
    config: AttentionConfig
    grid: tuple[int, int, int]
    kernel: object | None
    _runner: object | None
    _args: tuple

    @property
    def workspace_bytes(self) -> int:
        """Output storage only; the kernel has no scratch or packed KV."""
        return self.output.numel() * self.output.element_size()

    def run(self) -> torch.Tensor:
        """Launch cached pointer arguments on the caller's current stream."""
        if self.kernel is None:
            return self.output
        if torch.cuda.current_device() == self.q.device.index:
            self._runner(*self._args)
        else:
            with torch.cuda.device(self.q.device):
                self._runner(*self._args)
        return self.output


def _validate_candidate(q, segments, config):
    fixed_segments = _validate(q, segments, config)
    if (config.token_tile not in SUPPORTED_TOKEN_TILES
            or config.key_tile != KEY_TILE
            or config.head_group not in SUPPORTED_HEAD_GROUPS
            or config.warps not in SUPPORTED_WARPS
            or config.stages != PIPELINE_STAGES):
        raise ValueError(
            "M10 candidates are token_tile={16,32}, key_tile=64, "
            "head_group={1,2,4}, warps={4,8}, stages=2"
        )
    return fixed_segments


def prepare_unified_kernel(
    q,
    segments,
    config=AttentionConfig(),
) -> PreparedUnifiedAttention:
    """Validate, allocate, compile/warm and prepare a unified-KV launch."""
    fixed_segments = _validate_candidate(q, segments, config)
    head_group = config.head_group
    output = torch.empty_like(q)
    tokens = q.shape[0]
    grid = (triton.cdiv(tokens, config.token_tile), QUERY_HEADS // head_group, 1)
    if tokens == 0:
        return PreparedUnifiedAttention(
            q=q,
            segments=fixed_segments,
            output=output,
            config=config,
            grid=(0, QUERY_HEADS // head_group, 1),
            kernel=None,
            _runner=None,
            _args=(),
        )

    lengths = tuple(key.shape[0] for key, _ in fixed_segments)
    block_m = config.effective_block_m
    scale_log2 = SOFTMAX_SCALE_LOG2
    warm_args = (
        q,
        fixed_segments[0][0], fixed_segments[0][1],
        fixed_segments[1][0], fixed_segments[1][1],
        fixed_segments[2][0], fixed_segments[2][1],
        fixed_segments[3][0], fixed_segments[3][1],
        output,
        tokens, *lengths, head_group, config.token_tile, block_m, config.key_tile,
        QUERY_HEADS, KV_HEADS, HEAD_DIM, scale_log2,
    )
    with torch.cuda.device(q.device):
        kernel = _unified_attention_fwd[grid](
            *warm_args,
            num_warps=config.warps,
            num_stages=config.stages,
        )
        args = (
            q.data_ptr(),
            fixed_segments[0][0].data_ptr(), fixed_segments[0][1].data_ptr(),
            fixed_segments[1][0].data_ptr(), fixed_segments[1][1].data_ptr(),
            fixed_segments[2][0].data_ptr(), fixed_segments[2][1].data_ptr(),
            fixed_segments[3][0].data_ptr(), fixed_segments[3][1].data_ptr(),
            output.data_ptr(),
            tokens, *lengths, head_group, config.token_tile, block_m, config.key_tile,
            QUERY_HEADS, KV_HEADS, HEAD_DIM, scale_log2,
        )
        if len(kernel.src.signature) != len(args):
            raise RuntimeError("unexpected Triton 3.3.1 unified launcher signature")
        runner = kernel[grid]

    return PreparedUnifiedAttention(
        q=q,
        segments=fixed_segments,
        output=output,
        config=config,
        grid=grid,
        kernel=kernel,
        _runner=runner,
        _args=args,
    )
