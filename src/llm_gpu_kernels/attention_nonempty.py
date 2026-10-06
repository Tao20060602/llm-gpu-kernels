"""M15 four-segment attention with a contract-proven nonempty recurrence.

Padded Q rows use zero Q and compute an internal softmax; their loads and
stores remain masked. For T>0, every CTA has a positive current length and
its first executed block has a visible key for every logical row. Thus m
becomes finite and l positive before any later all-masked causal block.
T=0 never launches. Inputs are finite as in the numerical experiment.
"""
import math

import torch
import triton
import triton.language as tl

from .attention_segmented import AttentionConfig, PreparedSegmentedAttention, _validate


QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
SEGMENT_COUNT = 4
TOKEN_TILE = 32
KEY_TILE = 64
WARPS = 4
STAGES = 2
LOG2E = 1.4426950408889634
SOFTMAX_SCALE_LOG2 = (1.0 / math.sqrt(HEAD_DIM)) * LOG2E


@triton.jit
def _nonempty_update_segment(
    q,
    q_position,
    kv_head,
    key_ptr,
    value_ptr,
    segment_length: tl.constexpr,
    is_current: tl.constexpr,
    q_block_start,
    m_i,
    l_i,
    acc,
    token_tile: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    query_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale_log2: tl.constexpr,
):
    """Fold one segment into a shared per-query FP32 softmax state."""
    if is_current:
        stop = tl.minimum(segment_length, q_block_start + token_tile)
    else:
        stop = segment_length

    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, head_dim)
    if is_current:
        block_count = tl.cdiv(stop, block_n)
    else:
        block_count = tl.cdiv(segment_length, block_n)

    for block_index in range(block_count):
        key_position = block_index * block_n + offs_n
        key_valid = key_position < segment_length
        # Do not mask padded Q rows inside the recurrence. Q loads are zero
        # and final stores remain masked; each first block is nonempty.
        valid = key_valid[None, :]
        if is_current:
            valid = valid & (key_position[None, :] <= q_position[:, None])

        # [D, BN] K and [BN, D] V; segment strides are fixed contiguous
        # [L, KV_HEADS, HEAD_DIM], and offsets use int64 pointer arithmetic.
        key_base = (key_position.to(tl.int64) * kv_heads + kv_head.to(tl.int64)) * head_dim
        k = tl.load(
            key_ptr + key_base[None, :] + offs_d[:, None].to(tl.int64),
            key_valid[None, :] & (offs_d[:, None] < head_dim),
            other=0,
        )
        v = tl.load(
            value_ptr + key_base[:, None] + offs_d[None, :].to(tl.int64),
            key_valid[:, None] & (offs_d[None, :] < head_dim),
            other=0,
        )

        scores = tl.dot(q, k) * softmax_scale_log2
        masked_scores = tl.where(valid, scores, float("-inf"))
        block_max = tl.max(masked_scores, axis=1)
        m_new = tl.maximum(m_i, block_max)
        # First block: m_new is finite. Later all-masked current blocks keep
        # finite m_i, so exp2(-inf - m_new)=0 and alpha=1 are safe.
        p = tl.math.exp2(masked_scores - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        m_i = m_new

    return m_i, l_i, acc


@triton.jit
def _nonempty_attention_fwd(
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
    """One CTA per query-token tile and contiguous group of G query heads."""
    token_start = tl.program_id(0) * TOKEN_TILE_SIZE
    q_group = tl.program_id(1)
    row = tl.arange(0, BLOCK_M)
    d = tl.arange(0, HEAD_DIM_SIZE)

    # Flatten token-major rows: row = token_lane * G + group_lane.
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

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM_SIZE), tl.float32)

    # One shared normalization state spans selected, sink, recent, and current.
    m_i, l_i, acc = _nonempty_update_segment(
        q, q_position, kv_head, K0, V0, L0, False, token_start,
        m_i, l_i, acc, TOKEN_TILE_SIZE, BLOCK_M, BLOCK_N,
        QUERY_HEAD_COUNT, KV_HEAD_COUNT, HEAD_DIM_SIZE, SCALE_LOG2,
    )
    m_i, l_i, acc = _nonempty_update_segment(
        q, q_position, kv_head, K1, V1, L1, False, token_start,
        m_i, l_i, acc, TOKEN_TILE_SIZE, BLOCK_M, BLOCK_N,
        QUERY_HEAD_COUNT, KV_HEAD_COUNT, HEAD_DIM_SIZE, SCALE_LOG2,
    )
    m_i, l_i, acc = _nonempty_update_segment(
        q, q_position, kv_head, K2, V2, L2, False, token_start,
        m_i, l_i, acc, TOKEN_TILE_SIZE, BLOCK_M, BLOCK_N,
        QUERY_HEAD_COUNT, KV_HEAD_COUNT, HEAD_DIM_SIZE, SCALE_LOG2,
    )
    m_i, l_i, acc = _nonempty_update_segment(
        q, q_position, kv_head, K3, V3, L3, True, token_start,
        m_i, l_i, acc, TOKEN_TILE_SIZE, BLOCK_M, BLOCK_N,
        QUERY_HEAD_COUNT, KV_HEAD_COUNT, HEAD_DIM_SIZE, SCALE_LOG2,
    )

    output = (acc / l_i[:, None]).to(tl.bfloat16)
    out_offset = (
        q_position.to(tl.int64) * QUERY_HEAD_COUNT
        + q_head.to(tl.int64)
    ) * HEAD_DIM_SIZE
    tl.store(
        O + out_offset[:, None] + d[None, :].to(tl.int64),
        output,
        q_valid[:, None] & (d[None, :] < HEAD_DIM_SIZE),
    )


def prepare_nonempty_kernel(
    q,
    segments,
    config=AttentionConfig(),
) -> PreparedSegmentedAttention:
    """Validate, allocate output, compile/warm and cache one M15 attention launch.

    The first configuration is token tile 32, key tile 64, four warps, two
    stages, and head group one. Head groups two and four are exposed for
    correctness comparison; they have larger effective row tiles and are not
    predeclared performance winners.
    """
    fixed_segments = _validate(q, segments, config)
    head_group = config.head_group
    output = torch.empty_like(q)
    tokens = q.shape[0]
    grid = (triton.cdiv(tokens, config.token_tile), QUERY_HEADS // head_group, 1)
    if tokens == 0:
        return PreparedSegmentedAttention(
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
        kernel = _nonempty_attention_fwd[grid](
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
            raise RuntimeError("unexpected Triton 3.3.1 attention launcher signature")
        runner = kernel[grid]

    return PreparedSegmentedAttention(
        q=q,
        segments=fixed_segments,
        output=output,
        config=config,
        grid=grid,
        kernel=kernel,
        _runner=runner,
        _args=args,
    )
