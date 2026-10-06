"""Minimal one-kernel four-segment BF16 GQA attention prototype.

The online-softmax recurrence follows the official Triton 3.3.1 fused
attention tutorial.  This prototype's specific scope is reading the four KV
segments directly without packing them or expanding the KV heads.
"""
from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl


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
def _online_update_segment(
    q,
    q_position,
    q_valid,
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
        valid = q_valid[:, None] & key_valid[None, :]
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
        m_candidate = tl.maximum(m_i, block_max)
        block_active = tl.sum(valid.to(tl.int32), axis=1) > 0

        # Keep empty causal blocks and padded Q rows inert. Without this guard,
        # the first all-masked tile can evaluate -inf - -inf and poison state.
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

    return m_i, l_i, acc


@triton.jit
def _segmented_attention_fwd(
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
    m_i, l_i, acc = _online_update_segment(
        q, q_position, q_valid, kv_head, K0, V0, L0, False, token_start,
        m_i, l_i, acc, TOKEN_TILE_SIZE, BLOCK_M, BLOCK_N,
        QUERY_HEAD_COUNT, KV_HEAD_COUNT, HEAD_DIM_SIZE, SCALE_LOG2,
    )
    m_i, l_i, acc = _online_update_segment(
        q, q_position, q_valid, kv_head, K1, V1, L1, False, token_start,
        m_i, l_i, acc, TOKEN_TILE_SIZE, BLOCK_M, BLOCK_N,
        QUERY_HEAD_COUNT, KV_HEAD_COUNT, HEAD_DIM_SIZE, SCALE_LOG2,
    )
    m_i, l_i, acc = _online_update_segment(
        q, q_position, q_valid, kv_head, K2, V2, L2, False, token_start,
        m_i, l_i, acc, TOKEN_TILE_SIZE, BLOCK_M, BLOCK_N,
        QUERY_HEAD_COUNT, KV_HEAD_COUNT, HEAD_DIM_SIZE, SCALE_LOG2,
    )
    m_i, l_i, acc = _online_update_segment(
        q, q_position, q_valid, kv_head, K3, V3, L3, True, token_start,
        m_i, l_i, acc, TOKEN_TILE_SIZE, BLOCK_M, BLOCK_N,
        QUERY_HEAD_COUNT, KV_HEAD_COUNT, HEAD_DIM_SIZE, SCALE_LOG2,
    )

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
class AttentionConfig:
    """Fixed M9 launch configuration for the selected GQA grouping."""

    token_tile: int = TOKEN_TILE
    key_tile: int = KEY_TILE
    head_group: int = 1
    warps: int = WARPS
    stages: int = STAGES

    @property
    def effective_block_m(self) -> int:
        return self.token_tile * self.head_group

    @property
    def group_size(self) -> int:
        """Alias used by the kernel's constexpr name."""
        return self.head_group


@dataclass(frozen=True)
class PreparedSegmentedAttention:
    """Owns fixed input references, BF16 output, compiled kernel and runner.

    Input values can change in place. Tensor shapes, addresses, and segment
    order must stay fixed. Callers establish cross-stream dependencies and do
    not run the same output workspace concurrently.
    """

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
        """Owned BF16 output bytes; the kernel allocates no scratch workspace."""
        return self.output.numel() * self.output.element_size()

    def run(self) -> torch.Tensor:
        """Launch the cached kernel on the caller's current CUDA stream."""
        if self.kernel is None:
            return self.output
        if torch.cuda.current_device() == self.q.device.index:
            self._runner(*self._args)
        else:
            with torch.cuda.device(self.q.device):
                self._runner(*self._args)
        return self.output


def _validate(q, segments, config):
    if triton.__version__ != "3.3.1":
        raise RuntimeError("M9 prepared attention requires Triton 3.3.1")
    if not isinstance(config, AttentionConfig):
        raise TypeError("config must be an AttentionConfig")
    if config.head_group not in (1, 2, 4):
        raise ValueError("head_group must be one of 1, 2, or 4")
    for name, value in (("token_tile", config.token_tile), ("key_tile", config.key_tile)):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a positive power of two")
    if config.effective_block_m < 16 or config.key_tile < 16:
        raise ValueError("effective row and key tiles must each be at least 16")
    if config.warps not in (1, 2, 4, 8) or config.stages <= 0:
        raise ValueError("warps must be 1, 2, 4, or 8 and stages must be positive")
    if not isinstance(q, torch.Tensor) or q.ndim != 3:
        raise ValueError("q must have shape [T, 32, 128]")
    if tuple(q.shape[1:]) != (QUERY_HEADS, HEAD_DIM):
        raise ValueError(f"q must have shape [T, {QUERY_HEADS}, {HEAD_DIM}]")
    if not q.is_cuda or q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError("q must be contiguous CUDA BF16")
    if len(segments) != SEGMENT_COUNT:
        raise ValueError(f"exactly {SEGMENT_COUNT} K/V segments are required")

    fixed_segments = []
    for index, pair in enumerate(segments):
        if len(pair) != 2:
            raise ValueError(f"segment {index} must contain a K/V pair")
        key, value = pair
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError(f"segment {index} K and V must be tensors")
        if key.ndim != 3 or tuple(key.shape[1:]) != (KV_HEADS, HEAD_DIM):
            raise ValueError(f"segment {index} K must have shape [L, {KV_HEADS}, {HEAD_DIM}]")
        if value.shape != key.shape:
            raise ValueError(f"segment {index} V shape must match K")
        for name, tensor in (("K", key), ("V", value)):
            if tensor.device != q.device or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                raise ValueError(f"segment {index} {name} must be contiguous CUDA BF16 on q's device")
        fixed_segments.append((key, value))
    if fixed_segments[-1][0].shape[0] != q.shape[0]:
        raise ValueError("current K/V segment length must equal T")
    return tuple(fixed_segments)


def prepare_segmented_kernel(
    q,
    segments,
    config=AttentionConfig(),
) -> PreparedSegmentedAttention:
    """Validate, allocate output, compile/warm and cache one M9 attention launch.

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
        kernel = _segmented_attention_fwd[grid](
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
