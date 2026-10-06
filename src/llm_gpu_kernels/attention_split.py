"""M11 split-KV four-segment attention and stable FP32 partial-state merge."""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from .attention_segmented import AttentionConfig, QUERY_HEADS, KV_HEADS, HEAD_DIM
from .attention_unified import _validate_candidate, SOFTMAX_SCALE_LOG2
from .moe_prepared import _launch

@dataclass(frozen=True)
class SplitAttentionConfig(AttentionConfig):
    token_tile: int = 16
    head_group: int = 4
    warps: int = 8
    splits: int = 4

@triton.jit
def _split_attention_partial(
    Q,
    K0,
    V0,
    K1,
    V1,
    K2,
    V2,
    K3,
    V3,
    PM,
    PL,
    PA,
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
    SPLITS: tl.constexpr,
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
    partition = tl.program_id(2)
    total_blocks = tl.cdiv(total_length, BLOCK_N)
    block_begin = partition * total_blocks // SPLITS
    block_end = tl.minimum((partition + 1) * total_blocks // SPLITS, tl.cdiv(stop, BLOCK_N))

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM_SIZE), tl.float32)
    offs_n = tl.arange(0, BLOCK_N)

    l01 = L0 + L1
    l012 = history_length
    for block_index in range(block_begin, block_end):
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
        # A partition may have no visible keys for a valid Q row.
        m_safe = tl.where(m_candidate == float("-inf"), 0.0, m_candidate)
        p = tl.where(valid, tl.math.exp2(scores - m_safe[:, None]), 0.0)
        alpha = tl.where(l_i > 0.0, tl.math.exp2(m_i - m_safe), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        m_i = tl.where(l_i > 0.0, m_safe, float("-inf"))

    state_row = partition.to(tl.int64) * T * QUERY_HEAD_COUNT + q_position.to(tl.int64) * QUERY_HEAD_COUNT + q_head
    tl.store(PM + state_row, m_i, q_valid)
    tl.store(PL + state_row, l_i, q_valid)
    tl.store(PA + state_row[:, None] * HEAD_DIM_SIZE + d[None, :], acc,
             q_valid[:, None] & (d[None, :] < HEAD_DIM_SIZE))


@triton.jit
def _merge_partial(PM, PL, PA, O, ROWS: tl.constexpr, SPLITS: tl.constexpr,
                   D: tl.constexpr, BLOCK_R: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    partitions = tl.arange(0, SPLITS)
    dims = tl.arange(0, D)
    offsets = partitions[:, None].to(tl.int64) * ROWS + rows[None, :]
    l = tl.load(PL + offsets, rows[None, :] < ROWS, other=0.0)
    m = tl.load(PM + offsets, rows[None, :] < ROWS, other=float("-inf"))
    active = l > 0.0
    maximum = tl.max(tl.where(active, m, float("-inf")), axis=0)
    safe_maximum = tl.where(tl.sum(active.to(tl.int32), axis=0) > 0, maximum, 0.0)
    weights = tl.where(active, tl.math.exp2(m - safe_maximum[None, :]), 0.0)
    denominator = tl.sum(l * weights, axis=0)
    partial = tl.load(PA + offsets[:, :, None] * D + dims[None, None, :],
                      rows[None, :, None] < ROWS, other=0.0)
    numerator = tl.sum(partial * weights[:, :, None], axis=0)
    result = numerator / tl.where(denominator > 0.0, denominator, 1.0)[:, None]
    tl.store(O + rows[:, None].to(tl.int64) * D + dims[None, :],
             result.to(tl.bfloat16), rows[:, None] < ROWS)


@dataclass(frozen=True)
class PreparedSplitAttention:
    q: torch.Tensor
    segments: tuple
    output: torch.Tensor
    config: SplitAttentionConfig
    partial_m: torch.Tensor
    partial_l: torch.Tensor
    partial_acc: torch.Tensor
    launches: tuple

    @property
    def kernel(self):
        return self.launches[0].kernel if self.launches else None

    @property
    def merge_kernel(self):
        return self.launches[1].kernel if self.launches else None

    @property
    def scratch_bytes(self):
        return sum(x.numel()*x.element_size() for x in (self.partial_m,self.partial_l,self.partial_acc))

    @property
    def workspace_bytes(self):
        return self.scratch_bytes + self.output.numel()*self.output.element_size()

    def run(self):
        if not self.launches:return self.output
        if torch.cuda.current_device() == self.q.device.index:
            for launch in self.launches:launch.run()
        else:
            with torch.cuda.device(self.q.device):
                for launch in self.launches:launch.run()
        return self.output


def prepare_split_kernel(q, segments, config=SplitAttentionConfig()):
    if not isinstance(config, SplitAttentionConfig):raise TypeError('config must be SplitAttentionConfig')
    fixed=_validate_candidate(q,segments,config)
    if config.head_group!=4 or config.warps!=8 or config.splits not in (1,2,4,8):
        raise ValueError('M11 candidates require G4/warps8/splits1,2,4,8')
    tokens=q.shape[0];rows=tokens*QUERY_HEADS
    output=torch.empty_like(q)
    pm=torch.empty((config.splits,rows),device=q.device,dtype=torch.float32)
    pl=torch.empty_like(pm)
    pa=torch.empty((config.splits,rows,HEAD_DIM),device=q.device,dtype=torch.float32)
    if tokens==0:return PreparedSplitAttention(q,fixed,output,config,pm,pl,pa,())
    grid=(triton.cdiv(tokens,config.token_tile),QUERY_HEADS//config.head_group,config.splits)
    lengths=tuple(k.shape[0] for k,_ in fixed)
    tensors=(q,*(x for pair in fixed for x in pair),pm,pl,pa)
    constants=(tokens,*lengths,config.head_group,config.token_tile,config.effective_block_m,
               config.key_tile,QUERY_HEADS,KV_HEADS,HEAD_DIM,SOFTMAX_SCALE_LOG2,config.splits)
    with torch.cuda.device(q.device):
        kernel=_split_attention_partial[grid](*tensors,*constants,num_warps=config.warps,num_stages=config.stages)
        merge_grid=(triton.cdiv(rows,4),1,1)
        merge_constants=(rows,config.splits,HEAD_DIM,4)
        merge=_merge_partial[merge_grid](pm,pl,pa,output,*merge_constants,num_warps=4)
        first=_launch('split_attention_partial',kernel,grid,(*(x.data_ptr() for x in tensors),*constants))
        second=_launch('split_attention_merge',merge,merge_grid,(pm.data_ptr(),pl.data_ptr(),pa.data_ptr(),output.data_ptr(),*merge_constants))
    return PreparedSplitAttention(q,fixed,output,config,pm,pl,pa,(first,second))
