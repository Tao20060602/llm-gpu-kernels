"""BF16 tiled GEMM and host-planned ragged grouped GEMM on Ampere.

Original minimal implementation. Background: Triton v3.3.1 matrix multiplication
and grouped GEMM tutorials; this is an experimental baseline, not a new algorithm.
"""
from dataclasses import dataclass
import operator
import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class GemmConfig:
    block_m: int = 32
    block_n: int = 64
    block_k: int = 32
    warps: int = 4
    stages: int = 3

    def __post_init__(self):
        if any(x < 16 or x & (x - 1) for x in (self.block_m, self.block_n, self.block_k)):
            raise ValueError("tile dimensions must be powers of two >= 16")
        if self.warps not in (4, 8) or self.stages not in (2, 3, 4):
            raise ValueError("supported warps: 4/8; stages: 2/3/4")


@triton.jit
def _dense(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    cols = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
    ks = tl.arange(0, BK).to(tl.int64)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + ks
        a = tl.load(A + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(B + k[:, None] * N + cols[None, :],
                    (k[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    tl.store(C + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _grouped(A, W, C, OFFSETS, COUNTS, EXPERTS, STARTS,
             N: tl.constexpr, K: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    task = tl.program_id(0)
    expert = tl.load(EXPERTS + task)
    base = tl.load(OFFSETS + expert)
    count = tl.load(COUNTS + expert)
    local_rows = tl.load(STARTS + task) + tl.arange(0, BM)
    rows = base.to(tl.int64) + local_rows.to(tl.int64)
    cols = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
    ks = tl.arange(0, BK).to(tl.int64)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + ks
        a = tl.load(A + rows[:, None] * K + k[None, :],
                    (local_rows[:, None] < count) & (k[None, :] < K), other=0)
        w = tl.load(W + expert.to(tl.int64) * N * K + cols[None, :] * K + k[:, None],
                    (cols[None, :] < N) & (k[:, None] < K), other=0)
        acc = tl.dot(a, w, acc)
    tl.store(C + rows[:, None] * N + cols[None, :], acc,
             (local_rows[:, None] < count) & (cols[None, :] < N))


@dataclass(frozen=True)
class GroupedPlan:
    """Host-known expert counts; metadata built outside kernel timing.

    No live GPU router or tensor value readback is hidden in this API.
    Rebuild when counts or block_m change; creation/first use stay on one stream.
    """
    counts: tuple[int, ...]
    block_m: int
    offsets: torch.Tensor
    device_counts: torch.Tensor
    experts: torch.Tensor
    starts: torch.Tensor

    @classmethod
    def build(cls, counts, device, block_m=32):
        counts = tuple(operator.index(x) for x in counts)
        if not counts or any(x < 0 for x in counts) or sum(counts) >= 2**31:
            raise ValueError("nonempty nonnegative counts with total < 2**31 required")
        GemmConfig(block_m=block_m)
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("plan metadata must be on CUDA")
        offsets, experts, starts, total = [], [], [], 0
        for e, count in enumerate(counts):
            offsets.append(total)
            total += count
            for start in range(0, count, block_m):
                experts.append(e)
                starts.append(start)
        data = [torch.tensor(x, dtype=torch.int32, device=device)
                for x in (offsets, counts, experts, starts)]
        return cls(counts, block_m, *data)


def _input(tensor, ndim):
    if tensor.ndim != ndim or tensor.dtype != torch.bfloat16 or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"expected contiguous CUDA BF16 tensor with {ndim} dimensions")


def _output(out, shape, device, inputs):
    if out is None:
        return torch.empty(shape, device=device, dtype=torch.bfloat16)
    _input(out, 2)
    if tuple(out.shape) != shape or out.device != device:
        raise ValueError("output shape/device mismatch")
    if any(out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in inputs):
        raise ValueError("output must not alias inputs or plan metadata")
    return out


def dense_gemm(a, b, out=None, config=GemmConfig(), return_kernel=False):
    _input(a, 2)
    _input(b, 2)
    m, k = a.shape
    kb, n = b.shape
    if k != kb or k == 0 or n == 0 or a.device != b.device:
        raise ValueError("incompatible GEMM shapes/devices; N and K must be positive")
    out = _output(out, (m, n), a.device, (a, b))
    kernel = None
    if m:
        with torch.cuda.device(a.device):
            kernel = _dense[(triton.cdiv(m, config.block_m), triton.cdiv(n, config.block_n))](
                a, b, out, m, n, k, config.block_m, config.block_n, config.block_k,
                num_warps=config.warps, num_stages=config.stages)
    return (out, kernel) if return_kernel else out


def grouped_gemm(a, weights, plan, out=None, config=GemmConfig(), return_kernel=False):
    _input(a, 2)
    _input(weights, 3)
    e, n, k = weights.shape
    metadata = (plan.offsets, plan.device_counts, plan.experts, plan.starts)
    if (a.shape != (sum(plan.counts), k) or len(plan.counts) != e or n == 0 or k == 0
            or config.block_m != plan.block_m or weights.device != a.device
            or any(t.device != a.device for t in metadata)):
        raise ValueError("incompatible grouped shapes, metadata devices or tile plan")
    out = _output(out, (a.shape[0], n), a.device, (a, weights, *metadata))
    kernel = None
    if plan.experts.numel():
        with torch.cuda.device(a.device):
            kernel = _grouped[(plan.experts.numel(), triton.cdiv(n, config.block_n))](
                a, weights, out, *metadata, n, k, config.block_m, config.block_n, config.block_k,
                num_warps=config.warps, num_stages=config.stages)
    return (out, kernel) if return_kernel else out
