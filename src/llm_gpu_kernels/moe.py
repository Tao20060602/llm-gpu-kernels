"""M4 functional GPU fixed-route MoE baseline, with no host counts/readback.

Histogram/prefix/scatter grouping and dynamic task lookup are intentionally
separate stages. No router, communication, backward, or mature-backend claim.
"""
from dataclasses import dataclass, field
import torch
import triton
import triton.language as tl
from .gemm import GemmConfig, _input


@triton.jit
def _reset(COUNTS, STATUS, E: tl.constexpr, BE: tl.constexpr):
    e = tl.arange(0, BE)
    tl.store(COUNTS + e, 0, e < E)
    tl.store(STATUS, 0)


@triton.jit
def _histogram(IDS, RW, COUNTS, STATUS, R: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    q = tl.program_id(0) * B + tl.arange(0, B)
    ids = tl.load(IDS + q, q < R, other=0)
    rw = tl.load(RW + q, q < R, other=0)
    valid_id = (ids >= 0) & (ids < E)
    invalid_id = tl.sum(((q < R) & ~valid_id).to(tl.int32), 0) > 0
    invalid_weight = tl.sum(((q < R) & ~((rw == rw) & (tl.abs(rw) < float("inf")))).to(tl.int32), 0) > 0
    tl.atomic_or(STATUS, invalid_id.to(tl.int32) | (invalid_weight.to(tl.int32) * 2))
    safe_id = tl.where(valid_id, ids, 0)
    tl.atomic_add(COUNTS + safe_id, 1, (q < R) & valid_id, sem="relaxed")


@triton.jit
def _prefix(COUNTS, OFFSETS, CURSORS, TASKS, E: tl.constexpr, BM: tl.constexpr, BE: tl.constexpr):
    e = tl.arange(0, BE)
    counts = tl.load(COUNTS + e, e < E, other=0)
    offsets = tl.cumsum(counts, 0) - counts
    tasks = tl.cdiv(counts, BM)
    task_offsets = tl.cumsum(tasks, 0) - tasks
    tl.store(OFFSETS + e, offsets, e < E)
    tl.store(CURSORS + e, 0, e < E)
    tl.store(TASKS + e, task_offsets, e < E)
    tl.store(TASKS + E, tl.sum(tasks, 0))


@triton.jit
def _positions(IDS, OFFSETS, CURSORS, MAP, R: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    q = tl.program_id(0) * B + tl.arange(0, B)
    ids = tl.load(IDS + q, q < R, other=0)
    valid = (q < R) & (ids >= 0) & (ids < E)
    safe_id = tl.where(valid, ids, 0)
    slot = tl.atomic_add(CURSORS + safe_id, 1, valid, sem="relaxed")
    offset = tl.load(OFFSETS + safe_id)
    tl.store(MAP + q, tl.where(valid, offset + slot, -1), q < R)


@triton.jit
def _dispatch(X, MAP, A, R: tl.constexpr, H: tl.constexpr, TOPK: tl.constexpr,
              BR: tl.constexpr, BH: tl.constexpr):
    q = tl.program_id(0) * BR + tl.arange(0, BR)
    h = (tl.program_id(1) * BH + tl.arange(0, BH)).to(tl.int64)
    pos = tl.load(MAP + q, q < R, other=-1).to(tl.int64)
    token = (q // TOPK).to(tl.int64)
    value = tl.load(X + token[:, None] * H + h[None, :], (q[:, None] < R) & (h[None, :] < H), other=0)
    tl.store(A + pos[:, None] * H + h[None, :], value,
             (q[:, None] < R) & (pos[:, None] >= 0) & (h[None, :] < H))


@triton.jit
def _dynamic_grouped(A, W, C, OFFSETS, COUNTS, TASKS,
                     E: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BE: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    task = tl.program_id(0)
    total_tasks = tl.load(TASKS + E)
    if task < total_tasks:
        es = tl.arange(0, BE)
        ends = tl.load(TASKS + es + 1, es < E, other=2147483647)
        expert = tl.sum(((es < E) & (task >= ends)).to(tl.int32), 0)
        local_start = (task - tl.load(TASKS + expert)) * BM
        base = tl.load(OFFSETS + expert)
        count = tl.load(COUNTS + expert)
        local_rows = local_start + tl.arange(0, BM)
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


@triton.jit
def _swiglu(GU, ACT, R: tl.constexpr, I: tl.constexpr, B: tl.constexpr):
    j = tl.program_id(0) * B + tl.arange(0, B)
    row = (j // I).to(tl.int64)
    col = (j % I).to(tl.int64)
    gate = tl.load(GU + row * (2 * I) + col, j < R * I, other=0).to(tl.float32)
    up = tl.load(GU + row * (2 * I) + I + col, j < R * I, other=0).to(tl.float32)
    value = (gate / (1 + tl.exp(-gate))) * up
    tl.store(ACT + j.to(tl.int64), value, j < R * I)


@triton.jit
def _combine(DOWN, MAP, RW, OUT, T: tl.constexpr, H: tl.constexpr, TOPK: tl.constexpr, BH: tl.constexpr):
    token = tl.program_id(0)
    h = (tl.program_id(1) * BH + tl.arange(0, BH)).to(tl.int64)
    acc = tl.full((BH,), 0, tl.float32)
    for slot in tl.static_range(TOPK):
        q = token * TOPK + slot
        pos = tl.load(MAP + q).to(tl.int64)
        value = tl.load(DOWN + pos * H + h, (h < H) & (pos >= 0), other=0).to(tl.float32)
        weight = tl.load(RW + q)
        # Preserve a fixed slot order; do not permit FP32 atomic combine races.
        acc = acc + value * weight
    tl.store(OUT + token.to(tl.int64) * H + h, acc, h < H)


@dataclass
class PreparedMoE:
    """Fixed buffers/shapes; runtime IDs and values may change in place.

    IDs must be in [0,E), routing weights finite. Status flags invalid values;
    check_inputs is an explicit synchronizing debug operation, never in run.
    Caller establishes stream dependencies and forbids concurrent workspace use.
    """
    x: torch.Tensor
    gate_up: torch.Tensor
    down: torch.Tensor
    ids: torch.Tensor
    routing_weights: torch.Tensor
    output: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor
    cursors: torch.Tensor
    tasks: torch.Tensor
    mapping: torch.Tensor
    grouped_input: torch.Tensor
    projected: torch.Tensor
    activated: torch.Tensor
    down_output: torch.Tensor
    status: torch.Tensor
    config: GemmConfig
    kernels: dict = field(default_factory=dict)

    @property
    def workspace_bytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.counts, self.offsets, self.cursors, self.tasks, self.mapping, self.grouped_input,
                    self.projected, self.activated, self.down_output, self.status))

    def check_inputs(self):
        flags = int(self.status.item())
        if flags:
            raise ValueError(f"invalid runtime MoE inputs: GPU status={flags} (1=IDs, 2=nonfinite weights)")

    def dispatch(self):
        t, h = self.x.shape
        e, _, intermediate = self.down.shape
        topk = self.ids.shape[1]
        r = t * topk
        be = triton.next_power_of_2(e)
        self.kernels["reset"] = _reset[(1,)](self.counts, self.status, e, be)
        self.kernels["histogram"] = _histogram[(triton.cdiv(r, 256),)](
            self.ids, self.routing_weights, self.counts, self.status, r, e, 256)
        self.kernels["prefix"] = _prefix[(1,)](self.counts, self.offsets, self.cursors, self.tasks,
                                               e, self.config.block_m, be)
        self.kernels["positions"] = _positions[(triton.cdiv(r, 256),)](
            self.ids, self.offsets, self.cursors, self.mapping, r, e, 256)
        self.kernels["dispatch"] = _dispatch[(triton.cdiv(r, 16), triton.cdiv(h, 128))](
            self.x, self.mapping, self.grouped_input, r, h, topk, 16, 128)

    def _gemm(self, a, weights, out, name):
        e, n, k = weights.shape
        cfg = self.config
        # Sum ceil(count[e]/BM) <= ceil(R/BM)+E-1. No counts are read by host.
        task_bound = triton.cdiv(a.shape[0], cfg.block_m) + e - 1
        self.kernels[name] = _dynamic_grouped[(task_bound, triton.cdiv(n, cfg.block_n))](
            a, weights, out, self.offsets, self.counts, self.tasks,
            e, n, k, triton.next_power_of_2(e), cfg.block_m, cfg.block_n, cfg.block_k,
            num_warps=cfg.warps, num_stages=cfg.stages)

    def run(self):
        if self.x.shape[0] == 0:
            return self.output
        with torch.cuda.device(self.x.device):
            self.dispatch()
            self.project()
            self.activate()
            self.down_project()
            self.combine()
        return self.output

    def project(self):
        self._gemm(self.grouped_input, self.gate_up, self.projected, "gate_up")

    def activate(self):
        t, h = self.x.shape
        intermediate = self.down.shape[2]
        topk = self.ids.shape[1]
        self.kernels["swiglu"] = _swiglu[(triton.cdiv(t * topk * intermediate, 256),)](
            self.projected, self.activated, t * topk, intermediate, 256)

    def down_project(self):
        self._gemm(self.activated, self.down, self.down_output, "down")

    def combine(self):
        t, h = self.x.shape
        topk = self.ids.shape[1]
        self.kernels["combine"] = _combine[(t, triton.cdiv(h, 128))](
            self.down_output, self.mapping, self.routing_weights, self.output, t, h, topk, 128,
            enable_fp_fusion=False)


def prepare_moe(x, gate_up, down, ids, routing_weights):
    for tensor, rank in ((x, 2), (gate_up, 3), (down, 3)):
        _input(tensor, rank)
    e, h, intermediate = down.shape
    if (not 0 < e <= 256 or h <= 0 or intermediate <= 0 or x.shape[1] != h
            or gate_up.shape != (e, 2 * intermediate, h)):
        raise ValueError("invalid MoE shape; prototype requires E in [1,256], H/I positive")
    if (ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64) or ids.shape[0] != x.shape[0]
            or not 0 < ids.shape[1] <= 32 or routing_weights.shape != ids.shape
            or routing_weights.dtype != torch.float32):
        raise ValueError("expected CUDA IDs[T,topk<=32] and FP32 weights of identical shape")
    if any(t.device != x.device or not t.is_contiguous() for t in (gate_up, down, ids, routing_weights)):
        raise ValueError("all MoE tensors must be contiguous and on the same CUDA device")
    r = ids.numel()
    if r >= 2**31 or r * max(h, 2 * intermediate) >= 2**31:
        raise ValueError("prototype vector indexing requires routed/intermediate elements < 2**31")
    device = x.device
    bf16 = lambda shape: torch.empty(shape, device=device, dtype=torch.bfloat16)
    int32 = lambda shape: torch.zeros(shape, device=device, dtype=torch.int32)
    plan = PreparedMoE(x, gate_up, down, ids, routing_weights, bf16((x.shape[0], h)),
                       int32((e,)), int32((e,)), int32((e,)), int32((e + 1,)), int32((r,)),
                       bf16((r, h)), bf16((r, 2 * intermediate)), bf16((r, intermediate)),
                       bf16((r, h)), int32((1,)), GemmConfig())
    # Compile and populate metadata before capture; no device value validation here.
    plan.run()
    return plan
