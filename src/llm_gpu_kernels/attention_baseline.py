"""Prepared segmented-KV GQA input wrapper for the official FA2 cost baseline.

This module intentionally contains no custom attention kernel.  The timed full
path is a reusable-buffer K/V pack followed by the public FlashAttention 2 API.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch


EXPECTED_Q_HEADS = 32
EXPECTED_KV_HEADS = 8
EXPECTED_HEAD_DIM = 128
SEGMENT_COUNT = 4
SCALE = 1.0 / math.sqrt(EXPECTED_HEAD_DIM)


def _flash_attn_func() -> Callable:
    try:
        from flash_attn import flash_attn_func
    except Exception as exc:  # pragma: no cover - depends on the selected runtime overlay
        raise RuntimeError(
            "FlashAttention 2 is unavailable; source scripts/activate-flash-attn.sh first"
        ) from exc
    return flash_attn_func


@dataclass
class PreparedSegmentedAttention:
    """Fixed-address four-segment inputs and reusable packed K/V buffers.

    ``segments`` is ordered as selected history, sink, previous recent, current
    chunk.  The current segment has exactly the query length, giving FA2's
    bottom-right causal mask the required ``current_j <= query_i`` semantics.
    Values may be updated in place.  Shapes, storage, and segment order stay
    fixed for the lifetime of the plan; do not run one plan concurrently.
    """

    q: torch.Tensor
    segments: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    packed_k: torch.Tensor
    packed_v: torch.Tensor
    _q4: torch.Tensor
    _packed_k4: torch.Tensor
    _packed_v4: torch.Tensor
    _fa2: Callable
    output: torch.Tensor | None = None

    @property
    def query_length(self) -> int:
        return self.q.shape[0]

    @property
    def history_length(self) -> int:
        return sum(k.shape[0] for k, _ in self.segments[:3])

    @property
    def packed_workspace_bytes(self) -> int:
        """Bytes in reusable packed K/V only; excludes inputs and FA2 output."""
        return self.packed_k.numel() * self.packed_k.element_size() + self.packed_v.numel() * self.packed_v.element_size()

    def _device_guard(self):
        device_index = self.q.device.index
        if device_index is not None and torch.cuda.current_device() != device_index:
            return torch.cuda.device(self.q.device)
        return nullcontext()

    def pack(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate the four K/V segments into existing buffers in place."""
        if self.query_length == 0:
            return self.packed_k, self.packed_v
        with self._device_guard():
            torch.cat(tuple(k for k, _ in self.segments), dim=0, out=self.packed_k)
            torch.cat(tuple(v for _, v in self.segments), dim=0, out=self.packed_v)
        return self.packed_k, self.packed_v

    def attend_prepacked(self) -> torch.Tensor:
        """Run only the public FA2 call over the current packed buffers."""
        if self.query_length == 0:
            if self.output is None:
                self.output = torch.empty_like(self.q)
            return self.output
        with self._device_guard():
            result = self._fa2(
                self._q4,
                self._packed_k4,
                self._packed_v4,
                dropout_p=0.0,
                softmax_scale=SCALE,
                causal=True,
            )
        self.output = result.squeeze(0)
        return self.output

    def run(self) -> torch.Tensor:
        """Run the complete GPU-resident pack + FA2 chain."""
        if self.query_length == 0:
            if self.output is None:
                self.output = torch.empty_like(self.q)
            return self.output
        self.pack()
        return self.attend_prepacked()


def prepare_segmented_attention(
    q: torch.Tensor,
    segments: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    fa2_func: Callable | None = None,
) -> PreparedSegmentedAttention:
    """Validate fixed-shape inputs and allocate reusable packed K/V buffers.

    Inputs are unbatched contiguous BF16 tensors: Q[T,32,128] and four
    ``(K[L,8,128], V[L,8,128])`` pairs.  The returned views add a batch size of
    one without copying.  FA2 is imported lazily so other project modules do not
    require the optional official wheel.
    """
    if not isinstance(q, torch.Tensor) or q.ndim != 3:
        raise ValueError("q must have shape [T, 32, 128]")
    if tuple(q.shape[1:]) != (EXPECTED_Q_HEADS, EXPECTED_HEAD_DIM):
        raise ValueError(f"q must have shape [T, {EXPECTED_Q_HEADS}, {EXPECTED_HEAD_DIM}]")
    if len(segments) != SEGMENT_COUNT:
        raise ValueError(f"exactly {SEGMENT_COUNT} K/V segments are required")
    if not q.is_cuda or q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError("q must be contiguous CUDA BF16")

    fixed_segments: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, pair in enumerate(segments):
        if len(pair) != 2:
            raise ValueError(f"segment {index} must contain a K/V pair")
        k, v = pair
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise ValueError(f"segment {index} K and V must be tensors")
        if k.ndim != 3 or tuple(k.shape[1:]) != (EXPECTED_KV_HEADS, EXPECTED_HEAD_DIM):
            raise ValueError(f"segment {index} K must have shape [L, {EXPECTED_KV_HEADS}, {EXPECTED_HEAD_DIM}]")
        if v.shape != k.shape:
            raise ValueError(f"segment {index} V shape must match K")
        for name, tensor in (("K", k), ("V", v)):
            if tensor.device != q.device or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                raise ValueError(f"segment {index} {name} must be contiguous CUDA BF16 on q's device")
        fixed_segments.append((k, v))
    if fixed_segments[-1][0].shape[0] != q.shape[0]:
        raise ValueError("current K/V segment length must equal the Q length")

    total_length = 0 if q.shape[0] == 0 else sum(k.shape[0] for k, _ in fixed_segments)
    packed_k = torch.empty((total_length, EXPECTED_KV_HEADS, EXPECTED_HEAD_DIM), device=q.device, dtype=q.dtype)
    packed_v = torch.empty_like(packed_k)
    plan = PreparedSegmentedAttention(
        q=q,
        segments=tuple(fixed_segments),
        packed_k=packed_k,
        packed_v=packed_v,
        _q4=q.unsqueeze(0),
        _packed_k4=packed_k.unsqueeze(0),
        _packed_v4=packed_v.unsqueeze(0),
        _fa2=fa2_func or _flash_attn_func(),
        output=torch.empty_like(q) if q.shape[0] == 0 else None,
    )
    # Make the prepacked-only arm well-defined immediately after prepare.
    plan.pack()
    return plan


def segmented_attention_reference(
    q: torch.Tensor,
    segments: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Small FP32 semantic oracle with explicit GQA expansion and causal mask."""
    if q.shape[0] == 0:
        return torch.empty_like(q)
    if len(segments) != SEGMENT_COUNT:
        raise ValueError("reference expects exactly four K/V segments")
    query_length = q.shape[0]
    history_length = sum(k.shape[0] for k, _ in segments[:3])
    keys = torch.cat(tuple(k.float() for k, _ in segments), dim=0)
    values = torch.cat(tuple(v.float() for _, v in segments), dim=0)
    keys = keys.repeat_interleave(EXPECTED_Q_HEADS // EXPECTED_KV_HEADS, dim=1)
    values = values.repeat_interleave(EXPECTED_Q_HEADS // EXPECTED_KV_HEADS, dim=1)

    scores = torch.matmul(q.float().permute(1, 0, 2), keys.permute(1, 2, 0)) * SCALE
    key_positions = torch.arange(keys.shape[0], device=q.device)
    query_positions = torch.arange(query_length, device=q.device)
    history_visible = key_positions < history_length
    current_visible = (key_positions[None, :] >= history_length) & (
        key_positions[None, :] - history_length <= query_positions[:, None]
    )
    visible = history_visible[None, :] | current_visible
    scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    result = torch.matmul(probabilities, values.permute(1, 0, 2))
    return result.permute(1, 0, 2).contiguous()
