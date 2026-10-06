"""Readable reference paths. Expert loops are not strong performance baselines."""
import torch
from torch.nn import functional as F


def grouped_reference(a, weights, counts, out=None):
    if out is None:
        out = torch.empty((a.shape[0], weights.shape[1]), dtype=a.dtype, device=a.device)
    start = 0
    for e, count in enumerate(counts):
        if count:
            torch.mm(a[start:start + count], weights[e].T, out=out[start:start + count])
        start += count
    return out


def moe_reference(x, gate_up, down, expert_ids, routing_weights):
    """FP32 expert-loop oracle for fixed routing; no router, bias or renormalization.

    gate_up: [E, 2*I, H], gate followed by up. down: [E, H, I].
    Each route contributes weight * down(silu(gate(x)) * up(x)).
    Duplicate expert IDs within a token are allowed and contribute separately.
    """
    if x.ndim != 2 or gate_up.ndim != 3 or down.ndim != 3:
        raise ValueError("invalid ranks")
    e, h, intermediate = down.shape
    if (gate_up.shape != (e, 2 * intermediate, h) or x.shape[1] != h
            or expert_ids.ndim != 2 or expert_ids.shape != routing_weights.shape
            or expert_ids.shape[0] != x.shape[0] or expert_ids.shape[1] == 0):
        raise ValueError("incompatible MoE shapes")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("integer expert IDs required")
    if any(t.device != x.device for t in (gate_up, down, expert_ids, routing_weights)):
        raise ValueError("all tensors must use the same device")
    if torch.any((expert_ids < 0) | (expert_ids >= e)) or not torch.isfinite(routing_weights).all():
        raise ValueError("invalid expert IDs or nonfinite routing weights")
    output = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(e):
        tokens, slots = torch.where(expert_ids == expert)
        if tokens.numel():
            projected = x[tokens].float() @ gate_up[expert].float().T
            gate, up = projected.chunk(2, dim=-1)
            value = (F.silu(gate) * up) @ down[expert].float().T
            output.index_add_(0, tokens, value * routing_weights[tokens, slots, None].float())
    return output
