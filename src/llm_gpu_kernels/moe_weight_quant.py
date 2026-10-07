"""Offline M22 gate/up format preparation; never called inside timed forward.

Each signed INT4 q is encoded as q+8, eight low-to-high K nibbles per word.
Words use Torch int32 storage with the bit pattern of a uint32. A BF16 scale
belongs to one expert/output-channel/K128 group. CUDA consumption must unpack
and scale in its tile, not call the full-weight reference below at run time.
"""
from dataclasses import dataclass
import hashlib
import math

import torch

GROUP = 128
SHAPE = (128, 1536, 2048)


def _digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(tensor.dtype), tuple(tensor.shape))).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PackedGateUp:
    words: torch.Tensor
    scales: torch.Tensor
    metadata: dict

    def to(self, device):
        return PackedGateUp(self.words.to(device), self.scales.to(device), self.metadata)


def dequantize_gate_up_reference(packed):
    """Materialize BF16 weights for offline oracle/audit only, never hot path."""
    if not isinstance(packed, PackedGateUp):
        raise TypeError('PackedGateUp required')
    words, scales = packed.words, packed.scales
    if words.dtype != torch.int32 or scales.dtype != torch.bfloat16:
        raise ValueError('int32 bit-pattern words / BF16 scales required')
    if tuple(words.shape) != (128, 1536, 256) or tuple(scales.shape) != (128, 1536, 16):
        raise ValueError('unsupported packed shapes')
    if not words.is_contiguous() or not scales.is_contiguous() or words.device != scales.device:
        raise ValueError('same-device contiguous packed buffers required')
    output = torch.empty(SHAPE, dtype=torch.bfloat16, device=words.device)
    shifts = torch.arange(8, dtype=torch.int64, device=words.device) * 4
    for expert in range(128):
        nibble = (words[expert].long()[..., None] >> shifts) & 15
        q = (nibble.reshape(1536, 16, GROUP) - 8).float()
        output[expert].copy_((q * scales[expert].float()[..., None]).reshape(1536, 2048))
    return output


def quantize_gate_up(weight):
    """Prepare one finite, contiguous CPU BF16 E128/H2048/I768 weight tensor.

    Scale rounding is part of the format: quantization uses the stored BF16
    scale, with nearest-even integer rounding and clamp to [-7,7]. Preparation
    loops over experts to bound temporary host memory, not over timed GPU work.
    """
    if weight.device.type != 'cpu' or weight.dtype != torch.bfloat16:
        raise ValueError('offline packing requires CPU BF16 weights')
    if tuple(weight.shape) != SHAPE or not weight.is_contiguous():
        raise ValueError('expected contiguous [128,1536,2048] gate/up')
    if not torch.isfinite(weight).all().item():
        raise ValueError('nonfinite source weights')
    words = torch.empty((128, 1536, 256), dtype=torch.int32)
    scales = torch.empty((128, 1536, 16), dtype=torch.bfloat16)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    error_sum = reference_sum = 0.0
    error_max = reference_max = 0.0
    for expert in range(128):
        x = weight[expert].float().reshape(1536, 16, GROUP)
        max_abs = x.abs().amax(-1)
        scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / 7).to(torch.bfloat16)
        if not (torch.isfinite(scale) & (scale > 0)).all().item():
            raise ValueError('scale is not representable as finite positive BF16')
        q = torch.round(x / scale.float()[..., None]).clamp(-7, 7).long()
        word = ((q.reshape(1536, 256, 8) + 8) << shifts).sum(-1)
        words[expert].copy_(word.to(torch.int32))
        scales[expert].copy_(scale)
        decoded = (q.float() * scale.float()[..., None]).to(torch.bfloat16)
        if not torch.isfinite(decoded).all().item():
            raise ValueError('dequantized BF16 operand is not finite')
        original = x.double()
        difference = decoded.double() - original
        error_sum += difference.square().sum().item()
        reference_sum += original.square().sum().item()
        error_max = max(error_max, difference.abs().max().item())
        reference_max = max(reference_max, original.abs().max().item())
    metadata = dict(format_version=1, logical_shape=list(SHAPE), group_size=GROUP,
                    encoding='q+8, q in [-7,7], low nibble is earliest K',
                    word_storage='int32 storage of uint32 bit pattern', scale_dtype='bfloat16',
                    activation_dtype='bfloat16', accumulator_dtype='float32',
                    source_sha256=_digest(weight), words_sha256=_digest(words),
                    scales_sha256=_digest(scales),
                    source_weight_bytes=weight.numel() * weight.element_size(),
                    packed_weight_bytes=words.numel() * words.element_size(),
                    scale_bytes=scales.numel() * scales.element_size(),
                    weight_format_error=dict(relative_rms=math.sqrt(error_sum / reference_sum) if reference_sum else 0.0,
                                             normalized_max=error_max / reference_max if reference_max else 0.0,
                                             max_abs=error_max),
                    interpretation='format loss only; not kernel implementation error or model task quality')
    return PackedGateUp(words, scales, metadata)
