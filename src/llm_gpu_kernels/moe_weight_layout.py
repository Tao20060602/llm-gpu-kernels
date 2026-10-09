"""Offline bit-preserving M24 layout conversion for audited W4 gate/up data."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
import re

import torch

from .moe_weight_quant import GROUP, SHAPE, PackedGateUp


LAYOUT = (
    "words[E,projection,Ntile,K32,word,Nlane];"
    "scales[E,projection,Ntile,K128,Nlane];Nlane-contiguous"
)
PHYSICAL_WORDS_SHAPE = (128, 2, 12, 64, 4, 64)
PHYSICAL_SCALES_SHAPE = (128, 2, 12, 16, 64)
V1_WORDS_SHAPE = (128, 1536, 256)
V1_SCALES_SHAPE = (128, 1536, 16)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _digest(tensor: torch.Tensor) -> str:
    """Match the v1 digest encoding while streaming the CPU buffer to SHA-256."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(memoryview(value.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _check_digest(value, where: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class RepackedGateUp:
    """Physical M24 buffers plus their v1 canonical and layout provenance."""

    words: torch.Tensor
    scales: torch.Tensor
    metadata: dict

    def to(self, device):
        return RepackedGateUp(
            self.words.to(device), self.scales.to(device), deepcopy(self.metadata)
        )


def _validate_v1_cpu(packed: PackedGateUp) -> tuple[str, str]:
    if not isinstance(packed, PackedGateUp):
        raise TypeError("repack_gate_up requires an audited CPU PackedGateUp v1")
    if (packed.words.device.type != "cpu" or packed.scales.device.type != "cpu"
            or packed.words.device != packed.scales.device):
        raise ValueError("v1 words and scales must be on CPU before layout conversion")
    if (tuple(packed.words.shape) != V1_WORDS_SHAPE
            or tuple(packed.scales.shape) != V1_SCALES_SHAPE
            or packed.words.dtype != torch.int32
            or packed.scales.dtype != torch.bfloat16
            or not packed.words.is_contiguous()
            or not packed.scales.is_contiguous()):
        raise ValueError("v1 CPU buffers must be contiguous int32/BF16 with the audited shapes")

    metadata = packed.metadata
    if (not isinstance(metadata, dict)
            or metadata.get("format_version") != 1
            or metadata.get("group_size") != GROUP
            or tuple(metadata.get("logical_shape", ())) != SHAPE):
        raise ValueError("v1 metadata must declare format_version=1, W4G128 and E128/H1536/K2048")

    words_sha = _check_digest(metadata.get("words_sha256"), "v1 words_sha256")
    scales_sha = _check_digest(metadata.get("scales_sha256"), "v1 scales_sha256")
    if _digest(packed.words) != words_sha or _digest(packed.scales) != scales_sha:
        raise ValueError("v1 packed buffers do not match their recorded canonical digests")
    return words_sha, scales_sha


def repack_gate_up(packed: PackedGateUp) -> RepackedGateUp:
    """Permute an audited CPU v1 pack without requantizing or changing bits."""
    canonical_words_sha, canonical_scales_sha = _validate_v1_cpu(packed)

    words = packed.words.reshape(128, 2, 12, 64, 64, 4)
    words = words.permute(0, 1, 2, 4, 5, 3).contiguous()
    scales = packed.scales.reshape(128, 2, 12, 64, 16)
    scales = scales.permute(0, 1, 2, 4, 3).contiguous()

    canonical_words_bytes = packed.words.numel() * packed.words.element_size()
    canonical_scales_bytes = packed.scales.numel() * packed.scales.element_size()
    physical_words_bytes = words.numel() * words.element_size()
    physical_scales_bytes = scales.numel() * scales.element_size()
    physical_total_bytes = physical_words_bytes + physical_scales_bytes
    canonical_total_bytes = canonical_words_bytes + canonical_scales_bytes
    metadata = {
        "format_version": 2,
        "layout": LAYOUT,
        "logical_shape": list(SHAPE),
        "group_size": GROUP,
        "canonical_words_sha256": canonical_words_sha,
        "canonical_scales_sha256": canonical_scales_sha,
        "canonical_words_shape": list(V1_WORDS_SHAPE),
        "canonical_scales_shape": list(V1_SCALES_SHAPE),
        "physical_words_shape": list(PHYSICAL_WORDS_SHAPE),
        "physical_scales_shape": list(PHYSICAL_SCALES_SHAPE),
        "physical_words_sha256": _digest(words),
        "physical_scales_sha256": _digest(scales),
        "canonical_v1_metadata": deepcopy(packed.metadata),
        "byte_ledger": {
            "canonical_words_bytes": canonical_words_bytes,
            "canonical_scales_bytes": canonical_scales_bytes,
            "canonical_packed_total_bytes": canonical_total_bytes,
            "physical_words_bytes": physical_words_bytes,
            "physical_scales_bytes": physical_scales_bytes,
            "physical_packed_total_bytes": physical_total_bytes,
            "additional_repacked_tensor_bytes": physical_total_bytes,
            "canonical_plus_repacked_resident_bytes": canonical_total_bytes + physical_total_bytes,
            "physical_packed_total_mib": physical_total_bytes / (1024 * 1024),
            "interpretation": "layout-only storage; quantization and total packed bytes are unchanged",
        },
    }
    if physical_total_bytes != canonical_total_bytes:
        raise RuntimeError("layout conversion changed the packed+scale byte count")
    if not math.isclose(metadata["byte_ledger"]["physical_packed_total_mib"], 198.0):
        raise RuntimeError("unexpected physical W4G128 packed+scale size")
    return RepackedGateUp(words, scales, metadata)


def restore_v1_layout(repacked: RepackedGateUp) -> PackedGateUp:
    """Restore the canonical v1 bit patterns for offline reference use only."""
    if not isinstance(repacked, RepackedGateUp):
        raise TypeError("restore_v1_layout requires RepackedGateUp")
    metadata = repacked.metadata
    if (not isinstance(metadata, dict)
            or metadata.get("format_version") != 2
            or metadata.get("layout") != LAYOUT
            or metadata.get("group_size") != GROUP
            or tuple(metadata.get("logical_shape", ())) != SHAPE):
        raise ValueError("unsupported M24 physical layout metadata")
    if repacked.words.device.type != "cpu" or repacked.scales.device.type != "cpu":
        raise ValueError("restore_v1_layout is CPU-only offline reference preparation")
    if (tuple(repacked.words.shape) != PHYSICAL_WORDS_SHAPE
            or tuple(repacked.scales.shape) != PHYSICAL_SCALES_SHAPE
            or repacked.words.dtype != torch.int32
            or repacked.scales.dtype != torch.bfloat16
            or not repacked.words.is_contiguous()
            or not repacked.scales.is_contiguous()):
        raise ValueError("M24 physical buffers must be contiguous int32/BF16 with exact shapes")

    physical_words_sha = _check_digest(metadata.get("physical_words_sha256"), "physical_words_sha256")
    physical_scales_sha = _check_digest(metadata.get("physical_scales_sha256"), "physical_scales_sha256")
    if (_digest(repacked.words) != physical_words_sha
            or _digest(repacked.scales) != physical_scales_sha):
        raise ValueError("M24 physical buffers do not match their recorded digests")

    words = repacked.words.reshape(128, 2, 12, 64, 4, 64)
    words = words.permute(0, 1, 2, 5, 3, 4).contiguous().reshape(V1_WORDS_SHAPE)
    scales = repacked.scales.reshape(128, 2, 12, 16, 64)
    scales = scales.permute(0, 1, 2, 4, 3).contiguous().reshape(V1_SCALES_SHAPE)

    canonical_words_sha = _check_digest(metadata.get("canonical_words_sha256"), "canonical_words_sha256")
    canonical_scales_sha = _check_digest(metadata.get("canonical_scales_sha256"), "canonical_scales_sha256")
    if _digest(words) != canonical_words_sha or _digest(scales) != canonical_scales_sha:
        raise ValueError("restored v1 bit patterns do not match the canonical digests")

    v1_metadata = metadata.get("canonical_v1_metadata")
    if (not isinstance(v1_metadata, dict)
            or v1_metadata.get("format_version") != 1
            or v1_metadata.get("group_size") != GROUP
            or tuple(v1_metadata.get("logical_shape", ())) != SHAPE
            or v1_metadata.get("words_sha256") != canonical_words_sha
            or v1_metadata.get("scales_sha256") != canonical_scales_sha):
        raise ValueError("M24 metadata has no matching canonical v1 metadata snapshot")
    return PackedGateUp(words, scales, deepcopy(v1_metadata))
