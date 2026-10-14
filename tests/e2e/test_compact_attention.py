"""Public-path compact attention scenario on the project's SM86 GPU."""

from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from llm_gpu_kernels.attention_cuda_compact import prepare_cuda_gqa_compact


LIMIT = 0.008
TOKENS = 96
HISTORY_LENGTHS = (2049, 0, 575)
CURRENT_V_STRIDE = 6144


def _make_values(seed: int, tokens: int, lengths: tuple[int, ...], device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def random(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16
        )

    q = random((tokens, 32, 128))
    segments = []
    for length in (*lengths, tokens):
        key = random((length, 8, 128))
        value = random((length, 8, 128))
        segments.append((key, value))

    key, value = segments[-1]
    padded_value = torch.empty_strided(
        tuple(value.shape),
        (CURRENT_V_STRIDE, value.shape[2], 1),
        dtype=value.dtype,
        device=device,
    )
    padded_value.copy_(value)
    segments[-1] = (key, padded_value)
    return q, tuple(segments)


def _sdpa_reference(q: torch.Tensor, segments: tuple[tuple[torch.Tensor, torch.Tensor], ...]):
    """Independent FP32 reference through public PyTorch SDPA's math backend."""
    if q.shape[0] == 0:
        return torch.empty_like(q, dtype=torch.float32)

    history = sum(key.shape[0] for key, _ in segments[:3])
    keys = torch.cat(tuple(key.float() for key, _ in segments), dim=0)
    values = torch.cat(tuple(value.float() for _, value in segments), dim=0)
    keys = keys.repeat_interleave(4, dim=1).permute(1, 0, 2).unsqueeze(0)
    values = values.repeat_interleave(4, dim=1).permute(1, 0, 2).unsqueeze(0)
    query = q.float().permute(1, 0, 2).unsqueeze(0)

    query_positions = torch.arange(q.shape[0], device=q.device)
    current_positions = torch.arange(q.shape[0], device=q.device)
    allowed = torch.ones(
        (q.shape[0], keys.shape[-2]), dtype=torch.bool, device=q.device
    )
    allowed[:, history:] = current_positions[None, :] <= query_positions[:, None]

    with sdpa_kernel(SDPBackend.MATH):
        result = F.scaled_dot_product_attention(
            query,
            keys,
            values,
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(128.0),
        )
    return result.squeeze(0).permute(1, 0, 2)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if tuple(actual.shape) != tuple(expected.shape):
        return {
            "shape_match": False,
            "relative_rms": None,
            "normalized_max": None,
            "max_abs": None,
            "finite": False,
            "passed": False,
        }
    if actual.numel() == 0:
        finite = bool(torch.isfinite(actual).all())
        return {
            "shape_match": True,
            "relative_rms": 0.0,
            "normalized_max": 0.0,
            "max_abs": 0.0,
            "finite": finite,
            "passed": finite,
        }

    actual_f = actual.float()
    expected_f = expected.float()
    delta = actual_f - expected_f
    relative_rms = (
        delta.square().mean().sqrt()
        / expected_f.square().mean().sqrt().clamp_min(1e-12)
    ).item()
    normalized_max = (
        delta.abs().max() / expected_f.abs().max().clamp_min(1e-12)
    ).item()
    finite = bool(torch.isfinite(actual_f).all() and torch.isfinite(expected_f).all())
    return {
        "shape_match": True,
        "relative_rms": relative_rms,
        "normalized_max": normalized_max,
        "max_abs": delta.abs().max().item(),
        "finite": finite,
        "passed": finite and relative_rms <= LIMIT and normalized_max <= LIMIT,
    }


@pytest.mark.parametrize("stages", (1, 2))
def test_compact_attention_public_api_stream_and_graph_update(tmp_path, stages: int):
    assert torch.cuda.is_available(), "the compact attention scenario requires CUDA"
    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    assert capability == (8, 6), f"the compact attention scenario requires SM86, got {capability}"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    q, segments = _make_values(3700 + stages, TOKENS, HISTORY_LENGTHS, device)
    output = torch.empty_like(q)
    plan = prepare_cuda_gqa_compact(q, segments, output, stages=stages)
    buffers = (q, *(tensor for pair in segments for tensor in pair), output)
    fixed_addresses = tuple(tensor.data_ptr() for tensor in buffers)

    producer_stream = torch.cuda.current_stream(device)
    execution_stream = torch.cuda.Stream(device=device)
    execution_stream.wait_stream(producer_stream)
    non_default_stream = (
        execution_stream.cuda_stream
        != torch.cuda.default_stream(device).cuda_stream
    )

    with torch.cuda.stream(execution_stream):
        eager_initial = plan.run().clone()
        expected_initial = _sdpa_reference(q, segments)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=execution_stream):
            graph_output = plan.run()
        graph.replay()
        graph_initial = graph_output.clone()
    execution_stream.synchronize()

    updated_q, updated_segments = _make_values(
        4700 + stages, TOKENS, HISTORY_LENGTHS, device
    )
    producer_stream.wait_stream(execution_stream)
    with torch.cuda.stream(producer_stream):
        for target, source in zip(
            (q, *(tensor for pair in segments for tensor in pair)),
            (updated_q, *(tensor for pair in updated_segments for tensor in pair)),
        ):
            target.copy_(source)
    execution_stream.wait_stream(producer_stream)

    with torch.cuda.stream(execution_stream):
        eager_updated = plan.run().clone()
        expected_updated = _sdpa_reference(q, segments)
        graph.replay()
        graph_updated = graph_output.clone()
    execution_stream.synchronize()

    addresses_preserved = tuple(tensor.data_ptr() for tensor in buffers) == fixed_addresses
    output_changed = bool(torch.any(graph_updated != graph_initial))
    actual_checks = {
        "eager_initial_vs_fp32_sdpa_math": _metrics(eager_initial, expected_initial),
        "graph_initial_vs_fp32_sdpa_math": _metrics(graph_initial, expected_initial),
        "eager_updated_vs_fp32_sdpa_math": _metrics(eager_updated, expected_updated),
        "graph_updated_vs_fp32_sdpa_math": _metrics(graph_updated, expected_updated),
    }

    manifest = {
        "schema": 1,
        "scenario": "m37_compact_attention_public_api",
        "input_kind": "seeded synthetic BF16 generated by PyTorch",
        "stages": stages,
        "shape": {
            "query": list(q.shape),
            "segments": [list(key.shape) for key, _ in segments],
            "output": list(output.shape),
        },
        "stride": {
            "query": list(q.stride()),
            "segments": [
                {"key": list(key.stride()), "value": list(value.stride())}
                for key, value in segments
            ],
            "output": list(output.stride()),
        },
        "checks": {
            "complete_history_tiles": any(length >= 32 for length in HISTORY_LENGTHS),
            "partial_history_tiles": any(length % 32 != 0 for length in HISTORY_LENGTHS),
            "empty_middle_history_segment": HISTORY_LENGTHS[1] == 0,
            "query_tail": TOKENS % 64 != 0,
            "current_v_token_stride_6144": segments[-1][1].stride(0) == CURRENT_V_STRIDE,
            "non_default_stream": non_default_stream,
            "fixed_buffers_preserved": addresses_preserved,
            "updated_values_change_output": output_changed,
            **actual_checks,
        },
    }
    manifest_path = tmp_path / f"m37_compact_attention_s{stages}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    assert manifest["checks"]["complete_history_tiles"]
    assert manifest["checks"]["partial_history_tiles"]
    assert manifest["checks"]["empty_middle_history_segment"]
    assert manifest["checks"]["query_tail"]
    assert manifest["checks"]["current_v_token_stride_6144"]
    assert manifest["checks"]["non_default_stream"]
    assert manifest["checks"]["fixed_buffers_preserved"]
    assert manifest["checks"]["updated_values_change_output"]
    for name, metric in actual_checks.items():
        assert metric["passed"], f"{name} exceeded the {LIMIT} limits: {metric}"
