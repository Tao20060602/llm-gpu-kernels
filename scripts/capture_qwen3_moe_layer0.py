#!/usr/bin/env python3
"""Capture real layer-0 inputs and routing for pinned Qwen3-30B-A3B.

The capture uses the pinned Transformers implementation for embeddings,
attention, normalization, rotary embeddings, and routing. It replaces the
decoder layer's expert loop with a router-only stub, so it does not allocate
or execute the 128 experts. The artifacts are inputs for fixed-route operator
validation; they do not claim to represent full-model output or speed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_REPO_ID = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
NUM_EXPERTS = 128
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
TOP_K = 8
PROMPT_LENGTHS = (17, 193, 1024, 2049)
PROMPT_SEEDS = {
    "archive": (
        "Archive note: the river observatory stores daily gauge readings, "
        "calibration records, and maintenance summaries. Each entry includes "
        "a timestamp, a station identifier, and a brief account of the "
        "conditions observed by the field team."
    ),
    "code": (
        "def summarize_readings(records):\n"
        "    totals = {}\n"
        "    for record in records:\n"
        "        station = record['station']\n"
        "        totals.setdefault(station, []).append(record['reading'])\n"
        "    return {station: sum(values) / len(values) "
        "for station, values in totals.items() if values}\n"
    ),
}


class CaptureError(RuntimeError):
    """Raised when source provenance or the fixed capture contract fails."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor_bytes(tensor: torch.Tensor) -> str:
    if tensor.device.type != "cpu":
        tensor = tensor.detach().to("cpu")
    value = tensor.detach().contiguous()
    raw = value.view(torch.uint8).numpy()
    digest = hashlib.sha256()
    digest.update(memoryview(raw))
    return digest.hexdigest()


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CaptureError(f"{label} must contain a JSON object: {path}")
    return value


def _source_scalar(source: Any, name: str) -> str | None:
    if not isinstance(source, dict):
        return None
    if name == "blobId":
        value = source.get("blobId", source.get("blob_id"))
        return value if isinstance(value, str) else None
    lfs = source.get("lfs")
    if isinstance(lfs, dict):
        value = lfs.get("sha256")
        return value if isinstance(value, str) else None
    if isinstance(lfs, str) and re.fullmatch(r"[0-9a-fA-F]{64}", lfs):
        return lfs.lower()
    value = source.get("sha256")
    return value.lower() if isinstance(value, str) else None


def _verify_tokenizer_sources(tokenizer_dir: Path) -> tuple[dict[str, Path], dict[str, str]]:
    manifest_path = tokenizer_dir / "manifest.json"
    api_path = tokenizer_dir / "source-api.json"
    manifest = _load_json(manifest_path, "pinned tokenizer manifest")
    api = _load_json(api_path, "pinned tokenizer source API response")
    if (
        manifest.get("status") != "passed"
        or manifest.get("repo_id") != MODEL_REPO_ID
        or manifest.get("revision") != MODEL_REVISION
    ):
        raise CaptureError("tokenizer manifest does not identify the pinned model revision")
    if api.get("id") != MODEL_REPO_ID or api.get("sha") != MODEL_REVISION:
        raise CaptureError("tokenizer source API response does not identify the pinned revision")
    api_sha256 = _sha256_file(api_path)
    if manifest.get("metadata_sha256") != api_sha256:
        raise CaptureError("tokenizer source API response hash disagrees with its manifest")

    siblings_raw = api.get("siblings")
    if not isinstance(siblings_raw, list):
        raise CaptureError("tokenizer source API response has no siblings list")
    siblings = {
        item.get("rfilename"): item
        for item in siblings_raw
        if isinstance(item, dict) and isinstance(item.get("rfilename"), str)
    }
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list):
        raise CaptureError("tokenizer manifest has no files list")
    records = {
        item.get("name"): item
        for item in manifest_files
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }

    verified_paths: dict[str, Path] = {}
    verified_hashes: dict[str, str] = {}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        record = records.get(name)
        sibling = siblings.get(name)
        if not isinstance(record, dict) or not isinstance(sibling, dict):
            raise CaptureError(f"pinned tokenizer metadata is missing {name}")
        path = tokenizer_dir / name
        if not path.is_file():
            raise CaptureError(f"pinned tokenizer asset is missing: {path}")
        actual_size = path.stat().st_size
        actual_sha256 = _sha256_file(path)
        if record.get("bytes") != actual_size or record.get("sha256") != actual_sha256:
            raise CaptureError(f"tokenizer asset size or SHA256 mismatch: {name}")

        sibling_blob = sibling.get("blobId")
        sibling_lfs = sibling.get("lfs")
        record_blob = record.get("git_blob_sha1")
        # LFS blobId identifies the Git pointer, not the downloaded payload.
        if not isinstance(sibling_lfs, dict) and sibling_blob is not None and record_blob != sibling_blob:
            raise CaptureError(f"tokenizer git blob id mismatch: {name}")
        source = record.get("source")
        if isinstance(source, dict):
            source_blob = _source_scalar(source, "blobId")
            source_lfs = _source_scalar(source, "lfs")
            if source_blob is not None and source_blob != sibling_blob:
                raise CaptureError(f"tokenizer source blob metadata mismatch: {name}")
            if source_lfs is not None:
                if not isinstance(sibling_lfs, dict) or source_lfs != sibling_lfs.get("sha256"):
                    raise CaptureError(f"tokenizer source LFS metadata mismatch: {name}")
        if isinstance(sibling_lfs, dict):
            if sibling_lfs.get("sha256") != actual_sha256:
                raise CaptureError(f"tokenizer LFS content SHA256 mismatch: {name}")
            lfs_size = sibling_lfs.get("size")
            if lfs_size is not None and lfs_size != actual_size:
                raise CaptureError(f"tokenizer LFS content size mismatch: {name}")
        elif isinstance(sibling_blob, str):
            git_digest = hashlib.sha1(
                b"blob " + str(actual_size).encode("ascii") + b"\0"
                + path.read_bytes()
            ).hexdigest()
            if git_digest != sibling_blob or record_blob != git_digest:
                raise CaptureError(f"tokenizer Git blob content hash mismatch: {name}")

        verified_paths[name] = path
        verified_hashes[name] = actual_sha256

    verified_paths["manifest.json"] = manifest_path
    verified_paths["source-api.json"] = api_path
    verified_hashes["manifest.json"] = _sha256_file(manifest_path)
    verified_hashes["source-api.json"] = api_sha256
    return verified_paths, verified_hashes


def _validate_extraction_bundle(model_dir: Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    manifest_path = model_dir / "manifest.json"
    artifact_path = model_dir / "layer0.safetensors"
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    for path in (manifest_path, artifact_path, config_path, index_path):
        if not path.is_file():
            raise CaptureError(f"required pinned layer-0 source file is missing: {path}")
    manifest = _load_json(manifest_path, "layer-0 extraction manifest")
    source = manifest.get("source", {})
    artifact = manifest.get("artifact", {})
    if (
        manifest.get("status") != "passed"
        or source.get("repo") != MODEL_REPO_ID
        or source.get("revision") != MODEL_REVISION
        or artifact.get("file") != artifact_path.name
    ):
        raise CaptureError("layer-0 extraction manifest does not identify the pinned artifact")
    if artifact_path.stat().st_size != artifact.get("size_bytes"):
        raise CaptureError("layer-0 artifact size disagrees with its extraction manifest")
    artifact_sha256 = _sha256_file(artifact_path)
    if artifact_sha256 != artifact.get("sha256"):
        raise CaptureError("layer-0 artifact SHA256 disagrees with its extraction manifest")
    config_sha256 = _sha256_file(config_path)
    index_sha256 = _sha256_file(index_path)
    if config_sha256 != source.get("config_sha256"):
        raise CaptureError("config.json SHA256 disagrees with its extraction manifest")
    if index_sha256 != source.get("index_sha256"):
        raise CaptureError("model index SHA256 disagrees with its extraction manifest")

    paths = {
        "extraction_manifest": manifest_path,
        "layer0_safetensors": artifact_path,
        "model_config": config_path,
        "model_index": index_path,
    }
    hashes = {
        name: _sha256_file(path)
        for name, path in paths.items()
    }
    return manifest, paths, hashes


def _prepare_runtime(runtime_root: Path, repo_root: Path):
    if not runtime_root.is_dir():
        raise CaptureError(f"Transformers overlay is missing: {runtime_root}")
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(runtime_root))

    import safetensors
    from safetensors import safe_open
    import transformers
    from transformers import AutoTokenizer
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeDecoderLayer,
        Qwen3MoeRotaryEmbedding,
    )

    expected_runtime = {
        "python": (3, 12),
        "torch": "2.7.1+cu128",
        "transformers": "4.53.2",
        "safetensors": "0.8.0",
    }
    if sys.version_info[:2] != expected_runtime["python"]:
        raise CaptureError(f"expected Python 3.12, got {sys.version.split()[0]}")
    if torch.__version__ != expected_runtime["torch"]:
        raise CaptureError(f"expected torch {expected_runtime['torch']}, got {torch.__version__}")
    if transformers.__version__ != expected_runtime["transformers"]:
        raise CaptureError(
            f"expected transformers {expected_runtime['transformers']}, got {transformers.__version__}"
        )
    if safetensors.__version__ != expected_runtime["safetensors"]:
        raise CaptureError(
            f"expected safetensors {expected_runtime['safetensors']}, got {safetensors.__version__}"
        )
    return {
        "safe_open": safe_open,
        "AutoTokenizer": AutoTokenizer,
        "Qwen3MoeConfig": Qwen3MoeConfig,
        "Qwen3MoeDecoderLayer": Qwen3MoeDecoderLayer,
        "Qwen3MoeRotaryEmbedding": Qwen3MoeRotaryEmbedding,
        "create_causal_mask": create_causal_mask,
        "create_sliding_window_causal_mask": create_sliding_window_causal_mask,
        "transformers": transformers,
        "safetensors": safetensors,
    }


def _validate_config(config: Any) -> None:
    expected = {
        "model_type": "qwen3_moe",
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_intermediate_size": INTERMEDIATE_SIZE,
        "decoder_sparse_step": 1,
        "norm_topk_prob": True,
        "hidden_act": "silu",
        "max_position_embeddings": 40_960,
    }
    for name, expected_value in expected.items():
        actual = getattr(config, name, None)
        if actual != expected_value:
            raise CaptureError(f"pinned Qwen3-MoE config {name}={actual!r}, expected {expected_value!r}")
    if 0 in getattr(config, "mlp_only_layers", []):
        raise CaptureError("layer 0 is configured as dense, not sparse MoE")


def _required_tensor_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (151_936, HIDDEN_SIZE),
        "model.layers.0.input_layernorm.weight": (HIDDEN_SIZE,),
        "model.layers.0.post_attention_layernorm.weight": (HIDDEN_SIZE,),
        "model.layers.0.self_attn.q_proj.weight": (4_096, HIDDEN_SIZE),
        "model.layers.0.self_attn.k_proj.weight": (512, HIDDEN_SIZE),
        "model.layers.0.self_attn.v_proj.weight": (512, HIDDEN_SIZE),
        "model.layers.0.self_attn.o_proj.weight": (HIDDEN_SIZE, 4_096),
        "model.layers.0.self_attn.q_norm.weight": (128,),
        "model.layers.0.self_attn.k_norm.weight": (128,),
        "model.layers.0.mlp.gate.weight": (NUM_EXPERTS, HIDDEN_SIZE),
    }
    for expert in range(NUM_EXPERTS):
        prefix = f"model.layers.0.mlp.experts.{expert}."
        shapes[prefix + "gate_proj.weight"] = (INTERMEDIATE_SIZE, HIDDEN_SIZE)
        shapes[prefix + "up_proj.weight"] = (INTERMEDIATE_SIZE, HIDDEN_SIZE)
        shapes[prefix + "down_proj.weight"] = (HIDDEN_SIZE, INTERMEDIATE_SIZE)
    return shapes


def _validate_tensor_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_tensors = manifest.get("tensors")
    if not isinstance(source_tensors, dict):
        raise CaptureError("layer-0 manifest does not contain tensor records")
    expected_shapes = _required_tensor_shapes()
    if set(source_tensors) != set(expected_shapes):
        missing = sorted(set(expected_shapes) - set(source_tensors))[:4]
        extra = sorted(set(source_tensors) - set(expected_shapes))[:4]
        raise CaptureError(f"layer-0 tensor key set mismatch; missing={missing}, extra={extra}")
    for name, shape in expected_shapes.items():
        record = source_tensors[name]
        if record.get("shape") != list(shape) or record.get("dtype") != "BF16":
            raise CaptureError(f"layer-0 manifest has unexpected shape or dtype for {name}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise CaptureError(f"layer-0 manifest has invalid SHA256 for {name}")
    return source_tensors


def _read_safetensor(safe_file: Any, manifest_tensors: dict[str, dict[str, Any]], name: str,
                     expected_shape: tuple[int, ...]) -> torch.Tensor:
    value = safe_file.get_tensor(name)
    if tuple(value.shape) != expected_shape or value.dtype != torch.bfloat16:
        raise CaptureError(f"loaded source tensor has unexpected shape or dtype: {name}")
    expected_hash = manifest_tensors[name]["sha256"]
    if _sha256_tensor_bytes(value) != expected_hash:
        raise CaptureError(f"source tensor content hash mismatch: {name}")
    return value.contiguous()


def _write_shared_weights(
    artifact_path: Path,
    source_tensors: dict[str, dict[str, Any]],
    output_path: Path,
    safe_open: Any,
) -> dict[str, torch.Tensor]:
    gate_up = torch.empty(
        (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16, device="cpu"
    )
    down = torch.empty(
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device="cpu"
    )
    with safe_open(str(artifact_path), framework="pt", device="cpu") as safe_file:
        for expert in range(NUM_EXPERTS):
            prefix = f"model.layers.0.mlp.experts.{expert}."
            gate = _read_safetensor(
                safe_file, source_tensors, prefix + "gate_proj.weight",
                (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            )
            up = _read_safetensor(
                safe_file, source_tensors, prefix + "up_proj.weight",
                (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            )
            expert_down = _read_safetensor(
                safe_file, source_tensors, prefix + "down_proj.weight",
                (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            )
            gate_up[expert, :INTERMEDIATE_SIZE].copy_(gate)
            gate_up[expert, INTERMEDIATE_SIZE:].copy_(up)
            down[expert].copy_(expert_down)
            del gate, up, expert_down
    _atomic_torch_save(output_path, {"gate_up": gate_up, "down": down})
    return {"gate_up": gate_up, "down": down}


def _make_runtime_modules(
    config: Any,
    model_dir: Path,
    source_tensors: dict[str, dict[str, Any]],
    device: torch.device,
    safe_open: Any,
    decoder_layer_type: Any,
    rotary_type: Any,
) -> tuple[torch.Tensor, nn.Module, nn.Module, nn.Module]:
    artifact_path = model_dir / "layer0.safetensors"
    shapes = _required_tensor_shapes()
    with safe_open(str(artifact_path), framework="pt", device="cpu") as safe_file:
        embedding = _read_safetensor(
            safe_file, source_tensors, "model.embed_tokens.weight", shapes["model.embed_tokens.weight"]
        )
        input_norm_weight = _read_safetensor(
            safe_file, source_tensors, "model.layers.0.input_layernorm.weight", (HIDDEN_SIZE,)
        )
        post_norm_weight = _read_safetensor(
            safe_file, source_tensors, "model.layers.0.post_attention_layernorm.weight", (HIDDEN_SIZE,)
        )
        router_weight = _read_safetensor(
            safe_file, source_tensors, "model.layers.0.mlp.gate.weight", (NUM_EXPERTS, HIDDEN_SIZE)
        )
        attention_state: dict[str, torch.Tensor] = {}
        for suffix in (
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
            "q_norm.weight", "k_norm.weight",
        ):
            source_name = f"model.layers.0.self_attn.{suffix}"
            local_name = suffix
            attention_state[local_name] = _read_safetensor(
                safe_file, source_tensors, source_name, shapes[source_name]
            )

    config._attn_implementation = "eager"
    with torch.device("meta"):
        layer = decoder_layer_type(config, layer_idx=0)
    if not hasattr(layer.mlp, "gate") or len(layer.mlp.experts) != NUM_EXPERTS:
        raise CaptureError("Transformers constructed an unexpected layer-0 MLP implementation")

    layer.input_layernorm.load_state_dict(
        {"weight": input_norm_weight}, strict=True, assign=True
    )
    layer.post_attention_layernorm.load_state_dict(
        {"weight": post_norm_weight}, strict=True, assign=True
    )
    layer.self_attn.load_state_dict(attention_state, strict=True, assign=True)
    layer.mlp.gate.load_state_dict(
        {"weight": router_weight}, strict=True, assign=True
    )

    # Only real layer-0 attention/norm/router modules move to the requested
    # device. The 128 expert modules remain meta and are discarded below.
    layer.input_layernorm.to(device)
    layer.post_attention_layernorm.to(device)
    layer.self_attn.to(device)
    real_gate = layer.mlp.gate
    real_gate.to(device)
    top_k = layer.mlp.top_k
    norm_topk_prob = layer.mlp.norm_topk_prob
    layer.mlp = _RouterCaptureStub(real_gate, top_k, norm_topk_prob)
    del router_weight, input_norm_weight, post_norm_weight, attention_state
    layer.eval()

    rotary = rotary_type(config=config).to(device)
    embedding = embedding.contiguous()
    return embedding, layer, rotary, layer.mlp


class _RouterCaptureStub(nn.Module):
    """Keep the official router module while eliding the expert loop."""

    def __init__(self, gate: nn.Module, top_k: int, norm_topk_prob: bool):
        super().__init__()
        self.gate = gate
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.last_x: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor):
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        self.last_x = flat.detach()
        router_logits = self.gate(flat)
        return torch.zeros_like(hidden_states), router_logits


def _prompt_token_ids(tokenizer: Any, prompt: str, tokens: int) -> tuple[torch.Tensor, str]:
    seed = PROMPT_SEEDS[prompt]
    repeated = seed
    while True:
        token_ids = tokenizer.encode(repeated, add_special_tokens=False)
        if len(token_ids) >= tokens:
            break
        repeated = repeated + "\n" + repeated
        if len(repeated) > 4_000_000:
            raise CaptureError(f"could not construct {tokens} tokenizer IDs for prompt {prompt}")
    used_text = repeated
    result = torch.tensor(token_ids[:tokens], dtype=torch.long)
    return result, hashlib.sha256(used_text.encode("utf-8")).hexdigest()


def _capture_case(
    token_ids_cpu: torch.Tensor,
    *,
    device: torch.device,
    config: Any,
    layer: nn.Module,
    rotary: nn.Module,
    router_stub: _RouterCaptureStub,
    embedding_weight: torch.Tensor,
    create_causal_mask: Any,
    create_sliding_window_causal_mask: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = token_ids_cpu.numel()
    inputs_embeds = F.embedding(token_ids_cpu, embedding_weight).unsqueeze(0).to(device=device)
    cache_position = torch.arange(tokens, device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)
    mask_function = (
        create_causal_mask
        if config.sliding_window is None
        else create_sliding_window_causal_mask
    )
    attention_mask = mask_function(
        config=config,
        input_embeds=inputs_embeds,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )
    if attention_mask is None:
        raise CaptureError("official eager causal-mask helper returned no mask")
    position_embeddings = rotary(inputs_embeds, position_ids)
    router_stub.last_x = None
    outputs = layer(
        hidden_states=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=True,
        use_cache=False,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    router_logits = outputs[-1]
    x_device = router_stub.last_x
    if x_device is None or router_logits is None:
        raise CaptureError("official decoder path did not produce router input and logits")
    if x_device.dtype != torch.bfloat16 or router_logits.dtype != torch.bfloat16:
        raise CaptureError("pinned Qwen3 layer-0 is expected to route BF16 hidden states")

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
    if router_stub.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(x_device.dtype)
    route_weights_cpu = routing_weights.to(torch.float32).to("cpu").contiguous()
    return (
        x_device.to("cpu").contiguous(),
        selected_experts.to(torch.int32).to("cpu").contiguous(),
        route_weights_cpu,
    )


def _source_snapshot(
    named_paths: dict[str, Path],
    official_paths: dict[str, Path],
) -> tuple[dict[str, str], dict[str, str]]:
    all_paths = {**named_paths, **official_paths}
    hashes = {name: _sha256_file(path) for name, path in all_paths.items()}
    resolved = {name: str(path.resolve()) for name, path in all_paths.items()}
    return hashes, resolved


def _artifact_records(output_dir: Path, cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    paths = [output_dir / "weights.pt"]
    for case in cases:
        paths.append(output_dir / case["manifest"].replace(".json", ".pt"))
        paths.append(output_dir / case["manifest"])
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        result[path.relative_to(output_dir).as_posix()] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("/home/tmz/models/Qwen3-30B-A3B-layer0"),
        help="directory containing the verified layer0.safetensors extraction",
    )
    parser.add_argument(
        "--tokenizer-dir", type=Path,
        default=Path("/home/tmz/models/Qwen3-30B-A3B-layer0/tokenizer"),
        help="directory containing the pinned tokenizer assets and source manifest",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="new capture directory; it must be outside this Git repository",
    )
    parser.add_argument(
        "--transformers-root", type=Path,
        default=Path(os.environ.get(
            "VLLM_OPERATOR_RUNTIME", "/home/tmz/ai/baselines/vllm-0.10.0"
        )),
        help="existing vLLM overlay containing pinned Transformers 4.53.2",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def run(args: argparse.Namespace) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    tokenizer_dir = args.tokenizer_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    runtime_root = args.transformers_root.expanduser().resolve(strict=True)
    if output_dir == repo_root or repo_root in output_dir.parents:
        raise CaptureError("capture outputs must be outside the Git repository")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CaptureError(f"refusing to overwrite a non-empty output directory: {output_dir}")

    extraction_manifest, model_paths, model_hashes = _validate_extraction_bundle(model_dir)
    source_tensors = _validate_tensor_manifest(extraction_manifest)
    tokenizer_paths, tokenizer_hashes = _verify_tokenizer_sources(tokenizer_dir)
    runtime = _prepare_runtime(runtime_root, repo_root)
    config = runtime["Qwen3MoeConfig"].from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    _validate_config(config)
    tokenizer = runtime["AutoTokenizer"].from_pretrained(
        str(tokenizer_dir), local_files_only=True,
        trust_remote_code=False, use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise CaptureError("the pinned tokenizer must load through its fast tokenizer implementation")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CaptureError(f"requested CUDA device is unavailable: {device}")
    if device.type == "cuda":
        try:
            torch.cuda.get_device_properties(device)
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise CaptureError(f"requested CUDA device cannot be inspected: {device}") from exc

    official_root = Path(runtime["transformers"].__file__).resolve().parent
    official_paths = {
        "transformers_qwen3_moe_modeling": official_root / "models/qwen3_moe/modeling_qwen3_moe.py",
        "transformers_qwen3_moe_config": official_root / "models/qwen3_moe/configuration_qwen3_moe.py",
        "transformers_masking": official_root / "masking_utils.py",
        "transformers_rope": official_root / "modeling_rope_utils.py",
        "capture_script": Path(__file__).resolve(),
        "trace_writer": repo_root / "src/llm_gpu_kernels/moe_real_trace.py",
    }
    named_paths = {**model_paths, **tokenizer_paths}
    if any(not path.is_file() for path in official_paths.values()):
        missing = [str(path) for path in official_paths.values() if not path.is_file()]
        raise CaptureError(f"source files needed for provenance are missing: {missing}")
    source_hashes_before, source_paths = _source_snapshot(named_paths, official_paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema": "llm_gpu_kernels.qwen3_moe_layer0_capture",
        "schema_version": 1,
        "status": "running",
        "model": {
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "capture_scope": "layer0_moe_input_and_fixed_router",
            "num_experts": NUM_EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "top_k": TOP_K,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": runtime["transformers"].__version__,
            "safetensors": runtime["safetensors"].__version__,
            "device": str(device),
            "attention_backend": "eager",
        },
        "source_paths": source_paths,
        "source_hashes_before": source_hashes_before,
        "tokenizer": {
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "files": tokenizer_hashes,
        },
        "routing_capture": {
            "source": "Qwen3MoeSparseMoeBlock router gate and official forward semantics",
            "formula": "softmax(fp32 logits) -> topk(8) -> config norm_topk_prob -> BF16 -> exact FP32 promotion",
            "official_expert_loop": "not run; only true hidden input, router logits, IDs, and weights are captured",
        },
        "cases": [],
    }
    summary_path = output_dir / "capture-summary.json"
    _atomic_json_write(summary_path, summary)

    try:
        source_manifest_sha256 = source_hashes_before["extraction_manifest"]
        shared_weights = _write_shared_weights(
            model_dir / "layer0.safetensors",
            source_tensors,
            output_dir / "weights.pt",
            runtime["safe_open"],
        )
        weights_sha256 = _sha256_file(output_dir / "weights.pt")
        del shared_weights

        # The full embedding table stays on CPU; only prompt embeddings and
        # layer-0 execution are transferred to the selected compute device.
        embedding_weight, layer, rotary, router_stub = _make_runtime_modules(
            config,
            model_dir,
            source_tensors,
            device,
            runtime["safe_open"],
            runtime["Qwen3MoeDecoderLayer"],
            runtime["Qwen3MoeRotaryEmbedding"],
        )

        weight_relative = "../weights.pt"
        weight_refs = {
            "gate_up": {
                "path": weight_relative,
                "sha256": weights_sha256,
                "tensor_name": "gate_up",
                "shape": [NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE],
                "dtype": "torch.bfloat16",
                "source_manifest_sha256": source_manifest_sha256,
            },
            "down": {
                "path": weight_relative,
                "sha256": weights_sha256,
                "tensor_name": "down",
                "shape": [NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE],
                "dtype": "torch.bfloat16",
                "source_manifest_sha256": source_manifest_sha256,
            },
        }

        from llm_gpu_kernels.moe_real_trace import write_real_moe_trace

        with torch.inference_mode():
            for prompt in ("archive", "code"):
                for tokens in PROMPT_LENGTHS:
                    token_ids, prompt_seed_sha256 = _prompt_token_ids(
                        tokenizer, prompt, tokens
                    )
                    x, expert_ids, route_weights = _capture_case(
                        token_ids,
                        device=device,
                        config=config,
                        layer=layer,
                        rotary=rotary,
                        router_stub=router_stub,
                        embedding_weight=embedding_weight,
                        create_causal_mask=runtime["create_causal_mask"],
                        create_sliding_window_causal_mask=runtime[
                            "create_sliding_window_causal_mask"
                        ],
                    )
                    case_name = f"{prompt}-T{tokens}"
                    manifest_rel = f"traces/{case_name}.json"
                    trace_rel = f"traces/{case_name}.pt"
                    token_id_values = token_ids.tolist()
                    token_ids_sha256 = hashlib.sha256(
                        b"".join(int(item).to_bytes(8, "little", signed=False)
                                 for item in token_id_values)
                    ).hexdigest()
                    provenance = {
                        "layer": 0,
                        "capture_kind": "real_model_layer0_moe_input",
                        "prompt": prompt,
                        "tokens": tokens,
                        "prompt_seed_sha256": prompt_seed_sha256,
                        "tokenizer_repo_id": MODEL_REPO_ID,
                        "tokenizer_revision": MODEL_REVISION,
                        "tokenizer_asset_sha256": {
                            name: tokenizer_hashes[name]
                            for name in ("tokenizer.json", "tokenizer_config.json")
                        },
                        "token_ids": token_id_values,
                        "token_ids_sha256": token_ids_sha256,
                        "add_special_tokens": False,
                        "attention_backend": "eager",
                        "weight_concat_order": "gate_proj rows then up_proj rows",
                        "source_extraction_manifest_sha256": source_manifest_sha256,
                        "decoder_execution": (
                            "official Qwen3MoeDecoderLayer attention/norm path; router-only "
                            "MLP stub returns zero expert output and preserves official gate logits"
                        ),
                        "expert_outputs": "not captured; validate routed outputs with the independent FP32 oracle",
                    }
                    write_real_moe_trace(
                        output_dir / manifest_rel,
                        output_dir / trace_rel,
                        x=x,
                        expert_ids=expert_ids,
                        route_weights=route_weights,
                        provenance=provenance,
                        weights=weight_refs,
                    )
                    summary["cases"].append({
                        "name": case_name,
                        "manifest": manifest_rel,
                        "tokens": tokens,
                        "prompt": prompt,
                    })
                    summary["status"] = "running"
                    _atomic_json_write(summary_path, summary)
                    del x, expert_ids, route_weights, token_ids

        source_hashes_after, _ = _source_snapshot(named_paths, official_paths)
        summary["source_hashes_after"] = source_hashes_after
        if source_hashes_after != source_hashes_before:
            raise CaptureError("one or more source files changed during capture")
        expected_case_names = {
            f"{prompt}-T{tokens}"
            for prompt in ("archive", "code")
            for tokens in PROMPT_LENGTHS
        }
        if len(summary["cases"]) != 8 or {
            case["name"] for case in summary["cases"]
        } != expected_case_names:
            raise CaptureError("capture did not produce the complete two-prompt, four-length matrix")
        summary["artifact_hashes"] = _artifact_records(output_dir, summary["cases"])
        summary["status"] = "passed"
        _atomic_json_write(summary_path, summary)
        return summary_path
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        try:
            source_hashes_after, _ = _source_snapshot(named_paths, official_paths)
            summary["source_hashes_after"] = source_hashes_after
        except BaseException as hash_exc:
            summary["source_hashes_after_error"] = f"{type(hash_exc).__name__}: {hash_exc}"
        try:
            summary["artifact_hashes"] = _artifact_records(
                output_dir, summary["cases"]
            ) if (output_dir / "weights.pt").is_file() else {}
        except BaseException as hash_exc:
            summary["artifact_hashes_error"] = f"{type(hash_exc).__name__}: {hash_exc}"
        _atomic_json_write(summary_path, summary)
        raise


def main() -> int:
    args = _parser().parse_args()
    try:
        summary_path = run(args)
    except Exception as exc:
        print(f"capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"capture summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
