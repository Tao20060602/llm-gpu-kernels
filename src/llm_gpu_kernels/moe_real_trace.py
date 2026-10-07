"""Safe artifact I/O and validation for the pinned real Qwen3 MoE trace.

This module does not run a router or benchmark.  It accepts already captured
CPU tensors, stores only the trace tensors, and keeps the large extracted
expert-weight bundle as a hashed local reference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
from urllib.parse import urlsplit

import torch


SCHEMA = "llm_gpu_kernels.moe_real_trace"
SCHEMA_VERSION = 1
MODEL_REPO_ID = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
NUM_EXPERTS = 128
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
TOP_K = 8

TRACE_TENSOR_NAMES = ("x", "expert_ids", "route_weights")
WEIGHT_SHAPES = {
    "gate_up": (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
    "down": (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
}
_SHA256_LEN = 64


@dataclass(frozen=True)
class RealMoETrace:
    """A validated manifest and optionally materialized CPU tensor bundles."""

    manifest_path: Path
    manifest: dict[str, Any]
    tensors: dict[str, torch.Tensor] | None
    weights: dict[str, torch.Tensor] | None


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return a streaming SHA256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor values together with dtype and shape on CPU."""
    if tensor.device.type != "cpu":
        raise ValueError("artifact tensor hashing requires a CPU tensor")
    value = tensor.detach().contiguous()
    digest = hashlib.sha256()
    descriptor = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    digest.update(descriptor)
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _check_sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != _SHA256_LEN
            or any(char not in "0123456789abcdef" for char in value.lower())):
        raise ValueError(f"{label} must be a 64-character SHA256 hex digest")
    return value.lower()


def _cpu_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be copied to CPU before artifact validation")
    return value


def validate_trace_tensors(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Validate captured CPU inputs and return detached contiguous copies.

    The fixed-route artifact records the official router's BF16 weights after
    exact promotion to FP32.  They are preserved as supplied; no top-k
    renormalization or other transform is applied here.
    """
    x = _cpu_tensor(x, "x")
    expert_ids = _cpu_tensor(expert_ids, "expert_ids")
    route_weights = _cpu_tensor(route_weights, "route_weights")

    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != HIDDEN_SIZE:
        raise ValueError(f"x must have shape [T,{HIDDEN_SIZE}] with T > 0")
    tokens = x.shape[0]
    if x.dtype != torch.bfloat16:
        raise TypeError("x must use bfloat16")
    if not torch.isfinite(x).all():
        raise ValueError("x contains nonfinite values")

    expected_route_shape = (tokens, TOP_K)
    if tuple(expert_ids.shape) != expected_route_shape:
        raise ValueError(f"expert_ids must have shape {expected_route_shape}")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_ids must use int32 or int64")
    if torch.any((expert_ids < 0) | (expert_ids >= NUM_EXPERTS)):
        raise ValueError(f"expert_ids must lie in [0,{NUM_EXPERTS})")
    sorted_ids = expert_ids.sort(dim=1).values
    if torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
        raise ValueError("top-k expert IDs must be unique within each token")

    if tuple(route_weights.shape) != expected_route_shape:
        raise ValueError(f"route_weights must have shape {expected_route_shape}")
    if route_weights.dtype != torch.float32:
        raise TypeError("route_weights must be stored as float32")
    if not torch.isfinite(route_weights).all():
        raise ValueError("route_weights contains nonfinite values")
    if torch.any(route_weights < 0):
        raise ValueError("route_weights must be nonnegative")
    promoted = route_weights.to(torch.bfloat16).to(torch.float32)
    if not torch.equal(promoted, route_weights):
        raise ValueError("route_weights must be exact FP32 promotions of BF16 router outputs")

    return {
        "x": x.detach().contiguous().clone(),
        "expert_ids": expert_ids.detach().contiguous().clone(),
        "route_weights": route_weights.detach().contiguous().clone(),
    }


def validate_weight_tensors(weights: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Validate already materialized CPU `gate_up` and `down` tensors."""
    if not isinstance(weights, Mapping) or set(weights) != set(WEIGHT_SHAPES):
        raise ValueError("weights must contain exactly 'gate_up' and 'down'")
    result: dict[str, torch.Tensor] = {}
    for name, expected_shape in WEIGHT_SHAPES.items():
        tensor = _cpu_tensor(weights[name], name)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must use bfloat16")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains nonfinite values")
        result[name] = tensor
    return result


def _safe_local_path(value: Any, base_dir: Path, label: str, *, must_exist: bool) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{label} must be a local filesystem path")
    raw = os.fspath(value)
    parsed = urlsplit(raw)
    if parsed.scheme or "://" in raw:
        raise ValueError(f"{label} must not be a URL or remote artifact")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    try:
        resolved = path.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if must_exist and not resolved.is_file():
        raise ValueError(f"{label} must name a regular file: {resolved}")
    return resolved


def _manifest_path(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return str(path)


def _normalize_weight_refs(
    weights: Mapping[str, Mapping[str, Any]],
    base_dir: Path,
    *,
    hash_cache: dict[Path, str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(weights, Mapping) or set(weights) != set(WEIGHT_SHAPES):
        raise ValueError("weights references must contain exactly 'gate_up' and 'down'")
    result: dict[str, dict[str, Any]] = {}
    for name, expected_shape in WEIGHT_SHAPES.items():
        ref = weights[name]
        if not isinstance(ref, Mapping):
            raise TypeError(f"weights[{name!r}] must be a mapping")
        path = _safe_local_path(ref.get("path"), base_dir, f"weights[{name}].path", must_exist=True)
        if path.suffix.lower() != ".pt":
            raise ValueError("weight bundles must be local .pt files loaded with weights_only=True")
        tensor_name = ref.get("tensor_name")
        if tensor_name != name:
            raise ValueError(f"weights[{name}].tensor_name must be {name!r}")
        shape = ref.get("shape")
        if not isinstance(shape, (list, tuple)) or tuple(shape) != expected_shape:
            raise ValueError(f"weights[{name}].shape must be {list(expected_shape)}")
        dtype = str(ref.get("dtype", "")).removeprefix("torch.")
        if dtype != "bfloat16":
            raise ValueError(f"weights[{name}].dtype must be 'bfloat16'")
        source_manifest_sha256 = _check_sha256(
            ref.get("source_manifest_sha256"), f"weights[{name}].source_manifest_sha256"
        )
        if path not in hash_cache:
            supplied_hash = ref.get("sha256")
            if supplied_hash is None:
                hash_cache[path] = sha256_file(path)
            else:
                # Capturers may pass the digest they computed while saving the
                # bundle. Full re-verification remains available before use.
                hash_cache[path] = _check_sha256(supplied_hash, f"weights[{name}].sha256")
        elif ref.get("sha256") is not None:
            supplied_hash = _check_sha256(ref["sha256"], f"weights[{name}].sha256")
            if supplied_hash != hash_cache[path]:
                raise ValueError("shared weight references disagree on their SHA256")
        result[name] = {
            "path": _manifest_path(path, base_dir),
            "sha256": hash_cache[path],
            "format": "torch_weights_only_bundle",
            "tensor_name": tensor_name,
            "shape": list(expected_shape),
            "dtype": "bfloat16",
            "source_manifest_sha256": source_manifest_sha256,
        }
    return result


def _atomic_torch_save(path: Path, payload: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temp_path = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


def write_real_moe_trace(
    manifest_path: str | os.PathLike[str],
    trace_path: str | os.PathLike[str],
    *,
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    provenance: Mapping[str, Any],
    weights: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Write a JSON manifest and safe tensor-only trace bundle.

    `weights` references an already written local bundle with `gate_up` and
    `down` entries.  The bundle is not copied or loaded here, avoiding another
    large allocation; its supplied SHA256 is retained and can be rechecked by
    :func:`validate_real_moe_trace` before any benchmark timing.
    """
    manifest_target = Path(manifest_path).expanduser()
    if not manifest_target.is_absolute():
        manifest_target = Path.cwd() / manifest_target
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target = manifest_target.resolve()
    if manifest_target.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_target}")
    base_dir = manifest_target.parent
    trace_target = _safe_local_path(trace_path, base_dir, "trace_path", must_exist=False)
    if trace_target.suffix.lower() != ".pt":
        raise ValueError("trace bundle path must use the .pt extension")
    if trace_target == manifest_target:
        raise ValueError("trace_path and manifest_path must be different files")
    if trace_target.exists():
        raise FileExistsError(f"refusing to overwrite existing trace artifact: {trace_target}")

    trace_tensors = validate_trace_tensors(x, expert_ids, route_weights)
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a JSON-safe mapping")
    try:
        capture = json.loads(json.dumps(dict(provenance), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance must contain only finite JSON values") from exc

    hash_cache: dict[Path, str] = {}
    weight_records = _normalize_weight_refs(weights, base_dir, hash_cache=hash_cache)
    weight_paths = {
        _safe_local_path(weights[name]["path"], base_dir, f"weights[{name}].path", must_exist=True)
        for name in WEIGHT_SHAPES
    }
    if trace_target in weight_paths or manifest_target in weight_paths:
        raise ValueError("manifest and trace outputs must not overwrite the referenced weight bundle")
    _atomic_torch_save(trace_target, trace_tensors)
    trace_hash = sha256_file(trace_target)
    tensor_records = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "sha256": _tensor_sha256(tensor),
        }
        for name, tensor in trace_tensors.items()
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "model": {
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "num_experts": NUM_EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "top_k": TOP_K,
        },
        "routing_contract": {
            "source_dtype": "bfloat16",
            "stored_dtype": "float32",
            "conversion": "exact_bfloat16_to_float32_promotion",
            "renormalization": "none",
            "unique_expert_ids_per_token": True,
        },
        "trace": {
            "path": _manifest_path(trace_target, base_dir),
            "sha256": trace_hash,
            "format": "torch_weights_only_tensor_dict",
            "tensors": tensor_records,
        },
        "weights": weight_records,
        "capture": capture,
    }
    _atomic_json_write(manifest_target, manifest)
    return manifest


def _validate_manifest_shape(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be a JSON object")
    if manifest.get("schema") != SCHEMA or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported real-MoE trace manifest schema")
    if manifest.get("status") != "passed":
        raise ValueError("real-MoE trace manifest is not marked passed")
    expected_model = {
        "repo_id": MODEL_REPO_ID,
        "revision": MODEL_REVISION,
        "num_experts": NUM_EXPERTS,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "top_k": TOP_K,
    }
    if manifest.get("model") != expected_model:
        raise ValueError("manifest does not identify the pinned Qwen3-30B-A3B contract")
    expected_routing = {
        "source_dtype": "bfloat16",
        "stored_dtype": "float32",
        "conversion": "exact_bfloat16_to_float32_promotion",
        "renormalization": "none",
        "unique_expert_ids_per_token": True,
    }
    if manifest.get("routing_contract") != expected_routing:
        raise ValueError("manifest routing contract differs from the fixed trace contract")
    if not isinstance(manifest.get("capture"), dict):
        raise ValueError("manifest capture metadata must be a JSON object")

    trace_record = manifest.get("trace")
    if not isinstance(trace_record, dict):
        raise ValueError("manifest trace entry must be an object")
    if trace_record.get("format") != "torch_weights_only_tensor_dict":
        raise ValueError("trace bundle format must be weights-only tensor dictionary")
    _check_sha256(trace_record.get("sha256"), "trace.sha256")
    tensors = trace_record.get("tensors")
    if not isinstance(tensors, dict) or set(tensors) != set(TRACE_TENSOR_NAMES):
        raise ValueError("trace tensor metadata must contain x, expert_ids, route_weights")
    token_counts: dict[str, int] = {}
    expected_dtypes = {"x": "bfloat16", "expert_ids": None, "route_weights": "float32"}
    for name, record in tensors.items():
        if not isinstance(record, dict):
            raise ValueError(f"trace.tensors.{name} must be an object")
        shape = record.get("shape")
        if (not isinstance(shape, list) or any(type(size) is not int for size in shape)):
            raise ValueError(f"trace.tensors.{name}.shape must be a list")
        _check_sha256(record.get("sha256"), f"trace.tensors.{name}.sha256")
        if name == "x":
            if len(shape) != 2 or shape[0] <= 0 or shape[1] != HIDDEN_SIZE:
                raise ValueError("manifest x shape is incompatible with pinned hidden size")
            token_counts[name] = shape[0]
        elif len(shape) != 2 or shape[0] <= 0 or shape[1] != TOP_K:
            raise ValueError(f"manifest {name} shape must be [T,{TOP_K}]")
        else:
            token_counts[name] = shape[0]
        dtype = record.get("dtype")
        if name == "expert_ids":
            if dtype not in ("int32", "int64"):
                raise ValueError("manifest expert_ids dtype must be int32 or int64")
        elif dtype != expected_dtypes[name]:
            raise ValueError(f"manifest {name} dtype must be {expected_dtypes[name]}")
    if len(set(token_counts.values())) != 1:
        raise ValueError("all manifest trace tensors must have the same token count T")

    weight_records = manifest.get("weights")
    if not isinstance(weight_records, dict) or set(weight_records) != set(WEIGHT_SHAPES):
        raise ValueError("manifest weights must contain gate_up and down")
    source_manifest_hashes = set()
    for name, expected_shape in WEIGHT_SHAPES.items():
        record = weight_records[name]
        if not isinstance(record, dict):
            raise ValueError(f"manifest weights.{name} must be an object")
        if record.get("format") != "torch_weights_only_bundle":
            raise ValueError(f"manifest weights.{name} must declare a weights-only torch bundle")
        if record.get("tensor_name") != name:
            raise ValueError(f"manifest weights.{name}.tensor_name must match its role")
        if record.get("shape") != list(expected_shape) or record.get("dtype") != "bfloat16":
            raise ValueError(f"manifest weights.{name} shape/dtype violate the fixed contract")
        source_manifest_hashes.add(_check_sha256(
            record.get("source_manifest_sha256"), f"weights.{name}.source_manifest_sha256"
        ))
        _check_sha256(record.get("sha256"), f"weights.{name}.sha256")
    if len(source_manifest_hashes) != 1:
        raise ValueError("gate_up and down must cite the same layer-extraction manifest")
    if (weight_records["gate_up"].get("path") != weight_records["down"].get("path")
            or weight_records["gate_up"].get("sha256") != weight_records["down"].get("sha256")):
        raise ValueError("gate_up and down must reference the same shared weight bundle")
    return manifest


def validate_real_moe_trace(
    manifest_path: str | os.PathLike[str],
    *,
    verify_hashes: bool = True,
    load_trace: bool = True,
    load_weights: bool = False,
) -> RealMoETrace:
    """Validate a manifest, optionally safe-loading its CPU tensor bundles.

    All deserialization uses `weights_only=True`; remote URLs and pickle-style
    bundle formats are rejected. Hash verification is outside any operator
    timing and deduplicates shared paths within this validation call.
    """
    target = Path(manifest_path).expanduser().resolve(strict=True)
    if not target.is_file():
        raise ValueError("manifest_path must name a regular file")
    manifest = _validate_manifest_shape(json.loads(target.read_text(encoding="utf-8")))
    base_dir = target.parent

    records: dict[str, dict[str, Any]] = {"trace": manifest["trace"]}
    records.update({f"weights.{key}": value for key, value in manifest["weights"].items()})
    resolved: dict[str, Path] = {}
    for key, record in records.items():
        path = _safe_local_path(record.get("path"), base_dir, f"{key}.path", must_exist=True)
        if path.suffix.lower() != ".pt":
            raise ValueError(f"{key} artifact must be a local .pt weights-only bundle")
        resolved[key] = path
    if verify_hashes:
        hash_cache: dict[Path, str] = {}
        for key, record in records.items():
            path = resolved[key]
            if path not in hash_cache:
                hash_cache[path] = sha256_file(path)
            if hash_cache[path] != _check_sha256(record.get("sha256"), f"{key}.sha256"):
                raise ValueError(f"SHA256 mismatch for {key} artifact: {path}")

    trace_tensors: dict[str, torch.Tensor] | None = None
    if load_trace:
        payload = torch.load(resolved["trace"], map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != set(TRACE_TENSOR_NAMES):
            raise ValueError("trace .pt must contain only x, expert_ids, and route_weights tensors")
        trace_tensors = validate_trace_tensors(
            payload["x"], payload["expert_ids"], payload["route_weights"]
        )
        for name, tensor in trace_tensors.items():
            record = manifest["trace"]["tensors"][name]
            if list(tensor.shape) != record["shape"] or str(tensor.dtype).removeprefix("torch.") != record["dtype"]:
                raise ValueError(f"loaded trace tensor {name} disagrees with manifest metadata")
            if _tensor_sha256(tensor) != record["sha256"]:
                raise ValueError(f"tensor content hash mismatch for trace tensor {name}")

    weight_tensors: dict[str, torch.Tensor] | None = None
    if load_weights:
        loaded_bundles: dict[Path, Any] = {}
        weight_tensors = {}
        for name, record in manifest["weights"].items():
            path = resolved[f"weights.{name}"]
            if path not in loaded_bundles:
                loaded = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(loaded, dict):
                    raise ValueError("weight .pt must deserialize to a tensor dictionary")
                loaded_bundles[path] = loaded
            bundle = loaded_bundles[path]
            if set(bundle) != set(WEIGHT_SHAPES):
                raise ValueError("weight bundle must contain only gate_up and down")
            if record["tensor_name"] not in bundle:
                raise ValueError(f"weight bundle lacks tensor {record['tensor_name']!r}")
            weight_tensors[name] = bundle[record["tensor_name"]]
        weight_tensors = validate_weight_tensors(weight_tensors)

    return RealMoETrace(
        manifest_path=target,
        manifest=manifest,
        tensors=trace_tensors,
        weights=weight_tensors,
    )
