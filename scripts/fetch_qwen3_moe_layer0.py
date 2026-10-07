#!/usr/bin/env python3
"""Fetch only the tensors needed to run Qwen3-30B-A3B decoder layer 0.

The source repository and revision are intentionally pinned. The downloader
uses HTTP byte ranges against the official Hugging Face resolve endpoint and
refuses any tensor-data request that is not returned as the exact requested
206 response. It creates a valid safetensors subset with the source tensor
names preserved; it does not download the complete 30B checkpoint or a whole
source shard.

No network access or file writes occur at import time. Run explicitly with:

    python scripts/fetch_qwen3_moe_layer0.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


MODEL_ID = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/"
DEFAULT_OUTPUT = Path("/home/tmz/models/Qwen3-30B-A3B-layer0")

CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
SOURCE_SHARD = "model-00001-of-00016.safetensors"
OUTPUT_SHARD = "layer0.safetensors"

# Audited from the pinned repository revision. These protect against a changed
# source snapshot and against an unexpectedly different safetensors layout.
CONFIG_SHA256 = "2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297"
INDEX_SHA256 = "df0d481ec595c55a0ba58426d517390c6214a566ec4ff1c8fc4bbce9f57b3c24"
SHARD_SIZE_BYTES = 3_999_417_504
SHARD_ETAG = '"af78e9d0c41f3c2a670638e235791def9c89716420138516f396415a7360e65f"'
SHARD_HEADER_BYTES = 130_712
SHARD_HEADER_SHA256 = "b749467c61b8bc9b6d136ae30ee514fb76708a76915417d7744d91d02e139ee4"
FULL_CHECKPOINT_TENSOR_BYTES = 61_064_245_248
LAYER0_TENSOR_BYTES = 1_868_571_136

MAX_METADATA_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_DOWNLOAD_BYTES = 2 * 1024**3
DEFAULT_CHUNK_BYTES = 32 * 1024**2
ALLOWED_CHUNK_BYTES = (16 * 1024**2, 32 * 1024**2)
DEFAULT_WORKERS = 4
MAX_WORKERS = 4
MAX_RETRIES = 4

USER_AGENT = "llm-gpu-kernels-layer0-fetch/1.0 (pinned read-only source)"


class FetchError(RuntimeError):
    """Raised for a source, layout, integrity, or download-boundary failure."""


class RangeBudget:
    """Conservatively cap all ranged response bytes, including retries."""

    def __init__(self, maximum_bytes: int) -> None:
        self.maximum_bytes = maximum_bytes
        self.reserved_bytes = 0
        self.lock = threading.Lock()

    def reserve(self, requested_bytes: int) -> None:
        with self.lock:
            requested_total = self.reserved_bytes + requested_bytes
            if requested_total > self.maximum_bytes:
                raise FetchError(
                    f"range requests would exceed the remaining byte budget "
                    f"({requested_total} > {self.maximum_bytes})"
                )
            self.reserved_bytes = requested_total


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, start: int = 0, length: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as stream:
        stream.seek(start)
        while remaining is None or remaining:
            block_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            block = stream.read(block_size)
            if not block:
                if remaining not in (None, 0):
                    raise FetchError(f"short local read in {path} at byte {stream.tell()}")
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    request_headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if headers:
        request_headers.update(headers)
    return urllib.request.Request(url, headers=request_headers)


def _download_metadata(url: str, *, maximum_bytes: int, expected_sha256: str) -> tuple[bytes, dict[str, Any]]:
    try:
        with urllib.request.urlopen(_request(url), timeout=60) as response:
            if response.status != 200:
                raise FetchError(f"metadata fetch expected HTTP 200, got {response.status}: {url}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > maximum_bytes:
                raise FetchError(f"metadata exceeds {maximum_bytes} byte cap: {url}")
            if response.headers.get("Content-Encoding", "identity") not in ("", "identity"):
                raise FetchError(f"compressed metadata response is not accepted: {url}")
            data = response.read(maximum_bytes + 1)
            if len(data) > maximum_bytes:
                raise FetchError(f"metadata exceeds {maximum_bytes} byte cap: {url}")
            if content_length is not None and len(data) != int(content_length):
                raise FetchError(
                    f"metadata length mismatch: got {len(data)}, Content-Length {content_length}: {url}"
                )
            actual_sha256 = _sha256(data)
            if actual_sha256 != expected_sha256:
                raise FetchError(
                    f"pinned metadata SHA256 mismatch for {url}: {actual_sha256}"
                )
            return data, {
                "url": url,
                "http_status": response.status,
                "bytes": len(data),
                "sha256": actual_sha256,
                "etag": response.headers.get("ETag"),
            }
    except urllib.error.HTTPError as exc:
        raise FetchError(f"metadata HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise FetchError(f"metadata network error for {url}: {exc}") from exc


def _content_range(response: Any, start: int, end: int, total: int) -> None:
    expected = f"bytes {start}-{end}/{total}"
    actual = response.headers.get("Content-Range")
    if actual != expected:
        raise FetchError(f"unexpected Content-Range {actual!r}; expected {expected!r}")


def _download_range_once(url: str, start: int, end: int) -> tuple[bytes, str]:
    request = _request(url, {"Range": f"bytes={start}-{end}"})
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        # In particular, never accept or read a full-shard HTTP 200 fallback.
        if exc.code == 200:
            exc.close()
            raise FetchError("server ignored Range and returned HTTP 200; refusing the shard") from exc
        raise

    with response:
        if response.status != 206:
            # Close before reading a possible multi-gigabyte 200 response.
            raise FetchError(
                f"Range request expected HTTP 206, got {response.status}; refusing the response"
            )
        _content_range(response, start, end, SHARD_SIZE_BYTES)
        if response.headers.get("Content-Encoding", "identity") not in ("", "identity"):
            raise FetchError("compressed byte-range responses are not accepted")
        expected_length = end - start + 1
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != expected_length:
            raise FetchError(
                f"range Content-Length {content_length} does not match {expected_length} bytes"
            )
        etag = response.headers.get("ETag")
        if etag != SHARD_ETAG:
            raise FetchError(f"source shard ETag changed: expected {SHARD_ETAG!r}, got {etag!r}")
        data = response.read(expected_length + 1)
        if len(data) != expected_length:
            raise FetchError(f"short/long range body: got {len(data)}, expected {expected_length}")
        return data, etag


def _download_range(
    url: str, start: int, end: int, *, budget: RangeBudget,
) -> tuple[bytes, str]:
    last_error: BaseException | None = None
    for attempt in range(MAX_RETRIES):
        try:
            # Reserve a full expected response before each try. Failed attempts
            # are not refunded, bounding retries conservatively.
            budget.reserve(end - start + 1)
            return _download_range_once(url, start, end)
        except FetchError:
            raise
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (408, 425, 429, 500, 502, 503, 504):
                raise FetchError(f"range HTTP {exc.code} for bytes {start}-{end}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt + 1 < MAX_RETRIES:
            time.sleep(min(2**attempt, 8))
    raise FetchError(f"range request failed after {MAX_RETRIES} attempts: {last_error}")


def _write_at(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = os.pwrite(fd, view[written:], offset + written)
        if count <= 0:
            raise FetchError(f"short write at output offset {offset + written}")
        written += count


def _read_at(fd: int, length: int, offset: int) -> bytes:
    pieces: list[bytes] = []
    read = 0
    while read < length:
        piece = os.pread(fd, length - read, offset + read)
        if not piece:
            break
        pieces.append(piece)
        read += len(piece)
    return b"".join(pieces)


def _target_shapes() -> dict[str, tuple[int, ...]]:
    expected: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (151_936, 2_048),
        "model.layers.0.input_layernorm.weight": (2_048,),
        "model.layers.0.post_attention_layernorm.weight": (2_048,),
        "model.layers.0.self_attn.q_proj.weight": (4_096, 2_048),
        "model.layers.0.self_attn.k_proj.weight": (512, 2_048),
        "model.layers.0.self_attn.v_proj.weight": (512, 2_048),
        "model.layers.0.self_attn.o_proj.weight": (2_048, 4_096),
        "model.layers.0.self_attn.q_norm.weight": (128,),
        "model.layers.0.self_attn.k_norm.weight": (128,),
        "model.layers.0.mlp.gate.weight": (128, 2_048),
    }
    for expert in range(128):
        prefix = f"model.layers.0.mlp.experts.{expert}."
        expected[prefix + "gate_proj.weight"] = (768, 2_048)
        expected[prefix + "up_proj.weight"] = (768, 2_048)
        expected[prefix + "down_proj.weight"] = (2_048, 768)
    return expected


def _required_shard(index: dict[str, Any], header: dict[str, Any]) -> list[dict[str, Any]]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise FetchError("safetensors index is missing its weight_map")
    if index.get("metadata", {}).get("total_size") != FULL_CHECKPOINT_TENSOR_BYTES:
        raise FetchError("pinned full-checkpoint total_size differs from the audited revision")

    expected = _target_shapes()
    actual_layer0_experts = {
        key for key in weight_map
        if re.fullmatch(r"model\.layers\.0\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight", key)
    }
    required_experts = {
        key for key in expected if ".mlp.experts." in key
    }
    if actual_layer0_experts != required_experts:
        missing = sorted(required_experts - actual_layer0_experts)[:5]
        unexpected = sorted(actual_layer0_experts - required_experts)[:5]
        raise FetchError(f"expert key set mismatch; missing={missing}, unexpected={unexpected}")

    if set(expected) - set(weight_map):
        raise FetchError(f"required tensors missing from index: {sorted(set(expected) - set(weight_map))[:8]}")
    if set(expected) - set(header):
        raise FetchError(f"required tensors missing from safetensors header: {sorted(set(expected) - set(header))[:8]}")

    tensors: list[dict[str, Any]] = []
    for key, expected_shape in expected.items():
        if weight_map[key] != SOURCE_SHARD:
            raise FetchError(f"unexpected shard for {key}: {weight_map[key]}")
        info = header[key]
        shape = tuple(info.get("shape", ()))
        dtype = info.get("dtype")
        offsets = info.get("data_offsets")
        if dtype != "BF16" or shape != expected_shape:
            raise FetchError(
                f"unexpected {key} metadata: dtype={dtype!r}, shape={shape!r}, expected BF16/{expected_shape}"
            )
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise FetchError(f"invalid data_offsets for {key}: {offsets!r}")
        start, end = offsets
        expected_bytes = 2
        for dimension in shape:
            expected_bytes *= dimension
        if not (isinstance(start, int) and isinstance(end, int) and end - start == expected_bytes):
            raise FetchError(f"invalid byte range for {key}: {offsets!r}")
        tensors.append({
            "name": key,
            "dtype": dtype,
            "shape": list(shape),
            "num_bytes": expected_bytes,
            "source_data_offsets": [start, end],
            "source_file_offsets": [8 + SHARD_HEADER_BYTES + start, 8 + SHARD_HEADER_BYTES + end],
        })
    tensors.sort(key=lambda item: (item["source_data_offsets"][0], item["name"]))

    # Source tensor payloads in this snapshot are expected to be contiguous.
    # Requiring this keeps downloads restricted to exactly the selected tensors.
    cursor = 0
    for item in tensors:
        start, end = item["source_data_offsets"]
        if start != cursor:
            raise FetchError(
                f"unexpected source data gap/overlap before {item['name']}: {cursor} -> {start}"
            )
        cursor = end
    if cursor != LAYER0_TENSOR_BYTES:
        raise FetchError(f"selected tensor bytes {cursor} differ from audited {LAYER0_TENSOR_BYTES}")
    return tensors


def _safetensors_header(tensors: list[dict[str, Any]]) -> tuple[bytes, int]:
    header: dict[str, Any] = {}
    cursor = 0
    for item in tensors:
        start = cursor
        end = start + item["num_bytes"]
        item["output_data_offsets"] = [start, end]
        header[item["name"]] = {
            "dtype": item["dtype"],
            "shape": item["shape"],
            "data_offsets": [start, end],
        }
        cursor = end
    header["__metadata__"] = {
        "format": "pt",
        "source_repo": MODEL_ID,
        "source_revision": REVISION,
        "source_shard": SOURCE_SHARD,
    }
    encoded = _canonical_json(header)
    encoded += b" " * ((8 - (len(encoded) % 8)) % 8)
    return struct.pack("<Q", len(encoded)) + encoded, cursor


def _chunks(tensors: list[dict[str, Any]], *, data_start: int, chunk_size: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    job_id = 0
    # Adjacent safetensors payloads are coalesced before chunking, reducing
    # request count while avoiding every non-target source byte.
    interval_start: int | None = None
    interval_end: int | None = None
    interval_out_start: int | None = None
    interval_tensors: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []

    for item in tensors:
        start, end = item["source_data_offsets"]
        out_start, out_end = item["output_data_offsets"]
        if interval_end is None:
            interval_start, interval_end = start, end
            interval_out_start = out_start
            interval_tensors = [item]
        elif start == interval_end and out_start == interval_out_start + (interval_end - interval_start):
            interval_end = end
            interval_tensors.append(item)
        else:
            intervals.append({
                "source_start": interval_start,
                "source_end": interval_end,
                "output_start": interval_out_start,
                "tensors": interval_tensors,
            })
            interval_start, interval_end = start, end
            interval_out_start = out_start
            interval_tensors = [item]
    if interval_end is not None:
        intervals.append({
            "source_start": interval_start,
            "source_end": interval_end,
            "output_start": interval_out_start,
            "tensors": interval_tensors,
        })

    for interval in intervals:
        interval_bytes = interval["source_end"] - interval["source_start"]
        for relative in range(0, interval_bytes, chunk_size):
            length = min(chunk_size, interval_bytes - relative)
            source_data_start = interval["source_start"] + relative
            output_data_start = interval["output_start"] + relative
            source_file_start = 8 + SHARD_HEADER_BYTES + source_data_start
            output_file_start = data_start + output_data_start
            source_file_end = source_file_start + length
            output_file_end = output_file_start + length
            intersects = []
            for item in interval["tensors"]:
                tensor_start, tensor_end = item["source_data_offsets"]
                if tensor_start < source_data_start + length and tensor_end > source_data_start:
                    intersects.append(item["name"])
            jobs.append({
                "chunk_id": job_id,
                "source_file_offsets": [source_file_start, source_file_end],
                "output_file_offsets": [output_file_start, output_file_end],
                "source_data_offsets": [source_data_start, source_data_start + length],
                "output_data_offsets": [output_data_start, output_data_start + length],
                "num_bytes": length,
                "tensor_keys": intersects,
            })
            job_id += 1
    return jobs


def _padded_header(header_blob: bytes) -> tuple[bytes, int]:
    header_length = struct.unpack("<Q", header_blob[:8])[0]
    return header_blob, 8 + header_length


def _verify_resume_chunk(
    *, sidecar: Path, fd: int, partial_path: Path, job: dict[str, Any], fingerprint: str,
) -> str | None:
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if (record.get("fingerprint") != fingerprint
                or record.get("chunk_id") != job["chunk_id"]
                or record.get("source_file_offsets") != job["source_file_offsets"]
                or record.get("output_file_offsets") != job["output_file_offsets"]
                or record.get("num_bytes") != job["num_bytes"]):
            return None
        start, end = job["output_file_offsets"]
        data = _read_at(fd, job["num_bytes"], start)
        if len(data) != job["num_bytes"] or _sha256(data) != record.get("sha256"):
            return None
        return record["sha256"]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _write_chunk_sidecar(path: Path, record: dict[str, Any]) -> None:
    _atomic_write_json(path, record)


def _download_job(
    job: dict[str, Any], *, source_url: str, output_fd: int, output_path: Path,
    resume_dir: Path, fingerprint: str, range_budget: RangeBudget,
) -> dict[str, Any]:
    sidecar = resume_dir / f"chunk-{job['chunk_id']:05d}.json"
    existing_sha = _verify_resume_chunk(
        sidecar=sidecar, fd=output_fd, partial_path=output_path,
        job=job, fingerprint=fingerprint,
    )
    if existing_sha is not None:
        return {**job, "sha256": existing_sha, "resumed": True}

    start, end_exclusive = job["source_file_offsets"]
    data, etag = _download_range(
        source_url, start, end_exclusive - 1, budget=range_budget,
    )
    actual_sha = _sha256(data)
    output_start, _ = job["output_file_offsets"]
    _write_at(output_fd, data, output_start)
    os.fdatasync(output_fd)
    _write_chunk_sidecar(sidecar, {
        "fingerprint": fingerprint,
        "chunk_id": job["chunk_id"],
        "source_file_offsets": job["source_file_offsets"],
        "output_file_offsets": job["output_file_offsets"],
        "num_bytes": len(data),
        "sha256": actual_sha,
    })
    return {**job, "sha256": actual_sha, "resumed": False, "etag": etag}


def _tensor_sha256(path: Path, tensor: dict[str, Any], *, data_start: int) -> str:
    start, end = tensor["output_data_offsets"]
    return _sha256_file(path, start=data_start + start, length=end - start)


def _prepare_metadata(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_url = urllib.parse.urljoin(BASE_URL, CONFIG_NAME)
    index_url = urllib.parse.urljoin(BASE_URL, INDEX_NAME)
    config_bytes, config_provenance = _download_metadata(
        config_url, maximum_bytes=MAX_METADATA_BYTES, expected_sha256=CONFIG_SHA256,
    )
    index_bytes, index_provenance = _download_metadata(
        index_url, maximum_bytes=MAX_METADATA_BYTES, expected_sha256=INDEX_SHA256,
    )
    config = json.loads(config_bytes)
    expected_config = {
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "norm_topk_prob": True,
        "num_hidden_layers": 48,
        "decoder_sparse_step": 1,
        "hidden_act": "silu",
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        "max_position_embeddings": 40_960,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "rope_theta": 1_000_000.0,
        "mlp_only_layers": [],
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise FetchError(f"pinned config mismatch for {key}: {config.get(key)!r} != {expected!r}")
    index = json.loads(index_bytes)
    if index.get("weight_map", {}).get("model.embed_tokens.weight") != SOURCE_SHARD:
        raise FetchError("embedding tensor no longer maps to the audited source shard")
    _atomic_write_bytes(output_dir / CONFIG_NAME, config_bytes)
    _atomic_write_bytes(output_dir / INDEX_NAME, index_bytes)
    return config, index, {
        "config": config_provenance,
        "index": index_provenance,
    }


def _load_source_header(budget: RangeBudget) -> tuple[dict[str, Any], dict[str, Any]]:
    shard_url = urllib.parse.urljoin(BASE_URL, SOURCE_SHARD)
    prefix, etag = _download_range(shard_url, 0, 7, budget=budget)
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length != SHARD_HEADER_BYTES or header_length > MAX_METADATA_BYTES:
        raise FetchError(f"unexpected safetensors header length: {header_length}")
    header_blob, header_etag = _download_range(
        shard_url, 8, 7 + header_length, budget=budget,
    )
    if etag != SHARD_ETAG or header_etag != SHARD_ETAG:
        raise FetchError("source shard ETag does not match audited pinned revision")
    header_sha = _sha256(header_blob)
    if header_sha != SHARD_HEADER_SHA256:
        raise FetchError(f"source shard header SHA256 mismatch: {header_sha}")
    try:
        header = json.loads(header_blob)
    except json.JSONDecodeError as exc:
        raise FetchError("source safetensors header is not valid JSON") from exc
    return header, {
        "shard_path": SOURCE_SHARD,
        "shard_size_bytes": SHARD_SIZE_BYTES,
        "shard_etag": SHARD_ETAG,
        "header_size_bytes": header_length,
        "header_sha256": header_sha,
        "header_range_http_status": 206,
    }


def _validate_existing_complete(output_dir: Path) -> bool:
    final_path = output_dir / OUTPUT_SHARD
    manifest_path = output_dir / "manifest.json"
    pending_path = output_dir / ".manifest.pending"
    if final_path.exists() and not manifest_path.exists() and pending_path.exists():
        # Recover the narrow crash window after the atomic artifact rename but
        # before the final public manifest rename.
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        pending_artifact = pending.get("artifact", {})
        if (pending.get("status") != "passed"
                or pending.get("source", {}).get("repo") != MODEL_ID
                or pending.get("source", {}).get("revision") != REVISION
                or pending_artifact.get("file") != OUTPUT_SHARD
                or final_path.stat().st_size != pending_artifact.get("size_bytes")
                or _sha256_file(final_path) != pending_artifact.get("sha256")):
            raise FetchError("pending completion record does not verify the existing final artifact")
        os.replace(pending_path, manifest_path)
        _fsync_directory(output_dir)
    if not final_path.exists() and not manifest_path.exists():
        return False
    if not final_path.exists() or not manifest_path.exists():
        raise FetchError(
            f"found an incomplete final artifact in {output_dir}; inspect before removing or resuming"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "passed"
            or manifest.get("source", {}).get("repo") != MODEL_ID
            or manifest.get("source", {}).get("revision") != REVISION
            or manifest.get("source", {}).get("index_sha256") != INDEX_SHA256):
        raise FetchError("existing manifest is not a passed artifact for the pinned revision")
    artifact = manifest.get("artifact", {})
    if artifact.get("file") != OUTPUT_SHARD:
        raise FetchError("existing manifest names an unexpected output file")
    if final_path.stat().st_size != artifact.get("size_bytes"):
        raise FetchError("existing artifact size does not match its manifest")
    if _sha256_file(final_path) != artifact.get("sha256"):
        raise FetchError("existing artifact SHA256 does not match its manifest")
    print(f"Already present and verified: {final_path}")
    return True


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        output_dir.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise FetchError("model weights must be stored outside the Git repository")
    if args.workers < 1 or args.workers > MAX_WORKERS:
        raise FetchError(f"workers must be between 1 and {MAX_WORKERS}")
    if args.chunk_bytes not in ALLOWED_CHUNK_BYTES:
        raise FetchError("chunk size must be 16 or 32 MiB")

    output_dir.mkdir(parents=True, exist_ok=True)
    if _validate_existing_complete(output_dir):
        return output_dir / OUTPUT_SHARD

    config, index, small_metadata = _prepare_metadata(output_dir)
    range_budget = RangeBudget(
        args.max_total_bytes
        - small_metadata["config"]["bytes"]
        - small_metadata["index"]["bytes"]
    )
    source_header, header_provenance = _load_source_header(range_budget)
    tensors = _required_shard(index, source_header)
    if sum(item["num_bytes"] for item in tensors) != LAYER0_TENSOR_BYTES:
        raise FetchError("layer0 tensor budget differs from the audited pinned revision")

    output_header, payload_bytes = _safetensors_header(tensors)
    output_header_bytes, output_data_start = _padded_header(output_header)
    output_size = output_data_start + payload_bytes
    estimated_total = (
        small_metadata["config"]["bytes"] + small_metadata["index"]["bytes"]
        + 8 + header_provenance["header_size_bytes"] + payload_bytes
    )
    if estimated_total > args.max_total_bytes:
        raise FetchError(
            f"selected layer0 artifact requires {estimated_total} bytes, over the "
            f"{args.max_total_bytes} byte budget"
        )

    final_path = output_dir / OUTPUT_SHARD
    partial_path = output_dir / f".{OUTPUT_SHARD}.partial"
    manifest_path = output_dir / "manifest.json"
    resume_dir = output_dir / ".layer0-resume"
    resume_dir.mkdir(parents=True, exist_ok=True)
    header_sha = _sha256(output_header_bytes)
    fingerprint_payload = {
        "model": MODEL_ID,
        "revision": REVISION,
        "index_sha256": INDEX_SHA256,
        "source_header_sha256": SHARD_HEADER_SHA256,
        "output_header_sha256": header_sha,
        "chunk_bytes": args.chunk_bytes,
    }
    fingerprint = _sha256(_canonical_json(fingerprint_payload))
    fingerprint_path = resume_dir / "fingerprint.txt"
    if fingerprint_path.exists():
        old_fingerprint = fingerprint_path.read_text(encoding="ascii").strip()
        if old_fingerprint != fingerprint:
            raise FetchError(
                "resume state belongs to a different source/header/chunk size; "
                "preserve it or remove the hidden resume directory explicitly"
            )
    else:
        _atomic_write_bytes(fingerprint_path, (fingerprint + "\n").encode("ascii"))

    if partial_path.exists():
        if partial_path.stat().st_size not in (0, output_size):
            raise FetchError("partial output has an unexpected size; refusing to overwrite it")
        if partial_path.stat().st_size == output_size:
            with partial_path.open("rb") as stream:
                if stream.read(len(output_header_bytes)) != output_header_bytes:
                    raise FetchError("partial output header differs from the pinned artifact")
        else:
            fd = os.open(partial_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                _write_at(fd, output_header_bytes, 0)
                os.ftruncate(fd, output_size)
                os.fdatasync(fd)
            finally:
                os.close(fd)
    else:
        fd = os.open(partial_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            _write_at(fd, output_header_bytes, 0)
            os.ftruncate(fd, output_size)
            os.fdatasync(fd)
        finally:
            os.close(fd)

    jobs = _chunks(tensors, data_start=output_data_start, chunk_size=args.chunk_bytes)
    unique_data_bytes = sum(job["num_bytes"] for job in jobs)
    if unique_data_bytes != payload_bytes:
        raise FetchError("planned HTTP ranges do not cover exactly the selected tensor bytes")

    shard_url = urllib.parse.urljoin(BASE_URL, SOURCE_SHARD)
    output_fd = os.open(partial_path, os.O_RDWR)
    chunk_results: list[dict[str, Any] | None] = [None] * len(jobs)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_jobs = {
                pool.submit(
                    _download_job, job,
                    source_url=shard_url,
                    output_fd=output_fd,
                    output_path=partial_path,
                    resume_dir=resume_dir,
                    fingerprint=fingerprint,
                    range_budget=range_budget,
                ): job["chunk_id"]
                for job in jobs
            }
            for future in as_completed(future_jobs):
                chunk_id = future_jobs[future]
                result = future.result()
                chunk_results[chunk_id] = result
                print(
                    f"range {chunk_id + 1}/{len(jobs)} "
                    f"{result['num_bytes']:,} bytes "
                    f"{'(resumed)' if result.get('resumed') else ''}",
                    flush=True,
                )
        os.fdatasync(output_fd)
    finally:
        os.close(output_fd)

    if any(result is None for result in chunk_results):
        raise FetchError("one or more tensor byte ranges were not completed")
    completed_chunks = [result for result in chunk_results if result is not None]

    # Re-read all fetched slices and each tensor from disk before marking the
    # artifact passed. This validates resumed and newly downloaded ranges alike.
    verify_fd = os.open(partial_path, os.O_RDONLY)
    try:
        for job, result in zip(jobs, completed_chunks):
            start, end = job["output_file_offsets"]
            data = _read_at(verify_fd, job["num_bytes"], start)
            if len(data) != job["num_bytes"] or _sha256(data) != result["sha256"]:
                raise FetchError(f"post-download range SHA256 mismatch for chunk {job['chunk_id']}")
    finally:
        os.close(verify_fd)

    tensor_manifest: dict[str, Any] = {}
    for item in tensors:
        tensor_manifest[item["name"]] = {
            "dtype": item["dtype"],
            "shape": item["shape"],
            "num_bytes": item["num_bytes"],
            "source_data_offsets": item["source_data_offsets"],
            "source_file_offsets": item["source_file_offsets"],
            "output_data_offsets": item["output_data_offsets"],
            "sha256": _tensor_sha256(partial_path, item, data_start=output_data_start),
        }

    with partial_path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8 or struct.unpack("<Q", prefix)[0] != len(output_header_bytes) - 8:
            raise FetchError("constructed safetensors output has an invalid header length")
        constructed_header = stream.read(len(output_header_bytes) - 8)
    if constructed_header != output_header_bytes[8:]:
        raise FetchError("constructed safetensors header failed readback")
    artifact_sha256 = _sha256_file(partial_path)

    chunk_manifest = []
    for job, result in zip(jobs, completed_chunks):
        chunk_manifest.append({
            "chunk_id": job["chunk_id"],
            "source_file_offsets": job["source_file_offsets"],
            "output_file_offsets": job["output_file_offsets"],
            "num_bytes": job["num_bytes"],
            "sha256": result["sha256"],
            "tensor_keys": job["tensor_keys"],
        })

    manifest = {
        "schema": 1,
        "status": "passed",
        "scope": "only selected BF16 tensors required for Qwen3-30B-A3B decoder layer 0; not a full checkpoint",
        "source": {
            "repo": MODEL_ID,
            "revision": REVISION,
            "config_sha256": CONFIG_SHA256,
            "index_file": INDEX_NAME,
            "index_sha256": INDEX_SHA256,
            "index_bytes": small_metadata["index"]["bytes"],
            "full_checkpoint_tensor_bytes": FULL_CHECKPOINT_TENSOR_BYTES,
            "shard_path": SOURCE_SHARD,
            "shard_size_bytes": SHARD_SIZE_BYTES,
            "shard_etag": SHARD_ETAG,
            "shard_header_size_bytes": SHARD_HEADER_BYTES,
            "shard_header_sha256": SHARD_HEADER_SHA256,
            "shard_header_sha256_scope": "safetensors header bytes only; not the source shard payload",
            "base_url": BASE_URL,
        },
        "artifact": {
            "file": OUTPUT_SHARD,
            "format": "safetensors",
            "size_bytes": output_size,
            "sha256": artifact_sha256,
            "header_sha256": header_sha,
            "tensor_payload_bytes": payload_bytes,
        },
        "download": {
            "chunk_size_bytes": args.chunk_bytes,
            "workers": args.workers,
            "max_total_bytes": args.max_total_bytes,
            "metadata_bytes": small_metadata["config"]["bytes"] + small_metadata["index"]["bytes"] + 8 + SHARD_HEADER_BYTES,
            "unique_tensor_range_bytes": unique_data_bytes,
            "estimated_total_bytes": estimated_total,
            "range_http_status": 206,
            "range_request_bytes_reserved": range_budget.reserved_bytes,
            "chunks": chunk_manifest,
        },
        "tensors": tensor_manifest,
    }

    # The public manifest exists only after every range, tensor, and container
    # check has passed. The temporary manifest makes the final rename recoverable.
    pending_manifest = output_dir / ".manifest.pending"
    _atomic_write_json(pending_manifest, manifest)
    os.replace(partial_path, final_path)
    _fsync_directory(output_dir)
    os.replace(pending_manifest, manifest_path)
    _fsync_directory(output_dir)
    print(f"Saved verified layer0 tensor subset: {final_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Tensor payload: {payload_bytes:,} bytes; selected tensors: {len(tensors)}")
    return final_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"output directory outside the repository (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--chunk-mib", type=int, choices=(16, 32), default=32,
        help="resumable HTTP range chunk size in MiB (default: 32)",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"parallel range requests, 1-{MAX_WORKERS} (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--max-total-gib", type=int, choices=(1, 2), default=2,
        help="hard cap for selected artifact plus metadata (maximum: 2 GiB)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.chunk_bytes = args.chunk_mib * 1024**2
    args.max_total_bytes = args.max_total_gib * 1024**3
    try:
        run(args)
    except (FetchError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
