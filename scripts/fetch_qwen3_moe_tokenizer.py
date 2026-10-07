#!/usr/bin/env python3
"""Fetch the pinned Qwen3-30B-A3B tokenizer assets for M20 capture.

Only ``source-api.json``, ``tokenizer_config.json``, and ``tokenizer.json`` are
retrieved. This script never accesses model weights. Existing successful
snapshots are locally re-verified and reused; other non-empty output
directories are preserved and refused.

Run from the llm-gpu-kernels repository root:

    python scripts/fetch_qwen3_moe_tokenizer.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


MODEL_ID = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
DEFAULT_OUTPUT = Path("/home/tmz/models/Qwen3-30B-A3B-layer0/tokenizer")
API_URL = (
    f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
)
RESOLVE_BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/"
TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json")
OUTPUT_FILES = frozenset((*TOKENIZER_FILES, "manifest.json", "source-api.json"))
MAX_TOTAL_BYTES = 20 * 1024 * 1024
API_TIMEOUT_SECONDS = 45
FILE_TIMEOUT_SECONDS = 90
USER_AGENT = "llm-gpu-kernels-pinned-tokenizer-fetch/1.0"


class FetchError(RuntimeError):
    """Raised when pinned tokenizer source or local artifact validation fails."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    _atomic_write_bytes(path, encoded + b"\n")


def _http_get(url: str, *, byte_limit: int, expected_bytes: int | None = None,
              timeout: int, accept: str) -> tuple[bytes, dict[str, Any]]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise FetchError(f"refusing non-HTTPS source URL: {url}")
    if byte_limit < 0:
        raise FetchError("tokenizer download would exceed the 20 MiB total byte cap")

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": accept,
            "Accept-Encoding": "identity",
        },
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise FetchError(f"HTTP {exc.code} for pinned source URL: {url}") from exc
    except urllib.error.URLError as exc:
        raise FetchError(f"network error for pinned source URL {url}: {exc}") from exc
    except TimeoutError as exc:
        raise FetchError(f"request timed out for pinned source URL: {url}") from exc

    with response:
        if response.status != 200:
            raise FetchError(f"expected HTTP 200, got {response.status}: {url}")
        final_url = urllib.parse.urlsplit(response.geturl())
        if final_url.scheme != "https":
            raise FetchError(f"source redirected away from HTTPS: {response.geturl()}")
        encoding = response.headers.get("Content-Encoding", "identity").lower()
        if encoding not in ("", "identity"):
            raise FetchError(f"compressed response is not accepted for {url}")

        content_length = response.headers.get("Content-Length")
        declared_length: int | None = None
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise FetchError(f"invalid Content-Length for {url}: {content_length!r}") from exc
            if declared_length < 0 or declared_length > byte_limit:
                raise FetchError(f"response exceeds the remaining byte budget for {url}")
            if expected_bytes is not None and declared_length != expected_bytes:
                raise FetchError(
                    f"Content-Length mismatch for {url}: {declared_length} != {expected_bytes}"
                )

        data = response.read(byte_limit + 1)
        if len(data) > byte_limit:
            raise FetchError(f"response exceeds the remaining byte budget for {url}")
        if declared_length is not None and len(data) != declared_length:
            raise FetchError(f"short HTTP response for {url}: {len(data)} != {declared_length}")
        if expected_bytes is not None and len(data) != expected_bytes:
            raise FetchError(f"payload length mismatch for {url}: {len(data)} != {expected_bytes}")
        return data, {
            "url": url,
            "final_url": response.geturl(),
            "http_status": response.status,
            "bytes": len(data),
            "etag": response.headers.get("ETag"),
        }


def _parse_api(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FetchError("Hugging Face revision API returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise FetchError("Hugging Face revision API response must be a JSON object")
    if value.get("id") != MODEL_ID or value.get("sha") != REVISION:
        raise FetchError("revision API response does not match the pinned model and commit")
    siblings = value.get("siblings")
    if not isinstance(siblings, list):
        raise FetchError("revision API response has no siblings list")
    return value


def _expected_siblings(api: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in api["siblings"]:
        if isinstance(item, dict) and isinstance(item.get("rfilename"), str):
            indexed[item["rfilename"]] = item

    result: dict[str, dict[str, Any]] = {}
    for name in TOKENIZER_FILES:
        sibling = indexed.get(name)
        if not isinstance(sibling, dict):
            raise FetchError(f"pinned revision API is missing {name}")
        if sibling.get("rfilename") != name:
            raise FetchError(f"unexpected source filename for {name}")
        size = sibling.get("size")
        blob_id = sibling.get("blobId")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise FetchError(f"invalid pinned size for {name}")
        if size > MAX_TOTAL_BYTES:
            raise FetchError(f"pinned tokenizer file exceeds the 20 MiB limit: {name}")
        if not isinstance(blob_id, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", blob_id):
            raise FetchError(f"invalid pinned Git blob ID for {name}")

        lfs = sibling.get("lfs")
        if lfs is not None:
            if not isinstance(lfs, dict):
                raise FetchError(f"invalid LFS metadata for {name}")
            lfs_sha = lfs.get("sha256")
            lfs_size = lfs.get("size")
            if not isinstance(lfs_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", lfs_sha):
                raise FetchError(f"invalid pinned LFS SHA256 for {name}")
            if lfs_size != size:
                raise FetchError(f"pinned LFS size disagrees with file size for {name}")
        result[name] = sibling
    return result


def _file_url(name: str) -> str:
    if name not in TOKENIZER_FILES:
        raise FetchError(f"refusing unexpected tokenizer filename: {name}")
    return RESOLVE_BASE_URL + urllib.parse.quote(name, safe="")


def _validate_download(name: str, data: bytes,
                       sibling: dict[str, Any]) -> dict[str, Any]:
    url = _file_url(name)
    lfs = sibling.get("lfs")
    digest = _sha256(data)
    git_digest = _git_blob_sha1(data)
    if isinstance(lfs, dict):
        if len(data) != lfs["size"] or digest != lfs["sha256"].lower():
            raise FetchError(f"downloaded LFS payload hash or size mismatch: {name}")
    elif git_digest != sibling["blobId"].lower():
        raise FetchError(f"downloaded Git blob SHA1 mismatch: {name}")

    # Preserve the official sibling object verbatim. For an LFS file, the
    # sibling blobId names the Git pointer while git_blob_sha1 below hashes the
    # downloaded payload; the capture verifier uses the LFS SHA256 for content.
    source = json.loads(json.dumps(sibling, ensure_ascii=False))
    return {
        "name": name,
        "url": url,
        "bytes": len(data),
        "sha256": digest,
        "git_blob_sha1": git_digest,
        "source": source,
    }


def _validate_existing(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    api_path = output_dir / "source-api.json"
    if (manifest_path.is_symlink() or api_path.is_symlink()
            or not manifest_path.is_file() or not api_path.is_file()):
        raise FetchError("existing tokenizer metadata must be regular files, not symlinks")
    if manifest_path.stat().st_size > 1024 * 1024:
        raise FetchError("existing tokenizer manifest exceeds the 1 MiB local-read cap")
    if api_path.stat().st_size > MAX_TOTAL_BYTES:
        raise FetchError("existing source-api.json exceeds the 20 MiB local-read cap")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        api_bytes = api_path.read_bytes()
        api = _parse_api(api_bytes)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FetchError(f"non-empty output is not a readable verified snapshot: {output_dir}") from exc

    if not isinstance(manifest, dict):
        raise FetchError("existing tokenizer manifest is not a JSON object")
    if (manifest.get("status") != "passed" or manifest.get("repo_id") != MODEL_ID
            or manifest.get("revision") != REVISION):
        raise FetchError("existing tokenizer manifest is not passed for the pinned revision")
    api_sha = _sha256(api_bytes)
    if manifest.get("metadata_sha256") != api_sha:
        raise FetchError("existing source-api.json hash disagrees with tokenizer manifest")
    siblings = _expected_siblings(api)

    entries = set(os.listdir(output_dir))
    if entries != OUTPUT_FILES:
        raise FetchError("non-empty output has unexpected or missing files; refusing reuse")
    records_raw = manifest.get("files")
    if not isinstance(records_raw, list):
        raise FetchError("existing tokenizer manifest has no files list")
    records = {
        item.get("name"): item for item in records_raw
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if set(records) != set(TOKENIZER_FILES) or len(records_raw) != len(TOKENIZER_FILES):
        raise FetchError("existing tokenizer manifest does not describe exactly the two pinned files")

    total_bytes = len(api_bytes)
    for name in TOKENIZER_FILES:
        record = records[name]
        sibling = siblings[name]
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise FetchError(f"existing tokenizer file is missing: {name}")
        size = path.stat().st_size
        if record.get("url") != _file_url(name) or record.get("source") != sibling:
            raise FetchError(f"existing tokenizer source metadata disagrees with pinned API: {name}")
        if record.get("bytes") != size or size != sibling["size"]:
            raise FetchError(f"existing tokenizer size mismatch: {name}")
        if total_bytes + size > MAX_TOTAL_BYTES:
            raise FetchError("existing tokenizer snapshot exceeds the 20 MiB total size limit")
        data = path.read_bytes()
        verified = _validate_download(name, data, sibling)
        if record.get("sha256") != verified["sha256"]:
            raise FetchError(f"existing tokenizer SHA256 mismatch: {name}")
        recorded_git_sha1 = record.get("git_blob_sha1")
        if isinstance(sibling.get("lfs"), dict):
            # The LFS payload has a valid computed Git-object digest, while
            # blobId names the small pointer stored in the Git tree. The
            # capture verifier intentionally relies on the LFS SHA256 here.
            allowed_git_sha1 = {verified["git_blob_sha1"], sibling["blobId"]}
            if recorded_git_sha1 not in allowed_git_sha1:
                raise FetchError(f"existing tokenizer Git blob digest mismatch: {name}")
        elif recorded_git_sha1 != verified["git_blob_sha1"]:
            raise FetchError(f"existing tokenizer Git blob digest mismatch: {name}")
        total_bytes += size
    if total_bytes > MAX_TOTAL_BYTES:
        raise FetchError("existing tokenizer snapshot exceeds the 20 MiB total size limit")
    return manifest


def _failed_manifest(metadata_sha256: str | None, files: list[dict[str, Any]],
                     error: BaseException, bytes_received: int) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "status": "failed",
        "repo_id": MODEL_ID,
        "revision": REVISION,
        "files": files,
        "failed_at_utc": _utc_now(),
        "downloaded_bytes": bytes_received,
        "error": f"{type(error).__name__}: {error}",
    }
    if metadata_sha256 is not None:
        manifest["metadata_sha256"] = metadata_sha256
    return manifest


def run(output_dir: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = output_dir.expanduser().resolve()
    if output_dir == repo_root or repo_root in output_dir.parents:
        raise FetchError("tokenizer output must be outside the Git repository")
    if output_dir.exists() and not output_dir.is_dir():
        raise FetchError(f"output path is not a directory: {output_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        existing = _validate_existing(output_dir)
        print(f"Reusing locally verified pinned tokenizer snapshot: {output_dir}")
        print(f"Revision: {REVISION}; manifest status: {existing['status']}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FetchError(f"output directory changed during setup; refusing overwrite: {output_dir}")

    bytes_received = 0
    metadata_sha256: str | None = None
    records: list[dict[str, Any]] = []
    try:
        api_bytes, _ = _http_get(
            API_URL,
            byte_limit=MAX_TOTAL_BYTES,
            timeout=API_TIMEOUT_SECONDS,
            accept="application/json",
        )
        bytes_received += len(api_bytes)
        api = _parse_api(api_bytes)
        siblings = _expected_siblings(api)
        metadata_sha256 = _sha256(api_bytes)
        _atomic_write_bytes(output_dir / "source-api.json", api_bytes)

        for name in TOKENIZER_FILES:
            sibling = siblings[name]
            expected_size = sibling["size"]
            remaining = MAX_TOTAL_BYTES - bytes_received
            if expected_size > remaining:
                raise FetchError("pinned tokenizer assets exceed the 20 MiB total byte cap")
            data, _ = _http_get(
                _file_url(name),
                byte_limit=remaining,
                expected_bytes=expected_size,
                timeout=FILE_TIMEOUT_SECONDS,
                accept="application/octet-stream, application/json;q=0.9, */*;q=0.8",
            )
            bytes_received += len(data)
            record = _validate_download(name, data, sibling)
            _atomic_write_bytes(output_dir / name, data)
            records.append(record)

        manifest = {
            "status": "passed",
            "repo_id": MODEL_ID,
            "revision": REVISION,
            "metadata_sha256": metadata_sha256,
            "files": records,
        }
        _atomic_write_json(output_dir / "manifest.json", manifest)
        # Re-open and verify the exact on-disk layout before declaring success.
        verified = _validate_existing(output_dir)
        print(f"Saved pinned tokenizer snapshot: {output_dir}")
        print(f"Revision: {REVISION}; downloaded bytes: {bytes_received:,}; status: {verified['status']}")
        for record in records:
            print(f"{record['name']}: {record['bytes']:,} bytes, SHA256 {record['sha256']}")
        return output_dir
    except BaseException as exc:
        failure = _failed_manifest(metadata_sha256, records, exc, bytes_received)
        try:
            _atomic_write_json(output_dir / "manifest.json", failure)
        except BaseException as save_exc:
            print(f"Could not save failed tokenizer manifest: {save_exc}", file=sys.stderr)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"output directory (default: {DEFAULT_OUTPUT})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run(args.output)
    except (FetchError, OSError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("ERROR: interrupted; inspect the failed manifest before retrying", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
