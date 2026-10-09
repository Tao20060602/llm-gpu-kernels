"""M23 startup adapter for reusing the audited M22 native binaries.

This adapter imports the two exact, audited shared libraries directly and
primes the existing M22 loader caches before handing the original arguments
to the frozen benchmark worker. It does not invoke a compiler or change the
worker's implementation.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


ADAPTER_NAME = "moe_w4_repeat"
ADAPTER_VERSION = "1"
AUDITED_EXTENSIONS = {
    "w4": "llm_gpu_kernels_m22_w4_gate_up_sm86_v1.so",
    "bf16": "llm_gpu_kernels_m22_cuda_gate_up_sm86_v1.so",
}
REQUIRED_WORKER_PATH_OPTIONS = (
    "--w4-audit",
    "--bf16-audit",
    "--w4-extension-root",
    "--bf16-extension-root",
    "--output",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _option_values(arguments: list[str], option: str) -> list[str]:
    """Return explicit values without parsing or rewriting worker arguments."""
    values = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == option:
            if index + 1 >= len(arguments) or not arguments[index + 1]:
                raise ValueError(f"{option} requires an explicit value")
            value = arguments[index + 1]
            if value.startswith("--"):
                raise ValueError(f"{option} requires an explicit value before {value}")
            values.append(value)
            index += 2
        elif token.startswith(option + "="):
            value = token[len(option) + 1:]
            if not value:
                raise ValueError(f"{option} requires an explicit value")
            values.append(value)
            index += 1
        else:
            index += 1
    return values


def _one_explicit_value(arguments: list[str], option: str) -> str:
    values = _option_values(arguments, option)
    if not values:
        raise ValueError(f"forwarded worker argument {option} must be explicitly supplied")
    if len(values) != 1:
        raise ValueError(f"forwarded worker argument {option} must be supplied exactly once")
    return values[0]


def _load_passed_audit(path: Path, worker, label: str) -> tuple[dict, str, bytes]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} audit is not valid JSON: {resolved}") from exc
    if not isinstance(record, dict) or record.get("status") != "passed":
        raise RuntimeError(f"{label} audit must have top-level status='passed': {resolved}")

    # Reuse the frozen worker's audit schema and source-provenance checks.
    try:
        validated = worker.require_audit(resolved)
    except Exception as exc:
        raise RuntimeError(f"{label} audit rejected by the frozen worker: {exc}") from exc
    if validated != record or resolved.read_bytes() != raw:
        raise RuntimeError(f"{label} audit changed while it was being validated")

    return record, _sha256_bytes(raw), raw


def _resolve_audited_binary(audit: dict, root: Path, label: str) -> tuple[Path, str, str]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"{label} extension root is not a directory: {root}")

    binary_record = audit.get("binary")
    if not isinstance(binary_record, dict):
        raise RuntimeError(f"{label} audit has no binary record")
    filename = binary_record.get("name")
    expected_filename = AUDITED_EXTENSIONS[label]
    if filename != expected_filename or Path(filename).name != filename or not filename.endswith(".so"):
        raise RuntimeError(
            f"{label} audit binary name must be the fixed audited extension {expected_filename!r}"
        )

    before = binary_record.get("sha256_before")
    after = binary_record.get("sha256_after")
    if (not isinstance(before, str) or not SHA256_RE.fullmatch(before)
            or before != after or not isinstance(after, str) or not SHA256_RE.fullmatch(after)):
        raise RuntimeError(f"{label} audit binary SHA-256 is missing or changed")

    # This is the same cache layout used by moe_w4_benchmark.load_audited:
    # <extension-root>/<extension-name-without-.so>/<extension-name>.so.
    expected = root / filename[:-3] / filename
    if not expected.is_file():
        raise RuntimeError(f"{label} audited binary is absent from the explicit extension root: {expected}")
    resolved = expected.resolve(strict=True)
    actual_digest = _sha256_file(resolved)
    if actual_digest != after:
        raise RuntimeError(f"{label} extension binary does not match the audit SHA-256: {resolved}")

    module_name = Path(filename).stem
    return resolved, module_name, actual_digest


def _load_native_binary(path: Path, module_name: str, label: str):
    if module_name in sys.modules:
        raise RuntimeError(f"{label} native module name is already loaded: {module_name}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create a native-module import spec for {label}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise

    loaded_path = Path(module.__file__).resolve(strict=True)
    if loaded_path != path:
        raise RuntimeError(f"{label} native module loaded from an unexpected path: {loaded_path}")
    return module


def main(argv: list[str] | None = None) -> int:
    original_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--startup-receipt", type=Path, required=True)
    adapter_args, worker_args = parser.parse_known_args(original_argv)

    try:
        receipt_values = _option_values(original_argv, "--startup-receipt")
        if len(receipt_values) != 1:
            raise ValueError("--startup-receipt must be supplied exactly once")
        forwarded_paths = {
            option: Path(_one_explicit_value(worker_args, option))
            for option in REQUIRED_WORKER_PATH_OPTIONS
        }
    except ValueError as exc:
        parser.error(str(exc))

    receipt_path = adapter_args.startup_receipt.expanduser().resolve()
    output_path = forwarded_paths["--output"].resolve()
    if receipt_path == output_path:
        raise RuntimeError("--startup-receipt and the worker --output must be different paths")
    if (output_path.exists()
            or output_path.with_suffix(output_path.suffix + ".partial").exists()):
        raise RuntimeError("refusing overwrite worker output or partial output")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite startup receipt: {receipt_path}")

    # Keep runtime imports inside main so the adapter's only native loading
    # path is the explicit spec/module/exec sequence below.
    import torch
    from . import moe_w4_cuda
    from . import moe_cuda
    from . import moe_w4_benchmark as worker

    if moe_w4_cuda._EXTENSION is not None or moe_cuda._EXTENSION is not None:
        raise RuntimeError("refusing startup with a pre-populated M22 extension cache")

    audits = {}
    audit_hashes = {}
    audit_bytes = {}
    for label, option in (("w4", "--w4-audit"), ("bf16", "--bf16-audit")):
        audit, audit_hash, raw = _load_passed_audit(forwarded_paths[option], worker, label)
        audits[label] = audit
        audit_hashes[label] = audit_hash
        audit_bytes[label] = raw

    binaries = {}
    for label, option in (("w4", "--w4-extension-root"), ("bf16", "--bf16-extension-root")):
        path, module_name, digest = _resolve_audited_binary(
            audits[label], forwarded_paths[option], label
        )
        binaries[label] = {
            "resolved_path": path,
            "filename": AUDITED_EXTENSIONS[label],
            "module_name": module_name,
            "sha256_before_load": digest,
        }

    w4_module = _load_native_binary(
        binaries["w4"]["resolved_path"], binaries["w4"]["module_name"], "w4"
    )
    bf16_module = _load_native_binary(
        binaries["bf16"]["resolved_path"], binaries["bf16"]["module_name"], "bf16"
    )

    # Refuse any external cache mutation before installing the modules loaded
    # from the two audited files. The old loader then returns these cached
    # modules before reaching its cpp_extension.load branch.
    if moe_w4_cuda._EXTENSION is not None or moe_cuda._EXTENSION is not None:
        raise RuntimeError("M22 extension cache changed during audited native loading")
    moe_w4_cuda._EXTENSION = w4_module
    moe_cuda._EXTENSION = bf16_module

    for label, module in (("w4", w4_module), ("bf16", bf16_module)):
        binary = binaries[label]
        loaded_path = Path(module.__file__).resolve(strict=True)
        if loaded_path != binary["resolved_path"]:
            raise RuntimeError(f"{label} native module path changed after import")
        digest_after = _sha256_file(loaded_path)
        if digest_after != binary["sha256_before_load"]:
            raise RuntimeError(f"{label} native binary changed while loading")
        if digest_after != audits[label]["binary"]["sha256_after"]:
            raise RuntimeError(f"{label} loaded native binary no longer matches its audit")
        binary["sha256_after_load"] = digest_after

    for label, option in (("w4", "--w4-audit"), ("bf16", "--bf16-audit")):
        if forwarded_paths[option].resolve(strict=True).read_bytes() != audit_bytes[label]:
            raise RuntimeError(f"{label} audit changed before the startup receipt was written")

    adapter_source = Path(__file__).resolve(strict=True)
    worker_source = Path(worker.__file__).resolve(strict=True)
    receipt = {
        "schema": 1,
        "adapter": {"name": ADAPTER_NAME, "version": ADAPTER_VERSION},
        "adapter_source_sha256": _sha256_file(adapter_source),
        "audit_record_sha256": audit_hashes,
        "binaries": {
            label: {
                "resolved_path": str(binary["resolved_path"]),
                "filename": binary["filename"],
                "module_name": binary["module_name"],
                "sha256_before_load": binary["sha256_before_load"],
                "sha256_after_load": binary["sha256_after_load"],
            }
            for label, binary in binaries.items()
        },
        "worker_source_sha256": _sha256_file(worker_source),
        "torch_version": str(torch.__version__),
        "load_mode": "verified_native_binary_no_jit",
        "load_scope": "startup native binary loading only; worker Triton preparation is outside this scope",
        "native_cpp_extension_jit_invoked": False,
        "expected_output_path": str(output_path),
    }

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_receipt = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(encoded_receipt)
        stream.flush()

    # moe_w4_benchmark.run() owns the legacy argparse parser and reads
    # sys.argv itself. All original worker tokens are forwarded in order; only
    # --startup-receipt and its value were consumed by this adapter.
    program = sys.argv[0] if sys.argv else ADAPTER_NAME
    previous_argv = sys.argv
    sys.argv = [program, *worker_args]
    try:
        worker.run()
    finally:
        sys.argv = previous_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
