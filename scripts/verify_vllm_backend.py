"""Verify the unmodified upstream operator package against its official wheel."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile

FILENAME = "vllm-0.10.0-cp38-abi3-manylinux1_x86_64.whl"
URL = "https://files.pythonhosted.org/packages/a2/98/a9c7fd61e27fd766f3cb7022613b214f053836ba7749d12a4cd5ab4644fc/" + FILENAME
SHA256 = "8ca37559d82b43b5e8c8248d2e4a1ecb51d6d4e5d517491d656df6491ed93dab"


def verify(runtime):
    archive = Path.home() / "ai/baselines/downloads" / FILENAME
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as output:
            while block := response.read(4 * 1024**2):
                output.write(block)
        partial.rename(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("official wheel checksum mismatch; preserve archive for inspection")
    count = 0
    with zipfile.ZipFile(archive) as wheel:
        for name in wheel.namelist():
            if not name.startswith("vllm/") or name.endswith("/"):
                continue
            installed = runtime / name
            if not installed.is_file() or hashlib.sha256(wheel.read(name)).digest() != hashlib.sha256(installed.read_bytes()).digest():
                raise RuntimeError("installed upstream file differs from official wheel: " + name)
            count += 1
    print(f"Verified official wheel and {count} unmodified vLLM package files: {runtime}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=Path.home() / "ai/baselines/vllm-0.10.0")
    verify(parser.parse_args().runtime)
