"""Verify the pinned official FA2 wheel and unmodified operator overlay."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile

NAME = "flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
URL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/" + NAME.replace("+", "%2B")
SHA256 = "f87164bb919f5597cb94f7485196be6ac7842b71d7bf8e9f8e9133b9045ab5ea"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        while data := source.read(1024 * 1024):
            result.update(data)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path.home() / "ai/baselines/flash-attention-2.8.3.post1")
    parser.add_argument("--download-dir", type=Path, default=Path.home() / "ai/baselines/downloads")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    args.download_dir.mkdir(parents=True, exist_ok=True)
    wheel = args.download_dir / NAME
    if not wheel.exists():
        partial = wheel.with_suffix(".whl.partial")
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as destination:
            while data := response.read(1024 * 1024):
                destination.write(data)
        if digest(partial) != SHA256:
            raise RuntimeError("official FA2 wheel digest mismatch; partial file preserved")
        partial.rename(wheel)
    if digest(wheel) != SHA256:
        raise RuntimeError("cached FA2 wheel digest mismatch")
    if args.download_only:
        print("Official FA2 wheel SHA256 verified", wheel)
        return
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")
                 and (name.startswith("flash_attn/") or name.startswith("flash_attn_2_cuda"))]
        if not names:
            raise RuntimeError("FA2 package missing from pinned wheel")
        for name in names:
            installed = args.runtime / name
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            if not installed.is_file() or digest(installed) != expected:
                raise RuntimeError(f"installed FA2 file does not match official wheel: {name}")
    print("Official FA2 package files verified", len(names))


if __name__ == "__main__":
    main()
