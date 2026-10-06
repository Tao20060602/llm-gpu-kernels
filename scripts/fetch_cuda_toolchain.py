"""Fetch minimum NVIDIA CUDA 12.8.1 compiler components, with vendor SHA256.

No driver installation or system mutation. Uses NVIDIA's redistribution manifest.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
DEST = Path.home() / "ai/toolchains/cuda-12.8"


def main():
    downloads = DEST / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    url = BASE + "redistrib_12.8.1.json"
    raw = urllib.request.urlopen(url, timeout=60).read()
    metadata = json.loads(raw)
    (downloads / "redistrib_12.8.1.json").write_bytes(raw)
    record = {"manifest_url": url, "manifest_sha256": hashlib.sha256(raw).hexdigest(), "components": []}
    for name in ("cuda_nvcc", "cuda_cudart", "cuda_cccl"):
        item = metadata[name]["linux-x86_64"]
        archive = downloads / Path(item["relative_path"]).name
        if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != item["sha256"]:
            temporary = archive.with_suffix(".partial")
            print(f"Downloading {name}: {int(item['size']) / 1024**2:.1f} MiB", flush=True)
            with urllib.request.urlopen(BASE + item["relative_path"], timeout=60) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError(f"vendor SHA256 mismatch: {name}")
            temporary.replace(archive)
        package = DEST / "components" / name
        package.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as source:
            source.extractall(package, filter="data")
        entries = list(package.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise RuntimeError("unexpected component archive layout")
        root = entries[0]
        for child in root.iterdir():
            if child.is_dir():
                shutil.copytree(child, DEST / child.name, dirs_exist_ok=True, symlinks=True)
        record["components"].append({"name": name, "version": metadata[name]["version"], **item})
    (DEST / "source-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    if not (DEST / "lib64").exists():
        (DEST / "lib64").symlink_to("lib", target_is_directory=True)
    print(f"Prepared {DEST}", flush=True)


if __name__ == "__main__":
    main()
