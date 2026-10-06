"""Download/extract Ubuntu's GCC14 packages to user space; never install them."""
from pathlib import Path
import hashlib
import json
import subprocess


def main():
    root = Path.home() / "ai/toolchains/gcc14"
    packages = root / "packages"
    packages.mkdir(parents=True, exist_ok=True)
    version = "14.3.0-14ubuntu1"
    names = ("gcc-14-x86-64-linux-gnu", "g++-14-x86-64-linux-gnu", "cpp-14-x86-64-linux-gnu",
             "libgcc-14-dev", "libstdc++-14-dev", "gcc-14-base")
    subprocess.run(["apt-get", "download", *[name + "=" + version for name in names]], cwd=packages, check=True)
    record = []
    for path in sorted(packages.glob("*.deb")):
        pkg = subprocess.check_output(["dpkg-deb", "-f", str(path), "Package"], text=True).strip()
        ver = subprocess.check_output(["dpkg-deb", "-f", str(path), "Version"], text=True).strip()
        info = subprocess.check_output(["apt-cache", "show", pkg + "=" + ver], text=True)
        expected = [line.split(": ", 1)[1] for line in info.splitlines() if line.startswith("SHA256: ")][0]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected or pkg not in names or ver != version:
            raise RuntimeError(f"unexpected package or APT hash mismatch: {path}")
        subprocess.run(["dpkg-deb", "-x", str(path), str(root)], check=True)
        record.append({"package": pkg, "version": ver, "sha256": actual})
    (root / "source-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Extracted {len(record)} verified packages to {root}")


if __name__ == "__main__":
    main()
