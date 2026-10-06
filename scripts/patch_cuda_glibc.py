"""Record a local CUDA 12.8 header compatibility patch for glibc 2.43.

Only six exception specifications, no function bodies or GPU arithmetic changes.
Original vendor component headers and downloads remain intact.
Related upstream issue: https://github.com/NVIDIA/cuda-samples/issues/378
"""
import difflib
import hashlib
import json
from pathlib import Path
import re


def main():
    root = Path.home() / "ai/toolchains/cuda-12.8"
    header = root / "include/crt/math_functions.h"
    backup = header.with_suffix(".h.vendor-original")
    if not backup.exists():
        backup.write_bytes(header.read_bytes())
    original = backup.read_text()
    patched = original
    names = ("cospi", "cospif", "sinpi", "sinpif", "rsqrt", "rsqrtf")
    for name in names:
        pattern = rf"(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (?:double|float)\s+{name}\((?:double|float) x\));"
        patched, count = re.subn(pattern, r"\1 noexcept(true);", patched)
        if count != 1:
            raise RuntimeError(f"unexpected CUDA vendor declaration count for {name}: {count}")
    header.write_text(patched)
    diff = "".join(difflib.unified_diff(original.splitlines(keepends=True), patched.splitlines(keepends=True),
                                      fromfile="math_functions.h.vendor-original", tofile="math_functions.h"))
    (root / "glibc-compat.patch").write_text(diff)
    record = {"reason": "CUDA12.8 declarations vs glibc2.43 noexcept mismatch", "functions": names,
              "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
              "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
              "upstream_issue": "https://github.com/NVIDIA/cuda-samples/issues/378"}
    (root / "glibc-compat.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Recorded local header patch:", root / "glibc-compat.patch")


if __name__ == "__main__":
    main()
