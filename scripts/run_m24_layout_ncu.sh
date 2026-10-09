#!/usr/bin/env bash
# User-invoked M24 kernel-counter capture. The worker's timings are profile-only.
set -euo pipefail
umask 022

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"

ncu_bin=/home/tmz/ai/profilers/nsight-compute-2026.3.0/opt/nvidia/nsight-compute/2026.3.0/ncu
capture_root=/home/tmz/ai/traces/m20-qwen3-layer0-20261007-attempt3
packed="$repo_root/artifacts/m22-quant-pack-20261007-v1/gate-up-w4g128.pt"
pack_record="$repo_root/results/m22-quant-format-preparation.json"
w4_audit="$repo_root/artifacts/m22-w4-audit-20261007-v1/audit.json"
bf16_audit="$repo_root/artifacts/m22-cuda-audit-20261007-v2/audit.json"
layout_audit="$repo_root/artifacts/m24-w4-layout-audit-20261009-v1/audit.json"
w4_extension_root=/home/tmz/.cache/torch_extensions/py312_cu128
bf16_extension_root="$repo_root/artifacts/m22-cuda-audit-20261007-v2/torch-extensions"
layout_extension_root="$repo_root/artifacts/m24-w4-layout-audit-20261009-v1/torch-extensions"
vllm_runtime=/home/tmz/ai/baselines/vllm-0.10.0
python_bin="$repo_root/.venv/bin/python"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

for required in \
  "$ncu_bin" "$python_bin" "$capture_root/capture-summary.json" \
  "$packed" "$pack_record" "$w4_audit" "$bf16_audit" "$layout_audit" \
  "$vllm_runtime"; do
  [[ -e "$required" ]] || die "required audited input is missing: $required"
done
[[ -x "$ncu_bin" ]] || die "Nsight Compute is not executable: $ncu_bin"
[[ -x "$python_bin" ]] || die "repository venv Python is not executable: $python_bin"

verify_binary() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || die "audited extension binary is missing: $path"
  actual="$(sha256sum -- "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || die "audited extension SHA mismatch: $path"
}

verify_binary \
  "$w4_extension_root/llm_gpu_kernels_m22_w4_gate_up_sm86_v1/llm_gpu_kernels_m22_w4_gate_up_sm86_v1.so" \
  ba043552d76fdd139de2775863655fba43fb8087a9c10e20215591a2fc9b6f65
verify_binary \
  "$bf16_extension_root/llm_gpu_kernels_m22_cuda_gate_up_sm86_v1/llm_gpu_kernels_m22_cuda_gate_up_sm86_v1.so" \
  17eab52e5a5b45ac80817d0600b420dbc4c24366af59956de661f854020c08f9
verify_binary \
  "$layout_extension_root/llm_gpu_kernels_m24_w4_layout_sm86_v1/llm_gpu_kernels_m24_w4_layout_sm86_v1.so" \
  47611e5354d01efddcaa1d22d1a649d539810ff66439d3be8a3e9a350f2d04dc

# Match the already validated venv + vLLM 0.10.0 operator overlay. Keep HOME at
# tmz's path, but route mutable caches to this private profile directory; do not
# use env -i (which would discard the selected vLLM overlay and runtime context).
source "$repo_root/scripts/activate-vllm.sh"
python_path="$vllm_runtime:$repo_root/src"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
result_dir="$repo_root/artifacts/m24-layout-ncu-profile-only-$run_id"
[[ ! -e "$result_dir" ]] || die "refusing to reuse existing capture directory: $result_dir"
mkdir -m 700 -- "$result_dir"
mkdir -m 700 -- "$result_dir/reports" "$result_dir/csv" "$result_dir/logs" \
  "$result_dir/workers" "$result_dir/status" "$result_dir/triton-cache" \
  "$result_dir/cache" "$result_dir/torch-extensions"

# Cache inputs are copied by the unprivileged caller before NCU starts. By
# default the private Triton cache stays empty; M24 native binaries are loaded
# directly from the three audited roots and must not trigger native JIT.
triton_cache_dir="$result_dir/triton-cache"
xdg_cache_home="$result_dir/cache"
torch_extensions_dir="$result_dir/torch-extensions"
triton_cache_seed="${M24_TRITON_CACHE_SEED:-}"
if [[ -n "$triton_cache_seed" ]]; then
  [[ -d "$triton_cache_seed" ]] || die "M24_TRITON_CACHE_SEED is not a directory: $triton_cache_seed"
  cp -a -- "$triton_cache_seed/." "$triton_cache_dir/"
fi
if [[ -n "$triton_cache_seed" ]]; then
  triton_cache_seed_json="\"$triton_cache_seed\""
else
  triton_cache_seed_json=null
fi

ncu_version="$("$ncu_bin" --version | head -n 1)"
cat > "$result_dir/manifest.json" <<EOF
{
  "schema": 1,
  "status": "running",
  "experiment": "M24 exact-kernel NCU profile support",
  "classification": "profile-only; worker timing fields must never enter M24 performance summaries",
  "ncu": {
    "path": "$ncu_bin",
    "version": "$ncu_version",
    "profile_from_start": "on",
    "replay_mode": "kernel",
    "kernel_name_base": "function",
    "launch_skip": 2,
    "launch_count": 1,
    "launch_filter_semantics": "skip two matching launches, then profile one matching launch"
  },
  "native_loading": {
    "mode": "direct load of the three explicitly audited native binaries",
    "native_cpp_extension_jit_invoked": false
  },
  "triton_cache_policy": "isolated per-run cache; Triton preparation may recompile cache misses inside this private directory",
  "sections": [
    "MemoryWorkloadAnalysis",
    "ComputeWorkloadAnalysis",
    "LaunchStats",
    "Occupancy",
    "SchedulerStats",
    "SourceCounters"
  ],
  "cases": ["archive-T17", "code-T1024"],
  "kernel_filters": {
    "w4_sync": "cuda_gate_up_w4_sync",
    "w4_layout": "cuda_gate_up_w4_layout_sync"
  },
  "serial_workers": [
    "archive-T17-w4_sync",
    "archive-T17-w4_layout",
    "code-T1024-w4_sync",
    "code-T1024-w4_layout"
  ],
  "inputs": {
    "capture_root": "$capture_root",
    "packed": "$packed",
    "pack_record": "$pack_record",
    "w4_audit": "$w4_audit",
    "bf16_audit": "$bf16_audit",
    "layout_audit": "$layout_audit",
    "w4_extension_root": "$w4_extension_root",
    "bf16_extension_root": "$bf16_extension_root",
    "layout_extension_root": "$layout_extension_root",
    "vllm_runtime_overlay": "$vllm_runtime",
    "python": "$python_bin",
    "home": "/home/tmz",
    "torch_extensions_dir": "$torch_extensions_dir",
    "triton_cache_dir": "$triton_cache_dir",
    "triton_cache_seed": $triton_cache_seed_json,
    "xdg_cache_home": "$xdg_cache_home",
    "cuda_build_environment": "CUDA_HOME, CUDA_PATH, CUDACXX, LD_LIBRARY_PATH unset"
  },
  "outputs": {
    "reports": "reports/*.ncu-repz",
    "csv": "csv/*-raw.csv",
    "capture_logs": "logs/*-capture.log",
    "import_logs": "logs/*-import.stderr",
    "workers": "workers/*-PROFILE-ONLY-worker.json",
    "worker_status": "status/*.status"
  },
  "warning": "NCU replay perturbs execution. The full benchmark worker still runs all arms and writes latency fields; these are diagnostic-only and are excluded from unprofiled performance evidence."
}
EOF

mark_failed() {
  local exit_status="$?"
  trap - EXIT
  if [[ "$exit_status" -ne 0 && -f "$result_dir/manifest.json" ]]; then
    "$python_bin" - "$result_dir/manifest.json" "$exit_status" <<'PY' || true
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
manifest["status"] = "failed"
manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
manifest["failure_exit_code"] = int(sys.argv[2])
temporary = manifest_path.with_suffix(".json.partial")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, manifest_path)
PY
  fi
}
trap mark_failed EXIT

printf 'Prepared profile-only artifact directory: %s\n' "$result_dir"
printf 'The following step prompts for local sudo authentication before NCU collection.\n'
sudo -v

capture() {
  local case_name="$1" arm_name="$2" kernel_name="$3"
  local tag="$case_name-$arm_name"
  local report_stem="$result_dir/reports/$tag"
  local report="$report_stem.ncu-repz"
  local csv_file="$result_dir/csv/$tag-raw.csv"
  local worker_file="$result_dir/workers/$tag-PROFILE-ONLY-worker.json"
  local capture_log="$result_dir/logs/$tag-capture.log"
  local import_log="$result_dir/logs/$tag-import.stderr"
  local status_file="$result_dir/status/$tag.status"
  local status

  printf 'Profiling %s with exact kernel filter %s\n' "$tag" "$kernel_name"
  if sudo -- /usr/bin/env \
    -u CUDA_HOME -u CUDA_PATH -u CUDACXX -u LD_LIBRARY_PATH \
    HOME=/home/tmz PATH=/usr/bin:/bin LANG=C.UTF-8 \
    PYTHONPATH="$python_path" \
    VLLM_OPERATOR_RUNTIME="$vllm_runtime" \
    VLLM_LOGGING_LEVEL=WARNING VLLM_FUSED_MOE_CHUNK_SIZE=32768 \
    TORCH_CUDA_ARCH_LIST=8.6 \
    TORCH_EXTENSIONS_DIR="$torch_extensions_dir" \
    TRITON_CACHE_DIR="$triton_cache_dir" \
    XDG_CACHE_HOME="$xdg_cache_home" \
    "$ncu_bin" \
      --profile-from-start on --replay-mode kernel --target-processes application-only \
      --kernel-name-base function --kernel-name "$kernel_name" \
      --launch-skip 2 --launch-count 1 \
      --section MemoryWorkloadAnalysis \
      --section ComputeWorkloadAnalysis \
      --section LaunchStats \
      --section Occupancy \
      --section SchedulerStats \
      --section SourceCounters \
      --cache-control none --clock-control none \
      --export "$report_stem" \
      "$python_bin" -m llm_gpu_kernels.moe_w4_layout_benchmark \
      --capture-root "$capture_root" --case "$case_name" \
      --packed "$packed" --pack-record "$pack_record" \
      --w4-audit "$w4_audit" --bf16-audit "$bf16_audit" \
      --layout-audit "$layout_audit" \
      --w4-extension-root "$w4_extension_root" \
      --bf16-extension-root "$bf16_extension_root" \
      --layout-extension-root "$layout_extension_root" \
      --output "$worker_file" \
      > "$capture_log" 2>&1; then
    printf 'ncu_capture_exit=0\nworker_json=%s\nreport=%s\n' \
      "$worker_file" "$report" > "$status_file"
  else
    status="$?"
    printf 'ncu_capture_exit=%s\nlog=%s\n' "$status" "$capture_log" > "$status_file"
    printf 'Capture failed (%s); preserving logs and partial artifacts in %s\n' \
      "$status" "$result_dir" >&2
    return "$status"
  fi

  [[ -f "$report" ]] || die "NCU completed without the expected report file: $report"
  if "$ncu_bin" --import "$report" --csv --page raw \
    > "$csv_file" 2> "$import_log"; then
    printf 'csv=%s\n' "$csv_file" >> "$status_file"
  else
    status="$?"
    printf 'report_import_exit=%s\nimport_log=%s\n' "$status" "$import_log" \
      >> "$status_file"
    printf 'Report import failed (%s); preserving report and logs in %s\n' \
      "$status" "$result_dir" >&2
    return "$status"
  fi
}

# Each invocation uses the frozen full M24 worker for exactly one trace case;
# NCU filters the selected audited W4 kernel. Do not parallelize these workers.
capture archive-T17 w4_sync cuda_gate_up_w4_sync
capture archive-T17 w4_layout cuda_gate_up_w4_layout_sync
capture code-T1024 w4_sync cuda_gate_up_w4_sync
capture code-T1024 w4_layout cuda_gate_up_w4_layout_sync

"$python_bin" - "$result_dir/manifest.json" "$result_dir" <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys


manifest_path = Path(sys.argv[1])
result_dir = Path(sys.argv[2]).resolve(strict=True)
manifest = json.loads(manifest_path.read_text())
expected = (
    "archive-T17-w4_sync",
    "archive-T17-w4_layout",
    "code-T1024-w4_sync",
    "code-T1024-w4_layout",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path):
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


records = []
for tag in expected:
    worker_path = result_dir / "workers" / f"{tag}-PROFILE-ONLY-worker.json"
    report_path = result_dir / "reports" / f"{tag}.ncu-repz"
    csv_path = result_dir / "csv" / f"{tag}-raw.csv"
    capture_log = result_dir / "logs" / f"{tag}-capture.log"
    import_log = result_dir / "logs" / f"{tag}-import.stderr"
    status_path = result_dir / "status" / f"{tag}.status"
    paths = (worker_path, report_path, csv_path, capture_log, import_log, status_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("cannot mark NCU run complete; missing outputs: " + ", ".join(missing))
    worker = json.loads(worker_path.read_text())
    if worker.get("status") != "passed":
        raise SystemExit(f"cannot mark NCU run complete; worker status is not passed: {tag}")
    if report_path.stat().st_size == 0 or csv_path.stat().st_size == 0:
        raise SystemExit(f"cannot mark NCU run complete; empty report or CSV: {tag}")
    records.append(
        {
            "tag": tag,
            "worker_status": worker["status"],
            "worker": artifact_record(worker_path),
            "report": artifact_record(report_path),
            "csv": artifact_record(csv_path),
            "capture_log": artifact_record(capture_log),
            "import_log": artifact_record(import_log),
            "status_file": artifact_record(status_path),
        }
    )

manifest["status"] = "complete"
manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
manifest["results"] = records
temporary = manifest_path.with_suffix(".json.partial")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, manifest_path)
print(f"Verified four passed workers and report/CSV outputs; manifest sealed: {manifest_path}")
PY

printf 'All four serial profile-only captures completed: %s\n' "$result_dir"
