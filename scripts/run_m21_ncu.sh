#!/usr/bin/env bash
# User-invoked, five-kernel counter capture; no driver/global setting changes.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ncu_bin=/home/tmz/ai/profilers/nsight-compute-2026.3.0/opt/nvidia/nsight-compute/2026.3.0/ncu
capture_root=/home/tmz/ai/traces/m20-qwen3-layer0-20261007-attempt3
result_dir="$repo_root/artifacts/m21-ncu-user-$(date +%Y%m%dT%H%M%S)"
mkdir -- "$result_dir"
cd -- "$repo_root"
printf 'Capturing five selected kernels in three workers. Local evidence: %s\n' "$result_dir"
# Ask for the terminal password before redirecting the capture log.
sudo -v
# A clean explicit environment avoids forwarding account/API credentials and
# inherited CUDA 13.2 configuration. Administrator rights apply to this process.
capture() {
  local case_name="$1" arm_name="$2" pattern="$3" launch_count="$4"
  local stem="$result_dir/$case_name-$arm_name"
  printf 'Collecting %s / %s\n' "$case_name" "$arm_name"
  if sudo -- env -i PATH=/usr/bin:/bin LANG=C.UTF-8 \
    PYTHONPATH="$repo_root/src" TRITON_CACHE_DIR="$result_dir/triton-cache" \
    "$ncu_bin" --profile-from-start off --kernel-name-base function \
    --kernel-name "$pattern" --launch-count "$launch_count" \
    --section SpeedOfLight --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis --section LaunchStats --section Occupancy \
    --cache-control none --clock-control none \
    --export "$stem" \
    "$repo_root/.venv/bin/python" -m llm_gpu_kernels.moe_profile \
    --capture-root "$capture_root" --case "$case_name" \
    --phase profile --arm "$arm_name" --output "$stem-worker.json" \
    > "$stem-capture.log" 2>&1; then
    printf 'Finished %s / %s\n' "$case_name" "$arm_name"
  else
    local capture_status="$?"
    printf 'Capture failed (%s). Keep this log: %s\n' "$capture_status" "$stem-capture.log" >&2
    return "$capture_status"
  fi
}
# These are instrumented/replayed counter captures, not clean latency samples.
# Disable profiler cache flushing and clock locking; do not alter driver policy.
capture archive-T17-real frozen 'regex:.*gate_up.*' 1
gemm_pattern='regex:.*(_dynamic_grouped_gate_up_swiglu|_dynamic_grouped).*'
capture archive-T1024-real frozen "$gemm_pattern" 2
capture archive-T1024-real bn128 "$gemm_pattern" 2
printf 'Capture completed. Give Codex this directory: %s\n' "$result_dir"
