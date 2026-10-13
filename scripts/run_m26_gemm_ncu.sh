#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
NCU="/home/tmz/ai/profilers/nsight-compute-2026.3.0/opt/nvidia/nsight-compute/2026.3.0/ncu"
PYTHON="$REPO_ROOT/.venv/bin/python"
WORKER="$REPO_ROOT/scripts/profile_m26_gemm.py"
M25_BINARY="$REPO_ROOT/artifacts/m25-build/torch_extensions/llm_gpu_kernels_m25_bf16_mma_sm86_v1/llm_gpu_kernels_m25_bf16_mma_sm86_v1.so"
M26_BINARY="$REPO_ROOT/artifacts/m26-build/llm_gpu_kernels_m26_bf16_mma_tiled_sm86_v1.so"

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

[[ -x "$NCU" ]] || die "Nsight Compute is not executable: $NCU"
[[ -x "$PYTHON" ]] || die "repository Python is not executable: $PYTHON"
[[ -f "$WORKER" ]] || die "profile worker is missing: $WORKER"
[[ -s "$M25_BINARY" ]] || die "compiled M25 extension is missing or empty: $M25_BINARY"
[[ -s "$M26_BINARY" ]] || die "compiled M26 extension is missing or empty: $M26_BINARY"

ARTIFACTS="$REPO_ROOT/artifacts"
mkdir -p -- "$ARTIFACTS"
OUT_DIR="$ARTIFACTS/m26-ncu-$(date +%Y%m%dT%H%M%S)"
mkdir -- "$OUT_DIR" || die "cannot create a fresh output directory: $OUT_DIR"

printf 'Nsight Compute: %s\n' "$NCU"
printf 'Repository Python: %s\n' "$PYTHON"
printf 'M25 extension: %s\n' "$M25_BINARY"
printf 'M26 extension: %s\n' "$M26_BINARY"
printf 'Output directory: %s\n' "$OUT_DIR"
printf 'Nsight Compute may prompt for administrator credentials.\n'
sudo -v

report_path() {
    local stem="$1"
    if [[ -s "$stem.ncu-repz" ]]; then
        printf '%s\n' "$stem.ncu-repz"
        return 0
    fi
    if [[ -s "$stem.ncu-rep" ]]; then
        printf '%s\n' "$stem.ncu-rep"
        return 0
    fi
    printf 'no non-empty NCU report found for stem: %s\n' "$stem" >&2
    return 1
}

profile_one() {
    local arm="$1"
    local kernel_regex="$2"
    local stem="$OUT_DIR/$arm"
    local log="$stem.log"
    local report
    local worker_json="$stem.worker.json"
    local status
    local -a ncu_args=(
        --profile-from-start on
        --replay-mode kernel
        --target-processes application-only
        --kernel-name-base function
        --kernel-name "$kernel_regex"
        --launch-skip 3
        --launch-count 1
        --section MemoryWorkloadAnalysis
        --section ComputeWorkloadAnalysis
        --section LaunchStats
        --section Occupancy
        --section SchedulerStats
        --section SourceCounters
        --export "$stem"
        --force-overwrite
    )

    printf 'Profiling %s...\n' "$arm"
    if sudo /usr/bin/env \
        -u CUDA_HOME -u CUDA_PATH -u CUDACXX -u LD_LIBRARY_PATH \
        "PYTHONPATH=$REPO_ROOT/src" \
        PYTHONDONTWRITEBYTECODE=1 \
        PATH=/usr/bin:/bin \
        LANG=C.UTF-8 \
        "$NCU" "${ncu_args[@]}" \
        "$PYTHON" "$WORKER" --arm "$arm" --output "$worker_json" \
        >"$log" 2>&1; then
        :
    else
        status=$?
        printf 'Nsight Compute failed for %s (exit %s); recent log follows:\n' "$arm" "$status" >&2
        tail -n 80 -- "$log" >&2 || true
        return "$status"
    fi

    if ! report="$(report_path "$stem")"; then
        printf 'Nsight Compute produced no report for %s; recent log follows:\n' "$arm" >&2
        tail -n 80 -- "$log" >&2 || true
        return 1
    fi
    if [[ ! -s "$worker_json" ]]; then
        printf 'Profile worker produced no JSON for %s; recent log follows:\n' "$arm" >&2
        tail -n 80 -- "$log" >&2 || true
        return 1
    fi
}

profile_one m25 'regex:.*bf16_mma.*'
profile_one m26_padding 'regex:.*tiled_mma.*'
profile_one m26_swizzle 'regex:.*tiled_mma.*'

RESOLVED_OUT_DIR="$(realpath -- "$OUT_DIR")"
case "$RESOLVED_OUT_DIR" in
    "$REPO_ROOT"/artifacts/m26-ncu-*) ;;
    *) die "refusing to change ownership outside the new M26 result directory: $RESOLVED_OUT_DIR" ;;
esac
[[ -d "$RESOLVED_OUT_DIR" ]] || die "result directory disappeared: $RESOLVED_OUT_DIR"
[[ -n "$(find "$RESOLVED_OUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
    || die "refusing to change ownership of an empty result directory"
sudo chown -R "$(id -u):$(id -g)" -- "$RESOLVED_OUT_DIR"

import_report() {
    local arm="$1"
    local kernel_marker="$2"
    local stem="$OUT_DIR/$arm"
    local report
    local csv="$stem.csv"
    local log="$stem.import.log"
    local status

    if ! report="$(report_path "$stem")"; then
        printf 'Nsight Compute report is missing for %s\n' "$arm" >&2
        return 1
    fi

    if /usr/bin/env \
        -u CUDA_HOME -u CUDA_PATH -u CUDACXX -u LD_LIBRARY_PATH \
        PATH=/usr/bin:/bin \
        LANG=C.UTF-8 \
        "$NCU" --import "$report" --csv --page raw >"$csv" 2>"$log"; then
        :
    else
        status=$?
        printf 'Nsight Compute report import failed for %s (exit %s); recent log follows:\n' "$arm" "$status" >&2
        tail -n 80 -- "$log" >&2 || true
        return "$status"
    fi

    if [[ ! -s "$csv" ]] || ! grep -Fqi -- "$kernel_marker" "$csv"; then
        printf 'CSV for %s is empty or has no profiled kernel name (%s); recent output follows:\n' \
            "$arm" "$kernel_marker" >&2
        tail -n 80 -- "$csv" >&2 || true
        tail -n 40 -- "$log" >&2 || true
        return 1
    fi
}

import_report m25 bf16_mma
import_report m26_padding tiled_mma
import_report m26_swizzle tiled_mma

printf 'Completed: %s\n' "$OUT_DIR"
