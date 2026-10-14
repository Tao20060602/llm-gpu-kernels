#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
NCU="/home/tmz/ai/profilers/nsight-compute-2026.3.0/opt/nvidia/nsight-compute/2026.3.0/ncu"
PYTHON="$REPO_ROOT/.venv/bin/python"
WORKER="$REPO_ROOT/scripts/profile_m36_gqa.py"
M34_BINARY="$REPO_ROOT/artifacts/m34-build/llm_gpu_kernels_m34_cuda_gqa_sm86_v1.so"
M36_BINARY="$REPO_ROOT/artifacts/m36-build-v2/llm_gpu_kernels_m36_compact_gqa_sm86_v2.so"
TOKENS=256

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

if (($#)); then
    [[ $# -eq 2 && "$1" == "--tokens" ]] \
        || die "usage: $0 [--tokens 32|96|256]"
    TOKENS="$2"
fi
case "$TOKENS" in
    32|96|256) ;;
    *) die "unsupported token count: $TOKENS (choose 32, 96, or 256)" ;;
esac

[[ -x "$NCU" ]] || die "Nsight Compute is not executable: $NCU"
[[ -x "$PYTHON" ]] || die "repository Python is not executable: $PYTHON"
[[ -f "$WORKER" ]] || die "profile worker is missing: $WORKER"
[[ -s "$M34_BINARY" ]] || die "prebuilt M34 extension is missing or empty: $M34_BINARY"
[[ -s "$M36_BINARY" ]] || die "prebuilt M36 extension is missing or empty: $M36_BINARY"

ARTIFACTS="$REPO_ROOT/artifacts"
mkdir -p -- "$ARTIFACTS"
OUT_DIR="$ARTIFACTS/m36-gqa-ncu-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -- "$OUT_DIR" || die "cannot create a fresh output directory: $OUT_DIR"

printf 'Nsight Compute: %s\n' "$NCU"
printf 'Repository Python: %s\n' "$PYTHON"
printf 'M34 extension: %s\n' "$M34_BINARY"
printf 'M36 extension: %s\n' "$M36_BINARY"
printf 'Tokens: %s\n' "$TOKENS"
printf 'Output directory: %s\n' "$OUT_DIR"
printf 'Nsight Compute capture may prompt for administrator credentials.\n'
sudo -v

report_path() {
    local stem="$1"
    if [[ -s "$stem.ncu-rep" ]]; then
        printf '%s\n' "$stem.ncu-rep"
        return 0
    fi
    if [[ -s "$stem.ncu-repz" ]]; then
        printf '%s\n' "$stem.ncu-repz"
        return 0
    fi
    printf 'no non-empty NCU report found for stem: %s\n' "$stem" >&2
    return 1
}

profile_one() {
    local arm="$1"
    local stem="$OUT_DIR/$arm"
    local log="$stem.log"
    local worker_json="$stem.worker.json"
    local status
    local -a ncu_args=(
        --nvtx
        --nvtx-include 'profile_gqa/'
        --profile-from-start on
        --replay-mode kernel
        --target-processes application-only
        --section MemoryWorkloadAnalysis
        --section ComputeWorkloadAnalysis
        --section LaunchStats
        --section Occupancy
        --section SchedulerStats
        --section SourceCounters
        --export "$stem"
        --force-overwrite
    )

    printf 'Profiling %s at T=%s...\n' "$arm" "$TOKENS"
    if sudo /usr/bin/env \
        -u CUDA_HOME -u CUDA_PATH -u CUDACXX -u LD_LIBRARY_PATH \
        "PYTHONPATH=$REPO_ROOT/src" \
        PYTHONDONTWRITEBYTECODE=1 \
        PATH=/usr/bin:/bin \
        LANG=C.UTF-8 \
        "$NCU" "${ncu_args[@]}" \
        "$PYTHON" "$WORKER" --arm "$arm" --tokens "$TOKENS" --output "$worker_json" \
        >"$log" 2>&1; then
        :
    else
        status=$?
        printf 'Nsight Compute failed for %s (exit %s); recent log follows:\n' "$arm" "$status" >&2
        tail -n 80 -- "$log" >&2 || true
        return "$status"
    fi

    if ! report_path "$stem" >/dev/null; then
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

profile_one m34_s2
profile_one m36_s2
profile_one m34_s1

RESOLVED_OUT_DIR="$(realpath -- "$OUT_DIR")"
RESOLVED_ARTIFACTS="$(realpath -- "$ARTIFACTS")"
[[ "$(dirname -- "$RESOLVED_OUT_DIR")" == "$RESOLVED_ARTIFACTS" ]] \
    || die "refusing to change ownership outside the new M36 result directory: $RESOLVED_OUT_DIR"
[[ "$(basename -- "$RESOLVED_OUT_DIR")" == m36-gqa-ncu-* ]] \
    || die "refusing to change ownership outside the new M36 result directory: $RESOLVED_OUT_DIR"
sudo chown -R "$(id -u):$(id -g)" -- "$RESOLVED_OUT_DIR"

import_report() {
    local arm="$1"
    local stem="$OUT_DIR/$arm"
    local report
    local status

    if ! report="$(report_path "$stem")"; then
        printf 'Nsight Compute report is missing for %s\n' "$arm" >&2
        return 1
    fi
    if /usr/bin/env \
        -u CUDA_HOME -u CUDA_PATH -u CUDACXX -u LD_LIBRARY_PATH \
        PATH=/usr/bin:/bin \
        LANG=C.UTF-8 \
        "$NCU" --import "$report" --csv --page raw \
        >"$stem.csv" 2>"$stem.import.log"; then
        :
    else
        status=$?
        printf 'Nsight Compute report import failed for %s (exit %s); recent log follows:\n' "$arm" "$status" >&2
        tail -n 80 -- "$stem.import.log" >&2 || true
        return "$status"
    fi
    [[ -s "$stem.csv" ]] || die "raw CSV export is empty for $arm"
}

import_report m34_s2
import_report m36_s2
import_report m34_s1

printf 'Completed: %s\n' "$OUT_DIR"
