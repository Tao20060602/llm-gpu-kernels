#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
NCU="/home/tmz/ai/profilers/nsight-compute-2026.3.0/opt/nvidia/nsight-compute/2026.3.0/ncu"
PYTHON="$REPO_ROOT/.venv/bin/python"
WORKER="$REPO_ROOT/scripts/profile_m29_baseline.py"
M27_BINARY="$REPO_ROOT/artifacts/m27-build/llm_gpu_kernels_m27_bf16_mma_scheduled_sm86_v1.so"

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

[[ -x "$NCU" ]] || die "Nsight Compute is not executable: $NCU"
[[ -x "$PYTHON" ]] || die "repository Python is not executable: $PYTHON"
[[ -f "$WORKER" ]] || die "profile worker is missing: $WORKER"
[[ -s "$M27_BINARY" ]] || die "compiled M27 extension is missing or empty: $M27_BINARY"

ARTIFACTS="$REPO_ROOT/artifacts"
mkdir -p -- "$ARTIFACTS"
OUT_DIR="$ARTIFACTS/m29-baseline-ncu-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -- "$OUT_DIR" || die "cannot create a fresh output directory: $OUT_DIR"

printf 'Nsight Compute: %s\n' "$NCU"
printf 'Repository Python: %s\n' "$PYTHON"
printf 'M27 extension: %s\n' "$M27_BINARY"
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
        --nvtx-include 'profile_gemm/'
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

profile_one torch_default
profile_one torch_strict
profile_one m27

RESOLVED_OUT_DIR="$(realpath -- "$OUT_DIR")"
RESOLVED_ARTIFACTS="$(realpath -- "$ARTIFACTS")"
[[ "$(dirname -- "$RESOLVED_OUT_DIR")" == "$RESOLVED_ARTIFACTS" ]] \
    || die "refusing to change ownership outside the new M29 result directory: $RESOLVED_OUT_DIR"
[[ "$(basename -- "$RESOLVED_OUT_DIR")" == m29-baseline-ncu-* ]] \
    || die "refusing to change ownership outside the new M29 result directory: $RESOLVED_OUT_DIR"
sudo chown -R "$(id -u):$(id -g)" -- "$RESOLVED_OUT_DIR"

validate_csv() {
    local csv="$1"
    /usr/bin/env \
        -u CUDA_HOME -u CUDA_PATH -u CUDACXX -u LD_LIBRARY_PATH \
        PYTHONDONTWRITEBYTECODE=1 \
        "$PYTHON" - "$csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.reader(stream, strict=True) if row != []]
except (OSError, csv.Error) as exc:
    raise SystemExit(f"cannot parse NCU CSV {path}: {exc}")

if len(rows) < 3:
    raise SystemExit(f"NCU CSV needs header, units, and data rows: {path}")
header, units, *data_rows = rows
if len(header) < 2 or "Kernel Name" not in header:
    raise SystemExit(f"NCU CSV is missing the wide-format kernel header: {path}")
if len(units) != len(header):
    raise SystemExit(f"NCU CSV units row does not match the header width: {path}")
if any(len(row) != len(header) for row in data_rows):
    raise SystemExit(f"NCU CSV contains a row with a different width: {path}")
kernel_column = header.index("Kernel Name")
if not any(row[kernel_column].strip() for row in data_rows):
    raise SystemExit(f"NCU CSV has no kernel data rows: {path}")
print(f"Validated NCU CSV: {path} ({len(data_rows)} kernel rows)")
PY
}

import_report() {
    local arm="$1"
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

    if ! validate_csv "$csv"; then
        printf 'NCU CSV validation failed for %s; recent output follows:\n' "$arm" >&2
        tail -n 5 -- "$csv" >&2 || true
        tail -n 40 -- "$log" >&2 || true
        return 1
    fi
}

import_report torch_default
import_report torch_strict
import_report m27

printf 'Completed: %s\n' "$OUT_DIR"
