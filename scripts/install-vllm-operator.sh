#!/usr/bin/env bash
set -euo pipefail
# Operator-only overlay. Uses existing .venv torch/CUDA; no serving dependencies.
runtime="${VLLM_OPERATOR_RUNTIME:-$HOME/ai/baselines/vllm-0.10.0}"
uv_exe="${UV_EXE:-$HOME/.local/bin/uv}"
"$uv_exe" pip install --python .venv/bin/python --target "$runtime" --no-deps \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-vllm-operator.lock.txt
.venv/bin/python scripts/verify_vllm_backend.py --runtime "$runtime"
