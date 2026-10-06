#!/usr/bin/env bash
set -euo pipefail
runtime="${FLASH_ATTN_OPERATOR_RUNTIME:-$HOME/ai/baselines/flash-attention-2.8.3.post1}"
uv_exe="${UV_EXE:-$HOME/.local/bin/uv}"
wheel_name='flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
wheel_dir="${FLASH_ATTN_DOWNLOAD_DIR:-$HOME/ai/baselines/downloads}"
mkdir -p "$wheel_dir"
.venv/bin/python scripts/verify_flash_attn_backend.py --download-dir "$wheel_dir" --download-only
"$uv_exe" pip install --python .venv/bin/python --target "$runtime" --no-deps "$wheel_dir/$wheel_name"
"$uv_exe" pip install --python .venv/bin/python --target "$runtime" --no-deps \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple einops==0.8.2
.venv/bin/python scripts/verify_flash_attn_backend.py --download-dir "$wheel_dir" --runtime "$runtime"
