# Source from the operator repository root; preserve its torch/Triton runtime.
source scripts/activate.sh
export FLASH_ATTN_OPERATOR_RUNTIME="${FLASH_ATTN_OPERATOR_RUNTIME:-$HOME/ai/baselines/flash-attention-2.8.3.post1}"
export PYTHONPATH="$FLASH_ATTN_OPERATOR_RUNTIME:$PYTHONPATH"
