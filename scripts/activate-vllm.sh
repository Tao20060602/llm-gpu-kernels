# Run from this repository root. The baseline overlay leaves .venv unchanged.
source scripts/activate.sh
export VLLM_OPERATOR_RUNTIME="${VLLM_OPERATOR_RUNTIME:-$HOME/ai/baselines/vllm-0.10.0}"
export PYTHONPATH="$VLLM_OPERATOR_RUNTIME:$PYTHONPATH"
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_FUSED_MOE_CHUNK_SIZE=32768
