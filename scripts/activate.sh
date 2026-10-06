# Source this script from the repository root.
source .venv/bin/activate
unset CUDA_HOME CUDA_PATH CUDACXX LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST=8.6
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
