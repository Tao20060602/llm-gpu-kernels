# Source from the repository root; opt-in build environment for this baseline.
source scripts/activate.sh
export CUDA_HOME="$HOME/ai/toolchains/cuda-12.8"
export CUTLASS_PATH="$HOME/ai/sources/cutlass-v3.8.0"
export CC="$HOME/ai/toolchains/gcc14/usr/bin/x86_64-linux-gnu-gcc-14"
export CXX="$HOME/ai/toolchains/gcc14/usr/bin/x86_64-linux-gnu-g++-14"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1
