# Native Linux operator project

- Use this repository's `.venv`, Python 3.12, torch 2.7.1+cu128, Triton 3.3.1.
- Clear inherited CUDA_HOME, CUDA_PATH, CUDACXX and LD_LIBRARY_PATH from the shared CUDA 13.2 environment.
- GPU target is RTX 3080 Laptop, SM86. Read docs/CONTRACT.md before changing kernels.
- Keep this package independent of NanoKV, model weights and CPU KV offload.
- Preserve failed results. Separate pregrouped kernel timing from routing/whole-layer timing.
- A Python expert loop is a diagnostic baseline, not a mature grouped GEMM backend.
- Never claim production deployment, mature-backend superiority, or profiler utilization without evidence.
- User requests gpt-6-luna with max reasoning for any subagents.
