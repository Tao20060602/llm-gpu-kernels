# M32: CTA traversal for temporal input reuse

M31 larger output tiles gave only a small 4096³ difference and regressed on smaller sizes. The next hypothesis is reuse between CTAs, not more accumulator state. Keep CTA64×128/BK32, four warps, two shared stages, and G2 B-register prefetch. Change only the mapping from a linear block ID to the output tile coordinates.

The control runs all M tiles for one N tile before advancing N, matching the original two-dimensional grid's logical traversal. Grouped orders run a bounded M group across all N tiles, then advance to the next M group. Group sizes are 4/8/16; 0 denotes the control. This can bring reuse of the same A rows across different N tiles closer in launch order, while shortening the M span over which W is reused. It changes neither the arithmetic nor the total logical copy volume. Actual CTA execution order is scheduled by the GPU, so launch adjacency is a locality hint, not a synchronization or ordering guarantee.

For group size G, let `group_size=G*N_tiles`, `first_M=(block_id/group_size)*G`, and `active_M=min(G,M_tiles-first_M)`. Then `local=block_id%group_size`, `tile_M=first_M+local%active_M`, and `tile_N=local/active_M`. The last group uses its actual remaining M tiles. The grid has exactly `M_tiles*N_tiles` CTAs: no padded duplicate or out-of-range output tiles. Preparation checks the linear grid against the device and int32 limits.

For 4096³, four M64 tiles span 2 MiB of A, eight span 4 MiB, and sixteen span 8 MiB across K. These are logical working-set sizes, not guarantees about cache residency. Scheduling, simultaneous W traffic, cache policy, and other live data can all affect reuse. This is a limited locality experiment, not a claim that the kernel is exclusively DRAM-bound.

Eight numerical shapes cover M0, small inputs, non-full M groups, N tails, and K tails. All four configurations must pass the FP32 bounds and match M29 G2/mode1/S2 bitwise before timing. The 1024³/2048³/4096³ benchmark has seven arms: four traversal orders, matched M29/S2, and Torch strict/default, using the existing three-round Graph protocol with validation before/after timing.

Source: [CUDA](../src/llm_gpu_kernels/cuda_bf16_mma_raster.cu), [wrapper](../src/llm_gpu_kernels/gemm_mma_raster.py), [runner](../scripts/run_m32_gemm_check.py).
