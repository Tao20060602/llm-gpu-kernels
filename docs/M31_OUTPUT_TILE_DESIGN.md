# M31: increase output tile and input reuse

## Evidence that changed the priority

The [matched Torch capture](M29_BASELINE_NCU_METRICS.md) reports a `256x128 ... stages_32x3` kernel with 256 threads, while M27 has a 64×128 tile and 128 threads. Torch's achieved occupancy is lower, despite being faster in the non-profiled benchmark. Its captured L2 sector hit rate is also higher. These observations motivate a larger output tile, rather than maximizing occupancy or assuming that BK32 must be the problem. They do not independently prove the cause of the performance gap.

M30 BK64/S2 gives only a small observed gain; BK64/S3 regresses. Keep BK32 for this next comparison, matching the original primitives and the captured Torch K depth. M31 compares BM128/BM256 × two/three stages, with BN128, eight compute warps, and G2 B-register prefetch. It remains an independently written prototype, not a reimplementation of cuBLAS's complete kernel.

## Mapping and resource tradeoff

Eight warps form four M groups and two N groups. A warp computes `(BM/4)×64`: 32×64 for BM128 or 64×64 for BM256. Its M16-fragment count is `MF=BM/64`, and its row origin is `(warp/2)*MF*16`. Each thread holds 64 or 128 FP32 accumulator values respectively. A/W staging distributes aligned copies across all 256 threads; the existing BK32 shared swizzle and K16 MMA order are retained.

| BM×BN | Threads | Accumulators/thread | Input-only FLOP/byte | S2 shared | S3 shared |
|---|---:|---:|---:|---:|---:|
| 64×128 (M29) | 128 | 64 | 42.67 | 24,576 B | 36,864 B |
| 128×128 | 256 | 64 | 64.00 | 32,768 B | 49,152 B |
| 256×128 | 256 | 128 | 85.33 | 49,152 B | 73,728 B |

The intensity formula is `BM*BN/(BM+BN)` for BF16 inputs, excluding output stores and cache effects. For aligned 4096³, moving BM64→BM256 reduces the logical total input-copy volume by half and reduces the grid from 2048 to 512 CTAs. This does not establish a halving of DRAM traffic: actual cache behavior must be measured separately.

Larger BM reuses each W tile across more output rows, but adds accumulator state, may reduce resident CTAs, and increases masked work on small M/tails. There is no guarantee that BM256 or stage3 wins. Dynamic shared opt-in for BM256/S3 happens during preparation, not the timed launch.

CUDA12.8/GCC14 compilation reports 128/162 registers per thread for BM128 S2/S3, and 220/211 for BM256 S2/S3. All configurations report zero stack frame and zero spills. BM256/S3 using fewer registers than S2 illustrates that resource allocation depends on the compiler's full schedule, not just a source-level buffer count. Build evidence is retained locally at `artifacts/m31-build/build.log`.

## Validation and timing

Use the same eight shapes and FP32 error bounds as M30. All four candidates must match same-stage M29 G2/B-prefetch output bitwise; fail before timing if not. For 1024³/2048³/4096³, compare four M31 arms, two M29 references, and Torch strict/default in the existing eight-arm Graph protocol, with all outputs checked before/after timing. Preserve every configuration and regression; no default dispatch policy is selected from this diagnostic.

Source: [CUDA](../src/llm_gpu_kernels/cuda_bf16_mma_output_tile.cu), [wrapper](../src/llm_gpu_kernels/gemm_mma_output_tile.py), [runner](../scripts/run_m31_gemm_check.py).
