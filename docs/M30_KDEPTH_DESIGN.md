# M30: increase K depth without increasing the output tile

## Question

Does processing BK64 rather than BK32 reduce main-loop overhead enough to compensate for the larger shared-memory footprint? M29's register prefetch did not establish a stable gain. This experiment addresses a different hypothesis: fewer shared-stage commits, waits, and CTA barriers for the same dense GEMM.

The output tile stays 64×128, with four compute warps, G2 B-register prefetch, BF16 inputs, FP32 accumulation, and BF16 output. The comparison has only BK32/BK64 × two/three shared stages. BK32 is a control reproducing the M29 B-prefetch schedule; the Python wrapper defaults to BK32, not an observed winner.

## Costs and expected benefits

For K4096, BK32 has 128 outer iterations; BK64 has 64. Each outer iteration still has an arrival barrier and a reuse barrier. Each tile has BK/16 K16 slices, so doubling BK does not reduce the total MMA work. For aligned K, the total logical input-copy volume is also unchanged.

The input-only arithmetic intensity of the CTA is `2*BM*BN*BK / (2*(BM+BN)*BK) = 42.67 FLOP/byte` in either case. BK64 does not improve global-memory reuse merely by increasing K depth. Its potential advantage is amortizing loop/synchronization work and changing the amount of in-flight data.

| BK | Stages | Declared shared bytes/CTA | K4096 outer iterations |
|---:|---:|---:|---:|
| 32 | 2 | 24,576 | 128 |
| 32 | 3 | 36,864 | 128 |
| 64 | 2 | 49,152 | 64 |
| 64 | 3 | 73,728 | 64 |

Larger storage can reduce resident CTAs. The unrolled K body can also change compiler scheduling and register allocation even though the accumulator count is unchanged. Three stages are included to expose the tradeoff, not because deeper buffering is assumed to be better. Hardware occupancy is not inferred from these declared byte counts alone.

CUDA12.8/GCC14 compilation reports 128/164 registers per thread for BK32 S2/S3, and 160/168 for BK64 S2/S3. All four instantiations have zero stack frame and zero spill stores/loads. These are compiler resource counts, not measured occupancy. The build log is retained locally at `artifacts/m30-build/build.log`.

## Layout, tails, and synchronization

The shared index is `(row*BK+col) XOR ((row&7)*8)`. It preserves the low three BF16 address bits, so aligned eight-element copies remain contiguous. BK32 retains the old mapping; with BK64 the XOR acts within the row. Changing row pitch changes the physical mapping, so this is a K-depth/layout experiment, not a proof that any observed difference comes exclusively from fewer barriers.

The copy loop uses BK/8 vectors per row instead of assuming four. `ldmatrix` address calculations use the matching pitch. K is still allowed to be any positive multiple of 16, not only a multiple of BK. Invalid tail vectors are zero-filled, while every valid output follows the original K16 accumulation order. BK64 may execute more zero-filled slices than BK32 on some tails; this is acceptable for correctness but may hurt small/tail workloads.

The shared ring follows the existing prime/issue/wait/consume/reuse schedule. Register B prefetch stays inside each K16 slice. BK64/S3 requires dynamic shared-memory opt-in; the function attribute is set during plan preparation, outside timing and graph replay. Plans retain the existing fixed-buffer lifetime and current-stream launch semantics.

Source: [CUDA](../src/llm_gpu_kernels/cuda_bf16_mma_kdepth.cu), [wrapper](../src/llm_gpu_kernels/gemm_mma_kdepth.py), [runner](../scripts/run_m30_gemm_check.py).

## Comparison protocol

Eight shapes × four configurations cover empty M, small matrices, M/N tails, and K16/K48/K80/K112/K144/K208/K256. Each output must pass the existing FP32 oracle thresholds and be bitwise equal to the same-stage M29 G2/B-prefetch output. Stop before performance timing if that equivalence fails.

Benchmark 1024³, 2048³, and 4096³ with eight arms per size: four M30 candidates, two matched M29 arms, Torch default, and Torch strict. Keep the existing five-GEMM CUDA Graph and three forward/reverse/forward rounds with three samples per arm. Validate all arms before and after timing; compile and prepare outside timing. Results are diagnostic, not a frozen dispatch policy.
