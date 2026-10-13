# M32 Raster Diagnostic

Run date: 2026-10-13. The original run is preserved in `artifacts/m32-diagnostic/`; a fixed-path copy is [results/m32-raster-diagnostic.json](../results/m32-raster-diagnostic.json).

## Numerical checks and environment

Python 3.12.14, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 3080 Laptop GPU (SM86). TF32 was disabled and the FP32 oracle was `torch.mm(A.float(), W.float().T)`. The eight checked shapes `(M,N,K)` were `(0,17,16)`, `(1,1,16)`, `(65,129,48)`, `(257,257,80)`, `(577,513,112)`, `(1089,257,144)`, `(65,2177,208)`, and `(256,256,256)`.

All 32 M32 shape/configuration outputs were finite and passed both 0.008 FP32 error limits. The largest relative RMS was 0.001666; the largest `max_abs / max_abs(reference)` was 0.003102. All 32 outputs were bitwise equal to the matched M29 group2/mode1/stage2 output. All eight M29 reference checks passed.

All four M32 configurations use CTA `64×128×32`, group-N 2, B-only prefetch (mode 1), and two stages. Compiler output reported 128 registers per thread for every configuration, 24,576 bytes of dynamic shared memory, and zero stack/spill. Each CTA processes `ceil(K/32)` K tiles. The recorded M/N tile counts cover the requested two-, five-, ten-, and eighteen-M-tile cases, plus N tails and partial N-tile groups.

## CUDA Graph timings

Each benchmark shape used seven arms: four M32 group-M configurations, M29 group2/mode1/stage2, and Torch strict/default BF16 reduction. Each graph replay contained five GEMMs. Timing used three rounds in forward/reverse/forward arm order and three samples per arm per round. Every arm passed the FP32 check before and after timing. Values are medians of the nine raw samples, in microseconds per GEMM.

| Shape | M32 Gm0 | M32 Gm4 | M32 Gm8 | M32 Gm16 | M29 S2 | Torch strict | Torch default |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024³ | 98.89 | 98.71 | 99.11 | 99.12 | 98.71 | 141.31 | 74.34 |
| 2048³ | 641.82 | 637.75 | 634.68 | 641.02 | 644.92 | 564.43 | 559.48 |
| 4096³ | 5016.58 | 4969.68 | 4883.87 | 4867.48 | 5078.62 | 3827.51 | 3637.85 |

The JSON contains all raw timing samples, per-arm configurations, tile coverage, K-iteration counts, pre/post numerical metrics, and run environment fields.

## Interpretation

At 4096³, Gm16 reduces time by 2.97% relative to the same-batch linear Gm0 control, or 4.16% relative to M29/S2. At 2048³, Gm8 is the fastest observed order; at 1024³ the orders are essentially unchanged. The grouped order is therefore a modest locality candidate, not a universal improvement. It reaches about 74.7% of the same-batch default Torch throughput at 4096³.

The control differs from M29 in using a linear grid and coordinate arithmetic; report the grouped-versus-linear comparison separately rather than assigning the whole M29 difference to grouping. No M32 counters were captured, so timing alone does not establish increased L2 hit rate or lower DRAM traffic. The next comparison isolates [copy cache-policy hints](M33_CACHE_POLICY_DESIGN.md) with Gm16 fixed.
