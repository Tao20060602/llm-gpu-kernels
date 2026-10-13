# M33 Cache-Policy Diagnostic

Run date: 2026-10-13. The original run is preserved in `artifacts/m33-diagnostic/`; the fixed-path copy is [results/m33-cache-policy-diagnostic.json](../results/m33-cache-policy-diagnostic.json).

## Numerical checks and environment

Python 3.12.14, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 3080 Laptop GPU (SM86). TF32 was disabled and the FP32 oracle was `torch.mm(A.float(), W.float().T)`. The eight checked shapes `(M,N,K)` were `(0,17,16)`, `(1,1,16)`, `(65,129,48)`, `(257,257,80)`, `(577,513,112)`, `(1089,257,144)`, `(65,2177,208)`, and `(256,256,256)`.

All 32 M33 shape/configuration outputs were finite and passed both 0.008 FP32 error limits. The largest relative RMS was 0.001666; the largest `max_abs / max_abs(reference)` was 0.003102. All 32 outputs were bitwise equal to the matched M32 group_m=16 output. All eight M32 reference checks passed.

All configurations use M32 group_m=16, CTA `64×128×32`, group-N 2, B-only prefetch, and two stages. The cache modes are CA, CG, CA with an L2 128-byte modifier, and CG with an L2 128-byte modifier. SASS inspection confirmed the requested modifiers for all four configurations. Compiler output reported 128 registers per thread, 24,576 bytes of dynamic shared memory, and zero stack/spill for each configuration. Each CTA processes `ceil(K/32)` K tiles.

This experiment uses group_m=16 based on prior M32 observations. It is an exploratory follow-up, not a holdout or frozen-policy evaluation.

## CUDA Graph timings

Each benchmark shape used eight arms: four M33 cache modes, M32 group_m=16, M29 group2/mode1/stage2, and Torch strict/default BF16 reduction. Each graph replay contained five GEMMs. Timing used three rounds in forward/reverse/forward arm order and three samples per arm per round. Every arm passed the FP32 check before and after timing. Values are medians of the nine raw samples, in microseconds per GEMM.

| Shape | M33 CA | M33 CG | M33 CA + L2 128B | M33 CG + L2 128B | M32 G16 | M29 S2 | Torch strict | Torch default |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024³ | 98.68 | 96.82 | 98.51 | 96.46 | 98.89 | 98.51 | 140.90 | 74.55 |
| 2048³ | 626.87 | 619.72 | 631.96 | 617.68 | 635.70 | 640.82 | 553.16 | 553.78 |
| 4096³ | 4811.98 | 4702.21 | 4784.54 | 4663.91 | 4815.67 | 4987.06 | 3818.09 | 3639.08 |

The JSON contains all raw timing samples, arm configurations, tile coverage, K-iteration counts, pre/post numerical metrics, and run environment fields.

## Interpretation

At 4096³, CG+L2 128B is 3.08% faster than the same-source CA control, 3.15% faster than M32/G16, and 6.48% faster than M29/S2. It achieves about 78.0% of default Torch throughput and 81.9% of strict Torch throughput. The combined policy is also fastest among the four modes at 1024³ and 2048³ in this batch. It is an observed candidate, not a validated universal selection rule.

The isolated CG change improves 4096³ more than the isolated CA+L2 hint change; adding the L2 hint on top of CG gives a further small difference. All four compiled modes have the same reported register/shared/spill resources, and their load qualifiers differ in SASS. This supports the policy comparison, but does not measure cache-hit or traffic changes. No M33 counters were collected. See [design](M33_CACHE_POLICY_DESIGN.md) and [current status](GEMM_CURRENT_STATUS.md).
