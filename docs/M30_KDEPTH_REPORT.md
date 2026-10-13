# M30 K-Depth Diagnostic

Run date: 2026-10-13. The raw run is preserved in `artifacts/m30-diagnostic/`; a fixed-path copy is [results/m30-kdepth-diagnostic.json](../results/m30-kdepth-diagnostic.json).

## Numerical checks and environment

Python 3.12.14, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 3080 Laptop GPU (SM86). TF32 was disabled and the FP32 oracle was `torch.mm(A.float(), W.float().T)`. All 32 M30 shape/configuration outputs were finite and passed both 0.008 FP32 error limits. The largest relative RMS was 0.001714; the largest `max_abs / max_abs(reference)` was 0.003307. All 32 outputs were bitwise equal to the corresponding M29 group2/mode1 output at the same stage. The 16 M29 reference checks passed.

The four M30 configurations used CTA64×128 with group2 and B-only prefetch. Their recorded dynamic shared-memory sizes were: BK32/S2 24,576 bytes; BK32/S3 36,864 bytes; BK64/S2 49,152 bytes; BK64/S3 73,728 bytes. The diagnostic records `ceil(K/BK)` for each numerical and benchmark case. Compiler output reported 128/164 registers per thread for BK32/S2/S3 and 160/168 for BK64/S2/S3, with no spills.

## CUDA Graph timings

Each benchmark shape used eight arms: four M30 configurations, M29 group2/mode1 at stages 2 and 3, and Torch strict/default BF16 reduction. Each graph replay contained five GEMMs. Timing used three rounds in forward/reverse/forward arm order and three samples per arm per round. Every arm passed the FP32 check before and after timing. Values are medians of the nine raw samples, in microseconds per GEMM.

| Shape | M30 BK32/S2 | M30 BK32/S3 | M30 BK64/S2 | M30 BK64/S3 | M29 S2 | M29 S3 | Torch strict | Torch default |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024³ | 98.69 | 108.54 | 99.94 | 136.60 | 98.51 | 108.34 | 140.90 | 74.34 |
| 2048³ | 627.10 | 693.86 | 631.81 | 963.54 | 641.02 | 688.74 | 522.24 | 534.32 |
| 4096³ | 5223.63 | 5770.44 | 5054.87 | 7294.75 | 5143.14 | 5769.61 | 3808.05 | 3649.95 |

The JSON contains all raw timing samples, arm configurations, per-arm pre/post metrics, shared-memory sizes, K-iteration counts, and run environment fields.

## Interpretation

BK64/S2 is 1.72% faster than the same-batch M29/S2 at 4096³, but is slower at 1024³ and only 1.44% faster at 2048³. BK64/S3 regresses substantially on all three sizes. This is a small diagnostic gain, not a demonstrated broad or stable improvement; no default is changed.

The newly captured Torch baseline also uses BK32, with a much larger 256×128 output tile. Consequently, a small BK alone does not explain our performance gap. The next experiment varies the output tile and warp mapping to increase input reuse. See [M30 design](M30_KDEPTH_DESIGN.md) and [Torch counters](M29_BASELINE_NCU_METRICS.md).
