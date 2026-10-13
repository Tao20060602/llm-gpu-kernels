# M31 Output-Tile Diagnostic

Run date: 2026-10-13. The original run is preserved in `artifacts/m31-diagnostic/`; a fixed-path copy is [results/m31-output-tile-diagnostic.json](../results/m31-output-tile-diagnostic.json).

## Numerical checks and environment

Python 3.12.14, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 3080 Laptop GPU (SM86). TF32 was disabled and the FP32 oracle was `torch.mm(A.float(), W.float().T)`. The eight checked shapes were `(M,N,K)`: `(0,17,16)`, `(1,1,16)`, `(65,129,48)`, `(17,33,80)`, `(33,65,112)`, `(33,65,144)`, `(65,129,208)`, and `(256,256,256)`.

All 32 M31 shape/configuration outputs were finite and passed both 0.008 FP32 error limits. The largest relative RMS was 0.002493; the largest `max_abs / max_abs(reference)` was 0.003071. All 32 outputs were bitwise equal to the corresponding M29 group2/mode1 output at the same stage. The 16 M29 reference checks passed.

M31 uses CTA `BM×128×32`, eight warps, group2, and B-only prefetch. Dynamic shared memory follows `(BM + 128) × 32 × 2 × stages` bytes: BM128/S2 32,768; BM128/S3 49,152; BM256/S2 49,152; BM256/S3 73,728. Each CTA processes `ceil(K/32)` K tiles. Compiler output reported 128/162 registers per thread for BM128/S2/S3 and 220/211 for BM256/S2/S3, with no spills.

## CUDA Graph timings

Each benchmark shape used eight arms: four M31 configurations, M29 group2/mode1 at stages 2 and 3, and Torch strict/default BF16 reduction. Each graph replay contained five GEMMs. Timing used three rounds in forward/reverse/forward arm order and three samples per arm per round. Every arm passed the FP32 check before and after timing. Values are medians of the nine raw samples, in microseconds per GEMM.

| Shape | M31 BM128/S2 | M31 BM128/S3 | M31 BM256/S2 | M31 BM256/S3 | M29 S2 | M29 S3 | Torch strict | Torch default |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024³ | 121.24 | 142.11 | 138.44 | 143.36 | 99.12 | 108.93 | 141.11 | 74.55 |
| 2048³ | 652.86 | 751.21 | 726.22 | 748.13 | 629.15 | 694.47 | 519.78 | 553.98 |
| 4096³ | 5053.64 | 5771.26 | 5057.74 | 5080.06 | 5120.00 | 5827.17 | 3831.77 | 3653.02 |

The JSON contains all raw timing samples, arm configurations, per-arm pre/post metrics, shared-memory sizes, K-iteration counts, and run environment fields.

## Interpretation

BM128/S2 is only 1.30% faster than the same-batch M29/S2 at 4096³, while all four candidates regress at 1024³ and 2048³. Matching the captured Torch output-tile size alone did not reproduce its advantage: even BM256/S3 remains far slower than Torch. Larger tiles improve theoretical input reuse, but their resource and scheduling costs remain material. No default is changed.

The next bounded experiment keeps the smaller CTA and computation fixed and changes CTA traversal order, to investigate temporal input reuse separately from increasing the accumulator footprint. See [design](M31_OUTPUT_TILE_DESIGN.md) and [matched Torch counter evidence](M29_BASELINE_NCU_METRICS.md).
