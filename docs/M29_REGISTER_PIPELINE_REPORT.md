# M29 Register-Pipeline Diagnostic

Run date: 2026-10-13. Raw measurements and per-arm validation records are in [the result JSON](../results/m29-register-pipeline-diagnostic.json); the original run is preserved under `artifacts/m29-diagnostic/`.

## Environment and numerical checks

Python 3.12.14, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 3080 Laptop GPU (SM86). TF32 was disabled; the FP32 oracle was `torch.mm(A.float(), W.float().T)`. Both numerical limits were 0.008.

All 72 M29 configurations across the six declared shapes were finite and passed the FP32 checks. The largest observed relative RMS was 0.001724, and the largest `max_abs / max_abs(reference)` was 0.003258. All 72 outputs were bitwise equal to the M27 swizzle output with matching group and stage. The 24 matched M27 reference checks also passed.

## CUDA Graph timings

Each shape used 18 arms: 12 M29 configurations, four M27 swizzle references, and Torch strict/default BF16 reduction. Each graph replay contained five GEMMs. Timing used three rounds in forward/reverse/forward arm order and three samples per arm per round. Every arm passed the FP32 check before and after timing. Values below are medians of the nine raw samples, in microseconds per GEMM.

| Shape | Fastest M29 | M29 (µs) | Fastest M27 | M27 (µs) | Torch strict (µs) | Torch default (µs) |
|---|---|---:|---|---:|---:|---:|
| 1024³ | `g2/mode0/s2` | 98.92 | `g2/s2` | 99.10 | 141.11 | 75.38 |
| 2048³ | `g2/mode1/s2` | 647.99 | `g2/s2` | 646.55 | 570.98 | 561.36 |
| 4096³ | `g2/mode1/s2` | 5138.84 | `g2/s2` | 5156.66 | 3870.70 | 3688.86 |

The JSON contains every arm configuration, raw sample, pre/post numerical metric, and run environment field.

## Decision

At 4096³, the best M29 candidate is only 0.35% faster than the matched M27 arm; at 2048³ it is 0.22% slower. At 1024³ the fastest mode is the serial control. This diagnostic does not establish a stable register-prefetch gain, so M29 does not replace the previous implementation. At 4096³ it achieves about 71.8% of the default Torch throughput, or 75.3% of strict Torch throughput.

Compilation and SASS inspection show that a source-level pipeline is not itself evidence of an improved machine schedule. The implementation, resource evidence, and next comparison are explained in [the design note](M29_REGISTER_PIPELINE_DESIGN.md). No M29 hardware-counter capture has been performed. The prepared next capture compares Torch default, Torch strict, and M27 in the same 4096³ scene; its results remain pending.
