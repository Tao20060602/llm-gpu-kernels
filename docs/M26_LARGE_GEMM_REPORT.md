# M26 large GEMM diagnostic

## Result

All 120 numerical checks passed: five shapes across 24 tile/stage/layout configurations. The largest observed relative RMS error was `0.001674`; the largest `max_abs / max_abs(reference)` was `0.003301`. Both remain below the `0.008` limits.

The performance run used CUDA Graph timing only. Each arm was checked against the FP32 oracle before timing and after each of the three timing rounds; all 15 arms passed for every benchmark shape. A graph replay contained five calls, and the table reports median microseconds per call from nine samples per arm.

| Shape | Fastest M26 candidate observed | M26 (µs) | M25 stage 2 (µs) | Torch strict, reduced BF16 reduction off (µs) | Torch default, reduced BF16 reduction on (µs) |
|---|---|---:|---:|---:|---:|
| 1024³ | 64×64, stage 2, padding | 95.85 | 98.71 | 140.70 | 74.55 |
| 2048³ | 64×64, stage 2, swizzle | 662.94 | 784.77 | 509.34 | 510.33 |
| 4096³ | 64×128, stage 2, swizzle | 5350.40 | 7588.25 | 3904.92 | 3651.35 |

The fastest M26 candidate was selected after observing this sweep; it is not a frozen or held-out choice. It was about 2.9%, 15.5%, and 29.5% faster than M25 stage 2 at the three sizes, respectively. Torch's default baseline was faster than the fastest M26 candidate at all three sizes. The strict Torch baseline was faster at 2048³ and 4096³. The two Torch precision settings are separate measurements and should not be combined.

The winner changes layout across shapes, so these results do not establish a general swizzle advantage or prove that swizzling eliminates bank conflicts. No profiler was run; no utilization, occupancy, or peak-throughput claim is made.

## Reproduction record

- Seed: `426`; CUDA FP32 oracle with TF32 disabled.
- Numerical limits: relative RMS `≤ 0.008`; `max_abs / max_abs(reference) ≤ 0.008`.
- Environment: Python 3.12.14, PyTorch 2.7.1+cu128, CUDA toolkit 12.8.93, GCC 14, NVIDIA RTX 3080 Laptop GPU (SM86).
- Timing: three rounds in forward/reverse/forward arm order; three samples per arm per round; five calls per graph replay. Torch baselines explicitly set their BF16 reduction flag during warmup and capture, then restore it.
- Full candidate configurations, precision flags, input/source hashes, per-arm validation records, and all timing samples: [`results/m26-large-gemm-diagnostic.json`](../results/m26-large-gemm-diagnostic.json).
- An initial extension build failure under system GCC 15 was preserved separately; no kernel result was produced in that attempt: [`artifacts/m26-diagnostic/m26_gemm_check_20261013T061006Z.json`](../artifacts/m26-diagnostic/m26_gemm_check_20261013T061006Z.json).
