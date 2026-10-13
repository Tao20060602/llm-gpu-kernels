# M27 Scheduled BF16 GEMM Diagnostic

This report records the diagnostic run in [`results/m27-schedule-diagnostic.json`](../results/m27-schedule-diagnostic.json).

## Run configuration

- Seed: 427
- Device: NVIDIA GeForce RTX 3080 Laptop GPU (SM 8.6)
- PyTorch: 2.7.1+cu128; CUDA runtime: 12.8
- Numerical shapes: 0x17x16, 1x1x16, 65x129x48, 129x127x64, 256x256x256.
- Numerical checks: 80/80 passed; bitwise differences from matched M26 outputs: 0.
- FP32 tolerance: relative RMS <= 0.008; max-absolute/reference-maximum <= 0.008.
- Timing: CUDA Graph, 5 GEMMs per replay, 3 alternating rounds, 3 samples per arm per round; table values are median microseconds per GEMM.
- Benchmark arm set: 16 M27 configurations, four matched M26 configurations, M25 stage 2, and two Torch reduction settings (23 arms per shape).

## Graph median times

### 1024x1024x1024

| Arm | Median (us/call) |
|---|---:|
| `m25_stage_2` | 100.15 |
| `m26_64x128_s2_padding` | 97.28 |
| `m26_64x128_s2_swizzle` | 100.97 |
| `m26_64x128_s3_padding` | 107.52 |
| `m26_64x128_s3_swizzle` | 108.95 |
| `m27_group1_s2_padding` | 97.69 |
| `m27_group1_s2_swizzle` | 100.95 |
| `m27_group1_s3_padding` | 111.00 |
| `m27_group1_s3_swizzle` | 108.95 |
| `m27_group2_s2_padding` | 97.08 |
| `m27_group2_s2_swizzle` | 98.30 |
| `m27_group2_s3_padding` | 111.00 |
| `m27_group2_s3_swizzle` | 109.16 |
| `m27_group4_s2_padding` | 101.38 |
| `m27_group4_s2_swizzle` | 101.17 |
| `m27_group4_s3_padding` | 111.41 |
| `m27_group4_s3_swizzle` | 109.34 |
| `m27_group8_s2_padding` | 97.03 |
| `m27_group8_s2_swizzle` | 100.56 |
| `m27_group8_s3_padding` | 107.32 |
| `m27_group8_s3_swizzle` | 108.34 |
| `torch_default_bf16_reduction` | 73.93 |
| `torch_strict_bf16_reduction` | 141.11 |

### 2048x2048x2048

| Arm | Median (us/call) |
|---|---:|
| `m25_stage_2` | 789.71 |
| `m26_64x128_s2_padding` | 674.61 |
| `m26_64x128_s2_swizzle` | 671.95 |
| `m26_64x128_s3_padding` | 728.68 |
| `m26_64x128_s3_swizzle` | 715.37 |
| `m27_group1_s2_padding` | 656.59 |
| `m27_group1_s2_swizzle` | 628.10 |
| `m27_group1_s3_padding` | 680.76 |
| `m27_group1_s3_swizzle` | 729.70 |
| `m27_group2_s2_padding` | 681.57 |
| `m27_group2_s2_swizzle` | 652.08 |
| `m27_group2_s3_padding` | 729.29 |
| `m27_group2_s3_swizzle` | 732.98 |
| `m27_group4_s2_padding` | 642.25 |
| `m27_group4_s2_swizzle` | 673.18 |
| `m27_group4_s3_padding` | 723.76 |
| `m27_group4_s3_swizzle` | 729.70 |
| `m27_group8_s2_padding` | 678.71 |
| `m27_group8_s2_swizzle` | 670.11 |
| `m27_group8_s3_padding` | 734.82 |
| `m27_group8_s3_swizzle` | 727.24 |
| `torch_default_bf16_reduction` | 582.25 |
| `torch_strict_bf16_reduction` | 582.66 |

### 4096x4096x4096

| Arm | Median (us/call) |
|---|---:|
| `m25_stage_2` | 7943.58 |
| `m26_64x128_s2_padding` | 5634.87 |
| `m26_64x128_s2_swizzle` | 5535.74 |
| `m26_64x128_s3_padding` | 6088.29 |
| `m26_64x128_s3_swizzle` | 6114.92 |
| `m27_group1_s2_padding` | 5635.88 |
| `m27_group1_s2_swizzle` | 5445.84 |
| `m27_group1_s3_padding` | 6114.51 |
| `m27_group1_s3_swizzle` | 5986.10 |
| `m27_group2_s2_padding` | 5668.66 |
| `m27_group2_s2_swizzle` | 5357.36 |
| `m27_group2_s3_padding` | 5895.78 |
| `m27_group2_s3_swizzle` | 6045.70 |
| `m27_group4_s2_padding` | 5375.18 |
| `m27_group4_s2_swizzle` | 6147.28 |
| `m27_group4_s3_padding` | 6121.06 |
| `m27_group4_s3_swizzle` | 5990.40 |
| `m27_group8_s2_padding` | 5633.61 |
| `m27_group8_s2_swizzle` | 5590.43 |
| `m27_group8_s3_padding` | 6190.88 |
| `m27_group8_s3_swizzle` | 5929.78 |
| `torch_default_bf16_reduction` | 3721.01 |
| `torch_strict_bf16_reduction` | 4100.10 |
