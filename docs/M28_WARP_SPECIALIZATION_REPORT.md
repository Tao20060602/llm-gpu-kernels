# M28 Warp-Specialized BF16 GEMM 诊断报告

完整原始记录见 [`results/m28-warp-specialization-diagnostic.json`](../results/m28-warp-specialization-diagnostic.json)。

## 配置与数值结果

- 设备：NVIDIA GeForce RTX 3080 Laptop GPU，SM 8.6；PyTorch 2.7.1+cu128，CUDA 12.8。
- M28 候选：producer warps 1/2 × stages 2/3；固定 4 个 consumer/compute warps、CTA 64×128×32、swizzle、fragment group 2。
- 数值检查：28/28 通过 FP32 oracle 阈值（relative RMS ≤ 0.008，max-absolute/reference-maximum ≤ 0.008）。
- 同 shape、同输入 batch 下，与匹配 stage 的 M27 G2/swizzle 输出逐 bit 比较：28/28 一致；未对不同 batch 的结果做比值比较。
- 数值形状覆盖 CTA-K32 迭代数：1, 2, 3, 4, 8。
- Graph 计时：每次 replay 5 个 GEMM，3 轮交替顺序，每臂每轮 3 个样本；下表为每 GEMM 的中位数（µs）。所有 arm 的完整样本及计时前后校验均保存在 JSON 中。

## 所有 benchmark arms

| Arm | 1024³ (µs) | 2048³ (µs) | 4096³ (µs) |
|---|---:|---:|---:|
| M28 WS P1 S2 | 103.63 | 657.61 | 5142.73 |
| M28 WS P1 S3 | 125.13 | 715.37 | 6305.37 |
| M28 WS P2 S2 | 97.89 | 639.18 | 5153.16 |
| M28 WS P2 S3 | 100.56 | 645.53 | 5333.20 |
| M27 G2 S2 swizzle | 98.48 | 633.01 | 5163.62 |
| M27 G2 S3 swizzle | 108.34 | 712.09 | 5843.76 |
| M26 64x128 S2 swizzle | 101.38 | 670.72 | 5285.05 |
| M26 64x128 S3 swizzle | 108.74 | 719.67 | 5789.49 |
| Torch default reduction | 75.57 | 581.63 | 3649.33 |
| Torch strict reduction | 141.31 | 577.95 | 3805.18 |

## 结果判读与限制

- 4096³ 上 M28 WS P1 S2 为 5142.73 µs，匹配的 M27 S2 为 5163.62 µs，约低 0.4%；当前诊断不足以确认这是可靠收益。
- 2048³ 上四个 M28 WS arm 均慢于 M27 S2（633.01 µs）。1024³ 上 P2 S2 与 M27 S2 分别为 97.89 和 98.48 µs，差异较小。
- 本轮同 producer 的 S3 在三个尺寸上均慢于 S2。额外 producer warp 增加线程/寄存器预算和 barrier 资源占用；同 stage 的 shared-memory 容量不变，加深 stage 才增加该容量。本轮未对 M28 额外采集 NCU，不能据此断言实际 occupancy 瓶颈。
- 当前不将 WS 选作推荐配置，也不声称该实验已完成优化或超越成熟 cuBLAS。
