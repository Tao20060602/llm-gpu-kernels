# M25 MMA 小规模诊断报告

## 验证与计时

在 RTX 3080 Laptop（SM86）、CUDA 12.8 / PyTorch 2.7.1+cu128 上完成。28 组核心形状与 stage 检查全部通过；对照未舍入的 PyTorch FP32 oracle，TF32 关闭，误差阈值为 0.008。最差 relative RMS 为 0.0029263，最差 max_abs/max_abs(reference) 为 0.0033676。四个基准形状的 Torch BF16 baseline 和四种 stage 输出，在 API 与 CUDA Graph 计时前后也都通过有限值及误差检查。

比较基线关闭了 `allow_bf16_reduced_precision_reduction`，最终 JSON 中记录为 `false`。更早的初步计时未关闭该设置，不与下表混用。Graph 中位数如下，单位为 μs/调用；每个 arm 为 3 轮、每轮 5 个样本，每次 Graph replay 执行 10 次调用。

| M, N, K | Torch BF16 | stage 1 | stage 2 | stage 3 | stage 4 |
|---|---:|---:|---:|---:|---:|
| 17, 768, 2048 | 10.65 | 62.26 | 32.03 | 30.82 | 30.82 |
| 193, 768, 2048 | 26.83 | 73.83 | 40.04 | 40.86 | 40.24 |
| 1024, 768, 2048 | 131.48 | 145.41 | 146.94 | 147.46 | 158.41 |
| 1024, 1024, 1024 | 139.06 | 124.50 | 98.92 | 100.04 | 102.71 |

API 事件计时、原始样本和逐项校验记录在[完整 JSON 结果](../results/m25-mma-diagnostic.json)中。

## 观察

- M=17 时，stage 3 用时约为同步 stage 1 的 0.495 倍（少约 50.5%），但仍约为 Torch BF16 baseline 的 2.89 倍。
- M=193 时，stage 2 约为同步 stage 1 的 0.542 倍（少约 45.8%），但仍约为 Torch baseline 的 1.49 倍。
- M=1024、N=768 时，stage 2/3/4 均未优于同步 stage 1，Torch baseline 仍最快。
- M=N=1024 时，stage 2 为 98.92 μs，对比 Torch 的 139.06 μs，单次固定形状诊断中减少约 28.9%。

这些是四个固定随机输入形状上的小规模诊断，不是 holdout 评测；没有锁定 GPU 时钟或运行 NCU，也不代表 MoE、模型端到端表现，不能据此宣称普遍优于 cuBLAS。M=17 的当前网格为 1×12、共 12 个 CTA；并行度受限是待验证假设，不是 profiler 定位结果。当前内核还没有 shared-memory swizzle、warp specialization 或 MoE 集成。后续可先比较小 M 的 tile 与任务并行度、大 M 的资源复用，再评估布局和 warp specialization。
