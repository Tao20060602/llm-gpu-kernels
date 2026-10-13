# 大矩阵 GEMM 当前状态

更新：2026-10-13。硬件为 RTX 3080 Laptop / SM86，CUDA12.8/GCC14，BF16 输入、FP32 累加、BF16 输出，计算 `A[M,K] @ W[N,K].T`。这一主线是独立 dense GEMM 实验；MoE、量化和 Attention 的成果见 README 各自入口。

## 现在得到了什么

M33 将 M32 的分组 CTA 遍历与加载策略对照结合。该批 4096³ Graph 中位数为：

| 实现 | µs/GEMM |
|---|---:|
| M29：原有 CTA64×128/BK32/S2，G2 B 寄存器预取 | 4987.06 |
| M32：同计算，GroupM16 遍历 | 4815.67 |
| M33：GroupM16 + CG/L2::128B | 4663.91 |
| Torch strict BF16 reduction | 3818.09 |
| Torch default BF16 reduction | 3639.08 |

M33 相对同批 M29 耗时减少 **6.48%**，相对同批 M32 减少 **3.15%**，达到默认 Torch 吞吐的约 **78.0%**、strict Torch 的约 **81.9%**。仍未追平 Torch。这是固定合成输入的单批诊断，不是生产或全形状收益。

M30—M33 共新增 128 项配置/形状数值检查，全部通过原 FP32 误差界且与各自对照逐位一致。三个 benchmark 尺寸的每个 arm 都完成计时前后数值校验。完整原始样本、误差及退化都保存在 results，不仅保留最快配置。

## 为什么这样走

| 步骤 | 设计判断 | 观察和取舍 |
|---|---|---|
| M29 寄存器预取 | 给 ldmatrix/MMA 更多交错空间 | 4096³ 仅约 0.35% 差异，未确认稳定收益 |
| PyTorch NCU | 比较强基线的真实 kernel | Torch 为 256×128/BK32/S3，occupancy 更低但更快，L2 命中率更高 |
| M30 BK64 | 减少外层循环、wait 和 barrier | S2 对 M29 仅约 1.7% 改善；S3 共享内存/寄存器成本导致明显退化 |
| M31 大输出 tile | 提高 CTA 内输入复用 | 4096³ 仅约 1.3% 改善，小尺寸回归；单独匹配 tile 未复现 Torch 优势 |
| M32 分组遍历 | 改善 CTA 之间的时间局部性，不增加 accumulator | G16 对线性控制约 3.0% 改善，对同批 M29 约 4.2% |
| M33 加载策略 | 根据 Torch SASS 的 BYPASS/LTC128B 线索做二因素对照 | 四模式资源一致、SASS qualifiers 确实不同；CG/L2::128B 观察最快 |

各行百分比来自各自同批对照，不能相加，也不能把不同批次的绝对时间直接作速度对比。缓存 hit/traffic 对 M32/M33 尚未采集；速度改善不等于已经证明某一缓存指标变化。

## 源码和结果入口

- [Torch NCU 指标与静态 SASS](M29_BASELINE_NCU_METRICS.md)：原始 CSV/source 表已入仓库，大型 NCU report 仅本机保留。
- [M30 设计](M30_KDEPTH_DESIGN.md) / [结果](M30_KDEPTH_REPORT.md)：BK32/64、S2/S3。
- [M31 设计](M31_OUTPUT_TILE_DESIGN.md) / [结果](M31_OUTPUT_TILE_REPORT.md)：BM128/256、八 warp。
- [M32 设计](M32_RASTER_DESIGN.md) / [结果](M32_RASTER_REPORT.md)：GroupM0/4/8/16。
- [M33 设计](M33_CACHE_POLICY_DESIGN.md) / [结果](M33_CACHE_POLICY_REPORT.md)：CA/CG × 有无 L2 128B 提示。

M33 CUDA 源码为 [cuda_bf16_mma_cache_policy.cu](../src/llm_gpu_kernels/cuda_bf16_mma_cache_policy.cu)，实验入口为 [gemm_mma_cache_policy.py](../src/llm_gpu_kernels/gemm_mma_cache_policy.py)。显式选择 `cache_mode=3` 可以复现该候选；默认参数仍为控制模式，不将本次观察包装为通用最优策略。

## 后续优先级

1. 保留当前最快候选，下一轮集中比较向量化输出写回或主循环指令调度，不继续只叠加更深 buffer。
2. 若要解释缓存机制，采集匹配控制与候选的计数器，再判断 L2 hit、traffic 和 stall 的变化。
3. 有足够新形状结果后再考虑 dispatch 规则；当前不依据三个尺寸声称全形状最优。

无需更改模型、精度目标或硬件即可继续这些工作。TMA/TMEM/2-SM 属于其他架构，不能在本机实测冒充已实现。
