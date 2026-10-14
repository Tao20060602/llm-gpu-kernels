# CUDA Attention 当前状态

更新：2026-10-14。SM86 / CUDA12.8 / GCC14；固定 BF16 Q[T,32,128]、四段 KV[L,8,128]，FP32 online softmax，P 转 BF16 后做 PV，输出 BF16。历史全可见，current 因果，head 映射 qhead//4。旧 Triton 冻结策略和 NanoKV 默认入口继续使用各自已有实现。

## 现在新增了什么

- 手写 CUDA QK/PV `mma.sync`、`ldmatrix` 和 V 的转置加载。
- 每个 warp 独立负责 16 个 query，四线程一组做行 max/sum。
- P 经共享内存与 P 寄存器直通的对照。
- 单/双 K/V 缓冲的 `cp.async` 流水对照。
- 四段直接读取，包含 token stride6144 的 current V，不额外转成连续全局 buffer。
- prepared 入口持有输入/输出 owners 保证生命周期，并使用 current stream 与 Graph 重放；caller 必须保持 shape、stride、storage 固定，run 不承诺 runtime metadata 检查。

M34/M35/M36 共80项数值检查通过；M35/M36在所测输入上与对应M34配置逐位相同。测试包含T0、空历史、query/key尾块和padded V。三批诊断都验证所有性能臂在计时前后的独立FP32误差；不代表已经覆盖任意shape/stride或训练反向。

## 优化判断过程

| 阶段 | 假设和改动 | 结果 |
|---|---|---|
| M34 | 建立CUDA数据流；比较P直通、单/双缓冲 | 正确；部分场景超过指定Triton配置，仍慢FA2。寄存器223–237，无spill |
| M35 | 取消四段完全展开，减少live-state和静态代码体积 | 寄存器降到127–150，无spill；但匹配S1/direct在三个T均退化，保留失败 |
| M36 | 回到展开版本，删除direct-P未使用的4KiB共享暂存 | 寄存器不变、输出不变；S2在T96/T256显著改善，S1基本不变 |

M36 v1 的 T256 profile-only NCU 采集已完成，覆盖 M34 S1、M34 S2、M36 S2。M34 S2→M36 S2 的 worker 声明动态 SMEM 为 53,248→49,152 B；NCU 报告的共享内存 launch 限制为 1→2 blocks，寄存器限制均为 2 blocks，`sm__warps_active.avg.pct_of_peak_sustained_active` 为 8.333455%→14.273184%。这记录了该次 profile 的容量限制与 active-warp 计数变化；单次 profile 不能证明性能因果，NCU 时长也不替代未 profile 的 Graph benchmark。详见 [NCU 报告](M36_GQA_NCU_REPORT.md)和[汇总 JSON](../results/m36-gqa-ncu-summary.json)。

## M36 v1 历史性能

以下为 M36 v1 的独立历史批次：历史长度[2048,64,512]、padded current V 的Graph p50，单位µs/完整forward。不要与 M37 同批结果混算。

| T | M34 S1/direct | M34 S2/direct | M36 v1 compact S2 | raw-layout pack+FA2 |
|---:|---:|---:|---:|---:|
| 32 | 224.77 | 189.38 | 189.70 | 164.86 |
| 96 | 320.77 | 382.46 | 321.79 | 291.07 |
| 256 | 545.02 | 594.69 | 514.05 | 467.71 |

M36 v1 S2比匹配旧S2在T96/T256分别减少15.86%/13.56%；T256比同批旧S1/direct少5.68%，仍比FA2慢约9.9%。M36 v1 没有切换生产/引擎默认，也没有把局部forward结果写成TTFT收益。M34旧FA2 adapter有额外V转换，M35/M36已改用更强的raw-layout直接打包。

## M37 v2 同批诊断

M37 移除 compact `p_mode`/DirectP 选择和共享 P 路径；完整历史 tile 的 mask 状态由 kernel 按 tile 边界推导，partial query/key tiles 与 current causal tiles 保留原 mask。compact prepare 入口只保留 `stages` 配置，不改变默认实现。16 个数值 case 与 2 个 e2e case 均通过；e2e 验证固定 buffers 的值更新、非默认 stream 和 Graph。所有 diagnostic measured finite metrics 的最大 relative RMS / normalized max 为 0.0022691684 / 0.0035540883；两项 e2e 的最大值为 0.0022706119 / 0.0033076003。

以下是独立的 M37 批次，比较 v2 与明确加载的 M36 v1 control，单位 ms/完整 forward 的 Graph p50：

| T | Stage | M37 v2 | M36 v1 control | 差异 | raw-pack+FA2 |
|---:|---:|---:|---:|---:|---:|
| 32 | S1 | 0.227072 | 0.222968 | +1.84% | 0.164096 |
| 32 | S2 | 0.187648 | 0.187648 | 0.00% | 0.164096 |
| 96 | S1 | 0.321792 | 0.318720 | +0.96% | 0.290304 |
| 96 | S2 | 0.308992 | 0.320256 | -3.52% | 0.290304 |
| 256 | S1 | 0.529664 | 0.535296 | -1.05% | 0.459008 |
| 256 | S2 | 0.497368 | 0.506624 | -1.83% | 0.459008 |

M37 S2 在 T96/T256 的改善有限；S1 在 T32/T96 回归，T256 有小幅改善。raw-pack+FA2 在这三个尺寸仍更快。M37 保持为显式 operator 实验，不切换默认，不作稳定或全域收益结论；这些同批比较不能与上面的 M36 v1 历史表混算。详见 [M37 设计](M37_COMPACT_MASK_DESIGN.md)、[报告](M37_COMPACT_MASK_REPORT.md)及[e2e S1](../results/provenance/m37-e2e/m37_compact_attention_s1.json)/[S2](../results/provenance/m37-e2e/m37_compact_attention_s2.json)记录。

## 从源码学习

1. [M34 CUDA](../src/llm_gpu_kernels/cuda_gqa_attention.cu)：先看QK、行softmax和PV如何连接，尤其 `pack_bf16` 与 `load_value_b`。
2. [M34设计](M34_CUDA_GQA_DESIGN.md) / [来源与借鉴边界](M34_REFERENCE_NOTES.md)：算法、fragment布局、外部参考与本项目四段/stride适配。
3. [M35设计](M35_ROLLED_GQA_DESIGN.md) / [报告](M35_ROLLED_GQA_REPORT.md)：为什么更少寄存器的版本仍退化。
4. [M36 CUDA](../src/llm_gpu_kernels/cuda_gqa_attention_compact.cu) / [设计](M36_COMPACT_GQA_DESIGN.md) / [报告](M36_COMPACT_GQA_REPORT.md)：M36 v1 删除unused P scratch的历史容量取舍。
5. [M37 compact CUDA](../src/llm_gpu_kernels/cuda_gqa_attention_compact.cu) / [wrapper](../src/llm_gpu_kernels/attention_cuda_compact.py) / [设计](M37_COMPACT_MASK_DESIGN.md) / [报告](M37_COMPACT_MASK_REPORT.md)：内部推导完整历史 tile mask 与移除 compact `p_mode`。

## 下一步

M37 单轮 diagnostic 和 e2e 验证已完成；保留 v2 为显式实验，不切默认。后续可继续检查 Q/score 操作数复用与调度；现阶段不需要用户选择参数。
