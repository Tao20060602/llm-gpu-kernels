# CUDA Attention 当前状态

更新：2026-10-14。SM86 / CUDA12.8 / GCC14；固定 BF16 Q[T,32,128]、四段 KV[L,8,128]，FP32 online softmax，P 转 BF16 后做 PV，输出 BF16。历史全可见，current 因果，head 映射 qhead//4。旧 Triton 冻结策略和 NanoKV 默认入口继续使用各自已有实现。

## 现在新增了什么

- 手写 CUDA QK/PV `mma.sync`、`ldmatrix` 和 V 的转置加载。
- 每个 warp 独立负责 16 个 query，四线程一组做行 max/sum。
- P 经共享内存与 P 寄存器直通的对照。
- 单/双 K/V 缓冲的 `cp.async` 流水对照。
- 四段直接读取，包含 token stride6144 的 current V，不额外转成连续全局 buffer。
- 预分配输出、固定 owners/metadata/current-stream 入口和 Graph 重放。

M34/M35/M36 共80项数值检查通过；M35/M36在所测输入上与对应M34配置逐位相同。测试包含T0、空历史、query/key尾块和padded V。三批诊断都验证所有性能臂在计时前后的独立FP32误差；不代表已经覆盖任意shape/stride或训练反向。

## 优化判断过程

| 阶段 | 假设和改动 | 结果 |
|---|---|---|
| M34 | 建立CUDA数据流；比较P直通、单/双缓冲 | 正确；部分场景超过指定Triton配置，仍慢FA2。寄存器223–237，无spill |
| M35 | 取消四段完全展开，减少live-state和静态代码体积 | 寄存器降到127–150，无spill；但匹配S1/direct在三个T均退化，保留失败 |
| M36 | 回到展开版本，删除direct-P未使用的4KiB共享暂存 | 寄存器不变、输出不变；S2在T96/T256显著改善，S1基本不变 |

这两项实验说明：减少寄存器并不自动增加有效速度；几KiB共享容量的变化却可能跨过驻留限制。驻留机制尚需硬件计数器确认，不能仅凭速度和声明容量断言因果。

## 当前同批性能

以下均为历史长度[2048,64,512]、padded current V 的Graph p50，单位µs/完整forward：

| T | M34 S1/direct | M34 S2/direct | M36 compact S2 | raw-layout pack+FA2 |
|---:|---:|---:|---:|---:|
| 32 | 224.77 | 189.38 | 189.70 | 164.86 |
| 96 | 320.77 | 382.46 | 321.79 | 291.07 |
| 256 | 545.02 | 594.69 | 514.05 | 467.71 |

M36 S2比匹配旧S2在T96/T256分别减少15.86%/13.56%；T256比同批旧S1/direct少5.68%，仍比FA2慢约9.9%。没有切换生产/引擎默认，也没有把局部forward结果写成TTFT收益。M34旧FA2 adapter有额外V转换，M35/M36已改用更强的raw-layout直接打包；不得跨批混算速度。

## 从源码学习

1. [M34 CUDA](../src/llm_gpu_kernels/cuda_gqa_attention.cu)：先看QK、行softmax和PV如何连接，尤其 `pack_bf16` 与 `load_value_b`。
2. [M34设计](M34_CUDA_GQA_DESIGN.md) / [来源与借鉴边界](M34_REFERENCE_NOTES.md)：算法、fragment布局、外部参考与本项目四段/stride适配。
3. [M35设计](M35_ROLLED_GQA_DESIGN.md) / [报告](M35_ROLLED_GQA_REPORT.md)：为什么更少寄存器的版本仍退化。
4. [M36 CUDA](../src/llm_gpu_kernels/cuda_gqa_attention_compact.cu) / [设计](M36_COMPACT_GQA_DESIGN.md) / [报告](M36_COMPACT_GQA_REPORT.md)：删除unused P scratch的容量取舍。

## 下一步

先完成已有新旧S2的NCU计数器采集，确认寄存器/共享限制、实际occupancy和调度变化。脚本 `scripts/run_m36_gqa_ncu.sh` 只读取已有binary，串行3个kernel，不修改驱动。再根据证据选择mask完整块快路或Q/score操作数调度；暂不堆加Q寄存器常驻、更多stage或量化变动。
