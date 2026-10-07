# 两个项目的简历与面试证据

2026-10-07。项目独立组织，数字附适用范围和可追溯记录。

## 项目一：LLM GPU Kernels

可用的项目描述：

> 在SM86上实现BF16 Tensor Core MoE与四段KV GQA attention forward：GPU路由分组、
> compact dispatch、gate/up+SwiGLU局部融合、在线softmax及因果/GQA映射；构建独立
> FP32参考、CUDA Graph/stream与动态地址缓存验证，并与固定官方vLLM/FA2算子对照。

| 可写成果 | 必须同时说明的边界 | 证据 |
| --- | --- | --- |
| 完整MoE前向7次GPU启动，无CPU counts回读 | GPU处理给定route IDs/weights，不含模型router或serving | [M6](M6_RANKED_DISPATCH_REPORT.md) |
| 指定T1024 random场景Graph p50比固定vLLM默认算子少6.7–8.1% | 合成固定路由，E128/H2048/I768/topk8、两个seed；其他形状可能落后 | [M7](M7_FULL_MOE_OPTIMIZATION_REPORT.md) |
| 六项真实Qwen3-4B T96输入重放，动态API耗时比复用pack+FA2平均少33.17% | GPU-resident算子计时，包含V转换/绑定，排除模型、CPU gather/H2D；不是请求加速 | [M18](M18_DYNAMIC_BINDING_REPORT.md) |
| 编译模板LRU、独立workspace、新地址/值更新、固定图/stream和错误布局拒绝 | 图仍固定捕获地址，单GPU/stream顺序消费；LRU不限制Triton内部缓存 | [入口契约](M18_DYNAMIC_BINDING_USAGE.md) |
| PTX/cubin/SASS中的HMMA证据及寄存器/spill/shared-memory记录 | 静态证据，不等于Tensor Core利用率或occupancy | [M7](M7_FULL_MOE_OPTIMIZATION_REPORT.md)、[M17](M17_REAL_INPUT_BRIDGE_REPORT.md) |

面试可讲：为什么选择GQA head group、为何softmax状态跨四段共享、splitKV的merge成本、
小query收益与大query回退、路由倾斜的调度成本、BF16舍入和成熟后端不同、冻结策略后保留失败。

## 项目二：NanoKV 系统工程

可用的项目描述：

> 基于nano-vLLM迭代Qwen3-4B单请求稀疏推理原型，维护CPU完整KV历史、GPU代表检索与
> sink/recent保护窗口；实现pinned批量gather和实验算子接入，用逐层状态审计、源码/模型
> provenance及新进程配对实验区分局部算子收益与完整请求结果。

本轮具体证据：13个固定模型文件核对、1080项数值/864项窗口状态审计、9个新进程三arm
对照、首token与完整selected-ID记录、默认FA2的源代码分支保持。完整prefill没有稳定
收益，尾部后续层选择存在差异，因此保留实验入口并维持原默认。
系统历史中的CPU gather/selector性能与质量限制，使用NanoKV自己的README及M22收尾
报告作证据；不得把本仓库33.17%算子数字重复记为系统加速。

面试可讲：current store为何必须在attention之后、短尾recent复制的重叠/长度错误如何
修复、动态buffer生命周期、为何一套scratch可跨36层顺序复用、冷启动/warm请求区分、
为什么误差阈值通过后还检查后续检索集合与greedy。

## 当前成果验收

- 独立算子：限定场景的成熟对照收益、Tensor Core指令证据和复现材料已具备。
- 系统接入：工程路径可运行，性能/选择一致性证据不支持默认提升，结论公开。
- 两个项目都是单GPU实验原型；训练backward、多GPU/expert parallel、真实生产服务及
  广泛质量验收尚未得到验证。

完整系统边界见[M19](M19_NATIVE_INTEGRATION_BOUNDARY.md)。
