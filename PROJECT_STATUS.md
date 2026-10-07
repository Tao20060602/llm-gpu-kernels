# LLM GPU Kernels：项目阅读入口

面向 RTX3080 Laptop / Ampere SM86 的 GPU 算子项目，研究动态 MoE 和分段 GQA
Attention。项目独立于 NanoKV 推理引擎，算子 microbenchmark 与模型 TTFT 分开报告。

## 先看这些

- [M22 W4A16：INT4 权重在线解码与融合](docs/M22_W4_REPORT.md)
- [M22 BF16：手写 CUDA、tile 与流水线消融](docs/M22_BF16_REPORT.md)
- [M21 MoE：用 timeline / NCU 定位瓶颈](docs/M21_MOE_PROFILE_REPORT.md)
- [计算与运行契约](docs/CONTRACT.md)
- [简历表述与面试提纲](docs/RESUME_NOTES.md)
- [首版之后的任务与时间估计](docs/POST_RELEASE_PLAN.md)

## 主要设计

| 问题 | 实现 | 证据与取舍 |
|---|---|---|
| 动态 expert 分组 | GPU histogram→prefix→ranked dispatch | values/IDs 动态更新、重复 routes/空 experts/tails、Graph 和 stream 验证；不回读 CPU counts |
| Gate/up 中间流量 | gate/up 与 SwiGLU 融合 | 遵守 BF16 round-trip，完整链路七次 launch；阶段与全链路分别测量 |
| Tensor Core 供给 | CTA32×64×32 的 BF16 WMMA CUDA 核心、16B staging、双缓冲 cp.async | 同步/异步/无效 M16 跳过消融；实际 HMMA/LDGSTS 与静态资源，未获得整体 BF16 收益 |
| 小输入权重带宽 | K128 分组 W4A16、tile 内解码+SwiGLU融合 | gate/up 逻辑权重768→198MiB；down保持BF16，完整逻辑权重1152→582MiB；格式损失另报 |
| 四段 GQA KV 输入 | 直读分段 K/V、在线 softmax | 与 pack+FA2 强对照；算子收益不冒充推理引擎 TTFT 收益 |

## 阅读结果时的范围

首版W4完整MoE Graph在T17/T193四个小输入的配对p50几何平均耗时较官方BF16 vLLM
低10.05%（有格式损失），T17示例1.309→1.132ms。T1024有回归，全部六case总体
只低1.61%。采用相同量化权重的强对照、全部样本和回归见M22 W4报告。

完整 MoE 指给定 route IDs/weights 的 expert forward，不含 router/serving。官方 vLLM
0.10.0 对照采用未修改公共 API 和自动配置；CUDA Graph 模式各自保留 pool/output。
量化实现误差以相同还原 BF16 权重 oracle 检查，格式损失与原 BF16 的比较单列。
当前 W4 只量化真实 Qwen3-30B-A3B layer0 gate/up，没有完整 checkpoint 质量结论。
没有宣称生产部署或超过成熟 W4 MoE backend。

历史实验来源与原始样本以公开 JSON/hash 固定；模型权重、原始 trace tensor、profiler
二进制报告保留本地。M22 W4首版按六输入单批交付，额外复测和布局优化后续开展。
旧 README 及 M20/M21 冻结文件保留，最新阶段由本页进入。
