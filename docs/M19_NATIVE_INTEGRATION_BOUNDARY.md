# M19：Native NanoKV 接入后的收益与数值边界

2026-10-07。独立算子源码继续冻结在M18 `fa4da4d`；本阶段工程实现和原始请求数据
位于NanoKV，独立仓库只保存结果索引与结论，不混合引擎/模型依赖。

## 已完成

- 显式开启的adapter：缓存模板、新地址输入、同stream共享scratch；全36层共约35.19MiB
  新attention工作空间，避免每层长期存放大chunk拼接buffer。
- 固定Qwen3-4B、16480token=4096×4+96、Top32/sink64/recent512/summary1，保持既有检索
  算法和KV store次序。GPU计算正确恢复后运行，无系统驱动更改。
- 两种实验后端逐层同输入FA2/独立FP32比较及KV状态审计，1080项精度指标、864项状态
  检查通过。原.008双阈值不变，maxRMS .00177940、normalized max .00606061。
- 3组三arm新进程对照，9个独立模型进程、18项计时请求；完整记录编译/warmup和每层IDs。

## 完整模型结果

| 对照口径 | 完整prefill耗时几何平均比 | 本轮判断 |
| --- | ---: | --- |
| operator / 复用FA2 | 0.999172 | 少0.08%，方向不稳 |
| operator / 原FA2 | 1.005987 | 慢0.60%，六项配对都略慢 |
| 复用FA2 / 原FA2 | 1.006821 | 慢0.68%，复用不等于系统加速 |

18项首token与原FA2一致；只检查两个固定文本的首token，不等于一般质量验收。
所有4096后续主chunk选择一致；96尾chunk后续层出现重复的检索集合差异：archive11层、
code13层，每组都相同。24个prompt/layer差异各重复3次，不能计成72个独立质量案例。
selector源代码和预算不变。数值误差传播到后续Q/selector是由时序与差异数据支持的解释，
未做完整逐值因果trace。默认仍为原FA2，未以固定baseline IDs掩盖传播。

## 两种性能结论的边界

M18在GPU-resident真实tensor上，六项T96动态API比pack+FA2少33.17%；这是算子结论。
M19包含模型其余层、CPU gather/H2D/KV更新，完整prefill约5.8–5.9s且没有稳定收益。
不能把M18百分比写成TTFT、请求或模型加速，不能把局部数值容差写成模型无损。

本轮两个项目分别展示kernel优化与系统集成判断。没有通过新增模型工作负载重选M16
策略，已知回归/失败和不推广的理由均保留。

派生证据索引：[`m19-native-integration-summary.json`](../results/m19-native-integration-summary.json)。
系统端原始13份JSON、完整逐层选择与复现driver在NanoKV：
`benchmarks/results/operator_bridge/m19-native/`、`docs/NATIVE_SEGMENTED_ADAPTER_REPORT.md`。
源数据文件SHA256在派生索引中，默认接入/性能/质量未获推广，不是生产部署证明。
