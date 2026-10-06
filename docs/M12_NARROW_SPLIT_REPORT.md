# M12：窄域 split 仍未通过回归门槛

本轮完成新的T24/56训练与冻结验证，但没有启用split。四输入几何平均ratio为0.890775，最坏ratio1.251045超出1.05门槛，因此全域FA2回退。不能用平均约11%收益掩盖单场景25%回归。

## 假设、协议与结果

唯一固定候选token16/key64/G4/warps8/stages2/splits2，依据已知M11训练的T48收益，另设T≤64假设。新seed5109，T24/56 × selected257/1537，sink64/recent512/currentT。没有参数搜索，也没有因失败换候选。

训练前12功能场景、两次Graph值更新、非默认stream与恢复hash均通过。四训练输入的独立chunk64 FP32、API/Graph前后检查通过，原0.008界/finite。Graph paired4轮AB/BA/BA/AB，8样本、2replay、8forward；固定geomean<=.95/worst<=1.05，未合格所以不做长复测。

| T | selected | split/强FA2 Graph p50 ratio | split scratch bytes |
| ---: | ---: | ---: | ---: |
| 24 | 257 | 0.641694 | 798720 |
| 24 | 1537 | 0.722283 | 798720 |
| 56 | 257 | 1.085837 | 1863680 |
| 56 | 1537 | 1.251045 | 1863680 |

T24两输入约少28%–36%耗时；T56两输入反而慢约9%–25%。这个边界转移没有支持T≤64整体启用。冻结policy保留FA2；18项source hash与最终训练canonical hash/反向policy hash验证通过。

## 冻结后回退验证

seed6207，T17/33/49/63/64 × selected0/512/2049，共15输入，另T65/selected512域外诊断，全部独立FP32检查通过。三个arm为冻结FA2回退、M9G1、强FA2；API/Graph4轮正/逆8样本2调用Graph8。FA2回退与强FA2为同一算子不同buffers，微小差异不算split收益。全部p50/p95、raw event/wall、input hash、telemetry和峰值在JSON。

下一轮将验证已有M9G1的固定query64方案，而不是继续缩小split范围直到得到局部headline。明确训练新的KV长度、设metadata适用域，再测未参与选择的新seed及边界；其他形状维持成熟后端。split negative结果与核心实现完整保留。

证据：[计划](M12_NARROW_SPLIT_PLAN.md)、[训练](../results/m12-narrow-tuning-20261006T140708Z-61362601.json)、[冻结验证](../results/m12-narrow-holdout-20261006T141016Z-897113288.json)、[policy](../results/m12-narrow-policy.json)。

这些仅为SM86 GPU resident四段BF16 forward，不是NanoKV/模型/serving或硬件utilization成果。
