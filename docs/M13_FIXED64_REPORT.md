# M13：query64的新长度验证未通过最坏回归门槛

固定M9 G1的五个新训练输入全部正确，平均有收益，但selected3073慢约6.85%，超过预先规定的5%最坏回归上限。因此没有启用自定义路径，冻结为FA2。不是调整阈值以接受平均收益。

## 适用域假设与训练

基于M9已知Q64收益，候选固定token32/key64/G1/warps4/stages2，不是新kernel或搜索。假设域为query64、sink64/recent512/current64、总history≤4096；其余FA2。新seed7109、selected0/255/1023/2049/3073，五输入独立chunk64 FP32、API/Graph前后及input hash。

完整12功能场景、两次Graph更新、非默认stream和恢复hash通过。Graph paired四轮AB/BA/BA/AB、8样本2replay8forward。要求geomean≤.95且worst≤1.05，通过才同五输入长复测。geomean0.887256（平均约11.3%更快），worst1.068530，所以不长复测、不换候选。

| selected | candidate/pack+FA2 Graph p50 ratio | Candidate p50 / p95 ms | FA2 p50 / p95 ms |
| ---: | ---: | ---: | ---: |
| 0 | 0.711193 | 0.043456 / 0.043840 | 0.061103 / 0.061248 |
| 255 | 0.794189 | 0.062976 / 0.063341 | 0.079296 / 0.079773 |
| 1023 | 0.916196 | 0.110496 / 0.110848 | 0.120603 / 0.120960 |
| 2049 | 0.994394 | 0.167104 / 0.169494 | 0.168046 / 0.169773 |
| 3073 | 1.068530 | 0.231168 / 0.234605 | 0.216342 / 0.218304 |

启动前曾误引用计划文件名，修正后才进行GPU验证；[启动失败记录](../results/m13-startup-path-failure.json)保留，计算、矩阵和阈值没改。正式19项source hash、最终training canonical hash与反向policy hash冻结后验证一致。

## 冻结后回退验证

seed8207/9917，各Q64 selected0/129/512/1535/2048/3073/8193，共14输入；另seed8207 selected3520/3521检查history4096/4097，Q63/selected512检查query域外，共17。全部独立FP32/API/Graph前后通过。

实际选择全为FA2，故这不能称为fixed64的holdout性能证明；强FA2与冻结FA2只是独立buffers/pools的同一公开算子，微小差异不作为加速。M9G1诊断同时保留，完整API/Graph p50/p95和raw样本在JSON。

## 结论

目前attention具有可复现数学实现、Tensor Core编译证据、成熟完整链路对照，以及清楚的回归/回退边界；还没有一个满足本轮适用域门槛的冻结自定义性能策略。保留成熟FA2默认，MoE的M7结果则单独呈现。

继续改进时应另立实验问题，例如分析寄存器、数据布局和实际负载，而不是将平均收益写成可靠全域优化。当前没有counter或真实模型/NanoKV系统数据，不将算子结果外推为serving收益。

材料：[计划](M13_FIXED64_PLAN.md)、[训练JSON](../results/m13-fixed-tuning-20261006T141814Z-854724342.json)、[冻结回退JSON](../results/m13-fixed-holdout-20261006T142027Z-962062648.json)、[policy](../results/m13-fixed-policy.json)、[M9报告](M9_SEGMENTED_ATTENTION_REPORT.md)。
