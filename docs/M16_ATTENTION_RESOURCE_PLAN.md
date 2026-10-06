# M16：简化状态后有限资源分块搜索

不继续缩小旧规则适用域。本轮复用M15算法，在较宽的T32–4096域假设内研究key
分块32/64、query分块16/32、head_group1/2的八个组合；warps4/stages2固定。
较小key分块降低某些静态资源但增加循环次数，收益待测，不从spill推断性能。

训练seed12109：T96/384/1536 × selected257/2049，sink64/recent512/currentT，六输入、
48候选行。每配置T65/history33/17/41及T0数值检查。各arm API/Graph前后独立chunk64
FP32、原0.008界/finite/input hash；完整成本baseline复用pack+未修改公开FA2。
Graph四轮AB/BA/BA/AB，8样本2replay/8forward；长复测16样本4replay。

shape-only buckets small32–128、medium129–512、large513–4096；sink64/recent512、
currentT、总history≤4096，其他FA2。候选必须全部六输入正确；bucket两输入gm≤.95、
worst≤1.05，最低gm/index tie，长复测失败则FA2无次选。freeze最终training canonical
hash（仅排除顶层policy_sha256）和31项显式sources，在后续输入前完成冻结。

独立seed13207：T32/63/64/65/127/128/129/255/256/511/512/513/1024/4096 ×
selected0/512/2049，共42；另T31/T4097各selected512域外，共44。四arm：冻结选择/
M15默认/M9默认/完整FA2。API/Graph四轮正/逆，8样本2调用、Graph8forward。
全部独立FP32前后/hash、静态resources/PTX/cubin/SASS HMMA、原始event/wall/p50/p95。
每唯一选定配置12小功能场景、两次Graph值更新/非默认stream/恢复hash。
不使用holdout重新挑配置、调整阈值或域；有回归保持实验用途。

M15同配置固定A/B已知，不将本轮称为完全未知workload。单GPU SM86 BF16 forward，
GPU resident热重复，非model/NanoKV/H2D/RoPE/router/serving；输入scores有限，
不扩展nonfinite/overflow语义。静态资源不是occupancy/utilization。
