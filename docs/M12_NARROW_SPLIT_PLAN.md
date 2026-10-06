# M12：T≤64窄域 split-KV 验证

本轮新假设来自M11训练：token16/splits2在T48两输入对强FA2约少31%，但不能覆盖
全T≤256。因此M11保持全域FA2回退；这里另立窄域T≤64，并用新的训练/冻结边界。
这不是M11的预注册假设，也不是全新未见workload；不得回改M11数据或policy。

唯一候选固定token16/key64/G4/warps8/stages2/splits2，直接复用M11两个kernel。
保留partial FP32 scratch与merge、两次启动完整成本，不进行额外参数搜索。
T>64及无合格候选时回退复用buffer pack+未修改官方FA2公开API。

训练seed5109，T24/56 × selected257/1537，sink64/recent512/currentT，共4输入。
先使用M9完整12功能场景和两次Graph原地更新/非默认stream/恢复hash直接检查固定
split kernel（包含T65算术边界，即使policy不用于T65）。独立FP32原0.008 RMS/max
和finite；全部训练API/Graph前后chunk64 FP32检查与input hash。baseline失败停止。

Graph paired四轮AB/BA/BA/AB，8样本2replay8forward；唯一候选要全部4正确，
geomean ratio<=.95且worst<=1.05。通过才在同4训练输入长复测：4轮16样本4replay8
forward，同门槛。失败全域FA2回退，不换候选。冻结source hash、最终training canonical
hash（仅排除顶层policy_sha256）后再对照。

冻结后seed6207，T17/33/49/63/64 × selected0/512/2049共15新输入；另T65/selected512
域外FA2回退。三个arm：冻结选择、M9G1、强pack+FA2。API/Graph4轮正/逆8样本2调用
Graph8forward，全部前后独立chunk64 FP32、hash、telemetry、scratch/peak与partial/merge
编译证据。唯一选定split再12功能场景/两次graph更新与非默认stream，不根据对照
结果改配置/阈值。policy若fallback则相同FA2微小差异不解释为kernel收益。

不含模型、RoPE、H2D、NanoKV/serving，SM86单GPU forward；不由register/HMMA/CTA数
推断utilization、occupancy、DRAM或因果。保留所有失败/回退与完整成本。
