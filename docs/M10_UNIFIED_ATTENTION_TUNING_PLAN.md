# M10：统一逻辑 KV 遍历的有限搜索与冻结验证

本计划在M10 GPU测量前固定。M9已知结果支持新的假设：四段内联循环和较大head
分组的静态资源较高，统一逻辑KV单循环与不同warp数可能改善资源/完整延迟。
这不是已证实的硬件因果解释；保留M9全部回归，不修改旧代码或阈值。

## 候选与契约

新attention_unified.py逐个逻辑key lane按三个segment边界选择原始K/V指针，
单一循环处理四段；不cat、不repeatKV。跨segment block可读取不同原始buffer，
各lane局部offset和load mask必须正确，空段不被选中。历史全可见，current j<=q_i。
Q[T,32,128]、四段KV[L,8,128]、BF16、FP32 online m/l/acc和BF16 P的PV dot，
currentL=T，与M9数学契约相同。T0无启动，prepared owners/currentstream规则相同。

固定12候选：token_tile={16,32} × G={1,2,4} × warps={4,8}，key_tile64/stages2。
候选顺序依上述循环，完全相同的配置集合用于所有训练场景。每候选先检查
T65/history33/17/41非对齐段尾与T0；资源/数值失败保留并排除资格。禁止扩大0.008
RMS/normalized max界，所有输出finite。

## 训练与选择

seed1907，T={96,384,1536} × selected={256,1536}，sink64/recent512/currentT，
共6输入。同一T共享可重现Q和公共段。全部使用独立chunk64 dense FP32 oracle，
禁用TF32。对照仅为原M9 G1 token32/key64/warps4/stages2完整算子，不用FA2结果选配置。
每候选在每输入比较完整forward：计时前后API与Graph均FP32检查、输入hash、
resources和telemetry；总72行记录。baseline不正确时停止，不把问题算成候选失败。

Graph paired四轮AB/BA/BA/AB，每轮8样本，每样本2 replay，每replay8 forwards，
按完整forward归一化。选择基于Graph p50，不计编译、准备、oracle、hash、telemetry。

shape-only三个bucket：small T<=128、medium 128<T<=512、large T>512。
候选必须boundary和全部6输入正确且测量成功。每bucket两输入的paired ratio
几何平均<=0.95且最差ratio<=1.05者中选最低几何平均，index作为确定性tie-break。
没有合格者回退原M9 G1，不按KV内容或GPU统计值选配置。

初选后同bucket两训练输入长复测：四轮AB/BA/BA/AB，每轮16样本，每样本4 replay，
每replay8 forwards。使用同一5%几何平均和5%最坏回归限制；失败即fallback，不试
次优候选。只fallback的bucket保留原因，不为追求收益调整门槛。

冻结policy保存配置或M9fallback、明确源码hash、输入/训练来源与最终训练记录hash。
训练记录canonical JSON只排除top-level policy_sha256，避免与policy反向hash成环；
其他字段全部绑定，冻结后不可再改训练/选择。计时原始samples、编译/数值失败保留。

## 冻结后对照

seed2917，T64/256/1024/4096 × selected0/512/2048，sink64/recent512/currentT，
12输入。旧M9矩阵已经已知，新的候选结果未参与选择；不称完全未知workload。
三个arm：原M9 G1、冻结统一候选或fallback、复用buffer pack+未修改官方FA2公开API。
API/Graph同轮四次正/逆顺序，8样本/轮、2调用/样本，Graph8 forward/replay。
全矩阵独立chunk64 FP32检查所有arm的API/Graph前后输出，恢复input hash。

每种唯一冻结(mode,config)另外调用M9完整12-small功能矩阵、两次Graph值更新与
非默认stream、恢复hash；不通过时记录失败并停止，不能根据holdout改配置。
保存各配置PTX/cubin/SASS hash、HMMA、register/spill/shared metadata。无硬件
计数器时不宣称occupancy、DRAM流量或寄存器spill的因果贡献。

不含RoPE、CPU gather、H2D、router、模型或NanoKV系统执行。单GPU forward原型
不代表serving、真实模型质量或端到端收益。候选/训练/对照异常均先记录case状态、
配置、traceback和可得metrics；保留失败记录，不覆盖以前结果。
