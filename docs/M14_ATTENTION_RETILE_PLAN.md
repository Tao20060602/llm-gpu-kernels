# M14：query分块与流水线深度的有限对照

依据M9–M13已知资源/性能结果，本轮不改attention算法，仅调整现有四段直读kernel的
query tile、warps、pipeline stages。新候选固定六种：
(token32,G1,W4,S1)、(token32,G1,W8,S1)、(token64,G1,W4,S1)、
(token64,G1,W8,S1)、(token64,G1,W8,S2)、(token64,G2,W8,S1)，key_tile64。
候选没有因holdout结果而改变；pipeline/寄存器影响只是待验证假设。

训练seed9109：T48/80/192 × selected511/3073，sink64/recent512/currentT，共6。
先每candidate T65/history33/17/41和T0检查。全部API/Graph前后chunk64 FP32、finite、
原0.008界与hash。strongbaseline复用buffer pack+未修改FA2公开API。Graph paired4轮
AB/BA/BA/AB、8样本2replay8forward；36候选行完整成本，不计prepare/oracle/audit。

shape-only buckets为small T32–64、medium65–128、large129–256，限定sink64/recent512、
currentT、总history<=4096；其他形状FA2。每candidate必须全部6正确/测量成功，
每bucket两输入geomean ratio<=.95/worst<=1.05中选最低geomean/index tie。候选通过
同两训练输入长复测4轮16样本4replay8forward才冻结；失败FA2fallback无次选。
绑定最终training canonical hash（仅排除顶层policy_sha256）及source hashes，不修改旧实验。

冻结后seed10207：T32/63/64/65/127/128/129/193/256 × selected0/512/2049，共27。
另外T31与257各selected512两项域外回退。三arm为selected/strongFA2/M9G1。
API/Graph4轮正/逆8样本2调用Graph8forward，全部FP32前后/hash、resources与编译证据。
每唯一选定config全12功能场景/两次Graph值更新/非默认stream/恢复hash。
不依据对照改配置、阈值或适用域。若回归，报告并保持实验用途。

GPU resident热重复单GPU SM86 BF16 forward；无model/RoPE/H2D/NanoKV/serving。
不把静态资源推断成occupancy/utilization或硬件因果，不只报告单case赢家。
