# M13：固定query64直接读取的冻结适用域

候选为已有M9 G1 token32/key64/warps4/stages2，无新参数搜索。M9/M10已有不同seed的
Q64局部收益；M11/M12 split回归支持保留更简单的direct-read candidate。本轮是基于
已知数据的新鲁棒性/适用域验证，不称新算法或完全未知workload，不回改旧policy。

只在Q长度64、sink64、recent512、current64且总history<=4096的metadata域考虑候选；
其余输入统一回退复用buffer pack+未修改官方FA2公开API。selected可0，因此history
最少576。选择只读取host已知shape，准备时绑定owner/地址，值更新不改变选择。

seed7109训练Q64 selected0/255/1023/2049/3073，共5输入，独立chunk64 FP32原0.008
及finite、API/Graph前后与input hash。先M9全12功能/2次Graph值更新及nondefaultstream。
Graph paired4轮AB/BA/BA/AB，8样本2replay8forward；候选必须5输入正确，ratio
geomean<=.95/worst<=1.05，再同5输入长复测4轮16样本4replay8forward，同门槛。
失败全部FA2fallback，不换候选。source/training canonical hash（只排除policy_sha256）冻结。

冻结后seed8207和9917，Q64 selected0/129/512/1535/2048/3073/8193，共14输入；
8193为域外FA2诊断。另外seed8207 selected3520和3521检查history=4096与4097边界，
另Q63/selected512检查query域外，共17输入。强FA2、冻结选择、M9G1同轮API/Graph
4轮正/逆8样本2调用Graph8forward。全部前后FP32、hash、telemetry/peak、单kernel
编译产物。唯一direct选择再完整功能与stream/Graph检查，不因对照结果调整规则。

准备、输入、reference、warmup/capture和审计不计入；完整direct或pack+FA2均计入。
候选无KV pack或scratch，仅输出。全矩阵热重复输入；不含RoPE、H2D、gather、model、
NanoKV/serving。不由HMMA/资源推utilization，不把算子收益外推成系统收益。
