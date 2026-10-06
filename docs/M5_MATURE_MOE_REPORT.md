# M5：完整 MoE 成熟后端与入口缓存对照（2026-10-06）

状态：正式运行通过，9/9 functional cases、15/15 performance cases完成。结论仅覆盖本机
GPU、给定合成输入和完整fixed-route forward，不外推为真实模型或serving表现。

正式记录：[M5 JSON](../results/m5-mature-moe-20261006T101645Z-942093289.json)。
实验协议：[M5_EXPERIMENT_PLAN](M5_EXPERIMENT_PLAN.md)。记录包含逐文件源码哈希、
输入哈希、每case遥测、原始样本、p50/p95、vLLM自动配置和peak allocated memory。

## 1. 本轮要回答什么

M4已实现完整GPU固定路由forward和gate/up+SwiGLU融合；M5补上两个对照：

- **M4 Python fused**：`prepare_fused_moe(...).run()`，保留现有8-kernel计算路径。
- **Cached fused**：缓存M4已编译的完整launch序列、pointer和constexpr参数；计算不变。
- **vLLM public API**：直接调用官方`fused_experts`入口，包含上游对齐、dispatch、GEMM、
  activation、combine及其公共API分配行为。

本轮主要检验小批量时，缓存launch入口能否减少Python提交成本，以及完整GPU MoE与成熟
public API在预先给定形状上的实际差距。测试采用合成权重和固定合成IDs；不是实际模型
forward、router质量或serving负载。结果仅用于本机、所列shape和本轮工具链，不外推为普遍
实现优势。

## 2. 环境与成熟基线来源

| 项目 | 本轮配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU，SM86，16 GiB |
| Python | 3.12.14；路径记于正式JSON |
| PyTorch / CUDA runtime | 2.7.1+cu128 / 12.8 |
| Triton | 3.3.1 |
| NumPy | 2.5.3 |
| vLLM | 官方未修改0.10.0 wheel；tag commit `6d8d0a24c02bfd84d46b3016b865a44f048ae84b` |
| vLLM runtime | `/home/tmz/ai/baselines/vllm-0.10.0` 隔离operator overlay |

wheel SHA256为`8ca37559d82b43b5e8c8248d2e4a1ecb51d6d4e5d517491d656df6491ed93dab`，
overlay中1,237个上游文件已与官方wheel逐个核对。overlay只包含导入和运行该MoE operator
所需依赖；它不是完整vLLM serving安装。vLLM不做配置覆盖，使用public API的官方自动
配置选择；每个性能case记录所选config、config文件名和`packaged_config_found`。本轮15个
性能case的`packaged_config_found`均为false。官方selector仍返回shape相关配置：Qwen T1/T17
和small T1为`BLOCK_SIZE_M/N/K=16/32/64, GROUP_SIZE_M=1`，其他case为`64/64/32,
GROUP_SIZE_M=8`。本轮不宣称存在针对这块GPU和每个shape调优过的精确JSON配置，也不将
selector结果描述成最优配置。
更详细来源、wheel与扩展哈希见
[vLLM runtime provenance](../results/provenance/vllm-0.10.0-runtime.json)。
overlay安装和复现步骤见[vLLM算子环境说明](VLLM_OPERATOR_ENVIRONMENT.md)。

项目与NanoKV独立。没有加载NanoKV、模型权重或真实模型路由；M5使用的Qwen3-30B-A3B
形状只是`H=2048/I=768/E=128/topk=8`，并不表示在Qwen权重或模型质量上的测试。

## 3. 正确性验证

先复用M4的9个预先定义functional cases，覆盖空T、尾块、空expert、重复ID、热点/随机路由、
int32/int64 ID、非2次幂expert数和负routing weights。Python fused、cached fused、cached
unfused与vLLM public输出都分别对照完整FP32 reference；保留原阈值：
`relative_rms <= 0.008`、`normalized_max <= 0.008`且输出finite。cached fused和cached
unfused还逐bit比较；vLLM与本项目路径不要求逐bit相同。

舍入次序解释了vLLM输出与本项目可能存在的小差异。本项目的SwiGLU在BF16 gate/up
projection回读后用FP32计算SiLU乘up，再将activation写为BF16；down FP32累加先写BF16，
随后以FP32乘routing weight，并按route slot顺序FP32累加后写BF16。vLLM native SiLU分支
先舍入到BF16再乘up；down FP32累加直接乘route weight后写BF16；topk2/3/4 native sum按
slot顺序逐次BF16累加，其他topk走ATen `sum_out`。两者实现同一route加权MoE数学语义，
但activation、down/weight及combine的阶段舍入并不相同。因此分别以完整FP32 reference和
固定阈值判定，不要求vLLM与本项目逐bit相同。

固定地址Graph会原地更新`x`、IDs和routing weights，再验证新值下的输出；cached入口另在
非默认current stream上运行并核对数值与dispatch。计时case捕获Graph后同样做两轮可变值
检查，并恢复原始输入/route SHA256后才开始计时。更新、dispatch metadata readback、oracle、
hash和所有其他正确性诊断均不计入计时。

9/9 functional cases全部通过；cached fused/unfused输出在这些case中逐bit相同。包括空T、
不同ID类型、重复/热点route、非2次幂E、尾块与负权重的FP32检查。functional初始输出
最大relative RMS为0.004451112（vLLM），最大normalized max为0.007242441（本项目），
均低于原阈值。性能case的初始、更新及计时后FP32检查中，最大relative RMS为0.005269567，
最大normalized max为0.006259182；functional初始case保留的最大normalized max是
0.007242441。M4指定的`random_topk4_graph_update` functional case完成2次原地更新，cached
路径另通过1次非默认stream数值、dispatch和bitwise检查。15个performance Graph case又各
完成2次x/IDs/routing-weight更新，共30轮更新、每轮检查3个backend输出；全部通过FP32
检查，计时前输入和路由hash均恢复匹配。

## 4. 性能矩阵与计时边界

共15个完整forward case，权重为每case固定的合成BF16值，route由M4 helper按seed生成。
Qwen形状包含两个seed下的五个T值；两个热点case集中在16个expert；small组使用E16。

| 家族 | T | E | top-k | H/I | route | seed |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Qwen形状 | 1 | 128 | 8 | 2048/768 | random unique | 73 |
| Qwen形状 | 17 | 128 | 8 | 2048/768 | random unique | 73 |
| Qwen形状 | 193 | 128 | 8 | 2048/768 | random unique | 73 |
| Qwen形状 | 1024 | 128 | 8 | 2048/768 | random unique | 73 |
| Qwen形状 | 2049 | 128 | 8 | 2048/768 | random unique | 73 |
| Qwen形状 | 1 | 128 | 8 | 2048/768 | random unique | 991 |
| Qwen形状 | 17 | 128 | 8 | 2048/768 | random unique | 991 |
| Qwen形状 | 193 | 128 | 8 | 2048/768 | random unique | 991 |
| Qwen形状 | 1024 | 128 | 8 | 2048/768 | random unique | 991 |
| Qwen形状 | 2049 | 128 | 8 | 2048/768 | random unique | 991 |
| Qwen形状 | 193 | 128 | 8 | 2048/768 | 16-expert hot | 73 |
| Qwen形状 | 1024 | 128 | 8 | 2048/768 | 16-expert hot | 73 |
| Small | 1 | 16 | 4 | 128/64 | random unique | 73 |
| Small | 17 | 16 | 4 | 128/64 | random unique | 73 |
| Small | 193 | 16 | 4 | 128/64 | random unique | 73 |

API计时调用`optimization_benchmark.measure`：三个backend按正序和逆序循环轮换，共6轮；
每轮每backend 6个样本、每样本2次forward。完整保存CUDA-event原始样本、p50/p95及轮次顺序。
API范围包含各自`.run()`公共入口和Python提交间隙；M4两个wrapper使用prepare时分配的输出与
workspace，vLLM public API每次调用分配其输出和上游缓存。因此API比较有意保留真实入口
边界，但分配策略不同。

Graph分别捕获每个backend的完整forward，每次replay包含8次forward，CUDA-event结果除以8。
Graph地址、workspace与allocator pool固定；图提交间隙仍可能影响短case。图捕获及正确性
检查不计时。FLOPs统一按每forward `6 * T * topk * H * I` 估算；吞吐率是按该数学量推算，
不是硬件计数器读数。

### 4.1 全部15个Graph结果（p50 / p95，ms / complete forward）

| Case | M4 fused Python | Cached fused | vLLM public |
| --- | ---: | ---: | ---: |
| Qwen T1 random seed73 | 0.196640 / 0.199610 | 0.198336 / 0.200208 | 0.198459 / 0.198991 |
| Qwen T17 random seed73 | 1.898144 / 1.901546 | 1.898655 / 1.901352 | 1.873440 / 1.874688 |
| Qwen T193 random seed73 | 3.195290 / 3.199497 | 3.198654 / 3.202832 | 2.957088 / 2.963008 |
| Qwen T1024 random seed73 | 5.112956 / 5.119696 | 5.104857 / 5.109791 | 4.180448 / 4.267422 |
| Qwen T2049 random seed73 | 7.604446 / 7.724064 | 7.593952 / 7.709226 | 6.640160 / 6.784800 |
| Qwen T1 random seed991 | 0.194011 / 0.194736 | 0.193879 / 0.194828 | 0.190656 / 0.191288 |
| Qwen T17 random seed991 | 1.855836 / 1.859360 | 1.856576 / 1.860592 | 1.831392 / 1.833280 |
| Qwen T193 random seed991 | 3.197562 / 3.200464 | 3.202206 / 3.205712 | 2.956544 / 2.964634 |
| Qwen T1024 random seed991 | 5.112320 / 5.121753 | 5.108736 / 5.116191 | 4.145470 / 4.219536 |
| Qwen T2049 random seed991 | 7.680864 / 7.963073 | 7.658813 / 7.778096 | 6.786907 / 7.197679 |
| Qwen T193 hot16 seed73 | 0.742688 / 0.769598 | 0.741664 / 0.772464 | 0.707582 / 0.761991 |
| Qwen T1024 hot16 seed73 | 3.244032 / 3.295616 | 3.244093 / 3.337520 | 2.911813 / 3.061599 |
| Small T1 random seed73 | 0.012224 / 0.012288 | 0.012224 / 0.012288 | 0.009984 / 0.010048 |
| Small T17 random seed73 | 0.012412 / 0.012493 | 0.012352 / 0.012416 | 0.012992 / 0.013056 |
| Small T193 random seed73 | 0.015936 / 0.016015 | 0.015767 / 0.015808 | 0.023156 / 0.023230 |

### 4.2 Small组API入口比较（p50 / p95，ms / complete forward）

| Case | M4 fused Python | Cached fused | vLLM public | Cached p50相对M4 Python |
| --- | ---: | ---: | ---: | ---: |
| Small T1 | 0.148992 / 0.189056 | 0.046336 / 0.055296 | 0.225760 / 0.267008 | 减少68.90% |
| Small T17 | 0.131592 / 0.146432 | 0.039424 / 0.045568 | 0.193536 / 0.225536 | 减少70.04% |
| Small T193 | 0.124928 / 0.142808 | 0.038400 / 0.046476 | 0.184024 / 0.218880 | 减少69.26% |

缓存入口在small三例的API p50分别减少68.90%、70.04%和69.26%；所有15个case的API p50
也都低于M4 fused Python入口。Graph p50的small对应差异仅为0.00%、0.48%和1.06%。这与
“减少Python/API提交开销”的目标相符：明显收益出现在API入口，固定Graph中计算链没有
相应的大幅加速。变化百分比只描述本机这些case。

对比成熟后端时需单独看形状和计时边界。15个Graph case中，vLLM p50低于cached fused
12例；例如Qwen形状T1024 random seed73为4.180448 ms，cached fused为5.104857 ms。
small组里cached API明显快于vLLM public API，但Qwen大批量的Graph对照通常由vLLM领先。
因此本轮没有显示本项目完整MoE普遍优于成熟backend。API模式下每轮次与原始样本请见正式
JSON；小case尤其会受host-side公开API成本和分配边界影响。

## 5. 解释边界与后续

M4 cached入口不改变GPU数学kernel，目标是固定已编译的8个fused launch runner及其指针/
constexpr参数；unfused路径仅参加功能和bitwise验证。vLLM使用未修改官方operator及其默认
配置，不与M4内部kernel逐个对照。本实验没有tile sweep、配置搜索、完整serving栈或
expert-parallel通信，不测真实router，不使用Python expert-loop作为性能对照。

每case `peak_allocated_bytes` 覆盖该case创建的输入、workspace、oracle及CUDA Graph池/输出，
不是纯算子workspace占用或独立显存需求。温度、功耗和时钟逐case记录，但笔记本GPU未锁频；
固定输入反复执行的热缓存和时钟漂移仍是本地微基准限制。

M5表明入口缓存对small API调用有明确收益，但该收益在Graph中消失；本轮Qwen形状多数
Graph case仍由vLLM领先。下一步优先降低dispatch/launch组织成本，再结合阶段计时和编译产物
评估数学kernel与配置；不以入口缓存结果宣称成熟MoE整体更快或生产serving优势。

15个性能case的`peak_allocated_bytes`范围为9,396,224至1,880,303,104 bytes；这个峰值
包含当时存活的输入权重、oracle临时量、Graph池和workspace，不是纯kernel/workspace需求。
各case遥测与完整原始样本保存在结果JSON中。源码哈希已由正式运行记录并由root复核。
