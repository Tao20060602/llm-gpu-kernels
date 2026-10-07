# M22 W4A16：格式、融合与精度边界

首版同步 CUDA 已通过六个真实输入、每个七状态，共42个状态的数值与dispatch检查，
以及T0。性能按照 [运行前协议](M22_W4_PERFORMANCE_PLAN.md) 完成单批六个独立进程
测量；T17/T193有收益，T1024回归，旧 M7 policy保持不变。

## 从瓶颈到设计

M21 的小输入 gate/up 显示较高 DRAM 吞吐、较低 L2 hit；M22 BF16 的异步双缓冲
和无效 M16 算术跳过仍未带来全链路收益。因此从“重排相同 BF16 读取”转向“减少
gate/up 权重字节”，同时保留 BF16 down，便于隔离这个阶段的影响。

先选 weight-only INT4，而不是同时量化 activation：输入保持 BF16，tile 内还原 BF16
权重，再执行 BF16 Tensor Core / FP32 accumulate。这里是 W4A16 存储与解码，不能
称为原生 INT4×BF16 MMA；也不是完整 checkpoint 的量化推理实现。

| 设计 | 选择与代价 |
|---|---|
| 分组 | 每 expert、每 output-channel、K128 一组；更小 group 可减轻局部异常值影响，但增加 scale 字节和处理成本，本轮不扫 group size |
| 对称量化 | q∈[-7,7]、stored BF16 scale；无独立 zero-point tensor，少用一个负值编码换取对称规则 |
| 舍入 | 先把 scale 舍入 BF16，量化时用它 nearest-even round/clamp；q×scale FP32 后舍入 BF16，匹配实际 MMA operand |
| Packing | 八个 q+8 nibble / int32 bit-pattern word，低 nibble 对应最早 K；逻辑布局 E/N/K-word，离线准备并保存 hash |
| Tile | BM32/BN64/BK32、128 threads，与 BF16 sync 的计算组织相同；不同时引入 async 或 M16 skip |
| 融合 | 每次 K32 在 shared staging 内解码，再 WMMA；最后 gate/up 舍入及 SwiGLU 融合，没有完整 dense BF16 gate/up 热路径张量 |
| 实际链路 | 保留 GPU ranked routing、BF16 down 与 combine，完整七次 launch；固定 owners/current stream，無 CPU counts readback |

每个 thread 读取一个输出 channel 的16B packed vector、解出32个权重，共享 K128
scale。该映射简化格式对应与验证，但跨 channel 的读取有跨度；16B 单线程读取
不等于整个 warp 已合并访存。bit 解码、BF16 conversion、shared 写入及 scale 读取的
成本均包含在候选计时，后续布局改动需要另立格式版本和测量，不能据理论字节量
直接断言硬件带宽利用率。首版选择简单可审计的映射，保留负收益的可能。

## 存储账本

原 gate/up768MiB → packed192MiB + scale6MiB，即 gate/up 逻辑存储约缩小3.88倍。
原 down384MiB 不变，两个权重张量逻辑总量1152MiB →582MiB，约减少49.48%。
这不包含 workspace、Graph pool、allocator 或上传期间的副本，不是单路径峰值显存
结果。审计/性能进程同时持有完整参考权重；不拿它们的 allocator peak 展示显存收益。
CPU packing 与完整 reference 解量化均在准备阶段单独记录，热路径禁止调用它们。

## 精度的三层边界

1. **格式损失**：原 BF16 权重与还原权重，relative RMS0.127129、normalized max0.067361。
   这描述权重变化，不能按百分比直接换算语言模型任务得分。
2. **CUDA实现误差**：相对同一还原 BF16 gate/up、原 BF16 down 的独立 FP32 oracle，
   42个状态全部通过原0.008门槛。最大relative RMS0.00331937、normalized max0.00434174，
   max absolute0.00386453。没有因量化而放宽算子实现误差。
3. **模型任务质量**：需要完整 checkpoint、相同输入的任务分数。当前只研究layer0，
   尚不能证明满足任务得分相对完整attention下降≤10%。

原始真实输入的完整 layer0 forward 格式代理（量化权重 oracle / 原权重 oracle 差异）：

| 输入 | Relative RMS | Normalized max |
|---|---:|---:|
| archive-T17 | 0.0405434 | 0.00982006 |
| archive-T193 | 0.0434180 | 0.00985011 |
| archive-T1024 | 0.0448426 | 0.00985022 |
| code-T17 | 0.0261095 | 0.00200941 |
| code-T193 | 0.0335781 | 0.00319863 |
| code-T1024 | 0.0358667 | 0.00319713 |

实现 audit 还覆盖 IDs 变成0/127的重复 route、空 expert、tail、Graph值更新、非默认
stream及恢复输入；source/packed/binary/input/trace hashes前后稳定。pack format只读，
GPU words/scales digest与离线元数据一致。实际编译72 registers/thread、0 stack/spills，
动态 shared16KiB；这些是静态资源，不是运行时 occupancy或spill traffic 的量测。

## 性能结果：小输入收益，大输入回归

以下均为完整 expert forward 的 Graph p50，包含 GPU分组、在线解码、gate/up+
SwiGLU、BF16 down 与合并；不是仅已分组 GEMM。每个case一个进程、多轮event样本，
没有跨批稳定性声明。量化后还原成常驻BF16的对照使用相同量化operand。

| Case | W4同步 ms | 同量化vLLM ms | 同量化CUDA sync ms | 原BF16 vLLM ms | 原BF16冻结M7 ms |
|---|---:|---:|---:|---:|---:|
| archive-T17 | 1.1319 | 1.3083 | 1.3426 | 1.3089 | 1.3252 |
| code-T17 | 1.2102 | 1.3972 | 1.4692 | 1.3976 | 1.4118 |
| archive-T193 | 2.0198 | 2.1634 | 2.3458 | 2.1715 | 2.2074 |
| code-T193 | 1.9312 | 2.0568 | 2.2534 | 2.0549 | 2.1273 |
| archive-T1024 | 4.4086 | 3.8433 | 4.2968 | 3.8481 | 3.5998 |
| code-T1024 | 4.4094 | 3.6737 | 4.2932 | 3.6450 | 3.5911 |

archive-T17相对原BF16 vLLM的耗时降低约13.5%，相对同量化BF16 vLLM也降低约13.5%。
但在两个T1024输入，W4比同tile量化BF16 CUDA sync慢约2.6%–2.7%，比原BF16 M7
慢约22.5%–22.8%。相同tile对照说明，较大输入下存储压缩的收益未覆盖解码等新增
成本；W4与M7的更大差距还包含BN64/BN256组织差异，不能全部归到INT4。

六case等权配对Graph时间比的几何平均：相对同量化vLLM为0.9834（耗时降低1.66%），
相对原BF16 vLLM为0.9839（有格式损失的取舍，降低1.61%）。**总体收益很小**。
事后按输入长度描述：T17/T193四case相对同量化/原BF16 vLLM分别降低9.97%/10.05%；
T1024两case相对原BF16 vLLM平均慢17.72%，相对原BF16 M7平均慢22.63%。这不是
按新policy选择后的holdout结果，所有六case仍保留在总体汇总中。

全部API/Graph、阶段p50/p95、原始event样本与校验见
[性能摘要](../results/m22-w4-performance-summary.json)。
同量化CUDA实现正确性见 [audit摘要](../results/m22-w4-audit-summary.json)，
实际HMMA/资源见 [binary inspection](../results/m22-w4-binary-inspection.json)，
计时前cache失败及修复见 [preflight记录](../results/m22-w4-perf-preflight-failure.json)。

本版定位小token MoE的W4A16融合原型。没有隐藏T1024回归，也没有自动新建一个
同时常驻BF16与INT4权重的“最佳路径”选择器：那会改变权重存储账本。进一步的
packed layout、异步decode及同域Marlin比较另立实验，不阻碍首版简历交付。

## 公平对照与未完成项

相同量化权重预先还原 BF16 并常驻 GPU，分别由未修改官方 vLLM、冻结 M7、同 tile
CUDA sync执行；候选使用 packed 在线解码。前者没有每次转换开销，是强的缓存对照。
这取代草案中先考虑的逐 forward 完整解量化机制对照；转换成本实验若新增，会另立
协议，禁止拿 Python expert 解量化循环作为 headline 加速来源。

另列原 BF16 vLLM/M7，体现有格式损失的延迟取舍。vLLM0.10.0 fused Marlin MoE要求
gate/up和down共用量化类型，当前gate/up W4、down BF16不对应直接同域的完整入口；
因此本轮不声称优于成熟 W4 MoE 后端。尚未全模型量化、任务评估、cold-start测量，
也未将本候选自动替换到推理引擎。
