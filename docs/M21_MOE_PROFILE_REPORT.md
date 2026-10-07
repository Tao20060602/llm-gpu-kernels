# M21：MoE 阶段归因与下一步取舍

2026-10-07。当前状态：四个负载的净计时、十二组 Nsight Systems 阶段分析完成；
Nsight Compute 普通用户采集被 `ERR_NVGPUCTRPERM` 拒绝后，用户完成了管理员采集，
三组报告覆盖五个指定 kernel，硬件计数器与来源审计完成。
没有修改旧 kernel、M7 policy 或 M20 冻结发布文件。

## 为什么做这一轮

M20 已建立真实权重/输入、成熟后端和完整 expert-forward 对照，但这些结果不足以
回答“瓶颈在哪里、为什么这个 tile 有效”。本轮先固定有限对照，用时间线判断主要
阶段，再用硬件计数器判断机制。手写 CUDA 和量化需要有这个选择依据。

原方案已有 gate/up 两个 GEMM 与 SwiGLU 的局部融合：共用 A tile、两个 FP32
累加器、保留 gate/up 的 BF16 舍入后写 activation。它减少中间读写和 launch；
旧 projected workspace 仍分配，不能声称减少了这块已分配显存。当前尚无量化算子。

## 固定实验与覆盖

协议见 `M21_MOE_PROFILE_PLAN.md`。使用同一 pinned Qwen3-30B-A3B layer0 权重。
archive-T17 / archive-T1024 为真实路由；balanced128 / hot16 保持 T1024 的 X、
权重和 routing weights，只替换 expert IDs，是机制对照，不是新的真实 router trace。

三个 arm 都是 BM32/BK32/4 warps/3 stages：default BN64、诊断 BN128；
frozen 在 T17 为 BN64、T1024 为 BN256。T17 的 default/frozen 重复配置仅用于一致性。

四个 perf worker 和十二个独立 profile worker 都通过数值/dispatch 审计及
source/artifact/input 前后哈希核对。十二组分析每组覆盖 8 次完整 forward，
7 stages × 8 = 56 个关联 kernel；每个 stage NVTX 区间对应一个 CUDA API 和一个
GPU kernel。原始 SQLite/nsys-rep 留在 ignored 本地目录，不发布环境元数据。

数值界仍为 relative RMS / normalized max ≤ 0.008，全部 finite。
四个负载最大的 relative RMS 为 0.003090 以下；没有放宽算子误差来换速度。
推理引擎的任务质量下降 ≤10% 是另一层约束。

## 净计时：完整 forward

单位 ms，以下为未开启 profiler 的 Graph p50；API/p95/原始样本另存于
`results/m21-profile-summary.json`。这是一轮诊断测量，不是新 policy 验收。

| 负载 | default BN64 | frozen | BN128 |
|---|---:|---:|---:|
| T17 real | 1.3297 | 1.3310 | 1.3289 |
| T1024 real | 4.4214 | 3.6615 | 3.7187 |
| T1024 balanced128 | 5.0794 | 3.6970 | 4.1150 |
| T1024 hot16 | 3.3205 | 2.8946 | 2.8810 |

hot16 上 BN128 的小幅优势不能据此替换旧 policy。本轮未验证跨进程稳定性，
且另外两个 T1024 负载表现不同。

## 时间线告诉了什么

下表为 instrumented GPU kernel p50，单位 µs。它仅用于阶段归因，不能当作
净性能结论，也不把 CPU NVTX 提交时间当作 GPU 时长。

| 负载 / arm | gate/up + SwiGLU | down | dispatch | combine |
|---|---:|---:|---:|---:|
| T17 frozen | 875.03 | 437.86 | 1.95 | 3.38 |
| T1024 real default | 2444.83 | 1744.15 | 90.75 | 90.67 |
| T1024 real frozen | 2092.68 | 1082.06 | 90.48 | 90.79 |
| T1024 real BN128 | 2094.34 | 1266.25 | 90.58 | 90.72 |
| T1024 balanced128 frozen | 2195.72 | 1267.49 | 90.45 | 90.85 |
| T1024 hot16 frozen | 1646.98 | 861.66 | 90.50 | 91.86 |

T17 frozen 中 gate/up 约占已关联 GPU 执行时间总和的 66%，down 约占 33%；
GPU routing/prefix/combine 占比很小。因此本轮优先深挖 gate/up，暂不优化 histogram。
该占比按阶段 GPU duration sums 计算，不包含 host gap，不等于请求端到端占比。

T1024 real 上 BN256 与 BN128 的 gate/up 时间接近，但 BN256 的 down 更快。
完整链路收益不能全部归到 gate/up，两个 GEMM 也未必应该共享 N tile。
这是值得验证的候选假设，尚未实现或宣称独立阶段配置获益。

## Tile 的具体取舍

| gate/up 配置 | 静态 registers/thread | 静态 spills | shared bytes | T1024 gate/up grid CTAs |
|---|---:|---:|---:|---:|
| BN64 | 76 | 0 | 20480 | 4596 |
| BN128 | 144 | 0 | 36864 | 2298 |
| BN256 | 255 | 6 | 69632 | 1149 |

BN 增大减少 N 方向任务数，也可能增加 A 复用；代价是两个投影累加器同时变大，
寄存器和 shared memory 需求更高。这里的 spills 是编译资源信息，不是实际
local-memory 流量测量；不能仅凭它解释延迟或声称 occupancy 已被证明下降。

T17 的 136 条 route 分散到 59 个 active expert，BM32 对应 1,888 行 tile 容量，
有用行比例约 7.2%。这是 masked tile 的逻辑容量，并不表示真实分配了 padding
输入矩阵，更不是直接测出的多余 DRAM 字节。BN64→128 的完整延迟几乎不变，
小 M 的权重访问与 ragged 分块因此值得调查，尚不能认定为带宽瓶颈。

balanced128 和 hot16 都有 256 个有效 BM32 行 tile、8,192 条 route、无行尾填充，
但 active expert 数为 128 / 16，时长不同。两者改变权重工作集和跨 CTA 复用，
也改变每 expert 行数及调度分布；不能将它们当作只改变 atomic contention 的实验。

## 硬件计数器：改变了方向判断

用户顺序完成三组采集，选中 kernel 数为 1/2/2，实际名称/grid 与 worker 匹配。
三份报告及 worker 文件在离线导出前后哈希一致，完整安全指标见
`results/m21-ncu-metrics.json`。下表仅比较 gate/up，百分比保留原 metric 的归一化方式。

| NCU metric | T17 BN64 | T1024 BN256 | T1024 BN128 |
|---|---:|---:|---:|
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` (%) | 95.79 | 76.79 | 91.93 |
| `dram__bytes.sum.per_second` (Gbyte/s) | 428.75 | 343.70 | 411.48 |
| `lts__t_sector_hit_rate.pct` (%) | 0.51 | 64.01 | 59.09 |
| `sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed` (%) | 8.42 | 20.53 | 21.82 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` (%) | 31.69 | 8.33 | 16.48 |
| `launch__occupancy_limit_shared_mem` (blocks) | 4 | 1 | 2 |

T17 在这次采集中接近 DRAM 吞吐上限，L2 命中率很低；结合阶段占比，
优先减少 gate/up 的权重搬运量比继续扩大 tile 更有依据。
因此下一主线选择 **W4A16 在线解量化融合**，用手写 CUDA 建立搬运/解码与 BF16
Tensor Core 计算路径。先写同精度核心作为语义和流水线对照，不预期仅靠 BF16
pipeline 就能突破这次观测到的字节吞吐限制。

T1024 BN256 的共享内存上限为一个 block，active-warps 指标比 BN128 低；
其 L2 命中率却更高。这提供了“大 tile 复用与并发资源交换”的硬件佐证。
结合两者 gate/up 时间接近及 down 的差异，不能按 occupancy 高低直接选 winner。
计数器报告的 `derived__local_spilling_requests` 为 0；编译资源记录的潜在 spills=6
本身不足以认定产生了本负载下的实际 spilling 瓶颈。

这一轮使用 kernel replay、cache-control none、clock-control none，没有锁 GPU 时钟。
它提供具体采集条件下的机制证据，不代表完整 Graph 的相同缓存状态，也不替代
跨进程净性能与消融验证；不能将单次计数器值视为普遍架构结论。
这些限定与 NVIDIA 的 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
关于 replay/cache/clock 与指标归一化的说明一致。

## 下一步的选择依据

1. 已补充五个 kernel 的计数器：T17 frozen gate/up，以及 T1024 frozen/BN128 的
   gate/up 和 down。仅选 SpeedOfLight、ComputeWorkloadAnalysis、
   MemoryWorkloadAnalysis、LaunchStats、Occupancy。禁用 profiler 的 cache flush
   和 clock locking；kernel replay 的结果仍属于带扰动的计数器采集。
2. 小 M 采集显示高 DRAM 吞吐，优先研究 weight-only INT4 的在线解量化融合，
   将 packed 权重搬运、解量化和 Tensor Core 计算联起来，避免完整 BF16 权重落地。
   先实现匹配 BF16 语义的手写 CUDA gate/up 基础，随后在同一接口加入解量化。
   具体 tile/pipeline/格式取舍见 `M22_CUDA_QUANT_DESIGN_DRAFT.md`；当前没有对应
   实现或收益结论。
3. 手写 CUDA 首个核心只覆盖 SM86、已声明连续布局及固定 forward 域。
   在有依据的 tile/访存/pipeline 选择上做少量明确消融；不扩成大量参数盲搜。
4. 量化分别测格式误差、相同量化权重下的实现误差及模型任务质量。量化 reference
   与原 BF16 模型不能混为一谈；仅 expert-forward 不能证明完整模型质量。
5. 新候选在实现前冻结评价协议，保留失败，分别记录 pregrouped kernel 和含
   dispatch/down/combine 的完整 forward。与同精度成熟 backend 对照，不能把
   INT4 对 BF16 的加速全部当作 kernel 优化收益。

## 当前限制

这轮获得了具体负载的 DRAM/HMMA/active-warps 等计数器，尚未用新实现的消融证明
因果收益，也没有新 CUDA/INT4 kernel。计数器权限拒绝记录仍保留于
`results/m21-counter-access.json`，不能因后续成功而删除失败。用户提供的管理员终端
采集脚本为 `scripts/run_m21_ncu.sh`；它提升三个 NCU 及其 Python worker 采集作业，
不改驱动全局权限。
性能负载来自固定 layer0 expert-forward，不含 router 或整模型输出质量。
