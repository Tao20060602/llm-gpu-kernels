# M27 / M28 设计笔记

这份短说明记录 M27 与 M28 的设计、测量边界和当前取舍。

## M26 给出的线索

NCU 中 M25 到 M26 的变化值得作为线索记录，但每项指标只说明它本身：

下列计数来自 4096³ 的单 kernel profile。

| NCU 指标 | M25 | M26 padding | M26 swizzle |
|---|---:|---:|---:|
| `memory_l1_tag_requests_global` (sectors) | 104857600 | 54525952 | 54525952 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` (%) | 31.37 | 39.10 | 40.53 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` (%) | 49.24 | 24.64 | 24.48 |

global L1 tag sectors 下降不能解读为 DRAM bytes 下降。Tensor pipe active 上升与 achieved occupancy 下降同时出现，因此不要把“最大化 occupancy”当成单独目标。

从 M26 swizzle 的 source 页面按 `stall_math` 原始值排序，前五个 PC 全是 `HMMA.16816.F32.BF16`：

| PC | `stall_math` 原始值 | SASS 指令 |
|---|---:|---|
| `0x7976e728fda0` | 35564 | `HMMA.16816.F32.BF16 R60, R8, R110, R60` |
| `0x7976e7290080` | 8073 | `HMMA.16816.F32.BF16 R76, R20, R98, R76` |
| `0x7976e7290070` | 7351 | `HMMA.16816.F32.BF16 R80, R84, R94, R80` |
| `0x7976e7290090` | 7134 | `HMMA.16816.F32.BF16 R44, R20, R6, R44` |
| `0x7976e72900a0` | 6279 | `HMMA.16816.F32.BF16 R40, R20, R8, R40` |

这支持把 MMA 指令供给/调度列为 M27 的候选优化方向，但不能单凭这些采样断定根因。Nsight Compute 的 warp stall 采样关联到对应 PC 的 SASS 指令；它们不是耗时比例，也不能把各 PC 的值相加当成 runtime。`derived__memory_l1_conflicts_shared_nway` 的 CSV 单位为空且格式可疑，暂不解释为 bank-conflict 计数。[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

`launch__occupancy_limit_registers/shared_mem` 的原始 block 限额给出了具体资源取舍：M26 padding 为 4/3，swizzle 为 3/4。去掉 padding 后共享内存能容纳更多 block，但较高寄存器占用成为新的限制；两者的 achieved occupancy 因而接近。M27 缩短 B fragment 存活期，是尝试进一步改善这个资源平衡，运行时收益仍以基准为准。

## M27：保持 tile，调整短批调度

- 固定 tile 为 `64×128`；短批组大小 `G ∈ {1, 2, 4, 8}`。在 A tile 仍可复用时，将 B fragment 的加载与 MMA 工作交错安排。
- `G=8` 保持 M26 的算术顺序；每个输出 C 的 FP32 累加仍按原来的 K 顺序进行。
- 已报告的数值校验为 80/80 通过，与匹配的 M26 输出逐位一致。编译时，G2/S2/swizzle 使用 128 registers/thread，G8 使用 157；两者均无 spill。
- 同一 CUDA Graph schedule run 中每个尺寸都有 23 个 arms（含额外的 M25 arm）。每个尺寸的最佳 M27 arm不同，不能据此宣称有固定的最佳策略；默认 Torch 在这三种尺寸上仍更快。

以下只对照同尺寸的 M27 最佳结果、匹配 M26 arm 与默认 Torch，单位为 µs/GEMM：

| GEMM 尺寸 | M27 最佳 arm / 时间 | 匹配 M26 arm / 时间 | 默认 Torch / 时间 |
|---|---:|---:|---:|
| 1024³ | `G8/S2/padding` / 97.03 | `S2/padding` / 97.28 | 73.93 |
| 2048³ | `G1/S2/swizzle` / 628.10 | `S2/swizzle` / 671.95 | 582.25 |
| 4096³ | `G2/S2/swizzle` / 5357.36 | `S2/swizzle` / 5535.74 | 3721.01 |

完整 arms 和测量设置见 [M27 schedule report](M27_SCHEDULE_REPORT.md)。这些是同一 runner 的 CUDA Graph 对照值；不要把不同尺寸或更早运行中的 M26 数字跨批引用。

## M28：用 named barrier 做 warp specialization

M28 把 CTA 中的线程分成 producer 与 consumer：1 或 2 个 producer warp，加 4 个 consumer warp，对应 CTA 总计 160 或 192 个线程。producer 负责 A/W tile 的 `cp.async` 传输；consumer 读取 shared-memory fragments 并执行与 M27 相同的 G2 MMA。

对每个 stage 使用两组不同的 CTA named barrier：`ready = 0..S-1`，`free = S..2S-1`。例如 S2 使用 barrier IDs 0–3，S3 使用 0–5。每个 barrier 的到达线程数按完整 CTA 设置：160 或 192。

```text
producer(stage):
    若即将复用 stage: 等待 free[stage]
    发起 A/W tile 的 cp.async
    cp.async.wait_group 0
    bar.arrive ready[stage]       # 通知数据可读，不在这里等待 consumer

consumer(stage):
    bar.sync ready[stage]
    执行对应的 G2 MMA
    bar.arrive free[stage]        # 完成消费后释放 slot
```

`cp.async.wait_group 0` 等待该线程此前提交的 async-copy groups 完成；PTX 的 producer/consumer barrier 示例也展示了 producer 用 `bar.arrive` 继续工作、consumer 用 `bar.sync` 等待，以及消费者完成后反向通知 producer 的模式。[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) 每个 slot 只有收到 free 后才能复用。tail 路径需让参与者平衡到达，不自旋读 flag，也不覆盖仍在使用的 shared-memory slot。

S2 编译为 124 registers/thread、4 个 barrier IDs；S3 为 128 registers/thread、6 个 barrier IDs，两者均无 spill。M28 已完成 28 组 FP32 oracle 检查，并与同 stage 的 M27 G2/swizzle 输出逐位一致。三个大矩阵的同批 Graph 结果见 [M28 报告](M28_WARP_SPECIALIZATION_REPORT.md)：4096³ 的 P1/S2 相对 M27/S2 仅约 0.4% 改善，当前不足以确认可靠收益，因此不选为推荐路径。

producer 增加线程/寄存器预算和 barrier 资源；同 stage 的 shared-memory tile 容量不变，加深 stage 才增加该容量。当前未采集 M28 的计数器，不能把结构成本直接当成已测得的运行时瓶颈。保留无收益的候选，是为了说明 warp specialization 的适用条件，而非预设它一定获胜。
