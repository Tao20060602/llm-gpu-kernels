# M26：面向大矩阵的 SM86 BF16 GEMM

M26 是独立的大矩阵候选，保留 M25 的 32×64 CTA 作为参照，不修改 M25。输入和输出为 BF16，累加为 FP32；`BK=32` 固定，所有配置使用 `mma.sync` 的 `m16n8k16` 指令。

## Tile 与数据复用

每个 CTA 计算 `BM×BN` 输出块。一个 K32 tile 读取 A 的 `BM×BK` 和 W 的 `BN×BK` 个 BF16 值，复用比为：

`2 BM BN BK FLOP / [2 (BM + BN) BK input bytes] = BM BN / (BM + BN)`

该指标忽略输出、缓存命中和 shared-memory padding。

| 配置 | CTA tile | warp 排布 | 每 warp 输出 | 每线程累加器 | 复用比（FLOP/输入字节） |
|---|---:|---:|---:|---:|---:|
| M25 参照 | 32×64 | 4 warp | 32×16 | 16 个 FP32 | 21.33 |
| M26 | 64×64 | 2×2，共 4 warp | 32×32 | 32 个 FP32 | 32 |
| M26 | 64×128 | 2×2，共 4 warp | 32×64 | 64 个 FP32 | 42.67 |
| M26 | 128×128 | 4×2，共 8 warp | 32×64 | 64 个 FP32 | 64 |

更大的 tile 提高每次输入加载服务的输出量，目标是大矩阵。代价是每线程累加器和 CTA 资源增加；CTA 数量也会减少，尤其可能限制小 M 的并行度。

## Shared memory 与 pipeline

两种布局都保留 stages 1–4：stage 1 同步搬运，stage 2–4 使用 `cp.async`。主要诊断阶段为 2 和 3。

| Tile | Padding，3 stages | Swizzle，3 stages |
|---|---:|---:|
| 64×64 | 30,720 B | 24,576 B |
| 64×128 | 46,080 B | 36,864 B |
| 128×128 | 61,440 B | 49,152 B |

Padding 布局每行以 `LD=40` 个 BF16 存放 K32 数据；swizzle 使用紧凑的 `LD=32`，shared-memory 存储量减少 20%。它将逻辑索引映射为物理索引：

`index(row, col) = (row*32 + col) ^ ((row & 7)*8)`

异或不改变 BF16 索引的低 3 位，因此 8 个 BF16（16 字节）向量仍连续；在每组 8 行内映射是双射。它会混合行位，不能只描述为逐行列重排。搬运和 `ldmatrix` 地址都通过同一映射从逻辑坐标求物理地址。节省 padding 不等于已消除 bank conflict；没有 NCU 数据前不对冲突或性能收益作结论。

当单 CTA 动态 shared memory 超过 48 KiB 时，prepare 阶段通过 `cudaFuncSetAttribute` opt in；`run` 和 CUDA Graph 回放不修改该属性。3-stage 的 128×128 padding 为 61,440 B，需要 opt in；swizzle 为 49,152 B，未超过 48 KiB 阈值。

## 风险与状态

大 tile 会增加寄存器和 shared-memory 用量，可能降低驻留度；CTA 数减少，M/N 尾块的无效填充也可能增加。M26 优先探索大矩阵，这些代价仍需测量。以 M=17、N=768 为例，64×128 或 128×128 tile 会将网格从 M25 的 12 个 CTA 减为 6 个；并行度受限是结构性风险假设，不是 profiler 定位结果。

2026-10-13 已完成编译、120 组数值验证和 1024³/2048³/4096³ 的同批 CUDA Graph 诊断。结果见 [M26 报告](M26_LARGE_GEMM_REPORT.md)，strict 与默认 BF16 reduction 基线分别记录；M26 结果单独评估，不并入 M25 结论。

24 个编译 specialization 均无 stack frame 或 spill。每线程寄存器数如下，不能用静态资源数字代替运行时 occupancy：

| Tile | Layout | S1 | S2 | S3 | S4 |
|---|---|---:|---:|---:|---:|
| 64×64 | padding | 70 | 80 | 96 | 96 |
| 64×64 | swizzle | 72 | 94 | 120 | 120 |
| 64×128 | padding | 122 | 126 | 159 | 159 |
| 64×128 | swizzle | 126 | 157 | 164 | 164 |
| 128×128 | padding | 125 | 124 | 128 | 128 |
| 128×128 | swizzle | 124 | 156 | 162 | 162 |

下一步采集 4096³ 上 M25 S2、M26 64×128 S2 padding/swizzle 的 NCU 计数器，判断计算供给、共享内存读取和等待成本，再设计 warp specialization。采集脚本只加载已编译二进制，单独的 profiler 耗时不进入无 profiler 的性能结论。

## 本机计数器采集（待用户执行）

当前驱动 `RmProfilingAdminOnly=1`，采集需要本机管理员认证。运行以下命令，在终端输入 sudo 密码，完成后提供脚本打印的目录：

```bash
cd '/home/tmz/文档/ChatGPT/llm-gpu-kernels'
bash scripts/run_m26_gemm_ncu.sh
```

脚本串行采集三个 kernel，每个 worker 运行四次，跳过三个匹配启动后采集第四次。它不调用编译器、不修改驱动设置；保存 NCU 报告、CSV、日志和只含 profile 信息的 worker JSON。已完成脚本语法检查，计数器采集尚未执行。
