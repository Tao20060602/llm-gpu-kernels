# SM86 GQA 算子深化：先把复用假设算清楚

日期：2026-10-14。只读仓库记录、源码和一手资料；本笔记提出的 CUDA 布局、资源数和数值实验均未实现、编译、跑 GPU 或验证。目的不是列优化清单，而是找出一个能被反例推翻的假设。

## 已有证据与边界

- 当前显式 CUDA attention 的基础映射是每 CTA 一个 Q head、64 个 query rows、32 个 key rows、4 warps；`kv_head = q_head // 4`。每个 CTA 将一个 KV tile 从 global 请求并放入自己的 shared memory。M37 v2 的两阶段资源记录为 49,152 B shared、231 registers/thread。其完整 forward 在 T32/96/256 的同批结果仍慢于 raw-layout pack+官方 FA2。
- “按 GQA group 融合 query heads”已由 Triton M9 的 G=1/2/4 探过，不能称作新想法。M9 固定 token tile 32，使总 row tile 为 32G；G2/G4 在该编译证据中出现 220/234 spill bytes，且性能没有普遍胜过 G1。这个结果提醒我们资源成本真实存在，但不能直接预测不同 CUDA warp 映射的效果。
- M34/M37 已比较 P 的 shared 与 register 直通、K/V 一/两阶段、展开与 rolled traversal、移除未用 P scratch 和完整历史 tile mask；不要把这些重新包装成尚未做过的优化。M37 仍是显式实验，没有替换默认路径。
- 当前 attention 结果以合成输入为主。M17/M18 留有 Qwen3-4B 固定生成 prompt 的真实模型 Q/K/V 重放：T96 尾块及 T4096 主块、layers 0/17/35。它不是用户请求总体分布，也不是 M34 CUDA 的既有性能结论；适合成为后续数值稳健性输入。

## 要回答的窄问题：GQA 复用在哪一级发生？

`G=2` 只有在同一 CTA 的两个 Q heads 确实共用同一个 shared K/V tile 时，才省掉 CTA 之间重复的 global-to-shared 请求。每个 Q head 仍需独立的 score、FP32 `(m,l)` 和输出累加器；GQA 共享的是 K/V，不是 attention 概率或 softmax 状态。

必须分开记三种量：

1. **CTA → shared 请求**：同组 heads 同一 CTA 可直接共用一份 K/V tile，是布局能控制的量。
2. **L2 流量**：当前 `cp.async.cg` 绕过 L1；分开的 CTA 仍可能由 L2 命中供数。合组可能减少 L2 请求/SM 间搬运，但源码的重复 load 计数不能证明 L2 bytes 同比例下降。
3. **DRAM bytes**：如果当前重复请求已在 L2 命中，DRAM 读取可能几乎不变。只有同输入的 L2/DRAM transaction counters 才能回答 DRAM 是否减少。

### 两种容易混为一谈的对照

对 `T=256`，CTA 数近似为 `ceil(T / Q) × (32 / G)`，其中 `Q` 是每个 Q head 的 rows/CTA。以下数字按整齐 tile 计算：

| 映射 | Q rows/head | heads/CTA | CTA rows | CTA 数 | 能回答什么 |
|---|---:|---:|---:|---:|---|
| G1/Q64 | 64 | 1 | 64 | 128 | 现有 CUDA 基线 |
| G2/Q32 | 32 | 2 | 64 | 128 | 固定总 rows 的资源/行粒度对照；不能宣称减少总历史 KV staging |
| G4/Q16 | 16 | 4 | 64 | 128 | 同上；另可看较小 Q tile 对 current causal 遮蔽工作的影响 |
| G2/Q64 | 64 | 2 | 128 | 64 | 真正的跨 head staging 复用假设，CTA 数减半 |
| G4/Q64 | 64 | 4 | 256 | 32 | 不建议作为首轮候选，寄存器预算先已不成立 |

固定总 rows 的 G1/Q64、G2/Q32、G4/Q16 在 T256 都是 128 CTA；每个 CTA 对每个历史 key tile 仍只 stage 一份，所以总历史 K/V staging bytes 不变。它们是映射/资源 control，不是 KV 流量优化证明。Q 从64减到32/16 可能减少 current causal 段里“整块计算、再按 query mask 掉”的工作：M34 每个 query tile 会读到 tile 末端的 current prefix。这个收益要和更小 Q tile、更长 query-grid 的成本分别记账。

真正的复用候选是 G2/Q64：T256 时每 KV head 从 `4 个 token tiles × 4 个 Q heads = 16` 个 CTA 变成 `4 × 2 = 8` 个 CTA；82 个完整历史 key tiles（2048/32 + 64/32 + 512/32）对应的 global-to-shared K/V 请求量理想上减半。它不保证 DRAM bytes 减半，且 CTA 总数也减半。FA2 论文强调 attention 需要足够的 CTA 并行度；FlashInfer 的 head-group fusion 论文也说该策略主要对短 query 有用，query 维度本身足够大时复用不那么关键。这里 T32/96/256 都应覆盖，不能把 T256 的资源预测套给短尾块。

## G2/Q64 的资源账本：可测试，但未证明能跑赢

按 M37 已移除 P scratch 的路径估算，M34 当前每 head 的 Q tile 是 `64 × 128 × 2 B = 16 KiB`；S2 的 K+V 两级缓冲是 `2 × 2 × 32 × 128 × 2 B = 32 KiB`。G2/Q64 因此约需 `32 KiB Q + 32 KiB K/V = 64 KiB shared/CTA`，低于 SM86 的 99 KiB/CTA 上限，但按 100 KiB/SM 预算大致只容一 CTA。M36 的实测 NCU 曾显示 49,152 B 的 S2 可由 shared 限制驻留两 CTA；不能把它直接外推成 M37 或新布局的 occupancy 结论。

如果保持“每 warp 16 query rows”的分工，G2/Q64 需要 8 warps / 256 threads，而不是把每 head 的寄存器工作加在同一 warp 上。以当前 M37 S2 的 `231 registers/thread` 作 proxy，G2 估算为 `256 × 231 = 59,136 registers/CTA`，名义上低于 SM86 每 SM 64K 个 32-bit registers，但接近上限；新布局编译后的寄存器数未知。这里忽略寄存器分配粒度，数字只是预算预测。两个 4-warp CTA 与一个 8-warp CTA即使理论驻留 warp 数相近，CTA 调度粒度、barrier 代价和长循环延迟隐藏仍不同。

G4/Q64 会变成 16 warps / 512 threads；沿用 231 regs/thread 的粗估约 118,272 registers，超过 SM86 64K 寄存器文件预算。其约 96 KiB shared/CTA 虽接近 99 KiB 上限，寄存器已是更直接的否定信号。此处同样忽略分配粒度且未经编译；除非先证明寄存器布局能大幅改变，否则不值得首轮实现。

## 数值问题：BF16 P 的误差如何被真实 V 放大？

M34/M37 在 FP32 中算 score、online softmax 状态和归一化分母；把未归一化 probability 转 BF16 后做 PV MMA，再用 FP32 分母归一化。于是 P 的 BF16 舍入主要扰动分子。对一行，输出误差满足粗略上界 `||δO|| ≤ Σ_i |δp_i| ||V_i|| / l`；误差是否抵消取决于概率和 V 的相关性。FA2 的无近似算法描述不等于 BF16 中间量逐位等于 FP32。

现有 M34 的极端 finite-logit 数值 case 证明已测形状通过阈值，但没有隔离 P 舍入的方向性难例。先纠正一个容易误判的构造：当前 `P = exp(score - row_max)` 是未归一化 probability；等 logits 时每项 P 都是 1，BF16 可精确表示。即使可见 key 数不是 2 的幂，常量 V 也只是一个零 P 舍入 control，不能暴露分母与舍入后分子的差异。真正的分母隔离需要至少两档 logits，使非最大档 `exp(-delta)` 不能被 BF16 精确表示；常量 V 下，独立 reference 可比较 `sum(round_bf16(P))/sum(P)` 是否偏离 1。后续可以用三类互相解释的检查：

- **零误差 control 与分母隔离**：等 logits、常量 V 验证当前未归一化 P 的舍入 control；真正的分母隔离使用 max 档和至少一个非 max 档，令非 max `exp(-delta)` 不精确 BF16，再令 V 全相同。FP32 reference 的分母累加原始 P，PV reference 的分子累加 BF16-rounded P，比较理论上应为常量 V 的输出偏移。
- **舍入边界**：用可实现的 BF16 Q/K 构造 probability 落在 BF16 midpoint 两侧的 logits；再用 many-small-probability 与一个 dominant probability 的分布，看逐项舍入误差累积。先离线搜索 logits，再用真实 kernel 验证，避免只测抽象 score 张量。
- **V 相关性**：符号交替且幅值较大的 V 放大小概率变化；真实 V 检查模型中常见的误差抵消。独立 reference 可记录 FP32 P、BF16 P、归一化分母、未归一化输出及最终输出的 absolute/relative error，不只看总 RMS。不要假设这些中间量能从公开 kernel API 读取，也不要给热路径增加 debug 开关。

把这些构造测试与 M17/M18 的 Qwen3-4B 捕获输入并列：用独立 reference 按 layer 和 query row 统计 logits range、probability 在 BF16 rounding bin 的位置、舍入前后 PV 差及 V 范数。捕获来自固定 prompt，不能称为线上分布；synthetic worst case 也不能拿来声称真实模型精度有问题。要回答的假设是“当前容差由人为边界控制，还是在已有模型激活里可见的误差结构控制”。上述测试均未实施。

## 性能下限：先判计算、数据搬运还是 softmax

以现有 M34 T256 形状、历史长度 `2048+64+512=2624`、current 256 为例，完整可见 query-key pairs 为 `32 × (256×2624 + 256×257/2) = 22,548,480`。QK 与 PV 两次 D=128 BF16 矩阵乘约为 `22,548,480 × 4 × 128 = 11.545 GFLOP`（按乘加计2 FLOP）。这只是数学可见工作；32-key tile、current prefix 和 mask 会让实际 Tensor Core 指令执行更多 work。

一次理想的唯一全局数据流大约是 11.25 MiB K/V + 2 MiB Q + 2 MiB output。它给出 roofline 的带宽分子，不是实测 DRAM 字节。比较下限应写成 `max(11.545 GFLOP / 实测可持续 BF16 Tensor Core 吞吐, 15.25 MiB / 实测可持续 DRAM 带宽)`，再单列 exp/softmax/同步/launch 成本；未有 SM86 counters 时不报一个臆造的理论微秒值。若实际时间远高于两项下限，需判断是重复 L2/shared 搬运、低 CTA 并行度、softmax/同步，还是 launch/打包边界，而不是继续盲调 tile。

## 下一次值得做的单一实验

若继续，优先做 G2/Q64 对当前 G1/Q64：保持每 head 的 Q rows、key tile、mask、P 精度、段 stride 和 buffer policy 相同，只把两个同 KV head 的 Q head 放入一个 8-warp CTA。T256 首先能验证跨 head staging 预期；T32/96 是防止 query-grid 不足及 Q 尾块浪费的反例。同步记录 registers/spills/shared、CTA 数、完整 Graph forward、L2 请求/命中与 DRAM bytes，并用同一批输入做 P 数值检查。若 global-to-shared 请求减半但 L2/DRAM/延迟不动，复用并未触及瓶颈；若寄存器/CTA 压力导致延迟上升，也应保留为有解释的失败。

## 一手来源

1. [FlashAttention v2.8.3 官方 CUDA forward kernel](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/flash_fwd_kernel.h#L181)：online softmax、score/P 类型转换、PV 路径；算法“无近似”不表示中间 P 不舍入。
2. [FlashInfer 论文](https://homes.cs.washington.edu/~arvind/papers/flashinfer.pdf)：附录 A 的 GQA head-group fusion；同 CTA 一次 shared K/V load 服务一组 query heads，并限定其对短 query 的主要价值。
3. [NVIDIA CUTLASS Efficient GEMM 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)：threadblock、warp、thread 的分层复用，以及较大 tile 带来的并行度取舍。这里借用复用分析框架，不把 GEMM 映射当 attention 性能证据。
4. [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)：SM86 64K 32-bit registers/SM、100 KiB shared memory/SM、99 KiB/block、BF16 Tensor Core 和异步 global-to-shared copy 能力。
5. [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)：正文讨论减少非矩阵 FLOPs、提高 CTA/warp work partitioning，并指出即使单 head 也可沿 query blocks 并行以增加 occupancy；这些是并行度反例，不是 SM86 上的性能预测。
