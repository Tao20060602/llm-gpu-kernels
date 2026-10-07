# M22 设计草案：SM86 gate/up 核心与 W4A16

2026-10-07。本文件是实现前的候选设计，不是性能结果。M21 T17 gate/up 的
DRAM 吞吐约 95.8%、L2 hit 约 0.5%，因此主线优先减少权重搬运量：建立手写
BF16 核心作为对照，接着加入 W4A16 在线解量化。本轮仍保留旧 M7 policy，
新增入口用于对照，不替换已发布实现。

## 切入点与范围

M21 的 T17 时间线中 gate/up + SwiGLU 约占关联 kernel 时长的 66%。首个手写
CUDA 核心选择这个阶段；完整 forward 仍复用现有 GPU ranked dispatch、down、combine。
先支持实际 SM86、连续 BF16、E128/H2048/I768/topk8、BM32 task metadata、固定地址
forward。T0、空 expert 和最后一块不足 32 行必须正确；不宣称任意 shape/backward 支持。

初始同精度对照保持 BM32/BN64/BK32，不由 sweep 挑“最好”的大量配置。
保持 prefix 的 BM32，避免同时改变 down 分块而混淆 gate/up 的贡献。

## 手写 CUDA 的工作拆分

- CTA 128 threads / 4 warps，每 warp 负责 16 个输出列。
- 使用 BF16 × BF16、FP32 accumulator 的 WMMA 16×16×16；每 warp 计算两个 M16
  子块以及 gate/up 两个投影。四组累加器的资源成本必须实际编译与测量。
- 当前 expert 的有效行数不足 17 时，第二个 M16 子块在整个 CTA 一致的分支中跳过。
  不能在 warp lanes 内分歧调用 WMMA；该消融只减少无效算术，不声称减少权重字节。
- A 与 gate/up 权重用 16-byte coalesced 搬运。先做同步 shared staging 版本，
  再做双缓冲 `cp.async` 版本：计算当前 K32 时搬运下一个 K32，明确 commit/wait 和
  block barrier；尾行读取 zero-fill，并使用合法源指针，防止无效地址访问。
- 首版 shared stride 为 K32+8，即 40 个 BF16 元素；matrix tile 基址满足 WMMA 的
  对齐要求。padding 是否改善 shared 访问需要实测，不预先声称消除 bank conflict。
- K32 内分成两个 K16 WMMA；共享 B fragment 供两个 M16 子块使用。
- 最后复用已经消费完的 shared staging 空间存 FP32 gate/up epilogue，分别舍入为
  BF16 后转 FP32 计算 SwiGLU，再写 BF16 activation，遵守原 M4 合约。
- prepared 入口持有 buffer owners；run 在当前 stream 提交，不分配、不回读 GPU counts。
  CUDA Graph 地址固定，动态值/IDs 通过原 ranked metadata 链路更新。

选择 WMMA 首版是为了先得到可审计的手写访存/流水线实现，不等于已经手写每条
`mma.sync` 寄存器映射。记录编译后的 HMMA、异步搬运指令和资源；再按证据决定
是否值得替换为直接 PTX。编译指令存在与运行时利用率分别验证。

NVIDIA 官方说明 BF16 WMMA 要使用 FP32 accumulator，warp 全体参与矩阵操作；
Ampere 支持 global→shared 异步搬运。这些能力适用于本机，仍需实测实际编译产物。
参考：[CUDA 12.8 Programming Guide §7.24](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#warp-matrix-functions)、
[Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/12.5.0/ampere-tuning-guide/index.html)。

## W4A16 格式草案

先仅量化 gate/up 权重，down 保持 BF16，以便定位单阶段贡献。A 为 BF16，
计算前在 tile 内把 INT4 weight 解为 BF16，Tensor Core 计算仍为 BF16/FP32。
不能称为原生 INT4 × BF16 MMA，也不把它当作整数激活量化。

- 每 expert、每 output channel、沿 K 的每 128 个权重一组。
- 对称 q ∈ [-7,7]，8 个 q+8 nibble 打包成一个 uint32；低 4 bits 对应组内最早 K。
  零权重使用 q=0；不引入 zero-point tensor。
- scale 存 BF16，量化时使用已舍入的存储 scale，以 nearest-even rounding 后 clamp。
  全零组 scale=1；非零组若 scale 不可表示为 finite 正数则拒绝，不能静默改变精度契约。
- reference 的解量化为 FP32(q × stored_scale) 后舍入为 BF16，和实际 MMA operand
  对齐；SwiGLU、down、combine 的舍入点保持原合约。
- 原始 packed shape、group size、nibble order、scales dtype/shape、版本、权重 hashes
  一并保存。打包/格式准备成本单独记录，不塞入热路径，也不声称它免费。
- 热路径在 tile 内解量化，禁止先生成完整 BF16 gate/up 权重矩阵再把 dequant 成本
  排除在 benchmark 外。同步 staged decode 首版与可重叠 decode 版本分开比较。

仅 gate/up 理论存储从 768 MiB BF16 变为 192 MiB packed + 6 MiB scales；
这是形状与格式的计算值，不是已实现的显存峰值结果。原 down 的 384 MiB 仍在，
完整两个权重张量因此约为 582 MiB；workspace、allocator、源权重是否保留另报。

## 三层精度验证

1. 格式损失：解量化权重和原 BF16 权重的 relative RMS / normalized max，
   并记录量化后完整 expert-forward 相对原 BF16 forward 的变化，不设置虚假任务质量结论。
2. 实现误差：CUDA 相对同一解量化 BF16 权重的 FP32 oracle，沿用 finite、
   RMS / normalized max ≤0.008，不能因量化放宽算子实现容差。
3. 模型质量：完整 checkpoint 全相关层量化后才用相同模型/输入的任务分数评价。
   仅 layer0 expert-forward 无法证明“任务相对 dense 下降≤10%”。

## 实现前冻结的验证方向

先做同步版本数值及完整 forward，随后只增加双缓冲与可选 M16 skip 两个明确消融。
每次保留源码/输入/权重 hashes、全部采样、失败及资源。性能验收协议在运行前独立
冻结，使用多个新进程的成对顺序与原已固定 policy 比较，API 和 Graph 分开报告。
M21 的诊断 real/archive-T1024 不再作为未用过的调优验收输入。

CUDA BF16 首版对同精度 Triton/成熟 backend；W4A16 至少对“同一量化权重完整
dequant + 已固定 BF16 backend”计入转换的路径，并研究成熟 W4A16 backend 的布局、
BF16支持和共同 shape 域。没有共同精度/语义域时，明确不能据此声称优于成熟实现。
只报告完整 forward 正收益，或坦诚说明局部有收益、完整链路没收益及原因。
