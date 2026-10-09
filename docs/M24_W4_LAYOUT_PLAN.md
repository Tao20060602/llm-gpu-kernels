# M24 W4A16：保持量化算术的布局对照

2026-10-09。状态：设计方案，候选未获得 GPU 正确性或性能结果。

## 选择依据

M22 的 T1024 W4 完整链路慢于同 tile、同量化 BF16 CUDA 对照；小输入则有收益。
先完成 M23 跨进程复测，再决定该现象是否足以支撑进一步优化。

首版 words 为 `[E,N,Kword]`。一个线程读取一个 output channel 的四个连续
32-bit words，但同 warp 的线程跨 channel，地址间距为 1024 bytes。
16B 的单线程向量读取不保证 warp 的访存合并。候选将一个 K32 tile 的同一 word
排成连续 output channels，观察额外 load 指令与更连续访问之间的取舍。
这只是可检验假设，不从地址排列直接推断实际 DRAM 流量或加速。

## 唯一首轮改动

保持 E128/H2048/I768/topk8、BM32/BN64/BK32、128 threads、同步 shared staging、
BF16 WMMA/FP32 accumulate、BF16 gate/up 舍入与 SwiGLU。down 仍 BF16，完整
MoE 仍七次 launch。量化 q、K128 groups、BF16 scales 和 nibble order 不变。

新 layout 是原 v1 bit patterns 的离线排列：

- words：`[E,2,I/64,H/32,4,64]`，最后一轴是 N tile 内的 channel。
- scales：`[E,2,I/64,H/128,64]`，同一 K128 group 的 channels 连续。
- 对每个 K32，线程按 projection/channel 读四个 scalar words，随后保持原 shared
  地址、解码、WMMA 次序与 epilogue。activation 搬运保持原实现。

首轮不增加 async decode、不改 shared stride、不改变 tile、不跳过新的算术，
不缓存跨 K32 的 scale。后续每项另做消融，不能把多个变化合并成布局收益。
packed+scales 的逻辑字节仍是 198MiB；离线重排的时间和临时内存另报。

## Interface 与接入位置

候选使用独立的 layout metadata、CUDA extension 名称和 prepared 入口，保持
`prepare(...).run()` 的固定 shape/address、owners、current-stream 和无并发约束。
不让旧 `PackedGateUp` v1 自动接受不同形状，不覆盖旧 kernel 或 audit。

真实变化位置是 ranked dispatch 后、BF16 down 前的 gate/up+SwiGLU launch。
复用现有 GPU histogram/prefix/dispatch/down/combine 及它们的数值语义；prepare
持有新 packed layout，热路径不需要原 v1 words 或完整 BF16 gate/up。
候选入口不依赖调用旧 v1 prepare 后继续持有两套权重。

## 验证和计时顺序

1. 离线 bit-pattern round trip：新 words/scales 反排列必须与 v1 全部相等。
   重排不重新量化；canonical v1 digest 与新物理 layout digest 分别记录。
2. 六个原真实 case、七状态和 T0，检查完整 FP32 oracle、dispatch、Graph 值/ID
   更新和非默认 stream；沿用原 0.008 实现误差门槛，检查全部 finite。
3. 记录编译产物与资源。先通过正确性再测性能；不能拿“已写代码”当作验证完成。
4. 完整 forward 和 pregrouped gate/up 分开对照 v1 W4、同量化同 tile BF16 CUDA、
   同量化 vLLM。保留全部 case，特别是 T1024 和小输入回归；不拿阶段差值估计
   完整链路，不用更改量化误差换取“布局”收益。

M22/M23 的输入已用于诊断，因此本轮这些结果是机制对照，不能包装成新 holdout。
若要发布新的 shape policy，另冻结选择规则与新的评估输入。候选暂不替换默认入口。

首轮机制计时固定为一批六 case，每 case 新进程。完整链路按
`quant_vllm / quant_cuda_sync / w4_sync / w4_layout` 四 arms 测量，
每个正向和反向循环顺序合计八轮，每轮六样本、每样本两次调用。
Graph 每次 replay 含八次 forward，除以八归一化。阶段仅三个 CUDA arms，
固定已分组 metadata，六轮同样采样。全部准备、CPU 重排、编译均在计时外。
完整与阶段均记录 raw rounds、p50/p95、API/Graph；对六 case、T17/T193 四 case、
T1024 两 case 分别等权汇总配对耗时比，不汇总 p95、不丢弃回归。
计时前后及 Graph 输出另与原 W4 CUDA 输出逐 bit 比较并记录误差；若不同，
先停止性能采集检查原因，不能以双方都通过 oracle 门槛替代布局差分检查。
若失败保留 partial/日志，改动 source 后另开新实验目录并重新审计。

## 后续决策

布局和同步解码的对照不需要用户选择新的质量预算。改变 group size、异常值策略、
全层 down 量化或模型任务评估会改变目标和成本，届时单独说明取舍。
NCU 需要计数器权限时准备具体采集命令再请用户在本机输入密码；密码不进入聊天。
