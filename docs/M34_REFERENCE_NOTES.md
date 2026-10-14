# M34 外部源码参考与候选

本笔记对照当前 `cuda_gqa_attention.cu` 的 CTA64×K32、D128、四 warp、BF16 和四段直读结构。它记录源码模式与待验证实验，不代表建议已采用或已测出收益。M34 保留四段顺序、各段独立 row stride、current `key <= query` 因果边界及跨段在线 softmax 状态；外部 FA2 的单一连续 KV 遍历不能直接替代这套索引语义。

## 对照结论

官方 FlashAttention v2.8.3 的 SM80 kernel 提供可选 `Is_Q_in_regs` 路径：Q 从 shared memory 装入 MMA A fragment 后，后续 key tile 使用 `gemm<A_in_regs>`；但同版 launch template 的常见实例传 `false,false`，`true` 变体多为注释候选，不能把它写成默认获胜方案。M34 每个 key tile 都会从 `sq` 执行 `load_wide_a`，表面上有重复装载；然而当前四个变体已占 223–237 registers/thread、无 spill，D128 的 Q fragment 常驻会继续抬高寄存器压力，暂不应直接加入主线。先收短四段展开和局部 live range，再看编译资源，才值得做独立 Q-register A/B。

P 直通和 K/V 双缓冲则已在 M34 中实现。`DirectP` 将在线 softmax 后的 FP32 score 直接 pack 成 BF16 A fragment，送入 PV MMA；`Pshared` 把 P 写入 shared 后再读回，两者已有可匹配对照。FA2 也将 score 转成低精度 P fragment、调整为 A-operand layout 后调用 `gemm_rs`。Bias92 当前的 CUDA Tensor Core kernel 更直接：`accumulate_pv` 把 FP32 C fragments 的 P pack 成 FP16 A fragments，再与 V 做 MMA，源码直接佐证 DirectP 这类数据流。可迁移的是“P 不落 shared、就地重排/pack 成 MMA A operand”的结构；M34 仍须用 BF16，并保留四段指针、各自 stride 与段界。M34 的 S2 在计算当前 tile 时通过 `cp.async` 发起下一块 K/V 读取，并在下一轮等待；这属于现有实现，不是尚待加入的功能。CUTLASS CuTe 的 SM80 copy/MMA traits 可核对 16-byte `cp.async`、group fence/wait 与 BF16 `mma.sync` 的操作数/fragment 约定。M34 限定 SM86，适用 Ampere `cp.async`、`ldmatrix`、BF16 `mma.sync`；TMA 属于 Hopper 路径，WGMMA 源码由 SM90A 宏保护，不列为本机候选。

## 三项落地候选

| 项目 | 当前状态 | 建议的下一步 |
| --- | --- | --- |
| P 直接 pack 与 Pshared | M34 已有 `DirectP`/`Pshared` 两臂 | 保留为配对基线；对照转换指令、shared 写读与资源，不把外部实现当成收益证明。 |
| K/V `cp.async` 双缓冲 | M34 S2 已在 MMA 前预取下一 tile；四段边界会重启该段流水 | 将 S2 视为已实现基线；若继续调等待位置或段边界预取，单独做安全性与性能 A/B。 |
| 全历史块的 mask 快路 | 尚未实现 | 参考 FA2 将需 mask 的边缘/因果 tile 与完整内部 tile 分开循环；仅对段内完整历史 key tile 去掉 `key < length` 检查，保留尾 tile、Q 尾行和 current 因果 mask。按每段 stride 和实际长度独立判定。 |

FA2 没有四段专用的“取消四段展开”实现；其结构是一条 key-block loop，并把 mask-needed blocks 和完整无 mask blocks 分开。M34 可尝试降低四段 `#pragma unroll` 展开或缩短 score/P live range，但这只是资源压力实验，不能预设会减少寄存器或加速。Bias92 的 helper 和布局围绕 FP16 `mma.sync` 与单一连续 KV 区域组织；M34 应只借鉴 P fragment 的 pack/MMA 顺序，并自行适配 BF16、GQA head mapping、四段边界及 int64 row stride。

## 一手源码

- [FlashAttention v2.8.3 `flash_fwd_kernel.h`](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/flash_fwd_kernel.h#L181)：Q fragment、tile loop、mask split、P 转换及 `gemm_rs`。
- [FlashAttention v2.8.3 `kernel_traits.h`](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/kernel_traits.h#L48)：`Is_Q_in_regs` 和 SM80 kernel/layout traits。
- [FlashAttention v2.8.3 `flash_fwd_launch_template.h`](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/flash_fwd_launch_template.h#L182)：常见 dispatch 与被注释的 Q-register 变体。
- [FlashAttention v2.8.3 `softmax.h`](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/softmax.h#L136)：tile softmax 与在线 rescale。
- [CUTLASS v3.8 CuTe `copy_sm80.hpp`](https://github.com/NVIDIA/cutlass/blob/v3.8.0/include/cute/arch/copy_sm80.hpp#L72)、[BF16 `mma_sm80.hpp`](https://github.com/NVIDIA/cutlass/blob/v3.8.0/include/cute/arch/mma_sm80.hpp#L223) 与 [SM80 MMA traits/layout](https://github.com/NVIDIA/cutlass/blob/v3.8.0/include/cute/atom/mma_traits_sm80.hpp#L132)。
- [CUTLASS v3.8 SM90A WGMMA](https://github.com/NVIDIA/cutlass/blob/v3.8.0/include/cute/arch/mma_sm90_gmma.hpp#L39)；[NVIDIA Hopper tuning guide：TMA](https://docs.nvidia.com/cuda/archive/12.2.0/hopper-tuning-guide/index.html)。
- [Bias92 `flashattn-cuda` Tensor Core forward kernel](https://github.com/Bias92/flashattn-cuda/blob/main/cuda/flash_fwd_kernel.h#L57)：FP32 score fragment pack 为 FP16 A fragments，并用于 PV MMA。
