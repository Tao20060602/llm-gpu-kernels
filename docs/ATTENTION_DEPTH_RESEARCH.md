# Attention / FlashAttention 深化路线调查

更新：2026-10-14。仅阅读本仓库与一方论文、文档和源码；未运行 GPU、测试或 benchmark。

## 本项目已经做到哪里

现在的自定义 attention 是 Triton `tl.dot` 前向，不是手写 CUDA attention：[`attention_nonempty.py`](../src/llm_gpu_kernels/attention_nonempty.py) 对四段 BF16 KV 直接做 online softmax，支持本项目固定的 Q `[T,32,128]` / KV `[L,8,128]` GQA 映射和 current causal mask；[`attention_dynamic.py`](../src/llm_gpu_kernels/attention_dynamic.py) 增加编译模板缓存和同 metadata 下的新地址绑定。M16 冻结策略只在 `T≤128` 的小 query 域选自定义路径，大 query 仍走官方 FA2；NanoKV 默认 attention 仍为 FA2。[`ATTENTION_USAGE`](ATTENTION_USAGE.md) 描述了接口和契约。

现有实现已经包含分块 QK/PV、FP32 `(m,l,acc)` 归并和 Tensor Core 指令证据，但硬件矩阵乘由 Triton 编译。M11/M12 的 split-KV 是两个 kernel 加 FP32 partial/merge 的 prefill 实验：M11 的 8 种配置没有通过全域门槛；M12 固定一组配置虽在小样本平均快约 11%，最坏输入慢 25%，故均冻结为 FA2。无需再做同类宽泛 split 数搜索。M16 大 query 测试也显示未胜 FA2；M17/M18 的 T96 真实尾块局部收益没有转化为 M19 的整请求收益（operator/复用 FA2 几何平均比 0.999）。[M11](M11_SPLIT_KV_REPORT.md)、[M12](M12_NARROW_SPLIT_REPORT.md)、[M16](M16_ATTENTION_RESOURCE_REPORT.md)、[M17](M17_REAL_INPUT_BRIDGE_REPORT.md)、[M18](M18_DYNAMIC_BINDING_REPORT.md)、[M19](M19_NATIVE_INTEGRATION_BOUNDARY.md)。

## 外部实现对照

| 来源 | 可核对的技术内容 | 对本机项目的启发 |
|---|---|---|
| [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691) 与[官方实现](https://github.com/Dao-AILab/flash-attention) | IO-aware tiling、在线归一化；FA2 聚焦 CTA/warp work partitioning 和减少非矩阵乘开销。官方 CUDA 版支持 Ampere、BF16、GQA。 | 已有数学骨架；下一步应深入 Ampere warp 数据布局、Tensor Core 指令和异步搬运，而非只改 tile 数字。SM86 不适用 FA3/FA4 的 Hopper/Blackwell 专属路径。 |
| [Triton fused-attention tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py) | 官方教程展示 FA2 风格的在线 softmax 融合前向。 | 可作 Triton 基线与正确性参照；本项目已有更具体的四段/GQA/current-mask 语义。 |
| [FlashInfer attention API](https://docs.flashinfer.ai/api/attention.html) 与[KV layout 文档](https://docs.flashinfer.ai/tutorials/kv_layout.html) | 面向 serving 的单/批 decode、prefill/append、paged/ragged KV；batch decode workspace 存储 split-K partials，plan/run 分离。 | 若拓展，应作为独立的低 query、分页 KV decode 工作负载；它和 M11/M12 的多 query 四段 prefill split 不是同一个实验。当前[安装文档](https://docs.flashinfer.ai/installation.html)要求 CUDA 12.9/13.0/13.4，项目锁定 cu128，先作源码/API 对照，不作现成依赖。 |
| [SageAttention 作者仓库](https://github.com/thu-ml/SageAttention)、[`core.py` 架构分派](https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py) 与[论文](https://arxiv.org/abs/2410.02367) | 源码中 SM86 选择 `sageattn_qk_int8_pv_fp16_triton`，SM89 才选 CUDA QK-INT8/PV-FP8；API 面向连续 HND/NHD 张量，causal 只用于 Q/KV 等长。 | 可借鉴 INT8 QK 的误差/平滑思路；它不是本机的 CUDA attention 对手，也不能直接替代四段 current-causal 路径。 |
| [公开 CUDA 实现源码](https://github.com/Bias92/flashattn-cuda/blob/main/cuda/flash_fwd_kernel.h) / [搬运源码](https://github.com/Bias92/flashattn-cuda/blob/main/cuda/flash_fwd_memory.h) | 可读到 `mma.sync`、`ldmatrix`、在线 softmax、共享内存 K/V stage、`cp.async` 双缓冲；README 自述 RTX 4060 Ti，未核验其 benchmark 数字。公开[简历](https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/fcf88c87-b7a6-4b09-a74a-055d89bf5d8d.pdf)仅作项目表达样本，不用其自述数据作排名。 | 一条可复现的手写 CUDA 学习路线；重点是把每阶段源码、正确性和测量边界公开，而不是对齐简历数字。 |

## 首选路线：SM86 CUDA BF16 GQA 前向

从本仓库已验证的四段 GQA 语义起步，写一条 CUDA C++ 前向：QK 与 PV 用 `mma.sync` / `ldmatrix`，FP32 在线 softmax，K/V 以共享内存和 `cp.async` 分阶段搬运；第一版先限连续 BF16 `[T,32,128]` 与 `[L,8,128]`，随后接回四段直读和真实非连续 current-V。Ampere 8.x 提供 global-to-shared 异步拷贝；SM86 也可做软件 warp 分工，本仓库 M28 GEMM 已实现 producer/consumer warp 划分（[设计与报告](M27_M28_DESIGN.md)），但这不是本 attention kernel 的已有实现。应区分软件 warp 分工与硬件架构特性：TMA、WGMMA、TMEM 和 2-SM 设计不属于 SM86 能力，不能冒充本机已实现或实测。[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)、[PTX `mma.sync`](https://docs.nvidia.com/cuda/archive/12.6.3/pdf/ptx_isa_8.5.pdf)。

评价先固定与现有 Triton `tl.dot`、官方 FA2 三方的同输入比较，再逐阶段检查 Q/K/V layout、共享内存 swizzle、寄存器/溢出、同步和实际 profiler counters；保留不同 T、历史长度、分段边界与非默认 stream 的正确性。先瞄准已有真实 T96 尾块并如实报告局部成本，再确认完整请求是否有收益。若 CUDA 主线稳定，第二阶段再二选一研究 INT8 QK 计算精度，或 INT8 KV 对带宽/缓存的影响；分页 decode 是另一份工作负载，不叠加到第一版。

完成并独立验证后，可写成：“为 RTX 3080 Laptop（SM86）实现 CUDA BF16 GQA FlashAttention-2 风格前向，使用 MMA/在线 softmax/异步 K-V staging；在固定真实 NanoKV 尾块及合成边界上与 Triton 和官方 FA2 对照，报告精度、kernel 延迟和端到端边界。”没有测出的指标不要写进简历；M19 已证明仅凭算子局部加速不能声称请求加速。
