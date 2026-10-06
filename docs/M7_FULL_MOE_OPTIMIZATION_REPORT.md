# M7：完整 MoE 形状策略与 holdout 结果（2026-10-06）

状态：训练搜索、冻结策略、功能/边界检查和 15 个性能 holdout 全部通过。
本报告仅描述 RTX 3080 Laptop GPU 上的合成固定路由完整前向；它不代表真实
Qwen 权重、模型质量、NanoKV 集成或 serving 性能。

正式材料：
[训练记录](../results/m7-full-moe-tuning-20261006T103809Z-241373177.json)、
[冻结策略](../results/m7-policy.json)、
[holdout 记录](../results/m7-full-moe-holdout-20261006T104838Z-727559245.json)、
[静态编译证据](../results/m7-compile-evidence-20261006T105911Z.json)。
预先确定的搜索与选择方法见 [M7 计划](M7_FULL_MOE_TUNING_PLAN.md)。

## 1. 问题与方法

M6 将完整 GPU fixed-route forward 固定为 7 次启动。本轮保持这一 dispatch、融合
gate/up+SwiGLU、down 与 combine 数学路径不变，只为 gate/up 和 down 共用一组
`GemmConfig`，测试固定的 tile 配置是否能改善完整前向。每个候选都包含完整
MoE forward；不是单独 GEMM 测量。运行入口在计时前完成配置、编译和准备。

训练只使用 `E=128, H=2048, I=768, topk=8`，seed 409 下的 `T=64/512/4096`
与 `balanced/hot16` 两种路由，共 6 个训练 case。相同 T 的两个路由使用相同
X、gate/up 权重、down 权重和 routing weights，只改变 IDs。策略输入只使用
`mu = T × topk / E` 和形状；不读取 GPU counts，不按 hotness 选择配置。

扫描固定的 12 种配置，每个配置都在 6 个训练 case 上测量，共 72 行结果。
配置的资格要求是在全部训练输入上通过正确性与测量，相关 bucket 的几何平均
相对 baseline p50 不高于 0.95，且任一 case 比值不高于 1.05。每个 bucket 只对
初选配置做一次更长的 balanced/hot16 复核；复核失败就回退到 baseline，不尝试
下一个候选。策略冻结后才在独立 M5 性能矩阵上测量已选配置，不根据 holdout
结果再调阈值、候选或 tile。

这里的“holdout”特指 M7 新候选配置的测量没有参与策略选择，并在冻结后才执行；
不表示这些形状此前从未测过。M5 已经测过同一性能矩阵的 M4/vLLM baseline，
这些旧 baseline 结果在 M7 开始前已知，可用于提出搜索问题；M7 没用这些结果
挑选候选或配置。新策略的 M5 矩阵表现是冻结后验证，不是完全未见 workload。

## 2. 选择结果与复核

| Bucket | `mu` | 最终配置 `(BM, BN, BK, warps, stages)` | 训练 p50 几何平均比 | 复核几何平均比 | 复核最差 case 比 |
| --- | ---: | --- | ---: | ---: | ---: |
| small | `<= 8` | `(32, 64, 32, 4, 3)` baseline | 1.001199 | 1.002859 | 1.005348 |
| medium | `8 < mu <= 64` | `(32, 256, 32, 4, 3)`，候选 6 | 0.883550 | 0.881542 | 0.938422 |
| large | `mu > 64` | `(64, 128, 32, 4, 3)`，候选 9 | 0.811979 | 0.812310 | 0.830794 |

small bucket 没有达到至少 5% 的候选收益要求，因此保留原配置。medium 和 large
候选均通过更长复核；相对各自 paired baseline 的复核 p50 几何平均比分别为
0.8815 和 0.8123。最终策略只适用于上述 E/H/I/top-k 域，域外回退到原配置。
例如 M5 的 `E=16, H=128, I=64, topk=4` small case 使用 shape fallback，并不
应用 M7 的训练配置。

全部 72 个候选记录均通过，没有候选编译或资源失败；medium gate/up 的静态资源
记录仍有 6 个 spill（见第 5 节），这不同于候选运行失败。6 个更长复核均通过，
输入 SHA 与对应训练 case 匹配，未发生 baseline correctness failure。策略冻结
SHA256 为
`85040044bb4e817b9ceab770145f723741324067d79a94fcf8d305390a1c90a7`。

## 3. 正确性、策略与运行来源

M7 所有实现都用完整 FP32 MoE reference 检查，原定阈值保持不变：
`relative_rms <= 0.008`、`normalized_max <= 0.008`，输出必须 finite。72 行
训练候选检查全部通过；训练记录中的最大 relative RMS 为 `0.003710064`，最大
normalized max 为 `0.005033907`。正式 holdout 包含 9 个 M4 functional case、
3 个不同策略配置的边界 case，以及 15 个性能 case；共 297 项 FP32 数值指标
通过。holdout 的最大 relative RMS 为 `0.005269567`，最大 normalized max 为
`0.007242441`，均低于原阈值。

对每个不同的冻结配置，另以 `T=31, E=3, topk=2, H=131, I=67` 的 duplicate-ID
非整齐尾块输入运行边界检查。每个配置都验证初始输出、两次原地更新后的 Graph
输出、一个非默认 current stream 输出，以及恢复原值后的 Graph 输出；3 个配置的
数值、dispatch/status 与 hash 恢复检查全部通过。15 个性能 Graph case 也分别在
固定地址下两次更新 X/IDs/routing weights，检查三种 backend 输出，并在计时前
恢复原始输入/路由 hash。计时后的 eager 与 Graph 输出再次对照 FP32 reference。

本轮完整前向比较：

- **Default ranked**：M6 的原始 7-launch 路径与原 tile 配置。
- **Frozen-policy ranked**：同一 7-launch 路径，按冻结策略选择配置。
- **vLLM public**：官方未修改 vLLM 0.10.0 public `fused_experts`，使用其自动配置。

项目路径的舍入边界保持 M4 约定：gate/up FP32 accumulate 后 BF16；SwiGLU
FP32 计算后 BF16；down FP32 accumulate 后 BF16；route weight 与 slot-order
reduce 在 FP32 完成后写 BF16。vLLM native SiLU 分支在乘 up 前先 BF16 舍入；
down 累加乘 route weight 后再转 BF16；top-k 2/3/4 使用 native BF16 顺序累加，
其他 top-k 使用 ATen `sum_out`。两种实现按相同完整 FP32 阈值分别判定，不要求
逐 bit 相同；M7 更换 tile 后也不以跨配置逐 bit 相同作为通过条件。

环境为 Python 3.12.14、PyTorch 2.7.1+cu128 / CUDA runtime 12.8、Triton 3.3.1、
NumPy 2.5.3，NVIDIA GeForce RTX 3080 Laptop GPU（SM86、16 GiB）。vLLM 来自
官方 0.10.0 wheel/tag commit `6d8d0a24c02bfd84d46b3016b865a44f048ae84b`，
wheel SHA256 与 1,237 个上游文件校验结果见
[vLLM provenance](../results/provenance/vllm-0.10.0-runtime.json)。运行使用
`/home/tmz/ai/baselines/vllm-0.10.0` operator-only overlay；这不是完整 serving
安装。15/15 个 case 的 `packaged_config_found` 均为 false，未覆盖 vLLM 默认
配置，也不声称存在精确适配该 GPU 的调优 JSON。

冻结策略在 holdout 前后均通过 provenance 检查。policy/train record 的 15 个
源文件 hash 一致；holdout JSON 另外记录 20 个明确列出的源码与实验计划 hash。
完整 Python/NumPy/PyTorch/Triton 路径、vLLM 自动配置、输入 hash、telemetry、
逐轮原始 event/wall 样本和错误数据保留在结果 JSON。

## 4. Holdout 计时与结果

性能矩阵沿用 M5 的 15 个固定输入：Qwen3-30B-A3B **形状**（E128/H2048/I768/
topk8）含 T=1/17/193/1024/2049、seed 73 和 991；T=193/1024 的 hot16 路由
使用 seed 73；small E16/H128/I64/topk4 含 T=1/17/193、seed 73。权重均为合成
BF16 值，不使用模型 checkpoint 或真实 router。

API 与 Graph 都使用三个 backend、六轮正/逆 cyclic 顺序、6 个样本/轮、每个样本
2 次完整 forward。每个 Graph replay 包含 8 次完整 forward，结果除以 8。准备、
编译、warmup、正确性检查、hash、dispatch/status 读回和 telemetry 不计入计时。
API 保留各自 public `.run()` 开销；vLLM 每次调用会分配上游临时 buffers/output，
项目两个 ranked wrapper 使用 prepare 时分配的 workspace。Graph 使用各自固定
地址和 allocator pool。FLOPs 按每 forward `6 × T × topk × H × I` 计算，只是
数学运算量估值，不是硬件计数器读数。

### 4.1 全部 Graph 结果

单位为 **ms / complete forward**，每格写作 `p50 / p95`。p95 保留观察到的尾部
变化，尤其 T2049 的幅度应结合原始样本阅读。

| 场景 | 原始 ranked | 冻结策略 ranked | vLLM public |
| --- | ---: | ---: | ---: |
| Qwen T1 random seed73 | 0.194400 / 0.195360 | 0.194240 / 0.195084 | 0.191424 / 0.192014 |
| Qwen T17 random seed73 | 1.896320 / 1.898303 | 1.897664 / 1.901606 | 1.872832 / 1.874832 |
| Qwen T193 random seed73 | 3.199421 / 3.202128 | 2.979392 / 2.984144 | 2.964029 / 2.967872 |
| Qwen T1024 random seed73 | 5.123104 / 5.129264 | 3.836665 / 3.893152 | 4.172992 / 4.206048 |
| Qwen T2049 random seed73 | 7.584509 / 7.762896 | 6.513244 / 6.712736 | 6.652480 / 6.798157 |
| Qwen T1 random seed991 | 0.194560 / 0.198638 | 0.194560 / 0.198288 | 0.191456 / 0.194064 |
| Qwen T17 random seed991 | 1.856736 / 1.859120 | 1.857056 / 1.860594 | 1.831612 / 1.832912 |
| Qwen T193 random seed991 | 3.199584 / 3.202768 | 2.975744 / 2.981192 | 2.963360 / 2.968464 |
| Qwen T1024 random seed991 | 5.127648 / 5.134847 | 3.866944 / 3.937199 | 4.143200 / 4.169392 |
| Qwen T2049 random seed991 | 7.629500 / 7.817824 | 6.627424 / 6.763328 | 6.733918 / 6.880224 |
| Qwen T193 hot16 seed73 | 0.735131 / 0.744352 | 0.718464 / 0.750556 | 0.686968 / 0.696896 |
| Qwen T1024 hot16 seed73 | 3.239296 / 3.245472 | 2.981600 / 3.169584 | 2.909692 / 3.015695 |
| Small T1 random seed73 | 0.011264 / 0.011344 | 0.011328 / 0.011392 | 0.009943 / 0.009984 |
| Small T17 random seed73 | 0.013440 / 0.013504 | 0.013376 / 0.013436 | 0.015040 / 0.015040 |
| Small T193 random seed73 | 0.014903 / 0.014912 | 0.014912 / 0.014924 | 0.023161 / 0.023168 |

T1024 random 下，medium 配置相对原始 ranked 的 Graph p50 分别降低 25.11%
（seed73）和 24.59%（seed991）；相对 vLLM p50 分别低 8.06% 和 6.67%。T2049
large 配置相对原始 ranked 降低 14.12% 和 13.13%；相对 vLLM 只领先 2.09% 和
1.58%，差距较小，应结合 p95 和原始样本谨慎解读。T193 random medium 配置虽较
原始 ranked 更快，但 vLLM p50 仍快约 0.42–0.52%；两个 hot16 case 的 vLLM
p50 也领先冻结策略约 2.47–4.58%。因此 M7 显示了特定 random 大批量形状的配置
收益，但没有显示完整算子普遍胜过 vLLM。

### 4.2 三个 small case 的 API 结果

单位为 **µs / complete forward**，每格为 `p50 / p95`。这三个 E16 case 超出策略
的 E/H/I/top-k 域，冻结策略使用 baseline fallback；下表不能解释为 M7 tile 搜索
带来的 small-case 加速。

| 场景 | 原始 ranked | 冻结策略 ranked | vLLM public |
| --- | ---: | ---: | ---: |
| Small T1 random seed73 | 33.720 / 39.168 | 33.280 / 38.400 | 179.200 / 186.688 |
| Small T17 random seed73 | 34.232 / 43.680 | 33.792 / 38.772 | 182.520 / 198.560 |
| Small T193 random seed73 | 34.784 / 40.192 | 34.816 / 39.168 | 183.304 / 196.220 |

冻结策略与原始 ranked 使用相同 baseline 配置；三行 p50 的微小正负变化处于这组
测量的范围内，不作为调参收益。vLLM public API 的 small API p50 较高，而其
Graph结果依赖固定 graph pool，不能混为同一种调用开销或泛化成整体后端差距。
Qwen各case的API全样本仍在holdout JSON内。

## 5. 编译证据、资源与显存

编译证据对小/中/大三个 frozen config 各编译 gate/up+SwiGLU 与 down，并保存
PTX、cubin、SASS/哈希。六个kernel的 PTX 均含 MMA 指令，SASS 反汇编均含 HMMA；
这是编译产物中的 Tensor Core 指令静态证据，不是执行时利用率、occupancy 或
吞吐测量。

| Bucket / 配置 | Kernel | PTX MMA数 | SASS HMMA数 | Registers | Spills | Shared memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| small `(32,64,32,4,3)` | gate/up+SwiGLU | 16 | 16 | 76 | 0 | 20,480 B |
| small | down | 8 | 192 | 55 | 0 | 12,288 B |
| medium `(32,256,32,4,3)` | gate/up+SwiGLU | 64 | 64 | 255 | 6 | 69,632 B |
| medium | down | 32 | 32 | 168 | 0 | 36,864 B |
| large `(64,128,32,4,3)` | gate/up+SwiGLU | 64 | 64 | 255 | 0 | 40,960 B |
| large | down | 32 | 32 | 159 | 0 | 24,576 B |

medium gate/up 的 6 个 spill 和 69,632 B shared memory 是后续调查的依据，但不能
仅据此推断 occupancy 或将性能变化归因于 spill/shared-memory。large gate/up 为
255 registers、0 spills。所有配置都保持 7 次 ranked launch；各 M7 recheck 记录
的 workspace 为 small 6,559,752 B、medium 52,463,624 B、large 419,694,600 B，
这是对应 T=64/512/4096 训练输入上的 workspace 记录，不代表 holdout 所有 T 的
常量分配。

训练 case peak allocated 范围为 1,233,800,192–2,265,848,832 B（约 1.15–2.11
GiB）；holdout 性能 case 为 9,397,760–1,947,444,736 B（约 9.0 MiB–1.81 GiB）。
这些 peak 包含各自输入/权重、workspace、oracle temporaries 和 CUDA Graph pool，
不是纯 kernel workspace 或部署显存预算。

静态产物保存在 ignored 本地目录
`artifacts/m7-compile-evidence-20261006T105911Z/`；哈希、寄存器、shared memory
和指令计数汇总见[编译证据 JSON](../results/m7-compile-evidence-20261006T105911Z.json)。

## 6. 判断与下一步

本轮采用“全训练矩阵资格 + bucket 复核 + 冻结后 holdout”的选择流程，是因为
单一 T 或单一路由的胜出不足以支撑形状策略。medium/large 在训练 recheck 上
达到预设相对收益与 worst-case 条件，并在 M5 候选 holdout 上继续降低原始 ranked
的 p50；但 T193/hot16 仍落后于成熟 vLLM，T2049 相对 vLLM 的差值也较小。small
保留默认是有意结果，不为追求全局 headline 而放松 5% 选择门槛。

下一步应围绕 medium gate/up 的静态 spill 与大 T 的完整前向构成提出新的、预注册
候选实验，再分别确认 dispatch 和 GEMM 的影响。不要用当前 holdout 重选本策略；
任何新的选择都应另设训练与冻结边界，并保留完整 M7 配置的回归对照。当前证据
不能证明真实模型前向、serving latency、跨 GPU 性能或成熟 MoE backend 的通用
胜负，也没有 profiler/hardware-counter utilization 数据。

复现实验与审计文件：

- [M7 training JSON](../results/m7-full-moe-tuning-20261006T103809Z-241373177.json)
- [Frozen M7 policy](../results/m7-policy.json)
- [M7 holdout JSON](../results/m7-full-moe-holdout-20261006T104838Z-727559245.json)
- [M7 compile evidence JSON](../results/m7-compile-evidence-20261006T105911Z.json)
- [M5 baseline report](M5_MATURE_MOE_REPORT.md) and [M5 baseline JSON](../results/m5-mature-moe-20261006T101645Z-942093289.json)
- [Official vLLM operator provenance](../results/provenance/vllm-0.10.0-runtime.json)
