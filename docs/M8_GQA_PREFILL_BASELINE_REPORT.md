# M8 四段 KV GQA Prefill：FA2 基线结果

正式基线于 2026-10-06 通过。实验使用未修改的官方 FlashAttention 2.8.3.post1
公开算子，对固定形状的四段 KV GQA prefill 进行计时；不运行 NanoKV、完整模型或
serving 栈。

## 结果摘要

- 6 个功能场景、1 个固定地址 graph 更新/非默认 stream 场景，以及 12 个性能场景
  全部通过。
- 小形状 FP32 oracle 覆盖范围内，最大 relative RMS 为 0.002126，最大 normalized
  max 为 0.002961，低于 0.008 限值。各功能场景中 full-chain 与 prepacked FA2
  输出一致。
- 12 个性能场景的 full-chain 与 prepacked FA2 API/Graph arm 一致；大形状结果属于
  同一 FA2 算子的 arm-consistency 检查，不是独立数值 oracle 结果。
- Graph p50 的 full-chain 与 prepacked FA2 直接差异按 full-chain p50 归一化后，
  T=64 为约 29–36%，T=256 为约 14–16%，T=1024 约 5%，T=4096 约 2%。这些值描述
  本实验测到的两个执行路径差异，不等于可直接兑现的自定义 kernel 收益。

正式记录：[m8-gqa-baseline-20261006T105816Z-903442275.json](../results/m8-gqa-baseline-20261006T105816Z-903442275.json)。

## 固定输入与测试路径

查询张量为 `Q[T,32,128]`，四段 K/V 按 `[selected history, sink, previous recent,
current]` 顺序连接，每段形状为 `[L,8,128]`，全部为同设备、连续 BF16 tensor。
GQA 映射为 `kv_head = q_head // 4`；softmax scale 为 `1/sqrt(128)`；current 段采用
因果 mask，历史三段对所有 query 可见。输入由 seed 819 生成，T 取
64/256/1024/4096，selected-history 长度取 0/512/2048；sink 固定为 64，previous
recent 固定为 512，current 长度等于 T。

性能比较有三种独立测量：

1. **Full-chain Graph：** 把四段 K/V pack 到可复用预分配缓冲区，再调用 FA2
   公开算子。
2. **Prepacked FA2 Graph：** 用相同 packed K/V 调用同一 FA2 公开算子，pack 不计入
   该路径。
3. **Pack-only API：** 只测可复用缓冲区中的 K/V 拼接，不调用 attention。

Pack-only 是独立诊断；没有从 full-chain 时间中扣除 pack-only 时间。Graph p50
差异的表内比例定义为 `(full-chain p50 − prepacked p50) / full-chain p50`。
它包含本实验两个 Graph 路径的差异，不能当作纯 pack kernel 时间或实测 DRAM
流量。

API 和 Graph 每种对照路径使用 4 轮 `AB / BA / BA / AB`，每轮 8 个样本、每个样本
调用 2 次。Graph 每次 replay 含 8 次完整 forward，每个样本测 2 次 replay，并按
两次 replay 和 8 次 forward 归一化。表中 p50/p95 是 CUDA-event 毫秒/完整 forward；
同步 wall 样本也完整保留在 JSON。

## 性能数据

`selected history` 为所选历史段长度，`packed K/V` 为四段连接后的总长度。Pack-only
仅有 API 测量；full-chain 和 prepacked FA2 列为 Graph 测量。

### p50（毫秒/完整 forward）

| T | Selected history | Packed K/V | Full-chain Graph | Prepacked FA2 Graph | Graph 差异 / full-chain | Pack-only API |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0 | 640 | 0.060953 | 0.043289 | 29.0% | 0.024488 |
| 64 | 512 | 1152 | 0.085632 | 0.056384 | 34.2% | 0.035840 |
| 64 | 2048 | 2688 | 0.158528 | 0.101114 | 36.2% | 0.067584 |
| 256 | 0 | 832 | 0.132284 | 0.111455 | 15.7% | 0.026624 |
| 256 | 512 | 1344 | 0.205056 | 0.174848 | 14.7% | 0.037824 |
| 256 | 2048 | 2880 | 0.435484 | 0.372347 | 14.5% | 0.070144 |
| 1024 | 0 | 1600 | 0.619804 | 0.585056 | 5.6% | 0.046592 |
| 1024 | 512 | 2112 | 0.875648 | 0.827712 | 5.5% | 0.055808 |
| 1024 | 2048 | 3648 | 1.629041 | 1.548160 | 5.0% | 0.088064 |
| 4096 | 0 | 4672 | 5.125535 | 5.006873 | 2.3% | 0.115168 |
| 4096 | 512 | 5184 | 6.112544 | 5.997760 | 1.9% | 0.126464 |
| 4096 | 2048 | 6720 | 8.978016 | 8.823360 | 1.7% | 0.156120 |

### p95（毫秒/完整 forward）

| T | Selected history | Full-chain Graph | Prepacked FA2 Graph | Pack-only API |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0 | 0.061424 | 0.043595 | 0.025001 |
| 64 | 512 | 0.085884 | 0.056512 | 0.039011 |
| 64 | 2048 | 0.165440 | 0.104192 | 0.068838 |
| 256 | 0 | 0.132465 | 0.111738 | 0.030618 |
| 256 | 512 | 0.207069 | 0.175868 | 0.038630 |
| 256 | 2048 | 0.443553 | 0.373019 | 0.071398 |
| 1024 | 0 | 0.626995 | 0.587917 | 0.047313 |
| 1024 | 512 | 0.882819 | 0.831298 | 0.057062 |
| 1024 | 2048 | 1.634998 | 1.553347 | 0.089037 |
| 4096 | 0 | 5.151856 | 5.040019 | 0.118118 |
| 4096 | 512 | 6.127753 | 6.011940 | 0.129705 |
| 4096 | 2048 | 8.996780 | 8.857795 | 0.156902 |

## 正确性与运行环境

6 个功能场景覆盖空历史、非对齐段尾、极端 logits、GQA head 映射、T=0，以及
T=65/history=91 的小 FP32 oracle 边界。独立 oracle 仅用于 T≤65 且 history≤91；
不把它的覆盖范围外推到大形状。单独的 T=9 graph-update 场景在非默认 stream 更新
Q 和全部 K/V 段，检查更新后与恢复后的 FP32 oracle 结果及输入 hash。

运行环境为 Python 3.12.14、PyTorch 2.7.1+cu128、CUDA 12.8、NVIDIA GeForce RTX
3080 Laptop GPU（SM 8.6）。运行时从隔离 overlay 导入 FA2，发行版和模块版本均为
`2.8.3.post1`。Provenance 保存了官方 wheel 摘要、实际导入路径及 86 个逐文件匹配
结果；本次正式记录的 3 项源文件哈希在结束时复核一致。

官方 wheel build 字段 `2.8.3.post1+cu12torch2.7cxx11abitrue` 与发行版版本
`2.8.3.post1` 分开记录。较早的启动检查曾把两者错误地当作同一个版本字段；当前
运行时检查使用官方 METADATA/模块版本，并保留 build 字段作 wheel 来源标识。

此前的一份失败记录仍保留：[m8-gqa-baseline-20261006T105740Z-465462228.json](../results/m8-gqa-baseline-20261006T105740Z-465462228.json)。该次问题来自 graph capture
误用默认 stream；正式运行已改为 side stream 捕获。此失败不涉及数值计算失败，也
没有放宽阈值或改变预定输入。

FA2 wheel/runtime 的完整来源见
[环境说明](FLASH_ATTENTION_OPERATOR_ENVIRONMENT.md) 和
[provenance JSON](../results/provenance/flash-attention-2.8.3.post1-runtime.json)。
实验定义见 [M8 固定实验计划](M8_GQA_PREFILL_BASELINE_PLAN.md)。

## 结论范围与后续方向

结果建立的是指定分段长度和硬件上的 FA2 packed baseline。它不包含 selected-history
CPU gather、H2D、NanoKV store/update、模型执行或 serving；也没有实现或验证自定义
attention kernel。不能把 full-chain 与 prepacked FA2 的差异直接宣称为自定义 kernel
收益，更不能将 pack-only API 时间从 full-chain 中相减作为“纯 attention”延迟。

T=64/256 下 full-chain 与 prepacked Graph 的测量差距较大，支持进一步研究四段 KV
直接读取并复用 GQA 的候选路径；这只是由本次测量支持的后续方向，不是该路径已经
实现或获得加速的证据。
