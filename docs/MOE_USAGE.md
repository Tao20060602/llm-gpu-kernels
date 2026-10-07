# 完整 MoE prepared 接口使用指南

本文说明项目里的 GPU fixed-route MoE 完整前向如何准备、复用和检查。这里的
“完整”包含 GPU 上的 route 分组/dispatch、gate/up 与 SwiGLU、down projection
和 combine；输入已经带有 router 产生的 expert IDs 与 routing weights，因此不包含
router projection、softmax/top-k、attention、tokenization、模型调度或通信。

## 1. 输入张量契约

prepared 路径沿用 `prepare_moe` 的输入检查。令 `T` 为 token 数、`E` 为 expert 数、
`H` 为 hidden size、`I` 为 intermediate size、`K` 为 top-k：

| 参数 | 形状 | dtype / 布局 |
| --- | --- | --- |
| `x` | `[T, H]` | CUDA BF16、连续 |
| `gate_up` | `[E, 2I, H]` | CUDA BF16、连续；前 `I` 行是 gate，后 `I` 行是 up |
| `down` | `[E, H, I]` | CUDA BF16、连续 |
| `ids` | `[T, K]` | CUDA int32 或 int64、连续 |
| `routing_weights` | `[T, K]` | CUDA FP32、连续 |

五个 tensor 必须在同一 CUDA device 上。metadata 检查要求 `1 <= E <= 256`、
`H > 0`、`I > 0`、`1 <= K <= 32`、`x.shape[1] == H`，且 `ids` 与
`routing_weights` 的形状完全相同。`T=0` 可表示空输入；空输入返回空的 BF16
`[0,H]` 输出，不启动 MoE kernel。

另外，当前指针/向量索引实现要求 `R=T*K < 2^31`，并要求
`R * max(H, 2I) < 2^31`。实际可用的最大形状还受设备显存和编译资源限制。

`prepare_moe` 不会在 host 上读取 IDs 或 routing weights 来预先验证其数值。每次
forward 会在 GPU 上检查 ID 范围 `[0,E)` 和 routing weight 有限性，并更新 status。
IDs 不要求唯一：同一个 token 的多个 slot 可以指向同一 expert，各 slot 独立参与
计算和按 slot 顺序累加。routing weights 可为任意有限 FP32 数，包括负数；算子不会
重新归一化或限制它们的行和。为了得到有意义的数值输出，调用者仍应保证 `x`、
expert weights 与 routing weights 有限；GPU status 不检查 `x` 或权重矩阵中的 NaN/Inf。

M20 的 Qwen3 trace 是较窄的真实输入实例：top-k IDs 唯一，route weights 非负，且是官方
router BF16 输出精确晋升成 FP32后不再归一化。它们不是 prepared API 的额外限制；
通用接口继续支持重复 IDs 和有限 signed weights。详见 [M20 trace 契约](M20_REAL_MOE_TRACE_CONTRACT.md)。

## 2. 默认 ranked 与冻结策略入口

一般使用 M6 ranked 路径；需要沿用 M7 冻结的 shape-only 配置时，使用
`prepare_configured_moe`。两个入口都返回可复用的 prepared 对象：

```python
from pathlib import Path

import torch

from llm_gpu_kernels.moe_full_policy import FrozenMoEPolicy
from llm_gpu_kernels.moe_ranked import prepare_ranked_moe_launcher
from llm_gpu_kernels.moe_configured import prepare_configured_moe

# 假设五个 tensor 已按上一节的 dtype、shape、布局和 device 契约准备好。
device = x.device

# M6 默认 ranked 配置：完整路径为 7 个 GPU launches。
ranked = prepare_ranked_moe_launcher(x, gate_up, down, ids, routing_weights)
out = ranked.run()                  # BF16 [T, H]，由 prepared 对象持有
ranked.check_inputs()               # 显式同步诊断；不要放进计时或 CUDA Graph

# M7 frozen policy：先核验策略及其训练记录/源码 provenance，再按 shape 选择。
repo_root = Path.cwd().resolve()    # 从仓库根目录运行示例
policy = FrozenMoEPolicy.load(repo_root / "results/m7-policy.json")
policy_provenance = policy.verify_provenance(repo_root)
T, H = x.shape
E, _, I = down.shape
K = ids.shape[1]
config, bucket = policy.select(T, E, K, H, I)
configured = prepare_configured_moe(
    x, gate_up, down, ids, routing_weights, config
)
out = configured.run()
configured.check_inputs()
print(bucket, config)
```

策略以 `mu = T*K/E` 和 `(E,H,I,K)` 选择配置，不查看 IDs、route counts 或
实际 expert 热度。它的已训练域是 `E=128, H=2048, I=768, K=8`；域外调用返回
记录在策略里的 baseline fallback。若使用冻结策略做结果，应保留选择出的 bucket、
config、策略文件 hash 和 `verify_provenance` 返回值。`prepare_configured_moe` 的
`config` 参数必须是 `GemmConfig`。策略应在准备阶段选定；固定图的更新只改 tensor
内容，不改变 shape，因此不应在热路径中重新选策略。

`prepare_ranked_moe_launcher` 和 `prepare_configured_moe` 都使用固定地址的缓存
launchers。对 `T>0`，ranked 完整路径是七次 launch：reset、rank histogram、prefix、
rank dispatch、融合 gate/up+SwiGLU、down、combine。热路径在 GPU 上重建 counts 和
route mapping，不做 CPU counts readback，也不为每次 forward 创建 tensor。底层
`prepare_moe` / fused M4 和 cached M4 launcher 仍可用于既有基线；M6/M7 的 prepared
ranked 入口适用于这里说明的固定地址复用方式。cached launcher 当前绑定 Triton 3.3.1，
其他 Triton 版本会被入口拒绝；使用前确认项目的固定 runtime。

## 3. 生命周期、workspace 与 status

准备入口会分配 workspace、编译需要的 kernel，并执行用于编译/准备的前向。因此：

- 把准备、编译、首次执行和正确性检查放在性能计时之外；这些结果不能当作冷启动时间。
- 准备后保留 prepared 对象及其绑定的输入/权重 tensor。对象和 tensor 在所有异步
  launches 完成前都要保持存活。
- 同一 prepared workspace 不可被并发 forward、并发 graph replay 或另一个会写入它的
  调用共享。要重叠执行，给每个并发调用准备独立的输入和 workspace。
- 每次 `run()` 都覆盖对象内部的 BF16 输出和 workspace。需要保留多次结果时，在下一次
  forward 前把输出复制到调用者自己的 tensor。

prepared launchers 缓存 tensor 的地址和 shape。允许在原 tensor 上用 `copy_` 更新 `x`、
`ids` 和 `routing_weights` 的值；不要替换 tensor 对象、改变 shape/stride/device、resize
或搬到另一个地址。权重同样保持 shape、device 和地址不变。ID 的值可以改，更新后仍须
处于 `[0,E)`。若 `T` 或 `K` 改变，需重新准备 workspace、launcher 和 graph。

GPU status 会在 forward 开头清零，然后由 router-independent GPU 路径更新：bit 0（数值
`1`）表示发现越界 ID，bit 1（数值 `2`）表示发现非有限 routing weight；两者可同时置位。
发现非法数据时 kernel 会尝试避免越界地址访问，但其输出不是可用的 MoE 结果。应在该次
forward 后、下一次 forward 重置 status 前调用：

```python
out = configured.run()
configured.check_inputs()  # status.item() 会把 GPU 状态同步读回 host
```

有任意 status bit 时，`check_inputs()` 抛出 `ValueError` 并显示 status 值。该调用是显式
debug/validation 操作，会同步 GPU；不得放入性能计时或 graph capture。不要在一张包含多次
forward 的图结束后，试图用最终 status 诊断更早的一次非法 forward：下一次 forward 会重置
它。调试非法输入时，单独执行一次 `run()`，立刻同步并检查 status。初次 prepare 也会执行
准备前向，故准备输入也应满足 runtime 值契约。

## 4. CUDA Graph 与 stream 顺序

Graph 捕获的是固定 tensor 地址和固定 shape 下的 kernel 序列。捕获前先准备并预热，再
在所选 capture stream 上捕获；不要把 `check_inputs()`、`.item()`、CPU readback 或 oracle
放进捕获区域：

```python
for _ in range(3):
    configured.run()
torch.cuda.synchronize(device)

capture_stream = torch.cuda.Stream(device=device)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=capture_stream):
    for _ in range(8):
        graph_output = configured.run()

graph.replay()  # 在当前 replay stream 上排队；out buffer 地址固定
torch.cuda.synchronize(device)
configured.check_inputs()
```

改变输入时只原地写入捕获时绑定的 tensor，再重放 graph。若写入发生在另一个 stream，
必须建立“写入完成 → graph replay”的依赖；若另一个 stream 随后读取 graph 输出，也要
建立“replay 完成 → 读取”的依赖。下面的 event 示例把更新放在 producer stream，把 replay
放在另一条 stream：

```python
main_stream = torch.cuda.current_stream(device)
producer = torch.cuda.Stream(device=device)
replay_stream = torch.cuda.Stream(device=device)
ready = torch.cuda.Event()
done = torch.cuda.Event()

producer.wait_stream(main_stream)  # 等待此前在 main_stream 上准备好的源数据
with torch.cuda.stream(producer):
    x.copy_(x_next)
    ids.copy_(ids_next)
    routing_weights.copy_(weights_next)
    ready.record()

replay_stream.wait_event(ready)
with torch.cuda.stream(replay_stream):
    graph.replay()
    done.record()

main_stream.wait_event(done)       # main_stream 之后可安全消费 graph_output
torch.cuda.synchronize(device)
configured.check_inputs()
```

跨 stream 同步不是由 prepared 对象自动推断的。也不要在某个 stream 还在读取输入或写
workspace 时，于另一条 stream 改这些 buffer 或再次调用同一 prepared 对象。M6/M7 已有的
功能检查覆盖固定地址 graph 更新与显式 stream 依赖；实际使用仍须按应用的 producer/consumer
stream 建立同等依赖。

## 5. 验证和性能结果的适用范围

建议在集成或计时前，对每个准备对象和代表性输入执行独立 FP32 MoE reference、finite、
dispatch/status 验证；数值参考入口为 `llm_gpu_kernels.reference.moe_reference`。现有
M6/M7 实验使用固定阈值 `relative_rms <= 0.008`、`normalized_max <= 0.008` 且输出有限；
自定义 tile 可能改变累加分组，不要求不同配置逐 bit 相同。

- [M6 报告](M6_RANKED_DISPATCH_REPORT.md)记录 7-launch ranked 路径及其既定合成输入、
  graph、status、duplicate-ID 和 stream 验证；七次启动是路径结构，不单独证明加速。
- [M7 报告](M7_FULL_MOE_OPTIMIZATION_REPORT.md)说明冻结策略的训练/复核边界。它基于
  合成 fixed-route 输入选择，域外回退 baseline；不能把特定 bucket 的测量外推到所有
  token 长度、route 分布或设备。
- [M20 计划](M20_REAL_MOE_INPUT_PLAN.md)、[trace 契约](M20_REAL_MOE_TRACE_CONTRACT.md)
  和[报告](M20_REAL_MOE_REPORT.md)描述 pinned Qwen3 layer-0 真实 trace 的额外来源与范围。
  它属于真实固定路由算子验证，不等同于完整模型推理质量、router/attention/数据搬运成本、
  serving 延迟或生产吞吐；本指南不从该实验推导未记录的性能数字。

所有延迟结论都要绑定具体 GPU、软件版本、输入/路由、配置、测量边界和原始记录。API 与
CUDA Graph 是不同调用模式；Graph 结果不代表 eager/serving 的端到端成本。不要把准备/编译
之外的固定路由算子时间说成完整 MoE 子层或整模型时间，也不要仅凭 kernel launch 数或静态
Tensor Core 指令证据声称整体性能收益。
