# M4：GPU 路由的完整 MoE 基线与局部融合（2026-10-06）

状态：完整 GPU fixed-route forward 已实现；功能验证、动态值图重放和本项目
unfused/fused完整forward对照通过。尚未接入成熟完整MoE backend。
本阶段有端到端算子链收益，但不属于NanoKV端到端推理、真实模型或serving结果。

## 1. 从预分组 GEMM 变成完整 GPU 路径

`prepare_moe(X, gate_up, down, IDs, routing_weights)` 只绑定shape/buffers并分配workspace。
输入BF16，IDs int32/int64 CUDA，routing weights FP32；同地址的values/IDs可更换。
`run` 用GPU上当前IDs重新完成下面9个kernel：

1. reset counts/status。
2. histogram counts，并记录非法ID/非finite权重status。
3. GPU exclusive prefix rows/task offsets，清cursor。
4. atomic cursor生成原route -> packed row mapping。
5. 按mapping复制X到grouped input。
6. dynamic grouped gate/up GEMM。
7. SwiGLU。
8. dynamic grouped down GEMM。
9. 按原token/slot顺序的weighted combine。

不把counts读回CPU。GPU任务offsets把每个CTA映射到expert；固定grid上界为
`ceil(T*topk/BM)+E-1` 个M task，超出实际task的CTA退出。
不按最大expert容量填充矩阵，中间行数固定为T*topk；空expert跳过。
重复IDs作为不同route独立计算，combine按原slot收集，不依赖atomic dispatch顺序。

数学舍入边界明确：gate/up FP32 accumulate→BF16；SwiGLU FP32→BF16；
down FP32 accumulate→BF16；FP32 weighted reduce→BF16 output。
无bias、softmax/权重重归一化、learned router、通信、backward。
支持本prototype E<=256、topk<=32及连续同device buffers；实测topk1/2/4/8。
ID范围/finite是caller契约。GPUstatus对非法值置标记，显式`check_inputs()`同步
读取并报错；这个debug操作不在热路径或图内。

## 2. 怎么建立数值基线

- 保留独立完整FP32 expert oracle，又用PyTorch逐expert的BF16阶段oracle
  区分中间舍入与实现错误。二者都只检查正确性，不计作性能backend。
- 9个功能输入包括T0、空expert、重复/全热点/随机IDs、H/I/M尾块、
  int32/int64 IDs、非power-of-two E、单expert及负routing weights。
- 独立CPU histogram/offsets与GPU结果核对；mapping是route行的排列且落在
  正确expert段，逆向gather的grouped输入与原token相同。诊断readback不计时。
- 2种非法runtime值分别正确触发status1/2与debug异常。
- 12个性能场景各图捕获后做3次同地址values/IDs/权重更新；功能图另做3次更新，
  总39组图更新检查。捕获图中的reset、grouping和GEMM共同处理新的路由。
- 验证后恢复原seed/pattern/values并核对SHA再计时，保证eager/Graph同输入。
- 所有FP32对照最大relative RMS=0.004240、normalized max=0.007243，
  保持预定阈值0.008。逐阶段BF16 oracle最大RMS=0.000734、max=0.005814。

正式记录：[m4-moe-20261006T094645Z-376040740.json](../results/m4-moe-20261006T094645Z-376040740.json)。
源码hash已核对。显存峰值3.41GiB含oracle、replacement buffers与graphs，非纯workspace。

## 3. 完整 forward 功能基线（ms / forward，seed73）

| 场景 | Eager CUDA-event | Graph |
| --- | ---: | ---: |
| small_T17_random | 0.144608 | 0.014976 |
| small_T17_hot | 0.138080 | 0.014976 |
| small_T193_random | 0.159648 | 0.017280 |
| small_T193_hot | 0.140352 | 0.017920 |
| qwen3_30b_a3b_shape_T17_random | 2.003968 | 1.914980 |
| qwen3_30b_a3b_shape_T17_hot | 0.477184 | 0.387968 |
| qwen3_30b_a3b_shape_T193_random | 3.529952 | 3.362816 |
| qwen3_30b_a3b_shape_T193_hot | 0.820224 | 0.802304 |

small graph约15–18us，而eager约138–160us，提示多次Python/JIT提交占比高。
Qwen T193随机下，单阶段graph诊断 gate/up约2.21ms、down约1.11ms，
dispatch约25us、SwiGLU约17us、combine约18us。
small dispatch约7us，和整体约15us同量级；小矩阵下一步优先看启动/分组组织。
这些阶段独立重放改变了缓存/中间状态，不能严格相加还原完整forward时间。

Qwen形状取官方Qwen3-30B-A3B配置H2048/I768/E128/topk8；使用随机权重，
不是本地NanoKV Qwen3-4B模型的实测。热点使用16个随机expert。
完整forward包含分组，与M3预分组gate/up的数字不同，不能混写。

## 4. Gate/up + SwiGLU 融合

固定原tile32/64/32/4warps/3stages。单CTA有两个FP32 accumulator，计算同一
A的gate/up权重，各自BF16 round-trip后在CTA内做SwiGLU，直接写activation。
保留未融合的所有中间舍入边界；不是切换精度获得更快结果。
GPU dispatch、down、combine相同，完整链路从9个kernel变为8个。

消除projected数组一次写与一次读，逻辑流量共8*R*I bytes；T1024/topk8/I768
对应48MiB。两个投影还共享同CTA的A tile，改变了CTA数与资源使用。
因此实际收益同时可能来自输入复用、调度和流量；没有硬件计数器，不能把收益
全部归因于少一次启动，也不能把逻辑流量称为实测DRAM流量。

wrapper复用base，base.projected仍分配：workspace bytes两条路径完全相同。
Qwen fused编译资源76 registers、0 spills、20KiB shared，含BF16 HMMA；PTX
仍有BF16 rounding转换。保存完整PTX/cubin/SASS/hash，无occupancy/utilization测量。

预定8个场景，固定seed73、不调参或删慢case。4轮AB/BA/AB/BA，6样本/轮，
2调用/样本；Graph含8次完整forward。9个功能case的eager/更新图均通过；
8个性能case eager与Graph中fused/unfused输出均逐bit一致。

| 完整链路 A/B | Unfused Graph | Fused Graph | 耗时减少 |
| --- | ---: | ---: | ---: |
| T17, E16, H128/I64, random | 0.014656 | 0.014272 | 2.62% |
| T193, E16, H128/I64, random | 0.017088 | 0.015872 | 7.12% |
| T17, E128, H2048/I768, random | 1.916284 | 1.898208 | 0.94% |
| T17, E128, H2048/I768, hot | 0.387264 | 0.374517 | 3.29% |
| T193, E128, H2048/I768, random | 3.366464 | 3.198368 | 4.99% |
| T193, E128, H2048/I768, hot | 0.804448 | 0.732960 | 8.89% |
| T1024, E128, H2048/I768, random | 6.158560 | 5.104224 | 17.12% |
| T1024, E128, H2048/I768, hot | 3.784636 | 3.266624 | 13.69% |

T1024随机约少17%，热点约少14%；T17随机仅约1%，不能写成普遍17%收益。
T1024热点p95为unfused4.3333ms/fused3.4724ms，波动明显，保存全部样本。
融合前后相同数学/输入/固定tile，但只有本项目两实现对照，没有完整成熟MoE比较。

预先设计：[M4_FUSION_EXPERIMENT_PLAN](M4_FUSION_EXPERIMENT_PLAN.md)。
正式记录：[m4-fusion-20261006T095043Z.json](../results/m4-fusion-20261006T095043Z.json)。
源码hash已核对。基线GPU温度39→51°C，融合43→59°C，结束SM1695MHz；
笔记本未锁频/功耗、热输入反复复用，不能当作serving延迟或其他GPU保证。

## 5. 使用与下一步

```python
from llm_gpu_kernels.moe import prepare_moe
from llm_gpu_kernels.moe_fused import prepare_fused_moe
prepared = prepare_fused_moe(x, gate_up, down, ids, routing_weights)
out = prepared.run()
prepared.check_inputs()  # 可选同步debug；不要放在计时/Graph路径
```

```bash
source scripts/activate.sh
python -m llm_gpu_kernels.moe_benchmark
python -m llm_gpu_kernels.moe_fusion_benchmark
```

下一步重点：接入成熟完整MoE对照；减少small的分组与提交成本；取得profile证据后
再优化动态GEMM配置和down/combine组织。Attention仍属于后续独立阶段。
