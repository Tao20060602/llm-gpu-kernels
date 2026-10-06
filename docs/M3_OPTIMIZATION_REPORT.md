# M3：Prepared 入口与有限参数优化（2026-10-06）

状态：40配置训练扫描、训练复测、冻结规则、14个 holdout 场景、12组边界检查均完成。
训练和正式 holdout 的源码 SHA256 已逐项核对。不是完整 MoE 层性能结果。

## 做了什么、为什么

1. 原公开入口每次做 shape/alias 检查、JIT 查找和 launcher 参数转换。
   新 `prepare_grouped_gemm` 绑定 buffers/metadata，缓存已编译的 Triton3.3.1
   launcher 和指针。`run` 仍取当前 CUDA stream；输入 values 可变，但地址、
   shape 和 metadata 必须保持不变。此接口契约更窄，版本固定，不把它称为
   无条件兼容的原 API 加速。
2. 数学算法仍是原始 `tl.dot` tiled grouped GEMM，仅扫 BM/BN/BK/warp/stage。
   5个训练输入，每个40配置；195条通过、5条资源失败完整保留。
   一个超共享内存配置在所有训练场景失败。没有因慢或 spill 删除候选。
3. 训练规则按活跃 expert 平均行数分 <=8、(8,64]、>64 三组。
   只应用 N1536/K2048；其他形状明确回退原始配置。
   快速扫描后只在训练集做较长复测，至少改善5%才选入。
   small 保留原始32/64/32/4/3；medium选32/256/32/8/3；
   large选64/256/64/4/2。后两组训练复测几何平均耗时比分别0.8065、0.6728。
4. 冻结 JSON/hash 后才运行 holdout：T17/193/1024/2049、两seed随机唯一topk；
   热点改用随机16 expert，训练热点为固定8 expert；加down形状和small场景。
   根据 holdout 没有重新调参。

预先设计：[M3_EXPERIMENT_PLAN](M3_EXPERIMENT_PLAN.md)。
冻结规则：[m3-policy.json](../results/m3-policy.json)。
训练原始记录：[m3-scan-20261006T093119Z.json](../results/m3-scan-20261006T093119Z.json)。

## Graph p50（ms / GEMM）

| Holdout | 原始 prepared | 选择配置 prepared | CUTLASS 固定配置 | 相对原始耗时减少 |
| --- | ---: | ---: | ---: | ---: |
| T17 random, seed73 | 1.290624 | 1.291104 | 1.261984 | -0.0% |
| T193 random, seed73 | 2.213531 | 1.976793 | 1.911230 | 10.7% |
| T1024 random, seed73 | 3.886876 | 2.633404 | 2.574496 | 32.2% |
| T2049 random, seed73 | 5.828863 | 3.882587 | 4.315423 | 33.4% |
| T17 random, seed991 | 1.232762 | 1.232448 | 1.217821 | 0.0% |
| T193 random, seed991 | 2.218624 | 1.977248 | 1.917216 | 10.9% |
| T1024 random, seed991 | 3.971392 | 2.643584 | 2.673851 | 33.4% |
| T2049 random, seed991 | 5.835520 | 3.908384 | 4.365056 | 33.0% |
| T193 hot, seed73 | 0.493024 | 0.465119 | 0.407136 | 5.7% |
| T1024 hot, seed73 | 2.326880 | 1.640955 | 1.736029 | 29.5% |
| N2048/K768, T193 | 1.102749 | 1.107328 | 0.977600 | -0.4% |
| N2048/K768, T1024 | 1.994080 | 1.994784 | 1.382720 | -0.0% |
| N256/K128, T1 | 0.002496 | 0.002496 | 0.005696 | 0.0% |
| N256/K128, T64 | 0.002618 | 0.002560 | 0.005760 | 2.2% |

T193随机收益约11%，T1024和T2049随机约32–33%；T2049的选择配置比本轮
固定CUTLASS约少10%时间。T193随机仍落后CUTLASS约3–4%；T193热点落后约14%。
T1024两个seed相对CUTLASS排名不同，不能拿一个seed声称稳定胜出。

回退项和small本质仍是原kernel；0–2%差异是重复测量噪声，不归为tile优化。
热点T1024的p95波动较大（selected2.2295ms、original2.5437ms），不能只看p50。
所有p95/轮次/样本完整留存，未取最快样本。热buffers反复复用，无cache flush。

## Prepared 入口收益发生在哪里

small T64 API-event：公开0.043520ms、prepared0.009656ms，约少78%时间。
同一case Graph：公开0.002614ms、prepared0.002618ms，基本相同。
small T1同样是API从0.044008ms降到0.010152ms，Graph均0.002496ms。
这支持减少host入口成本，而不是GPU算术加速；CUDA-event API中含host提交空隙。
另外记录wall time，但含event记录/同步，不是独立纯host launch耗时。
大GEMM约ms级时，该入口节省约十几us，整体占比有限。

## 验证、编译资源与限制

- 三个最终配置各验证空输入、空expert、M/N/K尾块，更新输入values后在非默认
  stream运行，并做固定图重放；共12组全部通过。
- 14个holdout，4个入口的eager/Graph输出均通过FP32 oracle。
  最大relative RMS=0.001662，normalized max=0.003132（阈值均0.008）。
- medium：98 registers、0 spills、36KiB shared；large：255 registers、
  compiler报告14 spills、40KiB shared。large虽有spill仍在这些形状较快，
  不能从参数就断定实际occupancy或带宽。保存PTX/cubin/SASS/hash，含BF16 HMMA。
- 四入口正序/逆序循环轮换，每个位置出现两次（8轮），每轮6样本，2调用/样本。
  Graph每replay8个GEMM。GPU未锁频，温度40→62°C；部分热点样本受热/频率扰动。
  不具备ncu硬件计数器，访存/occupancy解释仍是待验证假设。
- 不含routing、dispatch、SwiGLU、down/combine；不是完整MoE加速或生产部署。
  CUTLASS为M2同一固定配置，本轮没有为CUTLASS调优。

MuseSpark只做方法建议，真实调用已成功，建议与usage存于
[审查记录](../results/provenance/m3-musespark-review.json)。采用了冻结规则、
循环计时、分离API/Graph和遥测；没有按其建议用holdout再挑down配置。

正式记录：[m3-holdout-20261006T093540Z.json](../results/m3-holdout-20261006T093540Z.json)。

## 使用与复跑

```python
from llm_gpu_kernels.gemm import GroupedPlan
from llm_gpu_kernels.prepared import prepare_grouped_gemm
from llm_gpu_kernels.selection import FrozenPolicy
policy = FrozenPolicy.load("results/m3-policy.json")
config, bucket = policy.select(counts, weights.shape[1], weights.shape[2])
plan = GroupedPlan.build(counts, a.device, config.block_m)
prepared = prepare_grouped_gemm(a, weights, plan, config=config)
output = prepared.run()
```

```bash
source scripts/activate-cutlass.sh
python -m llm_gpu_kernels.optimization_benchmark
```

训练执行器拒绝覆盖已有冻结规则；新一轮搜索需单独命名/保存新规则，不能
覆盖本轮的训练/holdout证据。下一阶段建立运行时GPU路由的完整MoE功能基线。
