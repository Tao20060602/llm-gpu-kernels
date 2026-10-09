# 复现与证据索引

2026-10-09。先查已有结果，再按需要重跑。独立算子与NanoKV使用各自checkout和venv。

## 1. 无GPU查阅与核对

从算子仓库根目录运行，Python 3.12标准库即可，不需要torch、模型、官方overlay或CUDA：

```bash
python3 scripts/check_evidence.py
```

同时核对NanoKV的M19已提交记录：

```bash
python3 scripts/check_evidence.py --nanokv-root /absolute/path/to/nanokv
```

入口只读已有文件，将JSON结果写到标准输出；失败返回非零退出码。独立检查的状态为
`passed_operator_only`，两仓库检查为`passed_combined`。它核对M7/M16冻结源码与
training/policy双向hash、M16–M18记录hash、M7/M17/M18数值指标，以及M19全部13份
归档记录、43/44份源码、9个worker、数值/状态计数、性能比例与检索差异。
M18动态API比例和M19各比例/差异计数从原始记录重算。

检查不加载外部包，不执行核函数，不验证当前驱动、模型文件或硬件稳定性。JSON内部
保留原采集机的路径，但检查使用仓库内已提交文件，迁移时不用重建这些绝对路径。
历史模型核对数量13表示当时审计结果；不是本次重新读取权重。

本机检查结果见[完整记录](../results/evidence-integrity-20261007.json)。它记录核对文件的
字节数与SHA256，包括检查入口自身；不替代原始测量或性能/质量验收。

## 2. 结果阅读顺序

| 内容 | 入口 | 当前可支持的结论 |
| --- | --- | --- |
| MoE完整forward | [M7报告](M7_FULL_MOE_OPTIMIZATION_REPORT.md) | 特定T1024随机路由，两seed，Graph p50比固定vLLM默认配置少6.7–8.1%；不是全域领先 |
| 真实MoE单层输入 | [M20报告](M20_REAL_MOE_REPORT.md)、[接口](MOE_USAGE.md) | 固定Qwen3-30B-A3B layer-0、真实权重/hidden/router，完整expert-forward；不含router或模型时间 |
| Attention冻结策略 | [M16报告](M16_ATTENTION_RESOURCE_REPORT.md) | 短query有限域规则，36项custom Graph比例几何平均0.7791；最坏慢2.17% |
| 真实模型输入与布局 | [M17报告](M17_REAL_INPUT_BRIDGE_REPORT.md) | 真实Q/K/V张量、非连续V转换成本；不是模型请求时间 |
| 动态地址与缓存 | [M18报告](M18_DYNAMIC_BINDING_REPORT.md)、[使用契约](M18_DYNAMIC_BINDING_USAGE.md) | 六项T96动态API比例0.6683，包含绑定/V转换；单stream顺序消费 |
| NanoKV系统接入 | [M19边界](M19_NATIVE_INTEGRATION_BOUNDARY.md) | 完整prefill未取得稳定收益，尾层选择有差异，默认FA2 |
| 求职项目表述 | [两项目证据](PORTFOLIO_EVIDENCE.md) | 实现、判断和限制分开说明，各项目使用各自测量数字 |
| W4跨进程复测 | [M23报告](M23_W4_REPEAT_REPORT.md)、[样本与审计汇总](../results/m23-w4-repeat-summary.json)、[发布清单](../results/m23-release-manifest.json) | 固定T17/T193四case三批复现；T1024两case持续回归；汇总保留逐case所有arm的p50/p95及原始samples |
| W4 packed-layout | [M24状态](M24_CURRENT_STATUS.md)、[性能报告](M24_W4_LAYOUT_PERFORMANCE_REPORT.md)、[NCU补充](M24_W4_LAYOUT_NCU_REPORT.md)、[NCU摘要](../results/m24-layout-ncu-summary.json) | 单批六case机制对照；NCU补充已采集，结论保留计数器定义和因果边界 |

比例均为实验耗时/对应基线耗时，低于1才是耗时减少。不能把不同实验的百分比累加。

## 3. 重跑GPU微基准

使用本仓库venv和对应官方overlay。具体环境见[vLLM](VLLM_OPERATOR_ENVIRONMENT.md)
和[FA2](FLASH_ATTENTION_OPERATOR_ENVIRONMENT.md)，冻结策略保持原样。

```bash
source scripts/activate-vllm.sh
python -m llm_gpu_kernels.moe_full_holdout
```

```bash
source scripts/activate-flash-attn.sh
python -m llm_gpu_kernels.attention_resource_holdout
python -m llm_gpu_kernels.attention_resource_boundary_audit
```

M17/M18重放另需要12份本机大型`.pt`采集张量及原manifest，详见M17/M18报告。
这些张量和PTX/cubin/SASS未入Git；记录中保留hash和静态指令/资源证据。
只拥有克隆仓库就能阅读/核对结果，但不能据此声称重新执行了张量重放或反汇编。

## 4. 重跑真实模型请求

在NanoKV checkout按其 `docs/NATIVE_SEGMENTED_ADAPTER_USAGE.md` 执行audit后再运行suite。
需要固定Qwen3-4B snapshot、model-local来源manifest、NanoKV自己的venv与算子checkout。
该driver只读取已提交manifest里的prompt IDs，不需要12份大型采集张量。
所有GPU工作串行运行，新目录保存结果；加载、预编译、warm请求和测量分别计时。

本机路径：NanoKV在 `/home/tmz/.codex/worktrees/nanokv-native-linux/nanokv`，
模型在 `/home/tmz/models/Qwen3-4B`；迁移后使用实际路径。不能用0.6B替代4B对照。
M19只生成首token、两条固定文本、三个新进程配对组；没有decode TPOT、广泛质量或生产验收。

## 5. M20 真实 MoE 单层重放

从算子仓库根目录执行。两个下载入口仅获取固定 revision 的第一层所需权重和 tokenizer，
不是完整 30B 模型。已有下载目录可按下载入口的验证规则复用。

```bash
source scripts/activate-vllm.sh
python scripts/fetch_qwen3_moe_layer0.py --output /home/tmz/models/Qwen3-30B-A3B-layer0 --chunk-mib 32 --workers 4 --max-total-gib 2
python scripts/fetch_qwen3_moe_tokenizer.py --output /home/tmz/models/Qwen3-30B-A3B-layer0/tokenizer
```

采集目录必须在仓库外且为空；benchmark 输出必须是新文件。以下示例路径用于新一轮重跑：

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/capture_qwen3_moe_layer0.py --model-dir /home/tmz/models/Qwen3-30B-A3B-layer0 --tokenizer-dir /home/tmz/models/Qwen3-30B-A3B-layer0/tokenizer --output-dir /home/tmz/ai/traces/m20-new-run
python -m llm_gpu_kernels.moe_real_input_benchmark --capture-root /home/tmz/ai/traces/m20-new-run --output results/m20-new-run.json
```

GPU 上串行执行；新进程复测仍使用同一组 capture。真实 tokenizer 与 HF 第一层 attention/router
只用于采集，benchmark 计时为给定路由的完整 expert-forward，不含 router 或请求。
本轮 [来源目录](../results/m20-provenance/) 保存小型 manifest、token IDs 和失败日志；大型权重、
trace tensors 和编译产物未入 Git。权重引用含采集机器绝对路径，移动整个 capture 目录后需要
重新采集或显式生成新的路径 manifest，不能把旧 artifact 当成可直接移植的包。

M20 的 CPU 核对独立于旧证据检查器，避免改写旧自哈希记录。`--record` 可重复；不指定
`--capture-root` 时只校验已归档记录与当前源码，外部大文件标记为 recorded-only。

```bash
python3 scripts/check_m20_evidence.py --record results/m20-real-input-20261007-attempt2.json --record results/m20-real-input-20261007-repeat-2.json --record results/m20-real-input-20261007-repeat-3.json --capture-root /home/tmz/ai/traces/m20-qwen3-layer0-20261007-attempt3 --output results/m20-new-audit.json
```

详细来源范围、逐项计时、复测与静态 Tensor Core 证据见 [M20 报告](M20_REAL_MOE_REPORT.md)。

## 6. M23 W4 跨进程复测

结果与审计见 [M23报告](M23_W4_REPEAT_REPORT.md)、[运行恢复记录](M23_RUNTIME_RECOVERY.md) 和
[汇总JSON](../results/m23-w4-repeat-summary.json)。汇总仅纳入固定首批六个worker与v4目录
两批共18个通过worker，追加批次的12份startup receipt也逐项验证；所有p50、p95和wall
p50均由worker原始samples复算。v1/v2/v3计时前失败产物保留但不作为耗时结果。

CPU-only汇总入口为 [scripts/summarize_m23_w4.py](../scripts/summarize_m23_w4.py)。要复算，
必须提供原worker JSON及receipt所在目录，并显式指定三个路径；脚本不从目录挑选结果，
且拒绝覆盖输出：

```bash
python3 scripts/summarize_m23_w4.py \
  --first-batch-dir /path/to/m22-w4-first-batch \
  --repeat-dir /path/to/m23-w4-v4 \
  --output results/m23-w4-recheck.json
```

公开汇总包含全部 raw timing samples、逐case各arm p50/p95及worker/receipt哈希；原worker、
receipt、日志、trace、权重和编译共享库留在本机，不随release manifest发布。克隆仓库
可查阅已归档的样本和统计，但不能据此声称重新执行了GPU benchmark。
