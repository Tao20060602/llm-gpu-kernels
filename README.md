# LLM GPU Kernels on Ampere

独立于 NanoKV 的算子项目：先做 **BF16 MoE grouped GEMM 与局部融合**，
包含四段KV GQA prefill attention。硬件为 RTX 3080 Laptop 16 GiB，SM86。

当前状态与M24最新结果见[M24当前入口](docs/M24_CURRENT_STATUS.md)；M23及更早阶段仍保留各自快照，历史项目脉络见[项目状态快照](docs/PROJECT_STATUS.md)，查阅与复现见[证据索引](docs/REPRODUCTION_INDEX.md)。完整MoE已取得特定形状的成熟对照收益；attention已实现短query冻结选择：合成验证矩阵中，相对完整FA2链路Graph耗时几何平均减少22.09%，最坏慢2.17%；其他形状回退FA2。M18另有真实模型张量重放，M19已完成系统接入对照，未取得稳定完整prefill收益。

真实 Qwen3-30B-A3B layer-0 的 expert 权重、hidden 与 router 路由重放见
[M20 报告](docs/M20_REAL_MOE_REPORT.md)；它与 M7 合成输入结果分开记录。
完整 MoE prepared 接口与缓冲区生命周期见 [使用说明](docs/MOE_USAGE.md)。

M25 正在加深 Ampere GEMM：新增直接 `ldmatrix` / `mma.sync` 的 BF16
`A @ W.T` 实验路径，以及同步和 2/3/4 级 `cp.async` 环形流水。
它已完成 28 组核心数值检查和四形状小规模基准，尚未接入默认 MoE 路径；
不同形状的最佳流水深度不同，小 M 仍慢于 PyTorch 对照。结果与范围见
[M25 诊断报告](docs/M25_MMA_DIAGNOSTIC_REPORT.md)，线程映射与缓冲生命周期见
[M25 设计讲义](docs/M25_MMA_PIPELINE_DESIGN.md)。

W4A16 首版与三批复测的差异、逐 case 回归及限制见
[M23 W4 复测报告](docs/M23_W4_REPEAT_REPORT.md)和[完整样本/审计汇总](results/m23-w4-repeat-summary.json)：
固定T17/T193四case的Full Graph p50几何平均相对同量化权重vLLM低10.00%，T1024两case
慢17.30%，不代表全域或生产收益。发布文件清单见[M23 release manifest](results/m23-release-manifest.json)。

M24 对 W4 packed gate/up 做了不改量化值与算术顺序的物理布局机制对照：单批六个固定
case 的 Full Graph p50 几何平均相对同量化 vLLM 低21.17%，相对旧 W4 低19.43%；T1024
对 vLLM 一快一慢，不能表述为长输入普遍改善。范围与限制见[M24当前入口](docs/M24_CURRENT_STATUS.md)、
[性能报告](docs/M24_W4_LAYOUT_PERFORMANCE_REPORT.md)、[NCU计数器补充](docs/M24_W4_LAYOUT_NCU_REPORT.md)和
[汇总JSON](results/m24-w4-layout-summary.json)。NCU补充采集及分析已完成，指标解释与因果边界见补充报告。

当前已完成 M0/M1 bootstrap、M2 固定 CUTLASS3.8 基线及 M3 参数优化：
原始 Triton dense/grouped GEMM、FP32 MoE 语义参考、边界数值检查、
API/CUDA Graph 对照计时与 Tensor Core 编译产物。M4已实现GPU动态值路由的
完整forward和gate/up+SwiGLU局部融合。M5已接入未修改的vLLM0.10.0完整MoE对照和缓存调用入口。
M6减少GPU调度，M7完成完整MoE分块搜索与冻结策略对照。attention已完成M8–M19有限范围实验；两个项目均为单GPU原型。

## 运行

```bash
cd /home/tmz/文档/ChatGPT/llm-gpu-kernels
source scripts/activate.sh
python -m llm_gpu_kernels.benchmark
```

运行环境：Python 3.12.14 / torch 2.7.1+cu128 / Triton 3.3.1，独立 `.venv`。
基础运行包来自本机 uv 缓存；M2另备用户目录内CUDA12.8/GCC14编译工具链。
精确包版本见 requirements.lock.txt，计算契约见 [CONTRACT](docs/CONTRACT.md)，
实际数值、性能和限制见 [首轮报告](docs/BOOTSTRAP_REPORT.md)。
固定成熟 grouped 实现的接入、结果与判断见 [M2报告](docs/M2_BASELINE_REPORT.md)。
Prepared 入口、冻结选择规则与随机 holdout 收益见 [M3报告](docs/M3_OPTIMIZATION_REPORT.md)。
全GPU固定路由MoE、动态路由值图重放和融合 A/B 见 [M4报告](docs/M4_MOE_REPORT.md)。
完整成熟MoE对照与入口开销见 [M5报告](docs/M5_MATURE_MOE_REPORT.md)。
GPU调度从8次减到7次的完整链路 A/B 见 [M6报告](docs/M6_RANKED_DISPATCH_REPORT.md)。
完整MoE冻结分块策略与vLLM对照见 [M7报告](docs/M7_FULL_MOE_OPTIMIZATION_REPORT.md)。
四段KV拼接与官方FA2的12场景成本基线见 [M8报告](docs/M8_GQA_PREFILL_BASELINE_REPORT.md)。
FA2使用独立overlay，来源与复现见 [环境说明](docs/FLASH_ATTENTION_OPERATOR_ENVIRONMENT.md)。
四段直读GQA与head分组的固定对照见 [M9报告](docs/M9_SEGMENTED_ATTENTION_REPORT.md)。
统一KV遍历的有限搜索、冻结后回归与边界见 [M10报告](docs/M10_UNIFIED_ATTENTION_REPORT.md)。
split-KV的完整成本与全域回退见 [M11报告](docs/M11_SPLIT_KV_REPORT.md)。
窄域split与固定query64的新长度回归见 [M12报告](docs/M12_NARROW_SPLIT_REPORT.md) 和 [M13报告](docs/M13_FIXED64_REPORT.md)。
query分块/流水线的新输入回归见 [M14报告](docs/M14_ATTENTION_RETILE_REPORT.md)。
softmax状态简化的固定同配置 A/B 见 [M15报告](docs/M15_NONEMPTY_ATTENTION_REPORT.md)。
短query资源搜索与44+25项冻结后验证见 [M16报告](docs/M16_ATTENTION_RESOURCE_REPORT.md)，接口与生命周期见 [使用说明](docs/ATTENTION_USAGE.md)。
真实Qwen3-4B输入与非连续V边界成本见 [M17报告](docs/M17_REAL_INPUT_BRIDGE_REPORT.md)。
vLLM算子依赖使用独立overlay，见 [环境说明](docs/VLLM_OPERATOR_ENVIRONMENT.md)。
源码支持通过 PYTHONPATH 运行；pyproject 提供后续标准安装入口。

## 路线

1. M0/M1：数值契约、基础 GEMM、ragged grouped GEMM 与 HMMA 证据。
2. M2：已接入固定 CUTLASS grouped 配置，完成 top-k2/4/8 和偏斜对照。
3. M3：已完成有限 tile/warp/stage 搜索；14个 holdout 与原实现/CUTLASS对照。
4. M4：已实现GPU dispatch、SwiGLU、down、combine全链路和gate/up+SwiGLU融合；成熟对照见M5。
5. M5：已完成vLLM0.10.0完整链路、cached launcher、15形状对照。
6. M6：已验证ranked dispatch将8次启动减到7次；小形状Graph减少约6–7%，大形状变化不足1%。
7. M7：完整MoE有限搜索、shape选择规则冻结与15场景对照；T1024随机路由比自身原配置减少约25%，比固定vLLM默认配置减少约7–8%。其他场景见报告，不能推广为全面胜出。
8. M8：四段GPU KV的复用缓冲区pack+FA2与预拼接FA2基线已完成；后续直接分段GQA kernel见M9–M18。
9. M9：四段直读GQA的G1/2/4对照、全矩阵独立FP32和stream/Graph验证已完成；G1仅T64优于强FA2链路，较大形状仍回归。
10. M10：统一KV遍历12候选训练与冻结验证完成；大T对自身首版减少约11–19%，T256回归，策略仅作实验记录。
11. M11：split-KV/merge的8候选完整成本已测，未通过T≤256全域门槛，冻结为FA2回退。
12. M12/M13：小query split与固定query64的新输入验证均因最坏回归未达门槛而保留FA2；全部数据与失败保留。
13. M14：六种query分块/流水线配置及冻结边界验证；训练选择在T127/128长历史回归约16%，保留实验策略，推荐默认FA2。
14. M15：在非空current契约下简化softmax状态，S2的13项同配置Graph耗时几何平均减少7.55%；成熟对照与大T回归完整保留。
15. M16：简化状态后的八种资源分块训练、冻结与69项验证；短query规则保留，其他形状FA2，最坏回归2.17%和适用边界完整记录。
16. M17：12组真实Qwen3-4B张量重放，6项96尾chunk计入V转换后Graph平均少37.26%；首次形状prepare约1.03s，不是公平的完全冷启动对照。后续系统接入见M19。
17. M18：编译模板LRU、动态地址与stream workspace；真实尾chunk动态API平均少33.17%，固定Graph少37.11%。后续完整prefill对照见M19。[报告](docs/M18_DYNAMIC_BINDING_REPORT.md) / [使用](docs/M18_DYNAMIC_BINDING_USAGE.md)。
18. M19：NanoKV实验接入与三组新进程对照完成；完整prefill没有稳定收益，后续尾层检索集合有差异，默认维持FA2。[边界报告](docs/M19_NATIVE_INTEGRATION_BOUNDARY.md)。
19. M20：pinned Qwen3-30B-A3B 第一层真实权重与路由输入、冻结 M7 策略的完整 expert-forward 对照；范围与原始记录见 [报告](docs/M20_REAL_MOE_REPORT.md)。
20. M22/M23：W4A16在线解码与融合完成三批固定输入复测；小组T17/T193收益复现，但T1024两输入持续回归。M24 packed-layout 已完成42状态与T0正确性审计及单批六case机制对照；没有改变默认实现，也不是新holdout。见 [M23报告](docs/M23_W4_REPEAT_REPORT.md)、[M24状态](docs/M24_CURRENT_STATUS.md) 与 [M24计划](docs/M24_W4_LAYOUT_PLAN.md)。

两个独立项目的可用表述、证据和限制见[简历与面试证据](docs/PORTFOLIO_EVIDENCE.md)。

成熟基线的来源与工具链准备见 [M2 准备](docs/M2_BASELINE_PLAN.md)。

首轮 Qwen-like grouped case 使用合成输入与 counts：E128、N1536、K2048、
总512 routed rows，模拟 T64/topk8 的均匀分组。它只有 gate/up 的形状，
不代表真实 router 输出、真实模型质量或完整 MoE 层性能。M2已固定官方
config revision/hash，详见 results/provenance。

## 来源与成果边界

- [Triton v3.3.1 GEMM tutorial](https://github.com/triton-lang/triton/blob/v3.3.1/python/tutorials/03-matrix-multiplication.py)
- [Triton v3.3.1 grouped GEMM tutorial](https://github.com/triton-lang/triton/blob/v3.3.1/python/tutorials/08-grouped-gemm.py)
- [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [Qwen3-30B-A3B config](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)

基础分块/`tl.dot` 技术为已有方法，本项目首版为自行编写的学习与测量起点。
只击败 Python expert 循环不能声称超过成熟 MoE。所有本机数字保留适用形状、
误差和计时边界；单 GPU prototype 不能说明 expert-parallel 或 serving 经验。
