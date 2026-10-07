# 当前算子项目：结果与复现入口

## 已有成果

独立仓库位于 `/home/tmz/文档/ChatGPT/llm-gpu-kernels`，与NanoKV引擎源码分开。
单GPU SM86、BF16 forward原型。MoE包含合成训练及真实Qwen3-30B-A3B第一层固定路由重放；attention包含合成训练和真实
Qwen3-4B张量重放。模型请求集成另有边界审计，尚未作为生产部署或广泛质量结果。

### 完整 MoE

- GPU histogram/rank、prefix、compact dispatch、gate/up+SwiGLU、down、combine，
  完整路径7次启动；GPU每次重新构造路由分组，不需要CPU counts回读。
- 重复ID、空expert/尾块、signed routing weights、非默认stream、同地址值更新与Graph
  检查。原0.008 FP32误差界与finite；保留BF16舍入契约。
- 固定官方vLLM0.10.0 public fused_experts作为成熟对照，保存未修改来源和默认配置。
- M7有限训练搜索和长复测后冻结shape规则。T1024 random、两个seed，相对自身原ranked
  Graph p50少约25%，相对该固定vLLM默认配置少6.7–8.1%。T193/hot等仍落后，不能说
  所有形状优于vLLM。详见[M7报告](M7_FULL_MOE_OPTIMIZATION_REPORT.md)。
- M20使用pinned真实Qwen3-30B-A3B第一层expert权重、官方HF路径的hidden与router；
  两条示例文本×四种长度，三次新进程共432项数值指标通过，冻结M7策略保持原样。
  T1024 Graph相对固定vLLM默认配置少3.20–6.99%；小输入Graph回归最多3.23%。
  全24项等权p50比例几何平均API少2.30%、Graph少1.13%，不代表生产流量加权收益。
  计时包含GPU分组、dispatch与expert forward，排除router/attention/模型。
  [来源、逐项结果与复测](M20_REAL_MOE_REPORT.md)、[prepared接口](MOE_USAGE.md)。

### 四段 KV GQA attention

- 直接四段读取的在线softmax、GQA head映射/current因果mask、BF16 Tensor Core dot；
  独立全矩阵chunk64 FP32参考，固定地址Graph值更新和非默认stream。
- 强对照是每次复用buffer pack+未修改官方FA2，不用弱PyTorch reference作性能对照。
- M9固定G1在T64若干场景有收益；M10统一遍历在大T对自身更快，但有边界回归。
  M11 split/merge、M12小query、M13固定Q64、M14分块/流水线都保留训练及新输入回归。
- M11–M13冻结为FA2回退；M10/M14冻结实验策略含自定义路径，但新输入存在回归。 局部收益不能拼成通用优化或可靠生产规则。
  这些历史实验的回归保留；后续可用短query路径见M16。
  [M9](M9_SEGMENTED_ATTENTION_REPORT.md)、[M10](M10_UNIFIED_ATTENTION_REPORT.md)、
  [M11](M11_SPLIT_KV_REPORT.md)、[M12](M12_NARROW_SPLIT_REPORT.md)、
  [M13](M13_FIXED64_REPORT.md)、[M14](M14_ATTENTION_RETILE_REPORT.md)。
- M15简化softmax状态，同stage对照13输入全部减少耗时，S2几何平均少7.55%；
  短query六项比强FA2少7.1–34.5%，大T仍落后。不是冻结域策略。
  [M15报告](M15_NONEMPTY_ATTENTION_REPORT.md)。
- **M16短query冻结路径通过当前验证矩阵。** T32–128、sink64/recent512/currentT、
  总history≤4096用M15 token32/key64/G1/W4/S2，其余FA2。44项原holdout加25项
  上限追加审计全部数值正确；36项实际custom路径Graph p50几何平均比完整FA2
  少22.09%，6项T127/128长历史小幅回归，最坏慢2.17%。没有按这些数据改策略。
  这是有限合成输入的独立prototype结论，NanoKV引擎默认仍未切换。
  [M16报告](M16_ATTENTION_RESOURCE_REPORT.md)、[使用契约](ATTENTION_USAGE.md)。
- M17在固定Qwen3-4B、两条实际模型请求采集的12组Q/K/V上重放；6项96尾chunk
  计入实际非连续V转换后，比强复用pack+FA2 Graph geomean少37.26%。原FA2/各arm
  独立FP32及204项精度检查通过。4096仍用FA2；首次形状prepare约1.03s，动态地址
  当时编译/绑定缓存与系统A/B尚未完成，后续分别见M18/M19。不能将该数字写成请求加速。
  [真实输入报告](M17_REAL_INPUT_BRIDGE_REPORT.md)。


- M18完成指针无关的编译模板LRU与stream独立动态workspace；12项真实输入、7项功能和
  3项布局审计共227项精度指标通过。六项尾chunk计入动态绑定/生命周期/V转换后，API
  比强FA2平均少33.17%，固定Graph少37.11%；动态API较M17固定入口多7.72%成本。
  fallback额外预留output和全冷启动未验证的限制公开；后续系统接入见M19。
  [动态绑定报告](M18_DYNAMIC_BINDING_REPORT.md)、[入口契约](M18_DYNAMIC_BINDING_USAGE.md)。

- M19完成NanoKV显式实验接入和9个新进程三arm对照，1080项精度/864项状态检查通过。
  完整prefill相对复用FA2平均仅少0.08%，相对原FA2慢0.60%；后续尾层检索集合变化，
  默认继续FA2。[系统边界](M19_NATIVE_INTEGRATION_BOUNDARY.md)、[简历证据](PORTFOLIO_EVIDENCE.md)。

## 复现冻结后的实验

先查阅[复现与证据索引](REPRODUCTION_INDEX.md)：无GPU的文件核对、GPU微基准和真实模型
请求分别有独立入口。以下命令重跑历史测量，不是查看已有结果所必需。

从本仓库根目录使用各自官方overlay；不重新覆盖冻结policy：

```bash
source scripts/activate-vllm.sh
python -m llm_gpu_kernels.moe_full_holdout
```

```bash
source scripts/activate-flash-attn.sh
python -m llm_gpu_kernels.attention_segmented_benchmark
python -m llm_gpu_kernels.attention_unified_holdout
python -m llm_gpu_kernels.attention_split_holdout
python -m llm_gpu_kernels.attention_narrow_holdout
python -m llm_gpu_kernels.attention_fixed_holdout
python -m llm_gpu_kernels.attention_retile_holdout
python -m llm_gpu_kernels.attention_nonempty_benchmark
python -m llm_gpu_kernels.attention_resource_holdout
python -m llm_gpu_kernels.attention_resource_boundary_audit
```

原始JSON保存每项输入/source hash、finite/误差、raw event/wall、p50/p95、telemetry、
allocator peak。冻结policy校验源码与training record；代码变动需要新实验而不是覆盖
旧record。PTX/cubin/SASS产物在本机ignored artifacts，hash/指令及资源在JSON。

## 环境与成果边界

项目独立venv：Python3.12.14、Torch2.7.1+cu128、Triton3.3.1，RTX3080 Laptop16GiB。
vLLM0.10.0与FA2 2.8.3.post1分别隔离安装，只用官方公开算子，不是完整serving安装。
环境说明：[vLLM](VLLM_OPERATOR_ENVIRONMENT.md)、[FA2](FLASH_ATTENTION_OPERATOR_ENVIRONMENT.md)。

HMMA只证明编译产物使用Tensor Core，不能代替utilization、occupancy或DRAM计数器。
本仓库微基准为hot repeated GPU resident输入：M7 MoE使用给定合成路由，M20使用真实第一层
权重与路由；M17/M18 attention
使用真实模型采集张量，计时不包括CPU gather/H2D/router/RoPE或模型请求。
M19请求测量在独立NanoKV仓库执行，包含该请求的数据路径；不含模型加载与warmup，
只生成首token。两者均没有多GPU/expert parallel或生产serving证据；不能重复使用算子
数字作为系统收益。
