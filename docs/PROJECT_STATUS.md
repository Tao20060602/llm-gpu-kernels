# 当前算子项目：结果与复现入口

## 已有成果

独立仓库位于 `/home/tmz/文档/ChatGPT/llm-gpu-kernels`，与NanoKV引擎源码分开。
单GPU SM86、BF16 forward原型，全部数字来自合成输入；不是生产部署或模型质量结果。

### 完整 MoE

- GPU histogram/rank、prefix、compact dispatch、gate/up+SwiGLU、down、combine，
  完整路径7次启动；GPU每次重新构造路由分组，不需要CPU counts回读。
- 重复ID、空expert/尾块、signed routing weights、非默认stream、同地址值更新与Graph
  检查。原0.008 FP32误差界与finite；保留BF16舍入契约。
- 固定官方vLLM0.10.0 public fused_experts作为成熟对照，保存未修改来源和默认配置。
- M7有限训练搜索和长复测后冻结shape规则。T1024 random、两个seed，相对自身原ranked
  Graph p50少约25%，相对该固定vLLM默认配置少6.7–8.1%。T193/hot等仍落后，不能说
  所有形状优于vLLM。详见[M7报告](M7_FULL_MOE_OPTIMIZATION_REPORT.md)。

### 四段 KV GQA attention

- 直接四段读取的在线softmax、GQA head映射/current因果mask、BF16 Tensor Core dot；
  独立全矩阵chunk64 FP32参考，固定地址Graph值更新和非默认stream。
- 强对照是每次复用buffer pack+未修改官方FA2，不用弱PyTorch reference作性能对照。
- M9固定G1在T64若干场景有收益；M10统一遍历在大T对自身更快，但有边界回归。
  M11 split/merge、M12小query、M13固定Q64、M14分块/流水线都保留训练及新输入回归。
- **当前推荐attention路径仍为FA2。** M11–M13冻结为回退；M10/M14冻结实验策略含自定义路径，但新输入存在回归。 局部收益不能拼成通用优化或可靠生产规则。
  实现/数学验证已完成，稳定胜过成熟后端的attention策略仍待下一轮研究。
  [M9](M9_SEGMENTED_ATTENTION_REPORT.md)、[M10](M10_UNIFIED_ATTENTION_REPORT.md)、
  [M11](M11_SPLIT_KV_REPORT.md)、[M12](M12_NARROW_SPLIT_REPORT.md)、
  [M13](M13_FIXED64_REPORT.md)、[M14](M14_ATTENTION_RETILE_REPORT.md)。

## 复现冻结后的实验

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
```

原始JSON保存每项输入/source hash、finite/误差、raw event/wall、p50/p95、telemetry、
allocator peak。冻结policy校验源码与training record；代码变动需要新实验而不是覆盖
旧record。PTX/cubin/SASS产物在本机ignored artifacts，hash/指令及资源在JSON。

## 环境与成果边界

项目独立venv：Python3.12.14、Torch2.7.1+cu128、Triton3.3.1，RTX3080 Laptop16GiB。
vLLM0.10.0与FA2 2.8.3.post1分别隔离安装，只用官方公开算子，不是完整serving安装。
环境说明：[vLLM](VLLM_OPERATOR_ENVIRONMENT.md)、[FA2](FLASH_ATTENTION_OPERATOR_ENVIRONMENT.md)。

HMMA只证明编译产物使用Tensor Core，不能代替utilization、occupancy或DRAM计数器。
实验为hot repeated GPU resident输入，不包括CPU gather/H2D/router/RoPE、真实模型、
多GPU/expert parallel或NanoKV请求。NanoKV系统项目另保留模型、KV状态/selection、
同步与端到端证据；不能重复使用算子数字作为系统收益。
