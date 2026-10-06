# LLM GPU Kernels on Ampere

独立于 NanoKV 的算子项目：先做 **BF16 MoE grouped GEMM 与局部融合**，
随后做 GQA prefill attention。硬件为 RTX 3080 Laptop 16 GiB，SM86。

当前已完成 M0/M1 bootstrap、M2 固定 CUTLASS3.8 基线及 M3 参数优化：
原始 Triton dense/grouped GEMM、FP32 MoE 语义参考、边界数值检查、
API/CUDA Graph 对照计时与 Tensor Core 编译产物。M4已实现GPU动态值路由的
完整forward和gate/up+SwiGLU局部融合。完整MoE成熟backend对照、attention
与生产部署尚未完成。

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
源码支持通过 PYTHONPATH 运行；pyproject 提供后续标准安装入口。

## 路线

1. M0/M1：数值契约、基础 GEMM、ragged grouped GEMM 与 HMMA 证据。
2. M2：已接入固定 CUTLASS grouped 配置，完成 top-k2/4/8 和偏斜对照。
3. M3：已完成有限 tile/warp/stage 搜索；14个 holdout 与原实现/CUTLASS对照。
4. M4：已实现GPU dispatch、SwiGLU、down、combine全链路和gate/up+SwiGLU融合；待成熟MoE对照。
5. M5：GQA prefill；NanoKV attention 适配是可选应用验证。

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
