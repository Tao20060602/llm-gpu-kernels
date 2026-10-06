# LLM GPU Kernels on Ampere

独立于 NanoKV 的算子项目：先做 **BF16 MoE grouped GEMM 与局部融合**，
随后做 GQA prefill attention。硬件为 RTX 3080 Laptop 16 GiB，SM86。

当前为 M0/M1 bootstrap：原始 Triton dense/grouped GEMM、FP32 MoE 语义参考，
边界数值检查、CUDA-event 诊断计时与 Tensor Core 编译产物。完整 MoE 优化、
成熟 backend 对照、attention 与生产部署尚未完成。

## 运行

```bash
cd /home/tmz/文档/ChatGPT/llm-gpu-kernels
source scripts/activate.sh
python -m llm_gpu_kernels.benchmark
```

运行环境：Python 3.12.14 / torch 2.7.1+cu128 / Triton 3.3.1，独立 `.venv`。
环境包从本机 uv 缓存安装，没有重新下载权重、CUDA toolkit 或改动 NanoKV。
精确包版本见 requirements.lock.txt，计算契约见 [CONTRACT](docs/CONTRACT.md)，
实际数值、性能和限制见 [首轮报告](docs/BOOTSTRAP_REPORT.md)。
源码支持通过 PYTHONPATH 运行；pyproject 提供后续标准安装入口。

## 路线

1. M0/M1：数值契约、基础 GEMM、ragged grouped GEMM 与 HMMA 证据。
2. M2：成熟 SM86 backend 接入，固定来源，扩大形状和偏斜路由矩阵。
3. M3：tile / warp / stage / 调度优化，保存全部对照和失败形状。
4. M4：GPU dispatch、SwiGLU、down、combine 全链路；profile 后决定融合。
5. M5：GQA prefill；NanoKV attention 适配是可选应用验证。

成熟基线的固定版本候选与工具链缺口见 [M2 准备](docs/M2_BASELINE_PLAN.md)。

首轮 Qwen-like grouped case 使用合成输入与 counts：E128、N1536、K2048、
总512 routed rows，模拟 T64/topk8 的均匀分组。它只有 gate/up 的形状，
不代表真实 router 输出、真实模型质量或完整 MoE 层性能。正式模型形状矩阵
仍需固定官方 config revision/hash。

## 来源与成果边界

- [Triton v3.3.1 GEMM tutorial](https://github.com/triton-lang/triton/blob/v3.3.1/python/tutorials/03-matrix-multiplication.py)
- [Triton v3.3.1 grouped GEMM tutorial](https://github.com/triton-lang/triton/blob/v3.3.1/python/tutorials/08-grouped-gemm.py)
- [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [Qwen3-30B-A3B config](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)

基础分块/`tl.dot` 技术为已有方法，本项目首版为自行编写的学习与测量起点。
只击败 Python expert 循环不能声称超过成熟 MoE。所有本机数字保留适用形状、
误差和计时边界；单 GPU prototype 不能说明 expert-parallel 或 serving 经验。
