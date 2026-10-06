# M2 成熟基线接入准备

2026-10-06，gpt-6-luna/max 子 agent 核查官方版本源码；尚未安装或编译基线。

## 主候选：CUTLASS v3.8.0 classic grouped GEMM

固定 `v3.8.0`，采用 [24_gemm_grouped](https://github.com/NVIDIA/cutlass/blob/v3.8.0/examples/24_gemm_grouped/gemm_grouped.cu)
的经典 `DefaultGemmGrouped` / `GemmGrouped` 路径。该样例有 Sm80 TensorOp
实例和 Ampere guard；不采用需要 Hopper/Blackwell 的新 grouped 路线。
样例本身为 FP16，BF16/FP32 累加实例在 SM86 的编译与正确性仍需本机验证。

接入只需一个小 CUDA extension：每个活跃 expert 的 problem M/N/K、
A/B/C device pointer、leading dimension、workspace、当前 PyTorch stream。
M=0 的 expert 排除。W[E,N,K] 存储方向与 CUTLASS B layout 必须显式匹配。
kernel-only 与 metadata/workspace 准备分别计时，不能把全 M padding 作为唯一对照。
参考 [grouped scheduler/API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)。

### 当前工具链前置缺口

项目有 torch CUDA 12.8 runtime，**没有 nvcc 12.8 toolkit**。共享 AI 环境的
nvcc 是 13.2.86，系统 GCC 15.2。不能称当前已有匹配编译链，也不能直接
把共享 CUDA 13.2 作为 torch cu128 extension 的默认 CUDA_HOME。
PyTorch v2.7.1 的 [cpp_extension 检查](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/utils/cpp_extension.py#L421-L441)
会拒绝这个 CUDA major version mismatch，因此当前不能直接构建该扩展。
CUDA 12.8 的 [Linux compiler policy](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html#host-compiler-support-policy)
列出 GCC 6.x–14.x；当前 GCC15 不在该范围。
需准备隔离的兼容 toolkit/host compiler，或先接入下面的纯 Triton 参考。
此项不影响已经运行的 Triton bootstrap；本轮未改系统 compiler 或驱动。

## 完整 MoE 的候选：vLLM v0.9.2 固定源码参考

[CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.9.2/requirements/cuda.txt)
固定 torch2.7.0；当前 torch2.7.1，不直接 pip 安装整包。
[fused_moe.py](https://github.com/vllm-project/vllm/blob/v0.9.2/vllm/model_executor/layers/fused_moe/fused_moe.py)
依赖 vLLM 的内部模块与自定义扩展，不能仅下载后直接 import。

可以评估按 Apache-2.0 保留来源/许可的纯 Triton kernel、轻 wrapper，及
[moe_align_block_size_triton](https://github.com/vllm-project/vllm/blob/v0.9.2/vllm/model_executor/layers/fused_moe/moe_align_block_size.py)
路径；不得依赖默认调用的自定义 CUDA routing extension。需要验证摘出后的
Triton3.3.1 编译、权重语义、padding 和 route/combine 数值。
这种 kernel 面向已排序 route，是完整 MoE 后端参考，不是通用 pointer-array
grouped GEMM。改动必须公开，并尽量保留上游算法与调度。

## 接入验收顺序

1. pin tag 和 commit，保存 build options 与编译链版本。
2. 在相同 BF16 输入上对照 FP32 oracle；核查空专家与尾块语义。
3. 对准数据布局、同 stream、workspace 生命周期与相同计时边界。
4. 完整记录原始样本、编译产物与失败形状，然后开始调优。

现有 Python expert-loop 继续作为诊断参考，不充当成熟 grouped/MoE 基线。
