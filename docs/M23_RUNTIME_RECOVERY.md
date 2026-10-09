# M23 运行环境恢复与启动方式

2026-10-09。本文记录重启后的实际诊断和复测入口，不包含新的性能结论。

## 已确认的问题

| 现象 | 证据与处理 |
| --- | --- |
| 最初 CUDA 无法初始化 | Driver API `cuInit=100`，PyTorch allocation/sync 失败；上次 boot 有 Xid119/GSP RPC 超时及 reset-required。用户重启后最小 CUDA 分配与同步恢复。具体驱动异常触发原因未确定。 |
| v1 扩展启动失败 | 清除继承 CUDA 环境后，旧 JIT loader 要求显式 CUDA12.8 路径；尚未计时。 |
| v2 编译失败 | 原 CUDA12.8 工具链存在；JIT loader 重写 ninja 后选择了不受支持的 host GCC。编译失败，两份审计 `.so` 的 SHA 未变化，尚未计时。 |
| v3 缺少 vLLM | 新入口成功加载原 `.so`，但 PYTHONPATH 未包含文档化的 operator overlay；尚未计时。 |
| v4 运行环境 | 使用已有 `activate-vllm.sh` 恢复 overlay/日志/chunk 设置，新入口直接加载审计二进制；新增两批 12/12 worker 通过。连同首批，18/18 记录与 12/12 启动收据已由汇总器校验，结果见 M23 报告。 |

v1/v2/v3 的失败 JSON、日志和 manifest 各自保留，没有用重试覆盖失败。
原算子源码、M22 worker、审计记录及两个 native binary 保持原哈希。

### 上次开机日志的时间顺序

上次 boot 的系统时钟曾跳变，因此按 journal 的 monotonic 时间核对：
约 +6s 已有 SBIOS 温度/平台电源数据的 `NV_ERR_INVALID_DATA`，
约 +72s 出现 `GFW_BOOT` timeout，+78–91s 出现 Xid119 GSP RPC timeout，
约 +92s 的 Xid154 要求 GPU reset。后续 CUDA 初始化仍报 reset-required。
没有在该日志中发现 suspend/resume、AER 或 PCIe Bus Error。
这些日志支持“驱动/固件在开机早期已异常”，不能证明具体应用、算子或电源策略
触发了异常，也不能仅凭某 RPC 行的进程名将责任归给该进程。
重启后的最小 CUDA 分配/同步成功；当前两批性能采集没有复现设备初始化失败。

## 为什么使用新的启动 Adapter

M22 的 JIT loader 在 prepare 阶段调用 `cpp_extension.load`，即使调用前的
`ninja -n` 无 action，loader 仍可能重新生成构建文件并触发编译。
复测需要复用已经审计的二进制，不需要重新构建算子。

`moe_w4_repeat` 从两份通过的审计记录验证源码与 `.so` SHA，再直接导入 native
module 并填入旧 loader 已有的模块缓存；未修改原函数、kernel 或计时 worker。
旧 worker 随后再次检查实际 binary 路径/SHA、输入、量化格式和全部数值结果。
每个进程另存启动收据，将新 Adapter 的源码身份和加载方式纳入复测 provenance。

收据的 `native_cpp_extension_jit_invoked=false` 仅描述 native extension 启动加载；
它不表示 worker 的 Triton 准备从未编译。准备均排除在性能计时之外。

## 单 case 复现

从仓库根目录启动独立 shell，保留主 `.venv` 和现有 vLLM0.10.0 operator overlay。
下面的输出路径必须为新路径；完整两批顺序以 M23 协议为准。

```bash
source scripts/activate-vllm.sh
env -u CUDA_HOME -u CUDA_PATH -u CUDACXX -u LD_LIBRARY_PATH \
  python -m llm_gpu_kernels.moe_w4_repeat \
  --capture-root /home/tmz/ai/traces/m20-qwen3-layer0-20261007-attempt3 \
  --case code-T1024 --batch 2 \
  --packed artifacts/m22-quant-pack-20261007-v1/gate-up-w4g128.pt \
  --pack-record results/m22-quant-format-preparation.json \
  --w4-audit artifacts/m22-w4-audit-20261007-v1/audit.json \
  --bf16-audit artifacts/m22-cuda-audit-20261007-v2/audit.json \
  --w4-extension-root /home/tmz/.cache/torch_extensions/py312_cu128 \
  --bf16-extension-root artifacts/m22-cuda-audit-20261007-v2/torch-extensions \
  --startup-receipt artifacts/m23-reproduction/batch2-code-T1024.startup.json \
  --output artifacts/m23-reproduction/batch2-code-T1024.json
```

外部 capture、pack 和审计缓存由本机证据提供，未包含在 GitHub 源码中。
其他机器须使用自己重新生成并审计的产物，不能把本机 SHA 当作其编译产物的证明。

## 后续新 kernel 的编译

本机已确认的工具链为 CUDA12.8.93 与 GCC14.3，分别位于
`/home/tmz/ai/toolchains/cuda-12.8` 和 `/home/tmz/ai/toolchains/gcc14`。
原 BF16 构建记录使用其中的 gcc/g++14 和 `-ccbin`。新的候选需在独立构建目录中
明确指定 CUDA_HOME、CC、CXX，重新记录源码/二进制/编译器并做对应正确性审计。
不得为绕过复测哈希检查而覆盖已审计二进制，也不需要向主 venv 另装 CUDA/PyTorch。
