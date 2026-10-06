# FlashAttention 2 官方算子环境（M8 四段 KV GQA）

本文记录 M8 四段 KV GQA prefill 基线使用的 FlashAttention 2（FA2）运行时、
官方 wheel 来源及复现方式。小规模 FP32 检查与 M8 正式基线均已通过；性能结果
见 [M8 基线报告](M8_GQA_PREFILL_BASELINE_REPORT.md)。

## 环境组成

FA2 以隔离 Python overlay 安装在：

```text
/home/tmz/ai/baselines/flash-attention-2.8.3.post1
```

项目仍使用仓库 `.venv` 中的 Python 3.12、PyTorch 2.7.1+cu128 和 Triton 3.3.1。
FA2 overlay 不安装或覆盖 PyTorch、Triton、CUDA 或项目依赖；安装的额外 Python
依赖为 `einops==0.8.2`。安装器从清华 PyPI 镜像获取该依赖。

`source scripts/activate-flash-attn.sh` 会先调用项目的 `scripts/activate.sh`：
启用 `.venv`、清除 `CUDA_HOME`、`CUDA_PATH`、`CUDACXX` 和
`LD_LIBRARY_PATH`、设置 `TORCH_CUDA_ARCH_LIST=8.6`，并加入项目 `src` 路径；
随后把上述 FA2 overlay 加到 `PYTHONPATH`。可在仓库根目录执行：

```bash
source scripts/activate-flash-attn.sh
python scripts/verify_flash_attn_backend.py
```

验证脚本默认检查 `~/ai/baselines/flash-attention-2.8.3.post1`，默认下载目录为
`~/ai/baselines/downloads`。调用验证脚本时，如需使用其他位置，应显式传入
`--runtime PATH` 和 `--download-dir PATH`。`FLASH_ATTN_OPERATOR_RUNTIME` 和
`FLASH_ATTN_DOWNLOAD_DIR` 环境变量由安装脚本和激活脚本读取，不会自动改变验证
脚本自身的命令行默认路径。

## 官方 wheel 来源与版本标识

使用的是 Dao-AILab [官方 release `v2.8.3.post1`](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post1)，对应上游 commit `a8aa52b`。
固定 wheel 文件名为：

```text
flash_attn-2.8.3.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

官方 release asset 的 SHA256 为：

```text
f87164bb919f5597cb94f7485196be6ac7842b71d7bf8e9f8e9133b9045ab5ea
```

wheel 文件名的 `+cu12torch2.7cxx11abiTRUE` 标识构建目标。Provenance 中记录的
`wheel_build_version` 为 `2.8.3.post1+cu12torch2.7cxx11abitrue`；这是 wheel 构建
变体标识。官方 wheel 的 METADATA 和 Python 模块报告的发行版版本均为
`2.8.3.post1`。不要将构建变体字符串误认为包的发行版或模块版本。

已记录的来源信息与校验结果见
[FA2 runtime provenance](../results/provenance/flash-attention-2.8.3.post1-runtime.json)。
目前记录显示 wheel 摘要与官方 release asset 一致，并且 overlay 内的 86 个
FA2 package files 均逐文件匹配该 wheel。

## 安装与复核

安装脚本会先下载官方 wheel 并核对完整 SHA256，再用项目 `.venv` 的 Python 配合
`uv pip --target` 把 wheel 安装到隔离 overlay，不安装其声明依赖；随后单独安装
`einops==0.8.2`，最后再次核对 wheel 摘要及 overlay 中的 FA2 文件内容：

```bash
bash scripts/install-flash-attn-operator.sh
```

脚本默认使用 `~/.local/bin/uv`、`~/ai/baselines/downloads` 和
`~/ai/baselines/flash-attention-2.8.3.post1`。可用 `UV_EXE`、
`FLASH_ATTN_DOWNLOAD_DIR`、`FLASH_ATTN_OPERATOR_RUNTIME` 覆盖相应路径。若下载中
断且摘要不匹配，校验脚本会保留 `.partial` 文件并失败，不会将其当作有效 wheel。

`scripts/verify_flash_attn_backend.py` 对下载 wheel 校验固定 SHA256；完成安装后，
它还会从 wheel 中读取 FA2 文件，并对 overlay 中对应文件逐个核对 SHA256。当前
provenance 记录的 86 个文件均匹配，未修改 FA2 源码，也未用本地编译结果替换官方
wheel。

## 使用范围与限制

M8 调用该隔离 overlay 中未修改的公开 FA2 算子，建立 segmented-KV GQA prefill 的
packed baseline，详见 [M8 固定实验计划](M8_GQA_PREFILL_BASELINE_PLAN.md)。这是一项
算子级基线：它不运行 NanoKV 或完整模型，不包括 serving 调度、模型其他层或完整
端到端推理，因此不能视为完整 serving 性能结果。

此文档说明如何固定和复核 FA2 operator runtime；它不代表安装了另一份 PyTorch，
也不改变项目既有 `.venv` 的 Torch/CUDA 配置。
