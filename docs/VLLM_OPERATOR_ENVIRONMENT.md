# vLLM MoE operator 隔离环境

本说明只用于导入和运行vLLM 0.10.0的公开`fused_experts` MoE operator，作为本项目
完整fixed-route forward的成熟backend对照。该overlay不是完整vLLM serving安装：不包含
完整服务端依赖、模型runner或部署配置，不用于启动vLLM server。

## 固定版本与隔离边界

- 项目主环境保持Python 3.12、PyTorch `2.7.1+cu128`、CUDA runtime 12.8和Triton `3.3.1`。
- vLLM采用未修改官方`0.10.0` PyPI wheel，对应tag commit
  `6d8d0a24c02bfd84d46b3016b865a44f048ae84b`。
- 默认overlay路径为`$HOME/ai/baselines/vllm-0.10.0`。它放在项目`.venv`之外，避免
  vLLM算子依赖替换项目固定的PyTorch、CUDA或Triton。
- 安装脚本以`--target`写入overlay并使用`--no-deps`；锁文件列出operator import所需的
 依赖，不会执行完整vLLM的依赖解析。
- `VLLM_OPERATOR_RUNTIME`可将overlay改到其他目录；安装和激活必须使用同一路径。

官方wheel地址与SHA256固定在`scripts/verify_vllm_backend.py`和
`results/provenance/vllm-0.10.0-runtime.json`中。本次官方wheel SHA256为
`8ca37559d82b43b5e8c8248d2e4a1ecb51d6d4e5d517491d656df6491ed93dab`。安装后校验脚本单独下载官方wheel（默认
缓存于`$HOME/ai/baselines/downloads/`），校验wheel SHA256，并逐文件比较安装overlay里的
vLLM package文件。本次运行验证了1,237个上游文件与官方wheel一致。pip镜像仅用于安装
依赖；对vLLM源文件的判据是官方wheel校验，不依赖镜像宣称。

## 安装或重建

从仓库根目录运行。需先创建固定Python3.12的`.venv`并安装`requirements.lock.txt`中的项目依赖；这些步骤不
安装vLLM serving栈。

```bash
source scripts/activate.sh
scripts/install-vllm-operator.sh
```

安装脚本默认用`$HOME/.local/bin/uv`，也可通过`UV_EXE`指定uv可执行文件。依赖从清华
PyPI镜像按`requirements-vllm-operator.lock.txt`安装到overlay；随即运行
`scripts/verify_vllm_backend.py`，其官方wheel下载和校验需网络可用。

锁文件没有单独安装torch或triton；它们由项目`.venv`提供。不要为了这个对照在项目环境
另装第二套CUDA/PyTorch，也不要把这些operator依赖解释成完整服务端依赖集合。

## 激活、检查与运行

```bash
source scripts/activate-vllm.sh
python -c 'import sys, torch, triton, vllm; print(sys.executable); print(torch.__version__, torch.__file__); print(triton.__version__, triton.__file__); print(vllm.__version__, vllm.__file__)'
python -m llm_gpu_kernels.mature_moe_benchmark
```

`activate-vllm.sh`先source项目的`activate.sh`，再把overlay置于`PYTHONPATH`前部，设置
日志等级和`VLLM_FUSED_MOE_CHUNK_SIZE=32768`。主`.venv`仍提供PyTorch/Triton；本次记录的
vLLM模块路径为`/home/tmz/ai/baselines/vllm-0.10.0/vllm/__init__.py`，且与provenance
中的runtime路径匹配。复现结果应保留解释器路径、模块版本/路径和运行JSON中的来源哈希。

不要使用`pip check`作为该overlay的验收标准。它会检查完整发行包元数据依赖，而本overlay
刻意省略了不属于operator运行范围的serving依赖；该结果不能判断已验证的`fused_experts`
路径是否正确。应使用上述官方wheel校验脚本和版本/路径检查，并实际运行项目的operator
验证。

## 配置和范围限制

本项目通过vLLM 0.10.0 public API调用官方`fused_experts`，不替换routing、alignment、GEMM、
activation或combine实现，不覆盖config。它依据官方配置选择逻辑尝试查找设备配置。M5的
15个case中，所请求的`E=128,N=768,device_name=NVIDIA_GeForce_RTX_3080_Laptop_GPU.json`和
`E=16,N=64,device_name=NVIDIA_GeForce_RTX_3080_Laptop_GPU.json`文件均未打包；每case仍记录
官方selector返回的tile和`packaged_config_found=false`。因此这组数据不是使用了专为该GPU
和shape准备的精确调优JSON，也不能称为vLLM当前版本的最优配置。

官方public API会为调用创建缓存、对齐后的IDs及输出；CUDA Graph内由捕获池复用。M5的
API和Graph结果反映各自公开入口边界，不等同于同一workspace分配方式。vLLM的阶段舍入与
本项目略有不同，报告按完整FP32 oracle的预设阈值比较，不把逐bit差异判作错误。见
[M5报告](M5_MATURE_MOE_REPORT.md)和[M5实验计划](M5_EXPERIMENT_PLAN.md)。
