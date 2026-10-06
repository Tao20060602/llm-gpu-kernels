# M2：CUTLASS grouped GEMM 基线（2026-10-06）

状态：编译、数值、固定图重放与十个形状的 A/B 已通过。本轮只建立基线，
没有对 Triton/CUTLASS 做参数调优，没有实现完整 MoE 层。

## 怎么做对照

1. CUTLASS v3.8.0 固定到 `afa1772203677c5118fcd82537a9c8fefbcc7008`。
   classic Sm80-template、实际编译SM86，CTA64x128x32、warp32x64x32、
   instruction16x8x16、3 stages、align8、DeviceOnly。
   Triton 保持 BM32/BN64/BK32、4 warps、3 stages。
2. 给两者相同的 BF16 A/W，W[E,N,K] 直接作为各 expert 的 column-major
   KxN B；都算 `A_e @ W_e.T`，FP32累加、BF16输出，无 M padding。
   输出预分配，metadata/workspace 准备不计入 GEMM 时间。
3. 合成 top-k2/4/8 路由，CPU生成 ID/order/counts，再把 token 行分组到 A。
   每token expert ID唯一；balanced 为 modular 分布，skewed 集中到
   max(topk,E//16) 个 expert。它们不是真实 router trace。
4. BF16 operands 转 FP32 的 expert reference 做 oracle，禁用 TF32、
   BF16 reduced-precision reduction。先验证 eager 输出，再测 API/Graph，
   最后重新验证 Graph 输出。relative RMS/normalized max 阈值各 <=0.008。
5. 四轮 AB/BA/AB/BA，每轮8样本、每样本4调用、10次warmup。
   API测正常调用的CUDA-event时间，包含wrapper检查与Python提交间隙。
   Graph每图捕获8次固定GEMM，event测replay后除以GEMM数。
   重复Graph的提交间隙仍可能计入，不称为 profiler 纯kernel时间。

统一排除 routing/dispatch、plan/workspace/output分配、SwiGLU、down、combine
和编译。首次prepare wall time另存单样本，不当作稳态分布。
只验证固定预分组GEMM图重放，没有完整MoE或动态routing图捕获验证。

## 形状、环境和来源

Qwen3-30B-A3B官方config固定 revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`，保存config和SHA256。
H2048、I768、E128、top-k8，只取gate/up的N1536/K2048；权重和输入随机生成，
没有下载整模型。见 [config provenance](../results/provenance/qwen3-30b-a3b-source.json)。

独立torch2.7.1+cu128环境、RTX3080Laptop/SM86，29个包的pip check通过。
用户目录内nvcc12.8.93/GCC14.3已编译运行。CUDA12.8与glibc2.43存在六个
noexcept声明冲突，在隔离工具链副本补齐标记；原始header/archives保留，
不修改函数体、GPU数学、CUTLASS源码或系统compiler/驱动。
工具链hash/diff在provenance，详见 [M2准备](M2_BASELINE_PLAN.md)。
不将该本机兼容补丁说成NVIDIA对Ubuntu26的完整官方认证。

## 数值和边界

- 两个backend的4组边界输入通过：空expert、M尾块、N/K尾块、全空输入。
- CUTLASS align8明确拒绝N/K非8倍数及连续但基址未16-byte对齐的view。
- 十个性能形状的eager和graph输出均通过FP32 oracle。
- 性能形状eager最大relative RMS=0.001686，normalized max=0.002859。
- 修复Triton空out误报alias：不同空storage的data_ptr都可能为0，改为比较
  storage identity；已验证全空输入复用显式out。

## Graph p50（毫秒 / GEMM）

| 场景 | T/E/top-k | N/K | Triton首版 | CUTLASS固定配置 |
| --- | --- | --- | ---: | ---: |
| small decode | 1/8/2 | 256/128 | 0.0026695 | 0.0064615 |
| small balanced | 64/8/2 | 256/128 | 0.0028160 | 0.0065280 |
| small skewed | 64/8/2 | 256/128 | 0.0028480 | 0.0066560 |
| medium top4 | 128/16/4 | 512/256 | 0.013310 | 0.017744 |
| Qwen balanced | 1/128/8 | 1536/2048 | 0.119408 | 0.121824 |
| Qwen balanced | 64/128/8 | 1536/2048 | 1.984446 | 1.882752 |
| Qwen skewed | 64/128/8 | 1536/2048 | 0.149024 | 0.132608 |
| Qwen balanced | 512/128/8 | 1536/2048 | 2.835107 | 1.962879 |
| Qwen skewed | 512/128/8 | 1536/2048 | 1.108752 | 0.798649 |
| Qwen balanced | 4096/128/8 | 1536/2048 | 9.703744 | 6.707792 |

API也完整保存。例small balanced：Triton API=0.033656ms，graph=0.002816ms；
CUTLASS API=0.007920ms，graph=0.006528ms。同形状两种模式的排名相反。
这提示host/API与GPU计算要分开研究，不能由差值精确归因某个函数。

笔记本GPU未锁频/功率或独占，整轮温度41→61°C，结束SM1575MHz、显存7001MHz。
四轮对称并不能完全消除热/频率变化。重复相同输入，无主动cache flush，
不模拟多层模型cache；保存p95和全部原始样本，未选最快样本。
峰值allocated约3.03GiB，包括oracle/graphs，不是纯kernel workspace。

## 判断与下一步假设

1. **小矩阵分开处理host和GPU时间。** 简单Triton kernel在小形状graph下
   较快，公开API却较慢。下一阶段用绑定buffers的入口单独测提交/检查成本，
   不改变数学语义。
2. **大形状有明确GPU优化目标。** T512/T4096均匀场景，固定CUTLASS graph
   耗时约少31%。下一阶段先扫tile/warp/stage/调度，再依据测量考虑融合。
   当前没有经过调优得到的收益可以写进简历。
3. **路由改变权重工作集。** 同T64/top-k8，均匀激活128专家，热点激活8专家；
   权重从768MiB降到48MiB，每expert行数从4升到64，FLOPs相同但时间差很大。
   推测访存/复用是关键因素，需profiler验证。这个单GPU结果不能推广为多卡
   expert-parallel的负载均衡结论。

CUTLASS这里只是成熟实现的一个固定配置，未做后端参数搜索，也不是完整vLLM MoE。
小形状胜出不能说超过所有成熟后端。后续加入随机routing、更多seed和未参与
调参的holdout shapes，再报告优化适用范围。

## 产物与复跑

CUTLASS库SASS有32条静态BF16 HMMA，Triton计时形状也有HMMA。
[CUTLASS SASS](../results/codegen/m2-cutlass.sass)纳入Git；完整.so/build.ninja
在`artifacts/<run_id>/cutlass/`，hash和编译flags纳入JSON。
未获得ncu occupancy/带宽/TensorCore utilization硬件计数器。

正式结果：[m2-20261006T082613Z-488194393.json](../results/m2-20261006T082613Z-488194393.json)。
源码hash已核对一致。探索轮[082149Z](../results/m2-20261006T082149Z-54895047.json)
保留；其3轮计时在native输入检查收尾期间运行，不用于正式引用。

```bash
cd /home/tmz/文档/ChatGPT/llm-gpu-kernels
source scripts/activate-cutlass.sh
python -m llm_gpu_kernels.baseline_benchmark
```

已有工具链可直接复跑。初始化入口：`scripts/fetch_cuda_toolchain.py`、
`scripts/fetch_gcc14.py`、`scripts/patch_cuda_glibc.py`，CUTLASS checkout固定到
上述commit。这些脚本针对当前Ubuntu26 APT版本，不是任意Linux的通用安装器。
