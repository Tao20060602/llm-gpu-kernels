# M0/M1 首轮记录（2026-10-06）

已建立独立本地 Git 仓库，分支 `codex/moe-bootstrap`，尚未创建 GitHub remote。
环境为独立 `.venv`，Python 3.12.14、torch 2.7.1+cu128、Triton 3.3.1。
`uv pip check`：27 个包兼容。GPU RTX 3080 Laptop，SM86，driver 595.91.07。

## 已完成

- BF16 dense GEMM / FP32 累加，支持 M/N/K 尾块。
- 无 M padding 的 grouped GEMM；host 构造 task table，一次 launch 计算多 expert。
- 空 expert、全空输入、偏斜 counts 与小尾块数值检查。
- 完整 MoE 公式的 FP32 reference，含重复 expert route 与 weighted combine。
- CUDA-event 原始样本、输入/源码 hash、PTX/cubin/SASS 和编译资源信息。

## 数值结果

4 个 dense 边界用例、4 个 grouped 用例、1 个 MoE reference 语义用例通过。
另有 5 个计时形状同时通过 FP32 oracle 对照。BF16 用例的最大 relative RMS
为 0.002213，最大 normalized max 为 0.003505；阈值事先设为两者 <= 0.008。
MoE FP32 reference 与独立逐 route 计算的最大绝对差为 1.91e-6。
这些检查还不覆盖完整优化 MoE 层、真实路由分布或模型质量。

## 诊断性能（p50，毫秒）

| 形状 | Triton | PyTorch | PyTorch 对照含义 |
| --- | ---: | ---: | --- |
| dense M=N=K=512 | 0.02392 | 0.01685 | 单次 torch.mm/cuBLAS |
| dense M=N=K=1024 | 0.09344 | 0.14295 | 单次 torch.mm/cuBLAS |
| grouped E8、每 expert M32、N512/K256 | 0.03123 | 0.12826 | Python expert 循环 |
| grouped E8、偏斜含空 expert、总 M256、N512/K256 | 0.03074 | 0.07194 | Python expert 循环 |
| 合成 Qwen-like E128、每 expert M4、N1536/K2048 | 1.99574 | 2.49366 | Python expert 循环 |

这是 hot repeated input、预分配输出、含 Python 提交间隙的 API 诊断，
不是纯 kernel profiler 时间。三轮 AB/BA/AB，10 次 warmup，每轮 12 样本，
每样本 4 调用，禁用 TF32 和 cuBLAS BF16 reduced-precision reduction。
排除编译、plan 构造、routing/dispatch、SwiGLU、down 和 combine。
小 grouped case 的 p95 明显高于 p50，不能由这些数字认定稳定硬件上限。
没有锁定频率或独占 GPU；GPU 末温 48°C，SM clock 1785MHz。

512 dense 暂时慢于 cuBLAS。1024 单形状的较快结果不代表通用 GEMM 优势。
grouped 只对照 Python expert 循环，不能据此宣称超过成熟 MoE backend。
每 expert 仅 M4 的场景在 BM32 tile 上存在大量无效计算，是后续重点。

## Tensor Core 与分析边界

五个计时形状的 PTX 均有 `mma.sync`；SASS 均有 BF16 HMMA 指令。
grouped baseline：56 registers、12KiB shared；Qwen-like：48 registers、12KiB shared。
这些是静态编译信息，HMMA 数量不是动态执行次数，也不是 utilization。
当前没有 `ncu`；驱动 `RmProfilingAdminOnly=1`，未获取硬件性能计数器。
实际 Tensor Core 利用率、occupancy、带宽与 cache hit 指标仍待 profile。

## 保存位置与下一步

当前结果：[bootstrap-20261006T080359Z-61205081.json](../results/bootstrap-20261006T080359Z-61205081.json)。
经 gpt-6-luna/max 子 agent 审查，修复大张量 pointer offset 的 int32 中间
乘法风险，提升地址计算至 int64 后重新跑完整 bootstrap；旧版本
[075818Z 结果](../results/bootstrap-20261006T075818Z-108833022.json) 继续保留。
另外修复 numerical checker 可能静默 broadcast 的漏洞，先检查 shape 完全一致；
[080110Z 结果](../results/bootstrap-20261006T080110Z-633691196.json) 为该检查前版本。
第一轮 [bootstrap.json](../results/bootstrap.json) 原样保留；它未关闭 cuBLAS
BF16 reduced-precision reduction，因此不是上表使用的正式 bootstrap 对照。
编译产物位于 `artifacts/<run_id>/<case>/`（Git ignored，本机保留），
结果 JSON 保存 hash、HMMA、registers/shared 与全部原始时间样本。
代表形状的 [PTX](../results/codegen/qwen_shape_grouped.ptx) 与
[SASS](../results/codegen/qwen_shape_grouped.sass) 纳入 Git，保留可移交的 HMMA 证据。

下一阶段先固定兼容 SM86 的成熟 backend，再比较小/大 routed rows、
均匀/偏斜、top-k2/4/8 和空 expert，随后依据 profile 调整 tile/调度与融合。
完整 MoE GPU 链路与 GQA attention 均尚未完成。
