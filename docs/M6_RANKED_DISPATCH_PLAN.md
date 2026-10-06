# M6 Ranked Dispatch 固定实验计划

## 目的与边界

比较 M4 的 cached fused 入口和 ranked-dispatch 入口在相同输入上的完整
fixed-route MoE forward。M5 已独立覆盖成熟后端，本实验无需 vLLM 运行时，也不
调用 vLLM 算子。两条路径共用 M4 的 fused gate/up+SwiGLU、down GEMM 和 combine；变化
限于路由分组：ranked 路径把 local-rank 分配并入 histogram，再把 route map
写入与 X copy 合并。

预期活跃输入由 8 次缓存 launch 减为 7 次，并为每条 route 增加一个 int32
rank（4R bytes）。这是结构预期，不是耗时、实际DRAM流量或总显存收益结论；
M4 其他浮点计算kernel和tile配置保持不变。结果只适用于本实验的 GPU、版本、
输入和计时边界，不代表 serving 延迟或生产部署表现。

## 固定实现与输入

- A：`prepare_moe_launcher(..., fused=True)` 的 cached fused M4 forward。
- B：`prepare_ranked_moe_launcher(...)` 的 ranked forward。
- 两者收到同一组五个 tensor 对象；计时期间 shape、地址和值固定。
- BF16 输入/权重、FP32 routing weights，TF32 与 BF16 reduced-precision
  reduction 关闭。FP32 oracle 阈值固定为 relative RMS ≤ 0.008、normalized
  max ≤ 0.008，且输出必须有限。A 与 B 另要求逐元素 bitwise 相等。
- 使用 M4 的全部 9 个功能 case。包括 T=0、尾块、空/热点 expert、重复 IDs、
  int32/int64 IDs、非二次幂 expert 数、signed weights。duplicate case 对同一
  输入额外重跑 3 次，以检查 atomic local-rank 顺序变化下的映射和结果。
- 每个有效功能 case 都检查 CPU histogram / offsets、map permutation、
  per-expert segment、rank 与 map 的对应关系、每个 expert 的 rank 集合为
  `[0, count)`，以及 grouped X 逆向 gather 与原输入一致。
- 指定的 graph-update 功能 case 在固定地址 graph 重放两次更新后的 X、IDs
  和 routing weights；另以相同 prepared buffers 在非默认 CUDA stream 上运行。
- 额外执行非法 ID 和非有限 routing weight 两类状态检查。两种实现均须由
  `check_inputs()` 报错，GPU status 分别为 1 和 2。

性能矩阵固定为 15 项，不删慢例、不因结果调整配置：

| 组 | E | H | I | top-k | T | route | seed |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| small | 16 | 128 | 64 | 4 | 1, 17, 193 | uniform random | 73 |
| Qwen shape | 128 | 2048 | 768 | 8 | 1, 17, 193, 1024, 2049 | uniform random | 73, 991 |
| Qwen hot16 | 128 | 2048 | 768 | 8 | 193, 1024 | 16-expert hot | 73 |

Qwen shape 仅指与 Qwen3-30B-A3B MoE 层一致的维度，不加载模型权重，也不代表
真实模型推理。所有数据均由固定 seed 生成。

## 验证与计时顺序

1. 创建结果 JSON 并保存 `running` 状态、实验计划 SHA256、相关源码 SHA256、
   Python/torch/Triton/CUDA 和 GPU 信息、limits 与计时设置。失败记录保留在
   `results/`。
2. 先运行完整功能矩阵、graph 更新、duplicate 重跑、stream 检查及非法输入
   状态检查。任何验证失败都会将整体标记为 failed，并在开始性能矩阵前终止。
3. 每个性能 case 只创建一组输入并把相同输入对象交给 A、B；记录原始输入与
   route hash。准备与预热不计时。捕获每条路径各自的 CUDA Graph，每次 replay
   包含 8 次完整 forward。
4. 每种计时模式使用 `optimization_benchmark.measure()` 的双实现平衡顺序：
   4 轮 `AB / BA / BA / AB`，每轮每实现 8 个样本，每样本调用 2 次。API
   模式含 Python 提交间隙；Graph 模式按 8 次 forward 归一化，仍可能含 graph
   replay 提交间隙。报告 p50/p95、wall p50 和每轮全部 event/wall 样本。
5. API 和 graph 计时后分别重跑并检查 FP32 oracle、A/B bitwise equality、
   dispatch/rank invariants 与输入 hash。记录开始/结束 telemetry、peak allocated
   bytes、workspace bytes、launch 数、rank workspace 增量和错误 traceback。

## 解释限制

不调 kernel tile、不扫描参数、不使用 holdout 选型。计时比较的是完整本项目
fixed-route forward，不隔离纯 dispatch 延迟。新增 4R bytes 是 rank tensor 的
逻辑 workspace 增量；总 workspace、runtime throughput 或 memory traffic 的变化
必须以实际记录为准。若 FP32 阈值、dispatch/rank 检查、状态检查或 bitwise
对照失败，不报告性能结论，保留失败 JSON 供后续诊断。
