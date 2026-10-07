# M21：先归因，再选择 CUDA / 量化实现

2026-10-07，测量前冻结。本轮是有限的机制诊断，不修改 M7 policy、旧 kernel 或 M20 发布清单。
用户要求加强算子深度，优先建立瓶颈证据，再做一个手写 CUDA 核心和 weight-only 量化融合。

## 固定负载与对照

模型权重和真实输入来自 M20 pinned Qwen3-30B-A3B layer0 attempt3 artifacts。
四个诊断负载：archive-T17-real、archive-T1024-real，以及同一个 archive-T1024 的 X、
weights、route weights 不变，仅将 IDs 改为 balanced128 或 hot16 的 modular distinct 路由。
替换路由是机制实验，不称为真实 router 输出。balanced/hot 的 slot ID 分别为 arange(T*8)%128 / %16。

三个 arm：原 ranked 默认 BM32/BN64/BK32/W4/S3；冻结 M7 shape policy；诊断 BN128，
保持 BM32/BK32/W4/S3。T17 的前两者配置相同，重复 arm 只检查测量一致性，不视为优化。
BN128 不是新冻结 policy，不能由这一批诊断输入决定或声称新 holdout 收益。

## 测量与正确性

- 单 GPU SM86 / 本仓库 venv / BF16 / 禁用 TF32 与 BF16 reduced reduction / 8 CPU threads。
- source、policy、manifest/trace/权重哈希在执行前后核对，输入 hash / actual counts / source HEAD 全部保存。
- 完整前向使用既有 FP32 oracle、finite、relative RMS 与 normalized max≤0.008，检查 ranked dispatch。
- 无 profile 的 perf phase 记录三 arm API 和 Graph 完整成本，沿用 forward/reverse cyclic 轮次，
  Graph 每 replay 8 forward。它是诊断测量，不是最终性能验收，不引入新的 unit tests。
- 独立 profile phase 只运行一个 case/arm，预热后捕获 8 个完整 7-launch forward。
  在每个实际 prepared launch 外加 NVTX，不改变 kernel，不在 ROI 回读 counts、hash 或计算参考。
- Nsight Systems 捕获 CUDA/NVTX 时间线，CPU sampling / context switch 关闭，不输出环境元数据。
  原始 nsys-rep / SQLite 只保留本地 ignored artifacts，分享固定白名单解析出的阶段统计。
- Nsight Compute 只对选定实际 gate/up 或 down kernel 采集。驱动当前管理员权限限制未解除；
  如果采集拒绝，保留拒绝记录并由用户完成管理员操作，不在后台变更驱动权限。

## 判读与后续设计

比较 histogram/prefix/dispatch/gate-up/down/combine 的 GPU 时间占比；CPU NVTX 是提交区间，
不能当作 GPU 执行时间。profile 带来扰动，不能作为净性能 headline。

BN 增大可能减少 N 方向任务数、复用 A，同时增加双累加器和资源需求；先观察固定输入下的
时间及静态资源，再用 counters 判断瓶颈，禁止从 spill 的存在直接推导其造成所有回退。
hot16 与 balanced128 改变活跃 expert 数、padding/task 分布和权重工作集，不能单独称为 atomic 对照。
分别记录 useful tasks、上界 grid、tile padding 和解析出的阶段成本，以区分这些因素。

CUDA 核心与量化格式在诊断后另立方案。量化正确性相对同一量化权重 oracle；对原权重的损失、
模型任务质量另外评估。推理项目已接受“各主要任务相对 dense 得分下降≤10%”，不放宽算子实现误差。
