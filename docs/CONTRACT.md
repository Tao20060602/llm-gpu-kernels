# M0/M1 计算契约

首版仅 forward、单 GPU、连续 BF16 输入与输出、FP32 累加，无 bias/backward。
N/K > 0，M 可为 0。输入不修改，out 不得与输入共享 storage。
跨 stream 的 metadata 构造/使用由调用者显式同步；首轮只用默认 stream。
未验证 CUDA Graph、非连续 stride、线程并发与多卡。

## Dense calibration

`dense_gemm(A[M,K], B[K,N]) -> C[M,N]`，C = A @ B。
这只是 Tensor Core 基础校准，非核心创新。

## Pregrouped ragged GEMM

`grouped_gemm(A[sum(M_e),K], W[E,N,K], plan) -> C[sum(M_e),N]`。
输入行已按 expert 连续分组，第 e 段计算 `A_e @ W_e.T`。
所有 expert 共享 N/K，可以有空 expert 和尾块；不使用 M padding。
`GroupedPlan.build` 接收 host 已知 counts，生成 int32 offset/count/task 元数据。
plan 和配置的 block_m 必须一致。调用者保持 plan 的 tensor 元数据只读。
这是 host 预分组接口，尚未把 GPU 动态路由分组纳入核心算子。

## 完整 MoE 语义（当前仅 FP32 reference）

X[T,H]，固定 IDs[T,topk] 与 routing weights[T,topk]。
Gate/up 权重 [E,2I,H]，先 gate 后 up；down 权重 [E,H,I]。
输出为所有 route 的 `weight * down(silu(gate(x)) * up(x))` 之和。
重复 expert ID 按两个 route 分别累加。不做 router、softmax 或权重重归一化。
验证 ID 范围和权重 finite；这些同步检查属于 reference，不是热路径性能。

## 首轮数值和计时

BF16 operands 转 FP32 做 oracle，禁用 TF32。预先设定 relative RMS <= 0.008，
max_abs/max_abs(reference) <= 0.008，检查 finite；不使用逐元素 relative error
处理接近零的元素。报告实际误差，不能仅报告通过。

固定 seed、保存输入 SHA256 和源码 SHA256。10 次 warmup，三轮交替 A/B
（AB/BA/AB），每轮 12 个样本、每个样本 4 次调用、CUDA event 时间除以 4。
预分配输出、重复相同输入、没有 CUDA Graph。Python 提交间隙可能计入事件，
特别是 expert 循环；首轮是 API 诊断时间，不是纯 GPU 硬件极限。
排除编译、metadata 生成、routing、dispatch、activation、down 和 combine。
grouped 的 torch expert 循环是弱诊断对照，成熟 backend 对照仍待接入。

保存 PTX/cubin/SASS 与 hash、寄存器和共享内存信息。HMMA 证明编译产物使用
Tensor Core 指令，不能替代 profiler utilization/occupancy 指标。
