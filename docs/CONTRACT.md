# M0/M1 计算契约

首版仅 forward、单 GPU、连续 BF16 输入与输出、FP32 累加，无 bias/backward。
N/K > 0，M 可为 0。输入不修改，out 不得与输入共享 storage。
跨 stream 的 metadata 构造/使用由调用者显式同步；首轮只用默认 stream。
M0/M1当时未验证 CUDA Graph；M2/M3补充固定图和stream验证见下文。
非连续 stride、线程并发与多卡不在当前范围。

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

## 完整 MoE 数学语义

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

## M2 对照契约

CUTLASS3.8 classic SM80-template，编译目标 SM86/device0，BF16/FP32 accumulate。
CTA64x128x32、warp32x64x32、instruction16x8x16、3 stages、align8、DeviceOnly。
要求 N/K 为8倍数，A/W/out 的基址16-byte aligned；拒绝不支持的连续偏移 view。
所有 M 可以 ragged/0，不做 M padding。和 Triton 比较时只取共同支持的形状。
plan 绑定具体 buffers，准备时构造/上传元数据；run 不重建plan或分配GPU输出。
首次准备和跨 stream 使用前显式完成数据依赖；不允许并发写同一 out。

两种计时都使用四轮 AB/BA/AB/BA，每轮8样本，每样本4调用。
API-mode 仍包含 Python 调用、backend wrapper 检查和提交间隙。
Graph-mode 每张固定图含8次 GEMM，按 GEMM 归一化；重复图提交的间隙仍可能计入。
记录 `samples_ms_per_graph_replay` 和 `samples_ms_per_gemm` 以明确单位。
只验证固定预分组 GEMM 的图重放，不代表动态路由或完整 MoE 可图捕获。
未做 tile sweep / 后端调优；本轮结果是两个固定配置的对照。

## M3 prepared / 选择规则

`prepare_grouped_gemm` 绑定全部输入/output/plan owners、固定shape和指针，
持有Triton3.3.1 CompiledKernel runner。输入values可变，buffers不可resize/替换，
metadata只读；调用者保证跨stream依赖、同一output无并发写。
runner每次获取当前stream。M3已验证非默认stream输入更新及固定图重放。

FrozenPolicy只适用N1536/K2048，按host已知counts平均活跃行数选3组tile；
其他shape回退原配置。所有选择发生在计时前，不包括counts readback或动态router。
BF16/FP32语义和误差阈值不变。详见M3_EXPERIMENT_PLAN与M3_OPTIMIZATION_REPORT。

## M4 GPU route forward

`prepare_moe` / `prepare_fused_moe` 接收上述X/IDs/routing weights及BF16权重，
返回绑定shape/buffers的workspace。IDs为CUDA int32/int64，routing weights为
CUDA FP32；输出BF16。原地更换values/IDs允许，不resize/替换buffer，不并发run。
当前prototype E<=256、topk<=32、N/H/I正数，T可0；连续同device输入。
vector中间元素数受int32限制，指针乘法为int64。counts由GPU在每个run重新构造，
所有workspace在prepare分配；run无tensor.item/CPU counts readback或GPU allocation。

gate/up累加FP32后BF16；SwiGLU在FP32运算后BF16；down累加FP32后BF16；
combine按原slot顺序FP32加权求和后BF16。重复expert IDs不合并，权重无重新归一化。
融合版本保留gate/up的BF16 round-trip后做相同SwiGLU，projected buffer仍分配。
调度使用GPU task offsets及固定上界grid，不按每expert最大容量做矩阵padding。

调用者保证IDs属于[0,E)、weights finite，并建立跨stream依赖；首轮完整链路只验证
默认调用stream与固定Graph。非法ID做地址mask并置device status，非finite也置status；
显式check_inputs同步检查，不能把非法输入下返回值当合法数学输出。
图重放已验证同shape/地址下更换输入、权重与路由值；不表示动态shape可共用一张图。

## M5 完整成熟后端 / 缓存入口

`prepare_moe_launcher(..., fused=True/False)`固定绑定Triton3.3.1编译产物、
完整参数和所有buffer owners，fused为8次launch、unfused为9次。
shape/address固定，输入值/IDs/weights可原地更新；当前stream每次调用获取。
不并发使用同一workspace，调用者显式建立跨stream依赖。
M5验证cached两种路径逐bit相同、非默认stream及更新后的Graph。

成熟对照调用未修改官方vLLM0.10.0 `fused_experts`，固定版本与默认自动config。
vLLM native activation、down权重乘法和topk2/3/4 reduction的BF16舍入顺序
与本项目不同；两者各自对完整FP32参考使用原0.008阈值，不要求彼此逐bit一致。
完整算子不包含router或serving。公共API有临时tensor分配，本项目cached入口
预分配；Graph模式缓存各自pool。计时包含这些不同接口成本，分别报告。
15性能场景各自两次原地改X/IDs/weights验证Graph，再恢复原输入hash计时；
Graph每replay8次完整forward，六轮正/逆循环，6样本/轮、2调用/样本。
保留逐forward归一化的全部event/wall原始样本、p50/p95及计时后数值检查。
