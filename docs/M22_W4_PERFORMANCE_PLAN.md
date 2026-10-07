# M22 W4A16：首版完整链路性能协议

运行前声明。只有六个真实 case、七状态、T0 的 W4 CUDA 审计全部通过，才测性能。
保持 W4G128 packed 格式与同步 BM32/BN64/BK32 CUDA 首版，不在本轮选择 tile。

## 两种问题与对照

相同量化算术的四个完整 forward arm：

- `quant_vllm`：同一 packed 权重预先还原成 BF16，官方 vLLM0.10.0 未修改公共 API。
- `quant_frozen`：相同还原 BF16 权重，冻结 M7 ranked Triton 配置。
- `quant_cuda_sync`：相同还原 BF16 权重，M22 同步 CUDA BM32/BN64/BK32。
- `w4_sync`：tile 内在线解码相同 packed 权重，融合 gate/up 与 SwiGLU，BF16 down 不变。

前三项还原在计时外且 BF16 权重常驻 GPU；它们是强的缓存对照，不每次重复 Python
解量化。相同量化权重的独立 FP32 oracle 用原 0.008 finite/RMS/normalized-max 门槛。
`quant_cuda_sync` 与 `w4_sync` 的算术 tile、7-launch DAG、down config 相同，用于隔离
存储格式与解码的组合影响。M7 的大 BN 与 vLLM 舍入/API 差异沿用已有说明。

另列 `bf16_vllm` 与 `bf16_frozen`：原 BF16 权重，分别对原 FP32 oracle 检查。
它们回答带格式损失的延迟取舍，不能标成同精度比较。按输入报告量化 oracle 与原
oracle 的差异；权重误差和 layer 输出误差都不是完整模型任务得分。

vLLM0.10.0 的 fused Marlin MoE 接口为 gate/up 和 down 共用量化类型，而这里 down
保持 BF16，本轮没有直接相同域的 Marlin 完整链路 arm。因此结果不代表胜过成熟
W4 MoE 后端；BF16 public fused_experts 仍按公共自动配置执行。

## 执行与保存

用户调整交付优先级后，在首次成功计时前收敛为单批：六个 pinned M20
archive/code × T17/193/1024，每 case 一个独立新进程，升序/archive-code交替。
这是首版简历交付测量，不声称三批稳定性。单 GPU 串行，不并行 packing/编译/profiler，
无锁钟或驱动设置更改。prepared buffers 固定；packed words/scales immutable。

每 worker 六 arm 全链路 API/Graph，三个本项目量化 arm 另列已分组 gate/up 阶段
API/Graph（vLLM 公共完整 API 没有对应阶段入口）。采用六轮正逆
循环、6 samples/round、2 calls/sample；Graph replay 内8 forward，以 forward/launch
归一化。阶段不含 routing/down/combine，不拿独立分段相减估计全链路瓶颈。

保存所有样本、p50/p95、数值/dispatch 前后与实际 Graph 输出检查；packed/input/
artifact/source/audit/binary hashes 前后稳定并与已审计版本一致。还原、上传、编译和
oracle 都在计时外；这不是冷启动或 serving TTFT benchmark。

首个 v1 worker 在计时前因默认 cache 新编译的 BF16 `.so` hash 不匹配而停止，
保留失败；原审计 cache 中的 BF16 binary 仍完好。v2 明确分别指定 BF16/W4 审计
对应的 extension root，加载前检查 cache 内 binary SHA、加载后再核对实际 path/SHA。
不重编译后静默接受新 hash。两份 audit 与两个 kernel source 均不改，预先定义的
计时 arms、输入和数值门槛不变；进入计时后冻结 v2 benchmark 源与本协议。

报告每个 case 的配对比值，再 equal-weight 几何平均及最坏回归，不挑最快单次。
packed gate/up 192MiB、scale6MiB、原 gate/up768MiB、down384MiB 为逻辑权重字节；
性能 worker 同时持有各对照权重/Graph pool，不用其 allocator 峰值证明单路径显存节约。
保留失败和负收益。没有完整模型评测，不声明满足任务质量下降≤10%。
