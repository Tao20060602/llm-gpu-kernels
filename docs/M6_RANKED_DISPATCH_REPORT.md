# M6 排名式分派：完整前向结果

正式运行于 2026-10-06 通过。以下方法与固定输入遵循预先确定的实验计划；测量值
来自所链接的结果 JSON。

## 方法与实现约定

M6 在相同输入张量上比较两种完整的固定路由前向：

- **Cached fused：** M4 cached fused 启动器。
- **Ranked dispatch：** histogram 为每条 route 分配 expert 内 rank；后续 kernel
  写入 route map，并将 X 复制为按 expert 分组的顺序。

两条路径使用相同的 M4 fused gate/up+SwiGLU、down projection 和 combine kernel，
并采用固定的 M4 tile 配置。对于有 token 的输入，启动数由 cached 路径的 8 次减为
ranked 路径的 7 次。ranked dispatch 为每条 route 增加一个 int32 值；当
`R = T × top-k` 时，逻辑 workspace 增量为 `4R` bytes。空 token 输入两条路径都
不启动 kernel。这些结构差异本身不能证明延迟或内存流量有所改善。

热路径在 GPU 上生成 counts 和 route mapping，不把 counts 读回 CPU。非法 ID 和
非有限 routing weight 会设置 device status bit；显式诊断 `check_inputs()` 在计时
之外读取 status 并抛出异常。基准记录实际 workspace 大小和 kernel 启动数，用于检查
实现约定。

## 正确性检查

预先确定的测试包含 M4 全部 9 个功能场景。每种实现均与独立 FP32 MoE oracle 比较，
要求 relative RMS ≤ 0.008、normalized max ≤ 0.008，且输出必须有限。Cached fused
与 ranked 的输出还必须逐 bit 相等。

每个有效场景都通过 dispatch 诊断将 GPU counts 和 offsets 与 CPU histogram 比较，
并检查 route map 是排列、每条 route 位于正确的 expert segment，且 grouped X 逆向
gather 后等于原 token 行。Ranked 路径还会检查每个 expert 的 ranks 从 0 开始、唯一
且连续，并验证每个 map 项等于对应 expert offset 加该 route 的 rank。功能矩阵中有
两个重复 ID 场景；每个场景都额外重跑 3 次，以检查 atomic 分配顺序变化时的映射与
结果。

指定的 graph-update 场景在原地址不变的 graph 中，两次原地更新 X、IDs 和
routing weights 后重放；同时还检查非默认 CUDA stream 上的执行。额外的非法输入检查
要求两种实现都由 `check_inputs()` 抛错：越界 ID 对应 status 1，非有限 routing
weight 对应 status 2。只有功能、graph、stream、dispatch、rank 和 status 检查全部
通过后才开始性能计时。

## 性能测试协议

固定矩阵共 15 个场景：small E16/H128/I64/top-4 随机路由，T=1/17/193，seed 73；
Qwen3-30B-A3B 形状 E128/H2048/I768/top-8 随机路由，T=1/17/193/1024/2049，
seed 73 和 991；以及两个 E128 hot16 场景，T=193/1024，seed 73。“Qwen 形状”只
表示维度相同；实验使用生成的输入和权重，不使用模型权重，也不测量模型推理。

每个场景都把相同的 tensor 对象交给两种实现。准备、编译、oracle 计算和 warmup
不计入计时。API 与 Graph 模式分别采用 4 轮平衡顺序 `AB / BA / BA / AB`；每轮每种
实现采集 8 个样本，每个样本调用 2 次。每次捕获的 Graph replay 含 8 次完整前向，
Graph 测量值除以 8。API 计时包括 Python 启动间隙；Graph 计时也可能包括 replay
提交间隙。结果 JSON 保留每轮全部 event 与 wall 样本、p50/p95 汇总、输入哈希、
telemetry、workspace 大小、源码哈希和正确性指标。

## 结果

正式运行通过了 M4 全部 9 个功能场景、两类非法输入 status 检查，以及全部 15 个
性能场景。记录的 79 次 cached-fused/ranked bitwise 比较全部通过。158 项 FP32
oracle 指标中的最大 relative RMS 为 0.0042392，最大 normalized max 为 0.0072424，
均低于固定阈值 0.008。两次 graph 输入更新和非默认 stream 检查均通过。

有 token 的路径使用 8 次 cached launch 和 7 次 ranked launch；空 token 场景两者都为
0 次。所有有 token 场景的 workspace 增量都等于 `4R`：例如 small T=1 时为 16 bytes，
Qwen 形状 T=1024、top-k 8 时为 32,768 bytes。这是新增 rank tensor 的逻辑存储，
不能据此推断总分配量或实测内存流量下降。

运行环境为 Python 3.12.14、PyTorch 2.7.1+cu128、CUDA 12.8、Triton 3.3.1，以及
NVIDIA GeForce RTX 3080 Laptop GPU（SM 8.6）。记录到的最大 allocated memory 为
1.73 GiB，包含输入、workspace、graph pool 和正确性 oracle。运行记录包含预先确定的
实验计划 SHA256 和 17 项相关源码哈希；运行结束后再次核对，17 项记录哈希均与对应源码一致。

正式记录：[m6-ranked-dispatch-20261006T102940Z-41281076.json](../results/m6-ranked-dispatch-20261006T102940Z-41281076.json)。

### API 和 Graph p50（毫秒/完整前向）

表中列出全部 15 个固定场景。变化率为 `(ranked / cached − 1) × 100%`；负值表示
该场景的 ranked dispatch p50 较低。

| 场景 | API cached_fused | API ranked | API变化 | Graph cached_fused | Graph ranked | Graph变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small_E16_T1_random_seed73 | 0.039424 | 0.035328 | −10.39% | 0.014064 | 0.013056 | −7.17% |
| small_E16_T17_random_seed73 | 0.038400 | 0.034312 | −10.65% | 0.014272 | 0.013376 | −6.28% |
| small_E16_T193_random_seed73 | 0.038400 | 0.034528 | −10.08% | 0.016000 | 0.014813 | −7.42% |
| qwen3_30b_a3b_T1_random_seed73 | 0.210928 | 0.207360 | −1.69% | 0.195101 | 0.194208 | −0.46% |
| qwen3_30b_a3b_T17_random_seed73 | 1.909808 | 1.908992 | −0.04% | 1.896192 | 1.896058 | −0.01% |
| qwen3_30b_a3b_T193_random_seed73 | 3.208712 | 3.211264 | +0.08% | 3.195288 | 3.197664 | +0.07% |
| qwen3_30b_a3b_T1024_random_seed73 | 5.111752 | 5.119224 | +0.15% | 5.102589 | 5.113600 | +0.22% |
| qwen3_30b_a3b_T2049_random_seed73 | 7.556352 | 7.537152 | −0.25% | 7.576288 | 7.569181 | −0.09% |
| qwen3_30b_a3b_T1_random_seed991 | 0.208832 | 0.206024 | −1.34% | 0.194202 | 0.193440 | −0.39% |
| qwen3_30b_a3b_T17_random_seed991 | 1.871360 | 1.867736 | −0.19% | 1.856191 | 1.855710 | −0.03% |
| qwen3_30b_a3b_T193_random_seed991 | 3.211520 | 3.212288 | +0.02% | 3.198300 | 3.199808 | +0.05% |
| qwen3_30b_a3b_T1024_random_seed991 | 5.115648 | 5.123840 | +0.16% | 5.099641 | 5.106107 | +0.13% |
| qwen3_30b_a3b_T2049_random_seed991 | 7.600856 | 7.571992 | −0.38% | 7.614240 | 7.590592 | −0.31% |
| qwen3_30b_a3b_T193_hot16_seed73 | 0.749312 | 0.736512 | −1.71% | 0.734624 | 0.728123 | −0.88% |
| qwen3_30b_a3b_T1024_hot16_seed73 | 3.226368 | 3.219712 | −0.21% | 3.239616 | 3.220064 | −0.60% |

### small 场景 API p50（毫秒/完整前向）

| 场景 | API cached_fused | API ranked | API变化 |
| --- | ---: | ---: | ---: |
| small_E16_T1_random_seed73 | 0.039424 | 0.035328 | −10.39% |
| small_E16_T17_random_seed73 | 0.038400 | 0.034312 | −10.65% |
| small_E16_T193_random_seed73 | 0.038400 | 0.034528 | −10.08% |

代表性 p95（毫秒/完整前向）：

| 场景 | API cached_fused | API ranked | Graph cached_fused | Graph ranked |
| --- | ---: | ---: | ---: | ---: |
| small_E16_T193_random_seed73 | 0.044706 | 0.035869 | 0.016064 | 0.014848 |
| qwen3_30b_a3b_T17_random_seed73 | 1.915103 | 1.917596 | 1.898867 | 1.898927 |
| qwen3_30b_a3b_T2049_random_seed73 | 9.318839 | 8.468249 | 7.720284 | 7.660061 |
| qwen3_30b_a3b_T1024_hot16_seed73 | 3.231188 | 3.225037 | 3.272534 | 3.263202 |

这些 p95 显示了样本尾部的波动，尤其是 T2049 API 场景；它们不替代也不筛除原始
样本。全部 15 个场景的 p95 汇总和逐轮样本都保留在 JSON 结果中。

### 正确性、资源与可复现性

- 正式功能/性能状态：通过，9/9 个功能场景、15/15 个性能场景。
- FP32 relative RMS 最大值 / normalized max 最大值：0.0042392 / 0.0072424；所有检查均通过。
- Cached 与 ranked 的 bitwise 比较：记录的 79 次比较全部通过。
- 实际启动数与 workspace 增量：有 token 时为 8→7 次启动、`4R` bytes；T=0 不启动 kernel，也不增加 rank workspace。
- 实验计划 SHA256：`41caa47e3dff0d9a23238a89d7a13c82c163ae21de8770a787136d75829c9b8e`；运行结束后核对的 17 项源码哈希全部一致。
- 结果 JSON：见上方链接。

## 结果解读

三个 small Graph 场景的 p50 低了 6.28–7.42%；small API p50 低了
10.08–10.65%。在本机的这些 small 固定形状上，观察到的方向一致。所有 Qwen 形状
Graph 场景的 p50 变化都小于 1%，且变化方向有正有负，现有结果不足以确认有明确
收益。本实验没有执行正式的统计等效性检验，也不据此声称两者统计等效。Qwen API
p50 在多数场景中变化小于 1%；T=1 随机路由和 T=193 hot16 场景的 p50 降幅较大，
表中也有小幅回归。整体结果不足以支持 Qwen 形状上的普遍提升结论。

这些测量比较的是此 GPU 和软件环境下、使用生成输入与权重的完整固定路由前向。
它们不能证明 serving 延迟、实测 DRAM 流量、模型级性能，也不能证明优于成熟 MoE
backend。
