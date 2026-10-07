# M20：真实 Qwen3 MoE layer-0 输入下的算子重放

日期：2026-10-07。这里的 M20 指 `llm-gpu-kernels` 独立算子项目，不是 NanoKV 的 M20。

状态：真实输入与路由采集已通过；三次新进程复测均通过。结果以本报告结果章节
及对应原始记录为准，失败尝试单独保留。

## 范围

本实验把固定公开 checkpoint 的真实 layer-0 MoE 输入、router 结果和 expert 权重接入已有
固定路由算子对照。它回答的是限定 checkpoint、两条固定示例文本、四种前缀长度及 layer 0
上的单层 operator 问题。它不测完整 30B 模型输出或质量，不包含模型服务、请求延迟、用户流量、
decode、prefill 全层成本或多卡 expert parallel。

M7 仍是独立的合成输入实验：随机 hidden/权重、人工 IDs 和 routing weights；其冻结 policy、
training、holdout 和源码均未由 M20 改写。M20 使用 M7 policy 原样选择配置，但不得将 M7 的
合成路由结果说成真实模型结果，也不得用本轮 trace 重新调 M7 policy。

## 来源与可追溯性

模型为 Hugging Face `Qwen/Qwen3-30B-A3B`，固定 revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`。固定配置为 `E=128, H=2048, I=768, top-k=8`，
`norm_topk_prob=true`。配置 SHA256 为
`2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297`，safetensors index
SHA256 为 `df0d481ec595c55a0ba58426d517390c6214a566ec4ff1c8fc4bbce9f57b3c24`。

只从 pinned checkpoint 提取 layer-0 所需的 394 个 BF16 张量，保存于仓库外
`/home/tmz/models/Qwen3-30B-A3B-layer0/`。它包括 embedding、layer-0 attention/norm/router
和 128 个 expert 的 gate/up/down，tensor payload 共 1,868,571,136 字节；未下载或装载完整
30B checkpoint。来源分片为 `model-00001-of-00016.safetensors`，分片长度
3,999,417,504 字节。下载器以 HTTP Range 获取所选区间并逐块核对响应和哈希；提取文件
`layer0.safetensors` 的 SHA256 为
`e8cc81d146128babd7279caf08ebbd50347a5462c173387a865a135fcac59c76`，提取 manifest
SHA256 为 `2e5553fd186cf32b614c3b774465d6204fd5a4bbaa4c855c8e541ca2c10e8ccc`。manifest 中的
源 shard header hash 只覆盖 SafeTensors header；分块和 tensor hash 只覆盖实际提取的数据，
这些都不是完整源 shard 的校验和。

采集使用同一 revision 的 tokenizer。`tokenizer.json` SHA256 为
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`，
`tokenizer_config.json` SHA256 为
`d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`。完整来源、API 响应、
各 trace manifest、token IDs、输入和权重文件大小/hash 记录在 capture summary 及其
8 个 case manifest 中，小型来源记录已归档于 [m20-provenance](../results/m20-provenance/)。

采集环境记录为 Python 3.12.14、PyTorch 2.7.1+cu128、Transformers 4.53.2、Safetensors 0.8.0，
attention backend 为 Transformers eager。summary 记录 tokenizer、checkpoint artifact、
capture 脚本、trace writer、Qwen3-MoE/config、masking 和 RoPE 源文件的采集前后 SHA256；
采集前后哈希相同。

| 证据 | 路径 | 状态/摘要 |
| --- | --- | --- |
| Layer-0 提取 | `/home/tmz/models/Qwen3-30B-A3B-layer0/manifest.json` | `passed`；revision、配置/index、header、chunk、tensor 与 artifact hash |
| Capture summary | `/home/tmz/ai/traces/m20-qwen3-layer0-20261007-attempt3/capture-summary.json` | `passed`，SHA256 `163be24068637aa6c0d29ca28fcc47bef0e805d6583798ddd5a4908a6471d61c` |
| Expert bundle | 同目录 `weights.pt` | 仅 `gate_up` 和 `down`；SHA256 `e703f1ea866455f43685a2f0fdd15055e40efb4f2086c876a812c3498317c7c1` |
| 8 个输入 trace | 同目录 `traces/{archive,code}-T{17,193,1024,2049}.{json,pt}` | 每项各有 manifest、tensor/content hash、token IDs 与路由输入 |

大型权重与 trace tensors 位于模型/trace 目录，不随 Git clone 自动提供。上述哈希用于绑定该次本地 artifact；
报告中的 hash 本身不代表后续机器已重新读取或重验这些外部文件。

## 输入构造

两条固定的示例 seed 文本标记为 `archive` 和 `code`，分别是短的记录摘要说明和 Python
代码片段；它们不是用户请求或生产流量。采集脚本用 pinned fast tokenizer、`add_special_tokens=false`
编码 seed，重复 seed 文本直到达到目标 token 数，再取前缀，得到每条文本的 `T=17、193、1024、2049`
四个 case。每个 case 的完整 token IDs、tokenizer hash 和构造来源写入 trace manifest。因此这些是
真实 tokenizer/checkpoint 下的确定输入，但长度扩展使用重复示例文本，不能解读成自然生产上下文分布。

layer-0 输入来自 pinned Transformers `Qwen3MoeDecoderLayer` 路径：真实 BF16 token embedding、
input RMSNorm、Q/K/V/O 投影、Q/K head norm、position IDs、RoPE 和 causal mask、attention 残差、
post-attention RMSNorm。保存的是 post-attention norm 后送入 router 的 `X[T,2048]`，不是
embedding 或 layer 的原始输入。

采集保留真实 router gate。对其 BF16 logits 执行 FP32 softmax、top-k 8、按配置
`norm_topk_prob` 归一化，再转回 BF16；artifact 将该 BF16 route score 精确提升到 FP32 保存，
benchmark 不重新归一化或重排。expert IDs 保存为 int32。采集替换了专家 MLP 执行环节的仅路由
stub：stub 记录真实 X 和 router logits，再返回零 expert output。**128 个 expert 没有在 capture
中运行；decoder layer 的 MLP 输出、层输出和后续模型状态不是模型真实输出。**

权重提取文件保留 checkpoint 原有独立 gate/up/down keys。capture 将每个 expert 的 gate 放在
up 前拼为 `gate_up[128,1536,2048]`，down 组成 `down[128,2048,768]`；router `gate.weight`
与 expert `gate_proj.weight` 是不同参数。

采集结果 `status=passed`，包含两条示例文本与四种长度共 8 个 case。具体 token IDs、route rows
和对应 hash 以 attempt3 capture summary/manifests 为准，不在本报告中重复展开。

## 固定路由对照方法

benchmark 将相同的 BF16 X、expert IDs、精确晋升的 FP32 route weights 和真实 layer-0 expert
weights 交给三臂：

- `ranked_default`：M6 原始 ranked 7-launch 路径。
- `ranked_selected`：M7 已冻结的 shape policy 选择；本实验不重调阈值或配置。
- `vllm_public`：官方 vLLM 0.10.0 public `fused_experts` 默认配置；对照来源和运行环境见
  [vLLM runtime provenance](../results/provenance/vllm-0.10.0-runtime.json)。

每一臂单独对同一个完整 FP32 fixed-route oracle 验证，阈值沿用项目契约：输出 finite，
`relative_rms <= 0.008` 且 `normalized_max <= 0.008`。另检查 ranked dispatch/status；Graph
在固定地址下原地换成相同长度的另一条真实 prompt 输入，再恢复原输入并核对 SHA256。保留初始、
更新、恢复及计时后正确性结果。各实现舍入次序可能不同，不以与 Transformers 完整 BF16 MoE
输出逐 bit 相同作为通过条件。

性能矩阵为两条 seed text × 四个 token 数，共 8 项。benchmark 计时前完成加载、编译、warmup、
hash、正确性和 dispatch/status 检查；API 使用六个样本/轮、每样本重复调用两次，按正反循环
backend 顺序采样；CUDA Graph 每次 replay 包含 8 个完整固定路由 expert forward，报告归一化后的
每 forward 数值和原始样本。API 与 Graph 分开解释。核心计时不包括模型/权重下载与加载、文本编码、
layer-0 attention/router、FP32 oracle、dispatch/status 读回、prepare/compile 或 telemetry；API
样本保留各 public/run 入口的 host 调用范围，Graph replay 不含每次 Python 重新绑定成本。

因此 benchmark 若通过，只能回答上述固定 trace 上的 GPU-resident fixed-route 单层 expert forward
表现。它不是包含 router 的完整 MoE 子层时间，更不是 layer-0 latency、模型 token latency 或服务
吞吐。若未来测 router+experts，必须将该组单独计时且对三臂统一边界。

benchmark runner 将真实 case 的 PTX/cubin/SASS、代码和资源信息绑定到结果；其中 MMA/HMMA
只能作为编译产物中存在 Tensor Core 指令的静态证据。当前没有 M20 的 Nsight Compute 运行时
Tensor Core utilization/occupancy 结论；如 profiling，必须与正常 event 计时分开记录，不能用
profiling 时间作竞速结论。

## 运行失败与保留记录

失败尝试不覆盖或删除；只有成功的 capture attempt3 用于当前 benchmark 输入。

1. 初始 capture 尝试的 tokenizer LFS pointer 校验逻辑错误，不能通过来源校验，没有有效 trace。
2. 后续 capture 尝试把 attention 子模块权重名带上了错误的 `self_attn.` 前缀，Transformers
   state dict 报告 Q/K/V/O 与 Q/K norm 权重缺失并将带前缀键列为 unexpected。修正后，attempt3
   采集通过。前一次错误尝试不作为 capture evidence。
3. benchmark 首次记录保留在
   [`m20-real-input-20261007.json`](../results/m20-real-input-20261007.json)，状态为 `failed`。
   `archive-T17` 的初始数值检查通过后，编译产物查找使用了不匹配的 `gate_up_swiglu` key，触发
   `KeyError` 并中止；它不是完整 holdout，不报告其计时为实验结果。
4. 新 benchmark attempt 2 保留在
   [`m20-real-input-20261007-attempt2.json`](../results/m20-real-input-20261007-attempt2.json)。
   8 项全部通过；之后用两个新进程重复同一组输入。成功记录与派生结果见下一节。

## 结果

三次独立新进程均完成8项，同一组输入与冻结配置；共 **432 项数值指标全部通过**，
最大 relative RMS **0.00286961**，最大 normalized max **0.00410477**，均低于固定0.008阈值。
初始、Graph初始、切换另一真实prompt、恢复、API后和Graph后六个阶段均逐臂核对。
每次都有24组 ranked dispatch 检查（两臂共48个检查结果），输入恢复hash及全部源码/外部artifact前后hash一致。

下表耗时为三个进程各自p50的中位数；减少百分比由三个进程的逐项 selected/vLLM p50比例
取几何平均计算。正值表示耗时减少，负值表示回归，百分比不能由表内四舍五入后的中位耗时重算。

| 输入 | 活跃experts / 最热route数 | API selected / vLLM ms | API减少 | Graph selected / vLLM ms | Graph减少 |
| --- | --- | --- | --- | --- | --- |
| archive-T17 | 59 / 8 | 1.3466 / 1.3857 | +2.81% | 1.3314 / 1.3091 | -1.72% |
| archive-T193 | 91 / 83 | 2.2190 / 2.2508 | +1.37% | 2.2014 / 2.1733 | -1.35% |
| archive-T1024 | 96 / 533 | 3.6851 / 3.9603 | +6.72% | 3.6533 / 3.9237 | +6.63% |
| archive-T2049 | 96 / 1092 | 6.4896 / 6.5436 | +0.80% | 6.2745 / 6.4525 | +2.83% |
| code-T17 | 63 / 7 | 1.4277 / 1.4740 | +3.86% | 1.4134 / 1.3975 | -1.16% |
| code-T193 | 87 / 70 | 2.1409 / 2.1240 | -0.21% | 2.1243 / 2.0580 | -3.17% |
| code-T1024 | 91 / 404 | 3.7307 / 3.7767 | +1.52% | 3.6485 / 3.8034 | +3.84% |
| code-T2049 | 95 / 813 | 6.5408 / 6.5989 | +1.35% | 6.2966 / 6.5102 | +2.77% |

全部24项同权重平均的p50比例几何平均：selected/vLLM **API 0.977007（少2.30%）**、
**Graph 0.988653（少1.13%）**；selected/原ranked为 **API 0.914007（少8.60%）**、
**Graph 0.897438（少10.26%）**。此聚合对每个测试case等权，不代表实际流量加权收益。

**判断：收益集中在T1024及T2049，不能声称全域超过vLLM。** archive-T1024三次Graph
分别少6.99%、6.30%、6.61%；code-T1024少3.20%、4.25%、4.07%。T17与T193的
Graph四项全部回归，最大单次慢3.23%；code-T193的API有两次回归一次小幅收益。
本轮没有针对这8个case调优，也没有据结果改变默认M7 policy或添加胜者选择。

所有臂共享同一真实route counts；例如archive-T2049最热expert有1092个route，而平均
128.0625个。这种真实倾斜与M7随机均匀输入不同，不能把T1024两条文本当成覆盖了真实生产分布。

### 编译产物与资源

每个case的两条ranked臂各保存gate/up+SwiGLU及down的实际launch kernel：
PTX/cubin SHA256、SASS反汇编返回码和HMMA指令计数、registers/shared-memory/spills。
32个case/kernel项（每进程）都存在MMA/HMMA；这只证明静态Tensor Core指令。

| 配置 / 阶段 | registers | shared bytes | spills | SASS HMMA count |
| --- | --- | --- | --- | --- |
| 原配置 / gate-up+SwiGLU | 76 | 20480 | 0 | 16 |
| 原配置 / down | 55 | 12288 | 0 | 192 |
| medium / gate-up+SwiGLU | 255 | 69632 | 6 | 64 |
| medium / down | 168 | 36864 | 0 | 32 |
| large / gate-up+SwiGLU | 255 | 40960 | 0 | 64 |
| large / down | 159 | 24576 | 0 | 32 |

small与原配置相同。medium融合kernel的spill已保留，不能把它无计数器地认定为回归的唯一
原因；没有运行时occupancy、DRAM流量或Tensor Core utilization测量。最大ranked workspace
209,950,792字节；总模型和GPU allocator峰值以逐case原始记录为准。

### 原始与派生记录

- [新进程1](../results/m20-real-input-20261007-attempt2.json)
- [新进程2](../results/m20-real-input-20261007-repeat-2.json)
- [新进程3](../results/m20-real-input-20261007-repeat-3.json)
- [独立CPU审计与派生比例](../results/m20-evidence-20261007.json)
- [旧M7–M19证据复核](../results/m20-historical-integrity-20261007.json)

原始记录保留每臂API/Graph的36个event样本和36个wall样本、六轮正反循环顺序、p50/p95、
workspace、峰值显存、GPU温度/时钟/功耗。CPU审计重算raw p50/p95及全部比例；
外部文件的重新核对与仅记录hash的模式明确分开。

## 阅读与复现

- [M20 输入计划](M20_REAL_MOE_INPUT_PLAN.md)：范围、下载和 Tensor Core 取证计划。
- [M20 trace 契约](M20_REAL_MOE_TRACE_CONTRACT.md)：artifact schema 与安全校验。
- [Capture 脚本](../scripts/capture_qwen3_moe_layer0.py) 与
  [固定路由 benchmark](../src/llm_gpu_kernels/moe_real_input_benchmark.py)。
- [固定 MoE 数学契约](CONTRACT.md) 与 [M7 冻结结果](M7_FULL_MOE_OPTIMIZATION_REPORT.md)。

本轮不改 M7–M18 的源码、artifact、policy 或历史结果，也不做 NanoKV 集成。完整单模型推理质量、
serving、真实请求分布、冷启动、多个模型层和多 GPU/expert parallel 均未在本实验范围内。
