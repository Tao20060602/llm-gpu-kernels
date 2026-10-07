# M20：独立算子项目的真实 Qwen3 MoE layer-0 输入计划

日期：2026-10-07。这里的“M20”是 `llm-gpu-kernels` 的独立算子实验编号，
与 NanoKV 仓库的 M20 无关。

状态：范围、数据源和有限 holdout 契约已确定；layer 权重下载、真实输入构造和
GPU 基准尚未执行。这个计划不改写 M7 的训练记录、冻结策略、holdout 或其源码哈希。

## 目标和可支持结论

M7 的 `E=128, H=2048, I=768, top-k=8` 与 Qwen3-30B-A3B 一层的维度相同，但
M7 使用随机 BF16 expert 权重、随机 hidden state、人工 IDs 和非归一化 FP32 路由系数。
它不能称为真实 Qwen 权重或真实路由。M7 的 T1024 random 两 seed 结果仍只支持：在
这组固定配置下，冻结策略的 Graph p50 比固定 vLLM 默认配置少 6.7–8.1%。

M20 的目标是用一个 pinned Qwen3-30B-A3B 检查点的一层真实参数，从固定自然语言
token IDs 生成 layer-0 router 输入，再对真实 MoE MLP 做独立算子比较。这个结果可以
支持“固定公开 checkpoint、固定 prompt、layer-0 的真实权重与路由输入”范围内的
operator 结论。它仍是单层实验，不是完整模型推理质量、在线服务、用户流量或请求
延迟证据。

本轮机器上只有 Qwen3-4B dense checkpoint。它没有 MoE router，不能当作 Qwen3-30B-A3B
数据使用，也不能据此声称 MoE 权重或路由是模型采集所得。

## 官方快照与安全取数

使用仓库现有 provenance 固定的 `Qwen/Qwen3-30B-A3B` revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`，配置 SHA256 为
`2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297`。
该快照配置为 H=2048、I=768、128 experts、top-k=8，`norm_topk_prob=true`，
48 层、每层均为 MoE。历史 M2 已固定同一 config revision；M20 下载脚本也会再次
核验 config 和 safetensors index 的 SHA256。

只下载 layer 0 所需张量，放在仓库外的
`/home/tmz/models/Qwen3-30B-A3B-layer0/`：

| 数据 | 张量数 | BF16 数据字节 |
| --- | ---: | ---: |
| `model.embed_tokens.weight` | 1 | 622,329,856 |
| layer-0 input/post-attention norms | 2 | 8,192 |
| layer-0 Q/K/V/O projections | 4 | 37,748,736 |
| layer-0 Q/K head norms | 2 | 512 |
| layer-0 router `mlp.gate.weight` | 1 | 524,288 |
| layer-0 128 experts 的 gate/up/down | 384 | 1,207,959,552 |
| **合计** | **394** | **1,868,571,136** |

固定 revision 的 index 文件为 1,699,758 B，列出 18,867 个张量和 16 个 safetensors
分片；完整 checkpoint 张量数据为 61,064,245,248 B。上表 394 个目标张量全部位于
`model-00001-of-00016.safetensors`，该分片大小为 3,999,417,504 B。HTTP Range
审计对 safetensors 前缀及 130,712 B header 观察到 HTTP 206 和正确的 `Content-Range`；
头部 SHA256 为 `b749467c61b8bc9b6d136ae30ee514fb76708a76915417d7744d91d02e139ee4`。
目标权重在源分片中的数据偏移从 0 连续至 1,868,571,136，没有未选张量间隙。
以上仅读取了 index/config/header 元数据，尚未取任何权重数据。

运行 [layer-0 fetcher](../scripts/fetch_qwen3_moe_layer0.py) 后，输出包含：

- `layer0.safetensors`：保留源 checkpoint 参数名的有效 SafeTensors 子集。
- `config.json` 与 `model.safetensors.index.json`：同一 pinned revision 的小型元数据。
- `manifest.json`：成功后才写入，含来源 revision、源 index/header hash、artifact hash、
  每个 tensor 的 shape/dtype/源和目标偏移/SHA256，以及每个范围块的 SHA256。

每次数据块必须收到精确的 HTTP 206、匹配请求的 `Content-Range`、字节数和固定 ETag；
服务器忽略 Range 返回 200 时立即拒绝。32 MiB 范围块、最多 4 个并发 worker，并支持
校验后续传。预检限制为 2 GiB。manifest 中的 `shard_header_sha256` 只校验 safetensors
header，`chunks[*].sha256` 和 `tensors[key].sha256` 只校验本次提取的字节；它们都不
冒充完整源 shard 的 checksum。

下载器把每个专家的 gate/up/down 保持为原始分开的键，例如
`model.layers.0.mlp.experts.0.gate_proj.weight`。构造算子输入时，对每个 expert 按
`gate_proj` 在前、`up_proj` 在后的顺序拼接为 `[E, 2I, H]`，down 组成 `[E, H, I]`。
不要假定源 safetensors 里已经有 `gate_up_proj` 打包张量。

Tokenizer 文件不由权重 fetcher 下载。capture 侧须使用同一 revision 的 tokenizer
配置，保存 tokenizer 文件 SHA256 和确切 token IDs；权重与 tokenizer 不得混用不同
revision。可在模型目录外保存固定短文本 prompt 和 token IDs；大权重和张量不进 Git，
小型来源manifest、数值及原始计时JSON会归档，保持证据可查。

## 真实 layer-0 输入的构造边界

真实 router input 不是 embedding 或 decoder layer 的原始输入。对固定 prompt token IDs，
layer 0 必须按 pinned Qwen3 实现运行：

1. 从真实 `model.embed_tokens.weight` 取 BF16 embedding。
2. 运行真实 `input_layernorm`。
3. 运行真实 self-attention：Q/K/V/O projection、Q/K head norm、使用 pinned config 的
   RoPE、真实 position IDs 和 causal mask。固定并记录 attention backend、数值类型和库版本。
4. 按模型 residual 顺序加 attention 输出，再运行真实 `post_attention_layernorm`。
   此处结果 `X[T,2048]` 才是 router 和 MoE experts 的输入。MoE 输出 residual 在 MoE
   operator 外，单独验证 layer 输出时再加回。
5. 使用真实 router 权重生成 logits、top-k IDs 和路由分值；或直接采集相同官方运行时
   的最终 IDs/分值。保存 route rows 和 SHA256。

官方 Transformers Qwen3-MoE 实现使用 `F.linear(X, gate.weight)`，FP32 softmax，top-k，
按 `norm_topk_prob` 归一化选中概率，再将路由分值转回 router logits 的 dtype。M20 应优先
保存这一步实际返回的 IDs 与 score；算子接口需要 FP32 weights 时，只将保存的 BF16 score
精确转成 FP32 表示，不再重新归一化或重排。官方 router 实现见
[Hugging Face Transformers v4.53.2 Qwen3-MoE](https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)。
配置细节以仓库 pin 的快照为准，而不是使用可变的 `main` 配置。

如果完整 layer-0 前置计算没有复现正确，禁止把合成 `X` 标成真实 hidden state。备选的
router-weight replay 可以用真实 `[128,2048]` router 权重配合人工 X 生成 IDs/weights，
但必须标注“真实 router 权重、合成 hidden state 和 expert 权重”；它只用于路由权重
加载/布局或路由实现诊断，不能代替本计划的真实 layer-0 输入路径。

## 有限 holdout 与计时契约

所有输入在测量前准备、验证和 hash；不修改 M7 artifacts 或根据 M20 timing 改 M7 policy。
首轮固定两条自含archive/code文本，对每条记录完整token IDs、tokenizer来源和hash；
对预定T=17、193、1024、2049分别执行真实layer-0前置计算，共八项。
这是人工文本的prefill单层输入，不是真实服务流量或单token decode；需要decode时，
必须另行生成带有效历史KV的模型输入。每项固定图另外原地换入同T另一条真实prompt的
X/IDs/weights，再恢复原值；不拿这些结果重新选择M7配置。

每个 T 都保存 BF16 router input、真实 route IDs/weights，以及实际权重和来源清单。运行
顺序先固定 route，然后比较：

- `ranked_default`：M6 原始 ranked 7-launch 路径。
- `ranked_selected`：不变更的 M7 FrozenMoEPolicy 配置选择。
- 官方 vLLM 0.10.0 public `fused_experts` 默认配置。

三臂使用完全相同的真实 X、IDs、route weights 和 layer-0 experts；每个 backend 单独对
同一个独立完整 FP32 oracle 检查 `relative_rms <= 0.008`、`normalized_max <= 0.008`、
finite。捕获阶段用router-only stub替换专家MLP，官方decoder的attention/residual/
postnorm顺序保持，仅输出真实router输入和沿官方公式构造的路由。没有执行完整HF
BF16专家循环或验证HF整层输出；不同BF16舍入边界以既有数学契约和完整FP32oracle
验收，不能宣称与HF完整layer输出逐bit一致。保留每项实际误差、原始计时样本、
输入/权重/hash、route counts与错误结果。

M7 计时的含义仍是 fixed-route 完整 expert forward：router projection、softmax/top-k、
prompt tokenization、layer-0 attention、下载、编译和准备都不计入核心 A/B。若希望报告
router+experts 的完整 MoE 子层时间，应单列另一组所有 backend 同样包含 router 的测量，
不能与 M7 fixed-route 数字直接合并。沿用 M7 的 API/Graph paired cyclic 顺序、p50/p95、
按完整 forward 归一化与计时后 correctness；报告范围只覆盖这组真实 prompt 与 layer 0。

M7 当前 T1024 冻结策略会选 medium config；本轮不因真实 trace 调整阈值或配置。如果
某个真实路由 case 回退或变慢，完整保留并报告，不重新挑选有利 prompt。

## Tensor Core 证据

M7 已保存其固定配置 Triton 产物的 PTX MMA 和 SASS HMMA 证据。HMMA 说明编译产物含
Tensor Core 指令，不说明真实模型权重执行时的 Tensor Core 利用率、occupancy 或服务
吞吐。M20 应对真实 layer-0 目标 shape/config 重新留存对应 PTX/cubin/SASS、hash、寄存器、
spill 和 shared-memory 记录，让证据绑定本轮源码与实际执行路径。

若 SM86 驱动和 Nsight Compute 可用，再单独 profile 一个预先选定的真实 trace：分别
报告 gate/up、down 与动态 dispatch kernel 的可用 Tensor pipeline/instruction counters、
occupancy/resource 指标以及 ncu 版本/命令/报告文件。先查询本机支持的 counter，再选用；
不支持的指标明确记为不可用。Nsight Compute 多 pass/replay 会扰动时间，profile 数据与
正常 CUDA event p50/p95 分开保存，不从 ncu 运行时间推算竞速结果。Tensor Core utilization
只能从受支持的运行时 counter 得出，不能由 HMMA 静态指令数量代替。

已有 M17/M18 attention 数据是固定 Qwen3-4B dense 模型的真实 Q/K/V 张量，不是 MoE
输入；其静态 HMMA 也不是运行时利用率。M20 不能把两类数据混写。Nsight Compute 的
replay、Graph profiling 和指标可用性说明见 [NVIDIA 官方指南](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

## 阶段边界

此阶段交付：固定来源、可审计的 layer-0 参数子集下载器、真实输入构造方案、有限 A/B
和 Tensor Core 取证契约。GPU 数据抓取/benchmark 由主 agent 串行完成；本计划不执行
benchmark、不调 kernel、不改 M7–M18、不做 NanoKV 集成，也不声称完整 30B model 或
production serving 验证。
