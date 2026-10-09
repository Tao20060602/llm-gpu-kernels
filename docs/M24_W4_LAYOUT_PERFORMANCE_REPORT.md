# M24 W4A16：保持量化算术的布局性能报告

日期：2026-10-09。状态：六个固定 case 的机制对照与 canonical summary 校验均已完成。汇总文件 SHA-256 为 `a5d494970ee745dd72676d314d6593c26cafb4d48c026d68bcd933dae447d33b`。本文不选择新 policy，也不替换默认实现。

## 范围与执行

本轮使用 M20 已采集的 Qwen3-30B-A3B layer0 固定输入，顺序为 `code-T1024`、`archive-T1024`、`code-T193`、`archive-T193`、`code-T17`、`archive-T17`；每个 case 一个新进程，单 GPU 串行。它是既有诊断输入上的一批布局机制对照，不是新的 holdout，也不支持生产流量结论。

四个完整 forward arms 为 `quant_vllm`、`quant_cuda_sync`、`w4_sync` 和 `w4_layout`。所有 arms 使用同一份量化值及 BF16 down；`quant_vllm` 是同量化权重的公共 vLLM baseline。阶段测量是固定 metadata 的 gate/up+SwiGLU，单独对照 `w4_sync` 和 `quant_cuda_sync`；vLLM 没有对应的阶段入口。Stage 与完整 forward 分开报告，不做相减推断。

Full API/Graph 各八轮，每轮六个样本、每样本两次调用；stage API/Graph 各六轮，同样每轮六样本、两次调用。Graph replay 含八次 forward，按八归一化。表中时间为逐 case 的 p50 / p95 毫秒，p95 不跨 case 汇总。比值为 `w4_layout p50 / reference p50`，小于 1 表示 layout 更快。

运行记录 manifest 与六份 worker JSON、原始日志仅保存在本机 `artifacts/m24-w4-layout-perf-20261009-v1/`，不随公开结果发布。manifest SHA-256 为 `e5b9cb7cc33f49f3bdb2ae2ed227246539fba72faa08fe80750ae08a437eea21`。公开的[canonical 逐 case 汇总](../results/m24-w4-layout-summary.json)包含原始计时样本及逐 case p50/p95，SHA-256 `a5d494970ee745dd72676d314d6593c26cafb4d48c026d68bcd933dae447d33b`。环境为 RTX 3080 Laptop GPU、Python 3.12.14、PyTorch 2.7.1+cu128、Triton 3.3.1 和官方 vLLM 0.10.0 overlay。

## 主要结果

完整 Graph 中，layout 对旧 W4 的六个比值都小于 1，范围为 0.7739–0.8536。对同量化 vLLM，四个 T17/T193 case 更快；`archive-T1024` 快约 3.2%，`code-T1024` 仍慢约 2.0%（比值 1.0201）。T1024 不能概括成普遍胜过 vLLM。

独立的 gate/up stage Graph 中，六个 case 的 layout 均快于旧 W4 和同量化 CUDA stage。该阶段结果只说明这个固定 metadata 阶段的对照，不能替代完整 forward 结论。

### Full API

| Case | Layout p50 / p95 ms | Old W4 p50 / p95 ms | 同量化 vLLM p50 / p95 ms | Layout / old W4 | Layout / vLLM |
|---|---:|---:|---:|---:|---:|
| code-T1024 | 3.700 / 3.893 | 4.332 / 4.722 | 3.662 / 3.783 | 0.8541 | 1.0104 |
| archive-T1024 | 3.737 / 3.900 | 4.368 / 4.548 | 3.920 / 4.139 | 0.8555 | 0.9532 |
| code-T193 | 1.549 / 1.571 | 1.929 / 1.941 | 2.121 / 2.233 | 0.8029 | 0.7303 |
| archive-T193 | 1.574 / 1.585 | 2.025 / 2.037 | 2.246 / 2.353 | 0.7775 | 0.7009 |
| code-T17 | 0.943 / 0.956 | 1.212 / 1.228 | 1.480 / 1.601 | 0.7780 | 0.6374 |
| archive-T17 | 0.888 / 0.903 | 1.132 / 1.153 | 1.391 / 1.492 | 0.7851 | 0.6388 |

### Full Graph

| Case | Layout p50 / p95 ms | Old W4 p50 / p95 ms | 同量化 vLLM p50 / p95 ms | Layout / old W4 | Layout / vLLM |
|---|---:|---:|---:|---:|---:|
| code-T1024 | 3.703 / 3.800 | 4.338 / 4.394 | 3.630 / 3.725 | 0.8536 | 1.0201 |
| archive-T1024 | 3.727 / 3.769 | 4.370 / 4.421 | 3.850 / 3.945 | 0.8528 | 0.9681 |
| code-T193 | 1.539 / 1.553 | 1.922 / 1.932 | 2.053 / 2.058 | 0.8005 | 0.7496 |
| archive-T193 | 1.563 / 1.580 | 2.017 / 2.033 | 2.167 / 2.170 | 0.7750 | 0.7212 |
| code-T17 | 0.936 / 0.960 | 1.209 / 1.218 | 1.398 / 1.400 | 0.7739 | 0.6695 |
| archive-T17 | 0.878 / 0.906 | 1.122 / 1.131 | 1.309 / 1.311 | 0.7825 | 0.6710 |

### Gate/up stage API

| Case | Layout p50 / p95 ms | Old W4 p50 / p95 ms | 同量化 CUDA p50 / p95 ms | Layout / old W4 | Layout / CUDA |
|---|---:|---:|---:|---:|---:|
| code-T1024 | 2.414 / 2.426 | 3.034 / 3.043 | 2.896 / 2.911 | 0.7957 | 0.8338 |
| archive-T1024 | 2.413 / 2.435 | 3.033 / 3.048 | 2.910 / 2.935 | 0.7956 | 0.8291 |
| code-T193 | 0.784 / 0.789 | 1.157 / 1.164 | 1.502 / 1.508 | 0.6775 | 0.5220 |
| archive-T193 | 0.775 / 0.798 | 1.227 / 1.236 | 1.555 / 1.562 | 0.6316 | 0.4986 |
| code-T17 | 0.461 / 0.472 | 0.725 / 0.732 | 0.998 / 1.007 | 0.6354 | 0.4615 |
| archive-T17 | 0.432 / 0.449 | 0.671 / 0.675 | 0.908 / 0.913 | 0.6436 | 0.4759 |

### Gate/up stage Graph

| Case | Layout p50 / p95 ms | Old W4 p50 / p95 ms | 同量化 CUDA p50 / p95 ms | Layout / old W4 | Layout / CUDA |
|---|---:|---:|---:|---:|---:|
| code-T1024 | 2.404 / 2.506 | 3.046 / 3.161 | 2.957 / 3.107 | 0.7891 | 0.8131 |
| archive-T1024 | 2.432 / 2.521 | 3.062 / 3.167 | 3.033 / 3.113 | 0.7944 | 0.8019 |
| code-T193 | 0.786 / 0.862 | 1.164 / 1.178 | 1.498 / 1.506 | 0.6757 | 0.5251 |
| archive-T193 | 0.786 / 0.908 | 1.234 / 1.278 | 1.551 / 1.561 | 0.6372 | 0.5068 |
| code-T17 | 0.462 / 0.524 | 0.732 / 0.751 | 0.995 / 1.004 | 0.6308 | 0.4641 |
| archive-T17 | 0.437 / 0.468 | 0.675 / 0.714 | 0.905 / 0.910 | 0.6480 | 0.4829 |

## 正确性与重排成本

M24 layout audit 的 42 个 case-state 及 T0 均通过；六个性能 worker 的原始 W4 对照与 layout 在 audit 前、Graph 计时前、完整 Graph 计时后和 audit 后共 24 次逐 bit 比较全部为 true。量化 oracle 的 finite、误差阈值与 dispatch 检查也通过。重排没有重新量化，canonical v1 words/scales 与物理 layout 使用各自 digest 记录。

原 gate/up packed words 与 scales 共 198 MiB（192 MiB words、6 MiB scales）；新物理布局字节数不变。本轮每个 worker 的 CPU-only repack 时间如下，均排除在 API/Graph 测量之外：

| Case | CPU repack ms |
|---|---:|
| code-T1024 | 246.925 |
| archive-T1024 | 271.100 |
| code-T193 | 274.254 |
| archive-T193 | 286.913 |
| code-T17 | 259.387 |
| archive-T17 | 284.567 |

重排额外消耗约 0.247–0.287 秒 CPU 时间；该时间没有被摊入单次 forward。本轮 benchmark 同时保留对照 arms，因此这些运行记录不是单路径 GPU 显存测量，也不证明线上常驻成本下降。

量化格式误差仍与 layout 差分分开。layout 的 CPU round trip 保持 canonical packed words/scales bit pattern，且性能 worker 对旧 W4 的四个检查位置都逐 bit 相等，因此这次物理重排没有引入相对旧 W4 的额外算术或量化取舍。worker 的格式 proxy 对照原 BF16 layer output，仅描述既有 W4 格式误差，不作为模型质量阈值；没有进行完整模型任务质量评估。

## 解释边界

首轮只改变 packed gate/up 的物理排列：v1 每线程对一个输出通道向量读四个 32-bit words，warp 内相邻线程跨 N 的地址步长较大；新布局让一个 K32 tile 的输出通道在内存中连续，每线程改为读取四个跨 N 连续的 scalar words。tile、同步 shared staging、WMMA/FP32 accumulate、舍入与 SwiGLU 算术顺序均未改变。观察到的延迟差异与这些访问方式相符，但本轮不证明具体 DRAM 流量或带宽是因果因素。

静态 SASS 中的 `LDG` 指令条数是指令清单，不等价于实际内存事务或 DRAM 字节数。此次没有采集该新 layout 的 NCU 计数器，不能用 M21 或其他 kernel 的旧 counter 推断本候选的吞吐或命中率。要判断访存因果，需要针对新 layout 单独采集 NCU，并把 profiler 下的扰动计时与本报告普通 worker 分开。

这批固定输入支持继续考察候选布局，但不据此改变 policy、量化格式、tile 或默认入口。尤其保留 `code-T1024` Full Graph 相对同量化 vLLM 慢约 2.0% 的结果。

## 跨 case 几何平均

以下是逐 case p50 候选/参考比值的等权几何平均；`T17/T193` 为四个小输入，`T1024` 为两个大输入。完整 forward 的同量化基线是 vLLM 公共算子；stage 没有 vLLM stage，使用同量化 CUDA stage。小于 1 表示 layout 更快。p95 只在前面的逐 case 表中报告，不跨 case 汇总。

| Section | Case scope | Layout / old W4 | Layout / same-quant baseline |
|---|---|---:|---:|
| Full API | all 6 | 0.8082 | 0.7652 |
| Full API | T17/T193 · 4 | 0.7858 | 0.6757 |
| Full API | T1024 · 2 | 0.8548 | 0.9814 |
| Full Graph | all 6 | 0.8057 | 0.7882 |
| Full Graph | T17/T193 · 4 | 0.7829 | 0.7020 |
| Full Graph | T1024 · 2 | 0.8532 | 0.9938 |
| Stage API | all 6 | 0.6930 | 0.5836 |
| Stage API | T17/T193 · 4 | 0.6468 | 0.4890 |
| Stage API | T1024 · 2 | 0.7956 | 0.8314 |
| Stage Graph | all 6 | 0.6925 | 0.5821 |
| Stage Graph | T17/T193 · 4 | 0.6477 | 0.4942 |
| Stage Graph | T1024 · 2 | 0.7918 | 0.8075 |

T1024 的 Full Graph 与同量化 vLLM 几何平均 0.9938 接近持平；这不代表两个 case 都有收益：`archive-T1024` 比值为 0.9681，而 `code-T1024` 为 1.0201，仍慢约 2.0%。
