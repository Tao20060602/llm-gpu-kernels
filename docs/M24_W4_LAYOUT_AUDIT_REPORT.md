# M24 W4 布局候选正确性审计

日期：2026-10-09。状态：正确性审计通过；审计阶段没有计时。后续独立性能测量见 [M24 W4 布局性能报告](M24_W4_LAYOUT_PERFORMANCE_REPORT.md)。

## 结果

六个固定真实输入（archive/code × T17/T193/T1024）的 42 个状态全部通过，42 个数值结果均为 finite，ranked dispatch 检查全部通过。相对原 v1 量化权重反量化得到的 BF16 oracle，最大误差为：

| 指标 | 42 状态最大值 | 门槛 |
|---|---:|---:|
| Relative RMS | 0.0033194 | 0.008 |
| Normalized max | 0.0043417 | 0.008 |
| Max absolute error | 0.0038645 | 信息项 |

T0 通过：输出为 BF16 [0, 2048]，launch count 为 0。每个输入都覆盖 API、Graph、两种重复 expert ID、非默认 stream 和恢复后的 API/Graph 状态。

v1 pack 到 v2 physical layout 的 words 与 scales 均 torch.equal，canonical 与 physical SHA 均验证通过。CPU 重排记录为 0.2464 秒，逆变换为 0.2485 秒；重排计时包括 v1 输入和 v2 输出 digest，不含逆变换和 H2D。packed+scales 仍为 198 MiB；完整 round-trip 同时持有 canonical、repacked 和恢复副本时，张量字节账本峰值为 594 MiB。

原 BF16 权重与同一量化格式的输出差异作为 format-loss proxy 单独记录，六 case 最大值为 relative RMS 0.0448426、normalized max 0.0098502、max absolute error 0.0091381。这些 proxy 不作为实现通过门槛、模型质量分数或质量结论。

## 编译与静态产物

环境为 Python 3.12.14、PyTorch 2.7.1+cu128、Triton 3.3.1、CUDA runtime/toolkit 12.8、GCC/G++ 14.3.0，RTX 3080 Laptop GPU，SM86。编译使用 nvcc 12.8.93、MAX_JOBS=1 和独立 TORCH_EXTENSIONS_DIR。

ptxas 报告 72 registers/thread、1 barrier、0 stack bytes、0 spill store/load，constant memory [0] 为 408 bytes。cuobjdump resource table 同样报告 REG 72、STACK/LOCAL/SHARED 0；这是二进制资源表的静态报告，不是运行时资源采样。

对 sm_86 SASS 的静态指令盘点为 16 条 HMMA.16816.F32.BF16、11 条 LDG（其中 1 条 LDG.E.128）、10 条 LDS、21 条 STS 和 3 条 BAR.SYNC。这确认编译镜像含 BF16 HMMA 和 global load 指令；指令数不代表实际 DRAM 流量、吞吐或性能。

编译使用的 cuobjdump/nvdisasm 来自本仓库 venv 的 Triton NVIDIA backend，版本均为 CUDA 12.8.55。工具 SHA 与编译资源见公开资源 JSON；完整 SASS、提取 cubin 和原始资源输出保存在本机 artifacts。

## 身份与产物

audit status 为 passed，source、binary、packed artifact、pack record、M7 policy 和训练记录的前后哈希一致。M24 源码及 M7 policy 声明的来源文件均纳入 audit provenance。

- 完整本机审计 JSON（本地文件）：`artifacts/m24-w4-layout-audit-20261009-v1/audit.json`
- 去除 binary.resolved_path 的公开副本：[m24-w4-layout-audit.json](../results/m24-w4-layout-audit.json)
- 汇总资源与静态指令：[m24-w4-layout-resources.json](../results/m24-w4-layout-resources.json)
- 本地运行环境与命令清单：`artifacts/m24-w4-layout-audit-20261009-v1/run_manifest.json`
- 本地编译日志：`artifacts/m24-w4-layout-audit-20261009-v1/logs/audit.log`
- 本地 cuobjdump SASS：`artifacts/m24-w4-layout-audit-20261009-v1/disassembly/m24-w4-layout-sm86.sass`
- 本地 nvdisasm SASS：`artifacts/m24-w4-layout-audit-20261009-v1/disassembly/m24-w4-layout-nvdisasm.sass`
- 本地 cuobjdump 资源输出：`artifacts/m24-w4-layout-audit-20261009-v1/disassembly/m24-w4-layout-resource-usage.txt`

本机完整 audit JSON SHA-256：bced7e90c17223a8f00c142953f09f076be1839159d3a4987085a95ab5b66517。公开副本只删除 binary.resolved_path，其 SHA-256 为 c0e1d86fe078894c4abc4bb41dd6fd1c2cd725200ae24ba03727f6f3c962a8e9。

编译产物 SHA-256：47611e5354d01efddcaa1d22d1a649d539810ff66439d3be8a3e9a350f2d04dc。

本次 audit 阶段只验证正确性并检查静态编译产物，没有计时或运行性能 benchmark。后续性能结果见 [M24 W4 布局性能报告](M24_W4_LAYOUT_PERFORMANCE_REPORT.md)。
