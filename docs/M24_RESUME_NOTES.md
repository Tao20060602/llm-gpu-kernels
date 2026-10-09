# M24 W4 packed-layout：简历表述

## 三点表述

- 实现 bit-preserving W4 gate/up packed-layout 候选：不改变量化值、BF16 WMMA、FP32 累加、舍入或 SwiGLU 顺序。
- 在 RTX 3080 Laptop / SM86 上，对复用的 Qwen3-30B-A3B layer-0 六个固定输入做单批 Full Graph 对照。候选相对旧 W4 的 p50 几何平均低 19.43%，相对同量化 vLLM 低 21.17%；T17/T193 四 case 对 vLLM 低 29.80%。
- 六输入 × 七状态及 T0 correctness audit 通过；六个性能 worker 的四阶段 bitwise 对照均通过。Packed words 与 scales 合计 198 MiB，CPU repack 约 0.247–0.287 秒，另行计量。

## 取舍与适用范围

- T1024 对 vLLM 一快一慢：`archive-T1024` 低约 3.19%，`code-T1024` 慢约 2.01%；两 case 几何平均仅低 0.62%，不代表长输入普遍改善。
- 结果来自既有 M20 输入的一批机制对照，不是新 holdout 或多批次复现；无 CI、锁钟、完整模型质量或生产评估。
- 新布局 NCU counters 已采集，结果待提取核验；在核验前不能把性能变化归因于 DRAM 流量、带宽或缓存命中。所有 benchmark arms 同时驻留，也不是单路径峰值显存测量。
- 候选未替换默认入口或冻结为 shape policy。完整逐 case 结果与限制见[M24 性能报告](M24_W4_LAYOUT_PERFORMANCE_REPORT.md)及[当前状态](M24_CURRENT_STATUS.md)。
