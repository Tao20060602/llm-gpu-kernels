# 当前状态：M24 W4 packed-layout

更新日期：2026-10-09。正确性审计与单批性能测量已完成；NCU counters 已采集，结果待提取和核验。候选仍是实验实现，未替换默认入口或冻结为新 policy。M23 阶段快照与 M22 基线保持不变。

## 六个固定 case 的 Full Graph p50

比值是候选/参考的等权几何平均；低于 1 表示候选更快。

| 范围 | 对同量化 vLLM | 对旧 W4 |
|---|---:|---:|
| 全部六个 case | 0.7882（低 21.17%） | 0.8057（低 19.43%） |
| T17/T193 四个 case | 0.7020（低 29.80%） | 0.7829（低 21.71%） |
| T1024 两个 case | 0.9938（低 0.62%） | 0.8532（低 14.68%） |

T1024 对 vLLM 一快一慢：`archive-T1024` 比值 0.9681，`code-T1024` 比值 1.0201，不能据此声称长输入普遍改善。完整 API/Graph、stage、逐 case p50/p95、raw rounds 与负收益见[性能报告](M24_W4_LAYOUT_PERFORMANCE_REPORT.md)和[canonical summary](../results/m24-w4-layout-summary.json)；p95 只逐 case 报告，不跨 case 聚合。

## 正确性、成本与范围

- 布局 audit 的六个输入 × 七个状态及 T0 全部通过；性能 worker 在四个阶段的 24 次候选/旧 W4 bitwise 对照均通过。
- Packed words 与 scales 合计 198 MiB，物理重排不改变字节数；CPU repack 每 worker 约 0.247–0.287 秒，未计入 API/Graph 计时。所有 benchmark arms 同时驻留，因此这不是单路径 VRAM 测量。
- 这是复用 M20 layer-0 输入的一批六 case 机制对照，不是新 holdout 或多批次复现；没有 CI、锁定 GPU 时钟、完整模型质量评估或生产评估。NCU counters 尚未提取核验，不能将收益归因于特定 DRAM、带宽或缓存变化。

候选保持独立，后续需单独采集 NCU 数据；部署 policy 需要新的选择规则和未参与诊断的输入。

## 记录

- [性能报告](M24_W4_LAYOUT_PERFORMANCE_REPORT.md) · [正确性审计](M24_W4_LAYOUT_AUDIT_REPORT.md) · [布局计划](M24_W4_LAYOUT_PLAN.md)
- [简历与面试笔记](M24_RESUME_NOTES.md) · [release manifest](../results/m24-release-manifest.json)
- [canonical summary](../results/m24-w4-layout-summary.json) · [资源记录](../results/m24-w4-layout-resources.json)
- 历史阶段：[M23 当前状态](M23_CURRENT_STATUS.md) · [M23 复测报告](M23_W4_REPEAT_REPORT.md)

Summary 公开保留 raw timing 和逐 case 结果；worker JSON、日志及带本机路径的运行 manifest 留在本机，不随发布清单分发。
