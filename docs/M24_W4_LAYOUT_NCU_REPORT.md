# M24 布局：计数器补充与取舍

2026-10-09。四次独立进程采集完成，精确匹配两个输入的新旧 gate/up kernel，
每次跳过两个匹配 launch、采集一个 launch。四个 worker 与 report/CSV 身份核验通过。
[原始指标、单位和来源哈希](../results/m24-layout-ncu-summary.json)。
这是 profiler replay 数据；普通性能结论仍使用
[未配置 profiler 的六输入测量](M24_W4_LAYOUT_PERFORMANCE_REPORT.md)。

## 关键观察

| NCU 原始指标 | archive-T17 旧 → 新 | code-T1024 旧 → 新 |
|---|---:|---:|
| `memory_l1_tag_requests_global`（sectors） | 7,467,641 → 2,939,271 | 44,368,851 → 13,296,313 |
| `lts__t_sectors_lookup_miss.sum`（sector） | 7,536,707 → 3,133,906 | 17,380,195 → 17,257,469 |
| L2 sector hit rate | 35.70% → 4.27% | 76.01% → 40.81% |
| NCU kernel duration | 672.160 → 425.984 µs | 3.012672 → 2.384128 ms |

新布局的 global tag sector 请求计数分别减少约 60.6% 和 70.0%，支持先前
“每线程向量读取仍可能产生分散的 warp 访问”的诊断。这个指标包含 global
sector 请求，不能将它当作单独的 global-load request 次数或实际 DRAM bytes。

T17 的 L2 lookup miss sector 计数明显降低；T1024 的该计数接近不变。
因此两个输入的收益不能统一描述为“DRAM 流量大幅减少”。L2 hit rate 反而下降，
也说明仅追求命中率百分比不能指导这项优化：访问请求集合与分母已经改变。

## 设计判断

保持量化值、tile、shared staging 和 WMMA 算术不变，只把同一 word 的 N channels
排成连续地址。每线程四个 scalar 读取增加了读取指令，却减少了上述 sector 请求。
未配置 profiler 的六输入完整链路均快于原 W4，24 次输出比较逐位一致。
据此保留此布局作为有效候选；不需要用新增量化误差换取这次收益。

这四份 CSV 未包含独立的 L1 global-load sectors/requests 对和 DRAM read/write
byte totals，不能计算 load-only sectors/request 或证明精确 DRAM 字节节省。
本轮结束于这些可支持的结论，不追加采集。NCU duration 不替换正式性能数字，
小输入对 vLLM 的收益与 T1024 一快一慢的边界保持原报告口径。
