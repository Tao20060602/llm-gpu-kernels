# 当前项目状态：M23 W4 复测完成

更新日期：2026-10-09。M23完成了M22 W4A16首批测量后的两批独立进程复测。
当前入口单独记录新状态；[M22原项目状态](../PROJECT_STATUS.md)和M22冻结成果保持历史快照。

## 复测结论

在固定的六个Qwen3-30B-A3B layer0输入、topk8、三批等权Full Graph p50比较中：

| 输入范围 | 相对同量化权重 vLLM | 相对原 BF16 vLLM |
|---|---:|---:|
| T17/T193四个case（12条case-run） | 低10.00% | 低10.07% |
| T1024两个case（6条case-run） | 慢17.30% | 慢17.58% |
| 全部18条case-run | 低1.69% | 低1.66% |

T17/T193小输入组的Full Graph比值每批均低于1；T1024两项在每批都回归。
总体混合几何平均不能掩盖大输入负收益，也不能推广为全形状规律。
原BF16比较单独体现量化格式取舍；同量化权重参考用于比较实现路径。
完整Full API/Graph、stage、p50、逐case p95和所有回归见
[M23复测报告](M23_W4_REPEAT_REPORT.md)。原始samples及provenance见
[汇总JSON](../results/m23-w4-repeat-summary.json)。

## 审计和限制

18/18 worker与12/12 repeat startup receipt均通过summary脚本验证；输入、trace、packed、
源码、binary与audit身份按协议核对。audit_before/after与Graph数值检查通过原0.008界，
所有适用项目arm的ranked dispatch通过。实现正确性指标与格式forward变化分别报告，
后者不是任务分数或模型质量评估。

没有统计置信区间，GPU时钟未锁定；这不是holdout。startup复测使用已审计native binary
adapter，首批使用legacy JIT preparation，两种准备都在计时外；“no JIT”仅指native
C++ extension启动加载，不涵盖Triton preparation。失败尝试与时间顺序见
[M23运行恢复记录](M23_RUNTIME_RECOVERY.md)。

本实验不支持production部署、全模型质量、全workload泛化或Marlin优越性声明。仅适用于
这组固定真实layer0输入。后续M24 packed-layout已有42状态与T0正确性审计通过，但性能
尚未测量；它仍是待测性能候选，未纳入M23复测，也不据正确性结果声称加速。

## 下一步

- 优先分析W4 NCU计数器，定位在线decode、scale/shared访问与权重流量成本。
- M24布局候选已通过42状态与T0正确性审计；后续需单独采集并报告完整性能样本，
  见[M24计划](M24_W4_LAYOUT_PLAN.md)。
- Group size、down量化与同域Marlin比较、完整checkpoint任务评估分别立项。

## 入口

- [M23结果与限制](M23_W4_REPEAT_REPORT.md)
- [简历表述稿](M23_RESUME_NOTES.md)
- [运行恢复与启动方式](M23_RUNTIME_RECOVERY.md)
- [冻结的M22首版报告](M22_W4_REPORT.md)
- [整体算子项目状态与历史结果](PROJECT_STATUS.md)
- [证据与复现索引](REPRODUCTION_INDEX.md)
- [M23 release manifest](../results/m23-release-manifest.json)
