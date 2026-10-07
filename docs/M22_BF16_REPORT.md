# M22 BF16：手写核心、消融与负收益

本阶段完成数值/Graph/stream 审计及六个真实输入×三批独立进程的性能测量。
结论是保留已冻结的 M7 Triton 默认路径：新增 CUDA BF16 核心尚未带来完整链路收益。
它作为 W4A16 的受控基础，不把“用了异步搬运/Tensor Core”写成加速结论。

## 为什么先做这一阶段

M21 显示 T17 gate/up 约占关联 GPU 时长之和的66%，NCU replay 下 DRAM 吞吐约
95.8%、L2 hit约0.5%。这支持优先研究减少权重读取量，但不能跳过同精度核心：先
固定计算组织，才能区分新 CUDA 的成本与后续 INT4 存储/解码效果。M21 counter 的
cache/replay 条件与本轮干净 Graph 测量不同，不能将它们拼成同一次运行的 Roofline。

## Tile 与布局

| 项目 | 选择与原因 |
|---|---|
| CTA | BM32/BN64/BK32、128 threads，沿用 ranked BM32 metadata 与 down 配置 |
| Warp | 每 warp 16 个 N 列，两个 M16 子块，gate/up 共四组 FP32 accumulator |
| Tensor Core | BF16 WMMA16×16×16，K32 分两次 K16；首版不手写全部 PTX fragment 映射 |
| Shared | K stride40 BF16，保持矩阵 tile 对齐；padding 不等于已证明没有 bank conflict |
| 同步版 | 一份12800B staging，最后复用为16384B FP32 epilogue，实际申请16384B |
| 异步版 | 两份 staging 合计25600B，计算当前 K32 时预取下一 K32，显式 commit/wait/barrier |
| 融合 | FP32 gate/up 分别舍入 BF16，再用 FP32 SwiGLU，写 BF16 activation；保留 M4 舍入契约 |

`async_skip` 只增加 CTA 一致的有效行分支：不足第二个 M16 时跳过该子块的矩阵
计算。它不减少权重读取量，也不改变 BM32 metadata；不是为了小输入悄悄修改精度。
三个版本完整链路均为 reset/histogram/prefix/dispatch/gate-up/down/combine 七次启动。
所有 buffer 在 prepare 持有，run 使用当前 stream，无 counts CPU 回读或热路径分配。

## 正确性及失败记录

首个尝试误把 offsets 检查为 E+1 项，触发 native constructor 的 metadata shape guard；
GPU prefix 实际 offsets/counts 为 E 项、tasks 为 E+1 项。修正 guard，保留原失败记录。
修复后的六 case×七状态×三个版本共126项数值与 dispatch 检查全部通过；包括动态
重复 IDs=0/127 的固定 Graph、非默认 stream、恢复输入及 T0 零 launch。

最大 relative RMS 为0.00331079，normalized max 为0.00472810，均低于预先声明的
0.008；最大绝对误差0.00381780。输入、源码、二进制与 artifact hashes 前后稳定。
见 [审计摘要](../results/m22-bf16-audit-summary.json) 与
[运行前审计协议](M22_CUDA_AUDIT_PLAN.md)。

## 性能怎么判断

使用原 Qwen3-30B-A3B layer0 的 archive/code × T17/193/1024，三批顺序不同的新进程。
完整 API 与 Graph 分开报告，另测已分组的 gate/up 阶段；所有候选和全部样本保留。
未锁钟、未并行 GPU 工作、未开启 profiler，也没有根据本轮重选 M7 policy。
见 [运行前性能协议](M22_BF16_PERFORMANCE_PLAN.md)。

18个配对 full Graph p50 比值（候选时间/vLLM 时间）的几何平均：sync1.0906、
async1.0816、async_skip1.0756；冻结 M7 为0.9959。新增版本整体比成熟 BF16 对照慢，
最小的 async_skip 也约慢7.56%。API 对应 sync1.0549、async1.0492、async_skip1.0417，
同样没有几何平均净收益。这里只是这六个输入三批的描述，不外推 serving。
全部样本、逐批 p50/p95、配对比值与完整性检查见
[性能摘要](../results/m22-bf16-performance-summary.json)，静态二进制证据见
[binary inspection](../results/m22-bf16-binary-inspection.json)。实际审核二进制中
sync/async/async_skip分别有16/16/16条HMMA、0/26/26条LDGSTS；后者说明机器码
存在异步搬运，不证明运行时利用率。动态 shared 分别16384/25600/25600B，与工具
显示的 static SHARED=0 不矛盾。

以 archive-T17 为例，三批 full Graph p50 中位数约为：冻结 M7 1.330ms、sync1.368ms、
async1.341ms、async_skip1.342ms。async 比 sync 局部改善，但仍未超越冻结路径。
archive-T1024 对应约3.600/4.307/4.296/4.235ms，大输入的差距更明显。

完整差异不能仅归因于 CUDA/Triton 或异步流水线。M7 中等输入 gate/up 为 BN256，
新 CUDA 固定 BN64：较小 N tile 减少资源需求，也增加 CTA 数和 A/metadata 重读。
stage-only 也显示新版本比冻结路径慢；不使用独立分段时间相减来声称每个瓶颈占比。
这里的五个阶段arm分别测量五种backend的gate/up单次启动，不是五次启动之和；
各自阶段Graph包含8次同backend启动，按一次launch归一化。

## 下一步取舍

不扩大 BF16 tile sweep 来寻找一个好看的单点结果。M21 的小输入带宽证据和本轮
完整链路负收益，使下一步更值得检验 W4A16：少搬 gate/up 权重，tile 内在线解码，
融合解码、gate/up 与 SwiGLU。INT4 存储带来 scale 读取、bit 解码、寄存器和访存布局
成本，不能把理论字节压缩3.88倍当成实测加速3.88倍。

格式损失、CUDA 实现误差、完整模型任务质量分别报告。只检查 layer0 无法证明任务
得分下降≤10%；W4 的对照先使用相同量化权重与保留的原 BF16权重，明确精度取舍。
