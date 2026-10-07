# M22 BF16 基础：净性能测量协议

2026-10-07，运行前声明。仅在 `M22_CUDA_AUDIT_PLAN.md` 全部数值/Graph/stream/dispatch
门槛通过后开始。目的为记录手写 CUDA 的基础成本与明确消融，不从这轮选择新的 M7 policy。

## 候选与对照

- 同一 pinned M20 archive/code × T17/193/1024 的六个输入。
- 六个完整 expert-forward arm：未修改的官方 vLLM0.10.0 public fused_experts、
  原 ranked default、冻结 M7、CUDA sync、CUDA async、CUDA async_skip。
- CUDA 三 arm 均使用 BM32/BN64/BK32 gate/up，保留同一输入对应的 M7 down tile 与
  BM32 prefix；其余 ranked launches 一致。M7 在中等输入的 gate/up 为 BN256，
  与新 CUDA BN64 不同，要明示此差异，不把完整差值仅归到编程语言或异步搬运。
- vLLM 的公共 API 分配、舍入语义和 Graph pool 行为沿用 M5/M20 的区别描述。
  禁止强制 config、替换其 dispatch 或称它是 serving benchmark。
- 两个 M21 诊断输入 archive-T17/1024 为已使用的设计证据；另外四个 real traces
  不参与 M21 配置诊断，但此前 M20 已测过，不称为从未见过的新 workload。

## 执行

三批独立新进程，每个 worker 只测一个 case。批一按 token升序/archive-code交替；
批二反序；批三将批一 case 顺序循环移动两位。单 GPU 串行，采集前停止 CPU 大型
packing/编译任务，不开启 profiler、没有锁钟或全局驱动设置改变。

每个 worker：输入权重/trace/source/policy/loaded binary hashes 前后核对，FP32
oracle 与 finite/.008 界前后复核，数值失败的 worker 不进入速度结论。
预热与 Graph 构建在计时外，prepared/run 持有固定 owners，不读 CPU counts。
API、Graph 使用既有 cyclic 正/逆轮次：每 arm 6 samples/round、2 calls/sample；
Graph 每 replay 含 8 个完整 forward，保存 replay 单位与逐 forward 归一化样本。
所有 raw samples、p50/p95、温度/时钟可用 telemetry 保存本地，不输出敏感环境信息。

另测五个本项目 gate/up launch 的固定已分组 metadata 成本（default/M7/三 CUDA），
单独捕获 stage Graph 并以每 launch 归一化。阶段计时不含 routing/down/combine，
不能冒充完整层，也不与其他独立分段计时做减法。

vLLM Graph output pointer 在捕获后显式持有；API 调用覆盖 wrapper.output 字段时，
仍核对原 Graph output，避免检查到另一次 API 的 output。

## 判读

逐输入、逐批报告全部 arm，不挑最好单次/最快路径。equal-weight geometric mean
与 worst regression 只用于总结，API/Graph 不混算。三批的配对比例全部展示；样本量
有限，不把它说成覆盖所有 serving workload 的稳定性证明。没有净收益也保留发布。

M21 T17 gate/up 约占 GPU duration sum 的66%，并观测到接近峰值 DRAM 吞吐。
减小 BF16 编程开销不能减少逻辑权重字节。仅 gate/up 从 768MiB→198MiB，若其耗时
理想同比降低且其他阶段完全不变，按该阶段占比粗算完整链路上限也仅约2×。
这是条件估计，不是 W4A16 实测收益；解码、scale访问与 Tensor Core 供给可能降低收益。
