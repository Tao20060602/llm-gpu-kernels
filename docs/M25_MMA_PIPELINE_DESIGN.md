# M25：SM86 显式 MMA 与多 stage GEMM 讲义

## 目标与状态

实验核计算 `C = A @ W.T`，其中 `A[M,K]`、`W[N,K]` 与 `C[M,N]` 为连续 BF16，累加为 FP32，写回 BF16。2026-10-13 已使用 CUDA 12.8 / GCC 14 完成 SM86 扩展编译和加载。**尚未运行数值验证，也未采集性能；不能据此宣称正确性通过或有性能收益。**

K 至少为 16 且是 16 的倍数，M 可为零，M/N 尾块补零并屏蔽无效输出；输入基址要求 16-byte 对齐。绑定的 buffers 不能 resize/替换，调用者负责跨 stream 依赖及避免同一输出并发写入。

编译器报告如下，均为零 stack frame、零 spill stores/loads。日志保存在本机 `artifacts/m25-build/build.log`；静态资源数字不代表运行时 occupancy 或速度。

| stages | registers/thread | dynamic shared bytes/CTA |
|---|---:|---:|
| 1（同步） | 48 | 7,680 |
| 2 | 58 | 15,360 |
| 3 | 70 | 23,040 |
| 4 | 70 | 30,720 |

当前边界：单 CTA 内计算，不做 shared-memory swizzle、不做 warp specialization、不接入 MoE，也不支持 W4。

## CTA、warp 与指令映射

- CTA tile 为 `BM×BN×BK = 32×64×32`，128 线程、四个 warp；每个 warp 负责 `32×16` 输出区域。
- 每条 MMA 指令计算 `m16n8k16`，输入 BF16、累加 FP32：`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`。每个 warp 持有两个 M16、两个 N8 的累加片段；每个 K32 tile 分两次 K16 更新累加器。
- A 从 shared memory 通过 `ldmatrix.sync.aligned.m8n8.x4.shared.b16` 装入寄存器；每个 N8 的 B 片段通过 `.x2` 装入。物理 `W[n,k]` 的 K 连续布局正好是逻辑 `B[k,n]` 的 column-major 视图，因此 B 的 `ldmatrix` 不加 `.trans`。
- 每线程 MMA 累加器含四个 FP32 值。lane 映射为：`row = mi*16 + lane/4 + (element/2)*8`，`col = warp*16 + ni*8 + (lane%4)*2 + element%2`。即 `lane/4` 选行，`lane%4` 选相邻两列中的列对；四个值覆盖两行（第二行偏移 8）与相邻两列。

官方 PTX 细节见 [PTX ISA：warp-level `mma` 与 `ldmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/)，其中 MMA 章节说明矩阵形状、寄存器片段和累加语义，`ldmatrix` 章节说明从 shared memory 按 warp 装载矩阵片段。

## Stage 与 shared memory

`LD=40` 个 BF16，A 与 W 共用一个 K32 stage：`(BM+BN)×LD×2 = 7,680` 字节。S 个环形槽需要 `7,680×S` 字节动态 shared memory；S=1 为同步搬运，S=2..4 使用 `cp.async`。

异步路径的顺序是：先预填 `S−1` 个 K32 stage；主循环为即将复用的槽发出 future tile 并 commit，再用 `cp.async.wait_group S−1` 等待最老一组完成，随后 CTA barrier 后消费当前 tile。没有更多 future tile 时用 `wait_group 0` 排空尾部，再经 CTA barrier。MMA 消费完当前槽后再做 CTA barrier，保证下一轮可安全覆盖该槽。Stage 1 使用同步向量化拷贝，并在消费前后使用 CTA barrier。

提高 S 会增加预取深度，也按每 stage 7.5 KiB 增加 shared 占用；它是否隐藏了更多延迟、是否损害驻留度，需要后续测量回答。

## 待验证接口示例

```python
from llm_gpu_kernels.gemm_mma import prepare_bf16_mma

plan = prepare_bf16_mma(a, w, stages=3)  # 待验证实验
out = plan.run()                         # C = A @ W.T，BF16 输出
```

## 后续阶段

按顺序单独推进：shared-memory swizzle → warp specialization → MoE 集成，再评估 W4 路径。每一步都需要独立数值检查与性能证据；本讲义不代表这些阶段已实现或已验证。
