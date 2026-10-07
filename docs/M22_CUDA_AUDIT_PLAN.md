# M22 BF16 基础实现：运行前正确性协议

2026-10-07。本阶段仅准备和审计，不选择性能 winner、不修改 M7 policy。
原始 pinned M20 layer0 的 archive/code × T17/193/1024 共六个真实输入，
三个已声明版本 sync、async、async_skip，全都跑相同 FP32 oracle 与 ranked dispatch 审计。
使用 SM86 / CUDA12.8 编译器 / 仓库 venv，禁止继承 CUDA13.2 配置。

每个输入记录以下状态，input/weight/source hashes 执行前后保存并核对：

1. 原真实输入：三个 CUDA 完整 forward 均检查 finite、relative RMS / normalized max≤0.008；
   检查 device counts、offsets、ranks 和 route map，不能只比局部 GEMM。
2. 捕获每个版本的固定地址 8-forward CUDA Graph，修改原地址中的 X、IDs、routing weights：
   X = 原 X×(-0.75) 后 BF16，IDs 全为0，weights=原值×0.5。
   这覆盖大量重复 expert IDs、空 experts 和最后 row tile 不满；重新计算 oracle 后 replay。
3. 将同一 IDs buffer 改为127，再次 Graph replay 与 oracle/dispatch 检查，验证 metadata
   来自新 GPU histogram/prefix，不是捕获时的 CPU counts 或固定 expert。
4. 非默认 stream 中写入原 X×0.5、IDs=原输入、routing weights=原值×(-0.5)，
   在同一 stream 调用各版本；同步后核对。调用前显式建立 stream 依赖。
5. 恢复所有原地址中的输入值，重新 forward/Graph 并检查原 oracle 与输入 hash。
6. 单独检查 T0 prepared wrapper 返回空 BF16 output、空 launch tuple，无 CUDA core 启动。

prepared 期间会初始化/编译现有 Triton 路径；新 CUDA plan 随后替换 gate/up，
各版本共享输入/权重，持有各自 output/metadata workspace。整个审计没有净性能计时。
任何失败写入新 ignored artifacts，保存实际值与源码版本，不覆盖旧失败。
数字和正确性记录可以发布；原模型权重和原始 trace tensors 保持本地。

BF16 基础通过后再冻结独立性能协议；W4A16 的核对使用相同量化权重的 oracle，
不能直接套用本轮原权重 oracle 来评价实现误差。
