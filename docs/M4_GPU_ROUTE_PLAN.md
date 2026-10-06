# M4：先建立完整 GPU 路径

M3 只测 host 已知 counts 的预分组 GEMM。M4 的第一步接收运行时 GPU IDs，
把 grouping、gate/up、SwiGLU、down、weighted combine 全部纳入一次 forward。
先固定原始 tile，不将 M3 的 host density 选择规则偷偷用于动态 GPU 路由。

## 预定实现

- 固定 shape/buffers，输入 values、IDs 和 routing weights 可更新。
- GPU histogram 得 counts；单 CTA prefix sum 得 rows offsets 和 GEMM task offsets。
- Atomic cursor 给每个 route 分配 expert 段内位置，复制 X 到 grouped buffer，
  同时保存原始 route -> grouped row 的 inverse map。duplicate IDs 作为独立 route。
- 固定的保守 task 上界 grid，通过 GPU task offsets 定位 expert，越界 CTA 退出。
  不读 counts 回 CPU，不按 expert 最大容量 padding；中间行数为 T*topk。
- gate/up GEMM BF16 输出；独立 SwiGLU kernel FP32 activation、BF16 输出；
  down GEMM BF16 输出；按原始 slot 顺序 gather 加权，FP32 reduce 后输出 BF16。
- 不在第一版引入原子浮点 combine，避免结果由 dispatch 的原子顺序决定。
- 所有临时 buffers 在 prepare 分配；run 无分配、无 tensor.item/CPU readback。
- caller 保证 IDs 属于 [0,E)、权重 finite，建立跨 stream 的读写依赖；不并发 run。

## 先验证什么

空 T、空 expert、重复 IDs、topk1/2/4/8、M/H/I 尾块、热点和随机 IDs。
与原有完整 FP32 MoE oracle 对照，阈值保持 relative RMS/normalized max <=0.008。
再用逐阶段 BF16 PyTorch oracle 辅助区分量化误差和实现错误。
固定 Graph 捕获后更新 IDs/输入 values，再重放并验证；不能只验证固定 IDs。
GPU histogram/counts 与 CPU 独立统计作诊断，统计 readback 不进入 forward/计时。

计时完整 forward 和主要阶段，公开 metadata/dispatch overhead。
第一版是功能基线，不声称超过成熟完整 MoE backend。单 GPU、forward only，
不包含 learned router、通信、serving、backward 或真实模型质量验证。
