# M18 编译缓存与动态地址绑定（预先声明）

日期：2026-10-07 Asia/Shanghai。延续 M17，冻结 M16 选择和全部 M17 源码。

## 工程判断

真实输入 V 为 stride6144；首次 T96/L2048 编译约1秒。不能每层/请求重建绑定并执行
warm attention。新模块使用 Triton3.3.1 warmup 只编译，缓存不持有输入或输出指针；
独立 workspace 保存输出/必要转换，每次运行检查元数据并重新绑定指针。

缓存键包括 device、能力、T、四段长度与 AttentionConfig。LRU 上限仅限制本模块
持有的模板；Triton 自身 JIT/disk cache 不受此上限控制。clear 不会销毁活跃 workspace
持有的模板。缓存未声明线程安全。

workspace 固定 shape/stride/device 和创建时 stream；调用者建立输入跨 stream 依赖。
另一 stream 必须使用独立 workspace。record_stream 保护输入释放后的 allocator
生命周期，不替代依赖同步。输出下一次运行覆盖，不可并发写。Graph 捕获后仍绑定
捕获时地址，动态 eager 地址替换不使已有 Graph 自动更新。

## 验证与成本

- 独立 FP32 oracle、原 .008 双阈值和 finite；新地址/新值、同址更新、独立 side stream、
  输入临时 owners 释放、布局/shape 错误拒绝、T0与域外 fallback、LRU 淘汰和 clear。
- 固定 Graph 地址下更新值并检查；更换地址只在 eager 或新捕获图进行。
- 重放 M17 全部12组真实模型输入；强对照为复用 pack + 官方 FA2，另列 M17 固定绑定
  与每次重新准备。所有动态检查/地址绑定/record_stream/V copy 纳入 API 热路径。
- 4轮正逆顺序、8样本、2调用，Graph8forward。编译和 workspace 准备分别记录，
  不冒充全冷启动；不将单个顺序首次准备时间当公平 A/B。
- 保存失败、原始样本、输入/源码/manifest hash；不改冻结策略、不声称完整请求收益。

本阶段完成后再准备 NanoKV 实验入口；默认 FA2、选择/Top-K/KV 语义保持既有约定。
