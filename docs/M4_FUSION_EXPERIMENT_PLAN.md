# M4 局部融合固定实验

在完整GPU route baseline之后，只改变gate/up投影与SwiGLU的局部组织。
固定BM32/BN64/BK32/4warps/3stages，不搜索新配置或根据结果挑形状。

每CTA同时计算gate/up两份FP32累加，各自BF16 round-trip后计算SwiGLU，
直接写activation。原有projected workspace仍分配，因此只可能减少流量/启动，
不能声称降低allocated workspace。全部GPU dispatch/down/combine不变。

功能输入沿用M4所有9种边界，再用固定Graph更换IDs/输入/权重验证。
对照完整FP32和逐阶段BF16 oracle，也对照unfused路径，记录实际误差。
性能预定8组，seed73：small(T17/193,E16,H128,I64,topk4)随机；
Qwen MoE(T17/193/1024,E128,H2048,I768,topk8)随机和16expert热点。
每组给两个实现同一组buffers，prepare/oracle不计时；4轮AB/BA/AB/BA，
每轮6样本，每样本2次调用，Graph每replay8次完整forward。
记录API/Graph/p95、全部样本、输入/源码hash、遥测和资源；慢例不删。
只比较本项目两种完整forward，不声称超过成熟完整MoE backend。
