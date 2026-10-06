# M17：真实 Qwen3-4B 输入与实际 V 布局成本

输入由NanoKV现有FA2路径采集：固定4B snapshot1cfa9a7208912126459214e8b04321603b3df60c，
两条16480-token生成文本prompt，chunk4096加96尾chunk，layers0/17/35共12项。
采集source与模型权重SHA256、prompt IDs、selection/protected IDs、状态和原strides
保存在manifest；CPU大tensor只在本地ignored bench_logs。数据采集带同步/磁盘I/O，
其步骤wall不能作为干净系统对照。

重放恢复Q/current K/V的原始strides（实际V token stride6144，连续V为1024）。
先独立FP32检查原FA2输出，再比较四个完整forward arm：

1. legacy_raw_pack：GPU resident分段输入，旧分配+逐段copy+公开FA2。
2. reusable_raw_pack：复用packed buffer、cat out支持strided V+公开FA2，强基线。
3. m16_boundary：冻结policy选择；direct模式每次复制必要的非连续Q/K/V到预分配
   contiguous buffer，再执行M15 kernel；fallback模式直接用独立raw pack+FA2。
4. m16_contiguous_core：预先连续化的冻结算子，仅用于拆解转换成本，不能替代arm3。

所有输入GPU resident，不计model/capture、CPU gather/H2D、加载/prepare/hash/oracle。
准备同步wall单列（本进程/已有编译缓存，不是首次安装或完整冷进程启动）。
API/Graph四轮正/逆，8样本2调用，Graph8forward；全部arm前后独立chunk64全heads
FP32、TF32 off、finite/原0.008界、输入hash，记录raw event/wall/p50/p95及峰值。
冻结M16 policy/source与本轮source hash运行前后校验；不据真实数据调配置/阈值/域。

这只是模型tensor上的算子边界重放；固定指针/shape，仍未实现动态地址adapter，
没有NanoKV feature-on/off或greedy输出对照，不能写成模型/请求加速或质量无损。
若精度失败，保留失败记录并先诊断，停止性能推广。
