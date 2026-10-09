# M23复测后的简历条目与面试提纲

更新日期：2026-10-09。性能表述限定于固定layer0 trace，不外推到全模型或生产流量。

## 项目名与技术栈

**面向 Ampere 的 LLM GPU 算子优化**

CUDA C++、Triton、BF16 Tensor Core、W4A16、CUDA Graph、Nsight Systems/Compute。
硬件为 RTX3080 Laptop 16GB / SM86；项目包含动态 MoE 与分段 GQA Attention forward。

## 精简版（三条）

- 构建动态 MoE expert-forward 算子链路，采用GPU histogram/prefix/ranked dispatch，
  融合gate/up与SwiGLU，完整七次kernel launch；固定workspace支持路由更新后的
  CUDA Graph与非默认stream，覆盖重复routes、空expert及tails。
- 根据Nsight带宽瓶颈设计SM86手写BF16 WMMA核心与W4A16在线解码融合，采用
  CTA32×64×32、K128分组；gate/up逻辑权重768→198MiB。模型提取的T17/T193
  四个固定输入经过三批复测后，相对原BF16 vLLM完整Graph p50等权几何平均耗时降低
  10.07%；两个T1024输入慢17.58%，18项混合总体仅低1.66%。相对同量化权重vLLM，
  小组低10.00%、T1024组慢17.30%。只描述这六个真实trace；详见[M23报告](M23_W4_REPEAT_REPORT.md)。
  同量化权重oracle最大实现RMS0.00332，量化层输出格式损失另报。
- 实现四段KV直读的GQA Attention与在线softmax，结合资源分析设计形状范围与FA2
  fallback；36个实际走自定义路径的合成输入，较pack+官方FA2的Graph p50配对
  几何平均耗时降低22.09%，最坏回归2.17%。

数字对应各报告的热重复算子级测量。W4的T1024回归、固定输入边界与Attention回退
范围须能在面试中解释，不能把上述小输入或自定义路径结果扩为完整服务收益。

## 可以直接使用的主体表述

- 实现 GPU histogram、prefix 与 ranked dispatch，支持重复路由、空 expert 和尾块；
  融合 gate/up 与 SwiGLU，完整 MoE expert forward 为七次 kernel launch，采用固定
  workspace 与当前 stream，支持输入/路由值更新后的 CUDA Graph 重放。
- 使用 Nsight Systems/Compute 定位小输入 MoE gate/up 的权重带宽瓶颈；在真实
  layer0 T17 场景中，该阶段占关联 GPU 时长约66%，NCU replay 下 DRAM 吞吐约
  峰值95.8%、L2 hit约0.5%，据此选择减少权重字节而非继续盲目扩大 BF16 tile。
- 手写 BF16 WMMA CUDA 核心，采用 CTA32×64×32、WMMA16×16×16、16B搬运及
  cp.async双缓冲，完成同步/异步/无效M16跳过消融；实现 K128 分组 INT4权重
  打包、tile 内在线解码与 gate/up+SwiGLU融合，BF16 activation/FP32累加。
- 真实 Qwen3-30B-A3B layer0 gate/up逻辑存储从768MiB降至192MiB packed+6MiB
  scale；down保持BF16，两个权重张量总量1152→582MiB。完成42个 W4动态路由、
  Graph和非默认stream状态的数值与dispatch检查，同量化权重oracle的最大relative RMS0.00332。

可补入的具体性能表述：**在 RTX3080 Laptop、固定Qwen3-30B-A3B layer0路由与输入、
topk8 的T17/T193四个case中，三批Full Graph p50等权几何平均相对同量化权重vLLM
降低10.00%，相对原BF16 vLLM降低10.07%；T1024两case分别慢17.30%和17.58%，18项
混合总体仅低1.69%和1.66%。** 这只适用于固定六个输入；没有统计置信区间，GPU时钟
未锁定。量化层输出相对原BF16的差异约2.6%–4.5%，不是任务分数。完整逐项结果、p95、
同量化强对照及负收益见[M23报告](M23_W4_REPEAT_REPORT.md)。

上述条目可按版面压缩成三条；硬件、scope与数值界要保留，避免写成全模型生产部署。

## 面试中的思路顺序

1. **先定义测什么**：给定固定 router 输出的完整 MoE，与仅已分组 GEMM 分开。
   Python expert loop只作诊断；使用官方 vLLM、冻结 Triton和同 tile CUDA为对照。
2. **先消除中间开销**：GPU分组、ranked dispatch、gate/up+SwiGLU融合，避免CPU
   counts回读和中间 projected tensor 的热路径依赖。
3. **看真实瓶颈**：NCU带宽/命中率与timeline决定下一步，不以Tensor Core指令
   存在就声称利用率高。T17稀疏路由的有效BM32行占比低，算术与权重流量分别考虑。
4. **验证直觉失败**：手写BF16 async有局部改进，六真实输入三批完整Graph仍比
   冻结路径慢。BN64与旧BN256在CTA数、A重读、shared/register成本上有取舍。
5. **改变流量账本**：W4减少权重字节，但付出bit解码、scale处理、布局和舍入成本；
   用同一量化权重的常驻BF16强对照验证，避免每次Python解量化的弱基线。
6. **区分两种误差**：实现误差≤0.008沿用不变；六真实layer输出格式代理RMS约
   2.6%–4.5%，不是完整模型任务得分，不据此声称满足任务分数下降≤10%。

## 不能写进简历的表述

- “显存峰值下降49.48%”：当前数字是逻辑权重存储，不含workspace/Graph pool。
- “INT4 Tensor Core原生计算”：实际是INT4存储，在线还原BF16后BF16 WMMA。
- “超过Marlin”或“生产环境部署”：当前没有相同混合量化域的完整Marlin对照或部署。
- “异步搬运使BF16整体加速”：已完成结果不支持此说法，保留负收益及其分析。
- “模型质量只降4.5%”：这里是layer0输出RMS，尚未测完整模型任务得分。

与推理引擎项目分别描述：CPU offload、KV检索与TTFT归NanoKV，算子仓库体现的是
tile、layout、pipeline、fusion、quantization、profiling及算子级基准验证。
