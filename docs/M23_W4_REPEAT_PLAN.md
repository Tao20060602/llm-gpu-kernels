# M23 W4A16：跨进程复测协议

2026-10-09。状态：运行前协议，尚未采集追加性能结果。

## 要回答的问题

M22 首版只完成六个真实 layer0 输入的单批测量：T17/T193 的四个输入有收益，
T1024 的两个输入回归。本轮追加两批独立进程，检查该长度相关现象是否重复出现，
为下一轮布局和解码优化提供基线。它不选择新 tile、不修改量化格式或数学语义。

本轮仅测给定路由的完整 expert forward 和已分组 gate/up；不包括 router、
模型任务质量、冷启动或推理引擎 TTFT。

## 冻结内容

- 继承 `M22_W4_PERFORMANCE_PLAN.md` 的六个 pinned 输入、六个完整 forward arm、
  三个已分组 gate/up arm、数值门槛及 API/Graph 计时方法。
- 保持 W4G128、同步 BM32/BN64/BK32、BF16 down、原 GPU ranked dispatch/combine。
- 使用 M22 已审计的 BF16/W4 extension cache 和实际 binary hashes；加载失败时停止，
  不重新编译后接受新的 hash。原始源码、审计、首批结果和冻结 M7 policy 不改。
- 每个 case 使用一个新进程；单 GPU 串行。准备、packing、上传、oracle、编译及
  profiler 不进入计时，保留每个 worker 的完整样本与前后检查。

## 两批追加运行

每批六个输入，继承 worker 的原有正逆 arm 顺序：完整链路记录十二轮、阶段记录
六轮，每轮六样本、每样本两调用。Graph 每 replay 八次 forward，计时按一次
forward 归一化。命令使用仓库 venv；先在命令环境清除继承的 CUDA 13.2
CUDA_HOME/CUDA_PATH/CUDACXX/LD_LIBRARY_PATH，再将 CUDA_HOME 显式指向与原审计
匹配、nvcc 实际报告 12.8 的工具链目录。其余三个变量保持清除，不修改 shell 配置。

case 顺序用于减轻温度和时间顺序影响，不能改变 worker 内的 arm 计时顺序：

1. 追加批次 2：首批六 case 顺序的反序。
2. 追加批次 3：按 code/archive 交替，从 T193、T1024、T17 依次运行。

每批使用新建输出目录；已有目录拒绝覆盖。记录 batch ID、执行顺序、驱动、
GPU 温度/时钟的可读信息和全部 worker 路径。未锁频，环境变化和异常须在报告中说明。

## 汇总和判断

先核对输入、packed 权重、kernel source、实际 binary 和 audit provenance 一致，
再逐 case、逐 batch 报告完整 API/Graph p50/p95 与 candidate/reference 时间比。
原首批单独列出，追加两批单独列出，三批总体为等权时间比的几何平均。
不得挑选最快批次、删除 T1024，或把独立阶段时间相减当作完整链路收益。

保留两种比较：同量化权重的 BF16 vLLM 对照用于比较实现路径；原 BF16 vLLM 对照
用于展示带格式损失的延迟取舍。量化权重的 BF16 对照预先还原并常驻 GPU，
不加入每次 forward 完整解量化的人为额外成本。

T17/T193 与 T1024 分层仅描述原报告已明确的长度差异，不构成新动态调度策略。
若任一批小输入组的 Graph 几何平均比值达到或超过 1，明确报告该组收益未在所有
批次复现；总体比值低于 1 也不能替代这个说明。三批不提供统计置信区间或普遍性保证。

## 异常处理和下一步

任何数值/provenance 失败均保留失败记录并停止该 worker，不放宽原 0.008 门槛。
CUDA 初始化失败时不采集性能、不生成虚拟结果，也不将失败列为零耗时。

初次运行前检查中，驱动日志出现 `NV_ERR_RESET_REQUIRED`，最小 CUDA 初始化失败。
用户重启后，最小 CUDA 分配与同步已恢复正常，设备恢复不涉及改动 kernel 或 venv。

追加采集 v1 在扩展加载前失败：清除继承环境后，加载器要求显式 CUDA 12.8 目录。
保留 v1 的失败 JSON、日志和 manifest。原审计实际使用的工具链是
`/home/tmz/ai/toolchains/cuda-12.8`，nvcc 12.8.93；缓存中的 build.ninja 记录该目录。
它并非 `/usr/local/cuda-12.8`，不根据惯用安装位置判断工具链不存在。

v2 指定正确工具链后，旧 JIT loader 重写 build.ninja 并触发编译，因当前 host GCC
版本高于 CUDA 12.8 支持范围而失败。此前 `ninja -n` 无 action 不能保证 Python JIT
loader 不重新生成构建规则。失败发生在计时前，两份审计 `.so` 的 SHA 仍匹配；
v2 的失败 JSON、日志和 manifest 保留。

v3 使用独立的复测启动 adapter，直接加载并核对两份已审计 `.so`，填入旧 loader
已经存在的模块缓存，再执行未修改的原 worker。启动阶段不调用 JIT、nvcc 或 ninja。
每个新 worker 另存 startup receipt，记录 adapter/worker/audit/binary 的 SHA、实际
加载路径和加载方式。该 adapter 不改 native 函数、输入、quantization 或计时逻辑。
原首批使用 JIT 准备，新两批使用已审计 native binary 准备；准备均在计时外，报告中
保留这个差异，不把新启动 adapter 伪装成原首批原样命令。

下一轮优先保持同一 W4 算术，研究 packed 访存映射与 decode/staging 的成本；
group size、异常值处理、down 量化、Marlin 和全模型任务质量分别立项，避免同时
改变布局、格式和算术而无法归因。
