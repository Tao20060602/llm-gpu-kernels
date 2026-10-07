# M22 W4A16 首版：实现误差审计协议

2026-10-07，运行前冻结。首版同步 tile 内解码与 BF16 WMMA；BM32/BN64/BK32，
两个 M16 都计算，保持与 CUDA sync BF16 基础的算术组织一致，不同时引入 M16 skip。
128 threads：每个 thread 解码一个 output-channel 的32个权重，读取一个16B packed
word vector及该K128组的BF16 scale；写 shared BF16 tile 后计算，不生成完整 BF16 权重。
它不是原生 INT4×BF16 MMA，也不是异步 decode 版本。

新 W4 workspace 独立分配 ranked metadata / grouped input / activation / down output，
复用原 histogram、prefix、dispatch、down、combine kernels；不借用需要原 BF16
gate/up tensor 的 preparer，亦不持有 projected scratch。固定地址/当前 stream/无热路径
分配与无 CPU counts readback的合约不变。不能因此将当前审计的CUDA allocator峰值
直接称为量化单路径峰值：审计另外持有完整 BF16参考权重。

使用已准备且核对过的 `gate-up-w4g128.pt`，与 format-preparation 记录核对文件hash、
words/scales数据hash、source BF16权重hash及格式字段。pack准备完全在计时外。
首版同一个 packed tensor在六个真实输入间复用，words/scales只读；down保持原BF16。

六个case同 `M22_CUDA_AUDIT_PLAN.md` 的七状态及T0，唯一候选 `w4_sync`：
原API/Graph、IDs=0/127动态Graph、非默认stream、恢复后的API/Graph。
独立FP32 oracle以**同一解量化BF16 gate/up权重**和原BF16 down为operand，
finite / relative RMS / normalized max≤0.008；每状态核对完整forward与ranked dispatch。
不以原BF16模型误差判断CUDA实现对不对。

每case原始真实输入另报：量化oracle相对原BF16 oracle、W4 CUDA相对原BF16 oracle
的relative RMS/normalized max，不按0.008判定格式损失，更不换算任务质量分数。
没有完整模型任务评测，就不声明满足任务得分下降≤10%。

输入、原weights/trace、packed artifact、源码、加载的binary hashes前后核对；
捕获Graph后words/scales hash不变。失败保留，本轮不作净性能或winner判定。
审计通过后再执行独立冻结的性能协议，包含转换在内的强对照另报。
