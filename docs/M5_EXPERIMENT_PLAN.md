# M5：完整成熟后端与入口缓存对照（先于计时固定）

完整成熟基线采用未修改的官方vLLM0.10.0 PyPI wheel，Git tag
6d8d0a24c02bfd84d46b3016b865a44f048ae84b。该版本要求PyTorch2.7.1，
使用原项目torch2.7.1+cu128/Triton3.3.1，依赖放到隔离overlay目录。
公共fused_experts包括native alignment、Triton GEMM、native activation/sum，
没有换成本项目dispatch或复制改写其kernel。记录官方自动配置和缺失的调优文件，
不覆盖默认config，也不宣称当前版本的最优实现。

vLLM在down FP32累加后乘route weight再转BF16，本项目先down BF16再FP32加权。
vLLM native SiLU先BF16舍入SiLU分支再乘up；topk2/3/4的native sum为BF16
逐slot累加，其他topk使用ATen sum_out。本项目SiLU乘up之后才BF16舍入、combine
统一FP32累加。分别记录这些边界，不把数学语义相同写成逐阶段舍入相同。
两者数学语义相同，舍入不同，各自对完整FP32 oracle用既定0.008误差阈值；
不要求它们逐bit相同。不能放宽阈值来掩盖失败。

入口优化只绑定M4已编译8/9个kernel和完整pointer/constexpr参数，GPU计算不变。
版本固定Triton3.3.1；输入values/IDs可更新，shape/address不可变；stream依赖显式。
保留所有owners，kernel runner每次取当前stream；无CPUcounts readback或GPU分配。

先用M4九个功能case验证cached fused/unfused及vLLM：空T、尾块、空expert、
duplicate IDs、不同整数dtype/非power-of-two expert、负routing weights。
固定图重放后更改输入/IDs/route weights，另验证cached当前非默认stream。

预定性能矩阵，seed73/991：Qwen H2048/I768/E128/topk8，T1/17/193/1024/2049
随机唯一topk；另T193/1024、seed73、16expert热点。
small E16/H128/I64/topk4，T1/17/193、seed73随机，共15case。
比较M4 fused Python入口、cached fused入口、vLLM公共API。三者同输入/数学。
六轮正序/逆序循环，让每个backend位次出现两次；6样本/轮，2调用/样本。
Graph每replay8次完整forward，先恢复原输入hash，再分别测API与Graph。
API公开各自检查/分配与host gap边界；Graph使用固定地址/pool，仍含图提交间隙。
保留全部原始样本、p95、误差、源码/输入/库hash、遥测和peak memory。
本轮不调GPUtile，不用holdout选择配置，不拿Python expert loop作性能对照。
