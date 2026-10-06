# M3 预先固定的实验设计（2026-10-06）

目标：分别测提交入口优化与同一 grouped GEMM kernel 的分块参数优化。
本轮保持数学语义、BF16/FP32、host-known pregrouping 与无 M padding 不变。

## 1. Prepared 入口

准备阶段调用现有公开入口做校验/编译，持有原 buffers、metadata 和 kernel。
运行时复用 Triton3.3.1 CompiledKernel launcher 与校验后的指针，仍取当前stream。
不重新做Python shape/alias检查或JIT缓存查找。要求调用者不resize/替换buffers、
不改变metadata，允许更新输入values；跨stream依赖显式建立。
这是固定buffer接口契约的变化，不能称为同一宽泛API的无代价加速。
分别比较public/prepared的API与Graph时间，确认收益发生在哪一部分。

## 2. 参数扫描（训练集）

40个预定配置：12种(BM,BN,BK)×(4warps/2stages、4/3、8/3)，加4个4warps/4stages。
包含原始32/64/32/4/3。明确记录编译、资源和数值失败，不删慢配置。
BM16/32/64、BN64/128/256、BK32/64；候选只是基础分块参数，没有新算法。

只用Qwen gate/up的N1536/K2048、E128/topk8：
T64 balanced、T64热点、T512 balanced、T512热点、T4096 balanced。
训练seed=20261006；路由沿用M2 modular/热点规则。
所有候选计时前对照FP32 oracle。快速扫描使用Graph含4次GEMM，2轮AB/BA，
每轮3样本；保存每条原始时间与原始配置配对比值。

根据活跃expert平均行数选择三个预定bucket：<=8、(8,64]、>64。
每bucket用训练形状的candidate/original时间比值的几何平均排名。
最佳配置需至少5%预测改善；否则保留原始配置。
最佳候选在训练集用更长4轮对称A/B复测，确认仍改善至少5%，再冻结选择规则。
规则只用于N1536/K2048；其他N/K预先规定回退原始配置，避免未经验证的泛化。

## 3. Holdout（规则冻结后才能计时）

未用于搜索的T17/T193/T1024/T2049，CPU随机无重复topk，seeds73/991。
增加热点随机分布及down N2048/K768场景（预定回退），small E8/topk2。
比较original prepared、selected prepared、固定CUTLASS。保存冻结规则hash、
形状/ID/counts/输入hash、FP32误差、API/Graph p50/p95和原始样本。
结果不满足预期时公开回归，不用holdout重新挑参数。

## 4. 验证和解释

选出的配置另验空输入、空expert、M/N/K尾块、输入values更新、非默认stream、
固定Graph重放。保存PTX/SASS、寄存器、spill和shared memory信息。
没有硬件计数器时，访存/occupancy原因只列为假设，不称为已profile验证。
笔记本热/时钟未锁定，配对轮次仍有噪声；报告范围限定于本机和预定输入。
