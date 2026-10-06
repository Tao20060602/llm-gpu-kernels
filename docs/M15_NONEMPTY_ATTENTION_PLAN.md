# M15：在现有契约下简化在线 softmax 状态

数学依据：T>0，每CTA token_start<T。所有历史段的首个实际循环block至少有一个
有效key并对全部query行可见。如果所有历史为空，current首block包含key0，
key0<=所有逻辑query位置；current循环stop=min(T,start+tile)>0。
因此首次执行block时所有内部行m_new有限、l>0。后来因果全遮罩block保持有限m/l，
exp2(-inf-m)=0、exp2(m-m)=1。填充query行的Q读为0、O写被mask，允许内部
fake行softmax；它们没有跨行影响。T0完全跳过kernel。

去掉q_valid在scores mask中的传播、valid整数归约、block_active/m_safe分支和最终
safe_l。其余four-segment索引、head映射、BF16 P、FP32状态、当前stream保持M9。
仅对有限输入研究，不扩展nonfinite语义；不改旧源码和冻结策略。

预先固定token32/key64/G1/warps4的stages1/2，无搜索/选择策略。各自与同配置原M9
对照，强baseline复用buffer pack+未修改公开FA2。五arm：old_s1/old_s2/new_s1/new_s2/fa2。
功能：new S1 G1与new S2 G1/2/4，各12小场景、两次Graph更新、非默认stream、恢复。
性能seed11109：T64/127/256/1024 × selected0/513/2049，共12，另T4096 selected513
长query检查共13。sink64/recent512/currentT。所有arm API/Graph前后独立chunk64 FP32、
finite、原0.008界、输入hash。四轮正/逆、8样本2调用，Graph8forward；不计准备/
oracle/hash/capture/编译。保存raw event/wall、p50/p95、resources、HMMA、source哈希。

本轮为固定配置算法简化A/B，结果不是选择后的holdout，不按其赢家改旧policy。
如有收益，后续需要另立冻结验证；推荐默认FA2不因单项实验自动改变。
hot GPU resident，非model/H2D/RoPE/NanoKV/serving；静态资源不是utilization/occupancy。
