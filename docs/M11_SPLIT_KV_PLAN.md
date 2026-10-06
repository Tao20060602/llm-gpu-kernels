# M11：短 query split-KV 与完整 merge 成本

在本轮测量前固定：只在T<=256研究短query，无Hopper机制，SM86 BF16 Tensor Core。
M10已说明G4在短T减少CTA，而静态资源/训练收益不能直接推广到T256。
本轮将逻辑KV block分为互不重叠的连续区间，增加独立CTA，写FP32 partial(m,l,acc)，
第二kernel按共同最大值重新加权归一化。两个启动、scratch读写和merge全部计入。
这仍是已有online-softmax/partition merge方法的固定四段GQA实现，不声称新算法。

契约与M9相同。原始K/V指针按逻辑lane选段，历史全可见/current因果。每split区间
与总KV block数成比例；clip到当前query tile可见的最后block。有的partial没有
任何可见key，必须输出l0/acc0/m=-inf，merge忽略该partial；有效query至少有self。
Q tail、空段、T0无启动，cached currentstream owners同前。scratch为
splits*T*32*(128+2)*4 bytes；输出另T*32*128*2 bytes。

固定8候选：token_tile16/32 × splits1/2/4/8，G4/warps8/key64/stages2。
即使splits1也计merge，不用省去第二kernel来制造不匹配对照。
所有候选先T65/history33/17/41与T0数值检查，FP32/finite原0.008界不变。

训练seed3109，T48/112/224 × selected383/1793，sink64/recent512/currentT，共6输入。
故意保留非整齐segment边界。成熟强baseline为每次复用buffer pack+未修改FA2公开API，
全部使用独立chunk64 FP32 reference。48候选行完整Graph paired四轮AB/BA/BA/AB，
8样本/轮、2replay/样本、8forward/replay，API/Graph前后检查与hash、资源、scratch。
baseline失败停止；候选失败保留并排除。选一个全域配置，要求全部6训练输入通过、
geomean ratio<=.95且worst<=1.05，最低geomean/index tie；没有合格者fallback强FA2。
初选在同6输入长复测：4轮16样本4replay8forward，门槛相同，失败fallback无次选。
冻结源文件、最终training canonicalhash（只排除顶层policy_sha256）后才对照。

冻结后seed4207，T17/65/129/193/256 × selected0/512/2049，共15输入；外域T257
再1个selected512 fallback诊断。sink64/recent512/currentT。对照为冻结split或FA2fallback、
M9G1、强pack+FA2，四轮正/逆8样本2调用Graph8。全部API/Graph前后FP32检查、input hash。
唯一选定split另跑M9的12功能输入、两次Graph原地值更新与非默认stream；保存两kernel
编译产物/HMMA与资源、明确scratch和进程peak。不得根据该矩阵重新选择旧policy。
无counter不证明occupancy/DRAM/causal解释；不运行model、RoPE、H2D、NanoKV/serving。
