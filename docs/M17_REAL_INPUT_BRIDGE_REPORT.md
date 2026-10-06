# M17：真实模型输入与非连续 V 的完整算子成本

12项真实Qwen3-4B张量重放全部数值通过。在6项96-token尾chunk中，包含每次current V连续化复制的冻结边界路径，相对复用pack+官方FA2的Graph p50几何平均减少37.26%；单项减少37.02%–37.41%。4096主chunk实际选择FA2，不计作自定义加速。尚未进行feature-on/off系统A/B。

## 真实输入的来源与状态

NanoKV native clone、固定Qwen3-4B snapshot1cfa9a7208912126459214e8b04321603b3df60c。archive/code两条固定生成文本prompt各16480 token；四个4096主chunk加96尾chunk，YaRN factor4、query summary1、Top32、sink64/recent512、index_select、TP1/eager、CPU8线程。不是实际用户请求分布或质量benchmark。

layers0/17/35的第二个4096 chunk与96尾chunk，共12输入。原FA2输出继续驱动模型，benchmark只包裹实例方法采集输入/状态。保存prompt IDs、完整selected/protected IDs、42项native source hash、权重hash及CPU Q/K/V/原输出。每请求144个later-layer调用，重置/selection、12项采集的post recent/sink检查通过。大tensor在ignored bench_logs，manifest保存绝对路径及每文件SHA256。

接入前发现并修复原NanoKV短tail recent重叠copy_错误及短历史窗口增长时源区间错误。30项精确recent/sink/valid_len检查通过；修复记录在NanoKV docs/NATIVE_SHORT_TAIL_STATE_FIX.md。这是窗口状态修复，不是新的retrieval质量优化。

数据从旧packed buffer读取，属于完全相同FA2数学输入；原current V token stride6144另存，重放恢复该布局，连续V的stride为1024。Q/K实际连续。12项segment lengths均[2048,64,512,T]，selected排除protected且无重复。输入已经过RoPE，kernel不重复处理位置。

## 四个 arm 与公平边界

| Arm | 每次forward内容 |
| --- | --- |
| legacy_raw_pack | GPU resident原strided输入，新分配K/V、逐段copy、未修改公开FA2 |
| reusable_raw_pack | 强基线：复用packed buffer，cat out可读取strided V、未修改公开FA2 |
| m16_boundary | 原冻结选择；direct每次将非连续current V复制到预分配连续buffer，然后M15 kernel；fallback直接用独立raw pack+FA2 |
| m16_contiguous_core | 输入提前连续化的冻结算子，诊断用，不可替代包含转换的边界路径 |

所有输入已GPU resident，不计CPU gather/H2D、模型、capture、加载、prepare、hash或oracle。legacy在这里仅重建GPU pack/attention成本，不能描述成完整旧runtime系统耗时。采集含同步/CPU/disk I/O，原engine step wall只作诊断，不作正式baseline或加速。

API/Graph各四轮正/逆、每轮8样本2调用，Graph8forward；单位为毫秒/完整forward。输入固定shape/address，hot重复。保存原始event/wall、p50/p95；wall包含事件创建/同步等，不是单独host launch成本。不同arm相减不能当作孤立复制成本。

## 全部 Graph p50 / p95

| Capture | mode | legacy | strong reusable | M16 boundary | contiguous core | boundary/strong |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| archive-L0-P4096-T4096 | fa2 | 8.973216 / 8.999216 | 8.993405 / 9.056365 | 8.987712 / 9.052656 | 8.963008 / 9.046528 | 0.999367 |
| archive-L17-P4096-T4096 | fa2 | 9.106688 / 9.134378 | 9.124896 / 9.157322 | 9.119932 / 9.173110 | 9.109183 / 9.140639 | 0.999456 |
| archive-L35-P4096-T4096 | fa2 | 9.230496 / 9.252826 | 9.237942 / 9.257842 | 9.247104 / 9.322723 | 9.249816 / 9.289292 | 1.000992 |
| archive-L0-P16384-T96 | direct | 0.273789 / 0.275602 | 0.276153 / 0.279994 | 0.173600 / 0.177146 | 0.170688 / 0.173933 | 0.628637 |
| archive-L17-P16384-T96 | direct | 0.272896 / 0.275533 | 0.275776 / 0.279296 | 0.172608 / 0.176423 | 0.170556 / 0.173478 | 0.625899 |
| archive-L35-P16384-T96 | direct | 0.275168 / 0.276717 | 0.277120 / 0.281309 | 0.174528 / 0.176871 | 0.173120 / 0.174960 | 0.629792 |
| code-L0-P4096-T4096 | fa2 | 9.177696 / 9.208679 | 9.201406 / 9.229597 | 9.197824 / 9.232618 | 9.181696 / 9.203261 | 0.999611 |
| code-L17-P4096-T4096 | fa2 | 9.207264 / 9.222608 | 9.225568 / 9.249795 | 9.233728 / 9.250539 | 9.213728 / 9.261230 | 1.000885 |
| code-L35-P4096-T4096 | fa2 | 9.309248 / 9.333118 | 9.335674 / 9.362093 | 9.335062 / 9.365223 | 9.329792 / 9.362531 | 0.999934 |
| code-L0-P16384-T96 | direct | 0.273088 / 0.276694 | 0.277535 / 0.281777 | 0.174112 / 0.178040 | 0.171225 / 0.173968 | 0.627352 |
| code-L17-P16384-T96 | direct | 0.273564 / 0.277242 | 0.277056 / 0.281181 | 0.173531 / 0.177620 | 0.173948 / 0.174999 | 0.626339 |
| code-L35-P16384-T96 | direct | 0.276544 / 0.278714 | 0.279704 / 0.283156 | 0.175232 / 0.179059 | 0.172288 / 0.175718 | 0.626491 |

## 全部 API p50 / p95

| Capture | legacy | strong reusable | M16 boundary | contiguous core | boundary/strong |
| --- | ---: | ---: | ---: | ---: | ---: |
| archive-L0-P4096-T4096 | 8.847360 / 9.172919 | 8.816088 / 9.174596 | 8.885240 / 9.177344 | 8.965848 / 9.150870 | 1.007844 |
| archive-L17-P4096-T4096 | 9.042944 / 9.153117 | 9.009920 / 9.070259 | 8.996096 / 9.135172 | 8.989432 / 9.145011 | 0.998466 |
| archive-L35-P4096-T4096 | 9.213952 / 9.336704 | 9.210056 / 9.310413 | 9.218560 / 9.392026 | 9.189888 / 9.251226 | 1.000923 |
| archive-L0-P16384-T96 | 0.309504 / 0.328076 | 0.284416 / 0.323790 | 0.177408 / 0.186266 | 0.171512 / 0.175334 | 0.623762 |
| archive-L17-P16384-T96 | 0.309248 / 0.313805 | 0.283600 / 0.286870 | 0.177152 / 0.181195 | 0.172480 / 0.177613 | 0.624654 |
| archive-L35-P16384-T96 | 0.310528 / 0.314776 | 0.285952 / 0.289280 | 0.178408 / 0.183474 | 0.172032 / 0.175299 | 0.623909 |
| code-L0-P4096-T4096 | 9.165776 / 9.260058 | 9.168840 / 9.216665 | 9.178144 / 9.223797 | 9.158912 / 9.206195 | 1.001015 |
| code-L17-P4096-T4096 | 9.232896 / 9.326311 | 9.232592 / 9.271910 | 9.228312 / 9.297152 | 9.223896 / 9.340212 | 0.999536 |
| code-L35-P4096-T4096 | 9.262080 / 9.487642 | 9.260024 / 9.377997 | 9.263360 / 9.401489 | 9.264896 / 9.368277 | 1.000360 |
| code-L0-P16384-T96 | 0.315136 / 0.430155 | 0.286208 / 0.335274 | 0.178040 / 0.202419 | 0.173752 / 0.178890 | 0.622065 |
| code-L17-P16384-T96 | 0.310992 / 0.324256 | 0.285440 / 0.291558 | 0.177152 / 0.184851 | 0.172544 / 0.180375 | 0.620628 |
| code-L35-P16384-T96 | 0.311808 / 0.321698 | 0.286896 / 0.306534 | 0.179424 / 0.185034 | 0.173568 / 0.177611 | 0.625397 |

尾chunk API边界路径geomean ratio 0.623401。4096回退行是同一公开FA2的不同buffers/pools，微小差异不作为收益。

## 准备成本、精度与资源

| Capture | legacy prepare ms | strong prepare ms | boundary prepare ms | core prepare ms |
| --- | ---: | ---: | ---: | ---: |
| archive-L0-P4096-T4096 | 0.025354 | 0.053008 | 0.077613 | 0.509894 |
| archive-L17-P4096-T4096 | 0.023243 | 0.055673 | 0.101802 | 0.361307 |
| archive-L35-P4096-T4096 | 0.033858 | 0.058139 | 0.079766 | 0.369473 |
| archive-L0-P16384-T96 | 0.057832 | 0.062763 | 1026.441347 | 0.608002 |
| archive-L17-P16384-T96 | 0.012788 | 0.022871 | 0.377196 | 0.266040 |
| archive-L35-P16384-T96 | 0.012106 | 0.022751 | 0.381891 | 0.284922 |
| code-L0-P4096-T4096 | 0.017698 | 0.038070 | 0.071265 | 0.309995 |
| code-L17-P4096-T4096 | 0.016433 | 0.039232 | 0.061597 | 0.354256 |
| code-L35-P4096-T4096 | 0.015317 | 0.034136 | 0.055113 | 0.293260 |
| code-L0-P16384-T96 | 0.015511 | 0.039371 | 0.433870 | 0.294255 |
| code-L17-P16384-T96 | 0.012339 | 0.023953 | 0.372004 | 0.276513 |
| code-L35-P16384-T96 | 0.013325 | 0.026968 | 0.391812 | 0.293468 |

第一个96形状boundary准备约1026ms，后续同形状约0.37–0.44ms；这是本进程同步prepare，受现有disk/JIT cache影响，不是完整模型冷启动。核心prepare排在boundary之后，不能将它较小的时间描述成公平首次编译比较。当前固定地址接口重新prepare每个动态地址会引入明显成本，接入前需形状编译缓存及安全重新绑定，不能把prepare放进模型请求后仍引用热kernel收益。

共204项finite/误差metrics，最大RMS 0.001730225、normalized max 0.003526490，全部保持原0.008界。包括原native FA2输出对FP32及四个arm API/Graph前后独立全heads chunk64 FP32。输入hash未改，34项operator source运行前后一致、M16原31项provenance通过。

| Capture | conversion count / bytes | Registers | Spills | Shared B | SASS HMMA |
| --- | ---: | ---: | ---: | ---: | ---: |
| archive-L0-P16384-T96 | 1 / 196608 | 255 | 26 | 45056 | 256 |
| archive-L17-P16384-T96 | 1 / 196608 | 255 | 26 | 45056 | 256 |
| archive-L35-P16384-T96 | 1 / 196608 | 255 | 26 | 45056 | 256 |
| code-L0-P16384-T96 | 1 / 196608 | 255 | 26 | 45056 | 256 |
| code-L17-P16384-T96 | 1 / 196608 | 255 | 26 | 45056 | 256 |
| code-L35-P16384-T96 | 1 / 196608 | 255 | 26 | 45056 | 256 |

每个尾chunk每forward复制一次current V，连续化workspace196608 B（192 KiB）；自定义kernel输出另计，无softmax scratch/最终pack。强基线packed K/V workspace11141120 B（10.625 MiB）。进程peak含所有arm、原始strided tensor、oracle、Graph pool，不是单operator预算。HMMA和静态资源不能代替utilization/occupancy/counter。

## 接入判断

真实Q/K/V上的数值与完整边界局部成本支持继续做受控adapter。当前接口是绑定owners/shape/address的实验bridge，没有解决动态模型tensor地址的缓存/重新绑定；也未执行feature-on/off的greedy/selection/system对照。

4096主chunk仍走FA2，只有自然短tail可能受益，因此不能把约37%写成整条模型请求的加速。下一步先完成动态绑定/编译缓存和强复用FA2/H2D基线，再按NanoKV handoff做至少三组fresh-process交替A/B，保持检索和KV状态契约。没有按真实输入重选M16配置、阈值或域。

材料：[计划](M17_REAL_INPUT_BRIDGE_PLAN.md)、[完整JSON](../results/m17-real-input-20261006T154812Z-504068096.json)、[边界实现](../src/llm_gpu_kernels/attention_raw_boundary.py)。

复现（先在NanoKV native clone以新的output目录运行capture driver，再在本仓库）：

```bash
source scripts/activate-flash-attn.sh
python -m llm_gpu_kernels.attention_real_input_benchmark --manifest /absolute/path/to/capture/manifest.json
```
