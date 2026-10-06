# M18：编译缓存与动态地址绑定

日期：2026-10-07 Asia/Shanghai。状态：独立算子入口实现、验证和真实输入重放完成。
NanoKV 实验入口及完整请求 A/B 尚未实施。

## 工程结论

M17 的固定 plan 不能随新请求直接替换输入地址，每次重建又执行一次 warm attention。
M18 新增不持有张量或地址的编译模板缓存，以及独立持有输出/转换 buffer 的 workspace。
同一元数据下每次 run 接受新地址，检查 shape/stride/device、登记输入 stream 生命周期，
必要时转换布局，然后启动原 M15 内核。M16 选择完全不变，M17 的34份源码逐项未变。

- 使用 Triton3.3.1 `warmup` 只编译、初始化 launcher，不执行 attention。
- 主重放缓存：1个形状模板、1次 local miss、13次 hit。hit 不再执行 warm attention。
- local LRU 上限16；Triton自己的 JIT/disk cache 和活跃 workspace 持有的模板不在此上限内。
- workspace 固定创建 stream，其他 stream 拒绝复用；独立 stream 创建独立 workspace。
  调用者仍须建立 producer 依赖。`record_stream` 保护 allocator 生命周期，不负责数据同步。
- Graph 仍固定捕获地址；动态 eager 地址替换不会更新已有 Graph。图及输入 owners 必须保留。

## 验证证据

主记录：[`m18-dynamic-20261006T163700Z-761619917.json`](../results/m18-dynamic-20261006T163700Z-761619917.json)。
补充布局审计：[`m18-layout-audit-20261006T163957Z.json`](../results/m18-layout-audit-20261006T163957Z.json)。
汇总：[`m18-summary.json`](../results/m18-summary.json)。

7项功能场景覆盖 T0/31/32/64/96/128/129、域外FA2、empty selected、新地址新值、原输入
恢复、固定图两次值更新、独立 side stream、输入临时 owners 释放后的 allocator pressure。
缓存上限2的审计实际发生一次淘汰；两个消费者共享模板但输出不同；clear后旧workspace仍能运行。

3项补充布局场景覆盖非连续Q、历史K、全部9个输入的 padded token stride；全部重新绑定地址。
6项错误拒绝覆盖合法但不同的shape、未对齐基址、head stride、重叠token stride、合法但
改变的stride及CPU输入。它们不扩展到任意strided tensor、多GPU或线程并发。

12项真实模型重放使用M17同一完整manifest和原始文件SHA256；恢复真实current V stride6144。
每项全部arm API/Graph前后分别对独立FP32 oracle检查，并额外更换模型张量地址。
总计227项数值指标全部finite、通过原 .008阈值：最大relative RMS **0.00226431**，
最大normalized max **0.00397238**。主记录37份源码和补充审计38份源码前后hash一致，
M16的31份冻结源码/training/policy provenance前后通过；未用本轮数据改配置。

## 成本与性能

同机同进程交替比较：4轮正逆顺序、8样本、每样本2次调用；Graph每次8forward。
API纳入动态元数据检查、指针提取、record_stream、布局转换和计算。Graph replay没有Python
重新绑定成本，只覆盖捕获好的固定地址。两者必须分开解释。

| 真实尾chunk | 动态API μs | 强复用pack+FA2 API μs | Graph动态/强基线 |
| --- | ---: | ---: | ---: |
| archive layer0 | 192.54 | 285.70 | 0.6297 |
| archive layer17 | 191.17 | 288.75 | 0.6279 |
| archive layer35 | 194.30 | 290.58 | 0.6333 |
| code layer0 | 198.14 | 294.66 | 0.6280 |
| code layer17 | 190.46 | 285.42 | 0.6268 |
| code layer35 | 190.72 | 286.52 | 0.6275 |

六项T96尾chunk API ratio几何平均 **0.668328（耗时少33.17%）**，全部少32.61–33.80%。
同步wall ratio几何平均0.671918（少32.81%）；该wall含event创建与提交，不是纯host开销。
Graph ratio几何平均 **0.628863（少37.11%）**。

相对M17已经绑定好地址的入口，动态API几何平均慢 **7.72%**，Graph慢约0.49%。
这是动态检查/生命周期处理的实际成本；本轮没有提出新数学内核优化。
相对每次重建M17 plan的诊断arm，动态API少52.44%，但该arm不能替代强FA2对照。

六项T4096保持FA2：API ratio几何平均1.003362，范围0.999146–1.007772；
Graph几何平均0.999126，范围0.991698–1.001340，接近基线，无自定义大chunk收益。

首次显式预编译本进程约 **321.76ms**，已经存在磁盘/JIT缓存，不能称为全冷启动，也不能
与M17顺序首次prepare约1.03s当公平A/B。预编译后六项尾chunk workspace准备约
**0.151–0.337ms**，仍含分配和验证，均在热forward计时之外。

直接路径每次只转换一个current V，复用转换空间 **192KiB**；另持有T96输出 **768KiB**。
FA2 fallback复用packed KV且使用官方公开API分配输出。目前该通用workspace还预留一份
未使用的Q形状output；T4096额外 **32MiB/实例**，接入引擎时应避免给fallback层长期持有
这份预留，不能把本入口不加改造地扩展为36层的生产内存方案。

## 使用与下一步

见 [`M18_DYNAMIC_BINDING_USAGE.md`](M18_DYNAMIC_BINDING_USAGE.md)。
本轮GPU测量开始前等待另一个OpenCode基准进程退出；在开始及中途抽查时仅见本基准计算进程，结束检查无其他计算进程。
运行环境沿用Python3.12.14、torch2.7.1+cu128、Triton3.3.1、官方FA2 2.8.3.post1，SM86。

下一阶段应做显式开启的NanoKV实验adapter：只在已验证形状复用direct workspace；
fallback使用独立强FA2拼接路径，避免上述预留输出与过多长期workspace。CPU gather、H2D、
选择IDs、保护窗口、KV store先后顺序和默认FA2保持既有约定。编译预热单列，逐步核对层
输出、选择与状态，再做至少3次新进程配对请求A/B。

本轮是GPU-resident真实张量重放，排除CPU gather/H2D、模型其他层、KV更新和请求启动。
**33.17%不能写成完整请求加速**；首次编译和只有尾chunk适用可能使实际收益很小。
没有新的profiler计数器，不作utilization/occupancy/DRAM结论。
