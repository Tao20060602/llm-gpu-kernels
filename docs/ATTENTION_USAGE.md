# 四段 GQA attention：M16 prototype 使用

当前验证硬件为RTX3080 Laptop/SM86；BF16 forward、单请求，Q[T,32,128]、
四个连续CUDA K/V[L,8,128]，current长度=T。历史全可见，current j≤query_i，
GQA qhead//4、scale=1/sqrt(128)。输入已完成位置处理，不含RoPE/dropout。
非finite/FP32 score overflow、其他stride/head布局、backward和多卡未得到支持验证。

## 运行环境与入口

从独立仓库根目录：

```bash
source scripts/activate-flash-attn.sh
python -m llm_gpu_kernels.attention_resource_holdout
python -m llm_gpu_kernels.attention_resource_boundary_audit
```

使用接口（q与四段KV由调用者创建）：

```python
from pathlib import Path
from llm_gpu_kernels.attention_resource_policy import FrozenResourcePolicy

root = Path('/home/tmz/文档/ChatGPT/llm-gpu-kernels')
policy = FrozenResourcePolicy.load(root / 'results/m16-resource-policy.json')
policy.verify_provenance(root)
plan = policy.prepare(q, segments)
out = plan.run()
```

T32–128、sink64/recent512、currentT、总history≤4096用自定义M15默认配置；
其它metadata选择FA2，但仍须满足上述四段基本数学/输入契约。无效shape不会因
回退自动受支持。策略不读取GPU counts/values来选择。

## 生命周期、stream 与 Graph

- prepare在计时外验证shape/dtype/stride/device，分配workspace并编译/warm。
  首次准备后显式完成依赖再捕获Graph；冷准备成本没有纳入本轮结果。
- plan持有输入owners与固定指针/shape，输入值允许原地更新；换地址、resize或
  变shape需重新prepare。run使用调用者当前CUDA stream。
- 跨stream的数据依赖由调用者显式建立；同一个plan/output不并发运行。
  自定义路径复用output，下一次run可覆盖上次结果；消费者需要正确同步。
- FA2路径复用packed KV buffer，公开API eager调用仍分配/返回输出；Graph各自
  使用捕获的输出/pool。不能假设两条路径的output分配行为完全一致。
- 图捕获包含固定shape/address的热forward，不代表可复用动态shape图。
  同地址值更新与非默认stream功能检查目前在T9小输入完成，未对每个大形状重复。

## 结果边界

44项原冻结后对照+25项追加上限审计均通过独立全heads FP32/finite/0.008界。
36项实际custom路径Graph p50几何平均比完整pack+FA2少22.09%，最坏慢2.17%；
不是任意连续形状空间的性能保证，也不是请求分布加权结果。所有回归和raw样本
保留在[M16报告](M16_ATTENTION_RESOURCE_REPORT.md)。

CPU gather/H2D、KV store、RoPE/router、真实模型、并发serving和多GPU不在
本轮时间内。NanoKV引擎默认入口没有切换；其系统集成需要另测状态和完整请求。
更改冻结源码需要另立实验，不能覆盖已有policy/training/holdout。
