# M18 动态入口使用契约

从独立算子仓库根目录使用项目venv和官方FA2 overlay：

```bash
source scripts/activate-flash-attn.sh
python -m llm_gpu_kernels.attention_dynamic_benchmark --manifest /absolute/path/to/m17/manifest.json
python -m llm_gpu_kernels.attention_dynamic_boundary_audit
```

接口示例（q/segments为已就绪的CUDA BF16张量）：

```python
from pathlib import Path
from llm_gpu_kernels.attention_resource_policy import FrozenResourcePolicy
from llm_gpu_kernels.attention_dynamic import AttentionCompileCache, DynamicAttentionWorkspace

root = Path('/absolute/path/to/llm-gpu-kernels')
policy = FrozenResourcePolicy.load(root / 'results/m16-resource-policy.json')
policy.verify_provenance(root)
cache = AttentionCompileCache(max_entries=16)
workspace = DynamicAttentionWorkspace(q, segments, policy, cache)
output = workspace.run(q, segments)
# 元数据/创建stream一致的新输入地址：
new_output = workspace.run(new_q, new_segments)
```

- Q[T,32,128]，四段K/V[L,8,128]，current L=T；顺序selected/sink/recent/current。
- CUDA BF16、同设备，head stride128/dim stride1；token stride至少一个完整token。
  基址16字节对齐，接受padding，包括模型current V stride6144。值须finite，遵守M15
  有限FP32 score实验契约。无梯度、dropout或RoPE。
- workspace固定shape、stride、device和创建stream；更改这些元数据需要独立workspace。
  cache可以跨workspace复用编译模板；它不保留输入地址，不记录选择或实际张量值。
- caller显式建立跨stream producer依赖。run会给输入record_stream，以保护释放后allocator
  生命周期；不替代wait_stream/event。workspace不可在线程或stream间并发写。
- direct输出复用并在下一次run覆盖；需保留结果时先clone或完成消费。
  FA2 fallback输出由官方API分配。目前fallback还持有未使用预留output，见报告内存限制。
- T0直接返回空output；domain外和medium/large回退FA2。不会按新性能结果改冻策略。
- Graph捕获使用在capture stream创建的独立workspace。必须持有图、输入和workspace，
  值可原地更新；新地址需重新捕获图。run换新指针只影响eager调用。
- LRU/clear只限制本模块持有的模板，不清除Triton内部缓存或销毁其他活跃workspace的模板。
  cache由caller串行访问。
- 可通过cache.get(cuda_device,T,lengths,config)提前只编译；shape长度必须事先已知。
  编译和分配不计入本报告的热forward数字，真实请求必须另外报告启动成本。

SM86单GPU实验入口，未作为NanoKV默认后端或完整serving部署。
