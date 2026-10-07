# M20：真实 Qwen3 MoE layer-0 trace 契约

本契约只定义固定路由 MoE 的真实输入/权重 artifact 保存与校验，不实现 router、
算子 dispatch 或 benchmark。模型来源固定为 `Qwen/Qwen3-30B-A3B` revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`，参数形状为 E=128、H=2048、I=768、
top-k=8。它不改变 M7–M18 的冻结文件或既有结果。

## 张量语义

- `x[T,2048]`：捕获的 layer-0 MoE 输入，BF16、有限值、T>0。
- `expert_ids[T,8]`：官方 layer-0 router 的 top-k expert ID，int32 或 int64，所有
  ID 在 `[0,128)`；同一 token 的 8 个 ID 必须互不重复。
- `route_weights[T,8]`：官方 router 输出的 BF16 权重精确晋升为 FP32。验证会确认每个
  FP32 值仍可精确表示为 BF16、有限且非负。原权重直接保留，不做 top-k 重归一化；
  不要求行和为 1，也不在本契约里猜测额外路由缩放上限。
- `gate_up[128,1536,2048]`：BF16，拼接顺序为 gate 在前、up 在后。
- `down[128,2048,768]`：BF16。

固定路由的数学语义沿用 [项目 MoE 参考契约](CONTRACT.md)：对每个 route 分别计算
`weight * down(silu(gate(x)) * up(x))` 并按 slot 累加；不在这里运行 router 或再归一化。

## Artifact 布局与安全边界

推荐每次捕获使用新目录：

```text
artifacts/m20_real_moe/<run_id>/
  manifest.json
  trace.pt       # 只含 x、expert_ids、route_weights
  weights.pt     # 共享 bundle，只含 gate_up、down
```

capture 脚本先写权重 bundle：

```python
torch.save({"gate_up": gate_up_cpu, "down": down_cpu}, weights_path)
```

trace 和权重 bundle 都是本地 tensor-only `.pt` 文件。模块拒绝 URL/远程路径以及其他
pickle 格式；所有反序列化统一使用 `torch.load(..., weights_only=True,
map_location="cpu")`。不要从用户提供的任意 checkpoint `.bin` 或不可信 pickle 加载。
`weights.pt` 必须来自已审计的本地 layer-0 提取步骤。`source_manifest_sha256` 记录该
提取 manifest 的 SHA256；gate_up/down 两条记录须引用同一个提取 manifest。

每个文件记录 SHA256。gate_up 与 down 可引用同一份共享 `weights.pt`，manifest 会保留
tensor key、shape、dtype、来源提取 manifest hash，而不会再次复制大权重。捕获时已有的
weight bundle SHA256 可直接传入 writer；benchmark 开始前用
`validate_real_moe_trace(..., verify_hashes=True)` 复核文件内容。共享路径在一次校验里
只散列一次。文件散列、安全反序列化和有限值检查均须发生在计时区外。

Manifest 固定写入模型 repo/revision、形状契约、routing 变换声明、trace/权重路径和
文件 hash，以及 trace 每个张量的 shape/dtype/content hash。`capture` 是调用者提供的
JSON-safe provenance；适合记录 layer、文本/token 来源标识、提取步骤与 source hash，
不要放入原始用户文本或访问凭据。writer 返回的 dict 就是写出的 manifest 内容；需要
manifest 文件本身的 hash 时，在写完后对 `manifest.json` 调用 `sha256_file`。

## Python API

```python
from llm_gpu_kernels.moe_real_trace import (
    sha256_file,
    write_real_moe_trace,
    validate_real_moe_trace,
)

weights_sha256 = sha256_file(weights_path)
weight_refs = {
    "gate_up": {
        "path": str(weights_path),
        "sha256": weights_sha256,
        "tensor_name": "gate_up",
        "shape": [128, 1536, 2048],
        "dtype": "torch.bfloat16",
        "source_manifest_sha256": extraction_manifest_sha256,
    },
    "down": {
        "path": str(weights_path),
        "sha256": weights_sha256,
        "tensor_name": "down",
        "shape": [128, 2048, 768],
        "dtype": "torch.bfloat16",
        "source_manifest_sha256": extraction_manifest_sha256,
    },
}
manifest = write_real_moe_trace(
    manifest_path, trace_path,
    x=x_cpu, expert_ids=expert_ids_cpu, route_weights=route_weights_cpu,
    provenance={"layer": 0, "capture_kind": "real_model_layer_input"},
    weights=weight_refs,
)

trace = validate_real_moe_trace(
    manifest_path, verify_hashes=True, load_trace=True, load_weights=False,
)
x = trace.tensors["x"]
weight_bundle_path = trace.manifest["weights"]["gate_up"]["path"]
weight_bundle_sha256 = trace.manifest["weights"]["gate_up"]["sha256"]
```

The writer requires CPU input tensors and writes a tensor-only trace dict with exactly
`x`, `expert_ids`, and `route_weights`. Relative artifact paths resolve from the manifest
directory; absolute local paths are also accepted. It returns the in-memory manifest dict.
`validate_real_moe_trace` returns a `RealMoETrace` with `.manifest_path`, `.manifest`,
`.tensors`, and `.weights`. With `load_weights=False`, `.weights` is `None`; this mode still
validates manifest fields and, when requested, artifact hashes, but intentionally does not
load the large weight bundle. Use `load_weights=True` for one-time CPU bundle materialization
and actual gate_up/down shape, dtype, and finite-value checks. A benchmark that loads weights
itself may instead do its own one-time checks and retain `.weights=None` during per-case trace
validation.

This artifact supports later fixed-route operator validation. It is not a router benchmark,
whole-model quality test, serving measurement, or evidence of end-to-end speedup.
