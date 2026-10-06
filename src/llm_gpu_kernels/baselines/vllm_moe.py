"""Unmodified vLLM 0.10.0 public BF16 fused_experts operator baseline.

Call the installed package, including its native alignment/activation/reduction.
No copied kernels, forced configuration, or replacement dispatch implementation.
"""
from dataclasses import dataclass
from functools import cache
from pathlib import Path
import torch
from ..gemm import _input


@cache
def _backend():
    import vllm
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    if vllm.__version__ != "0.10.0" or not torch.__version__.startswith("2.7.1"):
        raise RuntimeError("baseline requires vLLM0.10.0 and PyTorch2.7.1")
    return fused_experts


@dataclass
class VLLMMoE:
    x: torch.Tensor
    gate_up: torch.Tensor
    down: torch.Tensor
    ids: torch.Tensor
    routing_weights: torch.Tensor
    output: torch.Tensor | None = None

    def run(self):
        self.output = _backend()(self.x, self.gate_up, self.down, self.routing_weights, self.ids,
                                 inplace=False, activation="silu", apply_router_weight_on_input=False)
        return self.output

    def configuration(self):
        import importlib
        import vllm.envs as envs
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            get_config_file_name, get_config_dtype_str, try_get_optimal_moe_config)
        module = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
        dtype = get_config_dtype_str(self.x.dtype)
        filename = get_config_file_name(self.down.shape[0], self.down.shape[2], dtype)
        path = Path(module.__file__).parent / "configs" / filename
        config = try_get_optimal_moe_config(tuple(self.gate_up.shape), tuple(self.down.shape),
                                           self.ids.shape[1], dtype, min(self.x.shape[0], envs.VLLM_FUSED_MOE_CHUNK_SIZE))
        return {"selected": config, "config_filename": filename, "packaged_config_found": path.exists(),
                "chunk_size": envs.VLLM_FUSED_MOE_CHUNK_SIZE, "override": None,
                "route_weight_rounding": "FP32 down accumulator * route weight, then BF16 store, then sum routes",
                "activation_rounding": "SiLU branch BF16 round before multiplication by up",
                "sum_rounding": "topk2/3/4: native BF16 sequential accumulation; other topk: ATen sum_out",
                "allocation": "upstream public API allocates caches, aligned IDs and output per call; Graph uses captured pool"}


def prepare_vllm_moe(x, gate_up, down, ids, routing_weights):
    for t, rank in ((x, 2), (gate_up, 3), (down, 3)):
        _input(t, rank)
    e, h, intermediate = down.shape
    if x.shape[1] != h or gate_up.shape != (e, 2 * intermediate, h) or min(e, h, intermediate) <= 0:
        raise ValueError("incompatible MoE shapes")
    if (ids.ndim != 2 or ids.shape[0] != x.shape[0] or ids.shape[1] == 0
            or ids.dtype not in (torch.int32, torch.int64) or routing_weights.shape != ids.shape
            or routing_weights.dtype != torch.float32):
        raise ValueError("expected integer IDs[T,topk] and FP32 route weights")
    if any(t.device != x.device or not t.is_contiguous() for t in (gate_up, down, ids, routing_weights)):
        raise ValueError("contiguous same-device tensors required")
    _backend()
    wrapper = VLLMMoE(x, gate_up, down, ids, routing_weights)
    wrapper.run()  # Compile/warm outside timing.
    return wrapper
