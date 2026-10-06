"""M18 pointer-free compiled templates and stream-owned dynamic workspaces."""
from collections import OrderedDict
from dataclasses import dataclass

import torch
import triton
from flash_attn import flash_attn_func

from .attention_nonempty import _nonempty_attention_fwd, SOFTMAX_SCALE_LOG2


def _inputs(q, segments):
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or tuple(q.shape[1:]) != (32, 128):
        raise ValueError('expected Q[T,32,128]')
    segments = tuple(segments)
    if len(segments) != 4:
        raise ValueError('expected four K/V segments')
    flat = [q]
    for pair in segments:
        if len(pair) != 2:
            raise ValueError('expected K/V pair')
        k, v = pair
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise ValueError('expected K/V tensors')
        if k.ndim != 3 or tuple(k.shape[1:]) != (8, 128) or k.shape != v.shape:
            raise ValueError('expected matching K/V[L,8,128]')
        flat.extend(pair)
    if segments[-1][0].shape[0] != q.shape[0]:
        raise ValueError('current length must equal T')
    for t in flat:
        if not t.is_cuda or t.dtype != torch.bfloat16 or t.device != q.device:
            raise ValueError('expected CUDA BF16 on the same device')
        # M18 accepts contiguous or padded token stride, including QKV views.
        # Reject overlapping/arbitrary layouts rather than silently interpret them.
        if t.stride()[1:] != (128, 1) or t.stride(0) < t.shape[1]*128:
            raise ValueError('expected contiguous heads/dim and nonoverlapping token stride')
        if t.data_ptr() % 16:
            raise ValueError('expected 16-byte aligned input base')
    return segments, tuple(flat)


def _metadata(flat):
    return tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in flat)


@dataclass(frozen=True)
class CompiledTemplate:
    kernel: object
    runner: object
    constants: tuple
    grid: tuple


class AttentionCompileCache:
    """Bounded local LRU; Triton's own JIT/disk caches remain separate.

    Templates have no tensor owners/addresses. Caller serializes cache access.
    """
    def __init__(self, max_entries=16):
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError('max_entries must be a positive integer')
        if triton.__version__ != '3.3.1':
            raise RuntimeError('dynamic launcher requires Triton3.3.1')
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def get(self, device, tokens, lengths, config):
        device = torch.device(device)
        if device.type != 'cuda' or device.index is None:
            raise ValueError('expected explicit CUDA device index')
        if tokens <= 0 or len(lengths) != 4 or lengths[-1] != tokens or min(lengths) < 0:
            raise ValueError('expected positive T and four nonnegative lengths')
        if torch.cuda.get_device_capability(device) != (8, 6):
            raise RuntimeError('M18 validated only on SM86')
        # Only frozen M16 candidates may be requested by the workspace.
        from .attention_resource_policy import CANDIDATES
        if config not in CANDIDATES:
            raise ValueError('expected frozen M16 candidate')
        key = (device.index, (8, 6), tokens, tuple(lengths), config)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        constants = (tokens, *lengths, config.head_group, config.token_tile,
                     config.effective_block_m, config.key_tile, 32, 8, 128,
                     SOFTMAX_SCALE_LOG2)
        grid = (triton.cdiv(tokens, config.token_tile), 32//config.head_group, 1)
        with torch.cuda.device(device):
            # Dtype mock pointers: compile and initialize handles, NO attention launch.
            kernel = _nonempty_attention_fwd.warmup(
                *([torch.bfloat16]*10), *constants, grid=grid,
                num_warps=config.warps, num_stages=config.stages)
            if len(kernel.src.signature) != 10+len(constants):
                raise RuntimeError('unexpected Triton launcher signature')
            template = CompiledTemplate(kernel, kernel[grid], constants, grid)
        self.entries[key] = template
        if len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
            self.evictions += 1
        return template

    def clear(self):
        self.entries.clear()

    def stats(self):
        return dict(entries=len(self.entries), max_entries=self.max_entries,
                    hits=self.hits, misses=self.misses, evictions=self.evictions)


class DynamicAttentionWorkspace:
    """Fixed metadata/stream, replaceable addresses; inference forward only.

    Caller establishes producer dependencies. Inputs are record_stream'ed for
    allocator lifetime; output is overwritten next run. A captured Graph still
    needs its original input owners and addresses. No concurrent workspace use.
    """
    def __init__(self, q, segments, policy, cache):
        segments, flat = _inputs(q, segments)
        self.metadata = _metadata(flat)
        self.device = q.device
        self.stream = torch.cuda.current_stream(self.device)
        self.mode, self.config, self.bucket = policy.select(q.shape[0], tuple(k.shape[0] for k, _ in segments))
        self.output = torch.empty(q.shape, dtype=q.dtype, device=self.device)
        self.targets = tuple(None if t.is_contiguous() else torch.empty(t.shape, dtype=t.dtype, device=t.device)
                             for t in flat) if self.mode == 'direct' else ()
        self.template = cache.get(self.device, q.shape[0], tuple(k.shape[0] for k, _ in segments),
                                  self.config) if self.mode == 'direct' else None
        self.packed_k = self.packed_v = None
        if self.mode == 'fa2' and q.shape[0]:
            shape = (sum(k.shape[0] for k, _ in segments), 8, 128)
            self.packed_k = torch.empty(shape, dtype=q.dtype, device=self.device)
            self.packed_v = torch.empty_like(self.packed_k)
        for t in (self.output, *self.targets, self.packed_k, self.packed_v):
            if t is not None:
                t.record_stream(self.stream)

    @property
    def conversion_workspace_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.targets if t is not None)

    def run(self, q, segments):
        segments, flat = _inputs(q, segments)
        if _metadata(flat) != self.metadata:
            raise ValueError('workspace shape/stride/device changed; create another workspace')
        stream = torch.cuda.current_stream(self.device)
        if stream.cuda_stream != self.stream.cuda_stream:
            raise ValueError('workspace belongs to another stream; create an independent workspace')
        for t in flat:
            t.record_stream(stream)
        if q.shape[0] == 0:
            return self.output
        if self.mode == 'fa2':
            torch.cat(tuple(k for k, _ in segments), dim=0, out=self.packed_k)
            torch.cat(tuple(v for _, v in segments), dim=0, out=self.packed_v)
            return flash_attn_func(q.unsqueeze(0), self.packed_k.unsqueeze(0), self.packed_v.unsqueeze(0),
                                   softmax_scale=128**-.5, causal=True).squeeze(0)
        pointers = []
        for source, target in zip(flat, self.targets):
            if target is not None:
                target.copy_(source)
                source = target
            pointers.append(source.data_ptr())
        self.template.runner(*pointers, self.output.data_ptr(), *self.template.constants,
                             stream=stream.cuda_stream)
        return self.output
