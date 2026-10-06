"""Fixed-buffer real-layout bridge; explicit conversion costs belong to run."""
import torch
from flash_attn import flash_attn_func


class RawPackAttention:
    def __init__(self, q, segments, *, reusable):
        self.q = q
        self.segments = tuple(segments)
        self.reusable = reusable
        self.total = sum(k.shape[0] for k, _ in segments)
        shape = (self.total, 8, 128)
        self.packed_k = torch.empty(shape, dtype=q.dtype, device=q.device) if reusable else None
        self.packed_v = torch.empty(shape, dtype=q.dtype, device=q.device) if reusable else None

    @property
    def packed_workspace_bytes(self):
        return self.total*8*128*2*self.q.element_size() if self.reusable else 0

    def run(self):
        if self.reusable:
            k, v = self.packed_k, self.packed_v
            torch.cat(tuple(pair[0] for pair in self.segments), dim=0, out=k)
            torch.cat(tuple(pair[1] for pair in self.segments), dim=0, out=v)
        else:
            k = torch.empty((self.total, 8, 128), dtype=self.q.dtype, device=self.q.device)
            v = torch.empty_like(k)
            offset = 0
            for sk, sv in self.segments:
                length = sk.shape[0]
                k[offset:offset+length].copy_(sk)
                v[offset:offset+length].copy_(sv)
                offset += length
        return flash_attn_func(self.q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                               softmax_scale=128**-.5, causal=True).squeeze(0)


class FrozenRawBoundary:
    """Bound owners and addresses; shape changes require another instance.

    This laboratory bridge does not implement a general engine adapter.
    Caller owns cross-stream dependencies and forbids simultaneous writes.
    """
    def __init__(self, q, segments, policy):
        self.q = q
        self.segments = tuple(segments)
        self.mode, self.config, self.bucket = policy.select(
            q.shape[0], tuple(k.shape[0] for k, _ in segments))
        self.conversions = []
        if self.mode == 'fa2':
            self.plan = RawPackAttention(q, segments, reusable=True)
        else:
            def contiguous_target(source):
                if source.is_contiguous():
                    return source
                target = torch.empty(source.shape, dtype=source.dtype, device=source.device)
                target.copy_(source)
                self.conversions.append((source, target))
                return target
            cq = contiguous_target(q)
            cs = tuple((contiguous_target(k),contiguous_target(v)) for k,v in segments)
            self.plan = policy.prepare(cq, cs)

    @property
    def conversion_workspace_bytes(self):
        return sum(dst.numel()*dst.element_size() for _,dst in self.conversions)

    def run(self):
        for source, target in self.conversions:
            target.copy_(source)
        return self.plan.run()
