"""M18 additional padded-layout and invalid metadata audit; no policy selection."""
import json
import time
from pathlib import Path

import torch

from .attention_dynamic import AttentionCompileCache, DynamicAttentionWorkspace
from .attention_dynamic_benchmark import ROOT, SOURCES, make_inputs
from .attention_real_input_benchmark import file_hash
from .attention_resource_policy import FrozenResourcePolicy
from .attention_segmented_benchmark import _check, _chunked_fp32_reference
from .attention_baseline_benchmark import _flat_tensors, _tensor_hash
from .attention_unified_tuning import failure_fields


def run():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    policy = FrozenResourcePolicy.load(ROOT/'results/m16-resource-policy.json')
    cache = AttentionCompileCache()
    names = (*SOURCES,'src/llm_gpu_kernels/attention_dynamic_boundary_audit.py')
    hashes = {n:file_hash(ROOT/n) for n in names}
    path = ROOT/'results'/('m18-layout-audit-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json')
    record = dict(status='running',source_hashes=hashes,policy_sha256=policy.sha256,cases=[],rejections={})
    def save():
        path.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        for name in ('padded_q','padded_history_keys','all_padded'):
            q,segments=make_inputs(96,257,23001)
            flat=list(_flat_tensors(q,segments))
            for index,t in enumerate(flat):
                if name=='all_padded' or (name=='padded_q' and index==0) or (name=='padded_history_keys' and index in (1,3,5)):
                    target=torch.empty_strided(t.shape,(t.shape[1]*128*3,128,1),dtype=t.dtype,device=t.device)
                    target.copy_(t);flat[index]=target
            q=flat[0];segments=tuple(zip(flat[1::2],flat[2::2]))
            digest=_tensor_hash(*flat)
            workspace=DynamicAttentionWorkspace(q,segments,policy,cache)
            oracle=_chunked_fp32_reference(q,segments)
            checks={'initial':_check(workspace.run(q,segments),oracle,name)}
            # Replace all addresses, keeping each original stride exactly.
            new=[]
            for t in flat:
                target=torch.empty_strided(t.shape,t.stride(),dtype=t.dtype,device=t.device)
                target.copy_(t);new.append(target)
            checks['rebound']=_check(workspace.run(new[0],tuple(zip(new[1::2],new[2::2]))),oracle,name+' rebound')
            if digest != _tensor_hash(*flat):raise RuntimeError('input changed')
            record['cases'].append(dict(name=name,checks=checks,conversion_workspace_bytes=workspace.conversion_workspace_bytes,passed=True));save()
        q,segments=make_inputs(64,257,24001)
        workspace=DynamicAttentionWorkspace(q,segments,policy,cache)
        nq,ns=make_inputs(65,257,24002)
        offset_q=torch.empty(q.numel()+1,dtype=q.dtype,device=q.device)[1:].view(q.shape)
        bad_q=torch.empty_strided(q.shape,(32*256,256,1),dtype=q.dtype,device=q.device)
        overlap=segments[0][0].as_strided(segments[0][0].shape,(0,128,1))
        changed=(*segments[:-1],(segments[-1][0],segments[-1][1].contiguous()))
        invalid={
            'same_valid_contract_different_shape':(nq,ns),
            'misaligned_base':(offset_q,segments),
            'head_stride':(bad_q,segments),
            'overlap':(q,((overlap,segments[0][1]),*segments[1:])),
            'changed_valid_stride':(q,changed),
            'cpu_input':(q.cpu(),segments),
        }
        for name,(iq,isegments) in invalid.items():
            try:workspace.run(iq,isegments)
            except ValueError as exc:record['rejections'][name]=str(exc)
            else:raise RuntimeError('invalid accepted: '+name)
        after={n:file_hash(ROOT/n) for n in names}
        if after!=hashes:raise RuntimeError('sources changed')
        record.update(status='passed',source_hashes_after=after,cache_stats=cache.stats(),provenance=policy.verify_provenance(ROOT))
    except Exception as exc:
        record.update(failure_fields(exc));raise
    finally:
        save();print(path,flush=True)


if __name__=='__main__':run()
