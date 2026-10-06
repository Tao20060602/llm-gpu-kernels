"""Frozen M10 three-arm attention comparison, independent seed and full oracle."""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch

from .attention_baseline import prepare_segmented_attention
from .attention_baseline_benchmark import _graph, _make_inputs, _flat_tensors, _tensor_hash, _telemetry, _runtime_info
from .attention_segmented import prepare_segmented_kernel
from .attention_segmented_benchmark import _check, _chunked_fp32_reference, _resources, _functional
from .attention_unified import prepare_unified_kernel
from .attention_unified_policy import FrozenAttentionPolicy, DEFAULT
from .attention_unified_tuning import measure, prepare_choice, failure_fields
from .benchmark import compile_artifacts

ROOT=Path(__file__).resolve().parents[2]


def performance(tokens,selected,policy,run_id):
    torch.cuda.reset_peak_memory_stats()
    before=_telemetry()
    q,segments=_make_inputs(tokens,selected,seed=2917)
    inputs_hash=_tensor_hash(*_flat_tensors(q,segments))
    oracle=_chunked_fp32_reference(q,segments)
    mode,config,bucket=policy.select(tokens)
    old=prepare_segmented_kernel(q,segments,DEFAULT)
    chosen=prepare_choice(mode,q,segments,config)
    fa2=prepare_segmented_attention(q,segments)
    functions={'m9_default':old.run,'frozen_selected':chosen.run,'pack_plus_fa2':fa2.run}
    initial={name:_check(fn(),oracle,name+' initial') for name,fn in functions.items()}
    captures={name:_graph(fn,q.device) for name,fn in functions.items()}
    graph_initial={name:_check(output,oracle,name+' graph initial') for name,(_,output) in captures.items()}
    api=measure(functions,divisor=1)
    graph=measure({name:g.replay for name,(g,_) in captures.items()})
    post_api={name:_check(fn(),oracle,name+' post API') for name,fn in functions.items()}
    post_graph={}
    for name,(g,output) in captures.items():
        g.replay();post_graph[name]=_check(output,oracle,name+' post Graph')
    if _tensor_hash(*_flat_tensors(q,segments))!=inputs_hash:raise RuntimeError('holdout inputs modified')
    evidence=compile_artifacts(chosen.kernel,f'selected-T{tokens}-S{selected}',run_id)
    evidence['spills']=chosen.kernel.n_spills
    return {'status':'passed','passed':True,'T':tokens,'selected_history':selected,
            'segment_lengths':[k.shape[0] for k,_ in segments], 'seed':2917,
            'bucket':bucket,'selected_mode':mode,'selected_config':asdict(config),
            'input_sha256':inputs_hash,'input_hash_preserved':True,
            'initial_vs_fp32':initial,'initial_graph_vs_fp32':graph_initial,
            'post_api_vs_fp32':post_api,'post_graph_vs_fp32':post_graph,
            'api':api,'graph':graph,'default_resources':_resources(old),'selected_resources':_resources(chosen),
            'selected_owned_output_bytes':chosen.workspace_bytes,'selected_scratch_bytes':0,
            'baseline_packed_kv_bytes':fa2.packed_workspace_bytes,'compile_evidence':evidence,
            'process_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_scope':'inputs, all arms, graph pools, FP32 oracle and runtime; not kernel-only workspace',
            'telemetry_before':before,'telemetry_after':_telemetry()}


def run():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    gpu=torch.cuda.get_device_properties(0)
    if (gpu.major,gpu.minor)!=(8,6):raise RuntimeError('M10 scoped to SM86')
    policy=FrozenAttentionPolicy.load(ROOT/'results/m10-attention-policy.json')
    before_provenance=policy.verify_provenance(ROOT)
    run_id='m10-attention-holdout-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+str(time.time_ns()%10**9)
    path=ROOT/'results'/(run_id+'.json')
    record={'schema':1,'run_id':run_id,'status':'running','policy_sha256':policy.sha256,'policy':policy.data,
            'provenance_before':before_provenance,'seed':2917,
            'runtime':{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':gpu.name,'fa2':_runtime_info()},
            'source_hashes':policy.data['source_hashes'],
            'protocol':'4 balanced forward/reverse rounds,8 samples,2 calls,Graph8 forwards; per-forward units; no holdout tuning',
            'oracle':'independent dense FP32 over BF16 operands,TF32 off,query chunks64; all arm API/Graph pre/post checks',
            'limits':{'relative_rms':.008,'normalized_max':.008},
            'scope':'GPU resident four-segment forward vs public pack+FA2; no model,RoPE,H2D,router,NanoKV,serving',
            'functional':[],'cases':[],'telemetry_before':_telemetry()}
    def save():path.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        seen=set()
        for tokens in (64,256,1024):
            mode,config,bucket=policy.select(tokens)
            key=(mode,tuple(asdict(config).values()))
            if key in seen:continue
            seen.add(key)
            row={'mode':mode,'config':asdict(config),'bucket':bucket,'status':'running'}
            record['functional'].append(row);save()
            print('M10 functional',mode,asdict(config),flush=True)
            try:
                prepare=prepare_unified_kernel if mode=='unified' else prepare_segmented_kernel
                row.update(_functional(config,progress=row,prepare=prepare));row['status']='passed'
            except Exception as exc:
                row.update(failure_fields(exc))
                for group in ('cases','graph_updates'):
                    if row.get(group) and row[group][-1]['status']=='running':row[group][-1].update(failure_fields(exc))
                save();raise
            save();gc.collect();torch.cuda.empty_cache()
        for tokens in (64,256,1024,4096):
            for selected in (0,512,2048):
                row={'T':tokens,'selected_history':selected,'status':'running'}
                record['cases'].append(row);save()
                print(f'M10 holdout T={tokens} selected={selected}',flush=True)
                try:row.update(performance(tokens,selected,policy,run_id))
                except Exception as exc:
                    row.update(failure_fields(exc));save();raise
                save();gc.collect();torch.cuda.empty_cache()
        record.update(status='passed',provenance_after=policy.verify_provenance(ROOT))
    except Exception as exc:
        record.update(failure_fields(exc));raise
    finally:
        record['telemetry_after']=_telemetry();save()
        print('M10 holdout result:',path.relative_to(ROOT),flush=True)


if __name__=='__main__':run()
