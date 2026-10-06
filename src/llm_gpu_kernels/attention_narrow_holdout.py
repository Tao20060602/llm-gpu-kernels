"""M12 fixed short-query policy and out-of-domain fallback comparison."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import torch
from .attention_baseline import prepare_segmented_attention
from .attention_baseline_benchmark import _graph,_make_inputs,_flat_tensors,_tensor_hash,_telemetry,_runtime_info,_functional_validation
from .attention_segmented import prepare_segmented_kernel,AttentionConfig
from .attention_segmented_benchmark import _functional,_check,_chunked_fp32_reference,_resources
from .attention_split import prepare_split_kernel
from .attention_narrow_policy import FrozenNarrowPolicy
from .attention_unified_tuning import measure,failure_fields
from .benchmark import compile_artifacts

ROOT=Path(__file__).resolve().parents[2]


def performance(tokens,selected,policy,run_id):
    torch.cuda.reset_peak_memory_stats()
    before=_telemetry();q,segments=_make_inputs(tokens,selected,seed=6207)
    inputs_hash=_tensor_hash(*_flat_tensors(q,segments));oracle=_chunked_fp32_reference(q,segments)
    mode,config=policy.select(tokens)
    chosen=policy.prepare(q,segments);old=prepare_segmented_kernel(q,segments)
    fa2=prepare_segmented_attention(q,segments)
    functions={'frozen_selected':chosen.run,'m9_default':old.run,'pack_plus_fa2':fa2.run}
    initial={n:_check(f(),oracle,n+' initial') for n,f in functions.items()}
    captures={n:_graph(f,q.device) for n,f in functions.items()}
    graph_initial={n:_check(out,oracle,n+' initial Graph') for n,(_,out) in captures.items()}
    api=measure(functions,divisor=1)
    graph=measure({n:g.replay for n,(g,_) in captures.items()})
    post={n:_check(f(),oracle,n+' post API') for n,f in functions.items()}
    graph_post={}
    for n,(g,out) in captures.items():g.replay();graph_post[n]=_check(out,oracle,n+' post Graph')
    if inputs_hash!=_tensor_hash(*_flat_tensors(q,segments)):raise RuntimeError('holdout inputs modified')
    evidence={}
    resources=None
    if mode=='split':
        for name,kernel in (('partial',chosen.kernel),('merge',chosen.merge_kernel)):
            evidence[name]=compile_artifacts(kernel,f'{name}-T{tokens}-S{selected}',run_id)
            evidence[name]['spills']=kernel.n_spills
        resources={'partial':_resources(chosen),'merge':{'registers':chosen.merge_kernel.n_regs,'spills':chosen.merge_kernel.n_spills,'shared_memory_bytes':chosen.merge_kernel.metadata.shared}}
    return {'status':'passed','passed':True,'T':tokens,'selected_history':selected,'seed':6207,
            'selected_mode':mode,'selected_config':asdict(config) if config else None,'input_sha256':inputs_hash,
            'initial_vs_fp32':initial,'initial_graph_vs_fp32':graph_initial,'post_api_vs_fp32':post,
            'post_graph_vs_fp32':graph_post,'api':api,'graph':graph,'selected_resources':resources,
            'split_scratch_bytes':chosen.scratch_bytes if mode=='split' else 0,
            'selected_packed_kv_bytes':chosen.packed_workspace_bytes if mode=='fa2' else 0,
            'baseline_packed_kv_bytes':fa2.packed_workspace_bytes,'owned_output_bytes':q.numel()*q.element_size(),
            'compile_evidence':evidence,'process_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_scope':'inputs, all arms, oracle, graph pools, runtime; not single operator budget',
            'telemetry_before':before,'telemetry_after':_telemetry()}


def run():
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    gpu=torch.cuda.get_device_properties(0)
    if (gpu.major,gpu.minor)!=(8,6):raise RuntimeError('M12 SM86 only')
    policy=FrozenNarrowPolicy.load(ROOT/'results/m12-narrow-policy.json');provenance=policy.verify_provenance(ROOT)
    run_id='m12-narrow-holdout-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+str(time.time_ns()%10**9)
    path=ROOT/'results'/(run_id+'.json')
    record={'schema':1,'run_id':run_id,'status':'running','policy':policy.data,'policy_sha256':policy.sha256,
            'source_hashes':policy.data['source_hashes'],'provenance_before':provenance,'seed':6207,
            'runtime':{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':gpu.name,'fa2':_runtime_info()},
            'limits':{'relative_rms':.008,'normalized_max':.008},
            'protocol':'three-arm 4 balanced forward/reverse rounds,8 samples,2 calls;Graph8 forwards; per-forward units',
            'scope':'GPU resident partial+merge complete operator vs pack+officialFA2; no model/RoPE/H2D/router/NanoKV/serving',
            'functional':[],'cases':[],'telemetry_before':_telemetry()}
    def save():path.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        mode,config=policy.select(64)
        if mode=='split':
            row={'config':asdict(config),'status':'running'};record['functional'].append(row);save()
            try:
                row.update(_functional(config,progress=row,prepare=prepare_split_kernel));row['status']='passed'
            except Exception as exc:
                row.update(failure_fields(exc))
                for group in ('cases','graph_updates'):
                    if row.get(group) and row[group][-1]['status']=='running':row[group][-1].update(failure_fields(exc))
                save();raise
        else:
            row={'mode':'fa2','status':'running'};record['functional'].append(row);save()
            row.update(_functional_validation());row['status']='passed'
            if not row['passed']:raise RuntimeError('FA2 fallback functional failed')
        save();gc.collect();torch.cuda.empty_cache()
        specs=[(t,s) for t in (17,33,49,63,64) for s in (0,512,2049)]+[(65,512)]
        for tokens,selected in specs:
            row={'T':tokens,'selected_history':selected,'status':'running'};record['cases'].append(row);save()
            print(f'M12 holdout T={tokens} selected={selected}',flush=True)
            try:row.update(performance(tokens,selected,policy,run_id))
            except Exception as exc:row.update(failure_fields(exc));save();raise
            save();gc.collect();torch.cuda.empty_cache()
        record.update(status='passed',provenance_after=policy.verify_provenance(ROOT))
    except Exception as exc:record.update(failure_fields(exc));raise
    finally:
        record['telemetry_after']=_telemetry();save();print('M12 holdout result:',path.relative_to(ROOT),flush=True)


if __name__=='__main__':run()
