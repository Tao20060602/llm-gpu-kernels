"""M11 eight candidates against the strong full pack+FA2 operator baseline."""
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import torch

from .attention_baseline import prepare_segmented_attention
from .attention_baseline_benchmark import _graph,_make_inputs,_make_custom_inputs,_flat_tensors,_tensor_hash,_telemetry,_runtime_info
from .attention_segmented_benchmark import _check,_chunked_fp32_reference,_resources
from .attention_split import prepare_split_kernel
from .attention_split_policy import CANDIDATES,FrozenSplitPolicy
from .attention_unified_policy import canonical_training_hash
from .attention_unified_tuning import measure,failure_fields

ROOT=Path(__file__).resolve().parents[2]
SOURCE_FILES=(
 'docs/M11_SPLIT_KV_PLAN.md','src/llm_gpu_kernels/attention_split.py',
 'src/llm_gpu_kernels/attention_split_policy.py','src/llm_gpu_kernels/attention_split_tuning.py',
 'src/llm_gpu_kernels/attention_split_holdout.py','src/llm_gpu_kernels/attention_segmented.py',
 'src/llm_gpu_kernels/attention_segmented_benchmark.py','src/llm_gpu_kernels/attention_baseline.py',
 'src/llm_gpu_kernels/attention_baseline_benchmark.py','src/llm_gpu_kernels/attention_unified.py',
 'src/llm_gpu_kernels/attention_unified_policy.py','src/llm_gpu_kernels/attention_unified_tuning.py',
 'src/llm_gpu_kernels/moe_prepared.py','src/llm_gpu_kernels/benchmark.py')


def trial(q,segments,oracle,baseline,config,long=False):
    before=_tensor_hash(*_flat_tensors(q,segments))
    plan=prepare_split_kernel(q,segments,config)
    functions={'fa2_baseline':baseline.run,'split_candidate':plan.run}
    initial={n:_check(f(),oracle,n+' initial') for n,f in functions.items()}
    captures={n:_graph(f,q.device) for n,f in functions.items()}
    graph_initial={n:_check(out,oracle,n+' initial Graph') for n,(_,out) in captures.items()}
    graph=measure({n:g.replay for n,(g,_) in captures.items()},samples=16 if long else 8,calls=4 if long else 2)
    post={n:_check(f(),oracle,n+' post API') for n,f in functions.items()}
    graph_post={}
    for n,(g,out) in captures.items():g.replay();graph_post[n]=_check(out,oracle,n+' post Graph')
    if before!=_tensor_hash(*_flat_tensors(q,segments)):raise RuntimeError('training inputs modified')
    ratio=graph['summary']['split_candidate']['event_p50_ms_per_forward']/graph['summary']['fa2_baseline']['event_p50_ms_per_forward']
    return {'status':'passed','passed':True,'input_sha256':before,'config':asdict(config),
            'initial_vs_fp32':initial,'initial_graph_vs_fp32':graph_initial,'post_api_vs_fp32':post,
            'post_graph_vs_fp32':graph_post,'graph':graph,'ratio_over_fa2':ratio,'partial_resources':_resources(plan),
            'merge_resources':{'registers':plan.merge_kernel.n_regs,'spills':plan.merge_kernel.n_spills,'shared_memory_bytes':plan.merge_kernel.metadata.shared},
            'scratch_bytes':plan.scratch_bytes,'owned_output_and_scratch_bytes':plan.workspace_bytes}


def run():
    policy_path=ROOT/'results/m11-split-policy.json'
    if policy_path.exists():raise RuntimeError('frozen M11 policy exists; cannot overwrite')
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    gpu=torch.cuda.get_device_properties(0)
    if (gpu.major,gpu.minor)!=(8,6):raise RuntimeError('M11 SM86 only')
    run_id='m11-split-tuning-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+str(time.time_ns()%10**9)
    path=ROOT/'results'/(run_id+'.json')
    record={'schema':1,'run_id':run_id,'status':'running','seed':3109,'training_T':[48,112,224],
            'selected_lengths':[383,1793],'source_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCE_FILES},
            'runtime':{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':gpu.name,'fa2':_runtime_info()},
            'limits':{'relative_rms':.008,'normalized_max':.008},'candidates':[asdict(c) for c in CANDIDATES],
            'boundary':[],'cases':[],'rechecks':[],'telemetry_before':_telemetry()}
    def save():path.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        q,segments=_make_custom_inputs(65,(33,17,41,65),seed=3109);oracle=_chunked_fp32_reference(q,segments)
        for i,config in enumerate(CANDIDATES):
            row={'candidate_index':i,'config':asdict(config),'status':'running'};record['boundary'].append(row);save()
            try:
                plan=prepare_split_kernel(q,segments,config)
                row['vs_fp32']=_check(plan.run(),oracle,'split boundary')
                eq,es=_make_custom_inputs(0,(3,4,5,0),seed=3109);empty=prepare_split_kernel(eq,es,config)
                if empty.kernel is not None or empty.merge_kernel is not None or empty.run().shape!=eq.shape:raise RuntimeError('empty query launched')
                row.update(status='passed',passed=True,empty_no_launch=True,partial_resources=_resources(plan),scratch_bytes=plan.scratch_bytes)
                del plan,empty
            except Exception as exc:row.update(failure_fields(exc))
            save()
        del q,segments,oracle;gc.collect();torch.cuda.empty_cache()
        for tokens in (48,112,224):
            for selected in (383,1793):
                case={'T':tokens,'selected_history':selected,'status':'running','candidates':[],'telemetry_before':_telemetry()}
                record['cases'].append(case);save();print(f'M11 training T={tokens} selected={selected}',flush=True)
                q,segments=_make_inputs(tokens,selected,seed=3109);oracle=_chunked_fp32_reference(q,segments)
                baseline=prepare_segmented_attention(q,segments)
                case['baseline_vs_fp32']=_check(baseline.run(),oracle,'fa2 baseline')
                case['input_sha256']=_tensor_hash(*_flat_tensors(q,segments))
                for i,config in enumerate(CANDIDATES):
                    row={'candidate_index':i,'config':asdict(config),'status':'running'};case['candidates'].append(row);save()
                    if not record['boundary'][i].get('passed'):
                        row.update(status='excluded_boundary_failure',passed=False);save();continue
                    _check(baseline.run(),oracle,'fa2 baseline before candidate')
                    try:row.update(trial(q,segments,oracle,baseline,config))
                    except Exception as exc:
                        if str(exc).startswith('fa2'):raise
                        row.update(failure_fields(exc))
                    _check(baseline.run(),oracle,'fa2 baseline after candidate');save()
                case.update(status='completed',telemetry_after=_telemetry());save()
                del q,segments,oracle,baseline;gc.collect();torch.cuda.empty_cache()
        scores=[]
        for i,config in enumerate(CANDIDATES):
            rows=[c['candidates'][i] for c in record['cases']]
            if not all(r.get('passed') for r in rows):continue
            ratios=[r['ratio_over_fa2'] for r in rows];gm=math.exp(sum(math.log(x) for x in ratios)/len(ratios))
            scores.append({'candidate_index':i,'geomean_ratio':gm,'worst_ratio':max(ratios),'eligible':gm<=.95 and max(ratios)<=1.05})
        eligible=[s for s in scores if s['eligible']]
        selected_score=min(eligible,key=lambda x:(x['geomean_ratio'],x['candidate_index'])) if eligible else None
        record['scores']=scores;record['initial_selection']=selected_score;save()
        selected_config=CANDIDATES[selected_score['candidate_index']] if selected_score else None
        accepted=False
        if selected_config:
            ratios=[];failed=False
            for tokens in (48,112,224):
                for selected in (383,1793):
                    row={'T':tokens,'selected_history':selected,'status':'running'};record['rechecks'].append(row);save()
                    q,segments=_make_inputs(tokens,selected,seed=3109);oracle=_chunked_fp32_reference(q,segments)
                    baseline=prepare_segmented_attention(q,segments);_check(baseline.run(),oracle,'fa2 recheck baseline')
                    try:
                        row.update(trial(q,segments,oracle,baseline,selected_config,long=True))
                        expected=next(c['input_sha256'] for c in record['cases'] if c['T']==tokens and c['selected_history']==selected)
                        if row['input_sha256']!=expected:raise RuntimeError('recheck input differs')
                        ratios.append(row['ratio_over_fa2'])
                    except Exception as exc:
                        if str(exc).startswith('fa2'):raise
                        row.update(failure_fields(exc));failed=True
                    _check(baseline.run(),oracle,'fa2 after recheck');save()
                    del q,segments,oracle,baseline;gc.collect();torch.cuda.empty_cache()
            gm=math.exp(sum(math.log(x) for x in ratios)/len(ratios)) if ratios else None
            accepted=not failed and len(ratios)==6 and gm<=.95 and max(ratios)<=1.05
            record['recheck_summary']={'accepted':accepted,'geomean_ratio':gm,'worst_ratio':max(ratios) if ratios else None}
        record.update(status='passed',frozen_mode='split' if accepted else 'fa2',
                      frozen_config=asdict(selected_config) if accepted else None,telemetry_after=_telemetry());save()
        policy={'schema':1,'status':'frozen_before_holdout','max_T':256,'mode':record['frozen_mode'],
                'config':record['frozen_config'],'source_hashes':record['source_hashes'],
                'training_record_path':str(path.relative_to(ROOT)),'training_record_sha256':canonical_training_hash(record),
                'training_hash_scope':'canonical_json_without_top_level_policy_sha256'}
        policy_path.write_text(json.dumps(policy,indent=2)+'\n');record['policy_sha256']=hashlib.sha256(policy_path.read_bytes()).hexdigest();save()
        FrozenSplitPolicy.load(policy_path).verify_provenance(ROOT)
        print('M11 frozen:',record['frozen_mode'],record['frozen_config'],flush=True)
    except Exception as exc:
        for group in ('boundary','cases','rechecks'):
            if record.get(group) and record[group][-1]['status']=='running':record[group][-1].update(failure_fields(exc))
        record.update(failure_fields(exc),telemetry_after=_telemetry());save();raise
    print('M11 training result:',path.relative_to(ROOT),flush=True)


if __name__=='__main__':run()
