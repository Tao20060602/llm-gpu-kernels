"""M14 six candidates against the strong full pack+FA2 operator baseline."""
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
from .attention_segmented import prepare_segmented_kernel
from .attention_retile_policy import CANDIDATES,FrozenRetilePolicy,BUCKETS,DOMAIN,bucket_for
from .attention_unified_policy import canonical_training_hash
from .attention_unified_tuning import measure,failure_fields
from .attention_fixed_trial import trial

ROOT=Path(__file__).resolve().parents[2]
from .attention_fixed_tuning import SOURCE_FILES as OLD_SOURCES
SOURCE_FILES=(*OLD_SOURCES,'docs/M14_ATTENTION_RETILE_PLAN.md',
 'src/llm_gpu_kernels/attention_retile_policy.py','src/llm_gpu_kernels/attention_retile_tuning.py',
 'src/llm_gpu_kernels/attention_retile_holdout.py')




def run():
    policy_path=ROOT/'results/m14-retile-policy.json'
    if policy_path.exists():raise RuntimeError('frozen M14 policy exists; cannot overwrite')
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    gpu=torch.cuda.get_device_properties(0)
    if (gpu.major,gpu.minor)!=(8,6):raise RuntimeError('M14 SM86 only')
    run_id='m14-retile-tuning-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+str(time.time_ns()%10**9)
    path=ROOT/'results'/(run_id+'.json')
    record={'schema':1,'run_id':run_id,'status':'running','seed':9109,'training_T':[48,80,192],
            'selected_lengths':[511,3073],'source_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCE_FILES},
            'runtime':{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':gpu.name,'fa2':_runtime_info()},
            'limits':{'relative_rms':.008,'normalized_max':.008},'candidates':[asdict(c) for c in CANDIDATES],
            'boundary':[],'cases':[],'rechecks':[],'telemetry_before':_telemetry()}
    def save():path.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        q,segments=_make_custom_inputs(65,(33,17,41,65),seed=9109);oracle=_chunked_fp32_reference(q,segments)
        for i,config in enumerate(CANDIDATES):
            row={'candidate_index':i,'config':asdict(config),'status':'running'};record['boundary'].append(row);save()
            try:
                plan=prepare_segmented_kernel(q,segments,config)
                row['vs_fp32']=_check(plan.run(),oracle,'retile boundary')
                eq,es=_make_custom_inputs(0,(3,4,5,0),seed=9109);empty=prepare_segmented_kernel(eq,es,config)
                if empty.kernel is not None or empty.run().shape!=eq.shape:raise RuntimeError('empty query launched')
                row.update(status='passed',passed=True,empty_no_launch=True,candidate_resources=_resources(plan),owned_output_bytes=plan.workspace_bytes)
                del plan,empty
            except Exception as exc:row.update(failure_fields(exc))
            save()
        del q,segments,oracle;gc.collect();torch.cuda.empty_cache()
        for tokens in (48,80,192):
            for selected in (511,3073):
                case={'T':tokens,'selected_history':selected,'status':'running','candidates':[],'telemetry_before':_telemetry()}
                record['cases'].append(case);save();print(f'M14 training T={tokens} selected={selected}',flush=True)
                q,segments=_make_inputs(tokens,selected,seed=9109);oracle=_chunked_fp32_reference(q,segments)
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
        choices={}
        for bucket,tokens in zip(BUCKETS,(48,80,192)):
            scores=[]
            for i,config in enumerate(CANDIDATES):
                all_rows=[c['candidates'][i] for c in record['cases']]
                if not all(r.get('passed') for r in all_rows):continue
                ratios=[c['candidates'][i]['ratio_over_fa2'] for c in record['cases'] if c['T']==tokens]
                gm=math.exp(sum(math.log(x) for x in ratios)/len(ratios))
                scores.append({'candidate_index':i,'geomean_ratio':gm,'worst_ratio':max(ratios),'eligible':gm<=.95 and max(ratios)<=1.05})
            eligible=[x for x in scores if x['eligible']]
            selected_score=min(eligible,key=lambda x:(x['geomean_ratio'],x['candidate_index'])) if eligible else None
            record.setdefault('selection',{})[bucket]={'scores':scores,'initial_selection':selected_score};save()
            config=CANDIDATES[selected_score['candidate_index']] if selected_score else None
            accepted=False
            if config:
                ratios=[];failed=False
                for selected in (511,3073):
                    row={'bucket':bucket,'T':tokens,'selected_history':selected,'status':'running'};record['rechecks'].append(row);save()
                    q,segments=_make_inputs(tokens,selected,seed=9109);oracle=_chunked_fp32_reference(q,segments)
                    baseline=prepare_segmented_attention(q,segments);_check(baseline.run(),oracle,'fa2 recheck baseline')
                    try:
                        row.update(trial(q,segments,oracle,baseline,config,long=True))
                        expected=next(c['input_sha256'] for c in record['cases'] if c['T']==tokens and c['selected_history']==selected)
                        if row['input_sha256']!=expected:raise RuntimeError('recheck input differs')
                        ratios.append(row['ratio_over_fa2'])
                    except Exception as exc:
                        if str(exc).startswith('fa2'):raise
                        row.update(failure_fields(exc));failed=True
                    _check(baseline.run(),oracle,'fa2 after recheck');save()
                    del q,segments,oracle,baseline;gc.collect();torch.cuda.empty_cache()
                gm=math.exp(sum(math.log(x) for x in ratios)/len(ratios)) if ratios else None
                accepted=not failed and len(ratios)==2 and gm<=.95 and max(ratios)<=1.05
                record['selection'][bucket]['recheck']={'accepted':accepted,'geomean_ratio':gm,'worst_ratio':max(ratios) if ratios else None}
            choices[bucket]={'mode':'direct' if accepted else 'fa2','config':asdict(config) if accepted else None,
                             'candidate_index':selected_score['candidate_index'] if accepted else None}
        record.update(status='passed',frozen_choices=choices,telemetry_after=_telemetry());save()
        policy={'schema':1,'status':'frozen_before_holdout','domain':DOMAIN,'choices':choices,
                'source_hashes':record['source_hashes'],'training_record_path':str(path.relative_to(ROOT)),
                'training_record_sha256':canonical_training_hash(record),
                'training_hash_scope':'canonical_json_without_top_level_policy_sha256'}
        policy_path.write_text(json.dumps(policy,indent=2)+'\n');record['policy_sha256']=hashlib.sha256(policy_path.read_bytes()).hexdigest();save()
        FrozenRetilePolicy.load(policy_path).verify_provenance(ROOT)
        print('M14 frozen:',choices,flush=True)
    except Exception as exc:
        for group in ('boundary','cases','rechecks'):
            if record.get(group) and record[group][-1]['status']=='running':record[group][-1].update(failure_fields(exc))
        record.update(failure_fields(exc),telemetry_after=_telemetry());save();raise
    print('M14 training result:',path.relative_to(ROOT),flush=True)


if __name__=='__main__':run()
