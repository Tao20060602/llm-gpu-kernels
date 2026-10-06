"""M12 single fixed split candidate, new small-query training and recheck."""
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from .attention_baseline import prepare_segmented_attention
from .attention_baseline_benchmark import _make_inputs,_telemetry,_runtime_info,_flat_tensors,_tensor_hash
from .attention_segmented_benchmark import _functional,_check,_chunked_fp32_reference
from .attention_split import prepare_split_kernel,SplitAttentionConfig
from .attention_split_tuning import trial,SOURCE_FILES as OLD_SOURCES
from .attention_unified_policy import canonical_training_hash
from .attention_unified_tuning import failure_fields
from .attention_narrow_policy import FrozenNarrowPolicy

ROOT=Path(__file__).resolve().parents[2]
CONFIG=SplitAttentionConfig(token_tile=16,splits=2)
SOURCE_FILES=(*OLD_SOURCES,'docs/M12_NARROW_SPLIT_PLAN.md',
 'src/llm_gpu_kernels/attention_narrow_policy.py','src/llm_gpu_kernels/attention_narrow_tuning.py',
 'src/llm_gpu_kernels/attention_narrow_holdout.py')


def run():
    policy_path=ROOT/'results/m12-narrow-policy.json'
    if policy_path.exists():raise RuntimeError('frozen M12 policy exists; cannot overwrite')
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    gpu=torch.cuda.get_device_properties(0)
    if (gpu.major,gpu.minor)!=(8,6):raise RuntimeError('M12 SM86 only')
    run_id='m12-narrow-tuning-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+str(time.time_ns()%10**9)
    path=ROOT/'results'/(run_id+'.json')
    origin=ROOT/'results/m11-split-tuning-20261006T140016Z-227795202.json'
    record={'schema':1,'run_id':run_id,'status':'running','seed':5109,'training_T':[24,56],
            'selected_lengths':[257,1537],'config':asdict(CONFIG),
            'hypothesis_origin':str(origin.relative_to(ROOT)),'hypothesis_origin_sha256':hashlib.sha256(origin.read_bytes()).hexdigest(),
            'source_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCE_FILES},
            'runtime':{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':gpu.name,'fa2':_runtime_info()},
            'limits':{'relative_rms':.008,'normalized_max':.008},'functional':{},'cases':[],'rechecks':[],
            'telemetry_before':_telemetry()}
    def save():path.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        record['functional'].update(config=asdict(CONFIG),status='running');save()
        try:
            record['functional'].update(_functional(CONFIG,progress=record['functional'],prepare=prepare_split_kernel))
            record['functional']['status']='passed'
        except Exception as exc:
            row=record['functional'];row.update(failure_fields(exc))
            for group in ('cases','graph_updates'):
                if row.get(group) and row[group][-1]['status']=='running':row[group][-1].update(failure_fields(exc))
            save();raise
        save()
        for tokens in (24,56):
            for selected in (257,1537):
                row={'T':tokens,'selected_history':selected,'status':'running','telemetry_before':_telemetry()}
                record['cases'].append(row);save();print(f'M12 training T={tokens} selected={selected}',flush=True)
                q,segments=_make_inputs(tokens,selected,seed=5109);oracle=_chunked_fp32_reference(q,segments)
                baseline=prepare_segmented_attention(q,segments);_check(baseline.run(),oracle,'fa2 baseline')
                try:row.update(trial(q,segments,oracle,baseline,CONFIG))
                except Exception as exc:
                    if str(exc).startswith('fa2'):raise
                    row.update(failure_fields(exc))
                _check(baseline.run(),oracle,'fa2 baseline after candidate')
                row['telemetry_after']=_telemetry();save()
                del q,segments,oracle,baseline;gc.collect();torch.cuda.empty_cache()
        correct=all(row.get('passed') for row in record['cases'])
        ratios=[row['ratio_over_fa2'] for row in record['cases'] if row.get('passed')]
        gm=math.exp(sum(math.log(x) for x in ratios)/len(ratios)) if ratios else None
        eligible=correct and len(ratios)==4 and gm<=.95 and max(ratios)<=1.05
        record['training_summary']={'eligible':eligible,'geomean_ratio':gm,'worst_ratio':max(ratios) if ratios else None};save()
        accepted=False
        if eligible:
            ratios=[];failed=False
            for tokens in (24,56):
                for selected in (257,1537):
                    row={'T':tokens,'selected_history':selected,'status':'running'};record['rechecks'].append(row);save()
                    q,segments=_make_inputs(tokens,selected,seed=5109);oracle=_chunked_fp32_reference(q,segments)
                    baseline=prepare_segmented_attention(q,segments);_check(baseline.run(),oracle,'fa2 recheck baseline')
                    try:
                        row.update(trial(q,segments,oracle,baseline,CONFIG,long=True))
                        expected=next(c['input_sha256'] for c in record['cases'] if c['T']==tokens and c['selected_history']==selected)
                        if row['input_sha256']!=expected:raise RuntimeError('recheck training input mismatch')
                        ratios.append(row['ratio_over_fa2'])
                    except Exception as exc:
                        if str(exc).startswith('fa2'):raise
                        row.update(failure_fields(exc));failed=True
                    _check(baseline.run(),oracle,'fa2 after recheck');save()
                    del q,segments,oracle,baseline;gc.collect();torch.cuda.empty_cache()
            gm=math.exp(sum(math.log(x) for x in ratios)/len(ratios)) if ratios else None
            accepted=not failed and len(ratios)==4 and gm<=.95 and max(ratios)<=1.05
            record['recheck_summary']={'accepted':accepted,'geomean_ratio':gm,'worst_ratio':max(ratios) if ratios else None}
        record.update(status='passed',frozen_mode='split' if accepted else 'fa2',telemetry_after=_telemetry());save()
        policy={'schema':1,'status':'frozen_before_holdout','max_T':64,'mode':record['frozen_mode'],
                'config':asdict(CONFIG) if accepted else None,'source_hashes':record['source_hashes'],
                'training_record_path':str(path.relative_to(ROOT)),'training_record_sha256':canonical_training_hash(record),
                'training_hash_scope':'canonical_json_without_top_level_policy_sha256'}
        policy_path.write_text(json.dumps(policy,indent=2)+'\n');record['policy_sha256']=hashlib.sha256(policy_path.read_bytes()).hexdigest();save()
        FrozenNarrowPolicy.load(policy_path).verify_provenance(ROOT)
        print('M12 frozen:',record['frozen_mode'],record['training_summary'],record.get('recheck_summary'),flush=True)
    except Exception as exc:
        for group in ('cases','rechecks'):
            if record.get(group) and record[group][-1]['status']=='running':record[group][-1].update(failure_fields(exc))
        record.update(failure_fields(exc),telemetry_after=_telemetry());save();raise
    print('M12 training result:',path.relative_to(ROOT),flush=True)


if __name__=='__main__':run()
