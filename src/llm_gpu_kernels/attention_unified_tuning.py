"""Preregistered M10 bounded full-attention training and policy freeze."""
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch

from .attention_baseline_benchmark import _graph, _make_custom_inputs, _make_inputs, _flat_tensors, _tensor_hash, _telemetry, _runtime_info
from .attention_segmented import AttentionConfig, prepare_segmented_kernel
from .attention_segmented_benchmark import _check, _chunked_fp32_reference, _resources
from .attention_unified import prepare_unified_kernel
from .attention_unified_policy import CANDIDATES, DEFAULT, BUCKETS, DOMAIN, bucket_for, canonical_training_hash, FrozenAttentionPolicy

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    'docs/M10_UNIFIED_ATTENTION_TUNING_PLAN.md',
    'src/llm_gpu_kernels/attention_unified.py',
    'src/llm_gpu_kernels/attention_unified_policy.py',
    'src/llm_gpu_kernels/attention_unified_tuning.py',
    'src/llm_gpu_kernels/attention_unified_holdout.py',
    'src/llm_gpu_kernels/attention_segmented.py',
    'src/llm_gpu_kernels/attention_segmented_benchmark.py',
    'src/llm_gpu_kernels/attention_baseline.py',
    'src/llm_gpu_kernels/attention_baseline_benchmark.py',
    'src/llm_gpu_kernels/benchmark.py',
)


def measure(functions, samples=8, calls=2, divisor=8):
    names = list(functions)
    orders = (names, list(reversed(names)), list(reversed(names)), names)
    raw = {n: [] for n in names}
    walls = {n: [] for n in names}
    rounds = []
    for fn in functions.values():
        for _ in range(3): fn()
    torch.cuda.synchronize()
    for order in orders:
        row = {'order': order, 'event_ms_per_forward': {}, 'wall_ms_per_forward': {}}
        for name in order:
            events, wall = [], []
            for _ in range(samples):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter()
                start.record()
                for _ in range(calls): functions[name]()
                end.record(); end.synchronize()
                events.append(start.elapsed_time(end) / calls / divisor)
                wall.append((time.perf_counter() - begin) * 1000 / calls / divisor)
            raw[name].extend(events); walls[name].extend(wall)
            row['event_ms_per_forward'][name] = events
            row['wall_ms_per_forward'][name] = wall
        rounds.append(row)
    return {'rounds': rounds, 'samples_per_round': samples, 'calls_per_sample': calls,
            'forwards_per_call': divisor,
            'summary': {name: {'event_p50_ms_per_forward': statistics.median(values),
                               'event_p95_ms_per_forward': float(np.percentile(values, 95)),
                               'wall_p50_ms_per_forward': statistics.median(walls[name]),
                               'wall_p95_ms_per_forward': float(np.percentile(walls[name], 95))}
                        for name, values in raw.items()},
            'wall_scope': 'synchronized end-to-end sample including event creation and dispatch, not isolated host launch'}


def prepare_choice(mode, q, segments, config):
    return prepare_unified_kernel(q, segments, config) if mode == 'unified' else prepare_segmented_kernel(q, segments, config)


def pair_trial(q, segments, oracle, baseline, mode, config, *, long=False):
    inputs_hash = _tensor_hash(*_flat_tensors(q, segments))
    # Baseline checks must raise outside candidate-failure handling in caller.
    _check(baseline.run(), oracle, 'baseline initial')
    plan = prepare_choice(mode, q, segments, config)
    functions = {'m9_default': baseline.run, 'candidate': plan.run}
    initial = {name: _check(fn(), oracle, name+' initial') for name, fn in functions.items()}
    captures = {name: _graph(fn, q.device) for name, fn in functions.items()}
    graph_initial = {name: _check(output, oracle, name+' graph initial') for name, (_, output) in captures.items()}
    timing = measure({name: graph.replay for name, (graph, _) in captures.items()},
                     samples=16 if long else 8, calls=4 if long else 2)
    post = {name: _check(fn(), oracle, name+' post API') for name, fn in functions.items()}
    graph_post = {}
    for name, (graph, output) in captures.items():
        graph.replay()
        graph_post[name] = _check(output, oracle, name+' post Graph')
    if inputs_hash != _tensor_hash(*_flat_tensors(q, segments)):
        raise RuntimeError('training inputs modified')
    ratio = timing['summary']['candidate']['event_p50_ms_per_forward'] / timing['summary']['m9_default']['event_p50_ms_per_forward']
    return {'status': 'passed', 'passed': True, 'config': asdict(config), 'mode': mode,
            'input_sha256': inputs_hash, 'initial_vs_fp32': initial, 'initial_graph_vs_fp32': graph_initial,
            'post_api_vs_fp32': post, 'post_graph_vs_fp32': graph_post,
            'timing_graph': timing, 'ratio_candidate_over_m9': ratio,
            'resources': _resources(plan), 'owned_output_bytes': plan.workspace_bytes}


def failure_fields(exc):
    return {'status': 'failed', 'passed': False, 'failure': str(exc),
            'failure_metrics': getattr(exc, 'metrics', None), 'traceback': traceback.format_exc()}


def run():
    policy_path = ROOT / 'results/m10-attention-policy.json'
    if policy_path.exists():
        raise RuntimeError('M10 policy already exists; frozen policy is not overwritten')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    gpu = torch.cuda.get_device_properties(0)
    if (gpu.major, gpu.minor) != (8, 6): raise RuntimeError('M10 scoped to SM86')
    run_id = 'm10-attention-tuning-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + str(time.time_ns()%10**9)
    path = ROOT / 'results' / (run_id+'.json')
    record = {'schema': 1, 'run_id': run_id, 'status': 'running',
              'source_hashes': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCE_FILES},
              'runtime': {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'gpu': gpu.name, 'fa2': _runtime_info()},
              'seed': 1907, 'training_T': [96,384,1536], 'selected_lengths': [256,1536],
              'limits': {'relative_rms': .008, 'normalized_max': .008},
              'selection': 'all-six plus boundary correct; bucket graph geomean<=.95/worst<=1.05; long recheck else M9G1 fallback',
              'candidates': [asdict(c) for c in CANDIDATES], 'boundary': [], 'cases': [],
              'bucket_selection': {}, 'rechecks': [], 'telemetry_before': _telemetry()}
    def save(): path.write_text(json.dumps(record, indent=2)+'\n')
    save()
    try:
        q, segments = _make_custom_inputs(65, (33,17,41,65), seed=1907)
        oracle = _chunked_fp32_reference(q, segments)
        for index, config in enumerate(CANDIDATES):
            row = {'candidate_index': index, 'config': asdict(config), 'status': 'running'}
            record['boundary'].append(row); save()
            try:
                plan = prepare_unified_kernel(q, segments, config)
                row.update(vs_fp32=_check(plan.run(), oracle, 'candidate boundary'), resources=_resources(plan))
                emptyq, emptysegments = _make_custom_inputs(0,(3,4,5,0),seed=1907)
                empty = prepare_unified_kernel(emptyq, emptysegments, config)
                if empty.kernel is not None or empty.run().shape != emptyq.shape:
                    raise RuntimeError('candidate empty query launched or shape differs')
                row.update(status='passed', passed=True, empty_no_launch=True)
                del plan, empty
            except Exception as exc:
                row.update(failure_fields(exc))
            save()
        del q, segments, oracle
        gc.collect(); torch.cuda.empty_cache()
        for tokens in (96,384,1536):
            for selected in (256,1536):
                case = {'T': tokens, 'selected_history': selected, 'status': 'running',
                        'telemetry_before': _telemetry(), 'candidates': []}
                record['cases'].append(case); save()
                print(f'M10 training T={tokens} selected={selected}', flush=True)
                q, segments = _make_inputs(tokens, selected, seed=1907)
                oracle = _chunked_fp32_reference(q, segments)
                baseline = prepare_segmented_kernel(q, segments, DEFAULT)
                case['baseline_vs_fp32'] = _check(baseline.run(), oracle, 'training baseline')
                case['input_sha256'] = _tensor_hash(*_flat_tensors(q, segments))
                for index, config in enumerate(CANDIDATES):
                    row = {'candidate_index': index, 'config': asdict(config), 'status': 'running'}
                    case['candidates'].append(row); save()
                    if not record['boundary'][index].get('passed'):
                        row.update(status='excluded_boundary_failure', passed=False); save(); continue
                    # Check before the try so a broken baseline stops the run.
                    _check(baseline.run(), oracle, 'training baseline before candidate')
                    try:
                        row.update(pair_trial(q, segments, oracle, baseline, 'unified', config))
                    except Exception as exc:
                        if str(exc).startswith(('baseline', 'm9_default')):
                            raise
                        row.update(failure_fields(exc))
                    _check(baseline.run(), oracle, 'training baseline after candidate')
                    save()
                case.update(status='completed', all_candidates_passed=all(r.get('passed') for r in case['candidates']), telemetry_after=_telemetry())
                del baseline, q, segments, oracle
                gc.collect(); torch.cuda.empty_cache(); save()
        choices = {}
        for bucket, tokens in zip(BUCKETS,(96,384,1536)):
            scores = []
            for index, config in enumerate(CANDIDATES):
                all_rows = [case['candidates'][index] for case in record['cases']]
                if not all(row.get('passed') for row in all_rows): continue
                selected_rows = [case['candidates'][index] for case in record['cases'] if case['T']==tokens]
                ratios = [row['ratio_candidate_over_m9'] for row in selected_rows]
                gm = math.exp(sum(math.log(x) for x in ratios)/len(ratios))
                scores.append({'candidate_index': index, 'geomean_ratio': gm, 'worst_ratio': max(ratios),
                               'eligible': gm<=.95 and max(ratios)<=1.05})
            eligible = [x for x in scores if x['eligible']]
            chosen = min(eligible,key=lambda x:(x['geomean_ratio'],x['candidate_index'])) if eligible else None
            choices[bucket] = {'mode': 'unified' if chosen else 'm9',
                               'config': asdict(CANDIDATES[chosen['candidate_index']] if chosen else DEFAULT),
                               'candidate_index': chosen['candidate_index'] if chosen else None,
                               'reason': 'training_selected' if chosen else 'no_eligible_candidate'}
            record['bucket_selection'][bucket] = {'scores': scores, 'initial_choice': dict(choices[bucket])}
            save()
            if chosen:
                ratios=[]; failed=False
                for selected in (256,1536):
                    row={'bucket':bucket,'T':tokens,'selected_history':selected,'status':'running'}
                    record['rechecks'].append(row); save()
                    q,segments=_make_inputs(tokens,selected,seed=1907)
                    oracle=_chunked_fp32_reference(q,segments)
                    baseline=prepare_segmented_kernel(q,segments,DEFAULT)
                    _check(baseline.run(),oracle,'recheck baseline')
                    try:
                        row.update(pair_trial(q,segments,oracle,baseline,'unified',CANDIDATES[chosen['candidate_index']],long=True))
                        expected=next(case['input_sha256'] for case in record['cases'] if case['T']==tokens and case['selected_history']==selected)
                        if row['input_sha256']!=expected:raise RuntimeError('recheck training input hash mismatch')
                        ratios.append(row['ratio_candidate_over_m9'])
                    except Exception as exc:
                        if str(exc).startswith(('baseline', 'm9_default')):
                            raise
                        row.update(failure_fields(exc)); failed=True
                    _check(baseline.run(),oracle,'recheck baseline after candidate')
                    del q,segments,oracle,baseline
                    gc.collect();torch.cuda.empty_cache();save()
                gm=math.exp(sum(math.log(x) for x in ratios)/len(ratios)) if ratios else None
                accepted=not failed and len(ratios)==2 and gm<=.95 and max(ratios)<=1.05
                record['bucket_selection'][bucket]['recheck']={'accepted':accepted,'geomean_ratio':gm,'worst_ratio':max(ratios) if ratios else None}
                if not accepted:
                    choices[bucket]={'mode':'m9','config':asdict(DEFAULT),'candidate_index':None,'reason':'long_recheck_failed'}
                else: choices[bucket]['reason']='long_recheck_accepted'
        record.update(status='passed', frozen_choices=choices, telemetry_after=_telemetry())
        save()
        policy={'schema':1,'status':'frozen_before_holdout','domain':DOMAIN,'thresholds_T':[128,512],
                'choices':choices,'source_hashes':record['source_hashes'],
                'training_record_path':str(path.relative_to(ROOT)),
                'training_record_sha256':canonical_training_hash(record),
                'training_hash_scope':'canonical_json_without_top_level_policy_sha256'}
        policy_path.write_text(json.dumps(policy,indent=2)+'\n')
        record['policy_sha256']=hashlib.sha256(policy_path.read_bytes()).hexdigest(); save()
        FrozenAttentionPolicy.load(policy_path).verify_provenance(ROOT)
        print('M10 policy frozen:', choices, flush=True)
    except Exception as exc:
        for group in ('cases', 'rechecks', 'boundary'):
            if record.get(group) and record[group][-1]['status'] == 'running':
                record[group][-1].update(failure_fields(exc))
        record.update(failure_fields(exc), telemetry_after=_telemetry()); save()
        raise
    print('M10 training result:',path.relative_to(ROOT),flush=True)


if __name__=='__main__':run()
