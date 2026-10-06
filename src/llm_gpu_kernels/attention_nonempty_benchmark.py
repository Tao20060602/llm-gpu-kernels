"""M15 fixed same-configuration recurrence A/B with full FA2 comparison."""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
import triton

from .attention_baseline import prepare_segmented_attention
from .attention_baseline_benchmark import (
    _flat_tensors, _graph, _make_inputs, _runtime_info, _telemetry, _tensor_hash,
)
from .attention_segmented import AttentionConfig, prepare_segmented_kernel
from .attention_segmented_benchmark import (
    _functional, _check, _chunked_fp32_reference, _resources,
)
from .attention_nonempty import prepare_nonempty_kernel
from .attention_unified_tuning import measure, failure_fields
from .attention_retile_tuning import SOURCE_FILES as OLD_SOURCES
from .benchmark import compile_artifacts

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (*OLD_SOURCES, 'docs/M15_NONEMPTY_ATTENTION_PLAN.md',
                'src/llm_gpu_kernels/attention_nonempty.py',
                'src/llm_gpu_kernels/attention_nonempty_benchmark.py')
FUNCTIONAL_CONFIGS = (AttentionConfig(stages=1),
                      *(AttentionConfig(head_group=g) for g in (1, 2, 4)))


def performance(tokens, selected, run_id):
    torch.cuda.reset_peak_memory_stats()
    before = _telemetry()
    q, segments = _make_inputs(tokens, selected, seed=11109)
    inputs_hash = _tensor_hash(*_flat_tensors(q, segments))
    oracle = _chunked_fp32_reference(q, segments)
    plans = {}
    for stage in (1, 2):
        config = AttentionConfig(stages=stage)
        plans[f'old_s{stage}'] = prepare_segmented_kernel(q, segments, config)
        plans[f'new_s{stage}'] = prepare_nonempty_kernel(q, segments, config)
    fa2 = prepare_segmented_attention(q, segments)
    functions = {name: plan.run for name, plan in plans.items()}
    functions['pack_plus_fa2'] = fa2.run
    initial = {name: _check(fn(), oracle, name+' initial') for name, fn in functions.items()}
    captures = {name: _graph(fn, q.device) for name, fn in functions.items()}
    graph_initial = {name: _check(out, oracle, name+' initial Graph')
                     for name, (_, out) in captures.items()}
    api = measure(functions, divisor=1)
    graph = measure({name: g.replay for name, (g, _) in captures.items()})
    post = {name: _check(fn(), oracle, name+' post API') for name, fn in functions.items()}
    graph_post = {}
    for name, (capture, output) in captures.items():
        capture.replay()
        graph_post[name] = _check(output, oracle, name+' post Graph')
    if inputs_hash != _tensor_hash(*_flat_tensors(q, segments)):
        raise RuntimeError('performance inputs modified')
    evidence = {}
    for name, plan in plans.items():
        evidence[name] = compile_artifacts(plan.kernel, f'{name}-T{tokens}-S{selected}', run_id)
        evidence[name]['spills'] = plan.kernel.n_spills
    return {
        'status': 'passed', 'passed': True, 'T': tokens, 'selected_history': selected,
        'seed': 11109, 'input_sha256': inputs_hash, 'input_hash_preserved': True,
        'configs': {name: asdict(plan.config) for name, plan in plans.items()},
        'initial_vs_fp32': initial, 'initial_graph_vs_fp32': graph_initial,
        'post_api_vs_fp32': post, 'post_graph_vs_fp32': graph_post,
        'api': api, 'graph': graph,
        'resources': {name: _resources(plan) for name, plan in plans.items()},
        'compile_evidence': evidence,
        'owned_output_bytes': q.numel()*q.element_size(), 'candidate_scratch_bytes': 0,
        'baseline_packed_kv_bytes': fa2.packed_workspace_bytes,
        'process_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_scope': 'all arms, oracle, inputs, Graph pools/runtime; not single operator budget',
        'telemetry_before': before, 'telemetry_after': _telemetry(),
    }


def run():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    gpu = torch.cuda.get_device_properties(0)
    if (gpu.major, gpu.minor) != (8, 6):
        raise RuntimeError('M15 is scoped to SM86')
    run_id = 'm15-nonempty-attention-'+time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'-'+str(time.time_ns()%10**9)
    path = ROOT/'results'/(run_id+'.json')
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    record = {
        'schema': 1, 'run_id': run_id, 'status': 'running', 'source_hashes': hashes,
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'cuda_runtime': torch.version.cuda, 'triton': triton.__version__,
                    'gpu': gpu.name, 'sm': [gpu.major, gpu.minor], 'fa2': _runtime_info()},
        'seed': 11109, 'limits': {'relative_rms': .008, 'normalized_max': .008},
        'protocol': 'Fixed stages1/2, no tuning/selection; five-arm 4 balanced forward/reverse rounds,8 samples,2 calls;Graph8 forwards; per-forward units',
        'scope': 'GPU resident four-segment attention; no model/RoPE/H2D/NanoKV/serving',
        'functional': [], 'cases': [], 'telemetry_before': _telemetry(),
    }

    def save():
        path.write_text(json.dumps(record, indent=2)+'\n')

    save()
    try:
        for config in FUNCTIONAL_CONFIGS:
            row = {'config': asdict(config), 'status': 'running'}
            record['functional'].append(row); save()
            print('M15 functional', asdict(config), flush=True)
            try:
                row.update(_functional(config, progress=row, prepare=prepare_nonempty_kernel))
                row['status'] = 'passed'
            except Exception as exc:
                row.update(failure_fields(exc))
                for group in ('cases', 'graph_updates'):
                    if row.get(group) and row[group][-1]['status'] == 'running':
                        row[group][-1].update(failure_fields(exc))
                save(); raise
            save(); gc.collect(); torch.cuda.empty_cache()
        specs = [(t, s) for t in (64, 127, 256, 1024) for s in (0, 513, 2049)]+[(4096, 513)]
        for tokens, selected in specs:
            row = {'T': tokens, 'selected_history': selected, 'status': 'running'}
            record['cases'].append(row); save()
            print(f'M15 performance T={tokens} selected={selected}', flush=True)
            try:
                row.update(performance(tokens, selected, run_id))
            except Exception as exc:
                row.update(failure_fields(exc)); save(); raise
            save(); gc.collect(); torch.cuda.empty_cache()
        after = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
        if after != hashes:
            raise RuntimeError('experiment source changed during run')
        record.update(status='passed', source_hashes_after=after)
    except Exception as exc:
        record.update(failure_fields(exc)); raise
    finally:
        record['telemetry_after'] = _telemetry(); save()
        print('M15 result:', path.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    run()
