"""M9 exploratory four-segment attention: fixed G=1/2/4 vs strong FA2 chain."""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import torch
import triton

from .attention_baseline import prepare_segmented_attention, segmented_attention_reference
from .attention_baseline_benchmark import (
    _error, _flat_tensors, _graph, _make_custom_inputs, _make_inputs,
    _measure, _runtime_info, _telemetry, _tensor_hash,
)
from .attention_segmented import AttentionConfig, prepare_segmented_kernel
from .benchmark import compile_artifacts

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = tuple(AttentionConfig(head_group=g) for g in (1, 2, 4))


def _chunked_fp32_reference(q, segments, chunk=64):
    """Independent dense FP32 oracle with bounded temporary score storage."""
    if not q.shape[0]:
        return torch.empty_like(q, dtype=torch.float32)
    k = torch.cat(tuple(pair[0] for pair in segments)).float().repeat_interleave(4, dim=1).transpose(0, 1)
    v = torch.cat(tuple(pair[1] for pair in segments)).float().repeat_interleave(4, dim=1).transpose(0, 1)
    history = k.shape[1] - q.shape[0]
    keys = torch.arange(k.shape[1], device=q.device)
    out = torch.empty_like(q, dtype=torch.float32)
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        rows = torch.arange(start, end, device=q.device)
        scores = (q[start:end].float().transpose(0, 1) @ k.transpose(1, 2)) * (128 ** -0.5)
        scores.masked_fill_(keys[None, None, :] > (history + rows[None, :, None]), -float('inf'))
        out[start:end].copy_((torch.softmax(scores, dim=-1) @ v).transpose(0, 1))
    return out


def _resources(plan):
    if plan.kernel is None:
        return None
    return {"registers": plan.kernel.n_regs, "spills": plan.kernel.n_spills,
            "shared_memory_bytes": plan.kernel.metadata.shared}


def _check(actual, expected, message):
    metric = _error(actual, expected)
    if not metric['passed']:
        failure = RuntimeError(f'{message}: {metric}')
        failure.metrics = metric
        raise failure
    return metric


def _functional(config, progress=None, prepare=prepare_segmented_kernel):
    if progress is None:
        progress = {}
    specs = (
        ('empty_history', 7, (0, 0, 0, 7), False, False),
        ('segment_tails', 17, (5, 4, 7, 17), False, False),
        ('extreme_logits', 5, (2, 3, 4, 5), True, False),
        ('gqa_head_mapping', 6, (3, 2, 5, 6), False, True),
        ('empty_query', 0, (3, 4, 5, 0), False, False),
        ('tile_boundary', 65, (33, 17, 41, 65), False, False),
        ('empty_history_17', 17, (0, 0, 0, 17), False, False),
        ('short_tails_63', 63, (1, 5, 7, 63), False, False),
        ('cross_key_tile_65', 65, (3, 17, 5, 65), False, False),
        ('causal_boundary', 9, (2, 3, 5, 9), False, False),
        ('extreme_logits_9', 9, (2, 3, 5, 9), True, False),
        ('gqa_head_mapping_5', 5, (2, 2, 2, 5), False, True),
    )
    rows = progress.setdefault('cases', [])
    changes = progress.setdefault('graph_updates', [])
    progress['phase'] = 'functional_cases'
    for index, (name, tokens, lengths, extreme, head_map) in enumerate(specs):
        row = {'name': name, 'config': asdict(config), 'tokens': tokens,
               'lengths': lengths, 'status': 'running'}
        rows.append(row)
        q, segments = _make_custom_inputs(tokens, lengths, seed=1819 + index, extreme=extreme, head_map=head_map)
        if name == 'causal_boundary':
            q.zero_()
            offset = 0
            for k, v in segments:
                k.zero_()
                positions = torch.arange(offset, offset + k.shape[0], device=q.device)
                v.copy_((positions[:, None, None].float() / 32).expand_as(v).to(v.dtype))
                offset += k.shape[0]
        plan = prepare(q, segments, config=config)
        before = _tensor_hash(*_flat_tensors(q, segments))
        oracle = segmented_attention_reference(q, segments)
        fa2 = prepare_segmented_attention(q, segments).run()
        actual = plan.run()
        row.update({'vs_fp32': _check(actual, oracle, name + ' FP32'),
                     'vs_fa2': _check(actual, fa2, name + ' FA2'), 'resources': _resources(plan),
                     'empty_no_launch': tokens == 0 and plan.kernel is None,
                     'input_hash_preserved': before == _tensor_hash(*_flat_tensors(q, segments))})
        if not rows[-1]['input_hash_preserved']:
            raise RuntimeError('functional inputs modified')
        row['status'] = 'passed'
    # In-place values change, two updates, non-default stream plus fixed Graph.
    q, segments = _make_custom_inputs(9, (3, 4, 5, 9), seed=2829)
    progress['phase'] = 'graph_capture_and_updates'
    plan = prepare(q, segments, config=config)
    tensors = _flat_tensors(q, segments)
    original = tuple(x.clone() for x in tensors)
    original_hash = _tensor_hash(*tensors)
    graph, output = _graph(plan.run, q.device)
    current = torch.cuda.current_stream(q.device)
    for index in range(2):
        change = {'update': index, 'status': 'running'}
        changes.append(change)
        newq, newsegments = _make_custom_inputs(9, (3, 4, 5, 9), seed=3829 + index)
        stream = torch.cuda.Stream(device=q.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for target, value in zip(tensors, _flat_tensors(newq, newsegments)):
                target.copy_(value)
            side_output = plan.run()
        current.wait_stream(stream)
        oracle = segmented_attention_reference(q, segments)
        side_metric = _check(side_output, oracle, 'side stream update')
        graph.replay()
        change.update({'updated_hash': _tensor_hash(*tensors),
                       'side_stream_vs_fp32': side_metric,
                       'graph_vs_fp32': _check(output, oracle, 'graph update')})
        if change['updated_hash'] == original_hash or (index and change['updated_hash'] == changes[index-1]['updated_hash']):
            raise RuntimeError('graph update values did not change')
        change['status'] = 'passed'
    progress['phase'] = 'graph_restore'
    for target, value in zip(tensors, original):
        target.copy_(value)
    graph.replay()
    restore_metric = _check(output, segmented_attention_reference(q, segments), 'graph restored')
    restored_hash = _tensor_hash(*tensors)
    if restored_hash != original_hash:
        raise RuntimeError('graph input hash did not restore')
    return {'config': asdict(config), 'cases': rows, 'graph_updates': changes,
            'original_hash': original_hash, 'restored_hash': restored_hash,
            'restored_vs_fp32': restore_metric, 'passed': True}


def _performance(tokens, selected, run_id):
    torch.cuda.reset_peak_memory_stats()
    before_telemetry = _telemetry()
    q, segments = _make_inputs(tokens, selected)
    inputs_hash = _tensor_hash(*_flat_tensors(q, segments))
    oracle = _chunked_fp32_reference(q, segments)
    fa2 = prepare_segmented_attention(q, segments)
    plans = {f'segmented_G{c.head_group}': prepare_segmented_kernel(q, segments, config=c) for c in CONFIGS}
    functions = {'pack_plus_fa2': fa2.run, **{key: plan.run for key, plan in plans.items()}}
    initial = {name: _check(function(), oracle, name + ' initial') for name, function in functions.items()}
    captures = {name: _graph(function, q.device) for name, function in functions.items()}
    graphs = {name: _check(output, oracle, name + ' graph initial') for name, (_, output) in captures.items()}
    api = _measure(functions)
    graph = _measure({name: capture[0].replay for name, capture in captures.items()}, divisor=8)
    post = {name: _check(function(), oracle, name + ' post API') for name, function in functions.items()}
    post_graph = {}
    for name, (capture, output) in captures.items():
        capture.replay()
        post_graph[name] = _check(output, oracle, name + ' post graph')
    unchanged = inputs_hash == _tensor_hash(*_flat_tensors(q, segments))
    if not unchanged:
        raise RuntimeError('performance input hash changed')
    evidence = {}
    if tokens == 64 and selected == 512:
        for name, plan in plans.items():
            evidence[name] = compile_artifacts(plan.kernel, name, run_id)
            evidence[name]['spills'] = plan.kernel.n_spills
    return {'T': tokens, 'selected_history': selected, 'segment_lengths': [k.shape[0] for k, _ in segments],
            'status': 'passed', 'passed': True, 'input_sha256': inputs_hash, 'input_hash_preserved': unchanged,
            'initial_vs_fp32': initial, 'graph_vs_fp32': graphs, 'post_api_vs_fp32': post,
            'post_graph_vs_fp32': post_graph, 'api': api, 'graph': graph,
            'configs': {name: asdict(plan.config) for name, plan in plans.items()},
            'resources': {name: _resources(plan) for name, plan in plans.items()},
            'packed_kv_workspace_bytes': fa2.packed_workspace_bytes,
            'segmented_workspace_bytes': {name: plan.workspace_bytes for name, plan in plans.items()},
            'workspace_scope': 'segmented owns output only, zero scratch/pack; baseline packed KV excludes its output and internal state',
            'compile_evidence': evidence, 'process_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_memory_scope': 'includes independent FP32 oracle, all arms, graphs, inputs and runtime; not operator workspace',
            'telemetry_before': before_telemetry, 'telemetry_after': _telemetry()}


def run():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (gpu.major, gpu.minor) != (8, 6):
        raise RuntimeError('M9 is scoped to SM86')
    run_id = 'm9-segmented-attention-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + str(time.time_ns() % 10**9)
    files = ('docs/M9_SEGMENTED_ATTENTION_PLAN.md', 'src/llm_gpu_kernels/attention_segmented.py',
             'src/llm_gpu_kernels/attention_segmented_benchmark.py', 'src/llm_gpu_kernels/attention_baseline.py',
             'src/llm_gpu_kernels/attention_baseline_benchmark.py', 'src/llm_gpu_kernels/moe_prepared.py',
             'src/llm_gpu_kernels/benchmark.py')
    record = {'schema': 1, 'run_id': run_id, 'status': 'running',
              'date_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'runtime': {'fa2': _runtime_info(), 'python': platform.python_version(),
                          'torch': torch.__version__, 'torch_path': torch.__file__,
                          'cuda_runtime': torch.version.cuda, 'triton': triton.__version__,
                          'gpu': gpu.name, 'sm': [gpu.major, gpu.minor],
                          'tf32': False, 'bf16_reduced_precision_reduction': False},
              'source_hashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
              'configs': [asdict(c) for c in CONFIGS], 'limits': {'relative_rms': .008, 'normalized_max': .008},
              'protocol': 'Exploratory fixed G1/G2/G4, token_tile32 key_tile64 warps4 stages2; no candidate chosen from these measurements. Four balanced forward/reverse rounds, 8 samples, 2 calls, Graph8 forwards.',
              'scope': 'GPU resident full four-segment operator vs reusable pack + unmodified public FA2. No model, H2D, store, RoPE, router or serving.',
              'oracle': 'Every functional and performance case uses BF16 operands converted to independent dense FP32, TF32 off; production score tensors bounded by 64-query chunks. All pre/post API/Graph outputs checked.',
              'functional': [], 'cases': [], 'telemetry_before': _telemetry()}
    path = ROOT / 'results' / (run_id + '.json')
    def save():
        path.write_text(json.dumps(record, indent=2) + '\n')
    save()
    try:
        for config in CONFIGS:
            print('M9 functional', asdict(config), flush=True)
            functional_row = {'config': asdict(config), 'status': 'running'}
            record['functional'].append(functional_row); save()
            try:
                functional_row.update(_functional(config, progress=functional_row))
                functional_row['status'] = 'passed'
            except Exception as exc:
                functional_row.update(status='failed', failure=str(exc), traceback=traceback.format_exc())
                for group in ('cases', 'graph_updates'):
                    if functional_row.get(group) and functional_row[group][-1]['status'] == 'running':
                        functional_row[group][-1].update(status='failed', failure=str(exc), failure_metrics=getattr(exc, 'metrics', None))
                save()
                raise
            save()
            gc.collect(); torch.cuda.empty_cache()
        for tokens in (64, 256, 1024, 4096):
            for selected in (0, 512, 2048):
                row = {'T': tokens, 'selected_history': selected, 'status': 'running'}
                record['cases'].append(row); save()
                print(f'M9 performance T={tokens} selected={selected}', flush=True)
                try:
                    row.update(_performance(tokens, selected, run_id))
                except Exception as exc:
                    row.update(status='failed', passed=False, failure=str(exc), traceback=traceback.format_exc(), failure_metrics=getattr(exc, 'metrics', None))
                    save()
                    raise
                save()
                gc.collect(); torch.cuda.empty_cache()
        record['status'] = 'passed'
    except Exception as exc:
        record['status'] = 'failed'; record['failure'] = str(exc); record['traceback'] = traceback.format_exc()
        raise
    finally:
        record['telemetry_after'] = _telemetry(); save()
        print('M9 result:', path.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    run()
