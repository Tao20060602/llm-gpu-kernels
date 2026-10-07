"""M21 bounded full-MoE diagnostic; old kernels and policies stay frozen."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import traceback

import torch
import triton

from .benchmark import ROOT, LIMITS, tensor_hash
from .gemm import GemmConfig
from .mature_moe_benchmark import _capture_graph
from .moe_benchmark import safe_error, telemetry
from .moe_configured import prepare_configured_moe
from .moe_full_holdout import _ranked_state
from .moe_full_policy import FrozenMoEPolicy
from .moe_real_input_benchmark import owner_check
from .moe_real_trace import validate_real_moe_trace, sha256_file
from .optimization_benchmark import measure
from .reference import moe_reference


STAGES = ('reset', 'rank_histogram', 'prefix', 'rank_dispatch',
          'gate_up_swiglu', 'down', 'combine')
CASES = ('archive-T17-real', 'archive-T1024-real',
         'archive-T1024-balanced128', 'archive-T1024-hot16')
ARMS = ('default', 'frozen', 'bn128')
PROFILE_REPEATS = 8


def write_record(path, row):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(path)


def task_budget(counts, config, tokens):
    useful_rows = sum((int(n) + config.block_m - 1) // config.block_m for n in counts)
    bound = triton.cdiv(tokens * 8, config.block_m) + 127
    return dict(active_experts=sum(n > 0 for n in counts), route_rows=tokens * 8,
                maximum_routes_per_expert=max(counts), useful_row_tiles=useful_rows,
                scheduled_row_tile_bound=bound,
                padded_route_rows=useful_rows * config.block_m,
                gate_up_column_tiles=triton.cdiv(768, config.block_n),
                down_column_tiles=triton.cdiv(2048, config.block_n),
                gate_up_grid_ctas=bound * triton.cdiv(768, config.block_n),
                down_grid_ctas=bound * triton.cdiv(2048, config.block_n),
                note='Logical/launch counts, not hardware traffic or measured occupancy.')


def static_resources(backend):
    return {launch.name: dict(grid=list(launch.grid),
                             registers=getattr(launch.kernel, 'n_regs', None),
                             spills=getattr(launch.kernel, 'n_spills', None),
                             shared_bytes=getattr(launch.kernel.metadata, 'shared', None))
            for launch in backend.launches}


def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-root', type=Path, required=True)
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--phase', choices=('perf', 'profile'), required=True)
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if (a.phase == 'profile') != (a.arm is not None):
        p.error('profile requires exactly one --arm; perf compares all arms')
    if a.output.exists() or a.output.with_suffix(a.output.suffix + '.partial').exists():
        raise RuntimeError('refusing to overwrite existing diagnostic')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    r = dict(schema=1, experiment='M21 MoE mechanism diagnostic', status='running',
             phase=a.phase, case=a.case, arm=a.arm, stages=list(STAGES),
             profile_repeats=PROFILE_REPEATS, limits=LIMITS,
             scope='single-layer fixed-route expert-forward; not router/model/serving',
             runtime=dict(python=platform.python_version(), torch=torch.__version__,
                          cuda=torch.version.cuda, triton=triton.__version__),
             head=subprocess.check_output(['git', '-c', 'safe.directory=' + str(ROOT),
                                           'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write_record(a.output, r)
    try:
        owner_check()
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if torch.cuda.get_device_capability() != (8, 6):
            raise RuntimeError('requires actual SM86 device')
        policy = FrozenMoEPolicy.load(ROOT / 'results/m7-policy.json')
        r['policy_check_before'] = policy.verify_provenance(ROOT)
        sources = set(dict(policy.source_hashes)) | {
            'src/llm_gpu_kernels/moe_profile.py', 'docs/M21_MOE_PROFILE_PLAN.md',
            'src/llm_gpu_kernels/moe_real_trace.py',
            'src/llm_gpu_kernels/moe_real_input_benchmark.py',
            'src/llm_gpu_kernels/moe_full_holdout.py',
            'src/llm_gpu_kernels/optimization_benchmark.py',
            'src/llm_gpu_kernels/mature_moe_benchmark.py',
            'src/llm_gpu_kernels/moe_benchmark.py', 'src/llm_gpu_kernels/reference.py'}
        r['source_hashes_before'] = {name: sha256_file(ROOT / name) for name in sorted(sources)}
        capture_path = a.capture_root / 'capture-summary.json'
        capture = json.loads(capture_path.read_text())
        tokens = 17 if a.case == CASES[0] else 1024
        spec = next(row for row in capture['cases'] if row['prompt'] == 'archive' and row['tokens'] == tokens)
        manifest_path = (a.capture_root / spec['manifest']).resolve()
        manifest = json.loads(manifest_path.read_text())
        artifacts = [capture_path, manifest_path]
        for ref in (manifest['trace'], *manifest['weights'].values()):
            artifacts.append((manifest_path.parent / ref['path']).resolve())
        r['artifact_hashes_before'] = {str(path): sha256_file(path) for path in sorted(set(artifacts))}
        if capture['status'] != 'passed':
            raise RuntimeError('capture not passed')
        case = validate_real_moe_trace(manifest_path, load_weights=True)
        x = case.tensors['x'].cuda().contiguous()
        ids_cpu = case.tensors['expert_ids']
        if a.case.endswith('balanced128') or a.case.endswith('hot16'):
            span = 128 if a.case.endswith('balanced128') else 16
            ids_cpu = torch.arange(tokens * 8, dtype=ids_cpu.dtype).reshape(tokens, 8) % span
        ids = ids_cpu.cuda().contiguous()
        rw = case.tensors['route_weights'].cuda().contiguous()
        gate_up, down = (case.weights[n].cuda().contiguous() for n in ('gate_up', 'down'))
        buffers = (x, gate_up, down, ids, rw)
        counts = torch.bincount(ids_cpu.long().flatten(), minlength=128).tolist()
        selected, bucket = policy.select(tokens, 128, 8, 2048, 768)
        configs = dict(default=GemmConfig(), frozen=selected,
                       bn128=GemmConfig(block_m=32, block_n=128, block_k=32, warps=4, stages=3))
        names = (a.arm,) if a.arm else ARMS
        r.update(tokens=tokens, input_sha256=tensor_hash(x, ids, rw), route_counts=counts,
                 routing_kind='real' if a.case.endswith('real') else 'counterfactual modular distinct IDs',
                 selected_bucket=bucket, configs={n: asdict(configs[n]) for n in names},
                 budgets={n: task_budget(counts, configs[n], tokens) for n in names},
                 telemetry_before=telemetry())
        backends = {n: prepare_configured_moe(*buffers, configs[n]) for n in names}
        for backend in backends.values():
            if tuple(launch.name for launch in backend.launches) != STAGES:
                raise RuntimeError('unexpected prepared launch sequence')
        r['static_resources'] = {n: static_resources(b) for n, b in backends.items()}
        oracle = moe_reference(*buffers)
        def audit():
            output = {}
            for name, backend in backends.items():
                backend.run()
                torch.cuda.synchronize()
                output[name] = dict(numerical=safe_error(backend.base.output, oracle),
                                    dispatch=_ranked_state(backend, buffers))
                output[name]['dispatch']['ranked_dispatch'].pop('rank_values_by_expert', None)
                if not output[name]['numerical']['passed'] or not output[name]['dispatch']['passed']:
                    r['failed_audit'] = output
                    write_record(a.output, r)
                    raise RuntimeError('fixed numerical/dispatch gate failed')
            return output
        r['audit_before'] = audit()
        write_record(a.output, r)
        functions = {n: b.run for n, b in backends.items()}
        if a.phase == 'perf':
            graphs = {n: _capture_graph(f, 8, x.device) for n, f in functions.items()}
            flops = 6 * tokens * 8 * 2048 * 768
            r['api'] = measure(functions, flops, samples=6, repeats=2)
            r['graph'] = measure({n: g.replay for n, g in graphs.items()}, flops,
                                 divisor=8, samples=6, repeats=2)
            r['graph_vs_fp32'] = {}
            for n, g in graphs.items():
                g.replay(); torch.cuda.synchronize()
                r['graph_vs_fp32'][n] = safe_error(backends[n].base.output, oracle)
            if not all(m['passed'] for m in r['graph_vs_fp32'].values()):
                raise RuntimeError('Graph numerical gate failed')
        else:
            b = backends[a.arm]
            for _ in range(10):
                b.run()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            try:
                torch.cuda.nvtx.range_push('m21.moe.roi')
                try:
                    for _ in range(PROFILE_REPEATS):
                        for launch in b.launches:
                            torch.cuda.nvtx.range_push('m21.stage.' + launch.name)
                            try:
                                launch.run()
                            finally:
                                torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
                finally:
                    torch.cuda.nvtx.range_pop()
            finally:
                torch.cuda.cudart().cudaProfilerStop()
            r['timing_scope'] = 'instrumented profile, not clean latency benchmark'
        r['audit_after'] = audit()
        r['input_sha256_after'] = tensor_hash(x, ids, rw)
        r['source_hashes_after'] = {name: sha256_file(ROOT / name) for name in sorted(sources)}
        r['artifact_hashes_after'] = {str(path): sha256_file(path) for path in sorted(set(artifacts))}
        r['policy_check_after'] = policy.verify_provenance(ROOT)
        r['telemetry_after'] = telemetry()
        if r['source_hashes_after'] != r['source_hashes_before'] or r['artifact_hashes_after'] != r['artifact_hashes_before'] or r['input_sha256_after'] != r['input_sha256']:
            raise RuntimeError('source/artifact/input changed during measurement')
        r['status'] = 'passed'
    except Exception as exc:
        r.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_record(a.output, r)


if __name__ == '__main__':
    run()
