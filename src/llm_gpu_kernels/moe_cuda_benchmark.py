"""M22 BF16 six-arm real-input benchmark; no policy/winner selection."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import torch
import triton

from .baselines.vllm_moe import prepare_vllm_moe
from .benchmark import ROOT, LIMITS, tensor_hash
from .gemm import GemmConfig
from .mature_moe_benchmark import _capture_graph, VLLM_PROVENANCE
from .moe_benchmark import safe_error, telemetry
from .moe_configured import prepare_configured_moe
from .moe_cuda import load_cuda_gate_up, prepare_cuda_moe, VARIANTS
from .moe_full_holdout import _ranked_state
from .moe_full_policy import FrozenMoEPolicy
from .moe_profile import write_record
from .moe_real_input_benchmark import owner_check
from .moe_real_trace import sha256_file, validate_real_moe_trace
from .optimization_benchmark import measure
from .reference import moe_reference

CASES = tuple(f'{p}-T{t}' for t in (17, 193, 1024) for p in ('archive', 'code'))


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--audit-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    a.capture_root = a.capture_root.resolve()
    if a.output.exists() or a.output.with_suffix(a.output.suffix + '.partial').exists():
        raise RuntimeError('refusing overwrite')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    r = dict(schema=1, status='running', experiment='M22 BF16 full-forward/stage performance',
             case=a.case, batch=a.batch, limits=LIMITS,
             interpretation='all arms reported; no new policy or winner selection',
             graph_repeats=8, cases_are_previously_measured_m20_inputs=True,
             head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             runtime=dict(python=platform.python_version(), torch=torch.__version__,
                          cuda=torch.version.cuda, triton=triton.__version__))
    write_record(a.output, r)
    try:
        owner_check()
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if torch.cuda.get_device_capability() != (8, 6):
            raise RuntimeError('requires SM86')
        policy = FrozenMoEPolicy.load(ROOT / 'results/m7-policy.json')
        policy.verify_provenance(ROOT)
        audit = json.loads(a.audit_record.read_text())
        if (audit.get('status') != 'passed' or len(audit.get('cases', [])) != 6
                or not audit.get('zero_tokens', {}).get('passed')):
            raise RuntimeError('predeclared six-case CUDA audit not passed')
        for name, digest in audit['source_hashes_after'].items():
            if sha256_file(ROOT / name) != digest:
                raise RuntimeError('audit does not cover current source: ' + name)
        sources = set(audit['source_hashes_after']) | {
            'src/llm_gpu_kernels/moe_cuda_benchmark.py', 'docs/M22_BF16_PERFORMANCE_PLAN.md',
            'src/llm_gpu_kernels/baselines/vllm_moe.py', 'src/llm_gpu_kernels/optimization_benchmark.py',
            'src/llm_gpu_kernels/moe_real_input_benchmark.py',
            'results/provenance/vllm-0.10.0-runtime.json'}
        r['source_hashes_before'] = {n: sha256_file(ROOT / n) for n in sorted(sources)}
        r['audit_record_sha256'] = sha256_file(a.audit_record)
        extension = load_cuda_gate_up()
        binary_path = Path(extension.__file__)
        r['binary'] = dict(name=binary_path.name, sha256_before=sha256_file(binary_path))
        if r['binary']['sha256_before'] != audit['binary']['sha256_after']:
            raise RuntimeError('loaded binary differs from audited binary')
        capture_file = a.capture_root / 'capture-summary.json'
        capture = json.loads(capture_file.read_text())
        if capture['status'] != 'passed':
            raise RuntimeError('capture failed')
        spec = next(s for s in capture['cases'] if f"{s['prompt']}-T{s['tokens']}" == a.case)
        manifest_path = (a.capture_root / spec['manifest']).resolve()
        manifest = json.loads(manifest_path.read_text())
        artifacts = [capture_file, manifest_path, *( (manifest_path.parent / ref['path']).resolve()
                      for ref in (manifest['trace'], *manifest['weights'].values()))]
        r['artifact_hashes_before'] = {str(p.relative_to(a.capture_root)): sha256_file(p)
                                       for p in sorted(set(artifacts))}
        case = validate_real_moe_trace(manifest_path, load_weights=True)
        x = case.tensors['x'].cuda().contiguous()
        ids = case.tensors['expert_ids'].cuda().contiguous()
        rw = case.tensors['route_weights'].cuda().contiguous()
        gate, down = (case.weights[n].cuda().contiguous() for n in ('gate_up', 'down'))
        inputs = (x, gate, down, ids, rw)
        selected, bucket = policy.select(x.shape[0], 128, 8, 2048, 768)
        if selected.block_m != 32:
            raise RuntimeError('only declared BM32 domain')
        r.update(input_sha256=tensor_hash(x, ids, rw), selected_bucket=bucket,
                 frozen_config=asdict(selected), cuda_gate_up_config=dict(BM=32, BN=64, BK=32, warps=4),
                 telemetry_before=telemetry())
        backends = dict(vllm=prepare_vllm_moe(*inputs),
                        default=prepare_configured_moe(*inputs, GemmConfig()),
                        frozen=prepare_configured_moe(*inputs, selected))
        backends.update({n: prepare_cuda_moe(*inputs, variant=n, down_config=selected) for n in VARIANTS})
        provenance = json.loads(VLLM_PROVENANCE.read_text())
        loaded_vllm = Path(sys.modules['vllm'].__file__).resolve()
        if not loaded_vllm.is_relative_to(Path(provenance['runtime_path']).resolve()):
            raise RuntimeError('vLLM loaded from wrong runtime')
        r['vllm_configuration'] = backends['vllm'].configuration()
        oracle = moe_reference(*inputs)
        functions = {n: b.run for n, b in backends.items()}
        def audit_outputs():
            result = {}
            for n, backend in backends.items():
                output = backend.run()
                torch.cuda.synchronize()
                numerical = safe_error(output, oracle)
                result[n] = dict(numerical=numerical)
                if n != 'vllm':
                    state = _ranked_state(backend, inputs)
                    state['ranked_dispatch'].pop('rank_values_by_expert', None)
                    result[n]['dispatch'] = state
                    if not state['passed']:
                        raise RuntimeError('dispatch gate failed')
                if not numerical['passed']:
                    r['failed_audit'] = result
                    write_record(a.output, r)
                    raise RuntimeError('numerical gate failed')
            return result
        r['audit_before'] = audit_outputs()
        graphs, graph_outputs = {}, {}
        for n, function in functions.items():
            graphs[n] = _capture_graph(function, 8, x.device)
            graph_outputs[n] = (backends[n].output if n == 'vllm' else backends[n].base.output)
        flops = 6 * ids.numel() * 2048 * 768
        r['full_api'] = measure(functions, flops, samples=6, repeats=2)
        r['full_graph'] = measure({n: g.replay for n, g in graphs.items()}, flops,
                                  divisor=8, samples=6, repeats=2)
        r['graph_numerics'] = {}
        for n, graph in graphs.items():
            graph.replay(); torch.cuda.synchronize()
            r['graph_numerics'][n] = safe_error(graph_outputs[n], oracle)
        if not all(m['passed'] for m in r['graph_numerics'].values()):
            raise RuntimeError('actual graph output numerical gate failed')
        stages = {n: b.launches[4].run for n, b in backends.items() if n != 'vllm'}
        stage_graphs = {n: _capture_graph(f, 8, x.device) for n, f in stages.items()}
        stage_flops = 4 * ids.numel() * 2048 * 768
        r['stage_api'] = measure(stages, stage_flops, samples=6, repeats=2)
        r['stage_graph'] = measure({n: g.replay for n, g in stage_graphs.items()}, stage_flops,
                                   divisor=8, samples=6, repeats=2)
        r['stage_scope'] = 'pregrouped fixed metadata gate/up+SwiGLU only, excludes routing/down/combine'
        r['audit_after'] = audit_outputs()
        r['input_sha256_after'] = tensor_hash(x, ids, rw)
        r['source_hashes_after'] = {n: sha256_file(ROOT / n) for n in sorted(sources)}
        r['artifact_hashes_after'] = {str(p.relative_to(a.capture_root)): sha256_file(p)
                                      for p in sorted(set(artifacts))}
        r['binary']['sha256_after'] = sha256_file(binary_path)
        r['telemetry_after'] = telemetry()
        policy.verify_provenance(ROOT)
        if (r['source_hashes_before'] != r['source_hashes_after'] or
                r['artifact_hashes_before'] != r['artifact_hashes_after'] or
                r['input_sha256'] != r['input_sha256_after'] or
                r['binary']['sha256_before'] != r['binary']['sha256_after'] or
                r['audit_record_sha256'] != sha256_file(a.audit_record)):
            raise RuntimeError('source/input/artifact/binary/audit changed')
        r['status'] = 'passed'
    except Exception as exc:
        r.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_record(a.output, r)


if __name__ == '__main__':
    run()
