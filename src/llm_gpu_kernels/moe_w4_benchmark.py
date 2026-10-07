"""Frozen M22 W4 synchronous full-forward and fixed-metadata stage benchmark."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import torch
import triton

from .baselines.vllm_moe import prepare_vllm_moe
from .benchmark import ROOT, LIMITS, tensor_hash
from .mature_moe_benchmark import _capture_graph, VLLM_PROVENANCE
from .moe_benchmark import safe_error, telemetry
from .moe_configured import prepare_configured_moe
from .moe_cuda import load_cuda_gate_up, prepare_cuda_moe
from .moe_full_holdout import _ranked_state
from .moe_full_policy import FrozenMoEPolicy
from .moe_profile import write_record
from .moe_real_input_benchmark import owner_check
from .moe_real_trace import sha256_file, validate_real_moe_trace
from .moe_weight_quant import PackedGateUp, dequantize_gate_up_reference, _digest
from .moe_w4_cuda import load_w4_gate_up, prepare_w4_moe
from .optimization_benchmark import measure
from .reference import moe_reference

CASES = tuple(f'{p}-T{t}' for t in (17, 193, 1024) for p in ('archive', 'code'))
QUANT_ARMS = ('quant_vllm', 'quant_frozen', 'quant_cuda_sync', 'w4_sync')


def require_audit(path):
    audit = json.loads(path.read_text())
    if (audit.get('status') != 'passed' or len(audit.get('cases', [])) != 6
            or not audit.get('zero_tokens', {}).get('passed')):
        raise RuntimeError('six-case correctness audit not passed')
    for name, digest in audit['source_hashes_after'].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError('audit does not cover current source: ' + name)
    return audit


@contextmanager
def extension_root(root):
    """Select the certified cache for preparation, then restore caller state."""
    previous = os.environ.get('TORCH_EXTENSIONS_DIR')
    os.environ['TORCH_EXTENSIONS_DIR'] = str(root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('TORCH_EXTENSIONS_DIR', None)
        else:
            os.environ['TORCH_EXTENSIONS_DIR'] = previous


def load_audited(loader, audit, root):
    root = root.resolve(strict=True)
    name = audit['binary']['name']
    if Path(name).name != name or not name.endswith('.so'):
        raise RuntimeError('invalid audited extension basename')
    expected = root / name[:-3] / name
    if not expected.is_file() or sha256_file(expected) != audit['binary']['sha256_after']:
        raise RuntimeError('explicit cache does not contain audited binary: ' + name)
    with extension_root(root):
        module = loader()
    actual = Path(module.__file__).resolve()
    if actual != expected.resolve() or sha256_file(actual) != audit['binary']['sha256_after']:
        raise RuntimeError('extension cache changed or rebuilt the audited binary: ' + name)
    return module


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--batch', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--pack-record', type=Path, required=True)
    parser.add_argument('--w4-audit', type=Path, required=True)
    parser.add_argument('--bf16-audit', type=Path, required=True)
    parser.add_argument('--bf16-extension-root', type=Path, required=True)
    parser.add_argument('--w4-extension-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    a.capture_root = a.capture_root.resolve()
    if a.output.exists() or a.output.with_suffix(a.output.suffix + '.partial').exists():
        raise RuntimeError('refusing overwrite')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    r = dict(schema=1, status='running', experiment='M22 W4 full-forward/stage performance',
             case=a.case, batch=a.batch, limits=LIMITS, graph_repeats=8,
             quantized_arms=list(QUANT_ARMS), original_bf16_arms=['bf16_vllm', 'bf16_frozen'],
             interpretation='same-quantized-weight speed and lossy BF16 tradeoff separately; no task quality claim',
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
            raise RuntimeError('SM86 required')
        w4_audit, bf16_audit = require_audit(a.w4_audit), require_audit(a.bf16_audit)
        records = (a.w4_audit, a.bf16_audit, a.pack_record)
        r['record_hashes_before'] = {str(p): sha256_file(p) for p in records}
        sources = set(w4_audit['source_hashes_after']) | set(bf16_audit['source_hashes_after']) | {
            'src/llm_gpu_kernels/moe_w4_benchmark.py', 'docs/M22_W4_PERFORMANCE_PLAN.md',
            'src/llm_gpu_kernels/baselines/vllm_moe.py', 'src/llm_gpu_kernels/optimization_benchmark.py',
            'src/llm_gpu_kernels/moe_real_input_benchmark.py',
            'results/provenance/vllm-0.10.0-runtime.json'}
        r['source_hashes_before'] = {n: sha256_file(ROOT / n) for n in sorted(sources)}
        r['binaries'] = {}
        binary_paths = {}
        for name, loader, audit, cache in (('w4', load_w4_gate_up, w4_audit, a.w4_extension_root),
                                          ('bf16', load_cuda_gate_up, bf16_audit, a.bf16_extension_root)):
            module = load_audited(loader, audit, cache)
            p = Path(module.__file__)
            binary_paths[name] = p
            digest = sha256_file(p)
            if digest != audit['binary']['sha256_after']:
                raise RuntimeError('loaded binary differs from audit: ' + name)
            r['binaries'][name] = dict(name=p.name, sha256_before=digest)
        pack_record = json.loads(a.pack_record.read_text())
        r['packed_sha256_before'] = sha256_file(a.packed)
        if (pack_record.get('status') != 'passed' or
                r['packed_sha256_before'] != pack_record['artifact']['sha256'] or
                r['packed_sha256_before'] != w4_audit['packed_sha256_after']):
            raise RuntimeError('packed preparation/audit mismatch')
        data = torch.load(a.packed, map_location='cpu', weights_only=True)
        packed_cpu = PackedGateUp(data['words'], data['scales'], data['metadata'])
        if (_digest(packed_cpu.words) != packed_cpu.metadata['words_sha256'] or
                _digest(packed_cpu.scales) != packed_cpu.metadata['scales_sha256']):
            raise RuntimeError('packed payload digest mismatch')
        packed = packed_cpu.to('cuda')
        quant_gate = dequantize_gate_up_reference(packed)
        capture_file = a.capture_root / 'capture-summary.json'
        capture = json.loads(capture_file.read_text())
        if capture['status'] != 'passed':
            raise RuntimeError('capture failed')
        spec = next(s for s in capture['cases'] if f"{s['prompt']}-T{s['tokens']}" == a.case)
        manifest_path = (a.capture_root / spec['manifest']).resolve()
        manifest = json.loads(manifest_path.read_text())
        artifacts = [capture_file, manifest_path, *((manifest_path.parent / ref['path']).resolve()
                     for ref in (manifest['trace'], *manifest['weights'].values()))]
        r['artifact_hashes_before'] = {str(p.relative_to(a.capture_root)): sha256_file(p)
                                       for p in sorted(set(artifacts))}
        if r['artifact_hashes_before'].get('weights.pt') != pack_record['input']['manifest_weight_sha256']:
            raise RuntimeError('source checkpoint mismatch')
        case = validate_real_moe_trace(manifest_path, load_weights=True)
        if _digest(case.weights['gate_up']) != packed_cpu.metadata['source_sha256']:
            raise RuntimeError('BF16 source weight mismatch')
        x, ids, rw = (case.tensors[n].cuda().contiguous() for n in ('x', 'expert_ids', 'route_weights'))
        gate, down = (case.weights[n].cuda().contiguous() for n in ('gate_up', 'down'))
        quant_inputs, original_inputs = (x, quant_gate, down, ids, rw), (x, gate, down, ids, rw)
        policy = FrozenMoEPolicy.load(ROOT / 'results/m7-policy.json')
        policy.verify_provenance(ROOT)
        config, bucket = policy.select(x.shape[0], 128, 8, 2048, 768)
        r.update(input_sha256=tensor_hash(x, ids, rw), selected_bucket=bucket,
                 frozen_config=asdict(config), w4_config=dict(BM=32, BN=64, BK=32, warps=4, group_size=128),
                 telemetry_before=telemetry(), format_metadata=packed_cpu.metadata)
        backends = dict(quant_vllm=prepare_vllm_moe(*quant_inputs),
                        quant_frozen=prepare_configured_moe(*quant_inputs, config),
                        quant_cuda_sync=prepare_cuda_moe(*quant_inputs, variant='sync', down_config=config),
                        w4_sync=prepare_w4_moe(x, packed, down, ids, rw, down_config=config),
                        bf16_vllm=prepare_vllm_moe(*original_inputs),
                        bf16_frozen=prepare_configured_moe(*original_inputs, config))
        provenance = json.loads(VLLM_PROVENANCE.read_text())
        loaded_vllm = Path(sys.modules['vllm'].__file__).resolve()
        if not loaded_vllm.is_relative_to(Path(provenance['runtime_path']).resolve()):
            raise RuntimeError('vLLM loaded from wrong runtime')
        r['vllm_configuration'] = {n: backends[n].configuration() for n in ('quant_vllm', 'bf16_vllm')}
        quant_oracle, original_oracle = moe_reference(*quant_inputs), moe_reference(*original_inputs)
        r['format_forward_change'] = safe_error(quant_oracle, original_oracle)
        r['format_forward_change'].pop('passed', None)
        functions = {n: b.run for n, b in backends.items()}
        def audit_outputs():
            result = {}
            for n, backend in backends.items():
                output = backend.run()
                torch.cuda.synchronize()
                inputs = quant_inputs if n in QUANT_ARMS else original_inputs
                oracle = quant_oracle if n in QUANT_ARMS else original_oracle
                result[n] = dict(numerical=safe_error(output, oracle))
                if not n.endswith('vllm'):
                    state = _ranked_state(backend, inputs)
                    state['ranked_dispatch'].pop('rank_values_by_expert', None)
                    result[n]['dispatch'] = state
                    if not state['passed']:
                        raise RuntimeError('dispatch gate failed: ' + n)
                if not result[n]['numerical']['passed']:
                    r['failed_audit'] = result
                    write_record(a.output, r)
                    raise RuntimeError('numerical gate failed: ' + n)
            return result
        r['audit_before'] = audit_outputs()
        graphs, outputs = {}, {}
        for n, function in functions.items():
            graphs[n] = _capture_graph(function, 8, x.device)
            outputs[n] = backends[n].output if n.endswith('vllm') else backends[n].base.output
        flops = 6 * ids.numel() * 2048 * 768
        r['full_api'] = measure(functions, flops, samples=6, repeats=2)
        r['full_graph'] = measure({n: g.replay for n, g in graphs.items()}, flops, divisor=8, samples=6, repeats=2)
        r['graph_numerics'] = {}
        for n, graph in graphs.items():
            graph.replay(); torch.cuda.synchronize()
            r['graph_numerics'][n] = safe_error(outputs[n], quant_oracle if n in QUANT_ARMS else original_oracle)
        if not all(m['passed'] for m in r['graph_numerics'].values()):
            raise RuntimeError('actual Graph output numerical gate failed')
        stages = {n: backends[n].launches[4].run for n in ('quant_frozen', 'quant_cuda_sync', 'w4_sync')}
        stage_graphs = {n: _capture_graph(f, 8, x.device) for n, f in stages.items()}
        r['stage_api'] = measure(stages, 4 * ids.numel() * 2048 * 768, samples=6, repeats=2)
        r['stage_graph'] = measure({n: g.replay for n, g in stage_graphs.items()},
                                   4 * ids.numel() * 2048 * 768, divisor=8, samples=6, repeats=2)
        r['stage_scope'] = 'three fixed-metadata gate/up+SwiGLU arms; no routing/down/combine/official vLLM stage'
        r['audit_after'] = audit_outputs()
        r['input_sha256_after'] = tensor_hash(x, ids, rw)
        r['source_hashes_after'] = {n: sha256_file(ROOT / n) for n in sorted(sources)}
        r['artifact_hashes_after'] = {str(p.relative_to(a.capture_root)): sha256_file(p)
                                      for p in sorted(set(artifacts))}
        r['record_hashes_after'] = {str(p): sha256_file(p) for p in records}
        r['packed_sha256_after'] = sha256_file(a.packed)
        r['packed_gpu_digest_after'] = dict(words=_digest(packed.words), scales=_digest(packed.scales))
        for name, p in binary_paths.items():
            r['binaries'][name]['sha256_after'] = sha256_file(p)
        r['telemetry_after'] = telemetry()
        policy.verify_provenance(ROOT)
        if (r['source_hashes_before'] != r['source_hashes_after'] or
                r['artifact_hashes_before'] != r['artifact_hashes_after'] or
                r['record_hashes_before'] != r['record_hashes_after'] or
                r['input_sha256'] != r['input_sha256_after'] or
                r['packed_sha256_before'] != r['packed_sha256_after'] or
                any(b['sha256_before'] != b['sha256_after'] for b in r['binaries'].values()) or
                any(r['packed_gpu_digest_after'][n] != packed_cpu.metadata[n + '_sha256'] for n in ('words', 'scales'))):
            raise RuntimeError('source/input/packed/record/binary changed')
        r['status'] = 'passed'
    except Exception as exc:
        r.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_record(a.output, r)


if __name__ == '__main__':
    run()
