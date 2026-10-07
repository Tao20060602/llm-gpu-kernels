"""M22 W4A16 same-quantized-weight oracle audit; no net latency claims."""
import argparse
import gc
import json
from pathlib import Path
import traceback

import torch

from .benchmark import ROOT, LIMITS, tensor_hash
from .mature_moe_benchmark import _capture_graph
from .moe_benchmark import safe_error
from .moe_full_holdout import _ranked_state
from .moe_full_policy import FrozenMoEPolicy
from .moe_profile import write_record
from .moe_real_trace import sha256_file, validate_real_moe_trace
from .moe_weight_quant import PackedGateUp, dequantize_gate_up_reference, _digest
from .moe_w4_cuda import load_w4_gate_up, prepare_w4_moe
from .reference import moe_reference


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--pack-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    a.capture_root = a.capture_root.resolve()
    if a.output.exists() or a.output.with_suffix(a.output.suffix + '.partial').exists():
        raise RuntimeError('refusing overwrite')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    r = dict(schema=1, status='running', experiment='M22 W4A16 full-forward correctness', limits=LIMITS,
             cases=[], interpretation='same-dequantized-BF16-weight oracle; not model task quality or performance')
    write_record(a.output, r)
    try:
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if torch.cuda.get_device_capability() != (8, 6):
            raise RuntimeError('SM86 required')
        pack_record = json.loads(a.pack_record.read_text())
        if pack_record.get('status') != 'passed':
            raise RuntimeError('offline format preparation did not pass')
        r['packed_sha256_before'] = sha256_file(a.packed)
        if r['packed_sha256_before'] != pack_record['artifact']['sha256']:
            raise RuntimeError('packed artifact differs from audited preparation')
        if sha256_file(ROOT / 'src/llm_gpu_kernels/moe_weight_quant.py') != pack_record['format_source']['quantizer_sha256_before_after']['after']:
            raise RuntimeError('packing reference source differs from preparation')
        r['pack_record_sha256_before'] = sha256_file(a.pack_record)
        data = torch.load(a.packed, map_location='cpu', weights_only=True)
        packed_cpu = PackedGateUp(data['words'], data['scales'], data['metadata'])
        if (_digest(packed_cpu.words) != packed_cpu.metadata['words_sha256'] or
                _digest(packed_cpu.scales) != packed_cpu.metadata['scales_sha256']):
            raise RuntimeError('packed tensor hash mismatch')
        r['format_metadata'] = packed_cpu.metadata
        packed = packed_cpu.to('cuda')
        quantized_weight = dequantize_gate_up_reference(packed)
        if not torch.isfinite(quantized_weight).all().item():
            raise RuntimeError('nonfinite quantized reference')
        policy = FrozenMoEPolicy.load(ROOT / 'results/m7-policy.json')
        policy.verify_provenance(ROOT)
        source_names = set(dict(policy.source_hashes)) | {
            'src/llm_gpu_kernels/cuda_w4_gate_up.cu', 'src/llm_gpu_kernels/moe_w4_cuda.py',
            'src/llm_gpu_kernels/moe_w4_audit.py', 'src/llm_gpu_kernels/moe_weight_quant.py',
            'src/llm_gpu_kernels/moe_cuda.py', 'docs/M22_W4_IMPLEMENTATION_AUDIT_PLAN.md',
            'src/llm_gpu_kernels/moe_real_trace.py', 'src/llm_gpu_kernels/moe_full_holdout.py',
            'src/llm_gpu_kernels/reference.py', 'src/llm_gpu_kernels/moe_benchmark.py',
            'src/llm_gpu_kernels/mature_moe_benchmark.py', 'src/llm_gpu_kernels/moe_profile.py'}
        r['source_hashes_before'] = {n: sha256_file(ROOT / n) for n in sorted(source_names)}
        module = load_w4_gate_up()
        binary_path = Path(module.__file__)
        r['binary'] = dict(name=binary_path.name, sha256_before=sha256_file(binary_path))
        capture_file = a.capture_root / 'capture-summary.json'
        capture = json.loads(capture_file.read_text())
        specifications = sorted((s for s in capture['cases'] if s['tokens'] in (17, 193, 1024)),
                                key=lambda s: (s['prompt'], s['tokens']))
        if capture['status'] != 'passed' or len(specifications) != 6:
            raise RuntimeError('expected six passed real inputs')
        for spec in specifications:
            manifest_path = (a.capture_root / spec['manifest']).resolve()
            manifest = json.loads(manifest_path.read_text())
            paths = [capture_file, manifest_path, *((manifest_path.parent / ref['path']).resolve()
                     for ref in (manifest['trace'], *manifest['weights'].values()))]
            before = {str(p.relative_to(a.capture_root)): sha256_file(p) for p in sorted(set(paths))}
            if before.get('weights.pt') != pack_record['input']['manifest_weight_sha256']:
                raise RuntimeError('source checkpoint differs from format preparation')
            case = validate_real_moe_trace(manifest_path, load_weights=True)
            if _digest(case.weights['gate_up']) != packed_cpu.metadata['source_sha256']:
                raise RuntimeError('packed weights do not match pinned BF16 source')
            x = case.tensors['x'].cuda().contiguous()
            ids = case.tensors['expert_ids'].cuda().contiguous()
            rw = case.tensors['route_weights'].cuda().contiguous()
            down = case.weights['down'].cuda().contiguous()
            original_gate = case.weights['gate_up'].cuda().contiguous()
            originals = (x.clone(), ids.clone(), rw.clone())
            inputs = (x, quantized_weight, down, ids, rw)
            input_sha = tensor_hash(x, ids, rw)
            config, bucket = policy.select(x.shape[0], 128, 8, 2048, 768)
            row = dict(case=f"{spec['prompt']}-T{spec['tokens']}", selected_bucket=bucket,
                       input_sha256_before=input_sha, artifact_hashes_before=before,
                       audits=[], status='running')
            r['cases'].append(row)
            write_record(a.output, r)
            backend = prepare_w4_moe(x, packed, down, ids, rw, down_config=config)
            if tuple(launch.name for launch in backend.launches) != (
                    'reset', 'rank_histogram', 'prefix', 'rank_dispatch', 'gate_up_swiglu', 'down', 'combine'):
                raise RuntimeError('unexpected W4 launch DAG')
            def check(label, replay=None):
                reference = moe_reference(*inputs)
                (backend.run if replay is None else replay.replay)()
                torch.cuda.synchronize()
                numerical = safe_error(backend.base.output, reference)
                dispatch = _ranked_state(backend, inputs)
                dispatch['ranked_dispatch'].pop('rank_values_by_expert', None)
                result = dict(state=label, numerical=numerical, dispatch=dispatch)
                row['audits'].append(result)
                write_record(a.output, r)
                if not numerical['passed'] or not dispatch['passed']:
                    raise RuntimeError(f"{row['case']} / {label} failed")
                if label == 'original-api':
                    bf16_reference = moe_reference(x, original_gate, down, ids, rw)
                    row['format_forward_change'] = safe_error(reference, bf16_reference)
                    row['cuda_vs_original_bf16'] = safe_error(backend.base.output, bf16_reference)
                    row['format_forward_change'].pop('passed', None)
                    row['cuda_vs_original_bf16'].pop('passed', None)
                    row['format_forward_change_interpretation'] = 'proxy differences, not implementation pass/fail or task scores'
            check('original-api')
            graph = _capture_graph(backend.run, 8, x.device)
            check('original-graph', graph)
            x.copy_(originals[0] * -0.75); ids.fill_(0); rw.copy_(originals[2] * 0.5)
            check('duplicate-expert0-graph', graph)
            ids.fill_(127)
            check('duplicate-expert127-graph', graph)
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                x.copy_(originals[0] * 0.5); ids.copy_(originals[1]); rw.copy_(originals[2] * -0.5)
                backend.run()
            torch.cuda.synchronize()
            reference = moe_reference(*inputs)
            numerical = safe_error(backend.base.output, reference)
            dispatch = _ranked_state(backend, inputs)
            dispatch['ranked_dispatch'].pop('rank_values_by_expert', None)
            row['audits'].append(dict(state='nondefault-stream', numerical=numerical, dispatch=dispatch))
            if not numerical['passed'] or not dispatch['passed']:
                raise RuntimeError('nondefault-stream gate failed')
            for target, original in zip((x, ids, rw), originals):
                target.copy_(original)
            check('restored-api'); check('restored-graph', graph)
            row['input_sha256_after'] = tensor_hash(x, ids, rw)
            row['artifact_hashes_after'] = {str(p.relative_to(a.capture_root)): sha256_file(p)
                                          for p in sorted(set(paths))}
            if row['input_sha256_after'] != input_sha or row['artifact_hashes_after'] != before:
                raise RuntimeError('input/capture changed')
            row['status'] = 'passed'
            if r.get('zero_tokens') is None:
                empty = prepare_w4_moe(x[:0], packed, down, ids[:0], rw[:0])
                output = empty.run()
                if output.shape != (0, 2048) or output.dtype != torch.bfloat16 or empty.launches:
                    raise RuntimeError('T0 contract failed')
                r['zero_tokens'] = dict(passed=True, launch_count=0, dtype='bfloat16', shape=[0, 2048])
                del empty, output
            write_record(a.output, r)
            del graph, backend, inputs, originals, case, reference
            del x, ids, rw, down, original_gate
            gc.collect()
        r['source_hashes_after'] = {n: sha256_file(ROOT / n) for n in sorted(source_names)}
        r['binary']['sha256_after'] = sha256_file(binary_path)
        r['packed_sha256_after'] = sha256_file(a.packed)
        r['pack_record_sha256_after'] = sha256_file(a.pack_record)
        r['packed_gpu_digest_after'] = dict(words=_digest(packed.words), scales=_digest(packed.scales))
        policy.verify_provenance(ROOT)
        if (r['source_hashes_before'] != r['source_hashes_after'] or
                r['binary']['sha256_before'] != r['binary']['sha256_after'] or
                r['packed_sha256_before'] != r['packed_sha256_after'] or
                r['pack_record_sha256_before'] != r['pack_record_sha256_after'] or
                r['packed_gpu_digest_after']['words'] != packed_cpu.metadata['words_sha256'] or
                r['packed_gpu_digest_after']['scales'] != packed_cpu.metadata['scales_sha256']):
            raise RuntimeError('source/binary/packed inputs changed')
        r['status'] = 'passed'
    except Exception as exc:
        r.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_record(a.output, r)


if __name__ == '__main__':
    run()
