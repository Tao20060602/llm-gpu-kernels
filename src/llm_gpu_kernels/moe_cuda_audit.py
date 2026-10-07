"""M22 predeclared full-forward BF16 audit, no latency/winner selection."""
import argparse
import gc
import json
from pathlib import Path
import traceback

import torch

from .benchmark import ROOT, LIMITS, tensor_hash
from .gemm import GemmConfig
from .mature_moe_benchmark import _capture_graph
from .moe_cuda import prepare_cuda_moe, load_cuda_gate_up, VARIANTS
from .moe_full_holdout import _ranked_state
from .moe_full_policy import FrozenMoEPolicy
from .moe_profile import write_record
from .moe_real_trace import sha256_file, validate_real_moe_trace
from .moe_benchmark import safe_error
from .reference import moe_reference


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    a.capture_root = a.capture_root.resolve()
    if a.output.exists() or a.output.with_suffix(a.output.suffix + '.partial').exists():
        raise RuntimeError('refusing overwrite')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    r = dict(schema=1, status='running', experiment='M22 BF16 full-forward audit',
             limits=LIMITS, variants=list(VARIANTS), cases=[],
             interpretation='correctness only; no performance or model-quality conclusion')
    write_record(a.output, r)
    try:
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if torch.cuda.get_device_capability() != (8, 6):
            raise RuntimeError('SM86 required')
        capture_file = a.capture_root / 'capture-summary.json'
        capture = json.loads(capture_file.read_text())
        if capture['status'] != 'passed':
            raise RuntimeError('input capture did not pass')
        policy = FrozenMoEPolicy.load(ROOT / 'results/m7-policy.json')
        policy.verify_provenance(ROOT)
        source_names = set(dict(policy.source_hashes)) | {
            'src/llm_gpu_kernels/cuda_gate_up.cu', 'src/llm_gpu_kernels/moe_cuda.py',
            'src/llm_gpu_kernels/moe_cuda_audit.py', 'docs/M22_CUDA_AUDIT_PLAN.md',
            'src/llm_gpu_kernels/moe_real_trace.py', 'src/llm_gpu_kernels/moe_full_holdout.py',
            'src/llm_gpu_kernels/reference.py', 'src/llm_gpu_kernels/moe_benchmark.py',
            'src/llm_gpu_kernels/mature_moe_benchmark.py', 'src/llm_gpu_kernels/moe_profile.py'}
        r['source_hashes_before'] = {n: sha256_file(ROOT / n) for n in sorted(source_names)}
        extension = load_cuda_gate_up()
        binary_path = Path(extension.__file__)
        r['binary'] = dict(name=binary_path.name, sha256_before=sha256_file(binary_path))
        specifications = sorted((s for s in capture['cases'] if s['tokens'] in (17, 193, 1024)),
                                key=lambda s: (s['prompt'], s['tokens']))
        if len(specifications) != 6:
            raise RuntimeError('expected six declared real inputs')
        for spec in specifications:
            manifest_path = (a.capture_root / spec['manifest']).resolve()
            manifest = json.loads(manifest_path.read_text())
            paths = [capture_file, manifest_path]
            paths += [(manifest_path.parent / ref['path']).resolve()
                      for ref in (manifest['trace'], *manifest['weights'].values())]
            before = {str(p.relative_to(a.capture_root)): sha256_file(p) for p in sorted(set(paths))}
            case = validate_real_moe_trace(manifest_path, load_weights=True)
            x = case.tensors['x'].cuda().contiguous()
            ids = case.tensors['expert_ids'].cuda().contiguous()
            rw = case.tensors['route_weights'].cuda().contiguous()
            gate, down = (case.weights[n].cuda().contiguous() for n in ('gate_up', 'down'))
            originals = (x.clone(), ids.clone(), rw.clone())
            inputs = (x, gate, down, ids, rw)
            input_sha = tensor_hash(x, ids, rw)
            config, bucket = policy.select(x.shape[0], 128, 8, 2048, 768)
            if config.block_m != 32:
                raise RuntimeError('declared audit domain requires BM32')
            row = dict(case=f"{spec['prompt']}-T{spec['tokens']}", selected_bucket=bucket,
                       input_sha256_before=input_sha, artifact_hashes_before=before,
                       audits=[], status='running')
            r['cases'].append(row)
            write_record(a.output, r)
            backends = {name: prepare_cuda_moe(*inputs, variant=name, down_config=config) for name in VARIANTS}
            def check(label, *, replay=None):
                reference = moe_reference(*inputs)
                result = dict(state=label, variants={})
                row['audits'].append(result)
                for name, backend in backends.items():
                    (replay[name].replay if replay is not None else backend.run)()
                    torch.cuda.synchronize()
                    numerical = safe_error(backend.base.output, reference)
                    dispatch = _ranked_state(backend, inputs)
                    dispatch.get('ranked_dispatch', {}).pop('rank_values_by_expert', None)
                    result['variants'][name] = dict(numerical=numerical, dispatch=dispatch)
                    write_record(a.output, r)
                    if not numerical['passed'] or not dispatch['passed']:
                        raise RuntimeError(f"{row['case']} / {label} / {name} failed")
            check('original-api')
            graphs = {name: _capture_graph(b.run, 8, x.device) for name, b in backends.items()}
            check('original-graph', replay=graphs)
            x.copy_(originals[0] * -0.75); ids.fill_(0); rw.copy_(originals[2] * 0.5)
            check('duplicate-expert0-graph', replay=graphs)
            ids.fill_(127)
            check('duplicate-expert127-graph', replay=graphs)
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                x.copy_(originals[0] * 0.5); ids.copy_(originals[1]); rw.copy_(originals[2] * -0.5)
                for backend in backends.values():
                    backend.run()
            torch.cuda.synchronize()
            reference = moe_reference(*inputs)
            nondefault = dict(state='nondefault-stream', variants={})
            row['audits'].append(nondefault)
            for name, backend in backends.items():
                numerical = safe_error(backend.base.output, reference)
                dispatch = _ranked_state(backend, inputs)
                dispatch.get('ranked_dispatch', {}).pop('rank_values_by_expert', None)
                nondefault['variants'][name] = dict(numerical=numerical, dispatch=dispatch)
                if not numerical['passed'] or not dispatch['passed']:
                    raise RuntimeError('nondefault-stream gate failed')
            for target, original in zip((x, ids, rw), originals):
                target.copy_(original)
            check('restored-api')
            check('restored-graph', replay=graphs)
            row['input_sha256_after'] = tensor_hash(x, ids, rw)
            row['artifact_hashes_after'] = {str(p.relative_to(a.capture_root)): sha256_file(p)
                                          for p in sorted(set(paths))}
            if row['input_sha256_after'] != input_sha or row['artifact_hashes_after'] != before:
                raise RuntimeError('input/capture artifacts changed')
            row['status'] = 'passed'
            if r.get('zero_tokens') is None:
                empty = prepare_cuda_moe(x[:0], gate, down, ids[:0], rw[:0])
                output = empty.run()
                if output.shape != (0, 2048) or output.dtype != torch.bfloat16 or empty.launches:
                    raise RuntimeError('T0 contract failed')
                r['zero_tokens'] = dict(passed=True, launch_count=0, dtype='bfloat16', shape=[0, 2048])
                del empty, output
            write_record(a.output, r)
            del graphs, backends, inputs, originals, case, reference
            del x, ids, rw, gate, down, backend
            gc.collect()
        r['source_hashes_after'] = {n: sha256_file(ROOT / n) for n in sorted(source_names)}
        policy.verify_provenance(ROOT)
        if r['source_hashes_before'] != r['source_hashes_after']:
            raise RuntimeError('audit source changed')
        r['binary']['sha256_after'] = sha256_file(binary_path)
        if r['binary']['sha256_after'] != r['binary']['sha256_before']:
            raise RuntimeError('loaded extension file changed')
        r['status'] = 'passed'
    except Exception as exc:
        r.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_record(a.output, r)


if __name__ == '__main__':
    run()
