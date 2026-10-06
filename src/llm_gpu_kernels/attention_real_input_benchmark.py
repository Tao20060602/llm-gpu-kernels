"""M17 replay of captured model inputs with original current V stride."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from .attention_baseline_benchmark import (
    _graph, _flat_tensors, _tensor_hash, _telemetry, _runtime_info,
)
from .attention_segmented_benchmark import _check, _chunked_fp32_reference, _resources
from .attention_resource_policy import FrozenResourcePolicy
from .attention_resource_tuning import SOURCE_FILES as FROZEN_SOURCES
from .attention_unified_tuning import measure, failure_fields
from .attention_raw_boundary import RawPackAttention, FrozenRawBoundary
from .benchmark import compile_artifacts

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (*FROZEN_SOURCES, 'docs/M17_REAL_INPUT_BRIDGE_PLAN.md',
                'src/llm_gpu_kernels/attention_raw_boundary.py',
                'src/llm_gpu_kernels/attention_real_input_benchmark.py')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def restore_stride(tensor, stride):
    tensor = tensor.to('cuda')
    if tuple(tensor.stride()) == tuple(stride):
        return tensor
    raw = torch.empty_strided(tensor.shape, tuple(stride), dtype=tensor.dtype, device=tensor.device)
    raw.copy_(tensor)
    return raw


def performance(capture, policy, run_id, progress, save):
    path = Path(capture['path'])
    if file_hash(path) != capture['sha256']:
        raise RuntimeError('capture file SHA256 mismatch')
    if not capture['post_state_checked']:
        raise RuntimeError('capture state audit incomplete')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    stride = metadata['original_strides']
    q = restore_stride(payload['q'], stride['q'])
    segments = [(k.to('cuda'),v.to('cuda')) for k,v in payload['segments']]
    segments[3] = tuple(restore_stride(t, stride[n]) for n,t in zip(('k','v'),payload['segments'][3]))
    segments = tuple(segments)
    flat = _flat_tensors(q, segments)
    input_hash = _tensor_hash(*flat)
    if tuple(k.shape[0] for k,_ in segments) != tuple(metadata['lengths']):
        raise RuntimeError('capture metadata lengths differ from tensors')
    oracle = _chunked_fp32_reference(q, segments)
    progress.update(metadata=metadata, source_capture_sha256=capture['sha256'],
                    input_sha256=input_hash,
                    original_fa2_vs_fp32=_check(payload['original_output'].to('cuda'),oracle,'captured FA2'))
    save()
    plans = {}
    prep = {}
    constructors = {
        'legacy_raw_pack': lambda:RawPackAttention(q,segments,reusable=False),
        'reusable_raw_pack': lambda:RawPackAttention(q,segments,reusable=True),
        'm16_boundary': lambda:FrozenRawBoundary(q,segments,policy),
        'm16_contiguous_core': lambda:policy.prepare(q.contiguous(),tuple((k.contiguous(),v.contiguous()) for k,v in segments)),
    }
    for name, create in constructors.items():
        torch.cuda.synchronize(); start = time.perf_counter()
        plans[name] = create(); torch.cuda.synchronize()
        prep[name] = (time.perf_counter()-start)*1000
    functions = {n:p.run for n,p in plans.items()}
    initial = {n:_check(fn(),oracle,n+' initial') for n,fn in functions.items()}
    captures = {n:_graph(fn,q.device) for n,fn in functions.items()}
    graph_initial = {n:_check(out,oracle,n+' initial Graph') for n,(_,out) in captures.items()}
    api = measure(functions,divisor=1)
    graph = measure({n:g.replay for n,(g,_) in captures.items()})
    post = {n:_check(fn(),oracle,n+' post API') for n,fn in functions.items()}
    graph_post = {}
    for n,(g,out) in captures.items():
        g.replay();graph_post[n] = _check(out,oracle,n+' post Graph')
    if input_hash != _tensor_hash(*flat):
        raise RuntimeError('replay inputs changed')
    boundary = plans['m16_boundary']
    evidence = {}
    if boundary.mode == 'direct':
        evidence = compile_artifacts(boundary.plan.kernel,path.stem,run_id)
        evidence['spills'] = boundary.plan.kernel.n_spills
    return {
        'status':'passed','passed':True,'selected_mode':boundary.mode,
        'initial_vs_fp32':initial,'initial_graph_vs_fp32':graph_initial,
        'post_api_vs_fp32':post,'post_graph_vs_fp32':graph_post,'api':api,'graph':graph,
        'prepare_synchronized_wall_ms':prep,
        'prepare_scope':'this process with existing disk/JIT caches; not full cold startup',
        'conversion_workspace_bytes':boundary.conversion_workspace_bytes,
        'conversion_count_per_forward':len(boundary.conversions),
        'strong_packed_workspace_bytes':plans['reusable_raw_pack'].packed_workspace_bytes,
        'selected_resources':_resources(boundary.plan) if boundary.mode=='direct' else None,
        'compile_evidence':evidence,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_scope':'all inputs/arms/oracle/Graph pools; not per-operator workspace',
    }


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    gpu = torch.cuda.get_device_properties(0)
    if (gpu.major,gpu.minor)!=(8,6):
        raise RuntimeError('M17 SM86 only')
    manifest = json.loads(args.manifest.read_text())
    if manifest['status']!='passed' or manifest['source_hashes']!=manifest['source_hashes_after']:
        raise RuntimeError('capture manifest is not complete/audited')
    policy = FrozenResourcePolicy.load(ROOT/'results/m16-resource-policy.json')
    provenance = policy.verify_provenance(ROOT)
    run_id = 'm17-real-input-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+str(time.time_ns()%10**9)
    path = ROOT/'results'/(run_id+'.json')
    hashes = {n:file_hash(ROOT/n) for n in SOURCE_FILES}
    record = {
        'schema':1,'run_id':run_id,'status':'running','source_hashes':hashes,
        'capture_manifest_path':str(args.manifest.resolve()),
        'capture_manifest_sha256':file_hash(args.manifest),
        'capture_source_hashes':manifest['source_hashes'],
        'model_revision':manifest['model_revision'],'capture_config':manifest['config'],
        'policy_sha256':policy.sha256,'provenance_before':provenance,
        'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':gpu.name,'fa2':_runtime_info()},
        'limits':{'relative_rms':.008,'normalized_max':.008},
        'protocol':'four-arm 4 forward/reverse rounds,8 samples,2 calls;Graph8forward; per-forward units',
        'scope':'model tensor GPU-resident replay; excludes CPU gather/H2D/model/request; actual current V stride restored',
        'cases':[],'telemetry_before':_telemetry(),
    }

    def save():
        path.write_text(json.dumps(record,indent=2)+'\n')

    save()
    try:
        for capture in manifest['captures']:
            torch.cuda.reset_peak_memory_stats()
            row = {'capture_path':capture['path'],'status':'running','telemetry_before':_telemetry()}
            record['cases'].append(row);save()
            print('M17 replay',Path(capture['path']).name,flush=True)
            try:
                row.update(performance(capture,policy,run_id,row,save))
            except Exception as exc:
                row.update(failure_fields(exc));save();raise
            row['telemetry_after'] = _telemetry();save();gc.collect();torch.cuda.empty_cache()
        after = {n:file_hash(ROOT/n) for n in SOURCE_FILES}
        if hashes!=after:
            raise RuntimeError('replay sources changed during run')
        if file_hash(args.manifest)!=record['capture_manifest_sha256']:
            raise RuntimeError('capture manifest changed during run')
        record.update(status='passed',source_hashes_after=after,
                      provenance_after=policy.verify_provenance(ROOT))
    except Exception as exc:
        record.update(failure_fields(exc));raise
    finally:
        record['telemetry_after'] = _telemetry();save();print('M17 result',path,flush=True)


if __name__=='__main__':
    run()
