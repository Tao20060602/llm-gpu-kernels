"""M18 dynamic addresses, lifecycle contract and audited M17 input replay."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from .attention_dynamic import AttentionCompileCache, DynamicAttentionWorkspace
from .attention_resource_policy import FrozenResourcePolicy
from .attention_raw_boundary import RawPackAttention, FrozenRawBoundary
from .attention_real_input_benchmark import file_hash, restore_stride, SOURCE_FILES as M17_SOURCES
from .attention_baseline_benchmark import _graph, _make_custom_inputs, _flat_tensors, _tensor_hash, _telemetry, _runtime_info
from .attention_segmented_benchmark import _check, _chunked_fp32_reference
from .attention_unified_tuning import measure, failure_fields

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (*M17_SOURCES, 'docs/M18_DYNAMIC_BINDING_PLAN.md',
           'src/llm_gpu_kernels/attention_dynamic.py',
           'src/llm_gpu_kernels/attention_dynamic_benchmark.py')


def dynamic_graph(q, segments, policy, cache):
    """Independent stream workspace; captures still own fixed input pointers."""
    current = torch.cuda.current_stream(q.device)
    stream = torch.cuda.Stream(device=q.device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        workspace = DynamicAttentionWorkspace(q, segments, policy, cache)
        for _ in range(3):
            workspace.run(q, segments)
    current.wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(8):
            output = workspace.run(q, segments)
    current.wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    return graph, output, workspace, q, segments


def make_inputs(tokens, selected, seed):
    q, segments = _make_custom_inputs(tokens, (selected,64,512,tokens), seed=seed)
    k, v = segments[-1]
    raw_v = torch.empty_strided(v.shape, (6144,128,1), dtype=v.dtype, device=v.device)
    raw_v.copy_(v)
    return q, (*segments[:-1], (k, raw_v))


def functional(policy, save, record):
    cache = AttentionCompileCache(max_entries=2)
    rows = record.setdefault('functional', [])
    for tokens, selected in ((0,0),(31,7),(32,0),(64,257),(96,2048),(128,3521),(129,512)):
        q, segments = make_inputs(tokens,selected,18001+tokens)
        before = _tensor_hash(*_flat_tensors(q,segments))
        workspace = DynamicAttentionWorkspace(q,segments,policy,cache)
        row = dict(tokens=tokens,selected=selected,mode=workspace.mode,checks={})
        rows.append(row);save()
        row['checks']['initial'] = _check(workspace.run(q,segments),_chunked_fp32_reference(q,segments),'initial')
        new_q, new_segments = make_inputs(tokens,selected,19001+tokens)
        if tokens and new_q.data_ptr() == q.data_ptr():
            raise RuntimeError('new address validation reused Q address')
        row['checks']['new_addresses_values'] = _check(workspace.run(new_q,new_segments),
            _chunked_fp32_reference(new_q,new_segments),'new addresses')
        output_ptr = workspace.output.data_ptr()
        row['checks']['restore_original'] = _check(workspace.run(q,segments),
            _chunked_fp32_reference(q,segments),'restore')
        if workspace.output.data_ptr() != output_ptr:
            raise RuntimeError('workspace output changed address')
        if before != _tensor_hash(*_flat_tensors(q,segments)):
            raise RuntimeError('input modified')
        if tokens:
            captured = dynamic_graph(q,segments,policy,cache)
            graph, output = captured[:2]
            for iteration in range(2):
                updated_q, updated_segments = make_inputs(tokens,selected,20001+tokens+iteration)
                for dst, src in zip(_flat_tensors(q,segments),_flat_tensors(updated_q,updated_segments)):
                    dst.copy_(src)
                graph.replay()
                row['checks']['graph_value_update_'+str(iteration)] = _check(output,
                    _chunked_fp32_reference(q,segments),'graph update')
            del captured, graph, output
        # A different stream is explicitly refused by this workspace.
        temporary_q, temporary_segments = make_inputs(tokens,selected,21001+tokens)
        oracle = _chunked_fp32_reference(temporary_q,temporary_segments)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            try:
                workspace.run(q,segments)
            except ValueError as exc:
                row['other_stream_rejected'] = str(exc)
            else:
                raise RuntimeError('cross-stream reuse accepted')
            owner = DynamicAttentionWorkspace(q,segments,policy,cache)
            actual = owner.run(temporary_q,temporary_segments)
            del temporary_q, temporary_segments
        # Pressure on the original allocator stream after asynchronous submission.
        pressure = [torch.empty_like(q) for _ in range(4)]
        torch.cuda.current_stream().wait_stream(side)
        row['checks']['side_stream_temporary_owners'] = _check(actual,oracle,'side stream owners')
        del pressure
        # Shape/stride changes and unsupported metadata must fail before launch.
        for name, bad_q, bad_segments in (
            ('dtype',q.float(),segments),
            ('shape',q[:max(0,tokens-1)],segments) if tokens else ('segments',q,segments[:3]),
        ):
            try:
                workspace.run(bad_q,bad_segments)
            except ValueError:
                row[name+'_rejected'] = True
            else:
                raise RuntimeError(name+' accepted')
        row['passed'] = True;save()
    # Two independent consumers reuse a template but own distinct output buffers.
    q, segments = make_inputs(96,2048,22001)
    a = DynamicAttentionWorkspace(q,segments,policy,cache)
    b = DynamicAttentionWorkspace(q,segments,policy,cache)
    if a.template is not b.template or a.output.data_ptr() == b.output.data_ptr():
        raise RuntimeError('template/workspace separation failed')
    saved_template = a.template
    cache.clear()
    if cache.entries:
        raise RuntimeError('clear failed')
    metric = _check(a.run(q,segments),_chunked_fp32_reference(q,segments),'live workspace after clear')
    if a.template is not saved_template:
        raise RuntimeError('live template changed')
    record['cache_contract'] = dict(stats_after_clear=cache.stats(),live_workspace_after_clear=metric,
                                    shared_template=True,distinct_outputs=True)
    if cache.evictions == 0:
        raise RuntimeError('LRU eviction not exercised')
    save()


def performance(capture, policy, cache, row, save):
    if file_hash(capture['path']) != capture['sha256'] or not capture['post_state_checked']:
        raise RuntimeError('capture file/state mismatch')
    payload = torch.load(capture['path'],map_location='cpu',weights_only=True)
    stride = payload['metadata']['original_strides']
    q = restore_stride(payload['q'],stride['q'])
    segments = [(k.to('cuda'),v.to('cuda')) for k,v in payload['segments']]
    segments[3] = tuple(restore_stride(t,stride[n]) for n,t in zip(('k','v'),payload['segments'][3]))
    segments = tuple(segments)
    digest = _tensor_hash(*_flat_tensors(q,segments))
    oracle = _chunked_fp32_reference(q,segments)
    row.update(metadata=payload['metadata'],input_sha256=digest,source_capture_sha256=capture['sha256'])
    save()
    prep = {}
    plans = {}
    constructors = {
        'cached_dynamic':lambda:DynamicAttentionWorkspace(q,segments,policy,cache),
        'm17_fixed':lambda:FrozenRawBoundary(q,segments,policy),
        'strong_pack_fa2':lambda:RawPackAttention(q,segments,reusable=True),
    }
    for name, create in constructors.items():
        torch.cuda.synchronize();start = time.perf_counter()
        plans[name] = create();torch.cuda.synchronize()
        prep[name] = (time.perf_counter()-start)*1000
    # This arm deliberately includes the previous constructor + warm execution,
    # conversion allocation and run on EVERY eager invocation. Not graphed.
    def reprepare():
        return FrozenRawBoundary(q,segments,policy).run()
    functions = {'cached_dynamic':lambda:plans['cached_dynamic'].run(q,segments),
                 'm17_fixed':plans['m17_fixed'].run,
                 'strong_pack_fa2':plans['strong_pack_fa2'].run,
                 'm17_reprepare_each_call':reprepare}
    initial = {n:_check(fn(),oracle,n+' initial') for n,fn in functions.items()}
    captured_dynamic = dynamic_graph(q,segments,policy,cache)
    captures = {'cached_dynamic':captured_dynamic[:2],
                **{n:_graph(functions[n],q.device) for n in ('m17_fixed','strong_pack_fa2')}}
    graph_initial = {n:_check(out,oracle,n+' Graph initial') for n,(_,out) in captures.items()}
    api = measure(functions,divisor=1)
    graph = measure({n:g.replay for n,(g,_) in captures.items()})
    post = {n:_check(fn(),oracle,n+' post') for n,fn in functions.items()}
    graph_post = {}
    for n,(g,out) in captures.items():
        g.replay();graph_post[n] = _check(out,oracle,n+' Graph post')
    # Separate live owners/new pointer validation for every captured model tensor.
    nq = restore_stride(payload['q'],stride['q'])
    ns = [(k.to('cuda'),v.to('cuda')) for k,v in payload['segments']]
    ns[3] = tuple(restore_stride(t,stride[n]) for n,t in zip(('k','v'),payload['segments'][3]))
    ns = tuple(ns)
    if nq.data_ptr() == q.data_ptr():
        raise RuntimeError('real new pointer not exercised')
    rebound = _check(plans['cached_dynamic'].run(nq,ns),oracle,'real new pointers')
    if digest != _tensor_hash(*_flat_tensors(q,segments)):
        raise RuntimeError('input modified')
    row.update(status='passed',passed=True,mode=plans['cached_dynamic'].mode,
               prepare_synchronized_wall_ms=prep,cache_stats=cache.stats(),
               conversion_workspace_bytes=plans['cached_dynamic'].conversion_workspace_bytes,
               initial_vs_fp32=initial,graph_initial_vs_fp32=graph_initial,
               post_vs_fp32=post,graph_post_vs_fp32=graph_post,new_addresses_vs_fp32=rebound,
               api=api,graph=graph)
    save()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    policy = FrozenResourcePolicy.load(ROOT/'results/m16-resource-policy.json')
    manifest = json.loads(args.manifest.read_text())
    if manifest['status'] != 'passed' or manifest['source_hashes'] != manifest['source_hashes_after']:
        raise RuntimeError('incomplete capture manifest')
    run_id = 'm18-dynamic-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+str(time.time_ns()%10**9)
    path = ROOT/'results'/(run_id+'.json')
    hashes = {n:file_hash(ROOT/n) for n in SOURCES}
    record = dict(schema=1,status='running',run_id=run_id,source_hashes=hashes,
                  manifest_path=str(args.manifest.resolve()),manifest_sha256=file_hash(args.manifest),
                  provenance_before=policy.verify_provenance(ROOT),policy_sha256=policy.sha256,
                  runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,fa2=_runtime_info()),
                  limits=dict(relative_rms=.008,normalized_max=.008),cases=[],telemetry_before=_telemetry(),
                  scope='GPU-resident input replay; excludes CPU gather/H2D/model/request',
                  prepare_scope='existing disk/JIT caches, sequential constructors; not full cold startup',
                  graph_scope='fixed addresses, independent stream-owned workspace; no Python rebinding at replay')
    def save():
        path.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        # Isolate compilation first: no model attention launch and no input owners.
        cache = AttentionCompileCache()
        mode,config,_ = policy.select(96,(2048,64,512,96))
        assert mode == 'direct'
        torch.cuda.synchronize();start=time.perf_counter()
        template = cache.get(torch.device('cuda:0'),96,(2048,64,512,96),config)
        torch.cuda.synchronize()
        record['compile_without_launch_synchronized_ms']=(time.perf_counter()-start)*1000
        if cache.get(torch.device('cuda:0'),96,(2048,64,512,96),config) is not template:
            raise RuntimeError('cache hit failed')
        record['prewarm_cache_stats']=cache.stats();save()
        print('M18 functional',flush=True)
        functional(policy,save,record)
        gc.collect();torch.cuda.empty_cache()
        for capture in manifest['captures']:
            print('M18 real replay',Path(capture['path']).name,flush=True)
            row = dict(capture_path=capture['path'],status='running',telemetry_before=_telemetry())
            record['cases'].append(row);save()
            try:
                performance(capture,policy,cache,row,save)
            except Exception as exc:
                row.update(failure_fields(exc));save();raise
            row['telemetry_after']=_telemetry();save();gc.collect();torch.cuda.empty_cache()
        after={n:file_hash(ROOT/n) for n in SOURCES}
        if hashes != after or file_hash(args.manifest) != record['manifest_sha256']:
            raise RuntimeError('source/manifest changed')
        record.update(status='passed',source_hashes_after=after,
                      provenance_after=policy.verify_provenance(ROOT),cache_stats=cache.stats())
    except Exception as exc:
        record.update(failure_fields(exc));raise
    finally:
        record['telemetry_after']=_telemetry();save();print('M18 result',path,flush=True)


if __name__ == '__main__':
    run()
