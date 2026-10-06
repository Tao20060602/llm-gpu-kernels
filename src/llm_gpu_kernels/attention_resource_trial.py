"""Full M16 nonempty candidate versus reusable pack+official FA2 trial."""
from dataclasses import asdict
from .attention_nonempty import prepare_nonempty_kernel
from .attention_baseline_benchmark import _graph,_flat_tensors,_tensor_hash
from .attention_segmented_benchmark import _check,_resources
from .attention_unified_tuning import measure

def trial(q,segments,oracle,baseline,config,long=False):
    before=_tensor_hash(*_flat_tensors(q,segments))
    plan=prepare_nonempty_kernel(q,segments,config)
    functions={'fa2_baseline':baseline.run,'resource_candidate':plan.run}
    initial={n:_check(f(),oracle,n+' initial') for n,f in functions.items()}
    captures={n:_graph(f,q.device) for n,f in functions.items()}
    graph_initial={n:_check(out,oracle,n+' initial Graph') for n,(_,out) in captures.items()}
    graph=measure({n:g.replay for n,(g,_) in captures.items()},samples=16 if long else 8,calls=4 if long else 2)
    post={n:_check(f(),oracle,n+' post API') for n,f in functions.items()}
    graph_post={}
    for n,(g,out) in captures.items():g.replay();graph_post[n]=_check(out,oracle,n+' post Graph')
    if before!=_tensor_hash(*_flat_tensors(q,segments)):raise RuntimeError('training inputs modified')
    ratio=graph['summary']['resource_candidate']['event_p50_ms_per_forward']/graph['summary']['fa2_baseline']['event_p50_ms_per_forward']
    return {'status':'passed','passed':True,'input_sha256':before,'config':asdict(config),
            'initial_vs_fp32':initial,'initial_graph_vs_fp32':graph_initial,'post_api_vs_fp32':post,
            'post_graph_vs_fp32':graph_post,'graph':graph,'ratio_over_fa2':ratio,'candidate_resources':_resources(plan),
            'scratch_bytes':0,'owned_output_and_scratch_bytes':plan.workspace_bytes}
