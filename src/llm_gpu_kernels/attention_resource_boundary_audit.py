"""Post-freeze M16 history-domain edge audit; no policy edits or selection."""
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from .attention_resource_policy import FrozenResourcePolicy
from .attention_resource_holdout import performance
from .attention_baseline_benchmark import _runtime_info, _telemetry
from .attention_unified_tuning import failure_fields

ROOT = Path(__file__).resolve().parents[2]
AUDIT_SOURCES = ('docs/M16_BOUNDARY_AUDIT_PLAN.md',
                 'src/llm_gpu_kernels/attention_resource_boundary_audit.py')


def run():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    gpu = torch.cuda.get_device_properties(0)
    if (gpu.major, gpu.minor) != (8, 6):
        raise RuntimeError('M16 SM86 only')
    policy = FrozenResourcePolicy.load(ROOT/'results/m16-resource-policy.json')
    provenance = policy.verify_provenance(ROOT)
    run_id = 'm16-boundary-audit-'+time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'-'+str(time.time_ns()%10**9)
    path = ROOT/'results'/(run_id+'.json')
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in AUDIT_SOURCES}
    record = {
        'schema': 1, 'run_id': run_id, 'status': 'running', 'seed': 14307,
        'policy_sha256': policy.sha256, 'policy': policy.data,
        'source_hashes': policy.data['source_hashes'], 'audit_source_hashes': hashes,
        'provenance_before': provenance,
        'runtime': {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                    'gpu': gpu.name, 'fa2': _runtime_info()},
        'limits': {'relative_rms': .008, 'normalized_max': .008},
        'protocol': 'M16 same four-arm API/Graph protocol; post-freeze audit, no policy edits',
        'scope': 'GPU resident BF16 four-segment attention, no model/NanoKV/serving',
        'cases': [], 'telemetry_before': _telemetry(),
    }

    def save():
        path.write_text(json.dumps(record, indent=2)+'\n')

    save()
    try:
        specs = [(t, s) for t in (32, 48, 64, 96, 127, 128)
                 for s in (3073, 3519, 3520, 3521)]+[(64, 8193)]
        for tokens, selected in specs:
            row = {'T': tokens, 'selected_history': selected, 'status': 'running'}
            record['cases'].append(row); save()
            print(f'M16 edge audit T={tokens} selected={selected}', flush=True)
            try:
                row.update(performance(tokens, selected, policy, run_id, 14307))
                expected = 'fa2' if selected+576 > 4096 else policy.data['choices']['small']['mode']
                if row['selected_mode'] != expected:
                    raise RuntimeError('history boundary dispatch differs from frozen policy')
            except Exception as exc:
                row.update(failure_fields(exc)); save(); raise
            save(); gc.collect(); torch.cuda.empty_cache()
        after = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in AUDIT_SOURCES}
        if after != hashes:
            raise RuntimeError('audit source changed during run')
        record.update(status='passed', provenance_after=policy.verify_provenance(ROOT),
                      audit_source_hashes_after=after)
    except Exception as exc:
        record.update(failure_fields(exc)); raise
    finally:
        record['telemetry_after'] = _telemetry(); save()
        print('M16 edge audit result:', path.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    run()
