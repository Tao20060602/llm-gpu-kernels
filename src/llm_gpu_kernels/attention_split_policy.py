"""Immutable M11 policy; strong FA2 fallback outside the short-query domain."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from .attention_split import SplitAttentionConfig,prepare_split_kernel
from .attention_baseline import prepare_segmented_attention
from .attention_unified_policy import canonical_training_hash

CANDIDATES=tuple(SplitAttentionConfig(token_tile=t,splits=s) for t in (16,32) for s in (1,2,4,8))

@dataclass(frozen=True)
class FrozenSplitPolicy:
    payload:bytes
    sha256:str

    @property
    def data(self):return json.loads(self.payload)

    @classmethod
    def load(cls,path):
        payload=Path(path).read_bytes();data=json.loads(payload)
        if data.get('schema')!=1 or data.get('status')!='frozen_before_holdout' or data.get('max_T')!=256:
            raise ValueError('expected frozen short-query M11 policy')
        if data['mode']=='split':
            if SplitAttentionConfig(**data['config']) not in CANDIDATES:raise ValueError('unknown split candidate')
        elif data['mode']=='fa2':
            if data['config'] is not None:raise ValueError('FA2 fallback must not carry split config')
        else:raise ValueError('unknown policy mode')
        return cls(payload,hashlib.sha256(payload).hexdigest())

    def verify_provenance(self,root):
        root=Path(root);data=self.data
        for name,digest in data['source_hashes'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('frozen source changed: '+name)
        record=json.loads((root/data['training_record_path']).read_text())
        if record['status']!='passed' or canonical_training_hash(record)!=data['training_record_sha256'] or record['policy_sha256']!=self.sha256:
            raise RuntimeError('frozen M11 training/policy mismatch')
        return {'passed':True,'source_files_verified':len(data['source_hashes']),'policy_sha256':self.sha256}

    def select(self,tokens):
        data=self.data
        if tokens>data['max_T'] or data['mode']=='fa2':return 'fa2',None
        return 'split',SplitAttentionConfig(**data['config'])

    def prepare(self,q,segments):
        mode,config=self.select(q.shape[0])
        return prepare_split_kernel(q,segments,config) if mode=='split' else prepare_segmented_attention(q,segments)
