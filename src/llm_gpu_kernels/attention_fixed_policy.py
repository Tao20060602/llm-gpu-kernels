"""Immutable fixed-Q64 metadata policy with public FA2 fallback."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from .attention_segmented import AttentionConfig,prepare_segmented_kernel
from .attention_baseline import prepare_segmented_attention
from .attention_unified_policy import canonical_training_hash

DOMAIN={'T':64,'sink':64,'recent':512,'current':64,'max_history':4096}

@dataclass(frozen=True)
class FrozenFixedPolicy:
    payload:bytes
    sha256:str

    @property
    def data(self):return json.loads(self.payload)

    @classmethod
    def load(cls,path):
        payload=Path(path).read_bytes();data=json.loads(payload)
        if data.get('schema')!=1 or data.get('status')!='frozen_before_holdout' or data.get('domain')!=DOMAIN:
            raise ValueError('expected frozen M13 metadata policy')
        if data['mode']=='fixed64':
            if AttentionConfig(**data['config'])!=AttentionConfig():raise ValueError('unexpected fixed candidate')
        elif data['mode']=='fa2':
            if data['config'] is not None:raise ValueError('FA2 fallback carries config')
        else:raise ValueError('unknown mode')
        return cls(payload,hashlib.sha256(payload).hexdigest())

    def verify_provenance(self,root):
        root=Path(root);data=self.data
        for name,digest in data['source_hashes'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('frozen source changed: '+name)
        record=json.loads((root/data['training_record_path']).read_text())
        if record['status']!='passed' or canonical_training_hash(record)!=data['training_record_sha256'] or record['policy_sha256']!=self.sha256:
            raise RuntimeError('frozen training/policy mismatch')
        return {'passed':True,'source_files_verified':len(data['source_hashes']),'policy_sha256':self.sha256}

    def select(self,tokens,lengths):
        inside=(tokens==64 and len(lengths)==4 and tuple(lengths[1:])==(64,512,64) and sum(lengths[:3])<=4096)
        if inside and self.data['mode']=='fixed64':return 'fixed64',AttentionConfig()
        return 'fa2',None

    def prepare(self,q,segments):
        mode,config=self.select(q.shape[0],tuple(k.shape[0] for k,_ in segments))
        return prepare_segmented_kernel(q,segments,config) if mode=='fixed64' else prepare_segmented_attention(q,segments)
