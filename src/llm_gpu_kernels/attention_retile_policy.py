"""Immutable M14 metadata policy for the fixed finite tile experiment."""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
from .attention_segmented import AttentionConfig,prepare_segmented_kernel
from .attention_baseline import prepare_segmented_attention
from .attention_unified_policy import canonical_training_hash

CANDIDATES=(AttentionConfig(stages=1),AttentionConfig(warps=8,stages=1),
 AttentionConfig(token_tile=64,stages=1),AttentionConfig(token_tile=64,warps=8,stages=1),
 AttentionConfig(token_tile=64,warps=8),AttentionConfig(token_tile=64,head_group=2,warps=8,stages=1))
BUCKETS=('small','medium','large')
DOMAIN={'min_T':32,'max_T':256,'sink':64,'recent':512,'max_history':4096}

def bucket_for(tokens):return 'small' if tokens<=64 else 'medium' if tokens<=128 else 'large'

@dataclass(frozen=True)
class FrozenRetilePolicy:
    payload:bytes
    sha256:str
    @property
    def data(self):return json.loads(self.payload)
    @classmethod
    def load(cls,path):
        payload=Path(path).read_bytes();data=json.loads(payload)
        if data.get('schema')!=1 or data.get('status')!='frozen_before_holdout' or data.get('domain')!=DOMAIN:
            raise ValueError('expected M14 frozen metadata policy')
        if set(data['choices'])!=set(BUCKETS):raise ValueError('wrong buckets')
        for choice in data['choices'].values():
            if choice['mode']=='direct':
                if AttentionConfig(**choice['config']) not in CANDIDATES:raise ValueError('unknown candidate')
            elif choice['mode']=='fa2':
                if choice['config'] is not None:raise ValueError('fallback config must be None')
            else:raise ValueError('unknown mode')
        return cls(payload,hashlib.sha256(payload).hexdigest())
    def verify_provenance(self,root):
        root=Path(root);data=self.data
        for name,expected in data['source_hashes'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('frozen source changed: '+name)
        record=json.loads((root/data['training_record_path']).read_text())
        if record['status']!='passed' or canonical_training_hash(record)!=data['training_record_sha256'] or record['policy_sha256']!=self.sha256:
            raise RuntimeError('M14 frozen training/policy mismatch')
        return {'passed':True,'source_files_verified':len(data['source_hashes']),'policy_sha256':self.sha256}
    def select(self,tokens,lengths):
        inside=(32<=tokens<=256 and len(lengths)==4 and tuple(lengths[1:3])==(64,512)
                and lengths[3]==tokens and sum(lengths[:3])<=4096)
        if not inside:return 'fa2',None,None
        bucket=bucket_for(tokens);choice=self.data['choices'][bucket]
        return choice['mode'],AttentionConfig(**choice['config']) if choice['config'] else None,bucket
    def prepare(self,q,segments):
        mode,config,_=self.select(q.shape[0],tuple(k.shape[0] for k,_ in segments))
        return prepare_segmented_kernel(q,segments,config) if mode=='direct' else prepare_segmented_attention(q,segments)
