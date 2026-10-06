"""Frozen M10 shape-only policy, bound to complete training/source provenance."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .attention_segmented import AttentionConfig

DEFAULT = AttentionConfig()
CANDIDATES = tuple(AttentionConfig(token_tile=t, head_group=g, warps=w)
                   for t in (16, 32) for g in (1, 2, 4) for w in (4, 8))
BUCKETS = ('small', 'medium', 'large')
DOMAIN = {'q_heads': 32, 'kv_heads': 8, 'head_dim': 128, 'segments': 4}


def canonical_training_hash(record):
    copied = dict(record)
    copied.pop('policy_sha256', None)
    return hashlib.sha256(json.dumps(copied, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def bucket_for(tokens):
    return 'small' if tokens <= 128 else 'medium' if tokens <= 512 else 'large'


@dataclass(frozen=True)
class FrozenAttentionPolicy:
    payload: bytes
    sha256: str

    @property
    def data(self):
        # Return a fresh metadata object; callers cannot mutate frozen choices.
        return json.loads(self.payload)

    @classmethod
    def load(cls, path):
        payload = Path(path).read_bytes()
        data = json.loads(payload)
        if data.get('schema') != 1 or data.get('status') != 'frozen_before_holdout':
            raise ValueError('expected frozen M10 policy')
        if data.get('domain') != DOMAIN or data.get('thresholds_T') != [128, 512]:
            raise ValueError('unexpected attention policy domain/thresholds')
        if set(data.get('choices', {})) != set(BUCKETS):
            raise ValueError('expected three policy buckets')
        for choice in data['choices'].values():
            config = AttentionConfig(**choice['config'])
            if choice['mode'] == 'm9':
                if config != DEFAULT:
                    raise ValueError('fallback must be original M9 G1')
            elif choice['mode'] == 'unified':
                if config not in CANDIDATES:
                    raise ValueError('unknown candidate config')
            else:
                raise ValueError('unknown policy mode')
        if not data.get('source_hashes') or data.get('training_hash_scope') != 'canonical_json_without_top_level_policy_sha256':
            raise ValueError('missing source/training provenance')
        return cls(payload, hashlib.sha256(payload).hexdigest())

    def verify_provenance(self, root):
        root = Path(root)
        mismatches = [name for name, digest in self.data['source_hashes'].items()
                      if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
        if mismatches:
            raise RuntimeError(f'frozen sources changed: {mismatches}')
        record = json.loads((root / self.data['training_record_path']).read_text())
        if record['status'] != 'passed' or canonical_training_hash(record) != self.data['training_record_sha256']:
            raise RuntimeError('frozen training record mismatch')
        if record.get('policy_sha256') != self.sha256:
            raise RuntimeError('training policy back-reference mismatch')
        return {'passed': True, 'source_files_verified': len(self.data['source_hashes']),
                'training_record_verified': True, 'policy_sha256': self.sha256}

    def select(self, tokens):
        bucket = bucket_for(tokens)
        choice = self.data['choices'][bucket]
        return choice['mode'], AttentionConfig(**choice['config']), bucket
