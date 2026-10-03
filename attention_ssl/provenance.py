"""Executable dependencies and the immutable original reasoner fingerprint."""
from pathlib import Path
from sspredrnet.provenance import digest, hash_sources, state_hash


SOURCE_FILES = (
    'attention_ssl/model.py', 'attention_ssl/train.py', 'attention_ssl/evaluate.py',
    'attention_ssl/provenance.py', 'sspredrnet/model.py', 'sspredrnet/layers.py',
    'sspredrnet/views.py', 'sspredrnet/data.py', 'sspredrnet/energy.py',
    'sspredrnet/evaluate.py', 'sspredrnet/checkpoint.py', 'sspredrnet/provenance.py',
)


def source_hashes():
    return hash_sources(Path(__file__).resolve().parents[1], SOURCE_FILES)


def reasoner_hash(model):
    return state_hash(model.reasoner)
