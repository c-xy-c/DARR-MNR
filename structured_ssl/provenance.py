"""The complete executable dependency list of structured adaptation."""
from pathlib import Path
from sspredrnet.provenance import digest, hash_sources, state_hash


SOURCE_FILES = (
    'structured_ssl/__init__.py', 'structured_ssl/constants.py',
    'structured_ssl/blocks.py', 'structured_ssl/perception.py',
    'structured_ssl/pav.py', 'structured_ssl/objectives.py',
    'structured_ssl/model.py', 'structured_ssl/data.py', 'structured_ssl/train.py',
    'structured_ssl/evaluate.py', 'structured_ssl/runtime.py',
    'structured_ssl/provenance.py', 'structured_ssl/policy.py',
    'attention_ssl/model.py', 'sspredrnet/model.py', 'sspredrnet/layers.py',
    'sspredrnet/views.py', 'sspredrnet/data.py', 'sspredrnet/energy.py',
    'sspredrnet/checkpoint.py', 'sspredrnet/provenance.py',
)


def source_hashes():
    return hash_sources(Path(__file__).resolve().parents[1], SOURCE_FILES)


def anchor_hash(model):
    return state_hash(model.anchor)
