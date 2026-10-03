"""Content fingerprints for execution sources, checkpoints and frozen modules."""
import hashlib
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_sources(root, files):
    return {name: digest(root / name) for name in files}


def state_hash(module):
    result = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()
