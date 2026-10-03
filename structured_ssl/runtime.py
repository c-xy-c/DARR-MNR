"""Explicit execution device and complete device RNG for sealed adaptation."""
import torch

from sspredrnet.checkpoint import rng_state as inherited_rng_state
from sspredrnet.checkpoint import restore_rng as inherited_restore_rng


def resolve_device(name):
    device = torch.device(name)
    if device.type == 'cuda' and torch.cuda.is_available():
        return device
    if device.type == 'mps' and torch.backends.mps.is_available():
        return device
    raise RuntimeError(f'full experiment requires an available CUDA or Metal device: {name}')


def execution_record(device):
    return {'device': str(device), 'torch': str(torch.__version__),
            'precision': f'float16 autocast with {device.type} GradScaler',
            'frozen_teacher_and_evaluation_precision': 'float32',
            'hardware': (torch.cuda.get_device_name(device) if device.type == 'cuda' else 'Apple Metal'),
            'metal_recommended_max_bytes': (torch.mps.recommended_max_memory()
                                            if device.type == 'mps' else None)}


def rng_state(train_generator, val_generator, *, device):
    state = inherited_rng_state(train_generator, val_generator)
    state['execution_device'] = str(device)
    if device.type == 'mps':
        state['mps'] = torch.mps.get_rng_state()
    return state


def restore_rng(state, train_generator, val_generator, *, device):
    if state['execution_device'] != str(device):
        raise RuntimeError('resume changed execution device')
    inherited_restore_rng(state, train_generator, val_generator)
    if device.type == 'mps':
        torch.mps.set_rng_state(state['mps'].cpu())
