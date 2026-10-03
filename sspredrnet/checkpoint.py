"""Atomic checkpoints and complete training RNG state shared by both trainers."""
import os
import random

import numpy as np
import torch


def save_checkpoint(checkpoint, path):
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def rng_state(train_generator, val_generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
            'train_generator': train_generator.get_state(),
            'val_generator': val_generator.get_state()}


def restore_rng(state, train_generator, val_generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    train_generator.set_state(state['train_generator'].cpu())
    val_generator.set_state(state['val_generator'].cpu())
