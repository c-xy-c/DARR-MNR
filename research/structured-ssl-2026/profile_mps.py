"""Synthetic batch128 Metal capacity check; never a RAVEN accuracy result."""
import argparse
import json
from pathlib import Path
import time
import random

import numpy as np

import torch

from structured_ssl.check_contracts import fixture
from structured_ssl.data import to_device
from structured_ssl.data import ObjectRaven
from structured_ssl.model import StructuredCompletion
from structured_ssl.runtime import rng_state, restore_rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset-root')
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('Metal unavailable')
    torch.set_num_threads(4)
    torch.manual_seed(12345)
    print(json.dumps({'phase': 'load_cpu_model'}), flush=True)
    model = StructuredCompletion()
    model.load_anchor(torch.load(args.anchor, map_location='cpu', weights_only=False)['model'])
    model = model.to('mps')
    print(json.dumps({'phase': 'build_batch'}), flush=True)
    model.train()
    if args.dataset_root:
        dataset = ObjectRaven(args.dataset_root, 'train')
        indices = [(i % 7) * 6000 + (i // 7) * 31 for i in range(128)]
        examples = [dataset[i] for i in indices]
        pack = {k: torch.stack([example[0][k] for example in examples]) for k in examples[0][0]}
        configs = torch.tensor([example[2] for example in examples], device='mps')
        fixture_name = '128 official RAVEN training puzzles, seven configurations'
    else:
        pack = {k: v[:, :, :6].repeat((32,) + (1,) * (v.ndim - 1)) for k, v in fixture().items()}
        # Keep 128 distinct known targets in the synthetic negative pool.
        pack['views'][:, :, 5, 40, 40] = torch.arange(128)[:, None] / 127
        configs = torch.arange(128, device='mps') % 7
        fixture_name = 'synthetic full batch128, seven configuration IDs'
    pack = to_device(pack, torch.device('mps'))
    originals = {k: v.cpu().clone() for k, v in pack.items()}
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=3e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('mps')
    steps = []
    print(json.dumps({'phase': 'three_training_steps'}), flush=True)
    for _ in range(3):
        torch.mps.synchronize()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='mps', dtype=torch.float16):
            loss, statistics = model.ssl(pack, configs)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite Metal loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('nonfinite Metal gradients')
        scaler.step(optimizer)
        scaler.update()
        torch.mps.synchronize()
        record = {'step_seconds': time.monotonic() - started, 'loss': float(loss.detach()),
                  'driver_bytes': torch.mps.driver_allocated_memory(),
                  'current_bytes': torch.mps.current_allocated_memory(),
                  'statistics': statistics}
        if any(not torch.equal(value.cpu(), originals[k]) for k, value in pack.items()):
            raise RuntimeError('SSL modified the inputs')
        steps.append(record)
        print(json.dumps(record), flush=True)
    generator, validation_generator = torch.Generator(), torch.Generator()
    saved_rng = rng_state(generator, validation_generator, device=torch.device('mps'))
    def draws():
        return (random.random(), float(np.random.rand()), torch.rand(5),
                torch.rand(5, device='mps').cpu(), torch.rand(5, generator=generator))
    expected = draws()
    restore_rng(saved_rng, generator, validation_generator, device=torch.device('mps'))
    actual = draws()
    assert expected[:2] == actual[:2] and all(torch.equal(a, b) for a, b in zip(expected[2:], actual[2:]))
    model.eval()
    with torch.inference_mode():
        if args.dataset_root:
            dataset = ObjectRaven(args.dataset_root, 'val')
            examples = [dataset[i * 2000] for i in range(7)]
            validation = {k: torch.stack([example[0][k] for example in examples]) for k in examples[0][0]}
            scores = model(to_device(validation, torch.device('mps')))
        else:
            scores = model(to_device(fixture(), torch.device('mps')))
        assert torch.isfinite(scores).all()
    report = {'fixture': fixture_name,
              'full_raven_evaluation': False, 'device': 'mps', 'torch': str(torch.__version__),
              'activation_checkpointing': True, 'precision': 'float16 autocast with MPS GradScaler',
              'steps': steps, 'finite_forward_backward': True, 'pack_unchanged_after_ssl': True,
              'recommended_max_bytes': torch.mps.recommended_max_memory(),
              'metal_and_cpu_rng_restoration_bitwise_equal': True,
              'fp32_validation_forward_finite': True,
              'duplicate_filter': 'exact_cpu_byte_groups'}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
