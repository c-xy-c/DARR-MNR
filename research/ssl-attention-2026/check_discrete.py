"""CUDA boundary check for the isolated six-operator research comparator."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from attention_ssl.model import AttentionCompletion
from attention_ssl.train import reasoner_hash, source_hashes
from sspredrnet.model import SSPredRNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(12345)
    device = torch.device(args.device)
    state = torch.load(args.baseline, map_location=device, weights_only=False)['model']
    model = AttentionCompletion().to(device).eval()
    assert type(model.completion).__name__ == 'DiscreteSupport'
    model.load_baseline(state)
    original = SSPredRNet().to(device).eval()
    original.load_state_dict(state, strict=True)
    fingerprint = reasoner_hash(model)
    images = torch.rand(14, 2, 16, 80, 80, device=device) * 2 - 1
    configs = torch.arange(7, device=device).repeat_interleave(2)
    permutation = torch.tensor([6, 2, 7, 0, 1, 5, 3, 4], device=device)
    permuted = images.clone()
    permuted[:, :, 8:] = images[:, :, 8:][:, :, permutation]
    with torch.no_grad():
        initial = model(images)
        assert torch.equal(initial, original(images))
        features = model.features(images)
        support = model.completion.compile(features[:, :3])
        changed = features.clone()
        changed[:, 5] = torch.randn_like(changed[:, 5])
        changed[:, 8:] = torch.randn_like(changed[:, 8:])
        assert torch.equal(support, model.completion.compile(changed[:, :3]))
        model.completion.gate.fill_(.3)
        energy = model.completion_energy(features[:, :3], features[:, 6:8], features[:, 8:])[0]
        swapped = model.completion_energy(features[:, :3].roll(2, 0), features[:, 6:8], features[:, 8:])[0]
        dependence = float((energy - swapped).abs().max())
        assert dependence > 0
        model.completion.gate.zero_()
    try:
        model.ssl(images, configs)
        raise AssertionError('candidate-containing input accepted')
    except ValueError:
        pass
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    losses = []
    for _ in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            loss, _ = model.ssl(images[:, :, :6], configs)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        scaler.step(optimizer)
        scaler.update()
        assert reasoner_hash(model) == fingerprint
        losses.append(float(loss.detach()))
    assert model.completion.gate.grad.abs() > 0
    model.eval()
    with torch.no_grad():
        active = model(images)
        assert torch.allclose(model(permuted), active[:, permutation], atol=1e-6, rtol=1e-6)
        difference = float((active - initial).abs().max())
        assert difference > 0
        duplicate = images[:2, :, :6].clone()
        duplicate[1] = duplicate[0]
        loss, stats = model.ssl(duplicate, configs[:2])
        assert torch.isfinite(loss) and stats['eligible_component_queries'] == 0
    report = {'fixture': 'synthetic tensors', 'full_raven_evaluation': False,
              'torch': str(torch.__version__), 'device': str(device),
              'initial_native_baseline_bitwise_equal': True,
              'six_known_panels_only': True, 'targets_cannot_change_support_weights': True,
              'support_swap_changes_new_energy': dependence,
              'gradient_precision': 'float16' if device.type == 'cuda' else 'float32',
              'scaled_training_gradients_finite': True, 'frozen_baseline_unchanged': True,
              'candidate_permutation_equivariant': True,
              'post_update_max_score_change': difference, 'losses': losses,
              'trainable_parameters': sum(p.numel() for p in parameters),
              'source_sha256': source_hashes(),
              'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
