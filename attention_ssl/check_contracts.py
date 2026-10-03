"""Check the experimental support fit, task boundary and baseline anchoring."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from sspredrnet.model import SSPredRNet
from .model import AttentionCompletion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(12345)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    state = torch.load(args.baseline, map_location=device, weights_only=False)['model']
    original = SSPredRNet().to(device).eval()
    original.load_state_dict(state, strict=True)
    model = AttentionCompletion().to(device).eval()
    model.load_baseline(state)
    frozen = {k: v.clone() for k, v in model.reasoner.state_dict().items()}
    images = torch.rand(14, 2, 16, 80, 80, device=device) * 2 - 1
    configs = torch.arange(7, device=device).repeat_interleave(2)
    permutation = torch.tensor([6, 2, 7, 0, 1, 5, 3, 4], device=device)
    reordered = images.clone()
    reordered[:, :, 8:] = images[:, :, 8:][:, :, permutation]
    with torch.no_grad():
        baseline, initial = original(images), model(images)
        assert torch.equal(initial, baseline)
        assert torch.allclose(model(reordered), initial[:, permutation], atol=1e-6, rtol=1e-6)
        # Query targets are neither a key encoder input nor a support-fit input.
        features = model.features(images)
        first = model.completion.fit(features[:, :3])
        changed = features.clone()
        changed[:, 5] = torch.randn_like(changed[:, 5])
        changed[:, 8:] = torch.randn_like(changed[:, 8:])
        second = model.completion.fit(changed[:, :3])
        assert all(torch.equal(a, b) for a, b in zip(first, second))
        observed, _, unchanged_base = model.row_scores(features[:, :3], features[:, 6:8], features[:, 8:])
        swapped_delta, _ = model.completion_energy(features[:, :3].roll(2, 0), features[:, 6:8], features[:, 8:])
        assert torch.equal(observed, unchanged_base + model.evidence_weight * swapped_delta)
        model.completion.gate.fill_(.3)
        prediction, _ = model.completion(features[:, :3], features[:, 3:5])
        swapped, _ = model.completion(features[:, :3].roll(2, 0), features[:, 3:5])
        dependence = float((prediction - swapped).abs().max())
        assert dependence > 0
        model.completion.gate.zero_()
    model.train()
    try:
        model.ssl(images, configs)
        raise AssertionError('candidate-containing SSL input accepted')
    except ValueError:
        pass
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            loss, statistics = model.ssl(images[:, :, :6], configs)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        missing = [name for name, p in model.named_parameters()
                   if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        assert not missing, missing
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        assert not model.reasoner.training
        assert all(torch.equal(v, model.reasoner.state_dict()[k]) for k, v in frozen.items())
    assert model.completion.keys.attention.in_proj_weight.grad.norm() > 0
    assert model.completion.gate.grad.abs() > 0
    model.eval()
    with torch.no_grad():
        active = model(images)
        assert torch.allclose(model(reordered), active[:, permutation], atol=1e-6, rtol=1e-6)
        difference = float((active - baseline).abs().max())
        assert difference > 0
        # An entirely duplicated component has no valid negative and must stay finite.
        duplicate = images[:2, :, :6].clone()
        duplicate[1] = duplicate[0]
        loss, duplicate_stats = model.ssl(duplicate, configs[:2])
        assert torch.isfinite(loss)
        assert duplicate_stats['eligible_component_queries'] == 0
    report = {'device': str(device), 'torch': str(torch.__version__),
              'baseline_sha256': hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest(),
              'model_source_sha256': hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest(),
              'fixture': 'synthetic tensors', 'full_raven_evaluation': False,
              'initial_scores_bitwise_equal_to_baseline': True,
              'ssl_rejects_candidate_inputs': True,
              'query_target_cannot_change_support_fit': True,
              'zero_gate_support_swap_keeps_original_baseline_scores': True,
              'support_swap_changes_completion_max': dependence,
              'candidate_permutation_equivariant_before_and_after_training': True,
              'finite_training_gradients': True, 'frozen_baseline_weights_and_buffers_unchanged': True,
              'gradient_precision': 'float16' if device.type == 'cuda' else 'float32',
              'attention_and_gate_have_nonzero_gradients_after_updates': True,
              'losses': losses, 'last_training_statistics': statistics,
              'all_duplicate_targets_have_finite_loss_without_false_negatives': True,
              'post_update_max_score_change': difference}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
