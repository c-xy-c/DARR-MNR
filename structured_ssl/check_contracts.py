"""Boundary and mechanism checks; synthetic fixtures are not RAVEN accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import proposal_pack, proposals, to_device
from .model import StructuredCompletion, set_energy


def fixture():
    views = np.full((4, 2, 16, 80, 80), 255, np.float32)
    for sample in range(4):
        for component in range(2):
            for panel in range(16):
                cv2.rectangle(views[sample, component, panel], (8 + panel % 3, 9 + sample),
                              (25 + sample, 26 + panel % 4), 30 + panel * 5, 2)
                cv2.circle(views[sample, component, panel], (53, 53), 6 + sample, 60 + component * 30, -1)
    packs = [proposal_pack(v) for v in views]
    return {k: torch.stack([p[k] for p in packs]) for k in packs[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(12345)
    device = torch.device(args.device)
    model = StructuredCompletion().to(device)
    model.load_anchor(torch.load(args.anchor, map_location=device, weights_only=False)['model'])
    model.eval()
    frozen = {k: v.clone() for k, v in model.anchor.state_dict().items()}
    pack = to_device(fixture(), device)
    short = {k: v[:, :, :6].clone() for k, v in pack.items()}
    altered = {k: v.clone() for k, v in short.items()}
    altered['views'][:, :, 5] = -1
    altered['geometry'][:, :, 5] = .8
    altered['valid'][:, :, 5] = False
    permutation = torch.tensor([5, 1, 3, 0, 7, 4, 6, 2], device=device)
    reordered = {k: v.clone() for k, v in pack.items()}
    for k in reordered:
        reordered[k][:, :, 8:] = pack[k][:, :, 8:][:, :, permutation]
    with torch.no_grad():
        initial, anchor = model(pack), model.anchor(pack['views'])
        assert torch.equal(initial, anchor)
        before, _ = model.encode(short, 5)
        after, _ = model.encode(altered, 5)
        assert torch.equal(before, after), 'held-out target changed visible representation'
        factors = model.pav.compiler(before[:, :3], short['valid'].flatten(0, 1)[:, :3])
        factors_after = model.pav.compiler(after[:, :3], altered['valid'].flatten(0, 1)[:, :3])
        assert torch.equal(factors, factors_after)
        swap_effect = float((factors - factors.roll(2, 0)).abs().max())
        assert swap_effect > 0
        features = model.anchor.features(short['views'])
        fixed_objects, _ = model.targets(short, features)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
            protected_objects, _ = model.targets(short, features)
        assert protected_objects.dtype == torch.float32
        assert torch.equal(protected_objects, fixed_objects), 'AMP changed immutable teacher targets'
        captured = []
        handle = model.perception.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0].clone()))
        model.masked_loss(short, fixed_objects)
        handle.remove()
        masked_pixels = captured[-1].reshape(-1, 5, 80, 80)
        changed_panels = (masked_pixels != short['views'].flatten(0, 1)[:, :5]).flatten(2).any(-1).sum(-1)
        assert torch.all(changed_panels == 1), 'masking removed all known-panel context'
    try:
        model.ssl(pack, torch.tensor([0, 0, 1, 1], device=device))
        raise AssertionError('sixteen-panel SSL input was accepted')
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
            loss, statistics = model.ssl(short, torch.tensor([0, 0, 1, 1], device=device))
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        invalid = [k for k, p in model.named_parameters() if p.requires_grad and
                   (p.grad is None or not torch.isfinite(p.grad).all())]
        assert not invalid, invalid
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        assert all(torch.equal(v, model.anchor.state_dict()[k]) for k, v in frozen.items())
    gradients = {name: float(sum((p.grad.float().square().sum() for p in module.parameters() if p.grad is not None),
                                torch.zeros((), device=device)).sqrt())
                 for name, module in (('perception', model.perception), ('compiler', model.pav.compiler),
                                      ('pav', model.pav), ('masked_object', model.masked))}
    assert all(value > 0 for value in gradients.values())
    model.eval()
    with torch.no_grad():
        active = model(pack)
        candidate_equivariance = float((model(reordered) - active[:, permutation]).abs().max())
        assert candidate_equivariance < 1e-5
        change = float((active - anchor).abs().max())
        assert change > 0
        predicted = torch.randn(2, 10, 38, device=device)
        presence = torch.randn(2, 10, device=device)
        targets = torch.randn(2, 3, 10, 38, device=device)
        valid = torch.rand(2, 3, 10, device=device) > .5
        order = torch.randperm(10, device=device)
        expected = set_energy(predicted, presence, targets, valid)
        actual = set_energy(predicted, presence, targets[:, :, order], valid[:, :, order])
        object_equivariance = float((expected - actual).abs().max())
        assert object_equivariance < 1e-5
        blank = {k: v.clone() for k, v in short.items()}
        blank['views'].fill_(1)
        blank['masks'].zero_()
        blank['geometry'].zero_()
        blank['valid'].zero_()
        blank_loss, _ = model.ssl(blank, torch.zeros(4, dtype=torch.long, device=device))
        assert torch.isfinite(blank_loss)
    blank_region = np.full((80, 80), 255, np.uint8)
    assert not proposals(blank_region)[3].any()
    for y, x in [(5 + 10 * i, 5 + 12 * j) for i in range(4) for j in range(4)]:
        blank_region[y:y + 3, x:x + 3] = 0
    assert proposals(blank_region)[4], 'overflow not recorded'
    report = {'fixture': 'synthetic contours and tensors', 'full_raven_evaluation': False,
              'device': str(device), 'torch': str(torch.__version__),
              'anchor_sha256': hashlib.sha256(Path(args.anchor).read_bytes()).hexdigest(),
              'trainable_parameters': sum(p.numel() for p in parameters),
              'frozen_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad),
              'initial_scores_bitwise_equal_to_anchor': True,
              'heldout_target_cannot_change_visible_encoding_or_compilation': True,
              'object_masking_keeps_four_complete_visible_panels': True,
              'fixed_object_targets_are_fp32_and_bitwise_equal_under_autocast': True,
              'ssl_rejects_candidates': True, 'frozen_anchor_buffers_unchanged': True,
              'all_trainable_parameters_have_finite_gradients': True,
              'gradient_norms': gradients, 'losses': losses, 'training_statistics': statistics,
              'candidate_permutation_max_error': candidate_equivariance,
              'target_object_permutation_max_error': object_equivariance,
              'support_changes_dynamic_parameters_max': swap_effect,
              'post_update_score_change_max': change, 'trained_gate': float(model.gate.detach().tanh()),
              'blank_panels_and_no_legal_negatives_finite': True,
              'proposal_overflow_merges_and_records': True}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
