"""Boundary and mechanism checks; synthetic fixtures are not RAVEN accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import proposal_pack, proposals, to_device
from .model import StructuredCompletion, set_energy, pixel_duplicates, support_residuals


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
    untouched = {k: v.clone() for k, v in short.items()}
    altered = {k: v.clone() for k, v in short.items()}
    altered['views'][:, :, 5] = -1
    altered['geometry'][:, :, 5] = .8
    altered['valid'][:, :, 5] = False
    permutation = torch.tensor([5, 1, 3, 0, 7, 4, 6, 2], device=device)
    reordered = {k: v.clone() for k, v in pack.items()}
    for k in reordered:
        reordered[k][:, :, 8:] = pack[k][:, :, 8:][:, :, permutation]
    with torch.no_grad():
        equality_fixture = torch.randn(12, 6400, device=device)
        equality_fixture[3] = equality_fixture[1]
        equality_fixture[7] = equality_fixture[5]
        assert torch.equal(pixel_duplicates(equality_fixture), torch.cdist(equality_fixture, equality_fixture, p=0).eq(0))
        zero_fixture = torch.zeros(2, 10, device=device)
        zero_fixture[1] = -zero_fixture[1]
        assert pixel_duplicates(zero_fixture).all()
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
        fixed_objects, _ = model.targets(short)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
            protected_objects, _ = model.targets(short)
        assert protected_objects.dtype == torch.float32
        assert torch.equal(protected_objects, fixed_objects), 'AMP changed immutable teacher targets'
        outside_changed = {k: v.clone() for k, v in short.items()}
        x0, y0, x1, y1 = short['boxes'][0, 0, 0, 0].tolist()
        outside = torch.ones(80, 80, dtype=torch.bool, device=device)
        outside[y0:y1, x0:x1] = False
        outside_changed['views'][0, 0, 0].masked_fill_(outside, 1.)
        assert torch.equal(short['views'][0, 0, 0, y0:y1, x0:x1],
                           outside_changed['views'][0, 0, 0, y0:y1, x0:x1])
        modified_objects, _ = model.targets(outside_changed)
        assert torch.equal(fixed_objects[0, 0, 0], modified_objects[0, 0, 0]), 'outside pixels changed isolated target'
        dense_truth = torch.zeros(2, 32, 25, device=device)
        dense_truth[:, torch.arange(25), torch.arange(25)] = 1
        predicted_objects = torch.randn(2, 10, 38, device=device)
        predicted_presence = torch.randn(2, 10, device=device)
        support_targets = torch.randn(2, 10, 38, device=device)
        support_valid = torch.rand(2, 10, device=device) > .5
        aligned = support_residuals((dense_truth, predicted_objects, predicted_presence),
                                    dense_truth, support_targets, support_valid)
        permuted = support_residuals((dense_truth.roll(1, -1), predicted_objects, predicted_presence),
                                     dense_truth, support_targets, support_valid)
        assert aligned[0].eq(0).all()
        assert permuted[0][..., :32].abs().sum() > 0, 'spatial permutation escaped full feedback'
        assert permuted[0][..., 32:].eq(0).all(), 'fixture must preserve the pooled component'
        object_order = torch.randperm(10, device=device)
        reordered_residuals = support_residuals((dense_truth, predicted_objects, predicted_presence),
            dense_truth, support_targets[:, object_order], support_valid[:, object_order])
        residual_permutation_error = float((aligned[1] - reordered_residuals[1]).abs().max())
        assert residual_permutation_error < 1e-5, 'support feedback depends on target-object order'
        absent = support_residuals((dense_truth, predicted_objects, predicted_presence),
                                   dense_truth, support_targets, torch.zeros_like(support_valid))
        assert torch.equal(absent[1][..., -1], predicted_presence.sigmoid())
        assert absent[1][..., :38].eq(0).all(), 'null targets created appearance errors'
        feedback_state = torch.randn(2, 35, 96, device=device)
        zero_residuals = (torch.zeros(2, 25, 64, device=device), torch.zeros(2, 10, 39, device=device))
        assert model.pav.feedback[0](feedback_state, feedback_state, zero_residuals).eq(0).all()
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
    model.perception.train()
    perception_inputs = [short[key].flatten(0, 2)[:10]
                         for key in ('views', 'masks', 'geometry', 'valid')]
    perception_parameters = tuple(model.perception.parameters())
    direct = model.perception(*perception_inputs)
    direct_gradients = torch.autograd.grad(direct.square().mean(), perception_parameters)
    model.perception.activation_checkpointing = False
    stored = model.perception(*perception_inputs)
    stored_gradients = torch.autograd.grad(stored.square().mean(), perception_parameters)
    model.perception.activation_checkpointing = True
    assert torch.equal(direct, stored), 'activation checkpointing changed perception output'
    checkpoint_gradient_error = max(float((a - b).abs().max())
                                    for a, b in zip(direct_gradients, stored_gradients))
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-5)
               for a, b in zip(direct_gradients, stored_gradients)), 'checkpoint gradient mismatch'
    del direct, stored, direct_gradients, stored_gradients
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
        assert all(torch.equal(v, short[k]) for k, v in untouched.items()), 'SSL changed its input data'
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
              'ssl_does_not_mutate_input_pixels_or_proposals': True,
              'exact_pixel_groups_match_hamming_zero_including_signed_zero': True,
              'fixed_object_targets_are_fp32_and_bitwise_equal_under_autocast': True,
              'architecture_revision': 'isolated-target-structured-feedback-v2',
              'bbox_outside_intervention_preserves_object_target_bitwise': True,
              'structured_feedback_detects_spatial_permutation_with_identical_mean': True,
              'structured_feedback_target_object_permutation_max_error': residual_permutation_error,
              'null_object_targets_only_contribute_presence_feedback': True,
              'zero_structured_error_produces_zero_attention_feedback': True,
              'ssl_rejects_candidates': True, 'frozen_anchor_buffers_unchanged': True,
              'all_trainable_parameters_have_finite_gradients': True,
              'activation_checkpointing_preserves_outputs_and_gradients': True,
              'activation_checkpointing_gradient_max_error': checkpoint_gradient_error,
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
