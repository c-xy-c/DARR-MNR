"""CUDA checks for support dependence and target/candidate exclusion."""
import argparse
import json
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from sspredrnet.model import SSPredRNet
from .model import ComponentProgram
from .data import KnownRowTraining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(12345)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    state = torch.load(args.baseline, map_location='cuda', weights_only=False)['model']
    with tempfile.TemporaryDirectory(prefix='program-ssl-contract-') as temporary:
        directory = Path(temporary) / 'center_single'
        directory.mkdir()
        sample = directory / 'RAVEN_0_train.npz'
        pixels = np.random.default_rng(12345).integers(0, 256, (16, 160, 160), dtype=np.uint8)
        np.savez(sample, image=pixels)
        dataset = KnownRowTraining(temporary)
        random.seed(12345)
        known, _, _ = dataset[0]
        pixels[6:] = 0  # Both third-row panels and every candidate change.
        np.savez(sample, image=pixels)
        random.seed(12345)
        changed, _, _ = dataset[0]
        assert torch.equal(known, changed)
        assert tuple(known.shape) == (2, 6, 80, 80)
    original = SSPredRNet().cuda().eval()
    original.load_state_dict(state, strict=True)
    model = ComponentProgram().cuda().eval()
    model.load_baseline(state)
    images = torch.rand(4, 2, 16, 80, 80, device='cuda') * 2 - 1
    configurations = torch.zeros(4, dtype=torch.long, device='cuda')
    with torch.no_grad():
        base = original(images)
        removed = model(images, intervention='no_program')
        difference = float((base - removed).abs().max())
        assert torch.allclose(base, removed, atol=1e-6, rtol=1e-6)
        active = model(images)
        permutation = torch.tensor([6, 2, 7, 0, 1, 5, 3, 4], device='cuda')
        reordered = images.clone()
        reordered[:, :, 8:] = images[:, :, 8:][:, :, permutation]
        assert torch.allclose(model(reordered), active[:, permutation], atol=1e-6, rtol=1e-6)
    model.train()
    loss, alpha = model.ssl(images[:, :, :6], configurations)
    try:
        model.ssl(images, configurations)
        raise AssertionError('sixteen-panel input was accepted by SSL')
    except ValueError:
        pass
    modified = images[:, :, :6].clone()
    modified[:, :, 5] = torch.randn_like(modified[:, :, 5])
    first, second = model.features(images[:, :, :6]), model.features(modified)
    assert torch.equal(model.compile(first[:, :3]), model.compile(second[:, :3]))
    assert torch.equal(model.operators(first[:, 3:5]), model.operators(second[:, 3:5]))
    assert alpha.std(0).max() > 0
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss, _ = model.ssl(images[:, :, :6], configurations)
        assert torch.isfinite(loss)
        loss.backward()
        missing = [name for name, p in model.named_parameters()
                   if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        assert not missing, missing
        optimizer.step()
    assert model.operators.right.grad.norm() > 0
    assert model.prior.net[-1].weight.grad.norm() > 0
    model.eval()
    with torch.no_grad():
        active = model(images)
        unif = model(images, intervention='uniform_program')
        removed = model(images, intervention='no_program')
        assert float((active - unif).abs().max()) > 0
        assert float((active - removed).abs().max()) > 0
        assert torch.allclose(model(reordered), active[:, permutation], atol=1e-6, rtol=1e-6)
    report = {'removed_program_matches_baseline_max_difference': difference,
              'candidate_permutation_equivariant_before_and_after_updates': True,
              'ssl_rejects_candidate_containing_inputs': True,
              'training_preprocessing_independent_of_last_ten_panels': True,
              'training_sample_does_not_require_answer_field': True,
              'compiled_weights_and_query_predictions_independent_of_query_target': True,
              'support_programs_vary_before_training': True,
              'all_trainable_parameters_have_finite_fp16_gradients': True,
              'operators_and_prior_have_nonzero_gradients': True,
              'active_minus_uniform_max_score_change': float((active - unif).abs().max()),
              'active_minus_no_program_max_score_change': float((active - removed).abs().max())}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
