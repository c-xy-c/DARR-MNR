"""Compare cleaned code with the measured f43cf58 source without importing its CLI."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch

from program_ssl.data import KnownRowTraining
from program_ssl.evaluate import source_hashes
from program_ssl.model import ComponentProgram
from sspredrnet.data import CONFIGS, Raven, loader, normalize


def load_reference(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panels(configuration, variant):
    result = np.full((16, 80, 80), 255, np.uint8)
    for i, panel in enumerate(result):
        color, radius = 30 + 25 * ((i + variant) % 5), 4 + (i + variant) % 3
        if configuration.startswith('in_'):
            cv2.rectangle(panel, (9, 9), (71, 71), 0, 2)
            centers = [(40, 40)] if configuration.startswith('in_center') else [
                (29, 29), (51, 29), (29, 51), (51, 51)]
        elif configuration.startswith('left_'):
            centers = [(20, 40), (60, 40)]
        elif configuration.startswith('up_'):
            centers = [(40, 20), (40, 60)]
        elif configuration == 'distribute_four':
            centers = [(20, 20), (60, 20), (20, 60), (60, 60)]
        elif configuration == 'distribute_nine':
            centers = [(x, y) for x in (14, 40, 66) for y in (14, 40, 66)]
        else:
            centers = [(40, 40)]
        for center in centers:
            cv2.circle(panel, center, radius, color, -1)
    return np.stack([cv2.resize(p, (160, 160), interpolation=cv2.INTER_NEAREST) for p in result])


def assert_state_equal(first, second):
    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name]), name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-source', required=True)
    parser.add_argument('--run-root', default='program_ssl/results/support-program-v3')
    parser.add_argument('--baseline', default='sspredrnet/results/raven-20epoch/best.pt')
    parser.add_argument('--dataset-root', help='also compare complete real validation/test splits')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    reference_root = Path(args.reference_source)
    old_model = load_reference('cleanup_previous_model', reference_root / 'program_ssl/model.py')
    old_data = load_reference('cleanup_previous_data', reference_root / 'program_ssl/data.py')
    baseline = torch.load(args.baseline, map_location=device, weights_only=False)['model']
    torch.manual_seed(12345)
    old = old_model.ComponentProgram(program_weight=.05,
                                    verification_mode='support_evidence_ratio').to(device)
    old.load_baseline(baseline)
    torch.manual_seed(12345)
    new = ComponentProgram().to(device)
    new.load_baseline(baseline)
    assert_state_equal(old.state_dict(), new.state_dict())
    report = {'reference_commit': 'f43cf580250303f4c860c161df668502ea2aa1f3',
              'runtime_source_sha256': source_hashes(),
              'reference_source_sha256': {str(p.relative_to(reference_root)):
                  hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                  reference_root / 'program_ssl/model.py', reference_root / 'program_ssl/data.py')},
              'device': str(device), 'torch': str(torch.__version__),
              'initialization_and_parameter_keys_equal': True,
              'full_raven_evaluation': False}
    with tempfile.TemporaryDirectory(prefix='sspredrnet-cleanup-fixtures-') as temporary:
        for config in CONFIGS:
            folder = Path(temporary) / config
            folder.mkdir()
            for variant in range(3):
                image = panels(config, variant)
                np.savez(folder / f'RAVEN_{variant}_train.npz', image=image)
                np.savez(folder / f'RAVEN_{variant}_val.npz', image=image, target=variant)
        current, previous = KnownRowTraining(temporary), old_data.KnownRowTraining(temporary)
        images, configs = [], []
        for i in range(len(current)):
            random.seed(12345 + i)
            first = previous[i][0]
            random.seed(12345 + i)
            second, _, config = current[i]
            assert torch.equal(first, second), current.paths[i]
            images.append(second)
            configs.append(config)
        images = normalize(torch.stack(images).to(device))
        configs = torch.tensor(configs, device=device)
        report['seven_layout_training_views_exactly_equal'] = True
        opts = [torch.optim.Adam([p for p in m.parameters() if p.requires_grad],
                                 lr=3e-4, weight_decay=1e-5) for m in (old, new)]
        scalers = [torch.amp.GradScaler('cuda', enabled=device.type == 'cuda') for _ in opts]
        frozen = {k: v.clone() for k, v in new.reasoner.state_dict().items()}
        for _ in range(2):
            values = []
            for model, optimizer, scaler in zip((old, new), opts, scalers):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == 'cuda'):
                    loss, alpha = model.ssl(images, configs)
                assert torch.isfinite(loss)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                values.append((loss.detach(), alpha.detach()))
            assert torch.equal(values[0][0], values[1][0])
            assert torch.equal(values[0][1], values[1][1])
            for (name, before), (_, after) in zip(old.named_parameters(), new.named_parameters()):
                if before.requires_grad:
                    assert before.grad is not None and torch.equal(before.grad, after.grad), name
            for model, optimizer, scaler in zip((old, new), opts, scalers):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                scaler.step(optimizer)
                scaler.update()
            assert_state_equal(old.state_dict(), new.state_dict())
            assert_state_equal(frozen, new.reasoner.state_dict())
        report['two_loss_gradient_and_adam_steps_exactly_equal'] = True
        report['frozen_baseline_weights_and_buffers_unchanged'] = True
        dataset = Raven(temporary, 'val')
        evaluation_images = normalize(torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device))
        report['published_checkpoint_synthetic_scores'] = {}
        for name in ('best', 'final'):
            checkpoint = torch.load(Path(args.run_root) / 'program' / f'{name}.pt',
                                    map_location=device, weights_only=False)
            for model in (old, new):
                model.load_state_dict(checkpoint['model'], strict=True)
                model.eval()
            with torch.inference_mode():
                assert torch.equal(old(evaluation_images), new(evaluation_images))
                old.verification_mode, old.program_weight = 'posterior', .1
                assert torch.equal(old(evaluation_images), new.selection_scores(evaluation_images))
                old.verification_mode, old.program_weight = 'support_evidence_ratio', .05
            report['published_checkpoint_synthetic_scores'][name] = {
                'all_eight_candidate_scores_exactly_equal': True,
                'checkpoint_selection_scores_exactly_equal': True,
                'puzzles': len(dataset), 'configurations': len(CONFIGS)}
    if args.dataset_root:
        report['complete_real_splits'] = {}
        results = json.loads((Path(args.run_root) / 'test_result.json').read_text())
        for name in ('best', 'final'):
            state = torch.load(Path(args.run_root) / 'program' / f'{name}.pt',
                               map_location=device, weights_only=False)['model']
            for model in (old, new):
                model.load_state_dict(state, strict=True)
                model.eval()
            report['complete_real_splits'][name] = {}
            for split in ('val', 'test'):
                counts, total = [0] * 7, [0] * 7
                with torch.inference_mode():
                    for pixels, labels, configs in loader(Raven(args.dataset_root, split), 128, 8,
                                                          torch.Generator().manual_seed(12347)):
                        pixels = normalize(pixels.to(device))
                        original, cleaned = old(pixels), new(pixels)
                        assert torch.equal(original, cleaned), (name, split)
                        predictions = cleaned.argmin(1).cpu()
                        for index in range(7):
                            mask = configs == index
                            counts[index] += int(((predictions == labels) & mask).sum())
                            total[index] += int(mask.sum())
                assert total == [2000] * 7
                if split == 'test':
                    assert sum(counts) == results['studies']['program'][name]['test']['correct']
                report['complete_real_splits'][name][split] = {
                    'all_candidate_scores_exactly_equal': True, 'correct': sum(counts),
                    'total': sum(total), 'by_configuration': {
                        c: 100 * n / t for c, n, t in zip(CONFIGS, counts, total)}}
        report['full_raven_evaluation'] = True
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
