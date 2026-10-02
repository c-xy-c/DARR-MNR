"""Evaluate best/final checkpoints using the native reasoner directly."""
import argparse
import json
from pathlib import Path

import torch

from .data import CONFIGS, Raven, loader, normalize
from .model import Reasoner


def score(predict, dataset_root, split, device, *, batch_size=128, workers=8,
          generator=None):
    dataset = Raven(dataset_root, split)
    batches = loader(dataset, batch_size, workers, generator)
    correct, total = [0] * 7, [0] * 7
    with torch.inference_mode():
        for images, answers, configs in batches:
            images = normalize(images.to(device, non_blocking=True))
            predictions = predict(images).argmin(1).cpu()
            for index in range(7):
                mask = configs == index
                total[index] += int(mask.sum())
                correct[index] += int(((predictions == answers) & mask).sum())
    if total != [2000] * 7:
        raise RuntimeError(f'incomplete {split}: {total}')
    by_config = {c: 100 * n / t for c, n, t in zip(CONFIGS, correct, total)}
    return {'accuracy_macro': sum(by_config.values()) / 7,
            'accuracy_micro': 100 * sum(correct) / sum(total),
            'by_configuration': by_config, 'correct': sum(correct),
            'total': sum(total)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    root = Path(args.run_dir)
    expected = json.loads((root / 'result.json').read_text())
    history = [json.loads(s) for s in (root / 'history.jsonl').read_text().splitlines()]
    config = json.loads((root / 'config.json').read_text())
    if [r['epoch'] for r in history] != list(range(1, config['epochs'] + 1)):
        raise RuntimeError('incomplete training history')
    reasoner = Reasoner().to(device).eval()

    def predict(images):
        errors = reasoner(images.flatten(0, 1))[-1]
        return errors.reshape(len(images), 2, 8).mean(1)

    reports = {}
    for name in ('best', 'final'):
        checkpoint = torch.load(root / f'{name}.pt', map_location=device, weights_only=False)
        reasoner.load_state_dict({k.removeprefix('reasoner.'): v
                                 for k, v in checkpoint['model'].items()}, strict=True)
        observed = {'epoch': checkpoint['epoch']}
        for split, field in (('val', 'validation'), ('test', 'test')):
            observed[field] = score(predict, args.dataset_root, split, device,
                                    batch_size=config['batch_size'], workers=args.workers)
            reference = (history[checkpoint['epoch'] - 1]['validation'] if split == 'val'
                         else expected['test'][name])
            for key in ('correct', 'by_configuration'):
                if observed[field][key] != reference[key]:
                    raise RuntimeError(f'{name} {split} {key} mismatch: {observed[field]}')
        reports[name] = observed
    output = root / 'verification.json'
    output.write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
