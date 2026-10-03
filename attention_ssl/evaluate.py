"""Evaluate trained attention completion and its matched frozen baseline.

Validation must pass before opening test. Scores and intervention effects are
reported for both best and final, including real changes of candidate choices.
"""
import argparse
import json
from pathlib import Path

import torch

from sspredrnet.data import CONFIGS, Raven, loader, normalize
from .model import AttentionCompletion
from .provenance import digest, source_hashes, reasoner_hash


def summary(predictions, labels, configs):
    counts, total = [], []
    for index in range(7):
        mask = configs == index
        counts.append(int(((predictions == labels) & mask).sum()))
        total.append(int(mask.sum()))
    if total != [2000] * 7:
        raise RuntimeError('complete 14000-puzzle split required')
    by_config = {c: 100 * n / t for c, n, t in zip(CONFIGS, counts, total)}
    return {'accuracy_macro': sum(by_config.values()) / 7, 'correct': sum(counts),
            'total': sum(total), 'by_configuration': by_config}


def evaluate(model, dataset_root, split, device, workers):
    answers, configs, full, base, swapped = [], [], [], [], []
    max_change = 0.
    model.eval()
    with torch.inference_mode():
        for images, labels, configuration in loader(Raven(dataset_root, split), 128, workers,
                                                    torch.Generator().manual_seed(12347)):
            features = model.features(normalize(images.to(device)))
            current, original, counterfactual = [], [], []
            groups = configuration.repeat_interleave(2).to(device)
            ids = torch.arange(len(features), device=device)
            # Roll complete support rows only within layout and component.
            order = ids.clone()
            for layout in range(7):
                for component in range(2):
                    members = ids[(groups == layout) & (ids.remainder(2) == component)]
                    order[members] = members.roll(1)
            for indices in (slice(0, 3), slice(3, 6)):
                observed, _, baseline = model.row_scores(features[:, indices], features[:, 6:8], features[:, 8:])
                # Exchange only the new branch's support. Preserve the exact
                # original PaV score to isolate this mechanism's dependence.
                swapped_delta = model.completion_energy(features[order, indices], features[:, 6:8], features[:, 8:])[0]
                intervention = baseline + model.evidence_weight * swapped_delta
                current.append(observed)
                original.append(baseline)
                counterfactual.append(intervention)
            current = sum(current).reshape(len(images), 2, 8).mean(1)
            original = sum(original).reshape(len(images), 2, 8).mean(1)
            counterfactual = sum(counterfactual).reshape(len(images), 2, 8).mean(1)
            max_change = max(max_change, float((current - original).abs().max()))
            full.append(current.argmin(1).cpu())
            base.append(original.argmin(1).cpu())
            swapped.append(counterfactual.argmin(1).cpu())
            answers.append(labels)
            configs.append(configuration)
    full, base, swapped, answers, configs = [torch.cat(x) for x in (full, base, swapped, answers, configs)]
    result = {'full': summary(full, answers, configs), 'baseline': summary(base, answers, configs),
              'swapped_support': summary(swapped, answers, configs),
              'changed_answers_vs_baseline': int((full != base).sum()),
              'changed_answers_vs_swapped_support': int((full != swapped).sum()),
              'helped': int(((full == answers) & (base != answers)).sum()),
              'hurt': int(((full != answers) & (base == answers)).sum()),
              'max_score_change_vs_baseline': max_change,
              'completion_gate': float(model.completion.gate.detach().tanh()),
              'support_swap_preserves_baseline_scoring': True,
              'swap_scope': 'within evaluation batch, same public layout and component'}
    result['nonregression_and_active_branch'] = (result['full']['correct'] >= result['baseline']['correct']
                                                and result['changed_answers_vs_baseline'] > 0
                                                and result['changed_answers_vs_swapped_support'] > 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root, device = Path(args.run_dir), torch.device(args.device)
    output = root / f'{args.split}_evaluation.json'
    if output.exists():
        raise FileExistsError('evaluation record exists; preserve it instead of overwriting')
    seal = json.loads((root / 'training_complete.json').read_text())
    if seal.get('schema_version') != 1 or seal.get('method') != 'attention-support-completion':
        raise RuntimeError('completed attention-completion run required')
    if source_hashes() != seal['source_sha256']:
        raise RuntimeError('runtime changed after training was sealed')
    for name in ('best', 'final'):
        if digest(root / f'{name}.pt') != seal['checkpoint_sha256'][name]:
            raise RuntimeError(f'{name} checkpoint changed')
    if args.split == 'test':
        validation = json.loads((root / 'val_evaluation.json').read_text())
        if validation['seal'] != seal or not validation['checkpoints']['best']['nonregression_and_active_branch']:
            raise RuntimeError('best checkpoint has not passed validation nonregression and active-branch audit')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = AttentionCompletion(evidence_weight=seal['evidence_weight']).to(device).eval()
    report = {'split': args.split, 'seal': seal, 'device': str(device),
              'torch': str(torch.__version__), 'checkpoints': {}}
    for name in ('best', 'final'):
        checkpoint = torch.load(root / f'{name}.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        if reasoner_hash(model) != seal['frozen_reasoner_sha256']:
            raise RuntimeError(f'{name} changed the frozen baseline')
        report['checkpoints'][name] = {**evaluate(model, args.dataset_root, args.split, device, args.workers),
                                       'epoch': checkpoint['epoch'],
                                       'lineage_epoch': checkpoint['source_epoch'] + checkpoint['epoch']}
        print(json.dumps({'checkpoint': name, **report['checkpoints'][name]}), flush=True)
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
