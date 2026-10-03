"""Sealed best/final evaluation with active-branch and support interventions."""
import argparse
import json
from pathlib import Path

import torch

from sspredrnet.data import CONFIGS, loader
from .data import ObjectRaven, to_device
from .model import StructuredCompletion
from .train import anchor_hash, digest, source_hashes


def summary(predictions, labels, configs):
    correct, total = [], []
    for index in range(7):
        mask = configs == index
        correct.append(int(((predictions == labels) & mask).sum()))
        total.append(int(mask.sum()))
    if total != [2000] * 7:
        raise RuntimeError(f'incomplete evaluation split: {total}')
    accuracies = {name: 100 * n / t for name, n, t in zip(CONFIGS, correct, total)}
    return {'correct': sum(correct), 'total': sum(total), 'by_configuration': accuracies,
            'accuracy_macro': sum(accuracies.values()) / 7}


def evaluate(model, dataset_root, split, device, workers):
    records = {name: [] for name in ('full', 'anchor', 'swapped', 'answers', 'configs')}
    max_change, max_factor_change, overflow = 0., 0., 0
    model.eval()
    with torch.inference_mode():
        for pack, answers, configs in loader(ObjectRaven(dataset_root, split), 128, workers,
                                             torch.Generator().manual_seed(12347)):
            pack = to_device(pack, device)
            features = model.anchor.features(pack['views'])
            levels, visible = model.encode(pack, 8)
            objects, valid = model.targets(pack, features)
            ids = torch.arange(len(levels), device=device)
            groups = configs.repeat_interleave(2).to(device)
            order = ids.clone()
            for layout in range(7):
                for component in range(2):
                    members = ids[(groups == layout) & (ids.remainder(2) == component)]
                    order[members] = members.roll(1)
            current, reference, swapped = [], [], []
            for support in (slice(0, 3), slice(3, 6)):
                observed, _, evidence = model.row(levels, visible, features, objects, valid,
                                                   support, slice(6, 8), slice(8, 16))
                exchanged, _, counterfactual = model.row(levels, visible, features, objects, valid,
                                                        support, slice(6, 8), slice(8, 16), order=order)
                assert torch.equal(evidence['anchor'], counterfactual['anchor'])
                max_factor_change = max(max_factor_change, float((evidence['factors'] - counterfactual['factors']).abs().max()))
                current.append(observed)
                reference.append(evidence['anchor'])
                swapped.append(exchanged)
            current, reference, swapped = [(sum(scores).reshape(len(answers), 2, 8).mean(1))
                                           for scores in (current, reference, swapped)]
            max_change = max(max_change, float((current - reference).abs().max()))
            for name, value in (('full', current.argmin(1)), ('anchor', reference.argmin(1)),
                                ('swapped', swapped.argmin(1)), ('answers', answers), ('configs', configs)):
                records[name].append(value.cpu())
            overflow += int(pack['overflow'].sum())
    values = {name: torch.cat(parts) for name, parts in records.items()}
    full, anchor, swapped, answers, configs = [values[k] for k in ('full', 'anchor', 'swapped', 'answers', 'configs')]
    report = {'full': summary(full, answers, configs), 'anchor': summary(anchor, answers, configs),
              'swapped_support': summary(swapped, answers, configs),
              'changed_answers_vs_anchor': int((full != anchor).sum()),
              'changed_answers_vs_swapped_support': int((full != swapped).sum()),
              'helped': int(((full == answers) & (anchor != answers)).sum()),
              'hurt': int(((full != answers) & (anchor == answers)).sum()),
              'max_score_change_vs_anchor': max_change, 'max_factor_change_under_support_swap': max_factor_change,
              'gate': float(model.gate.detach().tanh()), 'proposal_overflow_panels': overflow,
              'support_swap_preserves_entire_attention_anchor': True}
    report['nonregression_and_active_branch'] = (report['full']['correct'] >= report['anchor']['correct'] and
                                                report['changed_answers_vs_anchor'] > 0 and
                                                report['changed_answers_vs_swapped_support'] > 0)
    return report


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
        raise FileExistsError('preserve existing evaluation record')
    seal = json.loads((root / 'training_complete.json').read_text())
    if seal.get('schema_version') != 1 or seal.get('method') != 'structured-object-support-pav':
        raise RuntimeError('structured-completion sealed training required')
    if source_hashes() != seal['source_sha256']:
        raise RuntimeError('runtime differs from trained source')
    for name in ('best', 'final'):
        if digest(root / f'{name}.pt') != seal['checkpoint_sha256'][name]:
            raise RuntimeError(f'{name} checkpoint bytes changed')
    if args.split == 'test':
        validation = json.loads((root / 'val_evaluation.json').read_text())
        if validation['seal'] != seal or not validation['checkpoints']['best']['nonregression_and_active_branch']:
            raise RuntimeError('best has not passed complete validation nonregression/active-support audit')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = StructuredCompletion().to(device).eval()
    report = {'split': args.split, 'seal': seal, 'torch': str(torch.__version__),
              'device': str(device), 'checkpoints': {}}
    for name in ('best', 'final'):
        checkpoint = torch.load(root / f'{name}.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        if anchor_hash(model) != seal['frozen_anchor_sha256']:
            raise RuntimeError('trained model changed its frozen anchor')
        report['checkpoints'][name] = {**evaluate(model, args.dataset_root, args.split, device, args.workers),
                                       'epoch': checkpoint['epoch'],
                                       'lineage_epoch': checkpoint['source_epoch'] + checkpoint['epoch']}
        print(json.dumps({'checkpoint': name, **report['checkpoints'][name]}), flush=True)
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
