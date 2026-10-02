"""Validation mechanism audit, then explicit checkpoint-locked test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from sspredrnet.data import CONFIGS, Raven, loader, normalize
from sspredrnet.evaluate import score
from .model import ComponentProgram


STUDIES = ('program', 'control', 'program-no-ssl', 'program-static')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_study(root, study, device, name='best'):
    directory = root / study
    config = json.loads((directory / 'config.json').read_text())
    history = [json.loads(s) for s in (directory / 'history.jsonl').read_text().splitlines()]
    if [r['epoch'] for r in history] != list(range(1, config['epochs'] + 1)):
        raise RuntimeError(f'{study}: incomplete history')
    if any(r['train_samples'] != 42000 or r['train_steps'] != 329 for r in history):
        raise RuntimeError(f'{study}: incomplete raw-puzzle epoch')
    best = max(history, key=lambda r: r['validation']['accuracy_macro'])
    checkpoint = torch.load(directory / f'{name}.pt', map_location=device, weights_only=False)
    if name == 'best' and checkpoint['epoch'] != best['epoch']:
        raise RuntimeError(f'{study}: checkpoint not selected by validation')
    model = ComponentProgram(use_program=study != 'control',
                             program_weight=config['program_weight'],
                             static_program=study == 'program-static').to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    return model, checkpoint, config


def summarize(correct, configurations):
    total, right = [0] * 7, [0] * 7
    for index in range(7):
        mask = configurations == index
        total[index], right[index] = int(mask.sum()), int(correct[mask].sum())
    if total != [2000] * 7:
        raise RuntimeError(f'incomplete evaluation: {total}')
    return {'accuracy_micro': float(100 * correct.mean()),
            'by_configuration': {c: 100 * n / t for c, n, t in zip(CONFIGS, right, total)},
            'correct': sum(right), 'total': sum(total)}


def paired_effect(full, ablated):
    help_count = int((full & ~ablated).sum())
    hurt_count = int((~full & ablated).sum())
    delta = (help_count - hurt_count) / len(full)
    draws = np.random.default_rng(12345).multinomial(
        len(full), [help_count / len(full), hurt_count / len(full),
                    1 - (help_count + hurt_count) / len(full)], size=10000)
    bounds = np.quantile(100 * (draws[:, 0] - draws[:, 1]) / len(full), [.025, .975])
    return {'full_minus_intervention_pp': 100 * delta,
            'paired_bootstrap_95_ci_pp': bounds.tolist(),
            'help': help_count, 'hurt': hurt_count}


def audit(root, dataset_root, device, workers):
    models = {s: load_study(root, s, device)[0] for s in STUDIES}
    base_state = models['control'].reasoner.state_dict()
    for study, model in models.items():
        for key, value in base_state.items():
            if key.startswith(('res', 'channel_reducer')):
                if not torch.equal(value, model.reasoner.state_dict()[key]):
                    raise RuntimeError(f'{study}: frozen perception changed at {key}')
    dataset = Raven(dataset_root, 'val')
    modes = [*STUDIES, 'no_program', 'uniform_program', 'swapped_program']
    predictions = {mode: [] for mode in modes}
    answers, configurations, alphas = [], [], []
    ssl_totals = {s: 0. for s in STUDIES if s != 'control'}
    total = 0
    score_deltas = {m: [] for m in modes if m not in STUDIES}
    with torch.inference_mode():
        for config_index, config_name in enumerate(CONFIGS):
            subset = Subset(dataset, [i for i, p in enumerate(dataset.paths)
                                      if p.parent.name == config_name])
            for images, labels, configs in loader(subset, 128, workers,
                                                  torch.Generator().manual_seed(12345)):
                batch = len(images)
                images = normalize(images.to(device, non_blocking=True))
                features = models['program'].features(images)
                scores = {}
                for mode in modes:
                    model = models[mode] if mode in STUDIES else models['program']
                    intervention = None if mode in STUDIES else mode
                    scores[mode] = model.predict_features(features, batch, intervention=intervention)
                    predictions[mode].append(scores[mode].argmin(1).cpu().numpy())
                    if mode not in STUDIES:
                        score_deltas[mode].append(float((scores[mode] - scores['program']).abs().mean()))
                for study, model in models.items():
                    if study == 'control':
                        continue
                    loss, alpha = model.ssl(images[:, :, :6], configs.to(device))
                    ssl_totals[study] += float(loss) * batch
                    if study == 'program':
                        alphas.append(alpha.cpu().numpy())
                answers.append(labels.numpy())
                configurations.append(configs.numpy())
                total += batch
    answers = np.concatenate(answers)
    configurations = np.concatenate(configurations)
    correct = {mode: np.concatenate(p) == answers for mode, p in predictions.items()}
    alpha = np.concatenate(alphas)
    report = {'split': 'val', 'perception_identical_across_studies': True,
              'intervention_swap_within_public_configuration_and_component': True,
              'accuracy': {m: summarize(c, configurations) for m, c in correct.items()},
              'paired_effects': {m: paired_effect(correct['program'], c)
                                 for m, c in correct.items() if m != 'program'},
              'candidate_free_completion_loss': {s: value / total for s, value in ssl_totals.items()},
              'alpha_mean': alpha.mean(0).tolist(), 'alpha_std': alpha.std(0).tolist(),
              'intervention_mean_absolute_score_change': {m: float(np.mean(v))
                                                         for m, v in score_deltas.items()},
              'scope': 'Frozen interventions measure immediate dependence, not retraining or semantic rule discovery.'}
    (root / 'validation_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    lock = {'selection_split': 'val', 'new_test_not_used_for_model_selection': True,
            'studies': {s: {name: digest(root / s / f'{name}.pt')
                            for name in ('best', 'final')} for s in STUDIES},
            'validation_audit_sha256': digest(root / 'validation_audit.json')}
    path = root / 'test_lock.json'
    if path.exists() and json.loads(path.read_text()) != lock:
        raise RuntimeError('an existing test lock differs')
    path.write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def evaluate_test(root, dataset_root, device, workers):
    lock = json.loads((root / 'test_lock.json').read_text())
    if digest(root / 'validation_audit.json') != lock['validation_audit_sha256']:
        raise RuntimeError('validation audit changed after lock')
    report = {'selection_split': 'val', 'source_training_budget_epochs': 20,
              'additional_training_epochs': 8, 'lineage_epochs': 25,
              'total_training_budget_epochs': 28, 'studies': {}}
    for study in STUDIES:
        report['studies'][study] = {}
        for name in ('best', 'final'):
            if digest(root / study / f'{name}.pt') != lock['studies'][study][name]:
                raise RuntimeError('checkpoint changed after lock')
            model, checkpoint, _ = load_study(root, study, device, name)
            result = {'additional_epoch': checkpoint['epoch'],
                      'checkpoint_sha256': lock['studies'][study][name],
                      'test': score(model, dataset_root, 'test', device, workers=workers,
                                    generator=torch.Generator().manual_seed(12347))}
            report['studies'][study][name] = result
            print(json.dumps({'study': study, 'checkpoint': name, **result}), flush=True)
    full = report['studies']['program']['best']['test']['accuracy_micro']
    report['requested_70_percent_gate_met'] = full >= 70.
    (root / 'test_result.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--experiment-root', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    root, device = Path(args.experiment_root), torch.device('cuda:0')
    if args.test:
        evaluate_test(root, args.dataset_root, device, args.workers)
    else:
        audit(root, args.dataset_root, device, args.workers)


if __name__ == '__main__':
    main()
