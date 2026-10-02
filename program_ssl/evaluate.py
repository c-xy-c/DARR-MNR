"""Validation mechanism audit, then explicit checkpoint-locked test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import cv2
import torch

from sspredrnet.data import CONFIGS, Raven, loader, normalize
from sspredrnet.evaluate import score as native_score
from .model import ComponentProgram, operator_energy
from sspredrnet.views import split_layers
from .feature_cache import FrozenFeatureCache


STUDIES = ('program', 'control', 'program-no-ssl', 'program-static')
CALIBRATION_WEIGHTS = (.05, .1, .2)  # Zero is a diagnostic, never selectable.
VERIFICATION_MODES = ('posterior', 'support_evidence_ratio')
EVALUATION_SOURCES = ('program_ssl/model.py', 'program_ssl/evaluate.py',
                      'program_ssl/feature_cache.py', 'sspredrnet/data.py',
                      'sspredrnet/views.py', 'sspredrnet/model.py',
                      'sspredrnet/layers.py', 'sspredrnet/evaluate.py')


def evaluation_hashes():
    package = Path(__file__).resolve().parents[1]
    return {name: digest(package / name) for name in EVALUATION_SOURCES}


class KnownRowValidation(Raven):
    """No target-dependent view fallback and no stochastic training flip.

    Kept in the auditor so the running training-source snapshot is immutable.
    """
    def __init__(self, root):
        super().__init__(root, 'val')

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            raw = data['image'].reshape(16, 160, 160)[:6]
        panels = np.stack([cv2.resize(p, (80, 80), interpolation=cv2.INTER_NEAREST)
                           for p in raw]).astype(np.uint8)
        layers = [split_layers(p, path.parent.name) for p in panels]
        if any(layer is None for layer in layers[:5]):
            views = np.stack((panels, panels))
        else:
            layers = [np.stack((p, p)) if layer is None else layer
                      for p, layer in zip(panels, layers)]
            views = np.stack(layers, axis=1)
        return torch.from_numpy(views.astype(np.float32)), -1, CONFIGS.index(path.parent.name)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_study(root, study, device, name='best'):
    directory = root / study
    config = json.loads((directory / 'config.json').read_text())
    complete = json.loads((directory / 'training_complete.json').read_text())
    if config['version'] != 3 or config['epochs'] != 8 or complete['epochs'] != 8:
        raise RuntimeError(f'{study}: repaired eight-epoch experiment required')
    history = [json.loads(s) for s in (directory / 'history.jsonl').read_text().splitlines()]
    if [r['epoch'] for r in history] != list(range(1, config['epochs'] + 1)):
        raise RuntimeError(f'{study}: incomplete history')
    if any(r['train_samples'] != 42000 or r['train_steps'] != 329 for r in history):
        raise RuntimeError(f'{study}: incomplete raw-puzzle epoch')
    best = max(history, key=lambda r: r['validation']['accuracy_macro'])
    checkpoint = torch.load(directory / f'{name}.pt', map_location=device, weights_only=False)
    if name == 'best' and checkpoint['epoch'] != best['epoch']:
        raise RuntimeError(f'{study}: checkpoint not selected by validation')
    if name == 'final' and checkpoint['epoch'] != config['epochs']:
        raise RuntimeError(f'{study}: final checkpoint is not final epoch')
    if digest(config['baseline']) != config['baseline_sha256']:
        raise RuntimeError(f'{study}: source checkpoint changed')
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


def program_energy(model, features, batch, intervention=None, verification_mode=None):
    first = model.compile(features[:, :3], intervention=intervention)
    second = model.compile(features[:, 3:6], intervention=intervention)
    joint = .5 * (first.clamp_min(1e-12).log() + second.clamp_min(1e-12).log())
    alpha = joint.softmax(-1)
    if intervention == 'swapped_program':
        alpha = alpha.roll(2, 0)
    if verification_mode is None:
        energy = model.verification_energy(features[:, 6:8], features[:, 8:], alpha)
    else:
        energy = model.verify(features[:, 6:8], features[:, 8:], alpha)
        if verification_mode == 'support_evidence_ratio':
            energy = energy - model.verify(features[:, 6:8], features[:, 8:],
                                           torch.full_like(alpha, 1 / 6))
    return energy.reshape(batch, 2, 8).mean(1)


def completion_metrics(model, features, mask, intervention=None, *, return_correct=False):
    alpha = model.compile(features[:, :3], intervention=intervention)
    energies = model.verify(features[:, 3:5], features[:, 5], alpha)
    logits = (-energies / model.verify_temperature).masked_fill(mask, -torch.inf)
    ids = torch.arange(len(features), device=features.device)
    positive = energies.diagonal().mean()
    nce = torch.nn.functional.cross_entropy(logits, ids)
    correct = logits.argmax(1) == ids
    metrics = {'positive_energy': float(positive), 'conditional_nce': float(nce),
               'loss': float(positive + nce),
               'retrieval_accuracy': float(correct.float().mean())}
    return (metrics, correct) if return_correct else metrics


def completion_effect(full, other):
    # Cluster the two component decisions from each raw puzzle together.
    delta = full.astype(float).mean(1) - other.astype(float).mean(1)
    values, counts = np.unique(delta, return_counts=True)
    draws = np.random.default_rng(12345).multinomial(len(delta), counts / len(delta), size=10000)
    bounds = np.quantile(100 * (draws @ values) / len(delta), [.025, .975])
    return {'full_minus_intervention_pp': float(100 * delta.mean()),
            'paired_bootstrap_95_ci_pp': bounds.tolist(),
            'bootstrap_unit': 'raw_puzzle_with_both_component_views',
            'raw_puzzles': len(delta)}


def audit(root, dataset_root, device, workers, cache_root):
    models = {s: load_study(root, s, device)[0] for s in STUDIES}
    configs_by_study = {s: json.loads((root / s / 'config.json').read_text()) for s in STUDIES}
    source = configs_by_study['program']['baseline']
    source_state = torch.load(source, map_location=device, weights_only=False)['model']
    for study, model in models.items():
        if study != 'control':
            for key, tensor in model.reasoner.state_dict().items():
                if not torch.equal(tensor, source_state['reasoner.' + key]):
                    raise RuntimeError(f'{study}: frozen baseline changed at {key}')
    base_state = models['control'].reasoner.state_dict()
    for study, model in models.items():
        for key, value in base_state.items():
            if key.startswith(('res', 'channel_reducer')):
                if not torch.equal(value, model.reasoner.state_dict()[key]):
                    raise RuntimeError(f'{study}: frozen perception changed at {key}')
    cache = FrozenFeatureCache(models['program'], dataset_root, 'val', cache_root,
                               device, workers=workers)
    modes = [*STUDIES, 'no_program', 'uniform_program', 'swapped_program']
    predictions = {mode: [] for mode in modes}
    answers, configurations, alphas = [], [], []
    auxiliary_modes = ('program', 'program-no-ssl', 'program-static',
                       'uniform_program')
    ssl_totals = {s: {m: 0. for m in ('positive_energy', 'conditional_nce',
                                     'loss', 'retrieval_accuracy')}
                  for s in auxiliary_modes}
    completion_correct = {s: [] for s in auxiliary_modes}
    total = 0
    score_deltas = {m: [] for m in modes if m not in STUDIES}
    stability_guaranteed, stability_unchanged, perturbation_ranges = [], [], []
    signature_distances = []
    base_scores, control_scores = [], []
    program_scores = {s: [] for s in STUDIES if s != 'control'}
    ratio_scores = {s: [] for s in STUDIES if s != 'control'}
    intervention_scores = {m: [] for m in ('uniform_program', 'swapped_program')}
    ratio_interventions = {m: [] for m in intervention_scores}
    with torch.inference_mode():
        for config_index in range(7):
            indices = torch.where(cache.configurations == config_index)[0]
            for start in range(0, len(indices), 128):
                ids = indices[start:start + 128]
                batch = len(ids)
                features = cache.features[ids.to(device)].flatten(0, 1)
                labels, configs = cache.answers[ids], cache.configurations[ids]
                base = models['program'].predict_features(features, batch, intervention='no_program')
                scores = {'no_program': base}
                for mode in modes:
                    model = models[mode] if mode in STUDIES else models['program']
                    intervention = None if mode in STUDIES else mode
                    if mode == 'control':
                        scores[mode] = model.predict_features(features, batch)
                        control_scores.append(scores[mode].cpu().numpy())
                    elif mode != 'no_program':
                        energy = program_energy(model, features, batch, intervention)
                        scores[mode] = base + model.program_weight * energy
                        if mode in program_scores:
                            program_scores[mode].append(energy.cpu().numpy())
                            ratio = program_energy(model, features, batch,
                                                   verification_mode='support_evidence_ratio')
                            ratio_scores[mode].append(ratio.cpu().numpy())
                            if total == 0:
                                model.verification_mode = 'support_evidence_ratio'
                                native_ratio = model.predict_features(features, batch)
                                model.verification_mode = 'posterior'
                                if not torch.equal(native_ratio, base + model.program_weight * ratio):
                                    raise RuntimeError(f'{mode}: evidence ratio factorization differs')
                        else:
                            intervention_scores[mode].append(energy.cpu().numpy())
                            ratio = program_energy(model, features, batch, intervention,
                                                   verification_mode='support_evidence_ratio')
                            ratio_interventions[mode].append(ratio.cpu().numpy())
                        if total == 0:
                            native = model.predict_features(features, batch, intervention=intervention)
                            if not torch.equal(native, scores[mode]):
                                raise RuntimeError(f'{mode}: energy factorization changes native scores')
                    predictions[mode].append(scores[mode].argmin(1).cpu().numpy())
                    if mode not in STUDIES:
                        score_deltas[mode].append(float((scores[mode] - scores['program']).abs().mean()))
                alphas.append(models['program'].compile(features[:, :3]).cpu().numpy())
                base_scores.append(base.cpu().numpy())
                base = scores['no_program']
                full = scores['program']
                margin = base.topk(2, largest=False).values.diff(dim=1)[:, 0]
                correction = full - base
                correction_range = correction.max(1).values - correction.min(1).values
                guaranteed = margin > correction_range + 1e-6
                unchanged = full.argmin(1) == base.argmin(1)
                if torch.any(guaranteed & ~unchanged):
                    raise RuntimeError('rank-stability bound violated')
                stability_guaranteed.append(guaranteed.cpu().numpy())
                stability_unchanged.append(unchanged.cpu().numpy())
                perturbation_ranges.append(correction_range.cpu().numpy())
                signatures = operator_energy(models['program'].operators(features[:, 6:8]),
                                             features[:, 8:])
                signatures = signatures - signatures.mean(-1, keepdim=True)
                distances = torch.cdist(signatures.float(), signatures.float()) / 8 ** .5
                pairs = torch.triu_indices(6, 6, offset=1, device=device)
                signature_distances.append(distances[:, pairs[0], pairs[1]].cpu().numpy())
                answers.append(labels.numpy())
                configurations.append(configs.numpy())
                total += batch
        # The auxiliary generative task needs its own target-safe views, not
        # the classifier's eight-observed-context fallback from Raven(val).
        auxiliary_set = KnownRowValidation(dataset_root)
        auxiliary_count = 0
        for images, _, configs in loader(auxiliary_set, 128, workers,
                                         torch.Generator().manual_seed(12345)):
            images = normalize(images.to(device, non_blocking=True))
            configs = configs.to(device)
            features = models['program'].features(images)
            ids = torch.arange(len(features), device=device)
            repeated_configs = configs.repeat_interleave(2)
            mask = ((ids[:, None] % 2 != ids[None, :] % 2) |
                    (repeated_configs[:, None] != repeated_configs[None, :]))
            pixels = images.flatten(0, 1)[:, 5].flatten(1)
            duplicates = torch.cdist(pixels.float(), pixels.float(), p=0).eq(0)
            eye = torch.eye(len(features), device=device, dtype=torch.bool)
            mask |= duplicates & ~eye
            for study in auxiliary_modes:
                model = models[study] if study in models else models['program']
                intervention = None if study in models else study
                values, retrieved = completion_metrics(model, features, mask, intervention,
                                                        return_correct=True)
                completion_correct[study].append(retrieved.reshape(-1, 2).cpu().numpy())
                if auxiliary_count == 0 and study in models:
                    native_loss, _ = model.ssl(images, configs)
                    if float(native_loss) != values['loss']:
                        raise RuntimeError(f'{study}: auxiliary loss decomposition differs')
                for metric, value in values.items():
                    ssl_totals[study][metric] += value * len(images)
            auxiliary_count += len(images)
        if auxiliary_count != 14000:
            raise RuntimeError('incomplete target-safe auxiliary validation')
    answers = np.concatenate(answers)
    configurations = np.concatenate(configurations)
    correct = {mode: np.concatenate(p) == answers for mode, p in predictions.items()}
    alpha = np.concatenate(alphas)
    guaranteed = np.concatenate(stability_guaranteed)
    unchanged = np.concatenate(stability_unchanged)
    ranges = np.concatenate(perturbation_ranges)
    distances = np.concatenate(signature_distances)
    base_scores = np.concatenate(base_scores)
    control_scores = np.concatenate(control_scores)
    energies = {s: np.concatenate(p) for s, p in program_scores.items()}
    ratios = {s: np.concatenate(p) for s, p in ratio_scores.items()}
    interventions = {m: np.concatenate(p) for m, p in intervention_scores.items()}
    ratio_interventions = {m: np.concatenate(p) for m, p in ratio_interventions.items()}
    calibration = {'selection_split': 'val', 'weights': list(CALIBRATION_WEIGHTS),
                   'verification_modes': list(VERIFICATION_MODES),
                   'verification_choice_added_after_validation_only_diagnosis': True,
                   'zero_weight_excluded': True, 'checkpoint_selected_before_calibration': True,
                   'studies': {}}
    calibrated_correct = {'control': control_scores.argmin(1) == answers,
                          'no_program': base_scores.argmin(1) == answers}
    calibrated_scores = {}
    for study, energy in energies.items():
        choices = []
        for verification in VERIFICATION_MODES:
            verification_energy = energy if verification == 'posterior' else ratios[study]
            for weight in CALIBRATION_WEIGHTS:
                observed = (base_scores + weight * verification_energy).argmin(1) == answers
                choices.append({'weight': weight, 'verification': verification,
                                **summarize(observed, configurations)})
        chosen = max(choices, key=lambda r: (r['correct'], -abs(r['weight'] - .1)))
        energy = energy if chosen['verification'] == 'posterior' else ratios[study]
        calibrated_scores[study] = base_scores + chosen['weight'] * energy
        calibrated_correct[study] = calibrated_scores[study].argmin(1) == answers
        calibration['studies'][study] = {'chosen_weight': chosen['weight'],
                                        'chosen_verification': chosen['verification'],
                                        'validation_grid': choices,
                                        'best_epoch': load_study(root, study, device)[1]['epoch']}
    weight = calibration['studies']['program']['chosen_weight']
    verification = calibration['studies']['program']['chosen_verification']
    for mode, energy in interventions.items():
        if verification == 'support_evidence_ratio':
            energy = ratio_interventions[mode]
        calibrated_correct[mode] = (base_scores + weight * energy).argmin(1) == answers
    calibrated = calibrated_scores['program']
    completion_correct = {s: np.concatenate(v) for s, v in completion_correct.items()}
    perturbation = calibrated - base_scores
    perturbation_range = np.ptp(perturbation, axis=1)
    base_sorted = np.sort(base_scores, axis=1)
    guaranteed_calibrated = base_sorted[:, 1] - base_sorted[:, 0] > perturbation_range + 1e-6
    unchanged_calibrated = calibrated.argmin(1) == base_scores.argmin(1)
    if np.any(guaranteed_calibrated & ~unchanged_calibrated):
        raise RuntimeError('calibrated rank-stability bound violated')
    if unchanged_calibrated.all():
        raise RuntimeError('calibrated program is inert on validation')
    report = {'split': 'val', 'perception_identical_across_studies': True,
              'evaluation_cache_fingerprint': cache.fingerprint,
              'evaluation_cache_encoder_sha256': cache.metadata['encoder_sha256'],
              'intervention_swap_within_public_configuration_and_component': True,
              'accuracy': {m: summarize(c, configurations) for m, c in correct.items()},
              'paired_effects': {m: paired_effect(correct['program'], c)
                                 for m, c in correct.items() if m != 'program'},
              'candidate_free_completion_loss': {s: values['loss'] / auxiliary_count
                                                 for s, values in ssl_totals.items()},
              'candidate_free_completion_metrics': {
                  s: {m: value / auxiliary_count for m, value in values.items()}
                  for s, values in ssl_totals.items()},
              'candidate_free_completion_paired_effects': {
                  s: completion_effect(completion_correct['program'], values)
                  for s, values in completion_correct.items() if s != 'program'},
              'auxiliary_validation_target_safe_views': True,
              'auxiliary_validation_count': auxiliary_count,
              'alpha_mean': alpha.mean(0).tolist(), 'alpha_std': alpha.std(0).tolist(),
              'intervention_mean_absolute_score_change': {m: float(np.mean(v))
                                                         for m, v in score_deltas.items()},
              'rank_stability': {'guaranteed_unchanged_count': int(guaranteed.sum()),
                                 'observed_unchanged_count': int(unchanged.sum()),
                                 'changed_answer_count': int((~unchanged).sum()),
                                 'total': len(guaranteed),
                                 'correction_range_mean': float(ranges.mean()),
                                 'correction_range_max': float(ranges.max()),
                                 'bound_verified_for_every_question': True},
              'operator_candidate_centered_energy_signature_distance': {
                  'mean_pairwise_rms': float(distances.mean()),
                  'minimum_pair_mean_rms': float(distances.mean(0).min()),
                  'pair_mean_rms': distances.mean(0).tolist()},
              'calibrated_accuracy': {m: summarize(c, configurations)
                                      for m, c in calibrated_correct.items()},
              'calibrated_paired_effects': {m: paired_effect(calibrated_correct['program'], c)
                                            for m, c in calibrated_correct.items() if m != 'program'},
              'calibrated_rank_stability': {
                  'chosen_weight': weight,
                  'chosen_verification': verification,
                  'guaranteed_unchanged_count': int(guaranteed_calibrated.sum()),
                  'changed_answer_count': int((~unchanged_calibrated).sum()),
                  'total': len(unchanged_calibrated),
                  'bound_verified_for_every_question': True},
              'scope': 'Frozen interventions measure immediate dependence, not retraining or semantic rule discovery.'}
    for study in STUDIES:
        directory = root / study
        checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
        history = [json.loads(s) for s in (directory / 'history.jsonl').read_text().splitlines()]
        expected = history[checkpoint['epoch'] - 1]['validation']
        if report['accuracy'][study]['correct'] != expected['correct'] or (
                report['accuracy'][study]['by_configuration'] != expected['by_configuration']):
            raise RuntimeError(f'{study}: audit does not reproduce selected checkpoint validation')
    (root / 'validation_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (root / 'calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    lock = {'selection_split': 'val', 'new_test_not_used_for_model_selection': True,
            'studies': {s: {name: digest(root / s / f'{name}.pt')
                            for name in ('best', 'final')} for s in STUDIES},
            'config_sha256': {s: digest(root / s / 'config.json') for s in STUDIES},
            'model_source_sha256': digest(Path(__file__).with_name('model.py')),
            'evaluation_source_sha256': digest(__file__),
            'evaluation_dependencies_sha256': evaluation_hashes(),
            'calibration_sha256': digest(root / 'calibration.json'),
            'validation_audit_sha256': digest(root / 'validation_audit.json')}
    path = root / 'test_lock.json'
    if path.exists() and json.loads(path.read_text()) != lock:
        raise RuntimeError('an existing test lock differs')
    path.write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def evaluate_test(root, dataset_root, device, workers, cache_root):
    lock = json.loads((root / 'test_lock.json').read_text())
    if digest(Path(__file__).with_name('model.py')) != lock['model_source_sha256'] or (
            digest(__file__) != lock['evaluation_source_sha256']):
        raise RuntimeError('evaluation source changed after lock')
    if digest(root / 'validation_audit.json') != lock['validation_audit_sha256']:
        raise RuntimeError('validation audit changed after lock')
    if digest(root / 'calibration.json') != lock['calibration_sha256']:
        raise RuntimeError('calibration changed after lock')
    if evaluation_hashes() != lock['evaluation_dependencies_sha256']:
        raise RuntimeError('evaluation dependency changed after lock')
    calibration = json.loads((root / 'calibration.json').read_text())
    for study in STUDIES:
        if digest(root / study / 'config.json') != lock['config_sha256'][study]:
            raise RuntimeError('configuration changed after lock')
        for name in ('best', 'final'):
            if digest(root / study / f'{name}.pt') != lock['studies'][study][name]:
                raise RuntimeError('checkpoint changed after lock')
    report = {'selection_split': 'val', 'source_training_budget_epochs': 20,
              'additional_training_epochs': 8, 'final_lineage_epochs': 25,
              'total_training_budget_epochs': 28, 'studies': {}}
    cache = None
    for study in STUDIES:
        if digest(root / study / 'config.json') != lock['config_sha256'][study]:
            raise RuntimeError('configuration changed after lock')
        report['studies'][study] = {}
        for name in ('best', 'final'):
            if digest(root / study / f'{name}.pt') != lock['studies'][study][name]:
                raise RuntimeError('checkpoint changed after lock')
            model, checkpoint, _ = load_study(root, study, device, name)
            if study != 'control':
                model.program_weight = calibration['studies'][study]['chosen_weight']
                model.verification_mode = calibration['studies'][study]['chosen_verification']
            if cache is None:
                # Test images and answer indices are first opened after every
                # checkpoint, source and validation calibration has been locked.
                cache = FrozenFeatureCache(model, dataset_root, 'test', cache_root,
                                           device, workers=workers)
                baseline = ComponentProgram(use_program=False).to(device).eval()
                source_path = json.loads((root / study / 'config.json').read_text())['baseline']
                baseline.load_baseline(torch.load(source_path, map_location=device,
                                                   weights_only=False)['model'])
                report['baseline_test'] = cache.score(baseline,
                    generator=torch.Generator().manual_seed(12347))
            result = {'additional_epoch': checkpoint['epoch'],
                      'checkpoint_lineage_epoch': checkpoint['source_epoch'] + checkpoint['epoch'],
                      'calibrated_program_weight': model.program_weight if study != 'control' else None,
                      'verification_mode': model.verification_mode if study != 'control' else None,
                      'checkpoint_sha256': lock['studies'][study][name],
                      'test': cache.score(model,
                                    generator=torch.Generator().manual_seed(12347))}
            if study == 'program' and name == 'best':
                native = native_score(model, dataset_root, 'test', device, workers=workers,
                                      generator=torch.Generator().manual_seed(12347))
                if native != result['test']:
                    raise RuntimeError('full native/cache test evaluation differs')
                result['native_test_cache_parity_verified'] = True
            report['studies'][study][name] = result
            print(json.dumps({'study': study, 'checkpoint': name, **result}), flush=True)
    full = report['studies']['program']['best']['test']['accuracy_micro']
    report['requested_70_percent_gate_met'] = full >= 70.
    report['evaluation_cache_fingerprint'] = cache.fingerprint
    (root / 'test_result.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--experiment-root', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--cache-root')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    root, device = Path(args.experiment_root), torch.device('cuda:0')
    cache_root = args.cache_root or root.parent / 'frozen-evaluation-feature-cache'
    if args.test:
        evaluate_test(root, args.dataset_root, device, args.workers, cache_root)
    else:
        audit(root, args.dataset_root, device, args.workers, cache_root)


if __name__ == '__main__':
    main()
