"""Verify sealed suite artifacts and compare the preregistered six jobs.

Does not load data or checkpoints into a model, select checkpoints, or open an
unmeasured test set. Missing artifacts remain pending rather than becoming zero
scores. This report does not decide whether the research goal is complete.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from sspredrnet.data import CONFIGS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def metric(value):
    if value['total'] != 14000 or not 0 <= value['correct'] <= 14000:
        raise RuntimeError('complete 14000-puzzle evaluation required')
    by_configuration = value['by_configuration']
    if set(by_configuration) != set(CONFIGS):
        raise RuntimeError('all seven configuration results required')
    if any(not math.isfinite(x) or not 0 <= x <= 100 for x in by_configuration.values()):
        raise RuntimeError('invalid configuration accuracy')
    implied = sum(round(x * 20) for x in by_configuration.values())
    if implied != value['correct'] or any(abs(x * 20 - round(x * 20)) > 1e-6
                                        for x in by_configuration.values()):
        raise RuntimeError('configuration percentages disagree with correct count')
    accuracy = 100 * value['correct'] / 14000
    if abs(value['accuracy_macro'] - accuracy) > 1e-8:
        raise RuntimeError('macro accuracy disagrees with balanced full split')
    return {'correct': value['correct'], 'total': 14000, 'accuracy': accuracy,
            'by_configuration': by_configuration}


def validate_training(run, source, expected_epochs, expected_seed, variant):
    config, seal = read(run / 'config.json'), read(run / 'training_complete.json')
    if (config['seed'] != expected_seed or config['epochs'] != expected_epochs
            or config['selection_split'] != 'val' or config['batch_size'] != 128
            or config['device'] != 'mps'
            or seal['epochs'] != expected_epochs
            or config['total_training_selection_budget_epochs'] != 44
            or config['source_sha256'] != seal['source_sha256']
            or seal['config_sha256'] != digest(run / 'config.json')):
        raise RuntimeError(f'protocol/seal mismatch: {run.name}')
    if config.get('experiment_variant', 'primary') != variant:
        raise RuntimeError('unexpected contribution variant')
    native = config['method'] == 'original-prb-continuation'
    reference = run.parent / ('reference-native-validation.json' if native else 'reference-anchor-validation.json')
    if digest(reference) != config['platform_validation_sha256']:
        raise RuntimeError('pretraining platform reference changed')
    initial = metric(read(run / 'initial_validation.json'))
    reference_count = config['platform_baseline_validation_correct'] if native else config['platform_anchor_validation_correct']
    if initial['correct'] != reference_count or reference_count != read(reference)['mps']['correct']:
        raise RuntimeError('initial validation did not replay the locked reference')
    if native:
        if (not config['cnn_and_batchnorm_frozen'] or not config['only_original_prbs_trainable']
                or not config['unlabeled_answer_candidates_used_as_negatives']):
            raise RuntimeError('native continuation recipe changed')
    else:
        if (config['method'] != 'structured-object-support-pav'
                or not config['baseline_and_batchnorm_frozen'] or not config['new_pixel_encoder_trainable']
                or config['answer_candidates_in_adaptation'] or config['answer_labels_or_xml_in_adaptation']
                or config['bidirectional_row_task'] or config['collapse_penalty']
                or config['loss_weights'] != {'ranking': 1., 'completion': .1,
                    'object_masking': 0. if variant == 'no_object_masking' else .1}):
            raise RuntimeError('declared self-supervised objective changed')
    for relative, expected in seal['source_sha256'].items():
        if digest(source / relative) != expected:
            raise RuntimeError(f'trained source changed: {relative}')
    for checkpoint in ('best', 'final'):
        if digest(run / f'{checkpoint}.pt') != seal['checkpoint_sha256'][checkpoint]:
            raise RuntimeError(f'checkpoint changed: {run.name}/{checkpoint}')
    history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
    if ([item['epoch'] for item in history] != list(range(1, expected_epochs + 1))
            or any(item['training_count'] != 42000 for item in history)):
        raise RuntimeError('incomplete training sequence')
    selected = max(history, key=lambda item: item['validation']['accuracy_macro'])['epoch']
    if selected != seal['best_epoch']:
        raise RuntimeError('checkpoint was not the first validation maximum')
    return config, seal, history


def collect(run_root):
    manifest = read(run_root / 'launch_manifest.json')
    expected = {
        'primary': {'seeds': [12345, 12346, 12347], 'epochs_each': 8},
        'original': {'seeds': [12345], 'epochs_each': 16},
        'contribution_controls': {'variants': ['no_object_masking', 'static_parameters'],
                                  'seed': 12345, 'epochs_each': 8}}
    if manifest['preregistered_jobs'] != expected:
        raise RuntimeError('this collector requires the declared six-job protocol')
    pending, jobs = {}, {}
    specs = [('primary', seed, 8) for seed in (12345, 12346, 12347)]
    specs += [('original', 12345, 16), ('no_object_masking', 12345, 8), ('static_parameters', 12345, 8)]
    for variant, seed, epochs in specs:
        name = f'{variant}-seed{seed}'
        run = run_root / name
        native = variant == 'original'
        required = ['training_complete.json', 'config.json', 'initial_validation.json', 'best.pt', 'final.pt', 'history.jsonl']
        required += ['evaluation.json'] if native else ['val_evaluation.json', 'test_evaluation.json']
        absent = [item for item in required if not (run / item).is_file()]
        rejected = not native and (run / 'test_deferred.txt').exists()
        if rejected and (native or variant != 'primary' or absent != ['test_evaluation.json']):
            raise RuntimeError('test deferral requires a complete primary validation audit')
        if absent:
            pending[name] = {'missing': absent, 'test_deferred': rejected,
                             'phase': 'validation_rejected' if rejected else 'training_or_evaluation_pending'}
            if not rejected:
                continue
        source = run_root / ('source-primary' if variant in ('primary', 'original')
                             else 'source-' + variant)
        if native and manifest.get('native_source_directory'):
            source = run_root / manifest['native_source_directory']
            migration = read(run / 'numerical_repair.json')
            prior = Path(migration['from_run'])
            if (migration['repair'] != 'exact_forward_l2_zero_subgradient'
                    or digest(prior / 'last.pt') != migration['original_last_sha256']
                    or digest(prior / 'config.json') != migration['original_config_sha256']
                    or migration['modified_source_files'] != ['sspredrnet/model.py']
                    or not migration['actual_failed_batch_bitwise_loss_and_finite_backward_verified']):
                raise RuntimeError('native numerical migration lineage changed')
        # Native continuation has no experiment_variant field.
        config, seal, history = validate_training(run, source, epochs, seed,
                                                  'primary' if native else variant)
        recorded = None if rejected else read(run / ('evaluation.json' if native else 'test_evaluation.json'))
        if recorded is not None and (recorded['seal'] != seal or recorded['device'] != config['device']):
            raise RuntimeError('test evaluation seal/device mismatch')
        validation = None if native else read(run / 'val_evaluation.json')
        if validation is not None and (validation['seal'] != seal or validation['device'] != config['device']):
            raise RuntimeError('validation audit seal/device mismatch')
        if validation is not None:
            for checkpoint in ('best', 'final'):
                if validation['checkpoints'][checkpoint]['anchor']['correct'] != config['platform_anchor_validation_correct']:
                    raise RuntimeError('validation changed its immutable attention anchor')
            passed = validation['checkpoints']['best']['nonregression_and_active_branch']
            if rejected and passed:
                raise RuntimeError('test was declared validation-rejected despite a passed gate')
            if variant == 'primary' and not rejected and not passed:
                raise RuntimeError('primary test was opened despite its failed validation gate')
        row = {'variant': variant, 'seed': seed, 'adaptation_epochs': epochs,
               'total_selection_budget_epochs': 44,
               'source_lineage_epoch': config['source_epoch'],
               'trainable_parameters': config['trainable_parameters'],
               'candidate_negatives_in_adaptation': native,
               'phase': 'validation_rejected' if rejected else 'evaluated', 'checkpoints': {}}
        for checkpoint, expected_epoch in (('best', seal['best_epoch']), ('final', epochs)):
            result = (validation if rejected else recorded)['checkpoints'][checkpoint]
            if result['epoch'] != expected_epoch:
                raise RuntimeError('evaluation checkpoint epoch mismatch')
            if result['lineage_epoch'] != config['source_epoch'] + expected_epoch:
                raise RuntimeError('evaluation parameter lineage mismatch')
            observed_validation = result['validation'] if native else validation['checkpoints'][checkpoint]['full']
            if metric(observed_validation) != metric(history[expected_epoch - 1]['validation']):
                raise RuntimeError('selected validation audit did not replay training history')
            observed_test = None if rejected else result['test'] if native else result['full']
            entry = {'epoch': expected_epoch, 'lineage_epoch': result['lineage_epoch'],
                     'validation': metric(observed_validation),
                     'test': None if rejected else metric(observed_test)}
            if rejected:
                entry['validation_gate_passed'] = result['nonregression_and_active_branch']
                entry['delta_vs_anchor_validation_pp'] = (entry['validation']['accuracy'] -
                                                         metric(result['anchor'])['accuracy'])
            if not native and not rejected:
                anchor = metric(result['anchor'])
                if (entry['test']['correct'] - anchor['correct'] != result['helped'] - result['hurt']
                        or result['helped'] + result['hurt'] > result['changed_answers_vs_anchor']
                        or not 0 <= result['helped'] <= 14000 or not 0 <= result['hurt'] <= 14000
                        or not 0 <= result['changed_answers_vs_anchor'] <= 14000
                        or not 0 <= result['changed_answers_vs_swapped_support'] <= 14000):
                    raise RuntimeError('paired change counts disagree with reported accuracy')
                active_and_preserved = (entry['test']['correct'] >= anchor['correct'] and
                    result['changed_answers_vs_anchor'] > 0 and result['changed_answers_vs_swapped_support'] > 0)
                if result['nonregression_and_active_branch'] != active_and_preserved:
                    raise RuntimeError('declared branch acceptance disagrees with observed counts')
                entry.update({'anchor_test': anchor,
                              'delta_vs_anchor_pp': entry['test']['accuracy'] - anchor['accuracy'],
                              'helped': result['helped'], 'hurt': result['hurt'],
                              'changed_answers_vs_anchor': result['changed_answers_vs_anchor'],
                              'changed_answers_vs_swapped_support': result['changed_answers_vs_swapped_support'],
                              'support_swap_test': metric(result['swapped_support']),
                              'gate': result['gate'],
                              'nonregression_and_active_branch': result['nonregression_and_active_branch']})
            row['checkpoints'][checkpoint] = entry
        jobs[name] = row
    report = {'all_required_results_present': not pending, 'pending': pending, 'jobs': jobs,
              'execution_platform': 'Metal', 'selection_split': 'val',
              'historical_cuda_discrepancy_fully_identified': False,
              'historical_attention_primary_test_accuracy': 71.23571428571428,
              'historical_original_continuation_final_test_accuracy': 71.41428571428571,
              'historical_official_three_block_reproduction_selected_on_test': True,
              'adaptation_seeds_share_one_foundation': True,
              'new_method_excludes_candidates_but_native_control_uses_unlabeled_candidate_negatives': True,
              'all_results_present_is_not_proof_of_research_goal_completion': True}
    if manifest.get('native_source_directory'):
        report['native_numerical_repair'] = read(run_root / 'original-seed12345/numerical_repair.json')
    names = [f'primary-seed{seed}' for seed in (12345, 12346, 12347)]
    if all(name in jobs and jobs[name]['phase'] == 'evaluated' for name in names):
        report['primary_three_seed'] = {}
        for checkpoint in ('best', 'final'):
            entries = [jobs[name]['checkpoints'][checkpoint] for name in names]
            accuracies = [entry['test']['accuracy'] for entry in entries]
            report['primary_three_seed'][checkpoint] = {
                'mean_accuracy': statistics.mean(accuracies), 'sample_std_pp': statistics.stdev(accuracies),
                'all_seeds_match_or_exceed_same_platform_anchor': all(entry['delta_vs_anchor_pp'] >= 0 for entry in entries),
                'all_seeds_have_active_support_branch': all(entry['changed_answers_vs_anchor'] > 0 and
                    entry['changed_answers_vs_swapped_support'] > 0 for entry in entries)}
    primary = jobs.get('primary-seed12345')
    if primary and primary['phase'] == 'evaluated':
        report['matched_primary_comparisons'] = {}
        for variant in ('original', 'no_object_masking', 'static_parameters'):
            control = jobs.get(f'{variant}-seed12345')
            if control and control['phase'] == 'evaluated':
                report['matched_primary_comparisons'][variant] = {
                    checkpoint: primary['checkpoints'][checkpoint]['test']['accuracy'] -
                    control['checkpoints'][checkpoint]['test']['accuracy'] for checkpoint in ('best', 'final')}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.run_root)
    # Preserve every observation; caller chooses a new path for a later report.
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'all_required_results_present': report['all_required_results_present'],
                      'verified_jobs': list(report['jobs']), 'pending_jobs': list(report['pending'])}))


if __name__ == '__main__':
    main()
