"""Audit full-run records and summarize completed studies without filling gaps."""
import hashlib
import json
from pathlib import Path
import statistics


BASELINE = '8a383fcc609127a7c35bfb9c6a2ffe064ce5285f66d8525b3668385ec915928c'
STUDIES = ('seed12345', 'seed12346', 'seed12347', 'random_negatives',
           'tokenwise_mlp', 'completion_energy_training', 'discrete_support')


def read(path):
    return json.loads(path.read_text())


def verify_scores(result):
    for name in ('full', 'baseline', 'swapped_support'):
        score = result[name]
        assert score['total'] == 14000
        assert len(score['by_configuration']) == 7
        counts = [accuracy * 20 for accuracy in score['by_configuration'].values()]
        assert all(abs(count - round(count)) < 1e-6 for count in counts)
        assert sum(round(count) for count in counts) == score['correct']
        assert abs(score['accuracy_macro'] - 100 * score['correct'] / 14000) < 1e-8
    assert result['helped'] - result['hurt'] == result['full']['correct'] - result['baseline']['correct']
    assert result['helped'] + result['hurt'] <= result['changed_answers_vs_baseline']
    accepted = (result['full']['correct'] >= result['baseline']['correct']
                and result['changed_answers_vs_baseline'] > 0
                and result['changed_answers_vs_swapped_support'] > 0)
    assert accepted == result['nonregression_and_active_branch']
    assert result['support_swap_preserves_baseline_scoring']


def audit(directory):
    config, seal = read(directory / 'config.json'), read(directory / 'training_complete.json')
    history = [json.loads(line) for line in (directory / 'history.jsonl').read_text().splitlines()]
    assert config['baseline_sha256'] == seal['baseline_sha256'] == BASELINE
    assert config['source_sha256'] == seal['source_sha256']
    assert config['frozen_reasoner_sha256'] == seal['frozen_reasoner_sha256']
    assert config['epochs'] == seal['epochs'] == 8
    assert [row['epoch'] for row in history] == list(range(1, 9))
    assert all(row['training_puzzles'] == 42000 for row in history)
    best = max(history, key=lambda row: row['validation']['accuracy_macro'])
    assert best['epoch'] == seal['best_epoch']
    validation = read(directory / 'val_evaluation.json')
    assert validation['seal'] == seal and validation['split'] == 'val'
    for name in ('best', 'final'):
        result = validation['checkpoints'][name]
        verify_scores(result)
        epoch = best['epoch'] if name == 'best' else 8
        assert result['epoch'] == epoch
        assert result['full']['correct'] == history[epoch - 1]['validation']['correct']
        checkpoint = directory / f'{name}.pt'
        if checkpoint.exists():
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == seal['checkpoint_sha256'][name]
    outcome = {'seed': config['seed'], 'variant': config.get('experiment_variant', 'primary'),
               'source_epoch': config['source_epoch'],
               'selection_budget_epochs': config['total_training_selection_budget_epochs'],
               'training_seconds': sum(row['wall_seconds'] for row in history),
               'trainable_runtime_source_sha256': seal['source_sha256']['attention_ssl/model.py'],
               'best_epoch': best['epoch'], 'validation': validation['checkpoints'],
               'checkpoint_files_verified': all((directory / f'{name}.pt').exists() for name in ('best', 'final'))}
    if (directory / 'test_evaluation.json').exists():
        assert validation['checkpoints']['best']['nonregression_and_active_branch']
        test = read(directory / 'test_evaluation.json')
        assert test['seal'] == seal and test['split'] == 'test'
        for name, result in test['checkpoints'].items():
            verify_scores(result)
            assert result['epoch'] == validation['checkpoints'][name]['epoch']
        outcome['test'] = test['checkpoints']
        outcome['state'] = 'complete'
    else:
        outcome['state'] = ('test_denied_by_validation' if not validation['checkpoints']['best']['nonregression_and_active_branch']
                            else 'test_missing')
    return outcome


def main():
    root = Path(__file__).resolve().parent / 'results'
    report = {'studies': {}, 'missing': [], 'all_main_seeds_verified_nonregression': False}
    for name in STUDIES:
        directory = root / name
        required = ('config.json', 'training_complete.json', 'history.jsonl', 'val_evaluation.json')
        if not all((directory / file).exists() for file in required):
            report['missing'].append(name)
            continue
        report['studies'][name] = audit(directory)
    main_names = [f'seed{seed}' for seed in (12345, 12346, 12347)]
    if all(name in report['studies'] and report['studies'][name]['state'] == 'complete' for name in main_names):
        studies = [report['studies'][name] for name in main_names]
        report['all_main_seeds_verified_nonregression'] = all(
            study['test']['best']['nonregression_and_active_branch'] for study in studies)
        accuracies = [study['test']['best']['full']['accuracy_macro'] for study in studies]
        report['primary_test_best_mean'] = statistics.mean(accuracies)
        report['primary_test_best_sample_std'] = statistics.stdev(accuracies)
        report['all_main_final_nonregression'] = all(
            study['test']['final']['nonregression_and_active_branch'] for study in studies)
        report['primary_frozen_baseline'] = studies[0]['test']['best']['baseline']
        assert all(study['test']['best']['baseline'] == report['primary_frozen_baseline'] for study in studies)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'completed_record_sets': list(report['studies']), 'missing': report['missing'],
                      'all_main_seeds_verified_nonregression': report['all_main_seeds_verified_nonregression']}))


if __name__ == '__main__':
    main()
