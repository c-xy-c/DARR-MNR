"""Apply a recorded user policy while executing an unchanged training export.

Only evaluation admission changes. Model code, objectives, source seals,
validation selection and best/final checkpoints remain exactly as trained.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def allows_test(report):
    if report['full']['total'] != 14000:
        raise RuntimeError('complete validation required')
    return (report['full']['correct'] >= 9800
            and report['changed_answers_vs_anchor'] > 0
            and report['changed_answers_vs_swapped_support'] > 0)


def policy_record(root):
    policy = read(root / 'evaluation_policy.json')
    if (policy['schema_version'] != 1 or policy['minimum_accuracy_percent'] != 70.
            or policy['selection_split'] != 'val'
            or not policy['active_support_branch_required']
            or not policy['sealed_training_sources_unchanged']
            or not policy['decided_before_first_revision_test']
            or policy['run_root'] != str(root.resolve())):
        raise RuntimeError('unexpected external evaluation policy')
    if digest(root / 'launch_manifest.json') != policy['original_launch_manifest_sha256']:
        raise RuntimeError('original experiment launch record changed')
    for name, expected in policy['policy_source_sha256'].items():
        if digest(root / 'suite.policy70/source' / name) != expected:
            raise RuntimeError(f'policy execution source changed: {name}')
    if digest(__file__) != policy['policy_source_sha256']['evaluate_sealed_policy.py']:
        raise RuntimeError('execute the sealed external policy evaluator')
    return policy, digest(root / 'evaluation_policy.json')


def validate_completed(run, source, policy):
    config, seal = read(run / 'config.json'), read(run / 'training_complete.json')
    variant = config.get('experiment_variant', 'primary')
    expected_source = run.parent / ('source-' + variant)
    if source != expected_source.resolve():
        raise RuntimeError('wrong contribution source export')
    if (config['method'] != 'structured-object-support-pav'
            or seal['method'] != config['method'] or seal['schema_version'] != 1
            or config['architecture_revision'] != policy['architecture_revision']
            or config['epochs'] != 8 or seal['epochs'] != 8
            or config['batch_size'] != 128 or config['device'] != 'mps'
            or config['training_count'] != 42000 or config['validation_count'] != 14000
            or config['selection_split'] != 'val'
            or config['source_sha256'] != seal['source_sha256']
            or seal['config_sha256'] != digest(run / 'config.json')):
        raise RuntimeError('sealed training protocol changed or incomplete')
    if (variant not in ('primary', 'no_object_masking', 'static_parameters')
            or config['seed'] not in ((12345, 12346, 12347) if variant == 'primary' else (12345,))
            or run.name != f"{variant}-seed{config['seed']}"):
        raise RuntimeError('job is outside the fixed seed/contribution protocol')
    for name, expected in seal['source_sha256'].items():
        if digest(source / name) != expected:
            raise RuntimeError(f'trained source changed: {name}')
    for name in ('best', 'final'):
        if digest(run / f'{name}.pt') != seal['checkpoint_sha256'][name]:
            raise RuntimeError(f'{name} checkpoint changed')
    history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
    if ([row['epoch'] for row in history] != list(range(1, 9))
            or any(row['training_count'] != 42000 or row['validation']['total'] != 14000 for row in history)):
        raise RuntimeError('incomplete training/validation history')
    best_epoch = max(history, key=lambda row: row['validation']['accuracy_macro'])['epoch']
    if seal['best_epoch'] != best_epoch:
        raise RuntimeError('best is not the first complete validation maximum')
    return config, seal, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    args = parser.parse_args()
    run, source = args.run_dir.resolve(), args.source_root.resolve()
    policy, policy_hash = policy_record(run.parent)
    config, seal, history = validate_completed(run, source, policy)
    output = run / f'{args.split}_evaluation.json'
    if output.exists():
        raise FileExistsError('preserve recorded evaluation')
    if args.split == 'test':
        validation = read(run / 'val_evaluation.json')
        if (validation['seal'] != seal or validation['device'] != config['device']
                or validation['evaluation_policy_sha256'] != policy_hash):
            raise RuntimeError('validation audit does not match this external policy and training seal')
        for name, epoch in (('best', seal['best_epoch']), ('final', 8)):
            observed = validation['checkpoints'][name]
            if (observed['epoch'] != epoch or observed['full'] != history[epoch - 1]['validation']
                    or observed['anchor']['correct'] != config['platform_anchor_validation_correct']):
                raise RuntimeError('validation did not replay sealed history and frozen anchor')
        if config.get('experiment_variant', 'primary') == 'primary' and not allows_test(validation['checkpoints']['best']):
            raise RuntimeError('best failed the 70-percent/active-support validation criterion')
    sys.path.insert(0, str(source))
    # Imports must come from the training export, never today's cleaned model.
    evaluator = importlib.import_module('structured_ssl.evaluate')
    trainer = importlib.import_module('structured_ssl.train')
    model_module = importlib.import_module('structured_ssl.model')
    for module in (evaluator, trainer, model_module):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError('evaluation imported an unsealed runtime')
    if trainer.source_hashes() != seal['source_sha256']:
        raise RuntimeError('runtime fingerprint differs from training')
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = evaluator.resolve_device(config['device'])
    model = model_module.StructuredCompletion().to(device).eval()
    report = {'split': args.split, 'seal': seal, 'torch': str(torch.__version__),
              'device': str(device), 'checkpoints': {},
              'evaluation_policy_sha256': policy_hash,
              'external_evaluator_sha256': digest(__file__),
              'minimum_accuracy_percent': 70., 'trained_runtime_unchanged': True,
              'admission_changed_by_direct_user_request': True}
    for name, expected_epoch in (('best', seal['best_epoch']), ('final', 8)):
        checkpoint = torch.load(run / f'{name}.pt', map_location=device, weights_only=False)
        if checkpoint['epoch'] != expected_epoch or checkpoint['source_epoch'] != config['source_epoch']:
            raise RuntimeError('checkpoint lineage/selection differs from completed history')
        model.load_state_dict(checkpoint['model'], strict=True)
        if trainer.anchor_hash(model) != seal['frozen_anchor_sha256']:
            raise RuntimeError('checkpoint changed its frozen anchor')
        observed = evaluator.evaluate(model, config['dataset_root'], args.split, device, config['workers'])
        if args.split == 'val' and (observed['full'] != history[expected_epoch - 1]['validation']
                or observed['anchor']['correct'] != config['platform_anchor_validation_correct']):
            raise RuntimeError('complete validation failed exact history/anchor replay')
        report['checkpoints'][name] = {**observed, 'epoch': expected_epoch,
            'lineage_epoch': checkpoint['source_epoch'] + expected_epoch,
            'meets_70_percent_target': observed['full']['correct'] >= 9800,
            'accuracy_target_and_active_branch': allows_test(observed)}
        print(json.dumps({'checkpoint': name, **report['checkpoints'][name]}), flush=True)
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
