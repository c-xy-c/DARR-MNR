"""Resume the failed fixed suite in a new directory with explicit L2 lineage.

Only the native source's zero-distance backward changes. The model, optimizer,
scaler and RNG checkpoint bytes are copied unchanged. All pending v1 jobs retain
their original source exports and epoch budgets. Failed records stay untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def hashes(source):
    return {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*.py'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--repair-diagnostic', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    prior, root = args.from_run_root.resolve(), args.run_root.resolve()
    native_prior = prior / 'original-seed12345'
    if (prior / 'suite.launch/exit_code.txt').read_text().strip() != '1':
        raise RuntimeError('requires a recorded terminal failure')
    diagnosis = json.loads(args.repair_diagnostic.read_text())
    repair = diagnosis['failure']['stable_l2_same_batch_retry']
    if (not repair['same_loss_bitwise'] or not repair['finite_gradient_norm']
            or repair['source_sha256'] != digest(repository / 'sspredrnet/model.py')
            or diagnosis['formal_checkpoint_sha256'] != digest(native_prior / 'last.pt')
            or diagnosis['replay_epoch'] != 8):
        raise RuntimeError('repair lacks verified failing-batch evidence')
    root.mkdir(parents=True, exist_ok=False)
    meta = root / 'suite.launch'
    meta.mkdir()
    write(meta / 'process.json', {'pid': os.getpid(), 'python': sys.executable})
    def phase(value):
        (meta / 'phase.txt').write_text(value + '\n')
        print(json.dumps({'phase': value}), flush=True)
    environment = {**os.environ, 'PYTHONPATH': '.'}
    def run(command, source, log):
        with log.open('a') as stream:
            subprocess.run([sys.executable, '-u', *command], cwd=source, env=environment,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    try:
        phase('copying_failed_suite_records_and_sealing_native_repair')
        for name in ('source-primary', 'source-no_object_masking', 'source-static_parameters',
                     'primary-seed12345', 'original-seed12345'):
            shutil.copytree(prior / name, root / name)
        for name in ('reference-anchor-validation.json', 'reference-native-validation.json'):
            shutil.copyfile(prior / name, root / name)
        native_source = root / 'source-native-numerical-repair'
        shutil.copytree(prior / 'source-primary', native_source)
        shutil.copyfile(repository / 'sspredrnet/model.py', native_source / 'sspredrnet/model.py')
        native = root / 'original-seed12345'
        config = json.loads((native / 'config.json').read_text())
        old_hashes = config['source_sha256']
        new_hashes = {name: digest(native_source / name) for name in old_hashes}
        changed = [name for name in old_hashes if old_hashes[name] != new_hashes[name]]
        if changed != ['sspredrnet/model.py']:
            raise RuntimeError('repair changed more than the verified numerical primitive')
        write(native / 'numerical_repair.json', {
            'repair': 'exact_forward_l2_zero_subgradient', 'from_run': str(native_prior),
            'original_last_sha256': digest(native_prior / 'last.pt'),
            'original_config_sha256': digest(native_prior / 'config.json'),
            'checkpoint_model_optimizer_scaler_rng_bytes_unchanged': True,
            'resume_after_completed_epoch': 7, 'fixed_total_epochs': 16,
            'failed_partial_epoch_replayed_from_last_completed_checkpoint': True,
            'modified_source_files': changed, 'prior_source_sha256': old_hashes,
            'repaired_source_sha256': new_hashes,
            'actual_failed_batch_bitwise_loss_and_finite_backward_verified': True,
            'diagnostic_sha256': digest(args.repair_diagnostic)})
        shutil.copyfile(args.repair_diagnostic, root / 'native-repair-diagnostic.json')
        config['run_dir'], config['source_sha256'] = str(native), new_hashes
        (native / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        if digest(native / 'last.pt') != digest(native_prior / 'last.pt'):
            raise RuntimeError('copied native checkpoint changed')
        sources = {name: root / ('source-' + name) for name in ('primary', 'no_object_masking', 'static_parameters')}
        locked = {name: hashes(source) for name, source in sources.items()}
        locked['native'] = hashes(native_source)
        manifest = json.loads((prior / 'launch_manifest.json').read_text())
        manifest.update({'run_root': str(root), 'source_primary': str(sources['primary']),
            'continued_from_failed_run_root': str(prior),
            'native_source_directory': native_source.name,
            'native_numerical_repair_sha256': digest(native / 'numerical_repair.json'),
            'source_sha256': locked, 'continuation_script_sha256': digest(__file__),
            'old_failure_exit_record_preserved': True})
        write(root / 'launch_manifest.json', manifest)
        phase('original:resume_epoch8_with_finite_l2_backward')
        native_command = ['research/structured-ssl-2026/original_continuation.py']
        for name in ('dataset_root', 'baseline', 'epochs', 'batch_size', 'workers', 'seed', 'lr', 'device', 'platform_validation'):
            native_command += ['--' + name.replace('_', '-'), str(config[name])]
        native_command += ['--run-dir', str(native), '--resume']
        run(native_command, native_source, root / 'original-seed12345.console.log')
        for variant, seed in (('no_object_masking', 12345), ('static_parameters', 12345),
                              ('primary', 12346), ('primary', 12347)):
            source = sources[variant]
            if hashes(source) != locked[variant]:
                raise RuntimeError('sealed pending source changed')
            job = root / f'{variant}-seed{seed}'
            log = root / (job.name + '.console.log')
            base = ['--dataset-root', config['dataset_root'], '--run-dir', str(job),
                    '--device', 'mps', '--workers', '2']
            phase(job.name + ':training')
            run(['-m', 'structured_ssl.train', *base, '--anchor', str(args.anchor.resolve()),
                 '--epochs', '8', '--seed', str(seed), '--platform-validation',
                 str(root / 'reference-anchor-validation.json')], source, log)
            phase(job.name + ':validation_audit')
            run(['-m', 'structured_ssl.evaluate', *base, '--split', 'val'], source, log)
            audit = json.loads((job / 'val_evaluation.json').read_text())
            if variant != 'primary' or audit['checkpoints']['best']['nonregression_and_active_branch']:
                phase(job.name + ':locked_test')
                run(['-m', 'structured_ssl.evaluate', *base, '--split', 'test'], source, log)
            else:
                (job / 'test_deferred.txt').write_text('Test deferred: complete validation failed nonregression/active-support gate.\n')
            (job / 'job_exit_code.txt').write_text('0\n')
        phase('collect_completed_continuation')
        run([str(repository / 'research/structured-ssl-2026/collect_results.py'),
             '--run-root', str(root), '--output', str(root / 'suite-results.json')],
            sources['primary'], root / 'collection.console.log')
        phase('fixed_queue_finished; inspect validation gates and sealed results')
    except BaseException:
        (meta / 'exit_code.txt').write_text('1\n')
        raise
    else:
        (meta / 'exit_code.txt').write_text('0\n')


if __name__ == '__main__':
    main()
