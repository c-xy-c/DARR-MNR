"""Seal v2 now, then run its fixed protocol after the current Metal suite.

The common native control is reused only after checking its completed seals and
unchanged source. All five new adaptation jobs run in separate source exports.
Failures preserve records and stop; a failed validation gate defers only test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from build_ablations import export


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-run-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    parent, root = args.after_run_root.resolve(), args.run_root.resolve()
    prior = json.loads((parent / 'launch_manifest.json').read_text())
    if prior['execution_device'] != 'mps' or prior['batch_size'] != 128 or prior['workers'] != 2:
        raise RuntimeError('requires the locked batch128 Metal parent protocol')
    if root == parent or not args.dataset_root.is_dir() or not args.anchor.is_file():
        raise RuntimeError('invalid revision destination, dataset or anchor')
    root.mkdir(parents=True, exist_ok=False)
    meta = root / 'suite.launch'
    meta.mkdir()
    write(meta / 'process.json', {'pid': os.getpid(), 'python': sys.executable})
    phase = meta / 'phase.txt'
    def set_phase(value):
        phase.write_text(value + '\n')
        print(json.dumps({'phase': value}), flush=True)
    environment = {**os.environ, 'PYTHONPATH': '.'}
    def run(command, source, log):
        with log.open('a') as stream:
            subprocess.run([sys.executable, '-u', *command], cwd=source, env=environment,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    try:
        set_phase('sealing_sources')
        primary = root / 'source-primary'
        primary.mkdir()
        for package in ('structured_ssl', 'attention_ssl', 'program_ssl', 'sspredrnet'):
            destination = primary / package
            destination.mkdir()
            for file in (repository / package).glob('*.py'):
                shutil.copyfile(file, destination / file.name)
        research = primary / 'research/structured-ssl-2026'
        research.mkdir(parents=True)
        for name in ('original_continuation.py', 'profile_mps.py', 'collect_results.py'):
            shutil.copyfile(repository / 'research/structured-ssl-2026' / name, research / name)
        sources = {'primary': primary}
        for variant in ('no_object_masking', 'static_parameters'):
            sources[variant] = root / ('source-' + variant)
            export(repository, sources[variant], variant)
        references = {}
        for name in ('reference-anchor-validation.json', 'reference-native-validation.json'):
            shutil.copyfile(parent / name, root / name)
            references[name] = digest(root / name)
        reference = json.loads((root / 'reference-anchor-validation.json').read_text())
        if digest(args.anchor) != reference['anchor_sha256'] or reference['test_opened']:
            raise RuntimeError('revision uses a different anchor or unsealed reference')
        locked = {variant: hashes(source) for variant, source in sources.items()}
        manifest = {'architecture_revision': 'isolated-target-structured-feedback-v2',
                    'run_root': str(root), 'after_run_root': str(parent),
                    'source_primary': str(primary), 'execution_device': 'mps',
                    'batch_size': 128, 'workers': 2,
                    'source_snapshots_locked_before_training': True,
                    'historical_validation_correct': prior['historical_validation_correct'],
                    'measured_platform_validation_correct': prior['measured_platform_validation_correct'],
                    'preregistered_jobs': prior['preregistered_jobs'],
                    'native_control_execution': 'reuse_identical_completed_parent_control',
                    'source_sha256': locked, 'reference_sha256': references,
                    'queue_script_sha256': digest(__file__),
                    'no_test_based_design_or_seed_selection': True,
                    'full_raven_result_claimed': False}
        write(root / 'launch_manifest.json', manifest)
        for variant, source in sources.items():
            set_phase(variant + ':cpu_preflight')
            run(['-m', 'structured_ssl.check_contracts', '--anchor', str(args.anchor.resolve()),
                 '--device', 'cpu', '--output', str(root / (variant + '-cpu-contracts.json'))],
                source, root / 'preflight.console.log')
        set_phase('waiting_for_parent_suite')
        exit_file = parent / 'suite.launch/exit_code.txt'
        while not exit_file.exists():
            time.sleep(30)
        if exit_file.read_text().strip() != '0':
            raise RuntimeError('parent suite failed; preserve records and repair it before revision training')
        for variant, source in sources.items():
            if hashes(source) != locked[variant]:
                raise RuntimeError('revision source export changed while waiting')
        set_phase('verify_completed_parent_suite')
        run(['research/structured-ssl-2026/collect_results.py', '--run-root', str(parent),
             '--output', str(root / 'parent-suite-results.json')], primary, root / 'preflight.console.log')
        native = parent / 'original-seed12345'
        native_seal = json.loads((native / 'training_complete.json').read_text())
        for name, expected in native_seal['source_sha256'].items():
            if digest(primary / name) != expected:
                raise RuntimeError('shared native control recipe changed')
        for name, expected in native_seal['checkpoint_sha256'].items():
            if digest(native / (name + '.pt')) != expected:
                raise RuntimeError('shared native checkpoint changed')
        if (json.loads((native / 'evaluation.json').read_text())['seal'] != native_seal
                or digest(native / 'config.json') != native_seal['config_sha256']):
            raise RuntimeError('shared native control is incomplete or modified')
        shutil.copytree(native, root / native.name)
        write(root / 'shared_native_control.json', {'from_run': str(native),
            'copied_files_sha256': {p.name: digest(p) for p in sorted(native.iterdir()) if p.is_file()},
            'actual_new_native_training_epochs': 0, 'shared_completed_native_adaptation_epochs': 16,
            'same_selection_budget_epochs': 44})
        set_phase('v2:real_batch128_metal_preflight')
        run(['research/structured-ssl-2026/profile_mps.py', '--anchor', str(args.anchor.resolve()),
             '--dataset-root', str(args.dataset_root.resolve()),
             '--output', str(root / 'real-batch128-mps-preflight.json')], primary, root / 'preflight.console.log')
        for variant, seed in (('primary', 12345), ('no_object_masking', 12345),
                              ('static_parameters', 12345), ('primary', 12346), ('primary', 12347)):
            source = sources[variant]
            if hashes(source) != locked[variant]:
                raise RuntimeError('sealed source changed before job')
            job = root / f'{variant}-seed{seed}'
            log = root / (job.name + '.console.log')
            base = ['--dataset-root', str(args.dataset_root.resolve()), '--run-dir', str(job),
                    '--device', 'mps', '--workers', '2']
            set_phase(job.name + ':training')
            run(['-m', 'structured_ssl.train', *base, '--anchor', str(args.anchor.resolve()),
                 '--epochs', '8', '--seed', str(seed), '--platform-validation',
                 str(root / 'reference-anchor-validation.json')], source, log)
            set_phase(job.name + ':validation_audit')
            run(['-m', 'structured_ssl.evaluate', *base, '--split', 'val'], source, log)
            audit = json.loads((job / 'val_evaluation.json').read_text())
            if variant != 'primary' or audit['checkpoints']['best']['nonregression_and_active_branch']:
                set_phase(job.name + ':locked_test')
                run(['-m', 'structured_ssl.evaluate', *base, '--split', 'test'], source, log)
            else:
                (job / 'test_deferred.txt').write_text('Test deferred: complete validation failed nonregression/active-support gate.\n')
            (job / 'job_exit_code.txt').write_text('0\n')
        set_phase('collect_completed_revision')
        run(['research/structured-ssl-2026/collect_results.py', '--run-root', str(root),
             '--output', str(root / 'suite-results.json')], primary, root / 'preflight.console.log')
        set_phase('fixed_queue_finished; inspect validation gates and sealed results')
    except BaseException:
        (meta / 'exit_code.txt').write_text('1\n')
        raise
    else:
        (meta / 'exit_code.txt').write_text('0\n')


if __name__ == '__main__':
    main()
