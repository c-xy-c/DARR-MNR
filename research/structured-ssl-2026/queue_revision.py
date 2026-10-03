"""Run an already sealed revision after the native control, then release v1.

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
import signal
import subprocess
import sys
import time

from structured_ssl import ARCHITECTURE_REVISION


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))}


def process(pid):
    value = subprocess.run(['ps', '-p', str(pid), '-o', 'state=,command='],
                           capture_output=True, text=True)
    if value.returncode:
        return None
    state, command = value.stdout.strip().split(None, 1)
    return state, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-run-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--sealed-queue-root', type=Path, required=True)
    parser.add_argument('--native-control-pid', type=int, required=True)
    parser.add_argument('--resume-parent-pid', type=int, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    parent, root = args.after_run_root.resolve(), args.run_root.resolve()
    queued = args.sealed_queue_root.resolve()
    queued_manifest = json.loads((queued / 'launch_manifest.json').read_text())
    parent_process = process(args.resume_parent_pid)
    native_process = process(args.native_control_pid)
    if (parent_process is None or 'T' not in parent_process[0]
            or 'resume_suite.py' not in parent_process[1] or str(parent) not in parent_process[1]
            or native_process is None or 'original_continuation.py' not in native_process[1]
            or str(parent) not in native_process[1]
            or queued_manifest['architecture_revision'] != ARCHITECTURE_REVISION
            or queued_manifest['after_run_root'] != str(parent)):
        raise RuntimeError('requires the preserved sealed queue, suspended v1 coordinator and live native child')
    parent_command = parent_process[1]
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
    def terminate_signal(_, __):
        raise KeyboardInterrupt('priority coordinator interrupted')
    signal.signal(signal.SIGTERM, terminate_signal)
    try:
        set_phase('copying_unchanged_sealed_revision_and_cpu_preflights')
        primary = root / 'source-primary'
        sources = {}
        for variant in ('primary', 'no_object_masking', 'static_parameters'):
            sources[variant] = root / ('source-' + variant)
            shutil.copytree(queued / sources[variant].name, sources[variant])
            if hashes(sources[variant]) != queued_manifest['source_sha256'][variant]:
                raise RuntimeError('previously sealed revision source changed')
            artifact = variant + '-cpu-contracts.json'
            contract = json.loads((queued / artifact).read_text())
            if contract['architecture_revision'] != ARCHITECTURE_REVISION or not contract['all_trainable_parameters_have_finite_gradients']:
                raise RuntimeError('previous sealed CPU preflight is incomplete')
            shutil.copyfile(queued / artifact, root / artifact)
        references = {}
        for name in ('reference-anchor-validation.json', 'reference-native-validation.json'):
            shutil.copyfile(parent / name, root / name)
            references[name] = digest(root / name)
        reference = json.loads((root / 'reference-anchor-validation.json').read_text())
        if digest(args.anchor) != reference['anchor_sha256'] or reference['test_opened']:
            raise RuntimeError('revision uses a different anchor or unsealed reference')
        locked = {variant: hashes(source) for variant, source in sources.items()}
        manifest = {'architecture_revision': ARCHITECTURE_REVISION,
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
                    'rescheduled_from_queue_root': str(queued),
                    'unchanged_prior_source_sha256': queued_manifest['source_sha256'],
                    'scheduling': 'native_control_then_v3_then_pending_v1_jobs',
                    'suspended_parent_pid': args.resume_parent_pid,
                    'native_control_pid': args.native_control_pid,
                    'no_test_based_design_or_seed_selection': True,
                    'full_raven_result_claimed': False}
        write(root / 'launch_manifest.json', manifest)
        set_phase('waiting_for_native_control_completion_and_gpu_release')
        native = parent / 'original-seed12345'
        while True:
            child = process(args.native_control_pid)
            if child is None or child[0].startswith('Z'):
                if not (native / 'evaluation.json').is_file():
                    raise RuntimeError('native child exited without complete sealed evaluation')
                break
            if 'original_continuation.py' not in child[1] or str(parent) not in child[1]:
                raise RuntimeError('native child PID was reused')
            time.sleep(30)
        for variant, source in sources.items():
            if hashes(source) != locked[variant]:
                raise RuntimeError('revision source export changed while waiting')
        set_phase('verify_completed_native_control_and_preserved_partial_v1_suite')
        run(['research/structured-ssl-2026/collect_results.py', '--run-root', str(parent),
             '--output', str(root / 'parent-suite-results.json')], primary, root / 'preflight.console.log')
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
        set_phase('revision:real_batch128_metal_preflight')
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
    finally:
        observed = process(args.resume_parent_pid)
        if observed is not None and observed[1] == parent_command:
            os.kill(args.resume_parent_pid, signal.SIGCONT)
            write(meta / 'parent_released.json', {'pid': args.resume_parent_pid,
                  'remaining_v1_jobs_resumed': True, 'fixed_budgets_and_sources_unchanged': True})


if __name__ == '__main__':
    main()
