"""Replace only coordination; an existing training child runs uninterrupted.

The original coordinator is suspended while its child runs, then retired after
verified completion. The v1 parent stays suspended until the five fixed v3 jobs
finish or a terminal failure leaves no live GPU child. Original sources and run
records are preserved. A separately sealed policy evaluator opens eligible test.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


JOBS = (('primary', 12345), ('no_object_masking', 12345),
        ('static_parameters', 12345), ('primary', 12346), ('primary', 12347))


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def process(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'state=,ppid=,command='],
                            capture_output=True, text=True)
    if result.returncode:
        return None
    state, parent, command = result.stdout.strip().split(None, 2)
    return {'pid': pid, 'state': state, 'ppid': int(parent), 'command': command}


def live(info):
    return info is not None and not info['state'].startswith('Z')


def same_process(pid, recorded):
    current = process(pid)
    return current if current is not None and current['command'] == recorded['command'] else None


def verify_exports(root):
    manifest = read(root / 'launch_manifest.json')
    for variant, expected in manifest['source_sha256'].items():
        source = root / ('source-' + variant)
        actual = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*.py'))}
        if actual != expected:
            raise RuntimeError(f'sealed {variant} export changed')
    return manifest


def launch(args):
    root = args.run_root.resolve()
    manifest = verify_exports(root)
    coordinator, child, parent = [process(pid) for pid in
                                  (args.coordinator_pid, args.training_pid, args.parent_pid)]
    job = root / 'primary-seed12345'
    if (not live(coordinator) or 'T' in coordinator['state']
            or 'queue_revision.py' not in coordinator['command']
            or str(root) not in coordinator['command']
            or not live(child) or child['ppid'] != args.coordinator_pid
            or '-m structured_ssl.train' not in child['command'] or str(job) not in child['command']
            or not live(parent) or 'T' not in parent['state']
            or 'resume_suite.py' not in parent['command']
            or args.parent_pid != manifest['suspended_parent_pid']):
        raise RuntimeError('handoff requires the identified live coordinator, live training child and suspended v1 parent')
    if manifest['execution_device'] != 'mps' or manifest['batch_size'] != 128 or manifest['workers'] != 2:
        raise RuntimeError('unexpected fixed execution protocol')
    if any((root / f'{variant}-seed{seed}/test_evaluation.json').exists() for variant, seed in JOBS):
        raise RuntimeError('policy must be recorded before first revision test')
    config = read(job / 'config.json')
    if (config['epochs'] != 8 or config['lr'] != .0003 or config['batch_size'] != 128
            or config['workers'] != 2 or config['seed'] != 12345
            or config['architecture_revision'] != manifest['architecture_revision']
            or config['selection_split'] != 'val' or config['answer_candidates_in_adaptation']):
        raise RuntimeError('unexpected live training objective/budget')
    old_script = subprocess.check_output(['git', 'show',
        '184f4d7bdfccbb3533f6e97c825de414ade21cce:research/structured-ssl-2026/queue_revision.py'],
        cwd=Path(__file__).resolve().parents[2])
    if hashlib.sha256(old_script).hexdigest() != manifest['queue_script_sha256']:
        raise RuntimeError('cannot recover the original coordinator source for its record')
    meta = root / 'suite.policy70'
    meta.mkdir(exist_ok=False)
    source = meta / 'source'
    source.mkdir()
    names = ('policy_handoff.py', 'evaluate_sealed_policy.py', 'collect_results.py')
    for name in names:
        shutil.copyfile(Path(__file__).with_name(name), source / name)
    (meta / 'retired_coordinator_source.py').write_bytes(old_script)
    partial = (job / 'history.jsonl').read_bytes()
    (meta / 'validation_history_at_policy_decision.jsonl').write_bytes(partial)
    history = [json.loads(line) for line in partial.decode().splitlines()]
    policy = {'schema_version': 1, 'run_root': str(root), 'created_at_utc': now(),
              'authority': 'direct_user_request', 'user_request': '没事上70就行了',
              'minimum_accuracy_percent': 70., 'selection_split': 'val',
              'active_support_branch_required': True,
              'decided_before_first_revision_test': True,
              'observed_completed_validation_epochs_at_policy_recording': len(history),
              'partial_history_sha256_at_policy_decision': hashlib.sha256(partial).hexdigest(),
              'change_is_after_partial_validation_not_a_pretraining_preregistration': True,
              'architecture_revision': config['architecture_revision'],
              'sealed_training_sources_unchanged': True,
              'original_launch_manifest_sha256': digest(root / 'launch_manifest.json'),
              'fixed_preregistered_jobs': manifest['preregistered_jobs'],
              'no_test_based_checkpoint_architecture_or_seed_selection': True,
              'policy_source_sha256': {name: digest(source / name) for name in names}}
    write(root / 'evaluation_policy.json', policy)
    write(meta / 'handoff.json', {'created_at_utc': now(), 'old_coordinator': coordinator,
        'existing_training_child': child, 'suspended_v1_parent': parent,
        'old_coordinator_source_sha256': hashlib.sha256(old_script).hexdigest(),
        'old_process_only_suspended_child_not_signalled': True,
        'training_exports_and_config_checkpoints_unchanged': True})
    os.kill(args.coordinator_pid, signal.SIGSTOP)
    stopped = same_process(args.coordinator_pid, coordinator)
    if stopped is None or 'T' not in stopped['state']:
        raise RuntimeError('old coordinator did not enter the required suspended state')
    worker = None
    try:
        with (meta / 'console.log').open('a') as log:
            worker = subprocess.Popen([sys.executable, '-u', str(source / 'policy_handoff.py'),
                '--mode', 'supervise', '--run-root', str(root)], cwd=source,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(100):
            if (meta / 'ready.json').exists():
                break
            if worker.poll() is not None:
                raise RuntimeError('new coordinator failed before handoff readiness')
            time.sleep(.1)
        else:
            raise RuntimeError('new coordinator has not acknowledged readiness')
        subprocess.Popen(['caffeinate', '-i', '-w', str(worker.pid)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(json.dumps({'new_coordinator_pid': worker.pid, 'existing_training_child_pid': args.training_pid,
                          'original_coordinator_suspended': True, 'training_child_interrupted': False,
                          'evaluation_policy_sha256': digest(root / 'evaluation_policy.json')}), flush=True)
    except BaseException:
        if worker is not None and worker.poll() is None:
            worker.kill()
            worker.wait()
        # Preserve the aborted decision without imposing it on the old queue.
        (root / 'evaluation_policy.json').rename(meta / 'aborted_evaluation_policy.json')
        observed = same_process(args.coordinator_pid, coordinator)
        if observed is not None:
            os.kill(args.coordinator_pid, signal.SIGCONT)
        raise


def supervise(args):
    # Resolve imports beside the pinned coordinator, never the mutable checkout.
    from evaluate_sealed_policy import policy_record, allows_test
    root = args.run_root.resolve()
    policy, policy_hash = policy_record(root)
    meta, handoff = root / 'suite.policy70', read(root / 'suite.policy70/handoff.json')
    old, existing, parent = [handoff[name] for name in
                             ('old_coordinator', 'existing_training_child', 'suspended_v1_parent')]
    if digest(__file__) != policy['policy_source_sha256']['policy_handoff.py']:
        raise RuntimeError('coordinator differs from recorded policy source')
    observed = same_process(old['pid'], old)
    if observed is None or 'T' not in observed['state']:
        raise RuntimeError('original coordinator is not safely suspended')
    if digest(root / 'launch_manifest.json') != policy['original_launch_manifest_sha256']:
        raise RuntimeError('original launch record changed')
    active = None
    write(meta / 'ready.json', {'pid': os.getpid(), 'created_at_utc': now(),
                              'policy_sha256': policy_hash, 'training_child_pid': existing['pid']})
    def phase(value):
        (meta / 'phase.txt').write_text(value + '\n')
        print(json.dumps({'phase': value, 'time_utc': now()}), flush=True)
    def execute(command, source, log_path):
        nonlocal active
        with log_path.open('a') as log:
            active = subprocess.Popen([sys.executable, '-u', *command], cwd=source,
                env={**os.environ, 'PYTHONPATH': str(source)}, stdout=log, stderr=subprocess.STDOUT)
            (meta / 'active_child.json').write_text(json.dumps({'pid': active.pid, 'command': command,
                                                              'source': str(source), 'started_at_utc': now()}, indent=2) + '\n')
            result = active.wait()
            active = None
            if result:
                raise subprocess.CalledProcessError(result, command)
    try:
        phase('waiting_for_existing_primary_training_child_without_interrupting_it')
        while live(same_process(existing['pid'], existing)):
            time.sleep(20)
        first = root / 'primary-seed12345'
        if not (first / 'training_complete.json').is_file():
            raise RuntimeError('existing training child ended without a completion seal; inspect its unchanged log')
        verify_exports(root)
        phase('retire_old_coordinator_after_existing_training_completion')
        observed = same_process(old['pid'], old)
        if observed is None or 'T' not in observed['state']:
            raise RuntimeError('old coordinator identity/state changed during training')
        os.kill(old['pid'], signal.SIGKILL)
        write(meta / 'old_coordinator_retired.json', {'pid': old['pid'], 'signal': 'SIGKILL',
            'reason': 'replace its evaluation admission after its training child completed',
            'existing_training_child_completed_before_retirement': True, 'time_utc': now(),
            'old_finally_not_executed_v1_parent_remains_suspended': True})
        for variant, seed in JOBS:
            verify_exports(root)
            job = root / f'{variant}-seed{seed}'
            source = root / ('source-' + variant)
            log = root / (job.name + '.console.log')
            if not (job / 'training_complete.json').exists():
                if job.exists():
                    raise RuntimeError('partial existing job requires diagnosis and exact resume, never overwrite')
                phase(job.name + ':training')
                config = read(first / 'config.json')
                execute(['-m', 'structured_ssl.train', '--dataset-root', config['dataset_root'],
                    '--run-dir', str(job), '--anchor', config['anchor'], '--epochs', '8',
                    '--seed', str(seed), '--batch-size', '128', '--workers', '2', '--lr', '.0003',
                    '--device', 'mps', '--platform-validation', str(root / 'reference-anchor-validation.json')], source, log)
            evaluator = str(meta / 'source/evaluate_sealed_policy.py')
            if not (job / 'val_evaluation.json').exists():
                phase(job.name + ':complete_validation_policy70')
                execute([evaluator, '--source-root', str(source), '--run-dir', str(job), '--split', 'val'], source, log)
            validation = read(job / 'val_evaluation.json')
            if validation['evaluation_policy_sha256'] != policy_hash:
                raise RuntimeError('evaluation belongs to a different policy')
            if variant != 'primary' or allows_test(validation['checkpoints']['best']):
                if not (job / 'test_evaluation.json').exists():
                    phase(job.name + ':locked_best_and_final_test_policy70')
                    execute([evaluator, '--source-root', str(source), '--run-dir', str(job), '--split', 'test'], source, log)
            else:
                (job / 'test_deferred.txt').write_text('Test deferred: full validation failed user-directed 70-percent/active-support criterion.\n')
            (job / 'job_exit_code.txt').write_text('0\n')
        phase('collect_fixed_revision_suite_with_recorded_policy70')
        execute([str(meta / 'source/collect_results.py'), '--run-root', str(root),
                 '--output', str(root / 'suite-results-policy70.json')], root / 'source-primary', meta / 'console.log')
        phase('fixed_v3_queue_finished_under_user_policy70')
        (meta / 'exit_code.txt').write_text('0\n')
    except BaseException:
        (meta / 'exit_code.txt').write_text('1\n')
        raise
    finally:
        # A coordinator failure must never start the v1 queue alongside a live GPU child.
        gpu_busy = live(same_process(existing['pid'], existing)) or (active is not None and active.poll() is None)
        observed = same_process(parent['pid'], parent)
        if not gpu_busy and observed is not None and 'T' in observed['state']:
            os.kill(parent['pid'], signal.SIGCONT)
            write(meta / 'parent_released.json', {'pid': parent['pid'], 'time_utc': now(),
                'remaining_fixed_v1_jobs_resumed': True, 'no_live_v3_gpu_child_at_release': True})
        elif gpu_busy:
            write(meta / 'parent_release_deferred.json', {'pid': parent['pid'], 'time_utc': now(),
                'reason': 'keep v1 suspended while an existing v3 GPU child remains live'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('launch', 'supervise'), required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--coordinator-pid', type=int)
    parser.add_argument('--training-pid', type=int)
    parser.add_argument('--parent-pid', type=int)
    args = parser.parse_args()
    if args.mode == 'launch':
        if any(pid is None for pid in (args.coordinator_pid, args.training_pid, args.parent_pid)):
            parser.error('launch requires all three existing process IDs')
        launch(args)
    else:
        supervise(args)


if __name__ == '__main__':
    main()
