"""Adapt attention completion on known panels; select with the deployed energy."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from program_ssl.data import KnownRowTraining
from sspredrnet.checkpoint import rng_state, restore_rng, save_checkpoint
from sspredrnet.data import Raven, loader, normalize
from sspredrnet.evaluate import score
from .model import AttentionCompletion


SOURCE_FILES = ('attention_ssl/model.py', 'attention_ssl/train.py', 'attention_ssl/evaluate.py',
                'program_ssl/model.py', 'program_ssl/data.py', 'sspredrnet/model.py',
                'sspredrnet/layers.py', 'sspredrnet/views.py', 'sspredrnet/data.py',
                'sspredrnet/evaluate.py', 'sspredrnet/checkpoint.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: digest(root / name) for name in SOURCE_FILES}


def reasoner_hash(model):
    fingerprint = hashlib.sha256()
    for name, tensor in model.reasoner.state_dict().items():
        fingerprint.update(name.encode())
        fingerprint.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return fingerprint.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('epochs, batch-size, lr must be positive; workers must be nonnegative')
    if not torch.cuda.is_available():
        raise RuntimeError('full RAVEN training requires CUDA')
    root, device = Path(args.run_dir), torch.device('cuda:0')
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = AttentionCompletion().to(device)
    baseline = torch.load(args.baseline, map_location=device, weights_only=False)
    model.load_baseline(baseline['model'])
    source_epoch = baseline.get('source_epoch', 0) + baseline['epoch']
    baseline_metadata_path = Path(args.baseline).with_name('config.json')
    baseline_metadata = json.loads(baseline_metadata_path.read_text()) if baseline_metadata_path.exists() else {}
    source_selection_budget = (baseline_metadata.get('source_training_budget_epochs', 0) + baseline_metadata['epochs']
                               if 'epochs' in baseline_metadata else None)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    train_set, val_set = KnownRowTraining(args.dataset_root), Raven(args.dataset_root, 'val')
    if len(train_set) != 42000 or len(val_set) != 14000:
        raise RuntimeError('complete RAVEN training and validation splits are required')
    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    config = {**vars(args), 'schema_version': 1, 'method': 'attention-support-completion',
              'source_epoch': source_epoch, 'baseline_checkpoint_epoch': baseline['epoch'],
              'baseline_source_epoch': baseline.get('source_epoch', 0),
              'source_training_selection_budget_epochs': source_selection_budget,
              'total_training_selection_budget_epochs': (source_selection_budget + args.epochs
                                                         if source_selection_budget is not None else None),
              'baseline_sha256': digest(args.baseline),
              'frozen_reasoner_sha256': reasoner_hash(model),
              'source_sha256': source_hashes(), 'evidence_weight': model.evidence_weight,
              'selection_energy_equals_deployed_energy': True, 'selection_split': 'val',
              'baseline_frozen': True, 'batchnorm_frozen': True,
              'adaptation_reads_only_six_known_panels': True, 'answer_candidates_used_in_adaptation': False,
              'training_count': 42000, 'validation_count': 14000,
              'gradient_clip_norm': 1., 'ridge': model.completion.ridge,
              'negative_mining': 'up_to_seven_same_layout_component_known_targets',
              'acceptance_requires_nonzero_branch_and_no_matched_test_regression': True}
    start, best_accuracy, best_epoch = 0, -1., 0
    if args.resume:
        previous = json.loads((root / 'config.json').read_text())
        for key, value in config.items():
            if key != 'resume' and previous.get(key) != value:
                raise ValueError(f'resume changed {key}')
        last = torch.load(root / 'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(last['model'], strict=True)
        if reasoner_hash(model) != config['frozen_reasoner_sha256']:
            raise RuntimeError('resume checkpoint changed the frozen baseline')
        optimizer.load_state_dict(last['optimizer'])
        scaler.load_state_dict(last['scaler'])
        restore_rng(last['rng'], train_generator, val_generator)
        start, best_accuracy, best_epoch = last['epoch'], last['best_accuracy'], last['best_epoch']
        if len((root / 'history.jsonl').read_text().splitlines()) != start:
            raise RuntimeError('checkpoint/history mismatch')
    else:
        (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        model.eval()
        initial = score(model, args.dataset_root, 'val', device, batch_size=args.batch_size,
                        workers=args.workers, generator=val_generator)
        (root / 'initial_validation.json').write_text(json.dumps(initial, indent=2) + '\n')
        print(json.dumps({'initial_validation': initial, 'config': config}), flush=True)
    for epoch in range(start, args.epochs):
        started = time.monotonic()
        model.train()
        samples, loss_sum, statistic_sums = 0, 0., {}
        for images, _, configs in loader(train_set, args.batch_size, args.workers, train_generator, training=True):
            images, configs = normalize(images.to(device)), configs.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss, statistics = model.ssl(images, configs)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            scaler.step(optimizer)
            scaler.update()
            batch = len(images)
            samples += batch
            loss_sum += float(loss.detach()) * batch
            for key, value in statistics.items():
                multiplier = 1 if key.endswith(('queries', 'correct')) else batch
                statistic_sums[key] = statistic_sums.get(key, 0.) + value * multiplier
        model.eval()
        validation = score(model, args.dataset_root, 'val', device, batch_size=args.batch_size,
                           workers=args.workers, generator=val_generator)
        improved = validation['accuracy_macro'] > best_accuracy
        if improved:
            best_accuracy, best_epoch = validation['accuracy_macro'], epoch + 1
        checkpoint = {'epoch': epoch + 1, 'source_epoch': source_epoch,
                      'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'scaler': scaler.state_dict(), 'rng': rng_state(train_generator, val_generator),
                      'best_accuracy': best_accuracy, 'best_epoch': best_epoch}
        save_checkpoint(checkpoint, root / 'last.pt')
        if improved:
            shutil.copyfile(root / 'last.pt', root / 'best.pt')
        if epoch + 1 == args.epochs:
            shutil.copyfile(root / 'last.pt', root / 'final.pt')
        record = {'epoch': epoch + 1, 'lineage_epoch': source_epoch + epoch + 1,
                  'validation': validation, 'best_accuracy': best_accuracy, 'best_epoch': best_epoch,
                  'ssl_loss': loss_sum / samples, 'training_puzzles': samples,
                  'wall_seconds': time.monotonic() - started,
                  'completion_gate': float(model.completion.gate.detach().tanh()),
                  'training_metrics': {key: value if key.endswith(('queries', 'correct')) else value / samples
                                       for key, value in statistic_sums.items()}}
        with (root / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)
    if reasoner_hash(model) != config['frozen_reasoner_sha256']:
        raise RuntimeError('frozen baseline changed during adaptation')
    seal = {'schema_version': 1, 'method': config['method'], 'source_sha256': source_hashes(),
            'baseline_sha256': config['baseline_sha256'], 'evidence_weight': model.evidence_weight,
            'frozen_reasoner_sha256': config['frozen_reasoner_sha256'],
            'checkpoint_sha256': {name: digest(root / f'{name}.pt') for name in ('best', 'final')},
            'best_epoch': best_epoch, 'epochs': args.epochs, 'test_opened': False}
    path = root / 'training_complete.json'
    if path.exists() and json.loads(path.read_text()) != seal:
        raise RuntimeError('completed-run seal differs')
    path.write_text(json.dumps(seal, indent=2) + '\n')


if __name__ == '__main__':
    main()
