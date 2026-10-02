"""Matched warm-start experiments. Test evaluation is a separate locked step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from sspredrnet.data import Raven, loader, normalize
from sspredrnet.evaluate import score
from sspredrnet.model import prediction_loss
from sspredrnet.train import save_checkpoint, rng_state, restore_rng
from .model import ComponentProgram
from .data import KnownRowTraining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--study', choices=('control', 'program', 'program-no-ssl', 'program-static'),
                        default='program')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--reasoning-lr', type=float, default=3e-5)
    parser.add_argument('--program-lr', type=float, default=3e-4)
    parser.add_argument('--program-weight', type=float, default=.1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(args.run_dir)
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    model = ComponentProgram(use_program=args.study != 'control',
                             program_weight=args.program_weight,
                             static_program=args.study == 'program-static').to(device)
    checkpoint = torch.load(args.baseline, map_location=device, weights_only=False)
    model.load_baseline(checkpoint['model'])
    source_epoch = checkpoint['epoch']
    groups = [{'params': [p for p in model.parameters() if p.requires_grad],
               'lr': args.reasoning_lr if args.study == 'control' else args.program_lr}]
    optimizer = torch.optim.Adam(groups, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    # Extra compiler initialization must not alter the matched dropout stream.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    include_candidates = args.study in ('control', 'program-no-ssl')
    train_set = KnownRowTraining(args.dataset_root, include_candidates=include_candidates)
    val_set = Raven(args.dataset_root, 'val')
    if len(train_set) != 42000 or len(val_set) != 14000:
        raise RuntimeError('full RAVEN train/validation splits required')
    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    start, best_accuracy, best_epoch = 0, -1., 0
    config = {**vars(args), 'source_epoch': source_epoch,
              'source_training_budget_epochs': 20,
              'baseline_sha256': hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest(),
              'perception_frozen': True, 'batchnorm_frozen': True,
              'gradient_clip_norm': 1., 'version': 3,
              'reasoner_frozen': args.study != 'control',
              'train_count': 42000, 'validation_count': 14000,
              'selection_split': 'val', 'test_opened_during_training': False,
              'adaptation_uses_candidates': args.study in ('control', 'program-no-ssl'),
              'training_split_confidence_uses_observed_five_only': True,
              'new_ssl_uses_candidates': False}
    if args.resume:
        saved = json.loads((root / 'config.json').read_text())
        for field, value in config.items():
            if field not in ('resume',) and value != saved[field]:
                raise ValueError(f'resume changed {field}')
        last = torch.load(root / 'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(last['model'], strict=True)
        optimizer.load_state_dict(last['optimizer'])
        scaler.load_state_dict(last['scaler'])
        restore_rng(last['rng'], train_generator, val_generator)
        start, best_accuracy, best_epoch = last['epoch'], last['best_accuracy'], last['best_epoch']
        records = (root / 'history.jsonl').read_text().splitlines()
        if len(records) != start:
            raise RuntimeError('checkpoint/history mismatch')
    else:
        (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        model.eval()
        initial = score(model, args.dataset_root, 'val', device,
                        batch_size=args.batch_size, workers=args.workers,
                        generator=val_generator)
        (root / 'initial_validation.json').write_text(json.dumps(initial, indent=2) + '\n')
        print(json.dumps({'initial_validation': initial, 'config': config}), flush=True)
    for epoch in range(start, args.epochs):
        started = time.monotonic()
        model.train()
        ranking_total, ssl_total, samples, steps = 0., 0., 0, 0
        alpha_sum = None
        alpha_square_sum = None
        for images, _, configurations in loader(train_set, args.batch_size, args.workers,
                                                train_generator, training=True):
            batch = len(images)
            images = normalize(images.to(device, non_blocking=True))
            configurations = configurations.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if args.study in ('program', 'program-static'):
                    ssl, alpha = model.ssl(images, configurations)
                    ranking = ssl.new_zeros(())
                    loss = ssl
                else:
                    output = model(images)
                    alpha = output['alpha']
                    ranking = prediction_loss(output['errors']) / batch
                    ssl = ranking.new_zeros(())
                    loss = ranking
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                           if p.requires_grad], 1.)
            scaler.step(optimizer)
            scaler.update()
            ranking_total += float(ranking.detach()) * batch
            ssl_total += float(ssl.detach()) * batch
            samples += batch
            steps += 1
            if alpha is not None:
                alpha = alpha.detach()
                if alpha_sum is None:
                    alpha_sum, alpha_square_sum = alpha.sum(0), alpha.square().sum(0)
                else:
                    alpha_sum += alpha.sum(0)
                    alpha_square_sum += alpha.square().sum(0)
        model.eval()
        validation = score(model, args.dataset_root, 'val', device,
                           batch_size=args.batch_size, workers=args.workers,
                           generator=val_generator)
        improved = validation['accuracy_macro'] > best_accuracy
        if improved:
            best_accuracy, best_epoch = validation['accuracy_macro'], epoch + 1
        saved = {'epoch': epoch + 1, 'source_epoch': source_epoch,
                 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scaler': scaler.state_dict(), 'rng': rng_state(train_generator, val_generator),
                 'best_accuracy': best_accuracy, 'best_epoch': best_epoch}
        save_checkpoint(saved, root / 'last.pt')
        if improved:
            shutil.copyfile(root / 'last.pt', root / 'best.pt')
        if epoch + 1 == args.epochs:
            shutil.copyfile(root / 'last.pt', root / 'final.pt')
        record = {'epoch': epoch + 1, 'lineage_epoch': source_epoch + epoch + 1,
                  'validation': validation, 'best_epoch': best_epoch,
                  'best_accuracy': best_accuracy, 'ranking_loss': ranking_total / samples,
                  'ssl_loss': ssl_total / samples, 'train_steps': steps,
                  'train_samples': samples, 'wall_seconds': time.monotonic() - started}
        if alpha_sum is not None:
            mean = alpha_sum / (2 * samples)
            deviation = (alpha_square_sum / (2 * samples) - mean.square()).clamp_min(0).sqrt()
            record['alpha_mean'], record['alpha_std'] = mean.tolist(), deviation.tolist()
            record['operator_right_norm'] = float(model.operators.right.detach().norm())
            record['operator_coefficients'] = model.operators.coefficients.detach().tolist()
        with (root / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)
    (root / 'training_complete.json').write_text(json.dumps({
        'epochs': args.epochs, 'best_epoch': best_epoch, 'best_validation_accuracy': best_accuracy,
        'test_evaluated': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
