"""One RAVEN self-supervised task; validation selects the checkpoint."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .data import Raven, loader, normalize
from .evaluate import score
from .model import SSPredRNet, prediction_loss


def save_checkpoint(checkpoint, path):
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def rng_state(train_generator, val_generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
            'train_generator': train_generator.get_state(),
            'val_generator': val_generator.get_state()}


def restore_rng(state, train_generator, val_generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    train_generator.set_state(state['train_generator'].cpu())
    val_generator.set_state(state['val_generator'].cpu())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--margin', type=float, default=.7)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('training requires CUDA')
    root = Path(args.run_dir)
    if args.resume:
        saved = json.loads((root / 'config.json').read_text())
        for field in ('dataset_root', 'epochs', 'batch_size', 'seed', 'lr',
                      'weight_decay', 'margin', 'fp16'):
            if saved[field] != getattr(args, field):
                raise ValueError(f'resume changed {field}')
    else:
        root.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = SSPredRNet(args.margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)
    train_set, val_set = Raven(args.dataset_root, 'train'), Raven(args.dataset_root, 'val')
    if len(train_set) != 42000 or len(val_set) != 14000:
        raise RuntimeError('expected the full RAVEN train/val splits')
    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    start, best_accuracy, best_epoch = 0, -1., 0
    if args.resume:
        checkpoint = torch.load(root / 'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        restore_rng(checkpoint['rng'], train_generator, val_generator)
        start, best_accuracy, best_epoch = (checkpoint['epoch'],
            checkpoint['best_accuracy'], checkpoint['best_epoch'])
        records = (root / 'history.jsonl').read_text().splitlines()
        if len(records) != start:
            raise RuntimeError('checkpoint/history epoch mismatch')
    else:
        (root / 'config.json').write_text(json.dumps({**vars(args),
            'train_count': len(train_set), 'validation_count': len(val_set),
            'selection_split': 'val', 'test_opened_during_training': False,
            'views_per_puzzle': 2, 'public_layout_prior': True}, indent=2) + '\n')
    for epoch in range(start, args.epochs):
        started = time.monotonic()
        model.train()
        total_loss, samples, steps = 0., 0, 0
        for images, _, _ in loader(train_set, args.batch_size, args.workers,
                                   train_generator, training=True):
            batch = len(images)
            images = normalize(images.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=args.fp16):
                loss = prediction_loss(model(images), args.margin)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            samples += batch
            steps += 1
        model.eval()
        validation = score(model, args.dataset_root, 'val', device,
                           batch_size=args.batch_size, workers=args.workers,
                           generator=val_generator)
        improved = validation['accuracy_macro'] > best_accuracy
        if improved:
            best_accuracy, best_epoch = validation['accuracy_macro'], epoch + 1
        save_checkpoint({'epoch': epoch + 1, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
            'rng': rng_state(train_generator, val_generator),
            'best_accuracy': best_accuracy, 'best_epoch': best_epoch}, root / 'last.pt')
        if improved:
            shutil.copyfile(root / 'last.pt', root / 'best.pt')
        if epoch + 1 == args.epochs:
            shutil.copyfile(root / 'last.pt', root / 'final.pt')
        record = {'epoch': epoch + 1, 'validation': validation,
                  'best_epoch': best_epoch, 'best_accuracy': best_accuracy,
                  'train_total_loss_per_sample': total_loss / samples,
                  'train_steps': steps, 'wall_seconds': time.monotonic() - started}
        with (root / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)
    # No test dataset is enumerated before the complete training run.
    model.eval()
    result = {'best_epoch': best_epoch, 'best_validation_accuracy': best_accuracy,
              'selection_split': 'val', 'test': {}}
    for name in ('final', 'best'):
        checkpoint = torch.load(root / f'{name}.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        result['test'][name] = score(model, args.dataset_root, 'test', device,
                                   batch_size=args.batch_size, workers=args.workers,
                                   generator=torch.Generator().manual_seed(args.seed + 2))
    (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
