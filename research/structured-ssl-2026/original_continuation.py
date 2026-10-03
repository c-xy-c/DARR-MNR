"""Matched original PRB continuation: frozen CNN/BN, no added network modules.

The initial checkpoint is the declared validation-selected native foundation.
Sixteen added epochs match the structured method's 44-epoch selection budget.
The original unlabeled-candidate margin task is retained and disclosed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
import torch

from sspredrnet.checkpoint import save_checkpoint
from structured_ssl.runtime import resolve_device, execution_record, rng_state, restore_rng
from sspredrnet.data import CONFIGS, Raven, loader, normalize
from sspredrnet.evaluate import score
from sspredrnet.model import SSPredRNet, prediction_loss
from sspredrnet.views import split_layers


FOUNDATION_SHA256 = '8a383fcc609127a7c35bfb9c6a2ffe064ce5285f66d8525b3668385ec915928c'
SOURCE_FILES = ('research/structured-ssl-2026/original_continuation.py',
                'structured_ssl/runtime.py',
                'sspredrnet/model.py', 'sspredrnet/layers.py', 'sspredrnet/views.py',
                'sspredrnet/data.py', 'sspredrnet/evaluate.py', 'sspredrnet/checkpoint.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {name: digest(root / name) for name in SOURCE_FILES}


class OriginalKnownRows(Raven):
    """Same five-visible-panel fallback as the measured earlier control."""
    def __init__(self, root):
        super().__init__(root, 'train')

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            # Positions six/seven are unused by the native training objective.
            indices = [0, 1, 2, 3, 4, 5, 0, 1, *range(8, 16)]
            raw = data['image'].reshape(16, 160, 160)[indices]
        panels = np.stack([cv2.resize(p, (80, 80), interpolation=cv2.INTER_NEAREST)
                           for p in raw]).astype(np.uint8)
        if random.random() < .5:
            panels = panels[:, :, ::-1].copy()
        layers = [split_layers(panel, path.parent.name) for panel in panels]
        if any(layer is None for layer in layers[:5]):
            views = np.stack((panels, panels))
        else:
            layers = [np.stack((panel, panel)) if layer is None else layer
                      for panel, layer in zip(panels, layers)]
            views = np.stack(layers, axis=1)
        return torch.from_numpy(views.astype(np.float32)), -1, CONFIGS.index(path.parent.name)


class OriginalContinuation(SSPredRNet):
    def __init__(self):
        super().__init__()
        for name, parameter in self.reasoner.named_parameters():
            if name.startswith(('res', 'channel_reducer')):
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for index in range(4):
            getattr(self.reasoner, f'res{index}').eval()
        self.reasoner.channel_reducer.eval()
        for module in self.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        return self


def perception_hash(model):
    result = hashlib.sha256()
    for name, tensor in model.reasoner.state_dict().items():
        if name.startswith(('res', 'channel_reducer')):
            result.update(name.encode())
            result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def batchnorm_hash(model):
    result = hashlib.sha256()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for key, value in module.named_buffers():
                result.update(f'{name}.{key}'.encode())
                result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs != 16 or args.batch_size < 1 or args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('declared native control uses 16 epochs; batch/lr positive, workers nonnegative')
    device = resolve_device(args.device)
    if digest(args.baseline) != FOUNDATION_SHA256:
        raise RuntimeError('baseline differs from the preregistered native foundation')
    root = Path(args.run_dir)
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = OriginalContinuation().to(device)
    foundation = torch.load(args.baseline, map_location=device, weights_only=False)
    model.load_state_dict(foundation['model'], strict=True)
    source_epoch = foundation['source_epoch'] + foundation['epoch']
    baseline_config = json.loads(Path(args.baseline).with_name('config.json').read_text())
    prior_budget = baseline_config['source_training_budget_epochs'] + baseline_config['epochs']
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(device.type)
    # Match the measured original continuation's dropout initialization stream.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train, validation = OriginalKnownRows(args.dataset_root), Raven(args.dataset_root, 'val')
    if len(train) != 42000 or len(validation) != 14000:
        raise RuntimeError('complete RAVEN training/validation required')
    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    config = {**vars(args), 'method': 'original-prb-continuation', 'schema_version': 1,
              'execution': execution_record(device),
              'source_epoch': source_epoch, 'source_training_selection_budget_epochs': prior_budget,
              'total_training_selection_budget_epochs': prior_budget + args.epochs,
              'baseline_sha256': FOUNDATION_SHA256, 'source_sha256': source_hashes(),
              'frozen_perception_sha256': perception_hash(model), 'selection_split': 'val',
              'frozen_batchnorm_buffers_sha256': batchnorm_hash(model),
              'cnn_and_batchnorm_frozen': True, 'only_original_prbs_trainable': True,
              'trainable_parameters': sum(p.numel() for p in parameters),
              'training_count': 42000, 'validation_count': 14000,
              'unlabeled_answer_candidates_used_as_negatives': True,
              'answer_indices_and_xml_used_in_training': False,
              'visible_five_decide_global_fallback': True, 'gradient_clip_norm': 1.,
              'loss_normalization': 'prediction_loss divided by puzzle batch, as earlier control'}
    start, best, best_epoch = 0, -1., 0
    if args.resume:
        previous = json.loads((root / 'config.json').read_text())
        if {k: v for k, v in previous.items() if k != 'resume'} != {k: v for k, v in config.items() if k != 'resume'}:
            raise RuntimeError('resume changed protocol or source')
        checkpoint = torch.load(root / 'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        restore_rng(checkpoint['rng'], train_generator, val_generator, device=device)
        start, best, best_epoch = checkpoint['epoch'], checkpoint['best_accuracy'], checkpoint['best_epoch']
        history = [json.loads(line) for line in (root / 'history.jsonl').read_text().splitlines()]
        if [record['epoch'] for record in history] != list(range(1, start + 1)):
            raise RuntimeError('checkpoint/history mismatch')
    else:
        (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        model.eval()
        initial = score(model, args.dataset_root, 'val', device, batch_size=args.batch_size,
                        workers=args.workers, generator=val_generator)
        if initial['correct'] != 9979:
            raise RuntimeError('declared foundation validation did not replay')
        (root / 'initial_validation.json').write_text(json.dumps(initial, indent=2) + '\n')
        print(json.dumps({'initial_validation': initial, 'config': config}), flush=True)
    for epoch in range(start, args.epochs):
        started, samples, summed = time.monotonic(), 0, 0.
        model.train()
        for images, _, _ in loader(train, args.batch_size, args.workers, train_generator, training=True):
            batch = len(images)
            images = normalize(images.to(device))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss = prediction_loss(model(images)) / batch
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite native loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm):
                raise RuntimeError('nonfinite native gradient')
            scaler.step(optimizer)
            scaler.update()
            samples += batch
            summed += float(loss.detach()) * batch
        if samples != 42000:
            raise RuntimeError('incomplete native epoch')
        model.eval()
        observed = score(model, args.dataset_root, 'val', device, batch_size=args.batch_size,
                         workers=args.workers, generator=val_generator)
        improved = observed['accuracy_macro'] > best
        if improved:
            best, best_epoch = observed['accuracy_macro'], epoch + 1
        checkpoint = {'epoch': epoch + 1, 'source_epoch': source_epoch, 'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                      'rng': rng_state(train_generator, val_generator, device=device), 'best_accuracy': best, 'best_epoch': best_epoch}
        save_checkpoint(checkpoint, root / 'last.pt')
        if improved:
            shutil.copyfile(root / 'last.pt', root / 'best.pt')
        if epoch + 1 == args.epochs:
            shutil.copyfile(root / 'last.pt', root / 'final.pt')
        record = {'epoch': epoch + 1, 'lineage_epoch': source_epoch + epoch + 1, 'validation': observed,
                  'best_accuracy': best, 'best_epoch': best_epoch, 'training_count': samples,
                  'loss': summed / samples, 'wall_seconds': time.monotonic() - started}
        with (root / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)
    if perception_hash(model) != config['frozen_perception_sha256']:
        raise RuntimeError('native continuation changed CNN weights or buffers')
    if batchnorm_hash(model) != config['frozen_batchnorm_buffers_sha256']:
        raise RuntimeError('native continuation changed frozen BatchNorm buffers')
    seal = {'method': config['method'], 'epochs': args.epochs, 'best_epoch': best_epoch,
            'source_sha256': source_hashes(), 'baseline_sha256': FOUNDATION_SHA256,
            'checkpoint_sha256': {name: digest(root / f'{name}.pt') for name in ('best', 'final')},
            'selection_split': 'val', 'test_opened': False}
    seal_path = root / 'training_complete.json'
    if seal_path.exists() and json.loads(seal_path.read_text()) != seal:
        raise RuntimeError('native training seal changed')
    seal_path.write_text(json.dumps(seal, indent=2) + '\n')
    output = root / 'evaluation.json'
    if output.exists():
        raise FileExistsError('preserve previous native evaluation')
    # The full epoch sequence and bytes are sealed before test enumeration.
    result = {'seal': seal, 'device': str(device), 'torch': str(torch.__version__), 'checkpoints': {}}
    for name in ('best', 'final'):
        checkpoint = torch.load(root / f'{name}.pt', map_location=device, weights_only=False)
        if digest(root / f'{name}.pt') != seal['checkpoint_sha256'][name] or source_hashes() != seal['source_sha256']:
            raise RuntimeError('native bytes/source changed before evaluation')
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        observed_validation = score(model, args.dataset_root, 'val', device, batch_size=args.batch_size, workers=args.workers)
        records = [json.loads(line) for line in (root / 'history.jsonl').read_text().splitlines()]
        if observed_validation != records[checkpoint['epoch'] - 1]['validation']:
            raise RuntimeError('native selected validation did not replay before test')
        result['checkpoints'][name] = {'epoch': checkpoint['epoch'],
            'lineage_epoch': source_epoch + checkpoint['epoch'],
            'validation': observed_validation,
            'test': score(model, args.dataset_root, 'test', device, batch_size=args.batch_size, workers=args.workers)}
        print(json.dumps({'checkpoint': name, **result['checkpoints'][name]}), flush=True)
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
