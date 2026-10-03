"""Full RAVEN adaptation; validation selects epochs, never test accuracy."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from sspredrnet.checkpoint import save_checkpoint
from sspredrnet.data import loader
from .data import ObjectRaven, to_device
from .model import StructuredCompletion
from .metrics import summarize_counts
from . import ARCHITECTURE_REVISION
from .runtime import resolve_device, execution_record, rng_state, restore_rng, validation_reference
from .policy import MINIMUM_ACCURACY_PERCENT
from .provenance import anchor_hash, digest, source_hashes


def score(model, dataset_root, split, device, batch_size, workers, generator):
    counts, totals = [0] * 7, [0] * 7
    with torch.inference_mode():
        for pack, labels, configs in loader(ObjectRaven(dataset_root, split), batch_size, workers, generator):
            predicted = model(to_device(pack, device)).argmin(1).cpu()
            for index in range(7):
                mask = configs == index
                totals[index] += int(mask.sum())
                counts[index] += int(((predicted == labels) & mask).sum())
    return summarize_counts(counts, totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--platform-validation')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('epochs, batch size and lr must be positive; workers nonnegative')
    device, root = resolve_device(args.device), Path(args.run_dir)
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = StructuredCompletion().to(device)
    anchor_seal = json.loads(Path(args.anchor).with_name('training_complete.json').read_text())
    anchor_validation = json.loads(Path(args.anchor).with_name('val_evaluation.json').read_text())
    if anchor_seal.get('method') != 'attention-support-completion' or anchor_validation['seal'] != anchor_seal:
        raise RuntimeError('declared attention anchor requires its sealed validation record')
    if digest(args.anchor) != anchor_seal['checkpoint_sha256']['best']:
        raise RuntimeError('anchor must be its validation-selected checkpoint')
    expected_anchor_correct = anchor_validation['checkpoints']['best']['full']['correct']
    platform_correct, platform_sha256 = validation_reference(args.platform_validation, args.anchor,
                                                             expected_anchor_correct, device)
    anchor = torch.load(args.anchor, map_location=device, weights_only=False)
    model.load_anchor(anchor['model'])
    anchor_config = json.loads(Path(args.anchor).with_name('config.json').read_text())
    source_epoch = anchor['source_epoch'] + anchor['epoch']
    selection_budget = anchor_config['total_training_selection_budget_epochs']
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(device.type)
    train, validation = ObjectRaven(args.dataset_root, 'train'), ObjectRaven(args.dataset_root, 'val')
    if len(train) != 42000 or len(validation) != 14000:
        raise RuntimeError('complete RAVEN 42000/14000 splits required')
    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    config = {**vars(args), 'schema_version': 1, 'method': 'structured-object-support-pav',
              'architecture_revision': ARCHITECTURE_REVISION,
              'parameter_packing': 'SHINE_rl_A_width_by_rank_B_rank_by_width',
              'compiler_normalization': 'post_norm',
              'compiler_memory_identity': 'learned_layer_and_token_positions_added_after_extraction_zero_initialized',
              'object_teacher_target': 'bbox_isolated_white_canvas_mask_pooled_frozen_cnn',
              'support_feedback': '25_local_and_pooled_dense_residuals_10_transport_aligned_object_presence_residuals',
              'execution': execution_record(device), 'perception_activation_checkpointing': True,
              'historical_anchor_validation_correct': expected_anchor_correct,
              'platform_anchor_validation_correct': platform_correct,
              'platform_validation_sha256': platform_sha256,
              'source_epoch': source_epoch, 'anchor_checkpoint_epoch': anchor['epoch'],
              'anchor_training_selection_budget_epochs': selection_budget,
              'total_training_selection_budget_epochs': selection_budget + args.epochs,
              'anchor_sha256': digest(args.anchor), 'frozen_anchor_sha256': anchor_hash(model),
              'source_sha256': source_hashes(), 'training_count': len(train),
              'validation_count': len(validation), 'selection_split': 'val',
              'evaluation_policy': {'minimum_accuracy_percent': MINIMUM_ACCURACY_PERCENT,
                                    'active_support_branch_required': True},
              'baseline_and_batchnorm_frozen': True, 'new_pixel_encoder_trainable': True,
              'answer_candidates_in_adaptation': False, 'answer_labels_or_xml_in_adaptation': False,
              'visible_panel_count': 5, 'target_panel_count': 1,
              'trainable_parameters': sum(p.numel() for p in parameters),
              'width': 96, 'perception_depth': 3, 'pav_stages': 3, 'rank': 8,
              'memory_tokens_per_stage': 48, 'axial_attention_depth': 4,
              'max_region_count': 10, 'evidence_weight': .2,
              'loss_weights': {'ranking': 1., 'completion': .1, 'object_masking': .1},
              'negative_mining': 'same_layout_component_known_targets_up_to_seven',
              'gradient_clip_norm': 1., 'collapse_penalty': False, 'bidirectional_row_task': False}
    start, best, best_epoch = 0, -1., 0
    if args.resume:
        previous = json.loads((root / 'config.json').read_text())
        if {k: v for k, v in previous.items() if k != 'resume'} != {k: v for k, v in config.items() if k != 'resume'}:
            raise RuntimeError('resume changed protocol or source')
        checkpoint = torch.load(root / 'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        if anchor_hash(model) != config['frozen_anchor_sha256']:
            raise RuntimeError('resume checkpoint changed anchor')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        restore_rng(checkpoint['rng'], train_generator, val_generator, device=device)
        start, best, best_epoch = checkpoint['epoch'], checkpoint['best_accuracy'], checkpoint['best_epoch']
        history = [json.loads(line) for line in (root / 'history.jsonl').read_text().splitlines()]
        if [line['epoch'] for line in history] != list(range(1, start + 1)):
            raise RuntimeError('history/checkpoint mismatch')
    else:
        (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        model.eval()
        initial = score(model, args.dataset_root, 'val', device, args.batch_size, args.workers, val_generator)
        (root / 'initial_validation.json').write_text(json.dumps(initial, indent=2) + '\n')
        if initial['correct'] != platform_correct:
            raise RuntimeError(f"initial validation {initial['correct']} did not replay platform anchor {platform_correct}")
        print(json.dumps({'config': config, 'initial_validation': initial}), flush=True)
    for epoch in range(start, args.epochs):
        started, samples, summed, metrics = time.monotonic(), 0, 0., {}
        model.train()
        for pack, _, configs in loader(train, args.batch_size, args.workers, train_generator, training=True):
            pack, configs = to_device(pack, device), configs.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss, statistics = model.ssl(pack, configs)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(grad_norm):
                raise RuntimeError('nonfinite gradient')
            scaler.step(optimizer)
            scaler.update()
            batch = len(pack['views'])
            samples += batch
            summed += float(loss.detach()) * batch
            for key, value in statistics.items():
                metrics[key] = metrics.get(key, 0.) + value * (1 if key.endswith(('queries', 'correct', 'panels')) else batch)
            if samples % (args.batch_size * 32) == 0:
                print(json.dumps({'phase': 'training', 'epoch': epoch + 1, 'training_seen': samples,
                                  'training_total': 42000, 'mean_loss': summed / samples,
                                  'elapsed_seconds': time.monotonic() - started}), flush=True)
        if samples != 42000:
            raise RuntimeError('incomplete training epoch')
        model.eval()
        observed = score(model, args.dataset_root, 'val', device, args.batch_size, args.workers, val_generator)
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
        record = {'epoch': epoch + 1, 'lineage_epoch': source_epoch + epoch + 1,
                  'validation': observed, 'best_accuracy': best, 'best_epoch': best_epoch,
                  'training_count': samples, 'loss': summed / samples,
                  'gate': float(model.gate.detach().tanh()), 'wall_seconds': time.monotonic() - started,
                  'training_metrics': {k: v if k.endswith(('queries', 'correct', 'panels')) else v / samples
                                       for k, v in metrics.items()}}
        with (root / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)
    if anchor_hash(model) != config['frozen_anchor_sha256']:
        raise RuntimeError('anchor changed during adaptation')
    seal = {'schema_version': 1, 'method': config['method'], 'source_sha256': source_hashes(),
            'config_sha256': digest(root / 'config.json'),
            'anchor_sha256': config['anchor_sha256'], 'frozen_anchor_sha256': config['frozen_anchor_sha256'],
            'checkpoint_sha256': {name: digest(root / f'{name}.pt') for name in ('best', 'final')},
            'best_epoch': best_epoch, 'epochs': args.epochs, 'test_opened': False}
    (root / 'training_complete.json').write_text(json.dumps(seal, indent=2) + '\n')


if __name__ == '__main__':
    main()
