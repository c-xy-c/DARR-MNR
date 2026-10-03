"""Replay the failed native epoch from its sealed source/checkpoint on Metal.

Diagnostic updates are discarded. On the first nonfinite gradient, retry the
same pixels, model and dropout RNG at lower loss scales. Also count zero L2
distances, since sqrt(sum(error**2)) has an undefined gradient at exact zero.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repaired-model', type=Path)
    args = parser.parse_args()
    source, run = args.source_root.resolve(), args.run_dir.resolve()
    config = json.loads((run / 'config.json').read_text())
    for name, expected in config['source_sha256'].items():
        if digest(source / name) != expected:
            raise RuntimeError('original native source changed')
    sys.path.insert(0, str(source))
    file = source / 'research/structured-ssl-2026/original_continuation.py'
    sys.path.insert(0, str(file.parent))
    native = importlib.import_module('original_continuation')
    if Path(native.__file__).resolve() != file:
        raise RuntimeError('replay imported a different native source')
    device = torch.device(config['device'])
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = native.OriginalContinuation().to(device)
    checkpoint = torch.load(run / 'last.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config['lr'], weight_decay=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler = torch.amp.GradScaler(device.type)
    scaler.load_state_dict(checkpoint['scaler'])
    train_generator, val_generator = torch.Generator(), torch.Generator()
    native.restore_rng(checkpoint['rng'], train_generator, val_generator, device=device)
    dataset = native.OriginalKnownRows(config['dataset_root'])
    distances = {}
    for index in (1, 2):
        def hook(_, inputs, output, index=index):
            distances[index] = output[1].detach()
        getattr(model.reasoner, f'prb{index}').register_forward_hook(hook)
    model.train()
    updates, failure = 0, None
    for batch_index, (images, _, _) in enumerate(native.loader(dataset, config['batch_size'], config['workers'], train_generator, training=True)):
        images = native.normalize(images.to(device))
        dropout_rng, cpu_rng = torch.mps.get_rng_state(), torch.get_rng_state()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            loss = native.prediction_loss(model(images)) / len(images)
        if not torch.isfinite(loss):
            raise RuntimeError('replay encountered a nonfinite forward loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm):
            zeros = {str(i): {'exact_zero_count': int(v.eq(0).sum()), 'total': v.numel(), 'dtype': str(v.dtype)}
                     for i, v in distances.items()}
            retries = []
            for scale in (65536., 32768., 1.):
                optimizer.zero_grad(set_to_none=True)
                torch.mps.set_rng_state(dropout_rng)
                torch.set_rng_state(cpu_rng)
                trial = torch.amp.GradScaler(device.type, init_scale=scale)
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    repeated = native.prediction_loss(model(images)) / len(images)
                trial.scale(repeated).backward()
                trial.unscale_(optimizer)
                retry_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                retries.append({'scale': scale, 'loss': float(repeated.detach()),
                                'same_loss_bitwise': bool(torch.equal(loss.detach(), repeated.detach())),
                                'finite_gradient_norm': bool(torch.isfinite(retry_norm))})
            repaired_retry = None
            if args.repaired_model:
                spec = importlib.util.spec_from_file_location('sspredrnet.repaired_model', args.repaired_model.resolve())
                repaired = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(repaired)
                original_blocks = importlib.import_module('sspredrnet.model')
                for module in model.modules():
                    if isinstance(module, original_blocks.PredictiveBlock):
                        module.forward = types.MethodType(repaired.PredictiveBlock.forward, module)
                optimizer.zero_grad(set_to_none=True)
                torch.mps.set_rng_state(dropout_rng)
                torch.set_rng_state(cpu_rng)
                stable_scaler = torch.amp.GradScaler(device.type, init_scale=scaler.get_scale())
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    stable_loss = native.prediction_loss(model(images)) / len(images)
                stable_scaler.scale(stable_loss).backward()
                stable_scaler.unscale_(optimizer)
                stable_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                repaired_retry = {'source_sha256': digest(args.repaired_model),
                                  'scale': stable_scaler.get_scale(),
                                  'same_loss_bitwise': bool(torch.equal(loss.detach(), stable_loss.detach())),
                                  'finite_gradient_norm': bool(torch.isfinite(stable_norm))}
                if not repaired_retry['same_loss_bitwise'] or not repaired_retry['finite_gradient_norm']:
                    raise RuntimeError('stable L2 did not preserve loss and repair the actual failing batch')
            failure = {'batch_index': batch_index, 'loss': float(loss.detach()),
                       'scale': scaler.get_scale(), 'zero_distance_counts': zeros,
                       'same_batch_lower_scale_retries': retries,
                       'stable_l2_same_batch_retry': repaired_retry}
            break
        scaler.step(optimizer)
        scaler.update()
        updates += 1
        if updates % 32 == 0:
            print(json.dumps({'replayed_updates': updates, 'epoch': checkpoint['epoch'] + 1}), flush=True)
    report = {'fixture': 'replay from actual native last checkpoint and full loader RNG',
              'formal_run_modified': False, 'formal_checkpoint_sha256': digest(run / 'last.pt'),
              'replay_epoch': checkpoint['epoch'] + 1, 'diagnostic_discarded_updates': updates,
              'failure': failure, 'test_opened': False, 'full_raven_evaluation': False,
              'source_sha256': config['source_sha256'], 'diagnostic_sha256': digest(__file__)}
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
