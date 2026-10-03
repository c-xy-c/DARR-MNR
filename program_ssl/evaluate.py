"""Evaluate the single CECS + SER-PaV model from a sealed run directory."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from sspredrnet.evaluate import score
from .feature_cache import FrozenFeatureCache
from .model import ComponentProgram


RUNTIME_SOURCES = (
    'program_ssl/model.py', 'program_ssl/data.py', 'program_ssl/train.py',
    'program_ssl/evaluate.py', 'program_ssl/feature_cache.py',
    'program_ssl/check_contracts.py', 'sspredrnet/checkpoint.py',
    'sspredrnet/model.py', 'sspredrnet/data.py', 'sspredrnet/views.py',
    'sspredrnet/layers.py', 'sspredrnet/evaluate.py', 'sspredrnet/train.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    package = Path(__file__).resolve().parents[1]
    return {name: digest(package / name) for name in RUNTIME_SOURCES}


def write_inference_config(root, program_weight, *, expected=None):
    """Seal checkpoint bytes and the inference weight before opening test."""
    root = Path(root)
    if not (root / 'training_complete.json').exists():
        raise RuntimeError('training must complete before sealing inference')
    config = {'schema_version': 1, 'method': 'cecs-serpav',
              'program_weight': program_weight, 'selection_split': 'val',
              'source_sha256': source_hashes(),
              'checkpoint_sha256': {n: digest(root / f'{n}.pt') for n in ('best', 'final')},
              'expected': expected or {}}
    path = root / 'inference.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError('existing inference configuration differs')
    path.write_text(json.dumps(config, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--cache-root', help='optional cache of frozen FP32 encoder features')
    parser.add_argument('--output', help='write this evaluation to a new JSON file')
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    root, device = Path(args.run_dir), torch.device(args.device)
    config = json.loads((root / 'inference.json').read_text())
    if config.get('schema_version') != 1 or config.get('method') != 'cecs-serpav':
        raise RuntimeError('a sealed CECS + SER-PaV run is required')
    if source_hashes() != config['source_sha256']:
        raise RuntimeError('runtime sources changed after inference was sealed')
    for name in ('best', 'final'):
        if digest(root / f'{name}.pt') != config['checkpoint_sha256'][name]:
            raise RuntimeError(f'{name} checkpoint changed after inference was sealed')
    if args.output and Path(args.output).exists():
        raise FileExistsError('choose a new output path to preserve previous evaluations')
    model = ComponentProgram(program_weight=config['program_weight']).to(device).eval()
    cache, report = None, {'split': args.split, 'method': config['method'],
                           'program_weight': model.program_weight, 'checkpoints': {}}
    for name in ('best', 'final'):
        checkpoint = torch.load(root / f'{name}.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        if args.cache_root:
            if cache is None:
                cache = FrozenFeatureCache(model, args.dataset_root, args.split,
                                           args.cache_root, device, workers=args.workers)
            observed = cache.score(model, generator=torch.Generator().manual_seed(12347))
        else:
            observed = score(model, args.dataset_root, args.split, device, workers=args.workers,
                             generator=torch.Generator().manual_seed(12347))
        expected = config['expected'].get(name, {}).get(args.split)
        if expected is not None:
            for field in ('correct', 'total', 'by_configuration'):
                if observed[field] != expected[field]:
                    raise RuntimeError(f'{name}: {args.split} {field} differs from verified results')
        report['checkpoints'][name] = {'adaptation_epoch': checkpoint['epoch'],
            'lineage_epoch': checkpoint['source_epoch'] + checkpoint['epoch'],
            'checkpoint_sha256': config['checkpoint_sha256'][name], 'accuracy': observed}
        print(json.dumps({'checkpoint': name, **report['checkpoints'][name]}), flush=True)
    if args.cache_root:
        report['cache_fingerprint'] = cache.fingerprint
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
