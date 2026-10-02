"""Exact FP32 evaluation cache for the immutable baseline image encoder.

Only validation/test evaluation is cached. Training views and gradients are
untouched. Test caches may be built only by the checkpoint-locked evaluator.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from sspredrnet.data import CONFIGS, Raven, loader, normalize
from sspredrnet.evaluate import score as native_score
from .model import ComponentProgram


def encoder_digest(model):
    digest = hashlib.sha256()
    for key, tensor in model.reasoner.state_dict().items():
        if key.startswith(('res', 'channel_reducer')):
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenFeatureCache:
    def __init__(self, model, dataset_root, split, cache_root, device,
                 *, batch_size=128, workers=8):
        if split not in ('val', 'test'):
            raise ValueError('only frozen evaluation features can be cached')
        self.device, self.batch_size, self.split = device, batch_size, split
        dataset = Raven(dataset_root, split)
        if len(dataset) != 14000:
            raise RuntimeError('a complete evaluation split is required')
        stamps = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in dataset.paths]
        package = Path(__file__).resolve().parents[1]
        code_paths = [package / 'sspredrnet' / name
                      for name in ('data.py', 'views.py', 'model.py', 'layers.py')]
        code_paths.append(Path(__file__).with_name('model.py'))
        metadata = {'encoder_sha256': encoder_digest(model), 'split': split,
                    'dataset_sha256': hashlib.sha256(json.dumps(stamps).encode()).hexdigest(),
                    'source_sha256': {str(p.relative_to(package)):
                        hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
                    'torch': str(torch.__version__), 'numpy': str(np.__version__),
                    'opencv': str(cv2.__version__), 'gpu': torch.cuda.get_device_name(device),
                    'cudnn_deterministic': torch.backends.cudnn.deterministic,
                    'cudnn_benchmark': torch.backends.cudnn.benchmark,
                    'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                    'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                    'batch_size': batch_size, 'dtype': 'float32'}
        self.fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        directory = Path(cache_root)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f'{split}-{self.fingerprint[:32]}.pt'
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not self.path.exists():
                self._build(model, dataset, metadata, workers)
            saved = torch.load(self.path, map_location='cpu', weights_only=True)
        if saved['metadata'] != metadata:
            raise RuntimeError('evaluation cache provenance mismatch')
        self.metadata = metadata
        self.features = saved['features'].to(device)
        self.answers, self.configurations = saved['answers'], saved['configurations']
        if tuple(self.features.shape) != (14000, 2, 16, 32, 25) or self.features.dtype != torch.float32:
            raise RuntimeError('evaluation cache shape or precision mismatch')

    def _build(self, model, dataset, metadata, workers):
        model.eval()
        features, answers, configurations = [], [], []
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            for images, labels, configs in loader(dataset, self.batch_size, workers,
                    torch.Generator().manual_seed(12346)):
                images = normalize(images.to(self.device, non_blocking=True))
                encoded = model.features(images).reshape(len(images), 2, 16, 32, 25)
                features.append(encoded.cpu())
                answers.append(labels)
                configurations.append(configs)
        temporary = self.path.with_suffix('.tmp')
        torch.save({'metadata': metadata, 'features': torch.cat(features),
                    'answers': torch.cat(answers),
                    'configurations': torch.cat(configurations)}, temporary)
        os.replace(temporary, self.path)

    def score(self, model, *, generator=None):
        model.eval()
        if encoder_digest(model) != self.metadata['encoder_sha256']:
            raise RuntimeError('image encoder changed after feature caching')
        # Native DataLoader evaluation consumes one CPU base-seed draw. Match
        # that draw so checkpoint RNG states retain the same val-generator path.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        correct, total = [0] * 7, [0] * 7
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            for start in range(0, len(self.features), self.batch_size):
                stop = min(start + self.batch_size, len(self.features))
                features = self.features[start:stop].flatten(0, 1)
                predictions = model.predict_features(features, stop - start).argmin(1).cpu()
                labels, configs = self.answers[start:stop], self.configurations[start:stop]
                for index in range(7):
                    mask = configs == index
                    total[index] += int(mask.sum())
                    correct[index] += int(((predictions == labels) & mask).sum())
        if total != [2000] * 7:
            raise RuntimeError('incomplete cached evaluation')
        by_config = {c: 100 * n / t for c, n, t in zip(CONFIGS, correct, total)}
        return {'accuracy_macro': sum(by_config.values()) / 7,
                'accuracy_micro': 100 * sum(correct) / sum(total),
                'by_configuration': by_config, 'correct': sum(correct), 'total': sum(total)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--program-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    baseline = ComponentProgram(use_program=False).to(device).eval()
    baseline.load_baseline(torch.load(args.baseline, map_location=device, weights_only=False)['model'])
    cache = FrozenFeatureCache(baseline, args.dataset_root, 'val', args.cache_root, device)
    program = ComponentProgram().to(device).eval()
    program.load_state_dict(torch.load(args.program_checkpoint, map_location=device,
                                       weights_only=False)['model'], strict=True)
    reports = {}
    for name, model, verification in (('baseline', baseline, 'posterior'),
                                     ('program', program, 'posterior'),
                                     ('evidence_ratio', program, 'support_evidence_ratio')):
        model.verification_mode = verification
        first = torch.Generator().manual_seed(12346)
        second = torch.Generator().manual_seed(12346)
        cuda_before = torch.cuda.get_rng_state_all()
        native = native_score(model, args.dataset_root, 'val', device, generator=first)
        cached = cache.score(model, generator=second)
        if native != cached:
            raise RuntimeError(f'{name}: full native/cache validation mismatch')
        if not torch.equal(first.get_state(), second.get_state()):
            raise RuntimeError('cached scoring changes validation RNG state')
        if not all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all())):
            raise RuntimeError('evaluation unexpectedly consumes CUDA RNG')
        reports[name] = {'complete_validation_equal': True, 'counts': native,
                         'validation_generator_equal': True, 'cuda_rng_unchanged': True}
    reports['cache_fingerprint'] = cache.fingerprint
    reports['training_source_unchanged'] = True
    Path(args.output).write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
