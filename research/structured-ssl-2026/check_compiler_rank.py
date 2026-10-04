"""Sample effective ranks of sealed support-generated P/G/V maps, validation only.

Loads the exact source export for a completed training job. The result describes
dynamic operators, not semantic rule discovery or benchmark accuracy.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = values.detach().cpu().numpy().astype(float)
    return {'median': float(np.median(a)), 'min': float(a.min()), 'max': float(a.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, run = args.source_root.resolve(), args.run_dir.resolve()
    seal = json.loads((run / 'training_complete.json').read_text())
    for name, expected in seal['source_sha256'].items():
        if digest(source / name) != expected:
            raise RuntimeError('sealed source changed')
    config = json.loads((run / 'config.json').read_text())
    if digest(run / 'config.json') != seal['config_sha256']:
        raise RuntimeError('sealed configuration changed')
    sys.path.insert(0, str(source))
    implementation = importlib.import_module('structured_ssl.model')
    dimensions = (importlib.import_module('structured_ssl.constants')
                  if (source / 'structured_ssl/constants.py').exists() else implementation)
    data = importlib.import_module('structured_ssl.data')
    if Path(implementation.__file__).resolve() != source / 'structured_ssl/model.py':
        raise RuntimeError('wrong implementation imported')
    torch.set_num_threads(2)
    rng = np.random.default_rng(23456)
    indices = [int(i * 2000 + j) for i in range(7) for j in rng.choice(2000, 3, replace=False)]
    dataset = data.ObjectRaven(args.dataset_root, 'val')
    items = [dataset[i] for i in indices]
    pack = data.to_device({k: torch.stack([x[0][k] for x in items]) for k in items[0][0]}, torch.device('cpu'))
    layout = torch.tensor([x[2] for x in items]).repeat_interleave(2)
    component = torch.arange(len(layout)).remainder(2)
    model = implementation.StructuredCompletion().eval()
    records = {}
    with torch.inference_mode():
        for checkpoint in ('best', 'final'):
            file = run / (checkpoint + '.pt')
            if digest(file) != seal['checkpoint_sha256'][checkpoint]:
                raise RuntimeError('checkpoint bytes changed')
            state = torch.load(file, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            if hasattr(model, 'encode_context'):
                context = model.encode_context(pack, 6)
                levels, valid = context.levels, context.valid
            else:  # Diagnosis of an immutable historical source export.
                levels, visible = model.encode(pack, 6)
                valid = visible['valid']
            matrices = []
            for row in (slice(0, 3), slice(3, 6)):
                factors = model.pav.compiler(levels[:, row], valid[:, row])
                a, b = factors.unbind(3)
                matrices.append(a.transpose(-1, -2) @ b / math.sqrt(dimensions.RANK))
            maps = torch.stack(matrices, 1)  # Component x support row x stage x role x 96 x 96.
            singular = torch.linalg.svdvals(maps)[..., :dimensions.RANK]
            distribution = singular / singular.sum(-1, keepdim=True).clamp_min(1e-8)
            effective_rank = (-(distribution * distribution.clamp_min(1e-12).log()).sum(-1)).exp()
            dominant_energy = singular[..., 0].square() / singular.square().sum(-1).clamp_min(1e-8)
            rows = {}
            for stage in range(dimensions.STAGES):
                for role, name in enumerate(('P', 'G', 'V')):
                    vectors = maps[:, :, stage, role].flatten(2)
                    differences, correlations = [], []
                    for group in range(7):
                        for view in range(2):
                            subset = vectors[(layout == group) & (component == view)]
                            rolled = subset.roll(1, 0)
                            correlations.append(F.cosine_similarity(subset, rolled, dim=-1))
                            differences.append((subset - rolled).norm(dim=-1) / subset.norm(dim=-1).clamp_min(1e-8))
                    rows[f'stage{stage + 1}_{name}'] = {
                        'effective_rank': stats(effective_rank[:, :, stage, role]),
                        'leading_direction_energy_fraction': stats(dominant_energy[:, :, stage, role]),
                        'same_layout_component_map_cosine': stats(torch.cat(correlations)),
                        'same_layout_component_map_relative_l2_change': stats(torch.cat(differences))}
            records[checkpoint] = {'epoch': state['epoch'], 'maps': rows}
    result = {'fixture': '21 sampled validation puzzles, three per layout, both complete support rows',
              'validation_indices': indices, 'seed': 23456, 'teacher_device': 'cpu',
              'declared_rank': dimensions.RANK, 'effective_rank_definition': 'exp entropy of top8 singular values',
              'support_variation_comparison': 'cyclic different puzzle within same layout and component',
              'run_dir': str(run), 'source_root': str(source), 'seal': seal,
              'diagnostic_source_sha256': digest(__file__), 'checkpoints': records,
              'test_opened': False, 'parameter_updates': 0, 'full_raven_evaluation': False,
              'limitations': ['Three puzzles per layout do not characterize the entire dataset.',
                              'Low effective rank does not prove an operator is useless.',
                              'Operator changes do not prove semantic rules or an accuracy benefit.',
                              'These diagnostics do not change any running or queued source export.']}
    with args.output.open('x') as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'checkpoints': records, 'full_raven_evaluation': False}), flush=True)


if __name__ == '__main__':
    main()
