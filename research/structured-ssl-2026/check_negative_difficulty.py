"""Compare frozen-teacher mining distances on sampled train/validation panels.

Training uses only known panels. Validation answer indices locate the true
candidate for this diagnostic, never for optimization. No parameters, running
sources or test data change. Distances do not measure reasoning accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from attention_ssl.model import AttentionCompletion
from program_ssl.data import KnownRowTraining
from sspredrnet.data import CONFIGS, Raven, normalize
from structured_ssl.model import dense_energy, pixel_duplicates


def statistics(values):
    array = values.detach().cpu().numpy().astype(float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {'count': 0, 'median': None, 'p10': None, 'p90': None}
    return {'count': len(array), 'median': float(np.median(array)),
            'p10': float(np.percentile(array, 10)), 'p90': float(np.percentile(array, 90))}


def encode(teacher, views):
    # Bound immutable CNN activation memory while the Metal suite runs.
    return torch.cat([teacher.features(batch) for batch in views.split(8)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    random.seed(12345)
    rng = np.random.default_rng(12345)
    torch.manual_seed(12345)
    torch.set_num_threads(2)
    teacher = AttentionCompletion().requires_grad_(False).eval()
    teacher.load_state_dict(torch.load(args.anchor, map_location='cpu', weights_only=False)['model'])
    training, validation = KnownRowTraining(args.dataset_root), Raven(args.dataset_root, 'val')
    if len(training) != 42000 or len(validation) != 14000:
        raise RuntimeError('official complete splits required')
    train_indices = [int(layout * 6000 + offset) for layout in range(7)
                     for offset in rng.choice(6000, 19 if layout < 2 else 18, replace=False)]
    rng.shuffle(train_indices)
    assert len(train_indices) == 128
    train_items = [training[index] for index in train_indices]
    train_views = normalize(torch.stack([item[0] for item in train_items]))
    train_configs = torch.tensor([item[2] for item in train_items]).repeat_interleave(2)
    validation_indices = [int(layout * 2000 + offset) for layout in range(7)
                          for offset in rng.choice(2000, 8, replace=False)]
    val_items = [validation[index] for index in validation_indices]
    val_views = normalize(torch.stack([item[0] for item in val_items]))
    val_labels = torch.tensor([item[1] for item in val_items]).repeat_interleave(2)
    val_configs = torch.tensor([item[2] for item in val_items]).repeat_interleave(2)
    with torch.inference_mode():
        train_features = encode(teacher, train_views)
        target = train_features[:, 5]
        distances = dense_energy(F.relu(target)[:, None], target)[:, 0]
        ids = torch.arange(len(target))
        groups = (train_configs[:, None].eq(train_configs[None]) &
                  ids[:, None].remainder(2).eq(ids[None].remainder(2)))
        duplicates = pixel_duplicates(train_views.flatten(0, 1)[:, 5].flatten(1))
        legal = groups & ~duplicates
        distances.masked_fill_(~legal, torch.inf)
        nearest = distances.topk(7, largest=False).values
        val_features = encode(teacher, val_views)
        val_ids = torch.arange(len(val_features))
        candidates = val_features[:, 8:]
        truth = candidates[val_ids, val_labels]
        candidate_distances = dense_energy(F.relu(truth)[:, None], candidates)[:, 0]
        positive = F.one_hot(val_labels, 8).bool()
        candidate_distances.masked_fill_(positive, torch.inf)
        val_pixels = val_views.flatten(0, 1)[:, 8:].flatten(2)
        pixel_truth = val_pixels[val_ids, val_labels]
        duplicate_candidates = val_pixels.eq(pixel_truth[:, None]).all(-1) & ~positive
        distinct = candidate_distances.masked_fill(duplicate_candidates, torch.inf)
        layouts = {}
        for index, name in enumerate(CONFIGS):
            train_mask, val_mask = train_configs == index, val_configs == index
            layouts[name] = {
                'train_known_sixth_nearest_negative': statistics(nearest[train_mask, 0]),
                'train_known_sixth_seven_nearest': statistics(nearest[train_mask].flatten()),
                'train_legal_donors': statistics(legal[train_mask].sum(-1)),
                'validation_true_answer_nearest_wrong_candidate': statistics(candidate_distances[val_mask].min(-1).values),
                'validation_true_answer_nearest_pixel_distinct_wrong_candidate': statistics(distinct[val_mask].min(-1).values),
                'validation_seven_wrong_candidates': statistics(candidate_distances[val_mask].flatten()),
                'validation_duplicate_component_candidates': int(duplicate_candidates[val_mask].sum())}
    report = {'fixture': '128 balanced sampled training puzzles and 56 sampled validation puzzles',
              'seed': 12345, 'training_indices': train_indices, 'validation_indices': validation_indices,
              'train_panels_per_component': 6, 'validation_panels_per_component': 16,
              'teacher_device': 'cpu', 'teacher_precision': 'float32',
              'anchor_sha256': hashlib.sha256(Path(args.anchor).read_bytes()).hexdigest(),
              'diagnostic_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'split_usage': {'train': 'known sixth only for mining, no answer candidates or indices',
                              'val': 'true answer index used only for candidate distance diagnostic'},
              'test_opened': False, 'parameter_updates': 0, 'full_raven_evaluation': False,
              'running_or_queued_design_modified': False, 'by_configuration': layouts,
              'limitations': ['Balanced donor pool is a sample, not the random training batch sequence.',
                              'Distances use the dense frozen-teacher mining metric, not the complete scoring model.',
                              'Component duplicates can be distinguished by the other component at puzzle level.',
                              'Known-sixth and true-third-cell images are different tasks; this comparison is descriptive.',
                              'A distance gap alone does not prove it causes validation regression.']}
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'configurations': layouts, 'full_raven_evaluation': False}), flush=True)


if __name__ == '__main__':
    main()
