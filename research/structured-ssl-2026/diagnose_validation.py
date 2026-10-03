"""Validation-only checks of object-target locality and support feedback.

Does not change a running experiment or inspect the test set. These mechanism
checks cannot establish why a whole benchmark regressed, or prove a fix helps.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from attention_ssl.model import AttentionCompletion
from sspredrnet.data import CONFIGS
from structured_ssl.data import ObjectRaven, to_device
from structured_ssl.model import dense_energy


def target_objects(features, masks, geometry, valid):
    """Historical v1 target, kept solely to reproduce the recorded diagnosis.

    The active production target now encodes isolated bbox images. This helper
    describes the immutable source-primary export of the first Metal suite.
    """
    n, panels = features.shape[:2]
    weights = F.avg_pool2d(masks.float().reshape(n * panels * 10, 1, 10, 10), 2)
    weights = weights.reshape(n, panels, 10, 25)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    appearance = weights @ F.relu(features.float()).transpose(2, 3)
    return torch.cat((appearance, geometry.float()), -1) * valid[..., None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    teacher = AttentionCompletion().requires_grad_(False).eval()
    teacher.load_state_dict(torch.load(args.anchor, map_location='cpu', weights_only=False)['model'])
    receptive_field, stride = 1, 1
    for index in range(4):
        for name in ('conv1', 'conv2', 'conv3'):
            conv = getattr(getattr(teacher.reasoner, f'res{index}'), name)[0]
            receptive_field += (conv.kernel_size[0] - 1) * stride
            stride *= conv.stride[0]
    assert receptive_field == 151 and stride == 16
    dataset = ObjectRaven(args.dataset_root, 'val')
    records = []
    with torch.inference_mode():
        for config in ('distribute_four', 'distribute_nine'):
            offset = CONFIGS.index(config) * 2000
            accepted = 0
            for index in range(offset, offset + 100):
                pack, _, _ = dataset[index]
                if int(pack['valid'][0, 0].sum()) < 2:
                    continue
                pack = to_device({k: v[None, :, :1] for k, v in pack.items()}, torch.device('cpu'))
                x0, y0, x1, y1 = pack['boxes'][0, 0, 0, 0].tolist()
                changed = pack['views'].clone()
                outside = torch.ones(80, 80, dtype=torch.bool)
                outside[y0:y1, x0:x1] = False
                changed[0, 0, 0].masked_fill_(outside, 1.)
                original_pixels = pack['views'][0, 0, 0]
                modified_pixels = changed[0, 0, 0]
                removed = int((original_pixels[outside] != modified_pixels[outside]).sum())
                if removed == 0:
                    continue
                assert torch.equal(original_pixels[y0:y1, x0:x1], modified_pixels[y0:y1, x0:x1])
                tensors = [pack[key].flatten(0, 1) for key in ('masks', 'geometry', 'valid')]
                original_target = target_objects(teacher.features(pack['views']), *tensors)[0, 0, 0]
                modified_target = target_objects(teacher.features(changed), *tensors)[0, 0, 0]
                assert torch.equal(original_target[32:], modified_target[32:])
                first, second = original_target[:32], modified_target[:32]
                records.append({'configuration': config, 'validation_index': index,
                                'target_bbox_pixels_bitwise_preserved': True,
                                'target_geometry_bitwise_preserved': True,
                                'outside_changed_pixels': removed,
                                'appearance_relative_l2_change': float((first - second).norm() / first.norm().clamp_min(1e-6)),
                                'appearance_normalized_squared_change': float((F.normalize(first, dim=0, eps=1e-3) -
                                                                                F.normalize(second, dim=0, eps=1e-3)).square().sum())})
                accepted += 1
                if accepted == 2:
                    break
        assert len(records) == 4
        target = torch.zeros(1, 32, 25)
        target[0, torch.arange(25), torch.arange(25)] = 1
        prediction = target.roll(1, -1)
        pooled_error = (prediction.mean(-1) - target.mean(-1)).square().mean()
        spatial_error = dense_energy(prediction[:, None], target[:, None])[0, 0, 0]
        assert pooled_error == 0 and spatial_error > 0
    history = [json.loads(line) for line in (Path(args.run_dir) / 'history.jsonl').read_text().splitlines()]
    report = {'split': 'val', 'test_opened': False, 'full_benchmark_result': False,
              'anchor_sha256': hashlib.sha256(Path(args.anchor).read_bytes()).hexdigest(),
              'cnn_theoretical_max_receptive_field_px': receptive_field, 'image_size_px': 80,
              'object_locality_interventions': records,
              'spatial_permutation_counterexample': {'mean_feedback_error': float(pooled_error),
                                                      'deployed_dense_discrepancy': float(spatial_error)},
              'observed_complete_epoch_validation': [{'epoch': row['epoch'],
                  'correct': row['validation']['correct'], 'total': row['validation']['total'],
                  'gate': row['gate'], 'masked_loss': row['training_metrics']['masked_object'],
                  'completion_loss': row['training_metrics']['completion']} for row in history],
              'limitations': ['Four locality interventions are not a whole-dataset causal attribution.',
                             'A mean-feedback counterexample demonstrates blindness, not the cause of all mistakes.',
                             'No alternative objective or network has been trained/evaluated by this script.']}
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
