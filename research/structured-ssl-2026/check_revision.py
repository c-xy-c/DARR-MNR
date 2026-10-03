"""Replay the four recorded validation interventions against isolated targets.

This checks target locality only. It does not train, use answer indices, open
test data, or estimate full RAVEN accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from structured_ssl.data import ObjectRaven, to_device
from structured_ssl.model import StructuredCompletion
from structured_ssl.train import source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--diagnosis', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = StructuredCompletion().eval()
    model.load_anchor(torch.load(args.anchor, map_location='cpu', weights_only=False)['model'])
    diagnosis = json.loads(Path(args.diagnosis).read_text())
    anchor_sha = hashlib.sha256(Path(args.anchor).read_bytes()).hexdigest()
    if diagnosis['split'] != 'val' or diagnosis['test_opened'] or diagnosis['anchor_sha256'] != anchor_sha:
        raise RuntimeError('requires the recorded validation-only diagnosis and same teacher')
    dataset = ObjectRaven(args.dataset_root, 'val')
    records = []
    with torch.inference_mode():
        for original in diagnosis['object_locality_interventions']:
            pack, _, _ = dataset[original['validation_index']]
            pack = to_device({k: v[None, :, :1] for k, v in pack.items()}, torch.device('cpu'))
            changed = {k: v.clone() for k, v in pack.items()}
            x0, y0, x1, y1 = pack['boxes'][0, 0, 0, 0].tolist()
            outside = torch.ones(80, 80, dtype=torch.bool)
            outside[y0:y1, x0:x1] = False
            changed['views'][0, 0, 0].masked_fill_(outside, 1.)
            removed = int((changed['views'][0, 0, 0] != pack['views'][0, 0, 0]).sum())
            if removed != original['outside_changed_pixels']:
                raise RuntimeError('intervention no longer matches recorded pixels')
            first, _ = model.targets(pack)
            second, _ = model.targets(changed)
            before, after = first[0, 0, 0], second[0, 0, 0]
            assert torch.equal(before, after), 'isolated target depends on pixels outside bbox'
            records.append({'configuration': original['configuration'],
                            'validation_index': original['validation_index'],
                            'outside_changed_pixels': removed,
                            'historical_v1_appearance_relative_l2_change': original['appearance_relative_l2_change'],
                            'revised_target_bitwise_unchanged': True,
                            'revised_appearance_relative_l2_change': float((before[:32] - after[:32]).norm())})
    report = {'architecture_revision': 'isolated-target-structured-feedback-v2',
              'split': 'val', 'test_opened': False, 'full_raven_evaluation': False,
              'anchor_sha256': anchor_sha, 'source_sha256': source_hashes(),
              'diagnosis_sha256': hashlib.sha256(Path(args.diagnosis).read_bytes()).hexdigest(),
              'interventions': records,
              'limitations': ['Targets still preserve location and scale, not proven semantic attributes.',
                              'Pixels of overlapping objects inside a bbox remain visible to the teacher.',
                              'Locality repair is not evidence of improved RAVEN accuracy.']}
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'interventions': len(records), 'all_isolated_targets_bitwise_unchanged': True,
                      'full_raven_evaluation': False}), flush=True)


if __name__ == '__main__':
    main()
