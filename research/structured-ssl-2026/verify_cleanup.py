"""Compare a source snapshot with the cleaned runtime in independent processes.

Checks exact scores, masked training losses, every gradient and optimizer step.
Only synthetic fixtures and optional validation samples are read. This is not
a full accuracy measurement or permission to reinterpret historical run seals.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


WORKER = r'''
import json, random, sys
from pathlib import Path
import torch
from structured_ssl.check_contracts import fixture
from structured_ssl.data import to_device, ObjectRaven
from structured_ssl.model import StructuredCompletion

torch.set_num_threads(2)
torch.manual_seed(12345)
model = StructuredCompletion()
model.load_state_dict(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['model'], strict=True)
pack = to_device(fixture(), torch.device('cpu'))
short = {k: v[:, :, :6].clone() for k, v in pack.items()}
model.eval()
with torch.inference_mode():
    scores = model(pack)
    objects = model.targets(short)[0]
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-5)
steps = []
for index in range(2):
    torch.manual_seed(800 + index)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, statistics = model.ssl(short, torch.tensor([0, 0, 1, 1]))
    loss.backward()
    gradients = {k: p.grad.clone() for k, p in model.named_parameters() if p.requires_grad}
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    optimizer.step()
    steps.append({'loss': loss.detach(), 'statistics': statistics, 'gradients': gradients,
                  'state': {k: v.clone() for k, v in model.state_dict().items()}})
observed = {'scores': scores, 'object_targets': objects, 'steps': steps}
if sys.argv[3]:
    model.eval()
    dataset = ObjectRaven(sys.argv[3], 'val')
    counts, selected = {}, []
    for i, path in enumerate(dataset.base.paths):
        layout = path.parent.name
        if counts.get(layout, 0) < 2:
            selected.append(i)
            counts[layout] = counts.get(layout, 0) + 1
    assert len(selected) == 14
    samples = [dataset[i][0] for i in selected]
    pack = to_device({k: torch.stack([p[k] for p in samples]) for k in samples[0]}, torch.device('cpu'))
    with torch.inference_mode():
        observed['real_validation_scores'] = model(pack)
    observed['real_validation_paths'] = [str(dataset.base.paths[i]) for i in selected]
torch.save(observed, sys.argv[2])
'''


def exact(first, second, name='root'):
    import torch
    if isinstance(first, torch.Tensor):
        if not torch.equal(first, second):
            raise AssertionError(f'{name}: max error {(first - second).abs().max().item()}')
    elif isinstance(first, dict):
        assert first.keys() == second.keys(), name
        for key in first:
            exact(first[key], second[key], f'{name}.{key}')
    elif isinstance(first, list):
        assert len(first) == len(second), name
        for index, (a, b) in enumerate(zip(first, second)):
            exact(a, b, f'{name}[{index}]')
    else:
        assert first == second, name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', default='')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='structured-cleanup-parity-') as temporary:
        outputs = []
        for name, source in (('reference', args.reference_source), ('cleaned', repository)):
            output = Path(temporary) / f'{name}.pt'
            subprocess.run([sys.executable, '-c', WORKER, str(args.checkpoint.resolve()),
                            str(output), args.dataset_root], cwd=source,
                           env={**os.environ, 'PYTHONPATH': str(source.resolve())}, check=True)
            import torch
            outputs.append(torch.load(output, map_location='cpu', weights_only=False))
        exact(*outputs)
    report = {'reference_source': str(args.reference_source.resolve()),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'source_sha256': {str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for package in ('structured_ssl', 'attention_ssl', 'sspredrnet')
                               for p in sorted((repository / package).glob('*.py'))},
              'checkpoint_load_strict_and_parameter_names_unchanged': True,
              'synthetic_candidate_scores_and_object_targets_bitwise_equal': True,
              'two_ssl_losses_all_gradients_and_adam_steps_bitwise_equal': True,
              'real_validation_sample_scores_bitwise_equal': bool(args.dataset_root),
              'real_validation_sample_count': 14 if args.dataset_root else 0,
              'test_opened': False, 'full_raven_evaluation': False,
              'historical_training_seals_must_use_their_original_source': True}
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
