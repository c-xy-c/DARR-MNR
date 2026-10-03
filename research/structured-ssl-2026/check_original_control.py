"""Replay the native control against retained f43cf58 code and data boundaries."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import tempfile

import cv2
import numpy as np
import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    current = load_module('native_current', Path(__file__).with_name('original_continuation.py'))
    torch.set_num_threads(4)
    torch.manual_seed(12345)
    state = torch.load(args.baseline, map_location='cpu', weights_only=False)['model']
    if current.digest(args.baseline) != current.FOUNDATION_SHA256:
        raise RuntimeError('declared native foundation required')
    with tempfile.TemporaryDirectory(prefix='structured-native-contracts-') as temporary:
        temporary = Path(temporary)
        hashes = {}
        for name in ('model', 'data'):
            content = subprocess.check_output(['git', 'show', f'f43cf58:program_ssl/{name}.py'], cwd=repository)
            (temporary / f'{name}.py').write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        # The inherited network/layout sources also need to be unchanged.
        for name in ('model', 'layers', 'views', 'data'):
            earlier = subprocess.check_output(['git', 'show', f'f43cf58:sspredrnet/{name}.py'], cwd=repository)
            if earlier != (repository / f'sspredrnet/{name}.py').read_bytes():
                raise RuntimeError(f'inherited source changed: {name}')
        old = load_module('native_reference', temporary / 'model.py')
        old_data = load_module('native_reference_data', temporary / 'data.py')
        native = current.OriginalContinuation()
        native.load_state_dict(state, strict=True)
        reference = old.ComponentProgram(use_program=False)
        reference.load_baseline(state)
        inputs = torch.rand(4, 2, 16, 80, 80) * 2 - 1
        inputs[:, :, 9] = inputs[:, :, 5]
        native.eval()
        reference.eval()
        with torch.no_grad():
            assert torch.equal(native(inputs), reference(inputs))
        native.train()
        reference.train()
        torch.manual_seed(55)
        errors = native(inputs)
        torch.manual_seed(55)
        earlier_errors = reference(inputs)['errors']
        assert all(torch.equal(a, b) for a, b in zip(errors, earlier_errors))
        loss = current.prediction_loss(errors) / len(inputs)
        earlier_loss = current.prediction_loss(earlier_errors) / len(inputs)
        assert torch.equal(loss, earlier_loss)
        loss.backward()
        earlier_loss.backward()
        reference_parameters = dict(reference.named_parameters())
        gradient_difference = max(float((p.grad - reference_parameters[name].grad).abs().max())
                                  for name, p in native.named_parameters() if p.requires_grad)
        assert gradient_difference == 0
        assert all(name.startswith('reasoner.prb') for name, p in native.named_parameters() if p.requires_grad)
        before_cnn, before_bn = current.perception_hash(native), current.batchnorm_hash(native)
        optimizer = torch.optim.Adam([p for p in native.parameters() if p.requires_grad], lr=3e-5, weight_decay=1e-5)
        torch.nn.utils.clip_grad_norm_([p for p in native.parameters() if p.requires_grad], 1.)
        optimizer.step()
        assert current.perception_hash(native) == before_cnn
        assert current.batchnorm_hash(native) == before_bn
        assert all(not m.training for m in native.modules() if isinstance(m, torch.nn.BatchNorm2d))
        for config in current.CONFIGS:
            directory = temporary / 'RAVEN' / config
            directory.mkdir(parents=True)
            panels = np.full((16, 160, 160), 255, np.uint8)
            for index, panel in enumerate(panels):
                if config.startswith('in_'):
                    cv2.rectangle(panel, (8, 8), (151, 151), 0, 3)
                    cv2.circle(panel, (80, 80), 10 + index % 3, 40 + index, 2)
                else:
                    cv2.rectangle(panel, (10, 55), (48 + index % 3, 100), 30 + index, 2)
                    cv2.circle(panel, (125, 80), 9 + index % 3, 60 + index, -1)
            # Accessing target would raise with allow_pickle=False. The native
            # task is allowed to use candidate images, never their answer index.
            np.savez(directory / 'RAVEN_0_train.npz', image=panels,
                     target=np.array([{'must_not_be_read': True}], dtype=object))
        dataset = current.OriginalKnownRows(temporary / 'RAVEN')
        earlier_dataset = old_data.KnownRowTraining(temporary / 'RAVEN', include_candidates=True)
        for index, path in enumerate(dataset.paths):
            random.seed(20 + index)
            observed = dataset[index]
            random.seed(20 + index)
            expected = earlier_dataset[index]
            assert torch.equal(observed[0], expected[0]) and observed[1:] == expected[1:]
            assert observed[1] == -1
            with np.load(path, allow_pickle=False) as archive:
                changed = archive['image'].copy()
            changed[5] = 255
            changed[8:] = 0
            np.savez(path, image=changed, target=np.array([{'must_not_be_read': True}], dtype=object))
            random.seed(20 + index)
            altered = dataset[index]
            assert torch.equal(observed[0][:, :5], altered[0][:, :5])
        report = {'fixture': 'synthetic tensors and NPZs', 'full_raven_evaluation': False,
                  'device': 'cpu', 'torch': str(torch.__version__), 'reference_commit': 'f43cf58',
                  'reference_source_sha256': hashes, 'native_source_sha256': current.source_hashes(),
                  'foundation_sha256': current.digest(args.baseline),
                  'native_eval_bitwise_equal_to_earlier_control': True,
                  'native_training_errors_bitwise_equal_to_earlier_control': True,
                  'native_margin_loss_equal': True, 'native_gradient_max_difference': gradient_difference,
                  'trainable_parameters': sum(p.numel() for p in native.parameters() if p.requires_grad),
                  'only_original_prbs_trainable': True, 'cnn_and_batchnorm_buffers_unchanged_after_update': True,
                  'all_seven_layouts_data_equal_to_earlier_control': True,
                  'target_or_candidates_cannot_change_five_visible_panels': True,
                  'object_dtype_answer_field_was_not_read': True}
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
