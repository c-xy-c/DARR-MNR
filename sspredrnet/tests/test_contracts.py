"""Task boundaries and aggregation contracts for the single SSL pipeline."""
import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np
import torch
from torch import nn

from sspredrnet.data import Raven
from sspredrnet.model import SSPredRNet, prediction_loss
from sspredrnet.views import component_views
from sspredrnet.checkpoint import rng_state, restore_rng


class FakeReasoner(nn.Module):
    def forward(self, images):
        if self.training:
            self.distances = torch.full((len(images), 9), .3, requires_grad=True)
            return [self.distances, self.distances]
        return [(torch.arange(len(images))[:, None] * 10 + torch.arange(8)).float()]


def fake_model(training):
    model = SSPredRNet.__new__(SSPredRNet)
    nn.Module.__init__(model)
    model.margin = .7
    model.reasoner = FakeReasoner()
    return model.train(training)


class Contracts(unittest.TestCase):
    def test_reconstruction_and_original_positions(self):
        panels = np.full((16, 80, 80), 255, np.uint8)
        panels[:, 20:30, 10:20] = 80
        panels[:, 20:30, 60:70] = 100
        views, separated = component_views(panels, 'left_center_single_right_center_single')
        self.assertTrue(separated)
        self.assertTrue(np.array_equal(np.minimum(*views), panels))
        self.assertTrue(np.all(views[0, :, :, 40:] == 255))
        self.assertTrue(np.all(views[1, :, :, :40] == 255))
        full, separated = component_views(panels, 'center_single')
        self.assertFalse(separated)
        self.assertTrue(np.array_equal(full[0], panels))

    def test_uncertain_candidate_does_not_change_other_inputs(self):
        panel = np.full((80, 80), 255, np.uint8)
        cv2.rectangle(panel, (8, 8), (72, 72), 0, 1)
        cv2.circle(panel, (40, 40), 5, 90, -1)
        panels = np.repeat(panel[None], 16, axis=0)
        before, _ = component_views(panels, 'in_center_single_out_center_single')
        panels[9] = 255
        after, separated = component_views(panels, 'in_center_single_out_center_single')
        self.assertTrue(separated)
        self.assertTrue(np.array_equal(before[:, :9], after[:, :9]))
        self.assertTrue(np.array_equal(before[:, 10:], after[:, 10:]))
        panels[3] = 255
        full, separated = component_views(panels, 'in_center_single_out_center_single')
        self.assertFalse(separated)
        self.assertTrue(np.array_equal(full[0], panels))

    def test_training_needs_no_answer_key(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'center_single'
            folder.mkdir()
            panels = np.full((16, 160, 160), 255, np.uint8)
            np.savez(folder / 'RAVEN_0_train.npz', image=panels)
            images, target, config = Raven(directory, 'train')[0]
            self.assertEqual(tuple(images.shape), (2, 16, 80, 80))
            self.assertEqual(target, -1)
            np.savez(folder / 'RAVEN_0_val.npz', image=panels, target=3)
            self.assertEqual(Raven(directory, 'val')[0][1], 3)

    def test_identical_known_positive_has_no_negative_gradient(self):
        images = torch.arange(16.).reshape(1, 1, 16, 1, 1).repeat(1, 2, 1, 80, 80)
        images[:, 0, 8] = images[:, 0, 5]
        model = fake_model(True)
        loss = prediction_loss(model(images))
        loss.backward()
        self.assertEqual(float(model.reasoner.distances.grad[0, 1]), 0.)
        self.assertLess(float(model.reasoner.distances.grad[1, 1]), 0.)
        self.assertGreater(float(model.reasoner.distances.grad[0, 0]), 0.)

    def test_average_keeps_parent_puzzle(self):
        scores = fake_model(False)(torch.zeros(2, 2, 16, 80, 80))
        self.assertTrue(torch.equal(scores[0], torch.arange(8) + 5))
        self.assertTrue(torch.equal(scores[1], torch.arange(8) + 25))

    def test_checkpoint_rng_can_be_loaded_on_gpu(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA is required for the mapped RNG-state check')
        first, second = torch.Generator().manual_seed(10), torch.Generator().manual_seed(20)
        saved = rng_state(first, second)
        expected = torch.rand(3, generator=first)
        saved['torch'] = saved['torch'].cuda()
        saved['train_generator'] = saved['train_generator'].cuda()
        saved['val_generator'] = saved['val_generator'].cuda()
        saved['cuda'] = [s.cuda() for s in saved['cuda']]
        restore_rng(saved, first, second)
        self.assertTrue(torch.equal(torch.rand(3, generator=first), expected))


if __name__ == '__main__':
    unittest.main()
