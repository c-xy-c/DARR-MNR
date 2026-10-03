"""Images only during training; canonical file order for evaluation."""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .views import component_views, split_layers

CONFIGS = (
    'center_single', 'distribute_four', 'distribute_nine',
    'in_center_single_out_center_single',
    'in_distribute_four_out_center_single',
    'left_center_single_right_center_single',
    'up_center_single_down_center_single',
)


class Raven(Dataset):
    def __init__(self, root, split):
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val, or test')
        self.split = split
        self.paths = [p for c in CONFIGS
                      for p in sorted((Path(root) / c).glob(f'*_{split}.npz'))]
        if not self.paths:
            raise FileNotFoundError(f'no {split} puzzles in {root}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            raw = data['image'].reshape(16, 160, 160)
            answer = int(data['target']) if self.split != 'train' else -1
        panels = np.stack([cv2.resize(p, (80, 80), interpolation=cv2.INTER_NEAREST)
                           for p in raw]).astype(np.uint8)
        if self.split == 'train' and random.random() < .5:
            panels = panels[:, :, ::-1].copy()
        views, _ = component_views(panels, path.parent.name)
        return (torch.from_numpy(views.astype(np.float32)), answer,
                CONFIGS.index(path.parent.name))


def normalize(images):
    return (images / 255. - .5) * 2


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def loader(dataset, batch_size, workers, generator, *, training=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=training,
                      num_workers=workers, pin_memory=True,
                      worker_init_fn=seed_worker, generator=generator)


class KnownRowTraining(Raven):
    def __init__(self, root):
        super().__init__(root, 'train')

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            raw = data['image'].reshape(16, 160, 160)[:6]
        panels = np.stack([cv2.resize(p, (80, 80), interpolation=cv2.INTER_NEAREST)
                           for p in raw]).astype(np.uint8)
        if random.random() < .5:
            panels = panels[:, :, ::-1].copy()
        # Only the five visible predictor inputs can decide a global fallback.
        # The known target at index five is not an observed predictor input.
        layers = [split_layers(p, path.parent.name) for p in panels]
        if any(layer is None for layer in layers[:5]):
            views = np.stack((panels, panels))
        else:
            # An ambiguous target falls back independently. It
            # cannot change the views of support or query-prefix panels.
            layers = [np.stack((p, p)) if layer is None else layer
                      for p, layer in zip(panels, layers)]
            views = np.stack(layers, axis=1)
        return (torch.from_numpy(views.astype(np.float32)), -1,
                CONFIGS.index(path.parent.name))
