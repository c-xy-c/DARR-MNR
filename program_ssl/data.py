"""Known-row adaptation without candidate-dependent preprocessing."""
import random
import cv2
import numpy as np
import torch
from sspredrnet.data import Raven, CONFIGS
from sspredrnet.views import component_views


class KnownRowTraining(Raven):
    def __init__(self, root, *, include_candidates=False):
        super().__init__(root, 'train')
        self.include_candidates = include_candidates

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            raw = data['image'].reshape(16, 160, 160)
            indices = [0, 1, 2, 3, 4, 5, 0, 1]
            indices += list(range(8, 16)) if self.include_candidates else [0] * 8
            raw = raw[indices]
        panels = np.stack([cv2.resize(p, (80, 80), interpolation=cv2.INTER_NEAREST)
                           for p in raw]).astype(np.uint8)
        if random.random() < .5:
            panels = panels[:, :, ::-1].copy()
        # Existing splitter checks eight context slots. Duplicating panels
        # zero/one makes its confidence depend on exactly the first six.
        views, _ = component_views(panels, path.parent.name)
        if not self.include_candidates:
            views = views[:, :6]
        return (torch.from_numpy(views.astype(np.float32)), -1,
                CONFIGS.index(path.parent.name))
