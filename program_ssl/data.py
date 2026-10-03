"""Known-row adaptation without candidate-dependent preprocessing."""
import random
import cv2
import numpy as np
import torch
from sspredrnet.data import Raven, CONFIGS
from sspredrnet.views import split_layers


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
