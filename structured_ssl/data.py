"""Independent pixel proposals; six known panels are the complete SSL boundary."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from program_ssl.data import KnownRowTraining
from sspredrnet.data import Raven


OBJECTS = 10


def proposals(panel):
    """Contour regions, not semantic labels or learned object discovery.

    Filled exterior contours include hollow interiors. Excess components merge
    into the last region without losing foreground; blank panels have no objects.
    Each panel is processed independently, including any completion target.
    """
    panel = np.asarray(panel)
    if panel.shape != (80, 80):
        raise ValueError('expected an 80px panel')
    ink = (panel < 255).astype(np.uint8)
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        region = np.zeros_like(ink)
        cv2.drawContours(region, [contour], -1, 1, thickness=cv2.FILLED)
        if np.any(region):
            regions.append(region)
    regions.sort(key=lambda r: tuple(np.argwhere(r).mean(0)))
    overflow = len(regions) > OBJECTS
    if overflow:
        regions = regions[:OBJECTS - 1] + [np.maximum.reduce(regions[OBJECTS - 1:])]
    masks = np.zeros((OBJECTS, 100), np.float32)
    geometry = np.zeros((OBJECTS, 6), np.float32)
    boxes = np.zeros((OBJECTS, 4), np.int64)
    valid = np.zeros(OBJECTS, bool)
    for i, region in enumerate(regions):
        ys, xs = np.nonzero(region)
        boxes[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        geometry[i] = (xs.mean() / 80, ys.mean() / 80,
                       (xs.max() - xs.min() + 1) / 80, (ys.max() - ys.min() + 1) / 80,
                       region.mean(), (255 - panel[region.astype(bool)]).mean() / 255)
        masks[i] = cv2.resize(region.astype(np.float32), (10, 10), interpolation=cv2.INTER_AREA).flatten()
        valid[i] = True
    return masks, geometry, boxes, valid, overflow


def proposal_pack(views):
    array = views.numpy() if isinstance(views, torch.Tensor) else np.asarray(views)
    shape = array.shape[:2]
    parsed = [proposals(panel.astype(np.uint8)) for panel in array.reshape(-1, 80, 80)]
    names = ('masks', 'geometry', 'boxes', 'valid', 'overflow')
    pack = {name: torch.from_numpy(np.stack([p[i] for p in parsed]).reshape(*shape, *np.shape(parsed[0][i])))
            for i, name in enumerate(names)}
    return {'views': torch.as_tensor(views), **pack}


class ObjectRaven(Dataset):
    def __init__(self, root, split):
        self.base = KnownRowTraining(root) if split == 'train' else Raven(root, split)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        views, label, configuration = self.base[index]
        return proposal_pack(views), label, configuration


def to_device(pack, device):
    # CUDA loader batches are pinned. Metal inputs are ordinary CPU allocations;
    # complete those copies before the caller can release the CPU batch.
    non_blocking = torch.device(device).type == 'cuda'
    return {k: ((v.to(device, non_blocking=non_blocking) / 255 - .5) * 2 if k == 'views'
                else v.to(device, non_blocking=non_blocking)) for k, v in pack.items()}
