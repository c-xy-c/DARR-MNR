"""Panel representations, immutable object targets and masked-object head."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .blocks import SelfBlock, CrossBlock
from .constants import WIDTH, STAGES, OBJECTS

class ObjectPerception(nn.Module):
    """Raw 8px patches -> three levels -> masked mean/cross-attention pooling."""
    def __init__(self):
        super().__init__()
        self.patches = nn.Conv2d(1, WIDTH, 8, stride=8)
        axis = torch.linspace(-1, 1, 10)
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('coordinates', torch.stack((x, y, x.square(), y.square()), -1).reshape(100, 4))
        self.position = nn.Linear(4, WIDTH)
        self.geometry = nn.Linear(6, WIDTH)
        self.layers = nn.ModuleList([SelfBlock() for _ in range(STAGES)])
        self.poolers = nn.ModuleList([CrossBlock() for _ in range(STAGES)])
        self.activation_checkpointing = True

    def forward(self, images, masks, geometry, valid):
        tokens = self.patches(images[:, None]).flatten(2).transpose(1, 2)
        tokens = tokens + self.position(self.coordinates.to(tokens))
        weights = masks.to(tokens) / masks.to(tokens).sum(-1, keepdim=True).clamp_min(1e-6)
        # Inactive objects use one harmless key, then their outputs are erased.
        first_key = torch.arange(100, device=masks.device).eq(0)
        outside = (masks <= 0) & (valid[..., None] | ~first_key)
        outside = outside.repeat_interleave(6, dim=0)
        levels = []
        for layer, pooler in zip(self.layers, self.poolers):
            recompute = self.training and torch.is_grad_enabled() and self.activation_checkpointing
            tokens = checkpoint(layer, tokens, use_reentrant=False) if recompute else layer(tokens)
            mean = weights @ tokens
            query = mean + self.geometry(geometry.to(tokens))
            objects = (checkpoint(pooler, query, tokens, mask=outside, use_reentrant=False)
                       if recompute else pooler(query, tokens, mask=outside))
            objects = objects * valid[..., None]
            scene = tokens.mean(1, keepdim=True)
            levels.append(torch.cat((objects, scene), 1))
        return torch.stack(levels, 1)


def target_objects(teacher, images, masks, boxes, geometry, valid):
    """Encode each bbox on a white panel before pooling its region features.

    Pixels, coordinates and scale inside the box stay unchanged. Context outside
    it cannot affect the appearance target. Overlapping boxes can still include
    another object's pixels; exterior contours are not semantic segmentation.
    Chunking bounds teacher activations without changing targets or gradients.
    """
    with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
        n, panels = images.shape[:2]
        pooled_masks = F.avg_pool2d(masks.float().reshape(n * panels * OBJECTS, 1, 10, 10), 2)
        weights = pooled_masks.reshape(-1, 25)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        appearance = torch.zeros(n * panels * OBJECTS, 32, device=images.device)
        active = valid.flatten().nonzero().flatten()
        flat_images, flat_boxes = images.float().flatten(0, 1), boxes.reshape(-1, 4)
        grid = torch.arange(80, device=images.device)
        for start in range(0, len(active), 128):
            ids = active[start:start + 128]
            x0, y0, x1, y1 = flat_boxes[ids].unbind(-1)
            inside = ((grid[None, None, :] >= x0[:, None, None]) &
                      (grid[None, None, :] < x1[:, None, None]) &
                      (grid[None, :, None] >= y0[:, None, None]) &
                      (grid[None, :, None] < y1[:, None, None]))
            isolated = flat_images[ids // OBJECTS].masked_fill(~inside, 1.)
            features = teacher.features(isolated[:, None, None])[:, 0]
            pooled = (weights[ids, None] @ F.relu(features).transpose(1, 2)).squeeze(1)
            appearance.index_copy_(0, ids, pooled)
        appearance = appearance.reshape(n, panels, OBJECTS, 32)
        return torch.cat((appearance, geometry.float()), -1) * valid[..., None]


class MaskedObjectPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, WIDTH) * .02)
        self.panel = nn.Parameter(torch.randn(5, WIDTH) * .02)
        self.center = nn.Linear(2, WIDTH)
        self.layers = nn.ModuleList([CrossBlock(), CrossBlock()])
        self.output = nn.Linear(WIDTH, 32)

    def forward(self, levels, valid, centers, panel_indices):
        n = len(levels)
        queries = self.query + self.panel[panel_indices] + self.center(centers)
        memory = (levels[:, :, -1] + self.panel[None, :, None]).flatten(1, 2)
        padding = torch.cat((~valid, torch.zeros(n, 5, 1, device=valid.device, dtype=torch.bool)), -1).flatten(1)
        for layer in self.layers:
            queries = layer(queries, memory, padding)
        return self.output(queries)
