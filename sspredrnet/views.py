"""Lossless component views using the public RAVEN layout geometry."""
import cv2
import numpy as np


def split_layers(panel, config):
    """Return two losslessly recomposable ink layers, or None if uncertain.

    Nested outer ink is one whole connected component, selected by its
    enclosing bounding box. Connected inner/outer ink is not disentangled.
    This geometric heuristic does not guarantee semantic object assignment.
    """
    panel = np.asarray(panel)
    if panel.shape != (80, 80) or panel.dtype != np.uint8:
        raise ValueError('expected one uint8 80px panel')
    if config == 'left_center_single_right_center_single':
        first = np.zeros_like(panel, bool)
        first[:, :40] = True
    elif config == 'up_center_single_down_center_single':
        first = np.zeros_like(panel, bool)
        first[:40] = True
    elif config.startswith('in_'):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (panel < 255).astype(np.uint8), connectivity=8)
        if count < 3:  # Cannot separate an outer component and inner ink.
            return None
        outer = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_WIDTH] *
                                  stats[1:, cv2.CC_STAT_HEIGHT]))
        x, y, width, height, _ = stats[outer]
        if width < 30 or height < 30:
            return None
        for index in range(1, count):
            if index == outer:
                continue
            ix, iy, iw, ih, _ = stats[index]
            if ix < x or iy < y or ix + iw > x + width or iy + ih > y + height:
                return None
        first = labels == outer
    else:
        return None
    layers = np.stack((np.where(first, panel, 255),
                       np.where(~first, panel, 255))).astype(np.uint8)
    if not np.array_equal(np.minimum(layers[0], layers[1]), panel):
        raise RuntimeError('layer decomposition lost original pixels')
    return layers


def component_views(panels, config):
    """Keep all positions/scales; split complete ink layers for every panel.

    If a context panel cannot be split, both views contain the full puzzle.
    Uncertain candidates retain their full image in both views, without
    changing other candidates or contexts. Single layouts have two identical
    full views. Two views let the original
    trainer retain a fixed tensor shape and average each raw puzzle's losses.
    Public layout geometry is an explicit prior; no rule or answer is read.
    """
    panels = np.asarray(panels)
    if panels.shape != (16, 80, 80) or panels.dtype != np.uint8:
        raise ValueError('sixteen uint8 80px panels required')
    layers = [split_layers(panel, config) for panel in panels]
    if any(layer is None for layer in layers[:8]):
        return np.stack((panels, panels)), False
    layers = [np.stack((panel, panel)) if layer is None else layer
              for panel, layer in zip(panels, layers)]
    views = np.stack(layers, axis=1)
    if not np.array_equal(np.minimum(views[0], views[1]), panels):
        raise RuntimeError('component views do not reconstruct original pixels')
    return views, True
