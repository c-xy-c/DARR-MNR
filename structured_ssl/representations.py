"""Separate trainable visible context from fixed scoring targets."""
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class VisibleContext:
    levels: Tensor
    valid: Tensor

    def select(self, panels, order=None):
        if order is None:
            return VisibleContext(self.levels[:, panels], self.valid[:, panels])
        return VisibleContext(self.levels[order, panels], self.valid[order, panels])


@dataclass(frozen=True)
class PanelTargets:
    dense: Tensor
    objects: Tensor
    valid: Tensor

    def select(self, panels, order=None):
        if order is None:
            return PanelTargets(self.dense[:, panels], self.objects[:, panels], self.valid[:, panels])
        return PanelTargets(self.dense[order, panels], self.objects[order, panels], self.valid[order, panels])
