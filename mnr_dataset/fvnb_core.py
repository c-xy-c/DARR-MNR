# -*- coding: utf-8 -*-
"""Core geometry and membership utilities for FVNB-MNR.

The functions in this module keep the fuzzy part deterministic: role
membership is computed from scene geometry rather than annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Tuple


ROLE_NAMES = ("outer", "inner", "boundary", "outside")


@dataclass(frozen=True)
class Rect:
    left: float
    top: float
    right: float
    bottom: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "left": float(self.left),
            "top": float(self.top),
            "right": float(self.right),
            "bottom": float(self.bottom),
        }

    def signed_distance(self, point: Tuple[float, float]) -> float:
        x, y = point
        inside = self.left <= x <= self.right and self.top <= y <= self.bottom
        if inside:
            return min(x - self.left, self.right - x, y - self.top, self.bottom - y)

        dx = max(self.left - x, 0.0, x - self.right)
        dy = max(self.top - y, 0.0, y - self.bottom)
        return -math.hypot(dx, dy)

    def boundary_distance(self, point: Tuple[float, float]) -> float:
        x, y = point
        if self.left <= x <= self.right and self.top <= y <= self.bottom:
            return min(abs(x - self.left), abs(x - self.right), abs(y - self.top), abs(y - self.bottom))

        def segment_distance(x1: float, y1: float, x2: float, y2: float) -> float:
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                return math.hypot(x - x1, y - y1)
            t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
            t = min(1.0, max(0.0, t))
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            return math.hypot(x - closest_x, y - closest_y)

        return min(
            segment_distance(self.left, self.top, self.right, self.top),
            segment_distance(self.right, self.top, self.right, self.bottom),
            segment_distance(self.right, self.bottom, self.left, self.bottom),
            segment_distance(self.left, self.bottom, self.left, self.top),
        )


@dataclass(frozen=True)
class Membership:
    raw: Dict[str, float]
    normalized: Dict[str, float]

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {"raw": dict(self.raw), "normalized": dict(self.normalized)}


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def normalize_scores(scores: Mapping[str, float], roles: Iterable[str] = ROLE_NAMES) -> Dict[str, float]:
    ordered = {role: max(float(scores.get(role, 0.0)), 0.0) for role in roles}
    total = sum(ordered.values())
    if total <= 0:
        return {role: 1.0 / len(ordered) for role in ordered}
    return {role: value / total for role, value in ordered.items()}


def containment_membership(
    point: Tuple[float, float],
    outer: Rect,
    inner: Rect,
    tau: float = 2.0,
    sigma: float = 1.8,
    margin: float = 0.0,
    boundary_weight: float = 5.0,
) -> Membership:
    """Compute deterministic fuzzy role membership for containment scenes."""
    outer_inside = sigmoid((outer.signed_distance(point) - margin) / tau)
    inner_inside = sigmoid((inner.signed_distance(point) - margin) / tau)
    boundary_distance = inner.boundary_distance(point)
    boundary = boundary_weight * math.exp(-(boundary_distance * boundary_distance) / (2.0 * sigma * sigma))

    raw = {
        "inner": inner_inside,
        "boundary": boundary,
        "outer": outer_inside * (1.0 - inner_inside),
        "outside": 1.0 - outer_inside,
    }
    return Membership(raw=raw, normalized=normalize_scores(raw))


def hard_role_for_membership(membership: Membership | Mapping[str, float]) -> str:
    values = membership.normalized if isinstance(membership, Membership) else membership
    return max(values.items(), key=lambda item: (item[1], item[0]))[0]
