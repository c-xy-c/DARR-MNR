# -*- coding: utf-8 -*-
"""1-5 continuous fuzzy quantity attributes for VQ-Expr.

The visual channel never uses Arabic digits, operators, object counts, or
position codes as quantity attributes. Each value is carried by one continuous
object attribute and decoded through a sample-local axis.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


QUANTITY_FAMILIES = (
    "size_level",
    "color_lightness",
    "stroke_width",
    "aspect_ratio",
)

VALUE_MIN = 1
VALUE_MAX = 5

ATTRIBUTE_AXIS_SPECS = {
    "size_level": {
        "axis_name": "radius",
        "visual_min": 0.17,
        "visual_max": 0.33,
        "sigma": 0.14,
    },
    "color_lightness": {
        "axis_name": "lightness",
        "visual_min": 0.88,
        "visual_max": 0.20,
        "sigma": 0.12,
    },
    "stroke_width": {
        "axis_name": "outline_width",
        "visual_min": 1.0,
        "visual_max": 6.0,
        "sigma": 0.13,
    },
    "aspect_ratio": {
        "axis_name": "width_to_height",
        "visual_min": 0.68,
        "visual_max": 1.36,
        "sigma": 0.16,
    },
}


@dataclass(frozen=True)
class RenderedQuantity:
    object_id: str
    family: str
    numeric_value: int
    observation: Mapping[str, object]
    primitives: List[Dict[str, object]]
    symbolic_parse: Mapping[str, object]
    fuzzy_value: Mapping[str, object]
    visual_confounds: Mapping[str, float]
    leakage_flags: Mapping[str, object]

    @property
    def component_parse(self) -> Mapping[str, object]:
        return self.symbolic_parse

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.object_id,
            "family": self.family,
            "numeric_value": self.numeric_value,
            "observation": dict(self.observation),
            "primitives": list(self.primitives),
            "symbolic_parse": dict(self.symbolic_parse),
            "component_parse": dict(self.symbolic_parse),
            "fuzzy_value": dict(self.fuzzy_value),
            "visual_confounds": dict(self.visual_confounds),
            "leakage_flags": dict(self.leakage_flags),
        }


def make_calibration_context(
    sample_id: str = "vqexpr_sample",
    seed: int = 0,
    black_digit: Optional[int] = None,
) -> Dict[str, object]:
    """Create a sample-local continuous-attribute decoder context."""

    _ = black_digit
    rng = random.Random(seed)
    axis_context = {}
    for family in QUANTITY_FAMILIES:
        spec = ATTRIBUTE_AXIS_SPECS[family]
        axis_context[family] = {
            "low_value": VALUE_MIN,
            "high_value": VALUE_MAX,
            "axis_name": spec["axis_name"],
            "visual_min": spec["visual_min"],
            "visual_max": spec["visual_max"],
            "sigma": spec["sigma"],
            "calibration_jitter": round(rng.uniform(-0.015, 0.015), 4),
            "monotonic_direction": "decreasing" if spec["visual_max"] < spec["visual_min"] else "increasing",
        }
    return {
        "sample_id": sample_id,
        "value_range": [VALUE_MIN, VALUE_MAX],
        "continuous_attributes": axis_context,
    }


def render_quantity(
    family: str,
    value: int,
    object_id: str,
    calibration_context: Optional[Mapping[str, object]] = None,
    style_seed: int = 0,
    fuzzy_severity: str = "medium",
) -> RenderedQuantity:
    if family not in QUANTITY_FAMILIES:
        raise ValueError("Unknown VQ-Expr quantity family: {0}".format(family))
    if not VALUE_MIN <= int(value) <= VALUE_MAX:
        raise ValueError("VQ-Expr quantity value must be in 1..5")

    context = calibration_context or make_calibration_context(seed=style_seed)
    value = int(value)
    observation, primitives, symbolic = _continuous_attribute(family, value, context)
    fuzzy_value = _fuzzy_value(value, fuzzy_severity)
    confounds = _visual_confounds(primitives, value, style_seed)
    leakage = {
        "contains_arabic_digit": False,
        "contains_operator_symbol": False,
        "text_ocr_risk": "none",
    }
    return RenderedQuantity(
        object_id=object_id,
        family=family,
        numeric_value=value,
        observation=observation,
        primitives=primitives,
        symbolic_parse=symbolic,
        fuzzy_value=fuzzy_value,
        visual_confounds=confounds,
        leakage_flags=leakage,
    )


def decode_observation(
    family: str,
    observation: Mapping[str, object],
    calibration_context: Mapping[str, object],
    fuzzy_severity: str = "medium",
) -> Dict[str, object]:
    """Decode a continuous visual attribute through the sample-local axis."""

    if family not in QUANTITY_FAMILIES:
        raise ValueError("Unknown family: {0}".format(family))
    low = int(calibration_context["continuous_attributes"][family]["low_value"])
    high = int(calibration_context["continuous_attributes"][family]["high_value"])
    axis_value = float(observation["attribute_axis_value"])
    value = int(round(low + (high - low) * axis_value))
    value = min(VALUE_MAX, max(VALUE_MIN, value))
    return _fuzzy_value(value, fuzzy_severity)


def _continuous_attribute(
    family: str,
    value: int,
    context: Mapping[str, object],
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, object]]:
    axis = context["continuous_attributes"][family]
    raw_axis = _axis_value(value)
    observed_axis = min(1.0, max(0.0, raw_axis + float(axis.get("calibration_jitter", 0.0))))
    visual_min = float(axis["visual_min"])
    visual_max = float(axis["visual_max"])
    visual_value = visual_min + (visual_max - visual_min) * observed_axis
    observation = {
        "attribute_family": family,
        "axis_name": axis["axis_name"],
        "attribute_axis_value": round(observed_axis, 4),
        "visual_value": round(visual_value, 4),
    }
    if family == "size_level":
        observation["radius_ratio"] = round(visual_value, 4)
    elif family == "color_lightness":
        observation["lightness"] = round(visual_value, 4)
    elif family == "stroke_width":
        observation["stroke_width"] = round(visual_value, 4)
    elif family == "aspect_ratio":
        observation["aspect_ratio"] = round(visual_value, 4)

    primitives = [
        {
            "kind": "continuous_attribute",
            "attribute_family": family,
            "axis_name": axis["axis_name"],
            "axis_value": round(observed_axis, 4),
            "visual_value": round(visual_value, 4),
            "area": round(12.0 + 8.0 * observed_axis, 4),
        }
    ]
    symbolic = {
        "attribute_family": family,
        "axis_name": axis["axis_name"],
        "axis_value": round(observed_axis, 4),
        "low_value": axis["low_value"],
        "high_value": axis["high_value"],
        "estimated_value": int(value),
    }
    return observation, primitives, symbolic


def _axis_value(value: int) -> float:
    if VALUE_MAX == VALUE_MIN:
        return 0.0
    return (float(value) - VALUE_MIN) / float(VALUE_MAX - VALUE_MIN)


def _fuzzy_value(value: int, severity: str = "medium") -> Dict[str, object]:
    radius = {"low": 1, "medium": 2, "high": 4}.get(severity, 2)
    support = {int(value)}
    for delta in range(1, radius + 1):
        if value - delta >= VALUE_MIN:
            support.add(value - delta)
        if value + delta <= VALUE_MAX:
            support.add(value + delta)
    membership = {str(candidate): _triangular_membership(value, candidate) for candidate in sorted(support)}
    membership[str(value)] = 1.0
    continuous = _axis_value(value)
    sigma = {"low": 0.10, "medium": 0.16, "high": 0.24}.get(severity, 0.16)
    return {
        "support": sorted(support),
        "membership": membership,
        "discrete_levels": list(range(VALUE_MIN, VALUE_MAX + 1)),
        "continuous_value": round(continuous, 4),
        "interpolation_sigma": sigma,
        "interpolation_kernel": "triangular_over_continuous_attribute_axis",
    }


def _triangular_membership(center: int, candidate: int) -> float:
    distance = abs(int(center) - int(candidate))
    if distance == 0:
        return 1.0
    return round(max(0.03, 1.0 / (distance + 1.8)), 4)


def _visual_confounds(primitives: Iterable[Mapping[str, object]], value: int, style_seed: int) -> Dict[str, float]:
    primitive_list = list(primitives)
    ink_area = sum(float(p.get("area", 1.0)) for p in primitive_list)
    hull = max(16.0, len(primitive_list) * 7.0 + (style_seed % 11))
    density = min(0.95, max(0.02, ink_area / (hull + 1e-6)))
    return {
        "primitive_count": float(len(primitive_list)),
        "ink_area": round(ink_area, 4),
        "convex_hull_area": round(hull, 4),
        "density": round(density, 4),
        "stroke_length": round(ink_area * 1.7 + value * 0.03, 4),
    }


def family_for_index(index: int) -> str:
    return QUANTITY_FAMILIES[index % len(QUANTITY_FAMILIES)]
