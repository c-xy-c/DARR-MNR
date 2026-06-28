# -*- coding: utf-8 -*-
"""1-9 calibrated fuzzy quantity attributes for VQ-Expr.

Presentation primitives intentionally avoid Arabic digits and arithmetic
symbols. Numeric values live in metadata for the oracle/debug view only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


QUANTITY_FAMILIES = (
    "local_place_value_codebook",
    "calibrated_metric_scale",
    "calibrated_area_grid",
    "chunked_path_topology",
    "part_ratio_dial",
)

VALUE_MIN = 1
VALUE_MAX = 9

TOKEN_ORDER = (
    "token_black",
    "token_red",
    "token_blue",
    "token_green",
    "token_yellow",
    "token_purple",
    "token_cyan",
    "token_orange",
    "token_gray",
    "token_white",
)

TOKEN_COLORS = {
    "token_black": (34, 36, 40),
    "token_red": (205, 76, 82),
    "token_blue": (68, 126, 191),
    "token_green": (82, 154, 103),
    "token_yellow": (218, 185, 73),
    "token_purple": (137, 92, 184),
    "token_cyan": (69, 169, 176),
    "token_orange": (218, 132, 65),
    "token_gray": (137, 146, 157),
    "token_white": (244, 244, 238),
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
            "fuzzy_value": {
                "support": list(self.fuzzy_value["support"]),
                "membership": dict(self.fuzzy_value["membership"]),
            },
            "visual_confounds": dict(self.visual_confounds),
            "leakage_flags": dict(self.leakage_flags),
        }


def make_calibration_context(
    sample_id: str = "vqexpr_sample",
    seed: int = 0,
    black_digit: Optional[int] = None,
) -> Dict[str, object]:
    """Create a sample-local decoder context.

    `black_digit` is exposed for tests and for the core calibration-OOD
    diagnostic: the same token can map to digit 9 in one sample and digit 0 in
    another sample.
    """

    rng = random.Random(seed)
    digits = list(range(10))
    tokens = list(TOKEN_ORDER)
    digit_to_token: Dict[int, str] = {}

    if black_digit is not None:
        if not 0 <= int(black_digit) <= 9:
            raise ValueError("black_digit must be in 0..9")
        digit_to_token[int(black_digit)] = "token_black"
        digits.remove(int(black_digit))
        tokens.remove("token_black")

    rng.shuffle(tokens)
    for digit, token in zip(digits, tokens):
        digit_to_token[digit] = token

    token_to_digit = {token: digit for digit, token in digit_to_token.items()}
    digit_codebook = {
        token: {
            "digit": digit,
            "mu_digit": _digit_membership(digit),
            "rgb": TOKEN_COLORS[token],
        }
        for token, digit in sorted(token_to_digit.items(), key=lambda item: item[1])
    }

    slot_mode = rng.choice(["left_right", "inner_outer", "upper_lower"])
    if slot_mode == "left_right":
        slot_decoder = {
            "slot_a": {"role": "tens", "mu_role": {"tens": 0.94, "ones": 0.06}},
            "slot_b": {"role": "ones", "mu_role": {"ones": 0.92, "tens": 0.08}},
            "layout": "left_right",
        }
    elif slot_mode == "inner_outer":
        slot_decoder = {
            "slot_a": {"role": "tens", "mu_role": {"tens": 0.91, "ones": 0.09}},
            "slot_b": {"role": "ones", "mu_role": {"ones": 0.93, "tens": 0.07}},
            "layout": "inner_outer",
        }
    else:
        slot_decoder = {
            "slot_a": {"role": "tens", "mu_role": {"tens": 0.93, "ones": 0.07}},
            "slot_b": {"role": "ones", "mu_role": {"ones": 0.91, "tens": 0.09}},
            "layout": "upper_lower",
        }

    orientation = rng.choice(["left_to_right", "right_to_left", "bottom_to_top", "clockwise_arc"])
    unit_cell = rng.choice(["square", "skewed_quad", "triangle"])
    chunk_proto = rng.choice(["capsule", "dotted_arc", "bead_chain"])
    active_region = rng.choice(["inner_ring", "outer_ring", "central_bar"])
    return {
        "sample_id": sample_id,
        "value_range": [VALUE_MIN, VALUE_MAX],
        "digit_codebook": digit_codebook,
        "digit_to_token": {str(digit): token for digit, token in digit_to_token.items()},
        "token_to_digit": {token: digit for token, digit in token_to_digit.items()},
        "slot_decoder": slot_decoder,
        "metric_scale": {
            "low_value": VALUE_MIN,
            "high_value": VALUE_MAX,
            "orientation": orientation,
            "sigma": 1.35,
        },
        "area_grid": {
            "unit_cell": unit_cell,
            "orientation": rng.choice(["axis_aligned", "skewed", "radial"]),
            "sigma": 1.55,
        },
        "chunked_path": {
            "chunk_proto": chunk_proto,
            "residual_proto": rng.choice(["thin_edge", "small_bead", "short_dash"]),
            "sigma": 1.15,
        },
        "part_ratio": {
            "low_value": VALUE_MIN,
            "high_value": VALUE_MAX,
            "active_region": active_region,
            "sigma": 1.8,
        },
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
        raise ValueError("VQ-Expr quantity value must be in 1..9")

    context = calibration_context or make_calibration_context(seed=style_seed)
    value = int(value)
    if family == "local_place_value_codebook":
        observation, primitives, symbolic = _place_value(value, context)
    elif family == "calibrated_metric_scale":
        observation, primitives, symbolic = _metric_scale(value, context)
    elif family == "calibrated_area_grid":
        observation, primitives, symbolic = _area_grid(value, context)
    elif family == "chunked_path_topology":
        observation, primitives, symbolic = _chunked_path(value, context)
    else:
        observation, primitives, symbolic = _part_ratio(value, context)

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
    """Decode an observation through a sample-local context."""

    if family == "local_place_value_codebook":
        token_to_digit = calibration_context["token_to_digit"]
        slot_a = str(observation["slot_tokens"]["slot_a"])
        slot_b = str(observation["slot_tokens"]["slot_b"])
        tens = int(token_to_digit[slot_a])
        ones = int(token_to_digit[slot_b])
        value = 10 * tens + ones
        if value == 0:
            value = 1
    elif family == "calibrated_metric_scale":
        t = float(observation["projected_coordinate"])
        low = int(calibration_context["metric_scale"]["low_value"])
        high = int(calibration_context["metric_scale"]["high_value"])
        value = int(round(low + (high - low) * t))
    elif family == "calibrated_area_grid":
        value = int(round(float(observation["mean_count"])))
    elif family == "chunked_path_topology":
        value = 10 * int(observation["chunk_count"]) + int(observation["residual_count"])
        value = max(1, value)
    elif family == "part_ratio_dial":
        low = int(calibration_context["part_ratio"]["low_value"])
        high = int(calibration_context["part_ratio"]["high_value"])
        rho = float(observation["filled_ratio"])
        value = int(round(low + rho * (high - low)))
    else:
        raise ValueError("Unknown family: {0}".format(family))
    value = min(VALUE_MAX, max(VALUE_MIN, value))
    return _fuzzy_value(value, fuzzy_severity)


def token_for_digit(calibration_context: Mapping[str, object], digit: int) -> str:
    return str(calibration_context["digit_to_token"][str(int(digit))])


def _place_value(value: int, context: Mapping[str, object]):
    tens = value // 10
    ones = value % 10
    token_a = token_for_digit(context, tens)
    token_b = token_for_digit(context, ones)
    slot_layout = str(context["slot_decoder"]["layout"])
    primitives = [
        {
            "kind": "place_slot",
            "slot_id": "slot_a",
            "token": token_a,
            "rgb": TOKEN_COLORS[token_a],
            "layout": slot_layout,
            "area": 8.0,
        },
        {
            "kind": "place_slot",
            "slot_id": "slot_b",
            "token": token_b,
            "rgb": TOKEN_COLORS[token_b],
            "layout": slot_layout,
            "area": 8.0,
        },
        {
            "kind": "slot_connector",
            "layout": slot_layout,
            "area": 2.0,
        },
    ]
    observation = {
        "slot_tokens": {"slot_a": token_a, "slot_b": token_b},
        "slot_layout": slot_layout,
    }
    symbolic = {
        "tens_digit": tens,
        "ones_digit": ones,
        "slot_roles": {
            "slot_a": context["slot_decoder"]["slot_a"]["role"],
            "slot_b": context["slot_decoder"]["slot_b"]["role"],
        },
        "token_ids": {"slot_a": token_a, "slot_b": token_b},
    }
    return observation, primitives, symbolic


def _metric_scale(value: int, context: Mapping[str, object]):
    scale = context["metric_scale"]
    low = int(scale["low_value"])
    high = int(scale["high_value"])
    raw = (value - low) / float(high - low)
    orientation = str(scale["orientation"])
    projected = 1.0 - raw if orientation in {"right_to_left", "bottom_to_top"} else raw
    primitives = [
        {"kind": "scale_axis", "orientation": orientation, "area": 18.0},
        {"kind": "scale_pointer", "projected_coordinate": projected, "area": 4.0},
        {"kind": "scale_band", "sigma": scale["sigma"], "area": 5.0},
    ]
    observation = {
        "scale_id": "metric_scale",
        "orientation": orientation,
        "projected_coordinate": raw,
        "visible_coordinate": projected,
    }
    symbolic = {
        "low_anchor": "scale_low",
        "high_anchor": "scale_high",
        "projected_coordinate": round(raw, 4),
        "estimated_value": value,
    }
    return observation, primitives, symbolic


def _area_grid(value: int, context: Mapping[str, object]):
    grid = context["area_grid"]
    width = 10
    filled = []
    for idx in range(value):
        filled.append({"cell_id": "c{0}".format(idx), "membership": 0.96 if idx < value - 1 else 0.88})
    primitives = [
        {
            "kind": "grid_lattice",
            "unit_cell": grid["unit_cell"],
            "orientation": grid["orientation"],
            "area": 24.0,
        }
    ]
    for idx in range(value):
        primitives.append(
            {
                "kind": "filled_cell",
                "cell_index": idx,
                "x": idx % width,
                "y": idx // width,
                "unit_cell": grid["unit_cell"],
                "area": 1.0,
            }
        )
    observation = {
        "unit_cell": grid["unit_cell"],
        "orientation": grid["orientation"],
        "mean_count": float(value),
    }
    symbolic = {
        "lattice_id": "grid_A",
        "unit_cell": grid["unit_cell"],
        "filled_cell_count": value,
        "filled_cell_memberships": {row["cell_id"]: row["membership"] for row in filled[:8]},
        "mean_count": float(value),
    }
    return observation, primitives, symbolic


def _chunked_path(value: int, context: Mapping[str, object]):
    path = context["chunked_path"]
    chunks = value // 10
    residual = value % 10
    primitives: List[Dict[str, object]] = [
        {
            "kind": "path_spine",
            "chunk_proto": path["chunk_proto"],
            "residual_proto": path["residual_proto"],
            "area": 8.0,
        }
    ]
    for idx in range(chunks):
        primitives.append({"kind": "path_chunk", "index": idx, "proto": path["chunk_proto"], "area": 5.0})
    for idx in range(residual):
        primitives.append({"kind": "residual_edge", "index": idx, "proto": path["residual_proto"], "area": 1.2})
    observation = {
        "chunk_proto": path["chunk_proto"],
        "residual_proto": path["residual_proto"],
        "chunk_count": chunks,
        "residual_count": residual,
    }
    symbolic = {
        "path_id": "path_A",
        "chunk_count": chunks,
        "residual_count": residual,
        "value_formula": "10 * chunk_count + residual_count",
        "topology": {"branches": 1 if value % 3 == 0 else 0, "crossings": 0},
    }
    return observation, primitives, symbolic


def _part_ratio(value: int, context: Mapping[str, object]):
    ratio = context["part_ratio"]
    low = int(ratio["low_value"])
    high = int(ratio["high_value"])
    rho = (value - low) / float(high - low)
    rho = min(1.0, max(0.0, rho))
    primitives = [
        {"kind": "dial_shell", "active_region": ratio["active_region"], "area": 20.0},
        {"kind": "dial_fill", "filled_ratio": rho, "area": 20.0 * rho},
        {"kind": "dial_uncertainty_band", "sigma": ratio["sigma"], "area": 3.0},
    ]
    observation = {
        "active_region": ratio["active_region"],
        "filled_ratio": round(rho, 4),
    }
    symbolic = {
        "dial_id": "dial_A",
        "active_region": ratio["active_region"],
        "low_value": low,
        "high_value": high,
        "filled_ratio": round(rho, 4),
        "estimated_value": value,
    }
    return observation, primitives, symbolic


def _digit_membership(digit: int) -> Dict[str, float]:
    membership = {str(digit): 1.0}
    if digit > 0:
        membership[str(digit - 1)] = 0.18
    if digit < 9:
        membership[str(digit + 1)] = 0.18
    return membership


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
    return {"support": sorted(support), "membership": membership}


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
