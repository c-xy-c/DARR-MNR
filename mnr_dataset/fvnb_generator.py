# -*- coding: utf-8 -*-
"""Programmatic FVNB-MNR v0 generator."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mnr_dataset.fvnb_core import Rect, containment_membership, hard_role_for_membership
from mnr_dataset.fvnb_oracle import FuzzyRule, RULE_ROLES, harden_panel_memberships


@dataclass
class FVNBConfig:
    panel_size: int = 128
    seed: int = 0
    theta_pos: float = 0.75
    delta: float = 0.20
    theta_neg: float = 0.45
    tau: float = 2.0
    sigma: float = 1.8
    margin: float = 0.0
    boundary_weight: float = 5.0
    t_norm: str = "product"
    epsilon: float = 0.001
    shuffle_candidates: bool = True


class FVNBGenerator:
    def __init__(self, config: FVNBConfig | None = None):
        self.config = config or FVNBConfig()
        self.random = random.Random(self.config.seed)
        size = self.config.panel_size
        outer_margin = self._scale_value(8)
        inner_left = self._scale_value(30)
        inner_right = self._scale_value(50)
        self.outer = Rect(outer_margin, outer_margin, size - outer_margin, size - outer_margin)
        self.inner = Rect(inner_left, inner_left, inner_right, inner_right)

    def generate_sample(self, sample_id: str, rule_id: str | None = None) -> Dict[str, object]:
        rule_id = rule_id or self.random.choice(list(RULE_VALUE_SAMPLERS.keys()))
        rule = FuzzyRule(rule_id, t_norm=self.config.t_norm, epsilon=self.config.epsilon)
        context_panels = [
            self._make_panel("context_{0}".format(index), rule_id, self._sample_values(rule_id), "context")
            for index in range(3)
        ]
        base_values = self._sample_values(rule_id)
        candidates = self._make_candidates(rule_id, base_values)
        self._score_panels(rule, context_panels)
        self._score_panels(rule, candidates)
        correct_index = 0
        if self.config.shuffle_candidates:
            indexed = list(enumerate(candidates))
            self.random.shuffle(indexed)
            correct_index = [old_index for old_index, _panel in indexed].index(0)
            candidates = [panel for _old_index, panel in indexed]

        metadata = {
            "sample_id": sample_id,
            "family": "containment",
            "rule": rule.as_dict(),
            "thresholds": {
                "theta_pos": self.config.theta_pos,
                "delta": self.config.delta,
                "theta_neg": self.config.theta_neg,
            },
            "scene": self._scene_dict(self.outer, self.inner),
            "scene_graph": self._scene_dict(self.outer, self.inner),
            "context": [self._panel_metadata(panel) for panel in context_panels],
            "candidates": [
                self._panel_metadata(panel, is_correct=(index == correct_index))
                for index, panel in enumerate(candidates)
            ],
            "validity": self._validity(rule, candidates, correct_index),
        }

        return {
            "context_images": np.array([panel["image"] for panel in context_panels], dtype=np.uint8),
            "answer_set_images": np.array([panel["image"] for panel in candidates], dtype=np.uint8),
            "correct_answer_image_index": correct_index,
            "metadata": metadata,
        }

    def _make_candidates(self, rule_id: str, values: Mapping[str, int]) -> List[Dict[str, object]]:
        wrong_boundary = dict(values)
        wrong_boundary["boundary"] = values["boundary"] + 1
        near_miss = dict(values)
        near_miss["outer"] = values["outer"] + 1

        return [
            self._make_panel("candidate_correct", rule_id, values, None, variant="crisp"),
            self._make_panel("candidate_same_number_shift", rule_id, values, "same_number_fuzzy_role_shift", variant="swap_outer_inner"),
            self._make_panel("candidate_same_coordinate_shift", rule_id, values, "same_coordinate_fuzzy_boundary_shift", variant="shifted_boundary"),
            self._make_panel("candidate_hard_role_flip", rule_id, values, "hard_role_invariant_fuzzy_flip", variant="soft_same_hard"),
            self._make_panel("candidate_fuzzy_false_positive", rule_id, values, "fuzzy_rule_false_positive", variant="wrong_membership"),
            self._make_panel("candidate_shortcut_false_positive", rule_id, values, "shortcut_consistent_false_positive", variant="all_outside"),
            self._make_panel("candidate_arithmetic_only", rule_id, wrong_boundary, "arithmetic_only_distractor", variant="crisp"),
            self._make_panel("candidate_near_miss", rule_id, near_miss, "near_miss_distractor", variant="crisp"),
        ]

    def _make_panel(
        self,
        panel_id: str,
        rule_id: str,
        values: Mapping[str, int],
        negative_type: str | None,
        variant: str = "crisp",
    ) -> Dict[str, object]:
        outer, inner, positions = self._variant_geometry(variant)
        anchors = [
            self._anchor("outer", values["outer"], positions["outer"], outer, inner),
            self._anchor("inner", values["inner"], positions["inner"], outer, inner),
            self._anchor("boundary", values["boundary"], positions["boundary"], outer, inner),
        ]
        panel = {
            "panel_id": panel_id,
            "rule_id": rule_id,
            "negative_type": negative_type,
            "scene": self._scene_dict(outer, inner),
            "anchors": anchors,
        }
        panel["image"] = self._render_panel(panel)
        return panel

    def _variant_geometry(self, variant: str) -> Tuple[Rect, Rect, Dict[str, Tuple[int, int]]]:
        positions = self._base_positions()
        crisp = {
            "outer": positions["outer"],
            "inner": positions["inner"],
            "boundary": positions["boundary"],
        }
        if variant == "crisp":
            return self.outer, self.inner, crisp
        if variant == "swap_outer_inner":
            return self.outer, self.inner, {
                "outer": positions["inner"],
                "inner": positions["outer"],
                "boundary": positions["boundary"],
            }
        if variant == "shifted_boundary":
            shifted_inner = Rect(
                self._scale_value(22),
                self._scale_value(30),
                self._scale_value(42),
                self._scale_value(50),
            )
            return self.outer, shifted_inner, crisp
        if variant == "soft_same_hard":
            return self.outer, self.inner, {
                "outer": positions["soft_outer"],
                "inner": positions["soft_inner"],
                "boundary": positions["soft_boundary"],
            }
        if variant == "wrong_membership":
            return self.outer, self.inner, {
                "outer": positions["inner"],
                "inner": self._point(30, 40),
                "boundary": positions["outer"],
            }
        if variant == "all_outside":
            return self.outer, self.inner, {
                "outer": self._point(4, 34),
                "inner": self._point(4, 46),
                "boundary": self._point(4, 58),
            }
        raise ValueError("Unknown FVNB variant: {0}".format(variant))

    def _scale_value(self, value: int) -> int:
        return int(round(value * self.config.panel_size / 80.0))

    def _point(self, x: int, y: int) -> Tuple[int, int]:
        return self._scale_value(x), self._scale_value(y)

    def _base_positions(self) -> Dict[str, Tuple[int, int]]:
        if self.config.panel_size <= 80:
            return {
                "outer": self._point(20, 40),
                "inner": self._point(40, 40),
                "boundary": self._point(30, 40),
                "soft_outer": self._point(7, 40),
                "soft_inner": self._point(34, 40),
                "soft_boundary": self._point(28, 40),
            }
        return {
            "outer": self._point(16, 40),
            "inner": self._point(45, 40),
            "boundary": self._point(30, 40),
            "soft_outer": self._point(7, 40),
            "soft_inner": self._point(34, 40),
            "soft_boundary": self._point(28, 40),
        }

    def _anchor(self, role: str, value: int, point: Tuple[int, int], outer: Rect, inner: Rect) -> Dict[str, object]:
        membership = containment_membership(
            point,
            outer,
            inner,
            tau=self.config.tau,
            sigma=self.config.sigma,
            margin=self.config.margin,
            boundary_weight=self.config.boundary_weight,
        )
        return {
            "id": "a_{0}".format(role),
            "intended_role": role,
            "value": int(value),
            "position": {"x": int(point[0]), "y": int(point[1])},
            "xy": [int(point[0]), int(point[1])],
            "membership": membership.as_dict(),
            "hard_role": hard_role_for_membership(membership),
            "rule_hard_role": max(RULE_ROLES, key=lambda item: membership.normalized.get(item, 0.0)),
        }

    def _score_panels(self, rule: FuzzyRule, panels: List[Dict[str, object]]) -> None:
        for panel in panels:
            evaluation = rule.evaluate(panel)
            hard_evaluation = rule.evaluate(harden_panel_memberships(panel))
            panel["truth_score"] = evaluation.score
            panel["truth_assignment"] = evaluation.assignment
            panel["arithmetic_truth"] = evaluation.arithmetic_truth
            panel["membership_truths"] = evaluation.membership_truths
            panel["hard_truth_score"] = hard_evaluation.score

    def _sample_values(self, rule_id: str) -> Dict[str, int]:
        return RULE_VALUE_SAMPLERS[rule_id](self.random)

    def _validity(self, rule: FuzzyRule, candidates: List[Dict[str, object]], correct_index: int) -> Dict[str, object]:
        scores = [float(candidate["truth_score"]) for candidate in candidates]
        correct_score = scores[correct_index]
        competing = [score for index, score in enumerate(scores) if index != correct_index]
        max_competing = max(competing)
        return {
            "unique_answer": correct_score - max_competing >= self.config.delta,
            "positive_threshold": correct_score >= self.config.theta_pos,
            "negative_threshold": max_competing <= self.config.theta_neg,
            "top_margin": correct_score - max_competing,
            "max_negative_score": max_competing,
            "hardening_gap": correct_score - float(candidates[correct_index]["hard_truth_score"]),
            "t_norm": rule.t_norm,
        }

    def _panel_metadata(self, panel: Mapping[str, object], is_correct: bool = False) -> Dict[str, object]:
        return {
            "panel_id": panel["panel_id"],
            "is_correct": bool(is_correct),
            "negative_type": panel["negative_type"],
            "scene": panel["scene"],
            "anchors": panel["anchors"],
            "truth_score": float(panel.get("truth_score", 0.0)),
            "hard_truth_score": float(panel.get("hard_truth_score", 0.0)),
            "truth_assignment": panel.get("truth_assignment", {}),
            "arithmetic_truth": float(panel.get("arithmetic_truth", 0.0)),
            "membership_truths": panel.get("membership_truths", {}),
            "shortcut_consistency": self._shortcut_consistency(panel),
        }

    def _shortcut_consistency(self, panel: Mapping[str, object]) -> Dict[str, bool]:
        negative_type = panel["negative_type"]
        fuzzy_rule = negative_type is None
        return {
            "number_only": negative_type in {
                None,
                "same_number_fuzzy_role_shift",
                "same_coordinate_fuzzy_boundary_shift",
                "hard_role_invariant_fuzzy_flip",
                "fuzzy_rule_false_positive",
                "shortcut_consistent_false_positive",
            },
            "fixed_coordinate": negative_type in {
                None,
                "same_coordinate_fuzzy_boundary_shift",
                "hard_role_invariant_fuzzy_flip",
                "shortcut_consistent_false_positive",
            },
            "hard_role": negative_type in {None, "hard_role_invariant_fuzzy_flip", "fuzzy_rule_false_positive"},
            "fuzzy_rule": fuzzy_rule,
        }

    def _scene_dict(self, outer: Rect, inner: Rect) -> Dict[str, object]:
        return {"outer_rect": outer.as_dict(), "inner_rect": inner.as_dict()}

    def _render_panel(self, panel: Mapping[str, object]) -> np.ndarray:
        render_scale = 4
        size = self.config.panel_size
        high_size = size * render_scale
        image = Image.new("L", (high_size, high_size), color=255)
        draw = ImageDraw.Draw(image)
        scene = panel["scene"]
        outer = self._scaled_rect(scene["outer_rect"], render_scale)
        inner = self._scaled_rect(scene["inner_rect"], render_scale)
        line = max(2, self._scale_value(2) * render_scale)
        boundary_band = max(3, self._scale_value(3) * render_scale)

        draw.rectangle(outer, fill=246)
        draw.rectangle(inner, fill=255)
        draw.rectangle(
            [
                inner[0] - boundary_band,
                inner[1] - boundary_band,
                inner[2] + boundary_band,
                inner[3] + boundary_band,
            ],
            outline=226,
            width=boundary_band,
        )
        draw.rectangle(outer, outline=22, width=line)
        draw.rectangle(inner, outline=22, width=line)

        font = self._load_font(max(10, int(round(size * 0.12))) * render_scale)
        for anchor in panel["anchors"]:
            self._draw_digit_token(draw, anchor, font, render_scale)
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        return np.array(image, dtype=np.uint8)

    def _draw_digit_token(self, draw: ImageDraw.ImageDraw, anchor: Mapping[str, object], font: ImageFont.ImageFont, scale: int) -> None:
        text = str(anchor["value"])
        x = int(anchor["position"]["x"]) * scale
        y = int(anchor["position"]["y"]) * scale
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        radius = max(self._scale_value(5) * scale, int(max(text_width, text_height) * 0.56))
        pad = self._scale_value(3) * scale
        image_size = self.config.panel_size * scale
        cx = min(max(x, radius + pad), image_size - radius - pad)
        cy = min(max(y, radius + pad), image_size - radius - pad)
        token_box = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.ellipse(token_box, fill=252, outline=16, width=max(1, self._scale_value(1) * scale))
        draw.text(
            (cx - text_width / 2 - bbox[0], cy - text_height / 2 - bbox[1]),
            text,
            fill=18,
            font=font,
        )

    def _scaled_rect(self, rect: Mapping[str, float], scale: int) -> List[int]:
        return [
            int(round(float(rect["left"]) * scale)),
            int(round(float(rect["top"]) * scale)),
            int(round(float(rect["right"]) * scale)),
            int(round(float(rect["bottom"]) * scale)),
        ]

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "DejaVuSans-Bold.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
        return ImageFont.load_default()


def _sample_diff(random_state: random.Random) -> Dict[str, int]:
    inner = random_state.randint(2, 8)
    boundary = random_state.randint(1, 8)
    return {"outer": inner + boundary, "inner": inner, "boundary": boundary}


def _sample_sum(random_state: random.Random) -> Dict[str, int]:
    inner = random_state.randint(2, 8)
    boundary = random_state.randint(1, 8)
    return {"outer": inner + boundary, "inner": inner, "boundary": boundary}


def _sample_ratio(random_state: random.Random) -> Dict[str, int]:
    inner = random_state.randint(2, 5)
    boundary = random_state.randint(2, 5)
    return {"outer": inner * boundary, "inner": inner, "boundary": boundary}


RULE_VALUE_SAMPLERS = {
    "diff_outer_inner_boundary": _sample_diff,
    "sum_outer_inner_boundary": _sample_sum,
    "ratio_outer_inner_boundary": _sample_ratio,
}


def save_sample_npz(sample: Mapping[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        context_images=sample["context_images"],
        answer_set_images=sample["answer_set_images"],
        correct_answer_image_index=np.array(sample["correct_answer_image_index"], dtype=np.int64),
        metadata_json=np.array(json.dumps(sample["metadata"], ensure_ascii=False, sort_keys=True)),
    )
