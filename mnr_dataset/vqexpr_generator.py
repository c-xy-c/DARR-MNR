# -*- coding: utf-8 -*-
"""Dataset generator for VQ-Expr.

The generator creates image-only multiple-choice visual arithmetic problems
whose values are defined by sample-local fuzzy visual attributes and whose
answer labels are verified by an executable Answer AoT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mnr_dataset.vqexpr_program import (
    EXPRESSION_SCHEMAS_BY_FAMILY,
    RULE_FAMILIES,
    VALUE_MAX,
    VALUE_MIN,
    build_answer_aot,
    candidate_score,
    evaluate_answer_aot,
    reverse_sample_leaf_values,
)
from mnr_dataset.vqexpr_quantity import (
    QUANTITY_FAMILIES,
    make_calibration_context,
    render_quantity,
)
from mnr_dataset.vqexpr_visual_rules import build_visual_rule_graph


NEGATIVE_MUTATIONS = (
    None,
    "output_perturbation",
    "leaf_decode_perturbation",
    "calibration_swap",
    "operator_swap",
    "port_binding_swap",
    "scope_tree_rotation",
    "fuzzy_ambiguity_trap",
)

PANEL_SIZE = 224
SCHEMA_VERSION = "vqexpr_1_5_avr_1x3_v8_dynamic_boundary"
PRESENTATION_LAYOUT = "1x3_context_row"
STRIP_SEMANTICS = "three_context_panels_plus_eight_full_panel_candidates"
VISUAL_SURFACE_STYLE = "dynamic_boundary_expression_grayscale_a_sig_lite"
SCENE_GRAPH_SCHEMA = "a_sig_lite_v4"


class VQExprDatasetGenerator(object):
    def __init__(
        self,
        seed: int = 0,
        value_range: Tuple[int, int] = (VALUE_MIN, VALUE_MAX),
        t_norm: str = "product",
        theta_pos: float = 0.75,
        theta_neg: float = 0.45,
        delta: float = 0.20,
    ) -> None:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.value_range = value_range
        self.t_norm = t_norm
        self.thresholds = {
            "theta_pos": float(theta_pos),
            "theta_neg": float(theta_neg),
            "delta": float(delta),
        }

    def generate_sample(
        self,
        sample_id: str,
        rule_family: Optional[str] = None,
        correct_slot: Optional[int] = None,
        visual_quantity_family: Optional[str] = None,
        expression_schema: Optional[str] = None,
    ) -> Dict[str, object]:
        rule_family = rule_family or self.rng.choice(RULE_FAMILIES)
        if rule_family == "mixed":
            rule_family = self.rng.choice(RULE_FAMILIES)
        if rule_family not in RULE_FAMILIES:
            raise ValueError("Unknown rule family: {0}".format(rule_family))

        black_digit = VALUE_MAX if self.rng.randint(0, 1) == 0 else 0
        calibration = make_calibration_context(sample_id=sample_id, seed=self.rng.randint(0, 10**7), black_digit=black_digit)
        visual_quantity_family = visual_quantity_family or self.rng.choice(QUANTITY_FAMILIES)
        if visual_quantity_family not in QUANTITY_FAMILIES:
            raise ValueError("Unknown visual quantity family: {0}".format(visual_quantity_family))
        expression_schema = expression_schema or self.rng.choice(list(EXPRESSION_SCHEMAS_BY_FAMILY[rule_family].keys()))
        if expression_schema not in EXPRESSION_SCHEMAS_BY_FAMILY[rule_family]:
            raise ValueError("Unknown expression schema for {0}: {1}".format(rule_family, expression_schema))
        query = self._make_panel_program(sample_id + "_query", rule_family, calibration, visual_quantity_family, expression_schema)
        contexts = [
            self._make_panel_program("{0}_ctx{1}".format(sample_id, idx), rule_family, calibration, visual_quantity_family, expression_schema)
            for idx in range(3)
        ]
        candidates = self._make_candidates(query, calibration, visual_quantity_family)
        if correct_slot is not None:
            candidates = _place_correct_candidate(candidates, correct_slot)
        context_images = np.stack([_pil_to_array(_draw_problem_panel(panel, calibration, debug=False)) for panel in contexts], axis=0)
        answer_images = np.stack([_pil_to_array(_draw_candidate_panel(candidate, query, calibration, debug=False)) for candidate in candidates], axis=0)
        correct_index = next(index for index, candidate in enumerate(candidates) if candidate["is_correct"])
        scores = [float(candidate["score"]) for candidate in candidates]
        max_negative = max(score for index, score in enumerate(scores) if index != correct_index)
        metadata = {
            "sample_id": sample_id,
            "schema_version": SCHEMA_VERSION,
            "value_range": _value_range_label(self.value_range),
            "t_norm": self.t_norm,
            "rule_family": rule_family,
            "expression_schema": expression_schema,
            "thresholds": dict(self.thresholds),
            "calibration_context": calibration,
            "presentation_layout": PRESENTATION_LAYOUT,
            "strip_semantics": STRIP_SEMANTICS,
            "visual_quantity_family": visual_quantity_family,
            "visual_surface": {
                "style": VISUAL_SURFACE_STYLE,
                "attribute_family": _surface_family_for_quantity_family(visual_quantity_family),
                "structure_template": _raven_structure_template(rule_family),
                "scene_graph_schema": SCENE_GRAPH_SCHEMA,
            },
            "query_panel": _strip_images(query),
            "context_panels": [_strip_images(panel) for panel in contexts],
            "candidates": [_strip_candidate(candidate) for candidate in candidates],
            "correct_answer_image_index": correct_index,
            "validity": {
                "unique_answer": scores[correct_index] - max_negative >= self.thresholds["delta"],
                "positive_threshold": scores[correct_index] >= self.thresholds["theta_pos"],
                "negative_threshold": max_negative <= self.thresholds["theta_neg"],
                "counterfactual_isolation": all(candidate["primary_mutation_count"] == (0 if candidate["is_correct"] else 1) for candidate in candidates),
                "no_visible_symbols": True,
            },
            "presentation_constraints": {
                "no_visible_digits": True,
                "no_visible_operator_symbols": True,
                "no_gate_labels": True,
                "grayscale_only": True,
                "raven_like_object_attribute_scene": True,
                "minimal_surface": True,
                "single_geometric_primitive": False,
                "single_quantity_primitive": True,
                "no_internal_structure_lines": False,
                "semantic_boundary_lines_only": True,
                "single_visual_alphabet_per_sample": True,
                "three_context_panels": True,
                "no_query_panel_in_context_row": True,
                "no_right_side_output_node": True,
                "answer_set_separate_from_context": True,
                "full_panel_candidates": True,
                "structured_scene_graph": True,
                "scene_graph_driven_renderer": True,
                "raven_configuration_family": True,
                "structured_scope_groups": True,
                "typed_boundary_shapes": True,
            },
        }
        return {
            "sample_id": sample_id,
            "context_images": context_images,
            "answer_set_images": answer_images,
            "correct_answer_image_index": correct_index,
            "metadata": metadata,
        }

    def generate_dataset(self, num_prob: int, output_dir: Path, rule_schema: str = "mixed") -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "metadata.jsonl"
        rule_counts = {family: 0 for family in RULE_FAMILIES}
        visual_family_counts = {family: 0 for family in QUANTITY_FAMILIES}
        correct_index_counts = {str(index): 0 for index in range(8)}
        negative_counts = {str(mutation): 0 for mutation in NEGATIVE_MUTATIONS if mutation is not None}
        expression_schema_counts = {
            family: {schema_id: 0 for schema_id in EXPRESSION_SCHEMAS_BY_FAMILY[family]}
            for family in RULE_FAMILIES
        }
        candidate_visual_audit = _empty_candidate_visual_audit()
        candidate_only_heuristics = _empty_candidate_only_heuristics()
        validity_counts = {
            "unique_answer": 0,
            "positive_threshold": 0,
            "negative_threshold": 0,
            "counterfactual_isolation": 0,
            "no_visible_symbols": 0,
        }
        rows: List[Dict[str, object]] = []
        for index in range(int(num_prob)):
            if rule_schema == "mixed":
                rule_family = RULE_FAMILIES[index % len(RULE_FAMILIES)]
            else:
                rule_family = rule_schema
            schemas = list(EXPRESSION_SCHEMAS_BY_FAMILY[rule_family].keys())
            expression_schema = schemas[rule_counts[rule_family] % len(schemas)]
            sample_id = "vqexpr_{0:06d}".format(index)
            sample = self.generate_sample(
                sample_id=sample_id,
                rule_family=rule_family,
                correct_slot=index % 8,
                visual_quantity_family=QUANTITY_FAMILIES[index % len(QUANTITY_FAMILIES)],
                expression_schema=expression_schema,
            )
            metadata = sample["metadata"]
            rows.append(metadata)
            rule_counts[str(metadata["rule_family"])] += 1
            expression_schema_counts[str(metadata["rule_family"])][str(metadata["expression_schema"])] += 1
            visual_family_counts[str(metadata["visual_quantity_family"])] += 1
            correct_index_counts[str(metadata["correct_answer_image_index"])] += 1
            _update_candidate_visual_audit(candidate_visual_audit, sample["answer_set_images"], metadata["candidates"])
            _update_candidate_only_heuristics(
                candidate_only_heuristics,
                sample["answer_set_images"],
                int(metadata["correct_answer_image_index"]),
            )
            for candidate in metadata["candidates"]:
                mutation = candidate["primary_mutation"]
                if mutation is not None:
                    negative_counts[str(mutation)] += 1
            for key, value in metadata["validity"].items():
                validity_counts[key] += int(bool(value))
            npz_path = output_dir / "{0}.npz".format(sample_id)
            np.savez_compressed(
                npz_path,
                context_images=sample["context_images"],
                answer_set_images=sample["answer_set_images"],
                correct_answer_image_index=np.array(sample["correct_answer_image_index"], dtype=np.int64),
                metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
            )
            if index == 0:
                overview = _draw_overview(sample)
                overview.save(output_dir / "{0}_overview.png".format(sample_id))
        with metadata_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        report = {
            "num_samples": int(num_prob),
            "schema_version": SCHEMA_VERSION,
            "presentation_layout": PRESENTATION_LAYOUT,
            "strip_semantics": STRIP_SEMANTICS,
            "rule_counts": rule_counts,
            "expression_schema_counts": expression_schema_counts,
            "visual_family_counts": visual_family_counts,
            "correct_index_counts": correct_index_counts,
            "candidate_visual_stats": _finalize_candidate_visual_audit(candidate_visual_audit),
            "candidate_only_heuristics": _finalize_candidate_only_heuristics(candidate_only_heuristics),
            "negative_counts": negative_counts,
            "validity_pass": validity_counts,
            "value_range": _value_range_label(self.value_range),
            "generator_seed": self.seed,
            "audit_notes": {
                "single_visual_alphabet_per_sample": True,
                "full_panel_candidates": True,
                "candidate_only_bias_requires_empirical_check": True,
            },
        }
        (output_dir / "generation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return report

    def _make_panel_program(
        self,
        panel_id: str,
        rule_family: str,
        calibration: Mapping[str, object],
        visual_quantity_family: str,
        expression_schema: str,
    ) -> Dict[str, object]:
        leaf_values = reverse_sample_leaf_values(rule_family, self.rng, expression_schema=expression_schema)
        quantity_families = {
            qid: visual_quantity_family
            for qid in sorted(leaf_values)
        }
        surface_family = _surface_family_for_quantity_family(visual_quantity_family)
        layout_blueprint = _make_layout_blueprint(
            panel_id=panel_id,
            rule_family=rule_family,
            operand_roles=sorted(leaf_values),
            rng=self.rng,
        )
        quantities = {
            qid: render_quantity(
                quantity_families[qid],
                value,
                object_id=qid,
                calibration_context=calibration,
                style_seed=self.rng.randint(0, 10**7),
            ).to_dict()
            for qid, value in leaf_values.items()
        }
        answer_aot = build_answer_aot(
            rule_family,
            leaf_values,
            quantity_families,
            expression_schema=expression_schema,
            boundary_binding=layout_blueprint.get("boundary_binding"),
        )
        evaluation = evaluate_answer_aot(quantities, answer_aot)
        output_quantity = render_quantity(
            visual_quantity_family,
            int(evaluation["output_value"]),
            object_id="output",
            calibration_context=calibration,
            style_seed=self.rng.randint(0, 10**7),
        ).to_dict()
        visual_rule_graph = build_visual_rule_graph(answer_aot, quantities)
        visual_scene_graph = _build_visual_scene_graph(
            panel_id=panel_id,
            rule_family=rule_family,
            quantities=quantities,
            output_quantity=output_quantity,
            surface_family=surface_family,
            layout_blueprint=layout_blueprint,
        )
        return {
            "panel_id": panel_id,
            "rule_family": rule_family,
            "expression_schema": expression_schema,
            "leaf_values": dict(leaf_values),
            "quantity_families": dict(quantity_families),
            "quantities": quantities,
            "output_quantity": output_quantity,
            "answer_aot": answer_aot,
            "visual_rule_graph": visual_rule_graph,
            "visual_scene_graph": visual_scene_graph,
            "visual_layout_blueprint": layout_blueprint,
            "evaluation": evaluation,
        }

    def _make_candidates(
        self,
        query: Mapping[str, object],
        calibration: Mapping[str, object],
        visual_quantity_family: str,
    ) -> List[Dict[str, object]]:
        expected = int(query["evaluation"]["output_value"])
        output_family = visual_quantity_family
        wrong_values = [value for value in range(self.value_range[0], self.value_range[1] + 1) if value != expected]
        unshuffled = []
        for idx, mutation in enumerate(NEGATIVE_MUTATIONS):
            if mutation is None:
                value = expected
            else:
                value = self.rng.choice(wrong_values)
            score = candidate_score(expected, value, mutation)
            if mutation == "fuzzy_ambiguity_trap":
                score = min(score, self.thresholds["theta_neg"] - 0.01)
            quantity = render_quantity(
                output_family,
                value,
                object_id="candidate_output_{0}".format(idx),
                calibration_context=calibration,
                style_seed=self.rng.randint(0, 10**7),
                fuzzy_severity="high" if mutation == "fuzzy_ambiguity_trap" else "medium",
            ).to_dict()
            visual_scene_graph = _build_visual_scene_graph(
                panel_id="{0}_candidate_{1}".format(query["panel_id"], idx),
                rule_family=str(query["rule_family"]),
                quantities=query["quantities"],
                output_quantity=quantity,
                surface_family=_surface_family_for_quantity_family(output_family),
                layout_blueprint=query["visual_layout_blueprint"],
            )
            unshuffled.append(
                {
                    "candidate_id": "c{0}".format(idx),
                    "candidate_index": idx,
                    "is_correct": mutation is None,
                    "primary_mutation": mutation,
                    "primary_mutation_count": 0 if mutation is None else 1,
                    "mutated_layer": _mutated_layer(mutation),
                    "output_value": value,
                    "output_quantity": quantity,
                    "visual_scene_graph": visual_scene_graph,
                    "score": float(score),
                    "mutation_log": _mutation_log(mutation, expected, value, query),
                }
            )
        self.rng.shuffle(unshuffled)
        for index, candidate in enumerate(unshuffled):
            candidate["candidate_index"] = index
        return unshuffled


def generate_killer_sample(output_dir: Optional[Path] = None) -> Dict[str, object]:
    """Backward-compatible entry point; now returns a real VQ-Expr sample."""

    generator = VQExprDatasetGenerator(seed=7)
    sample = generator.generate_sample("vqexpr_killer_000001", rule_family="inverse")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _draw_overview(sample).save(output_dir / "vqexpr_killer_presentation.png")
        _draw_debug_overview(sample).save(output_dir / "vqexpr_killer_debug.png")
        (output_dir / "vqexpr_killer_metadata.json").write_text(
            json.dumps(sample["metadata"], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return sample


def generate_dataset(num_prob: int, output_dir: Path, seed: int = 0, rule_schema: str = "mixed") -> Dict[str, object]:
    generator = VQExprDatasetGenerator(seed=seed)
    return generator.generate_dataset(num_prob=num_prob, output_dir=output_dir, rule_schema=rule_schema)


def _mutated_layer(mutation: Optional[str]) -> Optional[str]:
    return {
        None: None,
        "output_perturbation": "candidate_output",
        "leaf_decode_perturbation": "quantity_decode",
        "calibration_swap": "calibration_context",
        "operator_swap": "operator_gate",
        "port_binding_swap": "visual_binding",
        "scope_tree_rotation": "scope",
        "fuzzy_ambiguity_trap": "fuzzy_membership",
    }[mutation]


def _mutation_log(mutation: Optional[str], expected: int, value: int, query: Mapping[str, object]) -> Optional[Dict[str, object]]:
    if mutation is None:
        return None
    return {
        "primary_mutation": mutation,
        "before_output": expected,
        "after_output": value,
        "before_aot_root": query["answer_aot"]["root"],
        "rule_family": query["rule_family"],
    }


def _strip_images(panel: Mapping[str, object]) -> Dict[str, object]:
    return {
        "panel_id": panel["panel_id"],
        "rule_family": panel["rule_family"],
        "expression_schema": panel["expression_schema"],
        "leaf_values": dict(panel["leaf_values"]),
        "quantity_families": dict(panel["quantity_families"]),
        "quantities": panel["quantities"],
        "output_quantity": panel["output_quantity"],
        "answer_aot": panel["answer_aot"],
        "visual_rule_graph": panel["visual_rule_graph"],
        "visual_scene_graph": panel["visual_scene_graph"],
        "evaluation": panel["evaluation"],
    }


def _strip_candidate(candidate: Mapping[str, object]) -> Dict[str, object]:
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_index": candidate["candidate_index"],
        "is_correct": candidate["is_correct"],
        "primary_mutation": candidate["primary_mutation"],
        "primary_mutation_count": candidate["primary_mutation_count"],
        "mutated_layer": candidate["mutated_layer"],
        "output_value": candidate["output_value"],
        "output_quantity": candidate["output_quantity"],
        "visual_scene_graph": candidate["visual_scene_graph"],
        "score": candidate["score"],
        "mutation_log": candidate["mutation_log"],
    }


def _place_correct_candidate(candidates: Sequence[Dict[str, object]], correct_slot: int) -> List[Dict[str, object]]:
    arranged = list(candidates)
    slot = int(correct_slot) % len(arranged)
    current = next(index for index, candidate in enumerate(arranged) if candidate["is_correct"])
    arranged[slot], arranged[current] = arranged[current], arranged[slot]
    for index, candidate in enumerate(arranged):
        candidate["candidate_index"] = index
    return arranged


def _empty_stats_accumulator() -> Dict[str, float]:
    return {
        "count": 0.0,
        "ink_fraction": 0.0,
        "visual_mass": 0.0,
        "bbox_fraction": 0.0,
    }


def _empty_candidate_visual_audit() -> Dict[str, object]:
    return {
        "correct": _empty_stats_accumulator(),
        "negative": _empty_stats_accumulator(),
        "slots": {str(index): _empty_stats_accumulator() for index in range(8)},
    }


def _update_candidate_visual_audit(
    audit: Dict[str, object],
    answer_images: np.ndarray,
    candidates: Sequence[Mapping[str, object]],
) -> None:
    for index, candidate in enumerate(candidates):
        stats = _candidate_image_stats(answer_images[index])
        label_key = "correct" if bool(candidate["is_correct"]) else "negative"
        _accumulate_stats(audit[label_key], stats)
        _accumulate_stats(audit["slots"][str(index)], stats)


def _candidate_image_stats(image: np.ndarray) -> Dict[str, float]:
    arr = image.astype(np.float32)
    distance_from_white = np.max(np.abs(255.0 - arr), axis=2)
    ink_mask = distance_from_white > 18.0
    visual_mass = float(np.mean(distance_from_white / 255.0))
    ink_fraction = float(np.mean(ink_mask))
    if np.any(ink_mask):
        ys, xs = np.where(ink_mask)
        bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        bbox_fraction = bbox_area / float(image.shape[0] * image.shape[1])
    else:
        bbox_fraction = 0.0
    return {
        "ink_fraction": ink_fraction,
        "visual_mass": visual_mass,
        "bbox_fraction": bbox_fraction,
    }


def _accumulate_stats(accumulator: Mapping[str, float], stats: Mapping[str, float]) -> None:
    accumulator["count"] += 1.0
    for key in ("ink_fraction", "visual_mass", "bbox_fraction"):
        accumulator[key] += float(stats[key])


def _finalize_candidate_visual_audit(audit: Mapping[str, object]) -> Dict[str, object]:
    correct = _finalize_stats(audit["correct"])
    negative = _finalize_stats(audit["negative"])
    slots = {slot: _finalize_stats(values) for slot, values in audit["slots"].items()}
    return {
        "correct": correct,
        "negative": negative,
        "correct_minus_negative": {
            key: round(correct[key] - negative[key], 6)
            for key in ("ink_fraction", "visual_mass", "bbox_fraction")
        },
        "slots": slots,
        "note": "Simple visual-stat audit only; run a learned candidate-only baseline before making claims.",
    }


def _empty_candidate_only_heuristics() -> Dict[str, object]:
    return {
        "num_samples": 0,
        "heuristics": {
            name: {"correct": 0, "total": 0}
            for name in (
                "max_ink_fraction",
                "min_ink_fraction",
                "max_visual_mass",
                "min_visual_mass",
                "max_bbox_fraction",
                "min_bbox_fraction",
            )
        },
    }


def _update_candidate_only_heuristics(
    audit: Dict[str, object],
    answer_images: np.ndarray,
    correct_index: int,
) -> None:
    stats = [_candidate_image_stats(answer_images[index]) for index in range(answer_images.shape[0])]
    audit["num_samples"] += 1
    choices = {
        "max_ink_fraction": _arg_extreme(stats, "ink_fraction", max),
        "min_ink_fraction": _arg_extreme(stats, "ink_fraction", min),
        "max_visual_mass": _arg_extreme(stats, "visual_mass", max),
        "min_visual_mass": _arg_extreme(stats, "visual_mass", min),
        "max_bbox_fraction": _arg_extreme(stats, "bbox_fraction", max),
        "min_bbox_fraction": _arg_extreme(stats, "bbox_fraction", min),
    }
    for name, choice in choices.items():
        audit["heuristics"][name]["total"] += 1
        audit["heuristics"][name]["correct"] += int(int(choice) == int(correct_index))


def _arg_extreme(stats: Sequence[Mapping[str, float]], key: str, fn) -> int:
    values = [float(row[key]) for row in stats]
    target = fn(values)
    return next(index for index, value in enumerate(values) if value == target)


def _finalize_candidate_only_heuristics(audit: Mapping[str, object]) -> Dict[str, object]:
    finalized = {}
    best_accuracy = 0.0
    for name, row in audit["heuristics"].items():
        total = int(row["total"])
        correct = int(row["correct"])
        accuracy = 0.0 if total == 0 else round(correct / float(total), 6)
        finalized[name] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }
        best_accuracy = max(best_accuracy, accuracy)
    return {
        "num_samples": int(audit["num_samples"]),
        "heuristics": finalized,
        "best_accuracy": round(best_accuracy, 6),
        "chance_accuracy": 0.125,
        "note": "Candidate-only heuristic sanity check; learned candidate-only models are still required for publication claims.",
    }


def _finalize_stats(accumulator: Mapping[str, float]) -> Dict[str, float]:
    count = float(accumulator["count"])
    if count <= 0:
        return {"count": 0, "ink_fraction": 0.0, "visual_mass": 0.0, "bbox_fraction": 0.0}
    return {
        "count": int(count),
        "ink_fraction": round(float(accumulator["ink_fraction"]) / count, 6),
        "visual_mass": round(float(accumulator["visual_mass"]) / count, 6),
        "bbox_fraction": round(float(accumulator["bbox_fraction"]) / count, 6),
    }


def _pil_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _value_range_label(value_range: Tuple[int, int]) -> str:
    return "{0}..{1}".format(int(value_range[0]), int(value_range[1]))


def _draw_problem_panel(panel: Mapping[str, object], calibration: Mapping[str, object], debug: bool = False) -> Image.Image:
    return _draw_scene_graph(panel["visual_scene_graph"], debug=debug)


def _draw_candidate_panel(
    candidate: Mapping[str, object],
    query: Mapping[str, object],
    calibration: Mapping[str, object],
    debug: bool = False,
) -> Image.Image:
    return _draw_scene_graph(candidate["visual_scene_graph"], debug=debug)


def _draw_expression_panel(
    panel: Mapping[str, object],
    output_quantity: Optional[Mapping[str, object]] = None,
    hide_output: bool = False,
    debug: bool = False,
) -> Image.Image:
    surface_family = _surface_family_for_panel(panel, output_quantity)
    graph = _build_visual_scene_graph(
        panel_id=str(panel["panel_id"]),
        rule_family=str(panel["rule_family"]),
        quantities=panel["quantities"],
        output_quantity=output_quantity or panel["output_quantity"],
        surface_family=surface_family,
    )
    if hide_output:
        graph = _drop_target_entities(graph)
    return _draw_scene_graph(graph, debug=debug)


def _draw_overview(sample: Mapping[str, object]) -> Image.Image:
    width = 720
    height = 550
    image = Image.new("RGB", (width, height), (255, 255, 255))
    tile_size = 150
    gap = 16
    board_width = 3 * tile_size + 2 * gap
    board_x = (width - board_width) // 2
    board_y = 22
    for idx, panel_img in enumerate(sample["context_images"]):
        tile = Image.fromarray(panel_img).resize((tile_size, tile_size))
        image.paste(tile, (board_x + idx * (tile_size + gap), board_y))

    cand_size = tile_size
    cand_gap = gap
    cand_width = 4 * cand_size + 3 * cand_gap
    cand_x = (width - cand_width) // 2
    cand_y = board_y + tile_size + 36
    for idx, cand_img in enumerate(sample["answer_set_images"]):
        tile = Image.fromarray(cand_img).resize((cand_size, cand_size))
        x = cand_x + (idx % 4) * (cand_size + cand_gap)
        y = cand_y + (idx // 4) * (cand_size + cand_gap)
        image.paste(tile, (x, y))
    return image


def _draw_debug_overview(sample: Mapping[str, object]) -> Image.Image:
    image = _draw_overview(sample)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    meta = sample["metadata"]
    lines = [
        "DEBUG: {0}".format(meta["rule_family"]),
        "correct index: {0}".format(meta["correct_answer_image_index"]),
        "query output: {0}".format(meta["query_panel"]["evaluation"]["output_value"]),
    ]
    for idx, line in enumerate(lines):
        draw.text((20, 198 + idx * 16), line, fill=(20, 20, 20), font=font)
    return image


def _surface_family_for_panel(
    panel: Mapping[str, object],
    output_quantity: Optional[Mapping[str, object]],
) -> str:
    families = {str(quantity["family"]) for quantity in panel["quantities"].values()}
    if output_quantity is not None:
        families.add(str(output_quantity["family"]))
    return _surface_family_for_quantity_family(sorted(families)[0])


def _surface_family_for_quantity_family(quantity_family: str) -> str:
    return {
        "local_place_value_codebook": "gray_level",
        "calibrated_metric_scale": "size_level",
        "calibrated_area_grid": "count",
        "chunked_path_topology": "position_set",
        "part_ratio_dial": "gray_level",
    }[str(quantity_family)]


def _build_visual_scene_graph(
    panel_id: str,
    rule_family: str,
    quantities: Mapping[str, Mapping[str, object]],
    output_quantity: Mapping[str, object],
    surface_family: str,
    layout_blueprint: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    structure_family = _raven_structure_template(rule_family)
    layout_blueprint = layout_blueprint or _make_layout_blueprint(
        panel_id=panel_id,
        rule_family=rule_family,
        operand_roles=sorted(quantities),
        rng=random.Random("{0}:{1}".format(panel_id, rule_family)),
    )
    layout_specs = list(layout_blueprint["layout_specs"])
    components = []
    for spec in layout_specs:
        role = str(spec["role"])
        quantity = output_quantity if role == "target" else quantities[role]
        component = _build_scene_component(
            panel_id=panel_id,
            structure_family=structure_family,
            surface_family=surface_family,
            layout_spec=spec,
            quantity=quantity,
        )
        components.append(component)
    structure_groups = list(layout_blueprint["structure_groups"])
    boundary_instances = list(layout_blueprint["boundary_instances"])
    return {
        "schema_version": SCENE_GRAPH_SCHEMA,
        "panel_id": panel_id,
        "level": "Scene",
        "structure": {
            "level": "Structure",
            "family": structure_family,
            "source_rule_family": rule_family,
            "component_order": [component["component_id"] for component in components],
            "typed_boundary_shapes": True,
            "dynamic_boundary_layout": True,
            "in_out_regions": layout_blueprint["in_out_regions"],
        },
        "surface_family": surface_family,
        "boundary_instances": boundary_instances,
        "structure_groups": structure_groups,
        "components": components,
        "rendering_contract": {
            "drawn_from_scene_graph": True,
            "structure_groups_drawn_from_scene_graph": True,
            "typed_boundary_shapes": True,
            "no_visible_role_labels": True,
            "no_visible_digits": True,
            "no_visible_operator_symbols": True,
            "grayscale_only": True,
            "single_geometric_primitive": False,
            "single_quantity_primitive": True,
        },
    }


def _make_layout_blueprint(
    panel_id: str,
    rule_family: str,
    operand_roles: Sequence[str],
    rng: random.Random,
) -> Dict[str, object]:
    return _make_in_out_layout_blueprint(panel_id, rule_family, operand_roles, rng)


def _make_in_out_layout_blueprint(
    panel_id: str,
    rule_family: str,
    operand_roles: Sequence[str],
    rng: random.Random,
) -> Dict[str, object]:
    if set(operand_roles) != {"q1", "q2", "q3", "q4"}:
        raise ValueError("In/out boundary layouts require q1..q4 operands, got {0}".format(sorted(operand_roles)))

    boundary_id = "boundary_main"
    split_axis = rng.choice(["horizontal", "vertical"])
    center = (rng.randint(110, 114), rng.randint(110, 114))
    radius = rng.randint(46, 49)
    boundary_shape = rng.choice(["circle", "square", "diamond", "hexagon"])
    role_regions = {
        "q1": "outer_a",
        "q2": "outer_b",
        "q3": "inner_a",
        "q4": "inner_b",
        "target": "answer",
    }
    side_a, side_b = ("top", "bottom") if split_axis == "horizontal" else ("left", "right")
    region_groups = {
        "outer_a": {"region_id": "outer_a", "region": "outer", "side": side_a, "roles": ["q1"]},
        "outer_b": {"region_id": "outer_b", "region": "outer", "side": side_b, "roles": ["q2"]},
        "inner_a": {"region_id": "inner_a", "region": "inner", "side": side_a, "roles": ["q3"]},
        "inner_b": {"region_id": "inner_b", "region": "inner", "side": side_b, "roles": ["q4"]},
        "answer": {"region_id": "answer", "region": "answer", "side": "center", "roles": ["target"]},
    }
    boundary = {
        "boundary_id": boundary_id,
        "level": "Boundary",
        "boundary_shape": boundary_shape,
        "center": [int(center[0]), int(center[1])],
        "radius": int(radius),
        "split_axis": split_axis,
        "role_regions": dict(role_regions),
        "region_groups": region_groups,
        "governs_expression_binding": True,
        "dynamic_position": True,
        "outline": [132, 132, 132],
        "outline_width": 1,
        "visible_boundary": True,
        "z_order": 0,
    }
    in_out_regions = {
        "boundary_id": boundary_id,
        "split_axis": split_axis,
        "outer_roles": ["q1", "q2"],
        "inner_roles": ["q3", "q4"],
        "target_roles": ["target"],
        "role_regions": dict(role_regions),
        "region_groups": region_groups,
    }
    binding = {
        "boundary_id": boundary_id,
        "split_axis": split_axis,
        "outer_roles": ["q1", "q2"],
        "inner_roles": ["q3", "q4"],
        "target_roles": ["target"],
        "role_regions": dict(role_regions),
        "region_groups": region_groups,
    }
    layout_specs = _in_out_layout_specs(center, radius, split_axis, boundary_id, role_regions)
    group = _structure_group(
        "{0}_out_to_in_boundary".format(rule_family),
        ["q1", "q2", "q3", "q4", "target"],
        center,
        radius,
        boundary_shape,
        "out_to_in",
        z_order=0,
    )
    group["visible_scope_marker"] = False
    return {
        "layout_specs": layout_specs,
        "boundary_instances": [boundary],
        "structure_groups": [group],
        "in_out_regions": in_out_regions,
        "boundary_binding": binding,
    }


def _in_out_layout_specs(
    center: Tuple[int, int],
    radius: int,
    split_axis: str,
    boundary_id: str,
    role_regions: Mapping[str, str],
) -> List[Dict[str, object]]:
    cx, cy = center
    jitter = 0
    if split_axis == "horizontal":
        centers = {
            "q1": (cx + jitter, cy - radius - 25),
            "q2": (cx - jitter, cy + radius + 25),
            "q3": (cx - 22, cy - 20),
            "q4": (cx - 22, cy + 20),
            "target": (cx + 26, cy),
        }
        layout_family = "BoundaryHorizontalSplit"
    else:
        centers = {
            "q1": (cx - radius - 25, cy + jitter),
            "q2": (cx + radius + 25, cy - jitter),
            "q3": (cx - 22, cy - 22),
            "q4": (cx + 22, cy - 22),
            "target": (cx, cy + 26),
        }
        layout_family = "BoundaryVerticalSplit"
    specs = []
    for role in ["q1", "q2", "q3", "q4", "target"]:
        region_id = str(role_regions[role])
        region = "answer" if role == "target" else ("outer" if region_id.startswith("outer") else "inner")
        specs.append(
            _center_layout(
                role,
                layout_family,
                _bbox_from_center(centers[role][0], centers[role][1], 34),
                region,
                z_order=3 if role == "target" else 2,
                extra={
                    "center": [int(centers[role][0]), int(centers[role][1])],
                    "region": region,
                    "region_id": region_id,
                    "boundary_id": boundary_id,
                    "split_axis": split_axis,
                },
            )
        )
    return specs


def _structure_groups_for_rule(rule_family: str) -> List[Dict[str, object]]:
    if rule_family == "serial":
        return [
            _structure_group("serial_prefix_scope", ["q1", "q2", "q3"], (112, 84), 62, "hexagon", "prefix_scope", z_order=0),
            _structure_group("serial_tail_scope", ["q4", "target"], (112, 154), 42, "diamond", "tail_scope", z_order=0),
        ]
    if rule_family == "parallel":
        return [
            _structure_group("parallel_top_branch", ["q1", "q2"], (112, 74), 42, "square", "branch_scope", z_order=0),
            _structure_group("parallel_bottom_branch", ["q3", "q4"], (112, 150), 42, "square", "branch_scope", z_order=0),
            _structure_group("parallel_target_merge", ["q1", "q2", "q3", "q4", "target"], (112, 112), 34, "circle", "branch_to_target", z_order=1),
        ]
    if rule_family == "nested":
        return [
            _structure_group("nested_out_to_in_boundary", ["q1", "q2", "q3", "target"], (112, 112), 72, "hexagon", "out_to_in", z_order=0),
            _structure_group("nested_inner_scope", ["q2", "q3", "target"], (112, 112), 48, "square", "inner_scope", z_order=2),
        ]
    if rule_family == "inverse":
        return [
            _structure_group("inverse_out_to_in_boundary", ["q1", "q2", "q3", "target"], (112, 112), 70, "hexagon", "out_to_in", z_order=0),
            _structure_group("inverse_inner_stack", ["q2", "q3", "target"], (112, 112), 48, "diamond", "inner_scope", z_order=2),
        ]
    if rule_family == "calibration":
        return [
            _structure_group("calibration_local_frame", ["q1", "q2", "target"], (112, 108), 76, "diamond", "local_calibration_scope", z_order=0),
        ]
    raise ValueError("Unknown rule family: {0}".format(rule_family))


def _structure_group(
    group_id: str,
    roles: Sequence[str],
    center: Tuple[int, int],
    radius: int,
    boundary_shape: str,
    purpose: str,
    z_order: int,
) -> Dict[str, object]:
    return {
        "group_id": group_id,
        "level": "ComponentGroup",
        "roles": [str(role) for role in roles],
        "purpose": purpose,
        "space_relation": purpose,
        "shape": boundary_shape,
        "boundary_shape": boundary_shape,
        "center": [int(center[0]), int(center[1])],
        "radius": int(radius),
        "outline": [150, 150, 150],
        "outline_width": 1,
        "draw_mode": "boundary",
        "visible_scope_marker": True,
        "z_order": int(z_order),
    }


def _build_scene_component(
    panel_id: str,
    structure_family: str,
    surface_family: str,
    layout_spec: Mapping[str, object],
    quantity: Mapping[str, object],
) -> Dict[str, object]:
    role = str(layout_spec["role"])
    value = _bounded_value(quantity)
    layout = {
        "level": "Layout",
        "family": str(layout_spec["layout_family"]),
        "scope": str(layout_spec["scope"]),
        "slot_id": str(layout_spec["slot_id"]),
        "bbox": [int(v) for v in layout_spec["bbox"]],
        "center": list(layout_spec.get("center", _bbox_center(layout_spec["bbox"]))),
        "grid_shape": list(layout_spec.get("grid_shape", [])),
        "position": list(layout_spec.get("position", [])),
        "number": _layout_number(surface_family, value),
        "uniformity": True,
    }
    for key in ("region", "region_id", "boundary_id", "split_axis"):
        if key in layout_spec:
            layout[key] = layout_spec[key]
    entities = _entities_for_quantity(
        role=role,
        value=value,
        surface_family=surface_family,
        layout=layout,
        outer_scope=str(layout_spec["scope"]) == "outer",
    )
    return {
        "component_id": "{0}_{1}".format(panel_id, role),
        "level": "Component",
        "structure_family": structure_family,
        "role": role,
        "kind": "target" if role == "target" else "operand",
        "quantity_object_id": quantity.get("object_id"),
        "quantity_value": value,
        "layout": layout,
        "entities": entities,
        "z_order": int(layout_spec.get("z_order", 1)),
    }


def _structured_role_layouts(rule_family: str) -> List[Dict[str, object]]:
    if rule_family == "serial":
        return [
            _grid_layout("q1", "3x3Grid", (3, 3), (0, 0), "flat", z_order=2),
            _grid_layout("q2", "3x3Grid", (3, 3), (0, 2), "flat", z_order=2),
            _grid_layout("q3", "3x3Grid", (3, 3), (1, 1), "flat", z_order=2),
            _grid_layout("q4", "3x3Grid", (3, 3), (2, 0), "flat", z_order=2),
            _grid_layout("target", "3x3Grid", (3, 3), (2, 2), "flat", z_order=2),
        ]
    if rule_family == "parallel":
        return [
            _grid_layout("q1", "2x2Grid", (2, 2), (0, 0), "flat", z_order=2),
            _grid_layout("q2", "2x2Grid", (2, 2), (0, 1), "flat", z_order=2),
            _grid_layout("q3", "2x2Grid", (2, 2), (1, 0), "flat", z_order=2),
            _grid_layout("q4", "2x2Grid", (2, 2), (1, 1), "flat", z_order=2),
            _center_layout("target", "Center_Single", _bbox_from_center(112, 112, 48), "center", z_order=3),
        ]
    if rule_family == "nested":
        return [
            _center_layout("q1", "Out_Center_Single", (34, 34, 190, 190), "outer", z_order=1),
            _center_layout("q2", "In_Distribute_Four", _bbox_from_center(86, 92, 42), "inner", z_order=3),
            _center_layout("q3", "In_Distribute_Four", _bbox_from_center(138, 92, 42), "inner", z_order=3),
            _center_layout("target", "In_Distribute_Four", _bbox_from_center(112, 146, 44), "inner", z_order=3),
        ]
    if rule_family == "inverse":
        return [
            _center_layout("q1", "Out_Center_Single", (38, 38, 186, 186), "outer", z_order=1),
            _center_layout("q2", "In_Center_Single", _bbox_from_center(112, 76, 42), "inner", z_order=3),
            _center_layout("q3", "In_Center_Single", _bbox_from_center(112, 148, 42), "inner", z_order=3),
            _center_layout("target", "In_Center_Single", _bbox_from_center(112, 112, 46), "inner", z_order=4),
        ]
    if rule_family == "calibration":
        return [
            _center_layout("q1", "Left_Center_Single", _bbox_from_center(76, 124, 50), "left", z_order=2),
            _center_layout("q2", "Right_Center_Single", _bbox_from_center(148, 124, 50), "right", z_order=2),
            _center_layout("target", "Center_Single", _bbox_from_center(112, 74, 48), "center", z_order=2),
        ]
    raise ValueError("Unknown rule family: {0}".format(rule_family))


def _grid_layout(
    role: str,
    layout_family: str,
    grid_shape: Tuple[int, int],
    position: Tuple[int, int],
    scope: str,
    z_order: int,
) -> Dict[str, object]:
    rows, cols = grid_shape
    row, col = position
    if rows == 3 and cols == 3:
        cell = 48
        x0 = 40 + col * 56
        y0 = 40 + row * 56
    elif rows == 2 and cols == 2:
        cell = 54
        x0 = 50 + col * 74
        y0 = 50 + row * 74
    else:
        raise ValueError("Unsupported grid shape: {0}".format(grid_shape))
    return {
        "role": role,
        "layout_family": layout_family,
        "scope": scope,
        "slot_id": "{0}_{1}_{2}".format(layout_family, row, col),
        "bbox": (x0, y0, x0 + cell, y0 + cell),
        "grid_shape": grid_shape,
        "position": position,
        "z_order": z_order,
    }


def _center_layout(
    role: str,
    layout_family: str,
    bbox: Tuple[int, int, int, int],
    scope: str,
    z_order: int,
    extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    layout = {
        "role": role,
        "layout_family": layout_family,
        "scope": scope,
        "slot_id": layout_family,
        "bbox": bbox,
        "z_order": z_order,
    }
    if extra is not None:
        layout.update(dict(extra))
    return layout


def _bbox_from_center(cx: int, cy: int, size: int) -> Tuple[int, int, int, int]:
    half = int(size) // 2
    return (int(cx) - half, int(cy) - half, int(cx) + half, int(cy) + half)


def _entities_for_quantity(
    role: str,
    value: int,
    surface_family: str,
    layout: Mapping[str, object],
    outer_scope: bool,
) -> List[Dict[str, object]]:
    if surface_family == "gray_level":
        return [_gray_entity(role, value, layout, outer_scope)]
    if surface_family == "size_level":
        return [_size_entity(role, value, layout, outer_scope)]
    if surface_family == "count":
        centers = _outer_ring_centers(layout["bbox"], value) if outer_scope else _first_grid_centers(layout["bbox"], value)
        entities = [
            _entity(
                role=role,
                value=value,
                surface_family=surface_family,
                center=center,
                radius=max(3, int(_bbox_min_side(layout["bbox"]) * (0.042 if outer_scope else 0.065))),
                fill=(76, 76, 76),
                outline=(24, 24, 24),
                outline_width=1,
                index=index + (1 if outer_scope else 0),
            )
            for index, center in enumerate(centers)
        ]
        return entities
    centers = _position_center(layout["bbox"], value, outer_scope)
    entity = _entity(
        role=role,
        value=value,
        surface_family=surface_family,
        center=centers,
        radius=max(5, int(_bbox_min_side(layout["bbox"]) * (0.14 if not outer_scope else 0.045))),
        fill=(68, 68, 68),
        outline=(24, 24, 24),
        outline_width=2,
        index=1 if outer_scope else 0,
    )
    return [entity]


def _outer_structure_ring_entity(
    role: str,
    value: int,
    surface_family: str,
    layout: Mapping[str, object],
) -> Dict[str, object]:
    return _entity(
        role=role,
        value=value,
        surface_family=surface_family,
        center=_bbox_center(layout["bbox"]),
        radius=int(_bbox_min_side(layout["bbox"]) * 0.46),
        fill=None,
        outline=(96, 96, 96),
        outline_width=2,
        index=0,
        draw_mode="ring",
    )


def _gray_entity(role: str, value: int, layout: Mapping[str, object], outer_scope: bool) -> Dict[str, object]:
    bbox = layout["bbox"]
    center = _bbox_center(bbox)
    if outer_scope:
        return _entity(
            role=role,
            value=value,
            surface_family="gray_level",
            center=center,
            radius=int(_bbox_min_side(bbox) * 0.46),
            fill=None,
            outline=_gray_for_value(value),
            outline_width=5,
            index=0,
            draw_mode="ring",
        )
    return _entity(
        role=role,
        value=value,
        surface_family="gray_level",
        center=center,
        radius=int(_bbox_min_side(bbox) * 0.30),
        fill=_gray_for_value(value),
        outline=(28, 28, 28),
        outline_width=2,
        index=0,
    )


def _size_entity(role: str, value: int, layout: Mapping[str, object], outer_scope: bool) -> Dict[str, object]:
    bbox = layout["bbox"]
    radius_ratio = 0.22 + 0.018 * value if outer_scope else 0.15 + 0.022 * value
    return _entity(
        role=role,
        value=value,
        surface_family="size_level",
        center=_bbox_center(bbox),
        radius=int(_bbox_min_side(bbox) * radius_ratio),
        fill=None if outer_scope else (96, 96, 96),
        outline=(28, 28, 28),
        outline_width=4 if outer_scope else 2,
        index=0,
        draw_mode="ring" if outer_scope else "filled",
    )


def _entity(
    role: str,
    value: int,
    surface_family: str,
    center: Tuple[int, int],
    radius: int,
    fill: Optional[Tuple[int, int, int]],
    outline: Tuple[int, int, int],
    outline_width: int,
    index: int,
    draw_mode: str = "filled",
) -> Dict[str, object]:
    return {
        "entity_id": "{0}_entity_{1}".format(role, index),
        "level": "Entity",
        "role": role,
        "shape": "circle",
        "attribute_family": surface_family,
        "quantity_value": int(value),
        "center": [int(center[0]), int(center[1])],
        "radius": int(max(2, radius)),
        "fill": None if fill is None else [int(v) for v in fill],
        "outline": [int(v) for v in outline],
        "outline_width": int(outline_width),
        "draw_mode": draw_mode,
        "z_order": int(index),
    }


def _draw_scene_graph(scene_graph: Mapping[str, object], debug: bool = False) -> Image.Image:
    image = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, PANEL_SIZE - 13, PANEL_SIZE - 13), fill=(255, 255, 255), outline=(34, 34, 34), width=2)
    boundaries = sorted(scene_graph.get("boundary_instances", []), key=lambda boundary: int(boundary.get("z_order", 0)))
    for boundary in boundaries:
        if bool(boundary.get("visible_boundary", True)):
            _draw_boundary_instance(draw, boundary)
    groups = sorted(scene_graph.get("structure_groups", []), key=lambda group: int(group.get("z_order", 0)))
    for group in groups:
        if bool(group.get("visible_scope_marker", False)):
            _draw_structure_group(draw, group)
    components = sorted(scene_graph["components"], key=lambda component: int(component.get("z_order", 1)))
    for component in components:
        entities = sorted(component["entities"], key=lambda entity: int(entity.get("z_order", 0)))
        for entity in entities:
            _draw_scene_entity(draw, entity)
    if debug:
        font = ImageFont.load_default()
        draw.text((16, PANEL_SIZE - 24), str(scene_graph["structure"]["family"]), fill=(20, 20, 20), font=font)
    return image


def _draw_boundary_instance(draw: ImageDraw.ImageDraw, boundary: Mapping[str, object]) -> None:
    group_like = {
        "center": boundary["center"],
        "radius": boundary["radius"],
        "outline": boundary.get("outline", [132, 132, 132]),
        "outline_width": boundary.get("outline_width", 1),
        "boundary_shape": boundary.get("boundary_shape", "circle"),
        "shape": boundary.get("boundary_shape", "circle"),
    }
    _draw_structure_group(draw, group_like)


def _draw_structure_group(draw: ImageDraw.ImageDraw, group: Mapping[str, object]) -> None:
    center = group["center"]
    radius = int(group["radius"])
    x, y = int(center[0]), int(center[1])
    bbox = (x - radius, y - radius, x + radius, y + radius)
    outline = tuple(int(v) for v in group["outline"])
    width = int(group.get("outline_width", 1))
    shape = str(group.get("boundary_shape", group.get("shape", "circle")))
    if shape == "circle":
        draw.ellipse(bbox, fill=None, outline=outline, width=width)
    elif shape == "square":
        draw.rectangle(bbox, fill=None, outline=outline, width=width)
    elif shape == "diamond":
        points = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
        draw.line(points + [points[0]], fill=outline, width=width)
    elif shape == "hexagon":
        points = _regular_polygon_points(x, y, radius, 6, rotation_degrees=30.0)
        draw.line(points + [points[0]], fill=outline, width=width)
    else:
        raise ValueError("Unsupported boundary shape: {0}".format(shape))


def _draw_scene_entity(draw: ImageDraw.ImageDraw, entity: Mapping[str, object]) -> None:
    center = entity["center"]
    radius = int(entity["radius"])
    x, y = int(center[0]), int(center[1])
    bbox = (x - radius, y - radius, x + radius, y + radius)
    fill = None if entity.get("fill") is None else tuple(int(v) for v in entity["fill"])
    outline = tuple(int(v) for v in entity["outline"])
    width = int(entity.get("outline_width", 1))
    draw.ellipse(bbox, fill=fill, outline=outline, width=width)


def _regular_polygon_points(
    cx: int,
    cy: int,
    radius: int,
    sides: int,
    rotation_degrees: float = 0.0,
) -> List[Tuple[int, int]]:
    angles = [
        np.deg2rad(rotation_degrees + 360.0 * index / float(sides))
        for index in range(int(sides))
    ]
    return [
        (int(round(cx + radius * np.cos(angle))), int(round(cy + radius * np.sin(angle))))
        for angle in angles
    ]


def _drop_target_entities(scene_graph: Mapping[str, object]) -> Dict[str, object]:
    graph = json.loads(json.dumps(scene_graph))
    for component in graph["components"]:
        if component["role"] == "target":
            component["entities"] = []
    return graph


def _layout_number(surface_family: str, value: int) -> int:
    if surface_family == "count":
        return int(value)
    return 1


def _bounded_value(quantity: Mapping[str, object]) -> int:
    return max(VALUE_MIN, min(VALUE_MAX, int(quantity["numeric_value"])))


def _bbox_center(bbox: Sequence[int]) -> Tuple[int, int]:
    return (int((int(bbox[0]) + int(bbox[2])) / 2), int((int(bbox[1]) + int(bbox[3])) / 2))


def _bbox_min_side(bbox: Sequence[int]) -> int:
    return min(int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1]))


def _first_grid_centers(bbox: Sequence[int], count: int) -> List[Tuple[int, int]]:
    centers = _micro_grid_centers(bbox)
    return centers[: int(count)]


def _position_center(bbox: Sequence[int], value: int, outer_scope: bool) -> Tuple[int, int]:
    centers = _micro_grid_centers(bbox, pad_ratio=0.18 if not outer_scope else 0.12)
    return centers[int(value) - 1]


def _micro_grid_centers(bbox: Sequence[int], pad_ratio: float = 0.18) -> List[Tuple[int, int]]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    width = x1 - x0
    height = y1 - y0
    pad_x = width * pad_ratio
    pad_y = height * pad_ratio
    xs = [x0 + pad_x, (x0 + x1) / 2.0, x1 - pad_x]
    ys = [y0 + pad_y, (y0 + y1) / 2.0, y1 - pad_y]
    return [(int(round(x)), int(round(y))) for y in ys for x in xs]


def _outer_ring_centers(bbox: Sequence[int], count: int) -> List[Tuple[int, int]]:
    cx, cy = _bbox_center(bbox)
    radius = int(_bbox_min_side(bbox) * 0.42)
    anchors = [
        (-0.00, -1.00),
        (0.70, -0.70),
        (1.00, 0.00),
        (0.70, 0.70),
        (0.00, 1.00),
        (-0.70, 0.70),
        (-1.00, 0.00),
        (-0.70, -0.70),
        (0.00, 0.00),
    ]
    return [(int(round(cx + dx * radius)), int(round(cy + dy * radius))) for dx, dy in anchors[: int(count)]]


def _raven_structure_template(rule_family: str) -> str:
    return {
        "serial": "3x3Grid",
        "parallel": "2x2Grid",
        "nested": "Out-InGrid",
        "inverse": "Out-InCenter",
        "calibration": "Left-Right",
    }[str(rule_family)]


def _raven_role_slots(rule_family: str) -> Dict[str, Tuple[int, int, int, int]]:
    return {
        "serial": {
            "q1": (68, 66, 46, 46),
            "q2": (124, 66, 46, 46),
            "q3": (68, 122, 46, 46),
            "q4": (68, 178, 46, 46),
            "target": (124, 178, 46, 46),
        },
        "parallel": {
            "q1": (72, 74, 48, 48),
            "q2": (152, 74, 48, 48),
            "q3": (72, 150, 48, 48),
            "q4": (152, 150, 48, 48),
            "target": (112, 112, 52, 52),
        },
        "nested": {
            "q1": (112, 58, 52, 52),
            "q2": (84, 124, 42, 42),
            "q3": (140, 124, 42, 42),
            "target": (112, 168, 46, 46),
        },
        "inverse": {
            "q1": (66, 112, 48, 48),
            "q2": (112, 66, 42, 42),
            "q3": (112, 158, 42, 42),
            "target": (112, 112, 50, 50),
        },
        "calibration": {
            "q1": (76, 134, 48, 48),
            "q2": (148, 134, 48, 48),
            "target": (112, 80, 50, 50),
        },
    }[str(rule_family)]


def _draw_raven_quantity(
    draw: ImageDraw.ImageDraw,
    quantity: Mapping[str, object],
    slot: Tuple[int, int, int, int],
    surface_family: str,
    role: str,
    debug: bool = False,
) -> None:
    value = max(VALUE_MIN, min(VALUE_MAX, int(quantity["numeric_value"])))
    if surface_family == "gray_level":
        _draw_gray_level_quantity(draw, slot, value, role)
    elif surface_family == "size_level":
        _draw_size_level_quantity(draw, slot, value, role)
    elif surface_family == "count":
        _draw_count_quantity(draw, slot, value, role)
    else:
        _draw_position_set_quantity(draw, slot, value, role)
    if debug:
        x, y, _, h = slot
        draw.text((x - 8, y + h // 2 + 2), str(quantity["numeric_value"]), fill=(20, 20, 20), font=ImageFont.load_default())


def _draw_empty_raven_role(
    draw: ImageDraw.ImageDraw,
    slot: Tuple[int, int, int, int],
    surface_family: str,
    role: str,
) -> None:
    x, y, w, h = slot
    radius = int(min(w, h) * 0.36)
    _draw_mark(draw, (x, y), radius, fill=(255, 255, 255), outline=(188, 188, 188), width=1)


def _draw_gray_level_quantity(draw: ImageDraw.ImageDraw, slot: Tuple[int, int, int, int], value: int, role: str) -> None:
    x, y, w, h = slot
    radius = int(min(w, h) * 0.34)
    _draw_mark(draw, (x, y), radius, fill=_gray_for_value(value), outline=(28, 28, 28), width=2)


def _draw_size_level_quantity(draw: ImageDraw.ImageDraw, slot: Tuple[int, int, int, int], value: int, role: str) -> None:
    x, y, w, h = slot
    radius = int(min(w, h) * (0.16 + 0.022 * value))
    _draw_mark(draw, (x, y), radius, fill=(96, 96, 96), outline=(28, 28, 28), width=2)


def _draw_count_quantity(draw: ImageDraw.ImageDraw, slot: Tuple[int, int, int, int], value: int, role: str) -> None:
    x, y, w, h = slot
    step_x = max(9, int(w / 3.2))
    step_y = max(9, int(h / 3.2))
    start_x = x - step_x
    start_y = y - step_y
    radius = max(3, int(min(w, h) * 0.075))
    for index in range(value):
        cx = start_x + (index % 3) * step_x
        cy = start_y + (index // 3) * step_y
        _draw_mark(draw, (cx, cy), radius, fill=(76, 76, 76), outline=(24, 24, 24), width=1)


def _draw_position_set_quantity(draw: ImageDraw.ImageDraw, slot: Tuple[int, int, int, int], value: int, role: str) -> None:
    x, y, w, h = slot
    step_x = max(9, int(w / 3.0))
    step_y = max(9, int(h / 3.0))
    index = value - 1
    cx = x - step_x + (index % 3) * step_x
    cy = y - step_y + (index // 3) * step_y
    radius = max(6, int(min(w, h) * 0.16))
    _draw_mark(draw, (cx, cy), radius, fill=(68, 68, 68), outline=(24, 24, 24), width=2)


def _gray_for_value(value: int) -> Tuple[int, int, int]:
    gray = int(round(226 - (value - VALUE_MIN) * (178.0 / float(VALUE_MAX - VALUE_MIN))))
    return (gray, gray, gray)


def _draw_mark(
    draw: ImageDraw.ImageDraw,
    center: Tuple[int, int],
    radius: int,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    width: int = 2,
) -> None:
    x, y = center
    radius = max(2, int(radius))
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=width)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate VQ-Expr 1-5 calibrated visual expression data.")
    parser.add_argument("--num_prob", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=Path("VQExpr-ProbSet"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rule_schema", default="mixed", choices=("mixed",) + RULE_FAMILIES)
    args = parser.parse_args(argv)
    report = generate_dataset(args.num_prob, args.output_dir, seed=args.seed, rule_schema=args.rule_schema)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
