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
PANEL_CONTENT_MIN = 14
PANEL_CONTENT_MAX = 210
REGION_LAYOUT_BOX_SIZE = 42
MIN_ENTITY_RADIUS = 8
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

        calibration = make_calibration_context(sample_id=sample_id, seed=self.rng.randint(0, 10**7))
        visual_quantity_family = visual_quantity_family or self.rng.choice(QUANTITY_FAMILIES)
        if visual_quantity_family not in QUANTITY_FAMILIES:
            raise ValueError("Unknown visual quantity family: {0}".format(visual_quantity_family))
        expression_schema = expression_schema or self.rng.choice(list(EXPRESSION_SCHEMAS_BY_FAMILY[rule_family].keys()))
        if expression_schema not in EXPRESSION_SCHEMAS_BY_FAMILY[rule_family]:
            raise ValueError("Unknown expression schema for {0}: {1}".format(rule_family, expression_schema))
        split_axis = self.rng.choice(["horizontal", "vertical"])
        query = self._make_panel_program(
            sample_id + "_query",
            rule_family,
            calibration,
            visual_quantity_family,
            expression_schema,
            split_axis,
        )
        contexts = [
            self._make_panel_program(
                "{0}_ctx{1}".format(sample_id, idx),
                rule_family,
                calibration,
                visual_quantity_family,
                expression_schema,
                split_axis,
            )
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
            "boundary_split_axis": split_axis,
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
        split_axis: str,
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
            split_axis=split_axis,
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
    if str(quantity_family) not in QUANTITY_FAMILIES:
        raise ValueError("Unknown VQ-Expr quantity family: {0}".format(quantity_family))
    return str(quantity_family)


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
    split_axis: Optional[str] = None,
) -> Dict[str, object]:
    return _make_in_out_layout_blueprint(panel_id, rule_family, operand_roles, rng, split_axis=split_axis)


def _make_in_out_layout_blueprint(
    panel_id: str,
    rule_family: str,
    operand_roles: Sequence[str],
    rng: random.Random,
    split_axis: Optional[str] = None,
) -> Dict[str, object]:
    if set(operand_roles) != {"q1", "q2", "q3", "q4"}:
        raise ValueError("In/out boundary layouts require q1..q4 operands, got {0}".format(sorted(operand_roles)))

    boundary_id = "boundary_main"
    split_axis = split_axis or rng.choice(["horizontal", "vertical"])
    if split_axis not in {"horizontal", "vertical"}:
        raise ValueError("Unsupported split axis: {0}".format(split_axis))
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
    layout_specs = _in_out_layout_specs(center, radius, split_axis, boundary_id, boundary_shape, role_regions, rng)
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
    boundary_shape: str,
    role_regions: Mapping[str, str],
    rng: random.Random,
) -> List[Dict[str, object]]:
    sampled = _sample_in_out_role_centers(center, radius, split_axis, rng)
    if split_axis == "horizontal":
        layout_family = "BoundaryHorizontalSplit"
    else:
        layout_family = "BoundaryVerticalSplit"
    specs = []
    for role in ["q1", "q2", "q3", "q4", "target"]:
        region_id = str(role_regions[role])
        region = "answer" if role == "target" else ("outer" if region_id.startswith("outer") else "inner")
        sample_info = sampled[role]
        role_center = sample_info["center"]
        specs.append(
            _center_layout(
                role,
                layout_family,
                _bbox_from_center(role_center[0], role_center[1], REGION_LAYOUT_BOX_SIZE),
                region,
                z_order=3 if role == "target" else 2,
                extra={
                    "center": [int(role_center[0]), int(role_center[1])],
                    "region": region,
                    "region_id": region_id,
                    "boundary_id": boundary_id,
                    "split_axis": split_axis,
                    "position_sampling": "sampled_within_region",
                    "role_binding_source": "boundary_region",
                    "sampling_boundary_shape": boundary_shape,
                    "sampling_side": sample_info["side"],
                    "sampling_domain": sample_info["domain"],
                },
            )
        )
    return specs


def _sample_in_out_role_centers(
    center: Tuple[int, int],
    radius: int,
    split_axis: str,
    rng: random.Random,
) -> Dict[str, Dict[str, object]]:
    cx, cy = center
    answer_side = rng.choice(["left", "right"]) if split_axis == "horizontal" else rng.choice(["top", "bottom"])
    if split_axis == "horizontal":
        outer_locked = {"x": rng.randint(cx - 28, cx + 28)}
        inner_locked = {"x": rng.randint(cx - 10, cx + 10)}
        answer_locked = {"y": rng.randint(cy - 20, cy + 20)}
    else:
        outer_locked = {"y": rng.randint(cy - 28, cy + 28)}
        inner_locked = {"y": rng.randint(cy - 10, cy + 10)}
        answer_locked = {"x": rng.randint(cx - 20, cx + 20)}
    role_specs = {
        "q1": ("outer", "top" if split_axis == "horizontal" else "left", outer_locked),
        "q2": ("outer", "bottom" if split_axis == "horizontal" else "right", outer_locked),
        "q3": ("inner", "top" if split_axis == "horizontal" else "left", inner_locked),
        "q4": ("inner", "bottom" if split_axis == "horizontal" else "right", inner_locked),
        "target": ("answer", answer_side, answer_locked),
    }
    occupied: List[Tuple[int, int]] = []
    sampled: Dict[str, Dict[str, object]] = {}
    for role in ["q1", "q2", "q3", "q4", "target"]:
        region, side, locked_axis = role_specs[role]
        role_center, domain = _sample_region_center(center, radius, region, side, rng, occupied, locked_axis)
        occupied.append(role_center)
        sampled[role] = {
            "center": role_center,
            "side": side,
            "domain": domain,
        }
    return sampled


def _sample_region_center(
    boundary_center: Tuple[int, int],
    boundary_radius: int,
    region: str,
    side: str,
    rng: random.Random,
    occupied: Sequence[Tuple[int, int]],
    locked_axis: Optional[Mapping[str, int]] = None,
) -> Tuple[Tuple[int, int], Dict[str, object]]:
    domain = _region_sampling_domain(boundary_center, boundary_radius, region, side)
    locked_axis = locked_axis or {}
    locked_x = _locked_coordinate(locked_axis.get("x"), domain["x_range"])
    locked_y = _locked_coordinate(locked_axis.get("y"), domain["y_range"])
    for _ in range(128):
        x = locked_x if locked_x is not None else rng.randint(domain["x_range"][0], domain["x_range"][1])
        y = locked_y if locked_y is not None else rng.randint(domain["y_range"][0], domain["y_range"][1])
        candidate = (int(x), int(y))
        if _sampled_region_center_is_valid(candidate, boundary_center, boundary_radius, region, side, occupied):
            return candidate, domain
    fallback = _fallback_region_center(boundary_center, boundary_radius, region, side)
    if locked_x is not None:
        fallback = (locked_x, fallback[1])
    if locked_y is not None:
        fallback = (fallback[0], locked_y)
    return fallback, domain


def _region_sampling_domain(
    boundary_center: Tuple[int, int],
    boundary_radius: int,
    region: str,
    side: str,
) -> Dict[str, object]:
    cx, cy = boundary_center
    half = REGION_LAYOUT_BOX_SIZE // 2
    low = PANEL_CONTENT_MIN + half + 3
    high = PANEL_CONTENT_MAX - half - 3
    outer_gap = boundary_radius + 26
    inner_span = 20
    inner_gap = 18
    answer_span = max(24, boundary_radius - 10)

    if region == "inner":
        x_range = [cx - inner_span, cx + inner_span]
        y_range = [cy - inner_span, cy + inner_span]
        if side == "top":
            y_range = [cy - inner_span, cy - inner_gap]
        elif side == "bottom":
            y_range = [cy + inner_gap, cy + inner_span]
        elif side == "left":
            x_range = [cx - inner_span, cx - inner_gap]
        elif side == "right":
            x_range = [cx + inner_gap, cx + inner_span]
        else:
            raise ValueError("Unsupported inner side: {0}".format(side))
    elif region == "outer":
        x_range = [low, high]
        y_range = [low, high]
        if side == "top":
            y_range = [low, cy - outer_gap]
        elif side == "bottom":
            y_range = [cy + outer_gap, high]
        elif side == "left":
            x_range = [low, cx - outer_gap]
        elif side == "right":
            x_range = [cx + outer_gap, high]
        else:
            raise ValueError("Unsupported outer side: {0}".format(side))
    elif region == "answer":
        x_range = [cx - answer_span, cx + answer_span]
        y_range = [cy - answer_span, cy + answer_span]
        if side == "left":
            x_range = [low, cx - outer_gap]
        elif side == "right":
            x_range = [cx + outer_gap, high]
        elif side == "top":
            y_range = [low, cy - outer_gap]
        elif side == "bottom":
            y_range = [cy + outer_gap, high]
        else:
            raise ValueError("Unsupported answer side: {0}".format(side))
    else:
        raise ValueError("Unsupported region: {0}".format(region))

    x_range = _clamp_sampling_range(x_range, low, high)
    y_range = _clamp_sampling_range(y_range, low, high)
    return {
        "coordinate_frame": "panel_pixels",
        "region": region,
        "side": side,
        "x_range": x_range,
        "y_range": y_range,
        "boundary_center": [int(cx), int(cy)],
        "boundary_radius": int(boundary_radius),
    }


def _sampled_region_center_is_valid(
    point: Tuple[int, int],
    boundary_center: Tuple[int, int],
    boundary_radius: int,
    region: str,
    side: str,
    occupied: Sequence[Tuple[int, int]],
) -> bool:
    x, y = point
    cx, cy = boundary_center
    distance = _point_distance(point, boundary_center)
    if region == "inner":
        if distance + 19.0 > float(boundary_radius) - 3.0:
            return False
        if side == "top" and not y < cy:
            return False
        if side == "bottom" and not y > cy:
            return False
        if side == "left" and not x < cx:
            return False
        if side == "right" and not x > cx:
            return False
    else:
        clearance = 22.0
        if distance - clearance < float(boundary_radius) + 3.0:
            return False
        if region == "outer":
            if side == "top" and not y < cy:
                return False
            if side == "bottom" and not y > cy:
                return False
            if side == "left" and not x < cx:
                return False
            if side == "right" and not x > cx:
                return False
    for previous in occupied:
        if _point_distance(point, previous) < 30.0:
            return False
    return True


def _fallback_region_center(
    boundary_center: Tuple[int, int],
    boundary_radius: int,
    region: str,
    side: str,
) -> Tuple[int, int]:
    cx, cy = boundary_center
    if region == "inner":
        offset = max(18, min(24, boundary_radius - 20))
        if side == "top":
            return (cx, cy - offset)
        if side == "bottom":
            return (cx, cy + offset)
        if side == "left":
            return (cx - offset, cy)
        return (cx + offset, cy)
    offset = boundary_radius + 28
    if side == "top":
        return (cx, PANEL_CONTENT_MIN + REGION_LAYOUT_BOX_SIZE // 2)
    if side == "bottom":
        return (cx, PANEL_CONTENT_MAX - REGION_LAYOUT_BOX_SIZE // 2)
    if side == "left":
        return (PANEL_CONTENT_MIN + REGION_LAYOUT_BOX_SIZE // 2, cy)
    if side == "right":
        return (PANEL_CONTENT_MAX - REGION_LAYOUT_BOX_SIZE // 2, cy)
    return (cx + offset, cy)


def _clamp_sampling_range(values: Sequence[int], low: int, high: int) -> List[int]:
    start = max(int(low), min(int(values[0]), int(values[1])))
    end = min(int(high), max(int(values[0]), int(values[1])))
    if start > end:
        midpoint = int(round((float(values[0]) + float(values[1])) / 2.0))
        midpoint = max(int(low), min(int(high), midpoint))
        return [midpoint, midpoint]
    return [start, end]


def _locked_coordinate(value: Optional[int], valid_range: Sequence[int]) -> Optional[int]:
    if value is None:
        return None
    return max(int(valid_range[0]), min(int(valid_range[1]), int(value)))


def _point_distance(left: Tuple[int, int], right: Tuple[int, int]) -> float:
    dx = float(left[0] - right[0])
    dy = float(left[1] - right[1])
    return float((dx * dx + dy * dy) ** 0.5)


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
    for key in (
        "region",
        "region_id",
        "boundary_id",
        "split_axis",
        "position_sampling",
        "role_binding_source",
        "sampling_boundary_shape",
        "sampling_side",
        "sampling_domain",
    ):
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
    if surface_family == "size_level":
        return [_size_entity(role, value, layout, outer_scope)]
    if surface_family == "color_lightness":
        return [_color_lightness_entity(role, value, layout, outer_scope)]
    if surface_family == "stroke_width":
        return [_stroke_width_entity(role, value, layout, outer_scope)]
    if surface_family == "aspect_ratio":
        return [_aspect_ratio_entity(role, value, layout, outer_scope)]
    raise ValueError("Unknown surface family: {0}".format(surface_family))


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


def _color_lightness_entity(role: str, value: int, layout: Mapping[str, object], outer_scope: bool) -> Dict[str, object]:
    bbox = layout["bbox"]
    center = _bbox_center(bbox)
    if outer_scope:
        return _entity(
            role=role,
            value=value,
            surface_family="color_lightness",
            center=center,
            radius=int(_bbox_min_side(bbox) * 0.44),
            fill=None,
            outline=_gray_for_value(value),
            outline_width=5,
            index=0,
            draw_mode="ring",
        )
    return _entity(
        role=role,
        value=value,
        surface_family="color_lightness",
        center=center,
        radius=int(_bbox_min_side(bbox) * 0.32),
        fill=_gray_for_value(value),
        outline=(28, 28, 28),
        outline_width=2,
        index=0,
    )


def _size_entity(role: str, value: int, layout: Mapping[str, object], outer_scope: bool) -> Dict[str, object]:
    bbox = layout["bbox"]
    radius_ratio = 0.24 + 0.018 * value if outer_scope else 0.18 + 0.024 * value
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


def _stroke_width_entity(role: str, value: int, layout: Mapping[str, object], outer_scope: bool) -> Dict[str, object]:
    bbox = layout["bbox"]
    radius_ratio = 0.36 if outer_scope else 0.29
    return _entity(
        role=role,
        value=value,
        surface_family="stroke_width",
        center=_bbox_center(bbox),
        radius=int(_bbox_min_side(bbox) * radius_ratio),
        fill=None if outer_scope else (236, 236, 236),
        outline=(28, 28, 28),
        outline_width=1 + int(value),
        index=0,
        draw_mode="ring" if outer_scope else "filled",
    )


def _aspect_ratio_entity(role: str, value: int, layout: Mapping[str, object], outer_scope: bool) -> Dict[str, object]:
    bbox = layout["bbox"]
    base = int(_bbox_min_side(bbox) * (0.31 if outer_scope else 0.27))
    ratio = 0.68 + (int(value) - VALUE_MIN) * (0.68 / float(VALUE_MAX - VALUE_MIN))
    radius_x = max(MIN_ENTITY_RADIUS, int(round(base * ratio)))
    radius_y = max(MIN_ENTITY_RADIUS, int(round(base)))
    return _entity(
        role=role,
        value=value,
        surface_family="aspect_ratio",
        center=_bbox_center(bbox),
        radius=max(radius_x, radius_y),
        fill=None if outer_scope else (112, 112, 112),
        outline=(28, 28, 28),
        outline_width=3 if outer_scope else 2,
        index=0,
        draw_mode="ring" if outer_scope else "filled",
        radius_x=radius_x,
        radius_y=radius_y,
        shape="ellipse",
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
    radius_x: Optional[int] = None,
    radius_y: Optional[int] = None,
    shape: str = "circle",
) -> Dict[str, object]:
    row = {
        "entity_id": "{0}_entity_{1}".format(role, index),
        "level": "Entity",
        "role": role,
        "shape": shape,
        "attribute_family": surface_family,
        "quantity_value": int(value),
        "center": [int(center[0]), int(center[1])],
        "radius": int(max(MIN_ENTITY_RADIUS, radius)),
        "fill": None if fill is None else [int(v) for v in fill],
        "outline": [int(v) for v in outline],
        "outline_width": int(outline_width),
        "draw_mode": draw_mode,
        "z_order": int(index),
    }
    if radius_x is not None and radius_y is not None:
        row["radius_x"] = int(max(MIN_ENTITY_RADIUS, radius_x))
        row["radius_y"] = int(max(MIN_ENTITY_RADIUS, radius_y))
    return row


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
    radius_x = int(entity.get("radius_x", radius))
    radius_y = int(entity.get("radius_y", radius))
    x, y = int(center[0]), int(center[1])
    bbox = (x - radius_x, y - radius_y, x + radius_x, y + radius_y)
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
    return 1


def _bounded_value(quantity: Mapping[str, object]) -> int:
    return max(VALUE_MIN, min(VALUE_MAX, int(quantity["numeric_value"])))


def _bbox_center(bbox: Sequence[int]) -> Tuple[int, int]:
    return (int((int(bbox[0]) + int(bbox[2])) / 2), int((int(bbox[1]) + int(bbox[3])) / 2))


def _bbox_min_side(bbox: Sequence[int]) -> int:
    return min(int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1]))


def _raven_structure_template(rule_family: str) -> str:
    return {
        "serial": "3x3Grid",
        "parallel": "2x2Grid",
        "nested": "Out-InGrid",
        "inverse": "Out-InCenter",
        "calibration": "Left-Right",
    }[str(rule_family)]


def _gray_for_value(value: int) -> Tuple[int, int, int]:
    gray = int(round(226 - (value - VALUE_MIN) * (178.0 / float(VALUE_MAX - VALUE_MIN))))
    return (gray, gray, gray)


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
