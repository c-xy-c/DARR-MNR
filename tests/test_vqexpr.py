import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mnr_dataset.vqexpr_generator import (
    NEGATIVE_MUTATIONS,
    VQExprDatasetGenerator,
    generate_killer_sample,
)
from mnr_dataset.vqexpr_audit import audit_dataset
from mnr_dataset.vqexpr_program import RULE_FAMILIES, evaluate_answer_aot
from mnr_dataset.vqexpr_program import EXPRESSION_SCHEMAS_BY_FAMILY
from mnr_dataset.vqexpr_quantity import (
    QUANTITY_FAMILIES,
    decode_observation,
    make_calibration_context,
    render_quantity,
)


EXPECTED_STRUCTURE_FAMILIES = {
    "serial": "3x3Grid",
    "parallel": "2x2Grid",
    "nested": "Out-InGrid",
    "inverse": "Out-InCenter",
    "calibration": "Left-Right",
}

EXPECTED_MIN_STRUCTURE_GROUPS = {
    "serial": 1,
    "parallel": 1,
    "nested": 1,
    "inverse": 1,
    "calibration": 1,
}

BOUNDARY_SHAPES = {"circle", "square", "diamond", "hexagon"}
PANEL_CONTENT_MIN = 14
PANEL_CONTENT_MAX = 210
BOUNDARY_SPLIT_AXES = {"horizontal", "vertical"}


def _visible_text(primitives):
    texts = []
    for primitive in primitives:
        if "text" in primitive:
            texts.append(str(primitive["text"]))
        if "glyph" in primitive:
            texts.append(str(primitive["glyph"]))
    return "".join(texts)


def _max_channel_delta(images):
    arr = images.astype(np.int16)
    return int(
        max(
            np.max(np.abs(arr[..., 0] - arr[..., 1])),
            np.max(np.abs(arr[..., 1] - arr[..., 2])),
            np.max(np.abs(arr[..., 0] - arr[..., 2])),
        )
    )


def _scene_entities(graph):
    rows = []
    for component in graph["components"]:
        for entity in component["entities"]:
            rows.append((component, entity))
    return rows


def _assert_scene_geometry_contract(testcase, graph):
    entities = _scene_entities(graph)
    for component, entity in entities:
        cx, cy = [float(v) for v in entity["center"]]
        radius = float(entity["radius"])
        testcase.assertGreaterEqual(cx - radius, PANEL_CONTENT_MIN, entity["entity_id"])
        testcase.assertGreaterEqual(cy - radius, PANEL_CONTENT_MIN, entity["entity_id"])
        testcase.assertLessEqual(cx + radius, PANEL_CONTENT_MAX, entity["entity_id"])
        testcase.assertLessEqual(cy + radius, PANEL_CONTENT_MAX, entity["entity_id"])
        x0, y0, x1, y1 = [float(v) for v in component["layout"]["bbox"]]
        testcase.assertGreaterEqual(cx - radius, x0 - 1.0, entity["entity_id"])
        testcase.assertGreaterEqual(cy - radius, y0 - 1.0, entity["entity_id"])
        testcase.assertLessEqual(cx + radius, x1 + 1.0, entity["entity_id"])
        testcase.assertLessEqual(cy + radius, y1 + 1.0, entity["entity_id"])

    for left_idx, (_, left) in enumerate(entities):
        for _, right in entities[left_idx + 1 :]:
            left_center = np.array(left["center"], dtype=np.float32)
            right_center = np.array(right["center"], dtype=np.float32)
            distance = float(np.linalg.norm(left_center - right_center))
            min_distance = float(left["radius"] + right["radius"] + 2.0)
            testcase.assertGreaterEqual(
                distance,
                min_distance,
                "{0} overlaps {1}".format(left["entity_id"], right["entity_id"]),
            )

    boundary_by_id = {
        boundary["boundary_id"]: boundary
        for boundary in graph.get("boundary_instances", [])
    }
    for component, entity in entities:
        layout = component["layout"]
        boundary_id = layout.get("boundary_id")
        if not boundary_id:
            continue
        boundary = boundary_by_id[boundary_id]
        distance = float(
            np.linalg.norm(
                np.array(entity["center"], dtype=np.float32)
                - np.array(boundary["center"], dtype=np.float32)
            )
        )
        radius = float(entity["radius"])
        if layout["region"] == "inner":
            testcase.assertLessEqual(
                distance + radius,
                float(boundary["radius"]) - 3.0,
                "{0} crosses inner boundary".format(entity["entity_id"]),
            )
        elif layout["region"] == "outer":
            testcase.assertGreaterEqual(
                distance - radius,
                float(boundary["radius"]) + 3.0,
                "{0} crosses outer boundary".format(entity["entity_id"]),
            )


class TestVQExprQuantityAttributes(unittest.TestCase):
    def test_each_quantity_family_covers_1_to_5_with_continuous_fuzzy_interpolation(self):
        self.assertEqual(
            set(QUANTITY_FAMILIES),
            {
                "local_place_value_codebook",
                "calibrated_metric_scale",
                "calibrated_area_grid",
                "chunked_path_topology",
                "part_ratio_dial",
            },
        )
        context = make_calibration_context("coverage", seed=4, black_digit=5)
        for family in QUANTITY_FAMILIES:
            for value in [1, 2, 3, 4, 5]:
                with self.subTest(family=family, value=value):
                    rendered = render_quantity(family, value, object_id="q", calibration_context=context, style_seed=3)
                    self.assertEqual(rendered.numeric_value, value)
                    self.assertEqual(rendered.fuzzy_value["discrete_levels"], [1, 2, 3, 4, 5])
                    self.assertIn(value, rendered.fuzzy_value["support"])
                    self.assertIn("continuous_value", rendered.fuzzy_value)
                    self.assertGreaterEqual(rendered.fuzzy_value["continuous_value"], 0.0)
                    self.assertLessEqual(rendered.fuzzy_value["continuous_value"], 1.0)
                    self.assertIn("interpolation_sigma", rendered.fuzzy_value)
                    self.assertGreater(rendered.fuzzy_value["interpolation_sigma"], 0.0)
                    decoded = decode_observation(family, rendered.observation, context)
                    self.assertIn(str(value), decoded["membership"])
                    self.assertFalse(rendered.leakage_flags["contains_arabic_digit"])
                    self.assertFalse(rendered.leakage_flags["contains_operator_symbol"])
                    self.assertNotRegex(_visible_text(rendered.primitives), r"[0-9+\-*/=]")
                    self.assertIn("ink_area", rendered.visual_confounds)
                    self.assertIn("convex_hull_area", rendered.visual_confounds)

    def test_same_black_token_maps_to_5_and_zero_digit_across_samples(self):
        high_context = make_calibration_context("black_high", seed=1, black_digit=5)
        low_context = make_calibration_context("black_low", seed=2, black_digit=0)

        black_5 = render_quantity(
            "local_place_value_codebook",
            5,
            object_id="q_black_high",
            calibration_context=high_context,
        )
        self.assertEqual(black_5.observation["slot_tokens"]["slot_b"], "token_black")
        self.assertEqual(decode_observation("local_place_value_codebook", black_5.observation, high_context)["membership"]["5"], 1.0)

        black_zero_digit_value = render_quantity(
            "local_place_value_codebook",
            1,
            object_id="q_black_low",
            calibration_context=low_context,
        )
        self.assertEqual(low_context["token_to_digit"]["token_black"], 0)
        self.assertEqual(black_zero_digit_value.observation["slot_tokens"]["slot_a"], "token_black")
        decoded = decode_observation("local_place_value_codebook", black_zero_digit_value.observation, low_context)
        self.assertEqual(decoded["membership"]["1"], 1.0)


class TestVQExprAoTAndCandidates(unittest.TestCase):
    def test_each_composition_rule_generates_valid_answer_aot(self):
        generator = VQExprDatasetGenerator(seed=11)
        seen = set()
        for rule_family in RULE_FAMILIES:
            sample = generator.generate_sample("unit_{0}".format(rule_family), rule_family=rule_family)
            metadata = sample["metadata"]
            seen.add(metadata["rule_family"])
            self.assertEqual(metadata["value_range"], "1..5")
            self.assertEqual(metadata["rule_family"], rule_family)
            query = metadata["query_panel"]
            evaluation = evaluate_answer_aot(query["quantities"], query["answer_aot"])
            self.assertEqual(evaluation["output_value"], query["evaluation"]["output_value"])
            self.assertTrue(1 <= evaluation["output_value"] <= 5)
            self.assertIn("visual_rule_graph", query)
            self.assertIn("binding_edges", query["visual_rule_graph"])
            self.assertIn("scope_edges", query["visual_rule_graph"])
        self.assertEqual(seen, set(RULE_FAMILIES))

    def test_generated_sample_has_eight_counterfactual_candidates(self):
        generator = VQExprDatasetGenerator(seed=12)
        sample = generator.generate_sample("unit_candidates", rule_family="calibration")
        metadata = sample["metadata"]
        self.assertEqual(sample["context_images"].shape[0], 3)
        self.assertEqual(sample["answer_set_images"].shape[0], 8)
        self.assertEqual(len(metadata["candidates"]), 8)
        correct = [candidate for candidate in metadata["candidates"] if candidate["is_correct"]]
        self.assertEqual(len(correct), 1)
        self.assertEqual(metadata["correct_answer_image_index"], correct[0]["candidate_index"])

        mutations = {candidate["primary_mutation"] for candidate in metadata["candidates"]}
        self.assertEqual(mutations, set(NEGATIVE_MUTATIONS))
        for candidate in metadata["candidates"]:
            expected_count = 0 if candidate["is_correct"] else 1
            self.assertEqual(candidate["primary_mutation_count"], expected_count)
            if candidate["is_correct"]:
                self.assertGreaterEqual(candidate["score"], metadata["thresholds"]["theta_pos"])
            else:
                self.assertLessEqual(candidate["score"], metadata["thresholds"]["theta_neg"])
            self.assertTrue(1 <= candidate["output_value"] <= 5)
        self.assertTrue(all(metadata["validity"].values()))

    def test_each_sample_uses_one_visual_quantity_family(self):
        generator = VQExprDatasetGenerator(seed=22)
        sample = generator.generate_sample("unit_visual_alphabet", rule_family="nested")
        metadata = sample["metadata"]
        visual_family = metadata["visual_quantity_family"]
        self.assertIn(visual_family, QUANTITY_FAMILIES)
        self.assertEqual(metadata["presentation_layout"], "1x3_context_row")
        self.assertEqual(metadata["strip_semantics"], "three_context_panels_plus_eight_full_panel_candidates")
        self.assertTrue(metadata["presentation_constraints"]["single_visual_alphabet_per_sample"])
        self.assertTrue(metadata["presentation_constraints"]["grayscale_only"])
        self.assertTrue(metadata["presentation_constraints"]["raven_like_object_attribute_scene"])
        self.assertTrue(metadata["presentation_constraints"]["minimal_surface"])
        self.assertFalse(metadata["presentation_constraints"]["single_geometric_primitive"])
        self.assertTrue(metadata["presentation_constraints"]["single_quantity_primitive"])
        self.assertTrue(metadata["presentation_constraints"]["semantic_boundary_lines_only"])
        self.assertTrue(metadata["presentation_constraints"]["three_context_panels"])
        self.assertTrue(metadata["presentation_constraints"]["no_query_panel_in_context_row"])
        self.assertTrue(metadata["presentation_constraints"]["no_right_side_output_node"])
        self.assertTrue(metadata["presentation_constraints"]["answer_set_separate_from_context"])
        self.assertTrue(metadata["presentation_constraints"]["full_panel_candidates"])
        self.assertTrue(metadata["presentation_constraints"]["structured_scene_graph"])
        self.assertTrue(metadata["presentation_constraints"]["scene_graph_driven_renderer"])
        self.assertTrue(metadata["presentation_constraints"]["raven_configuration_family"])
        self.assertTrue(metadata["presentation_constraints"]["structured_scope_groups"])
        self.assertTrue(metadata["presentation_constraints"]["typed_boundary_shapes"])
        self.assertEqual(metadata["visual_surface"]["style"], "dynamic_boundary_expression_grayscale_a_sig_lite")
        self.assertIn(metadata["visual_surface"]["attribute_family"], {"gray_level", "size_level", "count", "position_set"})
        self.assertIn(metadata["visual_surface"]["structure_template"], {"3x3Grid", "2x2Grid", "Out-InGrid", "Out-InCenter", "Left-Right"})
        self.assertEqual(metadata["visual_surface"]["scene_graph_schema"], "a_sig_lite_v4")
        self.assertEqual(_max_channel_delta(sample["context_images"]), 0)
        self.assertEqual(_max_channel_delta(sample["answer_set_images"]), 0)

        panels = [metadata["query_panel"]] + metadata["context_panels"]
        for panel in panels:
            self.assertEqual(set(panel["quantity_families"].values()), {visual_family})
            self.assertEqual(panel["output_quantity"]["family"], visual_family)
        for candidate in metadata["candidates"]:
            self.assertEqual(candidate["output_quantity"]["family"], visual_family)
            self.assertIn("visual_scene_graph", candidate)

    def test_visual_scene_graph_is_structured_and_self_contained(self):
        generator = VQExprDatasetGenerator(seed=31)
        for rule_family, structure_family in EXPECTED_STRUCTURE_FAMILIES.items():
            with self.subTest(rule_family=rule_family):
                sample = generator.generate_sample("unit_scene_{0}".format(rule_family), rule_family=rule_family)
                metadata = sample["metadata"]
                graph = metadata["query_panel"]["visual_scene_graph"]
                self.assertEqual(graph["schema_version"], "a_sig_lite_v4")
                self.assertEqual(graph["structure"]["family"], structure_family)
                self.assertTrue(graph["structure"]["typed_boundary_shapes"])
                self.assertTrue(graph["structure"]["dynamic_boundary_layout"])
                self.assertEqual(graph["rendering_contract"]["drawn_from_scene_graph"], True)
                self.assertEqual(graph["rendering_contract"]["no_visible_role_labels"], True)
                self.assertEqual(graph["rendering_contract"]["typed_boundary_shapes"], True)
                roles = {component["role"] for component in graph["components"]}
                self.assertEqual(roles, set(metadata["query_panel"]["quantities"].keys()) | {"target"})
                boundary_instances = graph["boundary_instances"]
                self.assertGreaterEqual(len(boundary_instances), 1)
                for boundary in boundary_instances:
                    self.assertEqual(boundary["level"], "Boundary")
                    self.assertIn(boundary["boundary_shape"], BOUNDARY_SHAPES)
                    self.assertEqual(len(boundary["center"]), 2)
                    self.assertGreater(boundary["radius"], 0)
                    self.assertTrue(boundary["governs_expression_binding"])

                groups = graph["structure_groups"]
                self.assertGreaterEqual(len(groups), EXPECTED_MIN_STRUCTURE_GROUPS[rule_family])
                for group in groups:
                    self.assertEqual(group["level"], "ComponentGroup")
                    self.assertGreaterEqual(len(group["roles"]), 2)
                    self.assertEqual(group["draw_mode"], "boundary")
                    self.assertEqual(group["shape"], group["boundary_shape"])
                    self.assertIn(group["boundary_shape"], BOUNDARY_SHAPES)
                    self.assertIn("space_relation", group)
                    self.assertEqual(len(group["center"]), 2)
                    self.assertGreater(group["radius"], 0)
                    self.assertTrue(set(group["roles"]).issubset(roles))
                in_out = graph["structure"]["in_out_regions"]
                self.assertEqual(in_out["boundary_id"], boundary_instances[0]["boundary_id"])
                self.assertIn(in_out["split_axis"], BOUNDARY_SPLIT_AXES)
                self.assertEqual(set(in_out["outer_roles"]), {"q1", "q2"})
                self.assertEqual(set(in_out["inner_roles"]), {"q3", "q4"})
                self.assertEqual(in_out["target_roles"], ["target"])
                self.assertEqual(
                    set(in_out["role_regions"].values()),
                    {"outer_a", "outer_b", "inner_a", "inner_b", "answer"},
                )
                self.assertEqual(set(in_out["region_groups"]), {"outer_a", "outer_b", "inner_a", "inner_b", "answer"})
                if in_out["split_axis"] == "horizontal":
                    self.assertEqual(in_out["region_groups"]["outer_a"]["side"], "top")
                    self.assertEqual(in_out["region_groups"]["outer_b"]["side"], "bottom")
                    self.assertEqual(in_out["region_groups"]["inner_a"]["side"], "top")
                    self.assertEqual(in_out["region_groups"]["inner_b"]["side"], "bottom")
                else:
                    self.assertEqual(in_out["region_groups"]["outer_a"]["side"], "left")
                    self.assertEqual(in_out["region_groups"]["outer_b"]["side"], "right")
                    self.assertEqual(in_out["region_groups"]["inner_a"]["side"], "left")
                    self.assertEqual(in_out["region_groups"]["inner_b"]["side"], "right")
                boundary_binding = metadata["query_panel"]["answer_aot"]["boundary_binding"]
                self.assertEqual(boundary_binding["boundary_id"], in_out["boundary_id"])
                self.assertEqual(boundary_binding["split_axis"], in_out["split_axis"])
                self.assertEqual(boundary_binding["outer_roles"], in_out["outer_roles"])
                self.assertEqual(boundary_binding["inner_roles"], in_out["inner_roles"])
                self.assertEqual(boundary_binding["role_regions"], in_out["role_regions"])
                self.assertTrue(any(group["space_relation"] == "out_to_in" for group in groups))
                for component in graph["components"]:
                    if component["role"] in {"q1", "q2"}:
                        self.assertEqual(component["layout"]["region"], "outer")
                        self.assertIn(component["layout"]["region_id"], {"outer_a", "outer_b"})
                        self.assertEqual(component["layout"]["boundary_id"], in_out["boundary_id"])
                    elif component["role"] in {"q3", "q4"}:
                        self.assertEqual(component["layout"]["region"], "inner")
                        self.assertIn(component["layout"]["region_id"], {"inner_a", "inner_b"})
                        self.assertEqual(component["layout"]["boundary_id"], in_out["boundary_id"])
                    elif component["role"] == "target":
                        self.assertEqual(component["layout"]["region"], "answer")
                        self.assertEqual(component["layout"]["region_id"], "answer")
                        self.assertEqual(component["layout"]["boundary_id"], in_out["boundary_id"])

                for component in graph["components"]:
                    layout = component["layout"]
                    self.assertIn(layout["level"], {"Layout"})
                    self.assertEqual(len(layout["bbox"]), 4)
                    self.assertGreater(layout["bbox"][2], layout["bbox"][0])
                    self.assertGreater(layout["bbox"][3], layout["bbox"][1])
                    self.assertGreaterEqual(len(component["entities"]), 1)
                    for entity in component["entities"]:
                        self.assertEqual(entity["level"], "Entity")
                        self.assertEqual(entity["role"], component["role"])
                        self.assertIn(entity["attribute_family"], {"gray_level", "size_level", "count", "position_set"})
                        self.assertEqual(len(entity["center"]), 2)
                        self.assertGreater(entity["radius"], 0)

                for candidate in metadata["candidates"]:
                    candidate_graph = candidate["visual_scene_graph"]
                    self.assertEqual(candidate_graph["structure"]["family"], structure_family)
                    self.assertEqual(
                        [group["roles"] for group in candidate_graph["structure_groups"]],
                        [group["roles"] for group in groups],
                    )
                    target_components = [component for component in candidate_graph["components"] if component["role"] == "target"]
                    self.assertEqual(len(target_components), 1)
                    self.assertEqual(target_components[0]["quantity_value"], candidate["output_value"])

    def test_calibration_ood_breaks_global_attribute_shortcut_contract(self):
        generator = VQExprDatasetGenerator(seed=13)
        high = generator.generate_sample("black_high_sample", rule_family="calibration")["metadata"]["calibration_context"]
        low = make_calibration_context("manual_low", seed=13, black_digit=0)
        self.assertIn(high["token_to_digit"]["token_black"], [0, 5])
        self.assertEqual(low["token_to_digit"]["token_black"], 0)
        self.assertNotEqual(
            make_calibration_context("manual_high", seed=13, black_digit=5)["token_to_digit"]["token_black"],
            low["token_to_digit"]["token_black"],
        )


class TestVQExprArtifacts(unittest.TestCase):
    def test_dataset_writer_outputs_npz_metadata_report_and_overview(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            generator = VQExprDatasetGenerator(seed=21)
            report = generator.generate_dataset(num_prob=5, output_dir=out_dir, rule_schema="mixed")

            self.assertEqual(report["num_samples"], 5)
            self.assertEqual(report["schema_version"], "vqexpr_1_5_avr_1x3_v8_dynamic_boundary")
            for rule_family in RULE_FAMILIES:
                self.assertGreaterEqual(len(report["expression_schema_counts"][rule_family]), 2)
            self.assertEqual(report["presentation_layout"], "1x3_context_row")
            self.assertEqual(report["strip_semantics"], "three_context_panels_plus_eight_full_panel_candidates")
            self.assertEqual(sum(report["rule_counts"].values()), 5)
            self.assertEqual(sum(report["visual_family_counts"].values()), 5)
            self.assertEqual(sum(report["correct_index_counts"].values()), 5)
            self.assertTrue(report["audit_notes"]["single_visual_alphabet_per_sample"])
            visual_stats = report["candidate_visual_stats"]
            self.assertEqual(visual_stats["correct"]["count"], 5)
            self.assertEqual(visual_stats["negative"]["count"], 35)
            self.assertIn("correct_minus_negative", visual_stats)
            self.assertEqual(set(visual_stats["slots"].keys()), {str(index) for index in range(8)})
            self.assertTrue(all(slot_stats["count"] == 5 for slot_stats in visual_stats["slots"].values()))
            heuristics = report["candidate_only_heuristics"]
            self.assertEqual(heuristics["num_samples"], 5)
            self.assertEqual(heuristics["chance_accuracy"], 0.125)
            self.assertIn("max_ink_fraction", heuristics["heuristics"])
            self.assertTrue(all(row["total"] == 5 for row in heuristics["heuristics"].values()))
            self.assertTrue((out_dir / "metadata.jsonl").exists())
            self.assertTrue((out_dir / "generation_report.json").exists())
            self.assertTrue((out_dir / "vqexpr_000000_overview.png").exists())

            npz_path = out_dir / "vqexpr_000000.npz"
            self.assertTrue(npz_path.exists())
            data = np.load(npz_path, allow_pickle=True)
            self.assertEqual(data["context_images"].shape[0], 3)
            self.assertEqual(data["answer_set_images"].shape[0], 8)
            metadata = json.loads(str(data["metadata_json"]))
            self.assertEqual(metadata["value_range"], "1..5")
            self.assertEqual(metadata["presentation_layout"], "1x3_context_row")
            self.assertEqual(len(metadata["context_panels"]), 3)
            self.assertTrue(metadata["presentation_constraints"]["no_query_panel_in_context_row"])
            self.assertIn(metadata["rule_family"], RULE_FAMILIES)

            rows = (out_dir / "metadata.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 5)
            first_row = json.loads(rows[0])
            self.assertEqual(first_row["sample_id"], "vqexpr_000000")

    def test_expression_schema_is_not_fixed_within_rule_family(self):
        generator = VQExprDatasetGenerator(seed=41)
        for rule_family, schemas in EXPRESSION_SCHEMAS_BY_FAMILY.items():
            self.assertGreaterEqual(len(schemas), 2)
            seen = set()
            for schema_id in schemas:
                sample = generator.generate_sample(
                    "unit_expr_{0}_{1}".format(rule_family, schema_id),
                    rule_family=rule_family,
                    expression_schema=schema_id,
                )
                query = sample["metadata"]["query_panel"]
                self.assertEqual(query["answer_aot"]["expression_schema"], schema_id)
                self.assertEqual(query["answer_aot"]["template"], schemas[schema_id]["template"])
                self.assertEqual(query["expression_schema"], schema_id)
                seen.add(query["answer_aot"]["template"])
            self.assertGreaterEqual(len(seen), 2)

    def test_boundary_layout_is_dynamic_but_structured(self):
        generator = VQExprDatasetGenerator(seed=51)
        sample_a = generator.generate_sample("unit_boundary_a", rule_family="nested", expression_schema="nested_outer_minus_sum")
        sample_b = generator.generate_sample("unit_boundary_b", rule_family="nested", expression_schema="nested_outer_minus_sum")
        graph_a = sample_a["metadata"]["query_panel"]["visual_scene_graph"]
        graph_b = sample_b["metadata"]["query_panel"]["visual_scene_graph"]
        boundary_a = graph_a["boundary_instances"][0]
        boundary_b = graph_b["boundary_instances"][0]
        centers_or_shapes_differ = (
            boundary_a["center"] != boundary_b["center"]
            or boundary_a["radius"] != boundary_b["radius"]
            or boundary_a["boundary_shape"] != boundary_b["boundary_shape"]
        )
        self.assertTrue(centers_or_shapes_differ)

        def role_centers(graph):
            return {
                component["role"]: component["layout"]["center"]
                for component in graph["components"]
            }

        self.assertNotEqual(role_centers(graph_a), role_centers(graph_b))
        for graph in [graph_a, graph_b]:
            boundary = graph["boundary_instances"][0]
            for component in graph["components"]:
                layout = component["layout"]
                self.assertIn(layout["region"], {"inner", "outer", "answer", "boundary"})
                self.assertEqual(layout["boundary_id"], boundary["boundary_id"])
                distance = np.linalg.norm(np.array(layout["center"], dtype=np.float32) - np.array(boundary["center"], dtype=np.float32))
                if layout["region"] == "inner":
                    self.assertLess(distance, boundary["radius"])
                elif layout["region"] in {"outer", "boundary"}:
                    self.assertGreaterEqual(distance, boundary["radius"] * 0.55)
            _assert_scene_geometry_contract(self, graph)

    def test_boundary_split_axis_varies_across_samples(self):
        generator = VQExprDatasetGenerator(seed=71)
        axes = set()
        for index in range(12):
            sample = generator.generate_sample(
                "unit_boundary_axis_{0}".format(index),
                rule_family="nested",
                expression_schema="nested_outer_minus_sum",
            )
            graph = sample["metadata"]["query_panel"]["visual_scene_graph"]
            axis = graph["structure"]["in_out_regions"]["split_axis"]
            self.assertIn(axis, BOUNDARY_SPLIT_AXES)
            axes.add(axis)
        self.assertEqual(axes, BOUNDARY_SPLIT_AXES)

    def test_size_level_objects_do_not_overlap_or_cross_regions(self):
        generator = VQExprDatasetGenerator(seed=61)
        for rule_family in RULE_FAMILIES:
            sample = generator.generate_sample(
                "unit_size_geometry_{0}".format(rule_family),
                rule_family=rule_family,
                visual_quantity_family="calibrated_metric_scale",
            )
            graphs = [sample["metadata"]["query_panel"]["visual_scene_graph"]]
            graphs.extend(candidate["visual_scene_graph"] for candidate in sample["metadata"]["candidates"])
            for graph in graphs:
                self.assertTrue(
                    any(entity["attribute_family"] == "size_level" for _, entity in _scene_entities(graph))
                )
                _assert_scene_geometry_contract(self, graph)

    def test_dataset_writer_balances_correct_answer_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            generator = VQExprDatasetGenerator(seed=23)
            report = generator.generate_dataset(num_prob=16, output_dir=out_dir, rule_schema="mixed")
            self.assertEqual(report["correct_index_counts"], {str(index): 2 for index in range(8)})

    def test_dataset_writer_balances_visual_quantity_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            generator = VQExprDatasetGenerator(seed=24)
            report = generator.generate_dataset(num_prob=15, output_dir=out_dir, rule_schema="mixed")
            self.assertEqual(report["visual_family_counts"], {family: 3 for family in QUANTITY_FAMILIES})

    def test_posthoc_audit_recomputes_oracle_and_candidate_only_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            generator = VQExprDatasetGenerator(seed=25)
            generator.generate_dataset(num_prob=8, output_dir=out_dir, rule_schema="mixed")
            audit_path = out_dir / "audit_report.json"
            audit = audit_dataset(out_dir, output_path=audit_path)
            self.assertTrue(audit_path.exists())
            self.assertEqual(audit["num_samples"], 8)
            self.assertEqual(audit["metadata_oracle_accuracy"], 1.0)
            self.assertEqual(audit["score_argmax_accuracy"], 1.0)
            self.assertEqual(audit["candidate_visual_stats"]["correct"]["count"], 8)
            self.assertEqual(audit["candidate_visual_stats"]["negative"]["count"], 56)
            self.assertEqual(audit["candidate_only_heuristics"]["num_samples"], 8)
            self.assertEqual(audit["learned_candidate_only_probe"]["status"], "skipped")

    def test_posthoc_audit_runs_learned_candidate_only_probe_when_large_enough(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            generator = VQExprDatasetGenerator(seed=26)
            generator.generate_dataset(num_prob=24, output_dir=out_dir, rule_schema="mixed")
            audit = audit_dataset(out_dir, learned_baseline_min_samples=16)
            probe = audit["learned_candidate_only_probe"]
            self.assertEqual(probe["status"], "ok")
            self.assertEqual(probe["num_samples"], 24)
            self.assertIn("train_accuracy", probe)
            self.assertIn("test_accuracy", probe)
            self.assertEqual(probe["chance_accuracy"], 0.125)

    def test_killer_entrypoint_writes_presentation_debug_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            sample = generate_killer_sample(output_dir=out_dir)
            self.assertTrue((out_dir / "vqexpr_killer_presentation.png").exists())
            self.assertTrue((out_dir / "vqexpr_killer_debug.png").exists())
            self.assertTrue((out_dir / "vqexpr_killer_metadata.json").exists())
            self.assertEqual(sample["metadata"]["value_range"], "1..5")
            self.assertEqual(sample["answer_set_images"].shape[0], 8)


if __name__ == "__main__":
    unittest.main()
