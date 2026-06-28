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
    "serial": 2,
    "parallel": 2,
    "nested": 1,
    "inverse": 1,
    "calibration": 1,
}

BOUNDARY_SHAPES = {"circle", "square", "diamond", "hexagon"}


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


class TestVQExprQuantityAttributes(unittest.TestCase):
    def test_each_quantity_family_covers_1_to_9_without_visible_symbols(self):
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
        context = make_calibration_context("coverage", seed=4, black_digit=9)
        for family in QUANTITY_FAMILIES:
            for value in [1, 3, 5, 7, 9]:
                with self.subTest(family=family, value=value):
                    rendered = render_quantity(family, value, object_id="q", calibration_context=context, style_seed=3)
                    self.assertEqual(rendered.numeric_value, value)
                    self.assertIn(value, rendered.fuzzy_value["support"])
                    decoded = decode_observation(family, rendered.observation, context)
                    self.assertIn(str(value), decoded["membership"])
                    self.assertFalse(rendered.leakage_flags["contains_arabic_digit"])
                    self.assertFalse(rendered.leakage_flags["contains_operator_symbol"])
                    self.assertNotRegex(_visible_text(rendered.primitives), r"[0-9+\-*/=]")
                    self.assertIn("ink_area", rendered.visual_confounds)
                    self.assertIn("convex_hull_area", rendered.visual_confounds)

    def test_same_black_token_maps_to_9_and_zero_digit_across_samples(self):
        high_context = make_calibration_context("black_high", seed=1, black_digit=9)
        low_context = make_calibration_context("black_low", seed=2, black_digit=0)

        black_9 = render_quantity(
            "local_place_value_codebook",
            9,
            object_id="q_black_high",
            calibration_context=high_context,
        )
        self.assertEqual(black_9.observation["slot_tokens"]["slot_b"], "token_black")
        self.assertEqual(decode_observation("local_place_value_codebook", black_9.observation, high_context)["membership"]["9"], 1.0)

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
            self.assertEqual(metadata["value_range"], "1..9")
            self.assertEqual(metadata["rule_family"], rule_family)
            query = metadata["query_panel"]
            evaluation = evaluate_answer_aot(query["quantities"], query["answer_aot"])
            self.assertEqual(evaluation["output_value"], query["evaluation"]["output_value"])
            self.assertTrue(1 <= evaluation["output_value"] <= 9)
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
        self.assertEqual(metadata["visual_surface"]["style"], "typed_boundary_scope_grayscale_a_sig_lite")
        self.assertIn(metadata["visual_surface"]["attribute_family"], {"gray_level", "size_level", "count", "position_set"})
        self.assertIn(metadata["visual_surface"]["structure_template"], {"3x3Grid", "2x2Grid", "Out-InGrid", "Out-InCenter", "Left-Right"})
        self.assertEqual(metadata["visual_surface"]["scene_graph_schema"], "a_sig_lite_v3")
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
                self.assertEqual(graph["schema_version"], "a_sig_lite_v3")
                self.assertEqual(graph["structure"]["family"], structure_family)
                self.assertTrue(graph["structure"]["typed_boundary_shapes"])
                self.assertEqual(graph["rendering_contract"]["drawn_from_scene_graph"], True)
                self.assertEqual(graph["rendering_contract"]["no_visible_role_labels"], True)
                self.assertEqual(graph["rendering_contract"]["typed_boundary_shapes"], True)
                roles = {component["role"] for component in graph["components"]}
                self.assertEqual(roles, set(metadata["query_panel"]["quantities"].keys()) | {"target"})

                groups = graph["structure_groups"]
                self.assertGreaterEqual(len(groups), EXPECTED_MIN_STRUCTURE_GROUPS[rule_family])
                self.assertTrue(any(group["boundary_shape"] != "circle" for group in groups))
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
                if rule_family in {"serial", "parallel", "calibration"}:
                    self.assertTrue(any(group["visible_scope_marker"] for group in groups))
                if rule_family in {"nested", "inverse"}:
                    in_out = graph["structure"]["in_out_regions"]
                    self.assertEqual(in_out["out_roles"], ["q1"])
                    self.assertIn("target", in_out["in_roles"])
                    self.assertTrue(any(group["space_relation"] == "out_to_in" for group in groups))

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
        self.assertIn(high["token_to_digit"]["token_black"], [0, 9])
        self.assertEqual(low["token_to_digit"]["token_black"], 0)
        self.assertNotEqual(
            make_calibration_context("manual_high", seed=13, black_digit=9)["token_to_digit"]["token_black"],
            low["token_to_digit"]["token_black"],
        )


class TestVQExprArtifacts(unittest.TestCase):
    def test_dataset_writer_outputs_npz_metadata_report_and_overview(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            generator = VQExprDatasetGenerator(seed=21)
            report = generator.generate_dataset(num_prob=5, output_dir=out_dir, rule_schema="mixed")

            self.assertEqual(report["num_samples"], 5)
            self.assertEqual(report["schema_version"], "vqexpr_1_9_avr_1x3_v7_typed_boundary")
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
            self.assertEqual(metadata["value_range"], "1..9")
            self.assertEqual(metadata["presentation_layout"], "1x3_context_row")
            self.assertEqual(len(metadata["context_panels"]), 3)
            self.assertTrue(metadata["presentation_constraints"]["no_query_panel_in_context_row"])
            self.assertIn(metadata["rule_family"], RULE_FAMILIES)

            rows = (out_dir / "metadata.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 5)
            first_row = json.loads(rows[0])
            self.assertEqual(first_row["sample_id"], "vqexpr_000000")

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
            self.assertEqual(sample["metadata"]["value_range"], "1..9")
            self.assertEqual(sample["answer_set_images"].shape[0], 8)


if __name__ == "__main__":
    unittest.main()
