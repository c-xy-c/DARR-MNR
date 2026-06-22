import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mnr_dataset.fvnb_core import (
    Rect,
    containment_membership,
    hard_role_for_membership,
)
from mnr_dataset.fvnb_baselines import (
    audit_metadata_rows,
    predict_fuzzy_symbolic,
    score_fixed_coordinate_candidates,
    score_membership_shuffled_candidates,
    score_number_only_candidates,
    tied_best_candidates,
    tied_best_from_scores,
)
from mnr_dataset.fvnb_oracle import FuzzyRule, harden_panel_memberships
from mnr_dataset.fvnb_generator import FVNBConfig, FVNBGenerator, save_sample_npz
from mnr_dataset.fvnb_visualize import save_sample_grid


class TestFVNBMembership(unittest.TestCase):
    def test_rect_boundary_distance_handles_outside_points_on_all_sides(self):
        rect = Rect(10, 10, 20, 20)

        self.assertAlmostEqual(rect.boundary_distance((15, 5)), 5.0)
        self.assertAlmostEqual(rect.boundary_distance((15, 25)), 5.0)
        self.assertAlmostEqual(rect.boundary_distance((5, 15)), 5.0)
        self.assertAlmostEqual(rect.boundary_distance((25, 15)), 5.0)
        self.assertAlmostEqual(rect.boundary_distance((5, 5)), 50 ** 0.5)

    def test_containment_membership_assigns_inner_boundary_outer_and_outside_roles(self):
        outer = Rect(8, 8, 72, 72)
        inner = Rect(30, 30, 50, 50)

        inner_mu = containment_membership((40, 40), outer, inner, tau=2.0, sigma=1.8)
        boundary_mu = containment_membership((30, 40), outer, inner, tau=2.0, sigma=1.8)
        outer_mu = containment_membership((20, 40), outer, inner, tau=2.0, sigma=1.8)
        outside_mu = containment_membership((4, 40), outer, inner, tau=2.0, sigma=1.8)

        self.assertEqual(hard_role_for_membership(inner_mu), "inner")
        self.assertEqual(hard_role_for_membership(boundary_mu), "boundary")
        self.assertEqual(hard_role_for_membership(outer_mu), "outer")
        self.assertEqual(hard_role_for_membership(outside_mu), "outside")
        self.assertAlmostEqual(sum(inner_mu.normalized.values()), 1.0, places=6)
        self.assertGreater(inner_mu.normalized["inner"], 0.90)
        self.assertGreater(boundary_mu.normalized["boundary"], 0.50)


class TestFVNBOracle(unittest.TestCase):
    def test_fuzzy_rule_truth_changes_when_hard_roles_stay_the_same(self):
        rule = FuzzyRule("diff_outer_inner_boundary", t_norm="product", epsilon=0.001)
        crisp_panel = {
            "anchors": [
                {"id": "a_outer", "value": 9, "membership": {"outer": 0.95, "inner": 0.01, "boundary": 0.04}},
                {"id": "a_inner", "value": 4, "membership": {"outer": 0.02, "inner": 0.95, "boundary": 0.03}},
                {"id": "a_boundary", "value": 5, "membership": {"outer": 0.03, "inner": 0.02, "boundary": 0.95}},
            ]
        }
        fuzzy_panel = {
            "anchors": [
                {"id": "a_outer", "value": 9, "membership": {"outer": 0.55, "inner": 0.20, "boundary": 0.25}},
                {"id": "a_inner", "value": 4, "membership": {"outer": 0.15, "inner": 0.55, "boundary": 0.30}},
                {"id": "a_boundary", "value": 5, "membership": {"outer": 0.20, "inner": 0.25, "boundary": 0.55}},
            ]
        }

        crisp_score = rule.evaluate(crisp_panel).score
        fuzzy_score = rule.evaluate(fuzzy_panel).score
        crisp_hard_score = rule.evaluate(harden_panel_memberships(crisp_panel)).score
        fuzzy_hard_score = rule.evaluate(harden_panel_memberships(fuzzy_panel)).score

        self.assertGreater(crisp_score, 0.80)
        self.assertLess(fuzzy_score, 0.25)
        self.assertEqual(crisp_hard_score, 1.0)
        self.assertEqual(fuzzy_hard_score, 1.0)


class TestFVNBGenerator(unittest.TestCase):
    def test_generator_produces_eight_way_sample_with_fuzzy_metadata(self):
        generator = FVNBGenerator(FVNBConfig(seed=7, panel_size=80, shuffle_candidates=False))
        sample = generator.generate_sample("unit_000001", rule_id="diff_outer_inner_boundary")

        self.assertEqual(sample["context_images"].shape, (3, 80, 80))
        self.assertEqual(sample["answer_set_images"].shape, (8, 80, 80))
        self.assertEqual(len(sample["metadata"]["candidates"]), 8)
        self.assertEqual(sample["correct_answer_image_index"], 0)

        candidates = sample["metadata"]["candidates"]
        correct = candidates[sample["correct_answer_image_index"]]
        scores = [candidate["truth_score"] for candidate in candidates]
        negative_types = {candidate["negative_type"] for candidate in candidates[1:]}

        self.assertTrue(correct["is_correct"])
        self.assertGreaterEqual(correct["truth_score"], sample["metadata"]["thresholds"]["theta_pos"])
        self.assertGreaterEqual(
            correct["truth_score"] - max(scores[1:]),
            sample["metadata"]["thresholds"]["delta"],
        )
        self.assertIn("hard_role_invariant_fuzzy_flip", negative_types)
        self.assertIn("same_coordinate_fuzzy_boundary_shift", negative_types)
        self.assertIn("fuzzy_rule_false_positive", negative_types)
        self.assertIn("membership", candidates[0]["anchors"][0])
        self.assertIn("hard_truth_score", candidates[0])

    def test_symbolic_renderer_uses_antialiasing_and_region_tones(self):
        generator = FVNBGenerator(FVNBConfig(seed=7, panel_size=80, shuffle_candidates=False))
        sample = generator.generate_sample("unit_render_quality", rule_id="diff_outer_inner_boundary")
        image = sample["answer_set_images"][sample["correct_answer_image_index"]]
        unique_values = np.unique(image)

        self.assertGreaterEqual(len(unique_values), 8)
        self.assertTrue(any(15 < value < 245 for value in unique_values))
        self.assertLess(int(image.min()), 20)
        self.assertGreater(int(image.max()), 245)
        self.assertLess(int(image[16, 16]), 250)
        self.assertLess(int(image[30, 40]), 250)

    def test_generator_enforces_fuzzy_negative_threshold_and_structured_metadata(self):
        generator = FVNBGenerator(FVNBConfig(seed=17, panel_size=80, shuffle_candidates=False))
        sample = generator.generate_sample("unit_000003", rule_id="diff_outer_inner_boundary")
        metadata = sample["metadata"]
        candidates = metadata["candidates"]
        theta_neg = metadata["thresholds"]["theta_neg"]

        self.assertEqual(metadata["validity"]["negative_threshold"], True)
        self.assertIn("scene_graph", metadata)
        self.assertEqual(metadata["rule"]["expression"], ["=", ["-", "outer", "inner"], "boundary"])

        for candidate in candidates[1:]:
            self.assertLessEqual(candidate["truth_score"], theta_neg)
            self.assertIn("shortcut_consistency", candidate)

        hard_flip = next(
            candidate for candidate in candidates
            if candidate["negative_type"] == "hard_role_invariant_fuzzy_flip"
        )
        self.assertLessEqual(hard_flip["truth_score"], theta_neg)
        self.assertEqual(hard_flip["hard_truth_score"], 1.0)
        self.assertEqual(
            {anchor["id"]: anchor["rule_hard_role"] for anchor in hard_flip["anchors"]},
            {"a_outer": "outer", "a_inner": "inner", "a_boundary": "boundary"},
        )

        anchor = candidates[0]["anchors"][0]
        self.assertIn("xy", anchor)
        self.assertIn("raw", anchor["membership"])
        self.assertIn("normalized", anchor["membership"])
        self.assertAlmostEqual(sum(anchor["membership"]["normalized"].values()), 1.0, places=6)

    def test_t_norm_variants_keep_default_validity_thresholds(self):
        for t_norm in ("product", "min", "lukasiewicz"):
            with self.subTest(t_norm=t_norm):
                generator = FVNBGenerator(FVNBConfig(seed=19, panel_size=80, t_norm=t_norm, shuffle_candidates=False))
                sample = generator.generate_sample("unit_{0}".format(t_norm), rule_id="diff_outer_inner_boundary")
                validity = sample["metadata"]["validity"]

                self.assertTrue(validity["unique_answer"])
                self.assertTrue(validity["positive_threshold"])
                self.assertTrue(validity["negative_threshold"])

    def test_default_symbolic_size_preserves_hard_flip_diagnostic(self):
        generator = FVNBGenerator(FVNBConfig(seed=7, shuffle_candidates=False))
        sample = generator.generate_sample("unit_default_symbolic", rule_id="diff_outer_inner_boundary")
        hard_flip = next(
            candidate for candidate in sample["metadata"]["candidates"]
            if candidate["negative_type"] == "hard_role_invariant_fuzzy_flip"
        )

        self.assertLessEqual(hard_flip["truth_score"], sample["metadata"]["thresholds"]["theta_neg"])
        self.assertEqual(hard_flip["hard_truth_score"], 1.0)

    def test_save_sample_npz_includes_images_label_and_metadata_json(self):
        generator = FVNBGenerator(FVNBConfig(seed=11, panel_size=80, shuffle_candidates=False))
        sample = generator.generate_sample("unit_000002", rule_id="sum_outer_inner_boundary")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            save_sample_npz(sample, path)
            loaded = np.load(path, allow_pickle=False)

            self.assertEqual(loaded["context_images"].shape, (3, 80, 80))
            self.assertEqual(loaded["answer_set_images"].shape, (8, 80, 80))
            self.assertEqual(int(loaded["correct_answer_image_index"]), 0)
            metadata = json.loads(str(loaded["metadata_json"]))
            self.assertEqual(metadata["sample_id"], "unit_000002")
            self.assertEqual(metadata["rule"]["id"], "sum_outer_inner_boundary")

    def test_visualization_grid_writes_overview_png(self):
        generator = FVNBGenerator(FVNBConfig(seed=11, panel_size=80, shuffle_candidates=False))
        sample = generator.generate_sample("unit_visual", rule_id="sum_outer_inner_boundary")

        with tempfile.TemporaryDirectory() as tmp:
            sample_path = Path(tmp) / "sample.npz"
            output_path = Path(tmp) / "sample_grid.png"
            save_sample_npz(sample, sample_path)
            save_sample_grid(sample_path, output_path)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 1000)

    def test_cli_generates_npz_files_and_metadata_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable,
                "-m",
                "mnr_dataset.fvnb_main",
                "--num_prob",
                "2",
                "--output_dir",
                tmp,
                "--seed",
                "13",
                "--rule_schema",
                "diff_outer_inner_boundary",
                "--no_shuffle",
            ]
            result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            output = Path(tmp)
            generated = sorted(output.glob("*.npz"))
            self.assertEqual(len(generated), 2)
            loaded = np.load(generated[0], allow_pickle=False)
            self.assertEqual(loaded["context_images"].shape, (3, 128, 128))
            self.assertEqual(loaded["answer_set_images"].shape, (8, 128, 128))
            self.assertTrue((output / "metadata.jsonl").exists())
            rows = [json.loads(line) for line in (output / "metadata.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["rule"]["id"], "diff_outer_inner_boundary")
            report = json.loads((output / "generation_report.json").read_text())
            self.assertEqual(report["num_samples"], 2)
            self.assertEqual(report["rule_counts"]["diff_outer_inner_boundary"], 2)
            self.assertEqual(report["validity_pass"]["unique_answer"], 2)


class TestFVNBBaselines(unittest.TestCase):
    def test_fuzzy_oracle_and_hard_role_tie_expose_core_shortcut(self):
        generator = FVNBGenerator(FVNBConfig(seed=23, panel_size=80, shuffle_candidates=False))
        sample = generator.generate_sample("unit_000004", rule_id="diff_outer_inner_boundary")
        metadata = sample["metadata"]

        self.assertEqual(predict_fuzzy_symbolic(metadata), sample["correct_answer_image_index"])

        hard_ties = tied_best_candidates(metadata, "hard_truth_score")
        hard_tie_types = {metadata["candidates"][index]["negative_type"] for index in hard_ties}
        self.assertIn(None, hard_tie_types)
        self.assertIn("hard_role_invariant_fuzzy_flip", hard_tie_types)

        audit = audit_metadata_rows([metadata])
        self.assertEqual(audit["num_samples"], 1)
        self.assertEqual(audit["fuzzy_symbolic_accuracy"], 1.0)
        self.assertEqual(audit["validity_pass"]["negative_threshold"], 1)
        self.assertEqual(audit["negative_type_counts"]["hard_role_invariant_fuzzy_flip"], 1)
        self.assertIn("number_only_wrong_tie_rate", audit)
        self.assertIn("fixed_coordinate_wrong_tie_rate", audit)

    def test_shortcut_scores_expose_number_coordinate_and_membership_controls(self):
        generator = FVNBGenerator(FVNBConfig(seed=29, panel_size=80, shuffle_candidates=False))
        sample = generator.generate_sample("unit_000005", rule_id="diff_outer_inner_boundary")
        metadata = sample["metadata"]

        number_tie_types = {
            metadata["candidates"][index]["negative_type"]
            for index in tied_best_from_scores(score_number_only_candidates(metadata))
        }
        fixed_tie_types = {
            metadata["candidates"][index]["negative_type"]
            for index in tied_best_from_scores(score_fixed_coordinate_candidates(metadata))
        }
        shuffled_scores = score_membership_shuffled_candidates(metadata)

        self.assertIn("same_number_fuzzy_role_shift", number_tie_types)
        self.assertIn("same_coordinate_fuzzy_boundary_shift", fixed_tie_types)
        self.assertEqual(len(shuffled_scores), 8)
        self.assertLess(shuffled_scores[0], metadata["candidates"][0]["truth_score"])


if __name__ == "__main__":
    unittest.main()
