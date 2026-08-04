import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark import evaluate


class TestBenchmarkContract(unittest.TestCase):
    def write_sample(self, directory: Path, name: str, label: int = 2) -> None:
        np.savez(
            directory / name,
            context_images=np.zeros((3, 16, 16), dtype=np.uint8),
            answer_set_images=np.zeros((8, 16, 16), dtype=np.uint8),
            correct_answer_image_index=label,
        )

    def test_dataset_validation_reports_chance_without_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.write_sample(directory, "sample.npz")
            result = evaluate(directory, None)
            self.assertEqual(result["num_samples"], 1)
            self.assertEqual(result["chance_accuracy"], 0.125)

    def test_prediction_evaluation_requires_exact_sample_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.write_sample(directory, "sample.npz", label=2)
            predictions = directory / "predictions.csv"
            with predictions.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_id", "prediction"])
                writer.writerow(["sample.npz", "2"])
            result = evaluate(directory, predictions)
            self.assertEqual(result["correct"], 1)
            self.assertEqual(result["accuracy"], 1.0)
            self.assertEqual(result["coverage"], 1.0)

    def test_invalid_candidate_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            np.savez(
                directory / "invalid.npz",
                context_images=np.zeros((3, 16, 16), dtype=np.uint8),
                answer_set_images=np.zeros((7, 16, 16), dtype=np.uint8),
                correct_answer_image_index=0,
            )
            with self.assertRaises(ValueError):
                evaluate(directory, None)


if __name__ == "__main__":
    unittest.main()
