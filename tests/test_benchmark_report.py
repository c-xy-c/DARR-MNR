import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark_report import build_report


class TestBenchmarkReport(unittest.TestCase):
    def test_report_contains_required_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "test_set"
            split.mkdir()
            np.savez(
                split / "sample.npz",
                context_images=np.zeros((3, 8, 8), dtype=np.uint8),
                answer_set_images=np.zeros((8, 8, 8), dtype=np.uint8),
                correct_answer_image_index=0,
            )
            report = build_report(split, None, "unit-test")
            self.assertEqual(report["protocol_version"], "darr_mnr_v1")
            self.assertEqual(report["num_samples"], 1)
            self.assertEqual(report["chance_accuracy"], 0.125)
            self.assertEqual(report["runner"], "unit-test")


if __name__ == "__main__":
    unittest.main()
