import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark_run import main


class TestBenchmarkRun(unittest.TestCase):
    def test_baseline_run_writes_prediction_and_report(self):
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
            report = root / "report.json"
            import sys
            old_argv = sys.argv
            try:
                sys.argv = ["benchmark_run.py", str(split), "--baseline", "--report", str(report)]
                main()
            finally:
                sys.argv = old_argv
            self.assertTrue(report.exists())
            self.assertTrue(report.with_suffix(".csv").exists())
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(data["num_samples"], 1)
            self.assertEqual(data["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
