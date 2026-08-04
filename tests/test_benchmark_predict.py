import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark_predict import predict


class TestBenchmarkPredict(unittest.TestCase):
    def test_predict_writes_csv(self):
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
            output = root / "predictions.csv"
            predict(split, output, "first", 0)
            text = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(text[0], "sample_id,prediction")
            self.assertEqual(text[1], "sample.npz,0")


if __name__ == "__main__":
    unittest.main()
