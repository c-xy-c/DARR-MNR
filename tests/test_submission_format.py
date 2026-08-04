import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark_submission import build_template


class TestSubmissionFormat(unittest.TestCase):
    def test_template_contains_all_sample_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "test_set"
            split.mkdir()
            for idx in range(2):
                np.savez(
                    split / f"sample_{idx}.npz",
                    context_images=np.zeros((3, 8, 8), dtype=np.uint8),
                    answer_set_images=np.zeros((8, 8, 8), dtype=np.uint8),
                    correct_answer_image_index=0,
                )
            output = root / "template.csv"
            build_template(split, output)
            text = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(text[0], "sample_id,prediction")
            self.assertIn("sample_0.npz,0", text)
            self.assertIn("sample_1.npz,0", text)


if __name__ == "__main__":
    unittest.main()
