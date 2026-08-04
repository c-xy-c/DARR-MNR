"""Generate simple baselines for the DARR-MNR benchmark."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark import sample_files, validate_sample


def write_predictions(split_dir: Path, output_csv: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    files = sample_files(split_dir)
    if not files:
        raise ValueError(f"no .npz samples found in {split_dir}")
    rows = []
    for path in files:
        validate_sample(path)
        rows.append((path.name, int(rng.integers(0, 8))))
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "prediction"])
        writer.writerows(rows)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("random_predictions.csv"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = write_predictions(args.split_dir, args.output, args.seed)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
