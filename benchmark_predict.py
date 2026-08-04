"""Create benchmark predictions for a DARR-MNR split."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark import sample_files, validate_sample


def predict(split_dir: Path, output_csv: Path, strategy: str, seed: int) -> Path:
    files = sample_files(split_dir)
    if not files:
        raise ValueError(f"no .npz samples found in {split_dir}")
    rng = np.random.default_rng(seed)
    rows = []
    for path in files:
        validate_sample(path)
        if strategy == "random":
            prediction = int(rng.integers(0, 8))
        elif strategy == "first":
            prediction = 0
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        rows.append((path.name, prediction))
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "prediction"])
        writer.writerows(rows)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--strategy", choices=["random", "first"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = predict(args.split_dir, args.output, args.strategy, args.seed)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
