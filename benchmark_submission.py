"""Create or validate a DARR-MNR benchmark submission file."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchmark import sample_files, validate_sample


def build_template(split_dir: Path, output_csv: Path) -> Path:
    files = sample_files(split_dir)
    if not files:
        raise ValueError(f"no .npz samples found in {split_dir}")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "prediction"])
        for path in files:
            validate_sample(path)
            writer.writerow([path.name, "0"])
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("submission_template.csv"))
    args = parser.parse_args()
    output = build_template(args.split_dir, args.output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
