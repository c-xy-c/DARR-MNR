"""Run a benchmark evaluation and save a machine-readable report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import evaluate
from benchmark_baseline import write_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", type=Path, help="Official DARR-MNR split directory.")
    parser.add_argument("--predictions", type=Path, help="Optional prediction CSV to score.")
    parser.add_argument("--baseline", action="store_true", help="Generate random baseline predictions first.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random baseline.")
    parser.add_argument("--report", type=Path, default=Path("benchmark_report.json"), help="Output JSON report path.")
    args = parser.parse_args()

    prediction_file = args.predictions
    if args.baseline:
        prediction_file = write_predictions(args.split_dir, args.report.with_suffix(".csv"), args.seed)

    result = evaluate(args.split_dir, prediction_file)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()
