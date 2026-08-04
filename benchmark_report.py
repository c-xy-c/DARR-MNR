"""Create a structured benchmark run record for DARR-MNR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import evaluate


def build_report(split_dir: Path, predictions: Path | None, runner: str) -> dict[str, object]:
    result = evaluate(split_dir, predictions)
    report = {
        "protocol_version": "darr_mnr_v1",
        "split_dir": str(split_dir),
        "prediction_file": str(predictions) if predictions is not None else None,
        "runner": runner,
    }
    report.update(result)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--runner", type=str, default="benchmark_report.py")
    parser.add_argument("--output", type=Path, default=Path("benchmark_report.json"))
    args = parser.parse_args()
    report = build_report(args.split_dir, args.predictions, args.runner)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
