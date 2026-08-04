"""Validate and score the official DARR-MNR NPZ benchmark format."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

EXPECTED_CONTEXTS = 3
EXPECTED_CANDIDATES = 8


def sample_files(split_dir: Path) -> list[Path]:
    return sorted(split_dir.glob("*.npz"))


def validate_sample(path: Path) -> int:
    with np.load(path, allow_pickle=False) as sample:
        required = {"context_images", "answer_set_images", "correct_answer_image_index"}
        missing = required.difference(sample.files)
        if missing:
            raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
        contexts = sample["context_images"]
        candidates = sample["answer_set_images"]
        label = int(sample["correct_answer_image_index"])
    if contexts.shape[0] != EXPECTED_CONTEXTS:
        raise ValueError(f"{path.name}: expected 3 context panels, got {contexts.shape}")
    if candidates.shape[0] != EXPECTED_CANDIDATES:
        raise ValueError(f"{path.name}: expected 8 candidates, got {candidates.shape}")
    if not 0 <= label < EXPECTED_CANDIDATES:
        raise ValueError(f"{path.name}: label must be in [0, 7], got {label}")
    if not np.isfinite(contexts).all() or not np.isfinite(candidates).all():
        raise ValueError(f"{path.name}: images contain non-finite values")
    return label


def read_predictions(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames != ["sample_id", "prediction"]:
            raise ValueError("prediction CSV must have header: sample_id,prediction")
        predictions: dict[str, int] = {}
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id in predictions:
                raise ValueError(f"duplicate prediction for {sample_id}")
            prediction = int(row["prediction"])
            if not 0 <= prediction < EXPECTED_CANDIDATES:
                raise ValueError(f"prediction for {sample_id} must be in [0, 7]")
            predictions[sample_id] = prediction
    return predictions


def evaluate(split_dir: Path, prediction_file: Path | None) -> dict[str, float | int]:
    files = sample_files(split_dir)
    if not files:
        raise ValueError(f"no .npz samples found in {split_dir}")
    labels = {path.name: validate_sample(path) for path in files}
    if prediction_file is None:
        return {"num_samples": len(labels), "chance_accuracy": 0.125}
    predictions = read_predictions(prediction_file)
    if set(predictions) != set(labels):
        raise ValueError("prediction IDs must exactly match the evaluated split")
    correct = sum(predictions[name] == label for name, label in labels.items())
    return {"num_samples": len(labels), "correct": correct, "accuracy": correct / len(labels), "chance_accuracy": 0.125, "coverage": 1.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    for key, value in evaluate(args.split_dir, args.predictions).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
