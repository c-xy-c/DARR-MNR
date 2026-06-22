# -*- coding: utf-8 -*-
"""Post-hoc audits for generated VQ-Expr datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from mnr_dataset.vqexpr_generator import (
    _candidate_image_stats,
    _empty_candidate_only_heuristics,
    _empty_candidate_visual_audit,
    _finalize_candidate_only_heuristics,
    _finalize_candidate_visual_audit,
    _update_candidate_only_heuristics,
    _update_candidate_visual_audit,
)


def audit_dataset(
    dataset_dir: Path,
    output_path: Optional[Path] = None,
    learned_baseline_min_samples: int = 16,
    learned_baseline_seed: int = 0,
) -> Dict[str, object]:
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError("metadata.jsonl not found in {0}".format(dataset_dir))

    rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidate_visual_audit = _empty_candidate_visual_audit()
    candidate_only_heuristics = _empty_candidate_only_heuristics()
    metadata_oracle_correct = 0
    score_argmax_correct = 0
    learned_features = []
    learned_labels = []

    for row in rows:
        sample_id = str(row["sample_id"])
        npz_path = dataset_dir / "{0}.npz".format(sample_id)
        data = np.load(npz_path, allow_pickle=True)
        answer_images = data["answer_set_images"]
        candidates = row["candidates"]
        correct_index = int(row["correct_answer_image_index"])

        metadata_oracle_correct += int(bool(candidates[correct_index]["is_correct"]))
        scores = [float(candidate["score"]) for candidate in candidates]
        score_argmax_correct += int(int(np.argmax(scores)) == correct_index)
        _update_candidate_visual_audit(candidate_visual_audit, answer_images, candidates)
        _update_candidate_only_heuristics(candidate_only_heuristics, answer_images, correct_index)
        learned_features.append(_candidate_only_feature_matrix(answer_images))
        learned_labels.append(correct_index)

    num_samples = len(rows)
    report = {
        "num_samples": num_samples,
        "metadata_oracle_accuracy": _safe_accuracy(metadata_oracle_correct, num_samples),
        "score_argmax_accuracy": _safe_accuracy(score_argmax_correct, num_samples),
        "candidate_visual_stats": _finalize_candidate_visual_audit(candidate_visual_audit),
        "candidate_only_heuristics": _finalize_candidate_only_heuristics(candidate_only_heuristics),
        "learned_candidate_only_probe": _learned_candidate_only_probe(
            np.asarray(learned_features, dtype=np.float32),
            np.asarray(learned_labels, dtype=np.int64),
            min_samples=learned_baseline_min_samples,
            seed=learned_baseline_seed,
        ),
    }
    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


def _safe_accuracy(correct: int, total: int) -> float:
    if int(total) <= 0:
        return 0.0
    return round(int(correct) / float(total), 6)


def _candidate_only_feature_matrix(answer_images: np.ndarray) -> np.ndarray:
    rows = []
    for index in range(answer_images.shape[0]):
        stats = _candidate_image_stats(answer_images[index])
        slot = [1.0 if index == slot_index else 0.0 for slot_index in range(answer_images.shape[0])]
        rows.append(
            [
                float(stats["ink_fraction"]),
                float(stats["visual_mass"]),
                float(stats["bbox_fraction"]),
            ]
            + slot
            + [1.0]
        )
    return np.asarray(rows, dtype=np.float32)


def _learned_candidate_only_probe(
    features: np.ndarray,
    labels: np.ndarray,
    min_samples: int = 16,
    seed: int = 0,
) -> Dict[str, object]:
    num_samples = int(features.shape[0])
    if num_samples < int(min_samples):
        return {
            "status": "skipped",
            "reason": "need at least {0} samples".format(int(min_samples)),
            "num_samples": num_samples,
        }

    rng = np.random.default_rng(int(seed))
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    split = max(1, int(round(num_samples * 0.5)))
    split = min(split, num_samples - 1)
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_features = features[train_idx].copy()
    test_features = features[test_idx].copy()
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    mean = train_features.reshape(-1, train_features.shape[-1]).mean(axis=0)
    std = train_features.reshape(-1, train_features.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0
    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std

    weights = np.zeros(train_features.shape[-1], dtype=np.float32)
    learning_rate = 0.1
    reg = 0.01
    epochs = 300
    for _ in range(epochs):
        scores = np.einsum("nkd,d->nk", train_features, weights)
        probs = _softmax(scores)
        probs[np.arange(train_labels.shape[0]), train_labels] -= 1.0
        grad = np.einsum("nkd,nk->d", train_features, probs) / float(train_labels.shape[0])
        grad += reg * weights
        weights -= learning_rate * grad

    train_pred = _predict_candidate(train_features, weights)
    test_pred = _predict_candidate(test_features, weights)
    return {
        "status": "ok",
        "num_samples": num_samples,
        "train_samples": int(train_idx.shape[0]),
        "test_samples": int(test_idx.shape[0]),
        "train_accuracy": _safe_accuracy(int(np.sum(train_pred == train_labels)), int(train_labels.shape[0])),
        "test_accuracy": _safe_accuracy(int(np.sum(test_pred == test_labels)), int(test_labels.shape[0])),
        "chance_accuracy": 0.125,
        "features": ["ink_fraction", "visual_mass", "bbox_fraction", "slot_one_hot", "bias"],
        "note": "Linear candidate-only ranker over visual statistics and answer slot; it never sees context images or metadata rules.",
    }


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _predict_candidate(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = np.einsum("nkd,d->nk", features, weights)
    return np.argmax(scores, axis=1)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Audit a generated VQ-Expr dataset.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--learned_baseline_min_samples", type=int, default=16)
    parser.add_argument("--learned_baseline_seed", type=int, default=0)
    args = parser.parse_args(argv)
    report = audit_dataset(
        args.dataset_dir,
        output_path=args.output,
        learned_baseline_min_samples=args.learned_baseline_min_samples,
        learned_baseline_seed=args.learned_baseline_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
