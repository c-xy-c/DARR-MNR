# -*- coding: utf-8 -*-
"""Metadata-level baselines and audits for FVNB-MNR.

These helpers intentionally work on saved metadata rather than rendered images.
They are the cheap mechanism checks described in the dataset plan: fuzzy oracle
sanity, hard-role shortcut exposure, negative-type coverage, and validity rates.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import permutations
from typing import Dict, Iterable, List, Mapping

from mnr_dataset.fvnb_oracle import FuzzyRule, RULE_ROLES


def correct_index(metadata: Mapping[str, object]) -> int:
    for index, candidate in enumerate(metadata["candidates"]):
        if candidate.get("is_correct"):
            return index
    raise ValueError("FVNB metadata has no correct candidate")


def tied_best_candidates(metadata: Mapping[str, object], score_key: str, tolerance: float = 1e-9) -> List[int]:
    scores = [float(candidate.get(score_key, 0.0)) for candidate in metadata["candidates"]]
    return tied_best_from_scores(scores, tolerance=tolerance)


def tied_best_from_scores(scores: Iterable[float], tolerance: float = 1e-9) -> List[int]:
    scores = [float(score) for score in scores]
    best = max(scores)
    return [index for index, score in enumerate(scores) if abs(score - best) <= tolerance]


def predict_by_score(metadata: Mapping[str, object], score_key: str) -> int:
    return tied_best_candidates(metadata, score_key)[0]


def predict_fuzzy_symbolic(metadata: Mapping[str, object]) -> int:
    return predict_by_score(metadata, "truth_score")


def predict_hard_role(metadata: Mapping[str, object]) -> int:
    return predict_by_score(metadata, "hard_truth_score")


def score_number_only_candidates(metadata: Mapping[str, object]) -> List[float]:
    rule = _rule_from_metadata(metadata)
    scores: List[float] = []
    for candidate in metadata["candidates"]:
        values = [anchor["value"] for anchor in candidate["anchors"]]
        best = 0.0
        for ordered_values in permutations(values, len(RULE_ROLES)):
            role_values = dict(zip(RULE_ROLES, ordered_values))
            best = max(best, rule.arithmetic_truth(role_values))
        scores.append(best)
    return scores


def score_fixed_coordinate_candidates(metadata: Mapping[str, object]) -> List[float]:
    rule = _rule_from_metadata(metadata)
    scores: List[float] = []
    for candidate in metadata["candidates"]:
        role_values = {
            role: candidate["anchors"][index]["value"]
            for index, role in enumerate(RULE_ROLES)
        }
        scores.append(rule.arithmetic_truth(role_values))
    return scores


def score_membership_shuffled_candidates(metadata: Mapping[str, object]) -> List[float]:
    rule = _rule_from_metadata(metadata)
    scores: List[float] = []
    for candidate in metadata["candidates"]:
        shuffled = deepcopy(candidate)
        anchors = shuffled["anchors"]
        if anchors:
            memberships = [deepcopy(anchor["membership"]) for anchor in anchors]
            for index, anchor in enumerate(anchors):
                anchor["membership"] = memberships[(index + 1) % len(memberships)]
        scores.append(rule.evaluate(shuffled).score)
    return scores


def _predict_from_scores(scores: Iterable[float]) -> int:
    return tied_best_from_scores(scores)[0]


def _wrong_tie_rate(rows: List[Mapping[str, object]], score_rows: List[List[float]]) -> float:
    if not rows:
        return 0.0
    wrong_ties = 0
    for row, scores in zip(rows, score_rows):
        target = correct_index(row)
        ties = tied_best_from_scores(scores)
        wrong_ties += int(any(index != target for index in ties))
    return wrong_ties / len(rows)


def _rule_from_metadata(metadata: Mapping[str, object]) -> FuzzyRule:
    rule = metadata["rule"]
    return FuzzyRule(
        str(rule["id"]),
        t_norm=str(rule.get("t_norm", "product")),
        epsilon=float(rule.get("epsilon", 0.001)),
    )


def audit_metadata_rows(rows: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    rows = list(rows)
    rule_counts: Counter[str] = Counter()
    negative_type_counts: Counter[str] = Counter()
    validity_pass: Counter[str] = Counter()
    fuzzy_correct = 0
    hard_correct = 0
    number_correct = 0
    fixed_correct = 0
    shuffled_correct = 0
    hard_tie_with_wrong = 0
    number_score_rows: List[List[float]] = []
    fixed_score_rows: List[List[float]] = []
    shuffled_score_rows: List[List[float]] = []
    top_margins: List[float] = []
    max_negative_scores: List[float] = []

    for row in rows:
        rule_counts[str(row["rule"]["id"])] += 1
        target = correct_index(row)
        fuzzy_correct += int(predict_fuzzy_symbolic(row) == target)
        hard_correct += int(predict_hard_role(row) == target)
        hard_ties = tied_best_candidates(row, "hard_truth_score")
        hard_tie_with_wrong += int(any(index != target for index in hard_ties))
        number_scores = score_number_only_candidates(row)
        fixed_scores = score_fixed_coordinate_candidates(row)
        shuffled_scores = score_membership_shuffled_candidates(row)
        number_score_rows.append(number_scores)
        fixed_score_rows.append(fixed_scores)
        shuffled_score_rows.append(shuffled_scores)
        number_correct += int(_predict_from_scores(number_scores) == target)
        fixed_correct += int(_predict_from_scores(fixed_scores) == target)
        shuffled_correct += int(_predict_from_scores(shuffled_scores) == target)

        for candidate in row["candidates"]:
            negative_type = candidate.get("negative_type")
            if negative_type is not None:
                negative_type_counts[str(negative_type)] += 1

        validity = row.get("validity", {})
        for key, value in validity.items():
            if isinstance(value, bool) and value:
                validity_pass[str(key)] += 1
        if "top_margin" in validity:
            top_margins.append(float(validity["top_margin"]))
        if "max_negative_score" in validity:
            max_negative_scores.append(float(validity["max_negative_score"]))

    count = len(rows)
    return {
        "num_samples": count,
        "rule_counts": dict(sorted(rule_counts.items())),
        "negative_type_counts": dict(sorted(negative_type_counts.items())),
        "validity_pass": dict(sorted(validity_pass.items())),
        "fuzzy_symbolic_accuracy": fuzzy_correct / count if count else 0.0,
        "hard_role_accuracy": hard_correct / count if count else 0.0,
        "number_only_accuracy": number_correct / count if count else 0.0,
        "fixed_coordinate_accuracy": fixed_correct / count if count else 0.0,
        "membership_shuffled_accuracy": shuffled_correct / count if count else 0.0,
        "hard_role_wrong_tie_rate": hard_tie_with_wrong / count if count else 0.0,
        "number_only_wrong_tie_rate": _wrong_tie_rate(rows, number_score_rows),
        "fixed_coordinate_wrong_tie_rate": _wrong_tie_rate(rows, fixed_score_rows),
        "membership_shuffled_wrong_tie_rate": _wrong_tie_rate(rows, shuffled_score_rows),
        "min_top_margin": min(top_margins) if top_margins else None,
        "max_negative_score": max(max_negative_scores) if max_negative_scores else None,
    }
