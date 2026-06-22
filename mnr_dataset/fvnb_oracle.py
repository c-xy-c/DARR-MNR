# -*- coding: utf-8 -*-
"""Fuzzy rule semantics and symbolic oracles for FVNB-MNR."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Iterable, List, Mapping


RULE_ROLES = ("outer", "inner", "boundary")


@dataclass(frozen=True)
class RuleEvaluation:
    score: float
    assignment: Dict[str, str]
    arithmetic_truth: float
    membership_truths: Dict[str, float]


def tau_equal(left: float, right: float, epsilon: float = 0.001) -> float:
    diff = abs(float(left) - float(right))
    if epsilon <= 0:
        return 1.0 if diff == 0 else 0.0
    return max(0.0, 1.0 - diff / epsilon)


def apply_t_norm(values: Iterable[float], t_norm: str = "product") -> float:
    values = [max(0.0, min(1.0, float(value))) for value in values]
    if not values:
        return 0.0
    if t_norm == "product":
        result = 1.0
        for value in values:
            result *= value
        return result
    if t_norm in ("minimum", "min", "godel"):
        return min(values)
    if t_norm == "lukasiewicz":
        return max(0.0, sum(values) - len(values) + 1.0)
    raise ValueError("Unsupported t-norm: {0}".format(t_norm))


class FuzzyRule:
    """Role-membership-aware arithmetic rule."""

    def __init__(self, rule_id: str, t_norm: str = "product", epsilon: float = 0.001):
        if rule_id not in RULE_DEFINITIONS:
            raise ValueError("Unknown FVNB rule: {0}".format(rule_id))
        self.rule_id = rule_id
        self.t_norm = t_norm
        self.epsilon = epsilon
        self.definition = RULE_DEFINITIONS[rule_id]

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.rule_id,
            "description": self.definition["description"],
            "expression": self.definition["expression"],
            "roles": list(RULE_ROLES),
            "t_norm": self.t_norm,
            "epsilon": self.epsilon,
        }

    def arithmetic_truth(self, values: Mapping[str, float]) -> float:
        outer = values["outer"]
        inner = values["inner"]
        boundary = values["boundary"]
        if self.rule_id == "diff_outer_inner_boundary":
            return tau_equal(outer - inner, boundary, self.epsilon)
        if self.rule_id == "sum_outer_inner_boundary":
            return tau_equal(outer, inner + boundary, self.epsilon)
        if self.rule_id == "ratio_outer_inner_boundary":
            if inner == 0:
                return 0.0
            return tau_equal(outer / float(inner), boundary, self.epsilon)
        raise ValueError("Unknown FVNB rule: {0}".format(self.rule_id))

    def evaluate(self, panel: Mapping[str, object]) -> RuleEvaluation:
        anchors = list(panel["anchors"])
        best = RuleEvaluation(score=-1.0, assignment={}, arithmetic_truth=0.0, membership_truths={})
        for ordered_anchors in permutations(anchors, len(RULE_ROLES)):
            role_to_anchor = dict(zip(RULE_ROLES, ordered_anchors))
            values = {role: role_to_anchor[role]["value"] for role in RULE_ROLES}
            arithmetic = self.arithmetic_truth(values)
            membership_truths = {
                role: _membership_values(role_to_anchor[role]).get(role, 0.0)
                for role in RULE_ROLES
            }
            score = apply_t_norm(list(membership_truths.values()) + [arithmetic], self.t_norm)
            if score > best.score:
                best = RuleEvaluation(
                    score=score,
                    assignment={role: role_to_anchor[role]["id"] for role in RULE_ROLES},
                    arithmetic_truth=arithmetic,
                    membership_truths=membership_truths,
                )
        return best


RULE_DEFINITIONS = {
    "diff_outer_inner_boundary": {
        "description": "role(a, outer) AND role(b, inner) AND role(c, boundary) AND near_eq(a - b, c)",
        "expression": ["=", ["-", "outer", "inner"], "boundary"],
    },
    "sum_outer_inner_boundary": {
        "description": "role(a, outer) AND role(b, inner) AND role(c, boundary) AND near_eq(a, b + c)",
        "expression": ["=", "outer", ["+", "inner", "boundary"]],
    },
    "ratio_outer_inner_boundary": {
        "description": "role(a, outer) AND role(b, inner) AND role(c, boundary) AND near_eq(a / b, c)",
        "expression": ["=", ["/", "outer", "inner"], "boundary"],
    },
}


def _membership_values(anchor: Mapping[str, object]) -> Dict[str, float]:
    membership = anchor["membership"]
    if isinstance(membership, Mapping) and "normalized" in membership:
        membership = membership["normalized"]
    return {role: float(membership.get(role, 0.0)) for role in RULE_ROLES}


def harden_panel_memberships(panel: Mapping[str, object], roles: Iterable[str] = RULE_ROLES) -> Dict[str, object]:
    hardened = deepcopy(dict(panel))
    hardened_anchors: List[Dict[str, object]] = []
    for anchor in panel["anchors"]:
        copied = deepcopy(anchor)
        memberships = _membership_values(copied)
        hard_role = max(roles, key=lambda role: memberships.get(role, 0.0))
        hardened_membership = {role: 1.0 if role == hard_role else 0.0 for role in roles}
        if isinstance(copied["membership"], Mapping) and "normalized" in copied["membership"]:
            copied["membership"]["normalized"] = hardened_membership
        else:
            copied["membership"] = hardened_membership
        copied["hard_role"] = hard_role
        hardened_anchors.append(copied)
    hardened["anchors"] = hardened_anchors
    return hardened
