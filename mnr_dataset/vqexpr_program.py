# -*- coding: utf-8 -*-
"""Executable Answer AoT programs for VQ-Expr."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


RULE_FAMILIES = (
    "serial",
    "parallel",
    "nested",
    "inverse",
    "calibration",
)

OPS = ("Add", "Sub", "Mul", "Div")
VALUE_MIN = 1
VALUE_MAX = 9


def reverse_sample_leaf_values(rule_family: str, rng: random.Random) -> Dict[str, int]:
    """Sample leaf values whose final answer and intermediates stay in 1..9."""

    if rule_family == "serial":
        for _ in range(2000):
            q1 = rng.randint(1, 4)
            q2 = rng.randint(1, 4)
            q3 = rng.randint(1, 3)
            mid = (q1 + q2) * q3
            if VALUE_MIN + 1 <= mid <= VALUE_MAX:
                q4 = rng.randint(1, min(8, mid - 1))
                y = mid - q4
                if VALUE_MIN <= y <= VALUE_MAX:
                    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4}
    elif rule_family == "parallel":
        for _ in range(2000):
            q1 = rng.randint(1, 4)
            q2 = rng.randint(1, 4)
            q3 = rng.randint(2, 9)
            q4 = rng.randint(1, q3 - 1)
            y = (q1 + q2) + (q3 - q4)
            if VALUE_MIN <= y <= VALUE_MAX:
                return {"q1": q1, "q2": q2, "q3": q3, "q4": q4}
    elif rule_family == "nested":
        for _ in range(2000):
            q1 = rng.randint(4, 9)
            q2 = rng.randint(1, 4)
            q3 = rng.randint(1, 4)
            y = q1 - (q2 + q3)
            if VALUE_MIN <= y <= VALUE_MAX:
                return {"q1": q1, "q2": q2, "q3": q3}
    elif rule_family == "inverse":
        for _ in range(2000):
            q2 = rng.randint(1, 9)
            q1 = rng.randint(1, 9)
            if q1 * q2 <= VALUE_MAX:
                return {"q1": q1, "q2": q2, "q3": q2}
    elif rule_family == "calibration":
        for _ in range(2000):
            q1 = rng.randint(1, 8)
            q2 = rng.randint(1, 8)
            if q1 + q2 <= VALUE_MAX:
                return {"q1": q1, "q2": q2}
    raise ValueError("Could not sample values for rule family {0}".format(rule_family))


def build_answer_aot(
    rule_family: str,
    leaf_values: Mapping[str, int],
    quantity_families: Mapping[str, str],
) -> Dict[str, object]:
    if rule_family not in RULE_FAMILIES:
        raise ValueError("Unknown rule family: {0}".format(rule_family))

    nodes: List[Dict[str, object]] = []
    for qid in sorted(leaf_values):
        nodes.append(
            {
                "id": "d_{0}".format(qid),
                "type": "Decode",
                "quantity_id": qid,
                "quantity_family": quantity_families[qid],
            }
        )

    gates = _gate_nodes(rule_family)
    nodes.extend(gates)
    binding_edges = _binding_edges(rule_family)
    scope_edges = _scope_edges(rule_family)
    root = gates[-1]["id"]
    return {
        "rule_family": rule_family,
        "root": root,
        "nodes": nodes,
        "binding_edges": binding_edges,
        "scope_edges": scope_edges,
        "template": _template(rule_family),
        "value_range": "1..9",
    }


def evaluate_answer_aot(
    quantities: Mapping[str, Mapping[str, object]],
    answer_aot: Mapping[str, object],
) -> Dict[str, object]:
    node_outputs: Dict[str, int] = {}
    fuzzy_scores: Dict[str, float] = {}
    trace: List[Dict[str, object]] = []

    for node in answer_aot["nodes"]:
        node_id = str(node["id"])
        if node["type"] == "Decode":
            qid = str(node["quantity_id"])
            quantity = quantities[qid]
            value = int(quantity["numeric_value"])
            node_outputs[node_id] = value
            membership = quantity["fuzzy_value"]["membership"].get(str(value), 1.0)
            fuzzy_scores[node_id] = float(membership)
            trace.append({"node": node_id, "type": "Decode", "quantity_id": qid, "output": value})
        elif node["type"] == "Gate":
            left_id, right_id = node["inputs"]
            left = int(node_outputs[str(left_id)])
            right = int(node_outputs[str(right_id)])
            output = apply_op(str(node["op"]), left, right)
            if not VALUE_MIN <= output <= VALUE_MAX:
                raise ValueError("AoT output out of range: {0}".format(output))
            node_outputs[node_id] = output
            fuzzy_scores[node_id] = min(fuzzy_scores[str(left_id)], fuzzy_scores[str(right_id)], float(node.get("membership", 0.96)))
            trace.append(
                {
                    "node": node_id,
                    "type": "Gate",
                    "op": node["op"],
                    "inputs": [left, right],
                    "input_nodes": [left_id, right_id],
                    "output": output,
                }
            )
    root = str(answer_aot["root"])
    output_value = int(node_outputs[root])
    return {
        "output_value": output_value,
        "crisp_trace": trace,
        "fuzzy_output": _fuzzy_output(output_value),
        "program_score": round(float(fuzzy_scores.get(root, 1.0)), 4),
    }


def apply_op(op: str, left: int, right: int) -> int:
    if op == "Add":
        return left + right
    if op == "Sub":
        return left - right
    if op == "Mul":
        return left * right
    if op == "Div":
        if right == 0 or left % right != 0:
            raise ValueError("Division must be exact and non-zero")
        return left // right
    raise ValueError("Unknown op: {0}".format(op))


def candidate_score(
    expected_value: int,
    candidate_value: int,
    primary_mutation: Optional[str],
    base_membership: float = 0.94,
) -> float:
    if primary_mutation is None and int(candidate_value) == int(expected_value):
        return round(base_membership, 4)
    mutation_penalty = {
        "output_perturbation": 0.24,
        "leaf_decode_perturbation": 0.31,
        "calibration_swap": 0.28,
        "operator_swap": 0.27,
        "port_binding_swap": 0.26,
        "scope_tree_rotation": 0.30,
        "fuzzy_ambiguity_trap": 0.44,
    }.get(str(primary_mutation), 0.25)
    if int(candidate_value) == int(expected_value):
        return mutation_penalty
    distance = min(20, abs(int(candidate_value) - int(expected_value)))
    return round(max(0.05, mutation_penalty - 0.005 * distance), 4)


def _gate_nodes(rule_family: str) -> List[Dict[str, object]]:
    if rule_family == "serial":
        return [
            {"id": "g1", "type": "Gate", "op": "Add", "inputs": ["d_q1", "d_q2"], "membership": 0.97},
            {"id": "g2", "type": "Gate", "op": "Mul", "inputs": ["g1", "d_q3"], "membership": 0.96},
            {"id": "g3", "type": "Gate", "op": "Sub", "inputs": ["g2", "d_q4"], "membership": 0.96},
        ]
    if rule_family == "parallel":
        return [
            {"id": "g1", "type": "Gate", "op": "Add", "inputs": ["d_q1", "d_q2"], "membership": 0.97},
            {"id": "g2", "type": "Gate", "op": "Sub", "inputs": ["d_q3", "d_q4"], "membership": 0.96},
            {"id": "g3", "type": "Gate", "op": "Add", "inputs": ["g1", "g2"], "membership": 0.96},
        ]
    if rule_family == "nested":
        return [
            {"id": "g1", "type": "Gate", "op": "Add", "inputs": ["d_q2", "d_q3"], "membership": 0.97},
            {"id": "g2", "type": "Gate", "op": "Sub", "inputs": ["d_q1", "g1"], "membership": 0.95},
        ]
    if rule_family == "inverse":
        return [
            {"id": "g1", "type": "Gate", "op": "Mul", "inputs": ["d_q1", "d_q2"], "membership": 0.96},
            {"id": "g2", "type": "Gate", "op": "Div", "inputs": ["g1", "d_q3"], "membership": 0.95},
        ]
    if rule_family == "calibration":
        return [
            {"id": "g1", "type": "Gate", "op": "Add", "inputs": ["d_q1", "d_q2"], "membership": 0.96},
        ]
    raise ValueError("Unknown rule family: {0}".format(rule_family))


def _binding_edges(rule_family: str) -> List[Dict[str, object]]:
    edges = []
    for gate in _gate_nodes(rule_family):
        for port, src in zip(["left", "right"], gate["inputs"]):
            edges.append({"src": src, "dst": gate["id"], "port": port, "membership": 0.95})
    return edges


def _scope_edges(rule_family: str) -> List[Dict[str, object]]:
    if rule_family == "nested":
        return [
            {"scope": "inner_container", "contains": ["g1"], "membership": 0.95},
            {"scope": "outer_container", "contains": ["d_q1", "g1", "g2"], "membership": 0.94},
        ]
    if rule_family == "parallel":
        return [
            {"scope": "left_lane", "contains": ["d_q1", "d_q2", "g1"], "membership": 0.95},
            {"scope": "right_lane", "contains": ["d_q3", "d_q4", "g2"], "membership": 0.95},
        ]
    return [{"scope": "main_flow", "contains": [edge["dst"] for edge in _binding_edges(rule_family)], "membership": 0.94}]


def _template(rule_family: str) -> str:
    return {
        "serial": "Sub(Mul(Add(q1,q2),q3),q4)",
        "parallel": "Add(Add(q1,q2),Sub(q3,q4))",
        "nested": "Sub(q1,Add(q2,q3))",
        "inverse": "Div(Mul(q1,q2),q3)",
        "calibration": "Add(Decode_A(q1),Decode_B(q2))",
    }[rule_family]


def _fuzzy_output(value: int) -> Dict[str, object]:
    support = sorted({max(VALUE_MIN, value - 1), value, min(VALUE_MAX, value + 1)})
    membership = {str(v): (1.0 if v == value else 0.28) for v in support}
    return {"support": support, "membership": membership}


def evaluate_killer_expression(
    quantities: Mapping[str, Mapping[str, object]],
    visual_rule_program: Mapping[str, object],
) -> Dict[str, object]:
    """Backward-compatible evaluator for the old fixed probe."""

    values = {key: int(value["numeric_value"]) for key, value in quantities.items()}
    q1 = values.get("q1", 2)
    q2 = values.get("q2", 1)
    q3 = values.get("q3", 3)
    q4 = values.get("q4", 5)
    q5 = values.get("q5", 2)
    add = q1 + q2
    mul = add * q3
    sub = q4 - q5
    div = mul // sub
    return {
        "crisp_trace": [
            {"node": "g_add", "op": "Add", "inputs": [q1, q2], "output": add},
            {"node": "g_mul", "op": "Mul", "inputs": [add, q3], "output": mul},
            {"node": "g_sub", "op": "Sub", "inputs": [q4, q5], "output": sub},
            {"node": "g_div", "op": "Div", "inputs": [mul, sub], "output": div},
        ],
        "output_value": div,
        "fuzzy_output": _fuzzy_output(div),
    }
