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
VALUE_MAX = 5

EXPRESSION_SCHEMAS_BY_FAMILY: Dict[str, Dict[str, Dict[str, object]]] = {
    "serial": {
        "serial_add_mul_sub": {
            "roles": ["q1", "q2", "q3", "q4"],
            "template": "Sub(Mul(Add(q1,q2),q3),q4)",
            "gates": [
                {"id": "g1", "op": "Add", "inputs": ["d_q1", "d_q2"], "membership": 0.97},
                {"id": "g2", "op": "Mul", "inputs": ["g1", "d_q3"], "membership": 0.96},
                {"id": "g3", "op": "Sub", "inputs": ["g2", "d_q4"], "membership": 0.96},
            ],
        },
        "serial_sub_mul_add": {
            "roles": ["q1", "q2", "q3", "q4"],
            "template": "Add(Mul(Sub(q1,q2),q3),q4)",
            "gates": [
                {"id": "g1", "op": "Sub", "inputs": ["d_q1", "d_q2"], "membership": 0.97},
                {"id": "g2", "op": "Mul", "inputs": ["g1", "d_q3"], "membership": 0.96},
                {"id": "g3", "op": "Add", "inputs": ["g2", "d_q4"], "membership": 0.96},
            ],
        },
    },
    "parallel": {
        "parallel_sum_difference": {
            "roles": ["q1", "q2", "q3", "q4"],
            "template": "Add(Add(q1,q2),Sub(q3,q4))",
            "gates": [
                {"id": "g1", "op": "Add", "inputs": ["d_q1", "d_q2"], "membership": 0.97},
                {"id": "g2", "op": "Sub", "inputs": ["d_q3", "d_q4"], "membership": 0.96},
                {"id": "g3", "op": "Add", "inputs": ["g1", "g2"], "membership": 0.96},
            ],
        },
        "parallel_difference_sum": {
            "roles": ["q1", "q2", "q3", "q4"],
            "template": "Add(Sub(q1,q2),Add(q3,q4))",
            "gates": [
                {"id": "g1", "op": "Sub", "inputs": ["d_q1", "d_q2"], "membership": 0.97},
                {"id": "g2", "op": "Add", "inputs": ["d_q3", "d_q4"], "membership": 0.96},
                {"id": "g3", "op": "Add", "inputs": ["g1", "g2"], "membership": 0.96},
            ],
        },
    },
    "nested": {
        "nested_outer_minus_sum": {
            "roles": ["q1", "q2", "q3", "q4"],
            "template": "Sub(Add(q1,q2),Add(q3,q4))",
            "gates": [
                {"id": "g1", "op": "Add", "inputs": ["d_q1", "d_q2"], "membership": 0.97},
                {"id": "g2", "op": "Add", "inputs": ["d_q3", "d_q4"], "membership": 0.97},
                {"id": "g3", "op": "Sub", "inputs": ["g1", "g2"], "membership": 0.95},
            ],
        },
        "nested_pairwise_difference_sum": {
            "roles": ["q1", "q2", "q3", "q4"],
            "template": "Add(Sub(q1,q3),Sub(q2,q4))",
            "gates": [
                {"id": "g1", "op": "Sub", "inputs": ["d_q1", "d_q3"], "membership": 0.96},
                {"id": "g2", "op": "Sub", "inputs": ["d_q2", "d_q4"], "membership": 0.96},
                {"id": "g3", "op": "Add", "inputs": ["g1", "g2"], "membership": 0.95},
            ],
        },
    },
    "inverse": {
        "inverse_product_divisor": {
            "roles": ["q1", "q2", "q3", "q4"],
            "template": "Div(Mul(q1,q2),Add(q3,q4))",
            "gates": [
                {"id": "g1", "op": "Mul", "inputs": ["d_q1", "d_q2"], "membership": 0.96},
                {"id": "g2", "op": "Add", "inputs": ["d_q3", "d_q4"], "membership": 0.96},
                {"id": "g3", "op": "Div", "inputs": ["g1", "g2"], "membership": 0.95},
            ],
        },
        "inverse_sum_divisor": {
            "roles": ["q1", "q2", "q3", "q4"],
            "template": "Div(Add(q1,q2),Sub(q3,q4))",
            "gates": [
                {"id": "g1", "op": "Add", "inputs": ["d_q1", "d_q2"], "membership": 0.96},
                {"id": "g2", "op": "Sub", "inputs": ["d_q3", "d_q4"], "membership": 0.96},
                {"id": "g3", "op": "Div", "inputs": ["g1", "g2"], "membership": 0.95},
            ],
        },
    },
    "calibration": {
        "calibration_sum": {
            "roles": ["q1", "q2", "q3", "q4"],
            "template": "Add(Add(Decode_A(q1),Decode_B(q2)),Sub(Decode_A(q3),Decode_B(q4)))",
            "gates": [
                {"id": "g1", "op": "Add", "inputs": ["d_q1", "d_q2"], "membership": 0.96},
                {"id": "g2", "op": "Sub", "inputs": ["d_q3", "d_q4"], "membership": 0.96},
                {"id": "g3", "op": "Add", "inputs": ["g1", "g2"], "membership": 0.95},
            ],
        },
        "calibration_difference": {
            "roles": ["q1", "q2", "q3", "q4"],
            "template": "Add(Sub(Decode_A(q1),Decode_A(q3)),Sub(Decode_B(q2),Decode_B(q4)))",
            "gates": [
                {"id": "g1", "op": "Sub", "inputs": ["d_q1", "d_q3"], "membership": 0.96},
                {"id": "g2", "op": "Sub", "inputs": ["d_q2", "d_q4"], "membership": 0.96},
                {"id": "g3", "op": "Add", "inputs": ["g1", "g2"], "membership": 0.95},
            ],
        },
    },
}


def reverse_sample_leaf_values(
    rule_family: str,
    rng: random.Random,
    expression_schema: Optional[str] = None,
) -> Dict[str, int]:
    """Sample leaf values whose final answer and intermediates stay in 1..5."""

    _, schema = _resolve_expression_schema(rule_family, expression_schema, rng=rng)
    roles = [str(role) for role in schema["roles"]]
    for _ in range(8000):
        leaf_values = {role: rng.randint(VALUE_MIN, VALUE_MAX) for role in roles}
        try:
            _evaluate_schema_nodes(schema, leaf_values)
        except ValueError:
            continue
        return leaf_values
    raise ValueError("Could not sample values for rule family {0}".format(rule_family))


def build_answer_aot(
    rule_family: str,
    leaf_values: Mapping[str, int],
    quantity_families: Mapping[str, str],
    expression_schema: Optional[str] = None,
    boundary_binding: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    if rule_family not in RULE_FAMILIES:
        raise ValueError("Unknown rule family: {0}".format(rule_family))

    schema_id, schema = _resolve_expression_schema(rule_family, expression_schema)
    nodes: List[Dict[str, object]] = []
    for qid in schema["roles"]:
        nodes.append(
            {
                "id": "d_{0}".format(qid),
                "type": "Decode",
                "quantity_id": qid,
                "quantity_family": quantity_families[qid],
            }
        )

    gates = _gate_nodes(rule_family, schema_id)
    nodes.extend(gates)
    binding_edges = _binding_edges(gates)
    scope_edges = _scope_edges(rule_family, gates, boundary_binding)
    root = gates[-1]["id"]
    answer_aot = {
        "rule_family": rule_family,
        "expression_schema": schema_id,
        "root": root,
        "nodes": nodes,
        "binding_edges": binding_edges,
        "scope_edges": scope_edges,
        "template": str(schema["template"]),
        "value_range": "1..5",
    }
    if boundary_binding is not None:
        answer_aot["boundary_binding"] = dict(boundary_binding)
    return answer_aot


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


def _gate_nodes(rule_family: str, expression_schema: Optional[str] = None) -> List[Dict[str, object]]:
    _, schema = _resolve_expression_schema(rule_family, expression_schema)
    return [
        {
            "id": str(gate["id"]),
            "type": "Gate",
            "op": str(gate["op"]),
            "inputs": [str(value) for value in gate["inputs"]],
            "membership": float(gate.get("membership", 0.96)),
        }
        for gate in schema["gates"]
    ]


def _binding_edges(gates: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    edges = []
    for gate in gates:
        for port, src in zip(["left", "right"], gate["inputs"]):
            edges.append({"src": src, "dst": gate["id"], "port": port, "membership": 0.95})
    return edges


def _scope_edges(
    rule_family: str,
    gates: Sequence[Mapping[str, object]],
    boundary_binding: Optional[Mapping[str, object]] = None,
) -> List[Dict[str, object]]:
    gate_ids = [str(gate["id"]) for gate in gates]
    if rule_family == "nested":
        boundary_id = None if boundary_binding is None else boundary_binding.get("boundary_id")
        edges = [
            {"scope": "outer_pair", "contains": ["d_q1", "d_q2", gate_ids[0]], "membership": 0.95},
            {"scope": "inner_pair", "contains": ["d_q3", "d_q4", gate_ids[1]], "membership": 0.95},
            {"scope": "boundary_merge", "contains": gate_ids, "membership": 0.94},
        ]
        if boundary_id is not None:
            for edge in edges:
                edge["boundary_id"] = boundary_id
        return edges
    if rule_family == "inverse" and boundary_binding is not None:
        boundary_id = boundary_binding.get("boundary_id")
        return [
            {"scope": "outer_pair", "contains": ["d_q1", "d_q2", gate_ids[0]], "membership": 0.94, "boundary_id": boundary_id},
            {"scope": "inner_pair", "contains": ["d_q3", "d_q4", gate_ids[1]], "membership": 0.95, "boundary_id": boundary_id},
            {"scope": "boundary_merge", "contains": gate_ids, "membership": 0.94, "boundary_id": boundary_id},
        ]
    if rule_family == "parallel":
        return [
            {"scope": "left_lane", "contains": ["d_q1", "d_q2", gate_ids[0]], "membership": 0.95},
            {"scope": "right_lane", "contains": ["d_q3", "d_q4", gate_ids[1]], "membership": 0.95},
        ]
    return [{"scope": "main_flow", "contains": gate_ids, "membership": 0.94}]


def _template(rule_family: str, expression_schema: Optional[str] = None) -> str:
    _, schema = _resolve_expression_schema(rule_family, expression_schema)
    return str(schema["template"])


def _resolve_expression_schema(
    rule_family: str,
    expression_schema: Optional[str],
    rng: Optional[random.Random] = None,
) -> Tuple[str, Mapping[str, object]]:
    if rule_family not in EXPRESSION_SCHEMAS_BY_FAMILY:
        raise ValueError("Unknown rule family: {0}".format(rule_family))
    schemas = EXPRESSION_SCHEMAS_BY_FAMILY[rule_family]
    if expression_schema is None:
        schema_id = rng.choice(list(schemas.keys())) if rng is not None else next(iter(schemas.keys()))
    else:
        schema_id = str(expression_schema)
    if schema_id not in schemas:
        raise ValueError("Unknown expression schema for {0}: {1}".format(rule_family, schema_id))
    return schema_id, schemas[schema_id]


def _evaluate_schema_nodes(schema: Mapping[str, object], leaf_values: Mapping[str, int]) -> Dict[str, int]:
    node_outputs = {"d_{0}".format(role): int(leaf_values[str(role)]) for role in schema["roles"]}
    for gate in schema["gates"]:
        left_id, right_id = [str(value) for value in gate["inputs"]]
        left = int(node_outputs[left_id])
        right = int(node_outputs[right_id])
        output = apply_op(str(gate["op"]), left, right)
        if not VALUE_MIN <= output <= VALUE_MAX:
            raise ValueError("AoT output out of range: {0}".format(output))
        node_outputs[str(gate["id"])] = output
    return node_outputs


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
