# -*- coding: utf-8 -*-
"""Visual-rule graph compiler for VQ-Expr."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping


GATE_MORPHOLOGY = {
    "Add": "merge_funnel",
    "Sub": "cancellation_tray",
    "Mul": "repeater_loop",
    "Div": "partition_splitter",
}


def build_visual_rule_graph(
    answer_aot: Mapping[str, object],
    quantities: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    quantity_nodes = [
        {
            "id": qid,
            "kind": "quantity_object",
            "family": quantity["family"],
            "membership": 0.96,
        }
        for qid, quantity in sorted(quantities.items())
    ]
    gate_nodes = [
        {
            "id": node["id"],
            "kind": "operator_gate",
            "op": node["op"],
            "morphology": GATE_MORPHOLOGY[str(node["op"])],
            "membership": float(node.get("membership", 0.95)),
        }
        for node in answer_aot["nodes"]
        if node["type"] == "Gate"
    ]
    return {
        "family": "vqexpr_composition",
        "rule_family": answer_aot["rule_family"],
        "quantity_nodes": quantity_nodes,
        "gate_nodes": gate_nodes,
        "binding_edges": [dict(edge) for edge in answer_aot["binding_edges"]],
        "scope_edges": [dict(edge) for edge in answer_aot["scope_edges"]],
        "constraints": {
            "no_visible_digits": True,
            "no_visible_operator_symbols": True,
            "source": "visual rule graph compiled from Answer AoT",
        },
    }


def compile_visual_rules(
    primitives: Iterable[Dict[str, object]],
    family: str = "vqexpr_composition",
) -> Dict[str, object]:
    primitive_list = list(primitives)
    if family == "circuit_flow":
        return _compile_legacy_circuit_flow(primitive_list)
    quantities = sorted(
        [p for p in primitive_list if p.get("kind") == "quantity_carrier"],
        key=lambda item: str(item["id"]),
    )
    gates = sorted(
        [p for p in primitive_list if p.get("kind") == "operator_gate"],
        key=lambda item: str(item["id"]),
    )
    edges = [
        {
            "src": str(p["src"]),
            "dst": str(p["dst"]),
            "port": str(p["port"]),
            "membership": float(p.get("membership", p.get("confidence", 0.95))),
        }
        for p in primitive_list
        if p.get("kind") == "binding_edge"
    ]
    return {
        "family": family,
        "quantity_nodes": [{"id": str(q["id"]), "membership": float(q.get("membership", 0.96))} for q in quantities],
        "gate_nodes": [
            {
                "id": str(g["id"]),
                "op": str(g.get("op", "Add")),
                "morphology": str(g.get("morphology", "merge_funnel")),
                "membership": float(g.get("membership", g.get("confidence", 0.95))),
            }
            for g in gates
        ],
        "binding_edges": edges,
        "scope_edges": [],
        "constraints": {"no_visible_digits": True, "no_visible_operator_symbols": True},
    }


def _compile_legacy_circuit_flow(primitive_list: List[Dict[str, object]]) -> Dict[str, object]:
    quantities = sorted(
        [p for p in primitive_list if p.get("kind") == "quantity_carrier"],
        key=lambda item: str(item["id"]),
    )
    gates = sorted(
        [p for p in primitive_list if p.get("kind") == "operator_gate"],
        key=lambda item: str(item["id"]),
    )
    edges = [
        {
            "src": str(p["src"]),
            "dst": str(p["dst"]),
            "port": str(p["port"]),
            "confidence": float(p.get("confidence", 0.95)),
        }
        for p in primitive_list
        if p.get("kind") == "binding_edge"
    ]
    alternative_edges = [
        {
            "src": str(p["src"]),
            "dst": str(p["dst"]),
            "port": str(p["port"]),
            "confidence": float(p.get("confidence", 0.35)),
        }
        for p in primitive_list
        if p.get("kind") == "alternative_binding_edge"
    ]
    return {
        "family": "circuit_flow",
        "grouping": [
            {"rule": "common_region", "target": str(q["id"]), "confidence": float(q.get("confidence", 0.96))}
            for q in quantities
        ],
        "operator_gates": [
            {"id": str(g["id"]), "op": str(g["op"]), "morphology": str(g.get("morphology", "gate"))}
            for g in gates
        ],
        "binding_edges": edges,
        "alternative_binding_edges": alternative_edges,
        "execution_order": [["g_add"], ["g_mul", "g_sub"], ["g_div"]],
        "constraints": {
            "no_visible_digits": True,
            "no_visible_operator_symbols": True,
            "source": "visual primitives compiled from circuit-flow layout",
        },
    }
