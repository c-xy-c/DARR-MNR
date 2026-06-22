# AFVNB-MNR Implementation Plan

> For agentic workers: this is a TDD implementation plan for the AFVNB-MNR benchmark redesign. Implement task by task, and do not collapse the object-attribute ontology back into the FVNB v0 anchor schema.

**Goal:** Build the first executable AFVNB-MNR probe where numbers are `numeric_value` attributes of value-token objects, fuzzy visual predicates are object predicates, and labels are produced by fuzzy first-order programs plus deterministic counterfactual verification.

**Architecture:** Keep existing `fvnb_*` modules as v0 proof-of-concept. Add separate `afvnb_*` modules so the old anchor-centric implementation remains available for ablation and comparison. The new implementation must make the data contract explicit: `AFVNBScene -> FuzzyProgram.evaluate(scene) -> candidate scores -> verifier -> metadata`.

**Tech Stack:** Python standard library dataclasses, `numpy`, `Pillow`, existing `unittest` test style, existing `mnr_dataset` package layout.

**Evidence Protocol:** After implementation, evaluate the generated probe using [`doc/afvnb_experiment_protocol.md`](./afvnb_experiment_protocol.md). The implementation is not paper-ready merely because tests pass; it must also satisfy oracle recomputation, candidate-bias audit, hard-program falsifier, visual-dependency variants, and human sanity gates.

**Metadata Contract:** Implement all emitted sample metadata against [`doc/afvnb_metadata_schema.md`](./afvnb_metadata_schema.md), and do not prepare a public release until [`doc/afvnb_dataset_card_and_release_checklist.md`](./afvnb_dataset_card_and_release_checklist.md) passes.

---

## 0. Non-Negotiable Invariants

These invariants are the line between FVNB v0 and AFVNB:

| Invariant | Required behavior | Failure sign |
| --- | --- | --- |
| Numeric attribute ontology | Program evaluation reads `object.numeric_value` only. | Any new rule code reads `anchor["value"]`. |
| Object predicates | Role evidence is stored as `object.fuzzy_predicates["Outer"].normalized`, not as external role labels. | New metadata keeps only `anchors` or `intended_role`. |
| Executable program | Every label can be recomputed from scene graph + program JSON. | Metadata contains score fields that cannot be recomputed. |
| Counterfactual isolation | Every negative records a single main intervention and a verifier result. | Negative type is a string with no operator log. |
| Presentation discipline | Evaluation images do not display scores, negative type, or program labels. | Visualization leaks answer or diagnostic metadata. |
| v0 preservation | Existing `fvnb_*` tests continue to pass. | AFVNB change breaks old proof-of-concept. |

## 1. File Map

Create these files:

```text
mnr_dataset/afvnb_scene.py
mnr_dataset/afvnb_program.py
mnr_dataset/afvnb_generator.py
mnr_dataset/afvnb_verifier.py
mnr_dataset/afvnb_baselines.py
mnr_dataset/afvnb_visualize.py
mnr_dataset/afvnb_main.py
tests/test_afvnb_mnr.py
```

Modify these files:

```text
README.md
```

Do not modify these files except for compatibility imports or shared utility reuse:

```text
mnr_dataset/fvnb_core.py
mnr_dataset/fvnb_oracle.py
mnr_dataset/fvnb_generator.py
tests/test_fvnb_mnr.py
```

Reason: FVNB v0 is a useful ablation. AFVNB should supersede it conceptually, not erase it.

## 2. Task 1: Scene Graph Objects

**Files:**

```text
Create: mnr_dataset/afvnb_scene.py
Test: tests/test_afvnb_mnr.py
```

### Step 1: Write failing tests for object attributes

Add tests that prove `numeric_value` belongs to an object, not to an anchor.

```python
import unittest

from mnr_dataset.afvnb_scene import (
    AFVNBObject,
    AFVNBPredicateValue,
    AFVNBRegion,
    AFVNBScene,
)


class TestAFVNBScene(unittest.TestCase):
    def test_value_token_carries_numeric_value_and_predicates(self):
        obj = AFVNBObject(
            object_id="o_A",
            object_class="value_token",
            numeric_value=8,
            geometry={"center": [24, 64], "radius": 8},
            visual_attributes={"shape": "circle", "text": "8"},
            fuzzy_predicates={
                "Outer": AFVNBPredicateValue(raw=0.92, normalized=0.94),
                "Inner": AFVNBPredicateValue(raw=0.02, normalized=0.02),
                "Boundary": AFVNBPredicateValue(raw=0.04, normalized=0.04),
            },
        )

        self.assertEqual(obj.numeric_value, 8)
        self.assertEqual(obj.predicate("Outer"), 0.94)
        self.assertEqual(obj.value(), 8)
        self.assertNotIn("value", obj.to_dict())
        self.assertEqual(obj.to_dict()["numeric_value"], 8)

    def test_scene_filters_value_token_objects(self):
        scene = AFVNBScene(
            scene_id="scene_unit",
            objects=[
                AFVNBObject(
                    object_id="o_A",
                    object_class="value_token",
                    numeric_value=8,
                    geometry={"center": [24, 64]},
                    visual_attributes={"shape": "circle", "text": "8"},
                    fuzzy_predicates={"Outer": AFVNBPredicateValue(raw=1.0, normalized=1.0)},
                ),
                AFVNBObject(
                    object_id="r_marker",
                    object_class="region_marker",
                    numeric_value=None,
                    geometry={"center": [0, 0]},
                    visual_attributes={"shape": "rect"},
                    fuzzy_predicates={},
                ),
            ],
            regions=[AFVNBRegion(region_id="r_outer", region_class="container", geometry={"type": "rect"})],
            relations=[],
        )

        self.assertEqual([obj.object_id for obj in scene.value_tokens()], ["o_A"])
        self.assertEqual(scene.object_by_id("o_A").numeric_value, 8)
```

Run:

```bash
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_afvnb_mnr.TestAFVNBScene -v
```

Expected before implementation:

```text
ModuleNotFoundError: No module named 'mnr_dataset.afvnb_scene'
```

### Step 2: Implement scene dataclasses

Create `mnr_dataset/afvnb_scene.py` with:

```python
# -*- coding: utf-8 -*-
"""Object-level scene graph for AFVNB-MNR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional


@dataclass(frozen=True)
class AFVNBPredicateValue:
    raw: float
    normalized: float

    def to_dict(self) -> Dict[str, float]:
        return {"raw": float(self.raw), "normalized": float(self.normalized)}


@dataclass(frozen=True)
class AFVNBObject:
    object_id: str
    object_class: str
    numeric_value: Optional[int]
    geometry: Mapping[str, object]
    visual_attributes: Mapping[str, object]
    fuzzy_predicates: Mapping[str, AFVNBPredicateValue]

    def value(self) -> int:
        if self.numeric_value is None:
            raise ValueError("Object {0} has no numeric_value".format(self.object_id))
        return int(self.numeric_value)

    def predicate(self, name: str, use_raw: bool = False) -> float:
        predicate = self.fuzzy_predicates.get(name)
        if predicate is None:
            return 0.0
        return float(predicate.raw if use_raw else predicate.normalized)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "id": self.object_id,
            "class": self.object_class,
            "numeric_value": self.numeric_value,
            "geometry": dict(self.geometry),
            "visual_attributes": dict(self.visual_attributes),
            "fuzzy_predicates": {
                key: value.to_dict() for key, value in self.fuzzy_predicates.items()
            },
        }
        return payload


@dataclass(frozen=True)
class AFVNBRegion:
    region_id: str
    region_class: str
    geometry: Mapping[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.region_id,
            "class": self.region_class,
            "geometry": dict(self.geometry),
        }


@dataclass(frozen=True)
class AFVNBScene:
    scene_id: str
    objects: List[AFVNBObject]
    regions: List[AFVNBRegion]
    relations: List[Mapping[str, object]]

    def value_tokens(self) -> List[AFVNBObject]:
        return [obj for obj in self.objects if obj.object_class == "value_token"]

    def object_by_id(self, object_id: str) -> AFVNBObject:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        raise KeyError("Unknown object id: {0}".format(object_id))

    def to_dict(self) -> Dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "objects": [obj.to_dict() for obj in self.objects],
            "regions": [region.to_dict() for region in self.regions],
            "relations": [dict(relation) for relation in self.relations],
        }
```

### Step 3: Verify

Run:

```bash
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_afvnb_mnr.TestAFVNBScene -v
```

Expected after implementation:

```text
OK
```

## 3. Task 2: Fuzzy First-Order Program

**Files:**

```text
Create: mnr_dataset/afvnb_program.py
Modify: tests/test_afvnb_mnr.py
```

### Step 1: Write failing tests for program evaluation

Add:

```python
from mnr_dataset.afvnb_program import FuzzyProgram, make_program


class TestAFVNBProgram(unittest.TestCase):
    def _scene(self, outer_score=0.96, inner_score=0.94, boundary_score=0.91):
        return AFVNBScene(
            scene_id="program_scene",
            objects=[
                AFVNBObject(
                    object_id="o_A",
                    object_class="value_token",
                    numeric_value=8,
                    geometry={"center": [20, 40]},
                    visual_attributes={"text": "8"},
                    fuzzy_predicates={"Outer": AFVNBPredicateValue(raw=outer_score, normalized=outer_score)},
                ),
                AFVNBObject(
                    object_id="o_B",
                    object_class="value_token",
                    numeric_value=3,
                    geometry={"center": [45, 40]},
                    visual_attributes={"text": "3"},
                    fuzzy_predicates={"Inner": AFVNBPredicateValue(raw=inner_score, normalized=inner_score)},
                ),
                AFVNBObject(
                    object_id="o_C",
                    object_class="value_token",
                    numeric_value=5,
                    geometry={"center": [30, 40]},
                    visual_attributes={"text": "5"},
                    fuzzy_predicates={"Boundary": AFVNBPredicateValue(raw=boundary_score, normalized=boundary_score)},
                ),
            ],
            regions=[],
            relations=[],
        )

    def test_diff_program_evaluates_object_numeric_attributes(self):
        program = make_program("diff_outer_inner_boundary", t_norm="product", epsilon=0.001)
        evaluation = program.evaluate(self._scene())

        self.assertAlmostEqual(evaluation.score, 0.96 * 0.94 * 0.91, places=6)
        self.assertEqual(evaluation.assignment, {"x": "o_A", "y": "o_B", "z": "o_C"})
        self.assertEqual(evaluation.atom_truths["Equal(Value(x)-Value(y),Value(z))"], 1.0)

    def test_program_truth_changes_when_hard_roles_remain_but_membership_weakens(self):
        program = make_program("diff_outer_inner_boundary", t_norm="product", epsilon=0.001)

        high = program.evaluate(self._scene(0.96, 0.94, 0.91)).score
        low = program.evaluate(self._scene(0.54, 0.55, 0.52)).score

        self.assertGreater(high, 0.80)
        self.assertLess(low, 0.20)
```

Run:

```bash
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_afvnb_mnr.TestAFVNBProgram -v
```

Expected before implementation:

```text
ModuleNotFoundError: No module named 'mnr_dataset.afvnb_program'
```

### Step 2: Implement program primitives

Create `mnr_dataset/afvnb_program.py` with:

```python
# -*- coding: utf-8 -*-
"""Executable fuzzy first-order programs for AFVNB-MNR."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Iterable, List

from mnr_dataset.afvnb_scene import AFVNBObject, AFVNBScene


@dataclass(frozen=True)
class AFVNBProgramEvaluation:
    score: float
    assignment: Dict[str, str]
    atom_truths: Dict[str, float]


def tau_equal(left: float, right: float, epsilon: float = 0.001) -> float:
    diff = abs(float(left) - float(right))
    if epsilon <= 0:
        return 1.0 if diff == 0 else 0.0
    return max(0.0, 1.0 - diff / epsilon)


def apply_t_norm(values: Iterable[float], t_norm: str = "product") -> float:
    bounded = [max(0.0, min(1.0, float(value))) for value in values]
    if not bounded:
        return 0.0
    if t_norm == "product":
        result = 1.0
        for value in bounded:
            result *= value
        return result
    if t_norm in ("minimum", "min", "godel"):
        return min(bounded)
    if t_norm == "lukasiewicz":
        return max(0.0, sum(bounded) - len(bounded) + 1.0)
    raise ValueError("Unsupported t-norm: {0}".format(t_norm))


@dataclass(frozen=True)
class FuzzyProgram:
    program_id: str
    t_norm: str = "product"
    epsilon: float = 0.001

    def evaluate(self, scene: AFVNBScene) -> AFVNBProgramEvaluation:
        objects = scene.value_tokens()
        if len(objects) < 3:
            return AFVNBProgramEvaluation(score=0.0, assignment={}, atom_truths={})

        best = AFVNBProgramEvaluation(score=-1.0, assignment={}, atom_truths={})
        for x, y, z in permutations(objects, 3):
            atom_truths = self._atom_truths(x, y, z)
            score = apply_t_norm(atom_truths.values(), self.t_norm)
            if score > best.score:
                best = AFVNBProgramEvaluation(
                    score=score,
                    assignment={"x": x.object_id, "y": y.object_id, "z": z.object_id},
                    atom_truths=atom_truths,
                )
        return best

    def _atom_truths(self, x: AFVNBObject, y: AFVNBObject, z: AFVNBObject) -> Dict[str, float]:
        if self.program_id == "diff_outer_inner_boundary":
            arithmetic = tau_equal(x.value() - y.value(), z.value(), self.epsilon)
            arithmetic_name = "Equal(Value(x)-Value(y),Value(z))"
        elif self.program_id == "sum_outer_inner_boundary":
            arithmetic = tau_equal(x.value(), y.value() + z.value(), self.epsilon)
            arithmetic_name = "Equal(Value(x),Value(y)+Value(z))"
        elif self.program_id == "ratio_outer_inner_boundary":
            arithmetic = 0.0 if y.value() == 0 else tau_equal(x.value() / float(y.value()), z.value(), self.epsilon)
            arithmetic_name = "Equal(Value(x)/Value(y),Value(z))"
        else:
            raise ValueError("Unknown AFVNB program: {0}".format(self.program_id))

        return {
            "Outer(x)": x.predicate("Outer"),
            "Inner(y)": y.predicate("Inner"),
            "Boundary(z)": z.predicate("Boundary"),
            arithmetic_name: arithmetic,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "program_id": self.program_id,
            "variables": [
                {"name": "x", "domain": "value_token"},
                {"name": "y", "domain": "value_token"},
                {"name": "z", "domain": "value_token"},
            ],
            "aggregation": {
                "exists": "max",
                "and": self.t_norm,
                "epsilon": self.epsilon,
            },
        }


def make_program(program_id: str, t_norm: str = "product", epsilon: float = 0.001) -> FuzzyProgram:
    return FuzzyProgram(program_id=program_id, t_norm=t_norm, epsilon=epsilon)
```

### Step 3: Verify

Run:

```bash
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_afvnb_mnr.TestAFVNBProgram -v
```

Expected:

```text
OK
```

## 4. Task 3: Containment-Band Scene Builder

**Files:**

```text
Create: mnr_dataset/afvnb_generator.py
Modify: tests/test_afvnb_mnr.py
```

### Step 1: Write failing tests for generated object scene

Add:

```python
from mnr_dataset.afvnb_generator import AFVNBConfig, AFVNBGenerator


class TestAFVNBGenerator(unittest.TestCase):
    def test_generator_outputs_context_candidates_and_object_metadata(self):
        generator = AFVNBGenerator(AFVNBConfig(seed=7, panel_size=96, shuffle_candidates=False))
        sample = generator.generate_sample("afvnb_unit", program_id="diff_outer_inner_boundary")

        self.assertEqual(sample["context_images"].shape, (3, 96, 96))
        self.assertEqual(sample["answer_set_images"].shape, (8, 96, 96))
        self.assertEqual(sample["correct_answer_image_index"], 0)

        metadata = sample["metadata"]
        self.assertEqual(metadata["schema_version"], "afvnb_v1")
        self.assertEqual(metadata["program"]["program_id"], "diff_outer_inner_boundary")
        first_object = metadata["candidates"][0]["scene_graph"]["objects"][0]
        self.assertIn("numeric_value", first_object)
        self.assertIn("fuzzy_predicates", first_object)
        self.assertNotIn("anchors", metadata["candidates"][0])
```

Run:

```bash
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_afvnb_mnr.TestAFVNBGenerator -v
```

Expected before implementation:

```text
ModuleNotFoundError: No module named 'mnr_dataset.afvnb_generator'
```

### Step 2: Implement minimal generator

Implementation requirements:

```text
Use existing Rect and containment_membership from fvnb_core.
Convert each sampled point into AFVNBObject.
Render numeric_value as text inside the value-token object.
Return numpy image arrays plus metadata.
Do not expose negative type or truth score on evaluation images.
```

Minimal public API:

```python
@dataclass
class AFVNBConfig:
    panel_size: int = 128
    seed: int = 0
    theta_pos: float = 0.75
    theta_neg: float = 0.45
    delta: float = 0.20
    tau: float = 2.0
    sigma: float = 1.8
    boundary_weight: float = 5.0
    t_norm: str = "product"
    epsilon: float = 0.001
    shuffle_candidates: bool = True


class AFVNBGenerator:
    def generate_sample(self, sample_id: str, program_id: str) -> Dict[str, object]:
        ...
```

The first implementation may port geometry positions from `FVNBGenerator`, but the metadata must be object-centric:

```json
{
  "candidate_id": "candidate_correct",
  "counterfactual_operator": "correct",
  "scene_graph": {
    "objects": [
      {
        "id": "o_outer",
        "class": "value_token",
        "numeric_value": 8,
        "fuzzy_predicates": {
          "Outer": {"raw": 0.92, "normalized": 0.94}
        }
      }
    ]
  }
}
```

### Step 3: Verify old and new tests

Run:

```bash
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_afvnb_mnr.TestAFVNBGenerator -v
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_fvnb_mnr -v
```

Expected:

```text
OK
OK
```

## 5. Task 4: Counterfactual Operators and Verifier

**Files:**

```text
Create: mnr_dataset/afvnb_verifier.py
Modify: mnr_dataset/afvnb_generator.py
Modify: tests/test_afvnb_mnr.py
```

### Step 1: Write failing tests for operator logs

Add:

```python
from mnr_dataset.afvnb_verifier import verify_sample


class TestAFVNBCounterfactualVerifier(unittest.TestCase):
    def test_every_negative_has_operator_log_and_single_primary_change(self):
        generator = AFVNBGenerator(AFVNBConfig(seed=11, panel_size=96, shuffle_candidates=False))
        sample = generator.generate_sample("operator_unit", program_id="diff_outer_inner_boundary")
        report = verify_sample(sample["metadata"])

        self.assertTrue(report["unique_answer"])
        self.assertTrue(report["positive_threshold"])
        self.assertTrue(report["negative_threshold"])
        self.assertEqual(report["num_candidates"], 8)

        for candidate in sample["metadata"]["candidates"]:
            self.assertIn("counterfactual_operator", candidate)
            self.assertIn("operator_log", candidate)

        negatives = sample["metadata"]["candidates"][1:]
        self.assertTrue(all(item["operator_log"]["single_primary_intervention"] for item in negatives))
```

Expected before implementation:

```text
ModuleNotFoundError: No module named 'mnr_dataset.afvnb_verifier'
```

### Step 2: Implement verifier report

Create `mnr_dataset/afvnb_verifier.py`:

```python
# -*- coding: utf-8 -*-
"""Deterministic verification for AFVNB-MNR samples."""

from __future__ import annotations

from typing import Dict, Mapping


def verify_sample(metadata: Mapping[str, object]) -> Dict[str, object]:
    candidates = list(metadata["candidates"])
    scores = [float(candidate["truth_score"]) for candidate in candidates]
    correct_index = int(metadata["correct_answer_index"])
    thresholds = metadata["thresholds"]
    correct_score = scores[correct_index]
    max_negative = max(score for index, score in enumerate(scores) if index != correct_index)

    operator_logs_ok = all(
        "operator_log" in candidate and "single_primary_intervention" in candidate["operator_log"]
        for candidate in candidates
    )
    return {
        "num_candidates": len(candidates),
        "unique_answer": correct_score - max_negative >= float(thresholds["delta"]),
        "positive_threshold": correct_score >= float(thresholds["theta_pos"]),
        "negative_threshold": max_negative <= float(thresholds["theta_neg"]),
        "operator_logs_ok": operator_logs_ok,
        "max_negative_score": max_negative,
        "top_margin": correct_score - max_negative,
    }
```

### Step 3: Required counterfactual operators

Generator must produce these 8 slots when `shuffle_candidates=False`:

```text
0 correct
1 attribute_only_perturbation
2 predicate_only_perturbation
3 attribute_predicate_swap
4 hard_predicate_invariant_fuzzy_flip
5 program_atom_near_miss
6 candidate_only_decoy
7 low_margin_near_miss
```

Each candidate includes:

```json
"operator_log": {
  "operator": "predicate_only_perturbation",
  "kept_constant": ["numeric_values", "object_ids"],
  "changed": ["fuzzy_predicates"],
  "targets_shortcut": "number_only",
  "single_primary_intervention": true
}
```

### Step 4: Verify

Run:

```bash
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_afvnb_mnr.TestAFVNBCounterfactualVerifier -v
```

Expected:

```text
OK
```

## 6. Task 5: Baseline Audit

**Files:**

```text
Create: mnr_dataset/afvnb_baselines.py
Modify: tests/test_afvnb_mnr.py
```

### Step 1: Write failing tests for shortcut baselines

Add:

```python
from mnr_dataset.afvnb_baselines import (
    audit_afvnb_metadata_rows,
    predict_fuzzy_program,
    score_hard_program_candidates,
    score_number_only_candidates,
)


class TestAFVNBBaselines(unittest.TestCase):
    def test_fuzzy_program_oracle_matches_label_and_hard_program_exposes_gap(self):
        generator = AFVNBGenerator(AFVNBConfig(seed=17, panel_size=96, shuffle_candidates=False))
        sample = generator.generate_sample("baseline_unit", program_id="diff_outer_inner_boundary")
        metadata = sample["metadata"]

        self.assertEqual(predict_fuzzy_program(metadata), metadata["correct_answer_index"])

        hard_scores = score_hard_program_candidates(metadata)
        hard_flip_index = next(
            index for index, candidate in enumerate(metadata["candidates"])
            if candidate["counterfactual_operator"] == "hard_predicate_invariant_fuzzy_flip"
        )
        self.assertEqual(hard_scores[hard_flip_index], hard_scores[metadata["correct_answer_index"]])
        self.assertLess(metadata["candidates"][hard_flip_index]["truth_score"], metadata["thresholds"]["theta_neg"])

    def test_audit_reports_candidate_bias_and_shortcut_rates(self):
        generator = AFVNBGenerator(AFVNBConfig(seed=19, panel_size=96, shuffle_candidates=False))
        rows = [
            generator.generate_sample("audit_0", program_id="diff_outer_inner_boundary")["metadata"],
            generator.generate_sample("audit_1", program_id="sum_outer_inner_boundary")["metadata"],
        ]
        report = audit_afvnb_metadata_rows(rows)

        self.assertEqual(report["num_samples"], 2)
        self.assertEqual(report["fuzzy_program_accuracy"], 1.0)
        self.assertIn("number_only_wrong_tie_rate", report)
        self.assertIn("hard_program_wrong_tie_rate", report)
```

### Step 2: Implement baseline functions

Required functions:

```text
predict_fuzzy_program(metadata) -> int
score_number_only_candidates(metadata) -> list[float]
score_hard_program_candidates(metadata) -> list[float]
score_candidate_only_prior(metadata) -> list[float]
audit_afvnb_metadata_rows(rows) -> dict
```

Hard-program baseline definition:

```text
For each object, set the largest fuzzy predicate among Outer/Inner/Boundary to 1.0 and the others to 0.0.
Re-evaluate the same first-order program.
```

Number-only baseline definition:

```text
Score candidates by whether some permutation of numeric_value satisfies the arithmetic atom, ignoring fuzzy predicates.
```

Candidate-only audit definition:

```text
At minimum, report correct-index distribution and operator-slot distribution.
The score should not use context panels.
```

### Step 3: Verify

Run:

```bash
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_afvnb_mnr.TestAFVNBBaselines -v
```

Expected:

```text
OK
```

## 7. Task 6: NPZ Writer and CLI

**Files:**

```text
Create: mnr_dataset/afvnb_main.py
Modify: mnr_dataset/afvnb_generator.py
Modify: tests/test_afvnb_mnr.py
```

### Step 1: Write failing CLI test

Add:

```python
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


class TestAFVNBCLI(unittest.TestCase):
    def test_cli_generates_npz_metadata_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable,
                "-m",
                "mnr_dataset.afvnb_main",
                "--num_prob",
                "2",
                "--output_dir",
                tmp,
                "--seed",
                "23",
                "--program_schema",
                "diff_outer_inner_boundary",
                "--no_shuffle",
            ]
            result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            output = Path(tmp)
            generated = sorted(output.glob("*.npz"))
            self.assertEqual(len(generated), 2)
            loaded = np.load(generated[0], allow_pickle=False)
            metadata = json.loads(str(loaded["metadata_json"]))

            self.assertEqual(loaded["context_images"].shape[0], 3)
            self.assertEqual(loaded["answer_set_images"].shape[0], 8)
            self.assertEqual(metadata["schema_version"], "afvnb_v1")
            self.assertTrue((output / "metadata.jsonl").exists())
            self.assertTrue((output / "generation_report.json").exists())
```

### Step 2: Implement CLI contract

CLI args:

```text
--num_prob
--output_dir
--seed
--program_schema {mixed,diff_outer_inner_boundary,sum_outer_inner_boundary,ratio_outer_inner_boundary}
--panel_size
--theta_pos
--theta_neg
--delta
--t_norm {product,min,minimum,godel,lukasiewicz}
--no_shuffle
```

Output files:

```text
afvnb_000000.npz
metadata.jsonl
generation_report.json
```

NPZ arrays:

```text
context_images
answer_set_images
correct_answer_image_index
metadata_json
```

### Step 3: Verify

Run:

```bash
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_afvnb_mnr.TestAFVNBCLI -v
```

Expected:

```text
OK
```

## 8. Task 7: Visualization With Presentation and Debug Views

**Files:**

```text
Create: mnr_dataset/afvnb_visualize.py
Modify: tests/test_afvnb_mnr.py
```

### Step 1: Write failing visualization test

Add:

```python
from mnr_dataset.afvnb_generator import save_sample_npz
from mnr_dataset.afvnb_visualize import save_debug_view, save_presentation_view


class TestAFVNBVisualization(unittest.TestCase):
    def test_visualization_writes_presentation_and_debug_views(self):
        generator = AFVNBGenerator(AFVNBConfig(seed=31, panel_size=96, shuffle_candidates=False))
        sample = generator.generate_sample("viz_unit", program_id="diff_outer_inner_boundary")

        with tempfile.TemporaryDirectory() as tmp:
            sample_path = Path(tmp) / "sample.npz"
            presentation_path = Path(tmp) / "presentation.png"
            debug_path = Path(tmp) / "debug.png"
            save_sample_npz(sample, sample_path)

            save_presentation_view(sample_path, presentation_path)
            save_debug_view(sample_path, debug_path)

            self.assertTrue(presentation_path.exists())
            self.assertTrue(debug_path.exists())
            self.assertGreater(presentation_path.stat().st_size, 1000)
            self.assertGreater(debug_path.stat().st_size, presentation_path.stat().st_size)
```

### Step 2: Presentation view rules

Presentation view must show:

```text
3 context images
8 candidate images
clean answer-choice layout
no score text
no negative type text
no correct-answer highlight by default
```

Debug view may show:

```text
object ids
numeric_value
predicate bars
truth score
program assignment
operator log
correct candidate highlight
```

### Step 3: Verify

Run:

```bash
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_afvnb_mnr.TestAFVNBVisualization -v
```

Expected:

```text
OK
```

## 9. Task 8: README and Research Docs Alignment

**Files:**

```text
Modify: README.md
Modify: doc/afvnb_oral_benchmark_blueprint.md
```

### Step 1: README update contract

README must distinguish:

```text
FVNB v0:
  anchor-centric proof-of-concept

AFVNB v1:
  object-attribute fuzzy program benchmark
```

Include commands:

```bash
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m mnr_dataset.afvnb_main --num_prob 10 --output_dir AFVNB-ProbSet --seed 0 --program_schema mixed

/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m mnr_dataset.afvnb_visualize AFVNB-ProbSet/afvnb_000000.npz --presentation AFVNB-ProbSet/afvnb_000000_presentation.png --debug AFVNB-ProbSet/afvnb_000000_debug.png
```

### Step 2: Documentation sanity checks

Run:

```bash
rg -n "anchor-centric proof-of-concept|object-attribute fuzzy program|AFVNB" README.md doc/afvnb_oral_benchmark_blueprint.md
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 - <<'PY'
from pathlib import Path

patterns = ["TB" + "D", "TO" + "DO", "待" + "定", "占" + "位", "x" * 3, "X" * 3, "补" + "充"]
for path in [
    Path("README.md"),
    Path("doc/afvnb_oral_benchmark_blueprint.md"),
    Path("doc/afvnb_implementation_plan.md"),
]:
    text = path.read_text(encoding="utf-8")
    for pattern in patterns:
        if pattern in text:
            print("{0}: contains {1}".format(path, pattern))
PY
```

Expected:

```text
First command finds the distinction.
Second command prints no matches.
```

## 10. Full Verification Gate

Run all relevant tests:

```bash
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests.test_fvnb_mnr tests.test_afvnb_mnr -v
```

Generate a small probe:

```bash
rm -rf outputs/afvnb_probe
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m mnr_dataset.afvnb_main --num_prob 20 --output_dir outputs/afvnb_probe --seed 0 --program_schema mixed --no_shuffle
```

Inspect metadata:

```bash
/Users/lichengtai/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 - <<'PY'
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("outputs/afvnb_probe/metadata.jsonl").read_text().splitlines()]
print("num_rows", len(rows))
print("schema_versions", sorted({row["schema_version"] for row in rows}))
print("candidate_counts", sorted({len(row["candidates"]) for row in rows}))
print("has_anchor_key", any("anchors" in candidate for row in rows for candidate in row["candidates"]))
print("operators", sorted({candidate["counterfactual_operator"] for row in rows for candidate in row["candidates"]}))
PY
```

Expected:

```text
num_rows 20
schema_versions ['afvnb_v1']
candidate_counts [8]
has_anchor_key False
operators includes correct, attribute_only_perturbation, predicate_only_perturbation,
attribute_predicate_swap, hard_predicate_invariant_fuzzy_flip, program_atom_near_miss,
candidate_only_decoy, low_margin_near_miss
```

## 11. Stop Conditions

Stop and revise the benchmark design if any of these happen:

| Condition | Meaning | Required action |
| --- | --- | --- |
| Hard-program oracle solves hard-predicate invariant split | Fuzzy truth is not actually needed. | Redesign predicate membership intervention. |
| Number-only solver is high on predicate-only perturbation | Visual dependency is weak. | Strengthen predicate-only counterfactual and value balance. |
| Candidate-only model is high | RAVEN-FAIR style bias remains. | Rebalance candidate appearance and correct-index distribution. |
| Fuzzy oracle below 100% on generated metadata | Program/verifier bug. | Fix evaluator before generating more data. |
| Human sanity check below acceptable agreement | Images are not legible as semantic objects. | Simplify rendering and object schema. |
| Implementation reintroduces `anchor["value"]` | Ontology regression. | Reject change and restore object-attribute path. |

## 12. Completion Definition

AFVNB v1 probe is complete only when:

```text
All FVNB v0 tests pass.
All AFVNB v1 tests pass.
20-sample probe generation succeeds.
metadata contains object scene graphs, not anchors.
program evaluation recomputes labels.
verifier reports unique answer and valid thresholds.
hard-program oracle fails on hard-predicate invariant cases.
candidate-only audit does not show obvious shortcut.
presentation view and debug view are separated.
README explains FVNB v0 vs AFVNB v1.
```

Until these are true, the project is still at research maturity L3/L4 rather than evidence-ready L5.
