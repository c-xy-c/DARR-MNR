# AFVNB-MNR Metadata Schema Contract

日期：2026-06-18

本文档定义 AFVNB-MNR 的 metadata 合同。它的作用是防止实现时退回 FVNB v0 的 `anchor + value` 语义：在 AFVNB 中，数字必须是 `value_token` 对象的 `numeric_value` 属性，视觉结构必须以 `fuzzy_predicates` 进入 executable program truth。

## 1. Schema Version

每条样本 metadata 必须声明：

```json
{
  "schema_version": "afvnb_v1"
}
```

禁止使用：

```json
{
  "schema_version": "fvnb_v0"
}
```

AFVNB v1 兼容目标：

```text
FVNB v0 files may coexist in the repository.
AFVNB v1 metadata must not depend on FVNB v0 anchor fields.
```

## 2. Top-Level Sample Schema

Required top-level fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `schema_version` | string | yes | Must equal `afvnb_v1`. |
| `sample_id` | string | yes | Unique sample id, for example `afvnb_000001`. |
| `seed_id` | string | yes | Program seed id shared by paired variants. |
| `family` | string | yes | Visual family, for example `containment_band`. |
| `split_tags` | list[string] | yes | Split membership, for example `["IID", "Counterfactual-Stress"]`. |
| `program` | object | yes | Executable fuzzy first-order program. |
| `thresholds` | object | yes | Label construction thresholds. |
| `context` | list[panel] | yes | Exactly 3 context panels. |
| `candidates` | list[candidate] | yes | Exactly 8 answer candidates. |
| `correct_answer_index` | integer | yes | Index in `candidates`. |
| `validity` | object | yes | Verifier summary. |
| `generation_config` | object | yes | Parameters needed to reproduce sampling. |
| `release_flags` | object | yes | Visibility and leakage controls. |

Forbidden top-level fields:

```text
anchors
answer_set
negative_type
correct_answer_image_index
```

`correct_answer_image_index` is allowed inside `.npz` arrays for compatibility with existing loaders, but canonical metadata must use `correct_answer_index`.

## 3. Program Schema

Required `program` fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `program_id` | string | yes | `diff_outer_inner_boundary`, `sum_outer_inner_boundary`, or `ratio_outer_inner_boundary`. |
| `variables` | list[object] | yes | First-order variables over object domains. |
| `constraints` | list[object] | yes | Predicate, distinctness, and arithmetic atoms. |
| `aggregation` | object | yes | Existential max and t-norm semantics. |

Example:

```json
{
  "program_id": "diff_outer_inner_boundary",
  "variables": [
    {"name": "x", "domain": "value_token"},
    {"name": "y", "domain": "value_token"},
    {"name": "z", "domain": "value_token"}
  ],
  "constraints": [
    {"op": "distinct", "args": ["x", "y", "z"]},
    {"op": "predicate", "name": "Outer", "arg": "x"},
    {"op": "predicate", "name": "Inner", "arg": "y"},
    {"op": "predicate", "name": "Boundary", "arg": "z"},
    {
      "op": "near_equal",
      "left": {"op": "sub", "args": [{"value": "x"}, {"value": "y"}]},
      "right": {"value": "z"}
    }
  ],
  "aggregation": {
    "exists": "max",
    "and": "product",
    "epsilon": 0.001
  }
}
```

Program invariant:

```text
Every program variable binds to an object id.
Every arithmetic atom reads `numeric_value` through `Value(object)`.
No program atom reads visual text directly.
```

## 4. Panel Schema

Required panel fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `panel_id` | string | yes | Context or candidate panel id. |
| `scene_graph` | object | yes | Object-level scene graph. |
| `truth_score` | number | yes | Program truth score for this panel. |
| `oracle_assignment` | object | yes | Best variable-to-object assignment. |
| `atom_truths` | object | yes | Truth values for program atoms under oracle assignment. |
| `image_path` | string | release yes | Relative path to rendered panel image in released dataset. |

Context panels also include:

```text
panel_role = "context"
```

Candidate panels also include:

```text
panel_role = "candidate"
candidate_index
counterfactual_operator
operator_log
shortcut_consistency
```

Forbidden panel fields:

```text
anchors
intended_role
negative_type
hard_role as primary label
```

Hard-role summaries are allowed only under:

```text
diagnostics.hard_program
```

They must not be the canonical rule representation.

## 5. Scene Graph Schema

Required scene graph fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `scene_id` | string | yes | Unique panel scene id. |
| `renderer` | string | yes | Renderer id, for example `minimal_containment_v1`. |
| `objects` | list[object] | yes | Scene objects. |
| `regions` | list[object] | yes | Region objects defining predicates. |
| `relations` | list[object] | yes | Optional soft relations. |

### 5.1 Object schema

Required object fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `id` | string | yes | Object id used by program assignment. |
| `class` | string | yes | `value_token`, `container`, `region_marker`, or family-specific class. |
| `numeric_value` | integer or null | yes | Numeric attribute. Must be integer for `value_token`. |
| `visual_attributes` | object | yes | Shape, text rendering, fill, outline. |
| `geometry` | object | yes | Position and distance features. |
| `fuzzy_predicates` | object | yes | Predicate truth values. |

For `value_token`:

```text
numeric_value must be an integer.
visual_attributes.text must equal the rendered string of numeric_value.
fuzzy_predicates must include every predicate used by the program.
```

For non-value objects:

```text
numeric_value must be null.
```

Forbidden object fields:

```text
value
role
intended_role
membership as a replacement for fuzzy_predicates
```

### 5.2 Fuzzy predicate schema

Each predicate value:

```json
{
  "raw": 0.92,
  "normalized": 0.94,
  "source": "signed_distance_inner_boundary"
}
```

Required keys:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `raw` | number | yes | Unnormalized geometric evidence. |
| `normalized` | number | yes | Value used by program truth. |
| `source` | string | yes | How the predicate was computed. |

Predicate invariants:

```text
0 <= raw
0 <= normalized <= 1
For each value_token, normalized values over active role predicates should sum to 1 within tolerance 1e-6 when the family uses normalized roles.
```

## 6. Candidate Schema

Required candidate fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `panel_id` | string | yes | Candidate panel id. |
| `panel_role` | string | yes | Must equal `candidate`. |
| `candidate_index` | integer | yes | 0 to 7. |
| `scene_graph` | object | yes | Candidate scene. |
| `truth_score` | number | yes | Fuzzy program truth. |
| `hard_program_score` | number | yes | Diagnostic score after predicate hardening. |
| `oracle_assignment` | object | yes | Best variable-to-object assignment. |
| `atom_truths` | object | yes | Atom truth decomposition. |
| `counterfactual_operator` | string | yes | One of the eight operator names. |
| `operator_log` | object | yes | Single-intervention record. |
| `shortcut_consistency` | object | yes | Which shortcut would find this candidate plausible. |
| `diagnostics` | object | yes | Noncanonical helper values for audit. |

Valid `counterfactual_operator` values:

```text
correct
attribute_only_perturbation
predicate_only_perturbation
attribute_predicate_swap
hard_predicate_invariant_fuzzy_flip
program_atom_near_miss
candidate_only_decoy
low_margin_near_miss
```

Candidate invariant:

```text
Exactly one candidate has counterfactual_operator = correct.
correct_answer_index points to that candidate after shuffling.
All non-correct candidates must have operator_log.single_primary_intervention = true.
```

## 7. Operator Log Schema

Required `operator_log` fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `operator` | string | yes | Must equal `counterfactual_operator`. |
| `kept_constant` | list[string] | yes | Factors preserved by the operator. |
| `changed` | list[string] | yes | Factors intentionally changed. |
| `targets_shortcut` | string | yes | Shortcut being tested. |
| `single_primary_intervention` | boolean | yes | Whether the counterfactual is clean. |
| `verifier_checks` | object | yes | Boolean checks for isolation. |

Example:

```json
{
  "operator": "predicate_only_perturbation",
  "kept_constant": ["numeric_values", "object_ids", "value_multiset"],
  "changed": ["fuzzy_predicates", "region_geometry"],
  "targets_shortcut": "number_only",
  "single_primary_intervention": true,
  "verifier_checks": {
    "value_multiset_preserved": true,
    "object_count_preserved": true,
    "appearance_delta_within_tolerance": true,
    "truth_score_below_negative_threshold": true
  }
}
```

## 8. Validity Schema

Required `validity` fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `oracle_recomputed` | boolean | yes | Saved scores match recomputation. |
| `unique_answer` | boolean | yes | Top candidate unique under threshold and margin. |
| `positive_threshold` | boolean | yes | Correct truth >= theta_pos. |
| `negative_threshold` | boolean | yes | All negatives <= theta_neg. |
| `top_margin` | number | yes | Correct score minus top negative. |
| `max_negative_score` | number | yes | Highest negative score. |
| `candidate_balance_pass` | boolean | yes | Candidate statistics pass audit. |
| `operator_isolation_pass` | boolean | yes | All operator logs are clean. |
| `rejection_reason` | null or string | yes | Null for accepted samples. |

Accepted sample invariant:

```text
Every boolean in validity except diagnostic-only warnings must be true.
rejection_reason must be null.
```

Rejected samples may be stored in generator logs but must not be released as benchmark samples.

## 9. Release Flags

Required `release_flags`:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `evaluation_image_leakage_free` | boolean | yes | Evaluation images hide labels, scores, operators. |
| `debug_metadata_available` | boolean | yes | Debug fields exist outside evaluation image. |
| `language_prompt_role_free` | boolean | yes | Evaluation prompt does not reveal role names. |
| `contains_human_annotation` | boolean | yes | Whether human answers are attached. |
| `contains_model_outputs` | boolean | yes | Whether model predictions are attached. |

Release invariant:

```text
evaluation_image_leakage_free must be true for all public evaluation images.
```

## 10. Complete Metadata Example

```json
{
  "schema_version": "afvnb_v1",
  "sample_id": "afvnb_000001",
  "seed_id": "seed_000001",
  "family": "containment_band",
  "split_tags": ["IID", "Counterfactual-Stress"],
  "program": {
    "program_id": "diff_outer_inner_boundary",
    "variables": [
      {"name": "x", "domain": "value_token"},
      {"name": "y", "domain": "value_token"},
      {"name": "z", "domain": "value_token"}
    ],
    "constraints": [
      {"op": "distinct", "args": ["x", "y", "z"]},
      {"op": "predicate", "name": "Outer", "arg": "x"},
      {"op": "predicate", "name": "Inner", "arg": "y"},
      {"op": "predicate", "name": "Boundary", "arg": "z"},
      {
        "op": "near_equal",
        "left": {"op": "sub", "args": [{"value": "x"}, {"value": "y"}]},
        "right": {"value": "z"}
      }
    ],
    "aggregation": {"exists": "max", "and": "product", "epsilon": 0.001}
  },
  "thresholds": {"theta_pos": 0.75, "theta_neg": 0.45, "delta": 0.20},
  "context": [
    {
      "panel_id": "context_0",
      "panel_role": "context",
      "image_path": "images/afvnb_000001_context_0.png",
      "scene_graph": {
        "scene_id": "scene_context_0",
        "renderer": "minimal_containment_v1",
        "objects": [
          {
            "id": "o_A",
            "class": "value_token",
            "numeric_value": 8,
            "visual_attributes": {"shape": "circle", "text": "8", "fill": "white", "outline": "black"},
            "geometry": {"center": [24, 64], "radius": 8, "signed_distance_outer": 32.0, "signed_distance_inner": -11.0, "distance_to_boundary": 11.0},
            "fuzzy_predicates": {
              "Outer": {"raw": 0.92, "normalized": 0.94, "source": "signed_distance"},
              "Inner": {"raw": 0.02, "normalized": 0.02, "source": "signed_distance"},
              "Boundary": {"raw": 0.04, "normalized": 0.04, "source": "boundary_distance"}
            }
          }
        ],
        "regions": [
          {"id": "r_outer", "class": "container", "geometry": {"type": "rect", "bounds": [8, 8, 120, 120]}}
        ],
        "relations": [
          {"subject": "o_A", "relation": "inside_soft", "object": "r_outer", "truth": 0.94}
        ]
      },
      "truth_score": 0.821,
      "oracle_assignment": {"x": "o_A", "y": "o_B", "z": "o_C"},
      "atom_truths": {"Outer(x)": 0.96, "Inner(y)": 0.94, "Boundary(z)": 0.91, "Equal(Value(x)-Value(y),Value(z))": 1.0}
    }
  ],
  "candidates": [
    {
      "panel_id": "candidate_0",
      "panel_role": "candidate",
      "candidate_index": 0,
      "image_path": "images/afvnb_000001_candidate_0.png",
      "scene_graph": {"scene_id": "scene_candidate_0", "renderer": "minimal_containment_v1", "objects": [], "regions": [], "relations": []},
      "truth_score": 0.821,
      "hard_program_score": 1.0,
      "oracle_assignment": {"x": "o_A", "y": "o_B", "z": "o_C"},
      "atom_truths": {"Outer(x)": 0.96, "Inner(y)": 0.94, "Boundary(z)": 0.91, "Equal(Value(x)-Value(y),Value(z))": 1.0},
      "counterfactual_operator": "correct",
      "operator_log": {
        "operator": "correct",
        "kept_constant": ["program", "numeric_values", "fuzzy_predicates"],
        "changed": [],
        "targets_shortcut": "none",
        "single_primary_intervention": true,
        "verifier_checks": {"truth_score_above_positive_threshold": true}
      },
      "shortcut_consistency": {"number_only": true, "coordinate_order": true, "hard_program": true, "candidate_only": false},
      "diagnostics": {"hard_assignment": {"x": "o_A", "y": "o_B", "z": "o_C"}}
    }
  ],
  "correct_answer_index": 0,
  "validity": {
    "oracle_recomputed": true,
    "unique_answer": true,
    "positive_threshold": true,
    "negative_threshold": true,
    "top_margin": 0.40,
    "max_negative_score": 0.42,
    "candidate_balance_pass": true,
    "operator_isolation_pass": true,
    "rejection_reason": null
  },
  "generation_config": {
    "seed": 0,
    "panel_size": 128,
    "t_norm": "product",
    "tau": 2.0,
    "sigma": 1.8,
    "boundary_weight": 5.0
  },
  "release_flags": {
    "evaluation_image_leakage_free": true,
    "debug_metadata_available": true,
    "language_prompt_role_free": true,
    "contains_human_annotation": false,
    "contains_model_outputs": false
  }
}
```

The example uses abbreviated candidate scene objects for compactness. Real released metadata must include complete scene graphs for every context and candidate.

## 11. Serializer Rules

Serializer must output:

```text
metadata.jsonl: one top-level sample metadata object per line.
programs.json: unique program definitions and version ids.
splits.json: sample ids per split.
counterfactual_pairs.json: seed ids and paired variant ids.
generation_config.json: global generator configuration.
```

NPZ compatibility package may include:

```text
context_images
answer_set_images
correct_answer_image_index
metadata_json
```

But `metadata_json` remains the canonical source of benchmark truth.

## 12. Validator Checklist

A metadata validator must fail a sample if:

```text
schema_version != afvnb_v1
any candidate contains anchors
any value_token lacks numeric_value
any non-value object has numeric_value not null
any program variable assignment points to missing object id
any saved truth score differs from recomputation
correct_answer_index does not match argmax truth score
operator_log is missing for any candidate
any non-correct operator has single_primary_intervention = false
evaluation_image_leakage_free is false
```

Validator should warn if:

```text
normalized predicate sum differs from 1 by more than 1e-6
top margin is close to delta
candidate balance metrics approach release thresholds
human or model outputs are mixed into base release metadata
```

## 13. Reviewer-Facing Rationale

This schema is part of the benchmark contribution. It operationalizes the top-conference standards:

| Standard | Schema feature |
| --- | --- |
| CLEVR/GQA executable metadata | `program`, `scene_graph`, `oracle_assignment`, `atom_truths` |
| RAVEN-FAIR candidate bias control | `candidate_balance_pass`, `candidate_only` diagnostics |
| MathVerse/VC-Bench visual dependency | `split_tags`, information variants, `release_flags` |
| TACIT deterministic near-miss | `counterfactual_operator`, `operator_log`, `verifier_checks` |
| Fuzzy set / Real Logic semantics | `fuzzy_predicates`, first-order variables, t-norm aggregation |

The schema therefore is not bookkeeping. It is how AFVNB makes the claim testable:

```text
numeric_value is an object attribute;
fuzzy predicates are object truth values;
the program binds both before computing arithmetic truth.
```
