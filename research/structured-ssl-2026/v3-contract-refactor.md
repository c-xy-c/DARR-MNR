# V3 representation and ranking contracts, 2026-10-04

Reference: `4b2a2662071904bad9f05bf41553c4f7c274daa6`.

## Decisions

The predictor consumes visible context; the energy function consumes fixed
targets. These roles now have separate records in `representations.py`:

- `VisibleContext`: trainable network levels and valid visible regions.
- `PanelTargets`: fixed dense features, object features and valid target regions.

Both records slice all their fields together, preserving sample/panel alignment.
They hold existing tensors without copying, detaching or changing gradients.

`encode_context` and `fixed_targets` construct these records. `score_row` names
the support, prefix and completion roles explicitly; support exchange changes
the new branch only, preserving the frozen anchor. `rank` owns the two-support-row
sum and two-component-view mean. Model inference and intervention evaluation both
call it. `energy.py` owns the stage-averaged dense/object discrepancy.

The old `encode`, `targets` and long positional `row` interfaces are removed.
Production code has no compatibility aliases. Historical adaptations exist only
in the source-comparison and sealed-run diagnosis tools. The compiler-rank tool
now reads dimensions from their owning module when diagnosing split source exports.

The V3 architecture, parameter names, initialization, objective weights,
negative selection, teacher targets and checkpoint selection remain unchanged.

## Verification

- [Exact CPU parity](v3_contract_refactor_parity.json): identical initialization,
  strict checkpoint loading, scores, targets, support-swap factors/anchor/scores,
  two complete losses/gradients/Adam updates and 14 real validation scores.
- The same parity check runs both evaluators on a four-puzzle synthetic batch:
  predictions and all intervention statistics match. The full-split aggregation
  is replaced only for that fixture; this is not a full RAVEN evaluation.
- [Mechanism boundaries](v3_contract_refactor_contracts.json): finite gradients,
  input/anchor preservation, isolated targets and candidate/object permutations.
- [Control exports](v3_contract_refactor_controls.json): both same-capacity
  controls parse and pass CPU contracts with the new representation records.
- [Live source integrity](v3_contract_refactor_live_integrity.json): all 84 launch
  files and three pinned policy files remain unchanged; completed artifact
  collection passes. Training proceeds from its independent source export.

No test set was opened by these refactor checks. Reported benchmark results still
refer to their original sealed sources, not a new evaluation of this checkout.
