# V3 runtime cleanup, 2026-10-04

Reference: Git commit `5475c323137381d4f75ff533d74434a81857d719`.
The current implementation and supported commands retain V3.

## Current ownership

| Module | Owns |
| --- | --- |
| `data.py` | Image views, contour proposals and batch contracts |
| `perception.py` | Trainable object encoding and isolated fixed targets |
| `pav.py` | Support compilation and three-stage P/G/V execution |
| `energy.py` | Dense/object discrepancy and support residual feedback |
| `objectives.py` | Known-cell retrieval and whole-object masking losses |
| `model.py` | Composition and frozen-anchor candidate scoring |
| `metrics.py` | Shared complete seven-layout accuracy aggregation |
| `train.py` / `evaluate.py` | Optimization/validation selection and sealed evaluation |
| `runtime.py` / `provenance.py` / `policy.py` | Device/RNG, artifact identity and evaluation admission |

Inference feedback now depends on energy computation directly. Training and
evaluation share one accuracy definition, and runtime uses the shared file
digest helper. Parameter names, initialization, losses and scoring formulas
remain identical to the reference.

Removed the obsolete `queue_revision.py`, `resume_suite.py`,
`replay_native_overflow.py`, `diagnose_validation.py` and `check_revision.py`
entrypoints. Historical V1/V2 documents and 26 diagnostic artifacts are preserved
byte for byte under [archive](archive/README.md), with original paths and hashes
in its [manifest](archive/manifest.json). The one-off policy migration is archived;
the running coordinator executes its sealed copy under `runs/`.

## Current verification

- [Exact CPU parity](v3_cleanup_parity.json): strict checkpoint loading, identical
  scores and targets, two identical losses, all gradients and Adam updates;
  identical scores on 14 real validation puzzles spanning seven layouts.
- [Mechanism contracts](v3_cleanup_contracts.json): isolated teacher targets,
  no candidate training inputs, finite gradients and permutation behavior.
- [Both contribution controls](v3_cleanup_controls.json): updated exports parse
  and pass CPU contracts; masking gradients are zero only in that control, and
  static parameters do not change under support exchange. Capacity is preserved.
- [Metrics and archive checks](v3_cleanup_metrics_archive.json): exact original
  statistics on 14,000 synthetic predictions, incomplete-split rejection,
  valid supported-document links and 26 unchanged historical files.
- [Live integrity](v3_cleanup_live_integrity.json): all 84 launch source files
  and three policy files unchanged; the completed primary/native collector still
  passes and the V3 training child remains running.

These are refactor checks, not a new full accuracy evaluation. The measured V3
primary best/final results are retained unchanged in their separate evidence
files. Contribution gains and additional seeds remain pending.

---

# Earlier runtime cleanup, 2026-10-03

Reference: Git commit `184f4d7bdfccbb3533f6e97c825de414ade21cce`.

## Decisions

1. A single model composition belongs in `structured_ssl/model.py`. Pixel
   perception, support PaV execution and training objectives have their own files.
2. Necessary shared data and feature energy belong to `sspredrnet`, rather than
   importing an obsolete model package for two helper functions.
3. Frozen attention and original reasoner code remain, because they execute
   inside the current scoring anchor and define its teacher feature space.
4. Delete the discrete program runtime, train/evaluate commands, feature cache,
   contract CLI and obsolete one-off cleanup script. Retain historical artifacts;
   their sealed source is available from recorded revisions. The measured
   six-operator comparator is confined to its research export.
5. Remove the unused externally supplied `factors` argument from PaV. Support
   compilation is mandatory in the primary model. Contribution controls are
   separate source exports and preserve trainable capacity.
6. Source hashing belongs in provenance modules. Evaluation no longer imports
   the training CLI to fingerprint a model.
7. Accuracy admission is separate from numerical scoring. New runs declare the
   user's ≥70% target before training; nonregression remains a reported comparison.
   Historical training/evaluation seals retain their recorded policy.

No loss term, inference formula, parameter name, initialization order or optimizer
has changed. The shared feature energy explicitly disables autocast on its actual
device, preserving the FP32 scoring contract. CPU parity is directly verified;
an independent full Metal reevaluation after cleanup has not been run.

## Verification

- [Exact parity](cleanup_parity.json): strict load of the running experiment's
  best checkpoint; identical synthetic candidate scores and teacher targets;
  two identical loss/gradient/Adam steps; identical scores on 14 validation
  puzzles covering all seven layouts. No test data opened.
- [Boundary and mechanism checks](cleanup_contracts.json): finite trainable
  gradients, unchanged frozen anchor and inputs, isolated object targets,
  candidate/object permutations and no answer-candidate training inputs.
- [Contribution controls](cleanup_controls.json): both updated source exports
  pass their CPU mechanisms, including static support factors and a zero-weight
  object objective. Both retain 2,275,350 trainable parameters.
- [Live export integrity](cleanup_live_integrity.json): all 84 files across
  the three active experiment exports still match their launch hashes. The
  training child was not interrupted.
- Six structured/attention control exports parse successfully, training and
  evaluation CLI imports succeed, and the collector still validates completed
  historical native and rejected v1 results.

These checks establish refactor compatibility and boundary behavior. They do
not establish full RAVEN accuracy, contribution gains or semantic object learning.
