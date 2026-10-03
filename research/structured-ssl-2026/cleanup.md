# Runtime cleanup, 2026-10-03

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
