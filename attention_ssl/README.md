# Attention SSL + continuous support completion

Three complete eight-epoch RAVEN runs achieved **71.24%, 71.24%, and 71.17%**
test accuracy after validation selection, compared with **71.04%** for the same
frozen baseline. Their final checkpoints also exceeded that baseline. The
[measured results](../research/ssl-attention-2026/results.md) include final scores,
all seven configurations, contribution controls, and support interventions. The
[2026 paper references and experiment protocol](../research/ssl-attention-2026/design.md)
separate adopted mechanisms from claims not established by these experiments.

Training uses only six known panels. A learned relation-key encoder attends
across the 25 spatial locations. Each support row fits a regularized continuous
map from its prefix keys to its known completion residual. Candidates cannot
change that map. Both SSL and inference use the same baseline-plus-conditional
energy difference; SSL uses up to seven hard known-panel negatives of the same
layout/component and filters pixel duplicates.

The complete baseline, including BatchNorm, remains frozen. The gate starts
at zero for exact initial parity. All three trained best/final checkpoints
changed candidate choices and depended on support while preserving matched
full-test accuracy. No bidirectional task or collapse penalty is added.

From repository root:

```bash
python -m attention_ssl.check_contracts \
  --baseline program_ssl/results/support-program-v3/control/best.pt \
  --device cpu --output preflight.json

python -m attention_ssl.train \
  --dataset-root /path/to/RAVEN --run-dir runs/attention-completion \
  --baseline program_ssl/results/support-program-v3/control/best.pt \
  --epochs 8 --seed 12345

python -m attention_ssl.evaluate \
  --dataset-root /path/to/RAVEN --run-dir runs/attention-completion --split val

# Allowed only after best passes the full validation audit:
python -m attention_ssl.evaluate \
  --dataset-root /path/to/RAVEN --run-dir runs/attention-completion --split test
```

The primary published checkpoint is seed12345, chosen as the primary run before
its test evaluation. To replay it, use run directory
`research/ssl-attention-2026/results/seed12345`. Existing evaluation files are
preserved; replay in a separate copy of the run directory with those two
evaluation output files omitted, first with `--split val`, then `--split test`.
Do not choose another seed or epoch on test accuracy. Source and checkpoint
seals must match when evaluating.

Training requires CUDA and the complete 42000/14000 train/validation splits.
Best is selected only among trained epochs, using the deployed scoring formula.
Resume with `--resume` and unchanged arguments/source. A completed run seals the
runtime and best/final checkpoint hashes. Evaluation checks the seal before
opening data, refuses to overwrite records, and compares full, frozen baseline,
and within-layout/component support exchange for both best and final.

The example baseline already has historical 71.04% test accuracy. It was
selected using validation, and costs 28 training-selection epochs before this
adaptation; eight added epochs bring the method budget to 36. This baseline
accuracy is not a result of the new model. The original 70.38% release baseline
can be supplied to perform a separate matched comparison, with a separate run.

The added branch has 27,721 trainable parameters; the original 930,624 parameters
remain frozen. Hard negatives and spatial attention each showed small gains in
one-seed controls, which do not establish seed-stable individual contributions.
The CNN and layout-derived component views are inherited. The method is an
adaptation of a pretrained baseline, whose original training used candidates
as unlabeled negatives; only this adaptation excludes candidates. The historical
original-reasoner final checkpoint had 71.41% test accuracy, so these results
do not exceed every historical checkpoint.
