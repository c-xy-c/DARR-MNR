# Component-conditioned predictive programs

Research code based on the published SSPredRNet component-view checkpoint.
The [method and experiment contract](../research/component-program-ssl/design.md)
defines CECS self-supervision and support evidence ratio PaV verification.

This is an eight-epoch adaptation study initialized from the validation-selected
epoch-17 checkpoint of a completed 20-epoch run. It is not training from scratch.
The full method freezes the complete baseline and learns completion operators
using six known panels. Its earlier baseline pretraining used answer candidates
as unlabeled negatives.

```bash
python -m program_ssl.check_contracts \
  --baseline sspredrnet/results/raven-20epoch/best.pt --output preflight.json

python -m program_ssl.train \
  --dataset-root /path/to/RAVEN --run-dir runs/program \
  --baseline sspredrnet/results/raven-20epoch/best.pt --study program --epochs 8
```

Run the same command with separate output directories and studies `control`,
`program-no-ssl`, and `program-static` to complete the comparison.

After all four runs complete, validation audits lock the checkpoints before test:

```bash
python -m program_ssl.evaluate \
  --dataset-root /path/to/RAVEN --experiment-root runs
python -m program_ssl.evaluate \
  --dataset-root /path/to/RAVEN --experiment-root runs --test
```

The published `sspredrnet` package and its 70.38% result are unchanged. The new
model was evaluated after validation selection and checkpoint locking. Its
selected checkpoint scores **70.257143%** (9836/14000) on test. The final
checkpoint scores **70.264286%** (9837/14000). Both meet the requested 70% gate,
while remaining below the baseline and the original-reasoner continuation.
No result is inherited from the baseline or a rejected prototype.

The four studies, eight checkpoints, validation audit, selection grid and test
lock are included under [results/support-program-v3](results/support-program-v3).
See the [results and limits](../research/component-program-ssl/results.md) and
[two contribution definitions](../research/component-program-ssl/contributions.md).

To replay the locked published test evaluation from the repository root:

```bash
python -m program_ssl.evaluate \
  --dataset-root /path/to/RAVEN \
  --experiment-root program_ssl/results/support-program-v3 --test
```

The checkpoints retain training state. The duplicate remote `last.pt` files are omitted from the published
records. Remote run directories preserve full resume state.

Version 3 isolates preprocessing of the masked target. Only the five observed
predictor panels determine a global component fallback. An ambiguous target
falls back independently. Version 2 was rejected for violating this boundary.
Its records are preserved under `research/component-program-ssl/archive-v2`.

After checkpoint selection, the auditor calibrates the positive weight over
`[0.05, 0.1, 0.2]` and compares posterior energy with a support evidence ratio.
The latter subtracts uniform operator compatibility from posterior energy.
It was introduced after validation diagnostics showed that direct energy
addition could duplicate candidate appearance preferences. This is a six-item
validation grid, not an original preregistration. Checkpoints remain selected
by the original training validation metric. The chosen formula and weight are
locked for best and final test evaluation. Zero is only an intervention.

Evaluation can reuse exact FP32 features from the frozen CNN. The cache does
not affect training. Full native/cache validation counts and generator states
were checked for the baseline and program. Test features are first generated
after the evaluator checks the checkpoint and source lock. The selected full
model's complete native test evaluation also matched cached scoring exactly.

CECS improved held-out known-panel retrieval by 2.14 percentage points versus
training the same new branch with the original margin objective. This is
within-batch completion retrieval, not eight-choice RAVEN accuracy. Uniform
support interventions reduce that retrieval result. Candidate accuracy gains
from SER-PaV are not established. The proposal has one RAVEN seed and does not
establish semantic rule discovery or generalization beyond this experiment.
