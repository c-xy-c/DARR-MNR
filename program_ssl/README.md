# Component-conditioned predictive programs

Research code based on the published SSPredRNet component-view checkpoint.
The [method and experiment contract](../research/component-program-ssl/design.md)
defines CECS self-supervision and support-calibrated PaV verification.

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
model needs its own completed test evaluation before reporting a performance
claim. No result is inherited from the baseline or a rejected prototype.
