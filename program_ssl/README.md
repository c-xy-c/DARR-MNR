# CECS + SER-PaV

The runnable method has one self-supervised training task and one inference
formula, built on the component-view SSPredRNet checkpoint. See the
[method](../research/component-program-ssl/design.md) and
[measured results](../research/component-program-ssl/results.md).

## Pipeline

1. Training reads only the first six known panels and makes two component views.
   Only the five observed panels determine a global split fallback; an ambiguous
   target falls back independently. Answer indices and candidates are excluded.
2. The complete pretrained SSPredRNet, including its BatchNorm buffers, is frozen.
3. Six learned completion operators predict a query ending. Support-row
   prediction errors and a bounded prior determine the operator weights.
4. CECS learns the known sixth panel using positive completion energy and
   InfoNCE against other known targets of the same layout and component.
5. At inference, both complete rows determine fixed operator weights. The
   evidence ratio subtracts uniform-operator compatibility from posterior
   compatibility. Candidate energy is baseline PaV energy plus 0.05 times this
   ratio; the lowest of eight scores wins.

This is adaptation from a pretrained baseline, not training from scratch. The
baseline's original training used answer candidates as unlabeled negatives.
No collapse regularizer or bidirectional task is included.

## Commands

Run from the repository root with dependencies in
[sspredrnet/requirements.txt](../sspredrnet/requirements.txt).

```bash
python -m program_ssl.check_contracts \
  --baseline sspredrnet/results/raven-20epoch/best.pt \
  --device cpu --output preflight.json

python -m program_ssl.train \
  --dataset-root /path/to/RAVEN --run-dir runs/program \
  --baseline sspredrnet/results/raven-20epoch/best.pt \
  --epochs 8 --lr 3e-4

python -m program_ssl.evaluate \
  --dataset-root /path/to/RAVEN --run-dir runs/program \
  --split test --output runs/program/test-evaluation.json
```

Training requires CUDA. Contract checks support CPU or CUDA. For a new run,
validation selects the best checkpoint using baseline plus 0.1 times posterior
completion energy. Inference uses the fixed 0.05 evidence ratio. These separate
roles preserve the measured checkpoint-selection protocol; they are not
user-selectable scoring modes. Training never opens the test split.

Use `--resume` with the same arguments to resume a run made by this trainer
(config version 4). Historical version-3 runs require their original source.
Completed training writes `inference.json`, sealing source files, both
checkpoints and the inference weight. Evaluation verifies that seal before
opening data, and refuses to overwrite an explicitly named output file.
Optional `--cache-root /path/to/cache` reuses frozen FP32 evaluation features;
training does not use this cache.

## Published checkpoints and historical evidence

```bash
python -m program_ssl.evaluate \
  --dataset-root /path/to/RAVEN \
  --run-dir program_ssl/results/support-program-v3/program \
  --split test --output reproduced-test.json
```

The original completed experiment reports **70.257143%** (9836/14000) for the
validation-selected checkpoint and **70.264286%** (9837/14000) for the final
checkpoint. These are historical full-RAVEN measurements. They are below the
70.38% baseline; a candidate-ranking benefit is not established. CECS improved
held-out known-panel retrieval by 2.14 percentage points against the original
margin adaptation, which is a different metric.

The cleanup preserved checkpoint bytes and historical result/lock files.
[cleanup_verification.json](results/support-program-v3/cleanup_verification.json)
compares this implementation with commit `f43cf58`: preprocessing across all
seven layouts, two loss/gradient/Adam steps, and all eight candidate scores
from the published best and final checkpoints. These cleanup checks used
synthetic fixtures on CPU, not a renewed complete RAVEN/GPU evaluation.
[cleanup_contracts.json](results/support-program-v3/cleanup_contracts.json)
checks candidate exclusion, target isolation, permutation equivariance and
finite gradients. The separate `program/inference.json` seals the cleaned
runtime and includes the historical test counts as replay expectations. It
does not replace the original `test_lock.json` or retroactively preregister
the experiment.

The four-study trainer, interventions, calibration grid, legacy scoring
switches and server-specific launcher have been removed from current runtime.
Their measurements remain under [results/support-program-v3](results/support-program-v3).
To reproduce those historical studies and their original lock, use a separate
checkout of commit `f43cf580250303f4c860c161df668502ea2aa1f3`.
Rejected prototype records remain under `research/component-program-ssl/archive-v1`
and `archive-v2`; obsolete prototype executables are not part of the current
API. Checkpoints retain optimizer/RNG state; duplicate `last.pt` files are
omitted from the published records.
