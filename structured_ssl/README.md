# Structured object SSL and support-compiled SER-PaV

Current architecture: `isolated-target-structured-feedback-shine-rl-v3`.
It has **2,275,350 trainable** and **958,345 frozen** parameters. The primary V3
seed has completed: validation-selected best **71.2786% test**,
final **69.6429% test**. Controls and two further seeds remain in progress.

## Responsibilities

| File | Responsibility |
| --- | --- |
| `data.py` | Independent image proposals and training/evaluation packs |
| `constants.py` | Fixed v3 widths, stages, rank and region count |
| `blocks.py` | Shared self-attention and cross-attention blocks |
| `perception.py` | Three encoder levels, isolated fixed targets and masked-object head |
| `pav.py` | Support memory, SHINE rl factor packing, three-stage P/G/V and residual feedback |
| `energy.py` | Dense/object discrepancy and transport-aligned support residuals |
| `objectives.py` | Known-cell retrieval and complete-object masking losses |
| `metrics.py` | Shared complete seven-layout selection/evaluation statistics |
| `model.py` | Compose perception, PaV and frozen scoring anchor |
| `train.py` | Optimization, validation selection, exact resume and completion seal |
| `evaluate.py` | Best/final ranking and isolated support intervention |
| `policy.py` | User-directed 70% target and active-support validation criterion |
| `provenance.py` | Explicit source dependencies and immutable-anchor hashes |
| `runtime.py` | Device, precision, platform calibration and device RNG |

The original reasoner and its attention completion remain necessary parts of
the frozen anchor. The old discrete program runtime has been removed. Its
measured comparator exists only under `research/ssl-attention-2026`.

## Training and inference

**Training:** row1 complete + row2 first two panels → known sixth-panel features.
Targets come from the immutable teacher. Up to seven negatives are known sixth
panels from other training puzzles with the same layout/component. A second loss
masks one complete contour bbox in the first five panels and predicts its isolated
teacher appearance. The loss is retrieval + 0.1 completion + 0.1 masked object.
There is no bidirectional row objective or collapse regularizer.

**Inference:** row1 and row2 independently support prediction from row3's first
two panels. Three attention levels compile rank8 P/G/V updates. Dense spatial
and transport-aligned object residuals refine each prediction. Dense and object
errors rank eight candidates, with a bounded residual correction to the frozen
attention anchor. Candidates never enter support compilation or query encoding.

Contour bboxes are image priors, not verified semantic objects. The new
adaptation excludes answer candidates, labels and XML; historical anchor
pretraining used unlabeled candidate negatives. The sampled validation diagnosis
has not established that the new representation learns semantic rules.

## Commands

Run from the repository root:

```bash
python -m structured_ssl.check_contracts \
  --anchor research/ssl-attention-2026/results/seed12345/best.pt \
  --device cpu --output /tmp/structured-contracts.json

python -m structured_ssl.train \
  --dataset-root /path/to/RAVEN --run-dir runs/structured-raven8 \
  --anchor research/ssl-attention-2026/results/seed12345/best.pt \
  --device cuda:0 --epochs 8 --seed 12345

python -m structured_ssl.evaluate \
  --dataset-root /path/to/RAVEN --run-dir runs/structured-raven8 \
  --device cuda:0 --split val

python -m structured_ssl.evaluate \
  --dataset-root /path/to/RAVEN --run-dir runs/structured-raven8 \
  --device cuda:0 --split test
```

Full training requires 42,000 train and 14,000 validation puzzles. Complete
evaluation requires 2,000 puzzles per layout. Metal runs use `--device mps` and
require `--platform-validation` with the measured, immutable anchor's complete
validation record. Use unchanged arguments plus `--resume` to resume training.
Evaluation refuses to overwrite existing records or run with altered sources.

The current test admission criterion is validation-selected best ≥70%, with
actual changed answers under the new branch and an isolated support swap.
The old nonregression metric is still reported as a comparison. Controls report
declines too. No test result selects a checkpoint, architecture or seed.

## Sealed experiments

Refactoring keeps checkpoint names and CPU numerical behavior unchanged; old
training seals still require their original sources. Running jobs use independent
source exports under `runs/`. This checkout's cleanup does not modify those
exports, their objectives, checkpoints or logs. The user-directed 70% evaluation
admission was recorded separately before this revision's first test and executes
the unchanged training export. The V3 suite continues with two controls and two
further seeds; the unused V1 queue has been retired.

For full research sources and limitations, see [design](../research/structured-ssl-2026/design.md),
[v3 revision](../research/structured-ssl-2026/revision-v3.md) and
[contributions](../research/structured-ssl-2026/contributions.md).
