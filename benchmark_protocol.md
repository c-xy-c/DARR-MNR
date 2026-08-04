# DARR-MNR Benchmark Protocol

## 1. Scope

DARR-MNR evaluates machine number reasoning from three context panels and an eight-way answer panel. A submission receives the three context images and eight candidate images and returns one integer in `[0, 7]` for each sample.

This protocol applies only to the original DARR-MNR task. Experimental dataset variants are outside the official benchmark until separately versioned and documented.

## 2. Dataset contract

Each sample is an `.npz` file containing:

- `context_images`: exactly three context panels.
- `answer_set_images`: exactly eight candidate panels.
- `correct_answer_image_index`: the correct candidate index in `[0, 7]`.

The official evaluation script validates the shapes, label range, finite numeric values, and candidate count before scoring. Files that fail validation are rejected rather than silently skipped.

The recommended directory layout is:

```text
ProbSet/
?????? train_set/
?????? val_set/
?????? test_set/
```

A submission must not use test labels during model selection, threshold tuning, or checkpoint selection.

## 3. Submission contract

A prediction file is UTF-8 CSV with a header and one row per evaluated sample:

```text
sample_id,prediction
prob_000001.npz,3
```

`sample_id` is the exact `.npz` filename and `prediction` is an integer from `0` to `7`. Every dataset sample must occur exactly once, and no unknown sample may occur.

## 4. Official metric

The primary metric is eight-way answer accuracy:

\[
\mathrm{Accuracy} = \frac{1}{N}\sum_{i=1}^{N} 1[\hat y_i = y_i].
\]

The official report also includes the number of evaluated samples, correct predictions, chance accuracy (`1/8 = 0.125`), and prediction coverage. Accuracy is reported to six decimal places.

## 5. Baselines and reporting

Every reported result should state:

- DARR-MNR protocol version.
- Dataset split and number of samples.
- Model and checkpoint identifier.
- Whether external pretraining or additional data were used.
- Random seed(s).
- Mean and standard deviation over repeated runs when applicable.

At minimum, a benchmark release should report a random eight-way baseline, a deterministic heuristic baseline if one exists, the DARR model, and DARR ablations.

## 6. Reproducibility requirements

A valid release should provide the exact data-generation command or immutable dataset archive, environment requirements, training command, evaluation command, checkpoint metadata, and machine-readable prediction files. Changes to sample generation, split assignment, or label semantics require a new protocol version.

For the DARR model handoff, the model owner must provide the training entrypoint, inference entrypoint, checkpoint information, environment details, seed, external-data policy, and predictions in the submission format. See `docs_handoff_to_darr_owner.md` for the complete handoff checklist.

## 7. Limitations

This protocol standardizes the current data format and evaluation procedure; it does not by itself establish a new state-of-the-art result. Claims about benchmark superiority require controlled comparisons against published or reimplemented baselines under the same split and resource policy.
