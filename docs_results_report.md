# DARR-MNR Results Report Format

A benchmark run should be archived with a machine-readable JSON report and the exact prediction CSV that produced it.

## Required report fields

- `protocol_version`: benchmark protocol version string.
- `split_dir`: evaluated split path.
- `num_samples`: number of evaluated samples.
- `correct`: number of correct predictions.
- `accuracy`: top-1 accuracy.
- `chance_accuracy`: expected random baseline accuracy.
- `coverage`: fraction of evaluated samples that received predictions.
- `prediction_file`: path to the evaluated CSV, when applicable.
- `runner`: tool name or version used to produce the report.

## Recommended report workflow

1. Produce a prediction CSV.
2. Validate the CSV and dataset with `benchmark.py`.
3. Score the split with `benchmark_run.py`.
4. Store the JSON report together with the CSV and the model checkpoint metadata.

## Suggested archive layout

```text
runs/
  2026-08-04_darr_base/
    checkpoint.json
    predictions.csv
    benchmark_report.json
    stdout.log
```

## Reporting rules

- Report the exact protocol version used for the run.
- Report the exact split name and sample count.
- Do not mix predictions from different checkpoints in one report.
- Do not report a benchmark score without the associated prediction CSV.
- If repeated runs are used, report mean and standard deviation.
- The DARR model owner must include checkpoint, seed, environment, external-data policy, and inference command in the accompanying run metadata.

For the complete model-owner handoff procedure, see `docs_handoff_to_darr_owner.md`.
