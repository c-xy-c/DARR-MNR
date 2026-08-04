# DARR-MNR Submission Format

A valid benchmark submission is a UTF-8 CSV file with exactly two columns:

- `sample_id`
- `prediction`

The first row must be the header:

```text
sample_id,prediction
```

Each subsequent row must contain one `.npz` filename and one integer prediction in `[0, 7]`. Every evaluated sample must appear exactly once. Duplicate sample IDs, missing sample IDs, unknown sample IDs, and out-of-range predictions are invalid.

Recommended workflow:

1. Generate a submission template from the official split.
2. Fill in model predictions.
3. Validate the file with `benchmark.py`.
4. Score the file with `benchmark_run.py`.

Template command:

```bash
python benchmark_submission.py ProbSet/test_set --output submission_template.csv
```

Scoring command:

```bash
python benchmark_run.py ProbSet/test_set --predictions submission.csv --report benchmark_report.json
```
