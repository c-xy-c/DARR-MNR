#!/usr/bin/env bash
# Sequential fixed-budget jobs on one local Metal GPU; immutable source exports.
set -euo pipefail
if [[ $# -ne 6 ]]; then
    echo 'usage: run_metal_suite.sh SOURCE PYTHON RAVEN ANCHOR FOUNDATION RUN_ROOT' >&2
    exit 2
fi
source_dir=$1
task_python=$2
dataset_root=$3
anchor_checkpoint=$4
foundation=$5
run_root=$6
meta="$run_root/suite.launch"
mkdir "$meta"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$meta/exit_code.txt"' EXIT
export PYTHONPATH=.
adapt() {
    local variant=$1
    local seed=$2
    local work=$3
    local run="$run_root/$variant-seed$seed"
    cd "$work"
    printf '%s\n' "$variant-seed$seed:training" > "$meta/phase.txt"
    "$task_python" -u -m structured_ssl.train --dataset-root "$dataset_root" \
        --run-dir "$run" --anchor "$anchor_checkpoint" --epochs 8 --seed "$seed" \
        --device mps --workers 2 --platform-validation "$run_root/reference-anchor-validation.json" \
        > "$run.console.log" 2>&1
    printf '%s\n' "$variant-seed$seed:validation_audit" > "$meta/phase.txt"
    "$task_python" -u -m structured_ssl.evaluate --dataset-root "$dataset_root" \
        --run-dir "$run" --split val --device mps --workers 2 >> "$run.console.log" 2>&1
    if [[ "$variant" != primary ]] || "$task_python" -c \
        'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["checkpoints"]["best"]["nonregression_and_active_branch"] else 1)' \
        "$run/val_evaluation.json"; then
        printf '%s\n' "$variant-seed$seed:locked_test" > "$meta/phase.txt"
        "$task_python" -u -m structured_ssl.evaluate --dataset-root "$dataset_root" \
            --run-dir "$run" --split test --device mps --workers 2 >> "$run.console.log" 2>&1
    else
        printf '%s\n' 'Test deferred: complete validation failed nonregression/active-support gate.' \
            > "$run/test_deferred.txt"
    fi
    printf '%s\n' '0' > "$run/job_exit_code.txt"
}
adapt primary 12345 "$source_dir"
printf '%s\n' 'original:training_and_locked_evaluation' > "$meta/phase.txt"
cd "$source_dir"
"$task_python" -u research/structured-ssl-2026/original_continuation.py \
    --dataset-root "$dataset_root" --run-dir "$run_root/original-seed12345" \
    --baseline "$foundation" --epochs 16 --seed 12345 --device mps --workers 2 \
    --platform-validation "$run_root/reference-native-validation.json" \
    > "$run_root/original-seed12345.console.log" 2>&1
adapt no_object_masking 12345 "$run_root/source-no_object_masking"
adapt static_parameters 12345 "$run_root/source-static_parameters"
adapt primary 12346 "$source_dir"
adapt primary 12347 "$source_dir"
printf '%s\n' 'fixed_queue_finished; inspect validation gates and sealed results' > "$meta/phase.txt"
