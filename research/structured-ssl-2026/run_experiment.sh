#!/usr/bin/env bash
# Run inside a verified idle-GPU tmux session. No SSH credentials are handled.
set -euo pipefail
if [[ $# -ne 6 ]]; then
    echo 'usage: run_experiment.sh SOURCE PYTHON RAVEN ANCHOR RUN_DIR SEED' >&2
    exit 2
fi
: "${CUDA_VISIBLE_DEVICES:?Set a single verified idle GPU before launch}"
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
    echo 'CUDA_VISIBLE_DEVICES must identify one verified idle GPU' >&2
    exit 2
fi
source_dir=$1
task_python=$2
dataset_root=$3
anchor_checkpoint=$4
run_dir=$5
task_seed=$6
run_meta="${run_dir}.launch"
mkdir "$run_meta"
exec > >(tee "$run_meta/console.log") 2>&1
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$run_meta/exit_code.txt"' EXIT
cd "$source_dir"
printf 'preflight\n' > "$run_meta/phase.txt"
"$task_python" -m structured_ssl.check_contracts --anchor "$anchor_checkpoint" \
    --device cuda:0 --output "$run_meta/cuda_contracts.json"
printf 'training\n' > "$run_meta/phase.txt"
"$task_python" -m structured_ssl.train --dataset-root "$dataset_root" --run-dir "$run_dir" \
    --anchor "$anchor_checkpoint" --epochs 8 --seed "$task_seed"
printf 'validation_audit\n' > "$run_meta/phase.txt"
"$task_python" -m structured_ssl.evaluate --dataset-root "$dataset_root" --run-dir "$run_dir" --split val
printf 'locked_test\n' > "$run_meta/phase.txt"
# Main evaluator requires complete validation nonregression and a real active
# support branch. Isolated preregistered controls use their sealed full-val gate.
"$task_python" -m structured_ssl.evaluate --dataset-root "$dataset_root" --run-dir "$run_dir" --split test
printf 'complete\n' > "$run_meta/phase.txt"
