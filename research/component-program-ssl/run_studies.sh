#!/usr/bin/env bash
set -uo pipefail
cd /LCT-AVR/TTT/sspredrnet-runs/support-program-v2-20261003-source
experiment_root=/LCT-AVR/TTT/sspredrnet-runs/support-program-v2-20261003
mkdir -p "$experiment_root"
for study in program control program-no-ssl program-static; do
  /LCT-AVR/TTT/.venvs/sspredrnet/bin/python -u -m program_ssl.train \
    --dataset-root /LCT-AVR/TTT/datasets/RAVEN \
    --baseline sspredrnet/results/raven-20epoch/best.pt \
    --run-dir "$experiment_root/$study" --study "$study" --epochs 8 \
    > "$experiment_root/console-$study.log" 2>&1
  study_exit_code=$?
  printf '%s\n' "$study_exit_code" > "$experiment_root/exit-code-$study.txt"
  if [ "$study_exit_code" -ne 0 ]; then
    exit "$study_exit_code"
  fi
done
