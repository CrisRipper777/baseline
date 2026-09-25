#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
device="${DEVICE:-cuda:0}"

for dataset in Movies Toys Grocery ele-fashion Reddit-S; do
  for readout in terminal uniform gpr; do
    output_dir="outputs/multi_order_bank_nc/${dataset}/${readout}"
    mkdir -p "$output_dir"
    PYTHONPATH=src conda run --no-capture-output -n yhf_env python -m src.main \
      "dataset=${dataset}" task=nc model=multi_order_bank \
      "model.readout=${readout}" seed=42 num_runs=3 "device=${device}" \
      "task.save_ckpt_path=${output_dir}/best.pt" \
      "hydra.run.dir=${output_dir}/run"
  done
done
