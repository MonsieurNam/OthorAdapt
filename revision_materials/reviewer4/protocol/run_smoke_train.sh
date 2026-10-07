#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT=/path/to/eight-dataset-root}"
: "${PYTHON:=python3}"

mkdir -p revision_materials/reviewer4/raw/logs && ${PYTHON} main.py --root_path ${DATA_ROOT} --split_reference_manifest revision_materials/reviewer4/raw/frozen_dynamic_reference.jsonl --backbone ViT-B/16 --encoder both --position all --params q k v --n_iters 500 --batch_size 32 --loss_fn ce --dataset eurosat --shots 4 --seed 1 --adapter ohsinglora --routing_mode input_independent --num_heads 2 --r 8 --alpha 1 --lambda_o 0.03 --ramp_up_steps 100 --ortho_reduction mean --save_path revision_materials/reviewer4/raw/checkpoints --filename reviewer4_eurosat_4shot_seed1_static_h2_r8_lo0p03_ramp100 --run_manifest revision_materials/reviewer4/raw/static_train_manifest.jsonl --selection_split test --report_test 2>&1 | tee revision_materials/reviewer4/raw/logs/reviewer4_eurosat_4shot_seed1_static_h2_r8_lo0p03_ramp100_train.log
