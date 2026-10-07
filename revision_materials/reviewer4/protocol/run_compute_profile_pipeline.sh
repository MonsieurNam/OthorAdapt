#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:=python3}"
: "${DATA_ROOT:?Set DATA_ROOT=/path/to/eight-dataset-root}"

RAW=revision_materials/reviewer4/raw/profile
OUT=revision_materials/reviewer4/analysis
PAPER=revision_materials/reviewer4/paper_ready
mkdir -p "$RAW" "$OUT" "$PAPER"

for existing in "$RAW/compute_profile_train_manifest.jsonl" "$RAW/compute_microprofile.jsonl"; do
  if [[ -e "$existing" ]]; then
    echo "Refusing to append duplicate profiling evidence; archive or explicitly remove $existing first." >&2
    exit 1
  fi
done

bash revision_materials/reviewer4/protocol/run_compute_profile_training.sh
"$PYTHON" reviewer4_audit.py profile-environment \
  --profile-jsonl "$RAW/compute_profile_train_manifest.jsonl" \
  --output "$RAW/compute_profile_environment_audit.json"

bash revision_materials/reviewer4/protocol/run_compute_profile_benchmarks.sh

"$PYTHON" reviewer4_profile_compute.py merge \
  --training-manifest "$RAW/compute_profile_train_manifest.jsonl" \
  --micro-jsonl "$RAW/compute_microprofile.jsonl" \
  --raw-csv "$OUT/compute_profile_raw.csv" \
  --summary-csv "$OUT/compute_summary.csv" \
  --overhead-csv "$OUT/gating_overhead.csv" \
  --flops-csv "$OUT/flops_breakdown.csv" \
  --table-tex "$PAPER/computational_efficiency_table.tex"

nvidia-smi > revision_materials/reviewer4/raw/nvidia-smi-after-profiling.txt
"$PYTHON" -m pip freeze > revision_materials/reviewer4/raw/pip-freeze-after-profiling.txt
"$PYTHON" reviewer4_audit.py environment \
  --output revision_materials/reviewer4/raw/environment-after-profiling.json

echo "Compute profile pipeline passed and paper-ready table was generated."
