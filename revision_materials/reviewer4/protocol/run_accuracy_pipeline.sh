#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:=python3}"
: "${DATA_ROOT:?Set DATA_ROOT=/path/to/eight-dataset-root}"

P=revision_materials/reviewer4/protocol/reviewer4_gating_protocol.yaml
RAW=revision_materials/reviewer4/raw
GEN="$RAW/generated_commands"
mkdir -p "$GEN"

# 1. Static smoke: train, strict reload, and <=0.05 pp consistency.
"$PYTHON" reviewer4_gating_experiment.py --protocol "$P" --matrix smoke --stage train \
  --resume-manifest "$RAW/static_train_manifest.jsonl" --out "$GEN/smoke_train.sh"
bash "$GEN/smoke_train.sh"
"$PYTHON" reviewer4_gating_experiment.py --protocol "$P" --matrix smoke --stage eval \
  --resume-manifest "$RAW/static_eval_manifest.jsonl" --out "$GEN/smoke_eval.sh"
bash "$GEN/smoke_eval.sh"
"$PYTHON" reviewer4_audit.py smoke \
  --train-manifest "$RAW/static_train_manifest.jsonl" \
  --eval-manifest "$RAW/static_eval_manifest.jsonl" \
  --output "$RAW/smoke_audit.json"

# 2. Six dynamic sentinels. A failed gate selects the fresh-72 fallback.
"$PYTHON" reviewer4_gating_experiment.py --protocol "$P" --matrix sentinel --stage train \
  --resume-manifest "$RAW/dynamic_new_train_manifest.jsonl" --out "$GEN/dynamic_sentinel_train.sh"
bash "$GEN/dynamic_sentinel_train.sh"
"$PYTHON" reviewer4_gating_experiment.py --protocol "$P" --matrix sentinel --stage eval \
  --resume-manifest "$RAW/dynamic_new_eval_manifest.jsonl" --out "$GEN/dynamic_sentinel_eval.sh"
bash "$GEN/dynamic_sentinel_eval.sh"
if "$PYTHON" reviewer4_audit.py sentinel \
  --train-manifest "$RAW/dynamic_new_train_manifest.jsonl" \
  --eval-manifest "$RAW/dynamic_new_eval_manifest.jsonl" \
  --reference-manifest "$RAW/frozen_dynamic_reference.jsonl" \
  --output "$RAW/dynamic_sentinel_audit.json"; then
  DYNAMIC_REFERENCE="$RAW/frozen_dynamic_reference.jsonl"
  echo "Dynamic sentinel passed; retaining the frozen 72-run RTX 3090 reference."
else
  echo "Dynamic sentinel failed; producing the remaining 66 fresh dynamic runs."
  "$PYTHON" reviewer4_gating_experiment.py --protocol "$P" --matrix dynamic --stage train \
    --resume-manifest "$RAW/dynamic_new_train_manifest.jsonl" --out "$GEN/dynamic_remaining_train.sh"
  bash "$GEN/dynamic_remaining_train.sh"
  "$PYTHON" reviewer4_gating_experiment.py --protocol "$P" --matrix dynamic --stage eval \
    --resume-manifest "$RAW/dynamic_new_eval_manifest.jsonl" --out "$GEN/dynamic_remaining_eval.sh"
  bash "$GEN/dynamic_remaining_eval.sh"
  "$PYTHON" reviewer4_audit.py matrix \
    --train-manifest "$RAW/dynamic_new_train_manifest.jsonl" \
    --eval-manifest "$RAW/dynamic_new_eval_manifest.jsonl" \
    --routing-mode input_conditioned \
    --verify-checkpoint-files \
    --checkpoint-inventory "$RAW/dynamic_checkpoint_sha_inventory.json" \
    --output "$RAW/dynamic_full_coverage_audit.json"
  DYNAMIC_REFERENCE="$RAW/dynamic_new_eval_manifest.jsonl"
fi
printf '%s\n' "$DYNAMIC_REFERENCE" > "$RAW/selected_dynamic_reference_path.txt"

# 3. Static 72-run matrix, resuming past the completed smoke row.
"$PYTHON" reviewer4_gating_experiment.py --protocol "$P" --matrix static --stage train \
  --resume-manifest "$RAW/static_train_manifest.jsonl" --out "$GEN/static_remaining_train.sh"
bash "$GEN/static_remaining_train.sh"
"$PYTHON" reviewer4_gating_experiment.py --protocol "$P" --matrix static --stage eval \
  --resume-manifest "$RAW/static_eval_manifest.jsonl" --out "$GEN/static_remaining_eval.sh"
bash "$GEN/static_remaining_eval.sh"

"$PYTHON" reviewer4_audit.py matrix \
  --train-manifest "$RAW/static_train_manifest.jsonl" \
  --eval-manifest "$RAW/static_eval_manifest.jsonl" \
  --verify-checkpoint-files \
  --checkpoint-inventory "$RAW/checkpoint_sha_inventory.json" \
  --output "$RAW/coverage_audit.json"

echo "Accuracy pipeline passed. Dynamic evidence: $DYNAMIC_REFERENCE"
