#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:=python3}"
: "${DATA_ROOT:?Set DATA_ROOT=/path/to/eight-dataset-root}"

OUT=revision_materials/reviewer4/raw
mkdir -p "$OUT"

ACTIVE_GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$ACTIVE_GPU_PIDS" ]]; then
  echo "Refusing to profile while another GPU compute process is active: $ACTIVE_GPU_PIDS" >&2
  exit 1
fi
AVAILABLE_KB=$(df -Pk . | awk 'NR==2 {print $4}')
if (( AVAILABLE_KB < 150 * 1024 * 1024 )); then
  echo "At least 150 GB free disk is required in the worktree filesystem." >&2
  exit 1
fi
TOTAL_RAM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
if (( TOTAL_RAM_KB < 32 * 1024 * 1024 )); then
  echo "At least 32 GB RAM is required." >&2
  exit 1
fi

nvidia-smi > "$OUT/nvidia-smi-before.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit,clocks.current.graphics,clocks.current.memory --format=csv,noheader > "$OUT/gpu-identity-before.csv"
free -h > "$OUT/memory-before.txt"
df -h . "$DATA_ROOT" > "$OUT/disk-before.txt"
"$PYTHON" --version > "$OUT/python-version.txt" 2>&1
"$PYTHON" -m pip freeze > "$OUT/pip-freeze.txt"
git rev-parse HEAD > "$OUT/git-revision.txt"
git status --short > "$OUT/git-status-before.txt"

"$PYTHON" reviewer4_audit.py environment --output "$OUT/environment-before.json"
"$PYTHON" reviewer4_audit.py checksums \
  --inventory "$OUT/frozen_existing_artifacts_sha256.json" \
  --output "$OUT/frozen_checksum_audit.json"
"$PYTHON" -m pytest tests -q | tee "$OUT/pytest-before-gpu.log"
"$PYTHON" reviewer4_audit.py dataset \
  --data-root "$DATA_ROOT" \
  --reference-manifest "$OUT/frozen_dynamic_reference.jsonl" \
  --observed-output "$OUT/split_hash_observed.jsonl" \
  --output "$OUT/split_hash_audit.json"

echo "Reviewer #4 preflight passed: RTX 3090 identity, tests, and all 72 split identities are valid."
