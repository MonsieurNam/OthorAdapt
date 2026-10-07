#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:=python3}"

RAW=revision_materials/reviewer4/raw
ANALYSIS=revision_materials/reviewer4/analysis
PAPER=revision_materials/reviewer4/paper_ready
DYNAMIC_REFERENCE=$(tr -d '\r\n' < "$RAW/selected_dynamic_reference_path.txt")

mkdir -p "$ANALYSIS" "$PAPER"
"$PYTHON" reviewer4_analyze_evidence.py \
  --dynamic-manifest "$DYNAMIC_REFERENCE" \
  --static-manifest "$RAW/static_eval_manifest.jsonl" \
  --output-dir "$ANALYSIS" \
  --paper-dir "$PAPER" \
  --bootstrap-replicates 100000 \
  --bootstrap-seed 20261007

echo "Reviewer #4 statistical analysis and paper-ready artifacts were generated."
