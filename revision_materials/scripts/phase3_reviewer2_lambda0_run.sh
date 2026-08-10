#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT=/path/to/datasets}"
: "${PYTHON:=python3}"

MODE="${1:-all}"
MAX_RUNS_ARGS=()
if [[ -n "${MAX_RUNS:-}" ]]; then
  MAX_RUNS_ARGS=(--max-runs "${MAX_RUNS}")
fi

RUNNER="revision_materials/scripts/phase3_resumable_runner.py"
TRAIN_COMMANDS="revision_materials/scripts/phase3_reviewer2_lambda0_train_commands.sh"
TRAIN_MANIFEST="revision_materials/results/phase3_reviewer2_lambda0_train_results.jsonl"
EVAL_COMMANDS="revision_materials/scripts/phase3_reviewer2_lambda0_eval_commands.sh"
EVAL_MANIFEST="revision_materials/results/phase3_reviewer2_lambda0_eval_results.jsonl"
RUNTIME_SOURCE="revision_materials/results/phase3_reviewer2_singlora_train_results.jsonl"

run_phase() {
  local commands="$1"
  local manifest="$2"
  local runtime_args=()
  if [[ -f "${RUNTIME_SOURCE}" ]]; then
    runtime_args=(--runtime-source "${RUNTIME_SOURCE}")
  fi
  "${PYTHON}" "${RUNNER}" \
    --commands "${commands}" \
    --manifest "${manifest}" \
    "${runtime_args[@]}" \
    "${MAX_RUNS_ARGS[@]}"
}

case "${MODE}" in
  train)
    run_phase "${TRAIN_COMMANDS}" "${TRAIN_MANIFEST}"
    ;;
  eval)
    run_phase "${EVAL_COMMANDS}" "${EVAL_MANIFEST}"
    ;;
  all)
    run_phase "${TRAIN_COMMANDS}" "${TRAIN_MANIFEST}"
    run_phase "${EVAL_COMMANDS}" "${EVAL_MANIFEST}"
    ;;
  *)
    echo "Usage: $0 [train|eval|all]" >&2
    exit 2
    ;;
esac
