#!/usr/bin/env bash
# Install/remove @reboot cron entries for Phase 3 checker and Google Drive backup.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/OthorAdapt}"
DATA_ROOT="${DATA_ROOT:-/root/DATA}"
PYTHON="${PYTHON:-/opt/conda/bin/python}"
COST_PER_HOUR_VND="${COST_PER_HOUR_VND:-5000}"
RCLONE="${RCLONE:-/opt/conda/bin/rclone}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
GDRIVE_DIR="${GDRIVE_DIR:-RESEARCH/OHSinglora_CLIP/phase3_ramp100_backups}"
BACKUP_INTERVAL="${BACKUP_INTERVAL:-5h}"
INCLUDE_CHECKPOINTS="${INCLUDE_CHECKPOINTS:-1}"
REMOTE_RETENTION_DAYS="${REMOTE_RETENTION_DAYS:-7}"
PHASE3_COMMANDS="${PHASE3_COMMANDS:-revision_materials/scripts/phase3_main_commands.sh}"
PHASE3_MANIFEST="${PHASE3_MANIFEST:-revision_materials/results/phase3_main_ramp100_results.jsonl}"
PHASE3_RUNTIME_SOURCE="${PHASE3_RUNTIME_SOURCE:-revision_materials/results/validation_sweep_ramp100_results.jsonl}"
PHASE3_RUN_LABEL="${PHASE3_RUN_LABEL:-phase3_main_ramp100}"
PHASE3_TMUX_SESSION="${PHASE3_TMUX_SESSION:-run_ablation}"

BEGIN_MARKER="# OTHORADAPT_PHASE3_CRON_BEGIN"
END_MARKER="# OTHORADAPT_PHASE3_CRON_END"

print_help() {
  cat <<'EOF'
Usage:
  phase3_install_cron.sh install
  phase3_install_cron.sh remove
  phase3_install_cron.sh show

Installs @reboot cron entries that restart these loops after a server reboot:
  1. revision_materials/scripts/phase3_checker_loop.sh
  2. revision_materials/scripts/phase3_backup_to_gdrive.sh loop

The loops have their own lock directories, so re-running this installer or
starting a loop manually will not create duplicate long-running jobs.

Requires a working `crontab` and a running cron daemon. Containers without an
init system may not start cron at boot, so verify @reboot actually fires.

Common overrides:
  PROJECT_ROOT=/root/OthorAdapt
  DATA_ROOT=/root/DATA
  PYTHON=/opt/conda/bin/python
  RCLONE=/opt/conda/bin/rclone
  RCLONE_REMOTE=gdrive
  GDRIVE_DIR=RESEARCH/OHSinglora_CLIP/phase3_ramp100_backups
  BACKUP_INTERVAL=5h
  INCLUDE_CHECKPOINTS=1
  REMOTE_RETENTION_DAYS=7
  PHASE3_COMMANDS=revision_materials/scripts/phase3_main_commands.sh
  PHASE3_MANIFEST=revision_materials/results/phase3_main_ramp100_results.jsonl
  PHASE3_RUNTIME_SOURCE=revision_materials/results/validation_sweep_ramp100_results.jsonl
  PHASE3_RUN_LABEL=phase3_main_ramp100
  PHASE3_TMUX_SESSION=run_ablation

For OH-only rerun:
  PHASE3_COMMANDS=revision_materials/scripts/phase3_main_oh_ramp100_commands.sh
  PHASE3_RUN_LABEL=phase3_main_ramp100
EOF
}

checker_cron_line() {
  printf '@reboot cd %s && mkdir -p revision_materials/logs && PROJECT_ROOT=%s DATA_ROOT=%s PYTHON=%s COST_PER_HOUR_VND=%s PHASE3_COMMANDS=%s PHASE3_MANIFEST=%s PHASE3_RUNTIME_SOURCE=%s PHASE3_RUN_LABEL=%s PHASE3_TMUX_SESSION=%s CHECKER_LOCK_DIR=%s/revision_materials/logs/phase3_checker_loop.lock GDRIVE_BACKUP_STATUS_FILE=%s/revision_materials/logs/phase3_gdrive_backup_status.env bash revision_materials/scripts/phase3_checker_loop.sh >> revision_materials/logs/phase3_checker_cron.log 2>&1\n' \
    "$PROJECT_ROOT" "$PROJECT_ROOT" "$DATA_ROOT" "$PYTHON" "$COST_PER_HOUR_VND" "$PHASE3_COMMANDS" "$PHASE3_MANIFEST" "$PHASE3_RUNTIME_SOURCE" "$PHASE3_RUN_LABEL" "$PHASE3_TMUX_SESSION" "$PROJECT_ROOT" "$PROJECT_ROOT"
}

backup_cron_line() {
  printf '@reboot cd %s && mkdir -p revision_materials/logs && BASE_DIR=%s/revision_materials RCLONE=%s RCLONE_REMOTE=%s GDRIVE_DIR=%s INTERVAL=%s INCLUDE_CHECKPOINTS=%s REMOTE_RETENTION_DAYS=%s STATUS_FILE=%s/revision_materials/logs/phase3_gdrive_backup_status.env LOCK_DIR=%s/revision_materials/logs/phase3_backup_to_gdrive.lock bash revision_materials/scripts/phase3_backup_to_gdrive.sh loop >> revision_materials/logs/phase3_backup_cron.log 2>&1\n' \
    "$PROJECT_ROOT" "$PROJECT_ROOT" "$RCLONE" "$RCLONE_REMOTE" "$GDRIVE_DIR" "$BACKUP_INTERVAL" "$INCLUDE_CHECKPOINTS" "$REMOTE_RETENTION_DAYS" "$PROJECT_ROOT" "$PROJECT_ROOT"
}

without_managed_block() {
  sed "/^${BEGIN_MARKER}$/,/^${END_MARKER}$/d"
}

install_cron() {
  command -v crontab >/dev/null || { echo "ERROR: crontab not found; install cron first (apt-get install -y cron)." >&2; exit 1; }
  current="$(mktemp)"
  next="$(mktemp)"
  crontab -l > "$current" 2>/dev/null || true
  without_managed_block < "$current" > "$next"
  {
    printf '%s\n' "$BEGIN_MARKER"
    checker_cron_line
    backup_cron_line
    printf '%s\n' "$END_MARKER"
  } >> "$next"
  crontab "$next"
  rm -f "$current" "$next"
  echo "Installed Phase 3 @reboot cron entries."
}

remove_cron() {
  command -v crontab >/dev/null || { echo "ERROR: crontab not found." >&2; exit 1; }
  current="$(mktemp)"
  next="$(mktemp)"
  crontab -l > "$current" 2>/dev/null || true
  without_managed_block < "$current" > "$next"
  crontab "$next"
  rm -f "$current" "$next"
  echo "Removed Phase 3 managed cron entries."
}

show_cron() {
  echo "$BEGIN_MARKER"
  checker_cron_line
  backup_cron_line
  echo "$END_MARKER"
}

case "${1:-install}" in
  install)
    install_cron ;;
  remove)
    remove_cron ;;
  show)
    show_cron ;;
  --help|-h)
    print_help ;;
  *)
    echo "Unknown command: $1 (use: install | remove | show | --help)" >&2
    exit 1 ;;
esac
