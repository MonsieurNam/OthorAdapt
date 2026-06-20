#!/usr/bin/env bash
#
# Zip Phase 3B results/logs/checkpoints and upload to Google Drive every 5h.
set -euo pipefail

BASE_DIR="${BASE_DIR:-/root/OthorAdapt/revision_materials}"
RCLONE="${RCLONE:-/opt/conda/bin/rclone}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
GDRIVE_DIR="${GDRIVE_DIR:-RESEARCH/OHSinglora_CLIP/phase3b_same_param_backups}"
INTERVAL="${INTERVAL:-5h}"
STAGING_DIR="${STAGING_DIR:-/root/OthorAdapt/revision_materials/backups}"
RETENTION="${RETENTION:-12}"
REMOTE_RETENTION_DAYS="${REMOTE_RETENTION_DAYS:-7}"
INCLUDE_CHECKPOINTS="${INCLUDE_CHECKPOINTS:-1}"
CHECKPOINT_MIN_AGE_MINUTES="${CHECKPOINT_MIN_AGE_MINUTES:-10}"
STATUS_FILE="${STATUS_FILE:-${BASE_DIR}/logs/phase3b_gdrive_backup_status.env}"
LOCK_DIR="${LOCK_DIR:-${BASE_DIR}/logs/phase3b_backup_to_gdrive.lock}"

TARGETS=(results logs)
EXTRA_FILES=(
  plan/phase3b_same_param_protocol.yaml
  plan/phase3b_server_run_instructions.md
  scripts/phase3b_same_param_commands.sh
  scripts/phase3b_resumable_runner.py
  scripts/phase3b_checker_report.py
  scripts/phase3b_checker_loop.sh
)

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }

sanitize_value() {
  printf '%s' "${1:-}" | tr '\r\n' '  '
}

write_status() {
  local status="$1"
  local message="${2:-}"
  local archive="${3:-}"
  mkdir -p "$(dirname "$STATUS_FILE")"
  {
    printf 'STATUS=%s\n' "$(sanitize_value "$status")"
    printf 'TIMESTAMP_UTC=%s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    printf 'MESSAGE=%s\n' "$(sanitize_value "$message")"
    printf 'ARCHIVE=%s\n' "$(sanitize_value "$archive")"
    printf 'DESTINATION=%s\n' "$(sanitize_value "${RCLONE_REMOTE}:${GDRIVE_DIR}/")"
    printf 'INCLUDE_CHECKPOINTS=%s\n' "$(sanitize_value "$INCLUDE_CHECKPOINTS")"
  } > "${STATUS_FILE}.tmp"
  mv "${STATUS_FILE}.tmp" "$STATUS_FILE"
}

acquire_lock() {
  mkdir -p "$(dirname "$LOCK_DIR")"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "${LOCK_DIR}/pid"
    trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM
    return 0
  fi

  local old_pid
  old_pid="$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    log "Another Phase 3B backup loop is already running (pid=$old_pid); exiting."
    write_status "active" "backup loop already running (pid=$old_pid)" ""
    exit 0
  fi

  log "Removing stale backup lock: $LOCK_DIR"
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
  printf '%s\n' "$$" > "${LOCK_DIR}/pid"
  trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM
}

is_nonnegative_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

print_help() {
  cat <<'EOF'
Usage:
  phase3b_backup_to_gdrive.sh once
  phase3b_backup_to_gdrive.sh loop

Server-side Phase 3B backup loop. It creates timestamped zip snapshots under
STAGING_DIR and uploads them to Google Drive via rclone.

Default backup contents:
  revision_materials/results
  revision_materials/logs
  revision_materials/checkpoints files older than CHECKPOINT_MIN_AGE_MINUTES
  revision_materials/plan/phase3b_same_param_protocol.yaml
  revision_materials/plan/phase3b_server_run_instructions.md
  revision_materials/scripts/phase3b_same_param_commands.sh
  revision_materials/scripts/phase3b_resumable_runner.py
  revision_materials/scripts/phase3b_checker_report.py
  revision_materials/scripts/phase3b_checker_loop.sh

Recommended tmux loop:
  tmux new-session -d -s phase3b_backup
  tmux send-keys -t phase3b_backup 'cd /root/OthorAdapt && RCLONE_REMOTE=gdrive GDRIVE_DIR=RESEARCH/OHSinglora_CLIP/phase3b_same_param_backups INCLUDE_CHECKPOINTS=1 bash revision_materials/scripts/phase3b_backup_to_gdrive.sh loop' C-m
EOF
}

make_zip() {
  local stamp zip_path checksum_path filelist existing=()
  stamp="$(date '+%Y%m%d_%H%M%S')"
  zip_path="${STAGING_DIR}/phase3b_backup_${stamp}.zip"
  checksum_path="${zip_path}.sha256"
  filelist="${STAGING_DIR}/phase3b_backup_${stamp}.files"
  mkdir -p "$STAGING_DIR"

  for t in "${TARGETS[@]}"; do
    [[ -e "${BASE_DIR}/${t}" ]] && existing+=("$t")
  done
  for t in "${EXTRA_FILES[@]}"; do
    [[ -e "${BASE_DIR}/${t}" ]] && existing+=("$t")
  done

  : > "$filelist"
  if [[ "$INCLUDE_CHECKPOINTS" == "1" && -d "${BASE_DIR}/checkpoints/phase3b_same_param" ]]; then
    log "Checkpoint backup enabled; including Phase 3B checkpoint files older than ${CHECKPOINT_MIN_AGE_MINUTES} minute(s)."
    (
      cd "$BASE_DIR"
      if [[ "$CHECKPOINT_MIN_AGE_MINUTES" -eq 0 ]]; then
        find checkpoints/phase3b_same_param -type f -print
      else
        find checkpoints/phase3b_same_param -type f -mmin +"$CHECKPOINT_MIN_AGE_MINUTES" -print
      fi
    ) >> "$filelist"
  fi

  if [[ ${#existing[@]} -eq 0 && ! -s "$filelist" ]]; then
    log "ERROR: no backup targets exist under ${BASE_DIR}; skipping."
    rm -f "$filelist"
    return 1
  fi

  log "Zipping [${existing[*]:-none}] -> ${zip_path}"
  (
    cd "$BASE_DIR"
    if [[ ${#existing[@]} -gt 0 ]]; then
      zip -r -q "$zip_path" "${existing[@]}"
    fi
    if [[ -s "$filelist" ]]; then
      zip -q "$zip_path" -@ < "$filelist"
    fi
  )
  rm -f "$filelist"
  ( cd "$(dirname "$zip_path")" && sha256sum "$(basename "$zip_path")" > "$(basename "$checksum_path")" )
  log "Created $(du -h "$zip_path" | cut -f1) zip"
  log "Created checksum $(basename "$checksum_path")"
  printf '%s\n' "$zip_path"
}

upload_zip() {
  local zip_path="$1"
  local zip_name
  zip_name="$(basename "$zip_path")"
  log "Uploading to ${RCLONE_REMOTE}:${GDRIVE_DIR}/"
  "$RCLONE" copy "$(dirname "$zip_path")" "${RCLONE_REMOTE}:${GDRIVE_DIR}/" \
    --filter "+ ${zip_name}" --filter "+ ${zip_name}.sha256" --filter "- *" \
    --transfers 1 --retries 5 --low-level-retries 10 --stats-one-line
  log "Upload OK: $zip_name"
}

prune_local() {
  [[ "$RETENTION" -gt 0 ]] || return 0
  local files
  mapfile -t files < <(ls -1t "${STAGING_DIR}"/phase3b_backup_*.zip 2>/dev/null || true)
  if [[ ${#files[@]} -gt $RETENTION ]]; then
    for f in "${files[@]:$RETENTION}"; do
      log "Pruning old local zip: $(basename "$f")"
      rm -f "$f" "${f}.sha256"
    done
  fi
}

prune_remote() {
  [[ "$REMOTE_RETENTION_DAYS" -gt 0 ]] || return 0
  log "Pruning remote backups older than ${REMOTE_RETENTION_DAYS} day(s)."
  "$RCLONE" delete "${RCLONE_REMOTE}:${GDRIVE_DIR}/" \
    --filter '+ phase3b_backup_*.zip' \
    --filter '+ phase3b_backup_*.zip.sha256' \
    --filter '- *' \
    --min-age "${REMOTE_RETENTION_DAYS}d" \
    --retries 5 --low-level-retries 10
}

run_once() {
  local zip_path
  write_status "running" "backup cycle started" ""
  if ! zip_path="$(make_zip)"; then
    write_status "failure" "zip creation failed" ""
    return 1
  fi
  if ! upload_zip "$zip_path"; then
    write_status "failure" "upload failed" "$zip_path"
    return 1
  fi
  prune_local || { write_status "failure" "local pruning failed after upload" "$zip_path"; return 1; }
  prune_remote || { write_status "failure" "remote pruning failed after upload" "$zip_path"; return 1; }
  write_status "success" "upload ok" "$zip_path"
}

preflight() {
  is_nonnegative_int "$RETENTION" || { log "ERROR: RETENTION must be a non-negative integer, got '$RETENTION'."; exit 1; }
  is_nonnegative_int "$REMOTE_RETENTION_DAYS" || { log "ERROR: REMOTE_RETENTION_DAYS must be a non-negative integer, got '$REMOTE_RETENTION_DAYS'."; exit 1; }
  is_nonnegative_int "$CHECKPOINT_MIN_AGE_MINUTES" || { log "ERROR: CHECKPOINT_MIN_AGE_MINUTES must be a non-negative integer, got '$CHECKPOINT_MIN_AGE_MINUTES'."; exit 1; }
  [[ "$INCLUDE_CHECKPOINTS" == "0" || "$INCLUDE_CHECKPOINTS" == "1" ]] || { log "ERROR: INCLUDE_CHECKPOINTS must be 0 or 1, got '$INCLUDE_CHECKPOINTS'."; exit 1; }
  command -v zip >/dev/null || { log "ERROR: zip not found in PATH"; exit 1; }
  command -v sha256sum >/dev/null || { log "ERROR: sha256sum not found in PATH"; exit 1; }
  command -v "$RCLONE" >/dev/null || { log "ERROR: rclone not found at $RCLONE"; exit 1; }
  if ! "$RCLONE" listremotes 2>/dev/null | grep -qx "${RCLONE_REMOTE}:"; then
    log "ERROR: rclone remote '${RCLONE_REMOTE}:' not configured."
    exit 1
  fi
}

case "${1:-loop}" in
  --help|-h)
    print_help; exit 0 ;;
  once)
    acquire_lock; preflight; run_once ;;
  loop)
    acquire_lock
    preflight
    log "Starting Phase 3B backup loop (interval=${INTERVAL}). include_checkpoints=${INCLUDE_CHECKPOINTS}"
    while true; do
      run_once || log "Run failed; will retry next cycle."
      log "Sleeping ${INTERVAL}..."
      sleep "$INTERVAL"
    done ;;
  *)
    log "Unknown command: $1 (use: once | loop | --help)"; exit 1 ;;
esac
