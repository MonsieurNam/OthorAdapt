#!/usr/bin/env bash
#
# Watch phase3b_progress_notes.md and send the newest "## " section to Telegram
# whenever the file changes.
set -euo pipefail

NOTES_FILE="${NOTES_FILE:-/root/OthorAdapt/revision_materials/logs/phase3b_progress_notes.md}"
CONFIG_FILE="${CONFIG_FILE:-/root/OthorAdapt/revision_materials/scripts/telegram.conf}"
STATE_FILE="${STATE_FILE:-/root/OthorAdapt/revision_materials/logs/.phase3b_notes_telegram.state}"
POLL="${POLL:-30}"
LOCK_DIR="${LOCK_DIR:-/root/OthorAdapt/revision_materials/logs/phase3b_notes_telegram.lock}"

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }

print_help() {
  cat <<'EOF'
Usage:
  phase3b_notes_telegram.sh watch
  phase3b_notes_telegram.sh send
  phase3b_notes_telegram.sh test

Watch phase3b_progress_notes.md and send the newest "## " section to Telegram
whenever the note changes.

Required config file:
  /root/OthorAdapt/revision_materials/scripts/telegram.conf

Recommended tmux loop:
  tmux new-session -d -s phase3b_telegram
  tmux send-keys -t phase3b_telegram 'cd /root/OthorAdapt && bash revision_materials/scripts/phase3b_notes_telegram.sh watch >> revision_materials/logs/phase3b_notes_telegram.log 2>&1' C-m
EOF
}

if [[ "${1:-watch}" == "--help" || "${1:-watch}" == "-h" ]]; then
  print_help
  exit 0
fi

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
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
    log "Phase 3B Telegram notes watcher already running (pid=$old_pid); exiting."
    exit 0
  fi

  log "Removing stale Telegram watcher lock: $LOCK_DIR"
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
  printf '%s\n' "$$" > "${LOCK_DIR}/pid"
  trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM
}

[[ -n "$POLL" ]] && is_positive_int "$POLL" || { log "ERROR: POLL must be a positive integer, got '$POLL'."; exit 1; }
command -v curl >/dev/null || { log "ERROR: curl not found in PATH"; exit 1; }
command -v md5sum >/dev/null || { log "ERROR: md5sum not found in PATH"; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { log "ERROR: config not found: $CONFIG_FILE"; exit 1; }
mkdir -p "$(dirname "$STATE_FILE")"
# shellcheck disable=SC1090
source "$CONFIG_FILE"
: "${TELEGRAM_TOKEN:?TELEGRAM_TOKEN missing in $CONFIG_FILE}"
: "${CHAT_ID:?CHAT_ID missing in $CONFIG_FILE}"

latest_section() {
  [[ -f "$NOTES_FILE" ]] || return 0
  awk '
    /^## / { buf=$0"\n"; cap=1; next }
    cap    { buf=buf $0"\n" }
    END    { printf "%s", buf }
  ' "$NOTES_FILE"
}

send_telegram() {
  local text="$1" resp
  resp=$(curl -fsS -X POST \
    "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${CHAT_ID}" \
    --data-urlencode "text=${text}" \
    --data-urlencode "disable_web_page_preview=true" 2>&1) || {
      log "ERROR sending to Telegram: $resp"; return 1; }
  log "Sent to Telegram (chat ${CHAT_ID})."
}

notify_if_changed() {
  local section hash prev_hash
  section="$(latest_section)"
  [[ -n "$section" ]] || return 0
  hash="$(printf '%s' "$section" | md5sum | cut -d' ' -f1)"
  prev_hash="$(cat "$STATE_FILE" 2>/dev/null || true)"
  if [[ "$hash" != "$prev_hash" ]]; then
    if send_telegram "[Phase3B] progress (latest):

${section}"; then
      printf '%s' "$hash" > "$STATE_FILE"
    fi
  fi
}

case "${1:-watch}" in
  --help|-h) exit 0 ;;
  test)      send_telegram "[OK] Test: Telegram phase3b notes connection OK." ;;
  send)      notify_if_changed ;;
  watch)
    acquire_lock
    log "Watching $NOTES_FILE (poll=${POLL}s). Sends newest section on change."
    local_last=""
    while true; do
      if [[ -f "$NOTES_FILE" ]]; then
        cur="$(stat -c %Y "$NOTES_FILE" 2>/dev/null || echo 0)"
        if [[ "$cur" != "$local_last" ]]; then
          local_last="$cur"
          notify_if_changed || true
        fi
      fi
      sleep "$POLL"
    done ;;
  *) log "Unknown command: $1 (use: watch | send | test | --help)"; exit 1 ;;
esac
