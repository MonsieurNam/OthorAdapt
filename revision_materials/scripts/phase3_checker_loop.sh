#!/usr/bin/env bash
# Phase 3 checker loop.
#
# Every CHECK_INTERVAL_SECONDS, verify whether a Phase 3 train/runner process is
# still active. If none is active but the manifest still has pending runs, resume
# via phase3_resumable_runner.py (it skips completed runs and reruns incomplete
# commands). A progress note is appended every NOTE_EVERY cycles (default 30min
# x 2 = 1h), and always whenever a resume happens. When all runs are completed,
# write one final note and exit so the loop does not spam "completed" forever.
set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/OthorAdapt}
DATA_ROOT=${DATA_ROOT:-/root/DATA}
PYTHON=${PYTHON:-/opt/conda/bin/python}
CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-1800}   # check every 30 minutes
NOTE_EVERY=${NOTE_EVERY:-2}                              # note every 2 cycles = 1h
case "$NOTE_EVERY" in ''|*[!0-9]*) NOTE_EVERY=1 ;; esac  # must be a positive int
[ "$NOTE_EVERY" -lt 1 ] && NOTE_EVERY=1                  # guard div-by-zero in modulo
COST_PER_HOUR_VND=${COST_PER_HOUR_VND:-5000}
CHECKER_LOCK_DIR=${CHECKER_LOCK_DIR:-/root/OthorAdapt/revision_materials/logs/phase3_checker_loop.lock}
GDRIVE_BACKUP_STATUS_FILE=${GDRIVE_BACKUP_STATUS_FILE:-/root/OthorAdapt/revision_materials/logs/phase3_gdrive_backup_status.env}

cd "$PROJECT_ROOT"
export DATA_ROOT
export PYTHON

NOTE="$PROJECT_ROOT/revision_materials/logs/phase3_progress_notes.md"
RUNNER="revision_materials/scripts/phase3_resumable_runner.py"
REPORT="revision_materials/scripts/phase3_checker_report.py"

mkdir -p "$(dirname "$NOTE")"
if [ ! -f "$NOTE" ]; then
  echo "# Phase 3 progress notes (check every 30min, note every 1h)" > "$NOTE"
fi

acquire_checker_lock() {
  mkdir -p "$(dirname "$CHECKER_LOCK_DIR")"
  if mkdir "$CHECKER_LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "${CHECKER_LOCK_DIR}/pid"
    trap 'rm -rf "$CHECKER_LOCK_DIR"' EXIT INT TERM
    return 0
  fi

  old_pid=$(cat "${CHECKER_LOCK_DIR}/pid" 2>/dev/null || true)
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Phase 3 checker already running (pid=$old_pid); exiting."
    exit 0
  fi

  echo "Removing stale checker lock: $CHECKER_LOCK_DIR"
  rm -rf "$CHECKER_LOCK_DIR"
  mkdir "$CHECKER_LOCK_DIR"
  printf '%s\n' "$$" > "${CHECKER_LOCK_DIR}/pid"
  trap 'rm -rf "$CHECKER_LOCK_DIR"' EXIT INT TERM
}

# Local VN time without tzdata (server lacks /usr/share/zoneinfo). Prefer GNU
# `date -d`; fall back to epoch arithmetic on a minimal image lacking it.
now_vn() {
  if date -u -d '+7 hours' +'%Y-%m-%d %H:%M' 2>/dev/null; then
    return
  fi
  local epoch=$(( $(date -u +%s) + 7*3600 ))
  date -u -r "$epoch" +'%Y-%m-%d %H:%M' 2>/dev/null \
    || date -u +'%Y-%m-%d %H:%M (UTC; +7h failed)'
}

phase3_active_process_count() {
  ps -eo args \
    | grep -E 'python[0-9.]* +main\.py|phase3_resumable_runner\.py|phase3_main_commands\.sh' \
    | grep -E 'phase3_main|phase3_resumable_runner|phase3_main_commands' \
    | grep -v -E 'grep|phase3_checker_loop|phase3_checker_report' \
    | wc -l
}

gdrive_backup_process_count() {
  ps -eo args \
    | grep -E 'bash .*phase3_backup_to_gdrive\.sh|phase3_backup_to_gdrive\.sh +(loop|once)' \
    | grep -v -E 'grep|phase3_checker_loop|phase3_checker_report' \
    | wc -l
}

status_value() {
  key="$1"
  file="$2"
  awk -F= -v wanted="$key" '$1 == wanted {sub(/^[^=]*=/, ""); print; exit}' "$file" 2>/dev/null
}

write_gdrive_backup_note() {
  active_count=$(gdrive_backup_process_count)
  active_state="inactive"
  [ "${active_count:-0}" -gt 0 ] && active_state="active (${active_count} process(es))"

  if [ ! -f "$GDRIVE_BACKUP_STATUS_FILE" ]; then
    echo "- Google Drive backup: ${active_state}; no status file yet (${GDRIVE_BACKUP_STATUS_FILE})"
    return 0
  fi

  backup_status=$(status_value STATUS "$GDRIVE_BACKUP_STATUS_FILE")
  backup_time=$(status_value TIMESTAMP_UTC "$GDRIVE_BACKUP_STATUS_FILE")
  backup_message=$(status_value MESSAGE "$GDRIVE_BACKUP_STATUS_FILE")
  backup_archive=$(status_value ARCHIVE "$GDRIVE_BACKUP_STATUS_FILE")
  backup_dest=$(status_value DESTINATION "$GDRIVE_BACKUP_STATUS_FILE")

  [ -n "$backup_status" ] || backup_status="unknown"
  [ -n "$backup_time" ] || backup_time="unknown time"
  [ -n "$backup_message" ] || backup_message="no message"
  [ -n "$backup_dest" ] || backup_dest="unknown destination"

  echo "- Google Drive backup: ${active_state}; last_status=${backup_status}; last_update=${backup_time}; message=${backup_message}; destination=${backup_dest}"
  if [ -n "$backup_archive" ]; then
    echo "- Google Drive backup archive: ${backup_archive}"
  fi
}

pending_count() {
  "$PYTHON" "$REPORT" 2>/dev/null \
    | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin).get('pending', '?'))" 2>/dev/null \
    || echo "?"
}

resume_phase3() {
  # Safety re-check: only the dead-watchdog path reaches here, but a second
  # runner started next to a still-alive train process would train the same run
  # twice and create duplicate manifest rows. An automated watchdog must NOT kill
  # main.py (a count-pattern miss could otherwise reap a legitimate run); it only
  # re-checks for stragglers and SKIPS launching if any remain. Manual migration
  # is where killing belongs, not here.
  local stragglers
  stragglers=$(ps -eo args | grep -E 'python[0-9.]* +main\.py' | grep 'phase3_main' | grep -v grep | wc -l)
  if [ "$stragglers" -ne 0 ]; then
    echo "STRAGGLER_WARNING:$stragglers"
    return 1
  fi
  tmux has-session -t run_ablation 2>/dev/null || tmux new-session -d -s run_ablation
  tmux send-keys -t run_ablation \
    "cd '$PROJECT_ROOT' && DATA_ROOT='$DATA_ROOT' PYTHON='$PYTHON' '$PYTHON' '$RUNNER' 2>&1 | tee revision_materials/logs/phase3_resume_\$(date +%Y%m%d_%H%M%S).log" C-m
}

write_progress_note() {
  local action="$1"
  local report_json
  report_json=$(COST_PER_HOUR_VND="$COST_PER_HOUR_VND" "$PYTHON" "$REPORT" 2>/dev/null || echo '{}')
  {
    echo ""
    echo "## $(now_vn) VN ($(date -u +'%Y-%m-%d %H:%M') UTC)"
    echo "- Status: $action"
    echo "$report_json" | COST_PER_HOUR_VND="$COST_PER_HOUR_VND" "$PYTHON" -c '
import json, os, sys
try:
    data = json.load(sys.stdin)
except Exception:
    data = {}
if data:
    cost_per_hour = int(os.environ.get("COST_PER_HOUR_VND", "5000"))
    eta_hours = float(data.get("eta_hours", 0))
    cost_remaining = int(round(eta_hours) * cost_per_hour)
    print("- Completed: {}/{} runs".format(data["done"], data["total"]))
    print("- Pending: {} runs".format(data["pending"]))
    print("- Rate: {}".format(data["rate"]))
    print("- Estimated remaining time: {} (~{}h)".format(data["eta_human"], data["eta_hours"]))
    print("- Estimated finish: {} ({})".format(data.get("estimated_finish_vn", "?"), data.get("estimated_finish_utc", "?")))
    print("- Estimated remaining cost: {:,} VND".format(cost_remaining))
else:
    print("- Report unavailable")
'
    write_gdrive_backup_note
  } >> "$NOTE"
}

acquire_checker_lock

tick=0
while true; do
  tick=$((tick+1))
  ACTIVE_PROCS=$(phase3_active_process_count)

  did_resume=0
  all_done=0
  straggler=0
  if [ "${ACTIVE_PROCS:-0}" -eq 0 ]; then
    PENDING=$(pending_count)
    if [ "$PENDING" != "0" ] && [ "$PENDING" != "?" ]; then
      resume_out=$(resume_phase3)
      if printf '%s' "$resume_out" | grep -q 'STRAGGLER_WARNING'; then
        n=$(printf '%s' "$resume_out" | sed -n 's/.*STRAGGLER_WARNING:\([0-9]*\).*/\1/p')
        ACTION="Count saw 0 active but $n straggler train process(es) detected; SKIPPED launching runner (no kill) to avoid duplicate runs"
        straggler=1
      else
        ACTION="No active Phase 3 process; resume requested ($PENDING runs pending)"
        did_resume=1
      fi
    elif [ "$PENDING" = "0" ]; then
      ACTION="All Phase 3 runs completed; checker exiting"
      all_done=1
    else
      ACTION="Could not read pending count; skip resume this cycle"
    fi
  else
    ACTION="Phase 3 is active ($ACTIVE_PROCS process(es))"
  fi

  # Note on schedule (cycles 1,3,5...), on any resume, or once at completion.
  write_note=0
  [ $(( (tick-1) % NOTE_EVERY )) -eq 0 ] && write_note=1
  [ "$did_resume" -eq 1 ] && write_note=1
  [ "$all_done" -eq 1 ] && write_note=1
  [ "$straggler" -eq 1 ] && write_note=1
  [ "$write_note" -eq 1 ] && write_progress_note "$ACTION"

  # Stop the watchdog once everything is done.
  [ "$all_done" -eq 1 ] && break

  sleep "$CHECK_INTERVAL_SECONDS"
done
