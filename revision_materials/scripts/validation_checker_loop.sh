#!/usr/bin/env bash
# Watchdog for validation_sweep_ramp100: check active process, resume safely, write notes.
set -uo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/root/OthorAdapt}
DATA_ROOT=${DATA_ROOT:-/root/DATA}
PYTHON=${PYTHON:-/opt/conda/bin/python}
CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-1800}
NOTE_EVERY=${NOTE_EVERY:-2}
COST_PER_HOUR_VND=${COST_PER_HOUR_VND:-7000}
CHECKER_LOCK_DIR=${CHECKER_LOCK_DIR:-$PROJECT_ROOT/revision_materials/logs/validation_checker_loop.lock}
BACKUP_STATUS_FILE=${BACKUP_STATUS_FILE:-$PROJECT_ROOT/revision_materials/logs/validation_gdrive_backup_status.env}
cd "$PROJECT_ROOT" || exit 1
NOTE="$PROJECT_ROOT/revision_materials/logs/validation_progress_notes.md"
REPORT="revision_materials/scripts/validation_checker_report.py"
W3_REPORT="revision_materials/scripts/w3_headcount_h1_checker_report.py"
RESUME="revision_materials/scripts/resume_sweep.sh"
mkdir -p "$(dirname "$NOTE")"
[ -f "$NOTE" ] || echo "# Validation sweep progress notes" > "$NOTE"
case "$NOTE_EVERY" in ''|*[!0-9]*) NOTE_EVERY=1;; esac
[ "$NOTE_EVERY" -lt 1 ] && NOTE_EVERY=1
acquire_lock(){
  mkdir -p "$(dirname "$CHECKER_LOCK_DIR")"
  if mkdir "$CHECKER_LOCK_DIR" 2>/dev/null; then
    echo $$ > "$CHECKER_LOCK_DIR/pid"; trap 'rm -rf "$CHECKER_LOCK_DIR"' EXIT INT TERM; return 0
  fi
  old=$(cat "$CHECKER_LOCK_DIR/pid" 2>/dev/null || true)
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then echo "Validation checker already running pid=$old"; exit 0; fi
  rm -rf "$CHECKER_LOCK_DIR"; mkdir "$CHECKER_LOCK_DIR"; echo $$ > "$CHECKER_LOCK_DIR/pid"; trap 'rm -rf "$CHECKER_LOCK_DIR"' EXIT INT TERM
}
now_vn(){ date -u -d '+7 hours' +'%Y-%m-%d %H:%M' 2>/dev/null || date -u +'%Y-%m-%d %H:%M UTC'; }
active_count(){
  ps -eo args | grep -E 'python[0-9.]* +main\.py|validation_sweep_commands\.sh|resume_sweep\.sh' | grep -E 'validation_sweep_ramp100|resume_sweep|validation_sweep_commands' | grep -v -E 'grep|validation_checker_loop|validation_checker_report' | wc -l
}
pending_count(){ "$PYTHON" "$REPORT" 2>/dev/null | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("pending","?"))' 2>/dev/null || echo '?'; }
status_value(){ awk -F= -v k="$1" '$1==k{sub(/^[^=]*=/,"");print;exit}' "$2" 2>/dev/null; }
backup_note(){
  if [ ! -f "$BACKUP_STATUS_FILE" ]; then echo "- Google Drive backup: no status file yet ($BACKUP_STATUS_FILE)"; return; fi
  echo "- Google Drive backup: status=$(status_value STATUS "$BACKUP_STATUS_FILE"); update=$(status_value TIMESTAMP_UTC "$BACKUP_STATUS_FILE"); message=$(status_value MESSAGE "$BACKUP_STATUS_FILE"); dest=$(status_value DESTINATION "$BACKUP_STATUS_FILE")"
  a=$(status_value ARCHIVE "$BACKUP_STATUS_FILE"); [ -n "$a" ] && echo "- Google Drive backup archive: $a"
}
resume_validation(){
  strag=$(ps -eo args | grep -E 'python[0-9.]* +main\.py' | grep 'validation_sweep_ramp100' | grep -v grep | wc -l)
  if [ "$strag" -ne 0 ]; then echo "STRAGGLER_WARNING:$strag"; return 1; fi
  tmux has-session -t validation_ramp100 2>/dev/null || tmux new-session -d -s validation_ramp100
  tmux send-keys -t validation_ramp100 "cd '$PROJECT_ROOT' && DATA_ROOT='$DATA_ROOT' PYTHON='$PYTHON' bash '$RESUME' 2>&1 | tee revision_materials/logs/validation_resume_\$(date +%Y%m%d_%H%M%S).log" C-m
}
write_note(){
  action="$1"; report_json=$(COST_PER_HOUR_VND="$COST_PER_HOUR_VND" "$PYTHON" "$REPORT" 2>/dev/null || echo '{}')
  {
    echo ""; echo "## $(now_vn) VN ($(date -u +'%Y-%m-%d %H:%M') UTC)"; echo "- Status: $action"
    echo "$report_json" | COST_PER_HOUR_VND="$COST_PER_HOUR_VND" "$PYTHON" -c 'import json,os,sys; d=json.load(sys.stdin); c=int(os.environ.get("COST_PER_HOUR_VND","5000")); print(f"- Completed: {d.get('"'"'done'"'"','"'"'?'"'"')}/{d.get('"'"'total'"'"','"'"'?'"'"')} runs"); print(f"- Pending: {d.get('"'"'pending'"'"','"'"'?'"'"')} runs"); print(f"- Rate: {d.get('"'"'rate'"'"','"'"'unknown'"'"')}"); print(f"- Estimated remaining time: {d.get('"'"'eta_human'"'"','"'"'?'"'"')} (~{d.get('"'"'eta_hours'"'"','"'"'?'"'"')}h)"); print(f"- Estimated finish: {d.get('"'"'estimated_finish_vn'"'"','"'"'?'"'"')} ({d.get('"'"'estimated_finish_utc'"'"','"'"'?'"'"')})"); print(f"- Estimated remaining cost: {round(float(d.get('"'"'eta_hours'"'"',0)))*c:,} VND"); a=d.get('"'"'active_log'"'"') or {}; print(f"- Active progress: {a.get('"'"'iter'"'"','"'"'?'"'"')}/{a.get('"'"'total_iter'"'"','"'"'?'"'"')} ({a.get('"'"'percent'"'"','"'"'?'"'"')}%), active ETA {a.get('"'"'eta_human'"'"','"'"'?'"'"')}")'
    if [ -f "$W3_REPORT" ]; then
      w3_json=$("$PYTHON" "$W3_REPORT" 2>/dev/null || echo '{}')
      echo "$w3_json" | "$PYTHON" -c '
import json, sys
try:
    d=json.load(sys.stdin)
except Exception:
    d={}
if d and d.get("total", 0):
    print("- W3 H=1 add-on: {}/{} completed; pending={}; ETA={}; finish={}".format(d.get("done","?"), d.get("total","?"), d.get("pending","?"), d.get("eta_human","?"), d.get("estimated_finish_vn","?")))
    print("- W3 ETA basis: {}".format(d.get("rate", "unknown")))
    a=d.get("active_log") or {}
    print("- W3 active progress: {}/{} ({}%), active ETA {}".format(a.get("iter","?"), a.get("total_iter","?"), a.get("percent","?"), a.get("eta_human","?")))
'
    fi
    backup_note
  } >> "$NOTE"
}
acquire_lock
tick=0
while true; do
  tick=$((tick+1)); ACTIVE=$(active_count); did_resume=0; all_done=0; straggler=0
  if [ "${ACTIVE:-0}" -eq 0 ]; then
    PENDING=$(pending_count)
    if [ "$PENDING" = "0" ]; then ACTION="All validation sweep runs completed; checker exiting"; all_done=1
    elif [ "$PENDING" != "?" ]; then out=$(resume_validation); if printf '%s' "$out"|grep -q STRAGGLER_WARNING; then ACTION="Count saw 0 active but straggler exists; skipped resume"; straggler=1; else ACTION="No active validation process; resume requested ($PENDING pending)"; did_resume=1; fi
    else ACTION="Could not read pending count; skip resume"
    fi
  else ACTION="Validation sweep is active ($ACTIVE process(es))"; fi
  write_note_flag=0; [ $(((tick-1)%NOTE_EVERY)) -eq 0 ] && write_note_flag=1; [ "$did_resume" -eq 1 ] && write_note_flag=1; [ "$all_done" -eq 1 ] && write_note_flag=1; [ "$straggler" -eq 1 ] && write_note_flag=1
  [ "$write_note_flag" -eq 1 ] && write_note "$ACTION"
  [ "$all_done" -eq 1 ] && break
  sleep "$CHECK_INTERVAL_SECONDS"
done
