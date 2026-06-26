#!/usr/bin/env bash
# Tự tính báo cáo tiến độ validation sweep lúc 06:00 sáng giờ VN (= 23:00 UTC).
# Chạy độc lập trong tmux, không cần Claude.
set -uo pipefail

BASE=/root/OthorAdapt/revision_materials
MANIFEST=$BASE/results/validation_sweep_ramp100_results.jsonl
LOGDIR=$BASE/logs/validation_sweep_ramp100
OUT=$BASE/progress_report_6am.txt
TOTAL=120
RATE_VND=5000          # đồng / giờ
TZ_OFFSET=7            # VN = UTC+7
SWEEP_START=1781682227 # 2026-06-17 07:43:47 UTC (lúc bắt đầu sweep)

# Mục tiêu: 23:00 UTC ngày 17/06/2026 = 06:00 sáng 18/06 giờ VN
TARGET=$(date -u -d '2026-06-17 23:00:00' +%s)

now=$(date +%s)
sleep_for=$(( TARGET - now ))
if (( sleep_for > 0 )); then
  echo "Đợi tới 06:00 sáng VN ($(( sleep_for/60 )) phút nữa)..."
  sleep "$sleep_for"
fi

vn() { date -u -d "@$1" "+%Y-%m-%d %H:%M:%S" -d "@$(( $1 + TZ_OFFSET*3600 ))"; }

now=$(date +%s)
done_runs=$(wc -l < "$MANIFEST" 2>/dev/null || echo 0)
remaining=$(( TOTAL - done_runs ))

# tốc độ phút/run: từ lúc bắt đầu sweep tới run cuối hoàn tất
last_log_m=$(for f in "$LOGDIR"/*.log; do stat -c %Y "$f"; done 2>/dev/null | sort -n | tail -1)
if [[ -n "${last_log_m:-}" && "$done_runs" -gt 0 ]]; then
  sec_per_run=$(( (last_log_m - SWEEP_START) / done_runs ))
else
  sec_per_run=720
fi
eta_sec=$(( remaining * sec_per_run ))
finish=$(( now + eta_sec ))

# chi phí
elapsed_h=$(( (now - SWEEP_START) / 3600 ))
remain_h=$(( eta_sec / 3600 ))
total_h=$(( (finish - SWEEP_START) / 3600 ))
cost_remain=$(( remain_h * RATE_VND ))
cost_total=$(( total_h * RATE_VND ))

{
  echo "===== BÁO CÁO TIẾN ĐỘ (06:00 sáng VN 18/06) ====="
  echo "Thời điểm tính : $(vn "$now") VN"
  echo "Đã xong        : $done_runs / $TOTAL run"
  echo "Còn lại        : $remaining run"
  echo "Tốc độ         : $(( sec_per_run/60 )) phút/run (~${sec_per_run}s)"
  echo "Thời gian còn  : ~${remain_h} giờ"
  echo "Dự kiến xong   : $(vn "$finish") VN"
  echo "--- Chi phí (5.000đ/giờ) ---"
  echo "Đã chạy ~${elapsed_h}h"
  echo "Phần còn lại   : ~$(printf "%'d" $cost_remain) đ"
  echo "Toàn bộ sweep  : ~$(printf "%'d" $cost_total) đ (~${total_h}h)"
  echo "================================================"
  echo "(Run đang chạy:)"
  tmux capture-pane -t run -p 2>/dev/null | tail -2
} | tee "$OUT"

echo
echo ">>> Báo cáo đã lưu tại: $OUT"
echo ">>> Cửa sổ này giữ nguyên để bạn xem. Nhấn Ctrl-C để thoát."
sleep infinity
