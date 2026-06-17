#!/usr/bin/env bash
# Resume validation sweep: bỏ qua run đã có trong manifest, chỉ chạy phần còn lại.
# Idempotent: chạy lại nhiều lần đều an toàn (mỗi lần tự skip cái đã xong).
# Ghim conda python tuyệt đối -> KHÔNG đụng env opera / PATH.
set -uo pipefail

cd /root/OthorAdapt

export PYTHON=/opt/conda/bin/python
export DATA_ROOT=/root/DATA
export RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

SRC=revision_materials/scripts/validation_sweep_commands.sh
MANIFEST=revision_materials/results/validation_sweep_results.jsonl

# Tập filename đã completed trong manifest
mapfile -t DONE < <(
  "$PYTHON" - "$MANIFEST" <<'PY'
import json,sys
done=set()
try:
    for line in open(sys.argv[1]):
        line=line.strip()
        if not line: continue
        d=json.loads(line); c=d.get('command',[])
        if '--filename' in c and d.get('status')=='completed':
            done.add(c[c.index('--filename')+1])
except FileNotFoundError:
    pass
print("\n".join(sorted(done)))
PY
)
declare -A DONE_SET
for f in "${DONE[@]}"; do [[ -n "$f" ]] && DONE_SET["$f"]=1; done
echo ">>> Đã xong (skip): ${#DONE_SET[@]} run"

# Sanity check: conda python có torch không
if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
  echo "!!! LỖI: $PYTHON không có torch. Dừng lại." >&2
  exit 1
fi
echo ">>> Dùng python: $PYTHON ($("$PYTHON" -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())'))"

mkdir -p revision_materials/logs/validation_sweep

total=0; ran=0; skipped=0
# Đọc từng dòng lệnh main.py trong script gốc
while IFS= read -r line; do
  [[ "$line" == *"main.py"* ]] || continue
  total=$((total+1))
  # tách --filename <name>
  fname=$(sed -n 's/.*--filename \([A-Za-z0-9_]*\).*/\1/p' <<<"$line")
  if [[ -n "$fname" && -n "${DONE_SET[$fname]:-}" ]]; then
    skipped=$((skipped+1)); continue
  fi
  ran=$((ran+1))
  echo "=== [$ran] CHẠY: $fname  ($(date +%H:%M:%S)) ==="
  # eval để mở rộng ${DATA_ROOT} ${RUN_STAMP} $PYTHON trong dòng lệnh
  eval "$line"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "!!! run '$fname' thoát mã $rc — dừng để bạn kiểm tra." >&2
    exit $rc
  fi
done < "$SRC"

echo ">>> XONG. Tổng=$total | đã skip=$skipped | vừa chạy=$ran"
