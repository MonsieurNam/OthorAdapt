#!/bin/bash

# ==============================================================================
# SCRIPT CHẠY ABLATION STUDY VỀ LOSS FUNCTION (CÓ TÁCH FOLDER CHECKPOINT)
# ==============================================================================

# --- CẤU HÌNH DỮ LIỆU ---
DATASETS=( "fgvc" "eurosat" "food101" "oxford_pets" "oxford_flowers" "caltech101" "dtd" "ucf101" )
SHOTS=(1)

# --- CẤU HÌNH MODEL ---
ADAPTER_TYPE="ohsinglora"
LEARNING_RATE="2e-4"
RANK=2
ALPHA=1
RAMP_UP_STEPS=100
LORA_PARAMS="q k v"
BASE_ITERS=500
BATCH_SIZE=32
NUM_HEADS=2
LAMBDA_O=0.03

# --- ĐƯỜNG DẪN ---
ROOT_PATH="/root/DATA"
BASE_SAVE_PATH="./checkpoints" # Đường dẫn gốc để lưu weight

# --- LOGGING ---
LOG_DIR="results_logs_${ADAPTER_TYPE}_loss_ablation"
mkdir -p "${LOG_DIR}"

SUMMARY_CSV="${LOG_DIR}/summary_loss_ablation.csv"
# Thêm cột checkpoint_path vào header
if [ ! -f "$SUMMARY_CSV" ]; then
  echo "loss,dataset,shots,accuracy,log_file,checkpoint_path" > "${SUMMARY_CSV}"
fi

# ===================================================================
# DANH SÁCH LOSS FUNCTIONS
# ===================================================================
LOSSES=( "ce" "ce_ls" "arcface" "cosface" "ce_center" "ce_maxent" "focal" )

# ===================================================================
# HÀM TRỢ GIÚP: TRÍCH XUẤT ACCURACY (GIỮ NGUYÊN TỪ SCRIPT CỦA BẠN)
# ===================================================================
extract_accuracy_from_log() {
  local logfile="$1"
  local patterns=(
    "FINAL_ACC"
    "Test Acc[: ]"
    "Test accuracy[: ]"
    "Top-1[: ]"
    "Acc[: ]"
    "Accuracy[: ]"
    "Accuracy ="
  )

  for p in "${patterns[@]}"; do
    line=$(grep -i -E "${p}" "${logfile}" | tail -n 1)
    if [ -n "${line}" ]; then
      num=$(echo "${line}" | grep -o -E '[0-9]+([.][0-9]+)?' | tail -n1)
      if [ -n "${num}" ]; then
        echo "${num}"
        return 0
      fi
    fi
  done

  num=$(grep -o -E '[0-9]+([.][0-9]+)?' "${logfile}" | sort -n | tail -n1)
  if [ -n "${num}" ]; then
    echo "${num}"
    return 0
  fi

  echo ""
  return 1
}

# ===================================================================
# VÒNG LẶP CHÍNH
# ===================================================================
for DATASET in "${DATASETS[@]}"; do
  for SHOT in "${SHOTS[@]}"; do
    for LOSS in "${LOSSES[@]}"; do
      
      # Tạo tên định danh duy nhất
      CONFIG_NAME="${DATASET}_${SHOT}shot_${LOSS}_${ADAPTER_TYPE}"
      LOG_FILE="${LOG_DIR}/${CONFIG_NAME}.log"
      
      # --- [QUAN TRỌNG] TẠO ĐƯỜNG DẪN LƯU CHECKPOINT RIÊNG ---
      # Cấu trúc: ./checkpoints/caltech101/1shot_arcface_ohsinglora/
      CURRENT_SAVE_PATH="${BASE_SAVE_PATH}/${DATASET}/${SHOT}shot_${LOSS}_${ADAPTER_TYPE}"
      mkdir -p "${CURRENT_SAVE_PATH}"

      rm -f "${LOG_FILE}"

      # Xây dựng lệnh cơ bản
      COMMAND_BASE=( python3 main.py
        --dataset "${DATASET}"
        --root_path "${ROOT_PATH}"
        --shots "${SHOT}"
        --n_iters "${BASE_ITERS}"
        --num_heads "${NUM_HEADS}"
        --batch_size "${BATCH_SIZE}"
        --adapter "${ADAPTER_TYPE}"
        --lr "${LEARNING_RATE}"
        --r "${RANK}"
        --alpha "${ALPHA}"
        --ramp_up_steps "${RAMP_UP_STEPS}"
        --params ${LORA_PARAMS}
        --save_path "${CURRENT_SAVE_PATH}"  # <--- Dùng đường dẫn động
        --lambda_o "${LAMBDA_O}"
        --loss_fn "${LOSS}"
      )

      # --- THÊM THAM SỐ ĐẶC THÙ CHO TỪNG LOSS ---
      case "${LOSS}" in
        ce)
          ;;
        ce_ls)
          COMMAND_BASE+=( --label_smoothing 0.1 )
          ;;
        arcface)
          COMMAND_BASE+=( --metric_s 30.0 --metric_m 0.50 )
          ;;
        cosface)
          COMMAND_BASE+=( --metric_s 30.0 --metric_m 0.35 )
          ;;
        ce_center)
          COMMAND_BASE+=( --lambda_center 0.003 )
          ;;
        ce_maxent)
          COMMAND_BASE+=( --lambda_maxent 0.1 )
          ;;
        focal)
          COMMAND_BASE+=( --gamma 2.0 )
          ;;
        *)
          ;;
      esac

      # --- IN THÔNG TIN VÀ CHẠY ---
      echo "----------------------------------------------------------------------"
      echo "🚀 RUNNING: [Dataset: ${DATASET}] | [Loss: ${LOSS}]"
      echo "   Save Path: ${CURRENT_SAVE_PATH}"
      echo "----------------------------------------------------------------------"

      "${COMMAND_BASE[@]}" > "${LOG_FILE}" 2>&1

      # --- XỬ LÝ KẾT QUẢ ---
      ACC=$(extract_accuracy_from_log "${LOG_FILE}")
      if [ -z "${ACC}" ]; then
        ACC="N/A"
      fi

      # Ghi vào CSV (Kèm đường dẫn checkpoint)
      echo "${LOSS},${DATASET},${SHOT},${ACC},${LOG_FILE},${CURRENT_SAVE_PATH}" >> "${SUMMARY_CSV}"
      
      echo "✅ Result: Loss=${LOSS} -> Acc=${ACC}%"
      echo ""
    done
  done
done

echo "🎉 All experiments finished."
echo "📂 Summary CSV: ${SUMMARY_CSV}"