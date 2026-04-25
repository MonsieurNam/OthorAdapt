#!/bin/bash

# ==============================================================================
# SCRIPT TỰ ĐỘNG CHẠY VÀ TỔNG HỢP KẾT QUẢ (SCAN THEO CẤU HÌNH CỤ THỂ)
# ==============================================================================

# --- PHẦN 1: CẤU HÌNH THỬ NGHIỆM ---

# Các bộ dữ liệu cần chạy
DATASETS=("fgvc" "eurosat" "food101" "oxford_pets" "oxford_flowers" "caltech101" "dtd" "ucf101")

# Số lượng shots cần chạy (Scan cho cả 1, 4, 16 shots)
SHOTS=(1 4 16)

# --- ĐỊNH NGHĨA 5 CẤU HÌNH CẦN QUÉT ---
# Sử dụng 2 mảng song song tương ứng từng cặp (Rank, Heads)
CONF_RANKS=(2 2 4 4 4)
CONF_HEADS=(1 2 1 2 4)
# Giải thích thứ tự chạy:
# 1. R2 - H1
# 2. R2 - H2
# 3. R4 - H1
# 4. R4 - H2
# 5. R4 - H4

# Cấu hình CỐ ĐỊNH khác
ADAPTER_TYPE="ohsinglora"
LEARNING_RATE="2e-4"
ALPHA=1
RAMP_UP_STEPS=100
LORA_PARAMS="q k v"
BASE_ITERS=500
BATCH_SIZE=32
LAMBDA_O=0.03

# Đường dẫn
ROOT_PATH="/root/DATA"
SAVE_PATH="/root/checkpoints"

# --- THIẾT LẬP LOGGING ---
LOG_DIR="results_logs_scan_configs"
mkdir -p $LOG_DIR

# --- HÀM TRỢ GIÚP (TRÍCH XUẤT DỮ LIỆU) ---
extract_accuracy_from_log() {
  local logfile="$1"
  local acc=$(grep -i "Final test accuracy" "${logfile}" | tail -n 1 | grep -Eo '[0-9]+\.[0-9]+')
  if [[ -n "$acc" ]]; then echo "${acc}"; else echo "N/A"; fi
}

extract_params_from_log() {
  local logfile="$1"
  local params=$(grep "Number of trainable parameters:" "${logfile}" | head -n 1 | grep -Eo '[0-9]+')
  if [[ -n "$params" ]]; then echo "${params}"; else echo "N/A"; fi
}

extract_time_from_log() {
  local logfile="$1"
  local time_val=$(grep "Fine-tuning finished in" "${logfile}" | tail -n 1 | grep -Eo '[0-9]+\.[0-9]+')
  if [[ -n "$time_val" ]]; then echo "${time_val}"; else echo "N/A"; fi
}

# --- PHẦN 3: VÒNG LẶP THỰC THI ---

# Lặp qua từng số lượng shots
for SHOT in "${SHOTS[@]}"; do
  
  # Tạo file CSV tổng hợp cho từng mức shot
  SUMMARY_CSV="${LOG_DIR}/summary_${ADAPTER_TYPE}_${SHOT}shot.csv"
  
  # Ghi header nếu file chưa tồn tại
  if [ ! -f "$SUMMARY_CSV" ]; then
    echo "dataset,shots,adapter,rank,heads,lambda_o,params,time(s),accuracy,log_file" > "${SUMMARY_CSV}"
  fi

  # Lặp qua danh sách cấu hình (Rank, Heads) đã định nghĩa ở trên
  # ${!CONF_RANKS[@]} lấy index của mảng (0, 1, 2, 3, 4)
  for i in "${!CONF_RANKS[@]}"; do
    RANK=${CONF_RANKS[$i]}
    NUM_HEADS=${CONF_HEADS[$i]}

    # Lặp qua từng bộ dữ liệu
    for DATASET in "${DATASETS[@]}"; do

      # Tạo tên file log: Bao gồm Rank và Heads để phân biệt
      CONFIG_NAME="${ADAPTER_TYPE}_${DATASET}_${SHOT}shot_R${RANK}_Head${NUM_HEADS}"
      LOG_FILE="${LOG_DIR}/${CONFIG_NAME}.log"
      
      # Xóa log cũ để chạy mới
      rm -f "$LOG_FILE"

      # --- GHI HEADER VÀO LOG FILE ---
      {
          echo "############################################################"
          echo "# EXPERIMENT RUN INFO"
          echo "# ----------------------------------------------------------"
          echo "# ADAPTER      : ${ADAPTER_TYPE}"
          echo "# DATASET      : ${DATASET}"
          echo "# SHOTS        : ${SHOT}"
          echo "# RANK         : ${RANK}"
          echo "# HEADS        : ${NUM_HEADS}"
          echo "# LAMBDA_O     : ${LAMBDA_O}"
          echo "# DATE         : $(date)"
          echo "############################################################"
          echo ""
      } > "${LOG_FILE}"

      # --- HIỂN THỊ TRÊN CONSOLE ---
      echo "----------------------------------------------------------------------"
      echo "🚀 STARTING: [Dataset: ${DATASET}] | [Shots: ${SHOT}]"
      echo "   Config : Rank=${RANK}, Heads=${NUM_HEADS} (Config #$((i+1)))"
      echo "   Log File: ${LOG_FILE}"
      echo "----------------------------------------------------------------------"

      # Xây dựng lệnh python
      COMMAND=(python3 main.py 
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
        --save_path "${SAVE_PATH}" 
        --lambda_o "${LAMBDA_O}"
      )

      # Thực thi lệnh
      "${COMMAND[@]}" >> "${LOG_FILE}" 2>&1

      # --- TRÍCH XUẤT KẾT QUẢ ---
      ACC=$(extract_accuracy_from_log "${LOG_FILE}")
      PARAMS=$(extract_params_from_log "${LOG_FILE}")
      TIME_SEC=$(extract_time_from_log "${LOG_FILE}")

      # Ghi vào CSV
      echo "${DATASET},${SHOT},${ADAPTER_TYPE},${RANK},${NUM_HEADS},${LAMBDA_O},${PARAMS},${TIME_SEC},${ACC},${CONFIG_NAME}.log" >> "${SUMMARY_CSV}"

      echo "✅ FINISHED: [${DATASET} | ${SHOT}shot | R${RANK}-H${NUM_HEADS}]"
      echo "   -> Acc: ${ACC}% | Time: ${TIME_SEC}s"
      echo ""

    done # End Datasets
  done # End Configs (Rank/Heads)
done # End Shots

echo "🎉 ALL CONFIGURATIONS FINISHED."
echo "📂 Check results in directory: ${LOG_DIR}"