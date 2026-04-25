#!/bin/bash

# ==============================================================================
# SCRIPT TỰ ĐỘNG CHẠY VÀ TỔNG HỢP KẾT QUẢ (PHIÊN BẢN TÁCH FILE THEO SHOT)
# ==============================================================================

# --- PHẦN 1: CẤU HÌNH THỬ NGHIỆM ---

# Các bộ dữ liệu cần chạy
DATASETS=('caltech101' 'eurosat')

# Số lượng shots cần chạy
SHOTS=(16)

# Cấu hình CỐ ĐỊNH
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

# Đường dẫn GỐC (Sẽ tạo thư mục con bên trong này)
ROOT_PATH="/root/DATA"
BASE_SAVE_PATH="/root/checkpoints"  # <--- ĐÃ SỬA TÊN BIẾN

# --- THIẾT LẬP LOGGING ---
LOG_DIR="results_logs_${ADAPTER_TYPE}"
mkdir -p $LOG_DIR

# --- HÀM TRỢ GIÚP (TRÍCH XUẤT DỮ LIỆU) ---

# 1. Trích xuất Accuracy
extract_accuracy_from_log() {
  local logfile="$1"
  local acc=$(grep -i "Final test accuracy" "${logfile}" | tail -n 1 | grep -Eo '[0-9]+\.[0-9]+')
  if [[ -n "$acc" ]]; then echo "${acc}"; else echo "N/A"; fi
}

# 2. Trích xuất Trainable Parameters
extract_params_from_log() {
  local logfile="$1"
  local params=$(grep "Number of trainable parameters:" "${logfile}" | head -n 1 | grep -Eo '[0-9]+')
  if [[ -n "$params" ]]; then echo "${params}"; else echo "N/A"; fi
}

# 3. Trích xuất Training Time
extract_time_from_log() {
  local logfile="$1"
  local time_val=$(grep "Fine-tuning finished in" "${logfile}" | tail -n 1 | grep -Eo '[0-9]+\.[0-9]+')
  if [[ -n "$time_val" ]]; then echo "${time_val}"; else echo "N/A"; fi
}

# --- PHẦN 3: VÒNG LẶP THỰC THI ---

for SHOT in "${SHOTS[@]}"; do
  
  # --- CẤU HÌNH CSV RIÊNG CHO TỪNG SHOT ---
  SUMMARY_CSV="${LOG_DIR}/summary_${ADAPTER_TYPE}_${SHOT}shot.csv"

  if [ ! -f "$SUMMARY_CSV" ]; then
    echo "dataset,shots,adapter,rank,heads,lambda_o,params,time(s),accuracy,log_file,checkpoint_path" > "${SUMMARY_CSV}"
  fi

  for DATASET in "${DATASETS[@]}"; do

    # Tạo tên file log duy nhất
    CONFIG_NAME="${ADAPTER_TYPE}_${DATASET}_${SHOT}shot_LAMBDA_O${LAMBDA_O}"
    LOG_FILE="${LOG_DIR}/${CONFIG_NAME}.log"
    
    # --- [QUAN TRỌNG] TẠO ĐƯỜNG DẪN LƯU CHECKPOINT RIÊNG BIỆT ---
    # Cấu trúc: /root/checkpoints/caltech101/16shot_ohsinglora_r2/
    CURRENT_SAVE_PATH="${BASE_SAVE_PATH}/${DATASET}/${SHOT}shot_${ADAPTER_TYPE}_r${RANK}"
    
    # Tạo thư mục nếu chưa tồn tại
    mkdir -p "${CURRENT_SAVE_PATH}"

    # Xóa file log cũ
    rm -f "$LOG_FILE"

    # --- GHI HEADER VÀO LOG FILE ---
    {
        echo "############################################################"
        echo "# EXPERIMENT RUN INFO"
        echo "# ----------------------------------------------------------"
        echo "# DATASET      : ${DATASET}"
        echo "# SAVE PATH    : ${CURRENT_SAVE_PATH}"  # <--- Log đường dẫn lưu
        echo "# DATE         : $(date)"
        echo "############################################################"
        echo ""
    } > "${LOG_FILE}"

    # --- HIỂN THỊ TRÊN CONSOLE ---
    echo "----------------------------------------------------------------------"
    echo "🚀 STARTING: [Dataset: ${DATASET}] | [Shots: ${SHOT}]"
    echo "   Save Path: ${CURRENT_SAVE_PATH}"
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
      --save_path "${CURRENT_SAVE_PATH}"   # <--- SỬ DỤNG ĐƯỜNG DẪN ĐỘNG
      --lambda_o "${LAMBDA_O}"
    )

    # Thực thi lệnh
    "${COMMAND[@]}" >> "${LOG_FILE}" 2>&1

    # --- TRÍCH XUẤT KẾT QUẢ ---
    ACC=$(extract_accuracy_from_log "${LOG_FILE}")
    PARAMS=$(extract_params_from_log "${LOG_FILE}")
    TIME_SEC=$(extract_time_from_log "${LOG_FILE}")

    # Ghi vào CSV (Thêm cột checkpoint path để dễ tìm lại model sau này)
    echo "${DATASET},${SHOT},${ADAPTER_TYPE},${RANK},${NUM_HEADS},${LAMBDA_O},${PARAMS},${TIME_SEC},${ACC},${CONFIG_NAME}.log,${CURRENT_SAVE_PATH}" >> "${SUMMARY_CSV}"

    echo "✅ FINISHED: [Dataset: ${DATASET} - ${SHOT} shot]"
    echo "   -> Acc: ${ACC}%"
    echo ""

  done
done

echo "🎉 ALL EXPERIMENTS FINISHED."
echo "📂 Check results in directory: ${LOG_DIR}"