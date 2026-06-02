#!/bin/bash
# Libra-MIL 后台离线训练脚本
# 用法: ./train_background.sh
# 依次执行: 1-shot -> 4-shot -> 16-shot -> full
# 每个shot分别训练 k=4 和 k=10

# 公共参数
H5_FILE_DIR="/mnt/sda2/WSI/muti-modal/TCGA-RCC-fea/features"
INSTANCE_PATH="./text_prompt/TCGA_RCC_instance_prompt.json"
BAG_PATH="./text_prompt/TCGA_RCC_two_scale_text_prompt.csv"
TEXT_MODEL_WEIGHTS="/mnt/sda1/ln_workspace/CONCH/checkpoints/pytorch_model.bin"
NUM_CLASSES=3
EPOCHES=20

# 日志目录
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/train_all_${TIMESTAMP}.log"

echo "=========================================="
echo "Libra-MIL 后台离线训练"
echo "=========================================="
echo "启动时间: $(date)"
echo "主日志文件: ${MAIN_LOG}"
echo "=========================================="

# 启动后台训练
nohup bash -c '
    # 公共参数
    H5_FILE_DIR="/mnt/sda2/WSI/muti-modal/TCGA-RCC-fea/features"
    INSTANCE_PATH="./text_prompt/TCGA_RCC_instance_prompt.json"
    BAG_PATH="./text_prompt/TCGA_RCC_two_scale_text_prompt.csv"
    TEXT_MODEL_WEIGHTS="/mnt/sda1/ln_workspace/CONCH/checkpoints/pytorch_model.bin"
    NUM_CLASSES=3
    EPOCHES=20

    echo "=========================================="
    echo "Libra-MIL 离线训练开始"
    echo "=========================================="
    echo "开始时间: $(date)"
    echo ""

    # 训练任务: shot名称:数据目录:保存目录前缀
    # 顺序: 1-shot -> 4-shot -> 16-shot -> full
    # 每个shot依次训练 k=4 和 k=10

    # ==================== 1-shot ====================
    echo "=========================================="
    echo "[1/8] 1-shot, k=4"
    echo "=========================================="
    echo "开始时间: $(date)"
    python main.py \
        --data_split_json ./data_1shot/data_split.json \
        --data_csv ./data_1shot/labels.csv \
        --h5_file_dir ${H5_FILE_DIR} \
        --instance_path ${INSTANCE_PATH} \
        --bag_path ${BAG_PATH} \
        --text_model_weights_path ${TEXT_MODEL_WEIGHTS} \
        --save_dir ./results/TCGA_RCC_1shot_k=4 \
        --num_vis_prototypes 4 \
        --num_classes ${NUM_CLASSES} \
        --epoches ${EPOCHES}
    echo "完成时间: $(date)"
    echo ""

    echo "=========================================="
    echo "[2/8] 1-shot, k=10"
    echo "=========================================="
    echo "开始时间: $(date)"
    python main.py \
        --data_split_json ./data_1shot/data_split.json \
        --data_csv ./data_1shot/labels.csv \
        --h5_file_dir ${H5_FILE_DIR} \
        --instance_path ${INSTANCE_PATH} \
        --bag_path ${BAG_PATH} \
        --text_model_weights_path ${TEXT_MODEL_WEIGHTS} \
        --save_dir ./results/TCGA_RCC_1shot_k=10 \
        --num_vis_prototypes 10 \
        --num_classes ${NUM_CLASSES} \
        --epoches ${EPOCHES}
    echo "完成时间: $(date)"
    echo ""

    # ==================== 4-shot ====================
    echo "=========================================="
    echo "[3/8] 4-shot, k=4"
    echo "=========================================="
    echo "开始时间: $(date)"
    python main.py \
        --data_split_json ./data_4shot/data_split.json \
        --data_csv ./data_4shot/labels.csv \
        --h5_file_dir ${H5_FILE_DIR} \
        --instance_path ${INSTANCE_PATH} \
        --bag_path ${BAG_PATH} \
        --text_model_weights_path ${TEXT_MODEL_WEIGHTS} \
        --save_dir ./results/TCGA_RCC_4shot_k=4 \
        --num_vis_prototypes 4 \
        --num_classes ${NUM_CLASSES} \
        --epoches ${EPOCHES}
    echo "完成时间: $(date)"
    echo ""

    echo "=========================================="
    echo "[4/8] 4-shot, k=10"
    echo "=========================================="
    echo "开始时间: $(date)"
    python main.py \
        --data_split_json ./data_4shot/data_split.json \
        --data_csv ./data_4shot/labels.csv \
        --h5_file_dir ${H5_FILE_DIR} \
        --instance_path ${INSTANCE_PATH} \
        --bag_path ${BAG_PATH} \
        --text_model_weights_path ${TEXT_MODEL_WEIGHTS} \
        --save_dir ./results/TCGA_RCC_4shot_k=10 \
        --num_vis_prototypes 10 \
        --num_classes ${NUM_CLASSES} \
        --epoches ${EPOCHES}
    echo "完成时间: $(date)"
    echo ""

    # ==================== 16-shot ====================
    echo "=========================================="
    echo "[5/8] 16-shot, k=4"
    echo "=========================================="
    echo "开始时间: $(date)"
    python main.py \
        --data_split_json ./data_16shot/data_split.json \
        --data_csv ./data_16shot/labels.csv \
        --h5_file_dir ${H5_FILE_DIR} \
        --instance_path ${INSTANCE_PATH} \
        --bag_path ${BAG_PATH} \
        --text_model_weights_path ${TEXT_MODEL_WEIGHTS} \
        --save_dir ./results/TCGA_RCC_16shot_k=4 \
        --num_vis_prototypes 4 \
        --num_classes ${NUM_CLASSES} \
        --epoches ${EPOCHES}
    echo "完成时间: $(date)"
    echo ""

    echo "=========================================="
    echo "[6/8] 16-shot, k=10"
    echo "=========================================="
    echo "开始时间: $(date)"
    python main.py \
        --data_split_json ./data_16shot/data_split.json \
        --data_csv ./data_16shot/labels.csv \
        --h5_file_dir ${H5_FILE_DIR} \
        --instance_path ${INSTANCE_PATH} \
        --bag_path ${BAG_PATH} \
        --text_model_weights_path ${TEXT_MODEL_WEIGHTS} \
        --save_dir ./results/TCGA_RCC_16shot_k=10 \
        --num_vis_prototypes 10 \
        --num_classes ${NUM_CLASSES} \
        --epoches ${EPOCHES}
    echo "完成时间: $(date)"
    echo ""

    # ==================== full ====================
    echo "=========================================="
    echo "[7/8] full, k=4"
    echo "=========================================="
    echo "开始时间: $(date)"
    python main.py \
        --data_split_json ./data/data_split.json \
        --data_csv ./data/labels.csv \
        --h5_file_dir ${H5_FILE_DIR} \
        --instance_path ${INSTANCE_PATH} \
        --bag_path ${BAG_PATH} \
        --text_model_weights_path ${TEXT_MODEL_WEIGHTS} \
        --save_dir ./results/TCGA_RCC_full_k=4 \
        --num_vis_prototypes 4 \
        --num_classes ${NUM_CLASSES} \
        --epoches ${EPOCHES}
    echo "完成时间: $(date)"
    echo ""

    echo "=========================================="
    echo "[8/8] full, k=10"
    echo "=========================================="
    echo "开始时间: $(date)"
    python main.py \
        --data_split_json ./data/data_split.json \
        --data_csv ./data/labels.csv \
        --h5_file_dir ${H5_FILE_DIR} \
        --instance_path ${INSTANCE_PATH} \
        --bag_path ${BAG_PATH} \
        --text_model_weights_path ${TEXT_MODEL_WEIGHTS} \
        --save_dir ./results/TCGA_RCC_full_k=10 \
        --num_vis_prototypes 10 \
        --num_classes ${NUM_CLASSES} \
        --epoches ${EPOCHES}
    echo "完成时间: $(date)"
    echo ""

    echo "=========================================="
    echo "全部训练完成!"
    echo "结束时间: $(date)"
    echo "=========================================="
' > ${MAIN_LOG} 2>&1 &

PID=$!
echo ""
echo "后台进程已启动!"
echo "PID: ${PID}"
echo ""
echo "常用命令:"
echo "  查看日志: tail -f ${MAIN_LOG}"
echo "  检查进程: ps aux | grep ${PID}"
echo "  查看GPU:  watch -n 1 nvidia-smi"
echo ""
