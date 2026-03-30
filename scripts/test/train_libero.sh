#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_ROOT/runs/train}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/runs/cache/train}"
RUN_NAME="${RUN_NAME:-$(date -u +%Y-%m-%dT%H-%M-%SZ)}"
RUN_DIR="${RUN_DIR:-$RUNS_ROOT/$RUN_NAME}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$RUN_DIR/tensorboard}"

export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/project_rlinf_hs/yuanhuining/datasets/libero}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/checkpoints}"
export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-1}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
export SAVE_STEPS="${SAVE_STEPS:-2000}"
export SAVE_DEEPSPEED_CHECKPOINT="${SAVE_DEEPSPEED_CHECKPOINT:-false}"
export REPORT_TO="${REPORT_TO:-none}"
export TRAIN_ARCHITECTURE="${TRAIN_ARCHITECTURE:-lora}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
export WAN_CKPT_DIR="${WAN_CKPT_DIR:-/mnt/project_rlinf_hs/yuanhuining/models/Wan2.1-I2V-14B-480P}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-/mnt/project_rlinf_hs/yuanhuining/models/umt5-xxl}"
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export TENSORBOARD_DIR
export DEBUG_SAVE_DIR="${DEBUG_SAVE_DIR:-$RUN_DIR/debug/images}"
export DEBUG_MAX_SAVES="${DEBUG_MAX_SAVES:-5}"

mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$RUN_DIR" "$DEBUG_SAVE_DIR"
LOG_FILE="$RUN_DIR/train_libero.log"

exec > >(tee -a "$LOG_FILE") 2>&1

cat > "$RUN_DIR/run_meta.txt" <<EOF
script=scripts/test/train_libero.sh
train_architecture=$TRAIN_ARCHITECTURE
libero_data_root=$LIBERO_DATA_ROOT
output_dir=$OUTPUT_DIR
num_gpus=$NUM_GPUS
max_steps=$MAX_STEPS
save_strategy=$SAVE_STRATEGY
save_steps=$SAVE_STEPS
save_deepspeed_checkpoint=$SAVE_DEEPSPEED_CHECKPOINT
pretrained_model_path=$PRETRAINED_MODEL_PATH
report_to=$REPORT_TO
per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE
dataloader_num_workers=$DATALOADER_NUM_WORKERS
frame_seqlen=512
run_dir=$RUN_DIR
tensorboard_dir=$TENSORBOARD_DIR
cache_root=$CACHE_ROOT
debug_save_dir=$DEBUG_SAVE_DIR
debug_max_saves=$DEBUG_MAX_SAVES
EOF

case "$TRAIN_ARCHITECTURE" in
    lora)
        TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train/libero_training_lora.sh"
        ;;
    full|full_finetune)
        TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train/libero_training_full_finetune.sh"
        ;;
    *)
        echo "ERROR: unsupported TRAIN_ARCHITECTURE=$TRAIN_ARCHITECTURE"
        echo "Supported values: lora, full, full_finetune"
        exit 1
        ;;
esac

bash "$TRAIN_SCRIPT"
