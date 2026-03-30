#!/bin/bash
# DreamZero LIBERO Full Fine-Tuning Script
#
# Usage:
#   bash scripts/train/libero_training_full_finetune.sh

export HYDRA_FULL_ERROR=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_ROOT=${RUNS_ROOT:-"$PROJECT_ROOT/runs/train"}
CACHE_ROOT=${CACHE_ROOT:-"$PROJECT_ROOT/runs/cache/train"}
RUN_NAME=${RUN_NAME:-$(date -u +%Y-%m-%dT%H-%M-%SZ)}
RUN_DIR=${RUN_DIR:-"$RUNS_ROOT/$RUN_NAME"}
LOG_FILE=${LOG_FILE:-"$RUN_DIR/train.log"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$RUN_DIR/tensorboard"}
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"

LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-"/mnt/project_rlinf_hs/yuanhuining/datasets/libero"}
OUTPUT_DIR=${OUTPUT_DIR:-"$RUN_DIR/checkpoints"}
NUM_GPUS=${NUM_GPUS:-8}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
SAVE_STEPS=${SAVE_STEPS:-2000}
SAVE_DEEPSPEED_CHECKPOINT=${SAVE_DEEPSPEED_CHECKPOINT:-false}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-null}
REPORT_TO=${REPORT_TO:-none}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-1}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
TORCHRUN_BIN=${TORCHRUN_BIN:-/opt/venvs/dreamzero-train/bin/torchrun}

if [ "$MAX_STEPS" -le 0 ]; then
    echo "ERROR: MAX_STEPS must be > 0. For full LIBERO training, use a positive value such as 100000."
    exit 1
fi

if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -d "$LIBERO_DATA_ROOT" ]; then
    echo "ERROR: LIBERO dataset not found at $LIBERO_DATA_ROOT"
    exit 1
fi

mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$TENSORBOARD_DIR" "$HF_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cat > "$RUN_DIR/run_meta.txt" <<EOF
script=scripts/train/libero_training_full_finetune.sh
libero_data_root=$LIBERO_DATA_ROOT
output_dir=$OUTPUT_DIR
num_gpus=$NUM_GPUS
max_steps=$MAX_STEPS
save_strategy=$SAVE_STRATEGY
report_to=$REPORT_TO
per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE
dataloader_num_workers=$DATALOADER_NUM_WORKERS
image_resolution_width=256
image_resolution_height=256
num_views=2
frame_seqlen=512
run_dir=$RUN_DIR
log_file=$LOG_FILE
tensorboard_dir=$TENSORBOARD_DIR
cache_root=$CACHE_ROOT
EOF

exec > >(tee -a "$LOG_FILE") 2>&1

"$TORCHRUN_BIN" --nproc_per_node $NUM_GPUS --standalone --module groot.vla.experiment.experiment \
    report_to=$REPORT_TO \
    data=dreamzero/libero_relative \
    wandb_project=dreamzero \
    train_architecture=full \
    num_frames=33 \
    action_horizon=16 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=16 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2_offload.json" \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    training_args.logging_dir=$TENSORBOARD_DIR \
    per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=$DATALOADER_NUM_WORKERS \
    image_resolution_width=256 \
    image_resolution_height=256 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=512 \
    save_strategy=$SAVE_STRATEGY \
    save_deepspeed_checkpoint=$SAVE_DEEPSPEED_CHECKPOINT \
    pretrained_model_path=$PRETRAINED_MODEL_PATH \
    libero_data_root=$LIBERO_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
