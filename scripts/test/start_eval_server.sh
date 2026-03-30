#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/project_rlinf_hs/yuanhuining/models/DreamZero-DROID}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-libero_sim}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-/mnt/project_rlinf_hs/yuanhuining/datasets/libero}"
PORT="${PORT:-8010}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_ROOT/runs/serve}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/runs/cache/serve}"
LATEST_RUN_FILE="${LATEST_RUN_FILE:-$RUNS_ROOT/.latest_run}"
RUN_NAME="${RUN_NAME:-$(date -u +%Y-%m-%dT%H-%M-%SZ)}"
RUN_DIR="${RUN_DIR:-$RUNS_ROOT/$RUN_NAME}"
GENERATED_VIDEO_DIR="${GENERATED_VIDEO_DIR:-$RUN_DIR/generated_videos}"

export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export WAN_CKPT_DIR="${WAN_CKPT_DIR:-/mnt/project_rlinf_hs/yuanhuining/models/Wan2.1-I2V-14B-480P}"
export DREAMZERO_TOKENIZER_PATH="${DREAMZERO_TOKENIZER_PATH:-/mnt/project_rlinf_hs/yuanhuining/models/umt5-xxl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LIBERO_DATA_ROOT

mkdir -p "$HF_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$RUN_DIR"
mkdir -p "$GENERATED_VIDEO_DIR"
LOG_FILE="$RUN_DIR/eval_server.log"

printf '%s\n' "$RUN_DIR" > "$LATEST_RUN_FILE"

cat > "$RUN_DIR/server_meta.txt" <<EOF
script=scripts/test/start_eval_server.sh
model_path=$MODEL_PATH
embodiment_tag=$EMBODIMENT_TAG
libero_data_root=$LIBERO_DATA_ROOT
port=$PORT
num_gpus=$NUM_GPUS
master_addr=$MASTER_ADDR
master_port=$MASTER_PORT
run_dir=$RUN_DIR
log_file=$LOG_FILE
latest_run_file=$LATEST_RUN_FILE
cache_root=$CACHE_ROOT
generated_videos_dir=$GENERATED_VIDEO_DIR
EOF

echo "RUN_DIR=$RUN_DIR"
echo "LOG_FILE=$LOG_FILE"

exec /opt/venvs/dreamzero-train/bin/python -m torch.distributed.run \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="$NUM_GPUS" \
  "$PROJECT_ROOT/socket_test_optimized_AR.py" \
  --port "$PORT" \
  --enable-dit-cache \
  --model-path "$MODEL_PATH" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --output-dir "$GENERATED_VIDEO_DIR" \
  >> "$LOG_FILE" 2>&1
