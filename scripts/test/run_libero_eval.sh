#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
BENCHMARK="${BENCHMARK:-libero_goal}"
TASK_IDS="${TASK_IDS:-}"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-600}"
CONTROLLER="${CONTROLLER:-OSC_POSE}"
CONTROL_FREQ="${CONTROL_FREQ:-10}"
OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-5}"
# Set SYNC_EVAL=1 to use full-chunk sync mode (send 8 downsampled frames per chunk,
# no chunk overlap). Requires OPEN_LOOP_HORIZON=16 (full action chunk size).
# Leave unset or 0 for the original single-frame open-loop mode.
SYNC_EVAL="${SYNC_EVAL:-0}"
RESUME="${RESUME:-0}"
WAIT_FOR_SERVER="${WAIT_FOR_SERVER:-1}"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_ROOT/runs/libero_eval}"
LATEST_RUN_FILE="${LATEST_RUN_FILE:-$RUNS_ROOT/.latest_run}"
SERVER_RUNS_ROOT="${SERVER_RUNS_ROOT:-$PROJECT_ROOT/runs/serve}"
SERVER_LATEST_RUN_FILE="${SERVER_LATEST_RUN_FILE:-$SERVER_RUNS_ROOT/.latest_run}"

if [ -z "${RUN_DIR:-}" ] && [ -z "${RUN_NAME:-}" ] && [ "$RESUME" = "1" ] && [ -f "$LATEST_RUN_FILE" ]; then
  BASE_RUN_DIR="$(cat "$LATEST_RUN_FILE")"
  RUN_NAME="$(basename "$BASE_RUN_DIR")"
else
  RUN_NAME="${RUN_NAME:-$(date -u +%Y-%m-%dT%H-%M-%SZ)}"
  BASE_RUN_DIR="${BASE_RUN_DIR:-$RUNS_ROOT/$RUN_NAME}"
fi

RUN_DIR="${RUN_DIR:-$BASE_RUN_DIR/libero_${BENCHMARK}}"
LIBERO_EVAL_PYTHON="${LIBERO_EVAL_PYTHON:-/opt/venvs/dreamzero-train/bin/python}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$BASE_RUN_DIR/libero_config}"
MUJOCO_GL="${MUJOCO_GL:-osmesa}"

mkdir -p "$RUN_DIR"
printf '%s\n' "$BASE_RUN_DIR" > "$LATEST_RUN_FILE"
LOG_FILE="$RUN_DIR/eval.log"
mkdir -p "$LIBERO_CONFIG_PATH"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export MUJOCO_GL
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:${PYTHONPATH}}"

cat > "$RUN_DIR/run_meta.txt" <<EOF
script=scripts/test/run_libero_eval.sh
host=$HOST
port=$PORT
benchmark=$BENCHMARK
task_ids=$TASK_IDS
episodes=$EPISODES
max_steps=$MAX_STEPS
controller=$CONTROLLER
control_freq=$CONTROL_FREQ
open_loop_horizon=$OPEN_LOOP_HORIZON
run_dir=$RUN_DIR
log_file=$LOG_FILE
latest_run_file=$LATEST_RUN_FILE
libero_eval_python=$LIBERO_EVAL_PYTHON
libero_config_path=$LIBERO_CONFIG_PATH
base_run_dir=$BASE_RUN_DIR
resume=$RESUME
wait_for_server=$WAIT_FOR_SERVER
EOF

wait_for_server() {
  until ss -ltnp "( sport = :$PORT )" 2>/dev/null | grep -q ":$PORT"; do
    sleep 10
  done

  until "$LIBERO_EVAL_PYTHON" - "$HOST" "$PORT" >/dev/null 2>&1 <<'PY'
import inspect
import socket
import sys

import websockets.sync.client
from openpi_client import msgpack_numpy

host = sys.argv[1]
port = int(sys.argv[2])
uri = f"ws://{host}:{port}"
socket.setdefaulttimeout(5)

connect_kwargs = {
    "compression": None,
    "max_size": None,
}
connect_signature = inspect.signature(websockets.sync.client.connect)
if "open_timeout" in connect_signature.parameters:
    connect_kwargs["open_timeout"] = 5
if "ping_interval" in connect_signature.parameters:
    connect_kwargs["ping_interval"] = 60
if "ping_timeout" in connect_signature.parameters:
    connect_kwargs["ping_timeout"] = 600

conn = websockets.sync.client.connect(uri, **connect_kwargs)
msgpack_numpy.unpackb(conn.recv())
conn.close()
PY
  do
    sleep 10
  done
}

resolve_server_run_dir() {
  if [ -f "$SERVER_LATEST_RUN_FILE" ]; then
    cat "$SERVER_LATEST_RUN_FILE"
  fi
}

resolve_server_generated_dir() {
  if [ -n "${SERVER_GENERATED_DIR:-}" ]; then
    printf '%s\n' "$SERVER_GENERATED_DIR"
    return 0
  fi

  local server_run_dir=""
  server_run_dir="$(resolve_server_run_dir)"
  if [ -n "$server_run_dir" ] && [ -f "$server_run_dir/server_meta.txt" ]; then
    local configured_dir=""
    configured_dir="$(sed -n 's/^generated_videos_dir=//p' "$server_run_dir/server_meta.txt" | tail -n1)"
    if [ -n "$configured_dir" ]; then
      printf '%s\n' "$configured_dir"
      return 0
    fi
  fi

  if [ -n "$server_run_dir" ] && [ -f "$server_run_dir/eval_server.log" ]; then
    sed -n 's/^INFO:root:Videos will be saved to: //p' "$server_run_dir/eval_server.log" | tail -n1
  fi
}

capture_generated_video_state() {
  local state_file="$1"
  local generated_dir=""
  generated_dir="$(resolve_server_generated_dir)"
  : > "$state_file"
  if [ -n "$generated_dir" ] && [ -d "$generated_dir" ]; then
    find "$generated_dir" -maxdepth 1 -type f -name '*.mp4' | sort > "$state_file"
  fi
}

copy_new_generated_videos() {
  local before_state_file="$1"
  local dest_dir="$2"
  local start_episode_index="${3:-0}"
  local generated_dir=""
  local current_state_file="$BASE_RUN_DIR/.generated_after.txt"
  local episode_index="$start_episode_index"

  generated_dir="$(resolve_server_generated_dir)"
  if [ ! -f "$before_state_file" ]; then
    mkdir -p "$(dirname "$before_state_file")"
    : > "$before_state_file"
  fi
  : > "$current_state_file"
  if [ -z "$generated_dir" ] || [ ! -d "$generated_dir" ]; then
    return 0
  fi

  find "$generated_dir" -maxdepth 1 -type f -name '*.mp4' | sort > "$current_state_file"
  mkdir -p "$dest_dir"
  while IFS= read -r src; do
    [ -n "$src" ] || continue
    cp -f "$src" "$dest_dir/episode_${episode_index}_gen.mp4"
    rm -f "$src"
    episode_index=$((episode_index + 1))
  done < <(comm -13 "$before_state_file" "$current_state_file")
  rm -f "$current_state_file"
}

if [ "$WAIT_FOR_SERVER" = "1" ]; then
  wait_for_server
fi

export LIBERO_CONFIG_PATH

CAMERA_HEIGHT="${CAMERA_HEIGHT:-256}"
CAMERA_WIDTH="${CAMERA_WIDTH:-256}"

ARGS=(
  --host "$HOST"
  --port "$PORT"
  --benchmark "$BENCHMARK"
  --episodes "$EPISODES"
  --max-steps "$MAX_STEPS"
  --controller "$CONTROLLER"
  --control-freq "$CONTROL_FREQ"
  --open-loop-horizon "$OPEN_LOOP_HORIZON"
  --camera-height "$CAMERA_HEIGHT"
  --camera-width "$CAMERA_WIDTH"
  --rotate-images
  --output-dir "$RUN_DIR"
)

if [ -n "$TASK_IDS" ]; then
  ARGS+=(--task-ids "$TASK_IDS")
fi

if [ "$RESUME" = "1" ]; then
  ARGS+=(--resume)
fi

if [ "$SYNC_EVAL" = "1" ]; then
  ARGS+=(--sync-eval)
fi

"$LIBERO_EVAL_PYTHON" -m eval_utils.run_libero_eval "${ARGS[@]}" >> "$LOG_FILE" 2>&1
status=$?

exit "$status"
