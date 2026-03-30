#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESUME="${RESUME:-0}"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_ROOT/runs/eval}"
LATEST_RUN_FILE="${LATEST_RUN_FILE:-$RUNS_ROOT/.latest_run}"
WAIT_FOR_SERVER="${WAIT_FOR_SERVER:-1}"
SERVER_RUNS_ROOT="${SERVER_RUNS_ROOT:-$PROJECT_ROOT/runs/serve}"
SERVER_LATEST_RUN_FILE="${SERVER_LATEST_RUN_FILE:-$SERVER_RUNS_ROOT/.latest_run}"

if [ -z "${BASE_RUN_DIR:-}" ] && [ -z "${RUN_NAME:-}" ] && [ "$RESUME" = "1" ] && [ -f "$LATEST_RUN_FILE" ]; then
  BASE_RUN_DIR="$(cat "$LATEST_RUN_FILE")"
  RUN_NAME="$(basename "$BASE_RUN_DIR")"
else
  RUN_NAME="${RUN_NAME:-$(date -u +%Y-%m-%dT%H-%M-%SZ)}"
  BASE_RUN_DIR="${BASE_RUN_DIR:-$RUNS_ROOT/$RUN_NAME}"
fi

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8010}"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:--1}"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-yes}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PROJECT_ROOT:/opt/src/sim-evals/src:/opt/venvs/dreamzero-eval/lib/python3.11/site-packages/isaaclab/source/isaaclab_tasks${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$BASE_RUN_DIR"
printf '%s\n' "$BASE_RUN_DIR" > "$LATEST_RUN_FILE"
LOG_FILE="$BASE_RUN_DIR/batch_eval.log"

exec > >(tee -a "$LOG_FILE") 2>&1

cat > "$BASE_RUN_DIR/run_meta.txt" <<EOF
script=scripts/test/run_three_scenes_10_rollouts.sh
server_host=$SERVER_HOST
server_port=$SERVER_PORT
episodes=$EPISODES
max_steps=$MAX_STEPS
base_run_dir=$BASE_RUN_DIR
latest_run_file=$LATEST_RUN_FILE
resume=$RESUME
wait_for_server=$WAIT_FOR_SERVER
EOF

wait_for_existing_eval() {
  while pgrep -af "/opt/venvs/dreamzero-eval/bin/python .*(run_sim_eval.py|eval_utils.run_sim_eval)" >/dev/null 2>&1; do
    sleep 30
  done
}

wait_for_server() {
  until ss -ltnp "( sport = :$SERVER_PORT )" 2>/dev/null | grep -q ":$SERVER_PORT"; do
    sleep 10
  done

  until /opt/venvs/dreamzero-eval/bin/python - "$SERVER_HOST" "$SERVER_PORT" >/dev/null 2>&1 <<'PY'
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
  local start_episode_index="$3"
  local generated_dir=""
  local current_state_file="$BASE_RUN_DIR/.generated_after.txt"
  local episode_index="$start_episode_index"

  generated_dir="$(resolve_server_generated_dir)"
  : > "$current_state_file"
  if [ -z "$generated_dir" ] || [ ! -d "$generated_dir" ]; then
    return 0
  fi

  find "$generated_dir" -maxdepth 1 -type f -name '*.mp4' | sort > "$current_state_file"
  mkdir -p "$dest_dir"
  while IFS= read -r src; do
    [ -n "$src" ] || continue
    cp -f "$src" "$dest_dir/episode_${episode_index}_gen.mp4"
    episode_index=$((episode_index + 1))
  done < <(comm -13 "$before_state_file" "$current_state_file")
  rm -f "$current_state_file"
}

run_scene() {
  local scene="$1"
  local output_dir="$2"

  mkdir -p "$output_dir"

  while true; do
    local completed
    completed=$(find "$output_dir" -maxdepth 1 -type f -name 'episode_*.mp4' | wc -l)
    if [ "$completed" -ge "$EPISODES" ]; then
      break
    fi

    /opt/venvs/dreamzero-eval/bin/python "$PROJECT_ROOT/eval_utils/run_sim_eval.py" \
      --host "$SERVER_HOST" \
      --port "$SERVER_PORT" \
      --episodes "$EPISODES" \
      --scene "$scene" \
      --headless \
      --max-steps "$MAX_STEPS" \
      --start-episode "$completed" \
      --output-dir "$output_dir"
  done
}

wait_for_existing_eval
if [ "$WAIT_FOR_SERVER" = "1" ]; then
  wait_for_server
fi

run_scene 1 "${SCENE1_OUTPUT_DIR:-$BASE_RUN_DIR/scene1}"
run_scene 2 "${SCENE2_OUTPUT_DIR:-$BASE_RUN_DIR/scene2}"
run_scene 3 "${SCENE3_OUTPUT_DIR:-$BASE_RUN_DIR/scene3}"
