#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIM_EVALS_ROOT="${SIM_EVALS_ROOT:-/opt/src/sim-evals}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
SCENE="${SCENE:-1}"
EPISODES="${EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-8}"
# Set SYNC_EVAL=1 to use full-chunk sync mode (send 8 downsampled frames per chunk,
# no chunk overlap). Requires OPEN_LOOP_HORIZON=24 (full DROID action chunk size).
# Leave unset or 0 for the original single-frame open-loop mode.
SYNC_EVAL="${SYNC_EVAL:-0}"
RESUME="${RESUME:-0}"
WAIT_FOR_SERVER="${WAIT_FOR_SERVER:-1}"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_ROOT/runs/eval}"
LATEST_RUN_FILE="${LATEST_RUN_FILE:-$RUNS_ROOT/.latest_run}"
SERVER_RUNS_ROOT="${SERVER_RUNS_ROOT:-$PROJECT_ROOT/runs/serve}"
SERVER_LATEST_RUN_FILE="${SERVER_LATEST_RUN_FILE:-$SERVER_RUNS_ROOT/.latest_run}"

if [ -z "${RUN_DIR:-}" ] && [ -z "${RUN_NAME:-}" ] && [ "$RESUME" = "1" ] && [ -f "$LATEST_RUN_FILE" ]; then
  RUN_DIR="$(cat "$LATEST_RUN_FILE")"
  RUN_NAME="$(basename "$RUN_DIR")"
else
  RUN_NAME="${RUN_NAME:-$(date -u +%Y-%m-%dT%H-%M-%SZ)}"
  RUN_DIR="${RUN_DIR:-$RUNS_ROOT/$RUN_NAME}"
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-yes}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PROJECT_ROOT:${SIM_EVALS_ROOT}/src:/opt/venvs/dreamzero-eval/lib/python3.11/site-packages/isaaclab/source/isaaclab_tasks${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$LATEST_RUN_FILE"
LOG_FILE="$RUN_DIR/eval.log"
START_EPISODE="${START_EPISODE:-$(find "$RUN_DIR" -maxdepth 1 -type f -name 'episode_*.mp4' | wc -l)}"

cat > "$RUN_DIR/run_meta.txt" <<EOF
script=scripts/test/run_official_sim_eval.sh
host=$HOST
port=$PORT
scene=$SCENE
episodes=$EPISODES
max_steps=$MAX_STEPS
start_episode=$START_EPISODE
run_dir=$RUN_DIR
latest_run_file=$LATEST_RUN_FILE
resume=$RESUME
wait_for_server=$WAIT_FOR_SERVER
EOF

echo "RUN_DIR=$RUN_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "START_EPISODE=$START_EPISODE"

wait_for_server() {
  until ss -ltnp "( sport = :$PORT )" 2>/dev/null | grep -q ":$PORT"; do
    sleep 10
  done

  until /opt/venvs/dreamzero-eval/bin/python - "$HOST" "$PORT" >/dev/null 2>&1 <<'PY'
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
  local current_state_file="$RUN_DIR/.generated_after.txt"
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

if [ "$WAIT_FOR_SERVER" = "1" ]; then
  wait_for_server
fi

EVAL_ARGS=(
  --host "$HOST"
  --port "$PORT"
  --episodes "$EPISODES"
  --scene "$SCENE"
  --headless
  --max-steps "$MAX_STEPS"
  --start-episode "$START_EPISODE"
  --output-dir "$RUN_DIR"
  --open-loop-horizon "$OPEN_LOOP_HORIZON"
)

if [ "$SYNC_EVAL" = "1" ]; then
  EVAL_ARGS+=(--sync-eval)
fi

/opt/venvs/dreamzero-eval/bin/python "$PROJECT_ROOT/eval_utils/run_sim_eval.py" \
  "${EVAL_ARGS[@]}" \
  >> "$LOG_FILE" 2>&1
status=$?

exit "$status"
