#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-/opt/venvs/dreamzero-libero-eval}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
RECREATE="${RECREATE:-0}"

if [ "$RECREATE" = "1" ] && [ -d "$VENV_DIR" ]; then
  rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  libero==0.1.1 \
  mujoco==3.6.0 \
  robosuite==1.4.0 \
  robomimic==0.2.0 \
  openpi-client==0.1.1 \
  opencv-python \
  imageio \
  imageio-ffmpeg

echo "Created LIBERO eval environment at $VENV_DIR"
