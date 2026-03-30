# LIBERO Evaluation for DreamZero

This repository originally exposed DreamZero evaluation only through the DROID `sim_evals` path. The files below add a parallel `LIBERO` evaluation path without changing the DreamZero websocket server:

- `eval_utils/run_libero_eval.py`
- `scripts/test/run_libero_eval.sh`
- `scripts/test/setup_libero_eval_env.sh`

## What This Integration Assumes

The released DreamZero checkpoint and inference server still use the DROID observation/action contract:

- 2 exterior views + 1 wrist view
- joint-position arm state
- scalar gripper state
- `(N, 8)` action chunks: 7 arm joints + 1 gripper

To make `LIBERO` runnable with the same server, the runner maps `LIBERO` observations into that DROID-format request:

- `agentview_image` -> exterior view 0
- `sideview_image` -> exterior view 1 when available, otherwise reuses `agentview_image`
- `robot0_eye_in_hand_image` -> wrist view
- `robot0_joint_pos` -> joint state
- `robot0_gripper_qpos` -> scalar gripper state via mean reduction when the env exposes two finger joints

This means the cleanest setup is a `LIBERO` environment/controller whose action space matches DreamZero-DROID's joint-position output. The runner therefore defaults to:

- controller: `JOINT_POSITION`
- DreamZero open-loop horizon: `8`

If the installed `LIBERO` controller exposes a different action dimension, the runner will stop with an explicit error instead of silently misapplying actions.

## Environment Setup

Create a dedicated venv next to the existing ones under `/opt/venvs`:

```bash
bash scripts/test/setup_libero_eval_env.sh
```

By default this creates:

```bash
/opt/venvs/dreamzero-libero-eval
```

Override with `VENV_DIR=/path/to/venv` if needed.

## Running Evaluation

Start the DreamZero websocket server first, exactly as you do for the existing DROID sim eval:

```bash
PORT=8010 NUM_GPUS=1 bash scripts/test/start_eval_server.sh
```

Then run `LIBERO` evaluation:

```bash
BENCHMARK=libero_goal \
EPISODES=10 \
MAX_STEPS=600 \
bash scripts/test/run_libero_eval.sh
```

Optional task subset:

```bash
BENCHMARK=libero_goal \
TASK_IDS=0,1,2 \
EPISODES=5 \
bash scripts/test/run_libero_eval.sh
```

Outputs are written under `runs/libero_eval/<timestamp>/`:

- per-episode rollout videos
- `summary.json` with aggregate success/progress/return metrics
- `run_meta.txt` and `eval.log`
- `libero_config/` for the run-local LIBERO config

## Notes

- The runner is written against the public `LIBERO` API shape, but `LIBERO` has had minor import-path differences across revisions. `run_libero_eval.py` probes the installed package dynamically to tolerate those differences.
- The validated runtime path in this repo currently uses `/opt/venvs/dreamzero-train/bin/python`. `run_libero_eval.sh` defaults to that interpreter so it can reuse the working DreamZero inference stack.
- If you still want a separate environment, `scripts/test/setup_libero_eval_env.sh` creates `/opt/venvs/dreamzero-libero-eval` with the LIBERO runtime dependencies.
- LIBERO assets should live under `/opt/src/libero-assets`, matching the existing sim-eval asset convention under `/opt/src`. The installed `libero` package may still resolve assets from its package path, so keep the existing symlink from `site-packages/libero/libero/assets` to `/opt/src/libero-assets`.
- `run_libero_eval.sh` now defaults to `MAX_STEPS=600`.
- `run_libero_eval.sh` now writes `LIBERO_CONFIG_PATH` under the current run root as `runs/libero_eval/<timestamp>/libero_config`, so all files for one run stay together.
