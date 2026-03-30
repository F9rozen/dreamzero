# Train / Serve / Eval Usage

This repo now keeps training logs, server logs, sim-eval outputs, and LIBERO eval outputs under `runs/`, with persistent caches under `runs/cache/`.

## Assets and Persistent Paths

- DROID sim-eval assets: `/opt/src/sim-evals/assets`
- LIBERO assets: `/opt/src/libero-assets`
- Train cache root: `runs/cache/train`
- Serve cache root: `runs/cache/serve`

No downloaded assets are expected to live under `/tmp`.

## Training

Main training:

```bash
bash scripts/train/droid_training_lora.sh
```

Useful overrides:

```bash
NUM_GPUS=1 MAX_STEPS=100 DROID_DATA_ROOT=/path/to/data bash scripts/train/droid_training_lora.sh
```

Outputs:

- `runs/train/<timestamp>/train.log`
- `runs/train/<timestamp>/checkpoints/...`

Sanity test:

```bash
bash scripts/test/train_droid.sh
```

LIBERO sanity test:

```bash
bash scripts/test/train_libero.sh
```

## Serve

Start the DreamZero websocket server:

```bash
bash scripts/test/start_eval_server.sh
```

Useful overrides:

```bash
PORT=8010 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash scripts/test/start_eval_server.sh
```

Outputs:

- `runs/serve/<timestamp>/eval_server.log`
- `runs/serve/<timestamp>/server_meta.txt`

## Official Sim Eval

Run one scene:

```bash
bash scripts/test/run_official_sim_eval.sh
```

Common overrides:

```bash
SCENE=2 EPISODES=5 MAX_STEPS=9 bash scripts/test/run_official_sim_eval.sh
```

Resume the latest run:

```bash
RESUME=1 bash scripts/test/run_official_sim_eval.sh
```

Key env vars:

- `HOST`, `PORT`
- `SCENE`
- `EPISODES`
- `MAX_STEPS`
- `WAIT_FOR_SERVER=0|1`
- `RUN_NAME`, `RUN_DIR`

Outputs:

- `runs/eval/<timestamp>/episode_*.mp4`
- `runs/eval/<timestamp>/eval.log`

## Three-Scene Sim Eval

Run scenes 1, 2, 3 sequentially:

```bash
bash scripts/test/run_three_scenes_10_rollouts.sh
```

Example:

```bash
EPISODES=5 MAX_STEPS=9 bash scripts/test/run_three_scenes_10_rollouts.sh
```

Resume the latest batch:

```bash
RESUME=1 bash scripts/test/run_three_scenes_10_rollouts.sh
```

Key env vars:

- `SERVER_HOST`, `SERVER_PORT`
- `EPISODES`
- `MAX_STEPS`
- `WAIT_FOR_SERVER=0|1`
- `RUN_NAME`, `BASE_RUN_DIR`

Outputs:

- `runs/eval/<timestamp>/scene1/...`
- `runs/eval/<timestamp>/scene2/...`
- `runs/eval/<timestamp>/scene3/...`
- `runs/eval/<timestamp>/batch_eval.log`

## LIBERO Eval

Run one benchmark:

```bash
bash scripts/test/run_libero_eval.sh
```

By default this runs:

- `BENCHMARK=libero_goal`
- `EPISODES=10`
- `MAX_STEPS=600`

Common overrides:

```bash
BENCHMARK=libero_spatial EPISODES=1 TASK_IDS=0,1,2 bash scripts/test/run_libero_eval.sh
```

Resume the latest LIBERO run:

```bash
RESUME=1 bash scripts/test/run_libero_eval.sh
```

Key env vars:

- `HOST`, `PORT`
- `BENCHMARK`
- `TASK_IDS`
- `EPISODES`
- `MAX_STEPS`
- `WAIT_FOR_SERVER=0|1`
- `RUN_NAME`, `BASE_RUN_DIR`, `RUN_DIR`

Outputs:

- `runs/libero_eval/<timestamp>/libero_<benchmark>/...`
- `runs/libero_eval/<timestamp>/libero_config/`

## Recommended Flow

Start server:

```bash
bash scripts/test/start_eval_server.sh
```

Run sim eval:

```bash
EPISODES=5 bash scripts/test/run_three_scenes_10_rollouts.sh
```

Resume sim eval:

```bash
RESUME=1 bash scripts/test/run_three_scenes_10_rollouts.sh
```

Run LIBERO eval:

```bash
BENCHMARK=libero_goal bash scripts/test/run_libero_eval.sh
```

Resume LIBERO eval:

```bash
RESUME=1 BENCHMARK=libero_goal bash scripts/test/run_libero_eval.sh
```
