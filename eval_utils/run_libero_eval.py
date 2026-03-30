#!/usr/bin/env python3
"""Run DreamZero evaluation on LIBERO benchmarks through the websocket policy server.

This runner mirrors the existing DROID sim-eval flow:
1. Reset a LIBERO task environment.
2. Convert LIBERO observations into DreamZero-DROID server inputs.
3. Query the DreamZero websocket server for an action chunk.
4. Execute the actions in the LIBERO environment and track metrics.

The LIBERO Python API differs slightly across revisions. This file keeps the
integration resilient by probing the installed package at runtime instead of
hard-coding a single import path.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from openpi_client import image_tools

from eval_utils.policy_client import WebsocketClientPolicy


DEFAULT_IMAGE_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "sideview_image",
)
DEFAULT_JOINT_KEYS = (
    "robot0_joint_pos",
    "joint_pos",
    "robot_joint_pos",
    "qpos",
)
DEFAULT_GRIPPER_KEYS = (
    "robot0_gripper_qpos",
    "gripper_qpos",
    "robot0_gripper_pos",
    "gripper_pos",
)
DEFAULT_PROGRESS_KEYS = (
    "task_progress",
    "progress",
    "success_fraction",
    "completion",
)
DEFAULT_SUCCESS_KEYS = (
    "success",
    "task_success",
    "is_success",
)


class LiberoImportError(RuntimeError):
    pass


def _import_first(paths: list[str]) -> Any:
    last_error: Exception | None = None
    for path in paths:
        try:
            return importlib.import_module(path)
        except Exception as exc:  # pragma: no cover - import compatibility path
            last_error = exc
    raise LiberoImportError(f"Unable to import any of {paths}: {last_error}")


def ensure_libero_config() -> None:
    config_root = Path(os.environ.setdefault("LIBERO_CONFIG_PATH", "/tmp/libero-config"))
    config_root.mkdir(parents=True, exist_ok=True)
    config_file = config_root / "config.yaml"
    if config_file.exists():
        return

    libero_spec = importlib.util.find_spec("libero")
    if libero_spec is None or libero_spec.origin is None:
        raise LiberoImportError("Could not locate installed `libero` package")

    benchmark_root = Path(libero_spec.origin).resolve().parent / "libero"
    config_text = (
        f"benchmark_root: {benchmark_root}\n"
        f"bddl_files: {benchmark_root / 'bddl_files'}\n"
        f"init_states: {benchmark_root / 'init_files'}\n"
        f"datasets: {benchmark_root.parent / 'datasets'}\n"
        f"assets: {benchmark_root / 'assets'}\n"
    )
    config_file.write_text(config_text, encoding="utf-8")


def load_libero_modules() -> tuple[Any, Any | None]:
    ensure_libero_config()
    benchmark_module = _import_first(
        [
            "libero.libero.benchmark",
            "libero.benchmark",
        ]
    )

    env_module = None
    for path in (
        "libero.libero.envs",
        "libero.envs",
    ):
        try:
            env_module = importlib.import_module(path)
            break
        except Exception:
            continue

    return benchmark_module, env_module


def resolve_benchmark_suite(benchmark_module: Any, benchmark_name: str) -> Any:
    if hasattr(benchmark_module, "get_benchmark_dict"):
        benchmark_dict = benchmark_module.get_benchmark_dict()
        if benchmark_name not in benchmark_dict:
            raise KeyError(
                f"Benchmark '{benchmark_name}' not found. "
                f"Available: {sorted(benchmark_dict.keys())}"
            )
        return benchmark_dict[benchmark_name]()

    if hasattr(benchmark_module, "get_benchmark"):
        suite = benchmark_module.get_benchmark(benchmark_name)
        if suite is None:
            raise KeyError(f"Benchmark '{benchmark_name}' not found")
        return suite

    raise LiberoImportError(
        "Installed LIBERO benchmark module does not expose "
        "`get_benchmark_dict()` or `get_benchmark()`."
    )


def get_num_tasks(suite: Any) -> int:
    for attr in ("n_tasks", "num_tasks"):
        value = getattr(suite, attr, None)
        if isinstance(value, int):
            return value
    for method_name in ("get_num_tasks",):
        method = getattr(suite, method_name, None)
        if callable(method):
            return int(method())
    for attr in ("tasks", "task_names"):
        value = getattr(suite, attr, None)
        if value is not None:
            return len(value)
    raise RuntimeError("Could not determine number of tasks in LIBERO benchmark")


def get_task_object(suite: Any, task_id: int) -> Any:
    if hasattr(suite, "get_task"):
        return suite.get_task(task_id)
    tasks = getattr(suite, "tasks", None)
    if tasks is not None:
        return tasks[task_id]
    raise RuntimeError("LIBERO suite does not expose task objects")


def get_task_name(task: Any, task_id: int) -> str:
    for attr in ("name", "task_name"):
        value = getattr(task, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(task, dict):
        for key in ("name", "task_name"):
            value = task.get(key)
            if isinstance(value, str) and value:
                return value
    return f"task_{task_id:03d}"


def get_task_instruction(suite: Any, task: Any, task_id: int) -> str:
    for method_name in ("get_task_instruction", "get_task_language"):
        method = getattr(suite, method_name, None)
        if callable(method):
            value = method(task_id)
            if isinstance(value, str) and value:
                return value

    for attr in ("language", "instruction", "problem_desc", "task_description"):
        value = getattr(task, attr, None)
        if isinstance(value, str) and value:
            return value

    if isinstance(task, dict):
        for key in ("language", "instruction", "problem_desc", "task_description"):
            value = task.get(key)
            if isinstance(value, str) and value:
                return value

    return get_task_name(task, task_id).replace("_", " ")


def get_task_bddl_path(suite: Any, task: Any, task_id: int) -> str | None:
    for method_name in ("get_task_bddl_file_path", "get_bddl_file_path"):
        method = getattr(suite, method_name, None)
        if callable(method):
            value = method(task_id)
            if value:
                return str(value)

    for attr in ("bddl_file", "bddl_file_name", "problem_file"):
        value = getattr(task, attr, None)
        if value:
            value = Path(str(value))
            if value.is_absolute():
                return str(value)
            problem_folder = getattr(task, "problem_folder", None)
            if problem_folder:
                try:
                    from libero.libero import get_libero_path

                    return str(Path(get_libero_path("bddl_files")) / str(problem_folder) / value)
                except Exception:
                    pass
            return str(value)

    if isinstance(task, dict):
        for key in ("bddl_file", "bddl_file_name", "problem_file"):
            value = task.get(key)
            if value:
                return str(value)

    return None


def get_task_init_states(suite: Any, task: Any, task_id: int) -> list[Any] | None:
    for method_name in ("get_task_init_states", "get_init_states"):
        method = getattr(suite, method_name, None)
        if callable(method):
            value = method(task_id)
            if value is not None:
                return list(value)

    for attr in ("init_states", "initial_states"):
        value = getattr(task, attr, None)
        if value is not None:
            return list(value)

    if isinstance(task, dict):
        for key in ("init_states", "initial_states"):
            value = task.get(key)
            if value is not None:
                return list(value)

    return None


def load_controller_config(controller_name: str) -> Any:
    try:
        from robosuite.controllers import load_controller_config

        return load_controller_config(default_controller=controller_name)
    except Exception:
        return {"type": controller_name}


def make_offscreen_env(
    env_module: Any | None,
    *,
    bddl_file_name: str | None,
    camera_names: list[str],
    camera_height: int,
    camera_width: int,
    controller_name: str,
    control_freq: int,
) -> Any:
    if env_module is None or not hasattr(env_module, "OffScreenRenderEnv"):
        raise LiberoImportError(
            "Installed LIBERO package does not expose OffScreenRenderEnv. "
            "Please install a LIBERO version with offscreen env support."
        )

    kwargs = {
        "camera_heights": camera_height,
        "camera_widths": camera_width,
        "camera_names": camera_names,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "ignore_done": False,
        "control_freq": control_freq,
        "controller": controller_name,
    }
    if bddl_file_name is not None:
        kwargs["bddl_file_name"] = bddl_file_name

    return env_module.OffScreenRenderEnv(**kwargs)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_")


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[-1]
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0) if image.max() <= 1.0 else np.clip(image, 0.0, 255.0)
        image = (image * 255.0).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return np.ascontiguousarray(image)


def rotate_180(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(image, 2))


def find_first_array(obs: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        value = obs.get(key)
        if value is not None:
            return np.asarray(value)
    return None


def reduce_gripper(gripper_value: np.ndarray) -> np.ndarray:
    gripper_value = np.asarray(gripper_value).astype(np.float64).reshape(-1)
    if gripper_value.size == 0:
        return np.zeros((1,), dtype=np.float64)
    if gripper_value.size == 1:
        return gripper_value
    return np.array([gripper_value.mean()], dtype=np.float64)


def get_action_bounds(env: Any) -> tuple[np.ndarray, np.ndarray] | None:
    for candidate in (env, getattr(env, "env", None)):
        if candidate is None:
            continue
        action_spec = getattr(candidate, "action_spec", None)
        if isinstance(action_spec, tuple) and len(action_spec) == 2:
            low, high = action_spec
            return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)

        action_space = getattr(candidate, "action_space", None)
        if action_space is not None and hasattr(action_space, "low") and hasattr(action_space, "high"):
            return (
                np.asarray(action_space.low, dtype=np.float32),
                np.asarray(action_space.high, dtype=np.float32),
            )

    return None


def extract_progress(info: dict[str, Any], success: bool) -> float:
    for key in DEFAULT_PROGRESS_KEYS:
        value = info.get(key)
        if isinstance(value, (int, float, np.floating)):
            return float(value)
    return 1.0 if success else 0.0


def extract_success(env: Any, info: dict[str, Any], reward: float, done: bool) -> bool:
    for key in DEFAULT_SUCCESS_KEYS:
        value = info.get(key)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.floating)):
            return float(value) > 0

    if hasattr(env, "is_success"):
        try:
            value = env.is_success()
            if isinstance(value, dict):
                for key in DEFAULT_SUCCESS_KEYS:
                    if key in value:
                        return bool(value[key])
                return any(bool(v) for v in value.values())
            return bool(value)
        except Exception:
            pass

    if hasattr(env, "check_success"):
        try:
            return bool(env.check_success())
        except Exception:
            pass

    return bool(done and reward > 0)


def step_env(env: Any, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = result
    return obs, float(reward), bool(done), info


def reset_env_with_init_state(env: Any, init_state: Any | None) -> dict[str, Any]:
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    if init_state is None:
        return obs

    for method_name in ("set_init_state", "set_init_states"):
        method = getattr(env, method_name, None)
        if callable(method):
            maybe_obs = method(init_state)
            if isinstance(maybe_obs, tuple):
                maybe_obs = maybe_obs[0]
            return obs if maybe_obs is None else maybe_obs

    for method_name in ("reset_to",):
        method = getattr(env, method_name, None)
        if callable(method):
            maybe_obs = method(init_state)
            if isinstance(maybe_obs, tuple):
                maybe_obs = maybe_obs[0]
            return obs if maybe_obs is None else maybe_obs

    return obs


class DreamZeroLiberoClient:
    def __init__(
        self,
        remote_host: str,
        remote_port: int,
        *,
        open_loop_horizon: int,
        exterior_key: str,
        wrist_key: str,
        secondary_exterior_key: str | None,
        joint_keys: tuple[str, ...],
        gripper_keys: tuple[str, ...],
        gripper_open_threshold: float,
        rotate_images: bool = False,
        sync_eval: bool = False,
    ) -> None:
        self.client = WebsocketClientPolicy(remote_host, remote_port)
        self.sync_eval = sync_eval
        self.open_loop_horizon = open_loop_horizon
        self.exterior_key = exterior_key
        self.secondary_exterior_key = secondary_exterior_key
        self.wrist_key = wrist_key
        self.joint_keys = joint_keys
        self.gripper_keys = gripper_keys
        self.gripper_open_threshold = gripper_open_threshold
        self.rotate_images = rotate_images
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk: np.ndarray | None = None
        self.session_id = str(uuid.uuid4())
        # Per-camera observation buffers for full-chunk sync inference.
        self.obs_buffer_exterior: list[np.ndarray] = []
        self.obs_buffer_wrist: list[np.ndarray] = []

    def reset(self, gen_video_path: str = "") -> None:
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())
        self.obs_buffer_exterior = []
        self.obs_buffer_wrist = []
        reset_info: dict[str, Any] = {}
        if gen_video_path:
            reset_info["gen_video_path"] = gen_video_path
        try:
            self.client.reset(reset_info)
        except Exception:
            pass

    def _extract_observation(self, obs_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        exterior = obs_dict.get(self.exterior_key)
        if exterior is None:
            raise KeyError(
                f"Missing exterior image key '{self.exterior_key}'. "
                f"Available keys: {sorted(obs_dict.keys())}"
            )
        exterior = to_uint8_image(exterior)
        if self.rotate_images:
            exterior = rotate_180(exterior)

        if self.secondary_exterior_key is not None and self.secondary_exterior_key in obs_dict:
            exterior_2 = to_uint8_image(obs_dict[self.secondary_exterior_key])
            if self.rotate_images:
                exterior_2 = rotate_180(exterior_2)
        else:
            exterior_2 = exterior

        wrist = obs_dict.get(self.wrist_key)
        if wrist is None:
            wrist = exterior
        else:
            wrist = to_uint8_image(wrist)
            if self.rotate_images:
                wrist = rotate_180(wrist)

        joint_position = find_first_array(obs_dict, self.joint_keys)
        if joint_position is None:
            raise KeyError(
                f"Missing joint position keys {self.joint_keys}. "
                f"Available keys: {sorted(obs_dict.keys())}"
            )
        joint_position = np.asarray(joint_position).astype(np.float64).reshape(-1)

        gripper_position = find_first_array(obs_dict, self.gripper_keys)
        gripper_position = (
            np.zeros((1,), dtype=np.float64)
            if gripper_position is None
            else reduce_gripper(gripper_position)
        )

        return {
            "right_image": exterior,
            "left_image": exterior_2,
            "wrist_image": wrist,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

    def infer(self, obs: dict[str, Any], instruction: str) -> tuple[np.ndarray, np.ndarray]:
        curr_obs = self._extract_observation(obs)

        if self.sync_eval:
            # ── Sync mode: execute a full action chunk, then send all collected frames ──
            # Training uses VIDEO_CHUNK_OFFSETS=[0,2,4,6,8,10,12,14] at 10fps→5fps, so
            # we accumulate open_loop_horizon frames and downsample with stride 2 to get 8.
            ext_frame = image_tools.resize_with_pad(curr_obs["right_image"], 256, 256)
            wrist_frame = image_tools.resize_with_pad(curr_obs["wrist_image"], 256, 256)

            if self.actions_from_chunk_completed == 0:
                # Episode start: send a single frame so the server resets its AR state.
                request_data = {
                    "observation/exterior_image_0_left": ext_frame,
                    "observation/wrist_image_left": wrist_frame,
                    "observation/joint_position": curr_obs["joint_position"],
                    "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                    "observation/gripper_position": curr_obs["gripper_position"],
                    "prompt": instruction,
                    "session_id": self.session_id,
                }
                self.obs_buffer_exterior = [ext_frame]
                self.obs_buffer_wrist = [wrist_frame]
                result = self.client.infer(request_data)
                actions = result["actions"] if isinstance(result, dict) and "actions" in result else result
                actions = np.asarray(actions, dtype=np.float32)
                if actions.ndim != 2:
                    raise ValueError(f"Expected 2D action chunk, got {actions.shape}")
                self.pred_action_chunk = actions
                self.actions_from_chunk_completed = 0
            else:
                self.obs_buffer_exterior.append(ext_frame)
                self.obs_buffer_wrist.append(wrist_frame)

                if self.actions_from_chunk_completed >= self.open_loop_horizon:
                    # Full chunk done: downsample collected frames to 8 (stride 2).
                    ext_chunk = np.stack(self.obs_buffer_exterior[::2][:8])    # (8,H,W,3)
                    wrist_chunk = np.stack(self.obs_buffer_wrist[::2][:8])     # (8,H,W,3)
                    request_data = {
                        "observation/exterior_image_0_left": ext_chunk,
                        "observation/wrist_image_left": wrist_chunk,
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                        "session_id": self.session_id,
                    }
                    self.obs_buffer_exterior = [ext_frame]
                    self.obs_buffer_wrist = [wrist_frame]
                    result = self.client.infer(request_data)
                    actions = result["actions"] if isinstance(result, dict) and "actions" in result else result
                    actions = np.asarray(actions, dtype=np.float32)
                    if actions.ndim != 2:
                        raise ValueError(f"Expected 2D action chunk, got {actions.shape}")
                    self.pred_action_chunk = actions
                    self.actions_from_chunk_completed = 0
        else:
            # ── Original single-frame mode ────────────────────────────────────────────
            if (
                self.actions_from_chunk_completed == 0
                or self.actions_from_chunk_completed >= self.open_loop_horizon
            ):
                ext_frame = image_tools.resize_with_pad(curr_obs["right_image"], 256, 256)
                wrist_frame = image_tools.resize_with_pad(curr_obs["wrist_image"], 256, 256)
                request_data = {
                    "observation/exterior_image_0_left": ext_frame,
                    "observation/wrist_image_left": wrist_frame,
                    "observation/joint_position": curr_obs["joint_position"],
                    "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                    "observation/gripper_position": curr_obs["gripper_position"],
                    "prompt": instruction,
                    "session_id": self.session_id,
                }
                result = self.client.infer(request_data)
                actions = result["actions"] if isinstance(result, dict) and "actions" in result else result
                actions = np.asarray(actions, dtype=np.float32)
                if actions.ndim != 2:
                    raise ValueError(f"Expected 2D action chunk, got {actions.shape}")
                self.pred_action_chunk = actions
                self.actions_from_chunk_completed = 0

        assert self.pred_action_chunk is not None
        action = np.asarray(self.pred_action_chunk[self.actions_from_chunk_completed]).copy()
        self.actions_from_chunk_completed += 1

        action[-1] = 1.0 if action[-1] > self.gripper_open_threshold else 0.0

        viz = np.concatenate(
            [
                image_tools.resize_with_pad(curr_obs["right_image"], 224, 224),
                image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
            ],
            axis=1,
        )
        return action, viz


@dataclass
class EpisodeResult:
    benchmark: str
    task_id: int
    task_name: str
    instruction: str
    episode_index: int
    success: bool
    progress: float
    return_sum: float
    steps: int
    video_path: str | None


def run_task_episodes(
    env: Any,
    client: DreamZeroLiberoClient,
    task_id: int,
    task_name: str,
    instruction: str,
    start_episode: int,
    episodes: int,
    max_steps: int,
    output_dir: Path,
    benchmark_name: str,
    init_states: list[Any] | None,
) -> list[EpisodeResult]:
    bounds = get_action_bounds(env)
    results: list[EpisodeResult] = []
    task_dir = output_dir / safe_name(task_name)
    task_dir.mkdir(parents=True, exist_ok=True)

    for episode_index in range(start_episode, episodes):
        init_state = None if not init_states else init_states[episode_index % len(init_states)]
        obs = reset_env_with_init_state(env, init_state)
        # Pass the path for the *previous* episode's generated video so the server
        # saves it directly alongside the rendered episode video on reset.
        prev_gen_path = (
            str(task_dir / f"episode_{episode_index - 1:03d}_gen.mp4")
            if episode_index > start_episode
            else ""
        )
        client.reset(gen_video_path=prev_gen_path)
        frames: list[np.ndarray] = []
        return_sum = 0.0
        final_success = False
        final_progress = 0.0
        episode_steps = 0

        for step_index in range(max_steps):
            action, viz = client.infer(obs, instruction)
            if bounds is not None:
                low, high = bounds
                if action.shape[-1] != low.shape[-1]:
                    raise ValueError(
                        f"DreamZero produced {action.shape[-1]} action dims but the environment expects "
                        f"{low.shape[-1]}. Configure LIBERO with a controller that matches the "
                        "DreamZero output action space."
                    )
                action = np.clip(action, low, high)

            obs, reward, done, info = step_env(env, action)
            frames.append(viz)
            return_sum += reward
            episode_steps = step_index + 1

            final_success = extract_success(env, info, reward, done)
            final_progress = max(final_progress, extract_progress(info, final_success))
            if final_success or done:
                break

        video_path = task_dir / f"episode_{episode_index:03d}.mp4"
        if frames:
            imageio.mimsave(video_path, frames, fps=10)
            saved_video_path = str(video_path)
        else:
            saved_video_path = None

        results.append(
            EpisodeResult(
                benchmark=benchmark_name,
                task_id=task_id,
                task_name=task_name,
                instruction=instruction,
                episode_index=episode_index,
                success=final_success,
                progress=final_progress,
                return_sum=return_sum,
                steps=episode_steps,
                video_path=saved_video_path,
            )
        )

    # Flush the last episode's generated video directly to its target path.
    if episodes > start_episode:
        last_gen_path = str(task_dir / f"episode_{episodes - 1:03d}_gen.mp4")
        client.reset(gen_video_path=last_gen_path)

    return results


def summarize_results(results: list[EpisodeResult]) -> dict[str, Any]:
    if not results:
        return {"episodes": 0, "success_rate": 0.0, "avg_progress": 0.0, "avg_return": 0.0}

    return {
        "episodes": len(results),
        "success_rate": float(np.mean([r.success for r in results])),
        "avg_progress": float(np.mean([r.progress for r in results])),
        "avg_return": float(np.mean([r.return_sum for r in results])),
        "avg_steps": float(np.mean([r.steps for r in results])),
    }


def load_existing_episode_results(summary_path: Path) -> list[EpisodeResult]:
    if not summary_path.exists():
        return []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    results: list[EpisodeResult] = []
    for item in payload.get("episodes", []):
        try:
            results.append(EpisodeResult(**item))
        except Exception:
            continue
    return results


def parse_task_ids(task_ids_arg: str | None, n_tasks: int) -> list[int]:
    if not task_ids_arg:
        return list(range(n_tasks))
    task_ids: list[int] = []
    for chunk in task_ids_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        task_id = int(chunk)
        if task_id < 0 or task_id >= n_tasks:
            raise ValueError(f"Task id {task_id} out of range [0, {n_tasks})")
        task_ids.append(task_id)
    return task_ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DreamZero on LIBERO benchmarks")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--benchmark", default="libero_goal")
    parser.add_argument("--task-ids", default=None, help="Comma separated task ids. Default: all tasks.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--open-loop-horizon", type=int, default=5)
    parser.add_argument(
        "--sync-eval",
        action="store_true",
        default=False,
        help=(
            "Sync eval mode: execute the full action chunk before the next inference call and "
            "send all collected frames (downsampled to 8 at stride 2) to the server. "
            "Eliminates chunk overlap and aligns with training VIDEO_CHUNK_OFFSETS. "
            "Requires --open-loop-horizon to equal the action chunk size (16 for LIBERO). "
            "Default: off (original single-frame mode)."
        ),
    )
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument(
        "--camera-names",
        default="agentview,robot0_eye_in_hand,sideview",
        help="Comma separated LIBERO camera names for offscreen rendering.",
    )
    parser.add_argument("--exterior-image-key", default="agentview_image")
    parser.add_argument("--secondary-exterior-image-key", default="sideview_image")
    parser.add_argument("--wrist-image-key", default="robot0_eye_in_hand_image")
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rotate-images",
        action="store_true",
        default=False,
        help=(
            "Apply 180-degree rotation to all camera images before sending to the server. "
            "Enable this if the training dataset was collected with physically inverted cameras "
            "(e.g. real DROID). LIBERO simulation cameras are already upright, so this should "
            "be left off unless the training data was explicitly rotated during conversion."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    benchmark_module, env_module = load_libero_modules()
    suite = resolve_benchmark_suite(benchmark_module, args.benchmark)
    n_tasks = get_num_tasks(suite)
    task_ids = parse_task_ids(args.task_ids, n_tasks)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs") / datetime.utcnow().strftime("%Y-%m-%d") / datetime.utcnow().strftime("%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    existing_results = load_existing_episode_results(summary_path) if args.resume else []

    client = DreamZeroLiberoClient(
        remote_host=args.host,
        remote_port=args.port,
        open_loop_horizon=args.open_loop_horizon,
        exterior_key=args.exterior_image_key,
        secondary_exterior_key=args.secondary_exterior_image_key,
        wrist_key=args.wrist_image_key,
        joint_keys=DEFAULT_JOINT_KEYS,
        gripper_keys=DEFAULT_GRIPPER_KEYS,
        gripper_open_threshold=args.gripper_open_threshold,
        rotate_images=args.rotate_images,
        sync_eval=args.sync_eval,
    )

    all_episode_results: list[EpisodeResult] = [
        item
        for item in existing_results
        if item.benchmark == args.benchmark and item.task_id in task_ids
    ]
    per_task_summary: dict[str, Any] = {}
    camera_names = [name.strip() for name in args.camera_names.split(",") if name.strip()]

    for task_id in task_ids:
        task = get_task_object(suite, task_id)
        task_name = get_task_name(task, task_id)
        instruction = get_task_instruction(suite, task, task_id)
        init_states = get_task_init_states(suite, task, task_id)
        bddl_file_name = get_task_bddl_path(suite, task, task_id)
        existing_task_results = [
            item for item in existing_results if item.benchmark == args.benchmark and item.task_id == task_id
        ]
        start_episode = len(existing_task_results)

        env = make_offscreen_env(
            env_module,
            bddl_file_name=bddl_file_name,
            camera_names=camera_names,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            controller_name=args.controller,
            control_freq=args.control_freq,
        )
        try:
            task_results = run_task_episodes(
                env=env,
                client=client,
                task_id=task_id,
                task_name=task_name,
                instruction=instruction,
                start_episode=start_episode,
                episodes=args.episodes,
                max_steps=args.max_steps,
                output_dir=output_dir,
                benchmark_name=args.benchmark,
                init_states=init_states,
            )
            if task_results:
                all_episode_results = [
                    item for item in all_episode_results if not (item.benchmark == args.benchmark and item.task_id == task_id)
                ]
                all_episode_results.extend(existing_task_results)
                all_episode_results.extend(task_results)
            per_task_summary[task_name] = summarize_results(
                [item for item in all_episode_results if item.benchmark == args.benchmark and item.task_id == task_id]
            )
        finally:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()

    summary = {
        "benchmark": args.benchmark,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task_ids": task_ids,
        "aggregate": summarize_results(all_episode_results),
        "per_task": per_task_summary,
        "episodes": [asdict(item) for item in all_episode_results],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved detailed results to {summary_path}")


if __name__ == "__main__":
    main()
