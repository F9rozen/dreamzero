"""
Example script for running 10 rollouts of a DROID policy on the example environment.

Usage:

First, make sure you download the simulation assets and unpack them into the root directory of this package.

Then, in a separate terminal, launch the policy server on localhost:8000 
-- make sure to set XLA_PYTHON_CLIENT_MEM_FRACTION to avoid JAX hogging all the GPU memory.

For example, to launch a pi0-FAST-DROID policy (with joint position control), 
run the command below in a separate terminal from the openpi "karl/droid_policies" branch:

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid_jointpos --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos

Finally, run the evaluation script:

python run_eval.py --episodes 10 --headless
"""

import uuid

import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from openpi_client import image_tools
from sim_evals.inference.abstract_client import InferenceClient
from policy_client import WebsocketClientPolicy


class DreamZeroJointPosClient(InferenceClient):
    def __init__(self,
                remote_host:str = "localhost",
                remote_port:int = 6000,
                open_loop_horizon:int = 8,
                sync_eval:bool = False,
    ) -> None:
        self.client = WebsocketClientPolicy(remote_host, remote_port)
        self.sync_eval = sync_eval
        self.open_loop_horizon = open_loop_horizon
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())
        # Per-camera observation buffers for full-chunk sync inference.
        self.obs_buffer_right: list[np.ndarray] = []
        self.obs_buffer_left: list[np.ndarray] = []
        self.obs_buffer_wrist: list[np.ndarray] = []

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        right_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        left_img = image_tools.resize_with_pad(curr_obs["left_image"], 224, 224)
        combined = np.concatenate([right_img, wrist_img, left_img], axis=1)
        return combined

    def reset(self, gen_video_path: str = "") -> None:
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())
        self.obs_buffer_right = []
        self.obs_buffer_left = []
        self.obs_buffer_wrist = []
        reset_info: dict = {}
        if gen_video_path:
            reset_info["gen_video_path"] = gen_video_path
        try:
            self.client.reset(reset_info)
        except Exception:
            pass

    def infer(self, obs: dict, instruction: str) -> dict:
        """
        Infer the next action from the policy in a server-client setup.
        """
        curr_obs = self._extract_observation(obs)

        if self.sync_eval:
            # ── Sync mode: execute a full action chunk, then send all collected frames ──
            # DROID training uses 24 actions at 15fps → 8 video frames at 5fps (stride 3).
            right_frame = image_tools.resize_with_pad(curr_obs["right_image"], 180, 320)
            left_frame  = image_tools.resize_with_pad(curr_obs["left_image"],  180, 320)
            wrist_frame = image_tools.resize_with_pad(curr_obs["wrist_image"], 180, 320)

            if self.actions_from_chunk_completed == 0:
                # Episode start: single frame triggers server AR reset.
                request_data = {
                    "observation/exterior_image_0_left": right_frame,
                    "observation/exterior_image_1_left": left_frame,
                    "observation/wrist_image_left": wrist_frame,
                    "observation/joint_position": curr_obs["joint_position"].astype(np.float64),
                    "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                    "observation/gripper_position": curr_obs["gripper_position"].astype(np.float64),
                    "prompt": instruction,
                    "session_id": self.session_id,
                }
                self.obs_buffer_right = [right_frame]
                self.obs_buffer_left  = [left_frame]
                self.obs_buffer_wrist = [wrist_frame]
                result = self.client.infer(request_data)
                actions = result["actions"] if isinstance(result, dict) else result
                assert actions.ndim == 2 and actions.shape[-1] == 8
                self.pred_action_chunk = actions
                self.actions_from_chunk_completed = 0
            else:
                self.obs_buffer_right.append(right_frame)
                self.obs_buffer_left.append(left_frame)
                self.obs_buffer_wrist.append(wrist_frame)

                if self.actions_from_chunk_completed >= self.open_loop_horizon:
                    # Full chunk done: downsample to 8 frames with stride 3
                    # (24 frames at 15fps → 8 frames at 5fps).
                    right_chunk = np.stack(self.obs_buffer_right[::3][:8])   # (8,H,W,3)
                    left_chunk  = np.stack(self.obs_buffer_left[::3][:8])    # (8,H,W,3)
                    wrist_chunk = np.stack(self.obs_buffer_wrist[::3][:8])   # (8,H,W,3)
                    request_data = {
                        "observation/exterior_image_0_left": right_chunk,
                        "observation/exterior_image_1_left": left_chunk,
                        "observation/wrist_image_left": wrist_chunk,
                        "observation/joint_position": curr_obs["joint_position"].astype(np.float64),
                        "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                        "observation/gripper_position": curr_obs["gripper_position"].astype(np.float64),
                        "prompt": instruction,
                        "session_id": self.session_id,
                    }
                    self.obs_buffer_right = [right_frame]
                    self.obs_buffer_left  = [left_frame]
                    self.obs_buffer_wrist = [wrist_frame]
                    result = self.client.infer(request_data)
                    actions = result["actions"] if isinstance(result, dict) else result
                    assert actions.ndim == 2 and actions.shape[-1] == 8
                    self.pred_action_chunk = actions
                    self.actions_from_chunk_completed = 0
        else:
            # ── Original single-frame mode ────────────────────────────────────────────
            if (
                self.actions_from_chunk_completed == 0
                or self.actions_from_chunk_completed >= self.open_loop_horizon
            ):
                self.actions_from_chunk_completed = 0
                request_data = {
                    "observation/exterior_image_0_left": image_tools.resize_with_pad(curr_obs["right_image"], 180, 320),
                    "observation/exterior_image_1_left": image_tools.resize_with_pad(curr_obs["left_image"],  180, 320),
                    "observation/wrist_image_left":      image_tools.resize_with_pad(curr_obs["wrist_image"], 180, 320),
                    "observation/joint_position": curr_obs["joint_position"].astype(np.float64),
                    "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                    "observation/gripper_position": curr_obs["gripper_position"].astype(np.float64),
                    "prompt": instruction,
                    "session_id": self.session_id,
                }
                result = self.client.infer(request_data)
                actions = result["actions"] if isinstance(result, dict) else result
                assert actions.ndim == 2 and actions.shape[-1] == 8
                self.pred_action_chunk = actions

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        img3 = image_tools.resize_with_pad(curr_obs["left_image"], 224, 224)
        both = np.concatenate([img1, img2, img3], axis=1)

        return {"action": action, "viz": both}

    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        # Assign images
        right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        left_image = obs_dict["policy"]["external_cam_2"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "left_image": left_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }




def main(
        episodes: int = 10,
        scene: int = 1,
        headless: bool = True,
        host: str = "localhost",
        port: int = 6000,
        max_steps: int = -1,
        start_episode: int = 0,
        output_dir: str = "",
        open_loop_horizon: int = 8,
        sync_eval: bool = False,
        ):
    # launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # All IsaacLab dependent modules should be imported after the app is launched
    import sim_evals.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg


    # Initialize the env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    instruction = None
    match scene:
        case 1:
            instruction = "put the cube in the bowl"
        case 2:
            instruction = "pick up the can and put it in the mug"
        case 3:
            instruction = "put the banana in the bin"
        case _:
            raise ValueError(f"Scene {scene} not supported")
        
    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials
    client = DreamZeroJointPosClient(
        remote_host=host,
        remote_port=port,
        open_loop_horizon=open_loop_horizon,
        sync_eval=sync_eval,
    )


    video_dir = Path(output_dir) if output_dir else Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    rollout_steps = env.env.max_episode_length
    if max_steps > 0:
        rollout_steps = min(rollout_steps, max_steps)
    with torch.no_grad():
        for ep in range(start_episode, episodes):
            for _ in tqdm(range(rollout_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs, instruction)
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

            mediapy.write_video(
                video_dir / f"episode_{ep}.mp4",
                video,
                fps=15,
            )
            video = []
            client.reset(gen_video_path=str(video_dir / f"episode_{ep}_gen.mp4"))

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)
