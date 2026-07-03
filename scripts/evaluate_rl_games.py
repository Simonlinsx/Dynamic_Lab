#!/usr/bin/env python3
"""Evaluate an rl_games checkpoint on Revo2 static grasp tasks and save successful videos."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))
REPO_ROOT = Path(__file__).resolve().parents[1]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToolReal-Revo2-Franka-StaticBall-Grasp-Direct-v0", help="Gym task id.")
parser.add_argument("--checkpoint", required=True, help="rl_games .pth checkpoint.")
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=64, help="Eval env count.")
parser.add_argument("--episodes", type=int, default=128, help="Number of completed eval episodes.")
parser.add_argument("--max-steps", type=int, default=None, help="Safety cap for vectorized eval steps.")
parser.add_argument("--success-threshold", type=float, default=0.30, help="Required success rate.")
parser.add_argument(
    "--skip-vector-eval",
    action="store_true",
    help="Skip the non-rendered vector evaluation and only collect requested success videos.",
)
parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic policy actions.")
parser.add_argument("--stochastic", action="store_false", dest="deterministic", help="Use sampled policy actions.")
parser.add_argument("--rl-device", default=None, help="Device for policy inference. Defaults to --device.")
parser.add_argument(
    "--dynamic-curriculum-alpha",
    type=float,
    default=None,
    help="Override dynamic speed curriculum alpha for evaluation. Use 1.0 for full-speed eval.",
)
parser.add_argument(
    "--tabletop-asset-curriculum-alpha",
    type=float,
    default=None,
    help="Override tabletop asset curriculum alpha for evaluation. Use 1.0 to sample all tabletop assets.",
)
parser.add_argument(
    "--tabletop-motion-curriculum-alpha",
    type=float,
    default=None,
    help="Override tabletop motion-mode curriculum alpha for evaluation. Use 1.0 to sample all motion modes.",
)
parser.add_argument(
    "--output-dir",
    default=str(REPO_ROOT / "outputs" / "eval_rl_games"),
    help="Directory for JSON summaries and videos.",
)
parser.add_argument("--save-success-videos", type=int, default=1, help="Number of successful mp4 videos to save.")
parser.add_argument("--video-attempts", type=int, default=20, help="Max rendered rollout attempts to find successful videos.")
parser.add_argument(
    "--video-envs",
    type=int,
    default=4,
    help="Number of parallel environments shown in the render viewport while searching for success videos.",
)
parser.add_argument("--video-stride", type=int, default=2, help="Record every N control steps.")
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--video-max-steps", type=int, default=None, help="Optional control-step cap for rendered videos.")
parser.add_argument(
    "--save-trial-sequence-videos",
    type=int,
    default=0,
    help="Number of continuous mp4 videos to save. Each video contains multiple reset trials.",
)
parser.add_argument(
    "--trial-sequence-trials",
    type=int,
    default=20,
    help="Number of reset trials to concatenate into each trial-sequence video.",
)
parser.add_argument(
    "--trial-sequence-env-id",
    type=int,
    default=0,
    help="Rendered env id to record for trial-sequence videos.",
)
parser.add_argument(
    "--trial-sequence-pre-roll-frames",
    type=int,
    default=6,
    help="Number of reset-state frames to include before each trial starts.",
)
parser.add_argument(
    "--trial-sequence-gap-frames",
    type=int,
    default=6,
    help="Number of black separator frames inserted after each trial in a sequence video.",
)
parser.add_argument(
    "--video-post-success-steps",
    type=int,
    default=0,
    help="Continue recording this many control steps after first success before saving the video.",
)
parser.add_argument(
    "--video-require-post-success-stable",
    action="store_true",
    help="Only save a success video if the post-success trace remains stable enough.",
)
parser.add_argument(
    "--video-post-success-min-stable-frac",
    type=float,
    default=0.90,
    help="Minimum stable/lifted/true-grasp fraction in the post-success video window.",
)
parser.add_argument(
    "--video-post-success-require-final-stable",
    action="store_true",
    help="Require the final post-success trace sample to still be stable.",
)
parser.add_argument(
    "--save-rollout-videos-on-failure",
    type=int,
    default=0,
    help="Save up to N rollout videos before closing the rendered env when no success video is found.",
)
parser.add_argument(
    "--save-rollout-videos-each-failed-attempt",
    action="store_true",
    help="When enabled, save debug rollout videos for every failed video attempt instead of only the last attempt.",
)
parser.add_argument(
    "--allow-rollout-video-fallback",
    action="store_true",
    help="Treat saved rollout debug videos as a valid video artifact when no success video is found.",
)
parser.add_argument(
    "--video-debug-interval",
    type=int,
    default=0,
    help="Print rendered rollout success/lift/stable counts every N control steps when positive.",
)
parser.add_argument(
    "--video-camera-track-object",
    action="store_true",
    default=False,
    help="Keep the video camera aimed at the object. By default the camera is fixed in a stable third view.",
)
parser.add_argument(
    "--no-video-camera-track-object",
    action="store_false",
    dest="video_camera_track_object",
    help="Use a static video camera target instead of tracking the object.",
)
parser.add_argument(
    "--video-camera-update-every-frame",
    action="store_true",
    default=False,
    help="Refresh the tracking camera pose before each captured frame.",
)
parser.add_argument(
    "--no-video-camera-update-every-frame",
    action="store_false",
    dest="video_camera_update_every_frame",
    help="Set the tracking camera after reset only.",
)
parser.add_argument(
    "--video-camera-track-offset",
    type=float,
    nargs=3,
    default=(0.65, 0.95, 0.65),
    metavar=("X", "Y", "Z"),
    help="Camera position offset from the tracked target. Defaults to the IsaacGym-aligned Revo2 side view.",
)
parser.add_argument(
    "--video-camera-eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Explicit fixed camera eye in env-local coordinates. Overrides --video-camera-track-offset.",
)
parser.add_argument(
    "--video-camera-target",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Explicit fixed camera target in env-local coordinates. Overrides object/track target selection.",
)
parser.add_argument(
    "--video-camera-track-target-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.05),
    metavar=("X", "Y", "Z"),
    help="Target offset from the object position. Defaults to simtoolreal's wide video view.",
)
parser.add_argument(
    "--video-camera-focal-length",
    type=float,
    default=None,
    help="Optional focal length override for rendered success videos. Larger values zoom in.",
)
parser.add_argument(
    "--video-camera-resolution",
    type=int,
    nargs=2,
    default=None,
    metavar=("WIDTH", "HEIGHT"),
    help="Optional tiled-camera resolution override for rendered success videos.",
)
parser.add_argument("--wandb-project", default=None, help="Enable W&B logging under this project.")
parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team.")
parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name.")
parser.add_argument("--wandb-group", default=None, help="Optional W&B group.")
parser.add_argument("--wandb-tags", nargs="*", default=None, help="Optional W&B tags.")
parser.add_argument(
    "--wandb-mode",
    choices=("online", "offline", "disabled"),
    default=os.environ.get("WANDB_MODE", "online"),
    help="W&B mode. Defaults to WANDB_MODE or online.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if (args_cli.save_success_videos > 0 or args_cli.save_trial_sequence_videos > 0) and hasattr(
    args_cli, "enable_cameras"
):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper  # noqa: E402
from rl_games.algos_torch import torch_ext  # noqa: E402
from rl_games.algos_torch.players import PpoPlayerContinuous  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


def _trace(message: str) -> None:
    print(f"[EVAL] {message}", flush=True)


def _maybe_init_wandb(checkpoint: Path, output_dir: Path):
    if not args_cli.wandb_project or args_cli.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requested, but wandb is not installed in this environment.") from exc

    return wandb.init(
        project=args_cli.wandb_project,
        entity=args_cli.wandb_entity,
        name=args_cli.wandb_run_name or f"eval_{checkpoint.stem}",
        group=args_cli.wandb_group,
        tags=args_cli.wandb_tags,
        mode=args_cli.wandb_mode,
        dir=str(output_dir),
        job_type="eval",
        config={
            "task": args_cli.task,
            "checkpoint": str(checkpoint),
            "num_envs": args_cli.num_envs,
            "episodes": args_cli.episodes,
            "success_threshold": args_cli.success_threshold,
            "deterministic": bool(args_cli.deterministic),
            "dynamic_curriculum_alpha": args_cli.dynamic_curriculum_alpha,
            "tabletop_asset_curriculum_alpha": args_cli.tabletop_asset_curriculum_alpha,
            "tabletop_motion_curriculum_alpha": args_cli.tabletop_motion_curriculum_alpha,
            "device": args_cli.device,
            "rl_device": args_cli.rl_device or args_cli.device,
            "save_success_videos": args_cli.save_success_videos,
            "save_trial_sequence_videos": args_cli.save_trial_sequence_videos,
            "trial_sequence_trials": args_cli.trial_sequence_trials,
            "video_camera_track_offset": list(args_cli.video_camera_track_offset),
            "video_camera_track_target_offset": list(args_cli.video_camera_track_target_offset),
            "video_camera_eye": list(args_cli.video_camera_eye) if args_cli.video_camera_eye is not None else None,
            "video_camera_target": list(args_cli.video_camera_target)
            if args_cli.video_camera_target is not None
            else None,
            "video_camera_track_object": bool(args_cli.video_camera_track_object),
            "video_camera_update_every_frame": bool(args_cli.video_camera_update_every_frame),
            "video_camera_focal_length": args_cli.video_camera_focal_length,
            "video_camera_resolution": args_cli.video_camera_resolution,
        },
    )


def _log_summary_to_wandb(wandb_run, summary: dict, summary_path: Path) -> None:
    if wandb_run is None:
        return

    import wandb

    eval_summary = summary.get("eval", {})
    metrics = {
        "eval/passed": float(bool(summary.get("passed", False))),
        "eval/success_threshold": float(summary.get("success_threshold", 0.0)),
    }
    for key, value in eval_summary.items():
        if isinstance(value, bool):
            metrics[f"eval/{key}"] = float(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"eval/{key}"] = float(value)
    for key, value in eval_summary.get("last_log", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"eval/last_log/{key}"] = float(value)

    wandb.log(metrics)
    wandb.save(str(summary_path), base_path=str(summary_path.parent))
    for trace_path in summary.get("success_video_traces", []):
        wandb.save(str(trace_path), base_path=str(summary_path.parent))
    for index, video_path in enumerate(summary.get("success_videos", [])):
        wandb.log({f"videos/success_{index:03d}": wandb.Video(video_path, fps=args_cli.video_fps, format="mp4")})
    for index, video_path in enumerate(summary.get("rollout_videos", [])):
        wandb.log({f"videos/rollout_{index:03d}": wandb.Video(video_path, fps=args_cli.video_fps, format="mp4")})
    for trace_path in summary.get("trial_sequence_traces", []):
        wandb.save(str(trace_path), base_path=str(summary_path.parent))
    for index, video_path in enumerate(summary.get("trial_sequence_videos", [])):
        wandb.log(
            {f"videos/trial_sequence_{index:03d}": wandb.Video(video_path, fps=args_cli.video_fps, format="mp4")}
        )


def _load_agent_cfg(task: str) -> dict:
    cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    return copy.deepcopy(cfg)


def _make_env(task: str, agent_cfg: dict, num_envs: int, render_mode: str | None):
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=num_envs)
    if args_cli.dynamic_curriculum_alpha is not None and hasattr(
        env_cfg, "dynamic_grasp_speed_curriculum_override_alpha"
    ):
        env_cfg.dynamic_grasp_speed_curriculum_override_alpha = float(args_cli.dynamic_curriculum_alpha)
    if args_cli.tabletop_asset_curriculum_alpha is not None and hasattr(
        env_cfg, "tabletop_asset_curriculum_override_alpha"
    ):
        env_cfg.tabletop_asset_curriculum_override_alpha = float(args_cli.tabletop_asset_curriculum_alpha)
    if args_cli.tabletop_motion_curriculum_alpha is not None and hasattr(
        env_cfg, "tabletop_motion_mode_curriculum_override_alpha"
    ):
        env_cfg.tabletop_motion_mode_curriculum_override_alpha = float(args_cli.tabletop_motion_curriculum_alpha)
    if render_mode == "rgb_array" and hasattr(env_cfg, "video_camera_enabled"):
        env_cfg.video_camera_enabled = True
        if hasattr(env_cfg, "terminate_on_success"):
            env_cfg.terminate_on_success = False
        if args_cli.video_camera_focal_length is not None:
            env_cfg.video_camera.spawn.focal_length = float(args_cli.video_camera_focal_length)
        if args_cli.video_camera_resolution is not None:
            env_cfg.video_camera.width = int(args_cli.video_camera_resolution[0])
            env_cfg.video_camera.height = int(args_cli.video_camera_resolution[1])
    env = gym.make(task, cfg=env_cfg, render_mode=render_mode)
    env.unwrapped.sim._app_control_on_stop_handle = None
    if render_mode == "rgb_array":
        if not _set_video_camera_poses(env.unwrapped):
            _set_render_camera(env.unwrapped)
    env_block = agent_cfg["params"]["env"]
    rl_device = args_cli.rl_device or args_cli.device
    return RlGamesVecEnvWrapper(
        env,
        rl_device,
        float(env_block.get("clip_observations", 100.0)),
        float(env_block.get("clip_actions", 1.0)),
    )


def _set_video_camera_poses(unwrapped_env) -> bool:
    camera = getattr(unwrapped_env, "_video_camera", None)
    if camera is None:
        return False
    origins = unwrapped_env.scene.env_origins
    if args_cli.video_camera_eye is not None and args_cli.video_camera_target is not None:
        eye = torch.tensor(args_cli.video_camera_eye, device=origins.device, dtype=origins.dtype).unsqueeze(0)
        targets = torch.tensor(args_cli.video_camera_target, device=origins.device, dtype=origins.dtype).unsqueeze(0)
        camera.set_world_poses_from_view(origins + eye, origins + targets)
        if hasattr(camera, "_update_poses") and hasattr(camera, "_ALL_INDICES"):
            camera._update_poses(camera._ALL_INDICES)
        return True

    target_offset = torch.tensor(
        args_cli.video_camera_track_target_offset, device=origins.device, dtype=origins.dtype
    ).unsqueeze(0)
    camera_offset = torch.tensor(
        args_cli.video_camera_track_offset, device=origins.device, dtype=origins.dtype
    ).unsqueeze(0)

    object_root_pos = None
    if bool(args_cli.video_camera_track_object):
        object_asset = getattr(unwrapped_env, "object", None)
        object_data = getattr(object_asset, "data", None)
        object_root_pos = getattr(object_data, "root_pos_w", None)

    if object_root_pos is not None:
        targets = object_root_pos[:, :3] + target_offset
    else:
        try:
            object_init_pos = torch.tensor(
                unwrapped_env.cfg.object_cfg.init_state.pos, device=origins.device, dtype=origins.dtype
            ).unsqueeze(0)
        except AttributeError:
            object_init_pos = torch.tensor((0.61, -0.17, 0.326), device=origins.device, dtype=origins.dtype).unsqueeze(0)
        targets = origins + object_init_pos + target_offset
    camera.set_world_poses_from_view(targets + camera_offset, targets)
    if hasattr(camera, "_update_poses") and hasattr(camera, "_ALL_INDICES"):
        camera._update_poses(camera._ALL_INDICES)
    return True


def _set_render_camera(unwrapped_env, focus_env_id: int | None = None) -> None:
    origins = unwrapped_env.scene.env_origins.detach().cpu()
    if focus_env_id is None:
        min_xy = origins[:, :2].min(dim=0).values
        max_xy = origins[:, :2].max(dim=0).values
        center_xy = 0.5 * (min_xy + max_xy)
        extent = float(torch.linalg.norm(max_xy - min_xy).item())
        target = (float(center_xy[0]), float(center_xy[1]), 0.56)
        eye = (
            target[0] + 0.8 + 0.6 * extent,
            target[1] - 1.0 - 0.7 * extent,
            target[2] + 0.5 + 0.35 * extent,
        )
    else:
        origin = origins[int(focus_env_id)]
        target = (float(origin[0]), float(origin[1]), 0.56)
        eye = (float(origin[0]) + 0.66, float(origin[1]) - 0.84, 0.82)
    unwrapped_env.sim.set_camera_view(eye=eye, target=target)


def _make_player(agent_cfg: dict, wrapped_env: RlGamesVecEnvWrapper, checkpoint: Path) -> PpoPlayerContinuous:
    params = copy.deepcopy(agent_cfg["params"])
    if params["algo"]["name"] != "a2c_continuous":
        raise NotImplementedError(f"Unsupported rl_games algo: {params['algo']['name']}")
    config = params["config"]
    rl_device = args_cli.rl_device or args_cli.device
    config["device"] = rl_device
    config["device_name"] = rl_device
    config["num_actors"] = wrapped_env.num_envs
    config["env_info"] = wrapped_env.get_env_info()
    config["vec_env"] = wrapped_env
    config.setdefault("player", {})
    config["player"]["deterministic"] = bool(args_cli.deterministic)
    config["player"]["print_stats"] = False

    player = PpoPlayerContinuous(params)
    player.has_batch_dimension = True
    player.batch_size = wrapped_env.num_envs
    original_load_checkpoint = torch_ext.load_checkpoint

    def load_checkpoint_on_eval_device(filename):
        print("=> loading checkpoint '{}'".format(filename))
        map_location = args_cli.rl_device or args_cli.device
        return torch_ext.safe_filesystem_op(torch.load, filename, map_location=map_location)

    try:
        torch_ext.load_checkpoint = load_checkpoint_on_eval_device
        player.restore(str(checkpoint))
    finally:
        torch_ext.load_checkpoint = original_load_checkpoint
    player.reset()
    return player


def _tensor_bool(extras: dict, key: str, num_envs: int, device: str) -> torch.Tensor:
    value = extras.get(key)
    if value is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return value.bool()


def _run_vector_eval(agent_cfg: dict, checkpoint: Path) -> dict:
    wrapped_env = _make_env(args_cli.task, agent_cfg, args_cli.num_envs, render_mode=None)
    try:
        player = _make_player(agent_cfg, wrapped_env, checkpoint)
        obs = wrapped_env.reset()
        device = wrapped_env.unwrapped.device
        num_envs = wrapped_env.num_envs
        max_episode_length = int(wrapped_env.unwrapped.max_episode_length)
        max_steps = args_cli.max_steps or (max_episode_length * (math.ceil(args_cli.episodes / num_envs) + 2))

        episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        episode_true_grasp = torch.zeros_like(episode_success)
        episode_lifted = torch.zeros_like(episode_success)
        episode_stable_hold = torch.zeros_like(episode_success)
        completed = 0
        success_count = 0
        true_count = 0
        lifted_count = 0
        stable_count = 0
        total_reward = 0.0
        total_reward_count = 0
        last_episode_log = {}

        for step in range(max_steps):
            with torch.no_grad():
                actions = player.get_action(obs, is_deterministic=args_cli.deterministic)
            if torch.is_tensor(actions):
                actions = actions.detach().clone()
            obs, rewards, dones, extras = wrapped_env.step(actions)
            episode_success |= _tensor_bool(extras, "success_env", num_envs, device)
            episode_true_grasp |= _tensor_bool(extras, "true_grasp_env", num_envs, device)
            episode_lifted |= _tensor_bool(extras, "lifted_env", num_envs, device)
            episode_stable_hold |= _tensor_bool(extras, "stable_hold_env", num_envs, device)
            total_reward += float(rewards.sum().item())
            total_reward_count += int(rewards.numel())
            if "episode" in extras:
                last_episode_log = {
                    key: float(value.item()) for key, value in extras["episode"].items() if hasattr(value, "item")
                }

            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                completed += int(done_ids.numel())
                success_count += int(episode_success[done_ids].sum().item())
                true_count += int(episode_true_grasp[done_ids].sum().item())
                lifted_count += int(episode_lifted[done_ids].sum().item())
                stable_count += int(episode_stable_hold[done_ids].sum().item())
                episode_success[done_ids] = False
                episode_true_grasp[done_ids] = False
                episode_lifted[done_ids] = False
                episode_stable_hold[done_ids] = False
                _trace(
                    f"episodes={completed}/{args_cli.episodes} "
                    f"success_rate={success_count / max(completed, 1):.3f}"
                )
                if completed >= args_cli.episodes:
                    break
        else:
            _trace(f"hit max_steps={max_steps} before completing requested episodes")

        completed = max(completed, 1)
        return {
            "episodes": int(completed),
            "success_count": int(success_count),
            "success_rate": float(success_count / completed),
            "true_grasp_episode_rate": float(true_count / completed),
            "lifted_episode_rate": float(lifted_count / completed),
            "stable_hold_episode_rate": float(stable_count / completed),
            "mean_step_reward": float(total_reward / max(total_reward_count, 1)),
            "max_episode_length": max_episode_length,
            "last_log": last_episode_log,
        }
    finally:
        wrapped_env.close()


def _capture_frame(wrapped_env: RlGamesVecEnvWrapper, frames: list) -> None:
    frame = wrapped_env.env.unwrapped.render(recompute=True)
    if frame is not None and frame.size > 0:
        frames.append(frame)


def _capture_video_frames(wrapped_env: RlGamesVecEnvWrapper, frames_by_env: list[list[np.ndarray]]) -> None:
    unwrapped_env = wrapped_env.env.unwrapped
    camera = getattr(unwrapped_env, "_video_camera", None)
    if camera is None:
        frame = unwrapped_env.render(recompute=True)
        if frame is not None and frame.size > 0:
            for frames in frames_by_env:
                frames.append(frame.copy())
        return

    if bool(args_cli.video_camera_update_every_frame):
        _set_video_camera_poses(unwrapped_env)
    rgb = camera.data.output.get("rgb")
    if rgb is None or rgb.numel() == 0:
        return
    rgb = rgb[..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
    for env_id, frames in enumerate(frames_by_env):
        if env_id < rgb.shape[0] and rgb[env_id].size > 0:
            frames.append(rgb[env_id].copy())


def _frames_have_visual_signal(frames: list[np.ndarray]) -> bool:
    if not frames:
        return False
    stride = max(len(frames) // 5, 1)
    for frame in frames[::stride]:
        if float(frame.max()) > 10.0 and float(frame.std()) > 1.0:
            return True
    return False


def _trace_tensor(unwrapped_env, name: str, num_envs: int, default: float = 0.0) -> torch.Tensor:
    value = getattr(unwrapped_env, name, None)
    if value is None:
        return torch.full((num_envs,), default, device=unwrapped_env.device)
    if not torch.is_tensor(value):
        return torch.as_tensor(value, device=unwrapped_env.device)
    return value[:num_envs]


def _trace_vec_tensor(unwrapped_env, name: str, num_envs: int, width: int, default: float = 0.0) -> torch.Tensor:
    value = getattr(unwrapped_env, name, None)
    if value is None:
        return torch.full((num_envs, width), default, device=unwrapped_env.device)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=unwrapped_env.device)
    value = value[:num_envs]
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    if value.shape[-1] < width:
        pad = torch.full((value.shape[0], width - value.shape[-1]), default, device=unwrapped_env.device)
        value = torch.cat((value, pad), dim=-1)
    return value[..., :width]


def _append_video_trace(
    wrapped_env: RlGamesVecEnvWrapper,
    traces_by_env: list[list[dict]],
    extras: dict,
    step: int,
) -> None:
    unwrapped_env = wrapped_env.env.unwrapped
    num_envs = len(traces_by_env)
    device = wrapped_env.device
    actions = getattr(unwrapped_env, "actions", None)
    if torch.is_tensor(actions):
        actions_cpu = actions[:num_envs].detach().cpu()
    else:
        actions_cpu = None
    hand_action_start = min(7, int(actions_cpu.shape[-1])) if actions_cpu is not None and actions_cpu.ndim == 2 else 7

    hand_joint_names = list(getattr(unwrapped_env, "_sim_hand_joint_names", []))
    hand_joint_ids = list(getattr(unwrapped_env, "_control_hand_joint_ids", []))
    joint_targets = getattr(unwrapped_env, "_joint_targets", None)
    if torch.is_tensor(joint_targets) and hand_joint_ids:
        hand_joint_targets_cpu = joint_targets[:num_envs, hand_joint_ids].detach().cpu()
    else:
        hand_joint_targets_cpu = None
    robot_data = getattr(getattr(unwrapped_env, "robot", None), "data", None)
    joint_pos = getattr(robot_data, "joint_pos", None)
    if torch.is_tensor(joint_pos) and hand_joint_ids:
        hand_joint_pos_cpu = joint_pos[:num_envs, hand_joint_ids].detach().cpu()
    else:
        hand_joint_pos_cpu = None
    object_height_delta = _trace_tensor(unwrapped_env, "_object_height_delta", num_envs)
    object_palm_rel_vel = _trace_tensor(unwrapped_env, "_object_palm_rel_vel", num_envs)
    success_streak = _trace_tensor(unwrapped_env, "_success_streak", num_envs)
    object_pos_w = _trace_vec_tensor(unwrapped_env, "_object_pos_w", num_envs, 3)
    object_lin_vel_w = _trace_vec_tensor(unwrapped_env, "_object_lin_vel_w", num_envs, 3)
    object_ang_vel_w = _trace_vec_tensor(unwrapped_env, "_object_ang_vel_w", num_envs, 3)
    hover_target_pos_w = _trace_vec_tensor(unwrapped_env, "_object_hover_target_pos_w", num_envs, 3)
    hover_latched_value = getattr(unwrapped_env, "_object_hover_target_latched", None)
    if hover_latched_value is None:
        hover_latched = torch.zeros(num_envs, dtype=torch.bool, device=device)
    else:
        if not torch.is_tensor(hover_latched_value):
            hover_latched_value = torch.as_tensor(hover_latched_value, device=device)
        hover_latched = hover_latched_value[:num_envs].bool()
    hover_xy_dist = torch.norm(object_pos_w[:, :2] - hover_target_pos_w[:, :2], dim=-1)
    hover_z_error = torch.abs(object_pos_w[:, 2] - hover_target_pos_w[:, 2])
    hover_object_speed = torch.norm(object_lin_vel_w, dim=-1)
    hover_object_ang_speed = torch.norm(object_ang_vel_w, dim=-1)
    true_grasp = _tensor_bool(extras, "true_grasp_env", num_envs, device)
    lifted = _tensor_bool(extras, "lifted_env", num_envs, device)
    stable_hold = _tensor_bool(extras, "stable_hold_env", num_envs, device)
    success = _tensor_bool(extras, "success_env", num_envs, device)
    success_seen = _tensor_bool(extras, "success_seen_env", num_envs, device)

    object_height_delta_cpu = object_height_delta.detach().cpu()
    object_palm_rel_vel_cpu = object_palm_rel_vel.detach().cpu()
    success_streak_cpu = success_streak.detach().cpu()
    hover_latched_cpu = hover_latched.detach().cpu()
    hover_xy_dist_cpu = hover_xy_dist.detach().cpu()
    hover_z_error_cpu = hover_z_error.detach().cpu()
    hover_object_speed_cpu = hover_object_speed.detach().cpu()
    hover_object_ang_speed_cpu = hover_object_ang_speed.detach().cpu()
    hover_target_pos_w_cpu = hover_target_pos_w.detach().cpu()
    object_pos_w_cpu = object_pos_w.detach().cpu()
    true_grasp_cpu = true_grasp.detach().cpu()
    lifted_cpu = lifted.detach().cpu()
    stable_hold_cpu = stable_hold.detach().cpu()
    success_cpu = success.detach().cpu()
    success_seen_cpu = success_seen.detach().cpu()
    for env_id, env_trace in enumerate(traces_by_env):
        trace_item = {
            "step": int(step),
            "object_height_delta": float(object_height_delta_cpu[env_id].item()),
            "object_palm_rel_vel": float(object_palm_rel_vel_cpu[env_id].item()),
            "success_streak": int(success_streak_cpu[env_id].item()),
            "hover_latched": bool(hover_latched_cpu[env_id].item()),
            "hover_xy_dist": float(hover_xy_dist_cpu[env_id].item()),
            "hover_z_error": float(hover_z_error_cpu[env_id].item()),
            "hover_object_speed": float(hover_object_speed_cpu[env_id].item()),
            "hover_object_ang_speed": float(hover_object_ang_speed_cpu[env_id].item()),
            "hover_target_z": float(hover_target_pos_w_cpu[env_id, 2].item()),
            "object_z": float(object_pos_w_cpu[env_id, 2].item()),
            "true_grasp": bool(true_grasp_cpu[env_id].item()),
            "lifted": bool(lifted_cpu[env_id].item()),
            "stable_hold": bool(stable_hold_cpu[env_id].item()),
            "success_now": bool(stable_hold_cpu[env_id].item()),
            "success": bool(success_cpu[env_id].item()),
            "success_seen": bool(success_seen_cpu[env_id].item()),
        }
        if actions_cpu is not None and actions_cpu.ndim == 2:
            action = actions_cpu[env_id]
            trace_item["action"] = [float(value) for value in action.tolist()]
            trace_item["hand_action"] = [float(value) for value in action[hand_action_start:].tolist()]
        if hand_joint_names:
            trace_item["hand_joint_names"] = hand_joint_names
        if hand_joint_targets_cpu is not None:
            trace_item["hand_joint_target"] = [
                float(value) for value in hand_joint_targets_cpu[env_id].tolist()
            ]
        if hand_joint_pos_cpu is not None:
            trace_item["hand_joint_pos"] = [float(value) for value in hand_joint_pos_cpu[env_id].tolist()]
        env_trace.append(trace_item)


def _write_video_trace(
    video_path: Path,
    trace: list[dict],
    attempt: int,
    env_id: int,
    frame_count: int,
) -> Path:
    lift_values = [float(item["object_height_delta"]) for item in trace]
    success_steps = [int(item["step"]) for item in trace if item.get("success")]
    lifted_steps = [int(item["step"]) for item in trace if item.get("lifted")]
    stable_steps = [int(item["step"]) for item in trace if item.get("stable_hold")]
    hover_latched_steps = [int(item["step"]) for item in trace if item.get("hover_latched")]
    hover_xy_values = [float(item["hover_xy_dist"]) for item in trace if "hover_xy_dist" in item]
    hover_z_values = [float(item["hover_z_error"]) for item in trace if "hover_z_error" in item]
    hover_speed_values = [float(item["hover_object_speed"]) for item in trace if "hover_object_speed" in item]
    trace_payload = {
        "video_path": str(video_path),
        "attempt": int(attempt),
        "env_id": int(env_id),
        "frame_count": int(frame_count),
        "camera_track_object": bool(args_cli.video_camera_track_object),
        "camera_update_every_frame": bool(args_cli.video_camera_update_every_frame),
        "camera_track_offset": list(args_cli.video_camera_track_offset),
        "camera_track_target_offset": list(args_cli.video_camera_track_target_offset),
        "camera_focal_length": args_cli.video_camera_focal_length,
        "camera_resolution": args_cli.video_camera_resolution,
        "max_object_height_delta": max(lift_values) if lift_values else None,
        "final_object_height_delta": lift_values[-1] if lift_values else None,
        "first_lifted_step": lifted_steps[0] if lifted_steps else None,
        "first_stable_hold_step": stable_steps[0] if stable_steps else None,
        "first_success_step": success_steps[0] if success_steps else None,
        "first_hover_latched_step": hover_latched_steps[0] if hover_latched_steps else None,
        "min_hover_xy_dist": min(hover_xy_values) if hover_xy_values else None,
        "final_hover_xy_dist": hover_xy_values[-1] if hover_xy_values else None,
        "min_hover_z_error": min(hover_z_values) if hover_z_values else None,
        "final_hover_z_error": hover_z_values[-1] if hover_z_values else None,
        "min_hover_object_speed": min(hover_speed_values) if hover_speed_values else None,
        "final_hover_object_speed": hover_speed_values[-1] if hover_speed_values else None,
        "success_steps": success_steps,
        "trace": trace,
    }
    trace_path = video_path.with_suffix(".trace.json")
    trace_path.write_text(json.dumps(trace_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return trace_path


def _post_success_trace_passes(
    trace: list[dict],
    success_step: int,
    min_stable_frac: float,
    require_final_stable: bool,
    min_window_steps: int = 0,
) -> tuple[bool, str]:
    window = [item for item in trace if int(item.get("step", -1)) >= int(success_step)]
    if not window:
        return False, "empty post-success trace"
    observed_steps = int(window[-1].get("step", success_step)) - int(success_step)
    if min_window_steps > 0 and observed_steps < int(min_window_steps):
        return False, f"short post-success trace (steps={observed_steps}/{int(min_window_steps)})"
    denom = float(len(window))
    stable_frac = sum(1.0 for item in window if bool(item.get("stable_hold", False))) / denom
    grasp_frac = sum(1.0 for item in window if bool(item.get("true_grasp", False))) / denom
    lifted_frac = sum(1.0 for item in window if bool(item.get("lifted", False))) / denom
    min_stable_frac = max(0.0, min(float(min_stable_frac), 1.0))
    final = window[-1]
    final_ok = (
        bool(final.get("stable_hold", False))
        and bool(final.get("true_grasp", False))
        and bool(final.get("lifted", False))
    )
    ok = stable_frac >= min_stable_frac and grasp_frac >= min_stable_frac and lifted_frac >= min_stable_frac
    if require_final_stable:
        ok = ok and final_ok
    detail = (
        f"stable_frac={stable_frac:.3f} true_grasp_frac={grasp_frac:.3f} "
        f"lifted_frac={lifted_frac:.3f} final_ok={int(final_ok)}"
    )
    return ok, detail


def _save_success_videos(agent_cfg: dict, checkpoint: Path, output_dir: Path) -> tuple[list[str], list[str]]:
    if args_cli.save_success_videos <= 0:
        return [], []

    video_paths: list[str] = []
    rollout_video_paths: list[str] = []
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_envs = max(1, int(args_cli.video_envs))
    _trace(
        "video search: "
        f"attempts={args_cli.video_attempts} envs={video_envs} "
        f"save_success={args_cli.save_success_videos} "
        f"failure_rollouts={args_cli.save_rollout_videos_on_failure} "
        f"each_failed_attempt={bool(args_cli.save_rollout_videos_each_failed_attempt)}"
    )

    def write_video_search_progress(attempt: int) -> None:
        progress_path = output_dir / "video_search_progress.json"
        progress = {
            "attempts_completed": int(attempt + 1),
            "checkpoint": str(checkpoint),
            "success_videos": video_paths,
            "success_video_traces": [
                str(Path(video_path).with_suffix(".trace.json"))
                for video_path in video_paths
                if Path(video_path).with_suffix(".trace.json").exists()
            ],
            "rollout_videos": rollout_video_paths,
            "rollout_video_traces": [
                str(Path(video_path).with_suffix(".trace.json"))
                for video_path in rollout_video_paths
                if Path(video_path).with_suffix(".trace.json").exists()
            ],
        }
        progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    omni.usd.get_context().new_stage()
    _trace("creating rendered video env")
    wrapped_env = _make_env(args_cli.task, agent_cfg, video_envs, render_mode="rgb_array")
    _trace(
        "created rendered video env: "
        f"num_envs={wrapped_env.num_envs} max_episode_length={wrapped_env.unwrapped.max_episode_length}"
    )
    try:
        _trace("creating rl_games player")
        player = _make_player(agent_cfg, wrapped_env, checkpoint)
        _trace("created rl_games player")
        for attempt in range(args_cli.video_attempts):
            if len(video_paths) >= args_cli.save_success_videos:
                break
            _trace(f"video attempt {attempt:03d} reset")
            obs = wrapped_env.reset()
            _trace(f"video attempt {attempt:03d} reset done")
            _set_video_camera_poses(wrapped_env.env.unwrapped)
            player.reset()
            frames_by_env: list[list[np.ndarray]] = [[] for _ in range(video_envs)]
            traces_by_env: list[list[dict]] = [[] for _ in range(video_envs)]
            saved_env_ids: set[int] = set()
            pending_success_steps: dict[int, int] = {}
            max_steps = int(args_cli.video_max_steps or wrapped_env.unwrapped.max_episode_length)

            def save_video_for_env(env_id: int) -> None:
                frames = frames_by_env[env_id]
                if not _frames_have_visual_signal(frames):
                    _trace(
                        f"video attempt {attempt:03d} env {env_id:03d} "
                        "succeeded but produced blank frames"
                    )
                    saved_env_ids.add(env_id)
                    return
                video_path = video_dir / (
                    f"success_attempt_{attempt:03d}_env_{env_id:03d}_"
                    f"idx_{len(video_paths):03d}.mp4"
                )
                imageio.mimsave(video_path, frames, fps=args_cli.video_fps, macro_block_size=16)
                trace_path = _write_video_trace(video_path, traces_by_env[env_id], attempt, env_id, len(frames))
                video_paths.append(str(video_path))
                saved_env_ids.add(env_id)
                _trace(f"saved success video: {video_path} (env_id={env_id}, trace={trace_path})")

            with torch.no_grad():
                for step in range(max_steps):
                    if step % max(args_cli.video_stride, 1) == 0:
                        _capture_video_frames(wrapped_env, frames_by_env)
                    actions = player.get_action(obs, is_deterministic=args_cli.deterministic)
                    obs, _, dones, extras = wrapped_env.step(actions)
                    _append_video_trace(wrapped_env, traces_by_env, extras, step)
                    success_now = _tensor_bool(extras, "success_env", video_envs, wrapped_env.device)
                    if args_cli.video_debug_interval > 0 and step % args_cli.video_debug_interval == 0:
                        lifted_now = _tensor_bool(extras, "lifted_env", video_envs, wrapped_env.device)
                        stable_now = _tensor_bool(extras, "stable_hold_env", video_envs, wrapped_env.device)
                        _trace(
                            f"video attempt {attempt:03d} step={step:04d} "
                            f"success={int(success_now.sum().item())} "
                            f"lifted={int(lifted_now.sum().item())} "
                            f"stable={int(stable_now.sum().item())}"
                        )
                    if bool(success_now.any()):
                        _capture_video_frames(wrapped_env, frames_by_env)
                        for success_env_tensor in success_now.nonzero(as_tuple=False).squeeze(-1):
                            success_env_id = int(success_env_tensor.item())
                            if success_env_id in saved_env_ids:
                                continue
                            if args_cli.video_post_success_steps > 0:
                                pending_success_steps.setdefault(success_env_id, step)
                            else:
                                save_video_for_env(success_env_id)
                            if len(video_paths) >= args_cli.save_success_videos:
                                break
                        if len(video_paths) >= args_cli.save_success_videos:
                            break
                    if pending_success_steps:
                        ready_env_ids = [
                            env_id
                            for env_id, success_step in pending_success_steps.items()
                            if step - success_step >= args_cli.video_post_success_steps
                        ]
                        for env_id in ready_env_ids:
                            if env_id not in saved_env_ids and len(video_paths) < args_cli.save_success_videos:
                                if args_cli.video_require_post_success_stable:
                                    ok, detail = _post_success_trace_passes(
                                        traces_by_env[env_id],
                                        pending_success_steps[env_id],
                                        args_cli.video_post_success_min_stable_frac,
                                        args_cli.video_post_success_require_final_stable,
                                        args_cli.video_post_success_steps,
                                    )
                                    if not ok:
                                        _trace(
                                            f"rejected success video for env {env_id}: "
                                            f"post-success hold failed ({detail})"
                                        )
                                        pending_success_steps.pop(env_id, None)
                                        continue
                                save_video_for_env(env_id)
                            pending_success_steps.pop(env_id, None)
                        if len(video_paths) >= args_cli.save_success_videos:
                            break
                    if bool(dones.all()):
                        break
                _capture_video_frames(wrapped_env, frames_by_env)
                if pending_success_steps:
                    for env_id in list(pending_success_steps):
                        if env_id not in saved_env_ids and len(video_paths) < args_cli.save_success_videos:
                            if args_cli.video_require_post_success_stable:
                                ok, detail = _post_success_trace_passes(
                                    traces_by_env[env_id],
                                    pending_success_steps[env_id],
                                    args_cli.video_post_success_min_stable_frac,
                                    args_cli.video_post_success_require_final_stable,
                                    args_cli.video_post_success_steps,
                                )
                                if not ok:
                                    _trace(
                                        f"rejected success video for env {env_id}: "
                                        f"post-success hold failed ({detail})"
                                    )
                                    continue
                            save_video_for_env(env_id)

            is_last_attempt = attempt >= args_cli.video_attempts - 1
            should_save_failure_rollout = (
                args_cli.save_rollout_videos_on_failure > 0
                and (is_last_attempt or args_cli.save_rollout_videos_each_failed_attempt)
            )
            if not saved_env_ids:
                _trace(f"video attempt {attempt:03d} did not succeed")
                if should_save_failure_rollout:
                    debug_limit = min(int(args_cli.save_rollout_videos_on_failure), len(frames_by_env))
                    for debug_env_id in range(debug_limit):
                        frames = frames_by_env[debug_env_id]
                        if not _frames_have_visual_signal(frames):
                            continue
                        debug_path = video_dir / f"rollout_attempt_{attempt:03d}_env_{debug_env_id:03d}.mp4"
                        imageio.mimsave(debug_path, frames, fps=args_cli.video_fps, macro_block_size=16)
                        trace_path = _write_video_trace(
                            debug_path, traces_by_env[debug_env_id], attempt, debug_env_id, len(frames)
                        )
                        rollout_video_paths.append(str(debug_path))
                        _trace(
                            f"saved debug rollout video: {debug_path} "
                            f"(env_id={debug_env_id}, trace={trace_path})"
                        )
            write_video_search_progress(attempt)
    finally:
        wrapped_env.close()

    return video_paths, rollout_video_paths


def _write_trial_sequence_trace(
    video_path: Path,
    trials: list[dict],
    frame_count: int,
) -> Path:
    success_count = sum(1 for trial in trials if bool(trial.get("success", False)))
    raw_success_count = sum(1 for trial in trials if bool(trial.get("raw_success", False)))
    post_success_hold_count = sum(1 for trial in trials if bool(trial.get("post_success_hold_success", False)))
    payload = {
        "video_path": str(video_path),
        "checkpoint": str(Path(args_cli.checkpoint).expanduser().resolve()),
        "task": args_cli.task,
        "trials": trials,
        "trial_count": int(len(trials)),
        "success_count": int(success_count),
        "success_rate": float(success_count / max(len(trials), 1)),
        "raw_success_count": int(raw_success_count),
        "raw_success_rate": float(raw_success_count / max(len(trials), 1)),
        "post_success_hold_count": int(post_success_hold_count),
        "post_success_hold_rate": float(post_success_hold_count / max(len(trials), 1)),
        "frame_count": int(frame_count),
        "camera_track_object": bool(args_cli.video_camera_track_object),
        "camera_update_every_frame": bool(args_cli.video_camera_update_every_frame),
        "camera_eye": list(args_cli.video_camera_eye) if args_cli.video_camera_eye is not None else None,
        "camera_target": list(args_cli.video_camera_target) if args_cli.video_camera_target is not None else None,
        "camera_track_offset": list(args_cli.video_camera_track_offset),
        "camera_track_target_offset": list(args_cli.video_camera_track_target_offset),
        "camera_focal_length": args_cli.video_camera_focal_length,
        "camera_resolution": args_cli.video_camera_resolution,
    }
    trace_path = video_path.with_suffix(".trace.json")
    trace_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return trace_path


def _save_trial_sequence_videos(agent_cfg: dict, checkpoint: Path, output_dir: Path) -> tuple[list[str], list[str]]:
    if args_cli.save_trial_sequence_videos <= 0:
        return [], []

    sequence_paths: list[str] = []
    trace_paths: list[str] = []
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_envs = max(1, int(args_cli.video_envs))
    record_env_id = min(max(int(args_cli.trial_sequence_env_id), 0), video_envs - 1)
    trial_count = max(1, int(args_cli.trial_sequence_trials))
    pre_roll_frames = max(0, int(args_cli.trial_sequence_pre_roll_frames))
    gap_frames = max(0, int(args_cli.trial_sequence_gap_frames))
    stride = max(1, int(args_cli.video_stride))
    post_success_steps = max(0, int(args_cli.video_post_success_steps))

    _trace(
        "trial sequence video: "
        f"videos={args_cli.save_trial_sequence_videos} trials={trial_count} "
        f"envs={video_envs} record_env_id={record_env_id}"
    )
    omni.usd.get_context().new_stage()
    wrapped_env = _make_env(args_cli.task, agent_cfg, video_envs, render_mode="rgb_array")
    _trace(
        "created trial-sequence video env: "
        f"num_envs={wrapped_env.num_envs} max_episode_length={wrapped_env.unwrapped.max_episode_length}"
    )
    try:
        player = _make_player(agent_cfg, wrapped_env, checkpoint)
        for sequence_idx in range(int(args_cli.save_trial_sequence_videos)):
            tmp_path = video_dir / f"trial_sequence_{sequence_idx:03d}_pending.mp4"
            writer = imageio.get_writer(tmp_path, fps=args_cli.video_fps, macro_block_size=16)
            trials: list[dict] = []
            frame_count = 0
            last_frame: np.ndarray | None = None

            def append_frames(frames: list[np.ndarray]) -> None:
                nonlocal frame_count, last_frame
                for frame in frames:
                    writer.append_data(frame)
                    frame_count += 1
                    last_frame = frame

            try:
                for trial_idx in range(trial_count):
                    _trace(f"trial-sequence {sequence_idx:03d} trial {trial_idx + 1:02d}/{trial_count} reset")
                    obs = wrapped_env.reset()
                    _set_video_camera_poses(wrapped_env.env.unwrapped)
                    player.reset()
                    frames_by_env: list[list[np.ndarray]] = [[] for _ in range(video_envs)]
                    traces_by_env: list[list[dict]] = [[] for _ in range(video_envs)]
                    first_success_step: int | None = None
                    post_success_detail = ""
                    max_steps = int(args_cli.video_max_steps or wrapped_env.unwrapped.max_episode_length)

                    for _ in range(pre_roll_frames):
                        _capture_video_frames(wrapped_env, frames_by_env)

                    with torch.no_grad():
                        for step in range(max_steps):
                            if step % stride == 0:
                                _capture_video_frames(wrapped_env, frames_by_env)
                            actions = player.get_action(obs, is_deterministic=args_cli.deterministic)
                            obs, _, dones, extras = wrapped_env.step(actions)
                            _append_video_trace(wrapped_env, traces_by_env, extras, step)
                            success_now = _tensor_bool(extras, "success_env", video_envs, wrapped_env.device)
                            if bool(success_now[record_env_id].item()) and first_success_step is None:
                                first_success_step = int(step)
                            if first_success_step is not None and step - first_success_step >= post_success_steps:
                                break
                            if bool(dones[record_env_id].item()):
                                break
                    _capture_video_frames(wrapped_env, frames_by_env)

                    trace = traces_by_env[record_env_id]
                    raw_success = first_success_step is not None
                    success = bool(raw_success)
                    post_success_hold_success = bool(success)
                    if success and bool(args_cli.video_require_post_success_stable):
                        post_success_hold_success, post_success_detail = _post_success_trace_passes(
                            trace,
                            int(first_success_step),
                            args_cli.video_post_success_min_stable_frac,
                            bool(args_cli.video_post_success_require_final_stable),
                            post_success_steps,
                        )
                    frames = frames_by_env[record_env_id]
                    visual_ok = _frames_have_visual_signal(frames)
                    if visual_ok:
                        append_frames(frames)
                    elif last_frame is not None:
                        append_frames([last_frame.copy()])

                    if gap_frames > 0 and last_frame is not None and trial_idx < trial_count - 1:
                        gap_frame = np.zeros_like(last_frame)
                        append_frames([gap_frame.copy() for _ in range(gap_frames)])

                    trials.append(
                        {
                            "trial": int(trial_idx),
                            "success": bool(success),
                            "raw_success": bool(raw_success),
                            "post_success_hold_success": bool(post_success_hold_success),
                            "first_success_step": first_success_step,
                            "post_success_detail": post_success_detail,
                            "visual_signal": bool(visual_ok),
                            "frame_count": int(len(frames)),
                            "trace": trace,
                        }
                    )
                    _trace(
                        f"trial-sequence {sequence_idx:03d} trial {trial_idx + 1:02d}/{trial_count} "
                        f"success={int(success)} raw_success={int(raw_success)} "
                        f"post_hold={int(post_success_hold_success)} frames={len(frames)}"
                    )
            finally:
                writer.close()

            success_count = sum(1 for trial in trials if bool(trial.get("success", False)))
            final_path = video_dir / (
                f"trial_sequence_{sequence_idx:03d}_trials_{len(trials):03d}_"
                f"success_{success_count:03d}_sr_{success_count / max(len(trials), 1):.3f}.mp4"
            )
            if frame_count <= 0:
                tmp_path.unlink(missing_ok=True)
                _trace(f"trial-sequence {sequence_idx:03d} produced no frames")
                continue
            tmp_path.replace(final_path)
            trace_path = _write_trial_sequence_trace(final_path, trials, frame_count)
            sequence_paths.append(str(final_path))
            trace_paths.append(str(trace_path))
            _trace(f"saved trial sequence video: {final_path} (trace={trace_path})")
    finally:
        wrapped_env.close()

    return sequence_paths, trace_paths


def main() -> None:
    carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args_cli.output_dir).expanduser().resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = _maybe_init_wandb(checkpoint, output_dir)
    agent_cfg = _load_agent_cfg(args_cli.task)
    _trace(f"checkpoint={checkpoint}")
    try:
        if args_cli.skip_vector_eval:
            eval_summary = {
                "skipped": True,
                "episodes": 0,
                "success_count": None,
                "success_rate": None,
                "reason": "--skip-vector-eval",
            }
            passed = True
        else:
            eval_summary = _run_vector_eval(agent_cfg, checkpoint)
            passed = eval_summary["success_rate"] >= args_cli.success_threshold
        video_paths, rollout_video_paths = _save_success_videos(agent_cfg, checkpoint, output_dir) if passed else ([], [])
        trial_sequence_paths, trial_sequence_trace_paths = _save_trial_sequence_videos(
            agent_cfg, checkpoint, output_dir
        )
        video_trace_paths = [
            str(Path(video_path).with_suffix(".trace.json"))
            for video_path in video_paths
            if Path(video_path).with_suffix(".trace.json").exists()
        ]

        summary = {
            "task": args_cli.task,
            "checkpoint": str(checkpoint),
            "device": args_cli.device,
            "rl_device": args_cli.rl_device or args_cli.device,
            "deterministic": bool(args_cli.deterministic),
            "dynamic_curriculum_alpha": args_cli.dynamic_curriculum_alpha,
            "tabletop_asset_curriculum_alpha": args_cli.tabletop_asset_curriculum_alpha,
            "tabletop_motion_curriculum_alpha": args_cli.tabletop_motion_curriculum_alpha,
            "success_threshold": float(args_cli.success_threshold),
            "passed": bool(passed),
            "eval": eval_summary,
            "success_videos": video_paths,
            "success_video_traces": video_trace_paths,
            "rollout_videos": rollout_video_paths,
            "trial_sequence_videos": trial_sequence_paths,
            "trial_sequence_traces": trial_sequence_trace_paths,
            "output_dir": str(output_dir),
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _log_summary_to_wandb(wandb_run, summary, summary_path)
        _trace(json.dumps(summary, indent=2, sort_keys=True))
        if not passed:
            raise SystemExit(
                f"success_rate={eval_summary['success_rate']:.3f} below threshold={args_cli.success_threshold:.3f}"
            )
        fallback_ok = bool(args_cli.allow_rollout_video_fallback and rollout_video_paths)
        if args_cli.save_success_videos > 0 and len(video_paths) < args_cli.save_success_videos and not fallback_ok:
            raise SystemExit(
                "success threshold passed, but only "
                f"{len(video_paths)}/{args_cli.save_success_videos} successful videos were saved"
            )
        if (
            args_cli.save_trial_sequence_videos > 0
            and len(trial_sequence_paths) < args_cli.save_trial_sequence_videos
        ):
            raise SystemExit(
                "only "
                f"{len(trial_sequence_paths)}/{args_cli.save_trial_sequence_videos} trial-sequence videos were saved"
            )
    except BaseException as exc:
        error_payload = {
            "task": args_cli.task,
            "checkpoint": str(checkpoint),
            "output_dir": str(output_dir),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        error_path = output_dir / "error.json"
        error_path.write_text(json.dumps(error_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _trace(f"wrote eval error: {error_path}")
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
