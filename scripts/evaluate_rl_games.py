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

VIDEO_CAMERA_PRESETS = {
    "falling_reference_20260703": {
        "eye": (1.55, -1.75, 1.08),
        "target": (0.1, 0.2, 0.64),
        "focal_length": 18.0,
        "resolution": (960, 544),
        "track_object": False,
        "update_every_frame": False,
    },
}

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToolReal-Revo2-Franka-StaticBall-Grasp-Direct-v0", help="Gym task id.")
parser.add_argument("--checkpoint", required=True, help="rl_games .pth checkpoint.")
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=64, help="Eval env count.")
parser.add_argument("--episodes", type=int, default=128, help="Number of completed eval episodes.")
parser.add_argument("--seed", type=int, default=42, help="Deterministic environment/evaluation seed.")
parser.add_argument("--max-steps", type=int, default=None, help="Safety cap for vectorized eval steps.")
parser.add_argument("--success-threshold", type=float, default=0.30, help="Required success rate.")
parser.add_argument(
    "--include-per-env-stats",
    action="store_true",
    help="Include verbose per-environment episode metrics in summary.json.",
)
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
    "--dynamic-tabletop-speed-range",
    type=float,
    nargs=2,
    metavar=("MIN_MPS", "MAX_MPS"),
    default=None,
    help="Override the free-rolling tabletop initial speed range for low/high-speed evaluation.",
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
    "--tabletop-pregrasp-lead-time",
    type=float,
    default=None,
    help="Override the moving-object velocity lead time without changing the registered task.",
)
parser.add_argument(
    "--tabletop-pregrasp-ahead-distance",
    type=float,
    default=None,
    help="Override the fixed moving-object interception lead distance.",
)
parser.add_argument(
    "--scripted-action-prior-active-residual-scale",
    type=float,
    default=None,
    help="Override policy residual authority while a scripted action prior is active.",
)
parser.add_argument(
    "--scripted-relative-lift-target-scale",
    type=float,
    default=None,
    help="Scale the configured relative joint-space lift target delta.",
)
parser.add_argument(
    "--scripted-action-prior-lift-steps",
    type=int,
    default=None,
    help="Override the scripted lift duration in control steps.",
)
parser.add_argument(
    "--episode-length-s",
    type=float,
    default=None,
    help="Override the environment episode duration in seconds.",
)
parser.add_argument(
    "--dynamic-success-hold-steps",
    type=int,
    default=None,
    help="Override consecutive stable steps required before success latches.",
)
parser.add_argument(
    "--tabletop-post-success-hand-close-fraction",
    type=float,
    default=None,
    help="Override the semantic 6-DoF hand-close blend applied after success latches.",
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
    "--trial-sequence-cycle-envs",
    action="store_true",
    help="Cycle the recorded environment across trials to avoid a single-env reset-stream bias.",
)
parser.add_argument(
    "--trial-sequence-seed-mode",
    choices=("per_trial", "sequence"),
    default="per_trial",
    help=(
        "Randomization policy for a trial-sequence video. per_trial reseeds before every reset; "
        "sequence seeds once and lets consecutive resets advance the environment RNG stream."
    ),
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
    "--video-post-success-max-relative-drift",
    type=float,
    default=0.0,
    help=(
        "Optional object-to-palm drift tolerance in meters. When positive, continuous strict grasp "
        "plus bounded drift can satisfy stability despite contact-velocity flag jitter."
    ),
)
parser.add_argument(
    "--video-post-success-require-final-stable",
    action="store_true",
    help="Require the final post-success trace sample to still be stable.",
)
parser.add_argument(
    "--video-post-success-lift-mode",
    choices=("auto", "never", "always"),
    default="auto",
    help=(
        "Whether post-success filtering should require lifted=True. "
        "auto disables the lifted requirement for falling-baton tasks and keeps it for lift tasks."
    ),
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
    "--video-camera-preset",
    choices=("none", *VIDEO_CAMERA_PRESETS.keys()),
    default="none",
    help="Named fixed camera preset. Use falling_reference_20260703 to match the saved IsaacGym-style falling view.",
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
parser.add_argument(
    "--wandb-save-video-traces",
    action="store_true",
    help="Upload per-frame video trace JSON files to W&B. They are kept locally by default because long videos can produce large traces.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video_camera_preset != "none":
    preset = VIDEO_CAMERA_PRESETS[args_cli.video_camera_preset]
    args_cli.video_camera_eye = tuple(preset["eye"])
    args_cli.video_camera_target = tuple(preset["target"])
    args_cli.video_camera_focal_length = float(preset["focal_length"])
    args_cli.video_camera_resolution = tuple(preset["resolution"])
    args_cli.video_camera_track_object = bool(preset["track_object"])
    args_cli.video_camera_update_every_frame = bool(preset["update_every_frame"])
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
            "seed": args_cli.seed,
            "success_threshold": args_cli.success_threshold,
            "include_per_env_stats": bool(args_cli.include_per_env_stats),
            "deterministic": bool(args_cli.deterministic),
            "dynamic_curriculum_alpha": args_cli.dynamic_curriculum_alpha,
            "dynamic_tabletop_speed_range": args_cli.dynamic_tabletop_speed_range,
            "tabletop_asset_curriculum_alpha": args_cli.tabletop_asset_curriculum_alpha,
            "tabletop_motion_curriculum_alpha": args_cli.tabletop_motion_curriculum_alpha,
            "device": args_cli.device,
            "rl_device": args_cli.rl_device or args_cli.device,
            "save_success_videos": args_cli.save_success_videos,
            "save_trial_sequence_videos": args_cli.save_trial_sequence_videos,
            "trial_sequence_trials": args_cli.trial_sequence_trials,
            "video_post_success_steps": args_cli.video_post_success_steps,
            "video_post_success_min_stable_frac": args_cli.video_post_success_min_stable_frac,
            "video_post_success_max_relative_drift": args_cli.video_post_success_max_relative_drift,
            "video_post_success_require_final_stable": bool(
                args_cli.video_post_success_require_final_stable
            ),
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
    if bool(args_cli.wandb_save_video_traces):
        for trace_path in summary.get("success_video_traces", []):
            wandb.save(str(trace_path), base_path=str(summary_path.parent))
    for index, video_path in enumerate(summary.get("success_videos", [])):
        wandb.log({f"videos/success_{index:03d}": wandb.Video(video_path, fps=args_cli.video_fps, format="mp4")})
    for index, video_path in enumerate(summary.get("rollout_videos", [])):
        wandb.log({f"videos/rollout_{index:03d}": wandb.Video(video_path, fps=args_cli.video_fps, format="mp4")})
    if bool(args_cli.wandb_save_video_traces):
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
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = int(args_cli.seed)
    if args_cli.dynamic_curriculum_alpha is not None and hasattr(
        env_cfg, "dynamic_grasp_speed_curriculum_override_alpha"
    ):
        env_cfg.dynamic_grasp_speed_curriculum_override_alpha = float(args_cli.dynamic_curriculum_alpha)
    if args_cli.dynamic_tabletop_speed_range is not None and hasattr(
        env_cfg, "dynamic_tabletop_initial_speed_range"
    ):
        speed_min, speed_max = (float(value) for value in args_cli.dynamic_tabletop_speed_range)
        if speed_min < 0.0 or speed_max < speed_min:
            raise ValueError("--dynamic-tabletop-speed-range requires 0 <= MIN_MPS <= MAX_MPS")
        env_cfg.dynamic_tabletop_initial_speed_range = (speed_min, speed_max)
    if args_cli.tabletop_asset_curriculum_alpha is not None and hasattr(
        env_cfg, "tabletop_asset_curriculum_override_alpha"
    ):
        env_cfg.tabletop_asset_curriculum_override_alpha = float(args_cli.tabletop_asset_curriculum_alpha)
    if args_cli.tabletop_motion_curriculum_alpha is not None and hasattr(
        env_cfg, "tabletop_motion_mode_curriculum_override_alpha"
    ):
        env_cfg.tabletop_motion_mode_curriculum_override_alpha = float(args_cli.tabletop_motion_curriculum_alpha)
    if args_cli.tabletop_pregrasp_lead_time is not None and hasattr(
        env_cfg, "dynamic_tabletop_pregrasp_lead_time"
    ):
        env_cfg.dynamic_tabletop_pregrasp_lead_time = float(args_cli.tabletop_pregrasp_lead_time)
    if args_cli.tabletop_pregrasp_ahead_distance is not None and hasattr(
        env_cfg, "dynamic_tabletop_pregrasp_ahead_distance"
    ):
        env_cfg.dynamic_tabletop_pregrasp_ahead_distance = float(args_cli.tabletop_pregrasp_ahead_distance)
    if args_cli.scripted_action_prior_active_residual_scale is not None and hasattr(
        env_cfg, "scripted_action_prior_active_residual_scale"
    ):
        env_cfg.scripted_action_prior_active_residual_scale = float(
            args_cli.scripted_action_prior_active_residual_scale
        )
    if args_cli.scripted_relative_lift_target_scale is not None and hasattr(
        env_cfg, "scripted_tabletop_relative_lift_target_arm_delta"
    ):
        relative_scale = float(args_cli.scripted_relative_lift_target_scale)
        env_cfg.scripted_tabletop_relative_lift_target_arm_delta = tuple(
            relative_scale * float(value)
            for value in env_cfg.scripted_tabletop_relative_lift_target_arm_delta
        )
    if args_cli.scripted_action_prior_lift_steps is not None and hasattr(
        env_cfg, "scripted_action_prior_lift_steps"
    ):
        env_cfg.scripted_action_prior_lift_steps = int(args_cli.scripted_action_prior_lift_steps)
    if args_cli.episode_length_s is not None and hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = float(args_cli.episode_length_s)
    if args_cli.dynamic_success_hold_steps is not None and hasattr(
        env_cfg, "dynamic_success_hold_steps"
    ):
        env_cfg.dynamic_success_hold_steps = int(args_cli.dynamic_success_hold_steps)
    if args_cli.tabletop_post_success_hand_close_fraction is not None:
        env_cfg.tabletop_post_success_hand_close_fraction = float(
            args_cli.tabletop_post_success_hand_close_fraction
        )
    if render_mode == "rgb_array" and hasattr(env_cfg, "video_camera_enabled"):
        env_cfg.video_camera_enabled = True
        if hasattr(env_cfg, "video_camera"):
            env_step_dt = float(env_cfg.sim.dt) * max(int(getattr(env_cfg, "decimation", 1)), 1)
            env_cfg.video_camera.update_period = env_step_dt * max(int(args_cli.video_stride), 1)
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


def _tensor_float(extras: dict, key: str, num_envs: int, device: str, default: float = 0.0) -> torch.Tensor:
    value = extras.get(key)
    if value is None:
        if default is None:
            return None
        return torch.full((num_envs,), float(default), dtype=torch.float32, device=device)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32, device=device)
    return value[:num_envs].float()


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
        episode_max_height_delta = torch.full((num_envs,), -1.0e9, dtype=torch.float32, device=device)
        relative_lift_labels = getattr(
            wrapped_env.unwrapped, "_scripted_relative_lift_target_candidate_labels", None
        )
        lift_action_labels = getattr(wrapped_env.unwrapped, "_scripted_lift_action_candidate_labels", None)
        hand_action_labels = getattr(
            wrapped_env.unwrapped, "_scripted_tabletop_hand_grasp_memory_action_candidate_labels", None
        )
        candidate_labels = (
            relative_lift_labels
            if relative_lift_labels is not None
            else lift_action_labels
            if lift_action_labels is not None
            else hand_action_labels
        )
        candidate_stats: dict[str, dict[str, float]] = {}
        tabletop_asset_specs = tuple(getattr(wrapped_env.unwrapped, "_tabletop_asset_specs", ()))
        active_asset_ids = getattr(wrapped_env.unwrapped, "_tabletop_active_asset_ids", None)
        episode_asset_ids = active_asset_ids.detach().clone() if torch.is_tensor(active_asset_ids) else None
        asset_stats: dict[str, dict[str, float | str | int]] = {}
        completed = 0
        success_count = 0
        true_count = 0
        lifted_count = 0
        stable_count = 0
        total_reward = 0.0
        total_reward_count = 0
        last_episode_log = {}
        per_env_episodes = torch.zeros(num_envs, dtype=torch.long, device=device)
        per_env_success = torch.zeros(num_envs, dtype=torch.long, device=device)
        per_env_true_grasp = torch.zeros(num_envs, dtype=torch.long, device=device)
        per_env_lifted = torch.zeros(num_envs, dtype=torch.long, device=device)
        per_env_stable_hold = torch.zeros(num_envs, dtype=torch.long, device=device)
        per_env_max_height_sum = torch.zeros(num_envs, dtype=torch.float32, device=device)
        per_env_max_height_max = torch.full((num_envs,), -1.0e9, dtype=torch.float32, device=device)
        target_episodes_per_env = torch.full(
            (num_envs,),
            int(args_cli.episodes) // num_envs,
            dtype=torch.long,
            device=device,
        )
        target_episodes_per_env[: int(args_cli.episodes) % num_envs] += 1

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
            object_height_delta = getattr(wrapped_env.unwrapped, "_object_height_delta", None)
            if torch.is_tensor(object_height_delta):
                episode_max_height_delta = torch.maximum(
                    episode_max_height_delta,
                    object_height_delta[:num_envs].float(),
                )
            total_reward += float(rewards.sum().item())
            total_reward_count += int(rewards.numel())
            if "episode" in extras:
                last_episode_log = {
                    key: float(value.item()) for key, value in extras["episode"].items() if hasattr(value, "item")
                }

            all_done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if all_done_ids.numel() > 0:
                # Balance completed episodes across vector environments. Counting the first N
                # completion events over-represents short failed episodes when successful
                # post-hold episodes intentionally run for the full horizon.
                count_mask = (
                    per_env_episodes[all_done_ids]
                    < target_episodes_per_env[all_done_ids]
                )
                done_ids = all_done_ids[count_mask]
                completed += int(done_ids.numel())
                success_count += int(episode_success[done_ids].sum().item())
                true_count += int(episode_true_grasp[done_ids].sum().item())
                lifted_count += int(episode_lifted[done_ids].sum().item())
                stable_count += int(episode_stable_hold[done_ids].sum().item())
                per_env_episodes[done_ids] += 1
                per_env_success[done_ids] += episode_success[done_ids].long()
                per_env_true_grasp[done_ids] += episode_true_grasp[done_ids].long()
                per_env_lifted[done_ids] += episode_lifted[done_ids].long()
                per_env_stable_hold[done_ids] += episode_stable_hold[done_ids].long()
                per_env_max_height_sum[done_ids] += episode_max_height_delta[done_ids]
                per_env_max_height_max[done_ids] = torch.maximum(
                    per_env_max_height_max[done_ids],
                    episode_max_height_delta[done_ids],
                )
                if candidate_labels is not None:
                    for done_env_id in done_ids.detach().cpu().tolist():
                        if done_env_id >= len(candidate_labels):
                            continue
                        label = str(candidate_labels[done_env_id])
                        stats = candidate_stats.setdefault(
                            label,
                            {
                                "episodes": 0.0,
                                "success_count": 0.0,
                                "true_grasp_count": 0.0,
                                "lifted_count": 0.0,
                                "stable_hold_count": 0.0,
                                "max_height_delta_sum": 0.0,
                                "max_height_delta_max": -1.0e9,
                            },
                        )
                        max_height = float(episode_max_height_delta[done_env_id].item())
                        stats["episodes"] += 1.0
                        stats["success_count"] += float(episode_success[done_env_id].item())
                        stats["true_grasp_count"] += float(episode_true_grasp[done_env_id].item())
                        stats["lifted_count"] += float(episode_lifted[done_env_id].item())
                        stats["stable_hold_count"] += float(episode_stable_hold[done_env_id].item())
                        stats["max_height_delta_sum"] += max_height
                        stats["max_height_delta_max"] = max(stats["max_height_delta_max"], max_height)
                if episode_asset_ids is not None:
                    for done_env_id in done_ids.detach().cpu().tolist():
                        asset_index = int(episode_asset_ids[done_env_id].item())
                        if 0 <= asset_index < len(tabletop_asset_specs):
                            spec = tabletop_asset_specs[asset_index]
                            asset_id = str(spec.get("asset_id", f"asset_{asset_index}"))
                            label = str(spec.get("label", asset_id.rsplit("/", 1)[-1]))
                            category = str(spec.get("category", "unknown"))
                        else:
                            asset_id = f"asset_{asset_index}"
                            label = asset_id
                            category = "unknown"
                        stats = asset_stats.setdefault(
                            asset_id,
                            {
                                "asset_index": asset_index,
                                "label": label,
                                "category": category,
                                "episodes": 0.0,
                                "success_count": 0.0,
                                "true_grasp_count": 0.0,
                                "lifted_count": 0.0,
                                "stable_hold_count": 0.0,
                                "max_height_delta_sum": 0.0,
                                "max_height_delta_max": -1.0e9,
                            },
                        )
                        max_height = float(episode_max_height_delta[done_env_id].item())
                        stats["episodes"] += 1.0
                        stats["success_count"] += float(episode_success[done_env_id].item())
                        stats["true_grasp_count"] += float(episode_true_grasp[done_env_id].item())
                        stats["lifted_count"] += float(episode_lifted[done_env_id].item())
                        stats["stable_hold_count"] += float(episode_stable_hold[done_env_id].item())
                        stats["max_height_delta_sum"] += max_height
                        stats["max_height_delta_max"] = max(float(stats["max_height_delta_max"]), max_height)
                    current_asset_ids = getattr(wrapped_env.unwrapped, "_tabletop_active_asset_ids", None)
                    if torch.is_tensor(current_asset_ids):
                        episode_asset_ids[all_done_ids] = current_asset_ids[all_done_ids]
                episode_success[all_done_ids] = False
                episode_true_grasp[all_done_ids] = False
                episode_lifted[all_done_ids] = False
                episode_stable_hold[all_done_ids] = False
                episode_max_height_delta[all_done_ids] = -1.0e9
                _trace(
                    f"episodes={completed}/{args_cli.episodes} "
                    f"success_rate={success_count / max(completed, 1):.3f}"
                )
                if completed >= args_cli.episodes:
                    break
        else:
            _trace(f"hit max_steps={max_steps} before completing requested episodes")

        completed = max(completed, 1)
        candidate_summary = {}
        for label, stats in candidate_stats.items():
            episodes = max(float(stats["episodes"]), 1.0)
            candidate_summary[label] = {
                "episodes": int(stats["episodes"]),
                "success_count": int(stats["success_count"]),
                "success_rate": float(stats["success_count"] / episodes),
                "true_grasp_episode_rate": float(stats["true_grasp_count"] / episodes),
                "lifted_episode_rate": float(stats["lifted_count"] / episodes),
                "stable_hold_episode_rate": float(stats["stable_hold_count"] / episodes),
                "mean_max_height_delta": float(stats["max_height_delta_sum"] / episodes),
                "max_height_delta": float(stats["max_height_delta_max"]),
            }
        asset_summary = {}
        for asset_id, stats in asset_stats.items():
            episodes = max(float(stats["episodes"]), 1.0)
            asset_summary[asset_id] = {
                "asset_index": int(stats["asset_index"]),
                "label": str(stats["label"]),
                "category": str(stats["category"]),
                "episodes": int(stats["episodes"]),
                "success_count": int(stats["success_count"]),
                "success_rate": float(stats["success_count"] / episodes),
                "true_grasp_episode_rate": float(stats["true_grasp_count"] / episodes),
                "lifted_episode_rate": float(stats["lifted_count"] / episodes),
                "stable_hold_episode_rate": float(stats["stable_hold_count"] / episodes),
                "mean_max_height_delta": float(stats["max_height_delta_sum"] / episodes),
                "max_height_delta": float(stats["max_height_delta_max"]),
            }
        per_env_summary = []
        if bool(args_cli.include_per_env_stats):
            for env_id in range(num_envs):
                episodes = int(per_env_episodes[env_id].item())
                denom = max(float(episodes), 1.0)
                per_env_summary.append(
                    {
                        "env_id": int(env_id),
                        "episodes": episodes,
                        "success_count": int(per_env_success[env_id].item()),
                        "success_rate": float(per_env_success[env_id].item() / denom),
                        "true_grasp_episode_rate": float(per_env_true_grasp[env_id].item() / denom),
                        "lifted_episode_rate": float(per_env_lifted[env_id].item() / denom),
                        "stable_hold_episode_rate": float(per_env_stable_hold[env_id].item() / denom),
                        "mean_max_height_delta": float(per_env_max_height_sum[env_id].item() / denom),
                        "max_height_delta": (
                            float(per_env_max_height_max[env_id].item()) if episodes > 0 else None
                        ),
                    }
                )
        summary = {
            "episodes": int(completed),
            "episode_sampling": "balanced_per_env",
            "target_episodes_per_env_min": int(target_episodes_per_env.min().item()),
            "target_episodes_per_env_max": int(target_episodes_per_env.max().item()),
            "success_count": int(success_count),
            "success_rate": float(success_count / completed),
            "true_grasp_episode_rate": float(true_count / completed),
            "lifted_episode_rate": float(lifted_count / completed),
            "stable_hold_episode_rate": float(stable_count / completed),
            "mean_step_reward": float(total_reward / max(total_reward_count, 1)),
            "max_episode_length": max_episode_length,
            "last_log": last_episode_log,
            "candidate_stats": candidate_summary,
            "asset_stats": asset_summary,
        }
        if per_env_summary:
            summary["per_env_stats"] = per_env_summary
        return summary
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
    arm_joint_names = list(getattr(unwrapped_env.cfg, "arm_joint_names", []))
    arm_joint_ids = list(getattr(unwrapped_env, "_arm_joint_ids", []))
    joint_targets = getattr(unwrapped_env, "_joint_targets", None)
    if torch.is_tensor(joint_targets) and hand_joint_ids:
        hand_joint_targets_cpu = joint_targets[:num_envs, hand_joint_ids].detach().cpu()
    else:
        hand_joint_targets_cpu = None
    if torch.is_tensor(joint_targets) and arm_joint_ids:
        arm_joint_targets_cpu = joint_targets[:num_envs, arm_joint_ids].detach().cpu()
    else:
        arm_joint_targets_cpu = None
    robot_data = getattr(getattr(unwrapped_env, "robot", None), "data", None)
    joint_pos = getattr(robot_data, "joint_pos", None)
    if torch.is_tensor(joint_pos) and hand_joint_ids:
        hand_joint_pos_cpu = joint_pos[:num_envs, hand_joint_ids].detach().cpu()
    else:
        hand_joint_pos_cpu = None
    if torch.is_tensor(joint_pos) and arm_joint_ids:
        arm_joint_pos_cpu = joint_pos[:num_envs, arm_joint_ids].detach().cpu()
    else:
        arm_joint_pos_cpu = None
    action_prior = getattr(unwrapped_env, "_action_prior", None)
    if torch.is_tensor(action_prior):
        action_prior_cpu = action_prior[:num_envs].detach().cpu()
    else:
        action_prior_cpu = None
    scripted_lift_action = getattr(unwrapped_env, "_scripted_lift_action", None)
    if torch.is_tensor(scripted_lift_action):
        if scripted_lift_action.shape[0] == num_envs:
            scripted_lift_action_cpu = scripted_lift_action[:num_envs].detach().cpu()
        else:
            scripted_lift_action_cpu = scripted_lift_action.expand(num_envs, -1).detach().cpu()
    else:
        scripted_lift_action_cpu = None
    scripted_lift_labels = getattr(unwrapped_env, "_scripted_lift_action_candidate_labels", None)
    relative_lift_delta = getattr(unwrapped_env, "_scripted_relative_lift_target_delta", None)
    if torch.is_tensor(relative_lift_delta):
        relative_lift_delta_cpu = relative_lift_delta[:num_envs].detach().cpu()
    else:
        relative_lift_delta_cpu = None
    relative_lift_labels = getattr(unwrapped_env, "_scripted_relative_lift_target_candidate_labels", None)
    relative_lift_latched = getattr(unwrapped_env, "_scripted_relative_lift_target_latched", None)
    if torch.is_tensor(relative_lift_latched):
        relative_lift_latched_cpu = relative_lift_latched[:num_envs].detach().cpu()
    else:
        relative_lift_latched_cpu = None
    object_height_delta = _trace_tensor(unwrapped_env, "_object_height_delta", num_envs)
    object_palm_rel_vel = _trace_tensor(unwrapped_env, "_object_palm_rel_vel", num_envs)
    success_streak = _trace_tensor(unwrapped_env, "_success_streak", num_envs)
    object_pos_w = _trace_vec_tensor(unwrapped_env, "_object_pos_w", num_envs, 3)
    palm_pos_w = _trace_vec_tensor(unwrapped_env, "_palm_pos_w", num_envs, 3)
    object_lin_vel_w = _trace_vec_tensor(unwrapped_env, "_object_lin_vel_w", num_envs, 3)
    object_ang_vel_w = _trace_vec_tensor(unwrapped_env, "_object_ang_vel_w", num_envs, 3)
    hover_target_pos_w = _trace_vec_tensor(unwrapped_env, "_object_hover_target_pos_w", num_envs, 3)
    env_origins = getattr(getattr(unwrapped_env, "scene", None), "env_origins", None)
    if torch.is_tensor(env_origins):
        env_origins = env_origins[:num_envs]
    else:
        env_origins = torch.zeros_like(object_pos_w)
    surface_dist = getattr(unwrapped_env, "_surface_dist", None)
    if torch.is_tensor(surface_dist):
        surface_dist_cpu = surface_dist[:num_envs].detach().cpu()
    else:
        surface_dist_cpu = None
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
    dropped = _tensor_bool(extras, "dropped_env", num_envs, device)
    out_xy = _tensor_bool(extras, "out_xy_env", num_envs, device)
    time_out = _tensor_bool(extras, "time_out_env", num_envs, device)
    clearance_violation = _tensor_bool(extras, "tabletop_arm_clearance_violation_env", num_envs, device)
    thumb_contact = _tensor_bool(extras, "thumb_contact_env", num_envs, device)
    strict_thumb_contact = _tensor_bool(extras, "strict_thumb_contact_env", num_envs, device)
    strict_true_grasp = _tensor_bool(extras, "strict_true_grasp_env", num_envs, device)
    strict_opposing_contact = _tensor_bool(extras, "strict_opposing_contact_env", num_envs, device)
    finger_contacts = _tensor_float(extras, "finger_contacts_env", num_envs, device)
    non_thumb_contacts = _tensor_float(extras, "non_thumb_contacts_env", num_envs, device)
    strict_finger_contacts = _tensor_float(extras, "strict_finger_contacts_env", num_envs, device)
    strict_non_thumb_contacts = _tensor_float(extras, "strict_non_thumb_contacts_env", num_envs, device)
    min_surface_dist = _tensor_float(extras, "min_surface_dist_env", num_envs, device)
    tabletop_clearance_ok = _tensor_bool(extras, "tabletop_arm_clearance_ok_env", num_envs, device)
    tabletop_clearance_penalty = _tensor_float(
        extras,
        "tabletop_arm_clearance_penalty_env",
        num_envs,
        device,
        default=None,
    )
    if tabletop_clearance_penalty is None:
        tabletop_clearance_penalty = _trace_tensor(unwrapped_env, "_tabletop_arm_clearance_penalty", num_envs)
    tabletop_clearance_min_margin = _tensor_float(
        extras,
        "tabletop_arm_clearance_min_margin_env",
        num_envs,
        device,
        default=None,
    )
    if tabletop_clearance_min_margin is None:
        tabletop_clearance_min_margin = _trace_tensor(unwrapped_env, "_tabletop_arm_clearance_min_margin", num_envs)
    tabletop_clearance_active_fraction = _tensor_float(
        extras,
        "tabletop_arm_clearance_active_fraction_env",
        num_envs,
        device,
        default=None,
    )
    if tabletop_clearance_active_fraction is None:
        tabletop_clearance_active_fraction = _trace_tensor(
            unwrapped_env, "_tabletop_arm_clearance_active_fraction", num_envs
        )
    tabletop_underwrap_thumb_score = _tensor_float(
        extras,
        "tabletop_underwrap_thumb_score_env",
        num_envs,
        device,
        default=0.0,
    )
    tabletop_underwrap_non_thumb_score = _tensor_float(
        extras,
        "tabletop_underwrap_non_thumb_score_env",
        num_envs,
        device,
        default=0.0,
    )
    tabletop_underwrap_pair_score = _tensor_float(
        extras,
        "tabletop_underwrap_pair_score_env",
        num_envs,
        device,
        default=0.0,
    )
    tabletop_underwrap_progress_score = _tensor_float(
        extras,
        "tabletop_underwrap_progress_score_env",
        num_envs,
        device,
        default=0.0,
    )
    tabletop_underwrap_mean_score = _tensor_float(
        extras,
        "tabletop_underwrap_mean_score_env",
        num_envs,
        device,
        default=0.0,
    )
    tabletop_underwrap_min_tip_z_rel = _tensor_float(
        extras,
        "tabletop_underwrap_min_tip_z_rel_env",
        num_envs,
        device,
        default=0.0,
    )
    tabletop_underwrap_thumb_z_rel = _tensor_float(
        extras,
        "tabletop_underwrap_thumb_z_rel_env",
        num_envs,
        device,
        default=0.0,
    )
    tabletop_underwrap_min_non_thumb_z_rel = _tensor_float(
        extras,
        "tabletop_underwrap_min_non_thumb_z_rel_env",
        num_envs,
        device,
        default=0.0,
    )

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
    palm_pos_w_cpu = palm_pos_w.detach().cpu()
    env_origins_cpu = env_origins.detach().cpu()
    true_grasp_cpu = true_grasp.detach().cpu()
    lifted_cpu = lifted.detach().cpu()
    stable_hold_cpu = stable_hold.detach().cpu()
    success_cpu = success.detach().cpu()
    success_seen_cpu = success_seen.detach().cpu()
    dropped_cpu = dropped.detach().cpu()
    out_xy_cpu = out_xy.detach().cpu()
    time_out_cpu = time_out.detach().cpu()
    clearance_violation_cpu = clearance_violation.detach().cpu()
    thumb_contact_cpu = thumb_contact.detach().cpu()
    strict_thumb_contact_cpu = strict_thumb_contact.detach().cpu()
    strict_true_grasp_cpu = strict_true_grasp.detach().cpu()
    strict_opposing_contact_cpu = strict_opposing_contact.detach().cpu()
    finger_contacts_cpu = finger_contacts.detach().cpu()
    non_thumb_contacts_cpu = non_thumb_contacts.detach().cpu()
    strict_finger_contacts_cpu = strict_finger_contacts.detach().cpu()
    strict_non_thumb_contacts_cpu = strict_non_thumb_contacts.detach().cpu()
    min_surface_dist_cpu = min_surface_dist.detach().cpu()
    tabletop_clearance_ok_cpu = tabletop_clearance_ok.detach().cpu()
    tabletop_clearance_penalty_cpu = tabletop_clearance_penalty.detach().cpu()
    tabletop_clearance_min_margin_cpu = tabletop_clearance_min_margin.detach().cpu()
    tabletop_clearance_active_fraction_cpu = tabletop_clearance_active_fraction.detach().cpu()
    tabletop_underwrap_thumb_score_cpu = tabletop_underwrap_thumb_score.detach().cpu()
    tabletop_underwrap_non_thumb_score_cpu = tabletop_underwrap_non_thumb_score.detach().cpu()
    tabletop_underwrap_pair_score_cpu = tabletop_underwrap_pair_score.detach().cpu()
    tabletop_underwrap_progress_score_cpu = tabletop_underwrap_progress_score.detach().cpu()
    tabletop_underwrap_mean_score_cpu = tabletop_underwrap_mean_score.detach().cpu()
    tabletop_underwrap_min_tip_z_rel_cpu = tabletop_underwrap_min_tip_z_rel.detach().cpu()
    tabletop_underwrap_thumb_z_rel_cpu = tabletop_underwrap_thumb_z_rel.detach().cpu()
    tabletop_underwrap_min_non_thumb_z_rel_cpu = tabletop_underwrap_min_non_thumb_z_rel.detach().cpu()
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
            "object_pos_w": [float(value) for value in object_pos_w_cpu[env_id].tolist()],
            "palm_pos_w": [float(value) for value in palm_pos_w_cpu[env_id].tolist()],
            "object_pos_local": [
                float(value) for value in (object_pos_w_cpu[env_id] - env_origins_cpu[env_id]).tolist()
            ],
            "palm_pos_local": [
                float(value) for value in (palm_pos_w_cpu[env_id] - env_origins_cpu[env_id]).tolist()
            ],
            "object_minus_palm": [
                float(value) for value in (object_pos_w_cpu[env_id] - palm_pos_w_cpu[env_id]).tolist()
            ],
            "true_grasp": bool(true_grasp_cpu[env_id].item()),
            "lifted": bool(lifted_cpu[env_id].item()),
            "stable_hold": bool(stable_hold_cpu[env_id].item()),
            "thumb_contact": bool(thumb_contact_cpu[env_id].item()),
            "finger_contacts": float(finger_contacts_cpu[env_id].item()),
            "non_thumb_contacts": float(non_thumb_contacts_cpu[env_id].item()),
            "strict_true_grasp": bool(strict_true_grasp_cpu[env_id].item()),
            "strict_thumb_contact": bool(strict_thumb_contact_cpu[env_id].item()),
            "strict_finger_contacts": float(strict_finger_contacts_cpu[env_id].item()),
            "strict_non_thumb_contacts": float(strict_non_thumb_contacts_cpu[env_id].item()),
            "strict_opposing_contact": bool(strict_opposing_contact_cpu[env_id].item()),
            "min_surface_dist": float(min_surface_dist_cpu[env_id].item()),
            "tabletop_arm_clearance_ok": bool(tabletop_clearance_ok_cpu[env_id].item()),
            "tabletop_arm_clearance_penalty": float(tabletop_clearance_penalty_cpu[env_id].item()),
            "tabletop_arm_clearance_min_margin": float(tabletop_clearance_min_margin_cpu[env_id].item()),
            "tabletop_arm_clearance_active_fraction": float(
                tabletop_clearance_active_fraction_cpu[env_id].item()
            ),
            "tabletop_underwrap_thumb_score": float(tabletop_underwrap_thumb_score_cpu[env_id].item()),
            "tabletop_underwrap_non_thumb_score": float(
                tabletop_underwrap_non_thumb_score_cpu[env_id].item()
            ),
            "tabletop_underwrap_pair_score": float(tabletop_underwrap_pair_score_cpu[env_id].item()),
            "tabletop_underwrap_progress_score": float(
                tabletop_underwrap_progress_score_cpu[env_id].item()
            ),
            "tabletop_underwrap_mean_score": float(tabletop_underwrap_mean_score_cpu[env_id].item()),
            "tabletop_underwrap_min_tip_z_rel": float(tabletop_underwrap_min_tip_z_rel_cpu[env_id].item()),
            "tabletop_underwrap_thumb_z_rel": float(tabletop_underwrap_thumb_z_rel_cpu[env_id].item()),
            "tabletop_underwrap_min_non_thumb_z_rel": float(
                tabletop_underwrap_min_non_thumb_z_rel_cpu[env_id].item()
            ),
            "success_now": bool(stable_hold_cpu[env_id].item()),
            "success": bool(success_cpu[env_id].item()),
            "success_seen": bool(success_seen_cpu[env_id].item()),
            "dropped": bool(dropped_cpu[env_id].item()),
            "out_xy": bool(out_xy_cpu[env_id].item()),
            "time_out": bool(time_out_cpu[env_id].item()),
            "tabletop_arm_clearance_violation": bool(clearance_violation_cpu[env_id].item()),
        }
        if actions_cpu is not None and actions_cpu.ndim == 2:
            action = actions_cpu[env_id]
            trace_item["action"] = [float(value) for value in action.tolist()]
            trace_item["hand_action"] = [float(value) for value in action[hand_action_start:].tolist()]
        if action_prior_cpu is not None and action_prior_cpu.ndim == 2:
            prior = action_prior_cpu[env_id]
            trace_item["action_prior"] = [float(value) for value in prior.tolist()]
            trace_item["hand_action_prior"] = [float(value) for value in prior[hand_action_start:].tolist()]
        if scripted_lift_action_cpu is not None and scripted_lift_action_cpu.ndim == 2:
            lift_action = scripted_lift_action_cpu[env_id]
            trace_item["scripted_lift_action"] = [float(value) for value in lift_action.tolist()]
        if scripted_lift_labels is not None and env_id < len(scripted_lift_labels):
            trace_item["scripted_lift_action_label"] = str(scripted_lift_labels[env_id])
        if relative_lift_delta_cpu is not None and relative_lift_delta_cpu.ndim == 2:
            lift_delta = relative_lift_delta_cpu[env_id]
            trace_item["relative_lift_target_delta"] = [float(value) for value in lift_delta.tolist()]
        if relative_lift_labels is not None and env_id < len(relative_lift_labels):
            trace_item["relative_lift_target_label"] = str(relative_lift_labels[env_id])
        if relative_lift_latched_cpu is not None:
            trace_item["relative_lift_target_latched"] = bool(relative_lift_latched_cpu[env_id].item())
        if arm_joint_names:
            trace_item["arm_joint_names"] = arm_joint_names
        if arm_joint_targets_cpu is not None:
            trace_item["arm_joint_target"] = [
                float(value) for value in arm_joint_targets_cpu[env_id].tolist()
            ]
        if arm_joint_pos_cpu is not None:
            trace_item["arm_joint_pos"] = [float(value) for value in arm_joint_pos_cpu[env_id].tolist()]
        if hand_joint_names:
            trace_item["hand_joint_names"] = hand_joint_names
        if hand_joint_targets_cpu is not None:
            trace_item["hand_joint_target"] = [
                float(value) for value in hand_joint_targets_cpu[env_id].tolist()
            ]
        if hand_joint_pos_cpu is not None:
            trace_item["hand_joint_pos"] = [float(value) for value in hand_joint_pos_cpu[env_id].tolist()]
        if surface_dist_cpu is not None:
            trace_item["surface_distances"] = [
                float(value) for value in surface_dist_cpu[env_id].tolist()
            ]
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
    underwrap_progress_values = [
        float(item["tabletop_underwrap_progress_score"])
        for item in trace
        if "tabletop_underwrap_progress_score" in item
    ]
    underwrap_pair_values = [
        float(item["tabletop_underwrap_pair_score"])
        for item in trace
        if "tabletop_underwrap_pair_score" in item
    ]
    min_tip_z_rel_values = [
        float(item["tabletop_underwrap_min_tip_z_rel"])
        for item in trace
        if "tabletop_underwrap_min_tip_z_rel" in item
    ]
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
        "max_tabletop_underwrap_progress_score": (
            max(underwrap_progress_values) if underwrap_progress_values else None
        ),
        "max_tabletop_underwrap_pair_score": max(underwrap_pair_values) if underwrap_pair_values else None,
        "min_tabletop_underwrap_tip_z_rel": min(min_tip_z_rel_values) if min_tip_z_rel_values else None,
        "final_tabletop_underwrap_tip_z_rel": min_tip_z_rel_values[-1] if min_tip_z_rel_values else None,
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
    require_lifted: bool = True,
    max_relative_drift: float = 0.0,
) -> tuple[bool, str]:
    window = [item for item in trace if int(item.get("step", -1)) >= int(success_step)]
    if not window:
        return False, "empty post-success trace"
    observed_steps = int(window[-1].get("step", success_step)) - int(success_step)
    if min_window_steps > 0 and observed_steps < int(min_window_steps):
        return False, f"short post-success trace (steps={observed_steps}/{int(min_window_steps)})"
    denom = float(len(window))
    stable_frac = sum(1.0 for item in window if bool(item.get("stable_hold", False))) / denom
    grasp_key = "strict_true_grasp" if any("strict_true_grasp" in item for item in window) else "true_grasp"
    grasp_frac = sum(1.0 for item in window if bool(item.get(grasp_key, False))) / denom
    lifted_frac = sum(1.0 for item in window if bool(item.get("lifted", False))) / denom
    min_stable_frac = max(0.0, min(float(min_stable_frac), 1.0))
    relative_drift = math.inf
    relative_positions = [item.get("object_minus_palm") for item in window]
    if relative_positions and all(isinstance(position, list) and len(position) >= 3 for position in relative_positions):
        origin = relative_positions[0]
        relative_drift = max(
            math.sqrt(sum((float(position[axis]) - float(origin[axis])) ** 2 for axis in range(3)))
            for position in relative_positions
        )
    drift_limit = max(float(max_relative_drift), 0.0)
    kinematic_stable = (
        drift_limit > 0.0
        and relative_drift <= drift_limit
        and grasp_frac >= min_stable_frac
    )
    final = window[-1]
    final_ok = (
        (bool(final.get("stable_hold", False)) or kinematic_stable)
        and bool(final.get(grasp_key, False))
        and (bool(final.get("lifted", False)) if require_lifted else True)
    )
    ok = (stable_frac >= min_stable_frac or kinematic_stable) and grasp_frac >= min_stable_frac
    if require_lifted:
        ok = ok and lifted_frac >= min_stable_frac
    if require_final_stable:
        ok = ok and final_ok
    detail = (
        f"stable_frac={stable_frac:.3f} {grasp_key}_frac={grasp_frac:.3f} "
        f"lifted_frac={lifted_frac:.3f} relative_drift={relative_drift:.4f} "
        f"drift_limit={drift_limit:.4f} kinematic_stable={int(kinematic_stable)} "
        f"lift_required={int(require_lifted)} final_ok={int(final_ok)}"
    )
    return ok, detail


def _post_success_requires_lifted(task: str, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    compact_task = task.lower().replace("-", "").replace("_", "")
    return "fallingbaton" not in compact_task


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
                                        _post_success_requires_lifted(
                                            args_cli.task, args_cli.video_post_success_lift_mode
                                        ),
                                        args_cli.video_post_success_max_relative_drift,
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
                                    _post_success_requires_lifted(args_cli.task, args_cli.video_post_success_lift_mode),
                                    args_cli.video_post_success_max_relative_drift,
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
        "trial_sequence_cycle_envs": bool(args_cli.trial_sequence_cycle_envs),
        "trial_sequence_seed_mode": str(args_cli.trial_sequence_seed_mode),
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
            # Camera and renderer construction may consume global random state.
            # Reseed only after the complete video environment exists so a
            # fixed trial-sequence seed denotes the same reset distribution
            # regardless of rendering setup.
            sequence_seed = int(args_cli.seed) + sequence_idx * 100_003
            np.random.seed(sequence_seed)
            torch.manual_seed(sequence_seed)
            if hasattr(wrapped_env.unwrapped, "seed"):
                wrapped_env.unwrapped.seed(sequence_seed)
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
                    trial_env_id = (
                        (record_env_id + trial_idx) % video_envs
                        if bool(args_cli.trial_sequence_cycle_envs)
                        else record_env_id
                    )
                    trial_seed: int | None = None
                    if args_cli.trial_sequence_seed_mode == "per_trial":
                        trial_seed = sequence_seed + trial_idx
                        if hasattr(wrapped_env.unwrapped, "seed"):
                            wrapped_env.unwrapped.seed(trial_seed)
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
                            if bool(success_now[trial_env_id].item()) and first_success_step is None:
                                first_success_step = int(step)
                            if first_success_step is not None and step - first_success_step >= post_success_steps:
                                break
                            if bool(dones[trial_env_id].item()):
                                break
                    _capture_video_frames(wrapped_env, frames_by_env)

                    trace = traces_by_env[trial_env_id]
                    raw_success = first_success_step is not None
                    post_success_hold_success = bool(raw_success)
                    if raw_success and bool(args_cli.video_require_post_success_stable):
                        post_success_hold_success, post_success_detail = _post_success_trace_passes(
                            trace,
                            int(first_success_step),
                            args_cli.video_post_success_min_stable_frac,
                            bool(args_cli.video_post_success_require_final_stable),
                            post_success_steps,
                            _post_success_requires_lifted(args_cli.task, args_cli.video_post_success_lift_mode),
                            args_cli.video_post_success_max_relative_drift,
                        )
                    success = bool(raw_success) and (
                        bool(post_success_hold_success)
                        if bool(args_cli.video_require_post_success_stable)
                        else True
                    )
                    frames = frames_by_env[trial_env_id]
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
                            "sequence_seed": int(sequence_seed),
                            "seed_mode": str(args_cli.trial_sequence_seed_mode),
                            "seed": int(trial_seed) if trial_seed is not None else None,
                            "env_id": int(trial_env_id),
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
    np.random.seed(int(args_cli.seed))
    torch.manual_seed(int(args_cli.seed))
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args_cli.output_dir).expanduser().resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = _maybe_init_wandb(checkpoint, output_dir)
    agent_cfg = _load_agent_cfg(args_cli.task)
    agent_cfg["params"]["seed"] = int(args_cli.seed)
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
            vector_passed = True
        else:
            eval_summary = _run_vector_eval(agent_cfg, checkpoint)
            vector_passed = eval_summary["success_rate"] >= args_cli.success_threshold
        video_paths, rollout_video_paths = (
            _save_success_videos(agent_cfg, checkpoint, output_dir) if vector_passed else ([], [])
        )
        trial_sequence_paths, trial_sequence_trace_paths = _save_trial_sequence_videos(
            agent_cfg, checkpoint, output_dir
        )
        trial_sequence_results = []
        for trace_path in trial_sequence_trace_paths:
            trace_payload = json.loads(Path(trace_path).read_text(encoding="utf-8"))
            trial_sequence_results.append(
                {
                    "video": trace_payload.get("video_path"),
                    "trace": str(trace_path),
                    "trials": int(trace_payload.get("trial_count", 0)),
                    "success_count": int(trace_payload.get("success_count", 0)),
                    "success_rate": float(trace_payload.get("success_rate", 0.0)),
                    "raw_success_count": int(trace_payload.get("raw_success_count", 0)),
                    "raw_success_rate": float(trace_payload.get("raw_success_rate", 0.0)),
                    "post_success_hold_count": int(trace_payload.get("post_success_hold_count", 0)),
                    "post_success_hold_rate": float(trace_payload.get("post_success_hold_rate", 0.0)),
                }
            )
        trial_sequence_requested = args_cli.save_trial_sequence_videos > 0
        trial_sequence_passed = (
            len(trial_sequence_results) == int(args_cli.save_trial_sequence_videos)
            and all(
                result["success_rate"] >= float(args_cli.success_threshold)
                for result in trial_sequence_results
            )
            if trial_sequence_requested
            else True
        )
        passed = bool(vector_passed and trial_sequence_passed)
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
            "seed": int(args_cli.seed),
            "dynamic_curriculum_alpha": args_cli.dynamic_curriculum_alpha,
            "dynamic_tabletop_speed_range": (
                list(args_cli.dynamic_tabletop_speed_range)
                if args_cli.dynamic_tabletop_speed_range is not None
                else None
            ),
            "tabletop_asset_curriculum_alpha": args_cli.tabletop_asset_curriculum_alpha,
            "tabletop_motion_curriculum_alpha": args_cli.tabletop_motion_curriculum_alpha,
            "tabletop_pregrasp_lead_time": args_cli.tabletop_pregrasp_lead_time,
            "tabletop_pregrasp_ahead_distance": args_cli.tabletop_pregrasp_ahead_distance,
            "scripted_action_prior_active_residual_scale": (
                args_cli.scripted_action_prior_active_residual_scale
            ),
            "scripted_relative_lift_target_scale": args_cli.scripted_relative_lift_target_scale,
            "scripted_action_prior_lift_steps": args_cli.scripted_action_prior_lift_steps,
            "episode_length_s": args_cli.episode_length_s,
            "dynamic_success_hold_steps": args_cli.dynamic_success_hold_steps,
            "tabletop_post_success_hand_close_fraction": (
                args_cli.tabletop_post_success_hand_close_fraction
            ),
            "success_threshold": float(args_cli.success_threshold),
            "video_post_success_steps": int(args_cli.video_post_success_steps),
            "video_post_success_min_stable_frac": float(args_cli.video_post_success_min_stable_frac),
            "video_post_success_max_relative_drift": float(
                args_cli.video_post_success_max_relative_drift
            ),
            "video_post_success_require_final_stable": bool(
                args_cli.video_post_success_require_final_stable
            ),
            "trial_sequence_cycle_envs": bool(args_cli.trial_sequence_cycle_envs),
            "trial_sequence_seed_mode": str(args_cli.trial_sequence_seed_mode),
            "vector_passed": bool(vector_passed),
            "trial_sequence_passed": bool(trial_sequence_passed),
            "trial_sequence_results": trial_sequence_results,
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
            failure_reasons = []
            if not vector_passed:
                failure_reasons.append(
                    f"success_rate={eval_summary['success_rate']:.3f} "
                    f"below threshold={args_cli.success_threshold:.3f}"
                )
            if not trial_sequence_passed:
                rates = ", ".join(
                    f"{result['success_count']}/{result['trials']}={result['success_rate']:.3f}"
                    for result in trial_sequence_results
                ) or "no completed trial sequence"
                failure_reasons.append(
                    f"trial_sequence_success_rate={rates} below threshold={args_cli.success_threshold:.3f}"
                )
            raise SystemExit("; ".join(failure_reasons))
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
