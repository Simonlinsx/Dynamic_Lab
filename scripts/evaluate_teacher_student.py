#!/usr/bin/env python3
"""Evaluate a PointTemporalStudent checkpoint in an IsaacLab dynamic grasp env."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToolReal-Revo2-Franka-FallingBaton-Teacher-Direct-v0")
parser.add_argument("--checkpoint", required=True, help="PointTemporalStudent checkpoint.")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument(
    "--first-episode-per-env",
    action="store_true",
    help=(
        "Count at most the first completed episode from each vector environment. "
        "Use this with episodes <= num-envs for an unbiased one-trial-per-initial-state evaluation."
    ),
)
parser.add_argument("--success-threshold", type=float, default=0.0)
parser.add_argument(
    "--object-contact-force-diagnostics",
    action="store_true",
    help="Record filtered PhysX fingertip-to-object forces alongside geometric contact metrics.",
)
parser.add_argument(
    "--object-contact-force-threshold",
    type=float,
    default=0.05,
    help="Per-fingertip force threshold in newtons used only by contact diagnostics.",
)
parser.add_argument(
    "--skip-vector-eval",
    action="store_true",
    help="Skip non-rendered student eval and only run requested video attempts.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--pointcloud-source",
    choices=("auto", "clean", "rgbd_projected_mask"),
    default="auto",
    help="Student point-cloud source. auto reads checkpoint metadata.",
)
parser.add_argument(
    "--proprio-source",
    choices=("auto", "policy", "deployable_robot"),
    default="auto",
    help="Student proprioception source. auto reads checkpoint metadata and preserves legacy checkpoints.",
)
parser.add_argument("--history", type=int, default=None, help="Override checkpoint history.")
parser.add_argument(
    "--history-bootstrap",
    choices=("repeat_initial", "zero_pad"),
    default="repeat_initial",
    help=(
        "How to initialize a student's temporal history after reset. repeat_initial fills every "
        "slot with the first deployable observation; zero_pad preserves the legacy partial history."
    ),
)
parser.add_argument("--object-points", type=int, default=None, help="Override checkpoint point count.")
parser.add_argument("--rgbd-width", type=int, default=160)
parser.add_argument("--rgbd-height", type=int, default=120)
parser.add_argument("--rgbd-mask-points", type=int, default=768)
parser.add_argument("--rgbd-mask-dilation", type=int, default=1)
parser.add_argument("--rgbd-depth-tolerance", type=float, default=0.045)
parser.add_argument("--rgbd-min-valid-points", type=int, default=16)
parser.add_argument("--rgbd-clean-fallback", action="store_true", default=True)
parser.add_argument("--no-rgbd-clean-fallback", action="store_false", dest="rgbd_clean_fallback")
parser.add_argument(
    "--rgbd-temporal-fallback",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "For low-visibility RGB-D frames, reuse the last valid masked point cloud. "
        "This uses only observation history and does not expose simulator object state."
    ),
)
parser.add_argument(
    "--student-camera-track-object",
    action="store_true",
    default=False,
    help="Aim the student RGB-D camera at the live object. By default the camera is fixed in a stable third view.",
)
parser.add_argument("--student-camera-track-offset", type=float, nargs=3, default=(0.65, 0.95, 0.65))
parser.add_argument("--student-camera-track-target-offset", type=float, nargs=3, default=(0.0, 0.0, 0.05))
parser.add_argument(
    "--student-camera-eye",
    type=float,
    nargs=3,
    default=None,
    help="Fixed camera eye in environment-local coordinates. Must be paired with --student-camera-target.",
)
parser.add_argument(
    "--student-camera-target",
    type=float,
    nargs=3,
    default=None,
    help="Fixed camera target in environment-local coordinates. Must be paired with --student-camera-eye.",
)
parser.add_argument("--student-camera-focal-length", type=float, default=24.0)
parser.add_argument("--student-device", default=None, help="Torch device for student inference. Defaults to --device.")
parser.add_argument("--action-clamp", type=float, default=1.0)
parser.add_argument(
    "--student-action-mode",
    choices=("mean", "ppo_sample"),
    default="mean",
    help="Execute the deterministic student mean or sample from a PPO checkpoint's learned Gaussian.",
)
parser.add_argument(
    "--student-action-std-scale",
    type=float,
    default=1.0,
    help="Multiplier for the learned PPO action standard deviation in ppo_sample mode.",
)
parser.add_argument(
    "--action-ema-alpha",
    type=float,
    default=1.0,
    help="EMA coefficient for current student action. 1.0 disables smoothing.",
)
parser.add_argument(
    "--action-rate-limit",
    type=float,
    default=0.0,
    help="Optional max absolute per-step action delta after smoothing. 0 disables rate limiting.",
)
parser.add_argument(
    "--predicted-grasp-hold-threshold",
    type=float,
    default=0.0,
    help=(
        "If >0, latch a deployable hand-hold reflex when the student's predicted "
        "true_grasp/grasp_seen privileged outputs exceed this threshold."
    ),
)
parser.add_argument(
    "--predicted-grasp-hold-mode",
    choices=("max", "target", "learned_target", "target_plus_learned_residual"),
    default="max",
    help="Hand-hold reflex mode used after --predicted-grasp-hold-threshold latches.",
)
parser.add_argument(
    "--predicted-grasp-hold-target",
    type=float,
    nargs="*",
    default=None,
    help="Hand action target for hold-mode=target. Must have action_dim-7 values.",
)
parser.add_argument(
    "--predicted-grasp-hold-blend",
    type=float,
    default=1.0,
    help="Blend toward --predicted-grasp-hold-target on latched envs. 1.0 means full target.",
)
parser.add_argument(
    "--predicted-grasp-hold-rel-vel-threshold",
    type=float,
    default=0.0,
    help="Optional predicted object-palm relative velocity gate. 0 disables this extra gate.",
)
parser.add_argument(
    "--predicted-grasp-hold-learned-gate-threshold",
    type=float,
    default=0.0,
    help="Optional sigmoid(hold_logits) gate for learned_target mode. 0 disables this extra gate.",
)
parser.add_argument(
    "--predicted-grasp-hold-learned-target-clamp",
    type=float,
    default=1.0,
    help="Clamp learned hold_target values before blending. 0 disables target clamping.",
)
parser.add_argument(
    "--predicted-grasp-hold-learned-residual-scale",
    type=float,
    default=1.0,
    help="Scale learned residuals in target_plus_learned_residual mode.",
)
parser.add_argument(
    "--predicted-grasp-hold-learned-residual-clamp",
    type=float,
    default=0.25,
    help="Clamp learned residual values before adding them to the calibrated target. 0 disables residual clamping.",
)
parser.add_argument(
    "--predicted-grasp-arm-lock-delay-steps",
    type=int,
    default=-1,
    help=(
        "Deployable post-grasp arm stabilization delay. A non-negative value captures "
        "and holds the current arm action this many steps after the grasp latch; -1 disables it."
    ),
)
parser.add_argument(
    "--predicted-grasp-arm-lock-blend",
    type=float,
    default=1.0,
    help="Blend toward the captured post-grasp arm action after the lock delay.",
)
parser.add_argument(
    "--dynamic-curriculum-alpha",
    type=float,
    default=None,
    help="Override dynamic speed curriculum alpha for eval, when the env cfg exposes it.",
)
parser.add_argument("--tabletop-asset-curriculum-alpha", type=float, default=None)
parser.add_argument("--tabletop-motion-curriculum-alpha", type=float, default=None)
parser.add_argument("--episode-length-s", type=float, default=None)
parser.add_argument("--dynamic-success-hold-steps", type=int, default=None)
parser.add_argument("--stability-target-latch-min-success-streak", type=int, default=None)
parser.add_argument("--tabletop-pregrasp-lead-time", type=float, default=None)
parser.add_argument("--tabletop-pregrasp-ahead-distance", type=float, default=None)
parser.add_argument(
    "--output-dir",
    default="outputs/eval_teacher_student",
    help="Directory for JSON summaries and videos.",
)
parser.add_argument("--wandb-project", default=None)
parser.add_argument("--wandb-entity", default=None)
parser.add_argument("--wandb-run-name", default=None)
parser.add_argument("--wandb-group", default="deployable_student_eval")
parser.add_argument("--wandb-tags", nargs="*", default=None)
parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
parser.add_argument("--wandb-log-videos", action="store_true")
parser.add_argument("--save-success-videos", type=int, default=0)
parser.add_argument("--video-attempts", type=int, default=20)
parser.add_argument("--video-envs", type=int, default=4)
parser.add_argument("--video-stride", type=int, default=2)
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--video-max-steps", type=int, default=None)
parser.add_argument("--video-post-success-steps", type=int, default=60)
parser.add_argument(
    "--video-require-post-success-stable",
    action="store_true",
    help="Require post-success contact and stable object-palm motion before counting a video trial.",
)
parser.add_argument("--video-post-success-min-stable-frac", type=float, default=0.60)
parser.add_argument("--video-post-success-max-relative-drift", type=float, default=0.03)
parser.add_argument("--video-post-success-require-final-stable", action="store_true")
parser.add_argument(
    "--video-post-success-lift-mode",
    choices=("auto", "never", "always"),
    default="auto",
)
parser.add_argument(
    "--save-trial-sequence-videos",
    type=int,
    default=0,
    help="Save continuous videos containing repeated reset trials.",
)
parser.add_argument("--trial-sequence-trials", type=int, default=20)
parser.add_argument(
    "--trial-sequence-seed-mode",
    choices=("sequence", "per_trial"),
    default="sequence",
    help=(
        "Use one deterministic RNG stream across the continuous sequence, or reseed before "
        "each trial. sequence matches the validated teacher video evaluator."
    ),
)
parser.add_argument("--trial-sequence-pre-roll-frames", type=int, default=6)
parser.add_argument("--trial-sequence-gap-frames", type=int, default=6)
parser.add_argument("--save-rollout-videos-on-failure", type=int, default=0)
parser.add_argument(
    "--video-camera-track-object",
    action="store_true",
    default=False,
    help="Aim the video camera at the live object. By default it shares the fixed third-view student camera pose.",
)
parser.add_argument("--video-camera-track-offset", type=float, nargs=3, default=(0.65, 0.95, 0.65))
parser.add_argument("--video-camera-track-target-offset", type=float, nargs=3, default=(0.0, 0.0, 0.05))
parser.add_argument(
    "--video-camera-eye",
    type=float,
    nargs=3,
    default=None,
    help="Fixed video eye in environment-local coordinates. Defaults to --student-camera-eye.",
)
parser.add_argument(
    "--video-camera-target",
    type=float,
    nargs=3,
    default=None,
    help="Fixed video target in environment-local coordinates. Defaults to --student-camera-target.",
)
parser.add_argument("--video-camera-focal-length", type=float, default=24.0)
parser.add_argument("--video-camera-resolution", type=int, nargs=2, default=(960, 544))
parser.add_argument(
    "--video-pointcloud-visualization",
    choices=("none", "inset", "separate", "both"),
    default="none",
    help=(
        "Visualize the exact current point-cloud observation as a top-right inset, "
        "a synchronized companion MP4, or both."
    ),
)
parser.add_argument(
    "--video-pointcloud-panel-resolution",
    type=int,
    nargs=2,
    default=(320, 192),
    metavar=("WIDTH", "HEIGHT"),
)
parser.add_argument("--video-pointcloud-range", type=float, default=0.30)
parser.add_argument("--video-pointcloud-point-radius", type=int, default=2)
parser.add_argument("--video-pointcloud-inset-margin", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.first_episode_per_env and args_cli.episodes > args_cli.num_envs:
    parser.error("--first-episode-per-env requires --episodes <= --num-envs")

if (args_cli.student_camera_eye is None) != (args_cli.student_camera_target is None):
    parser.error("--student-camera-eye and --student-camera-target must be provided together")
if (args_cli.video_camera_eye is None) != (args_cli.video_camera_target is None):
    parser.error("--video-camera-eye and --video-camera-target must be provided together")

if not (0.0 < float(args_cli.action_ema_alpha) <= 1.0):
    raise ValueError("--action-ema-alpha must be in (0, 1].")
if float(args_cli.action_rate_limit) < 0.0:
    raise ValueError("--action-rate-limit must be non-negative.")
if float(args_cli.student_action_std_scale) < 0.0:
    raise ValueError("--student-action-std-scale must be non-negative.")
if any(int(value) <= 0 for value in args_cli.video_pointcloud_panel_resolution):
    raise ValueError("--video-pointcloud-panel-resolution values must be positive.")
if float(args_cli.video_pointcloud_range) <= 0.0:
    raise ValueError("--video-pointcloud-range must be positive.")
if int(args_cli.video_pointcloud_point_radius) <= 0:
    raise ValueError("--video-pointcloud-point-radius must be positive.")
if int(args_cli.video_pointcloud_inset_margin) < 0:
    raise ValueError("--video-pointcloud-inset-margin must be non-negative.")
if not (0.0 <= float(args_cli.predicted_grasp_hold_blend) <= 1.0):
    raise ValueError("--predicted-grasp-hold-blend must be in [0, 1].")
if int(args_cli.predicted_grasp_arm_lock_delay_steps) < -1:
    raise ValueError("--predicted-grasp-arm-lock-delay-steps must be >= -1.")
if not (0.0 <= float(args_cli.predicted_grasp_arm_lock_blend) <= 1.0):
    raise ValueError("--predicted-grasp-arm-lock-blend must be in [0, 1].")
if (
    float(args_cli.predicted_grasp_hold_threshold) > 0.0
    and args_cli.predicted_grasp_hold_mode in {"target", "target_plus_learned_residual"}
    and not args_cli.predicted_grasp_hold_target
):
    raise ValueError(
        "--predicted-grasp-hold-target is required when "
        "--predicted-grasp-hold-mode is target or target_plus_learned_residual."
    )

if (
    args_cli.pointcloud_source != "clean"
    or args_cli.save_success_videos > 0
    or args_cli.save_trial_sequence_videos > 0
):
    setattr(args_cli, "enable_cameras", True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402
from simtoolreal_lab.teacher_student import (  # noqa: E402
    PointTemporalStudent,
    deployable_robot_proprioception,
    masked_rgbd_object_points_in_palm_frame,
    object_points_in_palm_frame,
    primitive_surface_points_for_envs,
    quat_rotate_inverse,
    sample_box_surface_points,
    sample_unit_primitive_surface_points,
)
from simtoolreal_lab.teacher_student.evaluation import (  # noqa: E402
    EpisodeFunnelAccumulator,
    EpisodeFunnelTracker,
    flatten_numeric_metrics,
)
from simtoolreal_lab.teacher_student.visualization import (  # noqa: E402
    add_pointcloud_inset,
    render_pointcloud_panel,
)


def _trace(message: str) -> None:
    print(f"[STUDENT-EVAL] {message}", flush=True)


def _policy_tensor(obs) -> torch.Tensor:
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        if "obs" in obs:
            return obs["obs"]
    return obs


def _tensor_bool(extras: dict, key: str, num_envs: int, device: torch.device | str) -> torch.Tensor:
    value = extras.get(key)
    if value is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return value.bool()


def _load_student(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[PointTemporalStudent, dict, dict, torch.Tensor | None]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = dict(payload.get("metadata", {}))
        spec = dict(payload.get("spec", {}))
    else:
        state_dict = payload
        metadata = {}
        spec = {}

    log_std = None
    if isinstance(payload, dict):
        ppo_state_dict = payload.get("ppo_state_dict")
        if isinstance(ppo_state_dict, dict) and torch.is_tensor(ppo_state_dict.get("log_std")):
            log_std = ppo_state_dict["log_std"].detach().to(device=device, dtype=torch.float32).view(-1)

    history = int(args_cli.history or spec.get("history", metadata.get("history", 4)))
    num_points = int(args_cli.object_points or spec.get("num_object_points", metadata.get("object_points", 128)))
    point_feature_dim = int(spec.get("point_feature_dim", metadata.get("point_feature_dim", 3)))
    proprio_dim = int(spec.get("proprio_dim", metadata.get("proprio_dim", 76)))
    action_dim = int(spec.get("action_dim", metadata.get("action_dim", 0)))
    arm_dim = int(spec.get("arm_dim", metadata.get("arm_dim", 7)))
    privileged_dim = int(spec.get("compact_privileged_dim", metadata.get("compact_privileged_dim", 32)))
    geometric_summary = bool(spec.get("geometric_summary", False))
    privileged_action_conditioning = bool(spec.get("privileged_action_conditioning", False))
    if action_dim <= 0:
        raise RuntimeError("Student checkpoint does not contain a positive action_dim in spec/metadata.")

    model = PointTemporalStudent(
        history=history,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        privileged_dim=privileged_dim,
        arm_dim=arm_dim,
        point_input_dim=point_feature_dim,
        geometric_summary=geometric_summary,
        privileged_action_conditioning=privileged_action_conditioning,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = ("hold_head.", "hold_gate_head.")
    bad_missing = [name for name in missing if not name.startswith(allowed_missing_prefixes)]
    if bad_missing or unexpected:
        raise RuntimeError(
            "Student checkpoint state mismatch: "
            f"missing={bad_missing}, unexpected={list(unexpected)}"
        )
    if missing:
        _trace(f"checkpoint missing optional hold head weights; initialized {list(missing)}")
    normalization = payload.get("normalization") if isinstance(payload, dict) else None
    if normalization is not None:
        required_norm_keys = {
            "pointcloud_mean",
            "pointcloud_std",
            "proprio_mean",
            "proprio_std",
        }
        missing_norm_keys = required_norm_keys - set(normalization)
        if missing_norm_keys:
            raise RuntimeError(
                "Student checkpoint normalization is incomplete: "
                f"missing={sorted(missing_norm_keys)}"
            )
        model.input_normalization = {
            key: normalization[key].detach().to(device=device, dtype=torch.float32)
            for key in required_norm_keys
        }
        _trace("loaded checkpoint input normalization")
    else:
        model.input_normalization = None
    model.eval()

    resolved_spec = {
        "history": history,
        "num_object_points": num_points,
        "point_feature_dim": point_feature_dim,
        "proprio_dim": proprio_dim,
        "action_dim": action_dim,
        "arm_dim": arm_dim,
        "hand_dim": max(action_dim - arm_dim, 0),
        "compact_privileged_dim": privileged_dim,
        "geometric_summary": geometric_summary,
        "privileged_action_conditioning": privileged_action_conditioning,
    }
    if args_cli.student_action_mode == "ppo_sample":
        if log_std is None:
            raise RuntimeError(
                "--student-action-mode=ppo_sample requires a checkpoint containing "
                "ppo_state_dict['log_std']."
            )
        if log_std.numel() != action_dim:
            raise RuntimeError(
                f"PPO log_std dimension mismatch: expected {action_dim}, got {log_std.numel()}."
            )
        _trace(
            "using stochastic PPO action execution: "
            f"std_mean={float(log_std.exp().mean().cpu()):.5f} "
            f"std_scale={float(args_cli.student_action_std_scale):.3f}"
        )
    return model, metadata, resolved_spec, log_std


def _resolve_pointcloud_source(metadata: dict) -> str:
    if args_cli.pointcloud_source != "auto":
        return args_cli.pointcloud_source
    return str(metadata.get("pointcloud_source", "clean"))


def _resolve_proprio_source(metadata: dict) -> str:
    if args_cli.proprio_source != "auto":
        return args_cli.proprio_source
    return str(metadata.get("proprio_source", "policy"))


def _current_proprio(unwrapped_env, obs) -> torch.Tensor:
    if getattr(args_cli, "resolved_proprio_source", "policy") == "deployable_robot":
        return deployable_robot_proprioception(unwrapped_env)
    return _policy_tensor(obs).to(unwrapped_env.device)


def _make_env(task: str, num_envs: int, render_mode: str | None, pointcloud_source: str):
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=num_envs)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.dynamic_curriculum_alpha is not None and hasattr(
        env_cfg, "dynamic_grasp_speed_curriculum_override_alpha"
    ):
        env_cfg.dynamic_grasp_speed_curriculum_override_alpha = float(args_cli.dynamic_curriculum_alpha)
    if args_cli.tabletop_asset_curriculum_alpha is not None and hasattr(
        env_cfg, "tabletop_asset_curriculum_override_alpha"
    ):
        env_cfg.tabletop_asset_curriculum_override_alpha = float(
            args_cli.tabletop_asset_curriculum_alpha
        )
    if args_cli.tabletop_motion_curriculum_alpha is not None and hasattr(
        env_cfg, "tabletop_motion_mode_curriculum_override_alpha"
    ):
        env_cfg.tabletop_motion_mode_curriculum_override_alpha = float(
            args_cli.tabletop_motion_curriculum_alpha
        )
    if args_cli.episode_length_s is not None and hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = float(args_cli.episode_length_s)
    if args_cli.dynamic_success_hold_steps is not None and hasattr(
        env_cfg, "dynamic_success_hold_steps"
    ):
        env_cfg.dynamic_success_hold_steps = int(args_cli.dynamic_success_hold_steps)
    if args_cli.stability_target_latch_min_success_streak is not None:
        env_cfg.stability_target_latch_min_success_streak = int(
            args_cli.stability_target_latch_min_success_streak
        )
    if args_cli.tabletop_pregrasp_lead_time is not None and hasattr(
        env_cfg, "dynamic_tabletop_pregrasp_lead_time"
    ):
        env_cfg.dynamic_tabletop_pregrasp_lead_time = float(args_cli.tabletop_pregrasp_lead_time)
    if args_cli.tabletop_pregrasp_ahead_distance is not None and hasattr(
        env_cfg, "dynamic_tabletop_pregrasp_ahead_distance"
    ):
        env_cfg.dynamic_tabletop_pregrasp_ahead_distance = float(
            args_cli.tabletop_pregrasp_ahead_distance
        )
    if hasattr(env_cfg, "object_contact_force_diagnostics_enabled"):
        env_cfg.object_contact_force_diagnostics_enabled = bool(
            args_cli.object_contact_force_diagnostics
        )
        env_cfg.object_contact_force_threshold = float(args_cli.object_contact_force_threshold)

    if pointcloud_source == "rgbd_projected_mask":
        if not hasattr(env_cfg, "student_camera_enabled"):
            raise RuntimeError(f"Task {task} does not expose student_camera_enabled.")
        env_cfg.student_camera_enabled = True
        env_cfg.student_camera.data_types = ["rgb", "distance_to_image_plane"]
        env_cfg.student_camera.width = int(args_cli.rgbd_width)
        env_cfg.student_camera.height = int(args_cli.rgbd_height)
        env_cfg.student_camera.spawn.focal_length = float(args_cli.student_camera_focal_length)
        env_cfg.student_camera.return_latest_camera_pose = True

    if render_mode == "rgb_array" and hasattr(env_cfg, "video_camera_enabled"):
        env_cfg.video_camera_enabled = True
        if hasattr(env_cfg, "video_camera"):
            env_step_dt = float(env_cfg.sim.dt) * max(int(getattr(env_cfg, "decimation", 1)), 1)
            env_cfg.video_camera.update_period = env_step_dt * max(int(args_cli.video_stride), 1)
        if hasattr(env_cfg, "terminate_on_success"):
            env_cfg.terminate_on_success = False
        env_cfg.video_camera.spawn.focal_length = float(args_cli.video_camera_focal_length)
        env_cfg.video_camera.width = int(args_cli.video_camera_resolution[0])
        env_cfg.video_camera.height = int(args_cli.video_camera_resolution[1])

    env = gym.make(task, cfg=env_cfg, render_mode=render_mode)
    env.unwrapped.sim._app_control_on_stop_handle = None
    return env


def _set_tracking_camera(
    unwrapped_env,
    camera_name: str,
    track_offset: tuple[float, float, float],
    target_offset,
    *,
    track_object: bool,
    fixed_eye: tuple[float, float, float] | None = None,
    fixed_target: tuple[float, float, float] | None = None,
) -> bool:
    camera = getattr(unwrapped_env, camera_name, None)
    if camera is None:
        return False
    origins = unwrapped_env.scene.env_origins
    if fixed_eye is not None and fixed_target is not None:
        eye = torch.tensor(fixed_eye, device=origins.device, dtype=origins.dtype).unsqueeze(0)
        targets = torch.tensor(fixed_target, device=origins.device, dtype=origins.dtype).unsqueeze(0)
        camera.set_world_poses_from_view(origins + eye, origins + targets)
        if hasattr(camera, "_update_poses") and hasattr(camera, "_ALL_INDICES"):
            camera._update_poses(camera._ALL_INDICES)
        return True
    target_offset_t = torch.tensor(target_offset, device=origins.device, dtype=origins.dtype).unsqueeze(0)
    camera_offset_t = torch.tensor(track_offset, device=origins.device, dtype=origins.dtype).unsqueeze(0)

    object_root_pos = None
    if bool(track_object):
        object_asset = getattr(unwrapped_env, "object", None)
        object_data = getattr(object_asset, "data", None)
        object_root_pos = getattr(object_data, "root_pos_w", None)
    if object_root_pos is not None:
        targets = object_root_pos[:, :3] + target_offset_t
    else:
        object_init_pos = torch.tensor(
            unwrapped_env.cfg.object_cfg.init_state.pos, device=origins.device, dtype=origins.dtype
        ).unsqueeze(0)
        targets = origins + object_init_pos + target_offset_t

    camera.set_world_poses_from_view(targets + camera_offset_t, targets)
    if hasattr(camera, "_update_poses") and hasattr(camera, "_ALL_INDICES"):
        camera._update_poses(camera._ALL_INDICES)
    return True


def _set_student_camera_poses(unwrapped_env) -> bool:
    return _set_tracking_camera(
        unwrapped_env,
        "_student_camera",
        tuple(float(v) for v in args_cli.student_camera_track_offset),
        tuple(float(v) for v in args_cli.student_camera_track_target_offset),
        track_object=bool(args_cli.student_camera_track_object),
        fixed_eye=(
            tuple(float(v) for v in args_cli.student_camera_eye)
            if args_cli.student_camera_eye is not None
            else None
        ),
        fixed_target=(
            tuple(float(v) for v in args_cli.student_camera_target)
            if args_cli.student_camera_target is not None
            else None
        ),
    )


def _set_video_camera_poses(unwrapped_env) -> bool:
    fixed_eye = args_cli.video_camera_eye or args_cli.student_camera_eye
    fixed_target = args_cli.video_camera_target or args_cli.student_camera_target
    return _set_tracking_camera(
        unwrapped_env,
        "_video_camera",
        tuple(float(v) for v in args_cli.video_camera_track_offset),
        tuple(float(v) for v in args_cli.video_camera_track_target_offset),
        track_object=bool(args_cli.video_camera_track_object),
        fixed_eye=tuple(float(v) for v in fixed_eye) if fixed_eye is not None else None,
        fixed_target=tuple(float(v) for v in fixed_target) if fixed_target is not None else None,
    )


def _force_camera_update(camera, dt: float) -> None:
    if camera is None:
        return
    if hasattr(camera, "update"):
        camera.update(dt, force_recompute=True)


def _student_camera_rgbd(unwrapped_env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    camera = getattr(unwrapped_env, "_student_camera", None)
    if camera is None:
        raise RuntimeError("pointcloud-source=rgbd_projected_mask requires cfg.student_camera_enabled=True")
    _force_camera_update(camera, float(unwrapped_env.dt))
    data = camera.data
    rgb = data.output.get("rgb")
    depth = data.output.get("distance_to_image_plane")
    if depth is None:
        depth = data.output.get("depth")
    if rgb is None or depth is None:
        raise RuntimeError(f"student camera output missing rgb/depth; available keys={list(data.output.keys())}")
    return rgb, depth, data.pos_w, data.quat_w_ros, data.intrinsic_matrices


def _make_rgbd_mask_point_template(
    unwrapped_env,
    num_points: int,
    device: torch.device | str,
) -> torch.Tensor:
    if torch.is_tensor(getattr(unwrapped_env, "_active_object_size_tensor", None)):
        return sample_unit_primitive_surface_points(num_points, device)
    points, _ = sample_box_surface_points(
        tuple(float(v) for v in unwrapped_env.cfg.object_size),
        num_points,
        device,
    )
    return points


def _current_pointcloud(
    unwrapped_env,
    pointcloud_source: str,
    local_points: torch.Tensor,
    affordance_points: torch.Tensor | None,
    point_feature_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    unwrapped_env._compute_intermediate_values()
    clean_points = object_points_in_palm_frame(
        local_points,
        unwrapped_env._object_pos_w,
        unwrapped_env._object_quat_w,
        unwrapped_env._palm_pos_w,
        unwrapped_env._palm_quat_w,
    )
    if pointcloud_source == "clean":
        valid = torch.ones(clean_points.shape[:2], dtype=clean_points.dtype, device=clean_points.device)
        features = (
            torch.cat((clean_points, torch.zeros_like(clean_points)), dim=-1)
            if point_feature_dim == 6
            else clean_points
        )
        return features, valid, {"rgbd_valid_mean": float(valid.shape[-1]), "rgbd_fallback_envs": 0.0}

    assert affordance_points is not None
    mask_local_points = affordance_points
    mask_object_size: tuple[float, float, float] | torch.Tensor = tuple(
        float(v) for v in unwrapped_env.cfg.object_size
    )
    active_sizes = getattr(unwrapped_env, "_active_object_size_tensor", None)
    active_shape_codes = getattr(unwrapped_env, "_active_object_shape_codes_tensor", None)
    if (
        affordance_points.ndim == 3
        and affordance_points.shape[0] == 4
        and torch.is_tensor(active_sizes)
        and torch.is_tensor(active_shape_codes)
    ):
        mask_local_points = primitive_surface_points_for_envs(
            affordance_points,
            active_shape_codes,
            active_sizes,
        )
        mask_object_size = active_sizes
        clean_local_points = mask_local_points[:, : local_points.shape[0]]
        clean_points = object_points_in_palm_frame(
            clean_local_points,
            unwrapped_env._object_pos_w,
            unwrapped_env._object_quat_w,
            unwrapped_env._palm_pos_w,
            unwrapped_env._palm_quat_w,
        )
    _set_student_camera_poses(unwrapped_env)
    rgb, depth, camera_pos_w, camera_quat_w_ros, camera_intrinsics = _student_camera_rgbd(unwrapped_env)
    rgbd_points = masked_rgbd_object_points_in_palm_frame(
        mask_local_points,
        mask_object_size,
        depth,
        camera_pos_w,
        camera_quat_w_ros,
        camera_intrinsics,
        unwrapped_env._object_pos_w,
        unwrapped_env._object_quat_w,
        unwrapped_env._palm_pos_w,
        unwrapped_env._palm_quat_w,
        num_points=int(local_points.shape[0]),
        rgb=rgb if point_feature_dim == 6 else None,
        mask_dilation=args_cli.rgbd_mask_dilation,
        depth_tolerance=args_cli.rgbd_depth_tolerance,
    )
    points = rgbd_points["points_palm"]
    colors = rgbd_points["colors"]
    valid = rgbd_points["valid"]
    valid_counts = torch.sum(valid > 0.0, dim=-1)
    low_valid = valid_counts < int(args_cli.rgbd_min_valid_points)
    fallback_count = int(low_valid.sum().detach().cpu())
    if fallback_count > 0 and bool(args_cli.rgbd_clean_fallback):
        points[low_valid] = clean_points[low_valid]
        colors[low_valid] = 0.0
        valid[low_valid] = 1.0
    features = torch.cat((points, colors), dim=-1) if point_feature_dim == 6 else points
    return features, valid, {
        "rgbd_valid_mean": float(valid_counts.float().mean().detach().cpu()),
        "rgbd_fallback_envs": float(fallback_count),
    }


def _student_action(
    model: PointTemporalStudent,
    point_hist: torch.Tensor,
    valid_hist: torch.Tensor,
    proprio_hist: torch.Tensor,
    log_std: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    normalization = getattr(model, "input_normalization", None)
    if normalization is not None:
        point_hist = (point_hist - normalization["pointcloud_mean"]) / normalization[
            "pointcloud_std"
        ]
        proprio_hist = (proprio_hist - normalization["proprio_mean"]) / normalization[
            "proprio_std"
        ]
    out = model(point_hist, valid_hist, proprio_hist)
    action = out["action"]
    if args_cli.student_action_mode == "ppo_sample":
        if log_std is None:
            raise RuntimeError("PPO action sampling requested without a learned log_std tensor.")
        action = action + torch.randn_like(action) * log_std.exp().view(1, -1) * float(
            args_cli.student_action_std_scale
        )
    if args_cli.action_clamp > 0.0:
        action = torch.clamp(action, -float(args_cli.action_clamp), float(args_cli.action_clamp))
    return action, out["privileged"], out["hold_target"], out["hold_logits"]


def _apply_predicted_grasp_hold(
    action: torch.Tensor,
    previous_action: torch.Tensor,
    predicted_privileged: torch.Tensor,
    predicted_hold_target: torch.Tensor | None,
    predicted_hold_logits: torch.Tensor | None,
    hold_latch: torch.Tensor,
) -> torch.Tensor:
    threshold = float(args_cli.predicted_grasp_hold_threshold)
    if threshold <= 0.0:
        return action
    if predicted_privileged.shape[-1] <= 16 or action.shape[-1] <= 7:
        return action
    grasp_signal = torch.maximum(predicted_privileged[:, 15], predicted_privileged[:, 16])
    should_latch = grasp_signal > threshold
    rel_vel_threshold = float(args_cli.predicted_grasp_hold_rel_vel_threshold)
    if rel_vel_threshold > 0.0 and predicted_privileged.shape[-1] > 14:
        should_latch &= predicted_privileged[:, 14] < rel_vel_threshold
    learned_gate_threshold = float(args_cli.predicted_grasp_hold_learned_gate_threshold)
    if learned_gate_threshold > 0.0:
        if predicted_hold_logits is None:
            return action
        should_latch &= torch.sigmoid(predicted_hold_logits) > learned_gate_threshold
    hold_latch |= should_latch
    if bool(hold_latch.any()):
        held = action.clone()
        if args_cli.predicted_grasp_hold_mode == "max":
            held[hold_latch, 7:] = torch.maximum(action[hold_latch, 7:], previous_action[hold_latch, 7:])
        elif args_cli.predicted_grasp_hold_mode in {"target", "target_plus_learned_residual"}:
            target_values = args_cli.predicted_grasp_hold_target or []
            hand_dim = action.shape[-1] - 7
            if len(target_values) != hand_dim:
                raise RuntimeError(
                    f"--predicted-grasp-hold-target has {len(target_values)} values, expected hand_dim={hand_dim}."
                )
            target = torch.tensor(target_values, dtype=action.dtype, device=action.device).view(1, hand_dim)
            if args_cli.predicted_grasp_hold_mode == "target_plus_learned_residual":
                if predicted_hold_target is None:
                    return action
                if predicted_hold_target.shape[-1] != hand_dim:
                    raise RuntimeError(
                        f"learned hold residual has dim={predicted_hold_target.shape[-1]}, expected hand_dim={hand_dim}."
                    )
                residual = predicted_hold_target
                residual_clamp = float(args_cli.predicted_grasp_hold_learned_residual_clamp)
                if residual_clamp > 0.0:
                    residual = torch.clamp(residual, -residual_clamp, residual_clamp)
                target = target + float(args_cli.predicted_grasp_hold_learned_residual_scale) * residual
                target_clamp = float(args_cli.predicted_grasp_hold_learned_target_clamp)
                if target_clamp > 0.0:
                    target = torch.clamp(target, -target_clamp, target_clamp)
            blend = float(args_cli.predicted_grasp_hold_blend)
            selected_target = target[hold_latch] if target.shape[0] == action.shape[0] else target
            held[hold_latch, 7:] = (1.0 - blend) * action[hold_latch, 7:] + blend * selected_target
        else:
            if predicted_hold_target is None:
                return action
            hand_dim = action.shape[-1] - 7
            if predicted_hold_target.shape[-1] != hand_dim:
                raise RuntimeError(
                    f"learned hold target has dim={predicted_hold_target.shape[-1]}, expected hand_dim={hand_dim}."
                )
            target = predicted_hold_target
            target_clamp = float(args_cli.predicted_grasp_hold_learned_target_clamp)
            if target_clamp > 0.0:
                target = torch.clamp(target, -target_clamp, target_clamp)
            blend = float(args_cli.predicted_grasp_hold_blend)
            held[hold_latch, 7:] = (
                (1.0 - blend) * action[hold_latch, 7:] + blend * target[hold_latch]
            )
        action = held
    return action


def _apply_predicted_grasp_arm_lock(
    action: torch.Tensor,
    hold_latch: torch.Tensor,
    arm_lock_age: torch.Tensor,
    arm_lock_target: torch.Tensor,
    arm_dim: int,
) -> torch.Tensor:
    delay = int(args_cli.predicted_grasp_arm_lock_delay_steps)
    if delay < 0 or arm_dim <= 0 or not bool(hold_latch.any()):
        return action

    newly_latched = hold_latch & (arm_lock_age < 0)
    arm_lock_age[newly_latched] = 0
    capture = hold_latch & (arm_lock_age == delay)
    if bool(capture.any()):
        arm_lock_target[capture] = action[capture, :arm_dim]

    locked = hold_latch & (arm_lock_age >= delay)
    if bool(locked.any()):
        blend = float(args_cli.predicted_grasp_arm_lock_blend)
        held = action.clone()
        held[locked, :arm_dim] = (
            (1.0 - blend) * action[locked, :arm_dim]
            + blend * arm_lock_target[locked]
        )
        action = held
    arm_lock_age[hold_latch] += 1
    return action


def _adapt_action(action: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
    if float(args_cli.action_ema_alpha) < 1.0:
        alpha = float(args_cli.action_ema_alpha)
        action = alpha * action + (1.0 - alpha) * previous_action
    if float(args_cli.action_rate_limit) > 0.0:
        limit = float(args_cli.action_rate_limit)
        action = previous_action + torch.clamp(action - previous_action, -limit, limit)
    if args_cli.action_clamp > 0.0:
        action = torch.clamp(action, -float(args_cli.action_clamp), float(args_cli.action_clamp))
    return action


def _make_histories(num_envs: int, spec: dict, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(
            (
                num_envs,
                spec["history"],
                spec["num_object_points"],
                spec.get("point_feature_dim", 3),
            ),
            device=device,
        ),
        torch.zeros((num_envs, spec["history"], spec["num_object_points"]), device=device),
        torch.zeros((num_envs, spec["history"], spec["proprio_dim"]), device=device),
    )


def _roll_histories(
    point_hist: torch.Tensor,
    valid_hist: torch.Tensor,
    proprio_hist: torch.Tensor,
    current_points: torch.Tensor,
    valid: torch.Tensor,
    proprio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    point_hist = torch.roll(point_hist, shifts=-1, dims=1)
    valid_hist = torch.roll(valid_hist, shifts=-1, dims=1)
    proprio_hist = torch.roll(proprio_hist, shifts=-1, dims=1)
    point_hist[:, -1] = current_points
    valid_hist[:, -1] = valid
    proprio_hist[:, -1] = proprio
    return point_hist, valid_hist, proprio_hist


def _reset_histories(
    point_hist: torch.Tensor,
    valid_hist: torch.Tensor,
    proprio_hist: torch.Tensor,
    done: torch.Tensor,
) -> None:
    if bool(done.any()):
        point_hist[done] = 0.0
        valid_hist[done] = 0.0
        proprio_hist[done] = 0.0


def _apply_rgbd_temporal_fallback(
    current_points: torch.Tensor,
    valid: torch.Tensor,
    point_hist: torch.Tensor,
    valid_hist: torch.Tensor,
) -> int:
    if not bool(args_cli.rgbd_temporal_fallback):
        return 0
    min_points = max(int(args_cli.rgbd_min_valid_points), 1)
    low_valid = valid.sum(dim=-1) < min_points
    previous_valid = valid_hist[:, -1].sum(dim=-1) >= min_points
    use_previous = low_valid & previous_valid
    if bool(use_previous.any()):
        current_points[use_previous] = point_hist[use_previous, -1]
        valid[use_previous] = valid_hist[use_previous, -1]
    return int(use_previous.sum().item())


def _run_student_eval(
    model: PointTemporalStudent,
    spec: dict,
    pointcloud_source: str,
    num_envs: int,
    episodes: int,
    log_std: torch.Tensor | None,
    render_mode: str | None = None,
) -> tuple[dict, object | None]:
    env = _make_env(args_cli.task, num_envs, render_mode, pointcloud_source)
    unwrapped = env.unwrapped
    device = unwrapped.device
    local_points, _ = sample_box_surface_points(
        tuple(float(v) for v in unwrapped.cfg.object_size),
        int(spec["num_object_points"]),
        device,
    )
    rgbd_mask_points = None
    if pointcloud_source == "rgbd_projected_mask":
        rgbd_mask_points = _make_rgbd_mask_point_template(
            unwrapped,
            max(int(args_cli.rgbd_mask_points), int(spec["num_object_points"])),
            device,
        )
    obs, _ = env.reset(seed=args_cli.seed)
    if render_mode == "rgb_array":
        _set_video_camera_poses(unwrapped)
    if pointcloud_source == "rgbd_projected_mask":
        _set_student_camera_poses(unwrapped)

    point_hist, valid_hist, proprio_hist = _make_histories(num_envs, spec, device)
    previous_action = torch.zeros((num_envs, int(spec["action_dim"])), device=device)
    hold_latch = torch.zeros(num_envs, dtype=torch.bool, device=device)
    arm_lock_age = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    arm_lock_target = torch.zeros((num_envs, int(spec["arm_dim"])), device=device)
    history_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
    pending_reset = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_true_grasp = torch.zeros_like(episode_success)
    episode_lifted = torch.zeros_like(episode_success)
    episode_stable = torch.zeros_like(episode_success)
    episode_hold_latched = torch.zeros_like(episode_success)
    episode_hold_latched_before_strict_grasp = torch.zeros_like(episode_success)
    episode_first_hold_latch_step = torch.full(
        (num_envs,), -1, dtype=torch.long, device=device
    )
    episode_funnel = EpisodeFunnelTracker(num_envs, device)
    funnel_accumulator = EpisodeFunnelAccumulator(
        str(getattr(unwrapped.cfg, "task_family", ""))
    )
    counted_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
    completed = 0
    success_count = 0
    true_count = 0
    lifted_count = 0
    stable_count = 0
    hold_latch_episode_count = 0
    hold_latch_before_strict_grasp_count = 0
    first_hold_latch_step_sum = 0
    first_hold_latch_step_count = 0
    total_reward = 0.0
    total_reward_count = 0
    last_log = {}
    rgbd_valid_sum = 0.0
    rgbd_fallback_sum = 0.0
    rgbd_temporal_fallback_sum = 0
    rgbd_steps = 0
    asset_specs = tuple(getattr(unwrapped, "_tabletop_asset_specs", ()))
    asset_names = [
        str(spec_item.get("label", spec_item.get("asset_id", f"asset_{index}")))
        if isinstance(spec_item, dict)
        else f"asset_{index}"
        for index, spec_item in enumerate(asset_specs)
    ]
    asset_episode_counts = [0 for _ in asset_names]
    asset_success_counts = [0 for _ in asset_names]
    asset_lifted_counts = [0 for _ in asset_names]
    asset_stable_counts = [0 for _ in asset_names]
    max_episode_length = int(unwrapped.max_episode_length)
    max_steps = args_cli.max_steps or (max_episode_length * (math.ceil(episodes / num_envs) + 2))

    with torch.inference_mode():
        for step in range(max_steps):
            _reset_histories(point_hist, valid_hist, proprio_hist, pending_reset)
            if bool(pending_reset.any()):
                previous_action[pending_reset] = 0.0
                hold_latch[pending_reset] = False
                arm_lock_age[pending_reset] = -1
                arm_lock_target[pending_reset] = 0.0
            current_points, valid, pc_info = _current_pointcloud(
                unwrapped,
                pointcloud_source,
                local_points,
                rgbd_mask_points,
                int(spec.get("point_feature_dim", 3)),
            )
            if pointcloud_source == "rgbd_projected_mask":
                rgbd_temporal_fallback_sum += _apply_rgbd_temporal_fallback(
                    current_points,
                    valid,
                    point_hist,
                    valid_hist,
                )
            proprio = _current_proprio(unwrapped, obs)
            if proprio.shape[-1] != int(spec["proprio_dim"]):
                raise RuntimeError(
                    f"proprio_dim mismatch: source={args_cli.resolved_proprio_source} "
                    f"env={proprio.shape[-1]} checkpoint={spec['proprio_dim']}"
                )
            point_hist, valid_hist, proprio_hist = _roll_histories(
                point_hist,
                valid_hist,
                proprio_hist,
                current_points,
                valid,
                proprio,
            )
            if args_cli.history_bootstrap == "repeat_initial":
                bootstrap = ~history_initialized
                if bool(bootstrap.any()):
                    point_hist[bootstrap] = current_points[bootstrap].unsqueeze(1)
                    valid_hist[bootstrap] = valid[bootstrap].unsqueeze(1)
                    proprio_hist[bootstrap] = proprio[bootstrap].unsqueeze(1)
            history_initialized[:] = True
            actions, predicted_privileged, predicted_hold_target, predicted_hold_logits = _student_action(
                model, point_hist, valid_hist, proprio_hist, log_std
            )
            hold_latch_before = hold_latch.clone()
            actions = _apply_predicted_grasp_hold(
                actions,
                previous_action,
                predicted_privileged,
                predicted_hold_target,
                predicted_hold_logits,
                hold_latch,
            )
            newly_latched = hold_latch & (~hold_latch_before)
            if bool(newly_latched.any()):
                strict_grasp_now = getattr(
                    unwrapped, "_strict_true_grasp", unwrapped._true_grasp
                ).bool()
                episode_hold_latched |= newly_latched
                episode_hold_latched_before_strict_grasp |= newly_latched & (~strict_grasp_now)
                first_latch = newly_latched & (episode_first_hold_latch_step < 0)
                episode_first_hold_latch_step[first_latch] = unwrapped.episode_length_buf[
                    first_latch
                ].long()
            actions = _apply_predicted_grasp_arm_lock(
                actions,
                hold_latch,
                arm_lock_age,
                arm_lock_target,
                int(spec["arm_dim"]),
            )
            actions = _adapt_action(actions, previous_action)
            active_asset_ids = getattr(unwrapped, "_tabletop_active_asset_ids", None)
            completed_asset_ids = (
                active_asset_ids.detach().clone()
                if torch.is_tensor(active_asset_ids) and asset_names
                else None
            )
            obs, rewards, terminated, truncated, extras = env.step(actions)
            previous_action = actions.detach()
            dones = terminated | truncated

            episode_success |= _tensor_bool(extras, "success_env", num_envs, device)
            episode_true_grasp |= _tensor_bool(extras, "true_grasp_env", num_envs, device)
            episode_lifted |= _tensor_bool(extras, "lifted_env", num_envs, device)
            episode_stable |= _tensor_bool(extras, "stable_hold_env", num_envs, device)
            episode_funnel.update(extras)
            reward_mask = ~counted_envs if args_cli.first_episode_per_env else torch.ones_like(counted_envs)
            total_reward += float(rewards[reward_mask].sum().item())
            total_reward_count += int(reward_mask.sum().item())
            if pointcloud_source == "rgbd_projected_mask":
                rgbd_valid_sum += pc_info["rgbd_valid_mean"]
                rgbd_fallback_sum += pc_info["rgbd_fallback_envs"]
                rgbd_steps += 1
            if "episode" in extras:
                last_log = {key: float(value.item()) for key, value in extras["episode"].items() if hasattr(value, "item")}

            raw_done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if raw_done_ids.numel() > 0:
                if args_cli.first_episode_per_env:
                    done_ids = raw_done_ids[~counted_envs[raw_done_ids]]
                else:
                    done_ids = raw_done_ids
                remaining = max(int(episodes) - completed, 0)
                if remaining <= 0:
                    break
                if done_ids.numel() > remaining:
                    done_ids = done_ids[:remaining]
                if done_ids.numel() > 0:
                    funnel_accumulator.add(episode_funnel.snapshot(done_ids))
                    completed += int(done_ids.numel())
                    success_count += int(episode_success[done_ids].sum().item())
                    true_count += int(episode_true_grasp[done_ids].sum().item())
                    lifted_count += int(episode_lifted[done_ids].sum().item())
                    stable_count += int(episode_stable[done_ids].sum().item())
                    hold_latch_episode_count += int(
                        episode_hold_latched[done_ids].sum().item()
                    )
                    hold_latch_before_strict_grasp_count += int(
                        episode_hold_latched_before_strict_grasp[done_ids].sum().item()
                    )
                    valid_latch_steps = episode_first_hold_latch_step[done_ids]
                    valid_latch_steps = valid_latch_steps[valid_latch_steps >= 0]
                    first_hold_latch_step_sum += int(valid_latch_steps.sum().item())
                    first_hold_latch_step_count += int(valid_latch_steps.numel())
                    if args_cli.first_episode_per_env:
                        counted_envs[done_ids] = True
                    if completed_asset_ids is not None:
                        for env_id in done_ids.tolist():
                            asset_id = int(completed_asset_ids[env_id].item())
                            if 0 <= asset_id < len(asset_names):
                                asset_episode_counts[asset_id] += 1
                                asset_success_counts[asset_id] += int(episode_success[env_id].item())
                                asset_lifted_counts[asset_id] += int(episode_lifted[env_id].item())
                                asset_stable_counts[asset_id] += int(episode_stable[env_id].item())
                    _trace(
                        f"episodes={completed}/{episodes} "
                        f"success_rate={success_count / max(completed, 1):.3f}"
                    )
                episode_success[raw_done_ids] = False
                episode_true_grasp[raw_done_ids] = False
                episode_lifted[raw_done_ids] = False
                episode_stable[raw_done_ids] = False
                episode_hold_latched[raw_done_ids] = False
                episode_hold_latched_before_strict_grasp[raw_done_ids] = False
                episode_first_hold_latch_step[raw_done_ids] = -1
                episode_funnel.reset(raw_done_ids)
                if completed >= episodes:
                    break
            pending_reset = dones
            history_initialized[dones] = False
        else:
            _trace(f"hit max_steps={max_steps} before completing requested episodes")

    completed = max(completed, 1)
    summary = {
        "episodes": int(completed),
        "success_count": int(success_count),
        "success_rate": float(success_count / completed),
        "true_grasp_episode_rate": float(true_count / completed),
        "lifted_episode_rate": float(lifted_count / completed),
        "stable_hold_episode_rate": float(stable_count / completed),
        "predicted_hold_latch_episode_rate": float(hold_latch_episode_count / completed),
        "predicted_hold_pre_strict_grasp_latch_episode_rate": float(
            hold_latch_before_strict_grasp_count / completed
        ),
        "predicted_hold_pre_strict_grasp_fraction_of_latched": float(
            hold_latch_before_strict_grasp_count / max(hold_latch_episode_count, 1)
        ),
        "predicted_hold_first_latch_step_mean": (
            float(first_hold_latch_step_sum / first_hold_latch_step_count)
            if first_hold_latch_step_count > 0
            else None
        ),
        "mean_step_reward": float(total_reward / max(total_reward_count, 1)),
        "max_episode_length": max_episode_length,
        "last_log": last_log,
        "rgbd_valid_mean": float(rgbd_valid_sum / max(rgbd_steps, 1)) if pointcloud_source == "rgbd_projected_mask" else None,
        "rgbd_fallback_envs_per_step": float(rgbd_fallback_sum / max(rgbd_steps, 1)) if pointcloud_source == "rgbd_projected_mask" else None,
        "rgbd_temporal_fallback_envs_per_step": (
            float(rgbd_temporal_fallback_sum / max(rgbd_steps, 1))
            if pointcloud_source == "rgbd_projected_mask"
            else None
        ),
        "failure_funnel": funnel_accumulator.summary(),
    }
    if asset_names:
        summary["asset_metrics"] = {
            name: {
                "episodes": int(asset_episode_counts[index]),
                "success_count": int(asset_success_counts[index]),
                "success_rate": float(
                    asset_success_counts[index] / max(asset_episode_counts[index], 1)
                ),
                "lifted_rate": float(
                    asset_lifted_counts[index] / max(asset_episode_counts[index], 1)
                ),
                "stable_hold_rate": float(
                    asset_stable_counts[index] / max(asset_episode_counts[index], 1)
                ),
            }
            for index, name in enumerate(asset_names)
        }
    if render_mode is None:
        env.close()
        return summary, None
    return summary, env


def _frames_have_signal(frames: list[np.ndarray]) -> bool:
    if not frames:
        return False
    stride = max(len(frames) // 5, 1)
    for frame in frames[::stride]:
        if float(frame.max()) > 10.0 and float(frame.std()) > 1.0:
            return True
    return False


def _pointcloud_inset_enabled() -> bool:
    return args_cli.video_pointcloud_visualization in {"inset", "both"}


def _pointcloud_separate_enabled() -> bool:
    return args_cli.video_pointcloud_visualization in {"separate", "both"}


def _read_video_frames(env, num_envs: int) -> list[np.ndarray | None]:
    unwrapped = env.unwrapped
    camera = getattr(unwrapped, "_video_camera", None)
    if camera is None:
        frame = unwrapped.render(recompute=True)
        if frame is None or frame.size == 0:
            return [None for _ in range(num_envs)]
        return [frame.copy() for _ in range(num_envs)]
    _set_video_camera_poses(unwrapped)
    _force_camera_update(camera, float(unwrapped.dt))
    rgb = camera.data.output.get("rgb")
    if rgb is None or rgb.numel() == 0:
        return [None for _ in range(num_envs)]
    rgb = rgb[..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
    return [
        rgb[env_id].copy() if env_id < rgb.shape[0] and rgb[env_id].size > 0 else None
        for env_id in range(num_envs)
    ]


def _pointcloud_panels(
    current_points: torch.Tensor | None,
    valid: torch.Tensor | None,
    num_envs: int,
) -> list[np.ndarray]:
    if current_points is None or valid is None:
        features_np = np.zeros((num_envs, 1, 3), dtype=np.float32)
        valid_np = np.zeros((num_envs, 1), dtype=np.float32)
    else:
        features_np = current_points[:num_envs].detach().float().cpu().numpy()
        valid_np = valid[:num_envs].detach().float().cpu().numpy()
    resolution = tuple(int(value) for value in args_cli.video_pointcloud_panel_resolution)
    return [
        render_pointcloud_panel(
            features_np[env_id],
            valid_np[env_id],
            resolution=resolution,
            view_range=float(args_cli.video_pointcloud_range),
            point_radius=int(args_cli.video_pointcloud_point_radius),
        )
        for env_id in range(num_envs)
    ]


def _current_visualization_pointcloud(
    unwrapped,
    pointcloud_source: str,
    local_points: torch.Tensor,
    rgbd_mask_points: torch.Tensor | None,
    point_feature_dim: int,
    point_hist: torch.Tensor,
    valid_hist: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if args_cli.video_pointcloud_visualization == "none":
        return None, None
    current_points, valid, _ = _current_pointcloud(
        unwrapped,
        pointcloud_source,
        local_points,
        rgbd_mask_points,
        point_feature_dim,
    )
    if pointcloud_source == "rgbd_projected_mask":
        _apply_rgbd_temporal_fallback(current_points, valid, point_hist, valid_hist)
    return current_points, valid


def _capture_video_frames(
    env,
    frames_by_env: list[list[np.ndarray]],
    current_points: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    pointcloud_frames_by_env: list[list[np.ndarray]] | None = None,
) -> None:
    raw_frames = _read_video_frames(env, len(frames_by_env))
    visualize_pointcloud = args_cli.video_pointcloud_visualization != "none"
    panels = (
        _pointcloud_panels(current_points, valid, len(frames_by_env))
        if visualize_pointcloud
        else [None for _ in frames_by_env]
    )
    for env_id, frames in enumerate(frames_by_env):
        frame = raw_frames[env_id]
        if frame is None:
            continue
        panel = panels[env_id]
        if panel is not None and _pointcloud_inset_enabled():
            frame = add_pointcloud_inset(
                frame,
                panel,
                margin=int(args_cli.video_pointcloud_inset_margin),
            )
        frames.append(frame)
        if (
            panel is not None
            and _pointcloud_separate_enabled()
            and pointcloud_frames_by_env is not None
        ):
            pointcloud_frames_by_env[env_id].append(panel)


def _trace_tensor(unwrapped_env, name: str, num_envs: int, default: float = 0.0) -> torch.Tensor:
    value = getattr(unwrapped_env, name, None)
    if value is None:
        return torch.full((num_envs,), default, device=unwrapped_env.device)
    if not torch.is_tensor(value):
        return torch.as_tensor(value, device=unwrapped_env.device)
    return value[:num_envs]


def _append_video_trace(env, traces_by_env: list[list[dict]], extras: dict, step: int) -> None:
    unwrapped = env.unwrapped
    num_envs = len(traces_by_env)
    device = unwrapped.device
    object_height_delta = _trace_tensor(unwrapped, "_object_height_delta", num_envs)
    object_palm_rel_vel = _trace_tensor(unwrapped, "_object_palm_rel_vel", num_envs)
    success_streak = _trace_tensor(unwrapped, "_success_streak", num_envs)
    true_grasp = _tensor_bool(extras, "true_grasp_env", num_envs, device)
    strict_true_grasp = (
        _tensor_bool(extras, "strict_true_grasp_env", num_envs, device)
        if "strict_true_grasp_env" in extras
        else true_grasp
    )
    lifted = _tensor_bool(extras, "lifted_env", num_envs, device)
    stable = _tensor_bool(extras, "stable_hold_env", num_envs, device)
    success = _tensor_bool(extras, "success_env", num_envs, device)
    object_minus_palm = unwrapped._object_pos_w[:num_envs] - unwrapped._palm_pos_w[:num_envs]

    for env_id, trace in enumerate(traces_by_env):
        trace.append(
            {
                "step": int(step),
                "object_height_delta": float(object_height_delta[env_id].detach().cpu().item()),
                "object_palm_rel_vel": float(object_palm_rel_vel[env_id].detach().cpu().item()),
                "success_streak": int(success_streak[env_id].detach().cpu().item()),
                "true_grasp": bool(true_grasp[env_id].detach().cpu().item()),
                "strict_true_grasp": bool(strict_true_grasp[env_id].detach().cpu().item()),
                "lifted": bool(lifted[env_id].detach().cpu().item()),
                "stable_hold": bool(stable[env_id].detach().cpu().item()),
                "success": bool(success[env_id].detach().cpu().item()),
                "object_minus_palm": [
                    float(value) for value in object_minus_palm[env_id].detach().cpu().tolist()
                ],
            }
        )


def _write_video_trace(
    video_path: Path,
    trace: list[dict],
    attempt: int,
    env_id: int,
    frame_count: int,
    pointcloud_video_path: Path | None = None,
) -> Path:
    lift_values = [float(item["object_height_delta"]) for item in trace]
    success_steps = [int(item["step"]) for item in trace if item.get("success")]
    lifted_steps = [int(item["step"]) for item in trace if item.get("lifted")]
    stable_steps = [int(item["step"]) for item in trace if item.get("stable_hold")]
    payload = {
        "video_path": str(video_path),
        "pointcloud_video_path": (
            str(pointcloud_video_path) if pointcloud_video_path is not None else None
        ),
        "pointcloud_visualization": str(args_cli.video_pointcloud_visualization),
        "pointcloud_coordinate_frame": "palm",
        "pointcloud_source": str(
            getattr(args_cli, "resolved_pointcloud_source", args_cli.pointcloud_source)
        ),
        "attempt": int(attempt),
        "env_id": int(env_id),
        "frame_count": int(frame_count),
        "camera_track_object": bool(args_cli.video_camera_track_object),
        "camera_track_offset": list(args_cli.video_camera_track_offset),
        "camera_track_target_offset": list(args_cli.video_camera_track_target_offset),
        "max_object_height_delta": max(lift_values) if lift_values else None,
        "final_object_height_delta": lift_values[-1] if lift_values else None,
        "first_lifted_step": lifted_steps[0] if lifted_steps else None,
        "first_stable_hold_step": stable_steps[0] if stable_steps else None,
        "first_success_step": success_steps[0] if success_steps else None,
        "success_steps": success_steps,
        "trace": trace,
    }
    trace_path = video_path.with_suffix(".trace.json")
    trace_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    if relative_positions and all(
        isinstance(position, list) and len(position) >= 3 for position in relative_positions
    ):
        origin = relative_positions[0]
        relative_drift = max(
            math.sqrt(
                sum((float(position[axis]) - float(origin[axis])) ** 2 for axis in range(3))
            )
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


def _save_success_videos(
    model: PointTemporalStudent,
    spec: dict,
    pointcloud_source: str,
    output_dir: Path,
    log_std: torch.Tensor | None,
) -> tuple[list[str], list[str]]:
    if args_cli.save_success_videos <= 0:
        return [], []

    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "video_progress.jsonl"
    video_paths: list[str] = []
    pointcloud_video_paths: list[str] = []
    video_envs = max(1, int(args_cli.video_envs))

    env = _make_env(args_cli.task, video_envs, "rgb_array", pointcloud_source)
    unwrapped = env.unwrapped
    device = unwrapped.device
    local_points, _ = sample_box_surface_points(
        tuple(float(v) for v in unwrapped.cfg.object_size),
        int(spec["num_object_points"]),
        device,
    )
    rgbd_mask_points = None
    if pointcloud_source == "rgbd_projected_mask":
        rgbd_mask_points = _make_rgbd_mask_point_template(
            unwrapped,
            max(int(args_cli.rgbd_mask_points), int(spec["num_object_points"])),
            device,
        )

    try:
        for attempt in range(int(args_cli.video_attempts)):
            if len(video_paths) >= int(args_cli.save_success_videos):
                break
            with progress_path.open("a", encoding="utf-8") as progress_file:
                progress_file.write(
                    json.dumps(
                        {
                            "event": "attempt_start",
                            "attempt": int(attempt),
                            "video_attempts": int(args_cli.video_attempts),
                            "saved_videos": int(len(video_paths)),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            _trace(
                f"video attempt {attempt:03d}/{int(args_cli.video_attempts):03d} "
                f"start saved={len(video_paths)}/{int(args_cli.save_success_videos)}"
            )
            obs, _ = env.reset(seed=args_cli.seed + attempt)
            _set_video_camera_poses(unwrapped)
            if pointcloud_source == "rgbd_projected_mask":
                _set_student_camera_poses(unwrapped)
            point_hist, valid_hist, proprio_hist = _make_histories(video_envs, spec, device)
            previous_action = torch.zeros((video_envs, int(spec["action_dim"])), device=device)
            hold_latch = torch.zeros(video_envs, dtype=torch.bool, device=device)
            arm_lock_age = torch.full((video_envs,), -1, dtype=torch.long, device=device)
            arm_lock_target = torch.zeros((video_envs, int(spec["arm_dim"])), device=device)
            history_initialized = torch.zeros(video_envs, dtype=torch.bool, device=device)
            pending_reset = torch.zeros(video_envs, dtype=torch.bool, device=device)
            frames_by_env: list[list[np.ndarray]] = [[] for _ in range(video_envs)]
            pointcloud_frames_by_env: list[list[np.ndarray]] = [
                [] for _ in range(video_envs)
            ]
            traces_by_env: list[list[dict]] = [[] for _ in range(video_envs)]
            saved_env_ids: set[int] = set()
            pending_success_steps: dict[int, int] = {}
            max_steps = int(args_cli.video_max_steps or unwrapped.max_episode_length)

            def save_video(env_id: int) -> None:
                frames = frames_by_env[env_id]
                if not _frames_have_signal(frames):
                    _trace(f"video attempt {attempt:03d} env {env_id:03d} succeeded but frames were blank")
                    saved_env_ids.add(env_id)
                    return
                video_path = video_dir / (
                    f"success_attempt_{attempt:03d}_env_{env_id:03d}_idx_{len(video_paths):03d}.mp4"
                )
                imageio.mimsave(video_path, frames, fps=args_cli.video_fps, macro_block_size=16)
                pointcloud_video_path = None
                if _pointcloud_separate_enabled():
                    pointcloud_frames = pointcloud_frames_by_env[env_id]
                    if len(pointcloud_frames) != len(frames):
                        raise RuntimeError(
                            "Point-cloud and scene video frame counts diverged: "
                            f"{len(pointcloud_frames)} != {len(frames)}"
                        )
                    pointcloud_video_path = video_path.with_name(
                        f"{video_path.stem}_pointcloud.mp4"
                    )
                    imageio.mimsave(
                        pointcloud_video_path,
                        pointcloud_frames,
                        fps=args_cli.video_fps,
                        macro_block_size=16,
                    )
                    pointcloud_video_paths.append(str(pointcloud_video_path))
                trace_path = _write_video_trace(
                    video_path,
                    traces_by_env[env_id],
                    attempt,
                    env_id,
                    len(frames),
                    pointcloud_video_path,
                )
                video_paths.append(str(video_path))
                saved_env_ids.add(env_id)
                _trace(f"saved success video: {video_path} (trace={trace_path})")

            with torch.inference_mode():
                for step in range(max_steps):
                    _reset_histories(point_hist, valid_hist, proprio_hist, pending_reset)
                    if bool(pending_reset.any()):
                        previous_action[pending_reset] = 0.0
                        hold_latch[pending_reset] = False
                        arm_lock_age[pending_reset] = -1
                        arm_lock_target[pending_reset] = 0.0
                    current_points, valid, _pc_info = _current_pointcloud(
                        unwrapped,
                        pointcloud_source,
                        local_points,
                        rgbd_mask_points,
                        int(spec.get("point_feature_dim", 3)),
                    )
                    if pointcloud_source == "rgbd_projected_mask":
                        _apply_rgbd_temporal_fallback(
                            current_points,
                            valid,
                            point_hist,
                            valid_hist,
                        )
                    if step % max(int(args_cli.video_stride), 1) == 0:
                        _capture_video_frames(
                            env,
                            frames_by_env,
                            current_points,
                            valid,
                            pointcloud_frames_by_env,
                        )
                    proprio = _current_proprio(unwrapped, obs)
                    if proprio.shape[-1] != int(spec["proprio_dim"]):
                        raise RuntimeError(
                            f"proprio_dim mismatch: source={args_cli.resolved_proprio_source} "
                            f"env={proprio.shape[-1]} checkpoint={spec['proprio_dim']}"
                        )
                    point_hist, valid_hist, proprio_hist = _roll_histories(
                        point_hist,
                        valid_hist,
                        proprio_hist,
                        current_points,
                        valid,
                        proprio,
                    )
                    if args_cli.history_bootstrap == "repeat_initial":
                        bootstrap = ~history_initialized
                        if bool(bootstrap.any()):
                            point_hist[bootstrap] = current_points[bootstrap].unsqueeze(1)
                            valid_hist[bootstrap] = valid[bootstrap].unsqueeze(1)
                            proprio_hist[bootstrap] = proprio[bootstrap].unsqueeze(1)
                    history_initialized[:] = True
                    actions, predicted_privileged, predicted_hold_target, predicted_hold_logits = _student_action(
                        model, point_hist, valid_hist, proprio_hist, log_std
                    )
                    actions = _apply_predicted_grasp_hold(
                        actions,
                        previous_action,
                        predicted_privileged,
                        predicted_hold_target,
                        predicted_hold_logits,
                        hold_latch,
                    )
                    actions = _apply_predicted_grasp_arm_lock(
                        actions,
                        hold_latch,
                        arm_lock_age,
                        arm_lock_target,
                        int(spec["arm_dim"]),
                    )
                    actions = _adapt_action(actions, previous_action)
                    obs, _rewards, terminated, truncated, extras = env.step(actions)
                    previous_action = actions.detach()
                    dones = terminated | truncated
                    _append_video_trace(env, traces_by_env, extras, step)
                    success_now = _tensor_bool(extras, "success_env", video_envs, device)
                    if bool(success_now.any()):
                        post_points, post_valid = _current_visualization_pointcloud(
                            unwrapped,
                            pointcloud_source,
                            local_points,
                            rgbd_mask_points,
                            int(spec.get("point_feature_dim", 3)),
                            point_hist,
                            valid_hist,
                        )
                        _capture_video_frames(
                            env,
                            frames_by_env,
                            post_points,
                            post_valid,
                            pointcloud_frames_by_env,
                        )
                        for env_tensor in success_now.nonzero(as_tuple=False).squeeze(-1):
                            env_id = int(env_tensor.item())
                            if env_id in saved_env_ids:
                                continue
                            if int(args_cli.video_post_success_steps) > 0:
                                pending_success_steps.setdefault(env_id, step)
                            else:
                                save_video(env_id)
                            if len(video_paths) >= int(args_cli.save_success_videos):
                                break
                    if pending_success_steps:
                        ready = [
                            env_id
                            for env_id, success_step in pending_success_steps.items()
                            if step - success_step >= int(args_cli.video_post_success_steps)
                        ]
                        for env_id in ready:
                            if env_id not in saved_env_ids and len(video_paths) < int(args_cli.save_success_videos):
                                save_video(env_id)
                            pending_success_steps.pop(env_id, None)
                    if len(video_paths) >= int(args_cli.save_success_videos):
                        break
                    pending_reset = dones
                    history_initialized[dones] = False

                final_points, final_valid = _current_visualization_pointcloud(
                    unwrapped,
                    pointcloud_source,
                    local_points,
                    rgbd_mask_points,
                    int(spec.get("point_feature_dim", 3)),
                    point_hist,
                    valid_hist,
                )
                _capture_video_frames(
                    env,
                    frames_by_env,
                    final_points,
                    final_valid,
                    pointcloud_frames_by_env,
                )
                for env_id in list(pending_success_steps):
                    if env_id not in saved_env_ids and len(video_paths) < int(args_cli.save_success_videos):
                        save_video(env_id)

            if not saved_env_ids:
                _trace(f"video attempt {attempt:03d} did not succeed")
                if args_cli.save_rollout_videos_on_failure > 0:
                    for env_id in range(min(int(args_cli.save_rollout_videos_on_failure), video_envs)):
                        frames = frames_by_env[env_id]
                        if not _frames_have_signal(frames):
                            continue
                        debug_path = video_dir / f"rollout_attempt_{attempt:03d}_env_{env_id:03d}.mp4"
                        imageio.mimsave(debug_path, frames, fps=args_cli.video_fps, macro_block_size=16)
                        debug_pointcloud_path = None
                        if _pointcloud_separate_enabled():
                            if len(pointcloud_frames_by_env[env_id]) != len(frames):
                                raise RuntimeError(
                                    "Point-cloud and debug scene frame counts diverged: "
                                    f"{len(pointcloud_frames_by_env[env_id])} != {len(frames)}"
                                )
                            debug_pointcloud_path = debug_path.with_name(
                                f"{debug_path.stem}_pointcloud.mp4"
                            )
                            imageio.mimsave(
                                debug_pointcloud_path,
                                pointcloud_frames_by_env[env_id],
                                fps=args_cli.video_fps,
                                macro_block_size=16,
                            )
                        trace_path = _write_video_trace(
                            debug_path,
                            traces_by_env[env_id],
                            attempt,
                            env_id,
                            len(frames),
                            debug_pointcloud_path,
                        )
                        _trace(f"saved debug rollout video: {debug_path} (trace={trace_path})")
            with progress_path.open("a", encoding="utf-8") as progress_file:
                progress_file.write(
                    json.dumps(
                        {
                            "event": "attempt_end",
                            "attempt": int(attempt),
                            "saved_env_ids": sorted(saved_env_ids),
                            "saved_videos": int(len(video_paths)),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            _trace(
                f"video attempt {attempt:03d}/{int(args_cli.video_attempts):03d} "
                f"end saved={len(video_paths)}/{int(args_cli.save_success_videos)}"
            )
    finally:
        env.close()
    return video_paths, pointcloud_video_paths


def _write_trial_sequence_trace(
    video_path: Path,
    trials: list[dict],
    frame_count: int,
    input_normalization: bool,
    pointcloud_video_path: Path | None = None,
) -> Path:
    success_count = sum(1 for trial in trials if bool(trial.get("success", False)))
    raw_success_count = sum(1 for trial in trials if bool(trial.get("raw_success", False)))
    post_success_hold_count = sum(
        1 for trial in trials if bool(trial.get("post_success_hold_success", False))
    )
    video_eye = args_cli.video_camera_eye or args_cli.student_camera_eye
    video_target = args_cli.video_camera_target or args_cli.student_camera_target
    payload = {
        "video_path": str(video_path),
        "pointcloud_video_path": (
            str(pointcloud_video_path) if pointcloud_video_path is not None else None
        ),
        "pointcloud_visualization": str(args_cli.video_pointcloud_visualization),
        "pointcloud_coordinate_frame": "palm",
        "pointcloud_source": str(
            getattr(args_cli, "resolved_pointcloud_source", args_cli.pointcloud_source)
        ),
        "pointcloud_panel_resolution": list(args_cli.video_pointcloud_panel_resolution),
        "pointcloud_range": float(args_cli.video_pointcloud_range),
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
        "history_bootstrap": str(args_cli.history_bootstrap),
        "input_normalization": bool(input_normalization),
        "trial_sequence_seed_mode": str(args_cli.trial_sequence_seed_mode),
        "camera_track_object": bool(args_cli.video_camera_track_object),
        "camera_eye": list(video_eye) if video_eye is not None else None,
        "camera_target": list(video_target) if video_target is not None else None,
        "camera_track_offset": list(args_cli.video_camera_track_offset),
        "camera_track_target_offset": list(args_cli.video_camera_track_target_offset),
        "camera_focal_length": float(args_cli.video_camera_focal_length),
        "camera_resolution": list(args_cli.video_camera_resolution),
        "student_camera_eye": (
            list(args_cli.student_camera_eye) if args_cli.student_camera_eye is not None else None
        ),
        "student_camera_target": (
            list(args_cli.student_camera_target) if args_cli.student_camera_target is not None else None
        ),
    }
    trace_path = video_path.with_suffix(".trace.json")
    trace_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return trace_path


def _save_trial_sequence_videos(
    model: PointTemporalStudent,
    spec: dict,
    pointcloud_source: str,
    output_dir: Path,
    log_std: torch.Tensor | None,
) -> tuple[list[str], list[str], list[str]]:
    if args_cli.save_trial_sequence_videos <= 0:
        return [], [], []

    sequence_paths: list[str] = []
    trace_paths: list[str] = []
    pointcloud_sequence_paths: list[str] = []
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    trial_count = max(1, int(args_cli.trial_sequence_trials))
    pre_roll_frames = max(0, int(args_cli.trial_sequence_pre_roll_frames))
    gap_frames = max(0, int(args_cli.trial_sequence_gap_frames))
    stride = max(1, int(args_cli.video_stride))
    post_success_steps = max(0, int(args_cli.video_post_success_steps))

    env = _make_env(args_cli.task, 1, "rgb_array", pointcloud_source)
    unwrapped = env.unwrapped
    device = unwrapped.device
    local_points, _ = sample_box_surface_points(
        tuple(float(v) for v in unwrapped.cfg.object_size),
        int(spec["num_object_points"]),
        device,
    )
    rgbd_mask_points = None
    if pointcloud_source == "rgbd_projected_mask":
        rgbd_mask_points = _make_rgbd_mask_point_template(
            unwrapped,
            max(int(args_cli.rgbd_mask_points), int(spec["num_object_points"])),
            device,
        )

    try:
        for sequence_idx in range(int(args_cli.save_trial_sequence_videos)):
            sequence_seed = int(args_cli.seed) + sequence_idx * 100_003
            np.random.seed(sequence_seed)
            torch.manual_seed(sequence_seed)
            if hasattr(unwrapped, "seed"):
                unwrapped.seed(sequence_seed)
            tmp_path = video_dir / f"trial_sequence_{sequence_idx:03d}_pending.mp4"
            writer = imageio.get_writer(tmp_path, fps=args_cli.video_fps, macro_block_size=16)
            pointcloud_tmp_path = video_dir / (
                f"trial_sequence_{sequence_idx:03d}_pointcloud_pending.mp4"
            )
            pointcloud_writer = (
                imageio.get_writer(
                    pointcloud_tmp_path,
                    fps=args_cli.video_fps,
                    macro_block_size=16,
                )
                if _pointcloud_separate_enabled()
                else None
            )
            trials: list[dict] = []
            frame_count = 0
            last_frame: np.ndarray | None = None
            last_pointcloud_frame: np.ndarray | None = None

            def append_frames(
                frames: list[np.ndarray],
                pointcloud_frames: list[np.ndarray] | None = None,
            ) -> None:
                nonlocal frame_count, last_frame, last_pointcloud_frame
                if pointcloud_writer is not None and len(pointcloud_frames or []) != len(frames):
                    raise RuntimeError(
                        "Point-cloud and scene sequence frame counts diverged: "
                        f"{len(pointcloud_frames or [])} != {len(frames)}"
                    )
                for frame in frames:
                    writer.append_data(frame)
                    frame_count += 1
                    last_frame = frame
                if pointcloud_writer is not None:
                    for pointcloud_frame in pointcloud_frames or []:
                        pointcloud_writer.append_data(pointcloud_frame)
                        last_pointcloud_frame = pointcloud_frame

            try:
                obs, _ = env.reset()
                _set_video_camera_poses(unwrapped)
                if pointcloud_source == "rgbd_projected_mask":
                    _set_student_camera_poses(unwrapped)
                for trial_idx in range(trial_count):
                    _trace(
                        f"trial-sequence {sequence_idx:03d} "
                        f"trial {trial_idx + 1:02d}/{trial_count} reset"
                    )
                    trial_seed: int | None = None
                    if args_cli.trial_sequence_seed_mode == "per_trial":
                        trial_seed = sequence_seed + trial_idx
                        if hasattr(unwrapped, "seed"):
                            unwrapped.seed(trial_seed)
                        if trial_idx > 0:
                            obs, _ = env.reset()
                            _set_video_camera_poses(unwrapped)
                            if pointcloud_source == "rgbd_projected_mask":
                                _set_student_camera_poses(unwrapped)
                    point_hist, valid_hist, proprio_hist = _make_histories(1, spec, device)
                    previous_action = torch.zeros((1, int(spec["action_dim"])), device=device)
                    hold_latch = torch.zeros(1, dtype=torch.bool, device=device)
                    arm_lock_age = torch.full((1,), -1, dtype=torch.long, device=device)
                    arm_lock_target = torch.zeros((1, int(spec["arm_dim"])), device=device)
                    frames_by_env: list[list[np.ndarray]] = [[]]
                    pointcloud_frames_by_env: list[list[np.ndarray]] = [[]]
                    traces_by_env: list[list[dict]] = [[]]
                    first_success_step: int | None = None
                    post_success_detail = ""
                    recording_complete = False
                    trial_ended_with_done = False
                    max_steps = int(args_cli.video_max_steps or unwrapped.max_episode_length)

                    # Pre-roll precedes the student's first policy observation. Start
                    # point-cloud recording at policy step 0 so the video shows only
                    # observations that were actually consumed by the student.
                    preview_points, preview_valid = None, None
                    for _ in range(pre_roll_frames):
                        _capture_video_frames(
                            env,
                            frames_by_env,
                            preview_points,
                            preview_valid,
                            pointcloud_frames_by_env,
                        )

                    with torch.inference_mode():
                        for step in range(max_steps):
                            if step > 0 and step % 100 == 0:
                                _trace(
                                    f"trial-sequence {sequence_idx:03d} "
                                    f"trial {trial_idx + 1:02d}/{trial_count} "
                                    f"step {step:04d}/{max_steps:04d}"
                                )
                            if recording_complete:
                                # Let DirectRLEnv reach its real timeout/reset boundary without
                                # paying for another RGB-D projection and policy forward pass.
                                # Mutating episode_length_buf or calling reset mid-episode can
                                # terminate a singleton rendered IsaacLab app.
                                obs, _rewards, terminated, truncated, _extras = env.step(actions)
                                trial_ended_with_done = bool((terminated | truncated)[0].item())
                                if trial_ended_with_done:
                                    break
                                continue
                            current_points, valid, _pc_info = _current_pointcloud(
                                unwrapped,
                                pointcloud_source,
                                local_points,
                                rgbd_mask_points,
                                int(spec.get("point_feature_dim", 3)),
                            )
                            if pointcloud_source == "rgbd_projected_mask":
                                _apply_rgbd_temporal_fallback(
                                    current_points,
                                    valid,
                                    point_hist,
                                    valid_hist,
                                )
                            if step % stride == 0:
                                _capture_video_frames(
                                    env,
                                    frames_by_env,
                                    current_points,
                                    valid,
                                    pointcloud_frames_by_env,
                                )
                            proprio = _current_proprio(unwrapped, obs)
                            if proprio.shape[-1] != int(spec["proprio_dim"]):
                                raise RuntimeError(
                                    "proprio_dim mismatch: "
                                    f"source={args_cli.resolved_proprio_source} "
                                    f"env={proprio.shape[-1]} checkpoint={spec['proprio_dim']}"
                                )
                            point_hist, valid_hist, proprio_hist = _roll_histories(
                                point_hist,
                                valid_hist,
                                proprio_hist,
                                current_points,
                                valid,
                                proprio,
                            )
                            if args_cli.history_bootstrap == "repeat_initial" and step == 0:
                                point_hist[:] = current_points.unsqueeze(1)
                                valid_hist[:] = valid.unsqueeze(1)
                                proprio_hist[:] = proprio.unsqueeze(1)
                            actions, predicted_privileged, predicted_hold_target, predicted_hold_logits = (
                                _student_action(model, point_hist, valid_hist, proprio_hist, log_std)
                            )
                            actions = _apply_predicted_grasp_hold(
                                actions,
                                previous_action,
                                predicted_privileged,
                                predicted_hold_target,
                                predicted_hold_logits,
                                hold_latch,
                            )
                            actions = _apply_predicted_grasp_arm_lock(
                                actions,
                                hold_latch,
                                arm_lock_age,
                                arm_lock_target,
                                int(spec["arm_dim"]),
                            )
                            actions = _adapt_action(actions, previous_action)
                            obs, _rewards, terminated, truncated, extras = env.step(actions)
                            previous_action = actions.detach()
                            done = terminated | truncated
                            trial_ended_with_done = bool(done[0].item())
                            if not recording_complete:
                                _append_video_trace(env, traces_by_env, extras, step)
                                success_now = _tensor_bool(extras, "success_env", 1, device)
                                if bool(success_now[0].item()) and first_success_step is None:
                                    first_success_step = int(step)
                                if (
                                    first_success_step is not None
                                    and step - first_success_step >= post_success_steps
                                ):
                                    post_points, post_valid = _current_visualization_pointcloud(
                                        unwrapped,
                                        pointcloud_source,
                                        local_points,
                                        rgbd_mask_points,
                                        int(spec.get("point_feature_dim", 3)),
                                        point_hist,
                                        valid_hist,
                                    )
                                    _capture_video_frames(
                                        env,
                                        frames_by_env,
                                        post_points,
                                        post_valid,
                                        pointcloud_frames_by_env,
                                    )
                                    recording_complete = True
                            if trial_ended_with_done:
                                break
                    if not recording_complete:
                        with torch.inference_mode():
                            final_points, final_valid = _current_visualization_pointcloud(
                                unwrapped,
                                pointcloud_source,
                                local_points,
                                rgbd_mask_points,
                                int(spec.get("point_feature_dim", 3)),
                                point_hist,
                                valid_hist,
                            )
                        _capture_video_frames(
                            env,
                            frames_by_env,
                            final_points,
                            final_valid,
                            pointcloud_frames_by_env,
                        )

                    trace = traces_by_env[0]
                    raw_success = first_success_step is not None
                    post_success_hold_success = bool(raw_success)
                    if raw_success and bool(args_cli.video_require_post_success_stable):
                        post_success_hold_success, post_success_detail = _post_success_trace_passes(
                            trace,
                            int(first_success_step),
                            float(args_cli.video_post_success_min_stable_frac),
                            bool(args_cli.video_post_success_require_final_stable),
                            post_success_steps,
                            _post_success_requires_lifted(
                                args_cli.task, args_cli.video_post_success_lift_mode
                            ),
                            float(args_cli.video_post_success_max_relative_drift),
                        )
                    success = bool(raw_success) and (
                        bool(post_success_hold_success)
                        if bool(args_cli.video_require_post_success_stable)
                        else True
                    )
                    frames = frames_by_env[0]
                    pointcloud_frames = pointcloud_frames_by_env[0]
                    visual_ok = _frames_have_signal(frames)
                    if visual_ok:
                        append_frames(frames, pointcloud_frames)
                    elif last_frame is not None:
                        fallback_pointcloud_frames = (
                            [last_pointcloud_frame.copy()]
                            if pointcloud_writer is not None
                            and last_pointcloud_frame is not None
                            else None
                        )
                        append_frames([last_frame.copy()], fallback_pointcloud_frames)
                    if gap_frames > 0 and last_frame is not None and trial_idx < trial_count - 1:
                        gap_frame = np.zeros_like(last_frame)
                        gap_pointcloud_frames = None
                        if pointcloud_writer is not None and last_pointcloud_frame is not None:
                            gap_pointcloud_frame = np.zeros_like(last_pointcloud_frame)
                            gap_pointcloud_frames = [
                                gap_pointcloud_frame.copy() for _ in range(gap_frames)
                            ]
                        append_frames(
                            [gap_frame.copy() for _ in range(gap_frames)],
                            gap_pointcloud_frames,
                        )

                    trials.append(
                        {
                            "trial": int(trial_idx),
                            "sequence_seed": int(sequence_seed),
                            "seed_mode": str(args_cli.trial_sequence_seed_mode),
                            "seed": int(trial_seed) if trial_seed is not None else None,
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
                        f"trial-sequence {sequence_idx:03d} "
                        f"trial {trial_idx + 1:02d}/{trial_count} "
                        f"success={int(success)} raw_success={int(raw_success)} "
                        f"post_hold={int(post_success_hold_success)} frames={len(frames)}"
                    )
                    if (
                        args_cli.trial_sequence_seed_mode == "sequence"
                        and trial_idx < trial_count - 1
                        and not trial_ended_with_done
                    ):
                        raise RuntimeError(
                            "A sequence trial reached --video-max-steps before the environment "
                            "reset. Use a video horizon at least as long as max_episode_length."
                        )
            finally:
                writer.close()
                if pointcloud_writer is not None:
                    pointcloud_writer.close()

            success_count = sum(1 for trial in trials if bool(trial.get("success", False)))
            final_path = video_dir / (
                f"trial_sequence_{sequence_idx:03d}_trials_{len(trials):03d}_"
                f"success_{success_count:03d}_sr_{success_count / max(len(trials), 1):.3f}.mp4"
            )
            if frame_count <= 0:
                tmp_path.unlink(missing_ok=True)
                pointcloud_tmp_path.unlink(missing_ok=True)
                _trace(f"trial-sequence {sequence_idx:03d} produced no frames")
                continue
            tmp_path.replace(final_path)
            pointcloud_final_path = None
            if pointcloud_writer is not None:
                pointcloud_final_path = final_path.with_name(
                    f"{final_path.stem}_pointcloud.mp4"
                )
                pointcloud_tmp_path.replace(pointcloud_final_path)
                pointcloud_sequence_paths.append(str(pointcloud_final_path))
            trace_path = _write_trial_sequence_trace(
                final_path,
                trials,
                frame_count,
                bool(getattr(model, "input_normalization", None) is not None),
                pointcloud_final_path,
            )
            sequence_paths.append(str(final_path))
            trace_paths.append(str(trace_path))
            _trace(f"saved trial sequence video: {final_path} (trace={trace_path})")
    finally:
        env.close()

    return sequence_paths, trace_paths, pointcloud_sequence_paths


def _maybe_init_wandb(checkpoint: Path, metadata: dict, spec: dict, output_dir: Path):
    if not args_cli.wandb_project or args_cli.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=args_cli.wandb_project,
        entity=args_cli.wandb_entity,
        name=args_cli.wandb_run_name or f"student_eval_{args_cli.task}",
        group=args_cli.wandb_group,
        tags=args_cli.wandb_tags,
        mode=args_cli.wandb_mode,
        dir=str(output_dir),
        job_type="student_eval",
        config={
            "task": args_cli.task,
            "checkpoint": str(checkpoint),
            "seed": int(args_cli.seed),
            "num_envs": int(args_cli.num_envs),
            "episodes": int(args_cli.episodes),
            "max_steps": args_cli.max_steps,
            "first_episode_per_env": bool(args_cli.first_episode_per_env),
            "pointcloud_source": args_cli.pointcloud_source,
            "proprio_source": getattr(args_cli, "resolved_proprio_source", args_cli.proprio_source),
            "dynamic_curriculum_alpha": args_cli.dynamic_curriculum_alpha,
            "tabletop_asset_curriculum_alpha": args_cli.tabletop_asset_curriculum_alpha,
            "tabletop_motion_curriculum_alpha": args_cli.tabletop_motion_curriculum_alpha,
            "episode_length_s": args_cli.episode_length_s,
            "dynamic_success_hold_steps": args_cli.dynamic_success_hold_steps,
            "stability_target_latch_min_success_streak": (
                args_cli.stability_target_latch_min_success_streak
            ),
            "student_action_mode": args_cli.student_action_mode,
            "student_action_std_scale": float(args_cli.student_action_std_scale),
            "trial_sequence_seed_mode": args_cli.trial_sequence_seed_mode,
            "student_camera_eye": args_cli.student_camera_eye,
            "student_camera_target": args_cli.student_camera_target,
            "video_pointcloud_visualization": args_cli.video_pointcloud_visualization,
            "video_pointcloud_panel_resolution": args_cli.video_pointcloud_panel_resolution,
            "video_pointcloud_range": float(args_cli.video_pointcloud_range),
            "metadata": metadata,
            "spec": spec,
        },
    )


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    torch.manual_seed(args_cli.seed)
    student_device_name = args_cli.student_device or args_cli.device
    if student_device_name.startswith("cuda") and not torch.cuda.is_available():
        student_device_name = "cpu"
    student_device = torch.device(student_device_name)
    model, metadata, spec, log_std = _load_student(checkpoint, student_device)
    pointcloud_source = _resolve_pointcloud_source(metadata)
    if pointcloud_source not in ("clean", "rgbd_projected_mask"):
        raise RuntimeError(f"Unsupported pointcloud source in metadata: {pointcloud_source!r}")
    args_cli.resolved_pointcloud_source = pointcloud_source
    args_cli.resolved_proprio_source = _resolve_proprio_source(metadata)
    if args_cli.resolved_proprio_source not in ("policy", "deployable_robot"):
        raise RuntimeError(
            f"Unsupported proprio source in metadata: {args_cli.resolved_proprio_source!r}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args_cli.output_dir).expanduser().resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _maybe_init_wandb(checkpoint, metadata, spec, output_dir)

    _trace(
        f"task={args_cli.task} checkpoint={checkpoint} pointcloud_source={pointcloud_source} "
        f"proprio_source={args_cli.resolved_proprio_source} "
        f"spec={spec}"
    )
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
        eval_summary, _ = _run_student_eval(
            model,
            spec,
            pointcloud_source,
            int(args_cli.num_envs),
            int(args_cli.episodes),
            log_std,
            render_mode=None,
        )
        vector_passed = eval_summary["success_rate"] >= float(args_cli.success_threshold)
    video_paths, success_pointcloud_video_paths = _save_success_videos(
        model, spec, pointcloud_source, output_dir, log_std
    )
    trace_paths = [
        str(Path(path).with_suffix(".trace.json"))
        for path in video_paths
        if Path(path).with_suffix(".trace.json").exists()
    ]
    (
        trial_sequence_paths,
        trial_sequence_trace_paths,
        trial_sequence_pointcloud_paths,
    ) = _save_trial_sequence_videos(model, spec, pointcloud_source, output_dir, log_std)
    trial_sequence_results = []
    for trace_path in trial_sequence_trace_paths:
        trace_payload = json.loads(Path(trace_path).read_text(encoding="utf-8"))
        trial_sequence_results.append(
            {
                "video": trace_payload.get("video_path"),
                "pointcloud_video": trace_payload.get("pointcloud_video_path"),
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
    trial_sequence_requested = int(args_cli.save_trial_sequence_videos) > 0
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
    summary = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "seed": int(args_cli.seed),
        "num_envs": int(args_cli.num_envs),
        "episodes_requested": int(args_cli.episodes),
        "max_steps": args_cli.max_steps,
        "student_device": str(student_device),
        "device": args_cli.device,
        "pointcloud_source": pointcloud_source,
        "proprio_source": args_cli.resolved_proprio_source,
        "metadata": metadata,
        "spec": spec,
        "action_clamp": float(args_cli.action_clamp),
        "student_action_mode": str(args_cli.student_action_mode),
        "student_action_std_scale": float(args_cli.student_action_std_scale),
        "student_action_std_mean": (
            float(log_std.exp().mean().detach().cpu()) if log_std is not None else None
        ),
        "action_ema_alpha": float(args_cli.action_ema_alpha),
        "action_rate_limit": float(args_cli.action_rate_limit),
        "history_bootstrap": str(args_cli.history_bootstrap),
        "first_episode_per_env": bool(args_cli.first_episode_per_env),
        "object_contact_force_diagnostics": bool(args_cli.object_contact_force_diagnostics),
        "object_contact_force_threshold": float(args_cli.object_contact_force_threshold),
        "rgbd_clean_fallback": bool(args_cli.rgbd_clean_fallback),
        "rgbd_temporal_fallback": bool(args_cli.rgbd_temporal_fallback),
        "trial_sequence_seed_mode": str(args_cli.trial_sequence_seed_mode),
        "video_pointcloud_visualization": str(args_cli.video_pointcloud_visualization),
        "video_pointcloud_panel_resolution": [
            int(value) for value in args_cli.video_pointcloud_panel_resolution
        ],
        "video_pointcloud_range": float(args_cli.video_pointcloud_range),
        "video_pointcloud_point_radius": int(args_cli.video_pointcloud_point_radius),
        "video_pointcloud_inset_margin": int(args_cli.video_pointcloud_inset_margin),
        "predicted_grasp_hold_threshold": float(args_cli.predicted_grasp_hold_threshold),
        "predicted_grasp_hold_mode": args_cli.predicted_grasp_hold_mode,
        "predicted_grasp_hold_target": (
            [float(value) for value in args_cli.predicted_grasp_hold_target]
            if args_cli.predicted_grasp_hold_target is not None
            else None
        ),
        "predicted_grasp_hold_blend": float(args_cli.predicted_grasp_hold_blend),
        "predicted_grasp_hold_rel_vel_threshold": float(args_cli.predicted_grasp_hold_rel_vel_threshold),
        "predicted_grasp_hold_learned_gate_threshold": float(args_cli.predicted_grasp_hold_learned_gate_threshold),
        "predicted_grasp_hold_learned_target_clamp": float(args_cli.predicted_grasp_hold_learned_target_clamp),
        "predicted_grasp_hold_learned_residual_scale": float(
            args_cli.predicted_grasp_hold_learned_residual_scale
        ),
        "predicted_grasp_hold_learned_residual_clamp": float(
            args_cli.predicted_grasp_hold_learned_residual_clamp
        ),
        "predicted_grasp_arm_lock_delay_steps": int(
            args_cli.predicted_grasp_arm_lock_delay_steps
        ),
        "predicted_grasp_arm_lock_blend": float(args_cli.predicted_grasp_arm_lock_blend),
        "success_threshold": float(args_cli.success_threshold),
        "dynamic_curriculum_alpha": args_cli.dynamic_curriculum_alpha,
        "tabletop_asset_curriculum_alpha": args_cli.tabletop_asset_curriculum_alpha,
        "tabletop_motion_curriculum_alpha": args_cli.tabletop_motion_curriculum_alpha,
        "stability_target_latch_min_success_streak": (
            args_cli.stability_target_latch_min_success_streak
        ),
        "passed": bool(passed),
        "vector_passed": bool(vector_passed),
        "trial_sequence_passed": bool(trial_sequence_passed),
        "eval": eval_summary,
        "success_videos": video_paths,
        "success_pointcloud_videos": success_pointcloud_video_paths,
        "success_video_traces": trace_paths,
        "trial_sequence_videos": trial_sequence_paths,
        "trial_sequence_pointcloud_videos": trial_sequence_pointcloud_paths,
        "trial_sequence_traces": trial_sequence_trace_paths,
        "trial_sequence_results": trial_sequence_results,
        "video_post_success_steps": int(args_cli.video_post_success_steps),
        "video_require_post_success_stable": bool(args_cli.video_require_post_success_stable),
        "video_post_success_min_stable_frac": float(args_cli.video_post_success_min_stable_frac),
        "video_post_success_max_relative_drift": float(
            args_cli.video_post_success_max_relative_drift
        ),
        "video_post_success_require_final_stable": bool(
            args_cli.video_post_success_require_final_stable
        ),
        "video_post_success_lift_mode": args_cli.video_post_success_lift_mode,
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if wandb_run is not None:
        import wandb

        scalar_metrics = flatten_numeric_metrics(eval_summary, "eval")
        for trace_path in trial_sequence_trace_paths:
            trace_payload = json.loads(Path(trace_path).read_text(encoding="utf-8"))
            scalar_metrics["video/trial_success_rate"] = float(trace_payload["success_rate"])
            scalar_metrics["video/raw_success_rate"] = float(trace_payload["raw_success_rate"])
            scalar_metrics["video/post_success_hold_rate"] = float(
                trace_payload["post_success_hold_rate"]
            )
        scalar_metrics["eval/passed"] = int(bool(passed))
        if args_cli.wandb_log_videos:
            for index, video_path in enumerate(trial_sequence_paths):
                scalar_metrics[f"video/trial_sequence_{index:03d}"] = wandb.Video(
                    video_path, fps=int(args_cli.video_fps), format="mp4"
                )
            for index, video_path in enumerate(trial_sequence_pointcloud_paths):
                scalar_metrics[f"video/trial_sequence_pointcloud_{index:03d}"] = wandb.Video(
                    video_path, fps=int(args_cli.video_fps), format="mp4"
                )
        wandb.log(scalar_metrics)
        wandb_run.summary.update({"summary_path": str(summary_path), "passed": bool(passed)})
        wandb_run.finish()
    _trace(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        failures = []
        if not vector_passed:
            failures.append(
                f"vector success_rate={float(eval_summary['success_rate']):.3f} "
                f"below threshold={float(args_cli.success_threshold):.3f}"
            )
        if not trial_sequence_passed:
            failures.append(
                f"trial sequence did not meet threshold={float(args_cli.success_threshold):.3f}"
            )
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import traceback

        _trace(f"fatal evaluation exit: {type(exc).__name__}: {exc!r}")
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
