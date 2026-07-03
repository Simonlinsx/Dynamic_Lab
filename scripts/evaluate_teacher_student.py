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
parser.add_argument("--success-threshold", type=float, default=0.0)
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
parser.add_argument("--history", type=int, default=None, help="Override checkpoint history.")
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
    "--student-camera-track-object",
    action="store_true",
    default=False,
    help="Aim the student RGB-D camera at the live object. By default the camera is fixed in a stable third view.",
)
parser.add_argument("--student-camera-track-offset", type=float, nargs=3, default=(0.65, 0.95, 0.65))
parser.add_argument("--student-camera-track-target-offset", type=float, nargs=3, default=(0.0, 0.0, 0.05))
parser.add_argument("--student-camera-focal-length", type=float, default=24.0)
parser.add_argument("--student-device", default=None, help="Torch device for student inference. Defaults to --device.")
parser.add_argument("--action-clamp", type=float, default=1.0)
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
    "--dynamic-curriculum-alpha",
    type=float,
    default=None,
    help="Override dynamic speed curriculum alpha for eval, when the env cfg exposes it.",
)
parser.add_argument(
    "--output-dir",
    default="outputs/eval_teacher_student",
    help="Directory for JSON summaries and videos.",
)
parser.add_argument("--save-success-videos", type=int, default=0)
parser.add_argument("--video-attempts", type=int, default=20)
parser.add_argument("--video-envs", type=int, default=4)
parser.add_argument("--video-stride", type=int, default=2)
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--video-max-steps", type=int, default=None)
parser.add_argument("--video-post-success-steps", type=int, default=60)
parser.add_argument("--save-rollout-videos-on-failure", type=int, default=0)
parser.add_argument(
    "--video-camera-track-object",
    action="store_true",
    default=False,
    help="Aim the video camera at the live object. By default it shares the fixed third-view student camera pose.",
)
parser.add_argument("--video-camera-track-offset", type=float, nargs=3, default=(0.65, 0.95, 0.65))
parser.add_argument("--video-camera-track-target-offset", type=float, nargs=3, default=(0.0, 0.0, 0.05))
parser.add_argument("--video-camera-focal-length", type=float, default=24.0)
parser.add_argument("--video-camera-resolution", type=int, nargs=2, default=(960, 544))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not (0.0 < float(args_cli.action_ema_alpha) <= 1.0):
    raise ValueError("--action-ema-alpha must be in (0, 1].")
if float(args_cli.action_rate_limit) < 0.0:
    raise ValueError("--action-rate-limit must be non-negative.")
if not (0.0 <= float(args_cli.predicted_grasp_hold_blend) <= 1.0):
    raise ValueError("--predicted-grasp-hold-blend must be in [0, 1].")
if (
    float(args_cli.predicted_grasp_hold_threshold) > 0.0
    and args_cli.predicted_grasp_hold_mode in {"target", "target_plus_learned_residual"}
    and not args_cli.predicted_grasp_hold_target
):
    raise ValueError(
        "--predicted-grasp-hold-target is required when "
        "--predicted-grasp-hold-mode is target or target_plus_learned_residual."
    )

if args_cli.pointcloud_source != "clean" or args_cli.save_success_videos > 0:
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
    masked_rgbd_object_points_in_palm_frame,
    object_points_in_palm_frame,
    object_points_in_world_frame,
    sample_box_surface_points,
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


def _load_student(checkpoint_path: Path, device: torch.device) -> tuple[PointTemporalStudent, dict, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = dict(payload.get("metadata", {}))
        spec = dict(payload.get("spec", {}))
    else:
        state_dict = payload
        metadata = {}
        spec = {}

    history = int(args_cli.history or spec.get("history", metadata.get("history", 4)))
    num_points = int(args_cli.object_points or spec.get("num_object_points", metadata.get("object_points", 128)))
    proprio_dim = int(spec.get("proprio_dim", metadata.get("proprio_dim", 76)))
    action_dim = int(spec.get("action_dim", metadata.get("action_dim", 0)))
    arm_dim = int(spec.get("arm_dim", metadata.get("arm_dim", 7)))
    privileged_dim = int(spec.get("compact_privileged_dim", metadata.get("compact_privileged_dim", 32)))
    if action_dim <= 0:
        raise RuntimeError("Student checkpoint does not contain a positive action_dim in spec/metadata.")

    model = PointTemporalStudent(
        history=history,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        privileged_dim=privileged_dim,
        arm_dim=arm_dim,
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
    model.eval()

    resolved_spec = {
        "history": history,
        "num_object_points": num_points,
        "proprio_dim": proprio_dim,
        "action_dim": action_dim,
        "arm_dim": arm_dim,
        "hand_dim": max(action_dim - arm_dim, 0),
        "compact_privileged_dim": privileged_dim,
    }
    return model, metadata, resolved_spec


def _resolve_pointcloud_source(metadata: dict) -> str:
    if args_cli.pointcloud_source != "auto":
        return args_cli.pointcloud_source
    return str(metadata.get("pointcloud_source", "clean"))


def _make_env(task: str, num_envs: int, render_mode: str | None, pointcloud_source: str):
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=num_envs)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.dynamic_curriculum_alpha is not None and hasattr(
        env_cfg, "dynamic_grasp_speed_curriculum_override_alpha"
    ):
        env_cfg.dynamic_grasp_speed_curriculum_override_alpha = float(args_cli.dynamic_curriculum_alpha)

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
) -> bool:
    camera = getattr(unwrapped_env, camera_name, None)
    if camera is None:
        return False
    origins = unwrapped_env.scene.env_origins
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
    )


def _set_video_camera_poses(unwrapped_env) -> bool:
    return _set_tracking_camera(
        unwrapped_env,
        "_video_camera",
        tuple(float(v) for v in args_cli.video_camera_track_offset),
        tuple(float(v) for v in args_cli.video_camera_track_target_offset),
        track_object=bool(args_cli.video_camera_track_object),
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


def _current_pointcloud(
    unwrapped_env,
    pointcloud_source: str,
    local_points: torch.Tensor,
    affordance_points: torch.Tensor | None,
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
        return clean_points, valid, {"rgbd_valid_mean": float(valid.shape[-1]), "rgbd_fallback_envs": 0.0}

    assert affordance_points is not None
    clean_points_w = object_points_in_world_frame(
        local_points,
        unwrapped_env._object_pos_w,
        unwrapped_env._object_quat_w,
    )
    _set_student_camera_poses(unwrapped_env)
    _rgb, depth, camera_pos_w, camera_quat_w_ros, camera_intrinsics = _student_camera_rgbd(unwrapped_env)
    rgbd_points = masked_rgbd_object_points_in_palm_frame(
        affordance_points,
        tuple(float(v) for v in unwrapped_env.cfg.object_size),
        depth,
        camera_pos_w,
        camera_quat_w_ros,
        camera_intrinsics,
        unwrapped_env._object_pos_w,
        unwrapped_env._object_quat_w,
        unwrapped_env._palm_pos_w,
        unwrapped_env._palm_quat_w,
        num_points=int(local_points.shape[0]),
        mask_dilation=args_cli.rgbd_mask_dilation,
        depth_tolerance=args_cli.rgbd_depth_tolerance,
    )
    points = rgbd_points["points_palm"]
    valid = rgbd_points["valid"]
    valid_counts = torch.sum(valid > 0.0, dim=-1)
    low_valid = valid_counts < int(args_cli.rgbd_min_valid_points)
    fallback_count = int(low_valid.sum().detach().cpu())
    if fallback_count > 0 and bool(args_cli.rgbd_clean_fallback):
        points[low_valid] = clean_points[low_valid]
        valid[low_valid] = 1.0
        clean_points_w = clean_points_w  # keep the explicit clean fallback visible for parity with collection
    return points, valid, {
        "rgbd_valid_mean": float(valid_counts.float().mean().detach().cpu()),
        "rgbd_fallback_envs": float(fallback_count),
    }


def _student_action(
    model: PointTemporalStudent,
    point_hist: torch.Tensor,
    valid_hist: torch.Tensor,
    proprio_hist: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(point_hist, valid_hist, proprio_hist)
    action = out["action"]
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
        torch.zeros((num_envs, spec["history"], spec["num_object_points"], 3), device=device),
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


def _run_student_eval(
    model: PointTemporalStudent,
    spec: dict,
    pointcloud_source: str,
    num_envs: int,
    episodes: int,
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
        rgbd_mask_points, _ = sample_box_surface_points(
            tuple(float(v) for v in unwrapped.cfg.object_size),
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
    pending_reset = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_true_grasp = torch.zeros_like(episode_success)
    episode_lifted = torch.zeros_like(episode_success)
    episode_stable = torch.zeros_like(episode_success)
    completed = 0
    success_count = 0
    true_count = 0
    lifted_count = 0
    stable_count = 0
    total_reward = 0.0
    total_reward_count = 0
    last_log = {}
    rgbd_valid_sum = 0.0
    rgbd_fallback_sum = 0.0
    rgbd_steps = 0
    max_episode_length = int(unwrapped.max_episode_length)
    max_steps = args_cli.max_steps or (max_episode_length * (math.ceil(episodes / num_envs) + 2))

    with torch.inference_mode():
        for step in range(max_steps):
            _reset_histories(point_hist, valid_hist, proprio_hist, pending_reset)
            if bool(pending_reset.any()):
                previous_action[pending_reset] = 0.0
                hold_latch[pending_reset] = False
            current_points, valid, pc_info = _current_pointcloud(unwrapped, pointcloud_source, local_points, rgbd_mask_points)
            policy_obs = _policy_tensor(obs).to(device)
            if policy_obs.shape[-1] != int(spec["proprio_dim"]):
                raise RuntimeError(
                    f"proprio_dim mismatch: env obs={policy_obs.shape[-1]} checkpoint={spec['proprio_dim']}"
                )
            point_hist, valid_hist, proprio_hist = _roll_histories(
                point_hist,
                valid_hist,
                proprio_hist,
                current_points,
                valid,
                policy_obs,
            )
            actions, predicted_privileged, predicted_hold_target, predicted_hold_logits = _student_action(
                model, point_hist, valid_hist, proprio_hist
            )
            actions = _apply_predicted_grasp_hold(
                actions,
                previous_action,
                predicted_privileged,
                predicted_hold_target,
                predicted_hold_logits,
                hold_latch,
            )
            actions = _adapt_action(actions, previous_action)
            obs, rewards, terminated, truncated, extras = env.step(actions)
            previous_action = actions.detach()
            dones = terminated | truncated

            episode_success |= _tensor_bool(extras, "success_env", num_envs, device)
            episode_true_grasp |= _tensor_bool(extras, "true_grasp_env", num_envs, device)
            episode_lifted |= _tensor_bool(extras, "lifted_env", num_envs, device)
            episode_stable |= _tensor_bool(extras, "stable_hold_env", num_envs, device)
            total_reward += float(rewards.sum().item())
            total_reward_count += int(rewards.numel())
            if pointcloud_source == "rgbd_projected_mask":
                rgbd_valid_sum += pc_info["rgbd_valid_mean"]
                rgbd_fallback_sum += pc_info["rgbd_fallback_envs"]
                rgbd_steps += 1
            if "episode" in extras:
                last_log = {key: float(value.item()) for key, value in extras["episode"].items() if hasattr(value, "item")}

            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                remaining = max(int(episodes) - completed, 0)
                if remaining <= 0:
                    break
                if done_ids.numel() > remaining:
                    done_ids = done_ids[:remaining]
                completed += int(done_ids.numel())
                success_count += int(episode_success[done_ids].sum().item())
                true_count += int(episode_true_grasp[done_ids].sum().item())
                lifted_count += int(episode_lifted[done_ids].sum().item())
                stable_count += int(episode_stable[done_ids].sum().item())
                episode_success[done_ids] = False
                episode_true_grasp[done_ids] = False
                episode_lifted[done_ids] = False
                episode_stable[done_ids] = False
                _trace(f"episodes={completed}/{episodes} success_rate={success_count / max(completed, 1):.3f}")
                if completed >= episodes:
                    break
            pending_reset = dones
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
        "mean_step_reward": float(total_reward / max(total_reward_count, 1)),
        "max_episode_length": max_episode_length,
        "last_log": last_log,
        "rgbd_valid_mean": float(rgbd_valid_sum / max(rgbd_steps, 1)) if pointcloud_source == "rgbd_projected_mask" else None,
        "rgbd_fallback_envs_per_step": float(rgbd_fallback_sum / max(rgbd_steps, 1)) if pointcloud_source == "rgbd_projected_mask" else None,
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


def _capture_video_frames(env, frames_by_env: list[list[np.ndarray]]) -> None:
    unwrapped = env.unwrapped
    camera = getattr(unwrapped, "_video_camera", None)
    if camera is None:
        frame = unwrapped.render(recompute=True)
        if frame is not None and frame.size > 0:
            for frames in frames_by_env:
                frames.append(frame.copy())
        return
    _set_video_camera_poses(unwrapped)
    _force_camera_update(camera, float(unwrapped.dt))
    rgb = camera.data.output.get("rgb")
    if rgb is None or rgb.numel() == 0:
        return
    rgb = rgb[..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
    for env_id, frames in enumerate(frames_by_env):
        if env_id < rgb.shape[0] and rgb[env_id].size > 0:
            frames.append(rgb[env_id].copy())


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
    lifted = _tensor_bool(extras, "lifted_env", num_envs, device)
    stable = _tensor_bool(extras, "stable_hold_env", num_envs, device)
    success = _tensor_bool(extras, "success_env", num_envs, device)

    for env_id, trace in enumerate(traces_by_env):
        trace.append(
            {
                "step": int(step),
                "object_height_delta": float(object_height_delta[env_id].detach().cpu().item()),
                "object_palm_rel_vel": float(object_palm_rel_vel[env_id].detach().cpu().item()),
                "success_streak": int(success_streak[env_id].detach().cpu().item()),
                "true_grasp": bool(true_grasp[env_id].detach().cpu().item()),
                "lifted": bool(lifted[env_id].detach().cpu().item()),
                "stable_hold": bool(stable[env_id].detach().cpu().item()),
                "success": bool(success[env_id].detach().cpu().item()),
            }
        )


def _write_video_trace(video_path: Path, trace: list[dict], attempt: int, env_id: int, frame_count: int) -> Path:
    lift_values = [float(item["object_height_delta"]) for item in trace]
    success_steps = [int(item["step"]) for item in trace if item.get("success")]
    lifted_steps = [int(item["step"]) for item in trace if item.get("lifted")]
    stable_steps = [int(item["step"]) for item in trace if item.get("stable_hold")]
    payload = {
        "video_path": str(video_path),
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


def _save_success_videos(
    model: PointTemporalStudent,
    spec: dict,
    pointcloud_source: str,
    output_dir: Path,
) -> list[str]:
    if args_cli.save_success_videos <= 0:
        return []

    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "video_progress.jsonl"
    video_paths: list[str] = []
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
        rgbd_mask_points, _ = sample_box_surface_points(
            tuple(float(v) for v in unwrapped.cfg.object_size),
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
            pending_reset = torch.zeros(video_envs, dtype=torch.bool, device=device)
            frames_by_env: list[list[np.ndarray]] = [[] for _ in range(video_envs)]
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
                trace_path = _write_video_trace(video_path, traces_by_env[env_id], attempt, env_id, len(frames))
                video_paths.append(str(video_path))
                saved_env_ids.add(env_id)
                _trace(f"saved success video: {video_path} (trace={trace_path})")

            with torch.inference_mode():
                for step in range(max_steps):
                    _reset_histories(point_hist, valid_hist, proprio_hist, pending_reset)
                    if bool(pending_reset.any()):
                        previous_action[pending_reset] = 0.0
                        hold_latch[pending_reset] = False
                    if step % max(int(args_cli.video_stride), 1) == 0:
                        _capture_video_frames(env, frames_by_env)
                    current_points, valid, _pc_info = _current_pointcloud(
                        unwrapped, pointcloud_source, local_points, rgbd_mask_points
                    )
                    policy_obs = _policy_tensor(obs).to(device)
                    point_hist, valid_hist, proprio_hist = _roll_histories(
                        point_hist,
                        valid_hist,
                        proprio_hist,
                        current_points,
                        valid,
                        policy_obs,
                    )
                    actions, predicted_privileged, predicted_hold_target, predicted_hold_logits = _student_action(
                        model, point_hist, valid_hist, proprio_hist
                    )
                    actions = _apply_predicted_grasp_hold(
                        actions,
                        previous_action,
                        predicted_privileged,
                        predicted_hold_target,
                        predicted_hold_logits,
                        hold_latch,
                    )
                    actions = _adapt_action(actions, previous_action)
                    obs, _rewards, terminated, truncated, extras = env.step(actions)
                    previous_action = actions.detach()
                    dones = terminated | truncated
                    _append_video_trace(env, traces_by_env, extras, step)
                    success_now = _tensor_bool(extras, "success_env", video_envs, device)
                    if bool(success_now.any()):
                        _capture_video_frames(env, frames_by_env)
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

                _capture_video_frames(env, frames_by_env)
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
                        _trace(f"saved debug rollout video: {debug_path}")
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
    return video_paths


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    torch.manual_seed(args_cli.seed)
    student_device_name = args_cli.student_device or args_cli.device
    if student_device_name.startswith("cuda") and not torch.cuda.is_available():
        student_device_name = "cpu"
    student_device = torch.device(student_device_name)
    model, metadata, spec = _load_student(checkpoint, student_device)
    pointcloud_source = _resolve_pointcloud_source(metadata)
    if pointcloud_source not in ("clean", "rgbd_projected_mask"):
        raise RuntimeError(f"Unsupported pointcloud source in metadata: {pointcloud_source!r}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args_cli.output_dir).expanduser().resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    _trace(
        f"task={args_cli.task} checkpoint={checkpoint} pointcloud_source={pointcloud_source} "
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
        passed = True
    else:
        eval_summary, _ = _run_student_eval(
            model,
            spec,
            pointcloud_source,
            int(args_cli.num_envs),
            int(args_cli.episodes),
            render_mode=None,
        )
        passed = eval_summary["success_rate"] >= float(args_cli.success_threshold)
    video_paths = _save_success_videos(model, spec, pointcloud_source, output_dir)
    trace_paths = [
        str(Path(path).with_suffix(".trace.json"))
        for path in video_paths
        if Path(path).with_suffix(".trace.json").exists()
    ]
    summary = {
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "student_device": str(student_device),
        "device": args_cli.device,
        "pointcloud_source": pointcloud_source,
        "metadata": metadata,
        "spec": spec,
        "action_clamp": float(args_cli.action_clamp),
        "action_ema_alpha": float(args_cli.action_ema_alpha),
        "action_rate_limit": float(args_cli.action_rate_limit),
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
        "success_threshold": float(args_cli.success_threshold),
        "passed": bool(passed),
        "eval": eval_summary,
        "success_videos": video_paths,
        "success_video_traces": trace_paths,
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _trace(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(
            f"success_rate={eval_summary['success_rate']:.3f} below threshold={float(args_cli.success_threshold):.3f}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
