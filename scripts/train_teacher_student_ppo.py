#!/usr/bin/env python3
"""On-policy PPO fine-tuning for the RGB-D point-cloud student policy."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToolReal-Revo2-Franka-DynamicTabletop-Teacher-Direct-v0")
parser.add_argument("--init-checkpoint", required=True, help="Initial PointTemporalStudent checkpoint.")
parser.add_argument("--out-dir", default="outputs/teacher_student_ppo")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=200)
parser.add_argument("--horizon", type=int, default=64)
parser.add_argument("--ppo-epochs", type=int, default=4)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--lr", type=float, default=1.0e-5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-ratio", type=float, default=0.2)
parser.add_argument("--ppo-ratio-limit", type=float, default=10.0)
parser.add_argument("--value-loss-weight", type=float, default=1.0)
parser.add_argument("--entropy-weight", type=float, default=0.002)
parser.add_argument("--privileged-aux-weight", type=float, default=0.05)
parser.add_argument("--bc-anchor-weight", type=float, default=0.0)
parser.add_argument("--reward-scale", type=float, default=0.01)
parser.add_argument("--init-log-std", type=float, default=-2.0)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
parser.add_argument("--action-clamp", type=float, default=1.0)
parser.add_argument("--predicted-grasp-hold-threshold", type=float, default=0.0)
parser.add_argument(
    "--predicted-grasp-hold-mode",
    choices=("max", "target", "learned_target", "target_plus_learned_residual"),
    default="max",
)
parser.add_argument("--predicted-grasp-hold-target", type=float, nargs="*", default=None)
parser.add_argument("--predicted-grasp-hold-blend", type=float, default=1.0)
parser.add_argument("--predicted-grasp-hold-rel-vel-threshold", type=float, default=0.0)
parser.add_argument("--predicted-grasp-hold-learned-gate-threshold", type=float, default=0.0)
parser.add_argument("--predicted-grasp-hold-learned-target-clamp", type=float, default=1.0)
parser.add_argument("--predicted-grasp-hold-learned-residual-scale", type=float, default=1.0)
parser.add_argument("--predicted-grasp-hold-learned-residual-clamp", type=float, default=0.25)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--history-bootstrap",
    choices=("repeat_initial", "zero_pad"),
    default="repeat_initial",
    help=(
        "How to initialize temporal history after reset. repeat_initial matches student "
        "evaluation by filling every slot with the first deployable observation."
    ),
)
parser.add_argument("--pointcloud-source", choices=("clean", "rgbd_projected_mask"), default="rgbd_projected_mask")
parser.add_argument(
    "--proprio-source",
    choices=("auto", "policy", "deployable_robot"),
    default="auto",
    help="Student proprioception source. auto reads the initialization checkpoint metadata.",
)
parser.add_argument("--rgbd-width", type=int, default=96)
parser.add_argument("--rgbd-height", type=int, default=72)
parser.add_argument("--rgbd-mask-points", type=int, default=512)
parser.add_argument("--rgbd-mask-dilation", type=int, default=1)
parser.add_argument("--rgbd-depth-tolerance", type=float, default=0.045)
parser.add_argument("--rgbd-min-valid-points", type=int, default=16)
parser.add_argument("--rgbd-clean-fallback", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--rgbd-temporal-fallback",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "For low-visibility RGB-D frames, reuse the previous valid deployable point cloud. "
        "This does not expose simulator object state."
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
parser.add_argument("--dynamic-curriculum-alpha", type=float, default=None)
parser.add_argument("--tabletop-asset-curriculum-alpha", type=float, default=None)
parser.add_argument("--tabletop-motion-curriculum-alpha", type=float, default=None)
parser.add_argument("--episode-length-s", type=float, default=None)
parser.add_argument("--dynamic-success-hold-steps", type=int, default=None)
parser.add_argument("--tabletop-pregrasp-lead-time", type=float, default=None)
parser.add_argument("--tabletop-pregrasp-ahead-distance", type=float, default=None)
parser.add_argument("--save-every", type=int, default=10)
parser.add_argument("--wandb-project", default=None)
parser.add_argument("--wandb-entity", default=None)
parser.add_argument("--wandb-run-name", default=None)
parser.add_argument("--wandb-group", default="deployable_student_ppo")
parser.add_argument("--wandb-tags", nargs="*", default=None)
parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if (args_cli.student_camera_eye is None) != (args_cli.student_camera_target is None):
    parser.error("--student-camera-eye and --student-camera-target must be provided together")
if not (0.0 <= float(args_cli.predicted_grasp_hold_blend) <= 1.0):
    parser.error("--predicted-grasp-hold-blend must be in [0, 1]")
if (
    float(args_cli.predicted_grasp_hold_threshold) > 0.0
    and args_cli.predicted_grasp_hold_mode in {"target", "target_plus_learned_residual"}
    and not args_cli.predicted_grasp_hold_target
):
    parser.error(
        "--predicted-grasp-hold-target is required for target-based hold modes"
    )

if args_cli.pointcloud_source == "rgbd_projected_mask":
    setattr(args_cli, "enable_cameras", True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.distributions import Normal  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402
from simtoolreal_lab.teacher_student import (  # noqa: E402
    PointTemporalStudent,
    deployable_robot_proprioception,
    masked_rgbd_object_points_in_palm_frame,
    object_points_in_palm_frame,
    primitive_surface_points_for_envs,
    sample_box_surface_points,
    sample_unit_primitive_surface_points,
)


def _trace(message: str) -> None:
    print(f"[STUDENT-PPO] {message}", flush=True)


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def _student_forward(
    student: PointTemporalStudent,
    point_hist: torch.Tensor,
    valid_hist: torch.Tensor,
    proprio_hist: torch.Tensor,
):
    normalization = getattr(student, "input_normalization", None)
    if normalization is not None:
        point_hist = (point_hist - normalization["pointcloud_mean"]) / normalization[
            "pointcloud_std"
        ]
        proprio_hist = (proprio_hist - normalization["proprio_mean"]) / normalization[
            "proprio_std"
        ]
    return student(point_hist, valid_hist, proprio_hist)


class StudentActorCritic(nn.Module):
    def __init__(self, student: PointTemporalStudent, privileged_dim: int, init_log_std: float) -> None:
        super().__init__()
        self.student = student
        self.log_std = nn.Parameter(torch.full((student.action_dim,), float(init_log_std)))
        self.critic = _mlp([privileged_dim, 256, 256, 1])

    def actor(self, point_hist: torch.Tensor, valid_hist: torch.Tensor, proprio_hist: torch.Tensor):
        out = _student_forward(self.student, point_hist, valid_hist, proprio_hist)
        std = torch.exp(self.log_std).expand_as(out["action"])
        return out, Normal(out["action"], std)

    def value(self, privileged: torch.Tensor) -> torch.Tensor:
        return self.critic(privileged).squeeze(-1)


def _policy_tensor(obs) -> torch.Tensor:
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        if "obs" in obs:
            return obs["obs"]
    return obs


def _resolve_proprio_source(metadata: dict) -> str:
    if args_cli.proprio_source != "auto":
        return args_cli.proprio_source
    return str(metadata.get("proprio_source", "policy"))


def _current_proprio(unwrapped_env, obs, proprio_source: str) -> torch.Tensor:
    if proprio_source == "deployable_robot":
        return deployable_robot_proprioception(unwrapped_env)
    return _policy_tensor(obs).to(unwrapped_env.device)


def _tensor_bool(extras: dict, key: str, num_envs: int, device: torch.device | str) -> torch.Tensor:
    value = extras.get(key)
    if value is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return value.bool()


def _load_student(checkpoint_path: Path, device: torch.device) -> tuple[PointTemporalStudent, dict, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"Not a PointTemporalStudent checkpoint: {checkpoint_path}")
    metadata = dict(payload.get("metadata", {}))
    spec = dict(payload.get("spec", {}))
    history = int(spec.get("history", metadata.get("history", 4)))
    num_points = int(spec.get("num_object_points", metadata.get("object_points", 64)))
    point_feature_dim = int(spec.get("point_feature_dim", metadata.get("point_feature_dim", 3)))
    proprio_dim = int(spec.get("proprio_dim", metadata.get("proprio_dim", 91)))
    action_dim = int(spec.get("action_dim", metadata.get("action_dim", 18)))
    arm_dim = int(spec.get("arm_dim", metadata.get("arm_dim", 7)))
    privileged_dim = int(spec.get("compact_privileged_dim", metadata.get("compact_privileged_dim", 32)))
    geometric_summary = bool(spec.get("geometric_summary", False))
    privileged_action_conditioning = bool(spec.get("privileged_action_conditioning", False))
    student = PointTemporalStudent(
        history=history,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        privileged_dim=privileged_dim,
        arm_dim=arm_dim,
        point_input_dim=point_feature_dim,
        geometric_summary=geometric_summary,
        privileged_action_conditioning=privileged_action_conditioning,
    ).to(device)
    missing, unexpected = student.load_state_dict(payload["model_state_dict"], strict=False)
    allowed_missing_prefixes = ("hold_head.", "hold_gate_head.")
    bad_missing = [name for name in missing if not name.startswith(allowed_missing_prefixes)]
    if bad_missing or unexpected:
        raise RuntimeError(
            "Student checkpoint state mismatch: "
            f"missing={bad_missing}, unexpected={list(unexpected)}"
        )
    if missing:
        _trace(f"init checkpoint missing optional hold head weights; initialized {list(missing)}")
    normalization = payload.get("normalization")
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
        student.input_normalization = {
            key: normalization[key].detach().to(device=device, dtype=torch.float32)
            for key in required_norm_keys
        }
    else:
        student.input_normalization = None
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
    return student, metadata, resolved_spec


def _make_env(task: str, num_envs: int, pointcloud_source: str):
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
    if pointcloud_source == "rgbd_projected_mask":
        env_cfg.student_camera_enabled = True
        env_cfg.student_camera.data_types = ["rgb", "distance_to_image_plane"]
        env_cfg.student_camera.width = int(args_cli.rgbd_width)
        env_cfg.student_camera.height = int(args_cli.rgbd_height)
        env_cfg.student_camera.spawn.focal_length = float(args_cli.student_camera_focal_length)
        env_cfg.student_camera.return_latest_camera_pose = True
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    env.unwrapped.sim._app_control_on_stop_handle = None
    return env


def _set_tracking_camera(unwrapped_env, camera_name: str, track_offset: tuple[float, float, float], target_offset) -> bool:
    camera = getattr(unwrapped_env, camera_name, None)
    if camera is None:
        return False
    origins = unwrapped_env.scene.env_origins
    if args_cli.student_camera_eye is not None and args_cli.student_camera_target is not None:
        eye = torch.tensor(args_cli.student_camera_eye, device=origins.device, dtype=origins.dtype).unsqueeze(0)
        targets = torch.tensor(
            args_cli.student_camera_target, device=origins.device, dtype=origins.dtype
        ).unsqueeze(0)
        camera.set_world_poses_from_view(origins + eye, origins + targets)
        if hasattr(camera, "_update_poses") and hasattr(camera, "_ALL_INDICES"):
            camera._update_poses(camera._ALL_INDICES)
        return True
    target_offset_t = torch.tensor(target_offset, device=origins.device, dtype=origins.dtype).unsqueeze(0)
    camera_offset_t = torch.tensor(track_offset, device=origins.device, dtype=origins.dtype).unsqueeze(0)
    object_root_pos = None
    if bool(args_cli.student_camera_track_object):
        object_root_pos = getattr(getattr(getattr(unwrapped_env, "object", None), "data", None), "root_pos_w", None)
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
    )


def _force_camera_update(camera, dt: float) -> None:
    if camera is not None and hasattr(camera, "update"):
        camera.update(dt, force_recompute=True)


def _student_camera_rgbd(unwrapped_env):
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


def _make_rgbd_mask_point_template(unwrapped_env, num_points: int, device) -> torch.Tensor:
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
        return features, valid

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
    if bool(low_valid.any()) and bool(args_cli.rgbd_clean_fallback):
        points[low_valid] = clean_points[low_valid]
        colors[low_valid] = 0.0
        valid[low_valid] = 1.0
    features = torch.cat((points, colors), dim=-1) if point_feature_dim == 6 else points
    return features, valid


def _compact_privileged(unwrapped_env, success_env: torch.Tensor | None = None) -> torch.Tensor:
    rel_obj_palm = quat_rotate_inverse(
        unwrapped_env._palm_quat_w,
        unwrapped_env._object_pos_w - unwrapped_env._palm_pos_w,
    )
    rel_vel_palm = quat_rotate_inverse(
        unwrapped_env._palm_quat_w,
        unwrapped_env._object_lin_vel_w - unwrapped_env._palm_lin_vel_w,
    )
    ang_vel_palm = quat_rotate_inverse(unwrapped_env._palm_quat_w, unwrapped_env._object_ang_vel_w)
    if success_env is None:
        success_env = torch.zeros(unwrapped_env.num_envs, dtype=torch.bool, device=unwrapped_env.device)
    values = [
        rel_obj_palm,
        rel_vel_palm,
        ang_vel_palm,
        unwrapped_env._surface_dist,
        unwrapped_env._object_palm_rel_vel.unsqueeze(-1),
        unwrapped_env._true_grasp.float().unsqueeze(-1),
        unwrapped_env._grasp_seen.float().unsqueeze(-1),
        success_env.float().unsqueeze(-1),
        unwrapped_env.episode_length_buf.float().unsqueeze(-1) / float(unwrapped_env.max_episode_length),
    ]
    asset_obs = getattr(unwrapped_env, "_tabletop_asset_obs", None)
    if torch.is_tensor(asset_obs) and asset_obs.shape[0] == rel_obj_palm.shape[0]:
        values.append(asset_obs)
    compact = torch.cat(values, dim=-1)
    if compact.shape[-1] < 32:
        compact = torch.cat(
            [compact, torch.zeros((compact.shape[0], 32 - compact.shape[-1]), device=compact.device)],
            dim=-1,
        )
    return compact[:, :32]


def _make_histories(num_envs: int, spec: dict, device: torch.device | str):
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


def _roll_histories(point_hist, valid_hist, proprio_hist, current_points, valid, proprio):
    point_hist = torch.roll(point_hist, shifts=-1, dims=1)
    valid_hist = torch.roll(valid_hist, shifts=-1, dims=1)
    proprio_hist = torch.roll(proprio_hist, shifts=-1, dims=1)
    point_hist[:, -1] = current_points
    valid_hist[:, -1] = valid
    proprio_hist[:, -1] = proprio
    return point_hist, valid_hist, proprio_hist


def _reset_histories(point_hist, valid_hist, proprio_hist, done: torch.Tensor) -> None:
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


def _apply_predicted_grasp_hold(
    action: torch.Tensor,
    previous_action: torch.Tensor,
    out: dict[str, torch.Tensor],
    hold_latch: torch.Tensor,
) -> torch.Tensor:
    threshold = float(args_cli.predicted_grasp_hold_threshold)
    predicted_privileged = out["privileged"]
    if threshold <= 0.0 or predicted_privileged.shape[-1] <= 16 or action.shape[-1] <= 7:
        return action

    grasp_signal = torch.maximum(predicted_privileged[:, 15], predicted_privileged[:, 16])
    should_latch = grasp_signal > threshold
    rel_vel_threshold = float(args_cli.predicted_grasp_hold_rel_vel_threshold)
    if rel_vel_threshold > 0.0 and predicted_privileged.shape[-1] > 14:
        should_latch &= predicted_privileged[:, 14] < rel_vel_threshold
    gate_threshold = float(args_cli.predicted_grasp_hold_learned_gate_threshold)
    if gate_threshold > 0.0:
        should_latch &= torch.sigmoid(out["hold_logits"]) > gate_threshold
    hold_latch |= should_latch
    if not bool(hold_latch.any()):
        return action

    held = action.clone()
    mode = str(args_cli.predicted_grasp_hold_mode)
    if mode == "max":
        held[hold_latch, 7:] = torch.maximum(
            action[hold_latch, 7:], previous_action[hold_latch, 7:]
        )
        return held

    hand_dim = action.shape[-1] - 7
    if mode == "learned_target":
        target = out["hold_target"]
    else:
        target_values = args_cli.predicted_grasp_hold_target or []
        if len(target_values) != hand_dim:
            raise RuntimeError(
                f"--predicted-grasp-hold-target has {len(target_values)} values, "
                f"expected hand_dim={hand_dim}"
            )
        target = torch.tensor(
            target_values, dtype=action.dtype, device=action.device
        ).view(1, hand_dim)
        if mode == "target_plus_learned_residual":
            residual = out["hold_target"]
            residual_clamp = float(args_cli.predicted_grasp_hold_learned_residual_clamp)
            if residual_clamp > 0.0:
                residual = torch.clamp(residual, -residual_clamp, residual_clamp)
            target = target + float(
                args_cli.predicted_grasp_hold_learned_residual_scale
            ) * residual

    target_clamp = float(args_cli.predicted_grasp_hold_learned_target_clamp)
    if target_clamp > 0.0:
        target = torch.clamp(target, -target_clamp, target_clamp)
    selected_target = target[hold_latch] if target.shape[0] == action.shape[0] else target
    blend = float(args_cli.predicted_grasp_hold_blend)
    held[hold_latch, 7:] = (
        (1.0 - blend) * action[hold_latch, 7:] + blend * selected_target
    )
    return held


def _bootstrap_histories(
    point_hist,
    valid_hist,
    proprio_hist,
    reset_mask: torch.Tensor,
    current_points: torch.Tensor,
    valid: torch.Tensor,
    proprio: torch.Tensor,
) -> None:
    if args_cli.history_bootstrap != "repeat_initial" or not bool(reset_mask.any()):
        return
    point_hist[reset_mask] = current_points[reset_mask].unsqueeze(1)
    valid_hist[reset_mask] = valid[reset_mask].unsqueeze(1)
    proprio_hist[reset_mask] = proprio[reset_mask].unsqueeze(1)


def _save_checkpoint(path: Path, actor_critic: StudentActorCritic, metadata: dict, spec: dict, iteration: int, metrics: dict):
    normalization = getattr(actor_critic.student, "input_normalization", None)
    checkpoint = {
        "model_state_dict": actor_critic.student.state_dict(),
        "ppo_state_dict": actor_critic.state_dict(),
        "metadata": metadata,
        "spec": spec,
        "iteration": int(iteration),
        "metrics": metrics,
        "train_algorithm": "student_ppo_privileged_critic",
        "normalization": (
            {key: value.detach().cpu() for key, value in normalization.items()}
            if normalization is not None
            else None
        ),
    }
    torch.save(checkpoint, path)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    device = torch.device(args_cli.device if torch.cuda.is_available() or not args_cli.device.startswith("cuda") else "cpu")
    init_checkpoint = Path(args_cli.init_checkpoint).expanduser().resolve()
    student, metadata, spec = _load_student(init_checkpoint, device)
    proprio_source = _resolve_proprio_source(metadata)
    if proprio_source not in ("policy", "deployable_robot"):
        raise RuntimeError(f"Unsupported proprio source: {proprio_source!r}")
    metadata["proprio_source"] = proprio_source
    actor_critic = StudentActorCritic(
        student,
        privileged_dim=int(spec["compact_privileged_dim"]),
        init_log_std=float(args_cli.init_log_std),
    ).to(device)
    actor_parameters = list(actor_critic.student.parameters()) + [actor_critic.log_std]
    critic_parameters = list(actor_critic.critic.parameters())
    metadata["ppo_gradient_clip_mode"] = "separate_actor_critic"
    anchor_student = None
    if float(args_cli.bc_anchor_weight) > 0.0:
        anchor_student = copy.deepcopy(actor_critic.student).to(device)
        anchor_student.eval()
        for parameter in anchor_student.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(actor_critic.parameters(), lr=float(args_cli.lr), weight_decay=1.0e-5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args_cli.out_dir).expanduser().resolve() / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    progress_path = out_dir / "progress.jsonl"
    _trace(f"output_dir={out_dir}")
    _trace(f"init_checkpoint={init_checkpoint}")

    wandb_run = None
    if args_cli.wandb_project and args_cli.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args_cli.wandb_project,
            entity=args_cli.wandb_entity,
            name=args_cli.wandb_run_name or f"student_ppo_{args_cli.task}",
            group=args_cli.wandb_group,
            tags=args_cli.wandb_tags,
            mode=args_cli.wandb_mode,
            dir=str(out_dir),
            job_type="student_ppo",
            config={
                "task": args_cli.task,
                "init_checkpoint": str(init_checkpoint),
                "num_envs": int(args_cli.num_envs),
                "iterations": int(args_cli.iterations),
                "horizon": int(args_cli.horizon),
                "ppo_epochs": int(args_cli.ppo_epochs),
                "minibatches": int(args_cli.minibatches),
                "lr": float(args_cli.lr),
                "reward_scale": float(args_cli.reward_scale),
                "bc_anchor_weight": float(args_cli.bc_anchor_weight),
                "gradient_clip_mode": "separate_actor_critic",
                "history_bootstrap": str(args_cli.history_bootstrap),
                "pointcloud_source": args_cli.pointcloud_source,
                "proprio_source": proprio_source,
                "point_feature_dim": int(spec.get("point_feature_dim", 3)),
                "rgbd_resolution": [int(args_cli.rgbd_width), int(args_cli.rgbd_height)],
                "rgbd_clean_fallback": bool(args_cli.rgbd_clean_fallback),
                "rgbd_temporal_fallback": bool(args_cli.rgbd_temporal_fallback),
                "predicted_grasp_hold_threshold": float(
                    args_cli.predicted_grasp_hold_threshold
                ),
                "predicted_grasp_hold_mode": str(args_cli.predicted_grasp_hold_mode),
                "predicted_grasp_hold_blend": float(args_cli.predicted_grasp_hold_blend),
                "dynamic_curriculum_alpha": args_cli.dynamic_curriculum_alpha,
                "tabletop_asset_curriculum_alpha": args_cli.tabletop_asset_curriculum_alpha,
                "tabletop_motion_curriculum_alpha": args_cli.tabletop_motion_curriculum_alpha,
                "seed": int(args_cli.seed),
            },
        )
        _trace(f"wandb_run={wandb_run.url}")

    env = _make_env(args_cli.task, int(args_cli.num_envs), args_cli.pointcloud_source)
    unwrapped = env.unwrapped
    env_device = unwrapped.device
    local_points, _ = sample_box_surface_points(
        tuple(float(v) for v in unwrapped.cfg.object_size),
        int(spec["num_object_points"]),
        env_device,
    )
    rgbd_mask_points = None
    if args_cli.pointcloud_source == "rgbd_projected_mask":
        rgbd_mask_points = _make_rgbd_mask_point_template(
            unwrapped,
            max(int(args_cli.rgbd_mask_points), int(spec["num_object_points"])),
            env_device,
        )
    obs, _ = env.reset(seed=args_cli.seed)
    if args_cli.pointcloud_source == "rgbd_projected_mask":
        _set_student_camera_poses(unwrapped)

    num_envs = int(args_cli.num_envs)
    horizon = int(args_cli.horizon)
    point_hist, valid_hist, proprio_hist = _make_histories(num_envs, spec, env_device)
    pending_reset = torch.ones(num_envs, dtype=torch.bool, device=env_device)
    previous_action = torch.zeros((num_envs, int(spec["action_dim"])), device=env_device)
    hold_latch = torch.zeros(num_envs, dtype=torch.bool, device=env_device)
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=env_device)
    episode_lifted = torch.zeros(num_envs, dtype=torch.bool, device=env_device)
    episode_stable = torch.zeros(num_envs, dtype=torch.bool, device=env_device)

    best_success_rate = -1.0
    best_path = out_dir / "student_ppo_best.pt"
    last_path = out_dir / "student_ppo_last.pt"

    try:
        for iteration in range(1, int(args_cli.iterations) + 1):
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "iter_start", "iteration": int(iteration)}, sort_keys=True) + "\n")
            _trace(f"iter={iteration:04d} start")
            buffers = {
                "point": [],
                "valid": [],
                "proprio": [],
                "privileged": [],
                "raw_action": [],
                "log_prob": [],
                "reward": [],
                "done": [],
                "value": [],
            }
            completed = success_count = lifted_count = stable_count = 0
            reward_sum = 0.0
            reward_count = 0
            rgbd_valid_sum = 0.0
            rgbd_temporal_fallback_sum = 0
            rgbd_steps = 0
            hold_latch_sum = 0.0

            actor_critic.eval()
            with torch.no_grad():
                for _step in range(horizon):
                    if _step > 0 and _step % 16 == 0:
                        _trace(f"iter={iteration:04d} rollout_step={_step:04d}/{horizon:04d}")
                        with progress_path.open("a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "event": "rollout_step",
                                        "iteration": int(iteration),
                                        "step": int(_step),
                                        "horizon": int(horizon),
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                    _reset_histories(point_hist, valid_hist, proprio_hist, pending_reset)
                    if bool(pending_reset.any()):
                        previous_action[pending_reset] = 0.0
                        hold_latch[pending_reset] = False
                    current_points, valid = _current_pointcloud(
                        unwrapped,
                        args_cli.pointcloud_source,
                        local_points,
                        rgbd_mask_points,
                        int(spec.get("point_feature_dim", 3)),
                    )
                    if args_cli.pointcloud_source == "rgbd_projected_mask":
                        rgbd_temporal_fallback_sum += _apply_rgbd_temporal_fallback(
                            current_points,
                            valid,
                            point_hist,
                            valid_hist,
                        )
                        rgbd_valid_sum += float(valid.sum(dim=-1).mean().item())
                        rgbd_steps += 1
                    proprio = _current_proprio(unwrapped, obs, proprio_source)
                    if proprio.shape[-1] != int(spec["proprio_dim"]):
                        raise RuntimeError(
                            f"proprio_dim mismatch: source={proprio_source} "
                            f"env={proprio.shape[-1]} checkpoint={spec['proprio_dim']}"
                        )
                    _bootstrap_histories(
                        point_hist,
                        valid_hist,
                        proprio_hist,
                        pending_reset,
                        current_points,
                        valid,
                        proprio,
                    )
                    point_hist, valid_hist, proprio_hist = _roll_histories(
                        point_hist, valid_hist, proprio_hist, current_points, valid, proprio
                    )
                    unwrapped._compute_intermediate_values()
                    privileged = _compact_privileged(unwrapped)
                    out, dist = actor_critic.actor(point_hist, valid_hist, proprio_hist)
                    raw_action = dist.sample()
                    action = torch.clamp(
                        raw_action,
                        -float(args_cli.action_clamp),
                        float(args_cli.action_clamp),
                    )
                    action = _apply_predicted_grasp_hold(
                        action,
                        previous_action,
                        out,
                        hold_latch,
                    )
                    previous_action.copy_(action)
                    hold_latch_sum += float(hold_latch.float().mean().item())
                    log_prob = dist.log_prob(raw_action).sum(dim=-1)
                    value = actor_critic.value(privileged)

                    obs, rewards, terminated, truncated, extras = env.step(action)
                    dones = terminated | truncated
                    scaled_reward = rewards * float(args_cli.reward_scale)

                    buffers["point"].append(point_hist.detach().clone())
                    buffers["valid"].append(valid_hist.detach().clone())
                    buffers["proprio"].append(proprio_hist.detach().clone())
                    buffers["privileged"].append(privileged.detach().clone())
                    buffers["raw_action"].append(raw_action.detach().clone())
                    buffers["log_prob"].append(log_prob.detach().clone())
                    buffers["reward"].append(scaled_reward.detach().clone())
                    buffers["done"].append(dones.detach().clone())
                    buffers["value"].append(value.detach().clone())

                    episode_success |= _tensor_bool(extras, "success_env", num_envs, env_device)
                    episode_lifted |= _tensor_bool(extras, "lifted_env", num_envs, env_device)
                    episode_stable |= _tensor_bool(extras, "stable_hold_env", num_envs, env_device)
                    reward_sum += float(rewards.sum().item())
                    reward_count += int(rewards.numel())
                    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    if done_ids.numel() > 0:
                        completed += int(done_ids.numel())
                        success_count += int(episode_success[done_ids].sum().item())
                        lifted_count += int(episode_lifted[done_ids].sum().item())
                        stable_count += int(episode_stable[done_ids].sum().item())
                        episode_success[done_ids] = False
                        episode_lifted[done_ids] = False
                        episode_stable[done_ids] = False
                    pending_reset = dones

                unwrapped._compute_intermediate_values()
                next_value = actor_critic.value(_compact_privileged(unwrapped)).detach()

            rewards_t = torch.stack(buffers["reward"])
            dones_t = torch.stack(buffers["done"]).float()
            values_t = torch.stack(buffers["value"])
            advantages = torch.zeros_like(rewards_t)
            last_gae = torch.zeros(num_envs, device=env_device)
            for step in reversed(range(horizon)):
                if step == horizon - 1:
                    next_nonterminal = 1.0 - dones_t[step]
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones_t[step]
                    next_values = values_t[step + 1]
                delta = rewards_t[step] + float(args_cli.gamma) * next_values * next_nonterminal - values_t[step]
                last_gae = delta + float(args_cli.gamma) * float(args_cli.gae_lambda) * next_nonterminal * last_gae
                advantages[step] = last_gae
            returns = advantages + values_t
            raw_advantage_mean = float(advantages.mean().detach().cpu())
            raw_advantage_std = float(advantages.std().detach().cpu())
            return_mean = float(returns.mean().detach().cpu())
            return_std = float(returns.std().detach().cpu())
            advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1.0e-6)

            flat = {
                key: torch.stack(value).reshape(horizon * num_envs, *value[0].shape[1:])
                for key, value in buffers.items()
            }
            flat["advantage"] = advantages.reshape(horizon * num_envs)
            flat["return"] = returns.reshape(horizon * num_envs)
            batch_size = horizon * num_envs
            minibatch_size = max(1, batch_size // int(args_cli.minibatches))

            actor_critic.train()
            update_totals = {
                "policy": 0.0,
                "value": 0.0,
                "entropy": 0.0,
                "privileged_aux": 0.0,
                "bc_anchor": 0.0,
                "actor_grad_norm": 0.0,
                "critic_grad_norm": 0.0,
                "loss": 0.0,
            }
            update_count = 0
            for _epoch in range(int(args_cli.ppo_epochs)):
                permutation = torch.randperm(batch_size, device=env_device)
                for start in range(0, batch_size, minibatch_size):
                    ids = permutation[start : start + minibatch_size]
                    out, dist = actor_critic.actor(flat["point"][ids], flat["valid"][ids], flat["proprio"][ids])
                    new_log_prob = dist.log_prob(flat["raw_action"][ids]).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    log_ratio = torch.clamp(new_log_prob - flat["log_prob"][ids], -20.0, 20.0)
                    ratio_limit = max(float(args_cli.ppo_ratio_limit), 1.0)
                    ratio = torch.clamp(torch.exp(log_ratio), 1.0 / ratio_limit, ratio_limit)
                    unclipped = ratio * flat["advantage"][ids]
                    clipped = torch.clamp(
                        ratio,
                        1.0 - float(args_cli.clip_ratio),
                        1.0 + float(args_cli.clip_ratio),
                    ) * flat["advantage"][ids]
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_pred = actor_critic.value(flat["privileged"][ids])
                    value_loss = 0.5 * (value_pred - flat["return"][ids]).pow(2).mean()
                    privileged_aux_loss = (out["privileged"] - flat["privileged"][ids]).pow(2).mean()
                    if anchor_student is not None:
                        with torch.no_grad():
                            anchor_action = _student_forward(
                                anchor_student,
                                flat["point"][ids],
                                flat["valid"][ids],
                                flat["proprio"][ids],
                            )["action"]
                        bc_anchor_loss = (out["action"] - anchor_action).pow(2).mean()
                    else:
                        bc_anchor_loss = torch.zeros((), device=env_device)
                    loss = (
                        policy_loss
                        + float(args_cli.value_loss_weight) * value_loss
                        - float(args_cli.entropy_weight) * entropy
                        + float(args_cli.privileged_aux_weight) * privileged_aux_loss
                        + float(args_cli.bc_anchor_weight) * bc_anchor_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        actor_parameters, float(args_cli.max_grad_norm)
                    )
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        critic_parameters, float(args_cli.max_grad_norm)
                    )
                    optimizer.step()
                    update_totals["policy"] += float(policy_loss.detach().cpu())
                    update_totals["value"] += float(value_loss.detach().cpu())
                    update_totals["entropy"] += float(entropy.detach().cpu())
                    update_totals["privileged_aux"] += float(privileged_aux_loss.detach().cpu())
                    update_totals["bc_anchor"] += float(bc_anchor_loss.detach().cpu())
                    update_totals["actor_grad_norm"] += float(actor_grad_norm.detach().cpu())
                    update_totals["critic_grad_norm"] += float(critic_grad_norm.detach().cpu())
                    update_totals["loss"] += float(loss.detach().cpu())
                    update_count += 1

            metrics = {key: value / max(update_count, 1) for key, value in update_totals.items()}
            metrics |= {
                "iteration": int(iteration),
                "episodes": int(completed),
                "success_rate": float(success_count / max(completed, 1)),
                "lifted_rate": float(lifted_count / max(completed, 1)),
                "stable_hold_rate": float(stable_count / max(completed, 1)),
                "mean_step_reward": float(reward_sum / max(reward_count, 1)),
                "reward_scale": float(args_cli.reward_scale),
                "bc_anchor_weight": float(args_cli.bc_anchor_weight),
                "ppo_ratio_limit": float(args_cli.ppo_ratio_limit),
                "log_std_mean": float(actor_critic.log_std.detach().mean().cpu()),
                "raw_advantage_mean": raw_advantage_mean,
                "raw_advantage_std": raw_advantage_std,
                "return_mean": return_mean,
                "return_std": return_std,
                "rgbd_valid_mean": float(rgbd_valid_sum / max(rgbd_steps, 1)),
                "rgbd_temporal_fallback_envs_per_step": float(
                    rgbd_temporal_fallback_sum / max(rgbd_steps, 1)
                ),
                "predicted_grasp_hold_latch_fraction": float(
                    hold_latch_sum / max(horizon, 1)
                ),
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, sort_keys=True) + "\n")
            if wandb_run is not None:
                wandb_run.log(metrics, step=iteration)
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "event": "iter_end",
                            "iteration": int(iteration),
                            "episodes": int(completed),
                            "success_rate": float(metrics["success_rate"]),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            _trace(
                f"iter={iteration:04d} episodes={completed} success={metrics['success_rate']:.3f} "
                f"reward={metrics['mean_step_reward']:.3f} loss={metrics['loss']:.4f}"
            )
            _save_checkpoint(last_path, actor_critic, metadata, spec, iteration, metrics)
            if metrics["success_rate"] >= best_success_rate:
                best_success_rate = metrics["success_rate"]
                _save_checkpoint(best_path, actor_critic, metadata, spec, iteration, metrics)
            if int(args_cli.save_every) > 0 and iteration % int(args_cli.save_every) == 0:
                _save_checkpoint(out_dir / f"student_ppo_iter_{iteration:04d}.pt", actor_critic, metadata, spec, iteration, metrics)
    finally:
        env.close()

    _trace(f"saved best checkpoint: {best_path}")
    _trace(f"saved last checkpoint: {last_path}")
    if wandb_run is not None:
        import wandb

        wandb_run.summary.update(
            {
                "best_success_rate": float(best_success_rate),
                "best_checkpoint": str(best_path),
                "last_checkpoint": str(last_path),
            }
        )
        wandb.save(str(best_path), base_path=str(out_dir))
        wandb.save(str(last_path), base_path=str(out_dir))
        wandb_run.finish()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
