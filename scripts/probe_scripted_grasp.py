#!/usr/bin/env python3
"""Scripted close/lift probe for the Revo2 static grasp IsaacLab tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToolReal-Revo2-Franka-StaticBall-Grasp-Direct-v0", help="Gym task id.")
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=8, help="Number of envs.")
parser.add_argument("--pre-steps", type=int, default=100, help="Initial locked/open steps.")
parser.add_argument("--close-steps", type=int, default=90, help="Hand close steps.")
parser.add_argument("--lift-steps", type=int, default=50, help="Arm lift steps.")
parser.add_argument("--hold-steps", type=int, default=80, help="Hold steps after lift.")
parser.add_argument("--seed", type=int, default=None, help="Optional environment seed.")
parser.add_argument(
    "--hold-arm-mode",
    choices=("zero", "lift-target"),
    default="lift-target",
    help="Arm command during hold when a lift target exists.",
)
parser.add_argument("--lift-deltas", type=float, nargs=7, default=None, help="Optional 7-DoF arm lift target delta.")
parser.add_argument("--lift-joint2-delta", type=float, default=None, help="Legacy shortcut for joint2 delta.")
parser.add_argument("--lift-joint4-delta", type=float, default=None, help="Legacy shortcut for joint4 delta.")
parser.add_argument("--lift-joint6-delta", type=float, default=None, help="Legacy shortcut for joint6 delta.")
parser.add_argument("--hand-action", type=float, default=1.0, help="Hand action during close/lift/hold.")
parser.add_argument(
    "--hand-action-vector",
    type=float,
    nargs="+",
    default=None,
    help="Optional hand action vector for close/lift/hold. Use 6 values for semantic hands or 12 for physical Inspire.",
)
parser.add_argument(
    "--inspire-close-target",
    choices=("cfg", "safe", "sphere4cm", "p80"),
    default="cfg",
    help="Optional Inspire semantic close target override for diagnostic videos.",
)
parser.add_argument(
    "--inspire-close-target-values",
    type=float,
    nargs=12,
    default=None,
    help="Optional explicit 12-joint Inspire semantic close targets in sim_hand_joint_names order.",
)
parser.add_argument(
    "--reference-hand-fractions",
    type=float,
    nargs=6,
    default=None,
    help="Optional 6-DoF active hand close fractions for Revo2/Inspire semantic action probes.",
)
parser.add_argument("--object-pos", type=float, nargs=3, default=None, help="Override object start XYZ.")
parser.add_argument(
    "--object-pos-grid",
    type=float,
    nargs=7,
    default=None,
    metavar=("X_MIN", "X_MAX", "X_COUNT", "Y_MIN", "Y_MAX", "Y_COUNT", "Z"),
    help="Assign per-env object positions on an XY grid at fixed Z.",
)
parser.add_argument("--table-top-z", type=float, default=None, help="Override table top z.")
parser.add_argument("--table-pos-xy", type=float, nargs=2, default=None, help="Override table center XY.")
parser.add_argument(
    "--static-tabletop",
    action="store_true",
    help="Force dynamic tabletop tasks to use one static object with zero reset velocity.",
)
parser.add_argument(
    "--pregrasp-arm-pos",
    type=float,
    nargs=7,
    default=None,
    help="Optional absolute 7-DoF Franka pregrasp target reached during pre-steps.",
)
parser.add_argument(
    "--pregrasp-arm-pos-mode",
    choices=("target", "normalized-action"),
    default="target",
    help=(
        "Interpret --pregrasp-arm-pos as an absolute joint target, or as the arm-position "
        "candidate used by search_dynamic_revo2_arm_pregrasp.py and converted to a normalized action."
    ),
)
parser.add_argument(
    "--close-arm-pos",
    type=float,
    nargs=7,
    default=None,
    help="Optional absolute 7-DoF Franka target used during close; lift starts from this target if set.",
)
parser.add_argument(
    "--close-arm-pos-mode",
    choices=("target", "normalized-action"),
    default="target",
    help="Interpret --close-arm-pos as an absolute joint target or normalized action target.",
)
parser.add_argument(
    "--reset-object-after-pre",
    action="store_true",
    help="Restore the object pose/velocity after the pregrasp move so close/lift probes are not polluted by approach contact.",
)
parser.add_argument("--output-json", default=None, help="Optional summary JSON path.")
parser.add_argument("--video-path", default=None, help="Optional mp4 path for env0.")
parser.add_argument("--video-stride", type=int, default=2, help="Record every N control steps.")
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--video-camera-eye", type=float, nargs=3, default=(1.08, -0.86, 0.74))
parser.add_argument("--video-camera-target", type=float, nargs=3, default=(0.60, 0.00, 0.38))
parser.add_argument("--video-camera-focal-length", type=float, default=24.0)
parser.add_argument("--video-camera-resolution", type=int, nargs=2, default=(1280, 720))
parser.add_argument("--contact-distance", type=float, default=None, help="Optional reward/success contact distance override.")
parser.add_argument(
    "--strict-contact-distance",
    type=float,
    default=None,
    help="Optional strict success contact distance override.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video_path and hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402
from simtoolreal_lab.tasks.dynamic_dexterous_grasp.dynamic_dexterous_grasp_env_cfg import (  # noqa: E402
    INSPIRE_ANYDEX_P80_CLOSE_TARGETS,
    INSPIRE_ANYDEX_SPHERE_4CM_CLOSE_TARGETS,
    INSPIRE_ANYDEX_SPHERE_SAFE_CLOSE_TARGETS,
)


def _trace(message: str) -> None:
    print(f"[SCRIPTED] {message}", flush=True)


def _set_video_camera_pose(unwrapped_env) -> bool:
    camera = getattr(unwrapped_env, "_video_camera", None)
    if camera is None:
        return False
    origins = unwrapped_env.scene.env_origins
    eye = torch.tensor(args_cli.video_camera_eye, device=origins.device, dtype=origins.dtype).unsqueeze(0)
    target = torch.tensor(args_cli.video_camera_target, device=origins.device, dtype=origins.dtype).unsqueeze(0)
    camera.set_world_poses_from_view(origins + eye, origins + target)
    if hasattr(camera, "_update_poses") and hasattr(camera, "_ALL_INDICES"):
        camera._update_poses(camera._ALL_INDICES)
    return True


def _force_camera_update(camera, dt: float) -> None:
    if hasattr(camera, "update"):
        camera.update(dt, force_recompute=True)


def _record_frame(env, frames: list) -> None:
    unwrapped = env.unwrapped
    camera = getattr(unwrapped, "_video_camera", None)
    if camera is None:
        frame = unwrapped.render(recompute=True)
        if frame is not None and frame.size > 0:
            frames.append(frame.copy())
        return

    _force_camera_update(camera, float(getattr(unwrapped, "dt", unwrapped.step_dt)))
    rgb = camera.data.output.get("rgb")
    if rgb is None or rgb.numel() == 0:
        return
    frame = rgb[0, ..., :3].detach().cpu().numpy()
    if frame.dtype != np.uint8:
        max_value = float(frame.max()) if frame.size else 0.0
        frame = np.clip(frame * 255.0 if max_value <= 1.0 else frame, 0, 255).astype(np.uint8)
    frames.append(frame.copy())


def _step(env, actions: torch.Tensor, success_any: torch.Tensor, metric_max: dict[str, torch.Tensor]):
    _, _, terminated, truncated, extras = env.step(actions)
    for key in ("success", "true_grasp", "lifted", "stable_hold"):
        tensor = extras.get(f"{key}_env")
        if tensor is None:
            continue
        metric_max[key] = torch.maximum(metric_max[key], tensor.float())
    success = extras.get("success_env")
    if success is not None:
        success_any |= success.bool()
    return terminated | truncated, extras


def _mean_log(extras: dict) -> dict[str, float]:
    return {key: float(value.item()) for key, value in extras.get("log", {}).items() if hasattr(value, "item")}


def _clearance_debug(base) -> dict:
    if getattr(base.cfg, "task_family", None) != "dynamic_tabletop_grasp":
        return {}

    sample_names = []
    sample_pos_parts = []
    sample_margin_parts = []
    body_ids = list(getattr(base, "_tabletop_arm_clearance_body_ids", ()))
    body_margins = getattr(base, "_tabletop_arm_clearance_body_margins", None)
    if body_ids:
        sample_names.extend(base.robot.body_names[idx] for idx in body_ids)
        sample_pos_parts.append(base.robot.data.body_pos_w[:, body_ids])
        sample_margin_parts.append(body_margins.expand(base.num_envs, -1))

    if hasattr(base, "_fingertip_pos_w"):
        fingertip_margin = max(float(getattr(base.cfg, "tabletop_arm_clearance_fingertip_point_margin", 0.006)), 0.0)
        sample_names.extend(f"{name}:contact_point" for name in base.cfg.fingertip_body_names)
        sample_pos_parts.append(base._fingertip_pos_w)
        sample_margin_parts.append(
            torch.full(
                (base.num_envs, base._fingertip_pos_w.shape[1]),
                fingertip_margin,
                dtype=torch.float32,
                device=base.device,
            )
        )

    if hasattr(base, "_palm_pos_w"):
        palm_margin = max(float(getattr(base.cfg, "tabletop_arm_clearance_palm_point_margin", 0.012)), 0.0)
        sample_names.append(f"{base.cfg.palm_body_name}:palm_point")
        sample_pos_parts.append(base._palm_pos_w.unsqueeze(1))
        sample_margin_parts.append(torch.full((base.num_envs, 1), palm_margin, dtype=torch.float32, device=base.device))

    if not sample_pos_parts:
        return {}

    sample_pos = torch.cat(sample_pos_parts, dim=1)
    sample_margin = torch.cat(sample_margin_parts, dim=1)
    local_pos = sample_pos - base.scene.env_origins[:, None, :]
    required_z = base.scene.env_origins[:, 2:3] + float(base.cfg.table_top_z) + sample_margin
    margin = sample_pos[:, :, 2] - required_z
    env_margin = margin[0].detach().cpu()
    env_pos = sample_pos[0].detach().cpu()
    env_local = local_pos[0].detach().cpu()
    min_idx = int(torch.argmin(env_margin).item())
    return {
        "table_top_z": float(base.cfg.table_top_z),
        "clearance_ok_env0": bool(base._tabletop_arm_clearance_ok[0].detach().cpu()),
        "clearance_penalty_env0": float(base._tabletop_arm_clearance_penalty[0].detach().cpu()),
        "clearance_active_fraction_env0": float(base._tabletop_arm_clearance_active_fraction[0].detach().cpu()),
        "clearance_min_margin_env0": float(base._tabletop_arm_clearance_min_margin[0].detach().cpu()),
        "clearance_min_sample_env0": {
            "name": sample_names[min_idx],
            "margin": float(env_margin[min_idx]),
            "z": float(env_pos[min_idx, 2]),
            "required_z": float(required_z[0, min_idx].detach().cpu()),
            "pos": [float(v) for v in env_pos[min_idx]],
            "local_pos": [float(v) for v in env_local[min_idx]],
            "configured_margin": float(sample_margin[0, min_idx].detach().cpu()),
        },
    }


def _geometry_debug(env) -> dict:
    base = env.unwrapped
    base._compute_intermediate_values()
    origin = base.scene.env_origins[0]
    object_pos = base._object_pos_w[0]
    palm_pos = base._palm_pos_w[0]
    hand_names = set(getattr(base.cfg, "hand_joint_names", ())) | set(getattr(base.cfg, "sim_hand_joint_names", ()))
    control_names = set(getattr(base.cfg, "arm_joint_names", ())) | hand_names
    joint_targets = getattr(base, "_joint_targets", None)
    return {
        "palm_body": base.cfg.palm_body_name,
        "fingertip_bodies": list(base.cfg.fingertip_body_names),
        "env_origin_env0": [float(v) for v in origin.detach().cpu()],
        "object_pos_env0": [float(v) for v in object_pos.detach().cpu()],
        "object_pos_local_env0": [float(v) for v in (object_pos - origin).detach().cpu()],
        "palm_pos_env0": [float(v) for v in palm_pos.detach().cpu()],
        "palm_pos_local_env0": [float(v) for v in (palm_pos - origin).detach().cpu()],
        "object_minus_palm_env0": [float(v) for v in (object_pos - palm_pos).detach().cpu()],
        "object_height_delta_env0": float(base._object_height_delta[0].detach().cpu()),
        "true_grasp_env0": bool(base._true_grasp[0].detach().cpu()),
        "lifted_env0": bool(base._lifted[0].detach().cpu()),
        "stable_hold_env0": bool(base._stable_hold[0].detach().cpu()),
        "fingertip_pos_env0": [
            [float(v) for v in tip.detach().cpu()] for tip in base._fingertip_pos_w[0]
        ],
        "fingertip_pos_local_env0": [
            [float(v) for v in (tip - origin).detach().cpu()] for tip in base._fingertip_pos_w[0]
        ],
        "surface_dist_env0": [float(v) for v in base._surface_dist[0].detach().cpu()],
        "joint_pos_env0": {
            name: float(base.robot.data.joint_pos[0, idx].detach().cpu())
            for idx, name in enumerate(base.robot.joint_names)
            if name in control_names
        },
        "joint_target_env0": {
            name: float(joint_targets[0, idx].detach().cpu())
            for idx, name in enumerate(base.robot.joint_names)
            if joint_targets is not None and name in control_names
        },
        "clearance_debug_env0": _clearance_debug(base),
    }


def _object_positions_tensor(base, object_pos) -> torch.Tensor:
    if torch.is_tensor(object_pos):
        positions = object_pos.to(device=base.device, dtype=torch.float32)
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError(f"Expected object_pos tensor with shape (N, 3), got {tuple(positions.shape)}.")
        if positions.shape[0] < base.num_envs:
            repeat_count = (base.num_envs + positions.shape[0] - 1) // positions.shape[0]
            positions = positions.repeat((repeat_count, 1))
        return positions[: base.num_envs]
    return torch.tensor(object_pos, dtype=torch.float32, device=base.device).view(1, 3).expand(base.num_envs, -1)


def _reset_object_pose(env, object_pos) -> None:
    base = env.unwrapped
    env_ids = torch.arange(base.num_envs, device=base.device, dtype=torch.long)
    positions = _object_positions_tensor(base, object_pos)
    if hasattr(base, "_object_start_pos"):
        base._object_start_pos = positions.clone()

    root_quat = torch.tensor(
        getattr(base.cfg, "object_start_rot", (1.0, 0.0, 0.0, 0.0)),
        dtype=torch.float32,
        device=base.device,
    ).view(1, 4)

    if hasattr(base, "_tabletop_objects") and getattr(base, "_tabletop_objects"):
        active_ids = getattr(base, "_tabletop_active_asset_ids", torch.zeros(base.num_envs, device=base.device, dtype=torch.long))
        for asset_index, obj in enumerate(base._tabletop_objects):
            state = obj.data.default_root_state.clone()
            state[:, 0:3] = base.scene.env_origins + positions
            state[:, 3:7] = root_quat.expand(base.num_envs, -1)
            state[:, 7:] = 0.0
            inactive = active_ids != asset_index
            if torch.any(inactive) and hasattr(base, "_tabletop_inactive_asset_parking_local_pos"):
                parking = base._tabletop_inactive_asset_parking_local_pos(asset_index, int(inactive.sum().item()))
                state[inactive, 0:3] = base.scene.env_origins[inactive] + parking
                state[inactive, 3:7] = root_quat
            obj.write_root_pose_to_sim(state[:, :7], env_ids)
            obj.write_root_velocity_to_sim(state[:, 7:], env_ids)
    else:
        state = base.object.data.default_root_state.clone()
        state[:, 0:3] = base.scene.env_origins + positions
        state[:, 3:7] = root_quat.expand(base.num_envs, -1)
        state[:, 7:] = 0.0
        base.object.write_root_pose_to_sim(state[:, :7], env_ids)
        base.object.write_root_velocity_to_sim(state[:, 7:], env_ids)

    if hasattr(base, "_tabletop_cmd_lin_vel_w"):
        base._tabletop_cmd_lin_vel_w[:] = 0.0
    if hasattr(base, "_tabletop_cmd_yaw_rate"):
        base._tabletop_cmd_yaw_rate[:] = 0.0
    if hasattr(base, "_set_tabletop_hover_targets"):
        base._set_tabletop_hover_targets(env_ids, base.scene.env_origins + positions)
    base.scene.write_data_to_sim()
    base.sim.forward()
    base._compute_intermediate_values()


def _grid_positions(values: list[float]) -> list[tuple[float, float, float]]:
    x_min, x_max, x_count_f, y_min, y_max, y_count_f, z = values
    x_count = max(int(round(x_count_f)), 1)
    y_count = max(int(round(y_count_f)), 1)
    xs = torch.linspace(float(x_min), float(x_max), x_count).tolist()
    ys = torch.linspace(float(y_min), float(y_max), y_count).tolist()
    return [(float(x), float(y), float(z)) for y in ys for x in xs]


def _env_metrics(env, metric_max: dict[str, torch.Tensor]) -> list[dict]:
    base = env.unwrapped
    base._compute_intermediate_values()
    metrics = []
    origins = base.scene.env_origins
    for env_id in range(base.num_envs):
        metrics.append(
            {
                "env_id": env_id,
                "object_pos_local": [
                    float(v) for v in (base._object_pos_w[env_id] - origins[env_id]).detach().cpu()
                ],
                "palm_pos_local": [float(v) for v in (base._palm_pos_w[env_id] - origins[env_id]).detach().cpu()],
                "min_surface_dist": float(torch.min(base._surface_dist[env_id]).detach().cpu()),
                "mean_surface_dist": float(torch.mean(base._surface_dist[env_id]).detach().cpu()),
                "tabletop_arm_clearance_ok": bool(base._tabletop_arm_clearance_ok[env_id].detach().cpu()),
                "tabletop_arm_clearance_penalty": float(base._tabletop_arm_clearance_penalty[env_id].detach().cpu()),
                "tabletop_arm_clearance_min_margin": float(
                    base._tabletop_arm_clearance_min_margin[env_id].detach().cpu()
                ),
                "true_grasp": bool(base._true_grasp[env_id].detach().cpu()),
                "lifted": bool(base._lifted[env_id].detach().cpu()),
                "stable_hold": bool(base._stable_hold[env_id].detach().cpu()),
                "max_true_grasp": float(metric_max["true_grasp"][env_id].detach().cpu()),
                "max_lifted": float(metric_max["lifted"][env_id].detach().cpu()),
                "max_stable_hold": float(metric_max["stable_hold"][env_id].detach().cpu()),
                "max_success": float(metric_max["success"][env_id].detach().cpu()),
            }
        )
    return metrics


def main() -> None:
    carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.seed is not None:
        env_cfg.seed = int(args_cli.seed)
    inspire_target_map = {
        "safe": INSPIRE_ANYDEX_SPHERE_SAFE_CLOSE_TARGETS,
        "sphere4cm": INSPIRE_ANYDEX_SPHERE_4CM_CLOSE_TARGETS,
        "p80": INSPIRE_ANYDEX_P80_CLOSE_TARGETS,
    }
    if args_cli.inspire_close_target_values is not None and hasattr(env_cfg, "inspire_semantic_close_targets"):
        env_cfg.inspire_semantic_close_targets = tuple(float(v) for v in args_cli.inspire_close_target_values)
    elif args_cli.inspire_close_target != "cfg" and hasattr(env_cfg, "inspire_semantic_close_targets"):
        env_cfg.inspire_semantic_close_targets = inspire_target_map[args_cli.inspire_close_target]
    if args_cli.reference_hand_fractions is not None and hasattr(env_cfg, "reference_hand_fractions"):
        env_cfg.reference_hand_fractions = tuple(float(v) for v in args_cli.reference_hand_fractions)
    if args_cli.contact_distance is not None and hasattr(env_cfg, "contact_distance"):
        env_cfg.contact_distance = float(args_cli.contact_distance)
    if args_cli.strict_contact_distance is not None and hasattr(env_cfg, "strict_success_contact_distance"):
        env_cfg.strict_success_contact_distance = float(args_cli.strict_contact_distance)
    if args_cli.static_tabletop:
        if hasattr(env_cfg, "reset_object_pos_noise"):
            env_cfg.reset_object_pos_noise = (0.0, 0.0, 0.0)
        if hasattr(env_cfg, "dynamic_tabletop_persistent_motion"):
            env_cfg.dynamic_tabletop_persistent_motion = False
        if hasattr(env_cfg, "dynamic_tabletop_start_speed_range"):
            env_cfg.dynamic_tabletop_start_speed_range = (0.0, 0.0)
        if hasattr(env_cfg, "dynamic_tabletop_initial_speed_range"):
            env_cfg.dynamic_tabletop_initial_speed_range = (0.0, 0.0)
        if hasattr(env_cfg, "dynamic_tabletop_start_yaw_rate_range"):
            env_cfg.dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
        if hasattr(env_cfg, "dynamic_tabletop_initial_yaw_rate_range"):
            env_cfg.dynamic_tabletop_initial_yaw_rate_range = (0.0, 0.0)
        if hasattr(env_cfg, "dynamic_grasp_speed_curriculum"):
            env_cfg.dynamic_grasp_speed_curriculum = False
        if hasattr(env_cfg, "tabletop_asset_curriculum"):
            env_cfg.tabletop_asset_curriculum = False
        if hasattr(env_cfg, "tabletop_asset_curriculum_override_alpha"):
            env_cfg.tabletop_asset_curriculum_override_alpha = 0.0
        if hasattr(env_cfg, "tabletop_asset_curriculum_start_count"):
            env_cfg.tabletop_asset_curriculum_start_count = 1
        for name in (
            "scripted_action_prior_enabled",
            "scripted_tabletop_approach_action_prior_enabled",
            "scripted_tabletop_pregrasp_prior_enabled",
            "scripted_tabletop_lift_target_prior_enabled",
            "scripted_tabletop_hand_grasp_memory_prior_enabled",
        ):
            if hasattr(env_cfg, name):
                setattr(env_cfg, name, False)
        if hasattr(env_cfg, "terminate_on_success"):
            env_cfg.terminate_on_success = False
        if hasattr(env_cfg, "episode_length_s"):
            env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), 30.0)
        if hasattr(env_cfg, "workspace_xy_limit"):
            env_cfg.workspace_xy_limit = max(float(env_cfg.workspace_xy_limit), 2.0)
        if args_cli.pregrasp_arm_pos is not None and hasattr(env_cfg, "arm_target_clamp_delta"):
            env_cfg.arm_target_clamp_delta = (3.20,) * 7
    table_top_z = args_cli.table_top_z
    grid_positions = _grid_positions(args_cli.object_pos_grid) if args_cli.object_pos_grid is not None else None
    representative_object_pos = grid_positions[0] if grid_positions else args_cli.object_pos
    if representative_object_pos is not None:
        object_pos = tuple(float(v) for v in representative_object_pos)
        env_cfg.object_start_pos = object_pos
        env_cfg.object_cfg.init_state.pos = object_pos
        env_cfg.workspace_xy_limit = max(
            float(env_cfg.workspace_xy_limit),
            max(abs(object_pos[0]), abs(object_pos[1])) + 0.30,
        )
        if table_top_z is None:
            if env_cfg.object_shape == "sphere":
                table_top_z = object_pos[2] - float(env_cfg.object_radius)
            else:
                table_top_z = object_pos[2] - 0.5 * float(env_cfg.object_size[2])
    if table_top_z is not None:
        table_xy = args_cli.table_pos_xy
        if table_xy is None and representative_object_pos is not None:
            table_xy = representative_object_pos[:2]
        if table_xy is None:
            table_xy = env_cfg.table_cfg.init_state.pos[:2]
        env_cfg.table_top_z = float(table_top_z)
        env_cfg.table_cfg.init_state.pos = (float(table_xy[0]), float(table_xy[1]), float(table_top_z) - 0.0225)
    if args_cli.video_path and hasattr(env_cfg, "video_camera_enabled"):
        env_cfg.video_camera_enabled = True
        env_cfg.video_camera.data_types = ["rgb"]
        env_cfg.video_camera.spawn.focal_length = float(args_cli.video_camera_focal_length)
        env_cfg.video_camera.width = int(args_cli.video_camera_resolution[0])
        env_cfg.video_camera.height = int(args_cli.video_camera_resolution[1])
    render_mode = "rgb_array" if args_cli.video_path else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video_path:
        _set_video_camera_pose(env.unwrapped)

    try:
        obs, _ = env.reset()
        del obs
        if args_cli.video_path:
            _set_video_camera_pose(env.unwrapped)
        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs
        object_positions_tensor = None
        if grid_positions is not None:
            object_positions_tensor = torch.tensor(grid_positions, dtype=torch.float32, device=device)
            _reset_object_pose(env, object_positions_tensor)
            if args_cli.video_path:
                _set_video_camera_pose(env.unwrapped)
        actions = torch.zeros((num_envs, env.unwrapped.cfg.action_space), device=device)
        success_any = torch.zeros(num_envs, dtype=torch.bool, device=device)
        metric_max = {
            "success": torch.zeros(num_envs, device=device),
            "true_grasp": torch.zeros(num_envs, device=device),
            "lifted": torch.zeros(num_envs, device=device),
            "stable_hold": torch.zeros(num_envs, device=device),
        }
        frames = []

        dt = env.unwrapped.step_dt
        scale = env.unwrapped.cfg.arm_action_scale
        effective_arm_gain = max(scale * env.unwrapped.cfg.arm_moving_average, 1.0e-6)
        default_lift_delta = [-0.005224, -0.066232, 0.130747, 0.127955, 0.004971, -0.050877, 0.206977]
        lift_delta_values = list(args_cli.lift_deltas) if args_cli.lift_deltas is not None else default_lift_delta
        if args_cli.lift_joint2_delta is not None:
            lift_delta_values[1] = args_cli.lift_joint2_delta
        if args_cli.lift_joint4_delta is not None:
            lift_delta_values[3] = args_cli.lift_joint4_delta
        if args_cli.lift_joint6_delta is not None:
            lift_delta_values[5] = args_cli.lift_joint6_delta
        lift_delta = torch.tensor(lift_delta_values, device=device)
        lift_action = torch.clamp(
            lift_delta / max(effective_arm_gain * dt * max(args_cli.lift_steps, 1), 1.0e-6),
            -1.0,
            1.0,
        )
        arm_ids = env.unwrapped._arm_joint_ids
        pregrasp_target = None
        close_target = None
        lift_target = None
        if args_cli.pregrasp_arm_pos is not None:
            pregrasp_target = torch.tensor(args_cli.pregrasp_arm_pos, device=device).view(1, 7)
        if args_cli.close_arm_pos is not None:
            close_target = torch.tensor(args_cli.close_arm_pos, device=device).view(1, 7)
        lift_base_target = close_target if close_target is not None else pregrasp_target
        if lift_base_target is not None:
            lift_target = lift_base_target + lift_delta.view(1, 7)

        def arm_action_to_target(target: torch.Tensor, remaining_steps: int) -> torch.Tensor:
            current = env.unwrapped._joint_targets[:, arm_ids]
            lower = env.unwrapped._joint_lower_limits[arm_ids].unsqueeze(0)
            upper = env.unwrapped._joint_upper_limits[arm_ids].unsqueeze(0)
            center = torch.clamp(env.unwrapped._default_joint_pos[:, arm_ids], lower, upper)
            target_clamped = torch.clamp(target.expand(num_envs, -1), lower, upper)
            if env.unwrapped.cfg.policy_action_interface == "joint_target":
                desired_next = current + (target_clamped - current) / float(max(remaining_steps, 1))
                arm_ma = max(float(env.unwrapped.cfg.arm_moving_average), 1.0e-6)
                raw_target = (desired_next - (1.0 - arm_ma) * current) / arm_ma
                raw_target = torch.clamp(raw_target, lower, upper)
                positive_span = torch.clamp(upper - center, min=1.0e-6)
                negative_span = torch.clamp(center - lower, min=1.0e-6)
                delta = raw_target - center
                action = torch.where(delta >= 0.0, delta / positive_span, delta / negative_span)
            else:
                delta = target_clamped - current
                denom = max(effective_arm_gain * dt * max(remaining_steps, 1), 1.0e-6)
                action = delta / denom
            return torch.clamp(action, -1.0, 1.0)

        def arm_pos_to_normalized_action(target: torch.Tensor) -> torch.Tensor:
            lower = env.unwrapped._joint_lower_limits[arm_ids].unsqueeze(0)
            upper = env.unwrapped._joint_upper_limits[arm_ids].unsqueeze(0)
            center = torch.clamp(env.unwrapped._default_joint_pos[:, arm_ids], lower, upper)
            target = torch.clamp(target.expand(num_envs, -1), lower, upper)
            delta = target - center
            positive_span = torch.clamp(upper - center, min=1.0e-6)
            negative_span = torch.clamp(center - lower, min=1.0e-6)
            action = torch.where(delta >= 0.0, delta / positive_span, delta / negative_span)
            return torch.clamp(action, -1.0, 1.0)

        def hand_targets_to_normalized_action(target_values: tuple[float, ...]) -> torch.Tensor:
            hand_ids = env.unwrapped._control_hand_joint_ids
            lower = env.unwrapped._joint_lower_limits[hand_ids]
            upper = env.unwrapped._joint_upper_limits[hand_ids]
            center = torch.clamp(env.unwrapped._default_joint_pos[:, hand_ids], lower.unsqueeze(0), upper.unsqueeze(0))[0]
            target = torch.tensor(target_values, dtype=torch.float32, device=device)
            if target.numel() != len(hand_ids):
                raise ValueError(f"Expected {len(hand_ids)} Inspire hand targets, got {target.numel()}.")
            target = torch.clamp(target, lower, upper)
            if env.unwrapped.cfg.policy_action_interface == "joint_target":
                delta = target - center
                positive_span = torch.clamp(upper - center, min=1.0e-6)
                negative_span = torch.clamp(center - lower, min=1.0e-6)
                action = torch.where(delta >= 0.0, delta / positive_span, delta / negative_span)
            else:
                action = 2.0 * (target - lower) / torch.clamp(upper - lower, min=1.0e-6) - 1.0
            return torch.clamp(action, -1.0, 1.0)

        pregrasp_arm_action = None
        close_arm_action = None
        lift_arm_action = None
        if pregrasp_target is not None and args_cli.pregrasp_arm_pos_mode == "normalized-action":
            pregrasp_arm_action = arm_pos_to_normalized_action(pregrasp_target)
        if close_target is not None and args_cli.close_arm_pos_mode == "normalized-action":
            close_arm_action = arm_pos_to_normalized_action(close_target)
        if lift_target is not None and args_cli.hold_arm_mode == "lift-target" and (
            (close_target is not None and args_cli.close_arm_pos_mode == "normalized-action")
            or (close_target is None and args_cli.pregrasp_arm_pos_mode == "normalized-action")
        ):
            lift_arm_action = arm_pos_to_normalized_action(lift_target)

        hand_action_dim = int(env.unwrapped.cfg.action_space) - 7
        physical_close_target_values = None
        uses_active_hand = env.unwrapped._uses_active_hand_actions()
        if not uses_active_hand:
            if args_cli.inspire_close_target_values is not None:
                physical_close_target_values = tuple(float(v) for v in args_cli.inspire_close_target_values)
            elif args_cli.inspire_close_target != "cfg":
                physical_close_target_values = tuple(float(v) for v in inspire_target_map[args_cli.inspire_close_target])
        if args_cli.hand_action_vector is not None:
            close_hand_action = torch.tensor(args_cli.hand_action_vector, device=device, dtype=torch.float32)
            if close_hand_action.numel() != hand_action_dim:
                raise ValueError(f"Expected {hand_action_dim} hand action values, got {close_hand_action.numel()}.")
        elif physical_close_target_values is not None:
            close_hand_action = hand_targets_to_normalized_action(physical_close_target_values)
        else:
            close_hand_action = torch.full((hand_action_dim,), args_cli.hand_action, device=device)
        close_hand_action = torch.clamp(close_hand_action, -1.0, 1.0)
        open_hand_action = torch.full((hand_action_dim,), -1.0, device=device)

        phases = (
            (
                "pre",
                args_cli.pre_steps,
                torch.cat((torch.zeros(7, device=device), open_hand_action)).repeat(num_envs, 1),
            ),
            ("close", args_cli.close_steps, torch.cat((torch.zeros(7, device=device), close_hand_action)).repeat(num_envs, 1)),
            ("lift", args_cli.lift_steps, torch.cat((lift_action, close_hand_action)).repeat(num_envs, 1)),
            ("hold", args_cli.hold_steps, torch.cat((torch.zeros(7, device=device), close_hand_action)).repeat(num_envs, 1)),
        )

        step_count = 0
        last_extras = {}
        phase_logs = {}
        phase_geometry = {}
        for phase_name, phase_steps, phase_actions in phases:
            _trace(f"phase={phase_name} steps={phase_steps}")
            for phase_step in range(phase_steps):
                if args_cli.video_path and step_count % max(args_cli.video_stride, 1) == 0:
                    _record_frame(env, frames)
                step_actions = phase_actions
                remaining = phase_steps - phase_step
                if pregrasp_arm_action is not None and phase_name == "pre":
                    step_actions = phase_actions.clone()
                    step_actions[:, :7] = pregrasp_arm_action
                elif close_arm_action is not None and phase_name == "close":
                    step_actions = phase_actions.clone()
                    step_actions[:, :7] = close_arm_action
                elif lift_arm_action is not None and (
                    phase_name == "lift" or (phase_name == "hold" and args_cli.hold_arm_mode == "lift-target")
                ):
                    step_actions = phase_actions.clone()
                    step_actions[:, :7] = lift_arm_action
                elif pregrasp_target is not None and phase_name == "pre":
                    step_actions = phase_actions.clone()
                    step_actions[:, :7] = arm_action_to_target(pregrasp_target, remaining)
                elif close_target is not None and phase_name == "close":
                    step_actions = phase_actions.clone()
                    step_actions[:, :7] = arm_action_to_target(close_target, remaining)
                elif lift_target is not None and (
                    phase_name == "lift" or (phase_name == "hold" and args_cli.hold_arm_mode == "lift-target")
                ):
                    step_actions = phase_actions.clone()
                    step_actions[:, :7] = arm_action_to_target(lift_target, remaining)
                _, last_extras = _step(env, step_actions, success_any, metric_max)
                step_count += 1
            phase_logs[phase_name] = _mean_log(last_extras)
            phase_geometry[phase_name] = _geometry_debug(env)
            if phase_name == "pre" and args_cli.reset_object_after_pre and representative_object_pos is not None:
                if object_positions_tensor is not None:
                    _reset_object_pose(env, object_positions_tensor)
                else:
                    _reset_object_pose(env, tuple(float(v) for v in representative_object_pos))
                if args_cli.video_path:
                    _set_video_camera_pose(env.unwrapped)
                phase_geometry["pre_after_object_reset"] = _geometry_debug(env)

        if args_cli.video_path:
            _record_frame(env, frames)
            video_path = Path(args_cli.video_path).expanduser().resolve()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_path, frames, fps=args_cli.video_fps, macro_block_size=16)
            _trace(f"video={video_path}")

        summary = {
            "task": args_cli.task,
            "num_envs": int(num_envs),
            "steps": int(step_count),
            "success_rate": float(success_any.float().mean().item()),
            "success_count": int(success_any.sum().item()),
            "max_true_grasp": float(metric_max["true_grasp"].mean().item()),
            "max_lifted": float(metric_max["lifted"].mean().item()),
            "max_stable_hold": float(metric_max["stable_hold"].mean().item()),
            "phase_logs": phase_logs,
            "phase_geometry": phase_geometry,
            "last_log": _mean_log(last_extras),
            "geometry_debug": _geometry_debug(env),
            "pregrasp_arm_pos_mode": args_cli.pregrasp_arm_pos_mode,
            "close_arm_pos_mode": args_cli.close_arm_pos_mode,
            "hold_arm_mode": args_cli.hold_arm_mode,
            "seed": args_cli.seed,
            "pregrasp_arm_action_env0": (
                [float(v) for v in pregrasp_arm_action[0].detach().cpu()] if pregrasp_arm_action is not None else None
            ),
            "close_arm_action_env0": (
                [float(v) for v in close_arm_action[0].detach().cpu()] if close_arm_action is not None else None
            ),
            "lift_arm_action_env0": (
                [float(v) for v in lift_arm_action[0].detach().cpu()] if lift_arm_action is not None else None
            ),
            "override_object_pos": list(representative_object_pos) if representative_object_pos is not None else None,
            "object_pos_grid": grid_positions,
            "hand_action_vector": [float(v) for v in close_hand_action.detach().cpu()],
            "env_metrics": _env_metrics(env, metric_max),
            "override_table_top_z": table_top_z,
            "video_path": str(Path(args_cli.video_path).expanduser().resolve()) if args_cli.video_path else None,
        }
        _trace(json.dumps(summary, indent=2, sort_keys=True))
        if args_cli.output_json:
            output_json = Path(args_cli.output_json).expanduser().resolve()
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
