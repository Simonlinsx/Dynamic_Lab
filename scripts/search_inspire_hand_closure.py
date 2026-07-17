#!/usr/bin/env python3
"""Search 6-active-DOF Inspire hand closure vectors for a fixed tabletop sphere pose."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import traceback
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from isaaclab.app import AppLauncher

DEFAULT_TASK = "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereApproachBootstrap-Teacher-Direct-v0"
DEFAULT_PREGRASP = (0.0, -0.35, 0.0, -2.20, 0.0, 2.35, 0.7853981633974483)
DEFAULT_LIFT_DELTA = (-0.005224, -0.066232, 0.130747, 0.127955, 0.004971, -0.050877, 0.206977)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--pre-steps", type=int, default=120)
parser.add_argument("--close-steps", type=int, default=130)
parser.add_argument("--lift-steps", type=int, default=80)
parser.add_argument("--hold-steps", type=int, default=80)
parser.add_argument("--pregrasp-arm-pos", type=float, nargs=7, default=DEFAULT_PREGRASP)
parser.add_argument(
    "--pregrasp-joint-sweep",
    type=float,
    nargs=3,
    default=None,
    metavar=("JOINT_ID", "MIN", "MAX"),
    help="Sweep one 1-based Franka joint target uniformly across environments.",
)
parser.add_argument(
    "--close-arm-pos",
    type=float,
    nargs=7,
    default=None,
    help="Optional arm target used during close/lift. If omitted, close/lift use --pregrasp-arm-pos.",
)
parser.add_argument("--lift-deltas", type=float, nargs=7, default=DEFAULT_LIFT_DELTA)
parser.add_argument(
    "--lift-delta-candidate",
    action="append",
    type=float,
    nargs=7,
    default=[],
    metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    help="Explicit Franka joint-target lift delta; repeat to assign candidates across environments.",
)
parser.add_argument("--object-pos", type=float, nargs=3, default=(0.58, 0.0, 0.320))
parser.add_argument(
    "--object-radius",
    type=float,
    default=None,
    help="Override the primitive sphere radius for size-calibration probes.",
)
parser.add_argument(
    "--object-mass",
    type=float,
    default=None,
    help="Override object mass in kilograms for load-bearing probes.",
)
parser.add_argument(
    "--object-pos-grid",
    type=float,
    nargs=7,
    default=None,
    metavar=("X_MIN", "X_MAX", "X_COUNT", "Y_MIN", "Y_MAX", "Y_COUNT", "Z"),
    help="Assign per-env object positions on an XY grid at fixed Z.",
)
parser.add_argument("--table-top-z", type=float, default=0.298)
parser.add_argument("--table-pos-xy", type=float, nargs=2, default=(0.58, 0.0))
parser.add_argument(
    "--inspire-close-target",
    choices=("cfg", "safe", "sphere4cm", "p80"),
    default="cfg",
)
parser.add_argument(
    "--inspire-close-target-values",
    type=float,
    nargs="+",
    default=None,
    help="Explicit Inspire close targets in the task's physical hand-control joint order.",
)
parser.add_argument("--contact-distance", type=float, default=0.020)
parser.add_argument("--strict-contact-distance", type=float, default=0.008)
parser.add_argument("--hand-moving-average", type=float, default=None)
parser.add_argument("--hand-effort-limit", type=float, default=None)
parser.add_argument("--hand-stiffness", type=float, default=None)
parser.add_argument("--hand-damping", type=float, default=None)
parser.add_argument(
    "--disable-self-collision",
    action="store_true",
    help="Diagnostic only: disable articulation self-collision to isolate mimic kinematics.",
)
parser.add_argument(
    "--report-self-contacts",
    action="store_true",
    help="Collect PhysX contact-report pairs where both actors belong to env_0/Robot.",
)
parser.add_argument(
    "--extra-self-collision-filter-pair",
    action="append",
    nargs=2,
    default=[],
    metavar=("LINK_A", "LINK_B"),
    help="Diagnostic-only self-collision pair to filter; repeat for multiple pairs.",
)
parser.add_argument(
    "--fixed-hand-fraction",
    type=float,
    nargs="+",
    default=None,
    help="Repeat one hand fraction instead of sweeping closure candidates. Accepts semantic 6D or physical action-dim values.",
)
parser.add_argument(
    "--hand-fraction-candidate",
    action="append",
    type=float,
    nargs=6,
    default=[],
    metavar=("THUMB_YAW", "THUMB_FLEX", "INDEX", "MIDDLE", "RING", "PINKY"),
    help="Explicit semantic 6-motor close candidate; repeat to assign candidates across environments.",
)
parser.add_argument(
    "--pregrasp-hand-fraction",
    type=float,
    nargs="+",
    default=None,
    help=(
        "Optional hand fraction held before the object is inserted. This reproduces "
        "hardware grasp tables that configure a preshape before closing selected fingers."
    ),
)
parser.add_argument(
    "--progressive-close",
    action="store_true",
    help="Linearly interpolate from --pregrasp-hand-fraction to the requested close target.",
)
parser.add_argument(
    "--force-limited-close",
    action="store_true",
    help=(
        "Enable progressive closing and hold each finger motor when its real mesh "
        "reports object contact."
    ),
)
parser.add_argument(
    "--finger-stop-force",
    type=float,
    default=0.50,
    help="Object contact force in newtons that stops one closing finger motor.",
)
parser.add_argument(
    "--finger-stop-consecutive-steps",
    type=int,
    default=2,
    help="Consecutive contact steps required before a finger motor is held.",
)
parser.add_argument("--video-path", default=None, help="Optional fixed-third-view MP4.")
parser.add_argument("--video-env-id", type=int, default=0, help="Environment index recorded by --video-path.")
parser.add_argument("--video-stride", type=int, default=2)
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--video-camera-eye", type=float, nargs=3, default=(1.25, -1.05, 0.72))
parser.add_argument("--video-camera-target", type=float, nargs=3, default=(0.58, 0.0, 0.37))
parser.add_argument("--video-camera-focal-length", type=float, default=24.0)
parser.add_argument("--video-camera-resolution", type=int, nargs=2, default=(960, 544))
parser.add_argument("--top-k", type=int, default=24)
parser.add_argument("--output-json", required=True)
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
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from pxr import PhysicsSchemaTools, PhysxSchema  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402
from simtoolreal_lab.tasks.dynamic_dexterous_grasp.dynamic_dexterous_grasp_env_cfg import (  # noqa: E402
    INSPIRE_ANYDEX_P80_CLOSE_TARGETS,
    INSPIRE_ANYDEX_SPHERE_4CM_CLOSE_TARGETS,
    INSPIRE_ANYDEX_SPHERE_SAFE_CLOSE_TARGETS,
)


def _trace(message: str) -> None:
    print(f"[INSPIRE-CLOSURE] {message}", flush=True)


def _set_static_tabletop_cfg(env_cfg) -> None:
    env_cfg.seed = int(args_cli.seed)
    if args_cli.disable_self_collision:
        env_cfg.robot_cfg.spawn.self_collision = False
        env_cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions = False
    if hasattr(env_cfg, "robot_extra_self_collision_filter_pairs"):
        env_cfg.robot_extra_self_collision_filter_pairs = tuple(
            (str(link_a), str(link_b))
            for link_a, link_b in args_cli.extra_self_collision_filter_pair
        )
    if args_cli.inspire_close_target_values is not None and hasattr(env_cfg, "inspire_semantic_close_targets"):
        env_cfg.inspire_semantic_close_targets = tuple(float(v) for v in args_cli.inspire_close_target_values)
    elif args_cli.inspire_close_target != "cfg" and hasattr(env_cfg, "inspire_semantic_close_targets"):
        target_map = {
            "safe": INSPIRE_ANYDEX_SPHERE_SAFE_CLOSE_TARGETS,
            "sphere4cm": INSPIRE_ANYDEX_SPHERE_4CM_CLOSE_TARGETS,
            "p80": INSPIRE_ANYDEX_P80_CLOSE_TARGETS,
        }
        env_cfg.inspire_semantic_close_targets = target_map[args_cli.inspire_close_target]
    if args_cli.hand_moving_average is not None and hasattr(env_cfg, "hand_moving_average"):
        env_cfg.hand_moving_average = float(args_cli.hand_moving_average)
    for actuator_name, actuator_cfg in env_cfg.robot_cfg.actuators.items():
        if not actuator_name.startswith("inspire_"):
            continue
        if args_cli.hand_effort_limit is not None:
            actuator_cfg.effort_limit_sim = float(args_cli.hand_effort_limit)
        if args_cli.hand_stiffness is not None:
            actuator_cfg.stiffness = float(args_cli.hand_stiffness)
        if args_cli.hand_damping is not None:
            actuator_cfg.damping = float(args_cli.hand_damping)
    for name in (
        "scripted_action_prior_enabled",
        "scripted_tabletop_pregrasp_prior_enabled",
        "scripted_tabletop_lift_target_prior_enabled",
        "scripted_tabletop_hand_grasp_memory_prior_enabled",
        "dynamic_tabletop_persistent_motion",
        "dynamic_grasp_speed_curriculum",
        "tabletop_asset_curriculum",
    ):
        if hasattr(env_cfg, name):
            setattr(env_cfg, name, False)
    for name in (
        "dynamic_tabletop_start_speed_range",
        "dynamic_tabletop_initial_speed_range",
        "dynamic_tabletop_start_yaw_rate_range",
        "dynamic_tabletop_initial_yaw_rate_range",
    ):
        if hasattr(env_cfg, name):
            setattr(env_cfg, name, (0.0, 0.0))
    if hasattr(env_cfg, "reset_object_pos_noise"):
        env_cfg.reset_object_pos_noise = (0.0, 0.0, 0.0)
    if hasattr(env_cfg, "terminate_on_success"):
        env_cfg.terminate_on_success = False
    if hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), 30.0)
    if hasattr(env_cfg, "workspace_xy_limit"):
        env_cfg.workspace_xy_limit = max(float(env_cfg.workspace_xy_limit), 2.0)
    if hasattr(env_cfg, "arm_target_clamp_delta"):
        env_cfg.arm_target_clamp_delta = (3.20,) * 7
    if hasattr(env_cfg, "contact_distance"):
        env_cfg.contact_distance = float(args_cli.contact_distance)
    if hasattr(env_cfg, "strict_success_contact_distance"):
        env_cfg.strict_success_contact_distance = float(args_cli.strict_contact_distance)
    if args_cli.object_radius is not None:
        object_radius = float(args_cli.object_radius)
        if object_radius <= 0.0:
            raise ValueError("--object-radius must be positive.")
        spawn = getattr(getattr(env_cfg, "object_cfg", None), "spawn", None)
        if spawn is None or not hasattr(spawn, "radius"):
            raise ValueError("--object-radius requires a primitive object spawn with a radius field.")
        spawn.radius = object_radius
        env_cfg.object_radius = object_radius
        env_cfg.object_size = (2.0 * object_radius,) * 3
    if args_cli.object_mass is not None:
        object_mass = float(args_cli.object_mass)
        if object_mass <= 0.0:
            raise ValueError("--object-mass must be positive.")
        spawn = getattr(getattr(env_cfg, "object_cfg", None), "spawn", None)
        mass_props = getattr(spawn, "mass_props", None)
        if mass_props is None or not hasattr(mass_props, "mass"):
            raise ValueError("--object-mass requires object spawn mass properties.")
        mass_props.mass = object_mass
    object_pos = _grid_positions(args_cli.object_pos_grid)[0] if args_cli.object_pos_grid is not None else args_cli.object_pos
    env_cfg.object_start_pos = tuple(float(v) for v in object_pos)
    env_cfg.object_cfg.init_state.pos = env_cfg.object_start_pos
    env_cfg.table_top_z = float(args_cli.table_top_z)
    env_cfg.table_cfg.init_state.pos = (
        float(args_cli.table_pos_xy[0]),
        float(args_cli.table_pos_xy[1]),
        float(args_cli.table_top_z) - 0.0225,
    )
    if args_cli.video_path and hasattr(env_cfg, "video_camera_enabled"):
        env_cfg.video_camera_enabled = True
        env_cfg.video_camera.data_types = ["rgb"]
        env_cfg.video_camera.spawn.focal_length = float(args_cli.video_camera_focal_length)
        env_cfg.video_camera.width = int(args_cli.video_camera_resolution[0])
        env_cfg.video_camera.height = int(args_cli.video_camera_resolution[1])


def _grid_positions(values: list[float] | tuple[float, ...] | None) -> list[tuple[float, float, float]]:
    if values is None:
        return []
    x_min, x_max, x_count_f, y_min, y_max, y_count_f, z = values
    x_count = max(int(round(x_count_f)), 1)
    y_count = max(int(round(y_count_f)), 1)
    xs = torch.linspace(float(x_min), float(x_max), x_count).tolist()
    ys = torch.linspace(float(y_min), float(y_max), y_count).tolist()
    return [(float(x), float(y), float(z)) for y in ys for x in xs]


def _configured_object_positions(base) -> torch.Tensor:
    if args_cli.object_pos_grid is not None:
        positions = torch.tensor(_grid_positions(args_cli.object_pos_grid), dtype=torch.float32, device=base.device)
        repeat_count = (base.num_envs + positions.shape[0] - 1) // positions.shape[0]
        return positions.repeat((repeat_count, 1))[: base.num_envs]
    return torch.tensor(args_cli.object_pos, dtype=torch.float32, device=base.device).view(1, 3).expand(base.num_envs, -1)


def _configured_pregrasp_targets(base) -> torch.Tensor:
    targets = torch.tensor(
        args_cli.pregrasp_arm_pos, dtype=torch.float32, device=base.device
    ).view(1, 7).expand(base.num_envs, -1).clone()
    if args_cli.pregrasp_joint_sweep is None:
        return targets
    joint_id_f, min_value, max_value = args_cli.pregrasp_joint_sweep
    joint_id = int(round(joint_id_f))
    if joint_id < 1 or joint_id > 7 or abs(joint_id_f - joint_id) > 1.0e-6:
        raise ValueError("--pregrasp-joint-sweep JOINT_ID must be an integer from 1 to 7.")
    targets[:, joint_id - 1] = torch.linspace(
        float(min_value), float(max_value), base.num_envs, device=base.device
    )
    return targets


def _reset_object_pose(env, positions: torch.Tensor) -> None:
    base = env.unwrapped
    env_ids = torch.arange(base.num_envs, device=base.device, dtype=torch.long)
    if hasattr(base, "_object_start_pos"):
        base._object_start_pos = positions.clone()
    state = base.object.data.default_root_state.clone()
    state[:, 0:3] = base.scene.env_origins + positions
    state[:, 3:7] = torch.tensor(
        getattr(base.cfg, "object_start_rot", (1.0, 0.0, 0.0, 0.0)),
        dtype=torch.float32,
        device=base.device,
    ).view(1, 4).expand(base.num_envs, -1)
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


def _semantic_to_physical_fractions(semantic: torch.Tensor, joint_names: list[str]) -> torch.Tensor:
    values = {
        "thumb_proximal_yaw_joint": semantic[:, 0],
        "thumb_proximal_pitch_joint": semantic[:, 1],
        "thumb_intermediate_joint": semantic[:, 1],
        "thumb_distal_joint": semantic[:, 1],
        "index_proximal_joint": semantic[:, 2],
        "index_intermediate_joint": semantic[:, 2],
        "middle_proximal_joint": semantic[:, 3],
        "middle_intermediate_joint": semantic[:, 3],
        "ring_proximal_joint": semantic[:, 4],
        "ring_intermediate_joint": semantic[:, 4],
        "pinky_proximal_joint": semantic[:, 5],
        "pinky_intermediate_joint": semantic[:, 5],
    }
    return torch.stack([values[name] for name in joint_names], dim=-1)


def _candidate_fractions(count: int, seed: int, hand_dim: int, joint_names: list[str]) -> torch.Tensor:
    if args_cli.fixed_hand_fraction is not None:
        return _repeat_hand_fraction(
            args_cli.fixed_hand_fraction,
            count,
            hand_dim,
            joint_names,
            argument_name="--fixed-hand-fraction",
        )
    if args_cli.hand_fraction_candidate:
        semantic = torch.tensor(args_cli.hand_fraction_candidate, dtype=torch.float32)
        repeat_count = (count + semantic.shape[0] - 1) // semantic.shape[0]
        semantic = semantic.repeat((repeat_count, 1))[:count]
        semantic = torch.clamp(semantic, 0.0, 1.0)
        return semantic if hand_dim == 6 else _semantic_to_physical_fractions(semantic, joint_names)
    thumb_yaws = (0.0, 0.25, 0.50, 0.75, 1.0)
    thumb_flexes = (0.20, 0.45, 0.70, 0.90, 1.0)
    finger_levels = (0.45, 0.65, 0.85, 1.0)
    rows: list[tuple[float, float, float, float, float, float]] = []
    for yaw, flex, common in itertools.product(thumb_yaws, thumb_flexes, finger_levels):
        rows.append((yaw, flex, common, common, common, common))
    for yaw, flex, idx_mid, ring_pinky in itertools.product(
        (0.0, 0.35, 0.70, 1.0),
        (0.45, 0.70, 0.90, 1.0),
        (0.55, 0.80, 1.0),
        (0.55, 0.85, 1.0),
    ):
        rows.append((yaw, flex, idx_mid, idx_mid, ring_pinky, ring_pinky))
    base = torch.tensor(rows, dtype=torch.float32)
    if base.shape[0] >= count:
        semantic = base[:count]
        return semantic if hand_dim == 6 else _semantic_to_physical_fractions(semantic, joint_names)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    extra = torch.rand((count - base.shape[0], 6), generator=gen)
    mins = torch.tensor((0.0, 0.15, 0.30, 0.30, 0.30, 0.30), dtype=torch.float32)
    maxs = torch.ones(6, dtype=torch.float32)
    extra = mins + extra * (maxs - mins)
    semantic = torch.cat((base, extra), dim=0)[:count]
    return semantic if hand_dim == 6 else _semantic_to_physical_fractions(semantic, joint_names)


def _repeat_hand_fraction(
    values: list[float] | tuple[float, ...],
    count: int,
    hand_dim: int,
    joint_names: list[str],
    *,
    argument_name: str,
) -> torch.Tensor:
    fraction = torch.tensor(values, dtype=torch.float32).view(1, -1)
    if fraction.shape[-1] == 6 and hand_dim != 6:
        fraction = _semantic_to_physical_fractions(fraction, joint_names)
    if fraction.shape[-1] != hand_dim:
        raise ValueError(
            f"{argument_name} has {fraction.shape[-1]} values but this task expects {hand_dim}."
        )
    return torch.clamp(fraction, 0.0, 1.0).expand(count, -1).clone()


def _hand_fractions_to_actions(env, fractions: torch.Tensor) -> torch.Tensor:
    base = env.unwrapped
    if base._uses_active_hand_actions():
        reference = torch.tensor(base.cfg.reference_hand_fractions, dtype=torch.float32, device=base.device).view(1, -1)
        return torch.clamp(2.0 * fractions / torch.clamp(reference, min=1.0e-6) - 1.0, -1.0, 1.0)

    control_ids = base._control_hand_joint_ids
    lower = base._joint_lower_limits[control_ids].unsqueeze(0)
    upper = base._joint_upper_limits[control_ids].unsqueeze(0)
    center = torch.clamp(base._default_joint_pos[:, control_ids], lower, upper)
    close_cfg = getattr(base.cfg, "inspire_semantic_close_targets", None)
    if close_cfg is None:
        close_targets = upper.expand_as(center)
    else:
        close_targets = torch.tensor(close_cfg, dtype=torch.float32, device=base.device).view(1, -1)
        if close_targets.shape[-1] != len(control_ids):
            raise ValueError(
                "inspire_semantic_close_targets must match physical hand control joints for physical probing; "
                f"got {close_targets.shape[-1]} values for {len(control_ids)} joints."
            )
        close_targets = torch.clamp(close_targets.expand_as(center), lower, upper)
    targets = center + fractions.to(device=base.device) * (close_targets - center)
    positive_span = torch.clamp(upper - center, min=1.0e-6)
    negative_span = torch.clamp(center - lower, min=1.0e-6)
    delta = targets - center
    actions = torch.where(delta >= 0.0, delta / positive_span, delta / negative_span)
    return torch.clamp(actions, -1.0, 1.0)


def _zeros_stats(num_envs: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "success": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "true_grasp": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "strict_true_grasp": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "lifted": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "stable_hold": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "force_grasp": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "force_lifted": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "force_stable_hold": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "max_force_sum": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "thumb_contact": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "strict_thumb_contact": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "non_thumb_count": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "strict_non_thumb_count": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "finger_count": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "grasp_quality": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "weighted_opposition": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "object_height_delta": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "min_surface_dist": torch.full((num_envs,), float("inf"), dtype=torch.float32, device=device),
        "min_thumb_surface_dist": torch.full((num_envs,), float("inf"), dtype=torch.float32, device=device),
        "min_non_thumb_surface_dist": torch.full((num_envs,), float("inf"), dtype=torch.float32, device=device),
    }


def _update_stats(env, stats: dict[str, torch.Tensor], extras: dict | None = None) -> None:
    base = env.unwrapped
    base._compute_intermediate_values()
    if extras is not None and extras.get("success_env") is not None:
        stats["success"] = torch.maximum(stats["success"], extras["success_env"].float())
    if extras is not None:
        for name in ("force_grasp", "force_lifted", "force_stable_hold"):
            value = extras.get(f"{name}_env")
            if value is not None:
                stats[name] = torch.maximum(stats[name], value.float())
    force_per_tip = getattr(base, "_object_fingertip_contact_forces", None)
    if isinstance(force_per_tip, torch.Tensor):
        stats["max_force_sum"] = torch.maximum(stats["max_force_sum"], force_per_tip.sum(dim=-1))
    stats["true_grasp"] = torch.maximum(stats["true_grasp"], base._true_grasp.float())
    stats["strict_true_grasp"] = torch.maximum(stats["strict_true_grasp"], base._strict_true_grasp.float())
    stats["lifted"] = torch.maximum(stats["lifted"], base._lifted.float())
    stats["stable_hold"] = torch.maximum(stats["stable_hold"], base._stable_hold.float())
    stats["thumb_contact"] = torch.maximum(stats["thumb_contact"], base._thumb_contact.float())
    stats["strict_thumb_contact"] = torch.maximum(stats["strict_thumb_contact"], base._strict_thumb_contact.float())
    stats["non_thumb_count"] = torch.maximum(stats["non_thumb_count"], base._non_thumb_contact_count.float())
    stats["strict_non_thumb_count"] = torch.maximum(
        stats["strict_non_thumb_count"], base._strict_non_thumb_contact_count.float()
    )
    stats["finger_count"] = torch.maximum(stats["finger_count"], base._finger_contact_count.float())
    stats["grasp_quality"] = torch.maximum(stats["grasp_quality"], base._grasp_quality)
    stats["weighted_opposition"] = torch.maximum(stats["weighted_opposition"], base._weighted_opposition_score)
    stats["object_height_delta"] = torch.maximum(stats["object_height_delta"], base._object_height_delta)
    stats["min_surface_dist"] = torch.minimum(stats["min_surface_dist"], base._surface_dist.min(dim=-1).values)
    stats["min_thumb_surface_dist"] = torch.minimum(stats["min_thumb_surface_dist"], base._surface_dist[:, 0])
    stats["min_non_thumb_surface_dist"] = torch.minimum(
        stats["min_non_thumb_surface_dist"], base._surface_dist[:, 1:].min(dim=-1).values
    )


def _arm_action_to_target(env, target: torch.Tensor, remaining_steps: int) -> torch.Tensor:
    base = env.unwrapped
    arm_ids = base._arm_joint_ids
    current = base._joint_targets[:, arm_ids]
    lower = base._joint_lower_limits[arm_ids].unsqueeze(0)
    upper = base._joint_upper_limits[arm_ids].unsqueeze(0)
    center = torch.clamp(base._default_joint_pos[:, arm_ids], lower, upper)
    target = torch.clamp(target.expand(base.num_envs, -1), lower, upper)
    if base.cfg.policy_action_interface == "joint_target":
        desired_next = current + (target - current) / float(max(int(remaining_steps), 1))
        arm_ma = max(float(base.cfg.arm_moving_average), 1.0e-6)
        raw_target = (desired_next - (1.0 - arm_ma) * current) / arm_ma
        raw_target = torch.clamp(raw_target, lower, upper)
        positive_span = torch.clamp(upper - center, min=1.0e-6)
        negative_span = torch.clamp(center - lower, min=1.0e-6)
        delta = raw_target - center
        action = torch.where(delta >= 0.0, delta / positive_span, delta / negative_span)
    else:
        scale = float(base.cfg.arm_action_scale)
        gain = max(scale * float(base.cfg.arm_moving_average), 1.0e-6)
        denom = max(gain * float(base.dt) * max(int(remaining_steps), 1), 1.0e-6)
        action = (target - current) / denom
    return torch.clamp(action, -1.0, 1.0)


def _tensor_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def _set_video_camera_pose(base) -> bool:
    camera = getattr(base, "_video_camera", None)
    if camera is None:
        return False
    origins = base.scene.env_origins
    eye = torch.tensor(args_cli.video_camera_eye, device=base.device).view(1, 3)
    target = torch.tensor(args_cli.video_camera_target, device=base.device).view(1, 3)
    camera.set_world_poses_from_view(origins + eye, origins + target)
    if hasattr(camera, "_update_poses") and hasattr(camera, "_ALL_INDICES"):
        camera._update_poses(camera._ALL_INDICES)
    return True


def _record_video_frame(base, frames: list[np.ndarray]) -> None:
    camera = getattr(base, "_video_camera", None)
    if camera is None:
        return
    camera.update(float(base.dt), force_recompute=True)
    rgb = camera.data.output.get("rgb")
    if rgb is None or rgb.numel() == 0:
        return
    frame = rgb[int(args_cli.video_env_id), ..., :3].detach().cpu().numpy()
    if frame.dtype != np.uint8:
        max_value = float(frame.max()) if frame.size else 0.0
        frame = np.clip(frame * 255.0 if max_value <= 1.0 else frame, 0, 255).astype(np.uint8)
    frames.append(frame.copy())


def _phase_debug(base) -> dict:
    base._compute_intermediate_values()
    origins = base.scene.env_origins
    return {
        "arm_joint_pos": _tensor_list(base.robot.data.joint_pos[:, base._arm_joint_ids]),
        "arm_joint_target": _tensor_list(base._joint_targets[:, base._arm_joint_ids]),
        "palm_pos_local": _tensor_list(base._palm_pos_w - origins),
        "object_pos_local": _tensor_list(base._object_pos_w - origins),
        "object_fingertip_contact_forces_n": _tensor_list(
            base._object_fingertip_contact_forces
        ),
    }


def _usd_debug_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _usd_mimic_debug() -> list[dict]:
    """Export authored PhysX mimic properties from the live stage."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return []
    rows = []
    robot_prefix = "/World/envs/env_0/Robot"
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(robot_prefix):
            continue
        schemas = [str(schema) for schema in prim.GetAppliedSchemas()]
        attrs = {
            attr.GetName(): _usd_debug_value(attr.Get())
            for attr in prim.GetAttributes()
            if "mimic" in attr.GetName().lower()
        }
        relationships = {
            rel.GetName(): [str(target) for target in rel.GetTargets()]
            for rel in prim.GetRelationships()
            if "mimic" in rel.GetName().lower() or "reference" in rel.GetName().lower()
        }
        if any("mimic" in schema.lower() for schema in schemas) or attrs or relationships:
            rows.append(
                {
                    "path": path,
                    "applied_schemas": schemas,
                    "attributes": attrs,
                    "relationships": relationships,
                }
            )
    return rows


def _enable_self_contact_report(base):
    stage = omni.usd.get_context().get_stage()
    rows: dict[tuple[str, str, str, str], dict] = {}
    robot_prefix = "/World/envs/env_0/Robot/"
    for body_name in base.robot.body_names:
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{body_name}")
        if not prim.IsValid():
            continue
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report_api.CreateThresholdAttr().Set(0.0)

    def on_contact_report(contact_headers, contact_data):
        for header in contact_headers:
            actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            if not actor0.startswith(robot_prefix) or not actor1.startswith(robot_prefix):
                continue
            collider0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            collider1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            if actor1 < actor0:
                actor0, actor1 = actor1, actor0
                collider0, collider1 = collider1, collider0
            key = (actor0, actor1, collider0, collider1)
            row = rows.setdefault(
                key,
                {
                    "actor0": actor0,
                    "actor1": actor1,
                    "collider0": collider0,
                    "collider1": collider1,
                    "event_count": 0,
                    "contact_point_count": 0,
                    "min_separation_m": float("inf"),
                    "max_impulse_norm": 0.0,
                },
            )
            row["event_count"] += 1
            start = int(header.contact_data_offset)
            stop = start + int(header.num_contact_data)
            for index in range(start, stop):
                datum = contact_data[index]
                impulse = datum.impulse
                impulse_norm = (
                    float(impulse[0]) ** 2 + float(impulse[1]) ** 2 + float(impulse[2]) ** 2
                ) ** 0.5
                row["contact_point_count"] += 1
                row["min_separation_m"] = min(row["min_separation_m"], float(datum.separation))
                row["max_impulse_norm"] = max(row["max_impulse_norm"], impulse_norm)

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact_report)
    return subscription, rows


def _self_contact_rows(rows: dict) -> list[dict]:
    output = []
    for row in rows.values():
        item = dict(row)
        if item["min_separation_m"] == float("inf"):
            item["min_separation_m"] = None
        output.append(item)
    output.sort(
        key=lambda row: (row["max_impulse_norm"], row["contact_point_count"], row["event_count"]),
        reverse=True,
    )
    return output


def _object_debug(base) -> dict:
    spawn = getattr(getattr(base.cfg, "object_cfg", None), "spawn", None)
    mass_props = getattr(spawn, "mass_props", None)
    debug = {
        "cfg_object_shape": str(getattr(base.cfg, "object_shape", "")),
        "cfg_object_radius": float(getattr(base.cfg, "object_radius", 0.0)),
        "cfg_object_size": [float(v) for v in getattr(base.cfg, "object_size", ())],
        "cfg_object_start_pos": [float(v) for v in getattr(base.cfg, "object_start_pos", ())],
        "cfg_table_top_z": float(getattr(base.cfg, "table_top_z", 0.0)),
        "uses_tabletop_asset_set": bool(base._uses_tabletop_asset_set())
        if hasattr(base, "_uses_tabletop_asset_set")
        else False,
        "object_spawn_type": type(spawn).__name__ if spawn is not None else None,
        "cfg_object_mass": float(mass_props.mass)
        if mass_props is not None and mass_props.mass is not None
        else None,
    }
    if debug["cfg_object_shape"] == "sphere":
        expected_support_z = debug["cfg_table_top_z"] + debug["cfg_object_radius"]
        debug["cfg_expected_support_z"] = expected_support_z
        debug["cfg_support_height_offset"] = (
            debug["cfg_object_start_pos"][2] - expected_support_z
            if len(debug["cfg_object_start_pos"]) >= 3
            else None
        )
    for name in (
        "_active_object_radius_tensor",
        "_active_object_height_tensor",
        "_active_object_start_z",
        "_active_object_size_tensor",
        "_active_object_shape_codes_tensor",
    ):
        value = getattr(base, name, None)
        if isinstance(value, torch.Tensor):
            debug[name[1:] + "_env0"] = _tensor_list(value[0])
    return debug


def _hand_joint_debug(base, env_id: int = 0) -> dict:
    joint_ids = base._control_hand_joint_ids
    current = base.robot.data.joint_pos[env_id, joint_ids]
    target = base._joint_targets[env_id, joint_ids]
    debug = {
        "joint_names": list(base._sim_hand_joint_names),
        "joint_ids": [int(value) for value in joint_ids],
        "position": _tensor_list(current),
        "target": _tensor_list(target),
        "position_error": _tensor_list(target - current),
        "velocity": _tensor_list(base.robot.data.joint_vel[env_id, joint_ids]),
        "soft_lower_limit": _tensor_list(base._joint_lower_limits[joint_ids]),
        "soft_upper_limit": _tensor_list(base._joint_upper_limits[joint_ids]),
    }
    for name in ("applied_torque", "computed_torque"):
        value = getattr(base.robot.data, name, None)
        if isinstance(value, torch.Tensor):
            debug[name] = _tensor_list(value[env_id, joint_ids])
    joint_index = {name: index for index, name in enumerate(base.robot.joint_names)}
    mimic_specs = (
        ("thumb_intermediate_joint", "thumb_proximal_pitch_joint", 1.334, 0.0),
        ("thumb_distal_joint", "thumb_proximal_pitch_joint", 0.667, 0.0),
        ("index_intermediate_joint", "index_proximal_joint", 1.06399, -0.04545),
        ("middle_intermediate_joint", "middle_proximal_joint", 1.06399, -0.04545),
        ("ring_intermediate_joint", "ring_proximal_joint", 1.06399, -0.04545),
        ("pinky_intermediate_joint", "pinky_proximal_joint", 1.06399, -0.04545),
    )
    mimic_debug = {}
    for follower, active, multiplier, offset in mimic_specs:
        if follower not in joint_index or active not in joint_index:
            continue
        follower_pos = base.robot.data.joint_pos[env_id, joint_index[follower]]
        active_pos = base.robot.data.joint_pos[env_id, joint_index[active]]
        expected = multiplier * active_pos + offset
        mimic_debug[follower] = {
            "active_joint": active,
            "active_position": float(active_pos.detach().cpu()),
            "follower_position": float(follower_pos.detach().cpu()),
            "expected_position": float(expected.detach().cpu()),
            "residual": float((follower_pos - expected).detach().cpu()),
        }
    debug["mimic"] = mimic_debug
    return debug


def _final_rows(
    env,
    stats: dict[str, torch.Tensor],
    fractions: torch.Tensor,
    hand_actions: torch.Tensor,
    start_object_positions: torch.Tensor,
) -> list[dict]:
    base = env.unwrapped
    base._compute_intermediate_values()
    origins = base.scene.env_origins
    object_local = base._object_pos_w - origins
    palm_local = base._palm_pos_w - origins
    fingertip_local = base._fingertip_pos_w - origins[:, None, :]
    fingertip_rel_object = base._fingertip_pos_w - base._object_pos_w[:, None, :]
    surface = base._surface_dist
    score = (
        1000.0 * stats["success"]
        + 500.0 * stats["stable_hold"]
        + 300.0 * stats["lifted"]
        + 180.0 * stats["true_grasp"]
        + 120.0 * stats["strict_true_grasp"]
        + 30.0 * stats["thumb_contact"]
        + 20.0 * torch.clamp(stats["non_thumb_count"], 0.0, 4.0)
        + 10.0 * stats["grasp_quality"]
        - 20.0 * torch.relu(stats["min_thumb_surface_dist"])
    )
    rows = []
    for env_id in range(base.num_envs):
        rows.append(
            {
                "env_id": int(env_id),
                "score": float(score[env_id].detach().cpu()),
                "hand_fraction": [float(v) for v in fractions[env_id].detach().cpu()],
                "hand_action": [float(v) for v in hand_actions[env_id].detach().cpu()],
                "start_object_pos_local": [float(v) for v in start_object_positions[env_id].detach().cpu()],
                "max_success": float(stats["success"][env_id].detach().cpu()),
                "max_stable_hold": float(stats["stable_hold"][env_id].detach().cpu()),
                "max_force_grasp": float(stats["force_grasp"][env_id].detach().cpu()),
                "max_force_lifted": float(stats["force_lifted"][env_id].detach().cpu()),
                "max_force_stable_hold": float(stats["force_stable_hold"][env_id].detach().cpu()),
                "max_force_sum_n": float(stats["max_force_sum"][env_id].detach().cpu()),
                "max_lifted": float(stats["lifted"][env_id].detach().cpu()),
                "max_true_grasp": float(stats["true_grasp"][env_id].detach().cpu()),
                "max_strict_true_grasp": float(stats["strict_true_grasp"][env_id].detach().cpu()),
                "max_thumb_contact": float(stats["thumb_contact"][env_id].detach().cpu()),
                "max_strict_thumb_contact": float(stats["strict_thumb_contact"][env_id].detach().cpu()),
                "max_non_thumb_count": float(stats["non_thumb_count"][env_id].detach().cpu()),
                "max_strict_non_thumb_count": float(stats["strict_non_thumb_count"][env_id].detach().cpu()),
                "max_finger_count": float(stats["finger_count"][env_id].detach().cpu()),
                "max_grasp_quality": float(stats["grasp_quality"][env_id].detach().cpu()),
                "max_weighted_opposition": float(stats["weighted_opposition"][env_id].detach().cpu()),
                "max_object_height_delta": float(stats["object_height_delta"][env_id].detach().cpu()),
                "min_surface_dist": float(stats["min_surface_dist"][env_id].detach().cpu()),
                "min_thumb_surface_dist": float(stats["min_thumb_surface_dist"][env_id].detach().cpu()),
                "min_non_thumb_surface_dist": float(stats["min_non_thumb_surface_dist"][env_id].detach().cpu()),
                "final_surface_dist": [float(v) for v in surface[env_id].detach().cpu()],
                "final_object_pos_local": [float(v) for v in object_local[env_id].detach().cpu()],
                "final_palm_pos_local": [float(v) for v in palm_local[env_id].detach().cpu()],
                "final_fingertip_pos_local": _tensor_list(fingertip_local[env_id]),
                "final_fingertip_rel_object": _tensor_list(fingertip_rel_object[env_id]),
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def main() -> None:
    carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    _set_static_tabletop_cfg(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video_path else None)
    contact_report_subscription = None
    self_contact_debug = {}
    try:
        if args_cli.report_self_contacts:
            contact_report_subscription, self_contact_debug = _enable_self_contact_report(env.unwrapped)
        _trace("before env.reset")
        try:
            env.reset(seed=int(args_cli.seed))
        except BaseException:
            _trace("env.reset raised")
            traceback.print_exc()
            raise
        _trace("after env.reset")
        base = env.unwrapped
        if args_cli.video_path and not 0 <= int(args_cli.video_env_id) < int(base.num_envs):
            raise ValueError(f"--video-env-id must be in [0, {base.num_envs - 1}].")
        _set_video_camera_pose(base)
        device = base.device
        num_envs = base.num_envs
        object_positions = _configured_object_positions(base)
        hand_dim = int(getattr(base.cfg, "action_space", 13)) - len(base._arm_joint_ids)
        hand_joint_names = list(base._active_hand_joint_names if base._uses_active_hand_actions() else base._sim_hand_joint_names)
        fractions = _candidate_fractions(num_envs, int(args_cli.seed), hand_dim, hand_joint_names).to(device=device)
        requested_fractions = fractions.clone()
        hand_actions = _hand_fractions_to_actions(env, fractions)
        pregrasp_fractions = None
        if args_cli.pregrasp_hand_fraction is not None:
            pregrasp_fractions = _repeat_hand_fraction(
                args_cli.pregrasp_hand_fraction,
                num_envs,
                hand_dim,
                hand_joint_names,
                argument_name="--pregrasp-hand-fraction",
            ).to(device=device)
            pregrasp_hand_actions = _hand_fractions_to_actions(env, pregrasp_fractions)
            object_away_positions = object_positions.clone()
            object_away_positions[:, 0] = float(args_cli.table_pos_xy[0]) + 0.38
            object_away_positions[:, 1] = float(args_cli.table_pos_xy[1]) + 0.28
            _reset_object_pose(env, object_away_positions)
        else:
            pregrasp_hand_actions = torch.full(
                (num_envs, hand_dim), -1.0, dtype=torch.float32, device=device
            )
            _reset_object_pose(env, object_positions)
        pregrasp_target = _configured_pregrasp_targets(base)
        close_target = (
            torch.tensor(args_cli.close_arm_pos, dtype=torch.float32, device=device).view(1, 7)
            if args_cli.close_arm_pos is not None
            else pregrasp_target
        )
        if args_cli.lift_delta_candidate:
            lift_deltas = torch.tensor(
                args_cli.lift_delta_candidate, dtype=torch.float32, device=device
            )
            repeat_count = (num_envs + lift_deltas.shape[0] - 1) // lift_deltas.shape[0]
            lift_deltas = lift_deltas.repeat((repeat_count, 1))[:num_envs]
        else:
            lift_deltas = torch.tensor(
                args_cli.lift_deltas, dtype=torch.float32, device=device
            ).view(1, 7).expand(num_envs, -1)
        lift_target = close_target + lift_deltas
        _trace(f"pre steps={args_cli.pre_steps}")
        for step in range(int(args_cli.pre_steps)):
            actions = torch.cat(
                (
                    _arm_action_to_target(env, pregrasp_target, int(args_cli.pre_steps) - step),
                    pregrasp_hand_actions,
                ),
                dim=-1,
            )
            env.step(actions)
        base._compute_intermediate_values()
        pre_debug = {
            "object_pos_local_env0": [float(v) for v in (base._object_pos_w[0] - base.scene.env_origins[0]).detach().cpu()],
            "palm_pos_local_env0": [float(v) for v in (base._palm_pos_w[0] - base.scene.env_origins[0]).detach().cpu()],
            "surface_dist_env0": [float(v) for v in base._surface_dist[0].detach().cpu()],
            "touch_body_pos_local_env0": {
                body_name: [
                    float(v)
                    for v in (
                        base.robot.data.body_pos_w[0, base.robot.body_names.index(body_name)]
                        - base.scene.env_origins[0]
                    ).detach().cpu()
                ]
                for body_name in getattr(base.cfg, "touch_body_names", ())
            },
            "tabletop_clearance_env0": {
                "penalty": float(base._tabletop_arm_clearance_penalty[0].detach().cpu()),
                "min_margin": float(base._tabletop_arm_clearance_min_margin[0].detach().cpu()),
                "ok": bool(base._tabletop_arm_clearance_ok[0].detach().cpu()),
            },
            "object_debug": _object_debug(base),
            "physics_debug": {
                "robot_collision_disable_count": int(getattr(base, "_robot_collision_disable_count", 0)),
                "robot_collision_disable_fail_count": int(
                    getattr(base, "_robot_collision_disable_fail_count", 0)
                ),
                "robot_mimic_property_count": int(
                    getattr(base, "_robot_mimic_property_count", 0)
                ),
                "robot_mimic_property_fail_count": int(
                    getattr(base, "_robot_mimic_property_fail_count", 0)
                ),
                "self_collision_enabled": bool(base.cfg.robot_cfg.spawn.self_collision),
                "robot_material_bind_count": int(getattr(base, "_robot_material_bind_count", 0)),
                "robot_material_bind_fail_count": int(getattr(base, "_robot_material_bind_fail_count", 0)),
                "robot_self_collision_filter_pair_count": int(
                    getattr(base, "_robot_self_collision_filter_pair_count", 0)
                ),
                "robot_self_collision_filter_fail_count": int(
                    getattr(base, "_robot_self_collision_filter_fail_count", 0)
                ),
                "touch_body_names": list(getattr(base.cfg, "touch_body_names", ())),
                "fingertip_body_names": list(getattr(base.cfg, "fingertip_body_names", ())),
            },
            "hand_joint_debug": _hand_joint_debug(base),
        }
        _reset_object_pose(env, object_positions)
        _set_video_camera_pose(base)

        stats = _zeros_stats(num_envs, device)
        video_frames: list[np.ndarray] = []
        video_step = 0
        if args_cli.video_path:
            _record_video_frame(base, video_frames)
        latched_fingers = torch.zeros((num_envs, hand_dim), dtype=torch.bool, device=device)
        latch_streaks = torch.zeros((num_envs, 5), dtype=torch.long, device=device)
        final_command_fractions = fractions.clone()
        progressive_close = bool(args_cli.progressive_close or args_cli.force_limited_close)
        if progressive_close:
            if pregrasp_fractions is None:
                raise ValueError(
                    "--progressive-close and --force-limited-close require --pregrasp-hand-fraction."
                )
        if args_cli.force_limited_close:
            if hand_dim != 6:
                raise ValueError("--force-limited-close currently requires the RH56BFX 6-motor action contract.")
            if float(args_cli.finger_stop_force) < 0.0:
                raise ValueError("--finger-stop-force must be non-negative.")
            if int(args_cli.finger_stop_consecutive_steps) < 1:
                raise ValueError("--finger-stop-consecutive-steps must be at least one.")

        _trace(f"close steps={args_cli.close_steps}")
        for step in range(int(args_cli.close_steps)):
            if progressive_close:
                alpha = float(step + 1) / float(max(int(args_cli.close_steps), 1))
                proposed_fractions = pregrasp_fractions + alpha * (fractions - pregrasp_fractions)
                if args_cli.force_limited_close:
                    final_command_fractions = torch.where(
                        latched_fingers,
                        final_command_fractions,
                        proposed_fractions,
                    )
                else:
                    final_command_fractions = proposed_fractions
                hand_actions = _hand_fractions_to_actions(env, final_command_fractions)
            actions = torch.cat(
                (
                    _arm_action_to_target(env, close_target, int(args_cli.close_steps) - step),
                    hand_actions,
                ),
                dim=-1,
            )
            _, _, _, _, extras = env.step(actions)
            _update_stats(env, stats, extras)
            video_step += 1
            if args_cli.video_path and video_step % max(int(args_cli.video_stride), 1) == 0:
                _record_video_frame(base, video_frames)
            if args_cli.force_limited_close:
                force_contacts = base._object_fingertip_contact_forces >= float(args_cli.finger_stop_force)
                latch_streaks = torch.where(force_contacts, latch_streaks + 1, torch.zeros_like(latch_streaks))
                closing_dims = torch.abs(fractions - pregrasp_fractions) > 1.0e-6
                for touch_index, action_index in enumerate((1, 2, 3, 4, 5)):
                    newly_latched = (
                        (latch_streaks[:, touch_index] >= int(args_cli.finger_stop_consecutive_steps))
                        & closing_dims[:, action_index]
                        & ~latched_fingers[:, action_index]
                    )
                    latched_fingers[newly_latched, action_index] = True
        close_phase_debug = _phase_debug(base)
        close_hand_debug = _hand_joint_debug(base)
        _trace(f"lift steps={args_cli.lift_steps}")
        for step in range(int(args_cli.lift_steps)):
            actions = torch.cat((_arm_action_to_target(env, lift_target, int(args_cli.lift_steps) - step), hand_actions), dim=-1)
            _, _, _, _, extras = env.step(actions)
            _update_stats(env, stats, extras)
            video_step += 1
            if args_cli.video_path and video_step % max(int(args_cli.video_stride), 1) == 0:
                _record_video_frame(base, video_frames)
        lift_phase_debug = _phase_debug(base)
        lift_hand_debug = _hand_joint_debug(base)
        _trace(f"hold steps={args_cli.hold_steps}")
        for step in range(int(args_cli.hold_steps)):
            hold_arm_action = _arm_action_to_target(env, lift_target, 1)
            actions = torch.cat((hold_arm_action, hand_actions), dim=-1)
            _, _, _, _, extras = env.step(actions)
            _update_stats(env, stats, extras)
            video_step += 1
            if args_cli.video_path and video_step % max(int(args_cli.video_stride), 1) == 0:
                _record_video_frame(base, video_frames)
        hold_phase_debug = _phase_debug(base)
        hold_hand_debug = _hand_joint_debug(base)
        if args_cli.video_path:
            _record_video_frame(base, video_frames)
            video_path = Path(args_cli.video_path).expanduser().resolve()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(
                video_path,
                video_frames,
                fps=int(args_cli.video_fps),
                macro_block_size=16,
            )
            _trace(f"video={video_path}")

        rows = _final_rows(env, stats, final_command_fractions, hand_actions, object_positions)
        for row in rows:
            row["lift_delta"] = [
                float(v) for v in lift_deltas[int(row["env_id"])].detach().cpu()
            ]
        base._compute_intermediate_values()
        final_force_grasp = getattr(base, "_force_grasp", torch.zeros(num_envs, dtype=torch.bool, device=device))
        final_force_lifted = base._lifted & final_force_grasp
        final_force_stable_hold = final_force_lifted & (
            base._object_palm_rel_vel < float(base.cfg.stable_object_palm_vel)
        )
        summary = {
            "task": args_cli.task,
            "seed": int(args_cli.seed),
            "num_envs": int(num_envs),
            "object_pos": [float(v) for v in args_cli.object_pos],
            "object_radius_override": (
                float(args_cli.object_radius) if args_cli.object_radius is not None else None
            ),
            "object_pos_grid": _grid_positions(args_cli.object_pos_grid),
            "table_top_z": float(args_cli.table_top_z),
            "pregrasp_arm_pos": [float(v) for v in args_cli.pregrasp_arm_pos],
            "pregrasp_joint_sweep": (
                [float(v) for v in args_cli.pregrasp_joint_sweep]
                if args_cli.pregrasp_joint_sweep is not None
                else None
            ),
            "pregrasp_arm_targets": _tensor_list(pregrasp_target),
            "close_arm_pos": [float(v) for v in args_cli.close_arm_pos] if args_cli.close_arm_pos is not None else None,
            "lift_deltas": [float(v) for v in args_cli.lift_deltas],
            "lift_delta_candidates": _tensor_list(lift_deltas),
            "inspire_close_target": args_cli.inspire_close_target,
            "inspire_close_target_values": (
                [float(v) for v in args_cli.inspire_close_target_values]
                if args_cli.inspire_close_target_values is not None
                else None
            ),
            "pregrasp_hand_fraction": (
                [float(v) for v in args_cli.pregrasp_hand_fraction]
                if args_cli.pregrasp_hand_fraction is not None
                else None
            ),
            "pregrasp_hand_fractions": (
                _tensor_list(pregrasp_fractions) if pregrasp_fractions is not None else None
            ),
            "requested_hand_fractions": _tensor_list(requested_fractions),
            "progressive_close": progressive_close,
            "force_limited_close": bool(args_cli.force_limited_close),
            "finger_stop_force_n": float(args_cli.finger_stop_force),
            "finger_stop_consecutive_steps": int(args_cli.finger_stop_consecutive_steps),
            "hand_effort_limit": (
                float(args_cli.hand_effort_limit) if args_cli.hand_effort_limit is not None else None
            ),
            "hand_stiffness": (
                float(args_cli.hand_stiffness) if args_cli.hand_stiffness is not None else None
            ),
            "hand_damping": (
                float(args_cli.hand_damping) if args_cli.hand_damping is not None else None
            ),
            "video_path": (
                str(Path(args_cli.video_path).expanduser().resolve()) if args_cli.video_path else None
            ),
            "video_env_id": int(args_cli.video_env_id),
            "latched_fingers": _tensor_list(latched_fingers),
            "final_command_fractions": _tensor_list(final_command_fractions),
            "contact_distance": float(args_cli.contact_distance),
            "strict_contact_distance": float(args_cli.strict_contact_distance),
            "extra_self_collision_filter_pairs": [
                [str(link_a), str(link_b)]
                for link_a, link_b in args_cli.extra_self_collision_filter_pair
            ],
            "object_debug": _object_debug(base),
            "usd_mimic_debug": _usd_mimic_debug(),
            "self_contact_debug": _self_contact_rows(self_contact_debug),
            "pre_debug_env0": pre_debug,
            "hand_joint_debug": {
                "after_close": close_hand_debug,
                "after_lift": lift_hand_debug,
                "after_hold": hold_hand_debug,
            },
            "phase_debug": {
                "after_close": close_phase_debug,
                "after_lift": lift_phase_debug,
                "after_hold": hold_phase_debug,
            },
            "counts": {
                "success": int(stats["success"].sum().detach().cpu()),
                "stable_hold": int(stats["stable_hold"].sum().detach().cpu()),
                "force_grasp": int(stats["force_grasp"].sum().detach().cpu()),
                "force_lifted": int(stats["force_lifted"].sum().detach().cpu()),
                "force_stable_hold": int(stats["force_stable_hold"].sum().detach().cpu()),
                "lifted": int(stats["lifted"].sum().detach().cpu()),
                "true_grasp": int(stats["true_grasp"].sum().detach().cpu()),
                "strict_true_grasp": int(stats["strict_true_grasp"].sum().detach().cpu()),
                "thumb_contact": int(stats["thumb_contact"].sum().detach().cpu()),
                "strict_thumb_contact": int(stats["strict_thumb_contact"].sum().detach().cpu()),
                "non_thumb_ge2": int((stats["non_thumb_count"] >= 2.0).sum().detach().cpu()),
                "max_force_sum_n": float(stats["max_force_sum"].max().detach().cpu()),
            },
            "final_counts": {
                "lifted": int(base._lifted.sum().detach().cpu()),
                "stable_hold": int(base._stable_hold.sum().detach().cpu()),
                "force_grasp": int(final_force_grasp.sum().detach().cpu()),
                "force_lifted": int(final_force_lifted.sum().detach().cpu()),
                "force_stable_hold": int(final_force_stable_hold.sum().detach().cpu()),
            },
            "top": rows[: max(int(args_cli.top_k), 1)],
            "all_rows": rows,
        }
        out_path = Path(args_cli.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _trace(json.dumps({k: summary[k] for k in ("counts", "pre_debug_env0")}, indent=2, sort_keys=True))
        _trace(f"wrote {out_path}")
    finally:
        contact_report_subscription = None
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
