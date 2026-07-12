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
    "--close-arm-pos",
    type=float,
    nargs=7,
    default=None,
    help="Optional arm target used during close/lift. If omitted, close/lift use --pregrasp-arm-pos.",
)
parser.add_argument("--lift-deltas", type=float, nargs=7, default=DEFAULT_LIFT_DELTA)
parser.add_argument("--object-pos", type=float, nargs=3, default=(0.58, 0.0, 0.320))
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
    default="sphere4cm",
)
parser.add_argument(
    "--inspire-close-target-values",
    type=float,
    nargs=12,
    default=None,
    help="Explicit 12-joint Inspire semantic close targets in sim_hand_joint_names order.",
)
parser.add_argument("--contact-distance", type=float, default=0.020)
parser.add_argument("--strict-contact-distance", type=float, default=0.008)
parser.add_argument("--hand-moving-average", type=float, default=None)
parser.add_argument(
    "--fixed-hand-fraction",
    type=float,
    nargs="+",
    default=None,
    help="Repeat one hand fraction instead of sweeping closure candidates. Accepts semantic 6D or physical action-dim values.",
)
parser.add_argument("--top-k", type=int, default=24)
parser.add_argument("--output-json", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

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
    object_pos = _grid_positions(args_cli.object_pos_grid)[0] if args_cli.object_pos_grid is not None else args_cli.object_pos
    env_cfg.object_start_pos = tuple(float(v) for v in object_pos)
    env_cfg.object_cfg.init_state.pos = env_cfg.object_start_pos
    env_cfg.table_top_z = float(args_cli.table_top_z)
    env_cfg.table_cfg.init_state.pos = (
        float(args_cli.table_pos_xy[0]),
        float(args_cli.table_pos_xy[1]),
        float(args_cli.table_top_z) - 0.0225,
    )


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
        fixed = torch.tensor(args_cli.fixed_hand_fraction, dtype=torch.float32).view(1, -1)
        if fixed.shape[-1] == 6 and hand_dim != 6:
            fixed = _semantic_to_physical_fractions(fixed, joint_names)
        if fixed.shape[-1] != hand_dim:
            raise ValueError(f"--fixed-hand-fraction has {fixed.shape[-1]} values but this task expects {hand_dim}.")
        return torch.clamp(fixed, 0.0, 1.0).expand(count, -1).clone()
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


def _object_debug(base) -> dict:
    spawn = getattr(getattr(base.cfg, "object_cfg", None), "spawn", None)
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
    }
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
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        _trace("before env.reset")
        try:
            env.reset(seed=int(args_cli.seed))
        except BaseException:
            _trace("env.reset raised")
            traceback.print_exc()
            raise
        _trace("after env.reset")
        base = env.unwrapped
        device = base.device
        num_envs = base.num_envs
        object_positions = _configured_object_positions(base)
        _reset_object_pose(env, object_positions)
        hand_dim = int(getattr(base.cfg, "action_space", 13)) - len(base._arm_joint_ids)
        hand_joint_names = list(base._active_hand_joint_names if base._uses_active_hand_actions() else base._sim_hand_joint_names)
        fractions = _candidate_fractions(num_envs, int(args_cli.seed), hand_dim, hand_joint_names).to(device=device)
        hand_actions = _hand_fractions_to_actions(env, fractions)
        pregrasp_target = torch.tensor(args_cli.pregrasp_arm_pos, dtype=torch.float32, device=device).view(1, 7)
        close_target = (
            torch.tensor(args_cli.close_arm_pos, dtype=torch.float32, device=device).view(1, 7)
            if args_cli.close_arm_pos is not None
            else pregrasp_target
        )
        lift_target = close_target + torch.tensor(args_cli.lift_deltas, dtype=torch.float32, device=device).view(1, 7)
        open_hand = torch.full((num_envs, hand_dim), -1.0, dtype=torch.float32, device=device)
        zeros7 = torch.zeros((num_envs, 7), dtype=torch.float32, device=device)

        _trace(f"pre steps={args_cli.pre_steps}")
        for step in range(int(args_cli.pre_steps)):
            actions = torch.cat((_arm_action_to_target(env, pregrasp_target, int(args_cli.pre_steps) - step), open_hand), dim=-1)
            env.step(actions)
        base._compute_intermediate_values()
        pre_debug = {
            "object_pos_local_env0": [float(v) for v in (base._object_pos_w[0] - base.scene.env_origins[0]).detach().cpu()],
            "palm_pos_local_env0": [float(v) for v in (base._palm_pos_w[0] - base.scene.env_origins[0]).detach().cpu()],
            "surface_dist_env0": [float(v) for v in base._surface_dist[0].detach().cpu()],
            "object_debug": _object_debug(base),
        }
        _reset_object_pose(env, object_positions)

        stats = _zeros_stats(num_envs, device)
        _trace(f"close steps={args_cli.close_steps}")
        for step in range(int(args_cli.close_steps)):
            actions = torch.cat((_arm_action_to_target(env, close_target, int(args_cli.close_steps) - step), hand_actions), dim=-1)
            _, _, _, _, extras = env.step(actions)
            _update_stats(env, stats, extras)
        _trace(f"lift steps={args_cli.lift_steps}")
        for step in range(int(args_cli.lift_steps)):
            actions = torch.cat((_arm_action_to_target(env, lift_target, int(args_cli.lift_steps) - step), hand_actions), dim=-1)
            _, _, _, _, extras = env.step(actions)
            _update_stats(env, stats, extras)
        _trace(f"hold steps={args_cli.hold_steps}")
        for step in range(int(args_cli.hold_steps)):
            actions = torch.cat((zeros7, hand_actions), dim=-1)
            _, _, _, _, extras = env.step(actions)
            _update_stats(env, stats, extras)

        rows = _final_rows(env, stats, fractions, hand_actions, object_positions)
        summary = {
            "task": args_cli.task,
            "seed": int(args_cli.seed),
            "num_envs": int(num_envs),
            "object_pos": [float(v) for v in args_cli.object_pos],
            "object_pos_grid": _grid_positions(args_cli.object_pos_grid),
            "table_top_z": float(args_cli.table_top_z),
            "pregrasp_arm_pos": [float(v) for v in args_cli.pregrasp_arm_pos],
            "close_arm_pos": [float(v) for v in args_cli.close_arm_pos] if args_cli.close_arm_pos is not None else None,
            "lift_deltas": [float(v) for v in args_cli.lift_deltas],
            "inspire_close_target": args_cli.inspire_close_target,
            "inspire_close_target_values": (
                [float(v) for v in args_cli.inspire_close_target_values]
                if args_cli.inspire_close_target_values is not None
                else None
            ),
            "contact_distance": float(args_cli.contact_distance),
            "strict_contact_distance": float(args_cli.strict_contact_distance),
            "object_debug": _object_debug(base),
            "pre_debug_env0": pre_debug,
            "counts": {
                "success": int(stats["success"].sum().detach().cpu()),
                "stable_hold": int(stats["stable_hold"].sum().detach().cpu()),
                "lifted": int(stats["lifted"].sum().detach().cpu()),
                "true_grasp": int(stats["true_grasp"].sum().detach().cpu()),
                "strict_true_grasp": int(stats["strict_true_grasp"].sum().detach().cpu()),
                "thumb_contact": int(stats["thumb_contact"].sum().detach().cpu()),
                "strict_thumb_contact": int(stats["strict_thumb_contact"].sum().detach().cpu()),
                "non_thumb_ge2": int((stats["non_thumb_count"] >= 2.0).sum().detach().cpu()),
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
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
