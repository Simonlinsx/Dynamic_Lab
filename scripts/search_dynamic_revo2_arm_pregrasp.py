#!/usr/bin/env python3
"""Search V699 Revo2 arm pregrasp targets for tabletop ball lifting."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="SimToolReal-Revo2-Franka-DynamicTabletopRollingRecoveryTargetObs-Teacher-Direct-v0",
)
parser.add_argument("--num-candidates", type=int, default=512)
parser.add_argument("--seed", type=int, default=23)
parser.add_argument("--pre-steps", type=int, default=150)
parser.add_argument("--close-steps", type=int, default=110)
parser.add_argument("--lift-steps", type=int, default=120)
parser.add_argument("--hold-steps", type=int, default=40)
parser.add_argument("--hand-action", type=float, default=1.0)
parser.add_argument(
    "--lift-arm-delta",
    type=float,
    nargs=7,
    default=None,
    help="Optional joint-space lift delta. Defaults to the V327/Revo2 lift delta.",
)
parser.add_argument("--object-pos", type=float, nargs=3, default=(0.58, -0.05, 0.326))
parser.add_argument(
    "--target-object-pos",
    type=float,
    nargs=3,
    default=None,
    help="Optional geometric target XYZ used for pregrasp ranking while the simulated object can be parked elsewhere.",
)
parser.add_argument("--table-top-z", type=float, default=None, help="Optional table top z override.")
parser.add_argument("--table-pos-xy", type=float, nargs=2, default=None, help="Optional table center XY override.")
parser.add_argument("--no-table", action="store_true", help="Disable table creation for pure kinematic pregrasp search.")
parser.add_argument(
    "--rank-mode",
    choices=("grasp", "pregrasp", "contact"),
    default="grasp",
    help="Use task success/lift ranking, direct contact ranking, or pure geometric pregrasp ranking.",
)
parser.add_argument("--output-json", required=True)
parser.add_argument("--candidate-mode", choices=("v327", "broad"), default="v327")
parser.add_argument(
    "--arm-command-mode",
    choices=("normalized-action", "target"),
    default="normalized-action",
    help="Use old fixed normalized arm actions, or closed-loop tracking to each candidate joint target.",
)
parser.add_argument("--center-arm-pos", type=float, nargs=7, default=None)
parser.add_argument("--joint-scales", type=float, nargs=7, default=None)
parser.add_argument(
    "--reference-hand-fractions",
    type=float,
    nargs=6,
    default=(0.45, 0.55, 0.30, 0.30, 0.30, 0.30),
    help="6-DoF active hand close fractions used for semantic hand actions.",
)
parser.add_argument(
    "--inspire-close-target",
    choices=("cfg", "safe", "sphere4cm", "p80"),
    default="cfg",
    help="Optional Inspire semantic close target override.",
)
parser.add_argument(
    "--reset-object-after-pre",
    action="store_true",
    help="Restore object pose/velocity after pregrasp so close/lift ranking ignores approach bumps.",
)
parser.add_argument(
    "--valid-reset-min-surface-dist",
    type=float,
    default=0.005,
    help="Minimum fingertip-object surface distance before close for a clean reset/preclose state.",
)
parser.add_argument(
    "--max-pre-object-xy-displacement",
    type=float,
    default=0.030,
    help="Maximum allowed object XY displacement caused by the pregrasp approach for clean ranking.",
)
parser.add_argument(
    "--max-pre-clearance-penalty",
    type=float,
    default=float("inf"),
    help="Maximum allowed table-clearance penalty after the pregrasp phase.",
)
parser.add_argument(
    "--max-final-clearance-penalty",
    type=float,
    default=float("inf"),
    help="Maximum allowed table-clearance penalty after close/lift for clean contact ranking.",
)
parser.add_argument(
    "--contact-distance",
    type=float,
    default=None,
    help="Override the task fingertip-object contact distance used by reward/success metrics.",
)
parser.add_argument(
    "--strict-contact-distance",
    type=float,
    default=None,
    help="Override the strict fingertip-object contact distance used by strict success diagnostics.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402
from simtoolreal_lab.tasks.revo2_static_grasp.revo2_static_grasp_env_cfg import (  # noqa: E402
    ISAACLAB_BALL_PREGRASP_TARGET_REL_PALM,
    ISAACLAB_BALL_PREGRASP_TARGET_SCALE,
    V326_DEFAULT_ARM_POS,
    V327_LIFT_ARM_DELTA,
    V327_PREGRASP_ARM_POS,
)
from simtoolreal_lab.tasks.dynamic_dexterous_grasp.dynamic_dexterous_grasp_env_cfg import (  # noqa: E402
    INSPIRE_ANYDEX_P80_CLOSE_TARGETS,
    INSPIRE_ANYDEX_SPHERE_4CM_CLOSE_TARGETS,
    INSPIRE_ANYDEX_SPHERE_SAFE_CLOSE_TARGETS,
)

FRANKA_HOME_ARM_POS = (0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)
LOWER_SAFE_ARM_POS = (0.0, -1.20, 0.0, -2.20, 0.0, 1.40, 0.7853981633974483)
FRONT_ARM_POS = (-1.571, 1.571, 0.0, -1.376, 0.0, 1.485, 2.358)


def _trace(message: str) -> None:
    print(f"[ARM_SEARCH] {message}", flush=True)


def _build_candidates(
    num_candidates: int,
    seed: int,
    device: str,
    mode: str,
    center_arm_pos: list[float] | None,
    joint_scales_arg: list[float] | None,
) -> torch.Tensor:
    rng = random.Random(seed)
    base = torch.tensor(center_arm_pos or V327_PREGRASP_ARM_POS, dtype=torch.float32)
    anchors = [
        base.tolist(),
        list(V326_DEFAULT_ARM_POS),
        list(V327_PREGRASP_ARM_POS),
        list(FRONT_ARM_POS),
    ]
    if mode == "broad":
        anchors.extend(
            [
                list(LOWER_SAFE_ARM_POS),
                list(FRANKA_HOME_ARM_POS),
                [-0.35, 0.65, 0.25, -2.20, 0.10, 2.25, 0.85],
                [-0.75, 0.95, 0.85, -1.85, 0.35, 1.95, -0.25],
                [-1.05, 0.75, 0.35, -1.95, 0.15, 2.05, 0.35],
            ]
        )
    default_scales = [0.42, 0.42, 0.55, 0.38, 0.50, 0.34, 0.58]
    if mode == "broad":
        default_scales = [1.25, 1.45, 1.65, 1.05, 1.65, 1.20, 1.85]
    joint_scales = torch.tensor(joint_scales_arg or default_scales, dtype=torch.float32)
    candidates: list[list[float]] = []
    candidates.extend(anchors)
    for anchor_values in anchors:
        anchor = torch.tensor(anchor_values, dtype=torch.float32)
        for joint_id in range(7):
            for scale in (-1.0, -0.5, 0.5, 1.0):
                value = anchor.clone()
                value[joint_id] += scale * joint_scales[joint_id]
                candidates.append(value.tolist())
    anchor_tensors = [torch.tensor(anchor, dtype=torch.float32) for anchor in anchors]
    while len(candidates) < num_candidates:
        value = rng.choice(anchor_tensors).clone() if mode == "broad" else base.clone()
        for joint_id, scale in enumerate(joint_scales.tolist()):
            value[joint_id] += rng.uniform(-scale, scale)
        candidates.append(value.tolist())
    return torch.tensor(candidates[:num_candidates], dtype=torch.float32, device=device)


def _arm_pos_to_action(base, arm_pos: torch.Tensor) -> torch.Tensor:
    arm_ids = base._arm_joint_ids
    lower = base._joint_lower_limits[arm_ids].unsqueeze(0)
    upper = base._joint_upper_limits[arm_ids].unsqueeze(0)
    center = torch.clamp(base._default_joint_pos[:, arm_ids], lower, upper)
    target = torch.clamp(arm_pos, lower, upper)
    delta = target - center
    positive_span = torch.clamp(upper - center, min=1.0e-6)
    negative_span = torch.clamp(center - lower, min=1.0e-6)
    action = torch.where(delta >= 0.0, delta / positive_span, delta / negative_span)
    return torch.clamp(action, -1.0, 1.0)


def _set_object_pos(env, object_pos: tuple[float, float, float]) -> None:
    base = env.unwrapped
    env_ids = torch.arange(base.num_envs, dtype=torch.long, device=base.device)
    pos = torch.tensor(object_pos, dtype=torch.float32, device=base.device).view(1, 3).expand(base.num_envs, -1)
    base._object_start_pos = pos.clone()
    if hasattr(base, "_tabletop_active_asset_ids"):
        base._tabletop_active_asset_ids[:] = 0
        base._update_active_tabletop_asset_tensors(env_ids)
    object_state = base.object.data.default_root_state.clone()
    object_state[:, 0:3] = pos + base.scene.env_origins
    object_state[:, 3:7] = torch.tensor(base.cfg.object_start_rot, dtype=torch.float32, device=base.device).view(1, 4)
    object_state[:, 7:] = 0.0
    base.object.write_root_pose_to_sim(object_state[:, :7], env_ids=env_ids)
    base.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids=env_ids)
    base.scene.write_data_to_sim()
    base.sim.forward()
    base._compute_intermediate_values()


def _phase(env, actions: torch.Tensor, steps: int, metrics: dict[str, torch.Tensor]) -> dict:
    base = env.unwrapped
    extras = {}
    for _ in range(steps):
        _, _, _, _, extras = env.step(actions)
        base._compute_intermediate_values()
        _update_contact_metrics(base, metrics)
        _update_clearance_metrics(base, metrics)
        metrics["max_height_delta"] = torch.maximum(metrics["max_height_delta"], base._object_height_delta)
        for key, extra_key in (
            ("true_grasp", "true_grasp_env"),
            ("lifted", "lifted_env"),
            ("stable_hold", "stable_hold_env"),
            ("success", "success_env"),
        ):
            value = extras.get(extra_key)
            if value is not None:
                metrics[key] |= value.bool()
    return extras


def _update_contact_metrics(base, metrics: dict[str, torch.Tensor]) -> None:
    if hasattr(base, "_thumb_contact"):
        metrics["thumb_contact"] |= base._thumb_contact.bool()
    if hasattr(base, "_strict_thumb_contact"):
        metrics["strict_thumb_contact"] |= base._strict_thumb_contact.bool()
    if hasattr(base, "_non_thumb_contact_count"):
        metrics["non_thumb_count"] = torch.maximum(
            metrics["non_thumb_count"], base._non_thumb_contact_count.float()
        )
    if hasattr(base, "_strict_non_thumb_contact_count"):
        metrics["strict_non_thumb_count"] = torch.maximum(
            metrics["strict_non_thumb_count"], base._strict_non_thumb_contact_count.float()
        )
    if hasattr(base, "_finger_contact_count"):
        metrics["finger_count"] = torch.maximum(metrics["finger_count"], base._finger_contact_count.float())


def _update_clearance_metrics(base, metrics: dict[str, torch.Tensor]) -> None:
    penalty = getattr(base, "_tabletop_arm_clearance_penalty", None)
    if penalty is not None:
        metrics["max_clearance_penalty"] = torch.maximum(metrics["max_clearance_penalty"], penalty.float())
    margin = getattr(base, "_tabletop_arm_clearance_min_margin", None)
    if margin is not None:
        metrics["min_clearance_margin"] = torch.minimum(metrics["min_clearance_margin"], margin.float())


def _arm_action_to_target(base, target: torch.Tensor, remaining_steps: int) -> torch.Tensor:
    arm_ids = base._arm_joint_ids
    current = base._joint_targets[:, arm_ids]
    lower = base._joint_lower_limits[arm_ids].unsqueeze(0)
    upper = base._joint_upper_limits[arm_ids].unsqueeze(0)
    center = torch.clamp(base._default_joint_pos[:, arm_ids], lower, upper)
    target = torch.clamp(target.expand_as(current), lower, upper)
    if base.cfg.policy_action_interface == "joint_target":
        desired_next = current + (target - current) / float(max(remaining_steps, 1))
        arm_ma = max(float(base.cfg.arm_moving_average), 1.0e-6)
        raw_target = (desired_next - (1.0 - arm_ma) * current) / arm_ma
        raw_target = torch.clamp(raw_target, lower, upper)
        positive_span = torch.clamp(upper - center, min=1.0e-6)
        negative_span = torch.clamp(center - lower, min=1.0e-6)
        delta = raw_target - center
        action = torch.where(delta >= 0.0, delta / positive_span, delta / negative_span)
    else:
        dt = base.step_dt
        effective_arm_gain = max(float(base.cfg.arm_action_scale) * float(base.cfg.arm_moving_average), 1.0e-6)
        denom = max(effective_arm_gain * dt * max(remaining_steps, 1), 1.0e-6)
        action = (target - current) / denom
    return torch.clamp(action, -1.0, 1.0)


def _new_metrics(num_envs: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "max_height_delta": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "true_grasp": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "lifted": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "stable_hold": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "success": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "thumb_contact": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "strict_thumb_contact": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "non_thumb_count": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "strict_non_thumb_count": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "finger_count": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "max_clearance_penalty": torch.zeros(num_envs, dtype=torch.float32, device=device),
        "min_clearance_margin": torch.full((num_envs,), float("inf"), dtype=torch.float32, device=device),
    }


def _phase_to_target(
    env,
    base_actions: torch.Tensor,
    arm_targets: torch.Tensor,
    steps: int,
    metrics: dict[str, torch.Tensor],
) -> dict:
    base = env.unwrapped
    extras = {}
    for step_id in range(steps):
        actions = base_actions.clone()
        actions[:, :7] = _arm_action_to_target(base, arm_targets, steps - step_id)
        _, _, _, _, extras = env.step(actions)
        base._compute_intermediate_values()
        _update_contact_metrics(base, metrics)
        _update_clearance_metrics(base, metrics)
        metrics["max_height_delta"] = torch.maximum(metrics["max_height_delta"], base._object_height_delta)
        for key, extra_key in (
            ("true_grasp", "true_grasp_env"),
            ("lifted", "lifted_env"),
            ("stable_hold", "stable_hold_env"),
            ("success", "success_env"),
        ):
            value = extras.get(extra_key)
            if value is not None:
                metrics[key] |= value.bool()
    return extras


def main() -> None:
    carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_candidates)
    env_cfg.reset_object_pos_noise = (0.0, 0.0, 0.0)
    env_cfg.dynamic_tabletop_persistent_motion = False
    env_cfg.dynamic_grasp_speed_curriculum = False
    env_cfg.scripted_action_prior_enabled = False
    if str(getattr(env_cfg, "action_contract", "")) in ("revo2_semantic_13d", "inspire_semantic_13d"):
        env_cfg.reference_hand_fractions = tuple(float(v) for v in args_cli.reference_hand_fractions)
    if args_cli.contact_distance is not None and hasattr(env_cfg, "contact_distance"):
        env_cfg.contact_distance = float(args_cli.contact_distance)
    if args_cli.strict_contact_distance is not None and hasattr(env_cfg, "strict_success_contact_distance"):
        env_cfg.strict_success_contact_distance = float(args_cli.strict_contact_distance)
    target_map = {
        "safe": INSPIRE_ANYDEX_SPHERE_SAFE_CLOSE_TARGETS,
        "sphere4cm": INSPIRE_ANYDEX_SPHERE_4CM_CLOSE_TARGETS,
        "p80": INSPIRE_ANYDEX_P80_CLOSE_TARGETS,
    }
    if args_cli.inspire_close_target != "cfg" and hasattr(env_cfg, "inspire_semantic_close_targets"):
        env_cfg.inspire_semantic_close_targets = target_map[args_cli.inspire_close_target]
    env_cfg.terminate_on_success = False
    env_cfg.object_start_pos = tuple(float(v) for v in args_cli.object_pos)
    if hasattr(env_cfg, "workspace_xy_limit"):
        workspace_limit = float(getattr(env_cfg, "workspace_xy_limit"))
        workspace_limit = max(workspace_limit, max(abs(float(v)) for v in args_cli.object_pos[:2]) + 0.20)
        if args_cli.table_pos_xy is not None:
            workspace_limit = max(workspace_limit, max(abs(float(v)) for v in args_cli.table_pos_xy) + 0.20)
        env_cfg.workspace_xy_limit = workspace_limit
    if hasattr(env_cfg, "object_cfg"):
        env_cfg.object_cfg.init_state.pos = env_cfg.object_start_pos
    if args_cli.no_table and hasattr(env_cfg, "create_table"):
        env_cfg.create_table = False
    if args_cli.table_top_z is not None and hasattr(env_cfg, "table_top_z"):
        env_cfg.table_top_z = float(args_cli.table_top_z)
    if args_cli.table_pos_xy is not None and hasattr(env_cfg, "table_cfg"):
        table_z = float(getattr(env_cfg, "table_top_z", env_cfg.table_cfg.init_state.pos[2] + 0.0225))
        env_cfg.table_cfg.init_state.pos = (
            float(args_cli.table_pos_xy[0]),
            float(args_cli.table_pos_xy[1]),
            table_z - 0.0225,
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        base = env.unwrapped
        _set_object_pos(env, env_cfg.object_start_pos)
        device = base.device
        num_envs = base.num_envs
        arm_candidates = _build_candidates(
            num_envs,
            args_cli.seed,
            str(device),
            args_cli.candidate_mode,
            args_cli.center_arm_pos,
            args_cli.joint_scales,
        ).to(device)
        arm_ids = base._arm_joint_ids
        lower = base._joint_lower_limits[arm_ids].unsqueeze(0)
        upper = base._joint_upper_limits[arm_ids].unsqueeze(0)
        arm_candidates = torch.clamp(arm_candidates, lower, upper)
        lift_delta_cfg = args_cli.lift_arm_delta if args_cli.lift_arm_delta is not None else V327_LIFT_ARM_DELTA
        lift_delta = torch.tensor(lift_delta_cfg, dtype=torch.float32, device=device).view(1, 7)
        pre_actions = torch.zeros((num_envs, base.cfg.action_space), dtype=torch.float32, device=device)
        pre_actions[:, :7] = _arm_pos_to_action(base, arm_candidates)
        pre_actions[:, 7:] = -1.0
        close_actions = pre_actions.clone()
        if (
            not base._uses_active_hand_actions()
            and args_cli.inspire_close_target != "cfg"
            and len(base._control_hand_joint_ids) == len(target_map[args_cli.inspire_close_target])
        ):
            hand_ids = base._control_hand_joint_ids
            hand_lower = base._joint_lower_limits[hand_ids]
            hand_upper = base._joint_upper_limits[hand_ids]
            hand_center = torch.clamp(base._default_joint_pos[:, hand_ids], hand_lower.unsqueeze(0), hand_upper.unsqueeze(0))[0]
            hand_target = torch.tensor(
                target_map[args_cli.inspire_close_target],
                dtype=torch.float32,
                device=device,
            )
            hand_target = torch.clamp(hand_target, hand_lower, hand_upper)
            if base.cfg.policy_action_interface == "joint_target":
                hand_delta = hand_target - hand_center
                hand_positive_span = torch.clamp(hand_upper - hand_center, min=1.0e-6)
                hand_negative_span = torch.clamp(hand_center - hand_lower, min=1.0e-6)
                close_hand_action = torch.where(
                    hand_delta >= 0.0,
                    hand_delta / hand_positive_span,
                    hand_delta / hand_negative_span,
                )
            else:
                close_hand_action = 2.0 * (hand_target - hand_lower) / torch.clamp(hand_upper - hand_lower, min=1.0e-6) - 1.0
            close_actions[:, 7:] = torch.clamp(close_hand_action, -1.0, 1.0).view(1, -1)
        else:
            close_actions[:, 7:] = float(args_cli.hand_action)
        lift_actions = close_actions.clone()
        lift_actions[:, :7] = _arm_pos_to_action(base, arm_candidates + lift_delta)

        pre_metrics = _new_metrics(num_envs, device)
        metrics = _new_metrics(num_envs, device)

        _trace(f"candidates={num_envs} arm_command_mode={args_cli.arm_command_mode}")
        if args_cli.arm_command_mode == "target":
            _phase_to_target(env, pre_actions, arm_candidates, args_cli.pre_steps, pre_metrics)
        else:
            _phase(env, pre_actions, args_cli.pre_steps, pre_metrics)
        base._compute_intermediate_values()
        if args_cli.target_object_pos is None:
            target_object_pos_w = base._object_pos_w
        else:
            target_object_pos = torch.tensor(
                args_cli.target_object_pos, dtype=torch.float32, device=device
            ).view(1, 3)
            target_object_pos_w = base.scene.env_origins + target_object_pos.expand(num_envs, -1)
        rel_w = target_object_pos_w - base._palm_pos_w
        rel_palm_pre = quat_rotate_inverse(base._palm_quat_w, rel_w)
        target = torch.tensor(ISAACLAB_BALL_PREGRASP_TARGET_REL_PALM, dtype=torch.float32, device=device).view(1, 3)
        target_scale = torch.tensor(ISAACLAB_BALL_PREGRASP_TARGET_SCALE, dtype=torch.float32, device=device).view(1, 3)
        pregrasp_error_norm = torch.norm((rel_palm_pre - target) / target_scale, dim=-1)
        pre_palm_distance = torch.norm(target_object_pos_w - base._palm_pos_w, dim=-1)
        pre_min_surface = base._surface_dist.min(dim=-1).values
        pre_surface_dist = base._surface_dist.clone()
        pre_palm_local = base._palm_pos_w - base.scene.env_origins
        pre_object_local = base._object_pos_w - base.scene.env_origins
        pre_clearance_min_margin = getattr(base, "_tabletop_arm_clearance_min_margin", None)
        if pre_clearance_min_margin is None:
            pre_clearance_min_margin = torch.full((num_envs,), float("nan"), device=device)
        pre_clearance_penalty = getattr(base, "_tabletop_arm_clearance_penalty", None)
        if pre_clearance_penalty is None:
            pre_clearance_penalty = torch.full((num_envs,), float("nan"), device=device)
        target_object_local = torch.tensor(
            args_cli.target_object_pos or args_cli.object_pos, dtype=torch.float32, device=device
        ).view(1, 3)
        pre_object_xy_displacement = torch.norm(
            pre_object_local[:, :2] - target_object_local[:, :2], dim=-1
        )
        if args_cli.reset_object_after_pre and (args_cli.close_steps > 0 or args_cli.lift_steps > 0 or args_cli.hold_steps > 0):
            _set_object_pos(env, env_cfg.object_start_pos)
            base._compute_intermediate_values()
        reset_min_surface = base._surface_dist.min(dim=-1).values
        reset_clean = reset_min_surface >= float(args_cli.valid_reset_min_surface_dist)
        pre_object_clean = pre_object_xy_displacement <= float(args_cli.max_pre_object_xy_displacement)
        pre_clearance_ok = pre_clearance_penalty <= float(args_cli.max_pre_clearance_penalty)
        valid_preclose = reset_clean & pre_object_clean & pre_clearance_ok
        if args_cli.arm_command_mode == "target":
            _phase_to_target(env, close_actions, arm_candidates, args_cli.close_steps, metrics)
            _phase_to_target(env, lift_actions, arm_candidates + lift_delta, args_cli.lift_steps, metrics)
            _phase_to_target(env, lift_actions, arm_candidates + lift_delta, args_cli.hold_steps, metrics)
        else:
            _phase(env, close_actions, args_cli.close_steps, metrics)
            _phase(env, lift_actions, args_cli.lift_steps, metrics)
            _phase(env, lift_actions, args_cli.hold_steps, metrics)

        base._compute_intermediate_values()
        palm_distance = torch.norm(base._object_pos_w - base._palm_pos_w, dim=-1)
        min_surface = base._surface_dist.min(dim=-1).values
        final_surface_dist = base._surface_dist.clone()
        object_local = base._object_pos_w - base.scene.env_origins
        palm_local = base._palm_pos_w - base.scene.env_origins
        final_clearance_min_margin = getattr(base, "_tabletop_arm_clearance_min_margin", None)
        if final_clearance_min_margin is None:
            final_clearance_min_margin = torch.full((num_envs,), float("nan"), device=device)
        final_clearance_penalty = getattr(base, "_tabletop_arm_clearance_penalty", None)
        if final_clearance_penalty is None:
            final_clearance_penalty = torch.full((num_envs,), float("nan"), device=device)
        fingertip_local = base._fingertip_pos_w - base.scene.env_origins.unsqueeze(1)
        min_fingertip_z = fingertip_local[..., 2].min(dim=-1).values
        table_top_z = float(getattr(base.cfg, "table_top_z", args_cli.table_top_z or 0.0))
        min_fingertip_table_margin = min_fingertip_z - table_top_z
        final_object_xy_displacement = torch.norm(
            object_local[:, :2] - target_object_local[:, :2], dim=-1
        )
        final_object_z_delta = object_local[:, 2] - target_object_local[:, 2]
        rollout_clearance_penalty = torch.maximum(final_clearance_penalty, metrics["max_clearance_penalty"])
        rollout_clearance_min_margin = torch.minimum(final_clearance_min_margin, metrics["min_clearance_margin"])
        final_clearance_ok = rollout_clearance_penalty <= float(args_cli.max_final_clearance_penalty)
        clean_table_margin = torch.clamp(min_fingertip_table_margin - 0.015, min=-0.05, max=0.05)
        results = []
        for env_id in range(num_envs):
            results.append(
                {
                    "env_id": int(env_id),
                    "arm_pos": [float(v) for v in arm_candidates[env_id].detach().cpu()],
                    "pre_arm_action": [float(v) for v in pre_actions[env_id, :7].detach().cpu()],
                    "lift_arm_action": [float(v) for v in lift_actions[env_id, :7].detach().cpu()],
                    "pregrasp_error_norm": float(pregrasp_error_norm[env_id].detach().cpu()),
                    "pre_rel_palm": [float(v) for v in rel_palm_pre[env_id].detach().cpu()],
                    "pre_palm_distance": float(pre_palm_distance[env_id].detach().cpu()),
                    "pre_min_surface_dist": float(pre_min_surface[env_id].detach().cpu()),
                    "pre_surface_dist": [float(v) for v in pre_surface_dist[env_id].detach().cpu()],
                    "pre_tabletop_clearance_min_margin": float(
                        pre_clearance_min_margin[env_id].detach().cpu()
                    ),
                    "pre_tabletop_clearance_penalty": float(pre_clearance_penalty[env_id].detach().cpu()),
                    "pre_tabletop_clearance_ok": bool(pre_clearance_ok[env_id].detach().cpu()),
                    "pre_palm_pos_local": [float(v) for v in pre_palm_local[env_id].detach().cpu()],
                    "pre_object_xy_displacement": float(pre_object_xy_displacement[env_id].detach().cpu()),
                    "pre_true_grasp": bool(pre_metrics["true_grasp"][env_id].detach().cpu()),
                    "pre_thumb_contact": bool(pre_metrics["thumb_contact"][env_id].detach().cpu()),
                    "pre_max_finger_count": float(pre_metrics["finger_count"][env_id].detach().cpu()),
                    "reset_min_surface_dist": float(reset_min_surface[env_id].detach().cpu()),
                    "reset_clean": bool(reset_clean[env_id].detach().cpu()),
                    "pre_object_clean": bool(pre_object_clean[env_id].detach().cpu()),
                    "valid_preclose": bool(valid_preclose[env_id].detach().cpu()),
                    "true_grasp": bool(metrics["true_grasp"][env_id].detach().cpu()),
                    "lifted": bool(metrics["lifted"][env_id].detach().cpu()),
                    "stable_hold": bool(metrics["stable_hold"][env_id].detach().cpu()),
                    "success": bool(metrics["success"][env_id].detach().cpu()),
                    "thumb_contact": bool(metrics["thumb_contact"][env_id].detach().cpu()),
                    "strict_thumb_contact": bool(metrics["strict_thumb_contact"][env_id].detach().cpu()),
                    "max_non_thumb_count": float(metrics["non_thumb_count"][env_id].detach().cpu()),
                    "max_strict_non_thumb_count": float(metrics["strict_non_thumb_count"][env_id].detach().cpu()),
                    "max_finger_count": float(metrics["finger_count"][env_id].detach().cpu()),
                    "max_height_delta": float(metrics["max_height_delta"][env_id].detach().cpu()),
                    "final_height_delta": float(base._object_height_delta[env_id].detach().cpu()),
                    "final_object_z_delta": float(final_object_z_delta[env_id].detach().cpu()),
                    "final_object_xy_displacement": float(
                        final_object_xy_displacement[env_id].detach().cpu()
                    ),
                    "final_palm_distance": float(palm_distance[env_id].detach().cpu()),
                    "final_min_surface_dist": float(min_surface[env_id].detach().cpu()),
                    "final_surface_dist": [float(v) for v in final_surface_dist[env_id].detach().cpu()],
                    "final_min_fingertip_table_margin": float(
                        min_fingertip_table_margin[env_id].detach().cpu()
                    ),
                    "final_tabletop_clearance_min_margin": float(
                        final_clearance_min_margin[env_id].detach().cpu()
                    ),
                    "final_tabletop_clearance_penalty": float(final_clearance_penalty[env_id].detach().cpu()),
                    "rollout_tabletop_clearance_min_margin": float(
                        rollout_clearance_min_margin[env_id].detach().cpu()
                    ),
                    "rollout_tabletop_clearance_max_penalty": float(
                        rollout_clearance_penalty[env_id].detach().cpu()
                    ),
                    "final_tabletop_clearance_ok": bool(final_clearance_ok[env_id].detach().cpu()),
                    "clean_table_margin_score": float(clean_table_margin[env_id].detach().cpu()),
                    "final_object_pos_local": [float(v) for v in object_local[env_id].detach().cpu()],
                    "final_palm_pos_local": [float(v) for v in palm_local[env_id].detach().cpu()],
                }
            )
        if args_cli.rank_mode == "pregrasp":
            results.sort(
                key=lambda row: (
                    -row["pregrasp_error_norm"],
                    -abs(row["pre_palm_pos_local"][2] - float((args_cli.target_object_pos or args_cli.object_pos)[2]) - 0.125),
                    -row["pre_palm_distance"],
                ),
                reverse=True,
            )
        elif args_cli.rank_mode == "contact":
            results.sort(
                key=lambda row: (
                    row["success"],
                    row["stable_hold"],
                    row["valid_preclose"],
                    row["final_tabletop_clearance_ok"],
                    row["true_grasp"],
                    row["strict_thumb_contact"],
                    row["thumb_contact"],
                    row["max_strict_non_thumb_count"],
                    row["max_non_thumb_count"],
                    row["max_finger_count"],
                    row["final_object_z_delta"],
                    row["lifted"],
                    row["max_height_delta"],
                    -row["final_object_xy_displacement"],
                    -row["rollout_tabletop_clearance_max_penalty"],
                    row["clean_table_margin_score"],
                    -row["final_min_surface_dist"],
                    -row["pregrasp_error_norm"],
                ),
                reverse=True,
            )
        else:
            results.sort(
                key=lambda row: (
                    row["success"],
                    row["stable_hold"],
                    row["lifted"],
                    row["true_grasp"],
                    row["max_height_delta"],
                    -row["final_min_surface_dist"],
                    -row["pre_palm_distance"],
                    -row["pregrasp_error_norm"],
                    -row["final_palm_distance"],
                ),
                reverse=True,
            )
        summary = {
            "task": args_cli.task,
            "num_candidates": num_envs,
            "object_pos": list(env_cfg.object_start_pos),
            "target_object_pos": list(args_cli.target_object_pos) if args_cli.target_object_pos is not None else None,
            "rank_mode": args_cli.rank_mode,
            "arm_command_mode": args_cli.arm_command_mode,
            "reference_hand_fractions": list(env_cfg.reference_hand_fractions),
            "lift_arm_delta": [float(v) for v in lift_delta_cfg],
            "inspire_close_target": args_cli.inspire_close_target,
            "contact_distance": float(getattr(env_cfg, "contact_distance", float("nan"))),
            "strict_success_contact_distance": float(
                getattr(env_cfg, "strict_success_contact_distance", float("nan"))
            ),
            "success_count": sum(int(row["success"]) for row in results),
            "stable_hold_count": sum(int(row["stable_hold"]) for row in results),
            "lifted_count": sum(int(row["lifted"]) for row in results),
            "true_grasp_count": sum(int(row["true_grasp"]) for row in results),
            "thumb_contact_count": sum(int(row["thumb_contact"]) for row in results),
            "strict_thumb_contact_count": sum(int(row["strict_thumb_contact"]) for row in results),
            "non_thumb_ge2_count": sum(int(row["max_non_thumb_count"] >= 2.0) for row in results),
            "valid_preclose_count": sum(int(row["valid_preclose"]) for row in results),
            "pre_clearance_ok_count": sum(int(row["pre_tabletop_clearance_ok"]) for row in results),
            "final_clearance_ok_count": sum(int(row["final_tabletop_clearance_ok"]) for row in results),
            "top_results": results[:25],
            "results": results,
        }
        output_path = Path(args_cli.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _trace(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "success_count",
                        "stable_hold_count",
                        "lifted_count",
                        "true_grasp_count",
                        "thumb_contact_count",
                        "strict_thumb_contact_count",
                        "non_thumb_ge2_count",
                        "valid_preclose_count",
                        "pre_clearance_ok_count",
                        "final_clearance_ok_count",
                    )
                }
            )
        )
        _trace(json.dumps(summary["top_results"][:10], indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
