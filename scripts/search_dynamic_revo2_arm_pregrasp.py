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
parser.add_argument("--object-pos", type=float, nargs=3, default=(0.58, -0.05, 0.326))
parser.add_argument("--output-json", required=True)
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


def _trace(message: str) -> None:
    print(f"[ARM_SEARCH] {message}", flush=True)


def _build_candidates(num_candidates: int, seed: int, device: str) -> torch.Tensor:
    rng = random.Random(seed)
    base = torch.tensor(V327_PREGRASP_ARM_POS, dtype=torch.float32)
    anchors = [
        list(V327_PREGRASP_ARM_POS),
        list(V326_DEFAULT_ARM_POS),
        [-1.571, 1.571, 0.0, -1.376, 0.0, 1.485, 2.358],
    ]
    joint_scales = torch.tensor([0.42, 0.42, 0.55, 0.38, 0.50, 0.34, 0.58], dtype=torch.float32)
    candidates: list[list[float]] = []
    candidates.extend(anchors)
    for joint_id in range(7):
        for scale in (-1.0, -0.5, 0.5, 1.0):
            value = base.clone()
            value[joint_id] += scale * joint_scales[joint_id]
            candidates.append(value.tolist())
    while len(candidates) < num_candidates:
        value = base.clone()
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
    env_cfg.reference_hand_fractions = (0.45, 0.55, 0.30, 0.30, 0.30, 0.30)
    env_cfg.terminate_on_success = False
    env_cfg.object_start_pos = tuple(float(v) for v in args_cli.object_pos)
    if hasattr(env_cfg, "object_cfg"):
        env_cfg.object_cfg.init_state.pos = env_cfg.object_start_pos

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        base = env.unwrapped
        _set_object_pos(env, env_cfg.object_start_pos)
        device = base.device
        num_envs = base.num_envs
        arm_candidates = _build_candidates(num_envs, args_cli.seed, str(device)).to(device)
        lift_delta = torch.tensor(V327_LIFT_ARM_DELTA, dtype=torch.float32, device=device).view(1, 7)
        pre_actions = torch.zeros((num_envs, base.cfg.action_space), dtype=torch.float32, device=device)
        pre_actions[:, :7] = _arm_pos_to_action(base, arm_candidates)
        pre_actions[:, 7:] = -1.0
        close_actions = pre_actions.clone()
        close_actions[:, 7:] = float(args_cli.hand_action)
        lift_actions = close_actions.clone()
        lift_actions[:, :7] = _arm_pos_to_action(base, arm_candidates + lift_delta)

        metrics = {
            "max_height_delta": torch.zeros(num_envs, dtype=torch.float32, device=device),
            "true_grasp": torch.zeros(num_envs, dtype=torch.bool, device=device),
            "lifted": torch.zeros(num_envs, dtype=torch.bool, device=device),
            "stable_hold": torch.zeros(num_envs, dtype=torch.bool, device=device),
            "success": torch.zeros(num_envs, dtype=torch.bool, device=device),
        }

        _trace(f"candidates={num_envs}")
        _phase(env, pre_actions, args_cli.pre_steps, metrics)
        base._compute_intermediate_values()
        rel_w = base._object_pos_w - base._palm_pos_w
        rel_palm_pre = quat_rotate_inverse(base._palm_quat_w, rel_w)
        target = torch.tensor(ISAACLAB_BALL_PREGRASP_TARGET_REL_PALM, dtype=torch.float32, device=device).view(1, 3)
        target_scale = torch.tensor(ISAACLAB_BALL_PREGRASP_TARGET_SCALE, dtype=torch.float32, device=device).view(1, 3)
        pregrasp_error_norm = torch.norm((rel_palm_pre - target) / target_scale, dim=-1)
        _phase(env, close_actions, args_cli.close_steps, metrics)
        _phase(env, lift_actions, args_cli.lift_steps, metrics)
        _phase(env, lift_actions, args_cli.hold_steps, metrics)

        base._compute_intermediate_values()
        palm_distance = torch.norm(base._object_pos_w - base._palm_pos_w, dim=-1)
        min_surface = base._surface_dist.min(dim=-1).values
        object_local = base._object_pos_w - base.scene.env_origins
        palm_local = base._palm_pos_w - base.scene.env_origins
        results = []
        for env_id in range(num_envs):
            results.append(
                {
                    "env_id": int(env_id),
                    "arm_pos": [float(v) for v in arm_candidates[env_id].detach().cpu()],
                    "pregrasp_error_norm": float(pregrasp_error_norm[env_id].detach().cpu()),
                    "pre_rel_palm": [float(v) for v in rel_palm_pre[env_id].detach().cpu()],
                    "true_grasp": bool(metrics["true_grasp"][env_id].detach().cpu()),
                    "lifted": bool(metrics["lifted"][env_id].detach().cpu()),
                    "stable_hold": bool(metrics["stable_hold"][env_id].detach().cpu()),
                    "success": bool(metrics["success"][env_id].detach().cpu()),
                    "max_height_delta": float(metrics["max_height_delta"][env_id].detach().cpu()),
                    "final_height_delta": float(base._object_height_delta[env_id].detach().cpu()),
                    "final_palm_distance": float(palm_distance[env_id].detach().cpu()),
                    "final_min_surface_dist": float(min_surface[env_id].detach().cpu()),
                    "final_object_pos_local": [float(v) for v in object_local[env_id].detach().cpu()],
                    "final_palm_pos_local": [float(v) for v in palm_local[env_id].detach().cpu()],
                }
            )
        results.sort(
            key=lambda row: (
                row["success"],
                row["stable_hold"],
                row["lifted"],
                row["max_height_delta"],
                -row["pregrasp_error_norm"],
                -row["final_palm_distance"],
            ),
            reverse=True,
        )
        summary = {
            "task": args_cli.task,
            "num_candidates": num_envs,
            "object_pos": list(env_cfg.object_start_pos),
            "success_count": sum(int(row["success"]) for row in results),
            "stable_hold_count": sum(int(row["stable_hold"]) for row in results),
            "lifted_count": sum(int(row["lifted"]) for row in results),
            "true_grasp_count": sum(int(row["true_grasp"]) for row in results),
            "top_results": results[:25],
            "results": results,
        }
        output_path = Path(args_cli.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _trace(json.dumps({k: summary[k] for k in ("success_count", "stable_hold_count", "lifted_count", "true_grasp_count")}))
        _trace(json.dumps(summary["top_results"][:10], indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
