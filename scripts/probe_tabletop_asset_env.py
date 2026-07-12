#!/usr/bin/env python3
"""Debug probe for multi-asset tabletop dynamic tasks."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--summary-out", required=True)
parser.add_argument("--speed-curriculum-alpha", type=float, default=None)
parser.add_argument("--asset-curriculum-alpha", type=float, default=None)
parser.add_argument("--motion-curriculum-alpha", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


out_path = Path(args_cli.summary_out).expanduser().resolve()
events: list[dict] = []


def record(event: str, **payload) -> None:
    item = {"event": event, **payload}
    events.append(item)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"events": events}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[TABLETOP-ASSET-PROBE] {json.dumps(item, sort_keys=True)}", flush=True)


def tensor_mean(value) -> float | None:
    if value is None or not hasattr(value, "float"):
        return None
    return float(value.float().mean().detach().cpu().item())


def tensor_xy_speed_mean(value) -> float | None:
    if value is None or not hasattr(value, "float"):
        return None
    return float(torch.norm(value.float()[..., :2], dim=-1).mean().detach().cpu().item())


def tensor_histogram(value, bins: int) -> list[int] | None:
    if value is None or not hasattr(value, "long"):
        return None
    return torch.bincount(value.long(), minlength=bins).detach().cpu().tolist()


def main() -> None:
    carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    record("parse_cfg_start", task=args_cli.task)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.speed_curriculum_alpha is not None:
        env_cfg.dynamic_grasp_speed_curriculum_override_alpha = float(args_cli.speed_curriculum_alpha)
    if args_cli.asset_curriculum_alpha is not None:
        env_cfg.tabletop_asset_curriculum_override_alpha = float(args_cli.asset_curriculum_alpha)
    if args_cli.motion_curriculum_alpha is not None:
        env_cfg.tabletop_motion_mode_curriculum_override_alpha = float(args_cli.motion_curriculum_alpha)
    record(
        "parse_cfg_done",
        observation_space=int(env_cfg.observation_space),
        action_space=int(env_cfg.action_space),
        default_arm_pos=list(env_cfg.default_arm_pos),
        arm_action_scale=float(env_cfg.arm_action_scale),
        arm_moving_average=float(env_cfg.arm_moving_average),
        initial_arm_target_lock_steps=int(env_cfg.initial_arm_target_lock_steps),
        initial_hand_target_lock_steps=int(env_cfg.initial_hand_target_lock_steps),
        scripted_action_prior_enabled=bool(env_cfg.scripted_action_prior_enabled),
        scripted_tabletop_pregrasp_prior_enabled=bool(
            env_cfg.scripted_tabletop_pregrasp_prior_enabled
        ),
        scripted_tabletop_relative_lift_target_prior_enabled=bool(
            env_cfg.scripted_tabletop_relative_lift_target_prior_enabled
        ),
        scripted_tabletop_hand_grasp_memory_prior_enabled=bool(
            env_cfg.scripted_tabletop_hand_grasp_memory_prior_enabled
        ),
        asset_count=len(tuple(getattr(env_cfg, "tabletop_object_asset_specs", ()))),
        asset_ids=[
            str(spec.get("asset_id", ""))
            for spec in tuple(getattr(env_cfg, "tabletop_object_asset_specs", ()))
        ],
        persistent_motion=bool(getattr(env_cfg, "dynamic_tabletop_persistent_motion", False)),
        motion_modes=list(getattr(env_cfg, "tabletop_motion_modes", ())),
        success_requires_hover=bool(getattr(env_cfg, "tabletop_success_requires_hover_target", False)),
        hover_height_delta=float(getattr(env_cfg, "tabletop_hover_height_delta", 0.0)),
        speed_curriculum_alpha=getattr(env_cfg, "dynamic_grasp_speed_curriculum_override_alpha", None),
        asset_curriculum_alpha=getattr(env_cfg, "tabletop_asset_curriculum_override_alpha", None),
        motion_curriculum_alpha=getattr(env_cfg, "tabletop_motion_mode_curriculum_override_alpha", None),
    )
    env = None
    try:
        record("gym_make_start")
        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        record(
            "gym_make_done",
            num_envs=int(unwrapped.num_envs),
            has_asset_set=bool(getattr(unwrapped.cfg, "tabletop_asset_set_enabled", False)),
            object_count=len(getattr(unwrapped, "_tabletop_objects", [])),
        )
        record("reset_start")
        obs, _ = env.reset(seed=42)
        policy_obs = obs["policy"]
        log = dict(getattr(unwrapped, "extras", {}).get("log", {}))
        record(
            "reset_done",
            obs_shape=list(policy_obs.shape),
            active_asset_id_mean=tensor_mean(getattr(unwrapped, "_tabletop_active_asset_ids", None)),
            active_asset_histogram=tensor_histogram(
                getattr(unwrapped, "_tabletop_active_asset_ids", None),
                len(getattr(unwrapped, "_tabletop_objects", ())),
            ),
            object_z_mean=float((unwrapped._object_pos_w[:, 2] - unwrapped.scene.env_origins[:, 2]).mean().detach().cpu()),
            tabletop_cmd_speed=tensor_mean(log.get("tabletop_cmd_speed")),
            direct_cmd_speed=tensor_xy_speed_mean(getattr(unwrapped, "_tabletop_cmd_lin_vel_w", None)),
            object_xy_speed=tensor_xy_speed_mean(getattr(unwrapped, "_object_lin_vel_w", None)),
            asset_count=tensor_mean(log.get("tabletop_asset_count")),
            persistent_motion=tensor_mean(log.get("tabletop_persistent_motion")),
            palm_distance=tensor_mean(log.get("palm_distance")),
            strict_finger_contacts=tensor_mean(log.get("strict_finger_contacts")),
            strict_thumb_contact=tensor_mean(log.get("strict_thumb_contact")),
            strict_true_grasp=tensor_mean(log.get("strict_true_grasp")),
            hover_latched=tensor_mean(log.get("hover_latched")),
            hover_xy_dist=tensor_mean(log.get("hover_xy_dist")),
            hover_z_error=tensor_mean(log.get("hover_z_error")),
            hover_target_rew=tensor_mean(log.get("hover_target_rew")),
            hover_stability_rew=tensor_mean(log.get("hover_stability_rew")),
            hover_object_speed=tensor_mean(log.get("hover_object_speed")),
        )
        actions = torch.zeros((unwrapped.num_envs, unwrapped.cfg.action_space), device=unwrapped.device)
        for step in range(int(args_cli.steps)):
            obs, rew, terminated, truncated, extras = env.step(actions)
            log = dict(extras.get("log", {}))
            record(
                "step",
                step=step + 1,
                reward_mean=float(rew.mean().detach().cpu()),
                done_count=int((terminated | truncated).sum().detach().cpu()),
                obs_shape=list(obs["policy"].shape),
                active_asset_id_mean=tensor_mean(getattr(unwrapped, "_tabletop_active_asset_ids", None)),
                active_asset_histogram=tensor_histogram(
                    getattr(unwrapped, "_tabletop_active_asset_ids", None),
                    len(getattr(unwrapped, "_tabletop_objects", ())),
                ),
                object_z_mean=float((unwrapped._object_pos_w[:, 2] - unwrapped.scene.env_origins[:, 2]).mean().detach().cpu()),
                tabletop_cmd_speed=tensor_mean(log.get("tabletop_cmd_speed")),
                direct_cmd_speed=tensor_xy_speed_mean(getattr(unwrapped, "_tabletop_cmd_lin_vel_w", None)),
                object_xy_speed=tensor_xy_speed_mean(getattr(unwrapped, "_object_lin_vel_w", None)),
                asset_count=tensor_mean(log.get("tabletop_asset_count")),
                motion_mode_count=tensor_mean(log.get("tabletop_motion_mode_count")),
                palm_distance=tensor_mean(log.get("palm_distance")),
                strict_finger_contacts=tensor_mean(log.get("strict_finger_contacts")),
                strict_thumb_contact=tensor_mean(log.get("strict_thumb_contact")),
                strict_true_grasp=tensor_mean(log.get("strict_true_grasp")),
                hover_latched=tensor_mean(log.get("hover_latched")),
                hover_xy_dist=tensor_mean(log.get("hover_xy_dist")),
                hover_z_error=tensor_mean(log.get("hover_z_error")),
                hover_target_rew=tensor_mean(log.get("hover_target_rew")),
                hover_stability_rew=tensor_mean(log.get("hover_stability_rew")),
                hover_object_speed=tensor_mean(log.get("hover_object_speed")),
            )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        record("error", error_type=type(exc).__name__, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        simulation_app.close()
