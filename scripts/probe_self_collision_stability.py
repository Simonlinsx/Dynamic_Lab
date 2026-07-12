#!/usr/bin/env python3
"""Smoke-test robot self-collision stability for SimToolReal IsaacLab tasks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToolReal-Revo2-Franka-FallingBaton-Teacher-Direct-v0")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--action-mode", choices=("zeros", "close", "random"), default="close")
parser.add_argument("--close-start-step", type=int, default=20)
parser.add_argument("--random-scale", type=float, default=0.25)
parser.add_argument("--log-interval", type=int, default=60)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--summary-out", default=None)
parser.add_argument(
    "--disable-self-collision",
    action="store_true",
    help="Runtime override for A/B probing; leaves task config files unchanged.",
)
parser.add_argument(
    "--object-start-pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Runtime override for object_start_pos and object_cfg init pose.",
)
parser.add_argument(
    "--zero-object-pos-noise",
    action="store_true",
    help="Runtime override that disables reset_object_pos_noise.",
)
parser.add_argument(
    "--arm-pos",
    type=float,
    nargs=7,
    default=None,
    metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    help="Runtime override for Franka reset/default arm joint positions.",
)
parser.add_argument("--no-table", action="store_true", help="Runtime override that disables table creation.")
parser.add_argument(
    "--table-pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Runtime override for table_cfg init pose.",
)
parser.add_argument(
    "--table-size",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Runtime override for table cuboid size.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


def _trace(message: str) -> None:
    print(f"[SELF-COLLISION] {message}", flush=True)


def _finite_stats(name: str, tensor: torch.Tensor, summary: dict) -> None:
    finite = torch.isfinite(tensor)
    nonfinite = int((~finite).sum().item())
    if finite.any():
        finite_tensor = tensor[finite]
        max_abs = float(finite_tensor.abs().max().item())
        min_value = float(finite_tensor.min().item())
        max_value = float(finite_tensor.max().item())
    else:
        max_abs = math.inf
        min_value = math.nan
        max_value = math.nan
    summary[f"{name}_nonfinite"] = nonfinite
    summary[f"{name}_max_abs"] = max_abs
    summary[f"{name}_min"] = min_value
    summary[f"{name}_max"] = max_value


def _joint_peak_stats(prefix: str, values: torch.Tensor, joint_names: list[str] | tuple[str, ...], summary: dict) -> None:
    finite_values = torch.nan_to_num(values.detach(), nan=0.0, posinf=math.inf, neginf=-math.inf)
    flat_abs = finite_values.abs().flatten()
    if flat_abs.numel() == 0:
        return
    flat_id = int(torch.argmax(flat_abs).item())
    joint_count = int(values.shape[-1])
    env_id = flat_id // joint_count
    joint_id = flat_id % joint_count
    summary[f"{prefix}_peak_env"] = int(env_id)
    summary[f"{prefix}_peak_joint_id"] = int(joint_id)
    if joint_id < len(joint_names):
        summary[f"{prefix}_peak_joint"] = str(joint_names[joint_id])
    summary[f"{prefix}_peak_value"] = float(values.reshape(-1)[flat_id].item())


def _make_action(unwrapped, step: int) -> torch.Tensor:
    action_dim = int(unwrapped.cfg.action_space)
    actions = torch.zeros((unwrapped.num_envs, action_dim), device=unwrapped.device)
    if args_cli.action_mode == "random":
        actions = float(args_cli.random_scale) * (2.0 * torch.rand_like(actions) - 1.0)
    elif args_cli.action_mode == "close" and step >= int(args_cli.close_start_step) and action_dim > 7:
        actions[:, 7:] = 1.0
    return actions


def _collect_step_summary(unwrapped, step: int) -> dict:
    robot = unwrapped.robot
    summary = {"step": int(step)}
    _finite_stats("joint_pos", robot.data.joint_pos, summary)
    _finite_stats("joint_vel", robot.data.joint_vel, summary)
    _joint_peak_stats("joint_pos", robot.data.joint_pos, robot.joint_names, summary)
    _joint_peak_stats("joint_vel", robot.data.joint_vel, robot.joint_names, summary)
    if hasattr(robot.data, "root_state_w"):
        _finite_stats("robot_root_state", robot.data.root_state_w, summary)
    if hasattr(robot.data, "body_pos_w"):
        _finite_stats("body_pos", robot.data.body_pos_w, summary)
        body_pos_local = robot.data.body_pos_w - unwrapped.scene.env_origins[:, None, :]
        z_values = body_pos_local[..., 2]
        flat_id = int(torch.argmin(z_values.flatten()).item())
        body_count = int(z_values.shape[-1])
        body_id = flat_id % body_count
        summary["body_z_min"] = float(z_values.flatten()[flat_id].item())
        summary["body_z_min_env"] = int(flat_id // body_count)
        summary["body_z_min_body_id"] = int(body_id)
        if body_id < len(robot.body_names):
            summary["body_z_min_body"] = str(robot.body_names[body_id])
    obj = getattr(unwrapped, "object", None)
    if obj is not None and hasattr(obj.data, "root_state_w"):
        _finite_stats("object_root_state", obj.data.root_state_w, summary)
        object_pos_local = obj.data.root_pos_w - unwrapped.scene.env_origins
        summary["object_z_min"] = float(object_pos_local[:, 2].min().item())
        summary["object_z_max"] = float(object_pos_local[:, 2].max().item())
    if hasattr(unwrapped, "_compute_intermediate_values"):
        unwrapped._compute_intermediate_values()
    if hasattr(unwrapped, "_palm_pos_w"):
        palm_pos_local = unwrapped._palm_pos_w - unwrapped.scene.env_origins
        summary["palm_x_mean"] = float(palm_pos_local[:, 0].mean().item())
        summary["palm_y_mean"] = float(palm_pos_local[:, 1].mean().item())
        summary["palm_z_min"] = float(palm_pos_local[:, 2].min().item())
        summary["palm_z_max"] = float(palm_pos_local[:, 2].max().item())
    if hasattr(unwrapped, "_fingertip_pos_w"):
        fingertip_pos_local = unwrapped._fingertip_pos_w - unwrapped.scene.env_origins[:, None, :]
        tip_z = fingertip_pos_local[..., 2]
        flat_id = int(torch.argmin(tip_z.flatten()).item())
        finger_count = int(tip_z.shape[-1])
        finger_id = flat_id % finger_count
        summary["fingertip_z_min"] = float(tip_z.flatten()[flat_id].item())
        summary["fingertip_z_min_env"] = int(flat_id // finger_count)
        summary["fingertip_z_min_finger_id"] = int(finger_id)
        fingertip_names = list(getattr(unwrapped.cfg, "fingertip_body_names", ()))
        if finger_id < len(fingertip_names):
            summary["fingertip_z_min_finger"] = str(fingertip_names[finger_id])
        if hasattr(unwrapped.cfg, "table_top_z"):
            summary["fingertip_table_clearance_min"] = float(tip_z.min().item() - float(unwrapped.cfg.table_top_z))
    if hasattr(unwrapped, "_tabletop_arm_clearance_min_margin"):
        summary["tabletop_arm_clearance_min_margin"] = float(unwrapped._tabletop_arm_clearance_min_margin.min().item())
        summary["tabletop_arm_clearance_penalty_max"] = float(unwrapped._tabletop_arm_clearance_penalty.max().item())
        summary["tabletop_arm_clearance_ok_rate"] = float(unwrapped._tabletop_arm_clearance_ok.float().mean().item())
    if hasattr(unwrapped, "_surface_dist"):
        surface_dist = unwrapped._surface_dist
        summary["surface_dist_min"] = float(surface_dist.min().item())
        summary["surface_dist_mean"] = float(surface_dist.mean().item())
        flat_id = int(torch.argmin(surface_dist.flatten()).item())
        finger_count = int(surface_dist.shape[-1])
        summary["surface_dist_min_env"] = int(flat_id // finger_count)
        summary["surface_dist_min_finger_id"] = int(flat_id % finger_count)
        fingertip_names = list(getattr(unwrapped.cfg, "fingertip_body_names", ()))
        if summary["surface_dist_min_finger_id"] < len(fingertip_names):
            summary["surface_dist_min_finger"] = str(fingertip_names[summary["surface_dist_min_finger_id"]])
    if hasattr(unwrapped, "_contact_score"):
        contact_score = unwrapped._contact_score
        summary["contact_score_max"] = float(contact_score.max().item())
        summary["contact_score_mean"] = float(contact_score.mean().item())
    if hasattr(unwrapped, "_joint_targets"):
        _finite_stats("joint_targets", unwrapped._joint_targets, summary)
    if hasattr(unwrapped, "_control_hand_joint_ids"):
        hand_ids = list(unwrapped._control_hand_joint_ids)
        hand_names = [unwrapped.robot.joint_names[joint_id] for joint_id in hand_ids]
        hand_pos = robot.data.joint_pos[:, hand_ids]
        summary["hand_joint_names"] = hand_names
        summary["hand_joint_pos_env0"] = [float(v) for v in hand_pos[0].detach().cpu()]
        summary["hand_joint_lower"] = [
            float(v) for v in unwrapped._joint_lower_limits[hand_ids].detach().cpu()
        ]
        summary["hand_joint_upper"] = [
            float(v) for v in unwrapped._joint_upper_limits[hand_ids].detach().cpu()
        ]
        if hasattr(unwrapped, "_default_joint_pos"):
            summary["hand_joint_default_env0"] = [
                float(v) for v in unwrapped._default_joint_pos[0, hand_ids].detach().cpu()
            ]
        if hasattr(unwrapped, "_joint_targets"):
            summary["hand_joint_target_env0"] = [
                float(v) for v in unwrapped._joint_targets[0, hand_ids].detach().cpu()
            ]
            target_error = hand_pos[0] - unwrapped._joint_targets[0, hand_ids]
            summary["hand_joint_target_error_env0"] = [float(v) for v in target_error.detach().cpu()]
    for attr_name in ("_robot_self_collision_filter_pair_count", "_robot_self_collision_filter_fail_count"):
        if hasattr(unwrapped, attr_name):
            summary[attr_name[1:]] = int(getattr(unwrapped, attr_name))
    return summary


def _has_bad_values(summary: dict) -> bool:
    for key, value in summary.items():
        if key.endswith("_nonfinite") and int(value) > 0:
            return True
        if key.endswith("_max_abs") and (not math.isfinite(float(value)) or float(value) > 1.0e4):
            return True
    return False


def main() -> None:
    carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    torch.manual_seed(int(args_cli.seed))
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    if hasattr(env_cfg, "terminate_on_success"):
        env_cfg.terminate_on_success = False
    if args_cli.no_table and hasattr(env_cfg, "create_table"):
        env_cfg.create_table = False
    if args_cli.table_pos is not None and hasattr(env_cfg, "table_cfg"):
        env_cfg.table_cfg.init_state.pos = tuple(float(value) for value in args_cli.table_pos)
    if args_cli.table_size is not None and hasattr(env_cfg, "table_cfg"):
        table_spawn = getattr(env_cfg.table_cfg, "spawn", None)
        if table_spawn is not None and hasattr(table_spawn, "size"):
            table_spawn.size = tuple(float(value) for value in args_cli.table_size)
    if args_cli.zero_object_pos_noise and hasattr(env_cfg, "reset_object_pos_noise"):
        env_cfg.reset_object_pos_noise = (0.0, 0.0, 0.0)
    if args_cli.arm_pos is not None:
        arm_pos = tuple(float(value) for value in args_cli.arm_pos)
        if hasattr(env_cfg, "default_arm_pos"):
            env_cfg.default_arm_pos = arm_pos
        if hasattr(env_cfg, "robot_cfg") and hasattr(env_cfg.robot_cfg, "init_state"):
            joint_pos = dict(getattr(env_cfg.robot_cfg.init_state, "joint_pos", {}))
            for joint_name, value in zip(("panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"), arm_pos):
                joint_pos[joint_name] = value
            env_cfg.robot_cfg.init_state.joint_pos = joint_pos
    if args_cli.object_start_pos is not None:
        object_start_pos = tuple(float(value) for value in args_cli.object_start_pos)
        if hasattr(env_cfg, "object_start_pos"):
            env_cfg.object_start_pos = object_start_pos
        if hasattr(env_cfg, "object_cfg") and hasattr(env_cfg.object_cfg, "init_state"):
            env_cfg.object_cfg.init_state.pos = object_start_pos

    spawn_cfg = env_cfg.robot_cfg.spawn
    articulation_props = getattr(spawn_cfg, "articulation_props", None)
    if args_cli.disable_self_collision:
        spawn_cfg.self_collision = False
        if articulation_props is not None:
            articulation_props.enabled_self_collisions = False
    self_collision = bool(getattr(spawn_cfg, "self_collision", False))
    enabled_self_collisions = bool(getattr(articulation_props, "enabled_self_collisions", False))
    _trace(
        f"making env task={args_cli.task} num_envs={args_cli.num_envs} device={args_cli.device} "
        f"self_collision={self_collision} enabled_self_collisions={enabled_self_collisions}"
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    step_summaries = []
    bad = False
    completed_episodes = 0
    try:
        env.reset(seed=int(args_cli.seed))
        unwrapped = env.unwrapped
        reset_summary = _collect_step_summary(unwrapped, 0)
        step_summaries.append(reset_summary)
        bad |= _has_bad_values(reset_summary)
        _trace(f"reset {json.dumps(reset_summary, sort_keys=True)}")

        for step in range(int(args_cli.steps)):
            actions = _make_action(unwrapped, step)
            _, _, terminated, truncated, extras = env.step(actions)
            dones = terminated | truncated
            completed_episodes += int(dones.sum().item())
            should_log = (
                step == 0
                or step + 1 == int(args_cli.steps)
                or (args_cli.log_interval > 0 and (step + 1) % int(args_cli.log_interval) == 0)
                or bool(dones.any().item())
            )
            if should_log:
                summary = _collect_step_summary(unwrapped, step + 1)
                summary["done_count"] = int(dones.sum().item())
                for key in ("success_env", "true_grasp_env", "lifted_env", "dropped_env", "out_xy_env", "time_out_env"):
                    value = extras.get(key)
                    if value is not None and hasattr(value, "sum"):
                        summary[key] = int(value.sum().item())
                step_summaries.append(summary)
                bad |= _has_bad_values(summary)
                _trace(f"step {step + 1} {json.dumps(summary, sort_keys=True)}")

        final = _collect_step_summary(unwrapped, int(args_cli.steps))
        bad |= _has_bad_values(final)
        final_summary = {
            "task": args_cli.task,
            "action_mode": args_cli.action_mode,
            "num_envs": int(args_cli.num_envs),
            "steps": int(args_cli.steps),
            "self_collision": self_collision,
            "enabled_self_collisions": enabled_self_collisions,
            "completed_episodes": int(completed_episodes),
            "bad_values": bool(bad),
            "final": final,
            "logs": step_summaries,
        }
        print(json.dumps(final_summary, indent=2, sort_keys=True), flush=True)
        if args_cli.summary_out:
            out_path = Path(args_cli.summary_out).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(final_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if bad:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
