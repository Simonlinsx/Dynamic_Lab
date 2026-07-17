#!/usr/bin/env python3
"""Probe Revo2 static-grasp reset geometry and scripted-prior rollout metrics."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="SimToolReal-Revo2-Franka-StaticBall-ResidualGrasp-Direct-v0",
    help="Gym task id.",
)
parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs.")
parser.add_argument("--steps", type=int, default=0, help="Number of zero-policy control steps to run.")
parser.add_argument("--log-interval", type=int, default=50, help="Metric print interval while stepping.")
parser.add_argument("--object-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--table-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--robot-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument(
    "--robot-base-z-offset",
    type=float,
    default=0.0,
    help="Raise the fixed robot base relative to the configured table.",
)
parser.add_argument("--robot-rot-wxyz", type=float, nargs=4, default=None, metavar=("W", "X", "Y", "Z"))
parser.add_argument("--arm-pos", type=float, nargs=7, default=None, metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"))
parser.add_argument("--object-y-sweep", type=float, nargs="+", default=None)
parser.add_argument("--object-x-sweep", type=float, nargs="+", default=None)
parser.add_argument("--pregrasp-target-rel-palm", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--hand-start-step", type=int, default=None)
parser.add_argument("--hand-ramp-steps", type=int, default=None)
parser.add_argument("--lift-start-step", type=int, default=None)
parser.add_argument("--lift-steps", type=int, default=None)
parser.add_argument("--lift-requires-grasp", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--initial-arm-lock-steps", type=int, default=None)
parser.add_argument("--initial-hand-lock-steps", type=int, default=None)
parser.add_argument("--hand-action", type=float, default=None)
parser.add_argument(
    "--hand-fractions",
    type=float,
    nargs=6,
    default=None,
    metavar=("TH_META", "TH_FLEX", "INDEX", "MIDDLE", "RING", "PINKY"),
)
parser.add_argument("--lift-action", type=float, nargs=7, default=None, metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"))
parser.add_argument("--lift-action-scale-sweep", type=float, nargs="+", default=None)
parser.add_argument("--lift-action-basis-sweep", action="store_true")
parser.add_argument("--print-hand-state", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


def _trace(message: str) -> None:
    print(f"[PROBE] {message}", flush=True)


def _load_cfg(entry_point: str):
    module_name, class_name = entry_point.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


def _fmt_vec(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().tolist()
    return "[" + ", ".join(f"{float(value): .6f}" for value in values) + "]"


def _fmt_mat(tensor: torch.Tensor) -> str:
    rows = tensor.detach().cpu()
    return "[" + ", ".join(_fmt_vec(row) for row in rows) + "]"


def _print_state(env, label: str) -> None:
    unwrapped = env.unwrapped
    unwrapped._compute_intermediate_values()
    origins = unwrapped.scene.env_origins
    env_id = 0

    palm = unwrapped._palm_pos_w[env_id] - origins[env_id]
    obj = unwrapped._object_pos_w[env_id] - origins[env_id]
    touch = unwrapped._fingertip_pos_w[env_id] - origins[env_id]
    rel_w = unwrapped._object_pos_w[env_id] - unwrapped._palm_pos_w[env_id]
    rel_palm = quat_rotate_inverse(unwrapped._palm_quat_w[env_id].unsqueeze(0), rel_w.unsqueeze(0))[0]
    target = unwrapped._pregrasp_target_rel_palm[0]
    scale = unwrapped._pregrasp_target_scale[0]
    pregrasp_error = (rel_palm - target) / scale

    log = unwrapped.extras.get("log", {})
    log_items = []
    for key in ("true_grasp", "lifted", "stable_hold", "object_height_delta", "palm_distance", "min_surface_dist"):
        value = log.get(key)
        if value is not None and hasattr(value, "item"):
            log_items.append(f"{key}={float(value):.6f}")

    _trace(f"{label} palm_local={_fmt_vec(palm)} object_local={_fmt_vec(obj)}")
    _trace(f"{label} touch_local={_fmt_mat(touch)}")
    _trace(
        f"{label} object_minus_palm_world={_fmt_vec(rel_w)} "
        f"object_minus_palm_local={_fmt_vec(rel_palm)} "
        f"target={_fmt_vec(target)} scaled_error={_fmt_vec(pregrasp_error)}"
    )
    _trace(f"{label} surface_dist={_fmt_vec(unwrapped._surface_dist[env_id])}")
    if log_items:
        _trace(f"{label} " + " ".join(log_items))


def _print_sweep_state(env, label: str) -> None:
    unwrapped = env.unwrapped
    unwrapped._compute_intermediate_values()
    origins = unwrapped.scene.env_origins
    rel_w = unwrapped._object_pos_w - unwrapped._palm_pos_w
    rel_palm = quat_rotate_inverse(unwrapped._palm_quat_w, rel_w)
    palm = unwrapped._palm_pos_w - origins
    obj = unwrapped._object_pos_w - origins
    min_surface = unwrapped._surface_dist.min(dim=-1).values
    for env_id in range(unwrapped.num_envs):
        _trace(
            f"{label} env={env_id:02d} palm_local={_fmt_vec(palm[env_id])} "
            f"object_local={_fmt_vec(obj[env_id])} "
            f"object_minus_palm_world={_fmt_vec(rel_w[env_id])} "
            f"object_minus_palm_local={_fmt_vec(rel_palm[env_id])} min_surface={float(min_surface[env_id]):.6f} "
            f"height_delta={float(unwrapped._object_height_delta[env_id]):.6f} "
            f"true_grasp={int(unwrapped._true_grasp[env_id])} lifted={int(unwrapped._lifted[env_id])} "
            f"stable={int(unwrapped._stable_hold[env_id])}"
        )


def _print_diagnostics(env, label: str) -> None:
    unwrapped = env.unwrapped
    _trace(
        f"{label} robot_material_bind_count={getattr(unwrapped, '_robot_material_bind_count', -1)} "
        f"robot_material_bind_fail_count={getattr(unwrapped, '_robot_material_bind_fail_count', -1)}"
    )
    if not args_cli.print_hand_state:
        return
    hand_pos = unwrapped.robot.data.joint_pos[:, unwrapped._hand_joint_ids]
    hand_target = unwrapped._joint_targets[:, unwrapped._hand_joint_ids]
    arm_pos = unwrapped.robot.data.joint_pos[:, unwrapped._arm_joint_ids]
    arm_target = unwrapped._joint_targets[:, unwrapped._arm_joint_ids]
    for env_id in range(unwrapped.num_envs):
        _trace(
            f"{label} env={env_id:02d} arm_pos={_fmt_vec(arm_pos[env_id])} "
            f"arm_target={_fmt_vec(arm_target[env_id])}"
        )
        _trace(
            f"{label} env={env_id:02d} hand_pos={_fmt_vec(hand_pos[env_id])} "
            f"hand_target={_fmt_vec(hand_target[env_id])}"
        )
    if label == "reset":
        hand_ids = unwrapped._hand_joint_ids
        _trace(f"{label} hand_joint_names={unwrapped._hand_joint_names}")
        _trace(f"{label} hand_lower={_fmt_vec(unwrapped._joint_lower_limits[hand_ids])}")
        _trace(f"{label} hand_upper={_fmt_vec(unwrapped._joint_upper_limits[hand_ids])}")
        _trace(f"{label} hand_stiffness={_fmt_vec(unwrapped.robot.data.joint_stiffness[0, hand_ids])}")
        _trace(f"{label} hand_damping={_fmt_vec(unwrapped.robot.data.joint_damping[0, hand_ids])}")
        _trace(f"{label} hand_effort_limits={_fmt_vec(unwrapped.robot.data.joint_effort_limits[0, hand_ids])}")


def _done_count(extras: dict, key: str) -> int:
    value = extras.get(key)
    if value is None or not hasattr(value, "sum"):
        return 0
    return int(value.sum().item())


def main() -> None:
    spec = gym.spec(args_cli.task)
    cfg = _load_cfg(spec.kwargs["env_cfg_entry_point"])
    cfg.sim.device = args_cli.device
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.object_y_sweep is not None and args_cli.object_x_sweep is not None:
        cfg.scene.num_envs = len(args_cli.object_x_sweep) * len(args_cli.object_y_sweep)
        cfg.reset_object_pos_noise = (0.0, 0.0, 0.0)
    elif args_cli.object_y_sweep is not None:
        cfg.scene.num_envs = len(args_cli.object_y_sweep)
        cfg.reset_object_pos_noise = (0.0, 0.0, 0.0)
    if args_cli.lift_action_scale_sweep is not None:
        cfg.scene.num_envs = len(args_cli.lift_action_scale_sweep)
        cfg.reset_object_pos_noise = (0.0, 0.0, 0.0)
    if args_cli.lift_action_basis_sweep:
        cfg.scene.num_envs = 14
        cfg.reset_object_pos_noise = (0.0, 0.0, 0.0)
    if args_cli.object_pos is not None:
        cfg.object_start_pos = tuple(args_cli.object_pos)
        cfg.object_cfg.init_state.pos = tuple(args_cli.object_pos)
    if args_cli.table_pos is not None:
        cfg.table_cfg.init_state.pos = tuple(args_cli.table_pos)
    if args_cli.robot_pos is not None:
        cfg.robot_cfg.init_state.pos = tuple(args_cli.robot_pos)
    elif float(args_cli.robot_base_z_offset) != 0.0:
        root_pos = tuple(float(value) for value in cfg.robot_cfg.init_state.pos)
        cfg.robot_cfg.init_state.pos = (
            root_pos[0],
            root_pos[1],
            root_pos[2] + float(args_cli.robot_base_z_offset),
        )
    if args_cli.robot_rot_wxyz is not None:
        cfg.robot_cfg.init_state.rot = tuple(args_cli.robot_rot_wxyz)
    if args_cli.arm_pos is not None:
        cfg.default_arm_pos = tuple(args_cli.arm_pos)
        for joint_name, value in zip(cfg.arm_joint_names, args_cli.arm_pos):
            cfg.robot_cfg.init_state.joint_pos[joint_name] = value
    if args_cli.pregrasp_target_rel_palm is not None:
        cfg.pregrasp_target_rel_palm = tuple(args_cli.pregrasp_target_rel_palm)
    if args_cli.hand_start_step is not None:
        cfg.scripted_action_prior_hand_start_step = args_cli.hand_start_step
    if args_cli.hand_ramp_steps is not None:
        cfg.scripted_action_prior_hand_ramp_steps = args_cli.hand_ramp_steps
    if args_cli.lift_start_step is not None:
        cfg.scripted_action_prior_lift_start_step = args_cli.lift_start_step
    if args_cli.lift_steps is not None:
        cfg.scripted_action_prior_lift_steps = args_cli.lift_steps
    if args_cli.lift_requires_grasp is not None:
        cfg.scripted_action_prior_lift_requires_grasp = args_cli.lift_requires_grasp
    if args_cli.initial_arm_lock_steps is not None:
        cfg.initial_arm_target_lock_steps = args_cli.initial_arm_lock_steps
    if args_cli.initial_hand_lock_steps is not None:
        cfg.initial_hand_target_lock_steps = args_cli.initial_hand_lock_steps
    if args_cli.hand_action is not None:
        cfg.scripted_action_prior_hand_action = args_cli.hand_action
    if args_cli.hand_fractions is not None:
        cfg.reference_hand_fractions = tuple(args_cli.hand_fractions)
    if args_cli.lift_action is not None:
        cfg.scripted_action_prior_lift_action = tuple(args_cli.lift_action)
        cfg.lift_action_prior = tuple(args_cli.lift_action)

    _trace(f"making env: task={args_cli.task}, num_envs={cfg.scene.num_envs}, device={cfg.sim.device}")
    env = gym.make(args_cli.task, cfg=cfg)
    try:
        _trace(
            f"episode_length_s={env.unwrapped.cfg.episode_length_s} "
            f"max_episode_length={env.unwrapped.max_episode_length} "
            f"sim_dt={env.unwrapped.cfg.sim.dt} decimation={env.unwrapped.cfg.decimation}"
        )
        if args_cli.object_y_sweep is not None and args_cli.object_x_sweep is not None:
            base_pos = tuple(args_cli.object_pos or cfg.object_start_pos)
            positions = []
            for x_value in args_cli.object_x_sweep:
                for y_value in args_cli.object_y_sweep:
                    positions.append([x_value, y_value, base_pos[2]])
                    _trace(f"xy_sweep env={len(positions) - 1:02d} object_pos={[x_value, y_value, base_pos[2]]}")
            env.unwrapped._object_start_pos = torch.tensor(
                positions, device=env.unwrapped.device, dtype=torch.float32
            )
        elif args_cli.object_y_sweep is not None:
            base_pos = tuple(args_cli.object_pos or cfg.object_start_pos)
            positions = [[base_pos[0], y_value, base_pos[2]] for y_value in args_cli.object_y_sweep]
            env.unwrapped._object_start_pos = torch.tensor(
                positions, device=env.unwrapped.device, dtype=torch.float32
            )
        if args_cli.lift_action_scale_sweep is not None:
            base_action = torch.tensor(
                args_cli.lift_action or cfg.scripted_action_prior_lift_action,
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
            scales = torch.tensor(args_cli.lift_action_scale_sweep, device=env.unwrapped.device, dtype=torch.float32)
            swept_action = torch.clamp(scales.unsqueeze(-1) * base_action.unsqueeze(0), -1.0, 1.0)
            env.unwrapped._scripted_lift_action = swept_action
            env.unwrapped._lift_action_prior = swept_action
            env.unwrapped._lift_action_prior_den = torch.clamp(torch.sum(torch.abs(swept_action), dim=-1), min=1.0e-6)
        if args_cli.lift_action_basis_sweep:
            basis = torch.eye(7, device=env.unwrapped.device, dtype=torch.float32)
            swept_action = torch.cat((basis, -basis), dim=0)
            env.unwrapped._scripted_lift_action = swept_action
            env.unwrapped._lift_action_prior = swept_action
            env.unwrapped._lift_action_prior_den = torch.ones(env.unwrapped.num_envs, device=env.unwrapped.device)
            for env_id, action in enumerate(swept_action.detach().cpu().tolist()):
                _trace(f"basis env={env_id:02d} lift_action={[round(float(value), 3) for value in action]}")
        env.reset()
        if (
            args_cli.object_y_sweep is not None
            or args_cli.object_x_sweep is not None
            or args_cli.lift_action_scale_sweep is not None
            or args_cli.lift_action_basis_sweep
        ):
            _print_sweep_state(env, "reset")
        else:
            _print_state(env, "reset")
        _print_diagnostics(env, "reset")

        zero_actions = torch.zeros(
            (env.unwrapped.num_envs, env.unwrapped.cfg.action_space), device=env.unwrapped.device
        )
        for step in range(args_cli.steps):
            _, _, terminated, truncated, _ = env.step(zero_actions)
            done = terminated | truncated
            should_log = (
                step == 0
                or torch.any(done).item()
                or (args_cli.log_interval > 0 and (step + 1) % args_cli.log_interval == 0)
            )
            if should_log:
                if (
                    args_cli.object_y_sweep is not None
                    or args_cli.object_x_sweep is not None
                    or args_cli.lift_action_scale_sweep is not None
                    or args_cli.lift_action_basis_sweep
                ):
                    _print_sweep_state(env, f"step={step + 1:04d}")
                else:
                    _print_state(env, f"step={step + 1:04d}")
                _print_diagnostics(env, f"step={step + 1:04d}")
                _trace(
                    f"step={step + 1:04d} done={int(done.sum())} "
                    f"terminated={int(terminated.sum())} truncated={int(truncated.sum())} "
                    f"dropped={_done_count(env.unwrapped.extras, 'dropped_env')} "
                    f"out_xy={_done_count(env.unwrapped.extras, 'out_xy_env')} "
                    f"success={_done_count(env.unwrapped.extras, 'success_env')} "
                    f"time_out={_done_count(env.unwrapped.extras, 'time_out_env')} "
                    f"episode_length={_fmt_vec(env.unwrapped.episode_length_buf.float())}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
