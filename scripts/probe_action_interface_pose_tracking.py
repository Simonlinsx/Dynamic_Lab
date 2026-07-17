#!/usr/bin/env python3
"""Track one palm pose through either the joint-target or Cartesian policy interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
EXT_SOURCE = REPO_ROOT / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--warmup-steps", type=int, default=60)
parser.add_argument("--track-steps", type=int, default=180)
parser.add_argument("--target-offset", type=float, nargs=3, default=(0.06, 0.03, 0.05))
parser.add_argument("--hand-action", type=float, default=-1.0)
parser.add_argument("--damping", type=float, default=0.08)
parser.add_argument("--max-joint-delta", type=float, default=0.04)
parser.add_argument("--seed", type=int, default=142)
parser.add_argument(
    "--robot-base-z-offset",
    type=float,
    default=0.0,
    help="Raise the fixed robot base relative to the table for collision-free reset probes.",
)
parser.add_argument("--output-json", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    compute_pose_error,
    matrix_from_quat,
    subtract_frame_transforms,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


def _palm_jacobian(base, palm_quat_w: torch.Tensor) -> torch.Tensor:
    robot = base.robot
    arm_ids = base._arm_joint_ids
    palm_body_id = int(base._palm_body_id)
    jacobian_row = palm_body_id - 1 if robot.is_fixed_base else palm_body_id
    jacobian = robot.root_physx_view.get_jacobians()[
        :, jacobian_row, :, arm_ids
    ].clone()

    link_rotation_w = matrix_from_quat(palm_quat_w)
    com_pos_link = robot.data.com_pos_b[:, palm_body_id]
    palm_offset_link = base._palm_body_offset.expand_as(com_pos_link)
    palm_from_com_w = torch.bmm(
        link_rotation_w, (palm_offset_link - com_pos_link).unsqueeze(-1)
    ).squeeze(-1)
    angular_columns_w = jacobian[:, 3:, :].transpose(1, 2)
    jacobian[:, :3, :] += torch.cross(
        angular_columns_w,
        palm_from_com_w.unsqueeze(1).expand_as(angular_columns_w),
        dim=-1,
    ).transpose(1, 2)

    root_rotation_inv = matrix_from_quat(robot.data.root_quat_w).transpose(1, 2)
    jacobian[:, :3, :] = torch.bmm(root_rotation_inv, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(root_rotation_inv, jacobian[:, 3:, :])
    return jacobian


def _joint_target_action(
    base,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    palm_pos_w, palm_quat_w = base._current_palm_point_pose_w()
    position_error_w, rotation_error_w = compute_pose_error(
        palm_pos_w,
        palm_quat_w,
        target_pos_w,
        target_quat_w,
        rot_error_type="axis_angle",
    )
    root_rotation_inv = matrix_from_quat(base.robot.data.root_quat_w).transpose(1, 2)
    pose_error_b = torch.cat(
        (
            torch.bmm(root_rotation_inv, position_error_w.unsqueeze(-1)).squeeze(-1),
            torch.bmm(root_rotation_inv, rotation_error_w.unsqueeze(-1)).squeeze(-1),
        ),
        dim=-1,
    )
    jacobian = _palm_jacobian(base, palm_quat_w)
    jacobian_t = jacobian.transpose(1, 2)
    damping = max(float(args_cli.damping), 1.0e-4)
    regularizer = (damping * damping) * torch.eye(
        6, dtype=jacobian.dtype, device=base.device
    ).unsqueeze(0)
    correction = torch.bmm(
        jacobian_t,
        torch.linalg.solve(
            torch.bmm(jacobian, jacobian_t) + regularizer,
            pose_error_b.unsqueeze(-1),
        ),
    ).squeeze(-1)
    correction = torch.clamp(
        torch.nan_to_num(correction),
        -float(args_cli.max_joint_delta),
        float(args_cli.max_joint_delta),
    )

    arm_ids = base._arm_joint_ids
    lower = base._joint_lower_limits[arm_ids].unsqueeze(0)
    upper = base._joint_upper_limits[arm_ids].unsqueeze(0)
    center = torch.clamp(base._default_joint_pos[:, arm_ids], lower, upper)
    current_target = base._joint_targets[:, arm_ids]
    desired_target = torch.clamp(
        base.robot.data.joint_pos[:, arm_ids] + correction, lower, upper
    )
    moving_average = max(float(base.cfg.arm_moving_average), 1.0e-6)
    action_mode = str(
        getattr(base.cfg, "joint_target_arm_action_mode", "absolute")
    ).lower()
    if action_mode in {"incremental", "delta", "relative"}:
        delta_scale = max(
            float(getattr(base.cfg, "joint_target_arm_delta_scale", 0.025)),
            1.0e-6,
        )
        action = (desired_target - current_target) / (moving_average * delta_scale)
        return torch.clamp(action, -1.0, 1.0), position_error_w, rotation_error_w
    raw_target = torch.clamp(
        (desired_target - (1.0 - moving_average) * current_target) / moving_average,
        lower,
        upper,
    )
    delta = raw_target - center
    positive_span = torch.clamp(upper - center, min=1.0e-6)
    negative_span = torch.clamp(center - lower, min=1.0e-6)
    action = torch.where(delta >= 0.0, delta / positive_span, delta / negative_span)
    return torch.clamp(action, -1.0, 1.0), position_error_w, rotation_error_w


def _cartesian_action(
    base,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    palm_pos_w, palm_quat_w = base._current_palm_point_pose_w()
    position_error_w, rotation_error_w = compute_pose_error(
        palm_pos_w,
        palm_quat_w,
        target_pos_w,
        target_quat_w,
        rot_error_type="axis_angle",
    )
    interface = str(base.cfg.policy_action_interface)
    if interface == "cartesian_impedance":
        palm_pos_b, palm_quat_b = base._current_palm_point_pose_b()
        target_pos_b, target_quat_b = subtract_frame_transforms(
            base.robot.data.root_pos_w,
            base.robot.data.root_quat_w,
            target_pos_w,
            target_quat_w,
        )
        position_error_b, rotation_error_b = compute_pose_error(
            palm_pos_b,
            palm_quat_b,
            target_pos_b,
            target_quat_b,
            rot_error_type="axis_angle",
        )
        target_mode = str(
            getattr(base.cfg, "cartesian_impedance_target_mode", "integrated_delta")
        )
        if target_mode == "integrated_delta":
            command_position_error, command_rotation_error = compute_pose_error(
                base._cartesian_impedance_desired_pose_b[:, :3],
                base._cartesian_impedance_desired_pose_b[:, 3:],
                target_pos_b,
                target_quat_b,
                rot_error_type="axis_angle",
            )
        elif target_mode == "measured_delta":
            command_position_error = position_error_b
            command_rotation_error = rotation_error_b
        else:
            raise ValueError(f"Unsupported Cartesian impedance target mode: {target_mode!r}")
    else:
        target_mode = str(
            getattr(base.cfg, "cartesian_wrist_target_mode", "measured_delta")
        )
        if target_mode == "integrated_delta":
            command_position_error, command_rotation_error = compute_pose_error(
                base._cartesian_wrist_policy_target_pos_w,
                base._cartesian_wrist_policy_target_quat_w,
                target_pos_w,
                target_quat_w,
                rot_error_type="axis_angle",
            )
        elif target_mode == "measured_delta":
            command_position_error = position_error_w
            command_rotation_error = rotation_error_w
        else:
            raise ValueError(f"Unsupported Cartesian wrist target mode: {target_mode!r}")
    translation_scale = max(float(base.cfg.cartesian_wrist_translation_scale), 1.0e-6)
    rotation_scale = max(float(base.cfg.cartesian_wrist_rotation_scale), 1.0e-6)
    action = torch.cat(
        (
            command_position_error / translation_scale,
            command_rotation_error / rotation_scale,
        ),
        dim=-1,
    )
    return torch.clamp(action, -1.0, 1.0), position_error_w, rotation_error_w


def _local_list(value: torch.Tensor, origins: torch.Tensor) -> list[float]:
    return [float(v) for v in (value[0] - origins[0]).detach().cpu()]


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=int(args_cli.num_envs),
    )
    env_cfg.seed = int(args_cli.seed)
    env_cfg.reset_arm_pos_noise = 0.0
    env_cfg.reset_object_pos_noise = (0.0, 0.0, 0.0)
    if hasattr(env_cfg, "canonical_reset_arm_pos_noise"):
        env_cfg.canonical_reset_arm_pos_noise = 0.0
    env_cfg.terminate_on_success = False
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), 30.0)
    env_cfg.initial_arm_target_lock_steps = 0
    env_cfg.initial_hand_target_lock_steps = 0
    robot_root_pos = tuple(float(value) for value in env_cfg.robot_cfg.init_state.pos)
    env_cfg.robot_cfg.init_state.pos = (
        robot_root_pos[0],
        robot_root_pos[1],
        robot_root_pos[2] + float(args_cli.robot_base_z_offset),
    )
    for field_name in (
        "dynamic_tabletop_start_speed_range",
        "dynamic_tabletop_initial_speed_range",
        "dynamic_tabletop_start_yaw_rate_range",
        "dynamic_tabletop_initial_yaw_rate_range",
    ):
        if hasattr(env_cfg, field_name):
            setattr(env_cfg, field_name, (0.0, 0.0))

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset(seed=int(args_cli.seed))
        base = env.unwrapped
        interface = str(base.cfg.policy_action_interface)
        arm_dim = int(base._policy_arm_action_dim())
        action_dim = int(base.cfg.action_space)
        actions = torch.zeros((base.num_envs, action_dim), device=base.device)
        actions[:, arm_dim:] = float(args_cli.hand_action)
        for _ in range(max(int(args_cli.warmup_steps), 0)):
            env.step(actions)

        start_pos_w, start_quat_w = base._current_palm_point_pose_w()
        start_pos_w = start_pos_w.clone()
        start_quat_w = start_quat_w.clone()
        target_offset = torch.tensor(
            args_cli.target_offset, dtype=start_pos_w.dtype, device=base.device
        ).view(1, 3)
        target_pos_w = start_pos_w + target_offset
        target_quat_w = start_quat_w
        max_action_norm = torch.zeros(base.num_envs, device=base.device)
        terminated_any = False
        truncated_any = False
        initial_position_error = torch.linalg.norm(target_pos_w - start_pos_w, dim=-1)
        final_position_error = initial_position_error
        final_rotation_error = torch.zeros_like(initial_position_error)

        for _ in range(int(args_cli.track_steps)):
            if interface == "joint_target":
                arm_action, position_error, rotation_error = _joint_target_action(
                    base, target_pos_w, target_quat_w
                )
            elif interface in {"cartesian_wrist_delta", "cartesian_impedance"}:
                arm_action, position_error, rotation_error = _cartesian_action(
                    base, target_pos_w, target_quat_w
                )
            else:
                raise ValueError(f"Unsupported interface for this probe: {interface!r}")
            actions.zero_()
            actions[:, :arm_dim] = arm_action
            actions[:, arm_dim:] = float(args_cli.hand_action)
            max_action_norm = torch.maximum(
                max_action_norm, torch.linalg.norm(arm_action, dim=-1)
            )
            _, _, terminated, truncated, _ = env.step(actions)
            terminated_any |= bool(torch.any(terminated).detach().cpu())
            truncated_any |= bool(torch.any(truncated).detach().cpu())
            final_position_error = torch.linalg.norm(position_error, dim=-1)
            final_rotation_error = torch.linalg.norm(rotation_error, dim=-1)

        final_pos_w, final_quat_w = base._current_palm_point_pose_w()
        final_arm_joint_pos = base.robot.data.joint_pos[:, base._arm_joint_ids]
        final_position_error_w, final_rotation_error_w = compute_pose_error(
            final_pos_w,
            final_quat_w,
            target_pos_w,
            target_quat_w,
            rot_error_type="axis_angle",
        )
        summary = {
            "task": args_cli.task,
            "policy_action_interface": interface,
            "num_envs": int(base.num_envs),
            "seed": int(args_cli.seed),
            "robot_base_z_offset": float(args_cli.robot_base_z_offset),
            "warmup_steps": int(args_cli.warmup_steps),
            "track_steps": int(args_cli.track_steps),
            "target_offset": [float(v) for v in args_cli.target_offset],
            "start_palm_pos_local_env0": _local_list(start_pos_w, base.scene.env_origins),
            "target_palm_pos_local_env0": _local_list(target_pos_w, base.scene.env_origins),
            "final_palm_pos_local_env0": _local_list(final_pos_w, base.scene.env_origins),
            "final_arm_joint_pos_env0": [
                float(value) for value in final_arm_joint_pos[0].detach().cpu()
            ],
            "initial_position_error_mean_m": float(initial_position_error.mean().detach().cpu()),
            "final_position_error_mean_m": float(
                torch.linalg.norm(final_position_error_w, dim=-1).mean().detach().cpu()
            ),
            "final_position_error_max_m": float(
                torch.linalg.norm(final_position_error_w, dim=-1).max().detach().cpu()
            ),
            "final_rotation_error_mean_rad": float(
                torch.linalg.norm(final_rotation_error_w, dim=-1).mean().detach().cpu()
            ),
            "max_arm_action_norm_mean": float(max_action_norm.mean().detach().cpu()),
            "terminated_any": terminated_any,
            "truncated_any": truncated_any,
        }
        output_path = Path(args_cli.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
