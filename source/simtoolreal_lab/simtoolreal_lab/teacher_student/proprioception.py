"""Deployable robot-state observations for teacher-student policies."""

from __future__ import annotations

import torch

from .pointcloud import quat_rotate_inverse


def deployable_robot_proprioception(unwrapped_env) -> torch.Tensor:
    """Build robot-only proprioception shared by collection, evaluation, and RL.

    The object state is deliberately excluded. Object motion is provided only
    through the masked RGB-D point-cloud history.
    """

    unwrapped_env._compute_intermediate_values()
    robot = unwrapped_env.robot
    arm_ids = list(unwrapped_env._arm_joint_ids)
    hand_ids = list(
        getattr(unwrapped_env, "_active_hand_joint_ids", unwrapped_env._hand_joint_ids)
    )

    arm_pos = robot.data.joint_pos[:, arm_ids] - unwrapped_env._default_arm_pos
    arm_vel = 0.1 * robot.data.joint_vel[:, arm_ids]

    hand_pos = robot.data.joint_pos[:, hand_ids]
    hand_vel = 0.1 * robot.data.joint_vel[:, hand_ids]
    hand_lower = unwrapped_env._joint_lower_limits[hand_ids]
    hand_upper = unwrapped_env._joint_upper_limits[hand_ids]
    hand_pos_scaled = 2.0 * (hand_pos - hand_lower) / torch.clamp(
        hand_upper - hand_lower, min=1.0e-6
    ) - 1.0

    palm_pos_local = unwrapped_env._palm_pos_w - unwrapped_env.scene.env_origins
    palm_quat = unwrapped_env._palm_quat_w
    palm_lin_vel = 0.1 * unwrapped_env._palm_lin_vel_w
    palm_ang_vel = 0.1 * robot.data.body_ang_vel_w[:, unwrapped_env._palm_body_id]
    fingertip_rel_palm = quat_rotate_inverse(
        palm_quat.unsqueeze(1),
        unwrapped_env._fingertip_pos_w - unwrapped_env._palm_pos_w.unsqueeze(1),
    ).reshape(unwrapped_env.num_envs, -1)

    previous_action = getattr(unwrapped_env, "policy_actions", None)
    if previous_action is None:
        previous_action = unwrapped_env.actions

    return torch.cat(
        (
            arm_pos,
            arm_vel,
            hand_pos_scaled,
            hand_vel,
            palm_pos_local,
            palm_quat,
            palm_lin_vel,
            palm_ang_vel,
            fingertip_rel_palm,
            previous_action,
        ),
        dim=-1,
    )
