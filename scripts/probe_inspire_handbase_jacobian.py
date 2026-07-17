#!/usr/bin/env python3
"""Compare PhysX Jacobian rows with finite-difference RH56BFX link kinematics."""

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
parser.add_argument(
    "--task",
    default="SimToolReal-Inspire-Franka-RH56BFXFaithfulSphere60RollingCurriculum-Teacher-Direct-v0",
)
parser.add_argument("--epsilon", type=float, default=1.0e-3)
parser.add_argument("--output-json", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


def _positions(robot, *, com: bool) -> torch.Tensor:
    if com:
        robot.data._body_com_state_w.timestamp = -1.0
        return robot.data.body_com_state_w[:, :, :3].clone()
    robot.data._body_link_state_w.timestamp = -1.0
    return robot.data.body_link_state_w[:, :, :3].clone()


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.reset_arm_pos_noise = 0.0
    env_cfg.reset_object_pos_noise = (0.0, 0.0, 0.0)
    env_cfg.initial_arm_target_lock_steps = 0
    env_cfg.initial_hand_target_lock_steps = 0
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset(seed=17)
        base = env.unwrapped
        robot = base.robot
        arm_ids = list(base._arm_joint_ids)
        ee_body_id = int(robot.find_bodies("hand_base_link")[0][0])
        q0 = robot.data.joint_pos[:, arm_ids].clone()
        analytic = robot.root_physx_view.get_jacobians()[0, :, :3, arm_ids].clone()
        epsilon = float(args_cli.epsilon)

        numeric_link = torch.zeros(
            (robot.num_bodies, 3, len(arm_ids)), dtype=q0.dtype, device=q0.device
        )
        numeric_com = torch.zeros_like(numeric_link)
        zero_vel = torch.zeros_like(q0)
        for column in range(len(arm_ids)):
            q_plus = q0.clone()
            q_plus[:, column] += epsilon
            robot.write_joint_state_to_sim(q_plus, zero_vel, joint_ids=arm_ids)
            base.sim.forward()
            plus_link = _positions(robot, com=False)[0]
            plus_com = _positions(robot, com=True)[0]

            q_minus = q0.clone()
            q_minus[:, column] -= epsilon
            robot.write_joint_state_to_sim(q_minus, zero_vel, joint_ids=arm_ids)
            base.sim.forward()
            minus_link = _positions(robot, com=False)[0]
            minus_com = _positions(robot, com=True)[0]

            numeric_link[:, :, column] = (plus_link - minus_link) / (2.0 * epsilon)
            numeric_com[:, :, column] = (plus_com - minus_com) / (2.0 * epsilon)

        robot.write_joint_state_to_sim(q0, zero_vel, joint_ids=arm_ids)
        base.sim.forward()

        target_link = numeric_link[ee_body_id]
        target_com = numeric_com[ee_body_id]
        rows = []
        for row_id in range(analytic.shape[0]):
            expected_body_id = row_id + 1 if robot.is_fixed_base else row_id
            row = analytic[row_id]
            rows.append(
                {
                    "jacobian_row": row_id,
                    "expected_body_id": expected_body_id,
                    "expected_body_name": robot.body_names[expected_body_id],
                    "rms_error_to_handbase_link": float(torch.sqrt(torch.mean((row - target_link) ** 2))),
                    "rms_error_to_handbase_com": float(torch.sqrt(torch.mean((row - target_com) ** 2))),
                    "rms_error_to_expected_link": float(
                        torch.sqrt(torch.mean((row - numeric_link[expected_body_id]) ** 2))
                    ),
                    "rms_error_to_expected_com": float(
                        torch.sqrt(torch.mean((row - numeric_com[expected_body_id]) ** 2))
                    ),
                }
            )

        best_link = min(rows, key=lambda item: item["rms_error_to_handbase_link"])
        best_com = min(rows, key=lambda item: item["rms_error_to_handbase_com"])
        expected_row = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
        payload = {
            "task": args_cli.task,
            "epsilon_rad": epsilon,
            "fixed_base": bool(robot.is_fixed_base),
            "body_names": list(robot.body_names),
            "arm_joint_ids": arm_ids,
            "arm_joint_names": [robot.joint_names[index] for index in arm_ids],
            "hand_base_body_id": ee_body_id,
            "expected_hand_base_jacobian_row": expected_row,
            "best_row_for_handbase_link": best_link,
            "best_row_for_handbase_com": best_com,
            "expected_row_diagnostics": rows[expected_row],
            "rows": rows,
            "handbase_numeric_link_jacobian": target_link.detach().cpu().tolist(),
            "handbase_numeric_com_jacobian": target_com.detach().cpu().tolist(),
            "expected_row_analytic_jacobian": analytic[expected_row].detach().cpu().tolist(),
        }
        output_path = args_cli.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({key: payload[key] for key in (
            "hand_base_body_id",
            "expected_hand_base_jacobian_row",
            "best_row_for_handbase_link",
            "best_row_for_handbase_com",
            "expected_row_diagnostics",
        )}, indent=2))
        print(f"wrote {output_path}")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
