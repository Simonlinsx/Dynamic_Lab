#!/usr/bin/env python3
"""Probe how single Franka arm actions move the palm relative to the tabletop object."""

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
parser.add_argument("--task", required=True)
parser.add_argument("--steps", type=int, default=90)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--action-value", type=float, default=1.0)
parser.add_argument(
    "--action-vectors-json",
    default=None,
    help="Optional JSON file with candidate 7D arm action vectors. Accepts a label->vector dict or a list of {'label', 'vector'} entries.",
)
parser.add_argument("--summary-out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


def _snapshot(unwrapped) -> dict:
    palm = unwrapped._palm_pos_w - unwrapped.scene.env_origins
    obj = unwrapped._object_pos_w - unwrapped.scene.env_origins
    palm_to_obj = obj - palm
    surface = unwrapped._surface_dist
    return {
        "palm_xyz": palm.detach().cpu().tolist(),
        "object_xyz": obj.detach().cpu().tolist(),
        "palm_to_object_xyz": palm_to_obj.detach().cpu().tolist(),
        "palm_object_dist": torch.norm(palm_to_obj, dim=-1).detach().cpu().tolist(),
        "palm_object_xy_dist": torch.norm(palm_to_obj[:, :2], dim=-1).detach().cpu().tolist(),
        "min_surface_dist": surface.min(dim=-1).values.detach().cpu().tolist(),
    }


def _load_action_vectors(path: str | None) -> tuple[list[str], list[list[float]]] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    labels: list[str] = []
    vectors: list[list[float]] = []
    if isinstance(payload, dict):
        items = payload.items()
        for label, vector in items:
            labels.append(str(label))
            vectors.append([float(value) for value in vector])
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                label = str(item.get("label", f"candidate_{index:02d}"))
                vector = item.get("vector")
            else:
                label = f"candidate_{index:02d}"
                vector = item
            labels.append(label)
            vectors.append([float(value) for value in vector])
    else:
        raise ValueError("--action-vectors-json must contain a dict or a list")
    for label, vector in zip(labels, vectors):
        if len(vector) != 7:
            raise ValueError(f"Candidate {label!r} must have 7 values, got {len(vector)}")
    return labels, vectors


def main() -> None:
    vector_candidates = _load_action_vectors(args_cli.action_vectors_json)
    env_count = len(vector_candidates[0]) if vector_candidates is not None else 14
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=env_count)
    env_cfg.scene.num_envs = env_count
    env_cfg.seed = int(args_cli.seed)
    if hasattr(env_cfg, "terminate_on_success"):
        env_cfg.terminate_on_success = False
    if hasattr(env_cfg, "reset_object_pos_noise"):
        env_cfg.reset_object_pos_noise = (0.0, 0.0, 0.0)
    if hasattr(env_cfg, "dynamic_tabletop_start_speed_range"):
        env_cfg.dynamic_tabletop_start_speed_range = (0.0, 0.0)
    if hasattr(env_cfg, "dynamic_tabletop_initial_speed_range"):
        env_cfg.dynamic_tabletop_initial_speed_range = (0.0, 0.0)
    if hasattr(env_cfg, "dynamic_tabletop_start_yaw_rate_range"):
        env_cfg.dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    if hasattr(env_cfg, "dynamic_tabletop_initial_yaw_rate_range"):
        env_cfg.dynamic_tabletop_initial_yaw_rate_range = (0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset(seed=int(args_cli.seed))
        unwrapped = env.unwrapped
        unwrapped._compute_intermediate_values()
        start = _snapshot(unwrapped)

        actions = torch.zeros((env_count, int(unwrapped.cfg.action_space)), device=unwrapped.device)
        action_value = float(args_cli.action_value)
        if vector_candidates is None:
            labels = []
            for joint_id in range(7):
                plus_env = 2 * joint_id
                minus_env = plus_env + 1
                actions[plus_env, joint_id] = action_value
                actions[minus_env, joint_id] = -action_value
                labels.extend([f"j{joint_id + 1}:+", f"j{joint_id + 1}:-"])
        else:
            labels, vectors = vector_candidates
            vector_tensor = torch.tensor(vectors, dtype=torch.float32, device=unwrapped.device)
            actions[:, :7] = torch.clamp(vector_tensor, -1.0, 1.0)

        for _ in range(int(args_cli.steps)):
            env.step(actions)
        unwrapped._compute_intermediate_values()
        final = _snapshot(unwrapped)

        rows = []
        for env_id, label in enumerate(labels):
            rows.append(
                {
                    "label": label,
                    "start_palm_object_dist": start["palm_object_dist"][env_id],
                    "final_palm_object_dist": final["palm_object_dist"][env_id],
                    "delta_palm_object_dist": final["palm_object_dist"][env_id]
                    - start["palm_object_dist"][env_id],
                    "start_xy_dist": start["palm_object_xy_dist"][env_id],
                    "final_xy_dist": final["palm_object_xy_dist"][env_id],
                    "delta_xy_dist": final["palm_object_xy_dist"][env_id] - start["palm_object_xy_dist"][env_id],
                    "start_min_surface_dist": start["min_surface_dist"][env_id],
                    "final_min_surface_dist": final["min_surface_dist"][env_id],
                    "delta_min_surface_dist": final["min_surface_dist"][env_id]
                    - start["min_surface_dist"][env_id],
                    "final_palm_xyz": final["palm_xyz"][env_id],
                    "final_palm_to_object_xyz": final["palm_to_object_xyz"][env_id],
                }
            )
        arm_ids = unwrapped._arm_joint_ids
        hand_ids = unwrapped._control_hand_joint_ids
        joint_summary = {
            "arm_joint_names": [unwrapped.robot.joint_names[joint_id] for joint_id in arm_ids],
            "arm_default_joint_pos_env0": [
                float(v) for v in unwrapped._default_joint_pos[0, arm_ids].detach().cpu()
            ],
            "arm_cfg_default_arm_pos": [float(v) for v in unwrapped._default_arm_pos[0].detach().cpu()],
            "arm_lower_limits": [float(v) for v in unwrapped._joint_lower_limits[arm_ids].detach().cpu()],
            "arm_upper_limits": [float(v) for v in unwrapped._joint_upper_limits[arm_ids].detach().cpu()],
            "arm_clamp_delta": [float(v) for v in unwrapped._arm_clamp_delta[0].detach().cpu()],
            "hand_joint_names": [unwrapped.robot.joint_names[joint_id] for joint_id in hand_ids],
            "hand_default_joint_pos_env0": [
                float(v) for v in unwrapped._default_joint_pos[0, hand_ids].detach().cpu()
            ],
            "hand_lower_limits": [float(v) for v in unwrapped._joint_lower_limits[hand_ids].detach().cpu()],
            "hand_upper_limits": [float(v) for v in unwrapped._joint_upper_limits[hand_ids].detach().cpu()],
        }

        summary = {
            "task": args_cli.task,
            "steps": int(args_cli.steps),
            "action_value": action_value,
            "joint_summary": joint_summary,
            "rows": rows,
        }
        out_path = Path(args_cli.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
