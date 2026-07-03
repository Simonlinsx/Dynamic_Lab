#!/usr/bin/env python3
"""Search arm action directions that lift the Revo2 palm/object in IsaacLab."""

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
parser.add_argument("--task", default="SimToolReal-Revo2-Franka-StaticBall-Grasp-Direct-v0")
parser.add_argument("--num-candidates", type=int, default=128)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--pre-steps", type=int, default=100)
parser.add_argument("--close-steps", type=int, default=50)
parser.add_argument("--lift-steps", type=int, default=140)
parser.add_argument("--hold-steps", type=int, default=40)
parser.add_argument("--hand-action", type=float, default=0.9)
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


def _trace(message: str) -> None:
    print(f"[LIFT_SEARCH] {message}", flush=True)


def _build_candidates(num_candidates: int, seed: int, device: str) -> torch.Tensor:
    rng = random.Random(seed)
    candidates: list[list[float]] = []

    for joint_id in range(7):
        for sign in (-1.0, 1.0):
            action = [0.0] * 7
            action[joint_id] = sign
            candidates.append(action)

    sign_patterns = (
        (-1, -1, 1, 1, 0, -1, 1),
        (1, -1, 1, 1, 0, -1, 1),
        (-1, 1, 1, 1, 0, -1, 1),
        (-1, -1, -1, 1, 0, -1, 1),
        (-1, -1, 1, -1, 0, -1, 1),
        (-1, -1, 1, 1, 0, 1, 1),
        (-1, -1, 1, 1, 0, -1, -1),
    )
    candidates.extend([list(map(float, pattern)) for pattern in sign_patterns])

    while len(candidates) < num_candidates:
        action = [rng.uniform(-1.0, 1.0) for _ in range(7)]
        scale = max(max(abs(v) for v in action), 1.0e-6)
        candidates.append([v / scale for v in action])

    return torch.tensor(candidates[:num_candidates], dtype=torch.float32, device=device)


def _step(env, actions, metric_max):
    _, _, _, _, extras = env.step(actions)
    base = env.unwrapped
    base._compute_intermediate_values()
    metric_max["object_height_delta"] = torch.maximum(metric_max["object_height_delta"], base._object_height_delta)
    metric_max["palm_lift_delta"] = torch.maximum(
        metric_max["palm_lift_delta"],
        base._palm_pos_w[:, 2] - base._palm_start_z,
    )
    for key in ("success", "true_grasp", "lifted", "stable_hold"):
        tensor = extras.get(f"{key}_env")
        if tensor is not None:
            metric_max[key] = torch.maximum(metric_max[key], tensor.float())
    return extras


def main() -> None:
    carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_candidates)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        base = env.unwrapped
        device = base.device
        num_envs = base.num_envs
        candidates = _build_candidates(num_envs, args_cli.seed, device)
        actions = torch.zeros((num_envs, base.cfg.action_space), device=device)
        hand = torch.full((num_envs, 6), float(args_cli.hand_action), device=device)
        metric_max = {
            "object_height_delta": torch.zeros(num_envs, device=device),
            "palm_lift_delta": torch.zeros(num_envs, device=device),
            "success": torch.zeros(num_envs, device=device),
            "true_grasp": torch.zeros(num_envs, device=device),
            "lifted": torch.zeros(num_envs, device=device),
            "stable_hold": torch.zeros(num_envs, device=device),
        }

        phases = (
            ("pre", args_cli.pre_steps, torch.zeros_like(actions)),
            ("close", args_cli.close_steps, torch.cat((torch.zeros((num_envs, 7), device=device), hand), dim=-1)),
            ("lift", args_cli.lift_steps, torch.cat((candidates, hand), dim=-1)),
            ("hold", args_cli.hold_steps, torch.cat((torch.zeros((num_envs, 7), device=device), hand), dim=-1)),
        )
        last_extras = {}
        for phase_name, steps, phase_actions in phases:
            _trace(f"phase={phase_name} steps={steps}")
            for _ in range(steps):
                last_extras = _step(env, phase_actions, metric_max)

        base._compute_intermediate_values()
        results = []
        for env_id in range(num_envs):
            results.append(
                {
                    "env_id": int(env_id),
                    "arm_action": [float(v) for v in candidates[env_id].detach().cpu()],
                    "success": bool(metric_max["success"][env_id].item() > 0.5),
                    "lifted": bool(metric_max["lifted"][env_id].item() > 0.5),
                    "true_grasp": bool(metric_max["true_grasp"][env_id].item() > 0.5),
                    "stable_hold": bool(metric_max["stable_hold"][env_id].item() > 0.5),
                    "max_object_height_delta": float(metric_max["object_height_delta"][env_id].item()),
                    "max_palm_lift_delta": float(metric_max["palm_lift_delta"][env_id].item()),
                    "final_object_height_delta": float(base._object_height_delta[env_id].item()),
                    "final_palm_lift_delta": float((base._palm_pos_w[env_id, 2] - base._palm_start_z[env_id]).item()),
                    "final_true_grasp": bool(base._true_grasp[env_id].item()),
                    "final_object_pos_local": [
                        float(v) for v in (base._object_pos_w[env_id] - base.scene.env_origins[env_id]).detach().cpu()
                    ],
                    "final_palm_pos_local": [
                        float(v) for v in (base._palm_pos_w[env_id] - base.scene.env_origins[env_id]).detach().cpu()
                    ],
                }
            )

        results.sort(
            key=lambda r: (
                r["success"],
                r["stable_hold"],
                r["lifted"],
                r["max_object_height_delta"],
                r["max_palm_lift_delta"],
            ),
            reverse=True,
        )
        summary = {
            "task": args_cli.task,
            "num_candidates": num_envs,
            "seed": args_cli.seed,
            "success_count": sum(int(r["success"]) for r in results),
            "lifted_count": sum(int(r["lifted"]) for r in results),
            "top_results": results[:20],
            "results": results,
            "last_log": {
                key: float(value.item()) for key, value in last_extras.get("log", {}).items() if hasattr(value, "item")
            },
        }
        output_path = Path(args_cli.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _trace(json.dumps({k: summary[k] for k in ("success_count", "lifted_count", "top_results")}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
