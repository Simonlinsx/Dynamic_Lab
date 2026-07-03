#!/usr/bin/env python3
"""Vectorized grid search for scripted Revo2 ball close/lift geometry."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToolReal-Revo2-Franka-StaticBall-Grasp-Direct-v0", help="Gym task id.")
parser.add_argument("--x-values", type=float, nargs="+", default=[0.54, 0.57, 0.60, 0.63])
parser.add_argument("--y-values", type=float, nargs="+", default=[-0.24, -0.21, -0.18, -0.15, -0.12])
parser.add_argument("--z", type=float, default=0.326)
parser.add_argument("--hand-actions", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
parser.add_argument("--lift-scales", type=float, nargs="+", default=[1.0])
parser.add_argument("--pre-steps", type=int, default=100)
parser.add_argument("--close-steps", type=int, default=50)
parser.add_argument("--lift-steps", type=int, default=120)
parser.add_argument("--hold-steps", type=int, default=40)
parser.add_argument("--lift-deltas", type=float, nargs=7, default=None)
parser.add_argument("--max-candidates", type=int, default=None)
parser.add_argument("--output-json", default=None)
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
    print(f"[SEARCH] {message}", flush=True)


def _candidate_table() -> list[dict]:
    candidates = [
        {
            "x": float(x),
            "y": float(y),
            "z": float(args_cli.z),
            "hand_action": float(hand_action),
            "lift_scale": float(lift_scale),
        }
        for x, y, hand_action, lift_scale in itertools.product(
            args_cli.x_values, args_cli.y_values, args_cli.hand_actions, args_cli.lift_scales
        )
    ]
    if args_cli.max_candidates is not None:
        candidates = candidates[: args_cli.max_candidates]
    return candidates


def _tensor_bool(extras: dict, key: str, num_envs: int, device: torch.device) -> torch.Tensor:
    value = extras.get(key)
    if value is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return value.bool()


def _set_object_positions(env, positions: torch.Tensor) -> None:
    base = env.unwrapped
    num_envs = base.num_envs
    env_ids = torch.arange(num_envs, device=base.device, dtype=torch.long)
    base._object_start_pos = positions.clone()

    object_state = base.object.data.default_root_state.clone()
    object_state[:, 0:3] = positions + base.scene.env_origins
    object_state[:, 3:7] = torch.tensor(base.cfg.object_start_rot, device=base.device).unsqueeze(0)
    object_state[:, 7:] = 0.0
    base.object.write_root_pose_to_sim(object_state[:, :7], env_ids=env_ids)
    base.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids=env_ids)
    base.scene.write_data_to_sim()
    base.sim.forward()
    base._compute_intermediate_values()


def _run_phase(env, actions: torch.Tensor, steps: int, metric_max: dict[str, torch.Tensor], height_max: torch.Tensor):
    last_extras = {}
    base = env.unwrapped
    for _ in range(steps):
        _, _, _, _, last_extras = env.step(actions)
        metric_max["success"] |= _tensor_bool(last_extras, "success_env", base.num_envs, base.device)
        metric_max["true_grasp"] |= _tensor_bool(last_extras, "true_grasp_env", base.num_envs, base.device)
        metric_max["lifted"] |= _tensor_bool(last_extras, "lifted_env", base.num_envs, base.device)
        metric_max["stable_hold"] |= _tensor_bool(last_extras, "stable_hold_env", base.num_envs, base.device)
        base._compute_intermediate_values()
        height_max[:] = torch.maximum(height_max, base._object_height_delta)
    return last_extras


def main() -> None:
    carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    candidates = _candidate_table()
    if not candidates:
        raise ValueError("No search candidates were generated.")

    num_envs = len(candidates)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env_cfg.reset_object_pos_noise = (0.0, 0.0, 0.0)
    env_cfg.object_start_pos = (float(args_cli.x_values[0]), float(args_cli.y_values[0]), float(args_cli.z))
    env_cfg.object_cfg.init_state.pos = env_cfg.object_start_pos
    env_cfg.table_top_z = float(args_cli.z) - float(env_cfg.object_radius)
    mean_x = sum(c["x"] for c in candidates) / num_envs
    mean_y = sum(c["y"] for c in candidates) / num_envs
    env_cfg.table_cfg.init_state.pos = (mean_x, mean_y, env_cfg.table_top_z - 0.0225)
    env_cfg.workspace_xy_limit = max(
        float(env_cfg.workspace_xy_limit),
        max(max(abs(c["x"]), abs(c["y"])) for c in candidates) + 0.35,
    )

    _trace(f"candidates={num_envs} table_top_z={env_cfg.table_top_z:.4f}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        base = env.unwrapped
        device = base.device
        positions = torch.tensor([[c["x"], c["y"], c["z"]] for c in candidates], device=device)
        hand_actions = torch.tensor([c["hand_action"] for c in candidates], device=device)
        lift_scales = torch.tensor([c["lift_scale"] for c in candidates], device=device)
        _set_object_positions(env, positions)

        actions_zero = torch.zeros((num_envs, base.cfg.action_space), device=device)
        actions_close = torch.zeros_like(actions_zero)
        actions_close[:, 7:] = hand_actions[:, None]

        dt = base.step_dt
        effective_arm_gain = max(base.cfg.arm_action_scale * base.cfg.arm_moving_average, 1.0e-6)
        default_lift_delta = [-0.005224, -0.066232, 0.130747, 0.127955, 0.004971, -0.050877, 0.206977]
        lift_delta = torch.tensor(args_cli.lift_deltas or default_lift_delta, device=device)
        lift_action = torch.clamp(
            lift_delta / max(effective_arm_gain * dt * max(args_cli.lift_steps, 1), 1.0e-6),
            -1.0,
            1.0,
        )
        actions_lift = actions_close.clone()
        actions_lift[:, :7] = torch.clamp(lift_action.unsqueeze(0) * lift_scales.unsqueeze(1), -1.0, 1.0)

        metric_max = {
            "success": torch.zeros(num_envs, dtype=torch.bool, device=device),
            "true_grasp": torch.zeros(num_envs, dtype=torch.bool, device=device),
            "lifted": torch.zeros(num_envs, dtype=torch.bool, device=device),
            "stable_hold": torch.zeros(num_envs, dtype=torch.bool, device=device),
        }
        height_max = torch.zeros(num_envs, device=device)

        phase_logs = {}
        for phase_name, steps, phase_actions in (
            ("pre", args_cli.pre_steps, actions_zero),
            ("close", args_cli.close_steps, actions_close),
            ("lift", args_cli.lift_steps, actions_lift),
            ("hold", args_cli.hold_steps, actions_close),
        ):
            _trace(f"phase={phase_name} steps={steps}")
            extras = _run_phase(env, phase_actions, steps, metric_max, height_max)
            phase_logs[phase_name] = {
                key: float(value.item()) for key, value in extras.get("log", {}).items() if hasattr(value, "item")
            }

        base._compute_intermediate_values()
        object_local = base._object_pos_w - base.scene.env_origins
        palm_distance = torch.norm(base._object_pos_w - base._palm_pos_w, dim=-1)
        min_surface_dist = base._surface_dist.min(dim=-1).values

        results = []
        for i, candidate in enumerate(candidates):
            result = dict(candidate)
            result.update(
                {
                    "success": bool(metric_max["success"][i].detach().cpu()),
                    "true_grasp": bool(metric_max["true_grasp"][i].detach().cpu()),
                    "lifted": bool(metric_max["lifted"][i].detach().cpu()),
                    "stable_hold": bool(metric_max["stable_hold"][i].detach().cpu()),
                    "max_height_delta": float(height_max[i].detach().cpu()),
                    "final_height_delta": float(base._object_height_delta[i].detach().cpu()),
                    "final_palm_distance": float(palm_distance[i].detach().cpu()),
                    "final_min_surface_dist": float(min_surface_dist[i].detach().cpu()),
                    "final_object_pos_local": [float(v) for v in object_local[i].detach().cpu()],
                }
            )
            results.append(result)

        results_sorted = sorted(
            results,
            key=lambda r: (
                r["success"],
                r["stable_hold"],
                r["lifted"],
                r["max_height_delta"],
                r["true_grasp"],
                -r["final_palm_distance"],
            ),
            reverse=True,
        )
        summary = {
            "task": args_cli.task,
            "num_envs": num_envs,
            "phase_logs": phase_logs,
            "success_count": sum(r["success"] for r in results),
            "lifted_count": sum(r["lifted"] for r in results),
            "true_grasp_count": sum(r["true_grasp"] for r in results),
            "top_results": results_sorted[:20],
            "results": results,
        }
        _trace(json.dumps({k: summary[k] for k in ["num_envs", "success_count", "lifted_count", "true_grasp_count"]}))
        _trace(json.dumps(summary["top_results"], indent=2, sort_keys=True))
        if args_cli.output_json:
            output_json = Path(args_cli.output_json).expanduser().resolve()
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
