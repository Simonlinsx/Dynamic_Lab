#!/usr/bin/env python3
"""Scripted close/lift probe for the Revo2 static grasp IsaacLab tasks."""

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
parser.add_argument("--task", default="SimToolReal-Revo2-Franka-StaticBall-Grasp-Direct-v0", help="Gym task id.")
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=8, help="Number of envs.")
parser.add_argument("--pre-steps", type=int, default=100, help="Initial locked/open steps.")
parser.add_argument("--close-steps", type=int, default=90, help="Hand close steps.")
parser.add_argument("--lift-steps", type=int, default=50, help="Arm lift steps.")
parser.add_argument("--hold-steps", type=int, default=80, help="Hold steps after lift.")
parser.add_argument("--lift-deltas", type=float, nargs=7, default=None, help="Optional 7-DoF arm lift target delta.")
parser.add_argument("--lift-joint2-delta", type=float, default=None, help="Legacy shortcut for joint2 delta.")
parser.add_argument("--lift-joint4-delta", type=float, default=None, help="Legacy shortcut for joint4 delta.")
parser.add_argument("--lift-joint6-delta", type=float, default=None, help="Legacy shortcut for joint6 delta.")
parser.add_argument("--hand-action", type=float, default=1.0, help="Hand action during close/lift/hold.")
parser.add_argument("--object-pos", type=float, nargs=3, default=None, help="Override object start XYZ.")
parser.add_argument("--table-top-z", type=float, default=None, help="Override table top z.")
parser.add_argument("--table-pos-xy", type=float, nargs=2, default=None, help="Override table center XY.")
parser.add_argument("--output-json", default=None, help="Optional summary JSON path.")
parser.add_argument("--video-path", default=None, help="Optional mp4 path for env0.")
parser.add_argument("--video-stride", type=int, default=2, help="Record every N control steps.")
parser.add_argument("--video-fps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video_path and hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


def _trace(message: str) -> None:
    print(f"[SCRIPTED] {message}", flush=True)


def _record_frame(env, frames: list) -> None:
    frame = env.unwrapped.render(recompute=True)
    if frame is not None and frame.size > 0:
        frames.append(frame)


def _step(env, actions: torch.Tensor, success_any: torch.Tensor, metric_max: dict[str, torch.Tensor]):
    _, _, terminated, truncated, extras = env.step(actions)
    for key in ("success", "true_grasp", "lifted", "stable_hold"):
        tensor = extras.get(f"{key}_env")
        if tensor is None:
            continue
        metric_max[key] = torch.maximum(metric_max[key], tensor.float())
    success = extras.get("success_env")
    if success is not None:
        success_any |= success.bool()
    return terminated | truncated, extras


def _mean_log(extras: dict) -> dict[str, float]:
    return {key: float(value.item()) for key, value in extras.get("log", {}).items() if hasattr(value, "item")}


def _geometry_debug(env) -> dict:
    base = env.unwrapped
    base._compute_intermediate_values()
    origin = base.scene.env_origins[0]
    object_pos = base._object_pos_w[0]
    palm_pos = base._palm_pos_w[0]
    return {
        "palm_body": base.cfg.palm_body_name,
        "fingertip_bodies": list(base.cfg.fingertip_body_names),
        "env_origin_env0": [float(v) for v in origin.detach().cpu()],
        "object_pos_env0": [float(v) for v in object_pos.detach().cpu()],
        "object_pos_local_env0": [float(v) for v in (object_pos - origin).detach().cpu()],
        "palm_pos_env0": [float(v) for v in palm_pos.detach().cpu()],
        "palm_pos_local_env0": [float(v) for v in (palm_pos - origin).detach().cpu()],
        "object_minus_palm_env0": [float(v) for v in (object_pos - palm_pos).detach().cpu()],
        "object_height_delta_env0": float(base._object_height_delta[0].detach().cpu()),
        "true_grasp_env0": bool(base._true_grasp[0].detach().cpu()),
        "lifted_env0": bool(base._lifted[0].detach().cpu()),
        "stable_hold_env0": bool(base._stable_hold[0].detach().cpu()),
        "fingertip_pos_env0": [
            [float(v) for v in tip.detach().cpu()] for tip in base._fingertip_pos_w[0]
        ],
        "fingertip_pos_local_env0": [
            [float(v) for v in (tip - origin).detach().cpu()] for tip in base._fingertip_pos_w[0]
        ],
        "surface_dist_env0": [float(v) for v in base._surface_dist[0].detach().cpu()],
        "joint_pos_env0": {
            name: float(base.robot.data.joint_pos[0, idx].detach().cpu())
            for idx, name in enumerate(base.robot.joint_names)
            if name in set(base.cfg.arm_joint_names) or name in set(base.cfg.hand_joint_names)
        },
    }


def main() -> None:
    carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    table_top_z = args_cli.table_top_z
    if args_cli.object_pos is not None:
        object_pos = tuple(float(v) for v in args_cli.object_pos)
        env_cfg.object_start_pos = object_pos
        env_cfg.object_cfg.init_state.pos = object_pos
        env_cfg.workspace_xy_limit = max(
            float(env_cfg.workspace_xy_limit),
            max(abs(object_pos[0]), abs(object_pos[1])) + 0.30,
        )
        if table_top_z is None:
            if env_cfg.object_shape == "sphere":
                table_top_z = object_pos[2] - float(env_cfg.object_radius)
            else:
                table_top_z = object_pos[2] - 0.5 * float(env_cfg.object_size[2])
    if table_top_z is not None:
        table_xy = args_cli.table_pos_xy
        if table_xy is None and args_cli.object_pos is not None:
            table_xy = args_cli.object_pos[:2]
        if table_xy is None:
            table_xy = env_cfg.table_cfg.init_state.pos[:2]
        env_cfg.table_top_z = float(table_top_z)
        env_cfg.table_cfg.init_state.pos = (float(table_xy[0]), float(table_xy[1]), float(table_top_z) - 0.0225)
    render_mode = "rgb_array" if args_cli.video_path else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video_path:
        env.unwrapped.sim.set_camera_view(eye=(0.66, -0.84, 0.82), target=(0.0, 0.0, 0.56))

    try:
        obs, _ = env.reset()
        del obs
        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs
        actions = torch.zeros((num_envs, env.unwrapped.cfg.action_space), device=device)
        success_any = torch.zeros(num_envs, dtype=torch.bool, device=device)
        metric_max = {
            "success": torch.zeros(num_envs, device=device),
            "true_grasp": torch.zeros(num_envs, device=device),
            "lifted": torch.zeros(num_envs, device=device),
            "stable_hold": torch.zeros(num_envs, device=device),
        }
        frames = []

        dt = env.unwrapped.step_dt
        scale = env.unwrapped.cfg.arm_action_scale
        effective_arm_gain = max(scale * env.unwrapped.cfg.arm_moving_average, 1.0e-6)
        default_lift_delta = [-0.005224, -0.066232, 0.130747, 0.127955, 0.004971, -0.050877, 0.206977]
        lift_delta_values = list(args_cli.lift_deltas) if args_cli.lift_deltas is not None else default_lift_delta
        if args_cli.lift_joint2_delta is not None:
            lift_delta_values[1] = args_cli.lift_joint2_delta
        if args_cli.lift_joint4_delta is not None:
            lift_delta_values[3] = args_cli.lift_joint4_delta
        if args_cli.lift_joint6_delta is not None:
            lift_delta_values[5] = args_cli.lift_joint6_delta
        lift_delta = torch.tensor(lift_delta_values, device=device)
        lift_action = torch.clamp(
            lift_delta / max(effective_arm_gain * dt * max(args_cli.lift_steps, 1), 1.0e-6),
            -1.0,
            1.0,
        )

        phases = (
            ("pre", args_cli.pre_steps, torch.zeros_like(actions)),
            ("close", args_cli.close_steps, torch.cat((torch.zeros(7, device=device), torch.full((6,), args_cli.hand_action, device=device))).repeat(num_envs, 1)),
            ("lift", args_cli.lift_steps, torch.cat((lift_action, torch.full((6,), args_cli.hand_action, device=device))).repeat(num_envs, 1)),
            ("hold", args_cli.hold_steps, torch.cat((torch.zeros(7, device=device), torch.full((6,), args_cli.hand_action, device=device))).repeat(num_envs, 1)),
        )

        step_count = 0
        last_extras = {}
        phase_logs = {}
        phase_geometry = {}
        for phase_name, phase_steps, phase_actions in phases:
            _trace(f"phase={phase_name} steps={phase_steps}")
            for _ in range(phase_steps):
                if args_cli.video_path and step_count % max(args_cli.video_stride, 1) == 0:
                    _record_frame(env, frames)
                _, last_extras = _step(env, phase_actions, success_any, metric_max)
                step_count += 1
            phase_logs[phase_name] = _mean_log(last_extras)
            phase_geometry[phase_name] = _geometry_debug(env)

        if args_cli.video_path:
            _record_frame(env, frames)
            video_path = Path(args_cli.video_path).expanduser().resolve()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_path, frames, fps=args_cli.video_fps, macro_block_size=16)
            _trace(f"video={video_path}")

        summary = {
            "task": args_cli.task,
            "num_envs": int(num_envs),
            "steps": int(step_count),
            "success_rate": float(success_any.float().mean().item()),
            "success_count": int(success_any.sum().item()),
            "max_true_grasp": float(metric_max["true_grasp"].mean().item()),
            "max_lifted": float(metric_max["lifted"].mean().item()),
            "max_stable_hold": float(metric_max["stable_hold"].mean().item()),
            "phase_logs": phase_logs,
            "phase_geometry": phase_geometry,
            "last_log": _mean_log(last_extras),
            "geometry_debug": _geometry_debug(env),
            "override_object_pos": list(args_cli.object_pos) if args_cli.object_pos is not None else None,
            "override_table_top_z": table_top_z,
            "video_path": str(Path(args_cli.video_path).expanduser().resolve()) if args_cli.video_path else None,
        }
        _trace(json.dumps(summary, indent=2, sort_keys=True))
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
