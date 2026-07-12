#!/usr/bin/env python3
"""Probe IsaacLab falling-baton scene setup and reset samples."""

from __future__ import annotations

import argparse
import faulthandler
import json
import sys
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from isaaclab.app import AppLauncher

faulthandler.enable()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToolReal-Revo2-Franka-FallingBaton-Teacher-Direct-v0")
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--summary-out", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


def log(message: str) -> None:
    print(f"[probe] {message}", flush=True)


def main() -> None:
    log(f"parse_env_cfg task={args_cli.task} device={args_cli.device} num_envs={args_cli.num_envs}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    log("gym.make start")
    env = gym.make(args_cli.task, cfg=env_cfg)
    log("gym.make done")
    try:
        unwrapped = env.unwrapped
        log("reset start")
        observations, _ = env.reset()
        log("reset done")
        reset_object_local = (unwrapped.object.data.root_pos_w - unwrapped.scene.env_origins).clone()
        reset_object_lin_vel = unwrapped.object.data.root_lin_vel_w.clone()
        zero_action = torch.zeros((args_cli.num_envs, unwrapped.cfg.action_space), device=unwrapped.device)
        for _ in range(args_cli.steps):
            env.step(zero_action)
        log(f"stepped {args_cli.steps} zero-action frames")

        stage = unwrapped.scene.stage
        table_paths = [
            f"/World/envs/env_{idx}/Table"
            for idx in range(min(args_cli.num_envs, 4))
            if stage.GetPrimAtPath(f"/World/envs/env_{idx}/Table").IsValid()
        ]
        obj_local = unwrapped.object.data.root_pos_w - unwrapped.scene.env_origins
        obj_vel = unwrapped.object.data.root_lin_vel_w
        palm_local = unwrapped._palm_pos_w - unwrapped.scene.env_origins
        tabletop_cmd_vel = getattr(unwrapped, "_tabletop_cmd_lin_vel_w", None)
        tabletop_cmd_yaw_rate = getattr(unwrapped, "_tabletop_cmd_yaw_rate", None)
        grasp_seen = getattr(unwrapped, "_grasp_seen", None)
        marker_paths = {
            region: [
                f"/World/envs/env_{idx}/ObjectAffordance{region.title()}"
                for idx in range(min(args_cli.num_envs, 4))
                if stage.GetPrimAtPath(
                    f"/World/envs/env_{idx}/ObjectAffordance{region.title()}"
                ).IsValid()
            ]
            for region in ("positive", "neutral", "negative")
        }
        policy_obs = observations.get("policy") if isinstance(observations, dict) else None

        print(f"task={args_cli.task}", flush=True)
        print(f"create_table_cfg={unwrapped.cfg.create_table}", flush=True)
        print(f"valid_table_paths={table_paths}", flush=True)
        print(f"valid_marker_paths={marker_paths}", flush=True)
        print(f"reset_object_local={reset_object_local.detach().cpu().tolist()}", flush=True)
        print(f"reset_object_lin_vel={reset_object_lin_vel.detach().cpu().tolist()}", flush=True)
        print(f"object_local={obj_local.detach().cpu().tolist()}", flush=True)
        print(f"object_lin_vel={obj_vel.detach().cpu().tolist()}", flush=True)
        if tabletop_cmd_vel is not None:
            print(f"tabletop_cmd_lin_vel={tabletop_cmd_vel.detach().cpu().tolist()}", flush=True)
        if tabletop_cmd_yaw_rate is not None:
            print(f"tabletop_cmd_yaw_rate={tabletop_cmd_yaw_rate.detach().cpu().tolist()}", flush=True)
        if grasp_seen is not None:
            print(f"grasp_seen={grasp_seen.detach().cpu().tolist()}", flush=True)
        print(f"palm_local={palm_local.detach().cpu().tolist()}", flush=True)
        print(f"catch_success_min_z={unwrapped.cfg.catch_success_min_z}", flush=True)
        print(f"falling_drop_z={unwrapped.cfg.falling_drop_z}", flush=True)
        print(f"max_episode_length={unwrapped.max_episode_length}", flush=True)
        if args_cli.summary_out is not None:
            reset_local_cpu = reset_object_local.detach().cpu()
            reset_vel_cpu = reset_object_lin_vel.detach().cpu()
            summary = {
                "task": args_cli.task,
                "benchmark_protocol": getattr(unwrapped.cfg, "benchmark_protocol", None),
                "seed": args_cli.seed,
                "num_envs": args_cli.num_envs,
                "action_space": int(unwrapped.cfg.action_space),
                "observation_shape": list(policy_obs.shape) if policy_obs is not None else None,
                "active_hand_dofs": len(unwrapped.cfg.hand_joint_names),
                "create_table": bool(unwrapped.cfg.create_table),
                "valid_table_count": len(table_paths),
                "valid_marker_counts": {key: len(value) for key, value in marker_paths.items()},
                "reset_object_local_min": reset_local_cpu.amin(dim=0).tolist(),
                "reset_object_local_max": reset_local_cpu.amax(dim=0).tolist(),
                "reset_object_lin_vel_min": reset_vel_cpu.amin(dim=0).tolist(),
                "reset_object_lin_vel_max": reset_vel_cpu.amax(dim=0).tolist(),
                "curriculum_alpha": float(unwrapped._dynamic_speed_curriculum_alpha()),
                "catch_success_min_z": float(unwrapped.cfg.catch_success_min_z),
                "falling_drop_z": float(unwrapped.cfg.falling_drop_z),
                "max_episode_length": int(unwrapped.max_episode_length),
            }
            summary_path = args_cli.summary_out.expanduser().resolve()
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"summary_out={summary_path}", flush=True)
    finally:
        log("env.close")
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
