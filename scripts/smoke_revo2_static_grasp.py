#!/usr/bin/env python3
"""Headless random-action smoke test for the Revo2 static grasp tasks."""

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
    default="SimToolReal-Revo2-Franka-StaticBall-Grasp-Direct-v0",
    help="Gym task id.",
)
parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel envs.")
parser.add_argument("--steps", type=int, default=64, help="Number of random action steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


def _trace(message: str) -> None:
    print(f"[SMOKE] {message}", flush=True)


def _load_cfg(entry_point: str):
    module_name, class_name = entry_point.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


def main() -> None:
    _trace(f"loading task spec: {args_cli.task}")
    spec = gym.spec(args_cli.task)
    cfg = _load_cfg(spec.kwargs["env_cfg_entry_point"])
    cfg.sim.device = args_cli.device
    cfg.scene.num_envs = args_cli.num_envs
    _trace(f"making env: num_envs={cfg.scene.num_envs}, device={cfg.sim.device}")
    env = gym.make(args_cli.task, cfg=cfg)
    _trace(f"env made: device={env.unwrapped.device}, action_space={env.unwrapped.cfg.action_space}")
    _trace("reset")
    obs, _ = env.reset()
    _trace(f"reset obs shape: {tuple(obs['policy'].shape)}")
    for step in range(args_cli.steps):
        actions = 2.0 * torch.rand((env.unwrapped.num_envs, env.unwrapped.cfg.action_space), device=env.unwrapped.device) - 1.0
        obs, rew, terminated, truncated, extras = env.step(actions)
        if step == 0 or step == args_cli.steps - 1:
            log = extras.get("log", {})
            log_msg = ", ".join(
                f"{key}={float(value):.4f}" for key, value in log.items() if hasattr(value, "item")
            )
            _trace(
                f"step={step:04d} reward_mean={float(rew.mean()):.4f} "
                f"done={int((terminated | truncated).sum())} {log_msg}"
            )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
