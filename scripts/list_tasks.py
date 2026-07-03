#!/usr/bin/env python3
"""List SimToolReal IsaacLab Gym task registrations."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

import simtoolreal_lab  # noqa: F401,E402


def main() -> None:
    task_ids = sorted(spec.id for spec in gym.registry.values() if spec.id.startswith("SimToolReal-"))
    for task_id in task_ids:
        spec = gym.spec(task_id)
        print(task_id)
        print(f"  env_cfg_entry_point: {spec.kwargs.get('env_cfg_entry_point')}")
        print(f"  rl_games_cfg_entry_point: {spec.kwargs.get('rl_games_cfg_entry_point')}")


if __name__ == "__main__":
    main()
