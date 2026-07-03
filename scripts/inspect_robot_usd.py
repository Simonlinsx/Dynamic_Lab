#!/usr/bin/env python3
"""Inspect Robot USD prims for collision/material debugging."""

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
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--output-json", default=None, help="Optional output JSON path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from pxr import Usd, UsdPhysics, UsdShade  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


def _prim_record(prim) -> dict:
    record = {
        "path": prim.GetPath().pathString,
        "type": prim.GetTypeName(),
        "is_instance": prim.IsInstance(),
        "is_instance_proxy": prim.IsInstanceProxy(),
        "has_collision_api": prim.HasAPI(UsdPhysics.CollisionAPI),
        "has_material_binding_api": prim.HasAPI(UsdShade.MaterialBindingAPI),
    }
    if prim.IsInstance():
        prototype = prim.GetPrototype()
        record["prototype_path"] = prototype.GetPath().pathString if prototype.IsValid() else None
    return record


def main() -> None:
    carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        base = env.unwrapped
        stage = base.scene.stage
        robot_path = f"{base.scene.env_prim_paths[0]}/Robot"
        root_prim = stage.GetPrimAtPath(robot_path)

        all_records = []
        collision_records = []
        instance_records = []
        prototype_records = []
        seen_prototypes = set()

        for prim in Usd.PrimRange(root_prim):
            record = _prim_record(prim)
            all_records.append(record)
            if record["has_collision_api"]:
                collision_records.append(record)
            if record["is_instance"] or record["is_instance_proxy"]:
                instance_records.append(record)
            if prim.IsInstance():
                prototype = prim.GetPrototype()
                if prototype.IsValid():
                    prototype_path = prototype.GetPath().pathString
                    if prototype_path not in seen_prototypes:
                        seen_prototypes.add(prototype_path)
                        for prototype_prim in Usd.PrimRange(prototype):
                            prototype_record = _prim_record(prototype_prim)
                            prototype_records.append(prototype_record)

        summary = {
            "task": args_cli.task,
            "robot_path": robot_path,
            "all_count": len(all_records),
            "collision_count": len(collision_records),
            "instance_count": len(instance_records),
            "prototype_count": len(prototype_records),
            "prototype_collision_count": sum(r["has_collision_api"] for r in prototype_records),
            "first_all": all_records[:120],
            "collisions": collision_records[:120],
            "instances": instance_records[:120],
            "prototypes": prototype_records[:160],
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
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
