#!/usr/bin/env python3
"""Plan collision-aware Franka trajectories to DOMINO AnyDex Inspire grasps.

Run this helper with the DOMINO conda environment.  It deliberately plans only
the seven Franka joints; Isaac Lab remains responsible for the faithful
RH56BFX hand dynamics, contacts, and success checks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXT_SOURCE = REPO_ROOT / "source" / "simtoolreal_lab"
DOMINO_CUROBO_SOURCE = Path("/data1/linsixu/DOMINO/envs/curobo/src")
for source in (EXT_SOURCE, DOMINO_CUROBO_SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from curobo.geom.types import WorldConfig  # noqa: E402
from curobo.types.base import TensorDeviceType  # noqa: E402
from curobo.types.math import Pose  # noqa: E402
from curobo.types.robot import JointState  # noqa: E402
from curobo.wrap.reacher.motion_gen import (  # noqa: E402
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)


ADAPTER_PATH = EXT_SOURCE / "simtoolreal_lab" / "anydex" / "inspire_adapter.py"
adapter_spec = importlib.util.spec_from_file_location("simtoolreal_anydex_inspire_adapter", ADAPTER_PATH)
if adapter_spec is None or adapter_spec.loader is None:
    raise ImportError(f"Could not load AnyDex adapter from {ADAPTER_PATH}")
adapter_module = importlib.util.module_from_spec(adapter_spec)
sys.modules[adapter_spec.name] = adapter_module
adapter_spec.loader.exec_module(adapter_module)
AnyDexInspirePaths = adapter_module.AnyDexInspirePaths
load_anydex_candidates = adapter_module.load_anydex_candidates


FRANKA_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
DEFAULT_START_Q = (0.0, -0.35, 0.0, -2.20, 0.0, 2.39, 0.7853981633974483)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anydex-result",
        type=Path,
        default=REPO_ROOT / "outputs/anydex_inspire/20260713_sphere60/sphere60_result.json",
    )
    parser.add_argument("--candidate-ranks", type=int, nargs="+", default=(0,))
    parser.add_argument("--start-q", type=float, nargs=7, default=DEFAULT_START_Q)
    parser.add_argument("--lift-height", type=float, default=0.10)
    parser.add_argument("--pregrasp-distance", type=float, default=0.075)
    parser.add_argument("--depth-base-offset", type=float, default=0.010)
    parser.add_argument(
        "--frame-offset-xyz",
        type=float,
        nargs=3,
        default=(-0.00902886, 0.00683512, 0.02743894),
    )
    parser.add_argument(
        "--frame-offset-rpy",
        type=float,
        nargs=3,
        default=(0.10579421, 0.04518228, 1.4874773),
    )
    parser.add_argument("--grasp-type-override", type=int, choices=tuple(range(1, 9)), default=None)
    parser.add_argument("--width-override", type=float, default=None)
    parser.add_argument("--object-pos", type=float, nargs=3, default=(0.535, 0.075, 0.328))
    parser.add_argument("--object-shape", choices=("sphere", "box", "cylinder"), default="sphere")
    parser.add_argument("--object-size", type=float, nargs=3, default=(0.060, 0.060, 0.060))
    parser.add_argument("--object-radius", type=float, default=0.030)
    parser.add_argument("--table-center", type=float, nargs=3, default=(0.58, -0.08, 0.2755))
    parser.add_argument("--table-dims", type=float, nargs=3, default=(1.00, 0.80, 0.045))
    parser.add_argument(
        "--base-curobo-config",
        type=Path,
        default=Path("/data1/linsixu/RoboTwin/assets/embodiments/franka-inspire/curobo.yml"),
    )
    parser.add_argument(
        "--faithful-urdf",
        type=Path,
        default=REPO_ROOT
        / "assets/embodiments/franka-inspire-rh56bfx-mimic/franka_inspire_rh56bfx_mimic.urdf",
    )
    parser.add_argument("--interpolation-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--num-ik-seeds", type=int, default=32)
    parser.add_argument("--num-graph-seeds", type=int, default=4)
    parser.add_argument("--num-trajopt-seeds", type=int, default=4)
    parser.add_argument("--position-threshold", type=float, default=0.005)
    parser.add_argument("--rotation-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _matrix_to_pose(matrix: np.ndarray, tensor_args: TensorDeviceType) -> Pose:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    rotation = matrix[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            (
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.array(
                (
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                )
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.array(
                (
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                )
            )
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.array(
                (
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
    return Pose(
        position=tensor_args.to_device(matrix[None, :3, 3]),
        quaternion=tensor_args.to_device(quaternion[None]),
    )


def _world_payload(args: argparse.Namespace, include_object: bool) -> dict:
    payload = {
        "cuboid": {
            "table": {
                "dims": [float(value) for value in args.table_dims],
                "pose": [*map(float, args.table_center), 1.0, 0.0, 0.0, 0.0],
            }
        }
    }
    if include_object:
        if args.object_shape == "sphere":
            payload["sphere"] = {
                "target_object": {
                    "position": [float(value) for value in args.object_pos],
                    "radius": float(args.object_radius),
                }
            }
        else:
            # CuRobo's primitive world supports cuboids directly.  Use the
            # exact box or the cylinder's conservative bounding box while the
            # faithful Isaac Lab execution retains the true cylinder collider.
            payload["cuboid"]["target_object"] = {
                "dims": [float(value) for value in args.object_size],
                "pose": [*map(float, args.object_pos), 1.0, 0.0, 0.0, 0.0],
            }
    return payload


def _load_robot_payload(args: argparse.Namespace) -> dict:
    config_path = args.base_curobo_config.expanduser().resolve()
    faithful_urdf = args.faithful_urdf.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not faithful_urdf.exists():
        raise FileNotFoundError(faithful_urdf)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    kinematics = payload["robot_cfg"]["kinematics"]
    kinematics["urdf_path"] = str(faithful_urdf)
    kinematics["asset_root_path"] = str(faithful_urdf.parent)
    kinematics["ee_link"] = "hand_base_link"
    return payload


def _plan_segment(
    motion_gen: MotionGen,
    tensor_args: TensorDeviceType,
    start_q: np.ndarray,
    target_matrix: np.ndarray,
    plan_config: MotionGenPlanConfig,
) -> tuple[np.ndarray | None, dict]:
    start_state = JointState.from_position(
        tensor_args.to_device(np.asarray(start_q, dtype=np.float32).reshape(1, 7)),
        joint_names=list(FRANKA_JOINT_NAMES),
    )
    result = motion_gen.plan_single(
        start_state,
        _matrix_to_pose(target_matrix, tensor_args),
        plan_config,
    )
    success = bool(result.success.item())
    diagnostics = {
        "success": success,
        "status": str(result.status),
        "solve_time_s": float(result.solve_time),
        "motion_time_s": float(result.motion_time) if result.motion_time is not None else None,
        "target_matrix": np.asarray(target_matrix, dtype=float).tolist(),
    }
    if not success:
        return None, diagnostics
    trajectory = result.get_interpolated_plan().position.detach().cpu().numpy()
    trajectory = np.asarray(trajectory, dtype=np.float32).reshape(-1, 7)
    finite = np.all(np.isfinite(trajectory), axis=1)
    trajectory = trajectory[finite]
    if trajectory.shape[0] == 0:
        diagnostics["success"] = False
        diagnostics["status"] = "empty_interpolated_trajectory"
        return None, diagnostics
    diagnostics["waypoints"] = int(trajectory.shape[0])
    diagnostics["final_q"] = trajectory[-1].astype(float).tolist()
    return trajectory, diagnostics


def main() -> None:
    args = _parse_args()
    if args.interpolation_dt <= 0.0:
        raise ValueError("--interpolation-dt must be positive")
    object_size = np.asarray(args.object_size, dtype=np.float64)
    if np.any(object_size <= 0.0):
        raise ValueError("--object-size dimensions must be positive")
    if args.object_shape in {"sphere", "cylinder"} and args.object_radius <= 0.0:
        raise ValueError("--object-radius must be positive")
    if args.object_shape == "sphere" and not np.allclose(
        object_size,
        2.0 * float(args.object_radius),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("sphere --object-size must equal 2 * --object-radius on all axes")
    if args.object_shape == "cylinder" and not np.allclose(
        object_size[:2],
        2.0 * float(args.object_radius),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("cylinder x/y --object-size must equal 2 * --object-radius")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    paths = AnyDexInspirePaths()
    candidates = load_anydex_candidates(
        args.anydex_result,
        width_angle_table_path=paths.width_angle_table,
        frame_offset_xyz=args.frame_offset_xyz,
        frame_offset_rpy=args.frame_offset_rpy,
        pregrasp_distance=float(args.pregrasp_distance),
        depth_base_offset=float(args.depth_base_offset),
        grasp_type_override=args.grasp_type_override,
        width_override=args.width_override,
    )
    ranks = [int(rank) for rank in args.candidate_ranks]
    if any(rank < 0 or rank >= len(candidates) for rank in ranks):
        raise ValueError(f"Candidate ranks {ranks} outside [0, {len(candidates) - 1}]")

    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    pregrasp_world = _world_payload(args, include_object=True)
    table_world = _world_payload(args, include_object=False)
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        _load_robot_payload(args),
        pregrasp_world,
        tensor_args,
        interpolation_dt=float(args.interpolation_dt),
        num_ik_seeds=int(args.num_ik_seeds),
        num_graph_seeds=int(args.num_graph_seeds),
        num_trajopt_seeds=int(args.num_trajopt_seeds),
        position_threshold=float(args.position_threshold),
        rotation_threshold=float(args.rotation_threshold),
        collision_activation_distance=0.005,
        evaluate_interpolated_trajectory=True,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    plan_config = MotionGenPlanConfig(
        max_attempts=int(args.max_attempts),
        timeout=float(args.timeout),
        enable_graph_attempt=1,
        ik_fail_return=6,
    )

    arrays: dict[str, np.ndarray] = {}
    candidate_summaries: list[dict] = []
    for rank in ranks:
        candidate = candidates[rank]
        current_q = np.asarray(args.start_q, dtype=np.float32)
        lift_matrix = candidate.grasp_hand_base_matrix_world.copy()
        lift_matrix[2, 3] += float(args.lift_height)
        segment_targets = (
            ("pregrasp", candidate.pregrasp_hand_base_matrix_world, pregrasp_world),
            ("grasp", candidate.grasp_hand_base_matrix_world, table_world),
            ("lift", lift_matrix, table_world),
        )
        segment_summary: dict[str, dict] = {}
        candidate_ok = True
        for phase, target_matrix, world_payload in segment_targets:
            motion_gen.update_world(WorldConfig.from_dict(world_payload))
            trajectory, diagnostics = _plan_segment(
                motion_gen,
                tensor_args,
                current_q,
                target_matrix,
                plan_config,
            )
            segment_summary[phase] = diagnostics
            if trajectory is None:
                candidate_ok = False
                break
            arrays[f"rank_{rank}_{phase}"] = trajectory
            current_q = trajectory[-1]
        candidate_summaries.append(
            {
                "rank": rank,
                "success": candidate_ok,
                "candidate": candidate.as_dict(),
                "segments": segment_summary,
            }
        )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    summary = {
        "contract": "DOMINO AnyDex candidate plus CuRobo Franka motion planning; no RL",
        "anydex_result": str(args.anydex_result.expanduser().resolve()),
        "faithful_urdf": str(args.faithful_urdf.expanduser().resolve()),
        "base_curobo_config": str(args.base_curobo_config.expanduser().resolve()),
        "output": str(output_path),
        "start_q": [float(value) for value in args.start_q],
        "interpolation_dt": float(args.interpolation_dt),
        "table_center": [float(value) for value in args.table_center],
        "table_dims": [float(value) for value in args.table_dims],
        "object_pos": [float(value) for value in args.object_pos],
        "object_shape": str(args.object_shape),
        "object_size": [float(value) for value in args.object_size],
        "object_radius": float(args.object_radius),
        "curobo_object_collision_proxy": (
            "sphere" if args.object_shape == "sphere" else "exact_cuboid"
            if args.object_shape == "box"
            else "conservative_bounding_cuboid"
        ),
        "lift_height": float(args.lift_height),
        "candidate_ranks": ranks,
        "success_count": sum(int(item["success"]) for item in candidate_summaries),
        "candidates": candidate_summaries,
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": str(summary_path), "success_count": summary["success_count"]}))
    if summary["success_count"] != len(ranks):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
