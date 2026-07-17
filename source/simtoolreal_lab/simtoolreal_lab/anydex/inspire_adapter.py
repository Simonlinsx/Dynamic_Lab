"""Bridge DOMINO's AnyDex outputs to the faithful six-motor RH56BFX model."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


GRASP_TYPE_NAMES = {
    1: "Ring",
    2: "Prismatic_2_Finger",
    3: "Prismatic_3_Finger",
    4: "Large_Diameter",
    5: "Medium_Wrap",
    6: "Tripod",
    7: "Sphere_3_Finger",
    8: "Distal_Type",
}

# Minimum number of non-thumb digits implied by each AnyDex grasp family.
# Ring and Prismatic_2_Finger are pinch grasps; the remaining families are
# multi-finger grasps after projection onto RH56BFX's mimic manifold.
ANYDEX_REQUIRED_NON_THUMB_CONTACTS = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
}

ACTIVE_JOINT_NAMES = (
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)

ACTIVE_JOINT_LOWER = np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float64)
ACTIVE_JOINT_UPPER = np.array((1.308, 0.6, 1.47, 1.47, 1.47, 1.47), dtype=np.float64)

DEFAULT_FRAME_OFFSET_XYZ = (-0.00902886, 0.00683512, 0.02743894)
DEFAULT_FRAME_OFFSET_RPY = (0.10579421, 0.04518228, 1.4874773)
DEFAULT_PREGRASP_DISTANCE = 0.075
DEFAULT_DEPTH_BASE_OFFSET = 0.010


@dataclass(frozen=True)
class AnyDexInspirePaths:
    """External paths for the isolated Python 3.8 AnyDex inference process."""

    anydex_root: Path = Path("/data1/linsixu/AnyDexGrasp")
    domino_root: Path = Path("/data1/linsixu/DOMINO")
    python: Path = Path("/data1/linsixu/miniconda3/envs/anydex-torch/bin/python")

    @property
    def predictor(self) -> Path:
        return self.domino_root / "script" / "run_anydex_inspire_from_pointcloud.py"

    @property
    def checkpoint(self) -> Path:
        return self.anydex_root / "logs" / "model" / "checkpoint.tar.18"

    @property
    def decision_models(self) -> Path:
        return self.anydex_root / "logs" / "model" / "inspire_model" / "final_single_point" / "obj140"

    @property
    def width_angle_table(self) -> Path:
        return (
            self.anydex_root
            / "generate_mesh_and_pointcloud"
            / "inspire_urdf"
            / "width_12Dangle_6Dangle.json"
        )


@dataclass(frozen=True)
class AnyDexInspireCandidate:
    """One AnyDex candidate expressed as an Isaac Lab hand-base target."""

    rank: int
    score: float
    source_grasp_type: int
    source_width: float
    grasp_type: int
    grasp_type_name: str
    width: float
    depth: float
    active_joint_targets: np.ndarray
    angle_12d: np.ndarray
    two_finger_pose: np.ndarray
    inspire_pose: np.ndarray
    approach_direction_world: np.ndarray
    anydex_hand_matrix_world: np.ndarray
    grasp_hand_base_matrix_world: np.ndarray
    pregrasp_hand_base_matrix_world: np.ndarray

    def as_dict(self) -> dict:
        return {
            "rank": int(self.rank),
            "score": float(self.score),
            "source_grasp_type": int(self.source_grasp_type),
            "source_grasp_type_name": GRASP_TYPE_NAMES.get(self.source_grasp_type, "unknown"),
            "source_width": float(self.source_width),
            "grasp_type": int(self.grasp_type),
            "grasp_type_name": self.grasp_type_name,
            "width": float(self.width),
            "depth": float(self.depth),
            "active_joint_names": list(ACTIVE_JOINT_NAMES),
            "active_joint_targets": self.active_joint_targets.tolist(),
            "angle_12d": self.angle_12d.tolist(),
            "approach_direction_world": self.approach_direction_world.tolist(),
            "anydex_hand_matrix_world": self.anydex_hand_matrix_world.tolist(),
            "grasp_hand_base_matrix_world": self.grasp_hand_base_matrix_world.tolist(),
            "pregrasp_hand_base_matrix_world": self.pregrasp_hand_base_matrix_world.tolist(),
        }


def _rotation_matrix_from_rpy(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    rotation_y = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rotation_z = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rotation_z @ rotation_y @ rotation_x


def _transform_matrix(translation: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def _nearest_width_entry(
    table: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    grasp_type_name: str,
    width_m: float,
) -> Mapping[str, Sequence[float]]:
    width_table = table.get(grasp_type_name)
    if not width_table:
        raise KeyError(f"AnyDex width table has no grasp type {grasp_type_name!r}")
    width_cm = float(width_m) * 100.0
    width_key = min(width_table, key=lambda value: abs(float(value) - width_cm))
    return width_table[width_key]


def _faithful_active_targets(angle_12d: Sequence[float]) -> np.ndarray:
    """Project AnyDex's independently actuated 12D pose onto the official mimic manifold."""
    angle = np.asarray(angle_12d, dtype=np.float64).reshape(12)
    # AnyDex order is index(2), middle(2), ring(2), pinky(2), then thumb(4).
    targets = np.array((-angle[8], -angle[9], -angle[0], -angle[2], -angle[4], -angle[6]))
    return np.clip(targets, ACTIVE_JOINT_LOWER, ACTIVE_JOINT_UPPER)


def _parse_candidate(
    payload: Mapping[str, Sequence[float]],
    rank: int,
    width_angle_table: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    *,
    frame_offset_xyz: Sequence[float],
    frame_offset_rpy: Sequence[float],
    pregrasp_distance: float,
    depth_base_offset: float,
    grasp_type_override: int | None,
    width_override: float | None,
) -> AnyDexInspireCandidate:
    two_finger = np.asarray(payload["two_finger_pose"], dtype=np.float64).reshape(-1)
    inspire = np.asarray(payload["InspiredHandR_pose"], dtype=np.float64).reshape(-1)
    if two_finger.size < 17 or inspire.size < 23:
        raise ValueError("Malformed AnyDex candidate: expected 17D two-finger and 23D Inspire poses")

    source_grasp_type = int(round(float(inspire[2])))
    source_width = float(inspire[22])
    grasp_type = source_grasp_type if grasp_type_override is None else int(grasp_type_override)
    grasp_type_name = GRASP_TYPE_NAMES.get(grasp_type)
    if grasp_type_name is None:
        raise ValueError(f"Unsupported AnyDex Inspire grasp type {grasp_type}")
    width = source_width if width_override is None else float(width_override)
    if not 0.0 < width <= 0.11:
        raise ValueError(f"AnyDex Inspire width must be in (0, 0.11], got {width}")
    width_entry = _nearest_width_entry(width_angle_table, grasp_type_name, width)
    angle_12d = np.asarray(width_entry["12d"], dtype=np.float64).reshape(12)

    if grasp_type_override is None and width_override is None:
        anydex_hand_matrix = _transform_matrix(inspire[12:15], inspire[3:12])
    else:
        # DOMINO defines the multifinger base from the two-finger grasp and
        # the selected grasp-type/width table entry. Recompute that exact
        # relation when object geometry constrains the network's type/width.
        two_finger_matrix = _transform_matrix(two_finger[13:16], two_finger[4:13])
        inspire_offset = _transform_matrix(width_entry["translation"], width_entry["rotation"])
        anydex_hand_matrix = two_finger_matrix @ np.linalg.inv(inspire_offset)
    frame_offset = _transform_matrix(frame_offset_xyz, _rotation_matrix_from_rpy(frame_offset_rpy))
    grasp_hand_base = anydex_hand_matrix @ frame_offset
    approach_direction = two_finger[4:13].reshape(3, 3)[:, 0]
    approach_direction = approach_direction / max(float(np.linalg.norm(approach_direction)), 1.0e-12)
    grasp_hand_base[:3, 3] += approach_direction * (float(inspire[1]) + float(depth_base_offset))
    pregrasp_hand_base = grasp_hand_base.copy()
    pregrasp_hand_base[:3, 3] -= approach_direction * float(pregrasp_distance)

    return AnyDexInspireCandidate(
        rank=int(rank),
        score=float(inspire[0]),
        source_grasp_type=source_grasp_type,
        source_width=source_width,
        grasp_type=grasp_type,
        grasp_type_name=grasp_type_name,
        width=width,
        depth=float(inspire[1]),
        active_joint_targets=_faithful_active_targets(angle_12d),
        angle_12d=angle_12d,
        two_finger_pose=two_finger,
        inspire_pose=inspire,
        approach_direction_world=approach_direction,
        anydex_hand_matrix_world=anydex_hand_matrix,
        grasp_hand_base_matrix_world=grasp_hand_base,
        pregrasp_hand_base_matrix_world=pregrasp_hand_base,
    )


def load_anydex_candidates(
    result_path: str | Path,
    *,
    width_angle_table_path: str | Path,
    frame_offset_xyz: Sequence[float] = DEFAULT_FRAME_OFFSET_XYZ,
    frame_offset_rpy: Sequence[float] = DEFAULT_FRAME_OFFSET_RPY,
    pregrasp_distance: float = DEFAULT_PREGRASP_DISTANCE,
    depth_base_offset: float = DEFAULT_DEPTH_BASE_OFFSET,
    grasp_type_override: int | None = None,
    width_override: float | None = None,
) -> list[AnyDexInspireCandidate]:
    result_path = Path(result_path).expanduser().resolve()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    if not data.get("success", False):
        raise RuntimeError(f"AnyDex inference failed: {data.get('error', 'unknown error')}")
    raw_candidates = data.get("candidates") or [data.get("selected")]
    if not raw_candidates or raw_candidates[0] is None:
        raise RuntimeError("AnyDex result has no candidate")
    table = json.loads(Path(width_angle_table_path).read_text(encoding="utf-8"))
    return [
        _parse_candidate(
            candidate,
            rank,
            table,
            frame_offset_xyz=frame_offset_xyz,
            frame_offset_rpy=frame_offset_rpy,
            pregrasp_distance=pregrasp_distance,
            depth_base_offset=depth_base_offset,
            grasp_type_override=grasp_type_override,
            width_override=width_override,
        )
        for rank, candidate in enumerate(raw_candidates)
    ]


def _fibonacci_sphere_points(point_count: int, radius: float) -> np.ndarray:
    indices = np.arange(point_count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / float(point_count)
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = np.pi * (1.0 + np.sqrt(5.0)) * indices
    return float(radius) * np.stack(
        (radial * np.cos(azimuth), radial * np.sin(azimuth), z), axis=-1
    )


def _box_surface_points(point_count: int, size: np.ndarray) -> np.ndarray:
    indices = np.arange(point_count, dtype=np.float64)
    face = np.mod(indices.astype(np.int64), 6)
    sample = np.floor(indices / 6.0) + 0.5
    u = 2.0 * np.mod(sample * 0.6180339887498949, 1.0) - 1.0
    v = 2.0 * np.mod(sample * 0.4142135623730950, 1.0) - 1.0
    half = 0.5 * size
    points = np.zeros((point_count, 3), dtype=np.float64)
    for axis in range(3):
        for side in range(2):
            mask = face == 2 * axis + side
            other_axes = [candidate for candidate in range(3) if candidate != axis]
            points[mask, axis] = (-1.0 if side == 0 else 1.0) * half[axis]
            points[mask, other_axes[0]] = u[mask] * half[other_axes[0]]
            points[mask, other_axes[1]] = v[mask] * half[other_axes[1]]
    return points


def _cylinder_surface_points(point_count: int, radius: float, height: float) -> np.ndarray:
    side_area = 2.0 * np.pi * float(radius) * float(height)
    cap_area = 2.0 * np.pi * float(radius) ** 2
    side_count = int(round(point_count * side_area / max(side_area + cap_area, 1.0e-12)))
    side_count = min(max(side_count, 1), point_count - 2)
    cap_count = point_count - side_count

    side_index = np.arange(side_count, dtype=np.float64) + 0.5
    side_theta = 2.0 * np.pi * np.mod(side_index * 0.6180339887498949, 1.0)
    side_z = float(height) * (np.mod(side_index * 0.4142135623730950, 1.0) - 0.5)
    side = np.stack(
        (
            float(radius) * np.cos(side_theta),
            float(radius) * np.sin(side_theta),
            side_z,
        ),
        axis=-1,
    )

    cap_index = np.arange(cap_count, dtype=np.float64) + 0.5
    cap_theta = 2.0 * np.pi * np.mod(cap_index * 0.6180339887498949, 1.0)
    cap_radius = float(radius) * np.sqrt(np.mod(cap_index * 0.4142135623730950, 1.0))
    cap_z = np.where(np.mod(np.arange(cap_count), 2) == 0, -0.5 * float(height), 0.5 * float(height))
    caps = np.stack(
        (cap_radius * np.cos(cap_theta), cap_radius * np.sin(cap_theta), cap_z),
        axis=-1,
    )
    return np.concatenate((side, caps), axis=0)


def make_primitive_predictor_input(
    output_path: str | Path,
    *,
    object_center_world: Sequence[float],
    shape: str,
    size: Sequence[float],
    point_count: int = 4096,
    predictor_center: Sequence[float] = (0.0, 0.0, 0.49),
) -> Path:
    """Write an object-only primitive cloud expected by DOMINO's AnyDex predictor."""
    if point_count < 32:
        raise ValueError("point_count must be at least 32")
    shape = str(shape).lower()
    if shape == "cube":
        shape = "box"
    if shape not in {"sphere", "box", "cylinder"}:
        raise ValueError(f"Unsupported primitive shape {shape!r}")
    size_array = np.asarray(size, dtype=np.float64).reshape(3)
    if np.any(size_array <= 0.0):
        raise ValueError("all primitive dimensions must be positive")
    if shape == "sphere":
        if not np.allclose(size_array, size_array[0], rtol=0.0, atol=1.0e-9):
            raise ValueError("sphere size must contain three equal diameters")
        local_points = _fibonacci_sphere_points(point_count, 0.5 * float(size_array[0]))
    elif shape == "box":
        local_points = _box_surface_points(point_count, size_array)
    else:
        if not np.isclose(size_array[0], size_array[1], rtol=0.0, atol=1.0e-9):
            raise ValueError("cylinder x/y dimensions must be equal diameters")
        local_points = _cylinder_surface_points(
            point_count,
            0.5 * float(size_array[0]),
            float(size_array[2]),
        )
    predictor_center_array = np.asarray(predictor_center, dtype=np.float64).reshape(3)
    points = predictor_center_array + local_points

    # DOMINO's camera convention keeps x and flips predictor y/z into world y/z.
    world_from_predictor = np.eye(4, dtype=np.float64)
    world_from_predictor[:3, :3] = np.diag((1.0, -1.0, -1.0))
    world_center = np.asarray(object_center_world, dtype=np.float64).reshape(3)
    world_from_predictor[:3, 3] = world_center - world_from_predictor[:3, :3] @ predictor_center_array

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points=points.astype(np.float32),
        world_from_predictor=world_from_predictor,
        object_center_world=world_center,
        object_center_predictor=predictor_center_array,
        object_shape=np.array(shape),
        object_size=size_array,
        radius=np.array(0.5 * float(size_array[0]), dtype=np.float64),
        height=np.array(float(size_array[2]), dtype=np.float64),
    )
    return output_path


def make_sphere_predictor_input(
    output_path: str | Path,
    *,
    object_center_world: Sequence[float],
    radius: float,
    point_count: int = 4096,
    predictor_center: Sequence[float] = (0.0, 0.0, 0.49),
) -> Path:
    """Backward-compatible helper for a spherical AnyDex predictor input."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return make_primitive_predictor_input(
        output_path,
        object_center_world=object_center_world,
        shape="sphere",
        size=(2.0 * float(radius),) * 3,
        point_count=point_count,
        predictor_center=predictor_center,
    )


def build_predictor_command(
    input_path: str | Path,
    output_path: str | Path,
    *,
    paths: AnyDexInspirePaths = AnyDexInspirePaths(),
    candidate_limit: int = 10,
    network_candidate_limit: int = 500,
    visualization_dir: str | Path | None = None,
) -> list[str]:
    command = [
        str(paths.python),
        str(paths.predictor),
        "--input",
        str(Path(input_path).expanduser().resolve()),
        "--output",
        str(Path(output_path).expanduser().resolve()),
        "--checkpoint_path",
        str(paths.checkpoint),
        "--anydex_root",
        str(paths.anydex_root),
        "--inspire_model_path",
        str(paths.decision_models),
        "--use_graspnet_v2",
        "--candidate_limit",
        str(int(candidate_limit)),
        "--network_candidate_limit",
        str(int(network_candidate_limit)),
    ]
    if visualization_dir is not None:
        command.extend(("--visualization_dir", str(Path(visualization_dir).expanduser().resolve())))
    return command


def run_predictor(
    input_path: str | Path,
    output_path: str | Path,
    *,
    paths: AnyDexInspirePaths = AnyDexInspirePaths(),
    cuda_visible_devices: str | None = None,
    candidate_limit: int = 10,
    network_candidate_limit: int = 500,
    visualization_dir: str | Path | None = None,
) -> Path:
    for required_path in (paths.python, paths.predictor, paths.checkpoint, paths.decision_models):
        if not required_path.exists():
            raise FileNotFoundError(f"Required AnyDex path does not exist: {required_path}")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    environment.setdefault("OMP_NUM_THREADS", "12")
    subprocess.run(
        build_predictor_command(
            input_path,
            output_path,
            paths=paths,
            candidate_limit=candidate_limit,
            network_candidate_limit=network_candidate_limit,
            visualization_dir=visualization_dir,
        ),
        cwd=paths.domino_root,
        env=environment,
        check=True,
    )
    return output_path
