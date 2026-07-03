"""Export tabletop affordance-label meshes as lightweight URDF assets.

The tabletop transport task uses the affordance-label meshes as visual/collision
assets, while the RL observation still keeps compact size/shape statistics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LAB_ROOT = Path(__file__).resolve().parents[1]
AFFORDANCE_ROOT = LAB_ROOT / "assets" / "affordance_labels"
OUTPUT_ROOT = LAB_ROOT / "assets" / "generated" / "tabletop_affordance_meshes"


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    target_size: tuple[float, float, float]
    mass: float
    up_axis: str = "Z"


ASSET_SPECS = (
    AssetSpec("domino20/cup", (0.064, 0.064, 0.078), 0.040, "Y"),
    AssetSpec("domino20/mug", (0.070, 0.070, 0.080), 0.050, "Y"),
    AssetSpec("domino20/kettle", (0.095, 0.075, 0.082), 0.070, "Y"),
    AssetSpec("dextoolbench/hammer/claw_hammer", (0.195, 0.086, 0.032), 0.100, "Z"),
    AssetSpec("dextoolbench/hammer/mallet_hammer", (0.170, 0.070, 0.045), 0.090, "Z"),
    AssetSpec("dextoolbench/brush/blue_brush", (0.210, 0.070, 0.040), 0.070, "Z"),
    AssetSpec("dextoolbench/brush/red_brush", (0.210, 0.065, 0.045), 0.070, "Z"),
)


MATERIALS = {
    "aff_positive": (0.05, 0.78, 0.25),
    "aff_negative": (0.92, 0.07, 0.05),
    "aff_ignore": (0.70, 0.70, 0.68),
    "aff_neutral": (0.28, 0.42, 0.78),
}


def _asset_out_dir(asset_id: str, output_root: Path) -> Path:
    return output_root.joinpath(*asset_id.split("/"))


def _transform_vertices(vertices: np.ndarray, spec: AssetSpec) -> tuple[np.ndarray, dict]:
    vertices = np.asarray(vertices, dtype=np.float64)
    source_bounds_min = vertices.min(axis=0)
    source_bounds_max = vertices.max(axis=0)

    if spec.up_axis.upper() == "Y":
        vertices = vertices[:, [0, 2, 1]]
    elif spec.up_axis.upper() != "Z":
        raise ValueError(f"unsupported up_axis={spec.up_axis!r} for {spec.asset_id}")

    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    extents = np.maximum(bounds_max - bounds_min, 1.0e-9)
    target_size = np.asarray(spec.target_size, dtype=np.float64)
    scale = target_size / extents
    transformed = (vertices - center) * scale

    return transformed.astype(np.float32), {
        "source_bounds_min": source_bounds_min.tolist(),
        "source_bounds_max": source_bounds_max.tolist(),
        "source_up_axis": spec.up_axis,
        "target_size": list(spec.target_size),
        "scale": scale.tolist(),
    }


def _face_materials(data: np.lib.npyio.NpzFile) -> np.ndarray:
    faces = np.asarray(data["faces"], dtype=np.int64)
    pos = np.asarray(data["positive_mask"], dtype=bool)
    neg = np.asarray(data["negative_mask"], dtype=bool)
    ignore = np.asarray(data["ignore_mask"], dtype=bool)
    face_pos = pos[faces].sum(axis=1)
    face_neg = neg[faces].sum(axis=1)
    face_ignore = ignore[faces].sum(axis=1)
    materials = np.full(len(faces), "aff_neutral", dtype=object)
    materials[(face_ignore >= 2) & (face_pos == 0) & (face_neg == 0)] = "aff_ignore"
    materials[(face_neg > face_pos) & (face_neg > 0)] = "aff_negative"
    materials[(face_pos >= face_neg) & (face_pos > 0)] = "aff_positive"
    return materials


def _write_mtl(path: Path) -> None:
    lines: list[str] = []
    for name, color in MATERIALS.items():
        lines.extend(
            [
                f"newmtl {name}",
                f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}",
                "Ka 0.020000 0.020000 0.020000",
                "Ks 0.080000 0.080000 0.080000",
                "Ns 24.000000",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, face_materials: np.ndarray) -> None:
    mtl_name = f"{path.stem}.mtl"
    with path.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        for vertex in vertices:
            f.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        current_material = None
        for face, material in zip(faces, face_materials, strict=True):
            if material != current_material:
                f.write(f"usemtl {material}\n")
                current_material = material
            a, b, c = (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1)
            f.write(f"f {a} {b} {c}\n")


def _box_inertia(mass: float, size: tuple[float, float, float]) -> tuple[float, float, float]:
    sx, sy, sz = size
    ixx = mass * (sy * sy + sz * sz) / 12.0
    iyy = mass * (sx * sx + sz * sz) / 12.0
    izz = mass * (sx * sx + sy * sy) / 12.0
    return ixx, iyy, izz


def _write_urdf(path: Path, obj_name: str, spec: AssetSpec) -> None:
    ixx, iyy, izz = _box_inertia(spec.mass, spec.target_size)
    path.write_text(
        f"""<?xml version=\"1.0\"?>
<robot name=\"{obj_name}\">
  <link name=\"{obj_name}\">
    <visual>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry>
        <mesh filename=\"{obj_name}.obj\" scale=\"1 1 1\"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry>
        <mesh filename=\"{obj_name}.obj\" scale=\"1 1 1\"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <mass value=\"{spec.mass:.8f}\"/>
      <inertia ixx=\"{ixx:.10f}\" ixy=\"0\" ixz=\"0\" iyy=\"{iyy:.10f}\" iyz=\"0\" izz=\"{izz:.10f}\"/>
    </inertial>
  </link>
</robot>
""",
        encoding="utf-8",
    )


def export_asset(spec: AssetSpec, output_root: Path) -> dict:
    label_path = AFFORDANCE_ROOT.joinpath(*spec.asset_id.split("/")) / "grasp_affordance_clean_v2.npz"
    if not label_path.exists():
        raise FileNotFoundError(f"missing affordance mesh label: {label_path}")

    data = np.load(label_path, allow_pickle=True)
    vertices, transform_meta = _transform_vertices(data["vertices_obj"], spec)
    faces = np.asarray(data["faces"], dtype=np.int64)
    materials = _face_materials(data)

    out_dir = _asset_out_dir(spec.asset_id, output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_name = spec.asset_id.rsplit("/", 1)[-1]
    obj_path = out_dir / f"{obj_name}.obj"
    mtl_path = out_dir / f"{obj_name}.mtl"
    urdf_path = out_dir / f"{obj_name}.urdf"

    _write_mtl(mtl_path)
    _write_obj(obj_path, vertices, faces, materials)
    _write_urdf(urdf_path, obj_name, spec)

    material_counts = {
        name: int((materials == name).sum())
        for name in sorted(MATERIALS)
    }
    record = {
        "asset_id": spec.asset_id,
        "label_path": str(label_path),
        "obj_path": str(obj_path),
        "mtl_path": str(mtl_path),
        "urdf_path": str(urdf_path),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "material_face_counts": material_counts,
        "mass": float(spec.mass),
        **transform_meta,
    }
    (out_dir / "export_meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    records = [export_asset(spec, args.output_root) for spec in ASSET_SPECS]
    manifest_path = args.output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"assets": records}, indent=2), encoding="utf-8")
    print(f"exported {len(records)} assets -> {manifest_path}")


if __name__ == "__main__":
    main()
