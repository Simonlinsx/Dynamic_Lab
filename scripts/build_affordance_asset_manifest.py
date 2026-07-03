#!/usr/bin/env python
"""Build the asset manifest for offline affordance annotation.

This is intentionally separate from the training env. It records the meshes
that should be labeled first, plus where per-object labels should be stored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]


DEXTOOLBENCH_OBJECTS = [
    ("brush", "blue_brush"),
    ("brush", "red_brush"),
    ("eraser", "flat_eraser"),
    ("eraser", "handle_eraser"),
    ("hammer", "claw_hammer"),
    ("hammer", "mallet_hammer"),
    ("marker", "sharpie_marker"),
    ("marker", "staples_marker"),
    ("screwdriver", "long_screwdriver"),
    ("screwdriver", "short_screwdriver"),
    ("spatula", "flat_spatula"),
    ("spatula", "spoon_spatula"),
]


DOMINO20_OBJECTS = [
    {"label": "mug", "asset_modelname": "039_mug", "asset_model_id": 0},
    {"label": "cup", "asset_modelname": "021_cup", "asset_model_id": 0},
    {"label": "bottle", "asset_modelname": "001_bottle", "asset_model_id": 13},
    {"label": "drill", "asset_modelname": "030_drill", "asset_model_id": 0},
    {"label": "pill_bottle", "asset_modelname": "080_pillbottle", "asset_model_id": 1},
    {"label": "milk_box", "asset_modelname": "038_milk-box", "asset_model_id": 0},
    {"label": "can", "asset_modelname": "071_can", "asset_model_id": 0},
    {"label": "milk_tea", "asset_modelname": "101_milk-tea", "asset_model_id": 0},
    {"label": "sauce_can", "asset_modelname": "105_sauce-can", "asset_model_id": 0},
    {"label": "tea_box", "asset_modelname": "112_tea-box", "asset_model_id": 0},
    {"label": "bowl", "asset_modelname": "002_bowl", "asset_model_id": 4},
    {"label": "plate", "asset_modelname": "003_plate", "asset_model_id": 0},
    {"label": "hammer", "asset_modelname": "020_hammer", "asset_model_id": 0},
    {"label": "screwdriver", "asset_modelname": "032_screwdriver", "asset_model_id": 0},
    {"label": "apple", "asset_modelname": "035_apple", "asset_model_id": 0},
    {"label": "book", "asset_modelname": "043_book", "asset_model_id": 0},
    {"label": "mouse", "asset_modelname": "047_mouse", "asset_model_id": 0},
    {"label": "stapler", "asset_modelname": "048_stapler", "asset_model_id": 0},
    {"label": "dumbbell", "asset_modelname": "052_dumbbell", "asset_model_id": 0},
    {"label": "kettle", "asset_modelname": "091_kettle", "asset_model_id": 0},
]


PART_HINTS = {
    "brush": {
        "parts": ["handle", "bristles", "head"],
        "positive_grasp": "handle",
        "negative_grasp": "bristles",
        "tool_use_surface": "bristles",
    },
    "eraser": {
        "parts": ["handle", "eraser_body", "erasing_surface"],
        "positive_grasp": "handle or side body",
        "negative_grasp": "erasing_surface",
        "tool_use_surface": "erasing_surface",
    },
    "hammer": {
        "parts": ["handle", "head", "claw", "striking_face"],
        "positive_grasp": "handle",
        "negative_grasp": "head or claw",
        "tool_use_surface": "striking_face or claw",
    },
    "marker": {
        "parts": ["barrel", "cap", "tip"],
        "positive_grasp": "barrel",
        "negative_grasp": "tip",
        "tool_use_surface": "tip",
    },
    "screwdriver": {
        "parts": ["handle", "shaft", "tip"],
        "positive_grasp": "handle",
        "negative_grasp": "shaft or tip",
        "tool_use_surface": "tip",
    },
    "spatula": {
        "parts": ["handle", "blade", "scoop"],
        "positive_grasp": "handle",
        "negative_grasp": "blade or scoop",
        "tool_use_surface": "blade or scoop",
    },
}


DOMINO_PART_HINTS = {
    "mug": ["handle", "rim", "body"],
    "cup": ["rim", "body", "bottom"],
    "bottle": ["cap", "neck", "body"],
    "drill": ["handle", "trigger", "chuck", "body"],
    "pill_bottle": ["cap", "body"],
    "milk_box": ["body", "top"],
    "can": ["body", "top", "bottom"],
    "milk_tea": ["cap", "body"],
    "sauce_can": ["body", "top", "bottom"],
    "tea_box": ["body", "lid"],
    "bowl": ["rim", "inner_surface", "outer_surface", "bottom"],
    "plate": ["rim", "center", "bottom"],
    "hammer": ["handle", "head", "striking_face"],
    "screwdriver": ["handle", "shaft", "tip"],
    "apple": ["body", "stem"],
    "book": ["cover", "spine", "pages"],
    "mouse": ["body", "buttons", "scroll_wheel"],
    "stapler": ["body", "base", "metal_tip"],
    "dumbbell": ["handle", "weights"],
    "kettle": ["handle", "spout", "body", "lid"],
}


DOMINO_AFFORDANCE_HINTS = {
    "mug": {
        "positive_grasp": "handle",
        "negative_grasp": "rim",
        "tool_use_surface": "rim",
    },
    "cup": {
        "positive_grasp": "body",
        "negative_grasp": "rim",
        "tool_use_surface": "rim",
    },
    "bottle": {
        "positive_grasp": "body",
        "negative_grasp": "cap or neck",
        "tool_use_surface": "cap",
    },
    "drill": {
        "positive_grasp": "handle",
        "negative_grasp": "trigger or chuck",
        "tool_use_surface": "chuck",
    },
    "pill_bottle": {
        "positive_grasp": "body",
        "negative_grasp": "cap",
        "tool_use_surface": "cap",
    },
    "milk_box": {
        "positive_grasp": "body",
        "negative_grasp": "top",
        "tool_use_surface": "top",
    },
    "can": {
        "positive_grasp": "body",
        "negative_grasp": "top",
        "tool_use_surface": "top",
    },
    "milk_tea": {
        "positive_grasp": "body or cup",
        "negative_grasp": "cap",
        "tool_use_surface": "cap",
    },
    "sauce_can": {
        "positive_grasp": "body",
        "negative_grasp": "top",
        "tool_use_surface": "top",
    },
    "tea_box": {
        "positive_grasp": "body or box",
        "negative_grasp": "lid",
        "tool_use_surface": "lid",
    },
    "bowl": {
        "positive_grasp": "outer surface",
        "negative_grasp": "rim or inner surface",
        "tool_use_surface": "inner surface",
    },
    "plate": {
        "positive_grasp": "rim",
        "negative_grasp": "center",
        "tool_use_surface": "center",
    },
    "hammer": {
        "positive_grasp": "handle",
        "negative_grasp": "head or striking face",
        "tool_use_surface": "striking face",
    },
    "screwdriver": {
        "positive_grasp": "handle",
        "negative_grasp": "shaft or tip",
        "tool_use_surface": "tip",
    },
    "apple": {
        "positive_grasp": "body",
        "negative_grasp": "stem or top",
        "tool_use_surface": "body",
    },
    "book": {
        "positive_grasp": "spine or cover",
        "negative_grasp": "pages or page edges",
        "tool_use_surface": "pages or page edges",
    },
    "mouse": {
        "positive_grasp": "body",
        "negative_grasp": "buttons or scroll wheel",
        "tool_use_surface": "buttons",
    },
    "stapler": {
        "positive_grasp": "body or stapler",
        "negative_grasp": "metal tip or base",
        "tool_use_surface": "base or metal tip",
    },
    "dumbbell": {
        "positive_grasp": "handle",
        "negative_grasp": "weights",
        "tool_use_surface": "weights",
    },
    "kettle": {
        "positive_grasp": "handle",
        "negative_grasp": "spout or lid",
        "tool_use_surface": "spout",
    },
}


AFFORDANCE_VOCAB = [
    "positive_grasp",
    "negative_grasp",
    "tool_use_surface",
    "safe_contact",
]


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _obj_stats(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.suffix.lower() != ".obj":
        return {}

    vertex_count = 0
    face_count = 0
    min_xyz = [float("inf"), float("inf"), float("inf")]
    max_xyz = [float("-inf"), float("-inf"), float("-inf")]
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                fields = line.split()
                if len(fields) < 4:
                    continue
                xyz = [float(fields[1]), float(fields[2]), float(fields[3])]
                vertex_count += 1
                for idx, value in enumerate(xyz):
                    min_xyz[idx] = min(min_xyz[idx], value)
                    max_xyz[idx] = max(max_xyz[idx], value)
            elif line.startswith("f "):
                face_count += 1

    stats: Dict[str, Any] = {
        "vertex_count": vertex_count,
        "face_count": face_count,
    }
    if vertex_count > 0:
        stats["bounds_min"] = min_xyz
        stats["bounds_max"] = max_xyz
        stats["extent"] = [max_xyz[idx] - min_xyz[idx] for idx in range(3)]
    return stats


def _file_info(path: Path) -> Dict[str, Any]:
    info = {
        "exists": path.exists(),
        "path": str(path),
    }
    if path.exists() and path.is_file():
        info["size_bytes"] = path.stat().st_size
    return info


def _dextoolbench_entries(repo_root: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for category, name in DEXTOOLBENCH_OBJECTS:
        object_dir = repo_root / "assets" / "urdf" / "dextoolbench" / category / name
        mesh_path = object_dir / f"{name}.obj"
        urdf_path = object_dir / f"{name}.urdf"
        label_dir = (
            repo_root
            / "assets"
            / "affordance_labels"
            / "dextoolbench"
            / category
            / name
        )
        hint = PART_HINTS[category]
        entries.append(
            {
                "stage": "dextoolbench12",
                "source": "dextoolbench",
                "asset_id": f"dextoolbench/{category}/{name}",
                "label": name,
                "category": category,
                "urdf_path": _relative_or_absolute(urdf_path, repo_root),
                "visual_mesh_path": _relative_or_absolute(mesh_path, repo_root),
                "collision_mesh_path": _relative_or_absolute(mesh_path, repo_root),
                "annotation_dir": _relative_or_absolute(label_dir, repo_root),
                "part_vocab": hint["parts"],
                "affordance_vocab": AFFORDANCE_VOCAB,
                "suggested_prompts": {
                    "positive_grasp": f"{name.replace('_', ' ')} {hint['positive_grasp']}",
                    "negative_grasp": f"{name.replace('_', ' ')} {hint['negative_grasp']}",
                    "tool_use_surface": f"{name.replace('_', ' ')} {hint['tool_use_surface']}",
                },
                "files": {
                    "urdf": _file_info(urdf_path),
                    "visual_mesh": _file_info(mesh_path),
                },
                "mesh_stats": _obj_stats(mesh_path),
            }
        )
    return entries


def _domino20_entries(repo_root: Path, domino_root: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for item in DOMINO20_OBJECTS:
        label = item["label"]
        modelname = item["asset_modelname"]
        model_id = item["asset_model_id"]
        visual_mesh = domino_root / modelname / "visual" / f"base{model_id}.glb"
        collision_mesh = domino_root / modelname / "collision" / f"base{model_id}.glb"
        label_dir = repo_root / "assets" / "affordance_labels" / "domino20" / label
        hint = DOMINO_AFFORDANCE_HINTS.get(label, {})
        prompt_label = label.replace("_", " ")
        entries.append(
            {
                "stage": "domino20",
                "source": "domino20",
                "asset_id": f"domino20/{label}",
                "label": label,
                "asset_modelname": modelname,
                "asset_model_id": model_id,
                "visual_mesh_path": str(visual_mesh.resolve()),
                "collision_mesh_path": str(collision_mesh.resolve()),
                "annotation_dir": _relative_or_absolute(label_dir, repo_root),
                "part_vocab": DOMINO_PART_HINTS.get(label, []),
                "affordance_vocab": AFFORDANCE_VOCAB,
                "suggested_prompts": {
                    affordance: f"{prompt_label} {part_hint}"
                    for affordance, part_hint in hint.items()
                },
                "files": {
                    "visual_mesh": _file_info(visual_mesh),
                    "collision_mesh": _file_info(collision_mesh),
                },
            }
        )
    return entries


def _filter_entries(entries: Iterable[Dict[str, Any]], stage: str) -> List[Dict[str, Any]]:
    if stage == "all":
        return list(entries)
    return [entry for entry in entries if entry["stage"] == stage]


def _missing_files(entries: Iterable[Dict[str, Any]]) -> List[str]:
    missing = []
    for entry in entries:
        for key, info in entry.get("files", {}).items():
            if not info.get("exists", False):
                missing.append(f"{entry['asset_id']}:{key}:{info.get('path')}")
    return missing


def build_manifest(repo_root: Path, domino_root: Path, stage: str) -> Dict[str, Any]:
    all_entries = _dextoolbench_entries(repo_root) + _domino20_entries(
        repo_root, domino_root
    )
    entries = _filter_entries(all_entries, stage)
    missing = _missing_files(entries)
    return {
        "schema_version": 1,
        "description": "Offline affordance annotation manifest. DextoolBench 12 is stage 1; DOMINO20 is stage 2.",
        "repo_root": str(repo_root.resolve()),
        "domino_root": str(domino_root.resolve()),
        "stage": stage,
        "num_assets": len(entries),
        "missing_files": missing,
        "assets": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["dextoolbench12", "domino20", "all"],
        default="all",
        help="Subset of assets to include.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="simtoolreal repository root.",
    )
    parser.add_argument(
        "--domino-root",
        type=Path,
        default=Path("/data1/linsixu/DOMINO/assets/objects"),
        help="DOMINO/RoboTwin object asset root used by the current training config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "assets" / "affordance_labels" / "asset_manifest.json",
        help="Output manifest JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    domino_root = args.domino_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = repo_root / output

    manifest = build_manifest(repo_root=repo_root, domino_root=domino_root, stage=args.stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    by_stage: Dict[str, int] = {}
    for entry in manifest["assets"]:
        by_stage[entry["stage"]] = by_stage.get(entry["stage"], 0) + 1

    print(f"Wrote {manifest['num_assets']} assets to {output}")
    print(f"Stages: {by_stage}")
    if manifest["missing_files"]:
        print("Missing files:")
        for item in manifest["missing_files"]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
