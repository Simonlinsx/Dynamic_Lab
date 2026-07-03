#!/usr/bin/env python
"""Visualize affordance annotation coverage for a manifest.

If an object's annotation_dir contains affordance.npz, the script visualizes
the stored labels. Otherwise it renders an unlabeled mesh preview so the asset
can still be inspected before annotation.
"""

from __future__ import annotations

import argparse
import json
import os
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-simtoolreal")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


AFFORDANCE_COLORS = np.asarray(
    [
        [0.55, 0.55, 0.55],
        [0.12, 0.55, 0.90],
        [0.90, 0.20, 0.18],
        [0.10, 0.72, 0.36],
        [0.95, 0.68, 0.18],
        [0.56, 0.32, 0.76],
        [0.15, 0.65, 0.65],
    ],
    dtype=np.float32,
)


def _resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_obj_vertices(path: Path) -> np.ndarray:
    vertices: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) < 4:
                continue
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
    return np.asarray(vertices, dtype=np.float32)


def _load_label_points(label_path: Path) -> Tuple[np.ndarray, np.ndarray | None, str]:
    data = np.load(label_path, allow_pickle=True)
    points = None
    for key in ("points_obj", "vertices_obj", "points"):
        if key in data:
            points = np.asarray(data[key], dtype=np.float32)
            break
    if points is None:
        raise KeyError(f"{label_path} does not contain points_obj/vertices_obj/points")

    labels = None
    label_kind = "unlabeled"
    if "affordance_multihot" in data:
        multihot = np.asarray(data["affordance_multihot"], dtype=bool)
        labels = np.zeros((len(points),), dtype=np.int64)
        if multihot.ndim == 2 and multihot.shape[0] == len(points):
            active_count = multihot.sum(axis=1)
            single = active_count == 1
            labels[single] = multihot[single].argmax(axis=1) + 1
            labels[active_count > 1] = min(5, len(AFFORDANCE_COLORS) - 1)
            return points, labels, "affordance_multihot_overlap"
    for key in ("affordance_id", "affordance_label", "part_id", "part_label"):
        if key in data:
            labels = np.asarray(data[key]).reshape(-1)
            label_kind = key
            break
    return points, labels, label_kind


def _sample_points(
    points: np.ndarray,
    labels: np.ndarray | None,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray | None]:
    if len(points) <= max_points:
        return points, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    sampled_labels = labels[idx] if labels is not None and len(labels) == len(points) else labels
    return points[idx], sampled_labels


def _equalize_axes(ax, points: np.ndarray) -> None:
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-4)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _colors_for_labels(labels: np.ndarray | None, count: int) -> np.ndarray:
    if labels is None or len(labels) != count:
        return np.tile(np.asarray([[0.52, 0.52, 0.52]], dtype=np.float32), (count, 1))
    label_values = np.asarray(labels, dtype=np.int64)
    color_idx = np.mod(label_values, len(AFFORDANCE_COLORS))
    return AFFORDANCE_COLORS[color_idx]


def _entry_points(entry: Dict[str, Any], repo_root: Path, max_points: int, seed: int):
    label_dir = _resolve_path(entry["annotation_dir"], repo_root)
    label_path = label_dir / "affordance.npz"
    if label_path.exists():
        points, labels, label_kind = _load_label_points(label_path)
        points, labels = _sample_points(points, labels, max_points=max_points, seed=seed)
        return points, labels, label_kind, label_path

    mesh_path = _resolve_path(entry["visual_mesh_path"], repo_root)
    if mesh_path.suffix.lower() != ".obj":
        return np.zeros((0, 3), dtype=np.float32), None, "mesh_preview_unavailable", label_path
    points = _load_obj_vertices(mesh_path)
    points, labels = _sample_points(points, None, max_points=max_points, seed=seed)
    return points, labels, "unlabeled_mesh", label_path


def _plot_overview(
    entries: List[Dict[str, Any]],
    repo_root: Path,
    output_path: Path,
    max_points: int,
) -> List[Dict[str, Any]]:
    cols = 4
    rows = max(1, ceil(len(entries) / cols))
    fig = plt.figure(figsize=(cols * 4.1, rows * 3.9))
    summary: List[Dict[str, Any]] = []

    for idx, entry in enumerate(entries):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        points, labels, label_kind, label_path = _entry_points(
            entry, repo_root=repo_root, max_points=max_points, seed=20260608 + idx
        )
        status = "labeled" if label_path.exists() and label_kind != "unlabeled_mesh" else "unlabeled"
        if len(points) > 0:
            colors = _colors_for_labels(labels, len(points))
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors,
                s=1.0,
                linewidths=0,
                alpha=0.88,
            )
            _equalize_axes(ax, points)
        else:
            ax.text2D(0.18, 0.50, "preview unavailable", transform=ax.transAxes, fontsize=8)

        ax.view_init(elev=24, azim=-54)
        ax.set_title(f"{entry['label']}\n{status}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        summary.append(
            {
                "asset_id": entry["asset_id"],
                "label": entry["label"],
                "status": status,
                "label_kind": label_kind,
                "annotation_path": str(label_path),
                "num_visualized_points": int(len(points)),
            }
        )

    for idx in range(len(entries), rows * cols):
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.axis("off")

    fig.suptitle("Affordance Annotation Overview", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "assets" / "affordance_labels" / "asset_manifest.json",
        help="Manifest produced by build_affordance_asset_manifest.py.",
    )
    parser.add_argument(
        "--stage",
        choices=["dextoolbench12", "domino20", "all"],
        default="dextoolbench12",
        help="Manifest stage to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "affordance_labels" / "visualizations",
        help="Directory for overview images and summary JSON.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Maximum points per object in the overview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = Path(manifest.get("repo_root", REPO_ROOT)).resolve()
    entries = [
        entry
        for entry in manifest["assets"]
        if args.stage == "all" or entry.get("stage") == args.stage
    ]
    if not entries:
        raise ValueError(f"No entries found for stage {args.stage!r} in {manifest_path}")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    image_path = output_dir / f"{args.stage}_affordance_overview.png"
    summary_path = output_dir / f"{args.stage}_affordance_overview.json"

    summary_entries = _plot_overview(
        entries=entries,
        repo_root=repo_root,
        output_path=image_path,
        max_points=max(1, args.max_points),
    )
    labeled_count = sum(1 for item in summary_entries if item["status"] == "labeled")
    summary = {
        "manifest_path": str(manifest_path),
        "stage": args.stage,
        "num_assets": len(entries),
        "labeled_count": labeled_count,
        "unlabeled_count": len(entries) - labeled_count,
        "image_path": str(image_path),
        "assets": summary_entries,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote overview image: {image_path}")
    print(f"Wrote overview summary: {summary_path}")
    print(f"Labeled assets: {labeled_count}/{len(entries)}")


if __name__ == "__main__":
    main()
