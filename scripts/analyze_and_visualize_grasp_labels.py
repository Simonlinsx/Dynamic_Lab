#!/usr/bin/env python
"""Analyze and visualize cleaned grasp affordance labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-simtoolreal")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "assets" / "affordance_labels" / "asset_manifest.json"


def _resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _entries_for_stage(manifest: Mapping[str, Any], stage: str) -> List[Mapping[str, Any]]:
    return [
        entry
        for entry in manifest["assets"]
        if stage == "all" or str(entry.get("stage")) == stage
    ]


def _sample(points: np.ndarray, labels: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], labels[idx]


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


def _colors(labels: np.ndarray) -> np.ndarray:
    colors = np.tile(np.asarray([[0.62, 0.62, 0.62]], dtype=np.float32), (len(labels), 1))
    colors[labels == 0] = np.asarray([0.90, 0.20, 0.18], dtype=np.float32)
    colors[labels == 1] = np.asarray([0.12, 0.55, 0.90], dtype=np.float32)
    return colors


def _analyze_entry(entry: Mapping[str, Any], repo_root: Path, label_file: str) -> Dict[str, Any]:
    label_dir = _resolve_path(str(entry["annotation_dir"]), repo_root)
    label_path = label_dir / label_file
    result: Dict[str, Any] = {
        "asset_id": entry["asset_id"],
        "label": entry["label"],
        "stage": entry["stage"],
        "annotation_path": str(label_path),
        "status": "missing",
    }
    if not label_path.exists():
        return result
    data = np.load(label_path, allow_pickle=True)
    labels = np.asarray(data["grasp_label"], dtype=np.int8)
    total = int(len(labels))
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    ign = int((labels < 0).sum())
    warnings = []
    if pos < 50:
        warnings.append("few_positive_vertices")
    if neg < 50:
        warnings.append("few_negative_vertices")
    if ign / max(total, 1) > 0.85:
        warnings.append("mostly_ignore")
    result.update(
        {
            "status": "labeled",
            "num_vertices": total,
            "positive_vertices": pos,
            "negative_vertices": neg,
            "ignore_vertices": ign,
            "positive_fraction": pos / total if total else 0.0,
            "negative_fraction": neg / total if total else 0.0,
            "ignore_fraction": ign / total if total else 0.0,
            "warnings": warnings,
        }
    )
    return result


def _plot(entries: List[Mapping[str, Any]], repo_root: Path, label_file: str, output_path: Path, max_points: int) -> None:
    cols = 4
    rows = max(1, ceil(len(entries) / cols))
    fig = plt.figure(figsize=(cols * 4.1, rows * 3.9))
    for idx, entry in enumerate(entries):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        label_path = _resolve_path(str(entry["annotation_dir"]), repo_root) / label_file
        if label_path.exists():
            data = np.load(label_path, allow_pickle=True)
            points = np.asarray(data["vertices_obj"], dtype=np.float32)
            labels = np.asarray(data["grasp_label"], dtype=np.int8)
            points, labels = _sample(points, labels, max_points=max_points, seed=20260615 + idx)
            if len(points) > 0:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=_colors(labels), s=1.0, linewidths=0, alpha=0.88)
                _equalize_axes(ax, points)
            status = "labeled"
        else:
            ax.text2D(0.20, 0.50, "missing", transform=ax.transAxes, fontsize=8)
            status = "missing"
        ax.view_init(elev=24, azim=-54)
        ax.set_title(f"{entry['label']}\n{status}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    fig.suptitle(f"{Path(label_file).stem} Labels", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_csv(path: Path, assets: List[Mapping[str, Any]]) -> None:
    fields = [
        "asset_id",
        "label",
        "status",
        "num_vertices",
        "positive_vertices",
        "negative_vertices",
        "ignore_vertices",
        "positive_fraction",
        "negative_fraction",
        "ignore_fraction",
        "warnings",
        "annotation_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in assets:
            writer.writerow(
                {
                    "asset_id": item.get("asset_id", ""),
                    "label": item.get("label", ""),
                    "status": item.get("status", ""),
                    "num_vertices": item.get("num_vertices", 0),
                    "positive_vertices": item.get("positive_vertices", 0),
                    "negative_vertices": item.get("negative_vertices", 0),
                    "ignore_vertices": item.get("ignore_vertices", 0),
                    "positive_fraction": f"{float(item.get('positive_fraction', 0.0)):.6f}",
                    "negative_fraction": f"{float(item.get('negative_fraction', 0.0)):.6f}",
                    "ignore_fraction": f"{float(item.get('ignore_fraction', 0.0)):.6f}",
                    "warnings": ";".join(item.get("warnings", [])),
                    "annotation_path": item.get("annotation_path", ""),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", choices=["dextoolbench12", "domino20", "all"], default="dextoolbench12")
    parser.add_argument("--label-file", default="grasp_affordance_clean_v2.npz")
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "assets" / "affordance_labels" / "visualizations")
    parser.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "assets" / "affordance_labels" / "analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = Path(manifest.get("repo_root", REPO_ROOT)).resolve()
    entries = _entries_for_stage(manifest, args.stage)
    assets = [_analyze_entry(entry, repo_root, args.label_file) for entry in entries]
    labeled = [item for item in assets if item["status"] == "labeled"]
    aggregate = {
        "num_assets": len(assets),
        "num_labeled": len(labeled),
        "num_missing": len(assets) - len(labeled),
        "total_vertices": int(sum(item.get("num_vertices", 0) for item in labeled)),
        "total_positive_vertices": int(sum(item.get("positive_vertices", 0) for item in labeled)),
        "total_negative_vertices": int(sum(item.get("negative_vertices", 0) for item in labeled)),
        "total_ignore_vertices": int(sum(item.get("ignore_vertices", 0) for item in labeled)),
        "warning_counts": {},
    }
    for item in assets:
        for warning in item.get("warnings", []):
            aggregate["warning_counts"][warning] = aggregate["warning_counts"].get(warning, 0) + 1
    stem = f"{args.stage}_{Path(args.label_file).stem}"
    image_path = args.output_dir.resolve() / f"{stem}_overview.png"
    json_path = args.analysis_dir.resolve() / f"{stem}_analysis.json"
    csv_path = args.analysis_dir.resolve() / f"{stem}_analysis.csv"
    _plot(entries, repo_root, args.label_file, image_path, max_points=args.max_points)
    report = {
        "manifest_path": str(manifest_path),
        "stage": args.stage,
        "label_file": args.label_file,
        "image_path": str(image_path),
        "aggregate": aggregate,
        "assets": assets,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(csv_path, assets)
    print(json.dumps({"image_path": str(image_path), "json_path": str(json_path), "csv_path": str(csv_path), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
