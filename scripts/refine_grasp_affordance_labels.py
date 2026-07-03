#!/usr/bin/env python
"""Refine weak SAM3 affordance labels with lightweight geometry priors.

The SAM3 pass is useful for finding candidate parts, but text-only 2D masks can
over-label entire elongated objects. This script keeps the original clean labels
intact and writes a derived label file that uses simple mesh geometry to suppress
common false positives such as screwdriver shafts, marker tips, and tool heads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "assets" / "affordance_labels" / "asset_manifest.json"


def _resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _entries_for_args(
    manifest: Mapping[str, Any],
    stage: str,
    asset_ids: Sequence[str],
) -> List[Mapping[str, Any]]:
    entries = [
        entry
        for entry in manifest["assets"]
        if stage == "all" or str(entry.get("stage")) == stage
    ]
    if asset_ids:
        wanted = set(asset_ids)
        entries = [entry for entry in entries if entry.get("asset_id") in wanted or entry.get("label") in wanted]
    return entries


def _pca_frame(vertices: np.ndarray) -> Dict[str, np.ndarray | int]:
    points = np.asarray(vertices, dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh.T
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    dims = np.maximum(maxs - mins, 1e-9)
    mid = (mins + maxs) / 2.0
    norm = (coords - mid) / (dims / 2.0)
    long_axis = int(np.argmax(dims))
    short_axis = int(np.argmin(dims))
    other_axes = [idx for idx in range(3) if idx != long_axis]
    t = (coords[:, long_axis] - mins[long_axis]) / dims[long_axis]
    return {
        "axes": vh,
        "coords": coords,
        "dims": dims,
        "norm": norm,
        "long_axis": long_axis,
        "short_axis": short_axis,
        "other_axes": np.asarray(other_axes, dtype=np.int32),
        "long_t": t,
    }


def _smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values
    padded = np.pad(values, (1, 1), mode="edge")
    return 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]


def _width_profile(
    frame: Mapping[str, Any],
    num_bins: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(frame["coords"], dtype=np.float64)
    t = np.asarray(frame["long_t"], dtype=np.float64)
    long_axis = int(frame["long_axis"])
    minor_axes = [idx for idx in range(3) if idx != long_axis]
    minor = coords[:, minor_axes]
    bin_idx = np.clip((t * num_bins).astype(np.int32), 0, num_bins - 1)
    widths = np.zeros((num_bins,), dtype=np.float64)
    counts = np.zeros((num_bins,), dtype=np.int32)
    for idx in range(num_bins):
        mask = bin_idx == idx
        counts[idx] = int(mask.sum())
        if counts[idx] < 4:
            continue
        lo = np.percentile(minor[mask], 5, axis=0)
        hi = np.percentile(minor[mask], 95, axis=0)
        widths[idx] = float(np.linalg.norm(hi - lo))
    if widths.max() > 0:
        widths = _smooth(widths / widths.max())
    return widths[bin_idx], widths, counts


def _ensure_some_positive(
    old_positive: np.ndarray,
    new_positive: np.ndarray,
    score: np.ndarray,
    min_keep_fraction: float,
) -> np.ndarray:
    old_count = int(old_positive.sum())
    if old_count == 0 or new_positive.sum() >= max(20, int(round(old_count * min_keep_fraction))):
        return new_positive
    candidates = np.flatnonzero(old_positive)
    keep = max(20, int(round(old_count * min_keep_fraction)))
    order = candidates[np.argsort(score[candidates])[::-1][:keep]]
    repaired = new_positive.copy()
    repaired[order] = True
    return repaired


def _labels_from_masks(
    num_vertices: int,
    positive: np.ndarray,
    negative: np.ndarray,
) -> np.ndarray:
    labels = np.full((num_vertices,), -1, dtype=np.int8)
    negative = negative & ~positive
    labels[negative] = 0
    labels[positive] = 1
    return labels


def _refine_screwdriver(
    positive: np.ndarray,
    negative: np.ndarray,
    score: np.ndarray,
    frame: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    width_at_vertex, widths, counts = _width_profile(frame)
    handle_candidate = width_at_vertex >= 0.42
    shaft_candidate = width_at_vertex <= 0.30
    refined_positive = positive & handle_candidate
    refined_positive = _ensure_some_positive(positive, refined_positive, score, min_keep_fraction=0.35)
    refined_negative = negative | (positive & shaft_candidate)
    return refined_positive, refined_negative, {
        "rule": "screwdriver_width_handle",
        "handle_width_threshold": 0.42,
        "shaft_width_threshold": 0.30,
        "width_profile": widths.tolist(),
        "bin_counts": counts.tolist(),
    }


def _refine_marker(
    positive: np.ndarray,
    negative: np.ndarray,
    frame: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    t = np.asarray(frame["long_t"], dtype=np.float64)
    positive_band = (t >= 0.12) & (t <= 0.88)
    negative_band = (t <= 0.08) | (t >= 0.92)
    refined_positive = positive & positive_band
    refined_negative = negative | negative_band
    return refined_positive, refined_negative, {
        "rule": "marker_mid_body_with_negative_ends",
        "positive_long_axis_range": [0.12, 0.88],
        "negative_long_axis_ranges": [[0.0, 0.08], [0.92, 1.0]],
    }


def _refine_handle_tool(
    positive: np.ndarray,
    negative: np.ndarray,
    score: np.ndarray,
    frame: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    width_at_vertex, widths, counts = _width_profile(frame)
    head_candidate = width_at_vertex >= 0.62
    handle_candidate = width_at_vertex <= 0.58
    refined_positive = handle_candidate
    refined_positive = _ensure_some_positive(positive, refined_positive, score, min_keep_fraction=0.45)
    refined_negative = head_candidate
    return refined_positive, refined_negative, {
        "rule": "hard_narrow_handle_wide_head_tool",
        "handle_width_threshold": 0.58,
        "head_width_threshold": 0.62,
        "width_profile": widths.tolist(),
        "bin_counts": counts.tolist(),
    }


def _refine_flat_eraser(
    positive: np.ndarray,
    negative: np.ndarray,
    frame: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    refined_positive = np.ones_like(positive, dtype=bool)
    refined_negative = np.zeros_like(negative, dtype=bool)
    return refined_positive, refined_negative, {
        "rule": "flat_eraser_all_positive_graspable_body",
    }


def _refine_entry_masks(
    entry: Mapping[str, Any],
    vertices: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    category = str(entry.get("category", "")).lower()
    label = str(entry.get("label", "")).lower()
    frame = _pca_frame(vertices)
    info: Dict[str, Any] = {
        "category": category,
        "label": label,
        "pca_dims": np.asarray(frame["dims"], dtype=np.float64).tolist(),
        "long_axis": int(frame["long_axis"]),
        "short_axis": int(frame["short_axis"]),
    }
    if category == "screwdriver" or "screwdriver" in label:
        refined_positive, refined_negative, rule_info = _refine_screwdriver(positive, negative, score, frame)
    elif category == "marker" or "marker" in label:
        refined_positive, refined_negative, rule_info = _refine_marker(positive, negative, frame)
    elif label == "flat_eraser":
        refined_positive, refined_negative, rule_info = _refine_flat_eraser(positive, negative, frame)
    elif category in {"brush", "spatula", "hammer"} or any(token in label for token in ("brush", "spatula", "hammer")):
        refined_positive, refined_negative, rule_info = _refine_handle_tool(positive, negative, score, frame)
    else:
        refined_positive, refined_negative, rule_info = positive.copy(), negative.copy(), {"rule": "identity"}
    info.update(rule_info)
    return refined_positive, refined_negative, info


def _copy_optional(data: Mapping[str, Any], key: str) -> np.ndarray:
    if key in data:
        return np.asarray(data[key])
    return np.asarray([], dtype=np.float32)


def refine_entry(
    entry: Mapping[str, Any],
    repo_root: Path,
    input_name: str,
    output_name: str,
) -> Dict[str, Any]:
    annotation_dir = _resolve_path(str(entry["annotation_dir"]), repo_root)
    input_path = annotation_dir / input_name
    output_path = annotation_dir / output_name
    result: Dict[str, Any] = {
        "asset_id": entry["asset_id"],
        "label": entry.get("label"),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "missing_input",
    }
    if not input_path.exists():
        return result

    data = np.load(input_path, allow_pickle=True)
    vertices = np.asarray(data["vertices_obj"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int32)
    old_labels = np.asarray(data["grasp_label"], dtype=np.int8)
    score = np.asarray(data["positive_score"], dtype=np.float32) if "positive_score" in data else np.zeros(len(vertices), dtype=np.float32)
    negative_score = (
        np.asarray(data["negative_score"], dtype=np.float32)
        if "negative_score" in data
        else np.zeros(len(vertices), dtype=np.float32)
    )
    positive = old_labels == 1
    negative = old_labels == 0
    refined_positive, refined_negative, info = _refine_entry_masks(entry, vertices, positive, negative, score)
    refined_labels = _labels_from_masks(len(vertices), refined_positive, refined_negative)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        vertices_obj=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        grasp_label=refined_labels.astype(np.int8),
        positive_mask=(refined_labels == 1).astype(np.uint8),
        negative_mask=(refined_labels == 0).astype(np.uint8),
        ignore_mask=(refined_labels < 0).astype(np.uint8),
        positive_score=score.astype(np.float32),
        negative_score=negative_score.astype(np.float32),
        original_grasp_label=old_labels.astype(np.int8),
        label_vocab=np.asarray(["ignore", "negative_grasp", "positive_grasp"], dtype=object),
        source=np.asarray(["geometry_refined_from_positive_only_clean_v2"], dtype=object),
        refinement_info=np.asarray([json.dumps(info, sort_keys=True)], dtype=object),
    )
    result.update(
        {
            "status": "refined",
            "rule": info.get("rule"),
            "old_positive_vertices": int(positive.sum()),
            "old_negative_vertices": int(negative.sum()),
            "old_ignore_vertices": int((old_labels < 0).sum()),
            "positive_vertices": int((refined_labels == 1).sum()),
            "negative_vertices": int((refined_labels == 0).sum()),
            "ignore_vertices": int((refined_labels < 0).sum()),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", choices=["dextoolbench12", "domino20", "all"], default="dextoolbench12")
    parser.add_argument("--asset-id", action="append", default=[], help="Filter by asset_id or label. Can be repeated.")
    parser.add_argument("--input-name", default="grasp_affordance_clean_v2.npz")
    parser.add_argument("--output-name", default="grasp_affordance_refined_v2.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = Path(manifest.get("repo_root", REPO_ROOT)).resolve()
    entries = _entries_for_args(manifest, args.stage, args.asset_id)
    results = [refine_entry(entry, repo_root, args.input_name, args.output_name) for entry in entries]
    summary = {
        "manifest_path": str(manifest_path),
        "stage": args.stage,
        "input_name": args.input_name,
        "output_name": args.output_name,
        "num_requested": len(entries),
        "num_refined": int(sum(item["status"] == "refined" for item in results)),
        "results": results,
    }
    summary_dir = repo_root / "assets" / "affordance_labels" / "analysis"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{args.stage}_{Path(args.output_name).stem}_refinement_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "num_refined": summary["num_refined"]}, indent=2))


if __name__ == "__main__":
    main()
