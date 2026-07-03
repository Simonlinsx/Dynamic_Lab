#!/usr/bin/env python
"""Build cleaned binary grasp affordance labels from SAM3 multiview annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

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


def _build_entry(
    entry: Mapping[str, Any],
    repo_root: Path,
    min_vote_pixels: float,
    margin: float,
    include_tool_as_negative: bool,
    output_name: str,
) -> Dict[str, Any]:
    annotation_dir = _resolve_path(str(entry["annotation_dir"]), repo_root)
    source_path = annotation_dir / "affordance.npz"
    output_path = annotation_dir / output_name
    result: Dict[str, Any] = {
        "asset_id": entry["asset_id"],
        "label": entry["label"],
        "stage": entry["stage"],
        "source_path": str(source_path),
        "output_path": str(output_path),
        "status": "missing_source",
    }
    if not source_path.exists():
        return result

    data = np.load(source_path, allow_pickle=True)
    vertices = np.asarray(data["vertices_obj"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int32)
    scores = np.asarray(data["affordance_scores"], dtype=np.float32)
    positive_score = scores[:, 0]
    negative_score = scores[:, 1].copy()
    if include_tool_as_negative and scores.shape[1] >= 3:
        negative_score = np.maximum(negative_score, scores[:, 2])

    positive_mask = (
        (positive_score >= min_vote_pixels)
        & (positive_score > negative_score * margin)
    )
    negative_mask = (
        (negative_score >= min_vote_pixels)
        & (negative_score > positive_score * margin)
    )
    label = np.full((len(vertices),), -1, dtype=np.int8)
    label[negative_mask] = 0
    label[positive_mask] = 1
    conflict_mask = (
        (positive_score >= min_vote_pixels)
        & (negative_score >= min_vote_pixels)
        & ~(positive_mask | negative_mask)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        vertices_obj=vertices,
        faces=faces,
        grasp_label=label,
        positive_mask=positive_mask.astype(np.uint8),
        negative_mask=negative_mask.astype(np.uint8),
        ignore_mask=(label < 0).astype(np.uint8),
        conflict_mask=conflict_mask.astype(np.uint8),
        positive_score=positive_score.astype(np.float32),
        negative_score=negative_score.astype(np.float32),
        label_vocab=np.asarray(["ignore", "negative_grasp", "positive_grasp"], dtype=object),
        min_vote_pixels=np.asarray([min_vote_pixels], dtype=np.float32),
        margin=np.asarray([margin], dtype=np.float32),
        include_tool_as_negative=np.asarray([include_tool_as_negative], dtype=bool),
        source_path=np.asarray([str(source_path)], dtype=object),
        source=np.asarray(["clean_binary_grasp_from_sam3_scores"], dtype=object),
    )

    pos_count = int(positive_mask.sum())
    neg_count = int(negative_mask.sum())
    ignore_count = int((label < 0).sum())
    conflict_count = int(conflict_mask.sum())
    warnings = []
    if pos_count < 50:
        warnings.append("few_positive_vertices")
    if neg_count < 50:
        warnings.append("few_negative_vertices")
    if conflict_count > max(pos_count + neg_count, 1):
        warnings.append("large_conflict_region")
    result.update(
        {
            "status": "labeled",
            "num_vertices": int(len(vertices)),
            "positive_vertices": pos_count,
            "negative_vertices": neg_count,
            "ignore_vertices": ignore_count,
            "conflict_vertices": conflict_count,
            "positive_fraction": pos_count / len(vertices) if len(vertices) else 0.0,
            "negative_fraction": neg_count / len(vertices) if len(vertices) else 0.0,
            "ignore_fraction": ignore_count / len(vertices) if len(vertices) else 0.0,
            "warnings": warnings,
        }
    )
    return result


def _write_csv(path: Path, assets: List[Mapping[str, Any]]) -> None:
    fields = [
        "asset_id",
        "label",
        "status",
        "num_vertices",
        "positive_vertices",
        "negative_vertices",
        "ignore_vertices",
        "conflict_vertices",
        "positive_fraction",
        "negative_fraction",
        "ignore_fraction",
        "warnings",
        "output_path",
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
                    "conflict_vertices": item.get("conflict_vertices", 0),
                    "positive_fraction": f"{float(item.get('positive_fraction', 0.0)):.6f}",
                    "negative_fraction": f"{float(item.get('negative_fraction', 0.0)):.6f}",
                    "ignore_fraction": f"{float(item.get('ignore_fraction', 0.0)):.6f}",
                    "warnings": ";".join(item.get("warnings", [])),
                    "output_path": item.get("output_path", ""),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", choices=["dextoolbench12", "domino20", "all"], default="all")
    parser.add_argument("--min-vote-pixels", type=float, default=20.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--include-tool-as-negative", action="store_true")
    parser.add_argument("--output-name", default="grasp_affordance_binary.npz")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "affordance_labels" / "analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = Path(manifest.get("repo_root", REPO_ROOT)).resolve()
    entries = _entries_for_stage(manifest, args.stage)
    assets = [
        _build_entry(
            entry,
            repo_root=repo_root,
            min_vote_pixels=args.min_vote_pixels,
            margin=args.margin,
            include_tool_as_negative=args.include_tool_as_negative,
            output_name=args.output_name,
        )
        for entry in entries
    ]
    labeled = [item for item in assets if item["status"] == "labeled"]
    aggregate = {
        "num_assets": len(assets),
        "num_labeled": len(labeled),
        "num_missing": len(assets) - len(labeled),
        "total_vertices": int(sum(item.get("num_vertices", 0) for item in labeled)),
        "total_positive_vertices": int(sum(item.get("positive_vertices", 0) for item in labeled)),
        "total_negative_vertices": int(sum(item.get("negative_vertices", 0) for item in labeled)),
        "total_ignore_vertices": int(sum(item.get("ignore_vertices", 0) for item in labeled)),
        "total_conflict_vertices": int(sum(item.get("conflict_vertices", 0) for item in labeled)),
        "warning_counts": {},
        "min_vote_pixels": args.min_vote_pixels,
        "margin": args.margin,
        "include_tool_as_negative": bool(args.include_tool_as_negative),
    }
    for item in assets:
        for warning in item.get("warnings", []):
            aggregate["warning_counts"][warning] = aggregate["warning_counts"].get(warning, 0) + 1

    analysis_dir = args.analysis_dir.resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    suffix = "with_tool_negative" if args.include_tool_as_negative else "negative_only"
    stem = f"{args.stage}_binary_grasp_{suffix}_m{args.margin:g}"
    json_path = analysis_dir / f"{stem}.json"
    csv_path = analysis_dir / f"{stem}.csv"
    report = {
        "manifest_path": str(manifest_path),
        "stage": args.stage,
        "aggregate": aggregate,
        "assets": assets,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(csv_path, assets)
    print(json.dumps({"json_path": str(json_path), "csv_path": str(csv_path), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
