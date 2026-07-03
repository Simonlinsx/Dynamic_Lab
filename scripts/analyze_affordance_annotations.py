#!/usr/bin/env python
"""Summarize affordance annotation coverage and quality checks."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

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


def _load_vocab(data: Mapping[str, Any], score_width: int) -> List[str]:
    if "affordance_multihot_vocab" in data:
        return [str(x) for x in np.asarray(data["affordance_multihot_vocab"], dtype=object).tolist()]
    if "affordance_vocab" in data:
        vocab = [str(x) for x in np.asarray(data["affordance_vocab"], dtype=object).tolist()]
        if len(vocab) == score_width + 1:
            return vocab[1:]
        if len(vocab) == score_width:
            return vocab
    return [f"class_{idx}" for idx in range(score_width)]


def _primary_counts(primary: np.ndarray, vocab_with_background: Iterable[str]) -> Dict[str, int]:
    vocab = list(vocab_with_background)
    return {
        name: int((primary == idx).sum())
        for idx, name in enumerate(vocab)
    }


def _warning_flags(
    total_vertices: int,
    primary_labeled: int,
    multilabel_counts: Mapping[str, int],
    overlap_vertices: int,
    primary_counts: Mapping[str, int],
) -> List[str]:
    warnings: List[str] = []
    if total_vertices == 0:
        return ["empty_annotation"]
    coverage = primary_labeled / total_vertices
    if coverage < 0.05:
        warnings.append("low_primary_coverage")
    if coverage > 0.95:
        warnings.append("very_high_primary_coverage")
    if multilabel_counts.get("positive_grasp", 0) == 0:
        warnings.append("missing_positive_grasp")
    if multilabel_counts.get("negative_grasp", 0) == 0:
        warnings.append("missing_negative_grasp")
    if multilabel_counts.get("tool_use_surface", 0) == 0:
        warnings.append("missing_tool_use_surface")
    if (
        multilabel_counts.get("tool_use_surface", 0) > 0
        and primary_counts.get("tool_use_surface", 0) == 0
        and overlap_vertices > 0
    ):
        warnings.append("tool_use_hidden_by_primary_overlap")
    return warnings


def _analyze_entry(entry: Mapping[str, Any], repo_root: Path) -> Dict[str, Any]:
    annotation_dir = _resolve_path(str(entry["annotation_dir"]), repo_root)
    annotation_path = annotation_dir / "affordance.npz"
    result: Dict[str, Any] = {
        "asset_id": entry["asset_id"],
        "label": entry["label"],
        "stage": entry["stage"],
        "category": entry.get("category", ""),
        "annotation_path": str(annotation_path),
        "status": "missing",
    }
    if not annotation_path.exists():
        result["warnings"] = ["missing_annotation"]
        return result

    data = np.load(annotation_path, allow_pickle=True)
    vertices = np.asarray(data["vertices_obj"], dtype=np.float32)
    primary = np.asarray(data["affordance_id"], dtype=np.int64).reshape(-1)
    scores = np.asarray(data["affordance_scores"], dtype=np.float32)
    min_vote_pixels = float(np.asarray(data.get("min_vote_pixels", [20.0])).reshape(-1)[0])
    if "affordance_multihot" in data:
        multihot = np.asarray(data["affordance_multihot"], dtype=bool)
    else:
        multihot = scores >= min_vote_pixels

    score_vocab = _load_vocab(data, scores.shape[1])
    vocab_with_background = ["background"] + score_vocab
    primary_counts = _primary_counts(primary, vocab_with_background)
    multilabel_counts = {
        name: int(multihot[:, idx].sum())
        for idx, name in enumerate(score_vocab)
    }
    overlap_counts = {
        f"{score_vocab[left]}+{score_vocab[right]}": int((multihot[:, left] & multihot[:, right]).sum())
        for left, right in combinations(range(multihot.shape[1]), 2)
    }
    total_vertices = int(len(vertices))
    primary_labeled = int((primary > 0).sum())
    multilabel_labeled = int(multihot.any(axis=1).sum())
    overlap_vertices = int((multihot.sum(axis=1) > 1).sum())
    warnings = _warning_flags(
        total_vertices=total_vertices,
        primary_labeled=primary_labeled,
        multilabel_counts=multilabel_counts,
        overlap_vertices=overlap_vertices,
        primary_counts=primary_counts,
    )
    result.update(
        {
            "status": "labeled",
            "num_vertices": total_vertices,
            "primary_labeled_vertices": primary_labeled,
            "multilabel_labeled_vertices": multilabel_labeled,
            "primary_coverage": primary_labeled / total_vertices if total_vertices else 0.0,
            "multilabel_coverage": multilabel_labeled / total_vertices if total_vertices else 0.0,
            "overlap_vertices": overlap_vertices,
            "primary_counts": primary_counts,
            "multilabel_counts": multilabel_counts,
            "overlap_counts": overlap_counts,
            "warnings": warnings,
        }
    )
    return result


def _write_csv(path: Path, assets: List[Mapping[str, Any]]) -> None:
    fields = [
        "asset_id",
        "label",
        "category",
        "status",
        "num_vertices",
        "primary_labeled_vertices",
        "multilabel_labeled_vertices",
        "primary_coverage",
        "multilabel_coverage",
        "overlap_vertices",
        "positive_grasp_vertices",
        "negative_grasp_vertices",
        "tool_use_surface_vertices",
        "safe_contact_vertices",
        "warnings",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in assets:
            counts = item.get("multilabel_counts", {})
            row = {
                "asset_id": item.get("asset_id", ""),
                "label": item.get("label", ""),
                "category": item.get("category", ""),
                "status": item.get("status", ""),
                "num_vertices": item.get("num_vertices", 0),
                "primary_labeled_vertices": item.get("primary_labeled_vertices", 0),
                "multilabel_labeled_vertices": item.get("multilabel_labeled_vertices", 0),
                "primary_coverage": f"{float(item.get('primary_coverage', 0.0)):.6f}",
                "multilabel_coverage": f"{float(item.get('multilabel_coverage', 0.0)):.6f}",
                "overlap_vertices": item.get("overlap_vertices", 0),
                "positive_grasp_vertices": counts.get("positive_grasp", 0),
                "negative_grasp_vertices": counts.get("negative_grasp", 0),
                "tool_use_surface_vertices": counts.get("tool_use_surface", 0),
                "safe_contact_vertices": counts.get("safe_contact", 0),
                "warnings": ";".join(item.get("warnings", [])),
            }
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", choices=["dextoolbench12", "domino20", "all"], default="dextoolbench12")
    parser.add_argument(
        "--output-dir",
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
    assets = [_analyze_entry(entry, repo_root) for entry in entries]
    labeled = [item for item in assets if item["status"] == "labeled"]
    aggregate = {
        "num_assets": len(assets),
        "num_labeled": len(labeled),
        "num_missing": len(assets) - len(labeled),
        "total_vertices": int(sum(item.get("num_vertices", 0) for item in labeled)),
        "total_primary_labeled_vertices": int(sum(item.get("primary_labeled_vertices", 0) for item in labeled)),
        "total_multilabel_labeled_vertices": int(sum(item.get("multilabel_labeled_vertices", 0) for item in labeled)),
        "total_overlap_vertices": int(sum(item.get("overlap_vertices", 0) for item in labeled)),
        "warning_counts": {},
    }
    for item in assets:
        for warning in item.get("warnings", []):
            aggregate["warning_counts"][warning] = aggregate["warning_counts"].get(warning, 0) + 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.stage}_affordance_analysis.json"
    csv_path = output_dir / f"{args.stage}_affordance_analysis.csv"
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
