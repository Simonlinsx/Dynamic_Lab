#!/usr/bin/env python
"""Collect per-asset SAM3 summaries after selective reruns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", choices=["dextoolbench12", "domino20", "all"], required=True)
    parser.add_argument("--summary-name", default="sam3_summary.json")
    parser.add_argument("--output-name", default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "affordance_labels" / "sam3_runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = Path(manifest.get("repo_root", REPO_ROOT)).resolve()
    entries = _entries_for_stage(manifest, args.stage)
    summaries = []
    failures = []
    for entry in entries:
        summary_path = _resolve_path(str(entry["annotation_dir"]), repo_root) / args.summary_name
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        else:
            failures.append({"asset_id": entry.get("asset_id"), "error": f"Missing {summary_path}"})

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{args.stage}_sam3_collected_summary.json"
    output_path = output_dir / output_name
    result = {
        "manifest_path": str(manifest_path),
        "stage": args.stage,
        "num_requested": len(entries),
        "num_completed": len(summaries),
        "num_failed": len(failures),
        "summaries": summaries,
        "failures": failures,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(output_path), "num_completed": len(summaries), "num_failed": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
