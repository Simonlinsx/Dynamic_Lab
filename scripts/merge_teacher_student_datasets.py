#!/usr/bin/env python3
"""Merge exported teacher-student datasets along the sample dimension."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from simtoolreal_lab.teacher_student import default_dataset_spec, validate_student_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-names", nargs="*", default=None)
    return parser.parse_args()


def _trace(message: str) -> None:
    print(f"[MERGE] {message}", flush=True)


def _batch_size(payload: dict[str, Any]) -> int:
    for key in ("target", "pointcloud_seq", "proprio_seq"):
        value = payload.get(key)
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise RuntimeError("Could not infer dataset batch size.")


def _validate(payload: dict[str, Any]) -> None:
    metadata = dict(payload.get("metadata", {}))
    spec = default_dataset_spec(
        task_family=metadata.get("task_family", "falling_baton_grasp"),
        hand=metadata.get("hand_embodiment", "revo2"),
        action_contract=metadata.get("action_contract", "revo2_semantic_13d"),
        history=int(metadata.get("history", payload["pointcloud_seq"].shape[1])),
        num_object_points=int(metadata.get("object_points", payload["pointcloud_seq"].shape[2])),
        point_feature_dim=int(
            metadata.get("point_feature_dim", payload["pointcloud_seq"].shape[-1])
        ),
        proprio_dim=int(metadata.get("proprio_dim", payload["proprio_seq"].shape[-1])),
        compact_privileged_dim=int(metadata.get("compact_privileged_dim", payload["compact_privileged"].shape[-1])),
    )
    errors = validate_student_batch(payload, spec)
    if errors:
        raise RuntimeError("Merged dataset failed schema validation: " + "; ".join(errors))


def _compatible_metadata(first: dict[str, Any], current: dict[str, Any], path: Path) -> None:
    keys = (
        "task",
        "history",
        "object_points",
        "point_feature_dim",
        "proprio_dim",
        "action_dim",
        "compact_privileged_dim",
        "task_family",
        "hand_embodiment",
        "action_contract",
        "pointcloud_source",
    )
    mismatches = []
    for key in keys:
        if first.get(key) != current.get(key):
            mismatches.append(f"{key}: first={first.get(key)!r} current={current.get(key)!r}")
    if first.get("pointcloud_source") == "rgbd_projected_mask":
        first_rgbd = dict(first.get("rgbd_camera", {}))
        current_rgbd = dict(current.get("rgbd_camera", {}))
        rgbd_keys = (
            "width",
            "height",
            "focal_length",
            "track_object",
            "eye",
            "target",
            "mask_points",
            "mask_dilation",
            "depth_tolerance",
            "mask_geometry",
            "depth_matching",
            "clean_fallback_enabled",
            "temporal_fallback_enabled",
        )
        for key in rgbd_keys:
            if first_rgbd.get(key) != current_rgbd.get(key):
                mismatches.append(
                    f"rgbd_camera.{key}: first={first_rgbd.get(key)!r} "
                    f"current={current_rgbd.get(key)!r}"
                )
    if mismatches:
        raise RuntimeError(f"Dataset {path} is not metadata-compatible: " + "; ".join(mismatches))


def main() -> None:
    args = parse_args()
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in args.datasets]
    if not payloads:
        raise RuntimeError("No datasets provided.")

    first_metadata = dict(payloads[0].get("metadata", {}))
    for path, payload in zip(args.datasets, payloads, strict=True):
        _validate(payload)
        _compatible_metadata(first_metadata, dict(payload.get("metadata", {})), path)

    batch_sizes = [_batch_size(payload) for payload in payloads]
    common_tensor_keys = set.intersection(
        *[
            {
                key
                for key, value in payload.items()
                if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == _batch_size(payload)
            }
            for payload in payloads
        ]
    )
    required = {"pointcloud_seq", "pointcloud_valid_seq", "proprio_seq", "target", "compact_privileged"}
    missing_required = required - common_tensor_keys
    if missing_required:
        raise RuntimeError(f"Missing required common tensor keys: {sorted(missing_required)}")

    merged: dict[str, Any] = {}
    for key in sorted(common_tensor_keys):
        merged[key] = torch.cat([payload[key] for payload in payloads], dim=0)

    names = args.source_names or [path.stem for path in args.datasets]
    if len(names) != len(args.datasets):
        raise RuntimeError("--source-names length must match --datasets length.")
    source_ids = []
    for source_id, size in enumerate(batch_sizes):
        source_ids.append(torch.full((size,), source_id, dtype=torch.long))
    merged["source_dataset_id"] = torch.cat(source_ids, dim=0)

    metadata = dict(first_metadata)
    metadata["source"] = f"{metadata.get('source', 'teacher_student_dataset')}_merged"
    metadata["merged_datasets"] = [
        {"path": str(path), "name": str(name), "samples": int(size)}
        for path, name, size in zip(args.datasets, names, batch_sizes, strict=True)
    ]
    metadata["merged_sample_count"] = int(sum(batch_sizes))
    metadata["dropped_noncommon_tensor_keys"] = sorted(
        set().union(
            *[
                {
                    key
                    for key, value in payload.items()
                    if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == _batch_size(payload)
                }
                for payload in payloads
            ]
        )
        - common_tensor_keys
    )
    merged["metadata"] = metadata
    _validate(merged)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.out)
    _trace(f"merged {sum(batch_sizes)} samples from {len(payloads)} datasets")
    _trace(f"common tensor keys: {sorted(common_tensor_keys)}")
    _trace(f"saved merged dataset: {args.out}")


if __name__ == "__main__":
    main()
