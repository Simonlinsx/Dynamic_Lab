#!/usr/bin/env python3
"""Filter or rebalance an exported teacher-student dataset."""

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


COMPACT_LABEL_INDICES = {
    "object_palm_rel_vel": 14,
    "true_grasp": 15,
    "grasp_seen": 16,
    "success": 17,
    "episode_progress": 18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--require-label",
        action="append",
        choices=tuple(COMPACT_LABEL_INDICES) + ("episode_success",),
        default=[],
        help="Keep samples where each required label is above --threshold before optional negative sampling.",
    )
    parser.add_argument(
        "--anchor-label",
        choices=tuple(COMPACT_LABEL_INDICES) + ("episode_success",),
        default=None,
        help="Label used to define positives/negatives for --negative-random-fraction. Defaults to first required label.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--negative-random-fraction",
        type=float,
        default=0.0,
        help="Also keep this fraction of base-valid samples that fail the positive mask.",
    )
    parser.add_argument(
        "--positive-random-fraction",
        type=float,
        default=1.0,
        help="Keep this fraction of positive samples after label filtering.",
    )
    parser.add_argument("--min-current-valid-points", type=float, default=0.0)
    parser.add_argument("--min-history-valid-points", type=float, default=0.0)
    parser.add_argument("--phase-min", type=int, default=None)
    parser.add_argument("--phase-max", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--store-source-indices", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _trace(message: str) -> None:
    print(f"[FILTER] {message}", flush=True)


def _batch_size(payload: dict[str, Any]) -> int:
    for key in ("target", "pointcloud_seq", "proprio_seq"):
        value = payload.get(key)
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise RuntimeError("Could not infer dataset batch size from target/pointcloud_seq/proprio_seq.")


def _label_values(payload: dict[str, Any], label: str) -> torch.Tensor:
    if label == "episode_success":
        if label not in payload:
            raise KeyError("Dataset has no episode_success tensor.")
        return payload[label].float()
    if "compact_privileged" not in payload:
        raise KeyError("Dataset has no compact_privileged tensor.")
    index = COMPACT_LABEL_INDICES[label]
    compact = payload["compact_privileged"].float()
    if compact.shape[-1] <= index:
        raise IndexError(f"compact_privileged has dim {compact.shape[-1]}, cannot read {label} at index {index}.")
    return compact[:, index]


def _label_summary(payload: dict[str, Any], indices: torch.Tensor | None = None) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for label in tuple(COMPACT_LABEL_INDICES) + ("episode_success",):
        if label == "episode_success" and label not in payload:
            continue
        try:
            values = _label_values(payload, label)
        except (KeyError, IndexError):
            continue
        if indices is not None:
            values = values[indices]
        values = values.float()
        positive = values > 0.5
        summary[label] = {
            "mean": float(values.mean().item()) if values.numel() else 0.0,
            "positive_count": int(positive.sum().item()),
            "count": int(values.numel()),
        }
    if "pointcloud_valid_seq" in payload:
        valid = payload["pointcloud_valid_seq"].float()
        if indices is not None:
            valid = valid[indices]
        current_valid = valid[:, -1].sum(dim=-1)
        history_valid = valid.sum(dim=(1, 2))
        summary["valid_points"] = {
            "current_mean": float(current_valid.mean().item()) if current_valid.numel() else 0.0,
            "history_mean": float(history_valid.mean().item()) if history_valid.numel() else 0.0,
            "count": int(valid.shape[0]),
        }
    return summary


def _sample_mask(mask: torch.Tensor, fraction: float, generator: torch.Generator) -> torch.Tensor:
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        return torch.zeros_like(mask)
    fraction = max(0.0, min(1.0, float(fraction)))
    keep_count = int(round(float(indices.numel()) * fraction))
    keep_count = min(max(keep_count, 0), int(indices.numel()))
    selected = torch.zeros_like(mask)
    if keep_count == 0:
        return selected
    perm = torch.randperm(indices.numel(), generator=generator)[:keep_count]
    selected[indices[perm]] = True
    return selected


def _build_selection(payload: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    batch = _batch_size(payload)
    base = torch.ones(batch, dtype=torch.bool)

    if args.min_current_valid_points > 0.0 or args.min_history_valid_points > 0.0:
        valid = payload.get("pointcloud_valid_seq")
        if not torch.is_tensor(valid):
            raise KeyError("Validity filters require pointcloud_valid_seq.")
        valid = valid.float()
        if args.min_current_valid_points > 0.0:
            base &= valid[:, -1].sum(dim=-1) >= float(args.min_current_valid_points)
        if args.min_history_valid_points > 0.0:
            base &= valid.sum(dim=(1, 2)) >= float(args.min_history_valid_points)

    if args.phase_min is not None or args.phase_max is not None:
        phase = payload.get("phase")
        if not torch.is_tensor(phase):
            raise KeyError("Phase filters require phase tensor.")
        if args.phase_min is not None:
            base &= phase >= int(args.phase_min)
        if args.phase_max is not None:
            base &= phase <= int(args.phase_max)

    positive = base.clone()
    for label in args.require_label:
        positive &= _label_values(payload, label) > float(args.threshold)

    generator = torch.Generator().manual_seed(int(args.seed))
    selected = _sample_mask(positive, args.positive_random_fraction, generator)
    anchor = args.anchor_label or (args.require_label[0] if args.require_label else None)
    if args.negative_random_fraction > 0.0:
        if anchor is None:
            negative = base & ~selected
        else:
            negative = base & ~(_label_values(payload, anchor) > float(args.threshold))
        selected |= _sample_mask(negative, args.negative_random_fraction, generator)

    if not args.require_label and args.positive_random_fraction >= 1.0 and args.negative_random_fraction <= 0.0:
        selected = base

    if args.max_samples is not None:
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        max_samples = min(int(args.max_samples), int(indices.numel()))
        if max_samples < indices.numel():
            perm = torch.randperm(indices.numel(), generator=generator)[:max_samples]
            limited = torch.zeros_like(selected)
            limited[indices[perm]] = True
            selected = limited

    return selected


def _filter_payload(payload: dict[str, Any], indices: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    batch = _batch_size(payload)
    filtered: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "metadata":
            continue
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch:
            filtered[key] = value[indices]
        else:
            filtered[key] = value
    if args.store_source_indices:
        filtered["source_index"] = indices.clone()

    metadata = dict(payload.get("metadata", {}))
    source_filters = list(metadata.get("filters", []))
    source_filters.append(
        {
            "script": Path(__file__).name,
            "input_dataset": str(args.dataset),
            "selected_count": int(indices.numel()),
            "input_count": int(batch),
            "require_label": list(args.require_label),
            "anchor_label": args.anchor_label,
            "threshold": float(args.threshold),
            "negative_random_fraction": float(args.negative_random_fraction),
            "positive_random_fraction": float(args.positive_random_fraction),
            "min_current_valid_points": float(args.min_current_valid_points),
            "min_history_valid_points": float(args.min_history_valid_points),
            "phase_min": args.phase_min,
            "phase_max": args.phase_max,
            "max_samples": args.max_samples,
            "seed": int(args.seed),
            "shuffle": bool(args.shuffle),
            "label_summary_before": _label_summary(payload),
            "label_summary_after": _label_summary(payload, indices),
        }
    )
    metadata["filters"] = source_filters
    metadata["source"] = f"{metadata.get('source', 'teacher_student_dataset')}_filtered"
    filtered["metadata"] = metadata
    return filtered


def _validate(payload: dict[str, Any]) -> None:
    metadata = dict(payload.get("metadata", {}))
    spec = default_dataset_spec(
        task_family=metadata.get("task_family", "falling_baton_grasp"),
        hand=metadata.get("hand_embodiment", "revo2"),
        action_contract=metadata.get("action_contract", "revo2_semantic_13d"),
        history=int(metadata.get("history", payload["pointcloud_seq"].shape[1])),
        num_object_points=int(metadata.get("object_points", payload["pointcloud_seq"].shape[2])),
        proprio_dim=int(metadata.get("proprio_dim", payload["proprio_seq"].shape[-1])),
        compact_privileged_dim=int(metadata.get("compact_privileged_dim", payload["compact_privileged"].shape[-1])),
    )
    errors = validate_student_batch(payload, spec)
    if errors:
        raise RuntimeError("Filtered dataset failed schema validation: " + "; ".join(errors))


def main() -> None:
    args = parse_args()
    payload = torch.load(args.dataset, map_location="cpu", weights_only=False)
    batch = _batch_size(payload)
    selected = _build_selection(payload, args)
    indices = torch.nonzero(selected, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise RuntimeError("Filter selected zero samples.")
    if args.shuffle:
        generator = torch.Generator().manual_seed(int(args.seed) + 1009)
        indices = indices[torch.randperm(indices.numel(), generator=generator)]

    filtered = _filter_payload(payload, indices, args)
    _validate(filtered)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(filtered, args.out)

    summary_after = filtered["metadata"]["filters"][-1]["label_summary_after"]
    _trace(f"selected {indices.numel()} / {batch} samples")
    for label in ("true_grasp", "grasp_seen", "success", "episode_success", "valid_points"):
        if label in summary_after:
            _trace(f"{label}: {summary_after[label]}")
    _trace(f"saved filtered dataset: {args.out}")


if __name__ == "__main__":
    main()
