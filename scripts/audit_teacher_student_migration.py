#!/usr/bin/env python3
"""Audit the current dynamic-grasp teacher-student migration artifacts.

The audit is intentionally lightweight: it does not start IsaacLab.  It checks
the JSON summaries, checkpoints, success videos, traces, and student specs that
prove the migrated teacher-student stack is currently reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from simtoolreal_lab.teacher_student.schema import ACTION_CONTRACTS, default_dataset_spec, validate_student_batch


DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "case_id": "revo2_tabletop",
        "family": "dynamic_tabletop_grasp",
        "hand": "revo2",
        "task": "SimToolReal-Revo2-Franka-DynamicTabletop-Teacher-Direct-v0",
        "expected_action_dim": 13,
        "expected_proprio_dim": 76,
        "expected_arm_dim": 7,
        "expected_hand_dim": 6,
        "teacher_min_success_rate": 0.80,
        "student_min_success_rate": 0.45,
        "teacher_run_dir": "logs/rl_games/revo2_dynamic_tabletop_from_home_teacher_online_512env_2000ep_20260701",
        "student_dataset": "outputs/teacher_student/revo2_tabletop_best_fullspeed_teacher_rgbd_aligned_64env420_dataset.pt",
        "teacher_summary": (
            "outputs/eval_rl_games/tabletop_best_full_speed_vector_eval_online/"
            "20260701_180847/summary.json"
        ),
        "teacher_video_summary": (
            "outputs/eval_rl_games/tabletop_best_full_speed_success_video_side_camera_online/"
            "20260701_181119/summary.json"
        ),
        "student_summary": (
            "outputs/eval_teacher_student/"
            "revo2_tabletop_success_balanced_residual_hold_head_predpriv_gate_blend035_"
            "resclamp025_scale025_vector_max420_fixed64/20260702_004834/summary.json"
        ),
        "student_video_summary": (
            "outputs/eval_teacher_student/"
            "revo2_tabletop_success_balanced_residual_hold_head_scale025_success_video_"
            "isaacgym_side_16env_attempt0/20260702_010516/summary.json"
        ),
    },
    {
        "case_id": "revo2_falling_baton",
        "family": "falling_baton_grasp",
        "hand": "revo2",
        "task": "SimToolReal-Revo2-Franka-FallingBatonFullSpeedEval-Teacher-Direct-v0",
        "expected_action_dim": 13,
        "expected_proprio_dim": 76,
        "expected_arm_dim": 7,
        "expected_hand_dim": 6,
        "teacher_min_success_rate": 0.94,
        "student_min_success_rate": 0.65,
        "teacher_run_dir": (
            "logs/rl_games/revo2_falling_baton_full_no_table_from_easy_ep350_online_512env_"
            "2000ep_20260701"
        ),
        "student_dataset": (
            "outputs/teacher_student/revo2_falling_baton_fullspeed_ep400_teacher_rgbd_aligned_"
            "32env320_dataset.pt"
        ),
        "teacher_summary": (
            "outputs/eval_rl_games/falling_baton_full_speed_eval_ep400_vector_eval_online/"
            "20260701_171255/summary.json"
        ),
        "teacher_video_summary": (
            "outputs/eval_rl_games/falling_baton_full_speed_ep400_success_video_wide_side_eval_online/"
            "20260701_171630/summary.json"
        ),
        "student_summary": (
            "outputs/eval_teacher_student/"
            "revo2_falling_baton_fullspeed_ep400_rgbd_student_bc_aligned_ep20_vector_max420/"
            "20260701_220214/summary.json"
        ),
        "student_video_summary": (
            "outputs/eval_teacher_student/"
            "revo2_falling_baton_fullspeed_ep400_rgbd_student_bc_aligned_ep20_success_video_side/"
            "20260701_220308/summary.json"
        ),
    },
    {
        "case_id": "inspire_tabletop",
        "family": "dynamic_tabletop_grasp",
        "hand": "inspire",
        "task": "SimToolReal-Inspire-Franka-DynamicTabletop-DirectResidual-Teacher-Direct-v0",
        "expected_action_dim": 13,
        "expected_proprio_dim": 76,
        "expected_arm_dim": 7,
        "expected_hand_dim": 6,
        "teacher_min_success_rate": 0.34,
        "student_min_success_rate": 0.28,
        "teacher_run_dir": "logs/rl_games/inspire_tabletop_direct_residual_from_scratch_local_512env_1200ep_20260701",
        "student_dataset": (
            "outputs/teacher_student/inspire_tabletop_direct_residual_ep355_fullspeed_rgbd_"
            "teacher_aligned_plus_dagger_17664_dataset.pt"
        ),
        "teacher_summary": (
            "outputs/eval_rl_games/inspire_tabletop_direct_residual_ep355_vector_eval_fullspeed_local/"
            "20260701_201519/summary.json"
        ),
        "teacher_video_summary": (
            "outputs/eval_rl_games/inspire_tabletop_direct_residual_ep355_success_video_side_camera_fullspeed_local/"
            "20260701_201619/summary.json"
        ),
        "student_summary": (
            "outputs/eval_teacher_student/"
            "inspire_tabletop_direct_residual_fullspeed_rgbd_student_dagger_ft_aligned_plus_dagger_"
            "ep15_vector_max420/20260701_214931/summary.json"
        ),
        "student_video_summary": (
            "outputs/eval_teacher_student/"
            "inspire_tabletop_direct_residual_fullspeed_rgbd_student_dagger_ft_ep15_success_video_side/"
            "20260701_215124/summary.json"
        ),
    },
    {
        "case_id": "inspire_falling_baton",
        "family": "falling_baton_grasp",
        "hand": "inspire",
        "task": "SimToolReal-Inspire-Franka-FallingBaton-Teacher-Direct-v0",
        "expected_action_dim": 13,
        "expected_proprio_dim": 76,
        "expected_arm_dim": 7,
        "expected_hand_dim": 6,
        "teacher_min_success_rate": 0.95,
        "student_min_success_rate": 0.82,
        "teacher_run_dir": "logs/rl_games/inspire_falling_baton_from_smoke_teacher_online_512env_600ep_20260701",
        "student_dataset": (
            "outputs/teacher_student/inspire_falling_baton_fullspeed_600ep_teacher_rgbd_aligned_"
            "32env320_dataset.pt"
        ),
        "teacher_summary": (
            "outputs/eval_rl_games/inspire_falling_baton_from_smoke_teacher_600ep_vector_eval_online/"
            "20260701_185312/summary.json"
        ),
        "teacher_video_summary": (
            "outputs/eval_rl_games/inspire_falling_baton_from_smoke_teacher_600ep_success_video_side_camera_local/"
            "20260701_185427/summary.json"
        ),
        "student_summary": (
            "outputs/eval_teacher_student/"
            "inspire_falling_baton_fullspeed_600ep_rgbd_student_bc_aligned_vector_max420/"
            "20260701_222758/summary.json"
        ),
        "student_video_summary": (
            "outputs/eval_teacher_student/"
            "inspire_falling_baton_fullspeed_600ep_rgbd_student_bc_aligned_success_video_side/"
            "20260701_222859/summary.json"
        ),
    },
]


@dataclass
class CheckResult:
    ok: bool = True
    warnings: list[str] | None = None
    errors: list[str] | None = None

    def warn(self, message: str) -> None:
        if self.warnings is None:
            self.warnings = []
        self.warnings.append(message)

    def error(self, message: str) -> None:
        if self.errors is None:
            self.errors = []
        self.errors.append(message)
        self.ok = False


def _load_json(path: Path, result: CheckResult, label: str) -> dict[str, Any]:
    if not path.exists():
        result.error(f"{label} missing: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive audit path
        result.error(f"{label} cannot be read as JSON: {path} ({exc})")
        return {}


def _as_project_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _success_rate(summary: dict[str, Any]) -> float | None:
    value = summary.get("eval", {}).get("success_rate")
    return float(value) if isinstance(value, (int, float)) else None


def _eval_count(summary: dict[str, Any]) -> int | None:
    value = summary.get("eval", {}).get("success_count")
    return int(value) if isinstance(value, int) else None


def _episode_count(summary: dict[str, Any]) -> int | None:
    value = summary.get("eval", {}).get("episodes")
    return int(value) if isinstance(value, int) else None


def _check_success_summary(
    result: CheckResult,
    summary: dict[str, Any],
    label: str,
    min_success_rate: float,
) -> dict[str, Any]:
    rate = _success_rate(summary)
    episodes = _episode_count(summary)
    count = _eval_count(summary)
    if rate is None:
        result.error(f"{label} has no numeric eval.success_rate")
    elif rate < min_success_rate:
        result.error(f"{label} success_rate={rate:.6f} below min={min_success_rate:.6f}")
    if episodes is None or episodes <= 0:
        result.error(f"{label} has no positive eval.episodes")
    return {"success_rate": rate, "episodes": episodes, "success_count": count}


def _check_checkpoint(root: Path, result: CheckResult, summary: dict[str, Any], label: str) -> str | None:
    checkpoint = _as_project_path(root, summary.get("checkpoint"))
    if checkpoint is None:
        result.error(f"{label} has no checkpoint path")
        return None
    if not checkpoint.exists():
        result.error(f"{label} checkpoint missing: {checkpoint}")
    return str(checkpoint)


def _check_teacher_training(root: Path, result: CheckResult, case: dict[str, Any]) -> dict[str, Any]:
    run_dir = _as_project_path(root, case.get("teacher_run_dir"))
    if run_dir is None:
        result.error(f"{case['case_id']} missing teacher_run_dir")
        return {}
    record: dict[str, Any] = {"run_dir": str(run_dir)}
    if not run_dir.exists():
        result.error(f"{case['case_id']} teacher run dir missing: {run_dir}")
        return record

    event_files = sorted((run_dir / "summaries").glob("events.out.tfevents.*"))
    nn_checkpoints = sorted((run_dir / "nn").glob("*.pth"))
    wandb_files = sorted((run_dir / "wandb").glob("run-*/run-*.wandb"))
    record.update(
        {
            "event_files": [str(path) for path in event_files],
            "num_event_files": len(event_files),
            "nn_checkpoints": [str(path) for path in nn_checkpoints],
            "num_nn_checkpoints": len(nn_checkpoints),
            "wandb_run_files": [str(path) for path in wandb_files],
            "num_wandb_run_files": len(wandb_files),
        }
    )
    if not event_files:
        result.error(f"{case['case_id']} teacher run has no TensorBoard event files under {run_dir / 'summaries'}")
    if not nn_checkpoints:
        result.error(f"{case['case_id']} teacher run has no .pth checkpoints under {run_dir / 'nn'}")
    return record


def _batch_size(payload: dict[str, Any]) -> int | None:
    for key in ("target", "pointcloud_seq", "proprio_seq"):
        value = payload.get(key)
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
    return None


def _tensor_mean(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None or not hasattr(value, "numel") or int(value.numel()) == 0:
        return None
    return float(value.float().mean())


def _load_torch(path: Path, result: CheckResult, label: str) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment invariant
        result.error(f"torch is required to inspect {label}: {exc}")
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - defensive artifact path
        result.error(f"{label} cannot be loaded with torch: {path} ({exc})")
        return None


def _dataset_spec_from_payload(payload: dict[str, Any]) -> Any:
    metadata = dict(payload.get("metadata", {}))
    return default_dataset_spec(
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


def _check_student_dataset(root: Path, result: CheckResult, case: dict[str, Any]) -> dict[str, Any]:
    dataset_path = _as_project_path(root, case.get("student_dataset"))
    if dataset_path is None:
        result.error(f"{case['case_id']} missing student_dataset")
        return {}
    record: dict[str, Any] = {"path": str(dataset_path)}
    if not dataset_path.exists():
        result.error(f"{case['case_id']} student dataset missing: {dataset_path}")
        return record

    payload = _load_torch(dataset_path, result, f"{case['case_id']} student dataset")
    if not isinstance(payload, dict):
        result.error(f"{case['case_id']} student dataset is not a dictionary payload")
        return record
    try:
        spec = _dataset_spec_from_payload(payload)
        errors = validate_student_batch(payload, spec)
    except Exception as exc:
        result.error(f"{case['case_id']} student dataset schema setup failed: {exc}")
        errors = []
        spec = None
    if errors:
        result.error(f"{case['case_id']} student dataset schema errors: " + "; ".join(errors))

    metadata = dict(payload.get("metadata", {}))
    batch_size = _batch_size(payload)
    expected_action_dim = int(case["expected_action_dim"])
    expected_proprio_dim = int(case["expected_proprio_dim"])
    if metadata.get("action_dim") != expected_action_dim:
        result.error(
            f"{case['case_id']} dataset action_dim={metadata.get('action_dim')!r}, expected={expected_action_dim}"
        )
    if metadata.get("proprio_dim") != expected_proprio_dim:
        result.error(
            f"{case['case_id']} dataset proprio_dim={metadata.get('proprio_dim')!r}, expected={expected_proprio_dim}"
        )
    if metadata.get("pointcloud_source") != "rgbd_projected_mask":
        result.error(
            f"{case['case_id']} dataset pointcloud_source={metadata.get('pointcloud_source')!r}, "
            "expected='rgbd_projected_mask'"
        )
    if batch_size is None or batch_size <= 0:
        result.error(f"{case['case_id']} dataset has no positive batch size")

    record.update(
        {
            "metadata": metadata,
            "schema_valid": not errors,
            "num_samples": batch_size,
            "spec": {
                "history": getattr(spec, "history", None),
                "num_object_points": getattr(spec, "num_object_points", None),
                "proprio_dim": getattr(spec, "proprio_dim", None),
                "action_dim": getattr(spec, "action_dim", None),
                "compact_privileged_dim": getattr(spec, "compact_privileged_dim", None),
            }
            if spec is not None
            else {},
            "pointcloud_valid_mean": _tensor_mean(payload, "pointcloud_valid_seq"),
            "episode_success_mean": _tensor_mean(payload, "episode_success"),
            "episode_success_count": (
                int(payload["episode_success"].float().sum().item()) if "episode_success" in payload else None
            ),
            "hold_mask_mean": _tensor_mean(payload, "hold_mask"),
            "hold_mask_count": int(payload["hold_mask"].float().sum().item()) if "hold_mask" in payload else None,
            "has_point_flow_velocity": "point_flow_velocity" in payload,
            "has_affordance_region_labels": "affordance_region_labels" in payload,
        }
    )
    return record


def _check_student_training(
    root: Path,
    result: CheckResult,
    case: dict[str, Any],
    student_checkpoint: str | None,
) -> dict[str, Any]:
    if student_checkpoint is None:
        result.error(f"{case['case_id']} missing student checkpoint for training audit")
        return {}
    checkpoint_path = Path(student_checkpoint)
    train_dir = checkpoint_path.parent
    last_path = train_dir / "student_pretrain_last.pt"
    record: dict[str, Any] = {
        "train_dir": str(train_dir),
        "best_checkpoint": str(checkpoint_path),
        "last_checkpoint": str(last_path),
        "last_checkpoint_exists": last_path.exists(),
    }
    if not last_path.exists():
        result.error(f"{case['case_id']} student pretrain last checkpoint missing: {last_path}")

    payload = _load_torch(checkpoint_path, result, f"{case['case_id']} student checkpoint")
    if not isinstance(payload, dict):
        result.error(f"{case['case_id']} student checkpoint is not a dictionary payload")
        return record
    val_loss = payload.get("val_loss")
    if not isinstance(val_loss, (int, float)) or not math.isfinite(float(val_loss)):
        result.error(f"{case['case_id']} student checkpoint has invalid val_loss={val_loss!r}")
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or epoch <= 0:
        result.error(f"{case['case_id']} student checkpoint has invalid epoch={epoch!r}")
    record.update(
        {
            "epoch": epoch,
            "val_loss": float(val_loss) if isinstance(val_loss, (int, float)) else None,
            "spec": payload.get("spec", {}),
            "metadata": payload.get("metadata", {}),
            "hold_target_mode": payload.get("hold_target_mode"),
            "train_hold_head_only": payload.get("train_hold_head_only"),
        }
    )
    return record


def _resolved_spec(spec: dict[str, Any], metadata: dict[str, Any], expected_arm_dim: int) -> dict[str, Any]:
    resolved = dict(spec)
    action_dim = int(resolved.get("action_dim", 0) or 0)
    contract_name = str(metadata.get("action_contract", ""))
    contract = ACTION_CONTRACTS.get(contract_name)
    contract_note = None
    if contract is not None:
        contract_note = contract.name
        action_dim = action_dim or int(contract.action_dim)
        resolved.setdefault("action_dim", int(contract.action_dim))
    arm_dim = int(resolved.get("arm_dim", getattr(contract, "arm_dim", expected_arm_dim)) or expected_arm_dim)
    resolved.setdefault("arm_dim", arm_dim)
    if contract is not None:
        resolved.setdefault("hand_dim", int(contract.hand_dim))
    elif action_dim > 0:
        resolved.setdefault("hand_dim", max(action_dim - arm_dim, 0))
    if contract_note is not None:
        resolved.setdefault("resolved_from_action_contract", contract_note)
    return resolved


def _check_student_spec(
    result: CheckResult,
    summary: dict[str, Any],
    expected_action_dim: int,
    expected_proprio_dim: int,
    expected_arm_dim: int,
    expected_hand_dim: int,
) -> dict[str, Any]:
    spec = summary.get("spec") or {}
    if not isinstance(spec, dict):
        result.error("student spec is not a dictionary")
        return {}
    metadata = summary.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    resolved = _resolved_spec(spec, metadata, expected_arm_dim)
    for key, expected in (
        ("action_dim", expected_action_dim),
        ("proprio_dim", expected_proprio_dim),
        ("arm_dim", expected_arm_dim),
        ("hand_dim", expected_hand_dim),
    ):
        value = resolved.get(key)
        if value != expected:
            result.error(f"student spec {key}={value!r}, expected {expected!r}")
    for key in ("history", "num_object_points", "compact_privileged_dim"):
        if key not in resolved:
            result.warn(f"student spec missing {key}")
    for key in ("arm_dim", "hand_dim"):
        if key not in spec and not resolved.get("resolved_from_action_contract"):
            result.warn(f"student summary did not store {key}; resolved from action_dim/arm_dim")
    return resolved


def _video_frame_stats(path: Path, max_scan_frames: int, sample_frame_path: Path | None = None) -> dict[str, Any]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        chosen = None
        chosen_index = 0
        chosen_std = -1.0
        scanned_stds: list[float] = []
        for index, frame in enumerate(reader):
            frame_std = float(frame.std())
            scanned_stds.append(frame_std)
            if frame_std > chosen_std:
                chosen = frame
                chosen_index = index
                chosen_std = frame_std
            if index + 1 >= max_scan_frames:
                break
        if chosen is None:
            return {"error": "no frames"}
        if sample_frame_path is not None:
            sample_frame_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(sample_frame_path, chosen)
        return {
            "frame_index": chosen_index,
            "frames_scanned": len(scanned_stds),
            "shape": list(chosen.shape),
            "mean": float(chosen.mean()),
            "std": chosen_std,
            "scan_std_min": min(scanned_stds),
            "scan_std_max": max(scanned_stds),
            "min": int(chosen.min()),
            "max": int(chosen.max()),
            "sample_frame": str(sample_frame_path) if sample_frame_path else None,
        }
    finally:
        reader.close()


def _check_trace(path: Path, result: CheckResult, label: str) -> dict[str, Any]:
    trace = _load_json(path, result, f"{label} trace")
    if not trace:
        return {}
    trials = trace.get("trials")
    if isinstance(trials, list):
        successful_trials = [trial for trial in trials if bool(trial.get("success"))]
        if not successful_trials:
            result.error(f"{label} trial-sequence trace has no successful trial")
        return {
            "trial_count": int(trace.get("trial_count", len(trials))),
            "success_count": int(trace.get("success_count", len(successful_trials))),
            "success_rate": trace.get("success_rate"),
            "raw_success_count": trace.get("raw_success_count"),
            "post_success_hold_count": trace.get("post_success_hold_count"),
            "camera_eye": trace.get("camera_eye"),
            "camera_target": trace.get("camera_target"),
            "camera_track_object": trace.get("camera_track_object"),
        }

    first_success_step = trace.get("first_success_step")
    success_steps = trace.get("success_steps") or []
    if first_success_step is None and not success_steps:
        result.error(f"{label} trace has no first_success_step or success_steps")
    return {
        "first_success_step": first_success_step,
        "first_lifted_step": trace.get("first_lifted_step"),
        "first_stable_hold_step": trace.get("first_stable_hold_step"),
        "max_object_height_delta": trace.get("max_object_height_delta"),
        "camera_track_offset": trace.get("camera_track_offset"),
        "camera_track_target_offset": trace.get("camera_track_target_offset"),
    }


def _check_video_summary(
    root: Path,
    result: CheckResult,
    summary: dict[str, Any],
    label: str,
    args: argparse.Namespace,
    video_override: str | list[str] | None = None,
    trace_override: str | list[str] | None = None,
    require_trace: bool = True,
) -> dict[str, Any]:
    if video_override is None:
        videos = summary.get("trial_sequence_videos") or summary.get("success_videos") or []
    else:
        videos = video_override if isinstance(video_override, list) else [video_override]
    if trace_override is None:
        traces = summary.get("trial_sequence_traces") or summary.get("success_video_traces") or []
    else:
        traces = trace_override if isinstance(trace_override, list) else [trace_override]
    if not videos:
        result.error(f"{label} summary has no success_videos")
        return {"videos": [], "traces": []}
    video_records = []
    for video in videos:
        video_path = _as_project_path(root, video)
        if video_path is None:
            result.error(f"{label} has an empty video path")
            continue
        record: dict[str, Any] = {"path": str(video_path)}
        if not video_path.exists():
            result.error(f"{label} video missing: {video_path}")
        else:
            record["bytes"] = video_path.stat().st_size
            if record["bytes"] <= 0:
                result.error(f"{label} video is empty: {video_path}")
            if not args.skip_video_frame_check:
                try:
                    sample_frame_path = None
                    if not args.no_save_video_sample_frames:
                        safe_label = "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_")
                        sample_frame_dir = Path(args.sample_frame_dir)
                        sample_frame_path = sample_frame_dir / f"{safe_label}_{len(video_records):02d}.jpg"
                    stats = _video_frame_stats(video_path, int(args.video_scan_frames), sample_frame_path)
                    record["frame_stats"] = stats
                    if stats.get("sample_frame"):
                        record["sample_frame"] = stats["sample_frame"]
                    if "error" in stats:
                        result.error(f"{label} video frame check failed: {stats['error']}")
                    elif float(stats["std"]) < float(args.min_video_std):
                        result.error(
                            f"{label} video appears blank: std={stats['std']:.6f} "
                            f"below min={float(args.min_video_std):.6f}"
                        )
                except Exception as exc:  # pragma: no cover - optional codec path
                    result.warn(f"{label} video frame check failed for {video_path}: {exc}")
        video_records.append(record)

    trace_records = []
    if not traces and require_trace:
        result.error(f"{label} summary has no success_video_traces")
    for trace in traces:
        trace_path = _as_project_path(root, trace)
        if trace_path is None:
            result.error(f"{label} has an empty trace path")
            continue
        trace_records.append({"path": str(trace_path), **_check_trace(trace_path, result, label)})
    return {"videos": video_records, "traces": trace_records}


def _load_manifest(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_CASES
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("cases", []))
    if isinstance(payload, list):
        return payload
    raise TypeError("Manifest must be a JSON list or a dictionary with a 'cases' list.")


def _status(result: CheckResult, fail_on_warning: bool) -> str:
    if result.errors:
        return "FAIL"
    if result.warnings:
        return "FAIL" if fail_on_warning else "WARN"
    return "PASS"


def _audit_case(root: Path, case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = CheckResult()
    case_id = str(case["case_id"])
    student_required = bool(case.get("student_required", True))
    teacher_summary_path = _as_project_path(root, case.get("teacher_summary"))
    teacher_video_summary_path = _as_project_path(root, case.get("teacher_video_summary"))
    student_summary_path = _as_project_path(root, case.get("student_summary"))
    student_video_summary_path = _as_project_path(root, case.get("student_video_summary"))
    if teacher_summary_path is None:
        result.error(f"{case_id} missing teacher_summary")
        teacher_summary = {}
    else:
        teacher_summary = _load_json(teacher_summary_path, result, "teacher summary")
    if teacher_video_summary_path is None:
        result.error(f"{case_id} missing teacher_video_summary")
        teacher_video_summary = {}
    else:
        teacher_video_summary = _load_json(teacher_video_summary_path, result, "teacher video summary")
    if student_required and student_summary_path is None:
        result.error(f"{case_id} missing student_summary")
        student_summary = {}
    elif student_summary_path is not None:
        student_summary = _load_json(student_summary_path, result, "student summary")
    else:
        student_summary = {}
    if student_required and student_video_summary_path is None:
        result.error(f"{case_id} missing student_video_summary")
        student_video_summary = {}
    elif student_video_summary_path is not None:
        student_video_summary = _load_json(student_video_summary_path, result, "student video summary")
    else:
        student_video_summary = {}

    teacher_eval = _check_success_summary(
        result,
        teacher_summary,
        f"{case_id} teacher",
        float(case.get("teacher_min_success_rate", 0.0)),
    )
    teacher_checkpoint = _check_checkpoint(root, result, teacher_summary, f"{case_id} teacher")
    teacher_training = _check_teacher_training(root, result, case)
    teacher_video = _check_video_summary(
        root,
        result,
        teacher_video_summary,
        f"{case_id} teacher",
        args,
        video_override=case.get("teacher_video"),
        trace_override=case.get("teacher_video_trace"),
        require_trace=bool(case.get("teacher_video_trace_required", True)),
    )
    if student_required:
        student_eval = _check_success_summary(
            result,
            student_summary,
            f"{case_id} student",
            float(case.get("student_min_success_rate", 0.0)),
        )
        student_checkpoint = _check_checkpoint(root, result, student_summary, f"{case_id} student")
        student_dataset = _check_student_dataset(root, result, case)
        student_training = _check_student_training(root, result, case, student_checkpoint)
        student_spec = _check_student_spec(
            result,
            student_summary,
            int(case["expected_action_dim"]),
            int(case["expected_proprio_dim"]),
            int(case.get("expected_arm_dim", 7)),
            int(case["expected_hand_dim"]),
        )
        student_video = _check_video_summary(
            root,
            result,
            student_video_summary,
            f"{case_id} student",
            args,
            video_override=case.get("student_video"),
            trace_override=case.get("student_video_trace"),
            require_trace=bool(case.get("student_video_trace_required", True)),
        )
    else:
        student_eval = {"success_rate": None, "episodes": None, "success_count": None}
        student_checkpoint = None
        student_dataset = {}
        student_training = {}
        student_spec = {}
        student_video = {"videos": [], "traces": []}

    return {
        "case_id": case_id,
        "family": case.get("family"),
        "hand": case.get("hand"),
        "task": case.get("task"),
        "status": _status(result, bool(args.fail_on_warning)),
        "warnings": result.warnings or [],
        "errors": result.errors or [],
        "teacher": {
            "summary": str(teacher_summary_path) if teacher_summary_path else None,
            "checkpoint": teacher_checkpoint,
            "eval": teacher_eval,
            "min_success_rate": float(case.get("teacher_min_success_rate", 0.0)),
            "training": teacher_training,
            "video": teacher_video,
        },
        "student": {
            "required": student_required,
            "summary": str(student_summary_path) if student_summary_path else None,
            "checkpoint": student_checkpoint,
            "dataset": student_dataset,
            "eval": student_eval,
            "min_success_rate": float(case.get("student_min_success_rate", 0.0)),
            "spec": student_spec,
            "training": student_training,
            "video": student_video,
        },
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Teacher-Student Migration Audit",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Root: `{report['root']}`",
        f"- Overall status: `{report['overall_status']}`",
        "",
        "| Case | Family | Hand | Teacher SR | Student SR | Status |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for case in report["cases"]:
        teacher_rate = case["teacher"]["eval"].get("success_rate")
        student_rate = case["student"]["eval"].get("success_rate")
        teacher_text = "n/a" if teacher_rate is None else f"{teacher_rate:.6f}"
        student_text = "n/a" if student_rate is None else f"{student_rate:.6f}"
        lines.append(
            "| {case_id} | {family} | {hand} | {teacher_sr} | {student_sr} | {status} |".format(
                case_id=case["case_id"],
                family=case.get("family") or "",
                hand=case.get("hand") or "",
                teacher_sr=teacher_text,
                student_sr=student_text,
                status=case["status"],
            )
        )

    lines.extend(["", "## Details", ""])
    for case in report["cases"]:
        lines.extend(
            [
                f"### {case['case_id']}",
                "",
                f"- Task: `{case.get('task')}`",
                f"- Teacher summary: `{case['teacher']['summary']}`",
                f"- Teacher train dir: `{case['teacher']['training'].get('run_dir')}`",
                f"- Teacher train events/checkpoints: "
                f"`{case['teacher']['training'].get('num_event_files', 0)}` / "
                f"`{case['teacher']['training'].get('num_nn_checkpoints', 0)}`",
            ]
        )
        if bool(case["student"].get("required", True)):
            lines.extend(
                [
                    f"- Student summary: `{case['student']['summary']}`",
                    f"- Student dataset: `{case['student']['dataset'].get('path')}`",
                    f"- Student dataset samples: `{case['student']['dataset'].get('num_samples')}`",
                    f"- Student dataset success samples: "
                    f"`{case['student']['dataset'].get('episode_success_count')}`",
                    f"- Student checkpoint: `{case['student']['checkpoint']}`",
                    f"- Student train best epoch/loss: "
                    f"`{case['student']['training'].get('epoch')}` / "
                    f"`{case['student']['training'].get('val_loss')}`",
                    f"- Student spec: `{json.dumps(case['student']['spec'], sort_keys=True)}`",
                ]
            )
        teacher_videos = case["teacher"]["video"].get("videos", [])
        student_videos = case["student"]["video"].get("videos", [])
        if teacher_videos:
            lines.append(f"- Teacher video: `{teacher_videos[0]['path']}`")
            if teacher_videos[0].get("sample_frame"):
                lines.append(f"- Teacher sample frame: `{teacher_videos[0]['sample_frame']}`")
        if student_videos:
            lines.append(f"- Student video: `{student_videos[0]['path']}`")
            if student_videos[0].get("sample_frame"):
                lines.append(f"- Student sample frame: `{student_videos[0]['sample_frame']}`")
        for warning in case.get("warnings", []):
            lines.append(f"- Warning: {warning}")
        for error in case.get("errors", []):
            lines.append(f"- Error: {error}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/teacher_student_audit"))
    parser.add_argument("--output-prefix", default="teacher_student_migration_audit")
    parser.add_argument("--skip-video-frame-check", action="store_true")
    parser.add_argument("--no-save-video-sample-frames", action="store_true")
    parser.add_argument("--video-scan-frames", type=int, default=90)
    parser.add_argument("--min-video-std", type=float, default=1.0)
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--no-strict", action="store_true", help="Do not exit non-zero on failed checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out_dir = _as_project_path(root, str(args.out_dir))
    assert out_dir is not None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    args.sample_frame_dir = run_dir / "frames"

    cases = _load_manifest(args.manifest)
    audited = [_audit_case(root, case, args) for case in cases]
    failed = [case for case in audited if case["status"] == "FAIL"]
    warned = [case for case in audited if case["status"] == "WARN"]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "manifest": str(args.manifest.resolve()) if args.manifest else "default",
        "overall_status": "FAIL" if failed else ("WARN" if warned else "PASS"),
        "num_cases": len(audited),
        "num_failed": len(failed),
        "num_warned": len(warned),
        "cases": audited,
    }
    json_path = run_dir / f"{args.output_prefix}.json"
    md_path = run_dir / f"{args.output_prefix}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)

    print(json.dumps({"overall_status": report["overall_status"], "json": str(json_path), "markdown": str(md_path)}, indent=2))
    if failed and not args.no_strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
