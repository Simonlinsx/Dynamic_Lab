#!/usr/bin/env python3
"""Watch rl_games checkpoints and render periodic training videos."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


EPOCH_RE = re.compile(r"_ep_(\d+)_")


def _trace(message: str) -> None:
    print(f"[VIDEO-WATCHER] {message}", flush=True)


def _checkpoint_epoch(path: Path) -> int | None:
    match = EPOCH_RE.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"processed": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def _iter_ready_checkpoints(args, processed: set[str], processed_epochs: set[int]) -> list[tuple[int, Path]]:
    now = time.time()
    ready_by_epoch: dict[int, Path] = {}
    for checkpoint in sorted(args.checkpoint_dir.glob("*.pth")):
        epoch = _checkpoint_epoch(checkpoint)
        if epoch is None:
            continue
        if epoch in processed_epochs:
            continue
        if epoch < args.start_epoch:
            continue
        if epoch % args.epoch_interval != 0:
            continue
        if str(checkpoint.resolve()) in processed:
            continue
        if now - checkpoint.stat().st_mtime < args.checkpoint_min_age_sec:
            continue
        current = ready_by_epoch.get(epoch)
        if current is None or checkpoint.stat().st_mtime > current.stat().st_mtime:
            ready_by_epoch[epoch] = checkpoint
    return sorted(ready_by_epoch.items())


def _build_eval_command(args, checkpoint: Path, epoch: int) -> list[str]:
    output_dir = args.output_dir / f"ep_{epoch:04d}"
    run_name = f"{args.run_prefix}_video_ep_{epoch:04d}"
    cmd = [
        args.python,
        str(args.evaluate_script),
        "--task",
        args.task,
        "--checkpoint",
        str(checkpoint),
        "--skip-vector-eval",
        "--num-envs",
        str(args.num_envs),
        "--episodes",
        str(args.episodes),
        "--save-success-videos",
        str(args.save_success_videos),
        "--video-attempts",
        str(args.video_attempts),
        "--video-envs",
        str(args.video_envs),
        "--video-max-steps",
        str(args.video_max_steps),
        "--video-post-success-steps",
        str(args.video_post_success_steps),
        "--video-post-success-min-stable-frac",
        str(args.video_post_success_min_stable_frac),
        "--save-rollout-videos-on-failure",
        str(args.save_rollout_videos_on_failure),
        "--allow-rollout-video-fallback",
        "--video-debug-interval",
        str(args.video_debug_interval),
        "--video-camera-resolution",
        str(args.video_camera_resolution[0]),
        str(args.video_camera_resolution[1]),
        "--video-camera-focal-length",
        str(args.video_camera_focal_length),
        "--dynamic-curriculum-alpha",
        str(args.dynamic_curriculum_alpha),
        "--tabletop-asset-curriculum-alpha",
        str(args.tabletop_asset_curriculum_alpha),
        "--tabletop-motion-curriculum-alpha",
        str(args.tabletop_motion_curriculum_alpha),
        "--output-dir",
        str(output_dir),
        "--wandb-mode",
        args.wandb_mode,
        "--headless",
        "--device",
        args.device,
    ]
    if args.video_camera_eye is not None:
        cmd.extend(["--video-camera-eye", *(str(value) for value in args.video_camera_eye)])
    if args.video_camera_target is not None:
        cmd.extend(["--video-camera-target", *(str(value) for value in args.video_camera_target)])
    if args.video_camera_track_object:
        cmd.append("--video-camera-track-object")
    else:
        cmd.append("--no-video-camera-track-object")
    if args.video_camera_update_every_frame:
        cmd.append("--video-camera-update-every-frame")
    else:
        cmd.append("--no-video-camera-update-every-frame")
    if args.video_require_post_success_stable:
        cmd.append("--video-require-post-success-stable")
    if args.video_post_success_require_final_stable:
        cmd.append("--video-post-success-require-final-stable")
    if args.wandb_project:
        cmd.extend(["--wandb-project", args.wandb_project])
    if args.wandb_entity:
        cmd.extend(["--wandb-entity", args.wandb_entity])
    if args.wandb_group:
        cmd.extend(["--wandb-group", args.wandb_group])
    if args.wandb_tags:
        cmd.extend(["--wandb-tags", *args.wandb_tags])
    cmd.extend(["--wandb-run-name", run_name])
    if args.save_rollout_videos_each_failed_attempt:
        cmd.append("--save-rollout-videos-each-failed-attempt")
    if args.kit_args:
        cmd.extend(["--kit_args", args.kit_args])
    return cmd


def _has_success_video(output_dir: Path) -> bool:
    return any(output_dir.glob("*/videos/success_attempt*.mp4"))


def _run_eval_once(args, checkpoint: Path, epoch: int, process_attempt: int, process_attempts: int) -> dict:
    cmd = _build_eval_command(args, checkpoint, epoch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if process_attempts > 1:
        log_path = args.output_dir / f"eval_ep_{epoch:04d}_try_{process_attempt:02d}.log"
    else:
        log_path = args.output_dir / f"eval_ep_{epoch:04d}.log"
    if process_attempt == 0:
        _trace(f"rendering epoch {epoch}: checkpoint={checkpoint}")
    elif not _has_success_video(args.output_dir / f"ep_{epoch:04d}"):
        _trace(f"rendering epoch {epoch} extra try {process_attempt + 1}/{process_attempts}")
    _trace(f"log={log_path}")
    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, cwd=args.repo_root, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - start_time
    record = {
        "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "log": str(log_path),
        "process_attempt": int(process_attempt),
    }
    if proc.returncode == 0:
        _trace(f"epoch {epoch} try {process_attempt + 1} video eval finished in {elapsed:.1f}s")
    else:
        _trace(f"epoch {epoch} try {process_attempt + 1} video eval failed with code {proc.returncode}; see {log_path}")
    return record


def _run_eval(args, checkpoint: Path, epoch: int) -> dict:
    process_attempts = max(1, int(args.eval_process_attempts))
    epoch_output_dir = args.output_dir / f"ep_{epoch:04d}"
    records = []
    for process_attempt in range(process_attempts):
        if process_attempt > 0 and _has_success_video(epoch_output_dir):
            break
        records.append(_run_eval_once(args, checkpoint, epoch, process_attempt, process_attempts))
    success_found = _has_success_video(epoch_output_dir)
    if success_found:
        _trace(f"epoch {epoch} success video found under {epoch_output_dir}")
    elif process_attempts > 1:
        _trace(f"epoch {epoch} no success video after {len(records)} process tries")
    last = records[-1] if records else {}
    return {
        "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "returncode": int(last.get("returncode", 0)),
        "elapsed_sec": float(sum(record.get("elapsed_sec", 0.0) for record in records)),
        "log": str(last.get("log", "")),
        "process_attempts": len(records),
        "success_video_found": bool(success_found),
        "process_records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evaluate-script", type=Path, default=Path(__file__).resolve().parent / "evaluate_rl_games.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--poll-sec", type=float, default=90.0)
    parser.add_argument("--checkpoint-min-age-sec", type=float, default=20.0)
    parser.add_argument("--start-epoch", type=int, default=50)
    parser.add_argument("--epoch-interval", type=int, default=100)
    parser.add_argument("--max-evals", type=int, default=0)
    parser.add_argument(
        "--eval-process-attempts",
        type=int,
        default=1,
        help="Run this many independent evaluator processes per checkpoint unless a success video is found.",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--save-success-videos", type=int, default=1)
    parser.add_argument("--video-attempts", type=int, default=2)
    parser.add_argument(
        "--video-envs",
        type=int,
        default=1,
        help="Rendered env count. Keep this at one for clean third-view videos without neighboring envs.",
    )
    parser.add_argument("--video-max-steps", type=int, default=420)
    parser.add_argument("--video-post-success-steps", type=int, default=30)
    parser.add_argument("--video-require-post-success-stable", action="store_true")
    parser.add_argument("--video-post-success-min-stable-frac", type=float, default=0.90)
    parser.add_argument("--video-post-success-require-final-stable", action="store_true")
    parser.add_argument("--save-rollout-videos-on-failure", type=int, default=1)
    parser.add_argument(
        "--save-rollout-videos-each-failed-attempt",
        action="store_true",
        help="Save debug rollout videos for every failed rendered attempt.",
    )
    parser.add_argument("--video-debug-interval", type=int, default=120)
    parser.add_argument("--video-camera-resolution", type=int, nargs=2, default=(960, 544))
    parser.add_argument("--video-camera-focal-length", type=float, default=18.0)
    parser.add_argument("--video-camera-eye", type=float, nargs=3, default=None)
    parser.add_argument("--video-camera-target", type=float, nargs=3, default=None)
    parser.add_argument("--video-camera-track-object", action="store_true", default=False)
    parser.add_argument("--video-camera-update-every-frame", action="store_true", default=False)
    parser.add_argument("--dynamic-curriculum-alpha", type=float, default=1.0)
    parser.add_argument("--tabletop-asset-curriculum-alpha", type=float, default=1.0)
    parser.add_argument("--tabletop-motion-curriculum-alpha", type=float, default=1.0)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument(
        "--kit-args",
        default=None,
        help="Extra Kit arguments passed through to evaluate_rl_games.py, e.g. renderer GPU settings.",
    )
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.evaluate_script = args.evaluate_script.resolve()
    state_path = args.output_dir / "watcher_state.json"
    state = _load_state(state_path)
    processed = set(state.get("processed", []))
    processed_epochs = {int(record["epoch"]) for record in state.get("records", []) if "epoch" in record}
    eval_count = int(state.get("eval_count", 0))

    _trace(f"watching {args.checkpoint_dir}")
    _trace(f"videos -> {args.output_dir}")
    while True:
        ready = _iter_ready_checkpoints(args, processed, processed_epochs)
        for epoch, checkpoint in ready:
            record = _run_eval(args, checkpoint, epoch)
            resolved = str(checkpoint.resolve())
            processed.add(resolved)
            processed_epochs.add(epoch)
            state.setdefault("records", []).append(record)
            state["processed"] = sorted(processed)
            eval_count += 1
            state["eval_count"] = eval_count
            _save_state(state_path, state)
            if args.max_evals > 0 and eval_count >= args.max_evals:
                _trace(f"max evals reached: {eval_count}")
                return
        time.sleep(max(float(args.poll_sec), 1.0))


if __name__ == "__main__":
    main()
