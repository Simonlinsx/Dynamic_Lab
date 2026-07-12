#!/usr/bin/env python3
"""Evaluate a teacher checkpoint series under one fixed protocol."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/data1/linsixu/miniconda3/envs/dynamicvla-isaac/bin/python")
EPOCH_PATTERN = re.compile(r"_ep_(\d+)_rew_")


def _trace(message: str) -> None:
    print(f"[CHECKPOINT-AUDIT] {message}", flush=True)


def _checkpoint_epoch(path: Path) -> int | None:
    match = EPOCH_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def _discover_checkpoints(
    checkpoint_dir: Path, epochs: list[int] | None, include_final: bool
) -> list[tuple[str, int | None, Path]]:
    paths = sorted(checkpoint_dir.glob("*.pth"))
    epoch_paths: dict[int, list[Path]] = {}
    final_paths: list[Path] = []
    for path in paths:
        epoch = _checkpoint_epoch(path)
        if epoch is None:
            final_paths.append(path)
        else:
            epoch_paths.setdefault(epoch, []).append(path)

    selected = []
    requested_epochs = sorted(epoch_paths) if epochs is None else epochs
    for epoch in requested_epochs:
        candidates = epoch_paths.get(int(epoch), [])
        if not candidates:
            _trace(f"epoch {epoch} is absent from {checkpoint_dir}")
            continue
        # rl_games may leave a second filename with escaped reward punctuation.
        checkpoint = min(candidates, key=lambda path: (path.name.count("__"), len(path.name)))
        selected.append((f"ep_{epoch:04d}", int(epoch), checkpoint))
    if include_final and final_paths:
        checkpoint = min(final_paths, key=lambda path: (path.name.count("__"), len(path.name)))
        selected.append(("final", None, checkpoint))
    return selected


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _find_summary(eval_dir: Path) -> Path | None:
    candidates = list(eval_dir.glob("*/summary.json"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _extract_record(label: str, epoch: int | None, checkpoint: Path, summary_path: Path) -> dict:
    summary = _load_json(summary_path)
    evaluation = summary.get("eval", {})
    funnel = evaluation.get("failure_funnel", {})
    failure_rates = dict(funnel.get("primary_failure_rates", {}))
    failure_rates.pop("success", None)
    primary_failure = max(failure_rates, key=failure_rates.get) if failure_rates else None
    return {
        "label": label,
        "epoch": epoch,
        "checkpoint": str(checkpoint.resolve()),
        "summary": str(summary_path.resolve()),
        "episodes": int(evaluation.get("episodes", 0)),
        "success_rate": float(evaluation.get("success_rate", 0.0)),
        "strict_grasp_rate": float(
            funnel.get("stage_rates", {}).get(
                "strict_true_grasp", evaluation.get("true_grasp_episode_rate", 0.0)
            )
        ),
        "lifted_rate": float(evaluation.get("lifted_episode_rate", 0.0)),
        "stable_hold_rate": float(evaluation.get("stable_hold_episode_rate", 0.0)),
        "primary_failure": primary_failure,
        "primary_failure_rate": float(failure_rates.get(primary_failure, 0.0)),
        "conversion_rates": funnel.get("conversion_rates", {}),
    }


def _write_markdown(path: Path, payload: dict) -> None:
    records = sorted(payload["records"], key=lambda record: record["success_rate"], reverse=True)
    lines = [
        "# Teacher Checkpoint Audit",
        "",
        f"Task: `{payload['task']}`",
        "",
        "| Rank | Checkpoint | Episodes | Success | Strict grasp | Lifted | Stable | Main failure |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, record in enumerate(records, start=1):
        failure = record["primary_failure"] or "n/a"
        failure = f"{failure} ({record['primary_failure_rate']:.1%})"
        lines.append(
            f"| {rank} | `{record['label']}` | {record['episodes']} | "
            f"{record['success_rate']:.1%} | {record['strict_grasp_rate']:.1%} | "
            f"{record['lifted_rate']:.1%} | {record['stable_hold_rate']:.1%} | {failure} |"
        )
    lines.extend(["", f"Best checkpoint: `{payload.get('best_checkpoint', 'n/a')}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _init_wandb(args, checkpoints: list[tuple[str, int | None, Path]]):
    if args.wandb_mode == "disabled" or not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requested, but wandb is unavailable.") from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or f"checkpoint_audit_{args.checkpoint_dir.name}",
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        dir=str(args.output_dir),
        job_type="checkpoint_audit",
        config={
            "task": args.task,
            "checkpoint_dir": str(args.checkpoint_dir),
            "checkpoints": [str(path) for _, _, path in checkpoints],
            "num_envs": args.num_envs,
            "episodes": args.episodes,
            "seed": args.seed,
            "extra_eval_args": args.extra_eval_args,
        },
    )


def _log_record_to_wandb(wandb_run, record: dict, index: int) -> None:
    if wandb_run is None:
        return
    import wandb

    metrics = {
        "audit/checkpoint_index": index,
        "audit/epoch": record["epoch"] if record["epoch"] is not None else -1,
        "audit/success_rate": record["success_rate"],
        "audit/strict_grasp_rate": record["strict_grasp_rate"],
        "audit/lifted_rate": record["lifted_rate"],
        "audit/stable_hold_rate": record["stable_hold_rate"],
        "audit/primary_failure_rate": record["primary_failure_rate"],
    }
    for name, value in record["conversion_rates"].items():
        metrics[f"audit/conversion/{name}"] = float(value)
    wandb.log(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="*", default=None)
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--evaluate-script", type=Path, default=REPO_ROOT / "scripts/evaluate_rl_games.py")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-project", default="simtoolreal_lab")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="teacher_checkpoint_audit")
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument(
        "--extra-eval-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments forwarded verbatim to evaluate_rl_games.py; keep this option last.",
    )
    args = parser.parse_args()

    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.evaluate_script = args.evaluate_script.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)
    checkpoints = _discover_checkpoints(args.checkpoint_dir, args.epochs, args.include_final)
    if not checkpoints:
        raise RuntimeError(f"No requested checkpoints found under {args.checkpoint_dir}")

    state_path = args.output_dir / "audit_state.json"
    state = _load_json(state_path) if state_path.exists() else {"records": []}
    records_by_checkpoint = {record["checkpoint"]: record for record in state.get("records", [])}
    wandb_run = _init_wandb(args, checkpoints)
    try:
        for index, (label, epoch, checkpoint) in enumerate(checkpoints):
            resolved_checkpoint = str(checkpoint.resolve())
            previous = records_by_checkpoint.get(resolved_checkpoint)
            if previous and Path(previous.get("summary", "")).exists():
                _trace(f"resume: {label} already evaluated at {previous['success_rate']:.3f}")
                _log_record_to_wandb(wandb_run, previous, index)
                continue

            eval_dir = args.output_dir / "eval" / label
            eval_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(args.python),
                str(args.evaluate_script),
                "--task",
                args.task,
                "--checkpoint",
                str(checkpoint),
                "--num-envs",
                str(args.num_envs),
                "--episodes",
                str(args.episodes),
                "--seed",
                str(args.seed),
                "--success-threshold",
                "0.0",
                "--save-success-videos",
                "0",
                "--wandb-mode",
                "disabled",
                "--output-dir",
                str(eval_dir),
                "--headless",
                "--device",
                args.device,
                *args.extra_eval_args,
            ]
            log_path = args.output_dir / "logs" / f"{label}.log"
            _trace(f"evaluating {label}: {checkpoint.name}")
            with log_path.open("w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            summary_path = _find_summary(eval_dir)
            if result.returncode != 0 or summary_path is None:
                raise RuntimeError(
                    f"Evaluation failed for {checkpoint} with code {result.returncode}; see {log_path}"
                )
            record = _extract_record(label, epoch, checkpoint, summary_path)
            records_by_checkpoint[resolved_checkpoint] = record
            state["records"] = list(records_by_checkpoint.values())
            _write_json(state_path, state)
            _log_record_to_wandb(wandb_run, record, index)
            _trace(
                f"{label}: success={record['success_rate']:.3f} "
                f"stable={record['stable_hold_rate']:.3f} main_failure={record['primary_failure']}"
            )

        selected_checkpoints = [str(checkpoint.resolve()) for _, _, checkpoint in checkpoints]
        records = [
            records_by_checkpoint[checkpoint]
            for checkpoint in selected_checkpoints
            if checkpoint in records_by_checkpoint
        ]
        if len(records) != len(selected_checkpoints):
            raise RuntimeError("Checkpoint audit finished with incomplete selected records")
        best = max(records, key=lambda record: record["success_rate"])
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": args.task,
            "checkpoint_dir": str(args.checkpoint_dir),
            "protocol": {
                "num_envs": args.num_envs,
                "episodes": args.episodes,
                "seed": args.seed,
                "extra_eval_args": args.extra_eval_args,
            },
            "best_checkpoint": best["checkpoint"],
            "best_success_rate": best["success_rate"],
            "records": records,
        }
        report_path = args.output_dir / "checkpoint_audit.json"
        markdown_path = args.output_dir / "checkpoint_audit.md"
        _write_json(report_path, report)
        _write_markdown(markdown_path, report)
        _trace(f"best={best['label']} success={best['success_rate']:.3f}")
        _trace(f"report={report_path}")
        if wandb_run is not None:
            import wandb

            table = wandb.Table(
                columns=[
                    "label",
                    "epoch",
                    "success_rate",
                    "strict_grasp_rate",
                    "lifted_rate",
                    "stable_hold_rate",
                    "primary_failure",
                    "primary_failure_rate",
                    "checkpoint",
                ],
                data=[
                    [
                        record["label"],
                        record["epoch"] if record["epoch"] is not None else -1,
                        record["success_rate"],
                        record["strict_grasp_rate"],
                        record["lifted_rate"],
                        record["stable_hold_rate"],
                        record["primary_failure"],
                        record["primary_failure_rate"],
                        record["checkpoint"],
                    ]
                    for record in sorted(records, key=lambda item: item["success_rate"], reverse=True)
                ],
            )
            wandb.log({"audit/results": table})
            wandb.save(str(report_path), base_path=str(args.output_dir))
            wandb.save(str(markdown_path), base_path=str(args.output_dir))
            wandb_run.summary.update(
                {
                    "best_checkpoint": best["checkpoint"],
                    "best_success_rate": best["success_rate"],
                }
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
