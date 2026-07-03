#!/usr/bin/env python3
"""Backfill an rl-games TensorBoard run and eval summaries into W&B."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", required=True, help="rl-games run directory containing summaries/ and nn/.")
parser.add_argument("--wandb-project", required=True, help="W&B project to log into.")
parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team.")
parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name. Defaults to the run directory name.")
parser.add_argument("--wandb-group", default=None, help="Optional W&B group.")
parser.add_argument("--wandb-tags", nargs="*", default=None, help="Optional W&B tags.")
parser.add_argument(
    "--wandb-mode",
    choices=("online", "offline", "disabled"),
    default=os.environ.get("WANDB_MODE", "online"),
    help="W&B mode. Defaults to WANDB_MODE or online.",
)
parser.add_argument(
    "--eval-summary",
    action="append",
    default=[],
    help="Path to an evaluate_rl_games.py summary.json. Can be passed multiple times.",
)
parser.add_argument(
    "--eval-summary-glob",
    action="append",
    default=[],
    help="Glob for evaluate_rl_games.py summary.json files. Can be passed multiple times.",
)
args = parser.parse_args()


def _load_tensorboard_scalars(run_dir: Path) -> tuple[list[str], dict[int, dict[str, float]]]:
    event_files = sorted((run_dir / "summaries").glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {run_dir / 'summaries'}")

    records: dict[int, dict[str, float]] = defaultdict(dict)
    tags: set[str] = set()
    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            tags.add(tag)
            for scalar in accumulator.Scalars(tag):
                records[int(scalar.step)][tag] = float(scalar.value)
    return sorted(tags), dict(sorted(records.items()))


def _epoch_from_summary(summary: dict, summary_path: Path) -> int | None:
    candidates = [
        str(summary.get("checkpoint", "")),
        str(summary_path),
    ]
    for candidate in candidates:
        match = re.search(r"_ep_(\d+)_", candidate)
        if match:
            return int(match.group(1))
        match = re.search(r"[/_-]ep(\d+)(?:[/_-]|$)", candidate)
        if match:
            return int(match.group(1))
    return None


def _numeric_eval_metrics(summary: dict) -> dict[str, float]:
    eval_summary = summary.get("eval", {})
    metrics: dict[str, float] = {
        "eval/passed": float(bool(summary.get("passed", False))),
        "eval/success_threshold": float(summary.get("success_threshold", 0.0)),
    }
    for key, value in eval_summary.items():
        if isinstance(value, bool):
            metrics[f"eval/{key}"] = float(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"eval/{key}"] = float(value)
    for key, value in eval_summary.get("last_log", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"eval/last_log/{key}"] = float(value)
    return metrics


def _expand_eval_summary_paths() -> list[Path]:
    paths = [Path(path).expanduser().resolve() for path in args.eval_summary]
    for pattern in args.eval_summary_glob:
        paths.extend(sorted(Path().glob(pattern) if not pattern.startswith("/") else Path("/").glob(pattern[1:])))
    unique_paths = []
    seen = set()
    for path in paths:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return unique_paths


def main() -> None:
    if args.wandb_mode == "disabled":
        raise SystemExit("--wandb-mode=disabled leaves nothing to export")

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed in this environment.") from exc

    run_dir = Path(args.run_dir).expanduser().resolve()
    tags, records = _load_tensorboard_scalars(run_dir)

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or run_dir.name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        dir=str(run_dir),
        job_type="backfill",
        config={"run_dir": str(run_dir), "scalar_tags": tags},
    )
    try:
        wandb.define_metric("train/global_step")
        for tag in tags:
            wandb.define_metric(tag, step_metric="train/global_step")
        wandb.define_metric("eval/epoch")
        wandb.define_metric("eval/*", step_metric="eval/epoch")

        for global_step, scalars in records.items():
            wandb.log({"train/global_step": global_step, **scalars})

        for summary_path in _expand_eval_summary_paths():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            epoch = _epoch_from_summary(summary, summary_path)
            metrics = _numeric_eval_metrics(summary)
            if epoch is not None:
                metrics["eval/epoch"] = float(epoch)
            wandb.log(metrics)
            wandb.save(str(summary_path), base_path=str(summary_path.parent))

        print(f"Exported {len(records)} TensorBoard steps and {len(_expand_eval_summary_paths())} eval summaries.")
        print(f"W&B run: {wandb_run.url or wandb_run.name}")
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
