#!/usr/bin/env python3
"""Pretrain the temporal point-cloud student on an exported dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, random_split

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from simtoolreal_lab.teacher_student import PointTemporalStudent, default_dataset_spec, validate_student_batch

COMPACT_LABEL_INDICES = {
    "true_grasp": 15,
    "grasp_seen": 16,
    "success": 17,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional PointTemporalStudent checkpoint to fine-tune.")
    parser.add_argument("--flow-target", choices=("velocity", "delta"), default="velocity")
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--flow-loss-weight", type=float, default=0.25)
    parser.add_argument("--affordance-loss-weight", type=float, default=0.1)
    parser.add_argument("--privileged-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--hold-loss-weight",
        type=float,
        default=0.0,
        help="Supervise the learned deployable hold hand target head on positive hold samples.",
    )
    parser.add_argument(
        "--hold-gate-loss-weight",
        type=float,
        default=0.0,
        help="Binary supervision weight for the learned hold/reflex gate logit.",
    )
    parser.add_argument(
        "--hold-label-source",
        choices=("true_grasp_rel_vel", "grasp_seen_rel_vel", "success", "none"),
        default="true_grasp_rel_vel",
        help="Fallback label source when the dataset has no hold_mask tensor.",
    )
    parser.add_argument(
        "--hold-label-rel-vel-threshold",
        type=float,
        default=0.2,
        help="Fallback object-palm relative velocity threshold for *_rel_vel hold labels.",
    )
    parser.add_argument(
        "--hold-target-mode",
        choices=("absolute", "residual"),
        default="absolute",
        help="Train hold_target as an absolute hand action or as a residual around --hold-anchor-target.",
    )
    parser.add_argument(
        "--hold-anchor-target",
        type=float,
        nargs="*",
        default=None,
        help="Hand action anchor used when --hold-target-mode=residual. Must have action_dim-7 values.",
    )
    parser.add_argument(
        "--train-action-head-only",
        action="store_true",
        help="Freeze point/global encoders and auxiliary heads; only update the action head.",
    )
    parser.add_argument(
        "--train-hold-head-only",
        action="store_true",
        help="Freeze all non-hold-head parameters; useful for adding a hold/reflex head to old checkpoints.",
    )
    parser.add_argument(
        "--action-sample-weight-mode",
        choices=("none", "true_grasp", "grasp_seen", "success"),
        default="none",
        help="Upweight action imitation on samples with selected compact privileged label.",
    )
    parser.add_argument(
        "--action-positive-weight",
        type=float,
        default=3.0,
        help="Action-loss multiplier for positive samples selected by --action-sample-weight-mode.",
    )
    parser.add_argument(
        "--train-sampler-label",
        choices=("none", "true_grasp", "grasp_seen", "success", "episode_success"),
        default="none",
        help="Oversample training batches with the selected positive label.",
    )
    parser.add_argument(
        "--train-positive-sample-weight",
        type=float,
        default=1.0,
        help="WeightedRandomSampler weight for positive samples selected by --train-sampler-label.",
    )
    parser.add_argument(
        "--train-sampler-num-samples-multiplier",
        type=float,
        default=1.0,
        help="Number of samples drawn per epoch, as a multiple of the train split size.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument(
        "--wandb-metrics-only",
        action="store_true",
        help="Log scalar metrics only; omit dataset paths/metadata, checkpoint files, and console capture.",
    )
    parser.add_argument(
        "--wandb-save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload best/last checkpoints to W&B when W&B is enabled.",
    )
    return parser.parse_args()


def _trace(message: str) -> None:
    print(f"[STUDENT] {message}", flush=True)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.float()
    return ((pred - target).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def action_sample_weights(privileged: torch.Tensor, mode: str, positive_weight: float) -> torch.Tensor:
    """Build per-sample action imitation weights from compact privileged labels."""

    weights = torch.ones(privileged.shape[0], dtype=privileged.dtype, device=privileged.device)
    if mode == "none":
        return weights
    label_index = COMPACT_LABEL_INDICES[mode]
    if privileged.shape[-1] <= label_index:
        return weights
    positive = (privileged[:, label_index] > 0.5).to(dtype=privileged.dtype)
    return weights + positive * (float(positive_weight) - 1.0)


def replay_positive_labels(payload: dict, privileged: torch.Tensor, label: str) -> torch.Tensor | None:
    if label == "none":
        return None
    if label == "episode_success" and "episode_success" in payload:
        return payload["episode_success"].float().view(-1) > 0.5
    compact_label = "success" if label == "episode_success" else label
    label_index = COMPACT_LABEL_INDICES[compact_label]
    if privileged.shape[-1] <= label_index:
        return None
    return privileged[:, label_index].float().view(-1) > 0.5


def _maybe_init_wandb(args: argparse.Namespace, metadata: dict, spec):
    if not args.wandb_project or args.wandb_mode == "disabled":
        return None
    import wandb

    if args.wandb_metrics_only:
        config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "flow_target": args.flow_target,
            "history": spec.history,
            "num_object_points": spec.num_object_points,
            "proprio_dim": spec.proprio_dim,
            "action_dim": spec.action_dim,
            "compact_privileged_dim": spec.compact_privileged_dim,
            "action_sample_weight_mode": args.action_sample_weight_mode,
            "action_positive_weight": args.action_positive_weight,
            "train_sampler_label": args.train_sampler_label,
            "train_positive_sample_weight": args.train_positive_sample_weight,
            "train_sampler_num_samples_multiplier": args.train_sampler_num_samples_multiplier,
            "hold_loss_weight": args.hold_loss_weight,
            "hold_gate_loss_weight": args.hold_gate_loss_weight,
            "hold_label_source": args.hold_label_source,
            "hold_label_rel_vel_threshold": args.hold_label_rel_vel_threshold,
            "hold_target_mode": args.hold_target_mode,
            "hold_anchor_target": args.hold_anchor_target,
        }
    else:
        config = {
            "dataset": str(args.dataset),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "flow_target": args.flow_target,
            "action_sample_weight_mode": args.action_sample_weight_mode,
            "action_positive_weight": args.action_positive_weight,
            "train_sampler_label": args.train_sampler_label,
            "train_positive_sample_weight": args.train_positive_sample_weight,
            "train_sampler_num_samples_multiplier": args.train_sampler_num_samples_multiplier,
            "hold_loss_weight": args.hold_loss_weight,
            "hold_gate_loss_weight": args.hold_gate_loss_weight,
            "hold_label_source": args.hold_label_source,
            "hold_label_rel_vel_threshold": args.hold_label_rel_vel_threshold,
            "hold_target_mode": args.hold_target_mode,
            "hold_anchor_target": args.hold_anchor_target,
            "metadata": metadata,
        }
    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name or f"student_pretrain_{args.dataset.stem}",
        "group": args.wandb_group,
        "tags": args.wandb_tags,
        "mode": args.wandb_mode,
        "dir": str(args.out_dir),
        "job_type": "student_pretrain",
        "config": config,
    }
    if args.wandb_metrics_only:
        init_kwargs["settings"] = wandb.Settings(console="off", save_code=False)
    return wandb.init(**init_kwargs)


def hold_mask_from_privileged(privileged: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    source = str(args.hold_label_source)
    if source == "none":
        return torch.zeros(privileged.shape[0], dtype=privileged.dtype)
    if source == "success":
        if privileged.shape[-1] <= COMPACT_LABEL_INDICES["success"]:
            return torch.zeros(privileged.shape[0], dtype=privileged.dtype)
        return (privileged[:, COMPACT_LABEL_INDICES["success"]] > 0.5).to(dtype=privileged.dtype)
    if privileged.shape[-1] <= COMPACT_LABEL_INDICES["grasp_seen"]:
        return torch.zeros(privileged.shape[0], dtype=privileged.dtype)
    rel_vel = privileged[:, 14]
    label_key = "true_grasp" if source == "true_grasp_rel_vel" else "grasp_seen"
    grasp = privileged[:, COMPACT_LABEL_INDICES[label_key]] > 0.5
    stable = rel_vel < float(args.hold_label_rel_vel_threshold)
    return (grasp & stable).to(dtype=privileged.dtype)


def resolve_hold_tensors(payload: dict, compact_privileged: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    target = payload["target"].float()
    fallback_target = target[:, 7:].contiguous()
    hold_target = payload.get("hold_target")
    if hold_target is None:
        hold_target = fallback_target
    else:
        hold_target = hold_target.float()
        if tuple(hold_target.shape) != tuple(fallback_target.shape):
            raise RuntimeError(
                f"hold_target shape mismatch: expected {tuple(fallback_target.shape)}, got {tuple(hold_target.shape)}"
            )
    hold_mask = payload.get("hold_mask")
    if hold_mask is None:
        hold_mask = hold_mask_from_privileged(compact_privileged, args)
    else:
        hold_mask = hold_mask.float().view(-1)
        if hold_mask.shape[0] != target.shape[0]:
            raise RuntimeError(f"hold_mask length mismatch: expected {target.shape[0]}, got {hold_mask.shape[0]}")
    if args.hold_target_mode == "residual":
        anchor_values = args.hold_anchor_target or []
        if len(anchor_values) != fallback_target.shape[-1]:
            raise RuntimeError(
                f"--hold-anchor-target has {len(anchor_values)} values, expected hand_dim={fallback_target.shape[-1]}."
            )
        anchor = torch.tensor(anchor_values, dtype=hold_target.dtype).view(1, -1)
        hold_target = hold_target - anchor
    return hold_target, hold_mask


def main() -> None:
    args = parse_args()
    payload = torch.load(args.dataset, map_location="cpu", weights_only=False)
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
        raise RuntimeError("Dataset failed schema validation: " + "; ".join(errors))

    flow_key = "point_flow_velocity" if args.flow_target == "velocity" else "point_flow_delta"
    compact_privileged = payload["compact_privileged"].float()
    hold_target, hold_mask = resolve_hold_tensors(payload, compact_privileged, args)
    hold_positive_count = int((hold_mask > 0.5).sum().item())
    _trace(f"hold positives: {hold_positive_count}/{hold_mask.numel()} source={args.hold_label_source}")
    dataset = TensorDataset(
        payload["pointcloud_seq"].float(),
        payload["pointcloud_valid_seq"].float(),
        payload["proprio_seq"].float(),
        payload["target"].float(),
        payload[flow_key].float(),
        payload["affordance_region_labels"].float(),
        compact_privileged,
        hold_target.float(),
        hold_mask.float(),
    )
    val_count = int(round(len(dataset) * args.val_fraction))
    val_count = min(max(val_count, 1), max(len(dataset) - 1, 1))
    train_count = len(dataset) - val_count
    train_set, val_set = random_split(dataset, [train_count, val_count], generator=torch.Generator().manual_seed(42))
    positive_labels = replay_positive_labels(payload, compact_privileged, args.train_sampler_label)
    train_sampler = None
    sampler_stats = {
        "label": args.train_sampler_label,
        "positive_weight": float(args.train_positive_sample_weight),
        "num_samples_multiplier": float(args.train_sampler_num_samples_multiplier),
        "enabled": False,
    }
    if positive_labels is not None and float(args.train_positive_sample_weight) > 1.0:
        train_indices = torch.as_tensor(train_set.indices, dtype=torch.long)
        train_positive = positive_labels[train_indices]
        positive_count = int(train_positive.sum().item())
        train_weights = torch.ones(train_count, dtype=torch.double)
        train_weights[train_positive.cpu()] = float(args.train_positive_sample_weight)
        sampler_num_samples = max(1, int(round(train_count * float(args.train_sampler_num_samples_multiplier))))
        if positive_count > 0 and sampler_num_samples > 0:
            train_sampler = WeightedRandomSampler(
                train_weights,
                num_samples=sampler_num_samples,
                replacement=True,
                generator=torch.Generator().manual_seed(42),
            )
            expected_positive_fraction = float(
                train_weights[train_positive.cpu()].sum().item() / train_weights.sum().clamp_min(1.0).item()
            )
            sampler_stats |= {
                "enabled": True,
                "train_positive_count": positive_count,
                "train_count": int(train_count),
                "num_samples": int(sampler_num_samples),
                "expected_positive_fraction": expected_positive_fraction,
            }
            _trace(
                f"enabled train sampler label={args.train_sampler_label} "
                f"positive={positive_count}/{train_count} "
                f"weight={float(args.train_positive_sample_weight):.3f} "
                f"expected_positive_fraction={expected_positive_fraction:.3f}"
            )
        else:
            _trace(f"train sampler label={args.train_sampler_label} has no positives in train split; using shuffle")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = PointTemporalStudent(
        history=spec.history,
        proprio_dim=spec.proprio_dim,
        action_dim=spec.action_dim,
        privileged_dim=spec.compact_privileged_dim,
    ).to(device)
    if args.init_checkpoint is not None:
        init_payload = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if not isinstance(init_payload, dict) or "model_state_dict" not in init_payload:
            raise RuntimeError(f"--init-checkpoint is not a PointTemporalStudent checkpoint: {args.init_checkpoint}")
        init_spec = dict(init_payload.get("spec", {}))
        expected = {
            "history": spec.history,
            "num_object_points": spec.num_object_points,
            "proprio_dim": spec.proprio_dim,
            "action_dim": spec.action_dim,
            "compact_privileged_dim": spec.compact_privileged_dim,
        }
        mismatches = [
            f"{key}: init={init_spec.get(key)!r} current={value!r}"
            for key, value in expected.items()
            if key in init_spec and init_spec.get(key) != value
        ]
        if mismatches:
            raise RuntimeError("Init checkpoint spec mismatch: " + "; ".join(mismatches))
        missing, unexpected = model.load_state_dict(init_payload["model_state_dict"], strict=False)
        allowed_missing_prefixes = ("hold_head.", "hold_gate_head.")
        bad_missing = [name for name in missing if not name.startswith(allowed_missing_prefixes)]
        if bad_missing or unexpected:
            raise RuntimeError(
                "Init checkpoint state mismatch: "
                f"missing={bad_missing}, unexpected={list(unexpected)}"
            )
        if missing:
            _trace(f"init checkpoint missing optional hold head weights; initialized {list(missing)}")
        _trace(f"loaded init checkpoint: {args.init_checkpoint}")
    if args.train_action_head_only or args.train_hold_head_only:
        trainable_prefixes: list[str] = []
        if args.train_action_head_only:
            trainable_prefixes.append("action_head.")
        if args.train_hold_head_only:
            trainable_prefixes.extend(["hold_head.", "hold_gate_head."])
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(any(name.startswith(prefix) for prefix in trainable_prefixes))
        _trace(f"frozen parameters outside prefixes={trainable_prefixes}")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after applying freeze options.")
    _trace(f"trainable parameters: {trainable_count}/{total_count}")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=1.0e-4)
    wandb_run = _maybe_init_wandb(args, metadata, spec)

    def run_epoch(loader: DataLoader, train: bool) -> dict[str, float]:
        model.train(train)
        totals = {
            "loss": 0.0,
            "action": 0.0,
            "flow": 0.0,
            "affordance": 0.0,
            "privileged": 0.0,
            "hold": 0.0,
            "hold_gate": 0.0,
            "hold_positive_fraction": 0.0,
        }
        count = 0
        for pc, valid, proprio, action, flow, affordance, privileged, hold_target_batch, hold_mask_batch in loader:
            pc = pc.to(device)
            valid = valid.to(device)
            proprio = proprio.to(device)
            action = action.to(device)
            flow = flow.to(device)
            affordance = affordance.to(device)
            privileged = privileged.to(device)
            hold_target_batch = hold_target_batch.to(device)
            hold_mask_batch = hold_mask_batch.to(device).view(-1)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                out = model(pc, valid, proprio)
                current_valid = valid[:, -1]
                action_error = F.smooth_l1_loss(out["action"], action, reduction="none").mean(dim=-1)
                action_weights = action_sample_weights(
                    privileged,
                    args.action_sample_weight_mode,
                    args.action_positive_weight,
                )
                action_loss = (action_error * action_weights).sum() / action_weights.sum().clamp_min(1.0)
                flow_loss = masked_mse(out["flow"], flow, current_valid)
                aff_mask = current_valid * (affordance >= 0.0).float()
                affordance_loss = masked_bce(out["affordance_logits"], affordance.clamp(0.0, 1.0), aff_mask)
                privileged_loss = F.mse_loss(out["privileged"], privileged)
                if out["hold_target"].shape[-1] > 0:
                    hold_error = F.smooth_l1_loss(
                        out["hold_target"],
                        hold_target_batch,
                        reduction="none",
                    ).mean(dim=-1)
                    hold_loss = (hold_error * hold_mask_batch).sum() / hold_mask_batch.sum().clamp_min(1.0)
                else:
                    hold_loss = torch.zeros((), device=device)
                hold_gate_loss = F.binary_cross_entropy_with_logits(out["hold_logits"], hold_mask_batch.float())
                loss = (
                    args.action_loss_weight * action_loss
                    + args.flow_loss_weight * flow_loss
                    + args.affordance_loss_weight * affordance_loss
                    + args.privileged_loss_weight * privileged_loss
                    + args.hold_loss_weight * hold_loss
                    + args.hold_gate_loss_weight * hold_gate_loss
                )
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            batch_count = pc.shape[0]
            totals["loss"] += float(loss.detach().cpu()) * batch_count
            totals["action"] += float(action_loss.detach().cpu()) * batch_count
            totals["flow"] += float(flow_loss.detach().cpu()) * batch_count
            totals["affordance"] += float(affordance_loss.detach().cpu()) * batch_count
            totals["privileged"] += float(privileged_loss.detach().cpu()) * batch_count
            totals["hold"] += float(hold_loss.detach().cpu()) * batch_count
            totals["hold_gate"] += float(hold_gate_loss.detach().cpu()) * batch_count
            totals["hold_positive_fraction"] += float((hold_mask_batch > 0.5).float().mean().detach().cpu()) * batch_count
            count += batch_count
        return {key: value / max(count, 1) for key, value in totals.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = args.out_dir / "student_pretrain_best.pt"
    last_path = args.out_dir / "student_pretrain_last.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(train_loader, train=True)
        val_metrics = run_epoch(val_loader, train=False)
        _trace(
            f"epoch={epoch:04d} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_action={val_metrics['action']:.6f}"
        )
        if wandb_run is not None:
            import wandb

            wandb.log({f"train/{k}": v for k, v in train_metrics.items()} | {f"val/{k}": v for k, v in val_metrics.items()} | {"epoch": epoch})
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "spec": {
                "history": spec.history,
                "num_object_points": spec.num_object_points,
                "proprio_dim": spec.proprio_dim,
                "action_dim": spec.action_dim,
                "arm_dim": getattr(model, "arm_dim", 7),
                "hand_dim": getattr(model, "hand_dim", max(spec.action_dim - 7, 0)),
                "compact_privileged_dim": spec.compact_privileged_dim,
            },
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
            "train_sampler": sampler_stats,
            "train_action_head_only": bool(args.train_action_head_only),
            "train_hold_head_only": bool(args.train_hold_head_only),
            "hold_loss_weight": float(args.hold_loss_weight),
            "hold_gate_loss_weight": float(args.hold_gate_loss_weight),
            "hold_label_source": args.hold_label_source,
            "hold_label_rel_vel_threshold": float(args.hold_label_rel_vel_threshold),
            "hold_target_mode": args.hold_target_mode,
            "hold_anchor_target": (
                [float(value) for value in args.hold_anchor_target]
                if args.hold_anchor_target is not None
                else None
            ),
        }
        torch.save(checkpoint, last_path)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, best_path)

    if wandb_run is not None and args.wandb_save_checkpoints and not args.wandb_metrics_only:
        import wandb

        wandb.save(str(best_path), base_path=str(args.out_dir))
        wandb.save(str(last_path), base_path=str(args.out_dir))
    if wandb_run is not None:
        wandb_run.finish()
    _trace(f"saved best checkpoint: {best_path}")
    _trace(f"saved last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
