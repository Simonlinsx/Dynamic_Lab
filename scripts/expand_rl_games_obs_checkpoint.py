#!/usr/bin/env python3
"""Expand an rl_games MLP checkpoint to a larger observation dimension."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _expand_vector(tensor: torch.Tensor, old_dim: int, new_dim: int, fill: float) -> torch.Tensor:
    if tensor.shape != (old_dim,):
        raise ValueError(f"Expected vector shape {(old_dim,)}, got {tuple(tensor.shape)}")
    out = torch.full((new_dim,), fill, dtype=tensor.dtype, device=tensor.device)
    out[:old_dim] = tensor
    return out


def _expand_input_weight(tensor: torch.Tensor, old_dim: int, new_dim: int) -> torch.Tensor:
    if tensor.ndim != 2 or tensor.shape[1] != old_dim:
        raise ValueError(f"Expected input weight second dim {old_dim}, got {tuple(tensor.shape)}")
    out = torch.zeros((tensor.shape[0], new_dim), dtype=tensor.dtype, device=tensor.device)
    out[:, :old_dim] = tensor
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--old-obs-dim", type=int, default=86)
    parser.add_argument("--new-obs-dim", type=int, default=91)
    args = parser.parse_args()

    if args.new_obs_dim <= args.old_obs_dim:
        raise ValueError("--new-obs-dim must be larger than --old-obs-dim")

    checkpoint = torch.load(args.input, map_location="cpu")
    model = checkpoint["model"]
    old_dim = int(args.old_obs_dim)
    new_dim = int(args.new_obs_dim)

    model["running_mean_std.running_mean"] = _expand_vector(
        model["running_mean_std.running_mean"], old_dim, new_dim, 0.0
    )
    model["running_mean_std.running_var"] = _expand_vector(
        model["running_mean_std.running_var"], old_dim, new_dim, 1.0
    )
    model["a2c_network.actor_mlp.0.weight"] = _expand_input_weight(
        model["a2c_network.actor_mlp.0.weight"], old_dim, new_dim
    )

    optimizer = checkpoint.get("optimizer")
    if isinstance(optimizer, dict) and "state" in optimizer:
        first_layer_state = optimizer["state"].get(1)
        if isinstance(first_layer_state, dict):
            for key in ("exp_avg", "exp_avg_sq"):
                value = first_layer_state.get(key)
                if torch.is_tensor(value):
                    first_layer_state[key] = _expand_input_weight(value, old_dim, new_dim)

    checkpoint.setdefault("metadata", {})
    if isinstance(checkpoint["metadata"], dict):
        checkpoint["metadata"]["expanded_obs_dim"] = {
            "old": old_dim,
            "new": new_dim,
            "source": str(args.input),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved expanded checkpoint: {args.output}")


if __name__ == "__main__":
    main()
