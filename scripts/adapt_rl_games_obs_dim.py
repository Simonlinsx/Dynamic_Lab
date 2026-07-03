#!/usr/bin/env python3
"""Pad an rl_games checkpoint to a larger observation dimension."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _pad_vector(value: torch.Tensor, target_dim: int, fill: float) -> torch.Tensor:
    if value.ndim != 1:
        raise ValueError(f"expected a vector, got shape {tuple(value.shape)}")
    current_dim = value.shape[0]
    if current_dim == target_dim:
        return value.clone()
    if current_dim > target_dim:
        raise ValueError(f"cannot shrink vector from {current_dim} to {target_dim}")
    padded = torch.full((target_dim,), fill, dtype=value.dtype, device=value.device)
    padded[:current_dim] = value
    return padded


def _pad_first_layer(weight: torch.Tensor, target_dim: int) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(weight.shape)}")
    out_dim, current_dim = weight.shape
    if current_dim == target_dim:
        return weight.clone()
    if current_dim > target_dim:
        raise ValueError(f"cannot shrink first layer from {current_dim} to {target_dim}")
    padded = torch.zeros((out_dim, target_dim), dtype=weight.dtype, device=weight.device)
    padded[:, :current_dim] = weight
    return padded


def _pad_optimizer_state(optimizer: dict, source_dim: int, target_dim: int) -> None:
    for state in optimizer.get("state", {}).values():
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                continue
            if value.shape[1] == source_dim:
                state[key] = _pad_first_layer(value, target_dim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source rl_games checkpoint.")
    parser.add_argument("--output", required=True, help="Output padded checkpoint.")
    parser.add_argument("--target-obs-dim", type=int, required=True, help="New observation dimension.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    model = checkpoint.get("model")
    if not isinstance(model, dict):
        raise ValueError("checkpoint does not contain a rl_games 'model' state dict")

    first_layer_key = "a2c_network.actor_mlp.0.weight"
    if first_layer_key not in model:
        raise KeyError(f"missing first layer key: {first_layer_key}")
    source_dim = int(model[first_layer_key].shape[1])
    target_dim = int(args.target_obs_dim)

    model[first_layer_key] = _pad_first_layer(model[first_layer_key], target_dim)
    if "running_mean_std.running_mean" in model:
        model["running_mean_std.running_mean"] = _pad_vector(
            model["running_mean_std.running_mean"], target_dim, fill=0.0
        )
    if "running_mean_std.running_var" in model:
        model["running_mean_std.running_var"] = _pad_vector(
            model["running_mean_std.running_var"], target_dim, fill=1.0
        )

    optimizer = checkpoint.get("optimizer")
    if isinstance(optimizer, dict):
        _pad_optimizer_state(optimizer, source_dim, target_dim)
    if "last_mean_rewards" in checkpoint and hasattr(checkpoint["last_mean_rewards"], "item"):
        checkpoint["last_mean_rewards"] = float(checkpoint["last_mean_rewards"].item())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"adapted obs dim {source_dim} -> {target_dim}: {output_path}")


if __name__ == "__main__":
    main()
