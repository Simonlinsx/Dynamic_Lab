#!/usr/bin/env python3
"""Pad an rl_games MLP checkpoint with zero-initialized observation columns."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source rl_games checkpoint.")
    parser.add_argument("--output", required=True, type=Path, help="Expanded checkpoint path.")
    parser.add_argument("--target-observation-dim", required=True, type=int)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    model = checkpoint.get("model")
    if not isinstance(model, dict):
        raise ValueError("Checkpoint does not contain an rl_games model state dictionary.")

    mean_key = "running_mean_std.running_mean"
    var_key = "running_mean_std.running_var"
    first_layer_key = "a2c_network.actor_mlp.0.weight"
    for key in (mean_key, var_key, first_layer_key):
        if key not in model or not torch.is_tensor(model[key]):
            raise ValueError(f"Checkpoint model is missing tensor {key!r}.")

    old_dim = int(model[mean_key].numel())
    target_dim = int(args.target_observation_dim)
    if target_dim <= old_dim:
        raise ValueError(f"Target observation dim must exceed {old_dim}, got {target_dim}.")
    if tuple(model[var_key].shape) != (old_dim,):
        raise ValueError(f"Unexpected running variance shape: {tuple(model[var_key].shape)}")

    first_weight = model[first_layer_key]
    if first_weight.ndim != 2 or first_weight.shape[1] != old_dim:
        raise ValueError(f"Unexpected first actor layer shape: {tuple(first_weight.shape)}")

    extra_dim = target_dim - old_dim
    model[mean_key] = torch.cat((model[mean_key], model[mean_key].new_zeros(extra_dim)))
    model[var_key] = torch.cat((model[var_key], model[var_key].new_ones(extra_dim)))
    model[first_layer_key] = torch.cat(
        (first_weight, first_weight.new_zeros((first_weight.shape[0], extra_dim))),
        dim=1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Expanded observation dim {old_dim} -> {target_dim}: {args.output}")


if __name__ == "__main__":
    main()
