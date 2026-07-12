#!/usr/bin/env python3
"""Expand an rl_games checkpoint to a larger observation space without changing its policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source rl_games checkpoint.")
    parser.add_argument("--output", required=True, help="Expanded checkpoint path.")
    parser.add_argument("--new-observation-dim", type=int, required=True)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    model = checkpoint.get("model")
    if not isinstance(model, dict):
        raise RuntimeError(f"Checkpoint has no model state dict: {input_path}")

    mean_key = "running_mean_std.running_mean"
    var_key = "running_mean_std.running_var"
    if mean_key not in model or var_key not in model:
        raise RuntimeError("Checkpoint has no input RunningMeanStd tensors")
    old_dim = int(model[mean_key].numel())
    new_dim = int(args.new_observation_dim)
    if new_dim <= old_dim:
        raise ValueError(f"new observation dim must exceed {old_dim}, got {new_dim}")
    extra_dim = new_dim - old_dim

    expanded_weights = []
    for key, value in tuple(model.items()):
        if not torch.is_tensor(value) or value.ndim != 2 or int(value.shape[1]) != old_dim:
            continue
        padding = torch.zeros((value.shape[0], extra_dim), dtype=value.dtype, device=value.device)
        model[key] = torch.cat((value, padding), dim=1)
        expanded_weights.append(key)
    if not expanded_weights:
        raise RuntimeError(f"No network input weights with observation dim {old_dim} were found")

    mean = model[mean_key]
    variance = model[var_key]
    model[mean_key] = torch.cat((mean, torch.zeros(extra_dim, dtype=mean.dtype)), dim=0)
    model[var_key] = torch.cat((variance, torch.ones(extra_dim, dtype=variance.dtype)), dim=0)
    checkpoint["observation_expansion"] = {
        "source": str(input_path),
        "old_dim": old_dim,
        "new_dim": new_dim,
        "zero_padded_weight_keys": expanded_weights,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(
        f"expanded observation checkpoint {old_dim} -> {new_dim}: {output_path} "
        f"(weights={expanded_weights})"
    )


if __name__ == "__main__":
    main()
