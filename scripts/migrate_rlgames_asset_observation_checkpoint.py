#!/usr/bin/env python3
"""Prepare a fixed-asset rl_games checkpoint for multi-asset observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--asset-dim", type=int, default=10)
args = parser.parse_args()


INPUT_MEAN_KEY = "running_mean_std.running_mean"
INPUT_VAR_KEY = "running_mean_std.running_var"
FIRST_LAYER_KEY = "a2c_network.actor_mlp.0.weight"


def main() -> None:
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    model = checkpoint.get("model")
    if not isinstance(model, dict):
        raise KeyError("checkpoint does not contain a model state dict")
    for key in (INPUT_MEAN_KEY, INPUT_VAR_KEY, FIRST_LAYER_KEY):
        if key not in model:
            raise KeyError(f"checkpoint model is missing {key!r}")

    running_mean = model[INPUT_MEAN_KEY]
    running_var = model[INPUT_VAR_KEY]
    first_weight = model[FIRST_LAYER_KEY]
    asset_dim = int(args.asset_dim)
    if asset_dim <= 0 or asset_dim > running_mean.numel():
        raise ValueError(f"invalid asset dimension {asset_dim} for observation size {running_mean.numel()}")
    if running_var.shape != running_mean.shape or first_weight.shape[1] != running_mean.numel():
        raise ValueError(
            "incompatible running-stat/first-layer shapes: "
            f"mean={tuple(running_mean.shape)}, var={tuple(running_var.shape)}, "
            f"weight={tuple(first_weight.shape)}"
        )

    start = running_mean.numel() - asset_dim
    old_mean = running_mean[start:].clone()
    old_var = running_var[start:].clone()
    old_weight_norm = float(torch.linalg.vector_norm(first_weight[:, start:]).item())

    # The fixed-asset policy observed one constant metadata vector, so these
    # columns carry no learned variation. Ignore them at initialization and let
    # multi-asset continuation learn shape dependence from a neutral scale.
    running_mean[start:] = 0.0
    running_var[start:] = 1.0
    first_weight[:, start:] = 0.0

    checkpoint["simtoolreal_checkpoint_migration"] = {
        "kind": "fixed_asset_to_multi_asset_observation",
        "source": str(input_path),
        "observation_dim": int(running_mean.numel()),
        "asset_start_index": int(start),
        "asset_dim": int(asset_dim),
        "old_asset_running_mean": old_mean.tolist(),
        "old_asset_running_var": old_var.tolist(),
        "old_asset_first_layer_weight_norm": old_weight_norm,
        "new_asset_running_mean": running_mean[start:].tolist(),
        "new_asset_running_var": running_var[start:].tolist(),
        "new_asset_first_layer_weight_norm": float(
            torch.linalg.vector_norm(first_weight[:, start:]).item()
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    reloaded = torch.load(output_path, map_location="cpu", weights_only=False)
    reloaded_model = reloaded["model"]
    assert torch.count_nonzero(reloaded_model[FIRST_LAYER_KEY][:, start:]).item() == 0
    assert torch.all(reloaded_model[INPUT_MEAN_KEY][start:] == 0)
    assert torch.all(reloaded_model[INPUT_VAR_KEY][start:] == 1)
    print(checkpoint["simtoolreal_checkpoint_migration"], flush=True)
    print(f"saved migrated checkpoint: {output_path}", flush=True)


if __name__ == "__main__":
    main()
