#!/usr/bin/env python3
"""Rescale an rl_games continuous policy mean head in a copied checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source rl_games checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Destination checkpoint.")
    parser.add_argument("--mean-scale", type=float, required=True, help="Multiplier for actor mean weights and bias.")
    parser.add_argument("--force", action="store_true", help="Replace an existing destination.")
    args = parser.parse_args()

    if args.mean_scale <= 0.0:
        raise ValueError("--mean-scale must be positive")
    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.force:
        raise FileExistsError(f"destination already exists: {destination}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model = checkpoint.get("model")
    if not isinstance(model, dict):
        raise KeyError("checkpoint does not contain a model state dictionary")

    scaled_keys = [
        key for key in model if key.endswith(".mu.weight") or key.endswith(".mu.bias")
    ]
    if not scaled_keys:
        raise KeyError("no continuous actor mean parameters (*.mu.weight/bias) found")
    for key in scaled_keys:
        model[key] = model[key] * float(args.mean_scale)

    transform = {
        "type": "policy_mean_scale",
        "source": str(source),
        "mean_scale": float(args.mean_scale),
        "scaled_keys": scaled_keys,
    }
    checkpoint.setdefault("simtoolreal_checkpoint_transforms", []).append(transform)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    destination.with_suffix(destination.suffix + ".transform.json").write_text(
        json.dumps(transform, indent=2) + "\n",
        encoding="ascii",
    )
    print(json.dumps({"output": str(destination), **transform}, indent=2))


if __name__ == "__main__":
    main()
