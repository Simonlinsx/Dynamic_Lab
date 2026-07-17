from pathlib import Path
import subprocess
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "expand_rl_games_observation_checkpoint.py"


def test_expands_first_layer_and_running_stats_without_changing_old_policy_columns(tmp_path):
    source = tmp_path / "source.pth"
    output = tmp_path / "expanded.pth"
    first_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    checkpoint = {
        "model": {
            "running_mean_std.running_mean": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "running_mean_std.running_var": torch.tensor([2.0, 3.0, 4.0, 5.0]),
            "running_mean_std.count": torch.tensor(12.0),
            "a2c_network.actor_mlp.0.weight": first_weight,
        },
        "optimizer": {"state": {1: {"step": 3}}, "param_groups": []},
    }
    torch.save(checkpoint, source)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(source),
            "--output",
            str(output),
            "--target-observation-dim",
            "6",
        ],
        check=True,
    )

    expanded = torch.load(output, map_location="cpu")
    model = expanded["model"]
    assert torch.equal(model["a2c_network.actor_mlp.0.weight"][:, :4], first_weight)
    assert torch.count_nonzero(model["a2c_network.actor_mlp.0.weight"][:, 4:]) == 0
    assert torch.equal(
        model["running_mean_std.running_mean"],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0]),
    )
    assert torch.equal(
        model["running_mean_std.running_var"],
        torch.tensor([2.0, 3.0, 4.0, 5.0, 1.0, 1.0]),
    )
    assert expanded["optimizer"] == checkpoint["optimizer"]
