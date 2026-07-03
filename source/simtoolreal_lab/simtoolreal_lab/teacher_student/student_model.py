"""Temporal point-cloud student model for dynamic dexterous grasping."""

from __future__ import annotations

import torch
from torch import nn


def mlp(sizes: list[int], dropout: float = 0.0) -> nn.Sequential:
    """Build a SiLU MLP."""

    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class PointTemporalStudent(nn.Module):
    """WAM-like baseline over masked RGB-D object point-cloud sequences.

    Inputs:
        pointcloud_seq: [B, H, N, 3]
        valid_seq: [B, H, N]
        proprio_seq: [B, H, P]

    Outputs:
        action: [B, A]
        privileged: [B, C]
        flow: [B, N, 3]
        affordance_logits: [B, N]
        hold_target: [B, A - arm_dim]
        hold_logits: [B]
    """

    def __init__(
        self,
        history: int,
        proprio_dim: int,
        action_dim: int,
        privileged_dim: int,
        arm_dim: int = 7,
        point_dim: int = 128,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.history = int(history)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.privileged_dim = int(privileged_dim)
        self.arm_dim = int(arm_dim)
        self.hand_dim = max(self.action_dim - self.arm_dim, 0)

        self.point_encoder = mlp([4, point_dim, point_dim], dropout=dropout)
        global_dim = self.history * (2 * point_dim + self.proprio_dim)
        self.global_encoder = mlp([global_dim, hidden_dim, hidden_dim, latent_dim], dropout=dropout)
        self.action_head = mlp([latent_dim, hidden_dim, self.action_dim], dropout=dropout)
        self.privileged_head = mlp([latent_dim, hidden_dim, self.privileged_dim], dropout=dropout)
        self.flow_head = mlp([point_dim + latent_dim + 3, hidden_dim, hidden_dim // 2, 3], dropout=dropout)
        self.affordance_head = mlp([point_dim + latent_dim + 3, hidden_dim, hidden_dim // 2, 1], dropout=dropout)
        self.hold_head = mlp([latent_dim, hidden_dim, self.hand_dim], dropout=dropout)
        self.hold_gate_head = mlp([latent_dim, hidden_dim // 2, 1], dropout=dropout)

    def forward(
        self,
        pointcloud_seq: torch.Tensor,
        valid_seq: torch.Tensor,
        proprio_seq: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if pointcloud_seq.ndim != 4 or pointcloud_seq.shape[-1] != 3:
            raise ValueError(f"pointcloud_seq must have shape [B, H, N, 3], got {tuple(pointcloud_seq.shape)}")
        if valid_seq.shape != pointcloud_seq.shape[:-1]:
            raise ValueError(
                "valid_seq must match pointcloud_seq without xyz dimension: "
                f"got {tuple(valid_seq.shape)} vs {tuple(pointcloud_seq.shape[:-1])}"
            )
        if proprio_seq.ndim != 3 or proprio_seq.shape[:2] != pointcloud_seq.shape[:2]:
            raise ValueError(
                "proprio_seq must have shape [B, H, P] with matching B/H: "
                f"got {tuple(proprio_seq.shape)}"
            )

        batch_size, history, num_points, _ = pointcloud_seq.shape
        if history != self.history:
            raise ValueError(f"history mismatch: model={self.history}, input={history}")

        valid = (valid_seq > 0.0).float()
        point_input = torch.cat([pointcloud_seq, valid.unsqueeze(-1)], dim=-1)
        point_feat = self.point_encoder(point_input.reshape(batch_size * history * num_points, 4))
        point_feat = point_feat.reshape(batch_size, history, num_points, -1)

        mask = valid > 0.0
        mask_f = mask.unsqueeze(-1).float()
        mean_feat = (point_feat * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp_min(1.0)

        masked_feat = point_feat.masked_fill(~mask.unsqueeze(-1), torch.finfo(point_feat.dtype).min)
        max_feat = masked_feat.max(dim=2).values
        empty_frames = ~mask.any(dim=2)
        if empty_frames.any():
            max_feat = max_feat.masked_fill(empty_frames.unsqueeze(-1), 0.0)

        global_feat = torch.cat([mean_feat, max_feat, proprio_seq], dim=-1)
        latent = self.global_encoder(global_feat.reshape(batch_size, -1))

        current_point_feat = point_feat[:, -1]
        current_points = pointcloud_seq[:, -1]
        latent_points = latent[:, None, :].expand(-1, num_points, -1)
        per_point = torch.cat([current_point_feat, latent_points, current_points], dim=-1)

        return {
            "action": self.action_head(latent),
            "privileged": self.privileged_head(latent),
            "flow": self.flow_head(per_point),
            "affordance_logits": self.affordance_head(per_point).squeeze(-1),
            "hold_target": self.hold_head(latent),
            "hold_logits": self.hold_gate_head(latent).squeeze(-1),
        }
