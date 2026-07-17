"""Structured privileged-state actor-critic for controlled teacher ablations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder


STRUCTURED_STATE_NETWORK_NAME = "simtoolreal_structured_state_actor_critic"
FINGER_RELATION_RESIDUAL_NETWORK_NAME = "simtoolreal_finger_relation_residual_actor_critic"
STATIC_STATE_LAYOUT_VERSION = "static_privileged_v12"


def configure_static_structured_state_network(
    network_cfg: dict[str, Any],
    *,
    observation_dim: int,
    action_dim: int,
    branch_dim: int = 144,
) -> dict[str, Any]:
    """Configure the parameter-matched static privileged-state encoder in-place.

    The static teacher observation is intentionally unchanged. Only its inductive
    bias changes: robot, object, relative geometry, and contact/phase channels are
    encoded independently before actor-critic fusion.
    """

    observation_dim = int(observation_dim)
    action_dim = int(action_dim)
    previous_action_start = 63
    contact_phase_start = previous_action_start + action_dim
    expected_observation_dim = contact_phase_start + 19
    if observation_dim != expected_observation_dim:
        raise ValueError(
            "The structured static encoder requires the v12 privileged layout "
            f"(expected obs={expected_observation_dim} for action={action_dim}, "
            f"got obs={observation_dim})."
        )
    if branch_dim <= 0:
        raise ValueError(f"branch_dim must be positive, got {branch_dim}")

    network_cfg["name"] = STRUCTURED_STATE_NETWORK_NAME
    network_cfg["separate"] = False
    network_cfg["structured_state"] = {
        "layout_version": STATIC_STATE_LAYOUT_VERSION,
        "observation_dim": observation_dim,
        "branch_dim": int(branch_dim),
        "fusion_units": [512, 256, 128],
        "groups": {
            "robot": [[0, 26], [previous_action_start, contact_phase_start]],
            "object": [[26, 39]],
            "relation": [[39, previous_action_start]],
            "contact_phase": [[contact_phase_start, observation_dim]],
        },
    }
    return network_cfg


def configure_static_finger_relation_residual_network(
    network_cfg: dict[str, Any],
    *,
    observation_dim: int,
    action_dim: int,
    token_dim: int = 32,
) -> dict[str, Any]:
    """Add paired per-finger contact features without replacing the flat MLP."""

    observation_dim = int(observation_dim)
    action_dim = int(action_dim)
    contact_phase_start = 63 + action_dim
    expected_observation_dim = contact_phase_start + 19
    if observation_dim != expected_observation_dim:
        raise ValueError(
            "The finger-relation encoder requires the v12 privileged layout "
            f"(expected obs={expected_observation_dim} for action={action_dim}, "
            f"got obs={observation_dim})."
        )
    if token_dim <= 0:
        raise ValueError(f"token_dim must be positive, got {token_dim}")

    network_cfg["name"] = FINGER_RELATION_RESIDUAL_NETWORK_NAME
    network_cfg["separate"] = False
    network_cfg["finger_relation_residual"] = {
        "layout_version": STATIC_STATE_LAYOUT_VERSION,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "finger_count": 5,
        "token_dim": int(token_dim),
        "fingertip_relative_position_start": 42,
        "surface_distance_start": 57,
        "force_start": contact_phase_start + 4,
        "contact_flag_start": contact_phase_start + 9,
    }
    return network_cfg


def _validated_group_slices(
    groups: Mapping[str, Sequence[Sequence[int]]], observation_dim: int
) -> dict[str, tuple[tuple[int, int], ...]]:
    if len(groups) < 2:
        raise ValueError("structured_state.groups must contain at least two semantic groups")

    parsed: dict[str, tuple[tuple[int, int], ...]] = {}
    covered: list[int] = []
    for name, ranges in groups.items():
        if not ranges:
            raise ValueError(f"Semantic group {name!r} has no observation slices")
        parsed_ranges: list[tuple[int, int]] = []
        for bounds in ranges:
            if len(bounds) != 2:
                raise ValueError(f"Invalid slice for semantic group {name!r}: {bounds!r}")
            start, end = (int(value) for value in bounds)
            if not 0 <= start < end <= observation_dim:
                raise ValueError(
                    f"Slice [{start}, {end}) for semantic group {name!r} is outside "
                    f"observation width {observation_dim}"
                )
            parsed_ranges.append((start, end))
            covered.extend(range(start, end))
        parsed[str(name)] = tuple(parsed_ranges)

    if sorted(covered) != list(range(observation_dim)):
        raise ValueError(
            "Structured semantic slices must cover every observation column exactly once"
        )
    return parsed


class StructuredStateNetwork(NetworkBuilder.BaseNetwork):
    """Shared actor-critic trunk with explicit privileged-state semantic branches."""

    def __init__(self, params: dict[str, Any], **kwargs: Any):
        super().__init__()
        actions_num = int(kwargs.pop("actions_num"))
        input_shape = kwargs.pop("input_shape")
        self.value_size = int(kwargs.pop("value_size", 1))

        if isinstance(input_shape, Mapping) or len(input_shape) != 1:
            raise ValueError(f"Structured state network expects a flat observation, got {input_shape!r}")
        observation_dim = int(input_shape[0])
        if bool(params.get("separate", False)):
            raise ValueError("Structured state network currently requires a shared actor-critic trunk")
        if "continuous" not in params.get("space", {}):
            raise ValueError("Structured state network currently supports continuous actions only")

        structured_cfg = params.get("structured_state")
        if not isinstance(structured_cfg, Mapping):
            raise ValueError("Missing network.structured_state configuration")
        configured_dim = int(structured_cfg.get("observation_dim", observation_dim))
        if configured_dim != observation_dim:
            raise ValueError(
                f"Configured observation width {configured_dim} does not match runtime width {observation_dim}"
            )

        groups = _validated_group_slices(structured_cfg.get("groups", {}), observation_dim)
        branch_dim = int(structured_cfg.get("branch_dim", 144))
        fusion_units = [int(value) for value in structured_cfg.get("fusion_units", [512, 256, 128])]
        if branch_dim <= 0 or not fusion_units or any(value <= 0 for value in fusion_units):
            raise ValueError("Structured branch and fusion widths must be positive")

        mlp_cfg = params["mlp"]
        activation_name = mlp_cfg["activation"]
        self.group_slices = groups
        self.semantic_encoders = nn.ModuleDict()
        for name, ranges in groups.items():
            group_width = sum(end - start for start, end in ranges)
            self.semantic_encoders[name] = nn.Sequential(
                nn.Linear(group_width, branch_dim),
                self.activations_factory.create(activation_name),
            )

        self.fusion_mlp = self._build_mlp(
            input_size=branch_dim * len(groups),
            units=fusion_units,
            activation=activation_name,
            dense_func=nn.Linear,
            norm_func_name=mlp_cfg.get("normalization"),
            d2rl=False,
        )
        output_dim = fusion_units[-1]
        self.value = nn.Linear(output_dim, self.value_size)
        self.value_act = self.activations_factory.create(params.get("value_activation", "None"))

        continuous_cfg = params["space"]["continuous"]
        self.mu = nn.Linear(output_dim, actions_num)
        self.mu_act = self.activations_factory.create(continuous_cfg["mu_activation"])
        self.sigma_act = self.activations_factory.create(continuous_cfg["sigma_activation"])
        self.fixed_sigma = bool(continuous_cfg["fixed_sigma"])
        if self.fixed_sigma:
            self.sigma = nn.Parameter(torch.zeros(actions_num, dtype=torch.float32))
        else:
            self.sigma = nn.Linear(output_dim, actions_num)

        mlp_init = self.init_factory.create(**mlp_cfg["initializer"])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                mlp_init(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        mu_init = self.init_factory.create(**continuous_cfg["mu_init"])
        sigma_init = self.init_factory.create(**continuous_cfg["sigma_init"])
        mu_init(self.mu.weight)
        if self.fixed_sigma:
            sigma_init(self.sigma)
        else:
            sigma_init(self.sigma.weight)

    def forward(self, obs_dict: dict[str, torch.Tensor]):
        obs = obs_dict["obs"]
        encoded_groups = []
        for name, ranges in self.group_slices.items():
            group_obs = torch.cat([obs[..., start:end] for start, end in ranges], dim=-1)
            encoded_groups.append(self.semantic_encoders[name](group_obs))
        latent = self.fusion_mlp(torch.cat(encoded_groups, dim=-1))
        mu = self.mu_act(self.mu(latent))
        if self.fixed_sigma:
            logstd = mu * 0.0 + self.sigma_act(self.sigma)
        else:
            logstd = self.sigma_act(self.sigma(latent))
        value = self.value_act(self.value(latent))
        return mu, logstd, value, None


class StructuredStateNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.params: dict[str, Any] = {}

    def load(self, params: dict[str, Any]) -> None:
        self.params = params

    def build(self, name: str, **kwargs: Any) -> StructuredStateNetwork:
        del name
        return StructuredStateNetwork(self.params, **kwargs)


class FingerRelationResidualNetwork(A2CBuilder.Network):
    """Standard flat MLP plus a zero-initialized per-finger relation residual."""

    def __init__(self, params: dict[str, Any], **kwargs: Any):
        super().__init__(params, **kwargs)
        if self.separate or self.has_cnn or self.has_rnn or not self.is_continuous:
            raise ValueError(
                "Finger-relation residual requires a shared, feed-forward continuous actor-critic"
            )

        relation_cfg = params.get("finger_relation_residual")
        if not isinstance(relation_cfg, Mapping):
            raise ValueError("Missing network.finger_relation_residual configuration")
        self.relation_observation_dim = int(relation_cfg["observation_dim"])
        self.relation_finger_count = int(relation_cfg.get("finger_count", 5))
        self.relation_token_dim = int(relation_cfg.get("token_dim", 32))
        self.fingertip_relative_position_start = int(
            relation_cfg["fingertip_relative_position_start"]
        )
        self.surface_distance_start = int(relation_cfg["surface_distance_start"])
        self.force_start = int(relation_cfg["force_start"])
        self.contact_flag_start = int(relation_cfg["contact_flag_start"])
        if self.relation_finger_count != 5:
            raise ValueError("The static relation layout currently requires five fingers")

        # Preserve the global RNG state after the flat network is built. This
        # keeps initial flat weights and subsequent action sampling aligned with
        # the baseline; the residual projection starts at exactly zero.
        with torch.random.fork_rng(devices=[]):
            self.finger_relation_encoder = nn.Sequential(
                nn.Linear(6, self.relation_token_dim),
                self.activations_factory.create(self.activation),
                nn.Linear(self.relation_token_dim, self.relation_token_dim),
                self.activations_factory.create(self.activation),
            )
            pooled_dim = 3 * self.relation_token_dim
            self.finger_relation_projection = nn.Linear(pooled_dim, self.units[-1])

            mlp_init = self.init_factory.create(**self.initializer)
            for module in self.finger_relation_encoder.modules():
                if isinstance(module, nn.Linear):
                    mlp_init(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            nn.init.zeros_(self.finger_relation_projection.weight)
            nn.init.zeros_(self.finger_relation_projection.bias)

    def finger_relation_tokens(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.relation_observation_dim:
            raise ValueError(
                f"Expected observation width {self.relation_observation_dim}, got {obs.shape[-1]}"
            )
        count = self.relation_finger_count
        position_end = self.fingertip_relative_position_start + 3 * count
        fingertip_relative_position = obs[
            ..., self.fingertip_relative_position_start:position_end
        ].reshape(*obs.shape[:-1], count, 3)
        surface_distance = obs[
            ..., self.surface_distance_start : self.surface_distance_start + count
        ].unsqueeze(-1)
        force = obs[..., self.force_start : self.force_start + count].unsqueeze(-1)
        contact_flag = obs[
            ..., self.contact_flag_start : self.contact_flag_start + count
        ].unsqueeze(-1)
        return torch.cat(
            (fingertip_relative_position, surface_distance, force, contact_flag),
            dim=-1,
        )

    def forward(self, obs_dict: dict[str, torch.Tensor]):
        obs = obs_dict["obs"]
        flat_latent = self.actor_mlp(obs)

        finger_tokens = self.finger_relation_encoder(self.finger_relation_tokens(obs))
        thumb_token = finger_tokens[..., 0, :]
        non_thumb_tokens = finger_tokens[..., 1:, :]
        relation_latent = torch.cat(
            (
                thumb_token,
                non_thumb_tokens.mean(dim=-2),
                non_thumb_tokens.amax(dim=-2),
            ),
            dim=-1,
        )
        latent = flat_latent + self.finger_relation_projection(relation_latent)

        value = self.value_act(self.value(latent))
        if self.central_value:
            return value, None
        mu = self.mu_act(self.mu(latent))
        if self.fixed_sigma:
            logstd = mu * 0.0 + self.sigma_act(self.sigma)
        else:
            logstd = self.sigma_act(self.sigma(latent))
        return mu, logstd, value, None


class FingerRelationResidualNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.params: dict[str, Any] = {}

    def load(self, params: dict[str, Any]) -> None:
        self.params = params

    def build(self, name: str, **kwargs: Any) -> FingerRelationResidualNetwork:
        del name
        return FingerRelationResidualNetwork(self.params, **kwargs)


def register_structured_state_network() -> None:
    """Register the custom network with rl_games' process-local registry."""

    model_builder.register_network(STRUCTURED_STATE_NETWORK_NAME, StructuredStateNetworkBuilder)
    model_builder.register_network(
        FINGER_RELATION_RESIDUAL_NETWORK_NAME,
        FingerRelationResidualNetworkBuilder,
    )


__all__ = [
    "FINGER_RELATION_RESIDUAL_NETWORK_NAME",
    "STATIC_STATE_LAYOUT_VERSION",
    "STRUCTURED_STATE_NETWORK_NAME",
    "FingerRelationResidualNetwork",
    "FingerRelationResidualNetworkBuilder",
    "StructuredStateNetwork",
    "StructuredStateNetworkBuilder",
    "configure_static_finger_relation_residual_network",
    "configure_static_structured_state_network",
    "register_structured_state_network",
]
