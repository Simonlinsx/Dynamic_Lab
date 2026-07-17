"""Contract tests for the parameter-matched privileged teacher encoder."""

from __future__ import annotations

import copy

import pytest
import torch
from rl_games.algos_torch.network_builder import A2CBuilder

from simtoolreal_lab.teacher_student.structured_state_network import (
    STATIC_STATE_LAYOUT_VERSION,
    FingerRelationResidualNetworkBuilder,
    StructuredStateNetworkBuilder,
    configure_static_finger_relation_residual_network,
    configure_static_structured_state_network,
)


BASE_NETWORK_CFG = {
    "name": "actor_critic",
    "separate": False,
    "space": {
        "continuous": {
            "mu_activation": "None",
            "sigma_activation": "None",
            "mu_init": {"name": "default"},
            "sigma_init": {"name": "const_initializer", "val": 0},
            "fixed_sigma": True,
        }
    },
    "mlp": {
        "units": [512, 512, 256, 128],
        "activation": "elu",
        "d2rl": False,
        "initializer": {"name": "default"},
        "regularizer": {"name": "None"},
    },
}


def _build_structured(observation_dim: int, action_dim: int):
    cfg = copy.deepcopy(BASE_NETWORK_CFG)
    configure_static_structured_state_network(
        cfg,
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
    builder = StructuredStateNetworkBuilder()
    builder.load(cfg)
    return cfg, builder.build(
        "teacher",
        input_shape=(observation_dim,),
        actions_num=action_dim,
        value_size=1,
    )


def _build_finger_residual(observation_dim: int, action_dim: int):
    cfg = copy.deepcopy(BASE_NETWORK_CFG)
    configure_static_finger_relation_residual_network(
        cfg,
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
    builder = FingerRelationResidualNetworkBuilder()
    builder.load(cfg)
    return cfg, builder.build(
        "teacher",
        input_shape=(observation_dim,),
        actions_num=action_dim,
        value_size=1,
    )


@pytest.mark.parametrize(("observation_dim", "action_dim"), ((95, 13), (94, 12)))
def test_structured_teacher_forward_contract(observation_dim: int, action_dim: int):
    cfg, network = _build_structured(observation_dim, action_dim)
    mu, logstd, value, states = network({"obs": torch.randn(7, observation_dim)})

    assert cfg["structured_state"]["layout_version"] == STATIC_STATE_LAYOUT_VERSION
    assert mu.shape == (7, action_dim)
    assert logstd.shape == (7, action_dim)
    assert value.shape == (7, 1)
    assert states is None
    assert torch.isfinite(mu).all()
    assert torch.isfinite(value).all()


@pytest.mark.parametrize(("observation_dim", "action_dim"), ((95, 13), (94, 12)))
def test_structured_teacher_is_parameter_matched(observation_dim: int, action_dim: int):
    _, structured = _build_structured(observation_dim, action_dim)
    flat_builder = A2CBuilder()
    flat_builder.load(copy.deepcopy(BASE_NETWORK_CFG))
    flat = flat_builder.build(
        "teacher",
        input_shape=(observation_dim,),
        actions_num=action_dim,
        value_size=1,
    )

    structured_params = sum(parameter.numel() for parameter in structured.parameters())
    flat_params = sum(parameter.numel() for parameter in flat.parameters())
    assert abs(structured_params - flat_params) / flat_params < 0.01


def test_structured_teacher_layout_covers_each_channel_once():
    cfg, _ = _build_structured(95, 13)
    groups = cfg["structured_state"]["groups"]
    covered = [index for ranges in groups.values() for start, end in ranges for index in range(start, end)]

    assert sorted(covered) == list(range(95))
    assert len(covered) == len(set(covered))
    assert groups["robot"] == [[0, 26], [63, 76]]
    assert groups["contact_phase"] == [[76, 95]]


def test_structured_teacher_rejects_non_v12_observation_layout():
    with pytest.raises(ValueError, match="requires the v12 privileged layout"):
        configure_static_structured_state_network(
            copy.deepcopy(BASE_NETWORK_CFG),
            observation_dim=75,
            action_dim=13,
        )


@pytest.mark.parametrize(("observation_dim", "action_dim"), ((95, 13), (94, 12)))
def test_finger_relation_residual_forward_contract(observation_dim: int, action_dim: int):
    _, network = _build_finger_residual(observation_dim, action_dim)
    mu, logstd, value, states = network({"obs": torch.randn(7, observation_dim)})

    assert mu.shape == (7, action_dim)
    assert logstd.shape == (7, action_dim)
    assert value.shape == (7, 1)
    assert states is None


def test_finger_relation_residual_starts_exactly_as_flat_policy():
    torch.manual_seed(19)
    flat_builder = A2CBuilder()
    flat_builder.load(copy.deepcopy(BASE_NETWORK_CFG))
    flat = flat_builder.build("teacher", input_shape=(95,), actions_num=13, value_size=1)

    torch.manual_seed(19)
    _, residual = _build_finger_residual(95, 13)
    obs = torch.linspace(-1.0, 1.0, steps=3 * 95).reshape(3, 95)

    flat_outputs = flat({"obs": obs})
    residual_outputs = residual({"obs": obs})
    for flat_output, residual_output in zip(flat_outputs[:3], residual_outputs[:3], strict=True):
        assert torch.equal(flat_output, residual_output)
    assert torch.count_nonzero(residual.finger_relation_projection.weight) == 0
    assert torch.count_nonzero(residual.finger_relation_projection.bias) == 0


def test_finger_relation_tokens_pair_each_fingers_geometry_and_contact():
    _, network = _build_finger_residual(95, 13)
    obs = torch.zeros(1, 95)
    for finger_id in range(5):
        obs[0, 42 + 3 * finger_id : 45 + 3 * finger_id] = torch.tensor(
            [finger_id + 0.1, finger_id + 0.2, finger_id + 0.3]
        )
        obs[0, 57 + finger_id] = finger_id + 0.4
        obs[0, 80 + finger_id] = finger_id + 0.5
        obs[0, 85 + finger_id] = finger_id + 0.6

    tokens = network.finger_relation_tokens(obs)
    assert tokens.shape == (1, 5, 6)
    for finger_id in range(5):
        assert torch.allclose(
            tokens[0, finger_id],
            torch.tensor(
                [
                    finger_id + 0.1,
                    finger_id + 0.2,
                    finger_id + 0.3,
                    finger_id + 0.4,
                    finger_id + 0.5,
                    finger_id + 0.6,
                ]
            ),
        )


def test_finger_relation_residual_adds_less_than_four_percent_parameters():
    _, residual = _build_finger_residual(95, 13)
    flat_builder = A2CBuilder()
    flat_builder.load(copy.deepcopy(BASE_NETWORK_CFG))
    flat = flat_builder.build("teacher", input_shape=(95,), actions_num=13, value_size=1)

    residual_params = sum(parameter.numel() for parameter in residual.parameters())
    flat_params = sum(parameter.numel() for parameter in flat.parameters())
    assert flat_params < residual_params < 1.04 * flat_params
