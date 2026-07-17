from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from simtoolreal_lab.anydex.inspire_adapter import (
    ACTIVE_JOINT_NAMES,
    ANYDEX_REQUIRED_NON_THUMB_CONTACTS,
    AnyDexInspirePaths,
    build_predictor_command,
    load_anydex_candidates,
    make_primitive_predictor_input,
    make_sphere_predictor_input,
)


ROOT = Path(__file__).resolve().parents[1]
ANYDEX_ROOT = Path("/data1/linsixu/AnyDexGrasp")
WIDTH_TABLE = ANYDEX_ROOT / "generate_mesh_and_pointcloud/inspire_urdf/width_12Dangle_6Dangle.json"
REPLAY_RESULT = (
    ROOT
    / "outputs"
    / "anydex_inspire"
    / "20260713_predictor_replay"
    / "result.json"
)


def test_anydex_contact_contract_distinguishes_pinches_from_multifinger_grasps():
    assert ANYDEX_REQUIRED_NON_THUMB_CONTACTS[1] == 1
    assert ANYDEX_REQUIRED_NON_THUMB_CONTACTS[2] == 1
    assert all(
        ANYDEX_REQUIRED_NON_THUMB_CONTACTS[grasp_type] == 2
        for grasp_type in range(3, 9)
    )


def test_replayed_domino_result_maps_to_official_six_motor_contract():
    candidates = load_anydex_candidates(REPLAY_RESULT, width_angle_table_path=WIDTH_TABLE)
    selected = candidates[0]

    assert len(candidates) == 3
    assert selected.grasp_type == 3
    assert selected.grasp_type_name == "Prismatic_3_Finger"
    assert tuple(ACTIVE_JOINT_NAMES) == (
        "thumb_proximal_yaw_joint",
        "thumb_proximal_pitch_joint",
        "index_proximal_joint",
        "middle_proximal_joint",
        "ring_proximal_joint",
        "pinky_proximal_joint",
    )
    np.testing.assert_array_less(selected.active_joint_targets, np.array((1.309, 0.601, 1.471, 1.471, 1.471, 1.471)))
    np.testing.assert_array_less(np.full(6, -1.0e-12), selected.active_joint_targets)


def test_hand_base_conversion_matches_domino_raw_strict_contract():
    selected = load_anydex_candidates(REPLAY_RESULT, width_angle_table_path=WIDTH_TABLE)[0]
    expected_translation = np.array((-0.0479744922, -0.0415775338, 0.9857149477))
    np.testing.assert_allclose(selected.grasp_hand_base_matrix_world[:3, 3], expected_translation, atol=1.0e-8)
    expected_pregrasp = expected_translation - selected.approach_direction_world * 0.075
    np.testing.assert_allclose(selected.pregrasp_hand_base_matrix_world[:3, 3], expected_pregrasp, atol=1.0e-8)


def test_object_conditioned_sphere_reprojects_domino_candidate():
    raw = load_anydex_candidates(REPLAY_RESULT, width_angle_table_path=WIDTH_TABLE)[0]
    sphere = load_anydex_candidates(
        REPLAY_RESULT,
        width_angle_table_path=WIDTH_TABLE,
        grasp_type_override=7,
        width_override=0.06,
    )[0]

    assert sphere.source_grasp_type == raw.grasp_type
    assert sphere.source_width == raw.width
    assert sphere.grasp_type == 7
    assert sphere.grasp_type_name == "Sphere_3_Finger"
    assert sphere.width == 0.06
    assert not np.allclose(sphere.grasp_hand_base_matrix_world, raw.grasp_hand_base_matrix_world)
    np.testing.assert_array_less(np.full(6, -1.0e-12), sphere.active_joint_targets)


def test_sphere_predictor_input_preserves_world_center(tmp_path):
    center = np.array((0.535, 0.075, 0.328))
    path = make_sphere_predictor_input(tmp_path / "sphere.npz", object_center_world=center, radius=0.03)
    data = np.load(path)
    points = data["points"]
    transform = data["world_from_predictor"]
    world_points = (transform[:3, :3] @ points.T).T + transform[:3, 3]

    assert points.shape == (4096, 3)
    np.testing.assert_allclose(world_points.mean(axis=0), center, atol=2.0e-5)
    np.testing.assert_allclose(np.linalg.norm(world_points - center, axis=1), 0.03, atol=2.0e-6)


def test_box_predictor_input_matches_requested_extents(tmp_path):
    center = np.array((0.535, 0.075, 0.3275))
    size = np.array((0.055, 0.045, 0.055))
    path = make_primitive_predictor_input(
        tmp_path / "box.npz",
        object_center_world=center,
        shape="box",
        size=size,
    )
    data = np.load(path)
    points = data["points"]
    transform = data["world_from_predictor"]
    world_points = (transform[:3, :3] @ points.T).T + transform[:3, 3]

    assert str(data["object_shape"]) == "box"
    np.testing.assert_allclose(world_points.mean(axis=0), center, atol=8.0e-4)
    np.testing.assert_allclose(
        world_points.max(axis=0) - world_points.min(axis=0),
        size,
        atol=1.0e-6,
    )


def test_cylinder_predictor_input_matches_radius_and_height(tmp_path):
    center = np.array((0.535, 0.075, 0.3375))
    radius = 0.025
    height = 0.075
    path = make_primitive_predictor_input(
        tmp_path / "cylinder.npz",
        object_center_world=center,
        shape="cylinder",
        size=(2.0 * radius, 2.0 * radius, height),
    )
    data = np.load(path)
    points = data["points"]
    transform = data["world_from_predictor"]
    world_points = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    local = world_points - center

    assert str(data["object_shape"]) == "cylinder"
    assert np.max(np.linalg.norm(local[:, :2], axis=1)) <= radius + 1.0e-7
    np.testing.assert_allclose(local[:, 2].min(), -0.5 * height, atol=1.0e-7)
    np.testing.assert_allclose(local[:, 2].max(), 0.5 * height, atol=1.0e-7)


def test_predictor_command_uses_original_network_and_decision_models(tmp_path):
    paths = AnyDexInspirePaths()
    command = build_predictor_command(tmp_path / "input.npz", tmp_path / "result.json", paths=paths)
    assert "--use_graspnet_v2" in command
    assert command[command.index("--checkpoint_path") + 1] == str(paths.checkpoint)
    assert command[command.index("--inspire_model_path") + 1] == str(paths.decision_models)
    assert "--allow_random_inspire_decision" not in command


def test_saved_replay_is_a_real_anydex_result():
    payload = json.loads(REPLAY_RESULT.read_text(encoding="utf-8"))
    diagnostics = payload["diagnostics"]
    assert payload["success"] is True
    assert diagnostics["pipeline"] == "original_anydex_inspire_decision"
    assert diagnostics["allow_random_inspire_decision"] is False
    assert diagnostics["final_candidate_count"] == 3
