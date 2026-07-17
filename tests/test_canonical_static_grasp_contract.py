"""Contracts for the shared from-scratch static grasp baseline."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / (
    "source/simtoolreal_lab/simtoolreal_lab/tasks/dynamic_dexterous_grasp/"
    "dynamic_dexterous_grasp_env_cfg.py"
)
ENV_PATH = ROOT / (
    "source/simtoolreal_lab/simtoolreal_lab/tasks/dynamic_dexterous_grasp/"
    "dynamic_dexterous_grasp_env.py"
)
TASK_INIT_PATH = ROOT / (
    "source/simtoolreal_lab/simtoolreal_lab/tasks/dynamic_dexterous_grasp/__init__.py"
)


def _classes() -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in ast.parse(CFG_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef)
    }


def _assignment(node: ast.ClassDef, field_name: str):
    for statement in node.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(isinstance(target, ast.Name) and target.id == field_name for target in targets):
            return ast.literal_eval(statement.value)
    raise AssertionError(f"Missing {node.name}.{field_name}")


def test_canonical_contract_is_dense_from_scratch_and_strict_only_at_success():
    contract = _classes()["_StaticCanonicalFromScratchContract"]

    assert _assignment(contract, "canonical_static_reward_enabled") is True
    assert _assignment(contract, "benchmark_protocol") == (
        "static_canonical_progress_lift_target_v2_5"
    )
    assert _assignment(contract, "canonical_progress_only_approach") is True
    assert _assignment(contract, "canonical_reset_curriculum_enabled") is True
    assert _assignment(contract, "canonical_reset_curriculum_mixed_distribution") is True
    assert _assignment(contract, "canonical_reset_curriculum_pregrasp_anchor_fraction") == 0.20
    assert _assignment(contract, "canonical_reset_curriculum_hard_anchor_fraction") == 0.20
    assert _assignment(contract, "canonical_reset_hand_action") == 0.0
    assert _assignment(contract, "joint_target_arm_action_mode") == "incremental"
    assert _assignment(contract, "hand_action_semantics") == (
        "six_physical_motor_absolute_target"
    )
    assert _assignment(contract, "strict_reward_enabled") is False
    assert _assignment(contract, "strict_success_enabled") is True
    assert _assignment(contract, "tabletop_lift_requires_clean_grasp_latch") is False
    assert _assignment(contract, "tabletop_lift_gate_requires_force_grasp") is False
    assert _assignment(contract, "tabletop_lift_rewards_require_force_grasp") is False
    assert _assignment(contract, "tabletop_success_lift_height") == 0.10
    assert _assignment(contract, "tabletop_hover_latch_lift_progress") == 0.95
    assert _assignment(contract, "tabletop_hover_follow_object_xy_until_latch") is False
    assert _assignment(contract, "dynamic_success_hold_steps") == 180
    assert _assignment(contract, "tabletop_success_requires_force_grasp") is True
    assert _assignment(contract, "canonical_contact_rew_scale") == 0.0
    assert _assignment(contract, "canonical_grasp_quality_rew_scale") == 0.0
    assert _assignment(contract, "canonical_force_grasp_rew_scale") == 0.0
    assert 0.0 < _assignment(contract, "canonical_palm_reach_rew_scale") < 2.0
    assert 0.0 < _assignment(contract, "canonical_fingertip_reach_rew_scale") < 3.0
    assert _assignment(contract, "canonical_lift_progress_rew_scale") > 0.0
    assert _assignment(contract, "canonical_lift_milestone_bonus") > 0.0

    for field_name in (
        "scripted_action_prior_enabled",
        "scripted_tabletop_pregrasp_prior_enabled",
        "scripted_tabletop_approach_action_prior_enabled",
        "scripted_tabletop_lift_target_prior_enabled",
        "scripted_tabletop_relative_lift_target_prior_enabled",
        "scripted_tabletop_cartesian_lift_target_prior_enabled",
        "scripted_tabletop_hand_grasp_memory_prior_enabled",
        "tabletop_privileged_lift_target_obs_enabled",
    ):
        assert _assignment(contract, field_name) is False


def test_revo2_and_inspire_share_task_reward_reset_and_action_semantics():
    classes = _classes()
    expected_joint_bases = {
        "Revo2StaticCanonicalJointDeltaTeacherEnvCfg": (
            "_StaticCanonicalFromScratchContract",
            "Revo2DynamicTabletopTeacherEnvCfg",
        ),
        "InspireStaticCanonicalJointDeltaTeacherEnvCfg": (
            "_StaticCanonicalFromScratchContract",
            "InspireDynamicTabletopTeacherEnvCfg",
        ),
    }
    for class_name, expected_bases in expected_joint_bases.items():
        node = classes[class_name]
        assert tuple(ast.unparse(base) for base in node.bases) == expected_bases
        assert _assignment(node, "action_space") == 13
        assert _assignment(node, "observation_space") == 76
        assert _assignment(node, "policy_action_interface") == "joint_target"

    expected_cartesian_bases = {
        "Revo2StaticCanonicalCartesianImpedanceTeacherEnvCfg": (
            "_CartesianImpedanceDirectPolicyContract",
            "Revo2StaticCanonicalJointDeltaTeacherEnvCfg",
        ),
        "InspireStaticCanonicalCartesianImpedanceTeacherEnvCfg": (
            "_CartesianImpedanceDirectPolicyContract",
            "InspireStaticCanonicalJointDeltaTeacherEnvCfg",
        ),
    }
    for class_name, expected_bases in expected_cartesian_bases.items():
        node = classes[class_name]
        assert tuple(ast.unparse(base) for base in node.bases) == expected_bases
        assert _assignment(node, "observation_space") == 75
        assert _assignment(node, "cartesian_impedance_target_mode") == "measured_delta"


def test_robust_home_contract_is_shared_and_only_refines_existing_signals():
    classes = _classes()
    contract = classes["_StaticCanonicalRobustHomeContract"]

    assert _assignment(contract, "benchmark_protocol") == (
        "static_canonical_robust_home_stable_v2_6"
    )
    assert _assignment(contract, "canonical_reset_curriculum_pregrasp_anchor_fraction") == 0.10
    assert _assignment(contract, "canonical_reset_curriculum_hard_anchor_fraction") == 0.50
    assert _assignment(contract, "canonical_reset_arm_pos_noise") == 0.020
    assert _assignment(contract, "joint_target_arm_delta_scale") == 0.015
    assert _assignment(contract, "joint_target_arm_max_delta") == 0.015
    assert _assignment(contract, "canonical_hover_stable_rew_scale") > 0.0
    assert _assignment(contract, "canonical_success_now_rew_scale") > 0.0
    assert _assignment(contract, "canonical_hold_progress_rew_scale") > 0.0

    expected_bases = {
        "Revo2StaticCanonicalRobustJointDeltaTeacherEnvCfg": (
            "_StaticCanonicalRobustHomeContract",
            "Revo2StaticCanonicalJointDeltaTeacherEnvCfg",
        ),
        "Revo2StaticCanonicalRobustCartesianImpedanceTeacherEnvCfg": (
            "_StaticCanonicalRobustHomeContract",
            "Revo2StaticCanonicalCartesianImpedanceTeacherEnvCfg",
        ),
        "InspireStaticCanonicalRobustJointDeltaTeacherEnvCfg": (
            "_StaticCanonicalRobustHomeContract",
            "InspireStaticCanonicalJointDeltaTeacherEnvCfg",
        ),
        "InspireStaticCanonicalRobustCartesianImpedanceTeacherEnvCfg": (
            "_StaticCanonicalRobustHomeContract",
            "InspireStaticCanonicalCartesianImpedanceTeacherEnvCfg",
        ),
    }
    for class_name, bases in expected_bases.items():
        assert tuple(ast.unparse(base) for base in classes[class_name].bases) == bases


def test_incremental_arm_reset_curriculum_and_isolated_reward_are_executable():
    source = ENV_PATH.read_text(encoding="utf-8")

    assert "def _canonical_reset_curriculum_alpha" in source
    assert "def _apply_canonical_reset_curriculum" in source
    assert "use_mixed_distribution" in source
    assert "override_alpha is None" in source
    assert "self._apply_canonical_reset_curriculum(env_ids_tensor)" in source
    assert "neutral_hand_pos = self._active_hand_actions_to_sim_targets" in source
    assert "joint_pos[:, self._control_hand_joint_ids] = neutral_hand_pos" in source
    assert "def _write_canonical_mimic_reset_positions" in source
    assert "self._write_canonical_mimic_reset_positions(joint_pos)" in source
    assert 'arm_action_mode in {"incremental", "delta", "relative"}' in source
    assert "current_arm_targets + delta_scale * arm_actions" in source
    assert 'arm_action_mode in {"absolute", "normalized_absolute"}' in source

    canonical_reward = source.split("if canonical_reward_enabled:", 1)[1].split(
        "arm_target_tracking_error =", 1
    )[0]
    assert "canonical_palm_progress" in canonical_reward
    assert "canonical_fingertip_progress" in canonical_reward
    assert "canonical_lift_step_progress" in canonical_reward
    assert "_canonical_closest_mean_surface_distance" in canonical_reward
    assert "_canonical_max_object_height_delta" in canonical_reward
    assert "canonical_lift_milestone" in canonical_reward
    assert "canonical_pre_lift_phase" in canonical_reward
    assert "canonical_post_lift_phase" in canonical_reward
    assert "canonical_lift_height_reward\n                * canonical_pre_lift_phase" in canonical_reward
    assert "canonical_unsupported_lift" in canonical_reward
    assert "canonical_success_bonus" in canonical_reward
    assert "clean_lift_gate" not in canonical_reward
    assert "reward = canonical_reward" in canonical_reward


def test_all_four_canonical_tasks_are_registered():
    source = TASK_INIT_PATH.read_text(encoding="utf-8")
    for embodiment in ("Revo2", "Inspire"):
        for controller in ("JointDelta", "CartesianImpedance"):
            assert (
                f"SimToolReal-{embodiment}-Franka-StaticCanonical{controller}-"
                "Teacher-Direct-v0"
            ) in source
            assert (
                f"SimToolReal-{embodiment}-Franka-StaticCanonicalRobust{controller}-"
                "Teacher-Direct-v0"
            ) in source
