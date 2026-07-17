"""Contracts for the unified dynamic tasks selected by the static v2.6 audit."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / (
    "source/simtoolreal_lab/simtoolreal_lab/tasks/dynamic_dexterous_grasp/"
    "dynamic_dexterous_grasp_env_cfg.py"
)
TASK_INIT_PATH = ROOT / (
    "source/simtoolreal_lab/simtoolreal_lab/tasks/dynamic_dexterous_grasp/__init__.py"
)
ENV_PATH = ROOT / (
    "source/simtoolreal_lab/simtoolreal_lab/tasks/dynamic_dexterous_grasp/"
    "dynamic_dexterous_grasp_env.py"
)


def _classes() -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in ast.parse(CFG_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef)
    }


def _assignment_source(node: ast.ClassDef, field_name: str) -> str:
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value = statement.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == field_name for target in targets):
            assert value is not None
            return ast.unparse(value)
    raise AssertionError(f"Missing {node.name}.{field_name}")


def _literal(node: ast.ClassDef, field_name: str):
    return ast.literal_eval(_assignment_source(node, field_name))


def test_dynamic_v26_uses_the_selected_direct_six_motor_action_contract():
    contract = _classes()["_UnifiedDynamicJointDeltaV26PolicyContract"]

    assert _literal(contract, "action_space") == 13
    assert _literal(contract, "policy_action_interface") == "joint_target"
    assert _literal(contract, "joint_target_arm_action_mode") == "incremental"
    assert _literal(contract, "joint_target_arm_delta_scale") == 0.015
    assert _literal(contract, "joint_target_arm_max_delta") == 0.015
    assert _literal(contract, "joint_target_hand_max_delta") == 0.05
    assert _literal(contract, "arm_moving_average") == 0.20
    assert _literal(contract, "hand_moving_average") == 0.20
    assert _literal(contract, "hand_action_semantics") == (
        "six_physical_motor_absolute_target"
    )

    for field_name in (
        "scripted_action_prior_enabled",
        "scripted_tabletop_pregrasp_prior_enabled",
        "scripted_tabletop_approach_action_prior_enabled",
        "scripted_tabletop_lift_target_prior_enabled",
        "scripted_tabletop_relative_lift_target_prior_enabled",
        "scripted_tabletop_cartesian_lift_target_prior_enabled",
        "scripted_tabletop_hand_grasp_memory_prior_enabled",
        "tabletop_privileged_lift_target_obs_enabled",
        "tabletop_lift_hand_target_lock_enabled",
        "tabletop_post_success_arm_target_lock_enabled",
        "tabletop_post_success_hand_target_lock_enabled",
    ):
        assert _literal(contract, field_name) is False


def test_rolling_v26_is_static_to_fast_multishape_and_physically_strict():
    classes = _classes()
    contract = classes["_UnifiedRollingJointDeltaV26Contract"]

    assert _literal(contract, "benchmark_protocol") == (
        "rolling_multishape_jointdelta_v2_6"
    )
    assert _literal(contract, "canonical_reset_curriculum_enabled") is True
    assert _literal(contract, "canonical_reset_curriculum_mixed_distribution") is True
    assert _assignment_source(contract, "dynamic_tabletop_start_speed_range") == (
        "UNIFIED_ROLLING_START_SPEED_RANGE"
    )
    assert _assignment_source(contract, "dynamic_tabletop_initial_speed_range") == (
        "UNIFIED_ROLLING_TARGET_SPEED_RANGE"
    )
    assert _assignment_source(contract, "dynamic_tabletop_heading_range") == (
        "TABLETOP_FULL_HEADING_RANGE"
    )
    assert _literal(contract, "tabletop_success_requires_force_grasp") is True
    assert _literal(contract, "tabletop_success_requires_hover_target") is True
    assert _literal(contract, "dynamic_success_hold_steps") == 30

    expected_bases = {
        "Revo2UnifiedRollingJointDeltaV26TeacherEnvCfg": (
            "_UnifiedRollingJointDeltaV26Contract",
            "Revo2UnifiedRollingBenchmarkTeacherEnvCfg",
        ),
        "InspireUnifiedRollingJointDeltaV26TeacherEnvCfg": (
            "_UnifiedRollingJointDeltaV26Contract",
            "InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg",
        ),
    }
    for class_name, bases in expected_bases.items():
        assert tuple(ast.unparse(base) for base in classes[class_name].bases) == bases

    assert _assignment_source(
        classes["Revo2UnifiedRollingJointDeltaV26TeacherEnvCfg"], "sim_hand_joint_names"
    ) == "REVO2_HAND_JOINT_NAMES"
    assert "_v699_revo2_official_six_active_robot_cfg" in _assignment_source(
        classes["Revo2UnifiedRollingJointDeltaV26TeacherEnvCfg"], "robot_cfg"
    )
    assert _assignment_source(
        classes["InspireUnifiedRollingJointDeltaV26TeacherEnvCfg"], "sim_hand_joint_names"
    ) == "INSPIRE_ACTIVE_HAND_JOINT_NAMES"
    assert "_inspire_rh56bfx_mimic_robot_cfg" in _assignment_source(
        classes["InspireUnifiedRollingJointDeltaV26TeacherEnvCfg"], "robot_cfg"
    )


def test_falling_v26_shares_action_and_requires_force_backed_green_grasp():
    classes = _classes()
    contract = classes["_UnifiedFallingJointDeltaV26Contract"]

    assert _literal(contract, "benchmark_protocol") == (
        "falling_baton_affordance_jointdelta_v2_6"
    )
    assert _literal(contract, "falling_success_requires_force_grasp") is True
    assert _literal(contract, "falling_success_uses_strict_grasp") is True
    assert _literal(contract, "falling_success_requires_positive_affordance") is True

    expected_bases = {
        "Revo2UnifiedFallingJointDeltaV26TeacherEnvCfg": (
            "_UnifiedFallingJointDeltaV26Contract",
            "Revo2UnifiedFallingBatonBenchmarkTeacherEnvCfg",
        ),
        "InspireUnifiedFallingJointDeltaV26TeacherEnvCfg": (
            "_UnifiedFallingJointDeltaV26Contract",
            "InspireRH56BFXFaithfulUnifiedFallingTeacherEnvCfg",
        ),
    }
    for class_name, bases in expected_bases.items():
        assert tuple(ast.unparse(base) for base in classes[class_name].bases) == bases


def test_all_four_dynamic_v26_tasks_are_registered():
    source = TASK_INIT_PATH.read_text(encoding="utf-8")
    for embodiment in ("Revo2", "Inspire"):
        assert (
            f"SimToolReal-{embodiment}-Franka-UnifiedRollingJointDeltaV26-"
            "Teacher-Direct-v0"
        ) in source
        assert (
            f"SimToolReal-{embodiment}-Franka-UnifiedFallingBatonJointDeltaV26-"
            "Teacher-Direct-v0"
        ) in source


def test_rolling_v27_uses_lift_first_reward_and_sequential_success_gates():
    classes = _classes()
    contract = classes["_UnifiedRollingJointDeltaV27Contract"]

    assert _literal(contract, "canonical_static_reward_enabled") is True
    assert _literal(contract, "canonical_progress_only_approach") is True
    assert _literal(contract, "canonical_contact_rew_scale") == 0.0
    assert _literal(contract, "canonical_lift_progress_rew_scale") == 45.0
    assert _literal(contract, "canonical_hover_stable_rew_scale") == 90.0
    assert _literal(contract, "canonical_success_bonus") == 300.0
    assert _literal(contract, "canonical_reset_curriculum_mode") == "success_gate"
    assert _literal(contract, "canonical_reset_curriculum_metric") == "catch_hold"
    assert _literal(contract, "dynamic_speed_curriculum_min_canonical_reset_alpha") == 0.98

    expected_bases = {
        "Revo2UnifiedRollingJointDeltaV27TeacherEnvCfg": (
            "_UnifiedRollingJointDeltaV27Contract",
            "Revo2UnifiedRollingBenchmarkTeacherEnvCfg",
        ),
        "InspireUnifiedRollingJointDeltaV27TeacherEnvCfg": (
            "_UnifiedRollingJointDeltaV27Contract",
            "InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg",
        ),
    }
    for class_name, bases in expected_bases.items():
        assert tuple(ast.unparse(base) for base in classes[class_name].bases) == bases

    registrations = TASK_INIT_PATH.read_text(encoding="utf-8")
    for embodiment in ("Revo2", "Inspire"):
        assert (
            f"SimToolReal-{embodiment}-Franka-UnifiedRollingJointDeltaV27-"
            "Teacher-Direct-v0"
        ) in registrations

    env_source = ENV_PATH.read_text(encoding="utf-8")
    assert "def _update_canonical_reset_curriculum(" in env_source
    assert "dynamic_speed_curriculum_min_canonical_reset_alpha" in env_source
