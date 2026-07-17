"""Static contract tests for direct Cartesian-wrist teacher actions."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / (
    "source/simtoolreal_lab/simtoolreal_lab/tasks/dynamic_dexterous_grasp/"
    "dynamic_dexterous_grasp_env.py"
)
CFG_PATH = ROOT / (
    "source/simtoolreal_lab/simtoolreal_lab/tasks/dynamic_dexterous_grasp/"
    "dynamic_dexterous_grasp_env_cfg.py"
)
TASK_INIT_PATH = ROOT / (
    "source/simtoolreal_lab/simtoolreal_lab/tasks/dynamic_dexterous_grasp/__init__.py"
)
SCHEMA_PATH = ROOT / "source/simtoolreal_lab/simtoolreal_lab/teacher_student/schema.py"
EVAL_PATH = ROOT / "scripts/evaluate_rl_games.py"
POSE_TRACKING_PATH = ROOT / "scripts/probe_action_interface_pose_tracking.py"


def _class_node(path: Path, class_name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"Missing class {class_name}")


def _assignment_source(path: Path, class_name: str, field_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    node = _class_node(path, class_name)
    for statement in node.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(isinstance(target, ast.Name) and target.id == field_name for target in targets):
            return ast.get_source_segment(source, statement.value)
    raise AssertionError(f"Missing {class_name}.{field_name}")


def _module_assignment_source(path: Path, field_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(isinstance(target, ast.Name) and target.id == field_name for target in targets):
            return ast.get_source_segment(source, statement.value)
    raise AssertionError(f"Missing module assignment {field_name}")


def test_cartesian_policy_execution_is_single_step_direct_control():
    source = ENV_PATH.read_text(encoding="utf-8")

    assert 'self.cfg.policy_action_interface == "cartesian_wrist_delta"' in source
    assert "self._update_cartesian_wrist_policy_target()" in source
    assert "wrist_actions = self.actions[:, :6]" in source
    assert "hand_actions = self.actions[:, 6:]" in source
    assert 'target_mode == "measured_delta"' in source
    assert 'target_mode == "integrated_delta"' in source
    assert "target_base_pos_w + translation_scale" in source
    assert "quat_mul(delta_quat_w, target_base_quat_w)" in source
    assert "self._cartesian_wrist_policy_target_initialized[env_ids_tensor] = True" in source
    assert "robot.root_physx_view.get_jacobians()" in source
    assert "torch.linalg.solve(" in source
    assert "nullspace_projection" in source
    assert "desired_lift_action[:, 2] = 1.0" in source


def test_cartesian_contract_is_12d_and_forbids_every_scripted_prior():
    mixin = "_CartesianWristDirectPolicyContract"
    assert _assignment_source(CFG_PATH, mixin, "action_space") == "12"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, mixin, "policy_action_interface")
    ) == "cartesian_wrist_delta"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, mixin, "cartesian_wrist_target_mode")
    ) == "measured_delta"
    assert float(_assignment_source(CFG_PATH, mixin, "cartesian_wrist_translation_scale")) == 0.015
    assert float(_assignment_source(CFG_PATH, mixin, "cartesian_wrist_rotation_scale")) == 0.08
    assert float(_assignment_source(CFG_PATH, mixin, "cartesian_wrist_max_position_error")) == 0.12
    assert float(_assignment_source(CFG_PATH, mixin, "cartesian_wrist_max_rotation_error")) == 0.50
    assert float(_assignment_source(CFG_PATH, mixin, "cartesian_wrist_nullspace_gain")) == 0.05

    for field_name in (
        "scripted_action_prior_enabled",
        "scripted_tabletop_pregrasp_prior_enabled",
        "scripted_tabletop_approach_action_prior_enabled",
        "scripted_tabletop_lift_target_prior_enabled",
        "scripted_tabletop_relative_lift_target_prior_enabled",
        "scripted_tabletop_cartesian_lift_target_prior_enabled",
        "scripted_tabletop_hand_grasp_memory_prior_enabled",
    ):
        assert _assignment_source(CFG_PATH, mixin, field_name) == "False"

    env_source = ENV_PATH.read_text(encoding="utf-8")
    assert "Cartesian direct-from-scratch tasks cannot enable scripted action priors" in env_source

    persistent_mixin = "_PersistentCartesianWristDirectPolicyContract"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, persistent_mixin, "cartesian_wrist_target_mode")
    ) == "integrated_delta"


def test_simple_dynamic_tabletop_cartesian_ablation_only_overrides_action_contract():
    class_name = "Revo2DynamicTabletopCartesianTeacherEnvCfg"
    node = _class_node(CFG_PATH, class_name)
    assert [ast.unparse(base) for base in node.bases] == [
        "_PersistentCartesianWristDirectPolicyContract",
        "Revo2DynamicTabletopTeacherEnvCfg",
    ]
    assert ast.literal_eval(_assignment_source(CFG_PATH, class_name, "action_contract")) == (
        "revo2_cartesian_wrist_12d"
    )
    assert _assignment_source(CFG_PATH, class_name, "observation_space") == "75"

    source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert "SimToolReal-Revo2-Franka-DynamicTabletopCartesian-Teacher-Direct-v0" in source
    assert class_name in source


def test_inspire_falling_has_a_thin_persistent_cartesian_adapter():
    class_name = "InspireRH56BFXFaithfulUnifiedFallingCartesianTeacherEnvCfg"
    node = _class_node(CFG_PATH, class_name)
    assert [ast.unparse(base) for base in node.bases] == [
        "_PersistentCartesianWristDirectPolicyContract",
        "InspireRH56BFXFaithfulUnifiedFallingTeacherEnvCfg",
    ]
    assert ast.literal_eval(_assignment_source(CFG_PATH, class_name, "action_contract")) == (
        "inspire_cartesian_wrist_12d"
    )
    assert _assignment_source(CFG_PATH, class_name, "observation_space") == "75"

    source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert "SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedFallingCartesian-Teacher-Direct-v0" in source
    assert class_name in source


def test_both_hands_have_parallel_cartesian_curriculum_tasks():
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    class_names = (
        "Revo2UnifiedRollingStage1CartesianPhaseObsTeacherEnvCfg",
        "Revo2UnifiedRollingStage2HoldCartesianPhaseObsTeacherEnvCfg",
        "Revo2UnifiedRollingStage3CartesianPhaseObsTeacherEnvCfg",
        "InspireRH56BFXFaithfulUnifiedRollingStage1CartesianPhaseObsTeacherEnvCfg",
        "InspireRH56BFXFaithfulUnifiedRollingStage2HoldCartesianPhaseObsTeacherEnvCfg",
        "InspireRH56BFXFaithfulUnifiedRollingStage3CartesianPhaseObsTeacherEnvCfg",
    )
    for class_name in class_names:
        assert _assignment_source(CFG_PATH, class_name, "observation_space") == "89"
        assert _assignment_source(
            CFG_PATH, class_name, "tabletop_privileged_phase_obs_enabled"
        ) == "True"
        assert class_name in task_source

    assert "revo2_cartesian_wrist_12d" in _assignment_source(
        CFG_PATH,
        "Revo2UnifiedRollingStage1CartesianPhaseObsTeacherEnvCfg",
        "action_contract",
    )
    assert "inspire_cartesian_wrist_12d" in _assignment_source(
        CFG_PATH,
        "InspireRH56BFXFaithfulUnifiedRollingStage1CartesianPhaseObsTeacherEnvCfg",
        "action_contract",
    )


def test_acquisition_stage_removes_cartesian_object_lift_shortcuts():
    contract = "_CartesianWristAcquisitionStage1Contract"
    for field_name in (
        "palm_lift_rew_scale",
        "lift_progress_rew_scale",
        "quality_lift_progress_rew_scale",
        "lifted_true_grasp_rew_scale",
        "tabletop_grasped_palm_lift_rew_scale",
        "tabletop_stable_catch_rew_scale",
        "stable_hold_rew_scale",
        "hold_progress_rew_scale",
        "success_bonus",
        "tabletop_no_lift_after_grasp_penalty_scale",
    ):
        assert float(_assignment_source(CFG_PATH, contract, field_name)) == 0.0
    assert float(
        _assignment_source(
            CFG_PATH, contract, "tabletop_lift_without_current_grasp_penalty_scale"
        )
    ) > 0.0

    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    for class_name in (
        "Revo2UnifiedRollingStage1AcquisitionCartesianPhaseObsTeacherEnvCfg",
        "InspireRH56BFXFaithfulUnifiedRollingStage1AcquisitionCartesianPhaseObsTeacherEnvCfg",
    ):
        assert class_name in task_source


def test_static_pregrasp_has_no_synthetic_motion_direction():
    env_source = ENV_PATH.read_text(encoding="utf-8")

    assert 'dynamic_tabletop_pregrasp_direction_min_speed", 0.005' in env_source
    assert "object_xy_speed > direction_min_speed" in env_source
    assert "torch.zeros_like(object_xy_vel)" in env_source
    assert "fallback_xy_dir" not in env_source


def test_revo2_static_action_ablation_controls_every_non_action_variable():
    shared = "_Revo2StaticActionInterfaceAblationContract"
    clear_home = ast.literal_eval(
        _module_assignment_source(CFG_PATH, "REVO2_STATIC_CLEAR_HOME_ARM_POS")
    )
    assert clear_home == (0.0, -0.5471, 0.0, -2.749, 0.0, 2.9723, 0.7454398163397448)
    assert _assignment_source(CFG_PATH, shared, "robot_cfg").startswith(
        "_v699_revo2_robot_cfg(REVO2_STATIC_CLEAR_HOME_ARM_POS)"
    )
    for field_name in (
        "default_arm_pos",
        "scripted_tabletop_pregrasp_arm_pos",
        "tabletop_arm_lift_progress_baseline_pos",
    ):
        assert _assignment_source(CFG_PATH, shared, field_name) == (
            "REVO2_STATIC_CLEAR_HOME_ARM_POS"
        )
    calibrated_offset = ast.literal_eval(
        _module_assignment_source(CFG_PATH, "REVO2_CALIBRATED_GRASP_CENTER_OFFSET")
    )
    assert calibrated_offset == (0.025, -0.012, 0.044)
    assert _assignment_source(CFG_PATH, shared, "palm_offset") == (
        "REVO2_CALIBRATED_GRASP_CENTER_OFFSET"
    )
    joint_max_delta = float(
        _module_assignment_source(CFG_PATH, "REVO2_STATIC_ACTION_ABLATION_MAX_ARM_JOINT_DELTA")
    )
    cartesian_max_delta = float(
        _assignment_source(CFG_PATH, "_CartesianWristDirectPolicyContract", "cartesian_wrist_max_joint_delta")
    )
    assert joint_max_delta == cartesian_max_delta == 0.04
    assert _assignment_source(CFG_PATH, shared, "joint_target_arm_max_delta") == (
        "REVO2_STATIC_ACTION_ABLATION_MAX_ARM_JOINT_DELTA"
    )
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, shared, "tabletop_arm_lift_progress_mode")
    ) == "palm_z"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, shared, "joint_target_rate_limit_requires_lift_baseline")
    ) is False
    assert float(_assignment_source(CFG_PATH, shared, "tabletop_asset_curriculum_override_alpha")) == 0.0
    assert float(_assignment_source(CFG_PATH, shared, "dynamic_grasp_speed_curriculum_override_alpha")) == 0.0
    assert float(_assignment_source(CFG_PATH, shared, "dynamic_tabletop_pregrasp_lead_time")) == 0.0
    assert float(_assignment_source(CFG_PATH, shared, "dynamic_tabletop_pregrasp_ahead_distance")) == 0.0
    for field_name in (
        "action_penalty_scale",
        "arm_target_delta_penalty_scale",
        "tabletop_hover_post_latch_action_penalty_scale",
        "tabletop_hover_post_latch_target_delta_penalty_scale",
        "tabletop_lift_action_prior_rew_scale",
        "tabletop_lift_target_rew_scale",
        "tabletop_lift_without_current_grasp_penalty_scale",
        "tabletop_post_success_arm_target_drift_penalty_scale",
        "tabletop_post_success_action_penalty_scale",
        "tabletop_post_success_target_delta_penalty_scale",
    ):
        assert float(_assignment_source(CFG_PATH, shared, field_name)) == 0.0

    env_source = ENV_PATH.read_text(encoding="utf-8")
    assert "def _compute_tabletop_arm_lift_progress" in env_source
    assert env_source.count("self._compute_tabletop_arm_lift_progress()") == 2
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, shared, "tabletop_privileged_lift_target_obs_enabled")
    ) is False
    for field_name in (
        "scripted_action_prior_enabled",
        "scripted_tabletop_pregrasp_prior_enabled",
        "scripted_tabletop_approach_action_prior_enabled",
        "scripted_tabletop_lift_target_prior_enabled",
        "scripted_tabletop_relative_lift_target_prior_enabled",
        "scripted_tabletop_cartesian_lift_target_prior_enabled",
        "scripted_tabletop_hand_grasp_memory_prior_enabled",
    ):
        assert ast.literal_eval(_assignment_source(CFG_PATH, shared, field_name)) is False

    joint_class = "Revo2StaticActionInterfaceJointPhaseObsTeacherEnvCfg"
    cartesian_class = "Revo2StaticActionInterfaceCartesianPhaseObsTeacherEnvCfg"
    assert _assignment_source(CFG_PATH, joint_class, "action_space") == "13"
    assert ast.literal_eval(_assignment_source(CFG_PATH, joint_class, "policy_action_interface")) == "joint_target"
    assert _assignment_source(CFG_PATH, joint_class, "observation_space") == "90"
    assert ast.literal_eval(_assignment_source(CFG_PATH, cartesian_class, "action_contract")) == (
        "revo2_cartesian_wrist_12d"
    )
    assert _assignment_source(CFG_PATH, cartesian_class, "observation_space") == "89"

    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    for class_name in (joint_class, cartesian_class):
        base_names = {ast.unparse(base) for base in classes[class_name].bases}
        assert shared in base_names
        assert "_CartesianWristAcquisitionStage1Contract" in base_names
        assert class_name in TASK_INIT_PATH.read_text(encoding="utf-8")

    assert not classes[shared].bases


def test_revo2_static_action_ablation_has_paired_hold_and_lift_stages():
    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    paired_stages = (
        (
            "Revo2StaticActionInterfaceJointStage2HoldPhaseObsTeacherEnvCfg",
            "joint_target",
            "13",
            "90",
        ),
        (
            "Revo2StaticActionInterfaceCartesianStage2HoldPhaseObsTeacherEnvCfg",
            "cartesian_wrist_delta",
            "12",
            "89",
        ),
        (
            "Revo2StaticActionInterfaceJointStage3LiftPhaseObsTeacherEnvCfg",
            "joint_target",
            "13",
            "90",
        ),
        (
            "Revo2StaticActionInterfaceCartesianStage3LiftPhaseObsTeacherEnvCfg",
            "cartesian_wrist_delta",
            "12",
            "89",
        ),
    )
    for class_name, interface, action_space, observation_space in paired_stages:
        assert class_name in classes
        base_names = {ast.unparse(base) for base in classes[class_name].bases}
        assert "_Revo2StaticActionInterfaceAblationContract" in base_names
        assert _assignment_source(CFG_PATH, class_name, "observation_space") == observation_space
        if interface == "joint_target":
            assert _assignment_source(CFG_PATH, class_name, "action_space") == action_space
            assert ast.literal_eval(
                _assignment_source(CFG_PATH, class_name, "policy_action_interface")
            ) == interface
        else:
            assert "_CartesianWristDirectPolicyContract" in base_names
            assert _assignment_source(
                CFG_PATH, "_CartesianWristDirectPolicyContract", "action_space"
            ) == action_space
            assert ast.literal_eval(
                _assignment_source(
                    CFG_PATH, "_CartesianWristDirectPolicyContract", "policy_action_interface"
                )
            ) == interface
        assert class_name in task_source


def test_revo2_static_stage2_uses_shared_physical_contact_entry_for_both_interfaces():
    stage2_contract = "_Revo2StaticGraspHoldStage2Contract"
    assert float(
        _assignment_source(
            CFG_PATH,
            "_Revo2StaticActionInterfaceAblationContract",
            "dynamic_tabletop_palm_frame_pregrasp_rew_scale",
        )
    ) == 1800.0
    assert ast.literal_eval(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "pregrasp_target_rel_palm",
        )
    ) == (0.015, 0.016, 0.058)
    assert float(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "dynamic_tabletop_palm_frame_pregrasp_rew_scale",
        )
    ) == 1800.0
    assert float(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "tabletop_non_thumb_without_thumb_thumb_target",
        )
    ) == 0.67
    assert float(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "strict_touch_reward_target_distance",
        )
    ) == 0.002
    assert float(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "tabletop_non_thumb_without_thumb_penalty_scale",
        )
    ) == 6000.0
    assert float(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "strict_thumb_touch_rew_scale",
        )
    ) == 2000.0
    assert float(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "strict_touch_rew_scale",
        )
    ) == 3000.0
    assert float(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "strict_opposition_approach_rew_scale",
        )
    ) == 5000.0
    assert float(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "strict_opposition_touch_rew_scale",
        )
    ) == 12000.0
    assert ast.literal_eval(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "object_contact_force_diagnostics_enabled",
        )
    ) is True
    assert float(
        _assignment_source(CFG_PATH, stage2_contract, "tabletop_force_grasp_rew_scale")
    ) == 24000.0
    assert float(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "tabletop_force_thumb_contact_rew_scale",
        )
    ) == 16000.0
    assert int(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "object_force_grasp_min_non_thumb_contacts",
        )
    ) == 1
    assert float(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "tabletop_force_grasp_streak_rew_scale",
        )
    ) == 32000.0
    assert float(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "tabletop_force_stable_grasp_rew_scale",
        )
    ) == 42000.0
    assert ast.literal_eval(
        _assignment_source(
            CFG_PATH,
            stage2_contract,
            "tabletop_strict_grasp_loss_on_transition_only",
        )
    ) is True
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, stage2_contract, "benchmark_protocol")
    ) == "revo2_static_action_interface_ablation_v20_opposition_balanced_contact"

    env_source = ENV_PATH.read_text(encoding="utf-8")
    assert 'getattr(self.cfg, "strict_touch_reward_target_distance", None)' in env_source
    assert "self._surface_dist - strict_touch_target_distance" in env_source
    assert "strict_thumb_touch_rew = self._strict_thumb_touch_score * (1.0 - lift_progress)" in env_source
    assert 'getattr(self.cfg, "strict_thumb_touch_rew_scale", 0.0)' in env_source
    assert '"strict_thumb_touch_rew": strict_thumb_touch_rew.mean()' in env_source
    assert "tabletop_force_thumb_contact_rew = self._force_thumb_contact.float()" in env_source
    assert 'getattr(self.cfg, "tabletop_force_thumb_contact_rew_scale", 0.0)' in env_source
    assert '"tabletop_force_thumb_contact_rew": tabletop_force_thumb_contact_rew.mean()' in env_source

    for class_name in (
        "Revo2StaticActionInterfaceJointStage2HoldPhaseObsTeacherEnvCfg",
        "Revo2StaticActionInterfaceJointDeltaStage2HoldPhaseObsTeacherEnvCfg",
        "Revo2StaticActionInterfaceCartesianStage2HoldPhaseObsTeacherEnvCfg",
    ):
        class_node = _class_node(CFG_PATH, class_name)
        assert stage2_contract in {ast.unparse(base) for base in class_node.bases}


def test_revo2_static_joint_delta_matches_cartesian_target_step_and_isaacgym_semantics():
    contract = "_Revo2StaticJointDeltaPolicyContract"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, contract, "policy_action_interface")
    ) == "isaaclab_direct"
    assert ast.literal_eval(_assignment_source(CFG_PATH, contract, "action_contract")) == (
        "revo2_semantic_13d"
    )
    assert _assignment_source(CFG_PATH, contract, "action_space") == "13"
    assert _assignment_source(CFG_PATH, contract, "observation_space") == "90"
    scale = float(_assignment_source(CFG_PATH, contract, "arm_action_scale"))
    moving_average = float(_assignment_source(CFG_PATH, contract, "arm_moving_average"))
    tracking_error_limit = float(
        _assignment_source(CFG_PATH, contract, "arm_target_tracking_error_limit")
    )
    decimation = int(
        _assignment_source(CFG_PATH, "_Revo2StaticActionInterfaceAblationContract", "decimation")
    )
    assert decimation == 2
    assert abs(scale * (1.0 / 60.0) * moving_average * decimation - 0.04) < 1.0e-12
    assert tracking_error_limit == 0.04

    env_source = ENV_PATH.read_text(encoding="utf-8")
    assert "current_arm_targets + arm_actions * self.cfg.arm_action_scale * self.dt" in env_source
    assert 'getattr(self.cfg, "arm_target_tracking_error_limit", 0.0)' in env_source
    assert "measured_arm_pos - tracking_error_limit" in env_source
    assert "measured_arm_pos + tracking_error_limit" in env_source
    assert "self._joint_lower_limits[self._arm_joint_ids].unsqueeze(0)" in env_source
    assert "self._joint_upper_limits[self._arm_joint_ids].unsqueeze(0)" in env_source

    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    for class_name in (
        "Revo2StaticActionInterfaceJointDeltaPhaseObsTeacherEnvCfg",
        "Revo2StaticActionInterfaceJointDeltaStage2HoldPhaseObsTeacherEnvCfg",
        "Revo2StaticActionInterfaceJointDeltaStage3LiftPhaseObsTeacherEnvCfg",
    ):
        class_node = _class_node(CFG_PATH, class_name)
        assert contract in {ast.unparse(base) for base in class_node.bases}
        assert class_name in task_source


def test_torque_cartesian_impedance_contract_is_base_frame_and_physical():
    mixin = "_CartesianImpedanceDirectPolicyContract"
    assert _assignment_source(CFG_PATH, mixin, "action_space") == "12"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, mixin, "policy_action_interface")
    ) == "cartesian_impedance"
    assert float(
        _assignment_source(CFG_PATH, mixin, "cartesian_wrist_translation_scale")
    ) == 0.015
    assert float(
        _assignment_source(CFG_PATH, mixin, "cartesian_wrist_rotation_scale")
    ) == 0.08
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, mixin, "cartesian_impedance_target_mode")
    ) == "integrated_delta"
    assert float(
        _assignment_source(CFG_PATH, mixin, "cartesian_impedance_max_position_error")
    ) == 0.06
    assert float(
        _assignment_source(CFG_PATH, mixin, "cartesian_impedance_max_rotation_error")
    ) == 0.35
    assert ast.literal_eval(
        _assignment_source(
            CFG_PATH, mixin, "cartesian_impedance_arm_effort_limits"
        )
    ) == (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    assert ast.literal_eval(
        _assignment_source(
            CFG_PATH, mixin, "cartesian_impedance_gravity_compensation"
        )
    ) is False
    assert ast.literal_eval(
        _assignment_source(
            CFG_PATH, mixin, "cartesian_impedance_nullspace_control"
        )
    ) == "position"
    assert ast.literal_eval(
        _assignment_source(
            CFG_PATH, mixin, "cartesian_impedance_inertial_dynamics_decoupling"
        )
    ) is True

    for field_name in (
        "scripted_action_prior_enabled",
        "scripted_tabletop_pregrasp_prior_enabled",
        "scripted_tabletop_approach_action_prior_enabled",
        "scripted_tabletop_lift_target_prior_enabled",
        "scripted_tabletop_relative_lift_target_prior_enabled",
        "scripted_tabletop_cartesian_lift_target_prior_enabled",
        "scripted_tabletop_hand_grasp_memory_prior_enabled",
    ):
        assert _assignment_source(CFG_PATH, mixin, field_name) == "False"

    env_source = ENV_PATH.read_text(encoding="utf-8")
    assert 'target_mode == "measured_delta"' in env_source
    assert 'target_mode == "integrated_delta"' in env_source
    assert "target_base_pos_b = palm_pos_b" in env_source
    assert "target_base_quat_b = palm_quat_b" in env_source
    assert "target_base_pos_b + translation_scale" in env_source
    assert "quat_mul(delta_quat_b, target_base_quat_b)" in env_source
    assert "OperationalSpaceControllerCfg" in env_source
    assert "subtract_frame_transforms(" in env_source
    assert "self._cartesian_impedance_target_pose_b[:, :3]" in env_source
    assert "self._cartesian_impedance_target_pose_b[:, 3:]" in env_source
    assert "cartesian_impedance_max_position_error" in env_source
    assert "cartesian_impedance_max_rotation_error" in env_source
    assert "get_generalized_mass_matrices()" in env_source
    assert "get_gravity_compensation_forces()" in env_source
    assert "self._default_arm_pos.expand(self.num_envs, -1).clone()" in env_source
    assert "robot.set_joint_effort_target(joint_efforts, joint_ids=arm_ids)" in env_source
    assert "hand_actions = self.actions[:, 6:]" in env_source
    assert "robot.set_joint_position_target(" in env_source

    tracking_source = POSE_TRACKING_PATH.read_text(encoding="utf-8")
    assert 'interface == "cartesian_impedance"' in tracking_source
    assert '"cartesian_impedance_target_mode"' in tracking_source
    assert "base._current_palm_point_pose_b()" in tracking_source
    assert "base._cartesian_impedance_desired_pose_b[:, :3]" in tracking_source
    assert 'interface in {"cartesian_wrist_delta", "cartesian_impedance"}' in tracking_source


def test_inspire_static_impedance_ab_pair_shares_physics_and_reward_contract():
    joint_class = "InspireRH56BFXStaticJointTargetABTeacherEnvCfg"
    impedance_class = "InspireRH56BFXStaticCartesianImpedanceABTeacherEnvCfg"
    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    assert joint_class in classes
    assert impedance_class in classes
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, joint_class, "benchmark_protocol")
    ) == "inspire_static_joint_vs_torque_cartesian_impedance_v1"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, joint_class, "tabletop_arm_lift_progress_mode")
    ) == "palm_z"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, impedance_class, "action_contract")
    ) == "inspire_cartesian_impedance_12d"
    assert _assignment_source(CFG_PATH, impedance_class, "observation_space") == "75"
    assert joint_class in {ast.unparse(base) for base in classes[impedance_class].bases}
    assert "_CartesianImpedanceDirectPolicyContract" in {
        ast.unparse(base) for base in classes[impedance_class].bases
    }

    assert _assignment_source(CFG_PATH, joint_class, "robot_cfg").startswith(
        "_inspire_rh56bfx_mimic_robot_cfg("
    )
    assert _assignment_source(CFG_PATH, impedance_class, "robot_cfg").startswith(
        "_inspire_rh56bfx_impedance_robot_cfg("
    )
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    assert 'robot_cfg.actuators["franka_forearm"].effort_limit_sim = 12.0' in cfg_source
    assert 'robot_cfg.actuators["franka_shoulder"].stiffness = 0.0' in cfg_source
    assert 'robot_cfg.actuators["franka_forearm"].damping = 0.0' in cfg_source

    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert joint_class in task_source
    assert impedance_class in task_source


def test_revo2_static_torque_impedance_contract_is_registered_and_deployable():
    class_name = "Revo2StaticActionInterfaceCartesianImpedancePhaseObsTeacherEnvCfg"
    class_node = _class_node(CFG_PATH, class_name)
    bases = {ast.unparse(base) for base in class_node.bases}
    assert "_CartesianImpedanceDirectPolicyContract" in bases
    assert "_Revo2StaticActionInterfaceAblationContract" in bases
    assert "Revo2UnifiedRollingStage1TeacherEnvCfg" in bases
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, class_name, "action_contract")
    ) == "revo2_cartesian_impedance_12d"
    assert _assignment_source(CFG_PATH, class_name, "observation_space") == "89"
    assert _assignment_source(CFG_PATH, class_name, "robot_cfg").startswith(
        "_v699_revo2_impedance_robot_cfg("
    )
    assert "REVO2_STATIC_CLEAR_HOME_ARM_POS" in _assignment_source(
        CFG_PATH, class_name, "robot_cfg"
    )
    assert _assignment_source(CFG_PATH, class_name, "sim_hand_joint_names") == (
        "REVO2_V699_PHYSICAL_HAND_JOINT_NAMES"
    )
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, class_name, "robot_mimic_natural_frequency")
    ) is None
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, class_name, "robot_mimic_damping_ratio")
    ) is None
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    assert "robot_cfg.spawn.convert_mimic_joints_to_normal_joints = False" in cfg_source
    assert (
        "revo2_hand.joint_names_expr = list(REVO2_V699_PHYSICAL_HAND_JOINT_NAMES)"
        in cfg_source
    )
    assert 'revo2_hand.stiffness = 40.0' in cfg_source
    assert 'revo2_hand.damping = 4.0' in cfg_source
    assert class_name in TASK_INIT_PATH.read_text(encoding="utf-8")


def test_static_strict_from_scratch_ab_is_physical_prior_free_and_registered():
    contract = "_StaticStrictFromScratchControlABContract"
    expected = {
        "benchmark_protocol": "static_strict_from_scratch_control_ab_v1",
        "tabletop_asset_set_enabled": False,
        "dynamic_grasp_speed_curriculum": False,
        "scripted_action_prior_enabled": False,
        "scripted_tabletop_pregrasp_prior_enabled": False,
        "scripted_tabletop_lift_target_prior_enabled": False,
        "tabletop_privileged_lift_target_obs_enabled": False,
        # Lift under gravity plus a current strict opposed enclosure is the
        # physical gate. Distal-only force sensors are intentionally diagnostic.
        "tabletop_success_requires_force_grasp": False,
        "tabletop_post_success_stability_latch_enabled": False,
        "tabletop_post_success_arm_target_lock_enabled": False,
        "tabletop_post_success_hand_target_lock_enabled": False,
        "terminate_on_success": True,
    }
    for field_name, expected_value in expected.items():
        assert ast.literal_eval(_assignment_source(CFG_PATH, contract, field_name)) == expected_value

    assert float(_assignment_source(CFG_PATH, contract, "tabletop_success_lift_height")) == 0.04
    assert int(_assignment_source(CFG_PATH, contract, "dynamic_success_hold_steps")) == 12
    assert float(_assignment_source(CFG_PATH, contract, "strict_success_contact_distance")) == 0.008
    assert float(_assignment_source(CFG_PATH, contract, "object_contact_force_threshold")) == 0.05
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_lift_action_prior_rew_scale")) == 0.0
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_grasped_arm_lift_rew_scale")) == 0.0

    expected_classes = {
        "Revo2StaticStrictJointTargetABTeacherEnvCfg": (13, 80),
        "Revo2StaticStrictCartesianImpedanceABTeacherEnvCfg": (None, 79),
        "InspireRH56BFXStaticStrictJointTargetABTeacherEnvCfg": (13, 80),
        "InspireRH56BFXStaticStrictCartesianImpedanceABTeacherEnvCfg": (None, 79),
    }
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    for class_name, (action_dim, obs_dim) in expected_classes.items():
        assert class_name in task_source
        assert int(_assignment_source(CFG_PATH, class_name, "observation_space")) == obs_dim
        if action_dim is not None:
            assert int(_assignment_source(CFG_PATH, class_name, "action_space")) == action_dim

    for class_name in (
        "Revo2StaticStrictCartesianImpedanceABTeacherEnvCfg",
        "InspireRH56BFXStaticStrictCartesianImpedanceABTeacherEnvCfg",
    ):
        assert ast.literal_eval(
            _assignment_source(CFG_PATH, class_name, "cartesian_impedance_target_mode")
        ) == "measured_delta"

    assert _assignment_source(
        CFG_PATH, "Revo2StaticStrictJointTargetABTeacherEnvCfg", "robot_cfg"
    ).startswith("_v699_revo2_six_motor_robot_cfg(")
    assert _assignment_source(
        CFG_PATH, "Revo2StaticStrictCartesianImpedanceABTeacherEnvCfg", "robot_cfg"
    ).startswith("_v699_revo2_impedance_robot_cfg(")
    assert _assignment_source(
        CFG_PATH, "InspireRH56BFXStaticStrictJointTargetABTeacherEnvCfg", "robot_cfg"
    ).startswith("_inspire_rh56bfx_mimic_robot_cfg(")
    assert _assignment_source(
        CFG_PATH, "InspireRH56BFXStaticStrictCartesianImpedanceABTeacherEnvCfg", "robot_cfg"
    ).startswith("_inspire_rh56bfx_impedance_robot_cfg(")


def test_static_stable_hover_ab_rejects_ballistic_scoop_success():
    contract = "_StaticStableHoverControlABContract"
    expected = {
        "benchmark_protocol": "static_stable_hover_from_scratch_control_ab_v12_markov_phase_scalar",
        "tabletop_success_requires_hover_target": True,
        "tabletop_hover_success_requires_xy": True,
        "tabletop_hover_phase_out_lift_rewards": True,
        "tabletop_clean_grasp_latch_enabled": True,
        "tabletop_lift_requires_clean_grasp_latch": True,
        "tabletop_terminate_on_unclean_lift": True,
        "tabletop_success_requires_force_grasp": False,
        "tabletop_lift_gate_requires_force_grasp": False,
        "tabletop_lift_rewards_require_force_grasp": False,
        "tabletop_force_lift_curriculum_enabled": False,
        "tabletop_hover_latch_requires_force_grasp": False,
        "dynamic_success_hold_steps": 18,
    }
    for field_name, expected_value in expected.items():
        assert ast.literal_eval(_assignment_source(CFG_PATH, contract, field_name)) == expected_value

    assert float(_assignment_source(CFG_PATH, contract, "stable_object_palm_vel")) == 0.12
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_hover_height_delta")) == 0.08
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_hover_success_xy_tolerance")) == 0.06
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_hover_success_z_tolerance")) == 0.025
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_hover_success_object_speed")) == 0.12
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_hover_success_object_ang_speed")) == 2.0
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_hover_post_latch_lift_reward_floor")) == 1.0
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_hover_potential_progress_rew_scale")) == 8000.0
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_force_reward_post_lift_floor")) == 0.0
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_force_grasp_rew_scale")) == 0.0
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_clean_grasp_contact_distance")) == 0.003
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_unclean_lift_height")) == 0.012
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_success_max_palm_xy_drift")) == 0.03
    assert int(_assignment_source(CFG_PATH, contract, "tabletop_force_lift_curriculum_start_frames")) == 0
    assert int(_assignment_source(CFG_PATH, contract, "tabletop_force_lift_curriculum_end_frames")) == 3_000_000

    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    for class_name in (
        "Revo2StaticStableHoverJointTargetABTeacherEnvCfg",
        "Revo2StaticStableHoverCartesianImpedanceABTeacherEnvCfg",
        "InspireRH56BFXStaticStableHoverJointTargetABTeacherEnvCfg",
        "InspireRH56BFXStaticStableHoverCartesianImpedanceABTeacherEnvCfg",
    ):
        assert class_name in cfg_source
        assert class_name in task_source

    env_source = ENV_PATH.read_text(encoding="utf-8")
    assert 'getattr(self.cfg, "tabletop_hover_success_object_ang_speed", 0.0)' in env_source
    assert 'getattr(self.cfg, "tabletop_hover_phase_out_lift_rewards", False)' in env_source
    assert '"hover_lift_reward_phase": hover_lift_reward_phase.mean()' in env_source
    assert 'self._hover_potential_error_prev - hover_potential_error' in env_source
    assert '"hover_potential_progress_rew": hover_potential_progress_rew.mean()' in env_source
    assert 'getattr(self.cfg, "tabletop_hover_latch_requires_force_grasp", False)' in env_source
    assert 'getattr(self.cfg, "tabletop_force_reward_post_lift_floor", 0.0)' in env_source
    assert "def _tabletop_force_lift_curriculum_alpha" in env_source
    assert '"force_lift_curriculum_alpha": torch.tensor(' in env_source
    assert 'self.extras["strict_stable_hover_ang_speed_ok_env"]' in env_source


def test_revo2_official_static_ab_uses_one_native_mimic_hand_and_simple_reward():
    contract = "_StaticOfficialSixActiveFromScratchContract"
    joint_class = "Revo2StaticOfficialJointTargetTeacherEnvCfg"
    cartesian_class = "Revo2StaticOfficialCartesianImpedanceTeacherEnvCfg"

    assert ast.literal_eval(
        _assignment_source(CFG_PATH, contract, "benchmark_protocol")
    ) == "revo2_static_official_six_active_from_scratch_v2_measured_bounded"
    for field_name in (
        "dynamic_grasp_speed_curriculum",
        "dynamic_tabletop_persistent_motion",
        "scripted_action_prior_enabled",
        "scripted_tabletop_pregrasp_prior_enabled",
        "scripted_tabletop_lift_target_prior_enabled",
        "tabletop_post_success_stability_latch_enabled",
    ):
        assert ast.literal_eval(_assignment_source(CFG_PATH, contract, field_name)) is False
    for field_name in (
        "strict_success_enabled",
        "strict_reward_enabled",
        "object_contact_force_diagnostics_enabled",
        "tabletop_success_requires_force_grasp",
        "tabletop_success_requires_hover_target",
        "tabletop_hover_latch_requires_force_grasp",
        "terminate_on_success",
    ):
        assert ast.literal_eval(_assignment_source(CFG_PATH, contract, field_name)) is True
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_success_lift_height")) == 0.04
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_hover_height_delta")) == 0.08
    assert float(_assignment_source(CFG_PATH, contract, "arm_target_tracking_error_limit")) == 0.04
    assert ast.literal_eval(_assignment_source(CFG_PATH, contract, "object_start_pos")) == (
        0.58,
        -0.16,
        0.336,
    )
    assert int(_assignment_source(CFG_PATH, contract, "dynamic_success_hold_steps")) == 12
    assert float(_assignment_source(CFG_PATH, contract, "tabletop_force_grasp_rew_scale")) == 50.0
    assert float(_assignment_source(CFG_PATH, contract, "success_bonus")) == 1000.0

    assert _assignment_source(CFG_PATH, joint_class, "sim_hand_joint_names") == (
        "REVO2_HAND_JOINT_NAMES"
    )
    assert _assignment_source(CFG_PATH, joint_class, "robot_cfg").startswith(
        "_v699_revo2_official_six_active_robot_cfg("
    )
    assert _assignment_source(CFG_PATH, joint_class, "action_space") == "13"
    assert _assignment_source(CFG_PATH, joint_class, "observation_space") == "76"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, joint_class, "policy_action_interface")
    ) == "joint_target"

    cartesian_node = _class_node(CFG_PATH, cartesian_class)
    assert joint_class in {ast.unparse(base) for base in cartesian_node.bases}
    assert "_CartesianImpedanceDirectPolicyContract" in {
        ast.unparse(base) for base in cartesian_node.bases
    }
    assert _assignment_source(CFG_PATH, cartesian_class, "robot_cfg").startswith(
        "_v699_revo2_official_six_active_impedance_robot_cfg("
    )
    assert _assignment_source(CFG_PATH, cartesian_class, "observation_space") == "75"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, cartesian_class, "cartesian_impedance_target_mode")
    ) == "measured_delta"

    env_source = ENV_PATH.read_text(encoding="utf-8")
    active_start = env_source.index("    def _apply_active_joint_target_action")
    active_end = env_source.index("    def _active_hand_actions_to_sim_targets", active_start)
    active_source = env_source[active_start:active_end]
    assert 'getattr(self.cfg, "arm_target_tracking_error_limit", 0.0)' in active_source
    assert "measured_arm_pos - tracking_error_limit" in active_source
    assert "measured_arm_pos + tracking_error_limit" in active_source

    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert "SimToolReal-Revo2-Franka-StaticOfficialJointTarget-Teacher-Direct-v0" in task_source
    assert (
        "SimToolReal-Revo2-Franka-StaticOfficialCartesianImpedance-Teacher-Direct-v0"
        in task_source
    )


def test_inspire_official_static_ab_reuses_the_shared_task_and_action_semantics():
    contract = "_StaticOfficialSixActiveFromScratchContract"
    joint_class = "InspireStaticOfficialJointTargetTeacherEnvCfg"
    cartesian_class = "InspireStaticOfficialCartesianImpedanceTeacherEnvCfg"

    joint_node = _class_node(CFG_PATH, joint_class)
    joint_bases = {ast.unparse(base) for base in joint_node.bases}
    assert contract in joint_bases
    assert "InspireDynamicTabletopTeacherEnvCfg" in joint_bases
    assert ast.literal_eval(_assignment_source(CFG_PATH, joint_class, "action_contract")) == (
        "inspire_semantic_13d"
    )
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, joint_class, "policy_action_interface")
    ) == "joint_target"
    assert _assignment_source(CFG_PATH, joint_class, "action_space") == "13"
    assert _assignment_source(CFG_PATH, joint_class, "observation_space") == "76"
    assert _assignment_source(CFG_PATH, joint_class, "sim_hand_joint_names") == (
        "INSPIRE_ACTIVE_HAND_JOINT_NAMES"
    )
    assert _assignment_source(CFG_PATH, joint_class, "robot_cfg").startswith(
        "_inspire_rh56bfx_mimic_robot_cfg("
    )

    cartesian_node = _class_node(CFG_PATH, cartesian_class)
    cartesian_bases = {ast.unparse(base) for base in cartesian_node.bases}
    assert joint_class in cartesian_bases
    assert "_CartesianImpedanceDirectPolicyContract" in cartesian_bases
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, cartesian_class, "cartesian_impedance_target_mode")
    ) == "measured_delta"
    assert _assignment_source(CFG_PATH, cartesian_class, "observation_space") == "75"
    assert _assignment_source(CFG_PATH, cartesian_class, "robot_cfg").startswith(
        "_inspire_rh56bfx_impedance_robot_cfg("
    )

    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert "SimToolReal-Inspire-Franka-StaticOfficialJointTarget-Teacher-Direct-v0" in task_source
    assert (
        "SimToolReal-Inspire-Franka-StaticOfficialCartesianImpedance-Teacher-Direct-v0"
        in task_source
    )


def test_official_sphere_ab_enforces_one_global_action_choice_for_both_hands():
    contract = "_StaticOfficialSphereLiftControlABContract"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, contract, "benchmark_protocol")
    ) == "static_official_sphere_lift_global_action_ab_v1_acquisition"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, contract, "global_action_selection_scope")
    ) == "revo2_and_inspire"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, contract, "hand_action_semantics")
    ) == "six_physical_motor_absolute_target"
    assert float(
        _assignment_source(CFG_PATH, contract, "arm_target_tracking_error_limit")
    ) == 0.04

    lift_contract = "_StaticOfficialSphereForcePalmLiftStage2Contract"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, lift_contract, "tabletop_success_requires_force_grasp")
    ) is True
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, lift_contract, "tabletop_lift_gate_requires_force_grasp")
    ) is True
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, lift_contract, "tabletop_lift_rewards_require_force_grasp")
    ) is True
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, lift_contract, "tabletop_arm_lift_progress_mode")
    ) == "palm_z"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, lift_contract, "tabletop_arm_lift_progress_baseline_mode")
    ) == "first_force_grasp"
    assert float(
        _assignment_source(CFG_PATH, lift_contract, "tabletop_grasped_arm_lift_rew_scale")
    ) == 50000.0

    force_hold_contract = "_StaticOfficialSphereForceHoldStage2Contract"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, force_hold_contract, "benchmark_protocol")
    ) == "static_official_sphere_global_action_ab_v5_stage2_force_hold"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, force_hold_contract, "global_action_selection_scope")
    ) == "revo2_and_inspire"
    assert float(
        _assignment_source(CFG_PATH, force_hold_contract, "strict_success_contact_distance")
    ) == 0.002
    assert float(
        _assignment_source(CFG_PATH, force_hold_contract, "tabletop_grasped_arm_lift_rew_scale")
    ) == 0.0
    assert int(
        _assignment_source(CFG_PATH, force_hold_contract, "tabletop_force_grasp_streak_target")
    ) == 20
    for field_name in (
        "true_grasp_rew_scale",
        "grasp_quality_rew_scale",
        "tabletop_underwrap_rew_scale",
        "tabletop_strict_hold_rew_scale",
        "lift_progress_rew_scale",
        "tabletop_grasped_arm_lift_rew_scale",
        "tabletop_hover_goal_rew_scale",
    ):
        assert float(_assignment_source(CFG_PATH, force_hold_contract, field_name)) == 0.0

    pressure_hold_contract = "_StaticOfficialSphereOpposedPressureHoldStage2Contract"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, pressure_hold_contract, "benchmark_protocol")
    ) == "static_official_sphere_global_action_ab_v6_opposed_pressure_hold"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, pressure_hold_contract, "global_action_selection_scope")
    ) == "revo2_and_inspire"
    assert float(
        _assignment_source(CFG_PATH, pressure_hold_contract, "tabletop_opposed_force_pressure_target")
    ) == 5.0
    assert float(
        _assignment_source(
            CFG_PATH, pressure_hold_contract, "tabletop_opposed_force_pressure_rew_scale"
        )
    ) == 1000.0
    assert float(
        _assignment_source(CFG_PATH, pressure_hold_contract, "tabletop_force_grasp_loss_penalty_scale")
    ) == 0.0

    paired_classes = (
        "Revo2StaticOfficialSphereJointTargetTeacherEnvCfg",
        "Revo2StaticOfficialSphereCartesianImpedanceTeacherEnvCfg",
        "InspireStaticOfficialSphereJointTargetTeacherEnvCfg",
        "InspireStaticOfficialSphereCartesianImpedanceTeacherEnvCfg",
    )
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    for class_name in paired_classes:
        assert class_name in task_source

    for joint_class in (
        "Revo2StaticOfficialSphereJointTargetTeacherEnvCfg",
        "InspireStaticOfficialSphereJointTargetTeacherEnvCfg",
    ):
        assert contract in {
            ast.unparse(base) for base in _class_node(CFG_PATH, joint_class).bases
        }
        assert _assignment_source(CFG_PATH, joint_class, "observation_space") == "80"

    for cartesian_class, joint_class in (
        (
            "Revo2StaticOfficialSphereCartesianImpedanceTeacherEnvCfg",
            "Revo2StaticOfficialSphereJointTargetTeacherEnvCfg",
        ),
        (
            "InspireStaticOfficialSphereCartesianImpedanceTeacherEnvCfg",
            "InspireStaticOfficialSphereJointTargetTeacherEnvCfg",
        ),
    ):
        bases = {
            ast.unparse(base) for base in _class_node(CFG_PATH, cartesian_class).bases
        }
        assert joint_class in bases
        assert "_CartesianImpedanceDirectPolicyContract" in bases
        assert ast.literal_eval(
            _assignment_source(CFG_PATH, cartesian_class, "cartesian_impedance_target_mode")
        ) == "measured_delta"
        assert _assignment_source(CFG_PATH, cartesian_class, "observation_space") == "79"

    stage2_classes = (
        "Revo2StaticOfficialSphereLiftStage2JointTargetTeacherEnvCfg",
        "Revo2StaticOfficialSphereLiftStage2CartesianImpedanceTeacherEnvCfg",
        "InspireStaticOfficialSphereLiftStage2JointTargetTeacherEnvCfg",
        "InspireStaticOfficialSphereLiftStage2CartesianImpedanceTeacherEnvCfg",
    )
    for class_name in stage2_classes:
        assert class_name in task_source

    force_hold_stage2_classes = (
        "Revo2StaticOfficialSphereForceHoldStage2JointTargetTeacherEnvCfg",
        "Revo2StaticOfficialSphereForceHoldStage2CartesianImpedanceTeacherEnvCfg",
        "InspireStaticOfficialSphereForceHoldStage2JointTargetTeacherEnvCfg",
        "InspireStaticOfficialSphereForceHoldStage2CartesianImpedanceTeacherEnvCfg",
    )
    for class_name in force_hold_stage2_classes:
        assert class_name in task_source

    pressure_hold_stage2_classes = (
        "Revo2StaticOfficialSphereOpposedPressureHoldStage2JointTargetTeacherEnvCfg",
        "Revo2StaticOfficialSphereOpposedPressureHoldStage2CartesianImpedanceTeacherEnvCfg",
        "InspireStaticOfficialSphereOpposedPressureHoldStage2JointTargetTeacherEnvCfg",
        "InspireStaticOfficialSphereOpposedPressureHoldStage2CartesianImpedanceTeacherEnvCfg",
    )
    for class_name in pressure_hold_stage2_classes:
        assert class_name in task_source


def test_static_strict_ab_uses_the_same_global_action_family_for_both_hands():
    contract = "_StaticStrictFromScratchControlABContract"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, contract, "global_action_selection_scope")
    ) == "revo2_and_inspire"
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, contract, "hand_action_semantics")
    ) == "six_physical_motor_absolute_target"

    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    for class_name in (
        "Revo2StaticStrictJointTargetABTeacherEnvCfg",
        "Revo2StaticStrictCartesianImpedanceABTeacherEnvCfg",
        "InspireRH56BFXStaticStrictJointTargetABTeacherEnvCfg",
        "InspireRH56BFXStaticStrictCartesianImpedanceABTeacherEnvCfg",
    ):
        assert class_name in task_source


def test_revo2_static_from_scratch_contact_curriculum_is_matched_and_does_not_overpenalize_reach():
    contract = "_Revo2StaticFromScratchContactContract"
    assert ast.literal_eval(_assignment_source(CFG_PATH, contract, "benchmark_protocol")) == (
        "revo2_static_action_interface_ablation_v22_from_scratch_contact"
    )
    assert float(
        _assignment_source(CFG_PATH, contract, "tabletop_non_thumb_without_thumb_penalty_scale")
    ) == 300.0

    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    for class_name in (
        "Revo2StaticActionInterfaceJointDeltaFromScratchContactPhaseObsTeacherEnvCfg",
        "Revo2StaticActionInterfaceCartesianFromScratchContactPhaseObsTeacherEnvCfg",
    ):
        class_node = _class_node(CFG_PATH, class_name)
        assert contract in {ast.unparse(base) for base in class_node.bases}
        assert class_name in task_source


def test_revo2_static_stage2_micro_lift_bridge_is_state_based_and_shared():
    bridge = "_Revo2StaticMicroLiftVelocityBridgeContract"
    expected_transition_scales = {
        "contact_rew_scale": 120.0,
        "grasp_quality_rew_scale": 700.0,
        "opposition_rew_scale": 400.0,
        "true_grasp_rew_scale": 1500.0,
        "strict_touch_rew_scale": 1000.0,
        "strict_opposition_touch_rew_scale": 3000.0,
        "dynamic_tabletop_palm_frame_pregrasp_rew_scale": 600.0,
        "tabletop_underwrap_rew_scale": 850.0,
        "tabletop_strict_hold_rew_scale": 8000.0,
        "tabletop_no_lift_after_grasp_penalty_scale": 6000.0,
    }
    for name, expected in expected_transition_scales.items():
        assert float(_assignment_source(CFG_PATH, bridge, name)) == expected
    assert float(_assignment_source(CFG_PATH, bridge, "tabletop_palm_object_up_vel_rew_scale")) == 48000.0
    assert float(_assignment_source(CFG_PATH, bridge, "tabletop_vertical_palm_velocity_rew_scale")) == 96000.0
    assert float(_assignment_source(CFG_PATH, bridge, "tabletop_palm_object_up_vel_target")) == 0.08
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, bridge, "tabletop_strict_hold_reward_latches_at_target")
    ) is True
    assert float(
        _assignment_source(CFG_PATH, bridge, "tabletop_strict_hold_post_target_multiplier")
    ) == 0.05
    assert ast.literal_eval(_assignment_source(CFG_PATH, bridge, "benchmark_protocol")) == (
        "revo2_static_action_interface_ablation_v26_force_gated_micro_lift_balanced"
    )
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, bridge, "pregrasp_target_rel_palm")
    ) == (0.015, 0.016, 0.058)
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, bridge, "object_contact_force_diagnostics_enabled")
    ) is True
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, bridge, "tabletop_arm_lift_progress_baseline_mode")
    ) == "first_force_grasp"
    assert int(
        _assignment_source(CFG_PATH, bridge, "tabletop_arm_lift_progress_baseline_grasp_streak")
    ) == 4
    for flag in (
        "tabletop_lift_gate_requires_force_grasp",
        "tabletop_lift_rewards_require_force_grasp",
        "tabletop_lift_rewards_require_current_strict_grasp",
        "tabletop_no_lift_uses_force_grasp_gate",
        "tabletop_success_requires_force_grasp",
    ):
        assert ast.literal_eval(_assignment_source(CFG_PATH, bridge, flag)) is True
    assert float(_assignment_source(CFG_PATH, bridge, "tabletop_force_grasp_rew_scale")) == 1000.0
    assert float(
        _assignment_source(CFG_PATH, bridge, "tabletop_force_grasp_streak_rew_scale")
    ) == 1500.0
    assert float(
        _assignment_source(CFG_PATH, bridge, "tabletop_force_stable_grasp_rew_scale")
    ) == 2000.0
    assert float(
        _assignment_source(CFG_PATH, bridge, "tabletop_force_grasp_loss_penalty_scale")
    ) == 8000.0
    for class_name in (
        "Revo2StaticActionInterfaceJointDeltaStage2LiftBridgePhaseObsTeacherEnvCfg",
        "Revo2StaticActionInterfaceCartesianStage2LiftBridgePhaseObsTeacherEnvCfg",
    ):
        class_node = _class_node(CFG_PATH, class_name)
        assert bridge in {ast.unparse(base) for base in class_node.bases}
        assert class_name in TASK_INIT_PATH.read_text(encoding="utf-8")

    for class_name in (
        "Revo2StaticActionInterfaceJointDeltaStage2HoldPhaseObsTeacherEnvCfg",
        "Revo2StaticActionInterfaceCartesianStage2HoldPhaseObsTeacherEnvCfg",
    ):
        class_node = _class_node(CFG_PATH, class_name)
        assert bridge not in {ast.unparse(base) for base in class_node.bases}

    env_source = ENV_PATH.read_text(encoding="utf-8")
    assert "coupled_up_vel = torch.minimum(" in env_source
    assert "current_lift_grasp_gate" in env_source
    assert "vertical_palm_velocity_pose_score" in env_source
    assert 'getattr(self.cfg, "tabletop_strict_hold_reward_latches_at_target", False)' in env_source
    assert "self._tabletop_strict_hold_completed" in env_source
    assert '"tabletop_strict_hold_post_target_multiplier"' in env_source


def test_revo2_palm_frame_acquisition_is_state_based_shared_and_traceable():
    class_name = "Revo2UnifiedRollingStage1AcquisitionCartesianPalmFramePhaseObsTeacherEnvCfg"
    static_ablation_class = "_Revo2StaticActionInterfaceAblationContract"
    assert float(
        _assignment_source(CFG_PATH, "_UnifiedRollingRewardContract", "dynamic_tabletop_palm_frame_pregrasp_rew_scale")
    ) == 0.0
    assert float(
        _assignment_source(CFG_PATH, class_name, "dynamic_tabletop_palm_frame_pregrasp_rew_scale")
    ) > 0.0
    assert ast.literal_eval(_assignment_source(CFG_PATH, class_name, "pregrasp_target_rel_palm")) == (
        0.030,
        0.007,
        0.075,
    )
    assert float(
        _assignment_source(
            CFG_PATH,
            static_ablation_class,
            "dynamic_tabletop_palm_frame_pregrasp_rew_scale",
        )
    ) == 1800.0
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, static_ablation_class, "pregrasp_target_rel_palm")
    ) == (0.030, 0.007, 0.075)
    assert ast.literal_eval(
        _assignment_source(CFG_PATH, static_ablation_class, "pregrasp_target_scale")
    ) == (0.030, 0.025, 0.030)
    assert (
        _assignment_source(CFG_PATH, static_ablation_class, "touch_body_names")
        == "REVO2_DISTAL_CONTACT_BODY_NAMES"
    )
    assert (
        _assignment_source(
            CFG_PATH,
            static_ablation_class,
            "object_contact_force_diagnostics_enabled",
        )
        == "False"
    )

    env_source = ENV_PATH.read_text(encoding="utf-8")
    assert "object_pos_palm = quat_rotate_inverse(" in env_source
    assert "torch.reciprocal(1.0 + palm_frame_pregrasp_error_sq)" in env_source
    assert '"palm_frame_pregrasp_rew"' in env_source

    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert class_name in task_source
    eval_source = EVAL_PATH.read_text(encoding="utf-8")
    assert '"object_pos_palm"' in eval_source
    assert '"--diagnostic-hand-close-from-step"' in eval_source
    assert '"--diagnostic-hold-arm-while-closing"' in eval_source
    assert "actions[close_mask, arm_action_width:]" in eval_source
    assert "actions[close_mask, :arm_action_width] = 0.0" in eval_source


def test_teacher_student_schema_preserves_six_real_hand_motors():
    source = SCHEMA_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    contracts = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "ACTION_CONTRACTS":
                contracts = node.value
                break
    assert isinstance(contracts, ast.Dict)
    keys = [ast.literal_eval(key) for key in contracts.keys]
    for contract_name in (
        "revo2_cartesian_wrist_12d",
        "inspire_cartesian_wrist_12d",
    ):
        assert contract_name in keys
        value = contracts.values[keys.index(contract_name)]
        kwargs = {keyword.arg: keyword.value for keyword in value.keywords}
        assert ast.literal_eval(kwargs["action_dim"]) == 12
        assert ast.literal_eval(kwargs["arm_dim"]) == 6
        assert ast.literal_eval(kwargs["hand_dim"]) == 6


def test_existing_joint_target_curriculum_contract_is_unchanged():
    assert _assignment_source(
        CFG_PATH, "Revo2DynamicDexterousTeacherEnvCfg", "action_contract"
    ) == '"revo2_semantic_13d"'
    assert _assignment_source(
        CFG_PATH, "InspireDynamicDexterousTeacherEnvCfg", "action_space"
    ) == "13"
    assert _assignment_source(
        CFG_PATH, "InspireDynamicDexterousTeacherEnvCfg", "action_contract"
    ) == '"inspire_semantic_13d"'


def test_eval_trace_uses_the_runtime_arm_action_width():
    source = EVAL_PATH.read_text(encoding="utf-8")

    assert "def _policy_arm_action_width(" in source
    assert "arm_width = int(resolver())" in source
    assert "hand_action_start = _policy_arm_action_width(unwrapped_env, action_width)" in source
