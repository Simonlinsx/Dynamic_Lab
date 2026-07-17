import ast
from pathlib import Path


CFG_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "simtoolreal_lab"
    / "simtoolreal_lab"
    / "tasks"
    / "dynamic_dexterous_grasp"
    / "dynamic_dexterous_grasp_env_cfg.py"
)
ENV_PATH = CFG_PATH.with_name("dynamic_dexterous_grasp_env.py")


def _class_constants(class_name: str) -> dict[str, object]:
    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    constants: dict[str, object] = {}
    for statement in node.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            try:
                constants[target.id] = ast.literal_eval(statement.value)
            except ValueError:
                constants[target.id] = ast.unparse(statement.value)
    return constants


def _class_assignment_source(class_name: str, field_name: str) -> str:
    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    for statement in node.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == field_name
        ):
            return ast.unparse(statement.value)
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == field_name
            and statement.value is not None
        ):
            return ast.unparse(statement.value)
    raise AssertionError(f"{field_name!r} is not assigned in {class_name}")


def test_stage3_force_contacts_are_diagnostics_not_reward_or_success_gates():
    values = _class_constants("_UnifiedRollingLiftHoldStage3Contract")

    assert values["joint_target_arm_max_delta"] == 0.04
    assert values["joint_target_hand_max_delta"] == 0.05
    assert values["joint_target_rate_limit_requires_lift_baseline"] is True
    assert values["tabletop_lift_hand_target_lock_enabled"] is True
    assert values["tabletop_lift_hand_target_lock_blend"] == 1.0
    assert values["tabletop_lift_hand_target_close_fraction"] == 0.15
    assert values["object_contact_force_diagnostics_enabled"] is True
    assert values["tabletop_force_grasp_rew_scale"] == 0.0
    assert values["tabletop_force_grasp_streak_rew_scale"] == 0.0
    assert values["tabletop_force_stable_grasp_rew_scale"] == 0.0
    assert values["tabletop_force_grasp_loss_penalty_scale"] == 0.0
    assert values["tabletop_lift_rewards_require_force_grasp"] is False
    assert values["tabletop_lift_rewards_require_current_strict_grasp"] is True
    assert values["tabletop_lift_gate_requires_force_grasp"] is False
    assert values["tabletop_no_lift_uses_force_grasp_gate"] is False
    assert values["tabletop_success_requires_force_grasp"] is False
    assert values["tabletop_hover_latch_uses_grasp_seen"] is False
    assert values["tabletop_hover_reward_uses_grasp_seen"] is False
    assert values["tabletop_success_uses_grasp_seen"] is False
    assert values["tabletop_strict_hold_rew_scale"] == 0.0
    assert values["tabletop_strict_grasp_loss_penalty_scale"] == 12000.0
    assert values["tabletop_strict_grasp_loss_requires_lift_baseline"] is True
    assert values["tabletop_strict_grasp_loss_on_transition_only"] is True
    assert values["tabletop_hover_post_latch_speed_penalty_scale"] == 180.0
    assert values["tabletop_arm_lift_progress_baseline_mode"] == "first_strict_grasp"
    assert values["tabletop_arm_lift_progress_baseline_grasp_streak"] == 2


def test_stage3_carry_reward_keeps_current_strict_grasp_streak_during_lift():
    values = _class_constants("_UnifiedRollingLiftHoldStage3Contract")
    assert values["tabletop_object_carry_uses_lift_baseline_grasp_streak"] is True

    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
    carry_streak_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "carry_grasp_streak" for target in node.targets)
    ]
    assert any(
        isinstance(node.value, ast.Attribute)
        and node.value.attr == "_tabletop_arm_lift_baseline_grasp_streak"
        for node in carry_streak_assignments
    )
    assert any(
        isinstance(node, ast.Name) and node.id == "carry_grasp_streak"
        for assignment in ast.walk(tree)
        if isinstance(assignment, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "carry_streak_gate" for target in assignment.targets)
        for node in ast.walk(assignment.value)
    )


def test_stage3_strict_grasp_loss_uses_previous_step_and_resets_state():
    source = ENV_PATH.read_text(encoding="utf-8")

    assert "strict_grasp_loss_gate = self._strict_reward_grasp_prev.float()" in source
    assert "self._strict_reward_grasp_prev.copy_(reward_true_grasp)" in source
    assert "self._strict_reward_grasp_prev[env_ids] = False" in source


def test_joint_target_interface_rate_limits_arm_and_hand_targets():
    source = ENV_PATH.read_text(encoding="utf-8")

    assert source.count("self._rate_limit_joint_target_delta(") >= 4
    assert 'getattr(self.cfg, "joint_target_arm_max_delta", 0.0)' in source
    assert 'getattr(self.cfg, "joint_target_hand_max_delta", 0.0)' in source
    assert 'getattr(self.cfg, "joint_target_rate_limit_requires_lift_baseline", False)' in source


def test_stage3_latches_and_holds_hand_target_after_stable_grasp():
    source = ENV_PATH.read_text(encoding="utf-8")

    assert source.count("self._apply_tabletop_lift_hand_target_lock(") >= 2
    assert "self._tabletop_lift_hand_joint_target[baseline_latch_ids]" in source
    assert "self._tabletop_lift_hand_joint_target[env_ids]" in source
    assert "calibrated_close_targets = self._active_hand_actions_to_sim_targets" in source


def test_stage2_hold_contract_requires_sustained_contact_and_a_micro_lift():
    values = _class_constants("_UnifiedRollingGraspHoldStage2Contract")

    assert values["tabletop_strict_hold_rew_scale"] == 40000.0
    assert values["tabletop_strict_hold_uses_streak_progress"] is True
    assert values["tabletop_strict_hold_min_streak_multiplier"] == 0.125
    assert values["tabletop_strict_grasp_loss_penalty_scale"] == 12000.0
    assert values["tabletop_strict_grasp_hold_steps"] == 20
    assert values["tabletop_underwrap_rew_scale"] == 8500.0
    assert (
        _class_assignment_source(
            "_UnifiedRollingGraspHoldStage2Contract",
            "tabletop_underwrap_min_non_thumb_contacts",
        )
        == "UNIFIED_ROLLING_STRICT_MIN_NON_THUMB_CONTACTS"
    )
    assert values["tabletop_underwrap_progress_weight"] == 0.65
    assert values["tabletop_underwrap_pair_weight"] == 0.35
    assert values["tabletop_underwrap_uses_pregrasp_gate"] is False
    assert _class_assignment_source(
        "_UnifiedRollingGraspHoldStage2Contract", "lift_success_height"
    ) == "UNIFIED_ROLLING_LOAD_BEARING_LIFT_HEIGHT"
    assert _class_assignment_source(
        "_UnifiedRollingGraspHoldStage2Contract", "tabletop_success_lift_height"
    ) == "UNIFIED_ROLLING_LOAD_BEARING_LIFT_HEIGHT"
    assert values["lift_progress_rew_scale"] == 5000.0
    assert values["quality_lift_progress_rew_scale"] == 7000.0
    assert values["lifted_true_grasp_rew_scale"] == 14000.0
    assert values["tabletop_grasped_arm_lift_rew_scale"] == 60000.0
    assert values["tabletop_arm_lift_reward_object_margin"] == 0.16
    assert values["stable_hold_rew_scale"] == 16000.0
    assert values["hold_progress_rew_scale"] == 22000.0
    assert values["success_bonus"] == 36000.0
    assert values["tabletop_lift_rewards_require_current_strict_grasp"] is True
    assert values["tabletop_success_uses_grasp_seen"] is False


def test_stage2_hold_reward_uses_consecutive_strict_contact_progress():
    source = ENV_PATH.read_text(encoding="utf-8")

    assert "tabletop_strict_hold_streak_progress" in source
    assert "self._tabletop_strict_true_grasp_streak.float() + 1.0" in source
    assert 'getattr(self.cfg, "tabletop_strict_hold_uses_streak_progress", False)' in source
    assert '"tabletop_strict_hold_min_streak_multiplier"' in source


def test_stage3_lift_action_alignment_is_interface_aware_and_strict_grasp_gated():
    values = _class_constants("_UnifiedRollingLiftHoldStage3Contract")

    assert values["tabletop_lift_action_prior_rew_scale"] == 9000.0
    assert values["tabletop_lift_action_prior_gate_min"] == 0.0
    assert values["tabletop_lift_gate_requires_current_strict_grasp"] is True

    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
    interface_branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(item, ast.Attribute) and item.attr == "policy_action_interface"
            for item in ast.walk(node.test)
        )
        and any(
            isinstance(item, ast.Name) and item.id == "tabletop_lift_action_prior"
            for statement in node.body
            for item in ast.walk(statement)
        )
    ]
    assert interface_branches
    assert any(
        isinstance(item, ast.Attribute) and item.attr == "_tabletop_arm_lift_baseline_pos"
        for item in ast.walk(interface_branches[0])
    )
    assert any(
        isinstance(item, ast.Attribute) and item.attr == "_tabletop_arm_lift_baseline_latched"
        for item in ast.walk(interface_branches[0])
    )


def test_inspire_unified_reset_uses_table_clear_pose_and_matching_lift_adapter():
    class_name = "InspireUnifiedRollingBenchmarkTeacherEnvCfg"

    assert _class_assignment_source(class_name, "robot_cfg") == (
        "_inspire_z180_robot_cfg(INSPIRE_V341_CLEAR_ARM_POS)"
    )
    assert _class_assignment_source(class_name, "default_arm_pos") == "INSPIRE_V341_CLEAR_ARM_POS"
    assert _class_assignment_source(class_name, "tabletop_arm_lift_progress_baseline_pos") == (
        "INSPIRE_V341_CLEAR_ARM_POS"
    )
    assert _class_assignment_source(class_name, "lift_arm_delta") == "INSPIRE_V340_LIFT_ARM_DELTA"
    assert _class_assignment_source(class_name, "lift_action_prior") == "INSPIRE_V340_LIFT_ACTION_PRIOR"
