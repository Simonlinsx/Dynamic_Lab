"""Contracts for clean tabletop enclosure before vertical lift."""

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
EVALUATION_PATH = ROOT / (
    "source/simtoolreal_lab/simtoolreal_lab/teacher_student/evaluation.py"
)
TASK_INIT_PATH = ROOT / (
    "source/simtoolreal_lab/simtoolreal_lab/tasks/dynamic_dexterous_grasp/__init__.py"
)
TRAIN_SCRIPT_PATH = ROOT / "scripts/train_rl_games.py"
EVAL_SCRIPT_PATH = ROOT / "scripts/evaluate_rl_games.py"


def _classes(path: Path) -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef)
    }


def _assignment(class_node: ast.ClassDef, field_name: str):
    for statement in class_node.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(isinstance(target, ast.Name) and target.id == field_name for target in targets):
            return ast.literal_eval(statement.value)
    raise AssertionError(f"Missing {class_node.name}.{field_name}")


def test_static_hover_tasks_share_one_clean_air_grasp_contract():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticStableHoverControlABContract"]

    assert _assignment(contract, "tabletop_clean_grasp_latch_enabled") is True
    assert _assignment(contract, "tabletop_lift_requires_clean_grasp_latch") is True
    assert _assignment(contract, "tabletop_arm_lift_baseline_requires_clean_grasp_latch") is True
    assert _assignment(contract, "tabletop_terminate_on_unclean_lift") is True
    assert _assignment(contract, "tabletop_unclean_lift_height") == 0.012
    assert _assignment(contract, "tabletop_clean_grasp_contact_distance") <= 0.003
    assert _assignment(contract, "tabletop_clean_grasp_min_non_thumb_contacts") >= 2
    assert _assignment(contract, "tabletop_success_max_palm_xy_drift") <= 0.03
    assert _assignment(
        contract,
        "tabletop_opposed_force_pressure_requires_strict_opposition",
    ) is True
    assert _assignment(
        contract,
        "tabletop_opposed_force_pressure_requires_clearance",
    ) is True
    assert _assignment(contract, "tabletop_strict_hold_rew_scale") == 0.0
    assert _assignment(contract, "tabletop_clean_grasp_latch_bonus") >= 200000.0
    assert _assignment(contract, "success_bonus") >= 1000000.0
    assert _assignment(contract, "tabletop_acquisition_reward_post_clean_latch_floor") <= 0.05
    assert _assignment(contract, "tabletop_strict_grasp_milestone_enabled") is True
    assert _assignment(contract, "tabletop_strict_grasp_milestone_bonus") >= 100000.0
    assert _assignment(contract, "tabletop_acquisition_reward_post_strict_grasp_floor") <= 0.05
    assert _assignment(contract, "tabletop_clean_settle_strict_grasp_rew_scale") <= 250.0
    assert _assignment(contract, "tabletop_clean_grasp_readiness_rew_scale") > 0.0
    assert _assignment(contract, "tabletop_clean_grasp_readiness_potential_gamma") == 0.99

    for class_name in (
        "Revo2StaticStableHoverJointTargetABTeacherEnvCfg",
        "Revo2StaticStableHoverCartesianImpedanceABTeacherEnvCfg",
        "InspireRH56BFXStaticStableHoverJointTargetABTeacherEnvCfg",
        "InspireRH56BFXStaticStableHoverCartesianImpedanceABTeacherEnvCfg",
    ):
        bases = {ast.unparse(base) for base in classes[class_name].bases}
        assert "_StaticStableHoverControlABContract" in bases


def test_clean_grasp_requires_close_opposed_fingers_clearance_and_table_height():
    source = ENV_PATH.read_text(encoding="utf-8")

    assert "def _update_tabletop_clean_grasp_latch" in source
    assert "close_contacts[:, 0]" in source
    assert "close_contacts[:, 1:].sum(dim=-1) >= min_non_thumb" in source
    assert "& self._strict_opposing_contact" in source
    assert "& self._tabletop_arm_clearance_ok" in source
    assert "torch.abs(self._object_height_delta) <= max_height_delta" in source
    assert "self._tabletop_clean_grasp_latched |= " in source
    assert "self._tabletop_clean_grasp_just_latched.copy_(" in source


def test_acquisition_shaping_phases_out_after_clean_latch():
    source = ENV_PATH.read_text(encoding="utf-8")

    assert "clean_settle_reward_phase = post_latch_floor + (" in source
    assert "coarse_acquisition_reward_phase = post_strict_floor + (" in source
    assert "self._tabletop_strict_grasp_milestone_seen |= reward_true_grasp" in source
    assert "potential_gamma * clean_grasp_readiness" in source
    assert "- self._tabletop_clean_grasp_readiness_prev" in source
    assert "* acquisition_reward_phase" in source
    assert '"tabletop_acquisition_reward_phase": acquisition_reward_phase.mean()' in source


def test_unclean_lift_is_penalized_terminated_and_excluded_from_success():
    source = ENV_PATH.read_text(encoding="utf-8")

    assert "tabletop_unclean_lift_progress = lift_progress * (1.0 - clean_lift_gate)" in source
    assert "tabletop_unclean_lift_penalty_scale" in source
    assert "tabletop_terminate_on_unclean_lift" in source
    assert "terminated = terminated | unclean_lift" in source
    assert "hover_success_grasp = hover_success_grasp & clean_grasp_latched" in source
    assert "tabletop_success_grasp = tabletop_success_grasp & clean_grasp_latched" in source
    assert "baseline_grasp = baseline_grasp & self._tabletop_clean_grasp_latched" in source
    assert "no_lift_after_grasp_penalty * clean_lift_gate" in source
    assert "grasped_palm_lift_rew = grasped_palm_lift_rew * clean_lift_gate" in source


def test_clean_grasp_funnel_distinguishes_scoop_from_clean_lift():
    source = EVALUATION_PATH.read_text(encoding="utf-8")

    assert '"clean_grasp_latched": "tabletop_clean_grasp_latched_env"' in source
    assert '"unclean_lift": "tabletop_unclean_lift_env"' in source
    assert '("clean_grasp_to_clean_lift", "clean_grasp_latched", "clean_lifted")' in source
    assert '"clean_grasp_to_retained_grip"' in source
    assert '"retained_grip_to_clean_lift"' in source
    assert '("no_post_clean_grip_retention", "post_clean_grip_retained")' in source
    assert '("no_clean_grasp_latch", "clean_grasp_latched")' in source


def test_static_hover_teacher_has_shared_compact_contact_observation():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticStableHoverControlABContract"]

    assert _assignment(contract, "tabletop_privileged_contact_obs_enabled") is True
    assert _assignment(contract, "tabletop_privileged_contact_obs_dim") == 15
    assert _assignment(
        classes["Revo2StaticStableHoverJointTargetABTeacherEnvCfg"],
        "observation_space",
    ) == 95
    assert _assignment(
        classes["Revo2StaticStableHoverCartesianImpedanceABTeacherEnvCfg"],
        "observation_space",
    ) == 94
    assert _assignment(
        classes["InspireRH56BFXStaticStableHoverJointTargetABTeacherEnvCfg"],
        "observation_space",
    ) == 95
    assert _assignment(
        classes["InspireRH56BFXStaticStableHoverCartesianImpedanceABTeacherEnvCfg"],
        "observation_space",
    ) == 94

    source = ENV_PATH.read_text(encoding="utf-8")
    assert "tabletop_privileged_contact_obs_enabled" in source
    assert "self._object_fingertip_contact_forces / force_scale" in source
    assert "self._strict_opposing_contact.float().unsqueeze(-1)" in source
    assert "self._object_palm_rel_vel / relative_speed_scale" in source
    assert "self._tabletop_arm_clearance_ok.float().unsqueeze(-1)" in source
    assert "clean_grasp_phase_obs = 0.5 * (" in source
    assert "self._tabletop_strict_grasp_milestone_seen.float()" in source
    assert "self._tabletop_clean_grasp_latched.float()" in source


def test_v13_clean_contact_curriculum_tightens_to_the_v12_physical_gate():
    classes = _classes(CFG_PATH)
    strict_contract = classes["_StaticStableHoverControlABContract"]
    curriculum = classes["_StaticCleanGraspCurriculumContract"]

    assert _assignment(curriculum, "tabletop_clean_grasp_curriculum_enabled") is True
    assert _assignment(curriculum, "tabletop_clean_grasp_curriculum_override_alpha") is None
    assert _assignment(curriculum, "tabletop_clean_grasp_curriculum_start_frames") < _assignment(
        curriculum,
        "tabletop_clean_grasp_curriculum_end_frames",
    )
    assert _assignment(
        curriculum,
        "tabletop_clean_grasp_curriculum_start_contact_distance",
    ) > _assignment(strict_contract, "tabletop_clean_grasp_contact_distance")
    assert _assignment(
        curriculum,
        "tabletop_clean_grasp_curriculum_start_max_object_speed",
    ) > _assignment(strict_contract, "tabletop_clean_grasp_max_object_speed")
    assert _assignment(
        curriculum,
        "tabletop_clean_grasp_curriculum_start_max_relative_speed",
    ) > _assignment(strict_contract, "tabletop_clean_grasp_max_relative_speed")
    assert _assignment(
        curriculum,
        "tabletop_clean_grasp_curriculum_start_hold_steps",
    ) < _assignment(strict_contract, "tabletop_clean_grasp_hold_steps")

    expected_bases = {
        "Revo2StaticStableHoverJointTargetCleanCurriculumTeacherEnvCfg": (
            "Revo2StaticStableHoverJointTargetABTeacherEnvCfg"
        ),
        "InspireRH56BFXStaticStableHoverJointTargetCleanCurriculumTeacherEnvCfg": (
            "InspireRH56BFXStaticStableHoverJointTargetABTeacherEnvCfg"
        ),
    }
    for class_name, strict_base in expected_bases.items():
        bases = {ast.unparse(base) for base in classes[class_name].bases}
        assert "_StaticCleanGraspCurriculumContract" in bases
        assert strict_base in bases


def test_clean_contact_curriculum_is_observable_and_eval_defaults_to_strict():
    env_source = ENV_PATH.read_text(encoding="utf-8")
    train_source = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")
    eval_source = EVAL_SCRIPT_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")

    assert "def _tabletop_clean_grasp_curriculum_alpha" in env_source
    assert "def _tabletop_clean_grasp_effective_parameters" in env_source
    assert '"tabletop_clean_grasp_curriculum_alpha"' in env_source
    assert '"tabletop_clean_grasp_effective_contact_distance"' in env_source
    assert '"--clean-grasp-curriculum-alpha"' in train_source
    assert '"--clean-grasp-curriculum-alpha"' in eval_source
    assert "default=1.0" in eval_source
    assert "StaticStableHoverJointTargetCleanCurriculum" in task_source


def test_v14_dense_contact_reward_uses_the_required_finger_bottleneck():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticSynchronousContactRewardContract"]

    assert _assignment(contract, "strict_non_thumb_pair_score_reduction") == "min"
    assert _assignment(
        contract,
        "strict_opposition_touch_reward_requires_non_thumb_pair",
    ) is True

    expected_bases = {
        "Revo2StaticStableHoverJointTargetSynchronousContactTeacherEnvCfg": (
            "Revo2StaticStableHoverJointTargetCleanCurriculumTeacherEnvCfg"
        ),
        "InspireRH56BFXStaticStableHoverJointTargetSynchronousContactTeacherEnvCfg": (
            "InspireRH56BFXStaticStableHoverJointTargetCleanCurriculumTeacherEnvCfg"
        ),
    }
    for class_name, curriculum_base in expected_bases.items():
        bases = {ast.unparse(base) for base in classes[class_name].bases}
        assert "_StaticSynchronousContactRewardContract" in bases
        assert curriculum_base in bases

    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert "def _reduce_strict_non_thumb_pair_scores" in env_source
    assert "return required_scores.amin(dim=-1)" in env_source
    assert "strict_opposition_touch_reward_requires_non_thumb_pair" in env_source
    assert "* self._strict_non_thumb_pair_touch_score" in env_source
    assert "StaticStableHoverJointTargetSynchronousContact" in task_source


def test_v15_smoothly_transitions_from_exploration_to_bottleneck_reward():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticSynchronousContactCurriculumRewardContract"]

    assert _assignment(
        contract,
        "strict_non_thumb_pair_score_reduction",
    ) == "curriculum_mean_to_min"
    assert _assignment(
        contract,
        "strict_opposition_touch_pair_gate_mode",
    ) == "curriculum"
    assert _assignment(
        contract,
        "strict_opposition_touch_reward_requires_non_thumb_pair",
    ) is False

    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert '"curriculum_mean_to_min"' in env_source
    assert "mean_score + alpha * (min_score - mean_score)" in env_source
    assert "1.0\n                    + pair_gate_alpha" in env_source
    assert "StaticStableHoverJointTargetSynchronousCurriculum" in task_source


def test_v16_lift_bridge_is_clean_latched_attachment_preserving_and_action_agnostic():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticPostCleanLiftExplorationContract"]

    assert _assignment(
        contract,
        "tabletop_post_clean_palm_lift_exploration_rew_scale",
    ) > 0.0
    assert _assignment(
        contract,
        "tabletop_post_clean_palm_lift_attachment_scale",
    ) <= 0.012

    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert "tabletop_post_clean_palm_lift_exploration_rew = (" in env_source
    assert "clean_lift_gate" in env_source
    assert "tabletop_relative_palm_lift_progress" in env_source
    assert "tabletop_vertical_palm_pose_score" in env_source
    assert "post_clean_attachment_score" in env_source
    assert "self.actions" not in env_source[
        env_source.index("tabletop_post_clean_palm_lift_exploration_rew = (") :
        env_source.index("lift_schedule_unlocked = (")
    ]
    assert "StaticStableHoverJointTargetPostCleanLift" in task_source


def test_v17_upward_velocity_bridge_is_immediate_bounded_and_attachment_gated():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticPostCleanLiftVelocityContract"]

    assert _assignment(
        contract,
        "tabletop_post_clean_palm_up_velocity_rew_scale",
    ) > 0.0
    assert _assignment(
        contract,
        "tabletop_post_clean_palm_up_velocity_target",
    ) > 0.0

    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert "tabletop_post_clean_palm_up_velocity_rew = (" in env_source
    velocity_block = env_source[
        env_source.index("tabletop_post_clean_palm_up_velocity_rew = (") :
        env_source.index("object_palm_drift_tolerance = max(")
    ]
    assert "post_clean_palm_up_velocity_gate = clean_lift_gate" in env_source
    assert "post_clean_palm_up_velocity_gate" in velocity_block
    assert "torch.relu(self._palm_lin_vel_w[:, 2])" in velocity_block
    assert "post_clean_attachment_score" in velocity_block
    assert "vertical_palm_velocity_pose_score" in velocity_block
    assert "self.actions" not in velocity_block
    assert "StaticStableHoverJointTargetPostCleanLiftVelocity" in task_source


def test_v18_post_clean_grip_retention_uses_required_opposed_contacts_only():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticPostCleanGripRetentionContract"]

    assert _assignment(
        contract,
        "tabletop_post_clean_grip_retention_rew_scale",
    ) > 0.0
    assert _assignment(
        contract,
        "tabletop_post_clean_grip_loss_penalty_scale",
    ) > 0.0
    target = _assignment(contract, "tabletop_post_clean_grip_retention_target")
    assert 0.0 < target <= 1.0
    stationary_floor = _assignment(
        contract,
        "tabletop_post_clean_grip_stationary_reward_floor",
    )
    assert 0.0 < stationary_floor <= 0.10

    expected_bases = {
        "Revo2StaticStableHoverJointTargetPostCleanGripRetentionTeacherEnvCfg": (
            "Revo2StaticStableHoverJointTargetPostCleanLiftVelocityTeacherEnvCfg"
        ),
        "InspireRH56BFXStaticStableHoverJointTargetPostCleanGripRetentionTeacherEnvCfg": (
            "InspireRH56BFXStaticStableHoverJointTargetPostCleanLiftVelocityTeacherEnvCfg"
        ),
    }
    for class_name, velocity_base in expected_bases.items():
        bases = {ast.unparse(base) for base in classes[class_name].bases}
        assert "_StaticPostCleanGripRetentionContract" in bases
        assert velocity_base in bases

    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    start = env_source.index("post_clean_non_thumb_score = torch.topk(")
    end = env_source.index(
        "tabletop_post_clean_strict_grasp_retention_rew = (",
        start,
    )
    retention_block = env_source[start:end]
    assert "self._strict_contact_score[:, 1:]" in retention_block
    assert ".values.amin(dim=-1)" in retention_block
    assert "self._strict_contact_score[:, 0]" in retention_block
    assert "self._strict_opposition_touch_score" in retention_block
    assert "clean_lift_gate" in retention_block
    assert "tabletop_post_clean_palm_lift_exploration_rew" in retention_block
    assert "tabletop_post_clean_palm_up_velocity_rew" in retention_block
    assert "post_clean_grip_motion_score" in retention_block
    assert "tabletop_post_clean_grip_retained = clean_grasp_latched" in retention_block
    assert "self.actions" not in retention_block
    assert "StaticStableHoverJointTargetPostCleanGripRetention" in task_source


def test_v19_post_clean_hand_continuity_penalizes_only_opening_without_a_pose_prior():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticPostCleanActionContinuityContract"]

    assert _assignment(
        contract,
        "tabletop_post_clean_strict_grasp_retention_rew_scale",
    ) > 0.0
    assert _assignment(
        contract,
        "tabletop_post_clean_hand_opening_penalty_scale",
    ) > 0.0

    expected_bases = {
        "Revo2StaticStableHoverJointTargetPostCleanActionContinuityTeacherEnvCfg": (
            "Revo2StaticStableHoverJointTargetPostCleanGripRetentionTeacherEnvCfg"
        ),
        "InspireRH56BFXStaticStableHoverJointTargetPostCleanActionContinuityTeacherEnvCfg": (
            "InspireRH56BFXStaticStableHoverJointTargetPostCleanGripRetentionTeacherEnvCfg"
        ),
    }
    for class_name, retention_base in expected_bases.items():
        bases = {ast.unparse(base) for base in classes[class_name].bases}
        assert "_StaticPostCleanActionContinuityContract" in bases
        assert retention_base in bases

    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    start = env_source.index("post_clean_hand_opening_delta = torch.relu(")
    end = env_source.index("policy_action_penalty =", start)
    continuity_block = env_source[start:end]
    assert "self.prev_actions[:, hand_action_start:]" in continuity_block
    assert "self.actions[:, hand_action_start:]" in continuity_block
    assert "clean_lift_gate" in continuity_block
    assert "joint_targets" not in continuity_block
    assert "close_target" not in continuity_block
    assert "StaticStableHoverJointTargetPostCleanActionContinuity" in task_source


def test_v20_smooths_both_lift_phase_observation_channels_without_changing_dim():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticSmoothLiftPhaseObservationContract"]

    assert _assignment(contract, "tabletop_clean_lift_phase_ramp_enabled") is True
    assert _assignment(contract, "tabletop_clean_lift_phase_ramp_steps") >= 20

    expected_bases = {
        "Revo2StaticStableHoverJointTargetSmoothLiftPhaseTeacherEnvCfg": (
            "Revo2StaticStableHoverJointTargetPostCleanActionContinuityTeacherEnvCfg"
        ),
        "InspireRH56BFXStaticStableHoverJointTargetSmoothLiftPhaseTeacherEnvCfg": (
            "InspireRH56BFXStaticStableHoverJointTargetPostCleanActionContinuityTeacherEnvCfg"
        ),
    }
    for class_name, continuity_base in expected_bases.items():
        bases = {ast.unparse(base) for base in classes[class_name].bases}
        assert "_StaticSmoothLiftPhaseObservationContract" in bases
        assert continuity_base in bases

    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert "self._tabletop_clean_lift_phase_progress" in env_source
    assert "+ 1.0 / float(phase_ramp_steps)" in env_source
    assert "lift_phase_obs = self._tabletop_clean_lift_phase_progress" in env_source
    assert "clean_lift_phase_obs = self._tabletop_clean_lift_phase_progress" in env_source
    assert "StaticStableHoverJointTargetSmoothLiftPhase" in task_source


def test_v21_requires_sustained_force_before_clean_lift_for_both_hands():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticForceBackedCleanLiftContract"]

    assert _assignment(contract, "tabletop_clean_grasp_requires_force_grasp") is True
    assert _assignment(contract, "tabletop_clean_grasp_min_force_streak_steps") >= 4
    assert _assignment(contract, "tabletop_success_requires_force_grasp") is True
    assert _assignment(contract, "tabletop_lift_gate_requires_force_grasp") is True
    assert _assignment(contract, "tabletop_lift_rewards_require_force_grasp") is True
    assert _assignment(contract, "tabletop_no_lift_uses_force_grasp_gate") is True
    assert _assignment(contract, "tabletop_arm_lift_progress_baseline_mode") == "first_force_grasp"
    assert _assignment(contract, "tabletop_force_grasp_rew_scale") > 0.0
    assert _assignment(contract, "tabletop_force_grasp_streak_rew_scale") > 0.0
    assert _assignment(contract, "tabletop_force_grasp_loss_penalty_scale") > 0.0

    expected_bases = {
        "Revo2StaticStableHoverJointTargetForceBackedCleanLiftTeacherEnvCfg": (
            "Revo2StaticStableHoverJointTargetSmoothLiftPhaseTeacherEnvCfg"
        ),
        "InspireRH56BFXStaticStableHoverJointTargetForceBackedCleanLiftTeacherEnvCfg": (
            "InspireRH56BFXStaticStableHoverJointTargetSmoothLiftPhaseTeacherEnvCfg"
        ),
    }
    for class_name, smooth_phase_base in expected_bases.items():
        bases = {ast.unparse(base) for base in classes[class_name].bases}
        assert "_StaticForceBackedCleanLiftContract" in bases
        assert smooth_phase_base in bases

    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert "self._force_grasp_streak >= min_force_streak" in env_source
    assert task_source.count("StaticStableHoverJointTargetForceBackedCleanLift") >= 4


def test_v22_uses_the_same_force_backed_sphere_pinch_for_both_hands():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticForceBackedPinchLiftContract"]

    assert _assignment(contract, "tabletop_clean_grasp_min_non_thumb_contacts") == 1
    assert _assignment(contract, "object_force_grasp_min_non_thumb_contacts") == 1
    assert _assignment(contract, "tabletop_clean_grasp_min_force_streak_steps") >= 3
    assert _assignment(contract, "tabletop_arm_lift_progress_baseline_grasp_streak") == 1
    assert _assignment(contract, "tabletop_lift_gate_requires_current_strict_grasp") is False
    assert _assignment(contract, "tabletop_lift_rewards_require_current_strict_grasp") is False

    strict_contract = classes["_StaticStrictFromScratchControlABContract"]
    assert _assignment(strict_contract, "strict_success_min_finger_contacts") >= 3
    assert _assignment(strict_contract, "strict_success_min_non_thumb_contacts") >= 2

    expected_bases = {
        "Revo2StaticStableHoverJointTargetForceBackedPinchLiftTeacherEnvCfg": (
            "Revo2StaticStableHoverJointTargetForceBackedCleanLiftTeacherEnvCfg"
        ),
        "InspireRH56BFXStaticStableHoverJointTargetForceBackedPinchLiftTeacherEnvCfg": (
            "InspireRH56BFXStaticStableHoverJointTargetForceBackedCleanLiftTeacherEnvCfg"
        ),
    }
    for class_name, force_base in expected_bases.items():
        bases = {ast.unparse(base) for base in classes[class_name].bases}
        assert "_StaticForceBackedPinchLiftContract" in bases
        assert force_base in bases

    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert task_source.count("StaticStableHoverJointTargetForceBackedPinchLift") >= 4


def test_v23_pays_only_force_retained_object_coupled_lift():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticForceRetainedObjectLiftContract"]

    assert _assignment(contract, "tabletop_force_lift_curriculum_override_alpha") == 1.0
    assert _assignment(contract, "tabletop_clean_grasp_latch_bonus") < 200000.0
    assert _assignment(contract, "tabletop_acquisition_reward_post_clean_latch_floor") >= 0.25
    assert _assignment(contract, "tabletop_force_grasp_loss_penalty_scale") >= 100000.0
    for field in (
        "tabletop_post_clean_palm_lift_exploration_rew_scale",
        "tabletop_post_clean_palm_up_velocity_rew_scale",
        "tabletop_vertical_palm_velocity_rew_scale",
        "tabletop_grasped_palm_lift_rew_scale",
    ):
        assert _assignment(contract, field) == 0.0

    expected_bases = {
        "Revo2StaticStableHoverJointTargetForceRetainedObjectLiftTeacherEnvCfg": (
            "Revo2StaticStableHoverJointTargetForceBackedPinchLiftTeacherEnvCfg"
        ),
        "InspireRH56BFXStaticStableHoverJointTargetForceRetainedObjectLiftTeacherEnvCfg": (
            "InspireRH56BFXStaticStableHoverJointTargetForceBackedPinchLiftTeacherEnvCfg"
        ),
    }
    for class_name, pinch_base in expected_bases.items():
        bases = {ast.unparse(base) for base in classes[class_name].bases}
        assert "_StaticForceRetainedObjectLiftContract" in bases
        assert pinch_base in bases

    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert task_source.count("StaticStableHoverJointTargetForceRetainedObjectLift") >= 4


def test_v24_requires_a_sustained_force_hold_before_lift_for_both_hands():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticSustainedForceHoldBeforeLiftContract"]

    assert _assignment(contract, "tabletop_clean_grasp_min_force_streak_steps") >= 12
    assert _assignment(contract, "tabletop_force_grasp_streak_target") >= 12
    assert _assignment(contract, "tabletop_force_grasp_loss_penalty_scale") < 100000.0

    expected_bases = {
        "Revo2StaticStableHoverJointTargetSustainedForceLiftTeacherEnvCfg": (
            "Revo2StaticStableHoverJointTargetForceRetainedObjectLiftTeacherEnvCfg"
        ),
        "InspireRH56BFXStaticStableHoverJointTargetSustainedForceLiftTeacherEnvCfg": (
            "InspireRH56BFXStaticStableHoverJointTargetForceRetainedObjectLiftTeacherEnvCfg"
        ),
    }
    for class_name, retained_lift_base in expected_bases.items():
        bases = {ast.unparse(base) for base in classes[class_name].bases}
        assert "_StaticSustainedForceHoldBeforeLiftContract" in bases
        assert retained_lift_base in bases

    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert task_source.count("StaticStableHoverJointTargetSustainedForceLift") >= 4


def test_v25_micro_lift_is_force_gated_and_shared_by_both_hands():
    classes = _classes(CFG_PATH)
    contract = classes["_StaticForceCoupledMicroLiftContract"]

    assert _assignment(contract, "tabletop_post_clean_palm_up_velocity_rew_scale") > 0.0
    assert _assignment(contract, "tabletop_palm_object_up_vel_rew_scale") > 0.0
    assert _assignment(contract, "tabletop_object_carry_lift_rew_scale") > 0.0
    assert _assignment(contract, "tabletop_no_lift_after_grasp_grace_steps") <= 12
    assert _assignment(contract, "tabletop_no_lift_after_grasp_penalty_scale") > 0.0
    assert _assignment(contract, "tabletop_force_grasp_rew_scale") < 8000.0

    expected_bases = {
        "Revo2StaticStableHoverJointTargetForceCoupledMicroLiftTeacherEnvCfg": (
            "Revo2StaticStableHoverJointTargetSustainedForceLiftTeacherEnvCfg"
        ),
        "InspireRH56BFXStaticStableHoverJointTargetForceCoupledMicroLiftTeacherEnvCfg": (
            "InspireRH56BFXStaticStableHoverJointTargetSustainedForceLiftTeacherEnvCfg"
        ),
    }
    for class_name, force_hold_base in expected_bases.items():
        bases = {ast.unparse(base) for base in classes[class_name].bases}
        assert "_StaticForceCoupledMicroLiftContract" in bases
        assert force_hold_base in bases

    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    assert "post_clean_palm_up_velocity_gate * force_lift_gate" in env_source
    assert task_source.count("StaticStableHoverJointTargetForceCoupledMicroLift") >= 4
