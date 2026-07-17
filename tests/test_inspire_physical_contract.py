import ast
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = (
    ROOT
    / "source"
    / "simtoolreal_lab"
    / "simtoolreal_lab"
    / "tasks"
    / "dynamic_dexterous_grasp"
    / "dynamic_dexterous_grasp_env_cfg.py"
)
BASE_CFG_PATH = (
    ROOT
    / "source"
    / "simtoolreal_lab"
    / "simtoolreal_lab"
    / "tasks"
    / "revo2_static_grasp"
    / "revo2_static_grasp_env_cfg.py"
)
BASE_ENV_PATH = BASE_CFG_PATH.with_name("revo2_static_grasp_env.py")
DYNAMIC_ENV_PATH = CFG_PATH.with_name("dynamic_dexterous_grasp_env.py")
TASK_INIT_PATH = CFG_PATH.with_name("__init__.py")
EVAL_SCRIPT_PATH = ROOT / "scripts" / "evaluate_rl_games.py"
MIMIC_URDF_PATH = (
    ROOT
    / "assets"
    / "embodiments"
    / "franka-inspire-rh56bfx-mimic"
    / "franka_inspire_rh56bfx_mimic.urdf"
)


def _assignment_source(class_name: str, field_name: str) -> str:
    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    for node in class_node.body:
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.targets[0] if isinstance(node, ast.Assign) else node.target, ast.Name)
            and (node.targets[0] if isinstance(node, ast.Assign) else node.target).id == field_name
        ):
            return ast.unparse(node.value)
    raise AssertionError(f"{field_name!r} is not assigned in {class_name}")


def test_mesh_only_audit_disables_phantom_tip_collisions_and_uses_mesh_force_links():
    class_name = "InspirePhysicalAuditMeshOnlyTeacherEnvCfg"

    assert _assignment_source(class_name, "robot_collision_disabled_body_names") == (
        "INSPIRE_COLLISION_TIP_BODY_NAMES"
    )
    assert _assignment_source(class_name, "touch_body_names") == "INSPIRE_MESH_CONTACT_BODY_NAMES"
    assert _assignment_source(class_name, "object_contact_force_diagnostics_enabled") == "True"

    source = CFG_PATH.read_text(encoding="utf-8")
    for link_name in (
        "thumb_distal",
        "index_intermediate",
        "middle_intermediate",
        "ring_intermediate",
        "pinky_intermediate",
    ):
        assert f'"{link_name}"' in source


def test_phantom_tip_baseline_only_enables_force_diagnostics():
    class_name = "InspirePhysicalAuditPhantomTipsTeacherEnvCfg"
    assert _assignment_source(class_name, "object_contact_force_diagnostics_enabled") == "True"


def test_anydex_arm_residual_preserves_hand_closure_and_post_grasp_lift():
    class_name = "InspireRH56BFXAnyDexRing60ArmResidualTeacherEnvCfg"

    assert _assignment_source(class_name, "scripted_action_prior_active_arm_residual_scale") == "0.05"
    assert _assignment_source(class_name, "scripted_action_prior_active_hand_residual_scale") == "0.0"
    assert _assignment_source(class_name, "scripted_action_prior_post_grasp_arm_residual_scale") == "0.0"

    source = BASE_ENV_PATH.read_text(encoding="utf-8")
    assert 'getattr(self, "_tabletop_arm_lift_baseline_latched", None)' in source


def test_primitive3_load_bearing_stage_is_direct_rl_and_shape_normalized():
    class_name = "InspireRH56BFXFaithfulPrimitive3LoadBearingTeacherEnvCfg"

    assert _assignment_source(class_name, "tabletop_underwrap_rew_scale") == "16000.0"
    assert _assignment_source(class_name, "tabletop_underwrap_below_center_fraction") == "0.05"
    assert _assignment_source(class_name, "tabletop_underwrap_progress_weight") == "0.7"
    assert _assignment_source(class_name, "tabletop_underwrap_pair_weight") == "0.3"
    assert _assignment_source(class_name, "tabletop_no_lift_after_grasp_grace_steps") == "18"

    assert "active_radius = getattr(self, \"_active_object_radius_tensor\", None)" in DYNAMIC_ENV_PATH.read_text(
        encoding="utf-8"
    )
    parent_name = "InspireRH56BFXFaithfulSphere60ForceLiftTeacherEnvCfg"
    for field_name in (
        "scripted_action_prior_enabled",
        "scripted_tabletop_pregrasp_prior_enabled",
        "scripted_tabletop_relative_lift_target_prior_enabled",
        "scripted_tabletop_lift_target_prior_enabled",
        "scripted_tabletop_hand_grasp_memory_prior_enabled",
    ):
        assert _assignment_source(parent_name, field_name) == "False"


def test_legacy_actuation_audit_keeps_the_same_geometry_contract():
    class_name = "InspirePhysicalAuditMeshOnlyLegacyActuationTeacherEnvCfg"
    assert _assignment_source(class_name, "robot_cfg") == (
        "_inspire_z180_legacy_robot_cfg(INSPIRE_V340_KNOWN_GOOD_ARM_POS)"
    )


def test_collision_disable_is_applied_after_every_scene_clone():
    base_source = BASE_ENV_PATH.read_text(encoding="utf-8")
    dynamic_source = DYNAMIC_ENV_PATH.read_text(encoding="utf-8")

    assert "def _apply_robot_collision_disables(self)" in base_source
    assert "GetCollisionEnabledAttr().Set(False)" in base_source
    assert base_source.count("self._apply_robot_collision_disables()") == 1
    assert dynamic_source.count("self._apply_robot_collision_disables()") == 2


def test_mimic_properties_are_applied_after_every_scene_clone():
    base_source = BASE_ENV_PATH.read_text(encoding="utf-8")
    dynamic_source = DYNAMIC_ENV_PATH.read_text(encoding="utf-8")

    assert "def _apply_robot_mimic_joint_properties(self)" in base_source
    assert 'name.endswith(":naturalFrequency")' in base_source
    assert 'name.endswith(":dampingRatio")' in base_source
    assert base_source.count("self._apply_robot_mimic_joint_properties()") == 1
    assert dynamic_source.count("self._apply_robot_mimic_joint_properties()") == 2


def test_base_config_keeps_collision_disables_opt_in():
    tree = ast.parse(BASE_CFG_PATH.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Revo2StaticGraspEnvCfg"
    )
    field = next(
        node
        for node in class_node.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "robot_collision_disabled_body_names"
    )
    assert ast.literal_eval(field.value) == ()


def test_extra_self_collision_filters_are_opt_in_and_applied():
    tree = ast.parse(BASE_CFG_PATH.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Revo2StaticGraspEnvCfg"
    )
    field = next(
        node
        for node in class_node.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "robot_extra_self_collision_filter_pairs"
    )
    assert ast.literal_eval(field.value) == ()

    source = BASE_ENV_PATH.read_text(encoding="utf-8")
    assert 'getattr(self.cfg, "robot_extra_self_collision_filter_pairs", ())' in source
    assert "FilteredPairsAPI.Apply(prim_a)" in source
    assert "AddTarget(prim_b.GetPath())" in source
    assert "_collision_filter_prims_for_link" not in source


def test_rh56bfx_asset_restores_six_source_mimic_constraints():
    root = ET.parse(MIMIC_URDF_PATH).getroot()
    mimic_specs = {
        joint.attrib["name"]: (
            joint.find("mimic").attrib["joint"],
            float(joint.find("mimic").attrib["multiplier"]),
            float(joint.find("mimic").attrib["offset"]),
        )
        for joint in root.findall("joint")
        if joint.find("mimic") is not None
    }

    assert mimic_specs == {
        "thumb_intermediate_joint": ("thumb_proximal_pitch_joint", 1.334, 0.0),
        "thumb_distal_joint": ("thumb_proximal_pitch_joint", 0.667, 0.0),
        "index_intermediate_joint": ("index_proximal_joint", 1.06399, -0.04545),
        "middle_intermediate_joint": ("middle_proximal_joint", 1.06399, -0.04545),
        "ring_intermediate_joint": ("ring_proximal_joint", 1.06399, -0.04545),
        "pinky_intermediate_joint": ("pinky_proximal_joint", 1.06399, -0.04545),
    }


def test_rh56bfx_tip_frames_have_no_phantom_collision_geometry():
    root = ET.parse(MIMIC_URDF_PATH).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    for name in ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"):
        assert links[name].find("collision") is None


def test_rh56bfx_combined_asset_keeps_official_franka_arm_inertias():
    root = ET.parse(MIMIC_URDF_PATH).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    expected_masses = (
        2.8142028896,
        2.3599995791,
        2.379518833,
        2.6498823337,
        2.6948018744,
        2.9812816864,
        1.1285806309,
        0.4052912465,
    )
    for link_index, expected_mass in enumerate(expected_masses):
        inertial = links[f"panda_link{link_index}"].find("inertial")
        assert inertial is not None
        assert float(inertial.find("mass").attrib["value"]) == expected_mass
        assert inertial.find("inertia") is not None


def test_mimic_audit_controls_only_the_six_active_joints():
    class_name = "InspirePhysicalAuditMimicTeacherEnvCfg"
    assert _assignment_source(class_name, "sim_hand_joint_names") == "INSPIRE_ACTIVE_HAND_JOINT_NAMES"
    assert _assignment_source(class_name, "inspire_semantic_close_targets") == (
        "INSPIRE_RH56BFX_MIMIC_CLOSE_TARGETS"
    )
    robot_cfg_source = _assignment_source(class_name, "robot_cfg")
    assert "asset_path=INSPIRE_RH56BFX_MIMIC_URDF" in robot_cfg_source
    assert "actuate_mimic_followers=False" in robot_cfg_source
    assert "preserve_mimic_constraints=True" in robot_cfg_source
    assert _assignment_source(class_name, "robot_mimic_natural_frequency") == "0.0"
    assert _assignment_source(class_name, "robot_mimic_damping_ratio") == "0.0"
    offsets = ast.literal_eval(_assignment_source(class_name, "robot_mimic_offset_overrides_deg"))
    assert dict(offsets) == {
        "index_intermediate_joint": 2.604352333,
        "middle_intermediate_joint": 2.604352333,
        "ring_intermediate_joint": 2.604352333,
        "pinky_intermediate_joint": 2.604352333,
    }


def test_inspire_same_chain_collision_filters_cover_palm_descendants():
    source = BASE_ENV_PATH.read_text(encoding="utf-8")
    for pair in (
        '("hand_base_link", "thumb_distal")',
        '("hand_base_link", "index_intermediate")',
        '("hand_base_link", "middle_intermediate")',
        '("hand_base_link", "ring_intermediate")',
        '("hand_base_link", "pinky_intermediate")',
    ):
        assert pair in source


def test_faithful_teacher_tasks_share_the_six_motor_physical_contract():
    for class_name in (
        "InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg",
        "InspireRH56BFXFaithfulUnifiedFallingTeacherEnvCfg",
    ):
        assert _assignment_source(class_name, "sim_hand_joint_names") == (
            "INSPIRE_ACTIVE_HAND_JOINT_NAMES"
        )
        assert _assignment_source(class_name, "inspire_semantic_close_targets") == (
            "INSPIRE_ANYDEX_SPHERE_ACTIVE_CLOSE_TARGETS"
        )
        assert _assignment_source(class_name, "touch_body_names") == (
            "INSPIRE_MESH_CONTACT_BODY_NAMES"
        )
        assert _assignment_source(class_name, "object_contact_force_diagnostics_enabled") == "True"
        robot_cfg_source = _assignment_source(class_name, "robot_cfg")
        assert "_inspire_rh56bfx_mimic_robot_cfg" in robot_cfg_source


def test_faithful_teacher_success_requires_real_force_contact():
    assert _assignment_source(
        "InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg",
        "tabletop_success_requires_force_grasp",
    ) == "True"
    assert _assignment_source(
        "InspireRH56BFXFaithfulUnifiedFallingTeacherEnvCfg",
        "falling_success_requires_force_grasp",
    ) == "True"
    assert 'getattr(self.cfg, "falling_success_requires_force_grasp", False)' in (
        DYNAMIC_ENV_PATH.read_text(encoding="utf-8")
    )


def test_faithful_rolling_uses_rh56bfx_vertical_carry_direction():
    class_name = "InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg"
    for field_name in (
        "lift_arm_delta",
        "scripted_tabletop_lift_target_arm_delta",
        "scripted_tabletop_relative_lift_target_arm_delta",
    ):
        assert _assignment_source(class_name, field_name) == (
            "INSPIRE_RH56BFX_CARTESIAN_CARRY_ARM_DELTA"
        )


def test_faithful_unified_rolling_registers_the_three_stage_curriculum():
    cfg_tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
    expected_bases = {
        "InspireRH56BFXFaithfulUnifiedRollingStage1TeacherEnvCfg": (
            "InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg",
        ),
        "InspireRH56BFXFaithfulUnifiedRollingStage2HoldTeacherEnvCfg": (
            "_UnifiedRollingGraspHoldStage2Contract",
            "InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg",
        ),
        "InspireRH56BFXFaithfulUnifiedRollingStage3TeacherEnvCfg": (
            "_UnifiedRollingLiftHoldStage3Contract",
            "InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg",
        ),
    }
    for class_name, expected in expected_bases.items():
        class_node = next(
            node
            for node in cfg_tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        assert tuple(ast.unparse(base) for base in class_node.bases) == expected

    assert float(
        _assignment_source(
            "InspireRH56BFXFaithfulUnifiedRollingStage1TeacherEnvCfg",
            "strict_opposition_touch_rew_scale",
        )
    ) == 1200.0
    assert float(
        _assignment_source(
            "InspireRH56BFXFaithfulUnifiedRollingStage3TeacherEnvCfg",
            "tabletop_lift_hand_target_close_fraction",
        )
    ) == 0.15
    assert float(
        _assignment_source(
            "InspireRH56BFXFaithfulUnifiedRollingStage3TeacherEnvCfg",
            "tabletop_strict_hold_rew_scale",
        )
    ) == 12000.0
    assert float(
        _assignment_source(
            "InspireRH56BFXFaithfulUnifiedRollingStage3TeacherEnvCfg",
            "tabletop_underwrap_rew_scale",
        )
    ) == 4000.0

    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    for suffix in ("Stage1", "Stage2Hold", "Stage3"):
        assert (
            f'SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRolling{suffix}'
            '-Teacher-Direct-v0'
        ) in task_source


def test_faithful_phase_obs_curriculum_exposes_the_grasp_to_lift_state():
    source = DYNAMIC_ENV_PATH.read_text(encoding="utf-8")
    for signal in (
        "self._strict_true_grasp.float()",
        "self._tabletop_strict_true_grasp_streak.float()",
        "self._tabletop_arm_lift_baseline_latched.float()",
        "arm_lift_progress",
    ):
        assert signal in source

    for class_name in (
        "InspireRH56BFXFaithfulUnifiedRollingStage1PhaseObsTeacherEnvCfg",
        "InspireRH56BFXFaithfulUnifiedRollingStage2HoldPhaseObsTeacherEnvCfg",
        "InspireRH56BFXFaithfulUnifiedRollingStage3PhaseObsTeacherEnvCfg",
    ):
        assert _assignment_source(class_name, "observation_space") == "90"
        assert _assignment_source(class_name, "tabletop_privileged_phase_obs_enabled") == "True"

    stage3_class = "InspireRH56BFXFaithfulUnifiedRollingStage3PhaseObsTeacherEnvCfg"
    for field_name in (
        "lift_arm_delta",
        "scripted_tabletop_lift_target_arm_delta",
        "scripted_tabletop_relative_lift_target_arm_delta",
    ):
        assert _assignment_source(stage3_class, field_name) == (
            "INSPIRE_RH56BFX_SHORT_CARRY_ARM_DELTA"
        )

    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    for suffix in ("Stage1PhaseObs", "Stage2HoldPhaseObs", "Stage3PhaseObs"):
        assert (
            f'SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRolling{suffix}'
            '-Teacher-Direct-v0'
        ) in task_source

    target_class = "InspireRH56BFXFaithfulUnifiedRollingStage3TargetPhaseObsTeacherEnvCfg"
    assert float(_assignment_source(target_class, "tabletop_lift_action_prior_rew_scale")) == 0.0
    assert float(_assignment_source(target_class, "tabletop_lift_target_rew_scale")) == 60000.0
    assert float(_assignment_source(target_class, "tabletop_lift_target_error_scale")) == 0.25
    assert _assignment_source(target_class, "tabletop_lift_target_requires_current_grasp") == "False"
    assert ast.literal_eval(_assignment_source(target_class, "tabletop_lift_target_mode")) == "cartesian_ik"
    assert float(_assignment_source(target_class, "tabletop_lift_cartesian_target_height")) == 0.05
    assert float(
        _assignment_source(target_class, "tabletop_lift_cartesian_target_max_joint_delta")
    ) == 0.01
    assert _assignment_source(target_class, "observation_space") == "97"
    assert _assignment_source(
        target_class, "tabletop_privileged_lift_target_obs_enabled"
    ) == "True"
    assert float(_assignment_source(target_class, "tabletop_strict_hold_rew_scale")) == 28000.0
    assert (
        "SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage3TargetPhaseObs"
        "-Teacher-Direct-v0"
    ) in task_source
    assert "def _compute_tabletop_cartesian_lift_target_action(" in source
    assert "self._tabletop_lift_palm_baseline_quat_w" in source

    prior_class = (
        "InspireRH56BFXFaithfulUnifiedRollingStage3CartesianPriorTargetPhaseObsTeacherEnvCfg"
    )
    assert _assignment_source(prior_class, "scripted_action_prior_enabled") == "True"
    assert _assignment_source(
        prior_class, "scripted_action_prior_zero_passthrough_enabled"
    ) == "True"
    assert float(_assignment_source(prior_class, "scripted_action_prior_active_residual_scale")) == 0.0
    assert _assignment_source(
        prior_class, "scripted_action_prior_active_arm_residual_scale"
    ) == "None"
    assert float(
        _assignment_source(prior_class, "scripted_action_prior_active_hand_residual_scale")
    ) == 1.0
    assert _assignment_source(
        prior_class, "scripted_tabletop_cartesian_lift_target_prior_enabled"
    ) == "True"
    assert (
        "SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage3"
        "CartesianPriorTargetPhaseObs-Teacher-Residual-v0"
    ) in task_source
    assert '"scripted_tabletop_cartesian_lift_target_prior_enabled"' in source
    assert "self._compute_tabletop_cartesian_lift_target_action()" in source


def test_faithful_sphere60_bootstrap_uses_position_only_calibrated_prior():
    class_name = "InspireRH56BFXFaithfulSphere60RollingCurriculumTeacherEnvCfg"
    assert _assignment_source(class_name, "scripted_action_prior_hand_action_vector") == (
        "INSPIRE_RH56BFX_SPHERE_60MM_CALIBRATED_ACTION"
    )
    assert _assignment_source(
        class_name, "scripted_tabletop_hand_grasp_memory_action_vector"
    ) == "INSPIRE_RH56BFX_SPHERE_60MM_CALIBRATED_ACTION"
    assert _assignment_source(class_name, "tabletop_object_asset_specs") == (
        "(TABLETOP_INSPIRE_SPHERE_60MM_SPEC,)"
    )
    assert _assignment_source(class_name, "dynamic_tabletop_start_speed_range") == "(0.0, 0.0)"
    assert _assignment_source(class_name, "dynamic_tabletop_initial_speed_range") == (
        "UNIFIED_ROLLING_TARGET_SPEED_RANGE"
    )


def test_initial_hand_prior_accepts_per_motor_vector():
    source = BASE_ENV_PATH.read_text(encoding="utf-8")
    assert 'getattr(self.cfg, "scripted_action_prior_hand_action_vector", None)' in source
    assert "scripted_action_prior_hand_action_vector must contain" in source


def test_force_grasp_defaults_to_thumb_plus_two_non_thumb_contacts():
    assert _assignment_source(
        "Revo2DynamicDexterousTeacherEnvCfg",
        "object_force_grasp_min_non_thumb_contacts",
    ) == "2"
    source = DYNAMIC_ENV_PATH.read_text(encoding="utf-8")
    assert 'getattr(self.cfg, "object_force_grasp_min_non_thumb_contacts", 2)' in source


def test_anydex_ring60_bootstrap_is_scoped_to_verified_two_finger_contract():
    class_name = "InspireRH56BFXAnyDexRing60BootstrapTeacherEnvCfg"
    assert _assignment_source(class_name, "inspire_semantic_close_targets") == (
        "INSPIRE_ANYDEX_RING60_ACTIVE_CLOSE_TARGETS"
    )
    assert _assignment_source(class_name, "scripted_tabletop_pregrasp_arm_pos") == (
        "INSPIRE_ANYDEX_RING60_GRASP_ARM_POS"
    )
    assert _assignment_source(
        class_name, "scripted_tabletop_pregrasp_hold_after_track"
    ) == "True"
    assert _assignment_source(class_name, "scripted_tabletop_lift_target_arm_pos") == (
        "INSPIRE_ANYDEX_RING60_LIFT_ARM_POS"
    )
    assert _assignment_source(class_name, "min_finger_contacts") == "2"
    assert _assignment_source(class_name, "min_non_thumb_contacts") == "1"
    assert _assignment_source(class_name, "strict_success_min_finger_contacts") == "2"
    assert _assignment_source(class_name, "strict_success_min_non_thumb_contacts") == "1"
    assert _assignment_source(
        class_name, "object_force_grasp_min_non_thumb_contacts"
    ) == "1"
    assert _assignment_source(class_name, "tabletop_success_requires_force_grasp") == "True"


def test_faithful_force_curriculum_is_direct_rl_and_requires_load_bearing_contact():
    for class_name in (
        "InspireRH56BFXFaithfulSphere60ForceHoldTeacherEnvCfg",
        "InspireRH56BFXFaithfulSphere60ForceLiftTeacherEnvCfg",
    ):
        assert _assignment_source(class_name, "object_force_grasp_min_non_thumb_contacts") == "1"
        assert _assignment_source(class_name, "tabletop_success_requires_force_grasp") == "True"
        assert ast.literal_eval(
            _assignment_source(class_name, "tabletop_arm_lift_progress_baseline_mode")
        ) == "first_force_grasp"
        assert _assignment_source(class_name, "scripted_action_prior_enabled") == "False"
        assert _assignment_source(class_name, "scripted_tabletop_pregrasp_prior_enabled") == "False"
        assert _assignment_source(
            class_name, "scripted_tabletop_relative_lift_target_prior_enabled"
        ) == "False"
        assert _assignment_source(class_name, "scripted_tabletop_lift_target_prior_enabled") == "False"
        assert _assignment_source(
            class_name, "scripted_tabletop_hand_grasp_memory_prior_enabled"
        ) == "False"

    assert _assignment_source(
        "InspireRH56BFXFaithfulSphere60ForceLiftTeacherEnvCfg",
        "tabletop_lift_rewards_require_force_grasp",
    ) == "True"


def test_faithful_lift_commit_stage_rebalances_contact_toward_object_carry():
    class_name = "InspireRH56BFXFaithfulSphere60LiftCommitTeacherEnvCfg"
    force_contact_reward = sum(
        float(_assignment_source(class_name, field))
        for field in (
            "tabletop_force_grasp_rew_scale",
            "tabletop_force_grasp_streak_rew_scale",
            "tabletop_force_stable_grasp_rew_scale",
        )
    )
    object_carry_reward = sum(
        float(_assignment_source(class_name, field))
        for field in (
            "tabletop_object_up_vel_rew_scale",
            "tabletop_object_carry_lift_rew_scale",
            "lifted_true_grasp_rew_scale",
        )
    )
    assert object_carry_reward > 5.0 * force_contact_reward
    assert float(
        _assignment_source(class_name, "tabletop_no_lift_after_grasp_penalty_scale")
    ) > force_contact_reward


def test_primitive3_bridge_changes_shape_without_position_or_controller_changes():
    class_name = "InspireRH56BFXFaithfulPrimitive3LiftCommitTeacherEnvCfg"
    assert _assignment_source(class_name, "tabletop_object_asset_specs") == (
        "TABLETOP_INSPIRE_PRIMITIVE3_SPECS"
    )
    assert _assignment_source(class_name, "tabletop_asset_curriculum_start_count") == "1"
    assert _assignment_source(class_name, "tabletop_asset_curriculum_mode") == "'dynamic_speed'"
    assert _assignment_source(class_name, "reset_object_pos_noise") == "(0.0, 0.0, 0.0)"


def test_primitive3_sphere_balanced_sampler_keeps_all_three_assets():
    class_name = "InspireRH56BFXFaithfulPrimitive3SphereBalancedLiftCommitTeacherEnvCfg"
    weights = ast.literal_eval(_assignment_source(class_name, "tabletop_asset_sampling_weights"))
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1.0e-9
    assert weights[0] > weights[1] > weights[2]


def test_primitive3_cartesian_carry_rejects_wrist_flip_success_shortcut():
    class_name = "InspireRH56BFXFaithfulPrimitive3CartesianCarryTeacherEnvCfg"

    for field_name in (
        "lift_arm_delta",
        "scripted_tabletop_lift_target_arm_delta",
        "scripted_tabletop_relative_lift_target_arm_delta",
    ):
        assert _assignment_source(class_name, field_name) == (
            "INSPIRE_RH56BFX_CARTESIAN_CARRY_ARM_DELTA"
        )

    assert _assignment_source(
        class_name, "tabletop_success_requires_relative_palm_lift"
    ) == "True"
    assert float(
        _assignment_source(class_name, "tabletop_success_min_relative_palm_lift")
    ) == 0.045
    assert float(
        _assignment_source(class_name, "tabletop_success_max_object_palm_drift")
    ) == 0.045
    assert float(_assignment_source(class_name, "tabletop_palm_object_carry_rew_scale")) > 0.0
    assert float(
        _assignment_source(class_name, "tabletop_object_palm_drift_penalty_scale")
    ) > 0.0

    # The calibrated delta only defines a dense reward direction. The parent
    # contract keeps all scripted reach/close/lift controllers disabled.
    direct_parent = "InspireRH56BFXFaithfulSphere60ForceLiftTeacherEnvCfg"
    for field_name in (
        "scripted_action_prior_enabled",
        "scripted_tabletop_pregrasp_prior_enabled",
        "scripted_tabletop_relative_lift_target_prior_enabled",
        "scripted_tabletop_lift_target_prior_enabled",
        "scripted_tabletop_hand_grasp_memory_prior_enabled",
    ):
        assert _assignment_source(direct_parent, field_name) == "False"


def test_cartesian_carry_metrics_latch_palm_object_pose_and_gate_success():
    source = DYNAMIC_ENV_PATH.read_text(encoding="utf-8")

    assert "self._tabletop_lift_palm_baseline_pos_w[baseline_latch_ids]" in source
    assert "self._tabletop_lift_object_palm_rel_w[baseline_latch_ids]" in source
    assert "tabletop_relative_palm_lift / relative_palm_lift_target" in source
    assert "tabletop_object_palm_drift <= max_success_object_palm_drift" in source
    assert 'self.extras["tabletop_relative_palm_lift_env"]' in source
    assert 'self.extras["tabletop_object_palm_drift_env"]' in source


def test_palm_coupled_carry_removes_object_only_lift_shortcuts():
    class_name = "InspireRH56BFXFaithfulPrimitive3PalmCoupledCarryTeacherEnvCfg"

    for field_name in (
        "lift_progress_rew_scale",
        "quality_lift_progress_rew_scale",
        "lifted_true_grasp_rew_scale",
        "tabletop_grasped_palm_lift_rew_scale",
        "tabletop_hover_height_progress_rew_scale",
        "tabletop_object_up_vel_rew_scale",
    ):
        assert float(_assignment_source(class_name, field_name)) == 0.0
    assert float(
        _assignment_source(class_name, "tabletop_palm_object_up_vel_rew_scale")
    ) > 0.0
    assert float(_assignment_source(class_name, "tabletop_palm_object_carry_rew_scale")) > 0.0
    assert int(
        _assignment_source(class_name, "tabletop_arm_lift_progress_baseline_grasp_streak")
    ) >= 12
    assert _assignment_source(
        class_name, "tabletop_arm_lift_baseline_sync_target_to_measured"
    ) == "True"
    assert _assignment_source(
        class_name, "tabletop_lift_hand_target_latch_uses_measured_pos"
    ) == "True"
    assert float(
        _assignment_source(class_name, "tabletop_lift_hand_target_close_fraction")
    ) == 0.15

    source = DYNAMIC_ENV_PATH.read_text(encoding="utf-8")
    assert "self._joint_targets[target_rows, target_cols] = measured_arm_pos" in source
    assert "self._prev_joint_targets[target_rows, target_cols] = measured_arm_pos" in source
    assert '"tabletop_lift_hand_target_latch_uses_measured_pos"' in source
    assert "self.robot.data.joint_pos[baseline_latch_ids]" in source
    assert "torch.minimum(" in source
    assert "torch.relu(self._palm_lin_vel_w[:, 2])" in source
    assert "torch.relu(self._object_lin_vel_w[:, 2])" in source
    assert "torch.maximum(lift_progress, tabletop_relative_palm_lift_progress)" in source


def test_palm_coupled_carry_curriculum_increases_lift_and_tightens_drift():
    stage_15 = "InspireRH56BFXFaithfulPrimitive3PalmCoupledCarry15mmTeacherEnvCfg"
    stage_30 = "InspireRH56BFXFaithfulPrimitive3PalmCoupledCarry30mmTeacherEnvCfg"
    strict = "InspireRH56BFXFaithfulPrimitive3PalmCoupledCarryTeacherEnvCfg"

    lift_heights = (
        float(_assignment_source(stage_15, "tabletop_success_lift_height")),
        float(_assignment_source(stage_30, "tabletop_success_lift_height")),
        float(_assignment_source(strict, "tabletop_success_min_relative_palm_lift")),
    )
    max_drifts = (
        float(_assignment_source(stage_15, "tabletop_success_max_object_palm_drift")),
        float(_assignment_source(stage_30, "tabletop_success_max_object_palm_drift")),
        float(_assignment_source(strict, "tabletop_success_max_object_palm_drift")),
    )
    assert lift_heights[0] < lift_heights[1] < lift_heights[2]
    assert max_drifts[0] > max_drifts[1] > max_drifts[2]


def test_vertical_carry_stage_rewards_pose_preserving_palm_translation():
    class_name = "InspireRH56BFXFaithfulPrimitive3VerticalCarry15mmTeacherEnvCfg"

    assert _assignment_source(class_name, "tabletop_object_asset_specs") == (
        "(TABLETOP_INSPIRE_SPHERE_60MM_SPEC,)"
    )
    assert _assignment_source(class_name, "tabletop_asset_curriculum") == "False"
    assert float(
        _assignment_source(class_name, "tabletop_vertical_palm_carry_rew_scale")
    ) > float(_assignment_source(class_name, "tabletop_palm_object_carry_rew_scale"))
    assert float(
        _assignment_source(class_name, "tabletop_vertical_palm_velocity_rew_scale")
    ) > 0.0
    assert float(_assignment_source(class_name, "tabletop_grasped_arm_lift_rew_scale")) == 0.0
    assert float(_assignment_source(class_name, "tabletop_lift_action_prior_rew_scale")) == 0.0
    assert float(_assignment_source(class_name, "tabletop_object_carry_lift_rew_scale")) == 0.0
    assert float(_assignment_source(class_name, "tabletop_palm_object_carry_rew_scale")) == 0.0
    assert float(_assignment_source(class_name, "tabletop_palm_object_up_vel_rew_scale")) == 0.0
    assert float(_assignment_source(class_name, "tabletop_success_max_palm_xy_drift")) == 0.025
    assert float(
        _assignment_source(class_name, "tabletop_success_max_palm_orientation_drift")
    ) == 0.40

    source = DYNAMIC_ENV_PATH.read_text(encoding="utf-8")
    assert "self._tabletop_lift_palm_baseline_quat_w[baseline_latch_ids]" in source
    assert "tabletop_palm_xy_drift <= max_success_palm_xy_drift" in source
    assert (
        "tabletop_palm_orientation_drift <= max_success_palm_orientation_drift"
        in source
    )
    assert 'self.extras["tabletop_vertical_palm_carry_rew_env"]' in source
    assert 'self.extras["tabletop_vertical_palm_velocity_rew_env"]' in source
    assert "torch.linalg.norm(self._palm_lin_vel_w[:, :2], dim=-1)" in source
    assert "torch.linalg.norm(self._palm_ang_vel_w, dim=-1)" in source
    assert 'self.extras["tabletop_pre_pose_success_env"]' in source
    assert 'self.extras["tabletop_palm_xy_success_gate_env"]' in source


def test_cartesian_lift_diagnostic_uses_latched_palm_pose_and_com_shift():
    source = EVAL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"--diagnostic-post-grasp-cartesian-lift-height"' in source
    assert '"--diagnostic-post-grasp-min-episode-step"' in source
    assert "env.episode_length_buf[: actions.shape[0]] >= min_episode_step" in source
    assert "def _cartesian_lift_arm_targets" in source
    assert "target_pos_w = env._tabletop_lift_palm_baseline_pos_w.clone()" in source
    assert "env._tabletop_lift_palm_baseline_quat_w" in source
    assert "palm_offset_link - com_pos_link" in source
    assert "compute_pose_error(" in source
    assert "(target_arm_pos - (1.0 - blend) * current_arm_target) / blend" in source
    assert source.count("actions = _apply_diagnostic_action_override(actions, wrapped_env)") == 3


def test_anydex_object_relative_teacher_only_tracks_xy_and_keeps_verified_lift():
    class_name = "InspireRH56BFXAnyDexRing60ObjectRelativeTeacherEnvCfg"
    assert _assignment_source(
        class_name, "scripted_tabletop_object_relative_pregrasp_enabled"
    ) == "True"
    assert ast.literal_eval(
        _assignment_source(class_name, "scripted_tabletop_object_relative_pregrasp_mode")
    ) == "iterative_pose_ik"
    assert ast.literal_eval(
        _assignment_source(class_name, "scripted_tabletop_pregrasp_prior_control_mode")
    ) == "object_relative_ik"
    assert len(
        ast.literal_eval(
            _assignment_source(
                class_name, "scripted_tabletop_object_relative_pregrasp_hand_base_quat"
            )
        )
    ) == 4
    assert ast.literal_eval(
        _assignment_source(class_name, "scripted_tabletop_object_relative_pregrasp_track_axes")
    ) == (True, True, False)
    assert float(
        _assignment_source(class_name, "scripted_tabletop_object_relative_pregrasp_lead_time")
    ) >= 0.0
    assert _assignment_source(class_name, "scripted_action_prior_uses_force_grasp") == "True"
    assert _assignment_source(
        class_name, "scripted_action_prior_lift_relative_to_grasp"
    ) == "True"
    assert _assignment_source(
        class_name, "scripted_action_prior_hand_proximity_trigger_enabled"
    ) == "True"
    assert int(
        _assignment_source(class_name, "scripted_action_prior_hand_start_step")
    ) >= 9999
    assert float(
        _assignment_source(
            class_name, "scripted_action_prior_hand_proximity_position_threshold"
        )
    ) > 0.0
    assert int(
        _assignment_source(class_name, "scripted_action_prior_hand_proximity_ramp_steps")
    ) > 0
    assert _assignment_source(
        class_name, "scripted_action_prior_hand_proximity_speed_adaptive_enabled"
    ) == "True"
    assert float(
        _assignment_source(
            class_name, "scripted_action_prior_hand_proximity_position_threshold_max"
        )
    ) >= float(
        _assignment_source(
            class_name, "scripted_action_prior_hand_proximity_position_threshold"
        )
    )

    source = DYNAMIC_ENV_PATH.read_text(encoding="utf-8")
    assert "def _compute_object_relative_pregrasp_joint_correction" in source
    assert 'tracking_mode == "iterative_pose_ik"' in source
    assert "compute_pose_error(" in source
    assert "self._object_relative_pregrasp_latched_joint_correction" in source
    assert "lift_target_pos.expand(self.num_envs, -1) + object_relative_correction" in source
    assert "self._scripted_lift_grasp_trigger_step" in source
    assert "def _apply_proximity_triggered_hand_prior" in source
    assert "self._scripted_hand_close_trigger_step" in source
    assert "self._scripted_hand_close_trigger_speed" in source

    base_source = BASE_ENV_PATH.read_text(encoding="utf-8")
    assert 'getattr(self.cfg, "scripted_action_prior_lift_relative_to_grasp", False)' in base_source
    assert 'getattr(self.cfg, "scripted_action_prior_uses_force_grasp", False)' in base_source
