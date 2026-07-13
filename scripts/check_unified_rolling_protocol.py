#!/usr/bin/env python3
"""Verify that Revo2 and Inspire use the same rolling benchmark contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from isaaclab.app import AppLauncher


DEFAULT_REVO2_TASK = "SimToolReal-Revo2-Franka-UnifiedRollingStage3-Teacher-Direct-v0"
DEFAULT_INSPIRE_TASK = "SimToolReal-Inspire-Franka-UnifiedRollingStage3-Teacher-Direct-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--revo2-task", default=DEFAULT_REVO2_TASK)
parser.add_argument("--inspire-task", default=DEFAULT_INSPIRE_TASK)
parser.add_argument("--summary-out", type=Path, required=True)
parser.add_argument("--audit-all-simple", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


PROTOCOL_FIELDS = (
    "benchmark_protocol",
    "task_family",
    "observation_space",
    "action_space",
    "policy_action_interface",
    "arm_action_scale",
    "arm_moving_average",
    "joint_target_arm_max_delta",
    "joint_target_hand_max_delta",
    "joint_target_rate_limit_requires_lift_baseline",
    "initial_arm_target_lock_steps",
    "initial_hand_target_lock_steps",
    "tabletop_arm_lift_progress_baseline_mode",
    "tabletop_arm_lift_progress_baseline_grasp_streak",
    "object_contact_force_diagnostics_enabled",
    "object_contact_force_threshold",
    "create_table",
    "table_top_z",
    "workspace_xy_limit",
    "dynamic_tabletop_workspace_x",
    "dynamic_tabletop_workspace_y",
    "object_start_pos",
    "reset_object_pos_noise",
    "tabletop_asset_set_enabled",
    "tabletop_asset_obs_enabled",
    "tabletop_asset_sampling_weights",
    "tabletop_asset_curriculum",
    "tabletop_asset_curriculum_mode",
    "tabletop_asset_curriculum_start_count",
    "tabletop_asset_curriculum_steps",
    "tabletop_motion_modes",
    "tabletop_motion_mode_curriculum",
    "dynamic_tabletop_persistent_motion",
    "dynamic_tabletop_bounce_at_workspace",
    "dynamic_tabletop_release_motion_on_contact",
    "dynamic_tabletop_randomize_yaw",
    "dynamic_tabletop_heading_range",
    "dynamic_grasp_speed_curriculum",
    "dynamic_grasp_speed_curriculum_mode",
    "dynamic_grasp_speed_curriculum_metric",
    "dynamic_grasp_speed_curriculum_start_success",
    "dynamic_grasp_speed_curriculum_full_success",
    "dynamic_grasp_speed_curriculum_ema_alpha",
    "dynamic_grasp_speed_curriculum_alpha_rise",
    "dynamic_grasp_speed_curriculum_allow_decrease",
    "dynamic_grasp_speed_curriculum_steps",
    "dynamic_tabletop_speed_alpha_sample_enabled",
    "dynamic_tabletop_start_speed_range",
    "dynamic_tabletop_initial_speed_range",
    "dynamic_tabletop_start_yaw_rate_range",
    "dynamic_tabletop_initial_yaw_rate_range",
    "dynamic_tabletop_pregrasp_lead_time",
    "dynamic_tabletop_pregrasp_ahead_distance",
    "dynamic_tabletop_pregrasp_ready_distance",
    "contact_distance",
    "contact_score_scale",
    "min_finger_contacts",
    "min_non_thumb_contacts",
    "opposition_cos_threshold",
    "strict_success_enabled",
    "strict_success_contact_distance",
    "strict_success_min_finger_contacts",
    "strict_success_min_non_thumb_contacts",
    "strict_success_opposition_mode",
    "strict_success_opposition_cos_threshold",
    "tabletop_success_requires_hover_target",
    "tabletop_success_lift_height",
    "dynamic_success_hold_steps",
    "stable_object_palm_vel",
    "tabletop_hover_height_delta",
    "tabletop_hover_latch_lift_progress",
    "tabletop_hover_xy_distance_scale",
    "tabletop_hover_z_distance_scale",
    "tabletop_hover_object_speed_scale",
    "tabletop_hover_ang_speed_scale",
    "tabletop_hover_success_requires_xy",
    "tabletop_hover_success_xy_tolerance",
    "tabletop_hover_success_z_tolerance",
    "tabletop_hover_success_object_speed",
    "tabletop_success_requires_arm_clearance",
    "tabletop_terminate_on_arm_clearance_violation",
    "tabletop_arm_clearance_body_names",
    "tabletop_arm_clearance_body_margins",
    "tabletop_arm_clearance_xy_padding",
    "tabletop_arm_clearance_scale",
    "tabletop_arm_clearance_max_penalty",
    "tabletop_arm_clearance_ok_penalty_threshold",
    "tabletop_arm_clearance_include_fingertip_points",
    "tabletop_arm_clearance_fingertip_point_margin",
    "tabletop_arm_clearance_include_palm_point",
    "tabletop_arm_clearance_palm_point_margin",
    "terminate_on_success",
    "episode_length_s",
    "tabletop_post_success_stability_latch_enabled",
    "tabletop_post_success_arm_target_lock_enabled",
    "tabletop_post_success_arm_target_lock_blend",
    "tabletop_post_success_hand_target_lock_enabled",
    "tabletop_post_success_hand_target_lock_blend",
    "tabletop_post_success_hand_lock_uses_actual_joint_pos",
    "tabletop_post_success_hand_close_fraction",
    "affordance_label_mode",
)

REWARD_FIELD_NAMES = {
    "action_penalty_scale",
    "arm_target_delta_penalty_scale",
    "drop_penalty",
    "success_bonus",
}
REWARD_FIELD_SUFFIXES = ("_rew_scale", "_penalty_scale")
OBJECTIVE_SEMANTIC_FIELDS = (
    "contact_reward_requires_thumb_pair",
    "contact_reward_uses_opposition_product",
    "contact_reward_opposition_min_multiplier",
    "opposition_reward_uses_weighted_score",
    "true_grasp_score_requires_thumb_pair",
    "true_grasp_score_uses_opposition_product",
    "true_grasp_score_opposition_min_multiplier",
    "thumb_contact_reward_weight",
    "thumb_true_grasp_score_weight",
    "grasp_quality_finger_count_weight",
    "grasp_quality_non_thumb_weight",
    "grasp_quality_thumb_weight",
    "grasp_quality_opposition_weight",
    "reach_distance_scale",
    "fingertip_distance_scale",
    "palm_contact_distance",
    "palm_only_lift_dist",
    "dynamic_tabletop_gate_contact_rewards_by_pregrasp",
    "dynamic_tabletop_contact_pregrasp_gate_min",
    "dynamic_tabletop_low_palm_height_scale",
    "dynamic_tabletop_low_palm_max_penalty",
    "dynamic_tabletop_min_palm_height_offset",
    "dynamic_tabletop_pregrasp_height_offset",
    "dynamic_tabletop_pregrasp_height_scale",
    "dynamic_tabletop_pregrasp_xy_distance_scale",
    "dynamic_tabletop_speed_alpha_sample_full_fraction",
    "lift_reward_min_grasp_quality_multiplier",
    "lift_reward_uses_grasp_quality_gate",
    "lift_reward_min_opposition_multiplier",
    "lift_reward_uses_opposition_gate",
    "quality_lift_progress_min_opposition_multiplier",
    "quality_lift_progress_uses_opposition_gate",
    "tabletop_arm_lift_reward_object_margin",
    "tabletop_arm_object_lift_gap_margin",
    "tabletop_grasped_palm_lift_height",
    "tabletop_grasped_palm_lift_scale",
    "tabletop_lift_action_prior_gate_min",
    "tabletop_lift_gate_requires_current_strict_grasp",
    "tabletop_lift_gate_requires_force_grasp",
    "tabletop_lift_rewards_require_current_strict_grasp",
    "tabletop_lift_rewards_require_force_grasp",
    "tabletop_lift_use_grasp_seen_gate",
    "tabletop_strict_grasp_loss_requires_lift_baseline",
    "tabletop_strict_grasp_loss_on_transition_only",
    "tabletop_lift_without_current_grasp_min_progress",
    "tabletop_lift_without_current_grasp_ramp",
    "tabletop_lift_without_object_min_arm_progress",
    "tabletop_no_lift_after_grasp_grace_steps",
    "tabletop_no_lift_after_grasp_max_penalty",
    "tabletop_no_lift_after_grasp_ramp_steps",
    "tabletop_no_lift_min_progress",
    "tabletop_no_lift_soft_grasp_gate",
    "tabletop_no_lift_uses_soft_grasp_gate",
    "tabletop_no_lift_uses_force_grasp_gate",
    "tabletop_non_thumb_without_thumb_gate_start",
    "tabletop_non_thumb_without_thumb_gate_ramp",
    "tabletop_non_thumb_without_thumb_thumb_target",
    "tabletop_non_thumb_without_thumb_penalty_lift_gate_min",
    "tabletop_gate_boolean_grasp_rewards_by_clearance",
    "tabletop_gate_contact_rewards_by_clearance",
    "tabletop_contact_clearance_gate_min",
    "tabletop_contact_clearance_gate_scale",
    "tabletop_hover_latch_uses_grasp_seen",
    "tabletop_hover_reward_uses_grasp_seen",
    "tabletop_success_uses_grasp_seen",
    "tabletop_object_carry_grasp_seen_gate",
    "tabletop_object_carry_uses_grasp_seen",
    "tabletop_object_carry_min_grasp_streak",
    "tabletop_object_carry_stall_min_arm_progress",
    "tabletop_object_carry_stall_min_z_vel",
    "tabletop_object_carry_streak_ramp_steps",
    "tabletop_object_up_vel_scale",
    "tabletop_force_grasp_streak_target",
    "tabletop_success_requires_force_grasp",
    "tabletop_stable_catch_min_lift_multiplier",
    "tabletop_post_success_arm_target_drift_scale",
    "tabletop_post_success_arm_target_drift_tolerance",
    "tabletop_post_success_palm_drift_scale",
    "tabletop_post_success_palm_drift_tolerance",
    "tabletop_underwrap_below_center_fraction",
    "tabletop_underwrap_contact_margin",
    "tabletop_underwrap_contact_scale",
    "tabletop_underwrap_height_scale",
    "tabletop_underwrap_opposition_min_multiplier",
    "tabletop_underwrap_pair_weight",
    "tabletop_underwrap_progress_weight",
    "tabletop_underwrap_radial_fraction",
    "tabletop_underwrap_radial_scale",
    "tabletop_underwrap_uses_pregrasp_gate",
    "strict_approach_score_scale",
    "strict_reward_contact_score_scale",
    "strict_touch_reward_opposition_min_multiplier",
    "strict_touch_reward_requires_thumb_pair",
    "strict_touch_reward_uses_opposition_product",
    "strict_touch_score_scale",
    "scripted_action_prior_enabled",
    "scripted_tabletop_pregrasp_prior_enabled",
    "scripted_tabletop_relative_lift_target_prior_enabled",
    "scripted_tabletop_hand_grasp_memory_prior_enabled",
)


def _normalize(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _required_value(cfg, field: str):
    try:
        return getattr(cfg, field)
    except SystemExit as exc:
        raise RuntimeError(f"required config field {field!r} raised SystemExit") from exc


def _simple_config_snapshot(cfg) -> dict:
    snapshot = {}
    for name in dir(cfg):
        if name.startswith("_"):
            continue
        try:
            value = getattr(cfg, name)
        except (Exception, SystemExit):
            continue
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, list, tuple, dict, Path)) or value is None:
            snapshot[name] = _normalize(value)
    return snapshot


def _asset_snapshot(cfg) -> list[dict]:
    fields = (
        "asset_id",
        "category",
        "proxy_shape",
        "size",
        "radius",
        "height",
        "mass",
        "static_friction",
        "dynamic_friction",
        "restitution",
        "contact_offset",
        "rest_offset",
        "affordance_mode",
        "positive_fraction",
        "negative_fraction",
    )
    return [
        {field: _normalize(spec.get(field)) for field in fields}
        for spec in tuple(_required_value(cfg, "tabletop_object_asset_specs"))
    ]


def _camera_snapshot(camera_cfg) -> dict:
    return {
        "height": int(camera_cfg.height),
        "width": int(camera_cfg.width),
        "data_types": list(camera_cfg.data_types),
        "offset_pos": _normalize(camera_cfg.offset.pos),
        "offset_rot": _normalize(camera_cfg.offset.rot),
        "focal_length": float(camera_cfg.spawn.focal_length),
    }


def _snapshot(cfg, *, label: str) -> dict:
    print(f"[AUDIT] snapshot {label}: protocol fields", flush=True)
    snapshot = {field: _normalize(_required_value(cfg, field)) for field in PROTOCOL_FIELDS}
    print(f"[AUDIT] snapshot {label}: reward fields", flush=True)
    reward_fields = sorted(
        name
        for name in dir(cfg)
        if name in REWARD_FIELD_NAMES or name.endswith(REWARD_FIELD_SUFFIXES)
    )
    snapshot["reward_contract"] = {
        field: _normalize(_required_value(cfg, field)) for field in reward_fields
    }
    snapshot["objective_semantics"] = {
        field: _normalize(_required_value(cfg, field)) for field in OBJECTIVE_SEMANTIC_FIELDS
    }
    print(f"[AUDIT] snapshot {label}: assets and simulation", flush=True)
    snapshot["tabletop_object_asset_specs"] = _asset_snapshot(cfg)
    snapshot["sim_dt"] = float(cfg.sim.dt)
    snapshot["sim_render_interval"] = int(cfg.sim.render_interval)
    snapshot["decimation"] = int(cfg.decimation)
    snapshot["student_camera"] = _camera_snapshot(cfg.student_camera)
    snapshot["video_camera"] = _camera_snapshot(cfg.video_camera)
    return snapshot


def main() -> None:
    revo2_cfg = load_cfg_from_registry(args_cli.revo2_task, "env_cfg_entry_point")
    inspire_cfg = load_cfg_from_registry(args_cli.inspire_task, "env_cfg_entry_point")
    snapshots = {
        "revo2": _snapshot(revo2_cfg, label="revo2"),
        "inspire": _snapshot(inspire_cfg, label="inspire"),
    }
    differences = {
        key: {"revo2": snapshots["revo2"][key], "inspire": snapshots["inspire"][key]}
        for key in snapshots["revo2"]
        if snapshots["revo2"][key] != snapshots["inspire"][key]
    }
    result = {
        "passed": not differences,
        "tasks": {"revo2": args_cli.revo2_task, "inspire": args_cli.inspire_task},
        "protocol": snapshots["revo2"].get("benchmark_protocol"),
        "differences": differences,
        "snapshots": snapshots,
        "embodiment_adapters": {
            "revo2": {
                "hand_embodiment": revo2_cfg.hand_embodiment,
                "action_contract": revo2_cfg.action_contract,
                "palm_body_name": revo2_cfg.palm_body_name,
                "active_hand_dofs": len(revo2_cfg.hand_joint_names),
                "clearance_body_names": _normalize(revo2_cfg.tabletop_arm_clearance_body_names),
                "clearance_body_margins": _normalize(revo2_cfg.tabletop_arm_clearance_body_margins),
                "hand_moving_average": _normalize(revo2_cfg.hand_moving_average),
                "reference_hand_fractions": _normalize(revo2_cfg.reference_hand_fractions),
                "default_arm_pos": _normalize(revo2_cfg.default_arm_pos),
                "robot_arm_init_joint_pos": {
                    key: _normalize(value)
                    for key, value in sorted(revo2_cfg.robot_cfg.init_state.joint_pos.items())
                    if str(key).startswith("panda_joint")
                },
                "lift_baseline_pos": _normalize(revo2_cfg.tabletop_arm_lift_progress_baseline_pos),
                "lift_arm_delta": _normalize(revo2_cfg.lift_arm_delta),
                "lift_action_prior": _normalize(revo2_cfg.lift_action_prior),
            },
            "inspire": {
                "hand_embodiment": inspire_cfg.hand_embodiment,
                "action_contract": inspire_cfg.action_contract,
                "palm_body_name": inspire_cfg.palm_body_name,
                "active_hand_dofs": len(inspire_cfg.hand_joint_names),
                "clearance_body_names": _normalize(inspire_cfg.tabletop_arm_clearance_body_names),
                "clearance_body_margins": _normalize(inspire_cfg.tabletop_arm_clearance_body_margins),
                "hand_moving_average": _normalize(inspire_cfg.hand_moving_average),
                "reference_hand_fractions": _normalize(inspire_cfg.reference_hand_fractions),
                "semantic_close_targets": _normalize(inspire_cfg.inspire_semantic_close_targets),
                "default_arm_pos": _normalize(inspire_cfg.default_arm_pos),
                "robot_arm_init_joint_pos": {
                    key: _normalize(value)
                    for key, value in sorted(inspire_cfg.robot_cfg.init_state.joint_pos.items())
                    if str(key).startswith("panda_joint")
                },
                "lift_baseline_pos": _normalize(inspire_cfg.tabletop_arm_lift_progress_baseline_pos),
                "lift_arm_delta": _normalize(inspire_cfg.lift_arm_delta),
                "lift_action_prior": _normalize(inspire_cfg.lift_action_prior),
            },
        },
    }
    if args_cli.audit_all_simple:
        simple_snapshots = {
            "revo2": _simple_config_snapshot(revo2_cfg),
            "inspire": _simple_config_snapshot(inspire_cfg),
        }
        simple_keys = sorted(set(simple_snapshots["revo2"]) | set(simple_snapshots["inspire"]))
        result["all_simple_differences"] = {
            key: {
                "revo2": simple_snapshots["revo2"].get(key, "<missing>"),
                "inspire": simple_snapshots["inspire"].get(key, "<missing>"),
            }
            for key in simple_keys
            if simple_snapshots["revo2"].get(key, "<missing>")
            != simple_snapshots["inspire"].get(key, "<missing>")
        }
    out_path = args_cli.summary_out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "differences": differences}, sort_keys=True), flush=True)
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
