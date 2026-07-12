#!/usr/bin/env python3
"""Verify that Revo2 and Inspire share one falling-baton task contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from isaaclab.app import AppLauncher


DEFAULT_REVO2_TASK = "SimToolReal-Revo2-Franka-UnifiedFallingBatonBenchmark-Teacher-Direct-v0"
DEFAULT_INSPIRE_TASK = "SimToolReal-Inspire-Franka-UnifiedFallingBatonBenchmark-Teacher-Direct-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--revo2-task", default=DEFAULT_REVO2_TASK)
parser.add_argument("--inspire-task", default=DEFAULT_INSPIRE_TASK)
parser.add_argument("--summary-out", type=Path, required=True)
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
    "create_table",
    "episode_length_s",
    "object_shape",
    "object_radius",
    "object_size",
    "object_start_pos",
    "object_start_rot",
    "reset_object_pos_noise",
    "affordance_label_mode",
    "affordance_positive_fraction",
    "affordance_negative_fraction",
    "affordance_positive_end",
    "falling_baton_affordance_markers_enabled",
    "falling_baton_positive_marker_local_offset",
    "falling_baton_neutral_marker_local_offset",
    "falling_baton_negative_marker_local_offset",
    "falling_baton_palm_relative_spawn_enabled",
    "falling_baton_palm_relative_clamp_to_workspace",
    "falling_baton_spawn_x_range",
    "falling_baton_spawn_y_range",
    "falling_baton_spawn_z_range",
    "falling_baton_spawn_above_palm_enabled",
    "falling_baton_spawn_height_curriculum",
    "falling_baton_start_spawn_above_palm_range",
    "falling_baton_spawn_above_palm_range",
    "falling_baton_catch_center_finger_weight",
    "falling_baton_catch_center_forward_offset",
    "falling_baton_catch_center_world_offset",
    "falling_baton_palm_relative_start_x_range",
    "falling_baton_palm_relative_start_y_range",
    "falling_baton_randomize_orientation",
    "falling_baton_orientation_curriculum",
    "falling_baton_start_roll_range",
    "falling_baton_start_pitch_range",
    "falling_baton_start_yaw_range",
    "falling_baton_roll_range",
    "falling_baton_pitch_range",
    "falling_baton_yaw_range",
    "object_lin_vel_min",
    "object_lin_vel_max",
    "object_ang_vel_min",
    "object_ang_vel_max",
    "falling_baton_start_initial_xy_speed_range",
    "falling_baton_start_initial_z_speed_range",
    "falling_baton_start_initial_ang_vel_range",
    "dynamic_grasp_speed_curriculum",
    "dynamic_grasp_speed_curriculum_mode",
    "dynamic_grasp_speed_curriculum_metric",
    "dynamic_grasp_speed_curriculum_start_success",
    "dynamic_grasp_speed_curriculum_full_success",
    "dynamic_grasp_speed_curriculum_ema_alpha",
    "dynamic_grasp_speed_curriculum_alpha_rise",
    "dynamic_grasp_speed_curriculum_allow_decrease",
    "dynamic_grasp_speed_curriculum_override_alpha",
    "falling_success_uses_grasp_seen",
    "falling_success_uses_strict_grasp",
    "falling_success_requires_positive_affordance",
    "falling_success_max_palm_distance",
    "falling_success_min_finger_contacts",
    "catch_success_min_z",
    "falling_drop_z",
    "contact_distance",
    "contact_score_scale",
    "palm_contact_distance",
    "strict_reward_enabled",
    "strict_success_enabled",
    "strict_success_contact_distance",
    "strict_success_min_finger_contacts",
    "strict_success_min_non_thumb_contacts",
    "strict_success_opposition_mode",
    "strict_success_opposition_cos_threshold",
    "falling_affordance_reward_enabled",
    "falling_affordance_positive_requires_thumb_pair",
    "falling_affordance_positive_uses_opposition_product",
    "falling_affordance_positive_opposition_min_multiplier",
    "falling_affordance_distance_scale",
    "falling_affordance_contact_distance",
    "falling_affordance_radial_margin",
    "dynamic_success_hold_steps",
    "stable_object_palm_vel",
    "terminate_on_success",
    "falling_post_success_stability_enabled",
    "tabletop_post_success_stability_latch_enabled",
    "tabletop_post_success_arm_target_lock_enabled",
    "tabletop_post_success_arm_target_lock_blend",
    "tabletop_post_success_hand_target_lock_enabled",
    "tabletop_post_success_hand_target_lock_blend",
    "tabletop_post_success_hand_lock_uses_actual_joint_pos",
    "tabletop_post_success_hand_close_fraction",
    "scripted_action_prior_enabled",
    "scripted_tabletop_pregrasp_prior_enabled",
    "scripted_tabletop_relative_lift_target_prior_enabled",
    "scripted_tabletop_hand_grasp_memory_prior_enabled",
)

REWARD_FIELD_NAMES = {
    "action_penalty_scale",
    "arm_target_delta_penalty_scale",
    "drop_penalty",
    "success_bonus",
}
REWARD_FIELD_SUFFIXES = ("_rew_scale", "_penalty_scale")


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


def _camera_snapshot(camera_cfg) -> dict:
    return {
        "height": int(camera_cfg.height),
        "width": int(camera_cfg.width),
        "data_types": list(camera_cfg.data_types),
        "offset_pos": _normalize(camera_cfg.offset.pos),
        "offset_rot": _normalize(camera_cfg.offset.rot),
        "focal_length": float(camera_cfg.spawn.focal_length),
    }


def _object_snapshot(cfg) -> dict:
    spawn = cfg.object_cfg.spawn
    material = spawn.physics_material
    return {
        "size": _normalize(spawn.size),
        "mass": float(spawn.mass_props.mass),
        "contact_offset": float(spawn.collision_props.contact_offset),
        "rest_offset": float(spawn.collision_props.rest_offset),
        "static_friction": float(material.static_friction),
        "dynamic_friction": float(material.dynamic_friction),
        "restitution": float(material.restitution),
    }


def _marker_snapshot(cfg, region: str) -> dict:
    marker_cfg = getattr(cfg, f"falling_baton_{region}_marker_cfg")
    spawn = marker_cfg.spawn
    return {
        "size": _normalize(spawn.size),
        "color": _normalize(spawn.visual_material.diffuse_color),
        "local_offset": _normalize(getattr(cfg, f"falling_baton_{region}_marker_local_offset")),
    }


def _snapshot(cfg) -> dict:
    snapshot = {field: _normalize(getattr(cfg, field)) for field in PROTOCOL_FIELDS}
    reward_fields = sorted(
        name
        for name in dir(cfg)
        if name in REWARD_FIELD_NAMES or name.endswith(REWARD_FIELD_SUFFIXES)
    )
    snapshot["reward_contract"] = {
        field: _normalize(getattr(cfg, field)) for field in reward_fields
    }
    snapshot["object_physics"] = _object_snapshot(cfg)
    snapshot["affordance_markers"] = {
        region: _marker_snapshot(cfg, region) for region in ("positive", "neutral", "negative")
    }
    snapshot["sim_dt"] = float(cfg.sim.dt)
    snapshot["sim_render_interval"] = int(cfg.sim.render_interval)
    snapshot["decimation"] = int(cfg.decimation)
    snapshot["student_camera"] = _camera_snapshot(cfg.student_camera)
    snapshot["video_camera"] = _camera_snapshot(cfg.video_camera)
    return snapshot


def main() -> None:
    revo2_cfg = load_cfg_from_registry(args_cli.revo2_task, "env_cfg_entry_point")
    inspire_cfg = load_cfg_from_registry(args_cli.inspire_task, "env_cfg_entry_point")
    snapshots = {"revo2": _snapshot(revo2_cfg), "inspire": _snapshot(inspire_cfg)}
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
            },
            "inspire": {
                "hand_embodiment": inspire_cfg.hand_embodiment,
                "action_contract": inspire_cfg.action_contract,
                "palm_body_name": inspire_cfg.palm_body_name,
                "active_hand_dofs": len(inspire_cfg.hand_joint_names),
            },
        },
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
