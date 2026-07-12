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


DEFAULT_REVO2_TASK = "SimToolReal-Revo2-Franka-UnifiedRollingBenchmark-Teacher-Direct-v0"
DEFAULT_INSPIRE_TASK = "SimToolReal-Inspire-Franka-UnifiedRollingBenchmark-Teacher-Direct-v0"

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
        for spec in tuple(cfg.tabletop_object_asset_specs)
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


def _snapshot(cfg) -> dict:
    snapshot = {field: _normalize(getattr(cfg, field)) for field in PROTOCOL_FIELDS}
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
        "revo2": _snapshot(revo2_cfg),
        "inspire": _snapshot(inspire_cfg),
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
            },
            "inspire": {
                "hand_embodiment": inspire_cfg.hand_embodiment,
                "action_contract": inspire_cfg.action_contract,
                "palm_body_name": inspire_cfg.palm_body_name,
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
