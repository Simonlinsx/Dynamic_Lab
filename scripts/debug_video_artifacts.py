"""Write reproducible sidecars for simulation debug videos."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_VIDEO_ROOT = REPO_ROOT / "logs" / "debug_videos"


_ENV_FIELDS = (
    "reference_name",
    "benchmark_protocol",
    "hand_embodiment",
    "action_contract",
    "policy_action_interface",
    "action_space",
    "observation_space",
    "episode_length_s",
    "decimation",
    "default_arm_pos",
    "reset_arm_pos_noise",
    "reset_object_pos_noise",
    "arm_joint_names",
    "hand_joint_names",
    "sim_hand_joint_names",
    "touch_body_names",
    "palm_body_name",
    "palm_offset",
    "object_shape",
    "object_radius",
    "object_size",
    "object_start_pos",
    "table_top_z",
    "contact_distance",
    "contact_score_scale",
    "min_finger_contacts",
    "min_non_thumb_contacts",
    "strict_success_enabled",
    "strict_success_contact_distance",
    "strict_success_min_finger_contacts",
    "strict_success_min_non_thumb_contacts",
    "strict_success_opposition_mode",
    "strict_success_opposition_cos_threshold",
    "object_contact_force_diagnostics_enabled",
    "object_contact_force_threshold",
    "object_force_grasp_min_non_thumb_contacts",
    "tabletop_success_requires_force_grasp",
    "lift_success_height",
    "tabletop_success_lift_height",
    "stable_object_palm_vel",
    "dynamic_success_hold_steps",
    "terminate_on_success",
    "scripted_action_prior_enabled",
    "scripted_tabletop_pregrasp_prior_enabled",
    "scripted_tabletop_approach_action_prior_enabled",
    "scripted_tabletop_lift_target_prior_enabled",
    "tabletop_privileged_lift_target_obs_enabled",
    "joint_target_arm_max_delta",
    "joint_target_hand_max_delta",
    "cartesian_wrist_position_action_scale",
    "cartesian_wrist_rotation_action_scale",
    "cartesian_impedance_motion_stiffness",
    "cartesian_impedance_motion_damping_ratio",
    "cartesian_impedance_inertial_dynamics_decoupling",
    "cartesian_impedance_partial_inertial_dynamics_decoupling",
    "cartesian_impedance_nullspace_control",
    "cartesian_impedance_nullspace_stiffness",
    "dynamic_grasp_speed_curriculum",
    "dynamic_grasp_speed_curriculum_override_alpha",
    "dynamic_tabletop_initial_speed_range",
    "dynamic_tabletop_heading_range",
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _nested_attr(value: Any, path: str) -> Any:
    current = value
    for name in path.split("."):
        if current is None:
            return None
        current = getattr(current, name, None)
    return current


def collect_env_config(env_cfg: Any) -> dict[str, Any]:
    """Collect the control, physics, reward, and success contract actually used."""

    if env_cfg is None:
        return {}
    config = {
        field: _jsonable(getattr(env_cfg, field))
        for field in _ENV_FIELDS
        if hasattr(env_cfg, field)
    }
    nested_fields = {
        "sim_dt": "sim.dt",
        "sim_gravity": "sim.gravity",
        "robot_self_collision": "robot_cfg.spawn.articulation_props.enabled_self_collisions",
        "robot_init_joint_pos": "robot_cfg.init_state.joint_pos",
        "object_init_pos": "object_cfg.init_state.pos",
        "object_static_friction": "object_cfg.spawn.physics_material.static_friction",
        "object_dynamic_friction": "object_cfg.spawn.physics_material.dynamic_friction",
        "object_restitution": "object_cfg.spawn.physics_material.restitution",
    }
    for output_name, path in nested_fields.items():
        nested_value = _nested_attr(env_cfg, path)
        if nested_value is not None:
            config[output_name] = _jsonable(nested_value)

    reward_scales = {}
    for name in dir(env_cfg):
        if not (
            name.endswith("_rew_scale")
            or name.endswith("_penalty_scale")
            or name == "success_bonus"
        ):
            continue
        try:
            value = getattr(env_cfg, name)
        except Exception:
            continue
        if isinstance(value, (int, float)):
            reward_scales[name] = float(value)
    config["reward_scales"] = dict(sorted(reward_scales.items()))
    return config


def write_debug_video_artifacts(
    video_path: str | Path,
    *,
    method: str,
    args: argparse.Namespace | dict[str, Any] | None = None,
    env_cfg: Any = None,
    checkpoint: str | Path | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write config, metrics, and exact-command files beside one debug video."""

    path = Path(video_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    config_path = path.with_suffix(".config.json")
    metrics_path = path.with_suffix(".metrics.json")
    command_path = path.with_suffix(".command.txt")
    cli_args = vars(args) if isinstance(args, argparse.Namespace) else (args or {})
    command = shlex.join([sys.executable, *sys.argv])
    config = {
        "schema": "simtoolreal_lab_debug_video_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": str(method),
        "video_path": str(path),
        "checkpoint": str(Path(checkpoint).expanduser().resolve()) if checkpoint else None,
        "command": command,
        "cli": _jsonable(cli_args),
        "environment": collect_env_config(env_cfg),
        "extra": _jsonable(extra or {}),
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_path.write_text(
        json.dumps(_jsonable(metrics or {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command_path.write_text(command + "\n", encoding="utf-8")
    return {
        "config": str(config_path),
        "metrics": str(metrics_path),
        "command": str(command_path),
    }

