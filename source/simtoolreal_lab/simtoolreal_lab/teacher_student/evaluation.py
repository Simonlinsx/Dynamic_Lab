"""Episode-level diagnostics shared by teacher and student evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import torch


BOOL_EXTRA_KEYS = {
    "palm_reach": "palm_reach_env",
    "thumb_contact": "thumb_contact_env",
    "strict_thumb_contact": "strict_thumb_contact_env",
    "thumb_surface_dist_lt_002": "thumb_surface_dist_lt_002_env",
    "thumb_surface_dist_lt_004": "thumb_surface_dist_lt_004_env",
    "thumb_surface_dist_lt_006": "thumb_surface_dist_lt_006_env",
    "thumb_surface_dist_lt_008": "thumb_surface_dist_lt_008_env",
    "thumb_surface_dist_lt_010": "thumb_surface_dist_lt_010_env",
    "strict_opposing_contact": "strict_opposing_contact_env",
    "strict_true_grasp": "strict_true_grasp_env",
    "strict_grasp_hold": "strict_grasp_hold_env",
    "clean_grasp_candidate": "tabletop_clean_grasp_candidate_env",
    "clean_grasp_latched": "tabletop_clean_grasp_latched_env",
    "post_clean_grip_retained": "tabletop_post_clean_grip_retained_env",
    "unclean_lift": "tabletop_unclean_lift_env",
    "clean_lifted": "tabletop_clean_lifted_env",
    "clean_stable_hold": "tabletop_clean_stable_hold_env",
    "lifted": "lifted_env",
    "strict_lifted": "strict_lifted_env",
    "lifted_low_rel_vel": "lifted_low_rel_vel_env",
    "strict_stable_hold": "strict_stable_hold_env",
    "strict_stable_hover_latched": "strict_stable_hover_latched_env",
    "strict_stable_hover_target_ok": "strict_stable_hover_target_ok_env",
    "strict_stable_hover_speed_ok": "strict_stable_hover_speed_ok_env",
    "strict_stable_hover_ang_speed_ok": "strict_stable_hover_ang_speed_ok_env",
    "strict_stable_clearance_ok": "strict_stable_clearance_ok_env",
    "hover_latched": "hover_latched_env",
    "stable_hold": "stable_hold_env",
    "success": "success_env",
    "positive_affordance_contact": "positive_affordance_contact_env",
    "negative_affordance_contact": "negative_affordance_contact_env",
    "force_thumb_contact": "force_thumb_contact_env",
    "force_multifinger_contact": "force_multifinger_contact_env",
    "force_grasp": "force_grasp_env",
    "force_grasp_clearance_ok": "force_grasp_clearance_ok_env",
    "force_stable_grasp": "force_stable_grasp_env",
    "force_lifted": "force_lifted_env",
    "force_stable_hold": "force_stable_hold_env",
    "tabletop_arm_lift_baseline_latched": "tabletop_arm_lift_baseline_latched_env",
    "tabletop_pre_pose_success": "tabletop_pre_pose_success_env",
    "tabletop_object_palm_drift_success_gate": "tabletop_object_palm_drift_success_gate_env",
    "tabletop_palm_xy_success_gate": "tabletop_palm_xy_success_gate_env",
    "tabletop_palm_orientation_success_gate": "tabletop_palm_orientation_success_gate_env",
    "dropped": "dropped_env",
    "out_of_workspace": "out_xy_env",
    "time_out": "time_out_env",
}

MAX_EXTRA_KEYS = {
    "success_streak": "success_streak_env",
    "strict_finger_contacts": "strict_finger_contacts_env",
    "object_fingertip_force_sum": "object_fingertip_force_sum_env",
    "object_fingertip_force_max": "object_fingertip_force_max_env",
    "force_grasp_streak": "force_grasp_streak_env",
    "tabletop_arm_lift_progress": "tabletop_arm_lift_progress_env",
    "tabletop_relative_palm_lift": "tabletop_relative_palm_lift_env",
    "tabletop_relative_palm_lift_progress": "tabletop_relative_palm_lift_progress_env",
    "tabletop_object_palm_drift": "tabletop_object_palm_drift_env",
    "tabletop_palm_object_up_vel_rew": "tabletop_palm_object_up_vel_rew_env",
    "tabletop_lift_action_prior": "tabletop_lift_action_prior_env",
    "tabletop_lift_action_prior_gate": "tabletop_lift_action_prior_gate_env",
    "tabletop_lift_action_prior_rew": "tabletop_lift_action_prior_rew_env",
    "tabletop_lift_target_error": "tabletop_lift_target_error_env",
    "tabletop_lift_target_gate": "tabletop_lift_target_gate_env",
    "tabletop_lift_target_rew": "tabletop_lift_target_rew_env",
    "tabletop_lift_cartesian_position_error": (
        "tabletop_lift_cartesian_position_error_env"
    ),
    "tabletop_lift_cartesian_rotation_error": (
        "tabletop_lift_cartesian_rotation_error_env"
    ),
    "tabletop_object_up_vel_rew": "tabletop_object_up_vel_rew_env",
    "tabletop_object_carry_lift_rew": "tabletop_object_carry_lift_rew_env",
    "tabletop_post_clean_grip_score": "tabletop_post_clean_grip_score_env",
    "tabletop_post_clean_grip_retention_rew": (
        "tabletop_post_clean_grip_retention_rew_env"
    ),
    "tabletop_post_clean_grip_loss_penalty": (
        "tabletop_post_clean_grip_loss_penalty_env"
    ),
    "tabletop_post_clean_strict_grasp_retention_rew": (
        "tabletop_post_clean_strict_grasp_retention_rew_env"
    ),
    "tabletop_post_clean_hand_opening_penalty": (
        "tabletop_post_clean_hand_opening_penalty_env"
    ),
    "tabletop_clean_lift_phase_progress": (
        "tabletop_clean_lift_phase_progress_env"
    ),
    "tabletop_true_grasp_streak": "tabletop_true_grasp_streak_env",
    "tabletop_strict_true_grasp_streak": "tabletop_strict_true_grasp_streak_env",
    "tabletop_clean_grasp_streak": "tabletop_clean_grasp_streak_env",
    "object_height_delta": "object_height_delta_env",
    "strict_lift_height_delta": "strict_lift_height_delta_env",
    "strict_stable_height_delta": "strict_stable_height_delta_env",
    "force_lift_height_delta": "force_lift_height_delta_env",
    "tabletop_underwrap_pair_score": "tabletop_underwrap_pair_score_env",
    "tabletop_underwrap_progress_score": "tabletop_underwrap_progress_score_env",
}

MIN_EXTRA_KEYS = {
    "palm_distance": "palm_distance_env",
    "object_palm_rel_vel": "object_palm_rel_vel_env",
    "thumb_surface_dist": "thumb_surface_dist_env",
    "tabletop_underwrap_min_tip_z_rel": "tabletop_underwrap_min_tip_z_rel_env",
    "tabletop_underwrap_thumb_z_rel": "tabletop_underwrap_thumb_z_rel_env",
    "tabletop_underwrap_min_non_thumb_z_rel": "tabletop_underwrap_min_non_thumb_z_rel_env",
}


def flatten_numeric_metrics(
    payload: Mapping[str, object], prefix: str = ""
) -> dict[str, float]:
    """Flatten nested numeric mappings into slash-delimited metric names."""

    metrics: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            metrics.update(flatten_numeric_metrics(value, name))
        elif isinstance(value, bool):
            metrics[name] = float(value)
        elif isinstance(value, (int, float)):
            metrics[name] = float(value)
    return metrics


def _per_env_tensor(value, num_envs: int, device: torch.device, *, boolean: bool) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 0:
        tensor = tensor.expand(num_envs)
    if tensor.shape[0] < num_envs:
        raise ValueError(f"Diagnostic tensor has {tensor.shape[0]} envs, expected at least {num_envs}.")
    tensor = tensor[:num_envs]
    if tensor.ndim > 1:
        tensor = tensor.reshape(num_envs, -1)
        tensor = tensor.any(dim=-1) if boolean else tensor.max(dim=-1).values
    return tensor.bool() if boolean else tensor.float()


class EpisodeFunnelTracker:
    """Track whether each environment reaches diagnostic stages within an episode."""

    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.boolean = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in BOOL_EXTRA_KEYS
        }
        self.maximum = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for name in MAX_EXTRA_KEYS
        }
        self.minimum = {
            name: torch.full((self.num_envs,), torch.inf, dtype=torch.float32, device=self.device)
            for name in MIN_EXTRA_KEYS
        }
        self.available: set[str] = set()

    def update(self, extras: Mapping[str, object]) -> None:
        for name, key in BOOL_EXTRA_KEYS.items():
            if key not in extras:
                continue
            value = _per_env_tensor(extras[key], self.num_envs, self.device, boolean=True)
            self.boolean[name] |= value
            self.available.add(name)
        for name, key in MAX_EXTRA_KEYS.items():
            if key not in extras:
                continue
            value = _per_env_tensor(extras[key], self.num_envs, self.device, boolean=False)
            self.maximum[name] = torch.maximum(self.maximum[name], value)
            self.available.add(name)
        for name, key in MIN_EXTRA_KEYS.items():
            if key not in extras:
                continue
            value = _per_env_tensor(extras[key], self.num_envs, self.device, boolean=False)
            self.minimum[name] = torch.minimum(self.minimum[name], value)
            self.available.add(name)

    def snapshot(self, env_ids: torch.Tensor) -> dict[str, object]:
        return {
            "boolean": {name: values[env_ids] for name, values in self.boolean.items()},
            "maximum": {name: values[env_ids] for name, values in self.maximum.items()},
            "minimum": {name: values[env_ids] for name, values in self.minimum.items()},
            "available": set(self.available),
        }

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        for values in self.boolean.values():
            values[env_ids] = False
        for values in self.maximum.values():
            values[env_ids] = 0.0
        for values in self.minimum.values():
            values[env_ids] = torch.inf


class EpisodeFunnelAccumulator:
    """Aggregate tracker snapshots and assign one primary failure stage per episode."""

    def __init__(self, task_family: str):
        self.task_family = str(task_family)
        self.episodes = 0
        self.stage_counts: dict[str, int] = {}
        self.failure_counts: dict[str, int] = {}
        self.scalar_sums: dict[str, float] = {}
        self.scalar_maxima: dict[str, float] = {}
        self.transition_counts: dict[str, int] = {}
        self.transition_denominators: dict[str, int] = {}
        self.available: set[str] = set()

    def add(self, snapshot: Mapping[str, object]) -> None:
        boolean = snapshot["boolean"]
        maximum = snapshot["maximum"]
        minimum = snapshot["minimum"]
        available = set(snapshot["available"])
        if not boolean:
            return
        batch_size = int(next(iter(boolean.values())).shape[0])
        if batch_size == 0:
            return
        self.episodes += batch_size
        self.available.update(available)

        for name, values in boolean.items():
            if name in available:
                self.stage_counts[name] = self.stage_counts.get(name, 0) + int(values.sum().item())

        transition_pairs = []
        if self.task_family == "dynamic_tabletop_grasp":
            transition_pairs.extend(
                [
                    ("strict_grasp_to_lift", "strict_true_grasp", "lifted"),
                    ("strict_grasp_to_clean_grasp", "strict_true_grasp", "clean_grasp_latched"),
                    (
                        "clean_grasp_to_retained_grip",
                        "clean_grasp_latched",
                        "post_clean_grip_retained",
                    ),
                    (
                        "retained_grip_to_clean_lift",
                        "post_clean_grip_retained",
                        "clean_lifted",
                    ),
                    ("clean_grasp_to_clean_lift", "clean_grasp_latched", "clean_lifted"),
                    ("clean_lift_to_clean_stable_hold", "clean_lifted", "clean_stable_hold"),
                    ("strict_grasp_to_strict_lift", "strict_true_grasp", "strict_lifted"),
                    ("strict_lift_to_strict_stable_hold", "strict_lifted", "strict_stable_hold"),
                    ("force_grasp_to_force_lift", "force_grasp", "force_lifted"),
                    ("force_lift_to_force_stable_hold", "force_lifted", "force_stable_hold"),
                    ("lift_to_stable_hold", "lifted", "stable_hold"),
                ]
            )
        else:
            transition_pairs.append(
                ("strict_grasp_to_stable_hold", "strict_true_grasp", "stable_hold")
            )
        transition_pairs.extend(
            [
                ("stable_hold_to_success", "stable_hold", "success"),
                ("force_grasp_to_success", "force_grasp", "success"),
            ]
        )
        for transition_name, source_name, target_name in transition_pairs:
            if source_name not in available or target_name not in available:
                continue
            source = boolean[source_name]
            self.transition_denominators[transition_name] = (
                self.transition_denominators.get(transition_name, 0) + int(source.sum().item())
            )
            self.transition_counts[transition_name] = self.transition_counts.get(
                transition_name, 0
            ) + int((source & boolean[target_name]).sum().item())
        for name, values in {**maximum, **minimum}.items():
            if name not in available:
                continue
            finite = values[torch.isfinite(values)]
            if finite.numel() == 0:
                continue
            self.scalar_sums[name] = self.scalar_sums.get(name, 0.0) + float(finite.sum().item())
            self.scalar_maxima[name] = max(
                self.scalar_maxima.get(name, -float("inf")),
                float(finite.max().item()),
            )

        success = boolean["success"] if "success" in available else torch.zeros(batch_size, dtype=torch.bool, device=next(iter(boolean.values())).device)
        remaining = ~success
        self.failure_counts["success"] = self.failure_counts.get("success", 0) + int(success.sum().item())

        stage_order = [
            ("no_palm_reach", "palm_reach"),
            ("no_strict_thumb_contact", "strict_thumb_contact"),
            ("no_strict_opposition", "strict_opposing_contact"),
            ("no_strict_grasp", "strict_true_grasp"),
        ]
        if self.task_family == "dynamic_tabletop_grasp":
            stage_order.extend(
                [
                    ("no_clean_grasp_latch", "clean_grasp_latched"),
                    ("no_post_clean_grip_retention", "post_clean_grip_retained"),
                    ("no_lift", "lifted"),
                    ("no_stable_hold", "stable_hold"),
                ]
            )
        else:
            stage_order.extend(
                [
                    ("no_positive_affordance_contact", "positive_affordance_contact"),
                    ("no_stable_hold", "stable_hold"),
                ]
            )

        for failure_name, achieved_name in stage_order:
            if achieved_name not in available:
                continue
            failed_here = remaining & (~boolean[achieved_name])
            self.failure_counts[failure_name] = self.failure_counts.get(failure_name, 0) + int(failed_here.sum().item())
            remaining &= boolean[achieved_name]
        self.failure_counts["insufficient_success_streak"] = (
            self.failure_counts.get("insufficient_success_streak", 0) + int(remaining.sum().item())
        )

    def summary(self) -> dict[str, object]:
        episodes = max(self.episodes, 1)
        stage_rates = {
            name: count / episodes for name, count in sorted(self.stage_counts.items())
        }
        failure_rates = {
            name: count / episodes for name, count in sorted(self.failure_counts.items())
        }
        scalar_means = {
            name: total / episodes for name, total in sorted(self.scalar_sums.items())
        }
        conversions = {
            name: count / max(self.transition_denominators.get(name, 0), 1)
            for name, count in sorted(self.transition_counts.items())
        }

        return {
            "episodes": int(self.episodes),
            "stage_counts": dict(sorted(self.stage_counts.items())),
            "stage_rates": stage_rates,
            "conversion_rates": conversions,
            "conversion_counts": dict(sorted(self.transition_counts.items())),
            "conversion_denominators": dict(sorted(self.transition_denominators.items())),
            "primary_failure_counts": dict(sorted(self.failure_counts.items())),
            "primary_failure_rates": failure_rates,
            "episode_scalar_means": scalar_means,
            "episode_scalar_maxima": dict(sorted(self.scalar_maxima.items())),
            "available_signals": sorted(self.available),
        }
