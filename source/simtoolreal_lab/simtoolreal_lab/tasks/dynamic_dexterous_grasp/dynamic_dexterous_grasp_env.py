"""IsaacLab DirectRLEnv for dynamic dexterous grasp privileged teachers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse, sample_uniform

from simtoolreal_lab.tasks.revo2_static_grasp.revo2_static_grasp_env import Revo2StaticGraspEnv

from .dynamic_dexterous_grasp_env_cfg import (
    TABLETOP_AFFORDANCE_ROOT,
    Revo2DynamicDexterousTeacherEnvCfg,
    _object_cfg_from_tabletop_spec,
)


def _quat_from_euler_xyz_wxyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles to quaternions in IsaacLab's wxyz convention."""

    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr = torch.cos(half_roll)
    sr = torch.sin(half_roll)
    cp = torch.cos(half_pitch)
    sp = torch.sin(half_pitch)
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)
    return torch.cat(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def _spawn_local_ground(show_grid: bool = False) -> None:
    """Spawn a local ground plane so video runs do not depend on remote Isaac assets."""

    ground_cfg = sim_utils.CuboidCfg(
        size=(20.0, 20.0, 0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18), roughness=0.9),
    )
    ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, -0.011))

    if not show_grid:
        return

    line_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.62, 0.62), roughness=0.8)
    line_collision = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    grid_extent = 5.0
    grid_step = 0.25
    grid_count = int(grid_extent / grid_step)
    for line_idx in range(-grid_count, grid_count + 1):
        coord = line_idx * grid_step
        thickness = 0.018 if line_idx % 4 == 0 else 0.010
        x_line_cfg = sim_utils.CuboidCfg(
            size=(thickness, 2.0 * grid_extent, 0.003),
            collision_props=line_collision,
            visual_material=line_material,
        )
        y_line_cfg = sim_utils.CuboidCfg(
            size=(2.0 * grid_extent, thickness, 0.003),
            collision_props=line_collision,
            visual_material=line_material,
        )
        x_line_cfg.func(f"/World/ground_grid/x_{line_idx + grid_count:02d}", x_line_cfg, translation=(coord, 0.0, 0.002))
        y_line_cfg.func(f"/World/ground_grid/y_{line_idx + grid_count:02d}", y_line_cfg, translation=(0.0, coord, 0.002))


class DynamicDexterousGraspEnv(Revo2StaticGraspEnv):
    """Dynamic teacher task with privileged low-dimensional simulator state."""

    cfg: Revo2DynamicDexterousTeacherEnvCfg

    def __init__(self, cfg: Revo2DynamicDexterousTeacherEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._active_hand_joint_ids = list(self._hand_joint_ids)
        self._active_hand_joint_names = list(self._hand_joint_names)
        if self._uses_active_hand_actions():
            self._sim_hand_joint_ids, self._sim_hand_joint_names = self.robot.find_joints(
                list(self.cfg.sim_hand_joint_names), preserve_order=True
            )
            self._control_hand_joint_ids = list(self._sim_hand_joint_ids)
            self._controlled_joint_ids = self._arm_joint_ids + self._control_hand_joint_ids
        else:
            self._sim_hand_joint_ids = list(self._hand_joint_ids)
            self._sim_hand_joint_names = list(self._hand_joint_names)
            self._control_hand_joint_ids = list(self._hand_joint_ids)
        self._inspire_semantic_close_targets = None
        if self._uses_inspire_active_hand_actions():
            close_targets_cfg = getattr(self.cfg, "inspire_semantic_close_targets", None)
            if close_targets_cfg is not None:
                close_targets = torch.tensor(close_targets_cfg, dtype=torch.float32, device=self.device).flatten()
                if close_targets.numel() != len(self._control_hand_joint_ids):
                    raise ValueError(
                        "inspire_semantic_close_targets must match sim_hand_joint_names; "
                        f"got {close_targets.numel()} values for {len(self._control_hand_joint_ids)} joints "
                        f"{self._sim_hand_joint_names}."
                    )
                hand_lower = self._joint_lower_limits[self._control_hand_joint_ids]
                hand_upper = self._joint_upper_limits[self._control_hand_joint_ids]
                self._inspire_semantic_close_targets = torch.clamp(close_targets, hand_lower, hand_upper)
        self._dynamic_speed_curriculum_success_alpha = 0.0
        self._dynamic_speed_curriculum_metric_value = 0.0
        self._dynamic_speed_curriculum_metric_ema = 0.0
        self._dynamic_speed_curriculum_metric_initialized = False
        self._tabletop_cmd_lin_vel_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._tabletop_cmd_yaw_rate = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._tabletop_motion_mode_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._object_hover_target_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._object_hover_target_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._active_object_start_z = torch.full(
            (self.num_envs,),
            float(getattr(self.cfg, "object_start_pos", (0.0, 0.0, 0.0))[2]),
            dtype=torch.float32,
            device=self.device,
        )
        tabletop_lift_baseline = getattr(self.cfg, "tabletop_arm_lift_progress_baseline_pos", None)
        if tabletop_lift_baseline is None and bool(
            getattr(self.cfg, "scripted_tabletop_pregrasp_prior_enabled", False)
        ):
            tabletop_lift_baseline = getattr(self.cfg, "scripted_tabletop_pregrasp_arm_pos", None)
        if tabletop_lift_baseline is None:
            tabletop_lift_baseline_pos = self._default_arm_pos
        else:
            tabletop_lift_baseline_pos = torch.tensor(
                tabletop_lift_baseline,
                dtype=torch.float32,
                device=self.device,
            ).view(1, -1)
        self._tabletop_arm_lift_baseline_mode = str(
            getattr(self.cfg, "tabletop_arm_lift_progress_baseline_mode", "fixed")
        )
        if self._tabletop_arm_lift_baseline_mode not in {
            "fixed",
            "first_strict_grasp",
            "first_force_grasp",
        }:
            raise ValueError(
                "tabletop_arm_lift_progress_baseline_mode must be 'fixed', "
                "'first_strict_grasp', or 'first_force_grasp'; "
                f"got {self._tabletop_arm_lift_baseline_mode!r}."
            )
        self._tabletop_arm_lift_fixed_baseline_pos = tabletop_lift_baseline_pos
        self._tabletop_arm_lift_baseline_pos = tabletop_lift_baseline_pos.expand(
            self.num_envs, -1
        ).clone()
        self._tabletop_arm_lift_baseline_latched = torch.full(
            (self.num_envs,),
            self._tabletop_arm_lift_baseline_mode == "fixed",
            dtype=torch.bool,
            device=self.device,
        )
        self._tabletop_arm_lift_baseline_grasp_streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._tabletop_true_grasp_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._tabletop_strict_true_grasp_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._strict_reward_grasp_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._scripted_lift_grasp_recent_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._strict_grasp_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._post_success_stability_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._post_success_arm_joint_target = torch.zeros(
            (self.num_envs, len(self._arm_joint_ids)), dtype=torch.float32, device=self.device
        )
        self._post_success_hand_joint_target = torch.zeros(
            (self.num_envs, len(self._control_hand_joint_ids)), dtype=torch.float32, device=self.device
        )
        self._post_success_palm_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._object_fingertip_contact_forces = torch.zeros(
            (self.num_envs, len(self.cfg.touch_body_names)), dtype=torch.float32, device=self.device
        )
        self._force_thumb_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._force_multifinger_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._force_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._force_grasp_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._force_grasp_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._force_grasp_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._scripted_relative_lift_target_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._scripted_relative_lift_target_arm_pos = torch.zeros(
            (self.num_envs, len(self._arm_joint_ids)), dtype=torch.float32, device=self.device
        )
        self._scripted_relative_lift_target_delta = torch.zeros(
            (self.num_envs, len(self._arm_joint_ids)), dtype=torch.float32, device=self.device
        )
        self._scripted_relative_lift_target_candidate_labels = None
        clearance_names = tuple(getattr(self.cfg, "tabletop_arm_clearance_body_names", ()))
        clearance_margins = tuple(getattr(self.cfg, "tabletop_arm_clearance_body_margins", ()))
        default_clearance_margin = float(getattr(self.cfg, "tabletop_arm_clearance_margin", 0.040))
        clearance_pairs = []
        for body_index, name in enumerate(clearance_names):
            if name not in self.robot.body_names:
                continue
            margin = float(clearance_margins[body_index]) if body_index < len(clearance_margins) else default_clearance_margin
            clearance_pairs.append((self.robot.body_names.index(name), margin))
        self._tabletop_arm_clearance_body_ids = [body_id for body_id, _ in clearance_pairs]
        self._tabletop_arm_clearance_body_margins = torch.tensor(
            [margin for _, margin in clearance_pairs],
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)
        self._tabletop_arm_clearance_xy_center = torch.tensor(
            getattr(self.cfg, "tabletop_arm_clearance_xy_center", (0.58, -0.08)),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 2)
        self._tabletop_arm_clearance_xy_half_extent = torch.tensor(
            getattr(self.cfg, "tabletop_arm_clearance_xy_half_extent", (0.50, 0.40)),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 2)
        self._tabletop_arm_clearance_xy_padding = float(getattr(self.cfg, "tabletop_arm_clearance_xy_padding", 0.05))
        self._tabletop_arm_clearance_penalty = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._tabletop_arm_clearance_min_margin = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._tabletop_arm_clearance_active_fraction = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._tabletop_arm_clearance_ok = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._init_tabletop_asset_tensors()
        self._scripted_lift_action_candidate_labels = None
        self._scripted_tabletop_hand_grasp_memory_action = None
        self._scripted_tabletop_hand_grasp_memory_action_candidate_labels = None
        self._apply_scripted_lift_action_candidates()
        self._apply_scripted_relative_lift_target_candidates()
        self._apply_scripted_tabletop_hand_grasp_memory_action_candidates()

    def _apply_scripted_lift_action_candidates(self) -> None:
        candidate_cfg = getattr(self.cfg, "scripted_action_prior_lift_candidate_actions", None)
        if candidate_cfg is None:
            return
        candidates = torch.tensor(candidate_cfg, dtype=torch.float32, device=self.device)
        if candidates.ndim != 2 or candidates.shape[-1] != len(self._arm_joint_ids):
            raise ValueError(
                "scripted_action_prior_lift_candidate_actions must have shape "
                f"(N, {len(self._arm_joint_ids)}), got {tuple(candidates.shape)}"
            )
        if candidates.shape[0] <= 0:
            return
        env_candidate_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) % candidates.shape[0]
        self._scripted_lift_action = candidates[env_candidate_ids]

        labels_cfg = tuple(getattr(self.cfg, "scripted_action_prior_lift_candidate_labels", ()))
        if len(labels_cfg) != candidates.shape[0]:
            labels_cfg = tuple(f"candidate_{idx:02d}" for idx in range(candidates.shape[0]))
        self._scripted_lift_action_candidate_labels = [labels_cfg[int(idx)] for idx in env_candidate_ids.cpu().tolist()]

    def _apply_scripted_tabletop_hand_grasp_memory_action_candidates(self) -> None:
        candidate_cfg = getattr(self.cfg, "scripted_tabletop_hand_grasp_memory_action_candidates", None)
        if candidate_cfg is None:
            return
        hand_dim = int(getattr(self.cfg, "action_space", len(self._arm_joint_ids))) - len(self._arm_joint_ids)
        if hand_dim <= 0:
            return
        candidates = torch.tensor(candidate_cfg, dtype=torch.float32, device=self.device)
        if candidates.ndim == 1:
            candidates = candidates.view(1, -1)
        if candidates.ndim != 2 or candidates.shape[-1] != hand_dim:
            raise ValueError(
                "scripted_tabletop_hand_grasp_memory_action_candidates must have shape "
                f"(N, {hand_dim}), got {tuple(candidates.shape)}"
            )
        if candidates.shape[0] <= 0:
            return
        env_candidate_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) % candidates.shape[0]
        self._scripted_tabletop_hand_grasp_memory_action = candidates[env_candidate_ids]

        labels_cfg = tuple(getattr(self.cfg, "scripted_tabletop_hand_grasp_memory_action_candidate_labels", ()))
        if len(labels_cfg) != candidates.shape[0]:
            labels_cfg = tuple(f"hand_candidate_{idx:02d}" for idx in range(candidates.shape[0]))
        self._scripted_tabletop_hand_grasp_memory_action_candidate_labels = [
            labels_cfg[int(idx)] for idx in env_candidate_ids.cpu().tolist()
        ]

    def _apply_scripted_relative_lift_target_candidates(self) -> None:
        base_delta = torch.tensor(
            getattr(self.cfg, "scripted_tabletop_relative_lift_target_arm_delta", (0.0,) * len(self._arm_joint_ids)),
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)
        if base_delta.shape[-1] != len(self._arm_joint_ids):
            raise ValueError(
                "scripted_tabletop_relative_lift_target_arm_delta must contain "
                f"{len(self._arm_joint_ids)} values, got {base_delta.shape[-1]}"
            )
        self._scripted_relative_lift_target_delta[:] = base_delta.expand_as(
            self._scripted_relative_lift_target_delta
        )

        candidate_cfg = getattr(self.cfg, "scripted_tabletop_relative_lift_target_candidate_deltas", None)
        if candidate_cfg is None:
            return
        candidates = torch.tensor(candidate_cfg, dtype=torch.float32, device=self.device)
        if candidates.ndim != 2 or candidates.shape[-1] != len(self._arm_joint_ids):
            raise ValueError(
                "scripted_tabletop_relative_lift_target_candidate_deltas must have shape "
                f"(N, {len(self._arm_joint_ids)}), got {tuple(candidates.shape)}"
            )
        if candidates.shape[0] <= 0:
            return
        env_candidate_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) % candidates.shape[0]
        self._scripted_relative_lift_target_delta[:] = candidates[env_candidate_ids]

        labels_cfg = tuple(getattr(self.cfg, "scripted_tabletop_relative_lift_target_candidate_labels", ()))
        if len(labels_cfg) != candidates.shape[0]:
            labels_cfg = tuple(f"relative_candidate_{idx:02d}" for idx in range(candidates.shape[0]))
        self._scripted_relative_lift_target_candidate_labels = [
            labels_cfg[int(idx)] for idx in env_candidate_ids.cpu().tolist()
        ]

    def _uses_tabletop_asset_set_cfg(self) -> bool:
        specs = tuple(getattr(self.cfg, "tabletop_object_asset_specs", ()))
        return bool(getattr(self.cfg, "tabletop_asset_set_enabled", False)) and len(specs) > 0

    def _uses_tabletop_asset_set(self) -> bool:
        return self.cfg.task_family == "dynamic_tabletop_grasp" and self._uses_tabletop_asset_set_cfg()

    def _uses_falling_baton_affordance_markers_cfg(self) -> bool:
        return (
            self.cfg.task_family == "falling_baton_grasp"
            and bool(getattr(self.cfg, "falling_baton_affordance_markers_enabled", False))
            and (bool(getattr(self.cfg, "video_camera_enabled", False)) or bool(getattr(self.cfg, "student_camera_enabled", False)))
        )

    def _falling_baton_affordance_marker_cfgs(self):
        return (
            (
                "positive",
                getattr(self.cfg, "falling_baton_positive_marker_cfg", None),
                getattr(self.cfg, "falling_baton_positive_marker_local_offset", (0.0, 0.0, 0.0)),
            ),
            (
                "neutral",
                getattr(self.cfg, "falling_baton_neutral_marker_cfg", None),
                getattr(self.cfg, "falling_baton_neutral_marker_local_offset", (0.0, 0.0, 0.0)),
            ),
            (
                "negative",
                getattr(self.cfg, "falling_baton_negative_marker_cfg", None),
                getattr(self.cfg, "falling_baton_negative_marker_local_offset", (0.0, 0.0, 0.0)),
            ),
        )

    def _tabletop_spec_support_height(self, spec: dict) -> float:
        shape = str(spec.get("proxy_shape", "box")).lower()
        if shape == "sphere":
            return float(spec.get("radius", 0.03))
        if shape in {"cylinder", "cone"}:
            return 0.5 * float(spec.get("height", spec.get("size", (0.04, 0.04, 0.08))[2]))
        return 0.5 * float(spec.get("size", (0.04, 0.04, 0.08))[2])

    def _tabletop_spec_start_pos(self, spec: dict) -> tuple[float, float, float]:
        x, y, _ = self.cfg.object_start_pos
        return (float(x), float(y), float(self.cfg.table_top_z) + self._tabletop_spec_support_height(spec) + 0.002)

    def _create_object_contact_force_sensors(self, object_prim_paths: Sequence[str]) -> None:
        self._object_contact_force_sensors = []
        if not bool(getattr(self.cfg, "object_contact_force_diagnostics_enabled", False)):
            return
        filter_paths = [str(path) for path in object_prim_paths]
        robot_prim_path = str(self.cfg.robot_cfg.prim_path).rstrip("/")
        for body_name in self.cfg.touch_body_names:
            sensor_cfg = ContactSensorCfg(
                prim_path=f"{robot_prim_path}/{body_name}",
                update_period=0.0,
                history_length=1,
                track_air_time=False,
                filter_prim_paths_expr=filter_paths,
            )
            self._object_contact_force_sensors.append(ContactSensor(sensor_cfg))

    def _register_object_contact_force_sensors(self) -> None:
        for index, sensor in enumerate(getattr(self, "_object_contact_force_sensors", ())):
            self.scene.sensors[f"object_contact_tip_{index}"] = sensor

    def _update_object_contact_force_diagnostics(self) -> None:
        sensors = getattr(self, "_object_contact_force_sensors", ())
        if not sensors:
            self._object_fingertip_contact_forces.zero_()
            self._force_thumb_contact.zero_()
            self._force_multifinger_contact.zero_()
            self._force_grasp_prev.zero_()
            self._force_grasp.zero_()
            self._force_grasp_seen.zero_()
            self._force_grasp_streak.zero_()
            return

        self._force_grasp_prev.copy_(self._force_grasp)
        force_per_tip = []
        for sensor in sensors:
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is None:
                force_per_tip.append(torch.zeros(self.num_envs, dtype=torch.float32, device=self.device))
                continue
            filtered_force_norm = torch.linalg.vector_norm(force_matrix[:, 0], dim=-1)
            force_per_tip.append(filtered_force_norm.sum(dim=-1))
        self._object_fingertip_contact_forces = torch.stack(force_per_tip, dim=-1)
        threshold = max(float(getattr(self.cfg, "object_contact_force_threshold", 0.05)), 0.0)
        force_contacts = self._object_fingertip_contact_forces > threshold
        self._force_thumb_contact = force_contacts[:, 0]
        force_non_thumb_count = force_contacts[:, 1:].sum(dim=-1)
        self._force_multifinger_contact = force_contacts.sum(dim=-1) >= 3
        self._force_grasp = self._force_thumb_contact & (force_non_thumb_count >= 2)
        self._force_grasp_seen = self._force_grasp_seen | self._force_grasp
        self._force_grasp_streak = torch.where(
            self._force_grasp,
            self._force_grasp_streak + 1,
            torch.zeros_like(self._force_grasp_streak),
        )

    def _setup_scene(self):
        if not self._uses_tabletop_asset_set_cfg():
            self.robot = Articulation(self.cfg.robot_cfg)
            self.object = RigidObject(self.cfg.object_cfg)
            self.table = RigidObject(self.cfg.table_cfg) if self.cfg.create_table else None
            self._falling_baton_affordance_markers = []
            if self._uses_falling_baton_affordance_markers_cfg():
                for marker_name, marker_cfg, marker_offset in self._falling_baton_affordance_marker_cfgs():
                    if marker_cfg is None:
                        continue
                    self._falling_baton_affordance_markers.append(
                        (marker_name, RigidObject(marker_cfg), marker_offset)
                    )
            if self.cfg.video_camera_enabled:
                self._video_camera = TiledCamera(self.cfg.video_camera)
            if self.cfg.student_camera_enabled:
                self._student_camera = TiledCamera(self.cfg.student_camera)
            self._create_object_contact_force_sensors((self.cfg.object_cfg.prim_path,))
            self._bind_robot_physics_material()
            _spawn_local_ground(
                show_grid=bool(getattr(self.cfg, "video_camera_enabled", False))
                or bool(getattr(self.cfg, "student_camera_enabled", False))
            )
            self.scene.clone_environments(copy_from_source=False)
            self._apply_robot_self_collision_filters()
            self.scene.articulations["robot"] = self.robot
            self.scene.rigid_objects["object"] = self.object
            for marker_name, marker_object, _ in self._falling_baton_affordance_markers:
                self.scene.rigid_objects[f"falling_baton_affordance_{marker_name}"] = marker_object
            if self.table is not None:
                self.scene.rigid_objects["table"] = self.table
            if self.cfg.video_camera_enabled:
                self.scene.sensors["video_camera"] = self._video_camera
            if self.cfg.student_camera_enabled:
                self.scene.sensors["student_camera"] = self._student_camera
            self._register_object_contact_force_sensors()
            light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.78, 0.78, 0.76))
            light_cfg.func("/World/Light", light_cfg)
            return

        self.robot = Articulation(self.cfg.robot_cfg)
        self._tabletop_asset_specs = tuple(getattr(self.cfg, "tabletop_object_asset_specs", ()))
        self._falling_baton_affordance_markers = []
        self._tabletop_objects = []
        for asset_index, spec in enumerate(self._tabletop_asset_specs):
            object_cfg = _object_cfg_from_tabletop_spec(
                spec,
                pos=self._tabletop_spec_start_pos(spec),
                prim_suffix=f"Asset{asset_index}",
            )
            self._tabletop_objects.append(RigidObject(object_cfg))
        self.object = self._tabletop_objects[0]
        self.table = RigidObject(self.cfg.table_cfg) if self.cfg.create_table else None
        if self.cfg.video_camera_enabled:
            self._video_camera = TiledCamera(self.cfg.video_camera)
        if self.cfg.student_camera_enabled:
            self._student_camera = TiledCamera(self.cfg.student_camera)
        self._create_object_contact_force_sensors(tuple(obj.cfg.prim_path for obj in self._tabletop_objects))
        self._bind_robot_physics_material()
        _spawn_local_ground(
            show_grid=bool(getattr(self.cfg, "video_camera_enabled", False))
            or bool(getattr(self.cfg, "student_camera_enabled", False))
        )
        self.scene.clone_environments(copy_from_source=False)
        self._apply_robot_self_collision_filters()
        self.scene.articulations["robot"] = self.robot
        for asset_index, obj in enumerate(self._tabletop_objects):
            self.scene.rigid_objects[f"object_asset_{asset_index}"] = obj
        if self.table is not None:
            self.scene.rigid_objects["table"] = self.table
        if self.cfg.video_camera_enabled:
            self.scene.sensors["video_camera"] = self._video_camera
        if self.cfg.student_camera_enabled:
            self.scene.sensors["student_camera"] = self._student_camera
        self._register_object_contact_force_sensors()
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.78, 0.78, 0.76))
        light_cfg.func("/World/Light", light_cfg)

    def _shape_code_from_spec(self, spec: dict) -> int:
        shape = str(spec.get("proxy_shape", "box")).lower()
        return {"sphere": 0, "ball": 0, "box": 1, "cube": 1, "cylinder": 2, "cone": 3}.get(shape, 1)

    def _tabletop_affordance_label_path(self, spec: dict) -> Path:
        asset_id = str(spec.get("asset_id", ""))
        root = Path(getattr(self.cfg, "tabletop_affordance_label_root", TABLETOP_AFFORDANCE_ROOT))
        return root.joinpath(*asset_id.split("/")) / "grasp_affordance_clean_v2.npz"

    def _fit_affordance_vertices_to_spec(self, vertices: np.ndarray, spec: dict) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=np.float64)
        if str(spec.get("asset_id", "")).startswith("domino20/"):
            vertices = vertices[:, [0, 2, 1]]
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        center = 0.5 * (bounds_min + bounds_max)
        extents = np.maximum(bounds_max - bounds_min, 1.0e-9)
        target_size = np.asarray(spec.get("size", (0.05, 0.05, 0.05)), dtype=np.float64)
        return ((vertices - center) * (target_size / extents)).astype(np.float32)

    def _sample_affordance_region_points(self, points: np.ndarray, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        if points.size == 0:
            return np.zeros((count, 3), dtype=np.float32)
        if points.shape[0] >= count:
            indices = np.linspace(0, points.shape[0] - 1, count, dtype=np.int64)
        else:
            repeats = int(np.ceil(count / max(points.shape[0], 1)))
            indices = np.tile(np.arange(points.shape[0], dtype=np.int64), repeats)[:count]
        return np.asarray(points[indices], dtype=np.float32)

    def _load_tabletop_affordance_point_tensors(self, specs: tuple[dict, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        point_count = max(int(getattr(self.cfg, "tabletop_affordance_reward_points_per_region", 64)), 1)
        pos_points = []
        neg_points = []
        pos_valid = []
        neg_valid = []
        for spec in specs:
            label_path = self._tabletop_affordance_label_path(spec)
            if not label_path.exists():
                pos_points.append(np.zeros((point_count, 3), dtype=np.float32))
                neg_points.append(np.zeros((point_count, 3), dtype=np.float32))
                pos_valid.append(False)
                neg_valid.append(False)
                continue
            data = np.load(label_path, allow_pickle=True)
            vertices = self._fit_affordance_vertices_to_spec(data["vertices_obj"], spec)
            positive_mask = np.asarray(data["positive_mask"], dtype=bool)
            negative_mask = np.asarray(data["negative_mask"], dtype=bool)
            positive_points = vertices[positive_mask]
            negative_points = vertices[negative_mask]
            pos_points.append(self._sample_affordance_region_points(positive_points, point_count))
            neg_points.append(self._sample_affordance_region_points(negative_points, point_count))
            pos_valid.append(positive_points.shape[0] > 0)
            neg_valid.append(negative_points.shape[0] > 0)
        return (
            torch.tensor(np.stack(pos_points, axis=0), dtype=torch.float32, device=self.device),
            torch.tensor(np.stack(neg_points, axis=0), dtype=torch.float32, device=self.device),
            torch.tensor(pos_valid, dtype=torch.bool, device=self.device),
            torch.tensor(neg_valid, dtype=torch.bool, device=self.device),
        )

    def _init_tabletop_asset_tensors(self) -> None:
        if not self._uses_tabletop_asset_set():
            return
        specs = tuple(getattr(self, "_tabletop_asset_specs", tuple(getattr(self.cfg, "tabletop_object_asset_specs", ()))))
        self._tabletop_asset_specs = specs
        self._tabletop_asset_count = len(specs)
        self._tabletop_active_asset_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._active_object_shape_codes_tensor = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._active_object_size_tensor = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._active_object_radius_tensor = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._active_object_height_tensor = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._active_object_start_z = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._active_asset_positive_fraction = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._active_asset_negative_fraction = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._tabletop_asset_obs = torch.zeros((self.num_envs, 10), dtype=torch.float32, device=self.device)

        self._tabletop_asset_shape_codes = torch.tensor(
            [self._shape_code_from_spec(spec) for spec in specs], dtype=torch.long, device=self.device
        )
        self._tabletop_asset_sizes = torch.tensor(
            [spec["size"] for spec in specs], dtype=torch.float32, device=self.device
        )
        self._tabletop_asset_radii = torch.tensor(
            [float(spec["radius"]) for spec in specs], dtype=torch.float32, device=self.device
        )
        self._tabletop_asset_heights = torch.tensor(
            [float(spec["height"]) for spec in specs], dtype=torch.float32, device=self.device
        )
        self._tabletop_asset_support_heights = torch.tensor(
            [self._tabletop_spec_support_height(spec) for spec in specs], dtype=torch.float32, device=self.device
        )
        self._tabletop_asset_positive_fractions = torch.tensor(
            [float(spec.get("positive_fraction", 0.0)) for spec in specs], dtype=torch.float32, device=self.device
        )
        self._tabletop_asset_negative_fractions = torch.tensor(
            [float(spec.get("negative_fraction", 0.0)) for spec in specs], dtype=torch.float32, device=self.device
        )
        sampling_weights = getattr(self.cfg, "tabletop_asset_sampling_weights", None)
        if sampling_weights is None:
            self._tabletop_asset_sampling_weights = torch.ones(
                self._tabletop_asset_count, dtype=torch.float32, device=self.device
            )
        else:
            if len(sampling_weights) != self._tabletop_asset_count:
                raise ValueError(
                    "tabletop_asset_sampling_weights must contain one weight per tabletop asset "
                    f"({len(sampling_weights)} != {self._tabletop_asset_count})"
                )
            self._tabletop_asset_sampling_weights = torch.tensor(
                sampling_weights, dtype=torch.float32, device=self.device
            ).clamp_min(0.0)
            if float(self._tabletop_asset_sampling_weights.sum().item()) <= 0.0:
                raise ValueError("tabletop_asset_sampling_weights must contain at least one positive weight")
        (
            self._tabletop_affordance_pos_points_l,
            self._tabletop_affordance_neg_points_l,
            self._tabletop_affordance_pos_valid,
            self._tabletop_affordance_neg_valid,
        ) = self._load_tabletop_affordance_point_tensors(specs)
        self._update_active_tabletop_asset_tensors(torch.arange(self.num_envs, dtype=torch.long, device=self.device))

    def _uses_active_hand_actions(self) -> bool:
        return self._uses_revo2_active_hand_actions() or self._uses_inspire_active_hand_actions()

    def _uses_revo2_active_hand_actions(self) -> bool:
        return (
            str(getattr(self.cfg, "hand_embodiment", "")) == "revo2"
            and str(getattr(self.cfg, "action_contract", "")) == "revo2_semantic_13d"
        )

    def _uses_inspire_active_hand_actions(self) -> bool:
        return (
            str(getattr(self.cfg, "hand_embodiment", "")) == "inspire"
            and str(getattr(self.cfg, "action_contract", "")) == "inspire_semantic_13d"
        )

    def _tabletop_asset_curriculum_alpha(self) -> float:
        override_alpha = getattr(self.cfg, "tabletop_asset_curriculum_override_alpha", None)
        if override_alpha is not None:
            return min(max(float(override_alpha), 0.0), 1.0)
        if not bool(getattr(self.cfg, "tabletop_asset_curriculum", False)):
            return 1.0
        mode = str(getattr(self.cfg, "tabletop_asset_curriculum_mode", "steps")).lower()
        if mode in {"dynamic_speed", "shared_performance", "speed_curriculum"}:
            return self._dynamic_speed_curriculum_alpha()
        steps = int(getattr(self.cfg, "tabletop_asset_curriculum_steps", 0))
        if steps <= 0:
            return 1.0
        return self._frame_curriculum_alpha(steps)

    def _tabletop_motion_mode_curriculum_alpha(self) -> float:
        override_alpha = getattr(self.cfg, "tabletop_motion_mode_curriculum_override_alpha", None)
        if override_alpha is not None:
            return min(max(float(override_alpha), 0.0), 1.0)
        if not bool(getattr(self.cfg, "tabletop_motion_mode_curriculum", False)):
            return 1.0
        steps = int(getattr(self.cfg, "tabletop_motion_mode_curriculum_steps", 0))
        if steps <= 0:
            return 1.0
        return self._frame_curriculum_alpha(steps)

    def _frame_curriculum_alpha(self, frame_steps: int) -> float:
        if frame_steps <= 0:
            return 1.0
        env_count = max(int(getattr(self, "num_envs", 1)), 1)
        frame_count = float(self.common_step_counter) * float(env_count)
        return min(frame_count / float(frame_steps), 1.0)

    def _tabletop_current_asset_count(self) -> int:
        asset_count = int(getattr(self, "_tabletop_asset_count", 0))
        if asset_count <= 0:
            return 0
        start_count = max(1, min(int(getattr(self.cfg, "tabletop_asset_curriculum_start_count", 1)), asset_count))
        alpha = self._tabletop_asset_curriculum_alpha()
        return max(start_count, min(asset_count, start_count + int(torch.floor(torch.tensor((asset_count - start_count) * alpha)).item())))

    def _tabletop_current_motion_mode_count(self) -> int:
        mode_count = len(tuple(getattr(self.cfg, "tabletop_motion_modes", ("linear",))))
        if mode_count <= 0:
            return 1
        start_count = max(1, min(int(getattr(self.cfg, "tabletop_motion_mode_curriculum_start_count", 1)), mode_count))
        alpha = self._tabletop_motion_mode_curriculum_alpha()
        return max(start_count, min(mode_count, start_count + int(torch.floor(torch.tensor((mode_count - start_count) * alpha)).item())))

    def _sample_tabletop_asset_ids(self, count: int) -> torch.Tensor:
        current_count = max(1, self._tabletop_current_asset_count())
        weights = self._tabletop_asset_sampling_weights[:current_count]
        if bool(torch.all(weights == weights[0])):
            return torch.randint(0, current_count, (count,), dtype=torch.long, device=self.device)
        if float(weights.sum().item()) <= 0.0:
            raise ValueError("Active tabletop curriculum assets must include a positive sampling weight")
        return torch.multinomial(weights, count, replacement=True)

    def _sample_tabletop_motion_mode_ids(self, count: int) -> torch.Tensor:
        current_count = max(1, self._tabletop_current_motion_mode_count())
        return torch.randint(0, current_count, (count,), dtype=torch.long, device=self.device)

    def _set_tabletop_hover_targets(self, env_ids: torch.Tensor, object_pos_w: torch.Tensor) -> None:
        target = object_pos_w.clone()
        target[:, 2] = (
            self.scene.env_origins[env_ids, 2]
            + self._active_object_start_z[env_ids]
            + float(getattr(self.cfg, "tabletop_hover_height_delta", 0.16))
        )
        self._object_hover_target_pos_w[env_ids] = target
        self._object_hover_target_latched[env_ids] = False

    def _update_active_tabletop_asset_tensors(self, env_ids: torch.Tensor) -> None:
        if not self._uses_tabletop_asset_set():
            return
        asset_ids = self._tabletop_active_asset_ids[env_ids]
        self._active_object_shape_codes_tensor[env_ids] = self._tabletop_asset_shape_codes[asset_ids]
        self._active_object_size_tensor[env_ids] = self._tabletop_asset_sizes[asset_ids]
        self._active_object_radius_tensor[env_ids] = self._tabletop_asset_radii[asset_ids]
        self._active_object_height_tensor[env_ids] = self._tabletop_asset_heights[asset_ids]
        self._active_object_start_z[env_ids] = float(self.cfg.table_top_z) + self._tabletop_asset_support_heights[asset_ids] + 0.002
        self._active_asset_positive_fraction[env_ids] = self._tabletop_asset_positive_fractions[asset_ids]
        self._active_asset_negative_fraction[env_ids] = self._tabletop_asset_negative_fractions[asset_ids]
        shape_one_hot = torch.nn.functional.one_hot(
            self._active_object_shape_codes_tensor[env_ids].clamp(min=0, max=3), num_classes=4
        ).float()
        self._tabletop_asset_obs[env_ids] = torch.cat(
            (
                shape_one_hot,
                self._active_object_size_tensor[env_ids],
                self._active_object_radius_tensor[env_ids].unsqueeze(-1),
                self._active_asset_positive_fraction[env_ids].unsqueeze(-1),
                self._active_asset_negative_fraction[env_ids].unsqueeze(-1),
            ),
            dim=-1,
        )

    def _refresh_active_tabletop_object_state(self) -> None:
        env_index = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        asset_ids = self._tabletop_active_asset_ids
        self._object_pos_w = torch.stack([obj.data.root_pos_w for obj in self._tabletop_objects], dim=0)[asset_ids, env_index]
        self._object_quat_w = torch.stack([obj.data.root_quat_w for obj in self._tabletop_objects], dim=0)[asset_ids, env_index]
        self._object_lin_vel_w = torch.stack([obj.data.root_lin_vel_w for obj in self._tabletop_objects], dim=0)[
            asset_ids, env_index
        ]
        self._object_ang_vel_w = torch.stack([obj.data.root_ang_vel_w for obj in self._tabletop_objects], dim=0)[
            asset_ids, env_index
        ]

    def _clear_tabletop_arm_clearance(self) -> None:
        self._tabletop_arm_clearance_penalty.zero_()
        self._tabletop_arm_clearance_min_margin.zero_()
        self._tabletop_arm_clearance_active_fraction.zero_()
        self._tabletop_arm_clearance_ok.fill_(True)

    def _update_tabletop_arm_clearance(self) -> None:
        if self.cfg.task_family != "dynamic_tabletop_grasp":
            self._clear_tabletop_arm_clearance()
            return

        sample_pos_parts = []
        sample_margin_parts = []
        if self._tabletop_arm_clearance_body_ids:
            sample_pos_parts.append(self.robot.data.body_pos_w[:, self._tabletop_arm_clearance_body_ids])
            sample_margin_parts.append(self._tabletop_arm_clearance_body_margins.expand(self.num_envs, -1))

        if bool(getattr(self.cfg, "tabletop_arm_clearance_include_fingertip_points", True)) and hasattr(
            self, "_fingertip_pos_w"
        ):
            fingertip_margin = max(float(getattr(self.cfg, "tabletop_arm_clearance_fingertip_point_margin", 0.006)), 0.0)
            sample_pos_parts.append(self._fingertip_pos_w)
            sample_margin_parts.append(
                torch.full(
                    (self.num_envs, self._fingertip_pos_w.shape[1]),
                    fingertip_margin,
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        if bool(getattr(self.cfg, "tabletop_arm_clearance_include_palm_point", True)) and hasattr(self, "_palm_pos_w"):
            palm_margin = max(float(getattr(self.cfg, "tabletop_arm_clearance_palm_point_margin", 0.012)), 0.0)
            sample_pos_parts.append(self._palm_pos_w.unsqueeze(1))
            sample_margin_parts.append(
                torch.full((self.num_envs, 1), palm_margin, dtype=torch.float32, device=self.device)
            )

        if not sample_pos_parts:
            self._clear_tabletop_arm_clearance()
            return

        clearance_pos = torch.cat(sample_pos_parts, dim=1)
        clearance_margins = torch.cat(sample_margin_parts, dim=1)
        clearance_z = clearance_pos[:, :, 2]
        clearance_xy = clearance_pos[:, :, :2] - self.scene.env_origins[:, None, :2]
        active_xy = torch.all(
            torch.abs(clearance_xy - self._tabletop_arm_clearance_xy_center)
            <= (self._tabletop_arm_clearance_xy_half_extent + self._tabletop_arm_clearance_xy_padding),
            dim=-1,
        )
        min_z = self.scene.env_origins[:, 2:3] + float(self.cfg.table_top_z) + clearance_margins
        clearance_margin = clearance_z - min_z
        deficit = torch.where(active_xy, torch.relu(-clearance_margin), torch.zeros_like(clearance_margin))
        scale = max(float(getattr(self.cfg, "tabletop_arm_clearance_scale", 0.060)), 1.0e-6)
        max_penalty = float(getattr(self.cfg, "tabletop_arm_clearance_max_penalty", 2.0))
        self._tabletop_arm_clearance_penalty = torch.clamp(deficit / scale, 0.0, max_penalty).max(dim=-1).values
        masked_margin = torch.where(active_xy, clearance_margin, torch.full_like(clearance_margin, float("inf")))
        min_margin = masked_margin.min(dim=-1).values
        self._tabletop_arm_clearance_min_margin = torch.where(active_xy.any(dim=-1), min_margin, torch.zeros_like(min_margin))
        self._tabletop_arm_clearance_active_fraction = active_xy.float().mean(dim=-1)
        ok_threshold = max(float(getattr(self.cfg, "tabletop_arm_clearance_ok_penalty_threshold", 1.0e-4)), 0.0)
        self._tabletop_arm_clearance_ok = self._tabletop_arm_clearance_penalty <= ok_threshold

    def _compute_intermediate_values(self) -> None:
        if not self._uses_tabletop_asset_set():
            super()._compute_intermediate_values()
            self._update_tabletop_arm_clearance()
            self._compute_strict_contact_metrics_from_current_state()
            self._sync_falling_baton_affordance_markers()
            return

        self._refresh_active_tabletop_object_state()
        self._palm_body_pos_w = self.robot.data.body_pos_w[:, self._palm_body_id]
        self._palm_quat_w = self.robot.data.body_quat_w[:, self._palm_body_id]
        self._palm_lin_vel_w = self.robot.data.body_lin_vel_w[:, self._palm_body_id]
        self._palm_pos_w = self._palm_body_pos_w + quat_rotate(
            self._palm_quat_w,
            self._palm_body_offset.expand(self.num_envs, -1),
        )
        self._fingertip_body_pos_w = self.robot.data.body_pos_w[:, self._fingertip_body_ids]
        self._fingertip_quat_w = self.robot.data.body_quat_w[:, self._fingertip_body_ids]
        fingertip_offsets = self._fingertip_body_offsets.expand(self.num_envs, -1, -1)
        self._fingertip_pos_w = self._fingertip_body_pos_w + quat_rotate(
            self._fingertip_quat_w.reshape(-1, 4),
            fingertip_offsets.reshape(-1, 3),
        ).reshape(self.num_envs, len(self._fingertip_body_ids), 3)
        self._palm_min_z = torch.minimum(self._palm_min_z, self._palm_pos_w[:, 2])
        self._update_tabletop_arm_clearance()

        rel = self._fingertip_pos_w - self._object_pos_w.unsqueeze(1)
        self._surface_dist = self._object_surface_distance(rel)
        self._contact_score = torch.exp(-torch.relu(self._surface_dist) / self.cfg.contact_score_scale)
        contacts = self._surface_dist < self.cfg.contact_distance
        self._finger_contact_count = torch.sum(contacts.float(), dim=-1)
        self._thumb_contact = contacts[:, 0]
        non_thumb_contacts = contacts[:, 1:]
        self._non_thumb_contact_count = torch.sum(non_thumb_contacts, dim=-1)

        thumb_score = self._contact_score[:, 0]
        non_thumb_scores = self._contact_score[:, 1:]
        min_finger_contacts = max(float(getattr(self.cfg, "min_finger_contacts", 2)), 1.0)
        min_non_thumb_contacts = max(float(self.cfg.min_non_thumb_contacts), 1.0)
        self._finger_count_quality = torch.clamp(torch.sum(self._contact_score, dim=-1) / min_finger_contacts, 0.0, 1.0)
        self._non_thumb_quality = torch.clamp(torch.sum(non_thumb_scores, dim=-1) / min_non_thumb_contacts, 0.0, 1.0)

        thumb_vec = rel[:, 0]
        non_thumb_vec = rel[:, 1:]
        denom = torch.clamp(
            torch.norm(thumb_vec, dim=-1, keepdim=True) * torch.norm(non_thumb_vec, dim=-1),
            min=1.0e-6,
        )
        cos = torch.sum(thumb_vec.unsqueeze(1) * non_thumb_vec, dim=-1) / denom
        opposing = torch.relu(-cos) * non_thumb_contacts.float()
        self._opposition_score = torch.max(opposing, dim=-1).values
        threshold = float(self.cfg.opposition_cos_threshold)
        opposition_den = max(threshold + 1.0, 1.0e-6)
        opposition_progress = torch.clamp((threshold - cos) / opposition_den, 0.0, 1.0)
        self._weighted_opposition_score = torch.max(
            opposition_progress * non_thumb_scores * thumb_score.unsqueeze(-1),
            dim=-1,
        ).values
        self._opposing_contact = ((cos < threshold) & non_thumb_contacts & self._thumb_contact.unsqueeze(-1)).any(dim=-1)

        self._grasp_quality = torch.clamp(
            float(getattr(self.cfg, "grasp_quality_finger_count_weight", 0.30)) * self._finger_count_quality
            + float(getattr(self.cfg, "grasp_quality_non_thumb_weight", 0.25)) * self._non_thumb_quality
            + float(getattr(self.cfg, "grasp_quality_thumb_weight", 0.25)) * thumb_score
            + float(getattr(self.cfg, "grasp_quality_opposition_weight", 0.20)) * self._weighted_opposition_score,
            0.0,
            1.0,
        )

        palm_rel = self._palm_pos_w - self._object_pos_w
        self._palm_surface_dist = self._object_surface_distance(palm_rel)
        palm_contact_distance = max(float(getattr(self.cfg, "palm_contact_distance", 0.10)), 1.0e-6)
        self._palm_contact_score = torch.exp(-torch.relu(self._palm_surface_dist) / palm_contact_distance)
        self._palm_contact = self._palm_surface_dist < palm_contact_distance

        opposition_mode = str(getattr(self.cfg, "true_grasp_opposition_mode", "score")).lower()
        if opposition_mode in {"contact", "none", "off"}:
            opposition_gate = torch.ones_like(self._thumb_contact)
        elif opposition_mode == "dot":
            opposition_gate = self._opposing_contact
        else:
            opposition_gate = self._opposition_score > self.cfg.opposition_cos_threshold
        self._true_grasp = (
            self._thumb_contact
            & (self._non_thumb_contact_count >= self.cfg.min_non_thumb_contacts)
            & (self._finger_contact_count >= min_finger_contacts)
            & opposition_gate
        )
        strict_contact_distance = max(
            float(getattr(self.cfg, "strict_success_contact_distance", self.cfg.contact_distance)),
            1.0e-6,
        )
        strict_contacts = self._surface_dist < strict_contact_distance
        self._strict_thumb_contact = strict_contacts[:, 0]
        strict_non_thumb_contacts = strict_contacts[:, 1:]
        self._strict_finger_contact_count = torch.sum(strict_contacts.float(), dim=-1)
        self._strict_non_thumb_contact_count = torch.sum(strict_non_thumb_contacts.float(), dim=-1)
        strict_threshold = float(
            getattr(self.cfg, "strict_success_opposition_cos_threshold", self.cfg.opposition_cos_threshold)
        )
        self._strict_opposing_contact = (
            (cos < strict_threshold) & strict_non_thumb_contacts & self._strict_thumb_contact.unsqueeze(-1)
        ).any(dim=-1)
        strict_min_finger_contacts = max(
            float(getattr(self.cfg, "strict_success_min_finger_contacts", min_finger_contacts)),
            1.0,
        )
        strict_min_non_thumb_contacts = max(
            float(getattr(self.cfg, "strict_success_min_non_thumb_contacts", self.cfg.min_non_thumb_contacts)),
            1.0,
        )
        strict_opposition_mode = str(getattr(self.cfg, "strict_success_opposition_mode", "dot")).lower()
        if strict_opposition_mode in {"none", "off"}:
            strict_opposition_gate = torch.ones_like(self._strict_thumb_contact)
        elif strict_opposition_mode == "score":
            strict_opposition_gate = self._opposition_score > strict_threshold
        else:
            strict_opposition_gate = self._strict_opposing_contact
        self._strict_true_grasp = (
            self._strict_thumb_contact
            & (self._strict_non_thumb_contact_count >= strict_min_non_thumb_contacts)
            & (self._strict_finger_contact_count >= strict_min_finger_contacts)
            & strict_opposition_gate
        )
        strict_reward_score_scale = getattr(self.cfg, "strict_reward_contact_score_scale", None)
        if strict_reward_score_scale is None:
            strict_reward_score_scale = strict_contact_distance
        strict_reward_score_scale = max(float(strict_reward_score_scale), 1.0e-6)
        self._strict_contact_score = torch.exp(torch.neg(torch.relu(self._surface_dist) / strict_reward_score_scale))
        strict_approach_scale = max(float(getattr(self.cfg, "strict_approach_score_scale", 0.08)), 1.0e-6)
        strict_approach_scores = torch.exp(torch.neg(torch.relu(self._surface_dist) / strict_approach_scale))
        self._strict_approach_score = torch.max(strict_approach_scores, dim=-1).values
        self._strict_thumb_approach_score = strict_approach_scores[:, 0]
        strict_non_thumb_approach = strict_approach_scores[:, 1:]
        strict_pair_k = min(max(int(strict_min_non_thumb_contacts), 1), strict_non_thumb_approach.shape[-1])
        strict_non_thumb_pair = torch.topk(strict_non_thumb_approach, k=strict_pair_k, dim=-1).values.mean(dim=-1)
        self._strict_non_thumb_pair_approach_score = strict_non_thumb_pair
        self._strict_multifinger_approach_score = torch.clamp(
            0.25 * self._strict_approach_score
            + 0.25 * self._strict_thumb_approach_score
            + 0.25 * self._strict_non_thumb_pair_approach_score
            + 0.25 * torch.minimum(self._strict_thumb_approach_score, self._strict_non_thumb_pair_approach_score),
            0.0,
            1.0,
        )
        strict_touch_scale = max(float(getattr(self.cfg, "strict_touch_score_scale", 0.008)), 1.0e-6)
        strict_touch_gap = torch.relu(self._surface_dist - strict_contact_distance)
        strict_touch_scores = torch.exp(torch.neg(strict_touch_gap / strict_touch_scale))
        self._strict_thumb_touch_score = strict_touch_scores[:, 0]
        strict_non_thumb_touch = strict_touch_scores[:, 1:]
        strict_non_thumb_touch_pair = torch.topk(strict_non_thumb_touch, k=strict_pair_k, dim=-1).values.mean(dim=-1)
        self._strict_non_thumb_pair_touch_score = strict_non_thumb_touch_pair
        self._strict_multifinger_touch_score = torch.clamp(
            0.25 * torch.max(strict_touch_scores, dim=-1).values
            + 0.25 * self._strict_thumb_touch_score
            + 0.25 * self._strict_non_thumb_pair_touch_score
            + 0.25 * torch.minimum(self._strict_thumb_touch_score, self._strict_non_thumb_pair_touch_score),
            0.0,
            1.0,
        )
        self._strict_opposition_approach_score = torch.max(
            opposition_progress * strict_non_thumb_approach * self._strict_thumb_approach_score.unsqueeze(-1),
            dim=-1,
        ).values
        self._strict_opposition_touch_score = torch.max(
            opposition_progress * strict_non_thumb_touch * self._strict_thumb_touch_score.unsqueeze(-1),
            dim=-1,
        ).values
        strict_thumb_score = self._strict_contact_score[:, 0]
        strict_non_thumb_scores = self._strict_contact_score[:, 1:]
        self._strict_finger_count_quality = torch.clamp(
            torch.sum(self._strict_contact_score, dim=-1) / strict_min_finger_contacts,
            0.0,
            1.0,
        )
        self._strict_non_thumb_quality = torch.clamp(
            torch.sum(strict_non_thumb_scores, dim=-1) / strict_min_non_thumb_contacts,
            0.0,
            1.0,
        )
        self._strict_weighted_opposition_score = torch.max(
            opposition_progress * strict_non_thumb_scores * strict_thumb_score.unsqueeze(-1),
            dim=-1,
        ).values
        self._strict_grasp_quality = torch.clamp(
            float(getattr(self.cfg, "grasp_quality_finger_count_weight", 0.30)) * self._strict_finger_count_quality
            + float(getattr(self.cfg, "grasp_quality_non_thumb_weight", 0.25)) * self._strict_non_thumb_quality
            + float(getattr(self.cfg, "grasp_quality_thumb_weight", 0.25)) * strict_thumb_score
            + float(getattr(self.cfg, "grasp_quality_opposition_weight", 0.20))
            * self._strict_weighted_opposition_score,
            0.0,
            1.0,
        )
        self._grasp_seen = self._grasp_seen | self._true_grasp
        self._strict_grasp_seen = self._strict_grasp_seen | self._strict_true_grasp
        self._object_height_delta = self._object_pos_w[:, 2] - self._active_object_start_z
        self._lifted = self._object_height_delta > self.cfg.lift_success_height
        rel_vel = self._object_lin_vel_w - self._palm_lin_vel_w
        self._object_palm_rel_vel = torch.norm(rel_vel, dim=-1)
        self._stable_hold = self._lifted & self._true_grasp & (self._object_palm_rel_vel < self.cfg.stable_object_palm_vel)
        self._strict_stable_hold = (
            self._lifted & self._strict_true_grasp & (self._object_palm_rel_vel < self.cfg.stable_object_palm_vel)
        )
        palm_only_lift_dist = float(getattr(self.cfg, "palm_only_lift_dist", 0.12))
        self._scoop_lift = self._lifted & (~self._true_grasp)
        self._palm_only_lift = (
            self._lifted
            & (torch.norm(self._object_pos_w - self._palm_pos_w, dim=-1) < palm_only_lift_dist)
            & (
                (self._finger_contact_count < min_finger_contacts)
                | (self._non_thumb_contact_count < self.cfg.min_non_thumb_contacts)
                | (~self._thumb_contact)
            )
        )

    def _compute_strict_contact_metrics_from_current_state(self) -> None:
        rel = self._fingertip_pos_w - self._object_pos_w.unsqueeze(1)
        thumb_vec = rel[:, 0]
        non_thumb_vec = rel[:, 1:]
        denom = torch.clamp(
            torch.norm(thumb_vec, dim=-1, keepdim=True) * torch.norm(non_thumb_vec, dim=-1),
            min=1.0e-6,
        )
        cos = torch.sum(thumb_vec.unsqueeze(1) * non_thumb_vec, dim=-1) / denom

        strict_contact_distance = max(
            float(getattr(self.cfg, "strict_success_contact_distance", self.cfg.contact_distance)),
            1.0e-6,
        )
        strict_contacts = self._surface_dist < strict_contact_distance
        self._strict_thumb_contact = strict_contacts[:, 0]
        strict_non_thumb_contacts = strict_contacts[:, 1:]
        self._strict_finger_contact_count = torch.sum(strict_contacts.float(), dim=-1)
        self._strict_non_thumb_contact_count = torch.sum(strict_non_thumb_contacts.float(), dim=-1)

        strict_threshold = float(
            getattr(self.cfg, "strict_success_opposition_cos_threshold", self.cfg.opposition_cos_threshold)
        )
        self._strict_opposing_contact = (
            (cos < strict_threshold) & strict_non_thumb_contacts & self._strict_thumb_contact.unsqueeze(-1)
        ).any(dim=-1)
        min_finger_contacts = max(float(getattr(self.cfg, "min_finger_contacts", 2)), 1.0)
        strict_min_finger_contacts = max(
            float(getattr(self.cfg, "strict_success_min_finger_contacts", min_finger_contacts)),
            1.0,
        )
        strict_min_non_thumb_contacts = max(
            float(getattr(self.cfg, "strict_success_min_non_thumb_contacts", self.cfg.min_non_thumb_contacts)),
            1.0,
        )
        strict_opposition_mode = str(getattr(self.cfg, "strict_success_opposition_mode", "dot")).lower()
        if strict_opposition_mode in {"none", "off"}:
            strict_opposition_gate = torch.ones_like(self._strict_thumb_contact)
        elif strict_opposition_mode == "score":
            strict_opposition_gate = self._opposition_score > strict_threshold
        else:
            strict_opposition_gate = self._strict_opposing_contact
        self._strict_true_grasp = (
            self._strict_thumb_contact
            & (self._strict_non_thumb_contact_count >= strict_min_non_thumb_contacts)
            & (self._strict_finger_contact_count >= strict_min_finger_contacts)
            & strict_opposition_gate
        )

        strict_reward_score_scale = getattr(self.cfg, "strict_reward_contact_score_scale", None)
        if strict_reward_score_scale is None:
            strict_reward_score_scale = strict_contact_distance
        strict_reward_score_scale = max(float(strict_reward_score_scale), 1.0e-6)
        self._strict_contact_score = torch.exp(torch.neg(torch.relu(self._surface_dist) / strict_reward_score_scale))
        threshold = float(self.cfg.opposition_cos_threshold)
        opposition_den = max(threshold + 1.0, 1.0e-6)
        opposition_progress = torch.clamp((threshold - cos) / opposition_den, 0.0, 1.0)

        strict_approach_scale = max(float(getattr(self.cfg, "strict_approach_score_scale", 0.08)), 1.0e-6)
        strict_approach_scores = torch.exp(torch.neg(torch.relu(self._surface_dist) / strict_approach_scale))
        self._strict_approach_score = torch.max(strict_approach_scores, dim=-1).values
        self._strict_thumb_approach_score = strict_approach_scores[:, 0]
        strict_non_thumb_approach = strict_approach_scores[:, 1:]
        strict_pair_k = min(max(int(strict_min_non_thumb_contacts), 1), strict_non_thumb_approach.shape[-1])
        strict_non_thumb_pair = torch.topk(strict_non_thumb_approach, k=strict_pair_k, dim=-1).values.mean(dim=-1)
        self._strict_non_thumb_pair_approach_score = strict_non_thumb_pair
        self._strict_multifinger_approach_score = torch.clamp(
            0.25 * self._strict_approach_score
            + 0.25 * self._strict_thumb_approach_score
            + 0.25 * self._strict_non_thumb_pair_approach_score
            + 0.25 * torch.minimum(self._strict_thumb_approach_score, self._strict_non_thumb_pair_approach_score),
            0.0,
            1.0,
        )

        strict_touch_scale = max(float(getattr(self.cfg, "strict_touch_score_scale", 0.008)), 1.0e-6)
        strict_touch_gap = torch.relu(self._surface_dist - strict_contact_distance)
        strict_touch_scores = torch.exp(torch.neg(strict_touch_gap / strict_touch_scale))
        self._strict_thumb_touch_score = strict_touch_scores[:, 0]
        strict_non_thumb_touch = strict_touch_scores[:, 1:]
        strict_non_thumb_touch_pair = torch.topk(strict_non_thumb_touch, k=strict_pair_k, dim=-1).values.mean(dim=-1)
        self._strict_non_thumb_pair_touch_score = strict_non_thumb_touch_pair
        self._strict_multifinger_touch_score = torch.clamp(
            0.25 * torch.max(strict_touch_scores, dim=-1).values
            + 0.25 * self._strict_thumb_touch_score
            + 0.25 * self._strict_non_thumb_pair_touch_score
            + 0.25 * torch.minimum(self._strict_thumb_touch_score, self._strict_non_thumb_pair_touch_score),
            0.0,
            1.0,
        )
        self._strict_opposition_approach_score = torch.max(
            opposition_progress * strict_non_thumb_approach * self._strict_thumb_approach_score.unsqueeze(-1),
            dim=-1,
        ).values
        self._strict_opposition_touch_score = torch.max(
            opposition_progress * strict_non_thumb_touch * self._strict_thumb_touch_score.unsqueeze(-1),
            dim=-1,
        ).values

        strict_thumb_score = self._strict_contact_score[:, 0]
        strict_non_thumb_scores = self._strict_contact_score[:, 1:]
        self._strict_finger_count_quality = torch.clamp(
            torch.sum(self._strict_contact_score, dim=-1) / strict_min_finger_contacts,
            0.0,
            1.0,
        )
        self._strict_non_thumb_quality = torch.clamp(
            torch.sum(strict_non_thumb_scores, dim=-1) / strict_min_non_thumb_contacts,
            0.0,
            1.0,
        )
        self._strict_weighted_opposition_score = torch.max(
            opposition_progress * strict_non_thumb_scores * strict_thumb_score.unsqueeze(-1),
            dim=-1,
        ).values
        self._strict_grasp_quality = torch.clamp(
            float(getattr(self.cfg, "grasp_quality_finger_count_weight", 0.30)) * self._strict_finger_count_quality
            + float(getattr(self.cfg, "grasp_quality_non_thumb_weight", 0.25)) * self._strict_non_thumb_quality
            + float(getattr(self.cfg, "grasp_quality_thumb_weight", 0.25)) * strict_thumb_score
            + float(getattr(self.cfg, "grasp_quality_opposition_weight", 0.20))
            * self._strict_weighted_opposition_score,
            0.0,
            1.0,
        )
        self._strict_grasp_seen = self._strict_grasp_seen | self._strict_true_grasp
        if hasattr(self, "_lifted") and hasattr(self, "_object_palm_rel_vel"):
            self._strict_stable_hold = (
                self._lifted & self._strict_true_grasp & (self._object_palm_rel_vel < self.cfg.stable_object_palm_vel)
            )

    def _sync_falling_baton_affordance_markers(self) -> None:
        markers = getattr(self, "_falling_baton_affordance_markers", ())
        if not markers:
            return
        object_pos = self.object.data.root_pos_w
        object_quat = self.object.data.root_quat_w
        zero_vel = torch.zeros((self.num_envs, 6), dtype=object_pos.dtype, device=self.device)
        for _, marker_object, marker_offset in markers:
            local_offset = torch.tensor(marker_offset, dtype=object_pos.dtype, device=self.device).unsqueeze(0)
            marker_pos = object_pos + quat_rotate(object_quat, local_offset.expand(self.num_envs, -1))
            marker_pose = torch.cat((marker_pos, object_quat), dim=-1)
            marker_object.write_root_pose_to_sim(marker_pose)
            marker_object.write_root_velocity_to_sim(zero_vel)

    def _compute_tabletop_affordance_reward_terms(self) -> dict[str, torch.Tensor]:
        zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if not self._uses_tabletop_asset_set() or not bool(getattr(self.cfg, "tabletop_affordance_reward_enabled", False)):
            return {
                "pos_score": zeros,
                "neg_score": zeros,
                "pos_contact": zeros,
                "neg_contact": zeros,
                "pos_min_dist": zeros,
                "neg_min_dist": zeros,
                "thumb_pos_geom_score": zeros,
                "thumb_pos_score": zeros,
                "thumb_pos_contact": zeros,
            }

        asset_ids = self._tabletop_active_asset_ids
        pos_local = self._tabletop_affordance_pos_points_l[asset_ids]
        neg_local = self._tabletop_affordance_neg_points_l[asset_ids]
        point_count = pos_local.shape[1]
        quat = self._object_quat_w[:, None, :].expand(-1, point_count, -1).reshape(-1, 4)
        pos_world = self._object_pos_w[:, None, :] + quat_rotate(quat, pos_local.reshape(-1, 3)).reshape(
            self.num_envs, point_count, 3
        )
        neg_world = self._object_pos_w[:, None, :] + quat_rotate(quat, neg_local.reshape(-1, 3)).reshape(
            self.num_envs, point_count, 3
        )

        pos_dist = torch.cdist(self._fingertip_pos_w, pos_world).min(dim=-1).values
        neg_dist = torch.cdist(self._fingertip_pos_w, neg_world).min(dim=-1).values
        dist_scale = max(float(getattr(self.cfg, "tabletop_affordance_distance_scale", 0.035)), 1.0e-6)
        contact_dist = max(float(getattr(self.cfg, "tabletop_affordance_contact_distance", 0.045)), 1.0e-6)
        contact_weight = torch.clamp(self._contact_score, 0.0, 1.0)
        pos_tip_score = torch.exp(-pos_dist / dist_scale) * contact_weight
        neg_tip_score = torch.exp(-neg_dist / dist_scale) * contact_weight
        pos_valid = self._tabletop_affordance_pos_valid[asset_ids].float()
        neg_valid = self._tabletop_affordance_neg_valid[asset_ids].float()
        return {
            "pos_score": pos_tip_score.max(dim=-1).values * pos_valid,
            "neg_score": neg_tip_score.max(dim=-1).values * neg_valid,
            "pos_contact": ((pos_dist < contact_dist) & (contact_weight > 0.30)).float().max(dim=-1).values * pos_valid,
            "neg_contact": ((neg_dist < contact_dist) & (contact_weight > 0.30)).float().max(dim=-1).values * neg_valid,
            "pos_min_dist": pos_dist.min(dim=-1).values * pos_valid,
            "neg_min_dist": neg_dist.min(dim=-1).values * neg_valid,
        }

    def _compute_falling_baton_affordance_reward_terms(self) -> dict[str, torch.Tensor]:
        zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if self.cfg.task_family != "falling_baton_grasp" or not bool(
            getattr(self.cfg, "falling_affordance_reward_enabled", False)
        ):
            return {
                "pos_score": zeros,
                "neg_score": zeros,
                "pos_contact": zeros,
                "neg_contact": zeros,
                "pos_min_dist": zeros,
                "neg_min_dist": zeros,
                "thumb_pos_geom_score": zeros,
                "thumb_pos_score": zeros,
                "thumb_pos_contact": zeros,
            }

        quat = self._object_quat_w[:, None, :].expand(-1, self._fingertip_pos_w.shape[1], -1).reshape(-1, 4)
        rel = self._fingertip_pos_w - self._object_pos_w[:, None, :]
        tip_local = quat_rotate_inverse(quat, rel.reshape(-1, 3)).reshape_as(rel)
        radial = torch.norm(tip_local[..., :2], dim=-1)

        def segment_distance(center_z: float, half_z: float) -> torch.Tensor:
            axial = torch.relu(torch.abs(tip_local[..., 2] - center_z) - half_z)
            radial_out = torch.relu(radial - float(getattr(self.cfg, "falling_affordance_radial_margin", 0.020)))
            return torch.sqrt(axial * axial + radial_out * radial_out + 1.0e-8)

        pos_dist = segment_distance(
            float(getattr(self.cfg, "falling_affordance_positive_center_z", -0.05)),
            float(getattr(self.cfg, "falling_affordance_positive_half_z", 0.03)),
        )
        neg_dist = segment_distance(
            float(getattr(self.cfg, "falling_affordance_negative_center_z", 0.05)),
            float(getattr(self.cfg, "falling_affordance_negative_half_z", 0.035)),
        )
        dist_scale = max(float(getattr(self.cfg, "falling_affordance_distance_scale", 0.025)), 1.0e-6)
        contact_dist = max(float(getattr(self.cfg, "falling_affordance_contact_distance", 0.040)), 1.0e-6)
        contact_weight = torch.clamp(self._contact_score, 0.0, 1.0)
        pos_geom_score = torch.exp(-pos_dist / dist_scale)
        neg_geom_score = torch.exp(-neg_dist / dist_scale)
        pos_tip_score = pos_geom_score * contact_weight
        neg_tip_score = neg_geom_score * contact_weight
        thumb_pos_contact = ((pos_dist[:, 0] < contact_dist) & (contact_weight[:, 0] > 0.30)).float()
        return {
            "pos_score": pos_tip_score.max(dim=-1).values,
            "neg_score": neg_tip_score.max(dim=-1).values,
            "pos_contact": ((pos_dist < contact_dist) & (contact_weight > 0.30)).float().max(dim=-1).values,
            "neg_contact": ((neg_dist < contact_dist) & (contact_weight > 0.30)).float().max(dim=-1).values,
            "pos_min_dist": pos_dist.min(dim=-1).values,
            "neg_min_dist": neg_dist.min(dim=-1).values,
            "thumb_pos_geom_score": pos_geom_score[:, 0],
            "thumb_pos_score": pos_tip_score[:, 0],
            "thumb_pos_contact": thumb_pos_contact,
        }

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        if self.cfg.task_family == "dynamic_tabletop_grasp":
            if self._uses_tabletop_asset_set():
                self._park_inactive_tabletop_asset_objects()
            self._apply_tabletop_persistent_motion()

    def _apply_action(self) -> None:
        if self.cfg.policy_action_interface == "joint_target":
            self._apply_joint_target_action()
            return
        self._apply_isaaclab_direct_action()

    def _apply_joint_target_action(self) -> None:
        if self._uses_active_hand_actions():
            self._apply_active_joint_target_action()
            return

        control_ids = self._controlled_joint_ids
        lower = self._joint_lower_limits[control_ids].unsqueeze(0)
        upper = self._joint_upper_limits[control_ids].unsqueeze(0)
        center = torch.clamp(self._default_joint_pos[:, control_ids], lower, upper)

        positive_span = upper - center
        negative_span = center - lower
        raw_targets = torch.where(
            self.actions >= 0.0,
            center + self.actions * positive_span,
            center + self.actions * negative_span,
        )

        current_targets = self._joint_targets[:, control_ids]
        arm_dim = len(self._arm_joint_ids)
        smoothed = current_targets.clone()
        smoothed[:, :arm_dim] = (
            self.cfg.arm_moving_average * raw_targets[:, :arm_dim]
            + (1.0 - self.cfg.arm_moving_average) * current_targets[:, :arm_dim]
        )
        smoothed[:, arm_dim:] = (
            self.cfg.hand_moving_average * raw_targets[:, arm_dim:]
            + (1.0 - self.cfg.hand_moving_average) * current_targets[:, arm_dim:]
        )
        smoothed[:, :arm_dim] = self._rate_limit_joint_target_delta(
            smoothed[:, :arm_dim],
            current_targets[:, :arm_dim],
            float(getattr(self.cfg, "joint_target_arm_max_delta", 0.0)),
        )
        smoothed[:, arm_dim:] = self._rate_limit_joint_target_delta(
            smoothed[:, arm_dim:],
            current_targets[:, arm_dim:],
            float(getattr(self.cfg, "joint_target_hand_max_delta", 0.0)),
        )
        smoothed = torch.clamp(smoothed, lower, upper)

        self._prev_joint_targets[:] = self._joint_targets
        self._joint_targets[:, control_ids] = smoothed
        self._apply_initial_target_locks()
        self._apply_post_success_target_locks()
        self.robot.set_joint_position_target(self._joint_targets[:, control_ids], joint_ids=control_ids)

    def _rate_limit_joint_target_delta(
        self,
        proposed_targets: torch.Tensor,
        current_targets: torch.Tensor,
        max_delta: float,
    ) -> torch.Tensor:
        if max_delta <= 0.0:
            return proposed_targets
        delta = torch.clamp(proposed_targets - current_targets, -max_delta, max_delta)
        limited_targets = current_targets + delta
        if bool(getattr(self.cfg, "joint_target_rate_limit_requires_lift_baseline", False)):
            active = self._tabletop_arm_lift_baseline_latched.unsqueeze(-1)
            return torch.where(active, limited_targets, proposed_targets)
        return limited_targets

    def _compute_scripted_action_prior(self) -> torch.Tensor:
        action_prior = super()._compute_scripted_action_prior()
        if self.cfg.task_family != "dynamic_tabletop_grasp":
            return action_prior
        default_strict_grasp_prior = bool(
            getattr(self.cfg, "strict_reward_enabled", False) or getattr(self.cfg, "strict_success_enabled", False)
        )
        use_strict_grasp_prior = bool(
            getattr(self.cfg, "scripted_action_prior_uses_strict_grasp", default_strict_grasp_prior)
        )
        default_grasp = getattr(
            self, "_true_grasp", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        prior_true_grasp = getattr(self, "_strict_true_grasp", default_grasp) if use_strict_grasp_prior else default_grasp
        prior_grasp_seen = self._strict_grasp_seen if use_strict_grasp_prior else self._grasp_seen
        prior_grasp_streak = (
            self._tabletop_strict_true_grasp_streak
            if use_strict_grasp_prior
            else self._tabletop_true_grasp_streak
        )
        lift_recent_memory_cache: dict[int, torch.Tensor] = {}

        def lift_grasp_memory(min_memory_steps: int) -> torch.Tensor:
            min_memory_steps = int(min_memory_steps)
            requires_current_streak = bool(
                getattr(self.cfg, "scripted_action_prior_lift_memory_requires_streak", False)
            )
            if requires_current_streak and min_memory_steps > 0 and hasattr(self, "_tabletop_true_grasp_streak"):
                base_memory = prior_grasp_streak >= min_memory_steps
            elif min_memory_steps > 0 and hasattr(self, "_tabletop_true_grasp_streak"):
                base_memory = prior_grasp_seen | (prior_grasp_streak >= min_memory_steps)
            else:
                base_memory = prior_grasp_seen

            recent_steps = max(int(getattr(self.cfg, "scripted_action_prior_lift_grasp_recent_steps", 0)), 0)
            if recent_steps <= 0:
                return base_memory
            if min_memory_steps in lift_recent_memory_cache:
                recent_memory = lift_recent_memory_cache[min_memory_steps]
            else:
                if min_memory_steps > 0 and hasattr(self, "_tabletop_true_grasp_streak"):
                    recent_arm = prior_grasp_streak >= min_memory_steps
                else:
                    recent_arm = prior_true_grasp
                recent_value = torch.full_like(self._scripted_lift_grasp_recent_steps, recent_steps)
                self._scripted_lift_grasp_recent_steps[:] = torch.where(
                    recent_arm,
                    recent_value,
                    torch.clamp(self._scripted_lift_grasp_recent_steps - 1, min=0),
                )
                recent_memory = self._scripted_lift_grasp_recent_steps > 0
                lift_recent_memory_cache[min_memory_steps] = recent_memory
            if requires_current_streak:
                return recent_memory
            return base_memory | recent_memory

        if bool(getattr(self.cfg, "scripted_tabletop_approach_action_prior_enabled", False)):
            approach_start_step = int(getattr(self.cfg, "scripted_tabletop_approach_action_prior_start_step", 0))
            approach_steps = int(getattr(self.cfg, "scripted_tabletop_approach_action_prior_steps", 0))
            approach_stop_step = approach_start_step + max(approach_steps, 0)
            approach_unlocked = (self.episode_length_buf >= approach_start_step) & (
                (approach_steps <= 0) | (self.episode_length_buf < approach_stop_step)
            )
            arm_has_prior = torch.any(torch.abs(action_prior[:, :7]) > 1.0e-6, dim=-1)
            approach_write_mask = approach_unlocked & (~arm_has_prior)
            if torch.any(approach_write_mask):
                approach_action = torch.tensor(
                    getattr(self.cfg, "scripted_tabletop_approach_action_prior", (0.0,) * len(self._arm_joint_ids)),
                    dtype=torch.float32,
                    device=self.device,
                ).view(1, -1)
                if approach_action.shape[-1] != len(self._arm_joint_ids):
                    raise ValueError(
                        "scripted_tabletop_approach_action_prior must contain "
                        f"{len(self._arm_joint_ids)} values, got {approach_action.shape[-1]}"
                    )
                approach_ramp_steps = max(
                    int(getattr(self.cfg, "scripted_tabletop_approach_action_prior_ramp_steps", 1)), 1
                )
                approach_age = self.episode_length_buf[approach_write_mask] - approach_start_step
                approach_alpha = torch.clamp(
                    (approach_age + 1).float() / float(approach_ramp_steps), 0.0, 1.0
                ).unsqueeze(-1)
                action_prior[approach_write_mask, :7] = approach_alpha * approach_action.expand_as(
                    action_prior[:, :7]
                )[approach_write_mask]

        if not bool(getattr(self.cfg, "scripted_tabletop_pregrasp_prior_enabled", False)):
            if bool(getattr(self.cfg, "scripted_tabletop_relative_lift_target_prior_enabled", False)):
                lift_start_step = int(getattr(self.cfg, "scripted_action_prior_lift_start_step", 0))
                lift_steps = int(getattr(self.cfg, "scripted_action_prior_lift_steps", 0))
                lift_stop_step = lift_start_step + max(lift_steps, 0)
                lift_unlocked = (self.episode_length_buf >= lift_start_step) & (
                    (lift_steps <= 0) | (self.episode_length_buf < lift_stop_step)
                )
                if bool(getattr(self.cfg, "scripted_action_prior_lift_requires_grasp", False)):
                    min_memory_steps = int(getattr(self.cfg, "scripted_action_prior_lift_grasp_memory_min_steps", 0))
                    lift_unlocked = lift_unlocked & lift_grasp_memory(min_memory_steps)
                if torch.any(lift_unlocked):
                    arm_lower = self._joint_lower_limits[self._arm_joint_ids].unsqueeze(0)
                    arm_upper = self._joint_upper_limits[self._arm_joint_ids].unsqueeze(0)
                    current_arm_targets = self._joint_targets[:, self._arm_joint_ids]
                    new_latch = lift_unlocked & (~self._scripted_relative_lift_target_latched)
                    if torch.any(new_latch):
                        self._scripted_relative_lift_target_arm_pos[new_latch] = torch.clamp(
                            current_arm_targets[new_latch] + self._scripted_relative_lift_target_delta[new_latch],
                            arm_lower,
                            arm_upper,
                        )
                        self._scripted_relative_lift_target_latched[new_latch] = True

                    target_arm_pos = torch.clamp(self._scripted_relative_lift_target_arm_pos, arm_lower, arm_upper)
                    arm_center = torch.clamp(self._default_joint_pos[:, self._arm_joint_ids], arm_lower, arm_upper)
                    positive_span = torch.clamp(arm_upper - arm_center, min=1.0e-6)
                    negative_span = torch.clamp(arm_center - arm_lower, min=1.0e-6)
                    remaining_steps = torch.clamp(lift_stop_step - self.episode_length_buf, min=1).float().unsqueeze(-1)
                    if self.cfg.policy_action_interface == "joint_target":
                        desired_next = current_arm_targets + (target_arm_pos - current_arm_targets) / remaining_steps
                        arm_ma = max(float(self.cfg.arm_moving_average), 1.0e-6)
                        raw_target = (desired_next - (1.0 - arm_ma) * current_arm_targets) / arm_ma
                        raw_target = torch.clamp(raw_target, arm_lower, arm_upper)
                        delta = raw_target - arm_center
                        target_action = torch.where(delta >= 0.0, delta / positive_span, delta / negative_span)
                    else:
                        effective_gain = max(
                            float(self.cfg.arm_action_scale) * float(self.cfg.arm_moving_average) * self.dt,
                            1.0e-6,
                        )
                        target_action = (target_arm_pos - current_arm_targets) / (effective_gain * remaining_steps)
                    action_prior[lift_unlocked, :7] = torch.clamp(target_action, -1.0, 1.0)[lift_unlocked]
            if bool(getattr(self.cfg, "scripted_action_prior_lift_uses_grasp_memory", False)):
                lift_start_step = int(getattr(self.cfg, "scripted_action_prior_lift_start_step", 0))
                lift_steps = int(getattr(self.cfg, "scripted_action_prior_lift_steps", 0))
                lift_stop_step = lift_start_step + max(lift_steps, 0)
                lift_unlocked = (self.episode_length_buf >= lift_start_step) & (
                    (lift_steps <= 0) | (self.episode_length_buf < lift_stop_step)
                )
                min_memory_steps = int(getattr(self.cfg, "scripted_action_prior_lift_grasp_memory_min_steps", 0))
                lift_unlocked = lift_unlocked & lift_grasp_memory(min_memory_steps)
                arm_has_prior = torch.any(torch.abs(action_prior[:, :7]) > 1.0e-6, dim=-1)
                lift_write_mask = lift_unlocked & (~arm_has_prior)
                if torch.any(lift_write_mask):
                    if self._scripted_lift_action.shape[0] == self.num_envs:
                        action_prior[lift_write_mask, :7] = self._scripted_lift_action[lift_write_mask]
                    else:
                        action_prior[lift_write_mask, :7] = self._scripted_lift_action
            if bool(getattr(self.cfg, "scripted_tabletop_hand_grasp_memory_prior_enabled", False)):
                hand_start_step = int(getattr(self.cfg, "scripted_tabletop_hand_grasp_memory_prior_start_step", 0))
                hand_steps = int(getattr(self.cfg, "scripted_tabletop_hand_grasp_memory_prior_steps", 0))
                hand_stop_step = hand_start_step + max(hand_steps, 0)
                hand_unlocked = (self.episode_length_buf >= hand_start_step) & (
                    (hand_steps <= 0) | (self.episode_length_buf < hand_stop_step)
                )
                min_memory_steps = int(
                    getattr(self.cfg, "scripted_tabletop_hand_grasp_memory_min_steps", 0)
                )
                if min_memory_steps > 0 and hasattr(self, "_tabletop_true_grasp_streak"):
                    grasp_memory = prior_grasp_seen | (prior_grasp_streak >= min_memory_steps)
                else:
                    grasp_memory = prior_grasp_seen
                hand_unlocked = hand_unlocked & grasp_memory
                if torch.any(hand_unlocked):
                    hand_dim = action_prior.shape[-1] - len(self._arm_joint_ids)
                    hand_action_tensor = getattr(
                        self, "_scripted_tabletop_hand_grasp_memory_action", None
                    )
                    if torch.is_tensor(hand_action_tensor):
                        hand_action = hand_action_tensor[hand_unlocked]
                    else:
                        hand_action_vector = getattr(
                            self.cfg, "scripted_tabletop_hand_grasp_memory_action_vector", None
                        )
                        if hand_action_vector is None:
                            hand_action = torch.full(
                                (int(hand_unlocked.sum().item()), hand_dim),
                                float(getattr(self.cfg, "scripted_tabletop_hand_grasp_memory_action", 1.0)),
                                dtype=torch.float32,
                                device=self.device,
                            )
                        else:
                            hand_action = torch.tensor(
                                hand_action_vector, dtype=torch.float32, device=self.device
                            ).view(1, -1)
                            if hand_action.shape[-1] != hand_dim:
                                raise ValueError(
                                    "scripted_tabletop_hand_grasp_memory_action_vector must contain "
                                    f"{hand_dim} values, got {hand_action.shape[-1]}"
                                )
                            hand_action = hand_action.expand(int(hand_unlocked.sum().item()), -1)
                    hand_ramp_steps = int(
                        getattr(self.cfg, "scripted_tabletop_hand_grasp_memory_ramp_steps", 0)
                    )
                    if hand_ramp_steps > 0:
                        hand_age = self.episode_length_buf[hand_unlocked] - hand_start_step
                        hand_alpha = torch.clamp(
                            (hand_age + 1).float() / float(hand_ramp_steps), 0.0, 1.0
                        ).unsqueeze(-1)
                        action_prior[hand_unlocked, 7:] = -1.0 + hand_alpha * (hand_action + 1.0)
                    else:
                        action_prior[hand_unlocked, 7:] = hand_action
            return action_prior

        start_step = int(getattr(self.cfg, "scripted_tabletop_pregrasp_prior_start_step", 0))
        ramp_steps = max(int(getattr(self.cfg, "scripted_tabletop_pregrasp_prior_ramp_steps", 1)), 1)
        pregrasp_unlocked = self.episode_length_buf >= start_step
        if not torch.any(pregrasp_unlocked):
            return action_prior

        desired_arm_pos = torch.tensor(
            getattr(self.cfg, "scripted_tabletop_pregrasp_arm_pos", self.cfg.default_arm_pos),
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)
        if desired_arm_pos.shape[-1] != len(self._arm_joint_ids):
            raise ValueError(
                "scripted_tabletop_pregrasp_arm_pos must contain "
                f"{len(self._arm_joint_ids)} values, got {desired_arm_pos.shape[-1]}"
            )

        arm_lower = self._joint_lower_limits[self._arm_joint_ids].unsqueeze(0)
        arm_upper = self._joint_upper_limits[self._arm_joint_ids].unsqueeze(0)
        arm_center = torch.clamp(self._default_joint_pos[:, self._arm_joint_ids], arm_lower, arm_upper)
        positive_span = torch.clamp(arm_upper - arm_center, min=1.0e-6)
        negative_span = torch.clamp(arm_center - arm_lower, min=1.0e-6)
        prior_control_mode = str(getattr(self.cfg, "scripted_tabletop_pregrasp_prior_control_mode", "normalized_action"))

        def arm_pos_to_action(target_arm_pos: torch.Tensor) -> torch.Tensor:
            target_arm_pos = torch.clamp(target_arm_pos.expand_as(arm_center), arm_lower, arm_upper)
            delta = target_arm_pos - arm_center
            target_action = torch.where(delta >= 0.0, delta / positive_span, delta / negative_span)
            return torch.clamp(target_action, -1.0, 1.0)

        def arm_target_track_action(target_arm_pos: torch.Tensor, stop_step: int) -> torch.Tensor:
            target_arm_pos = torch.clamp(target_arm_pos.expand_as(arm_center), arm_lower, arm_upper)
            current_arm_targets = self._joint_targets[:, self._arm_joint_ids]
            remaining_steps = torch.clamp(stop_step - self.episode_length_buf, min=1).float().unsqueeze(-1)
            if self.cfg.policy_action_interface == "joint_target":
                desired_next = current_arm_targets + (target_arm_pos - current_arm_targets) / remaining_steps
                arm_ma = max(float(self.cfg.arm_moving_average), 1.0e-6)
                raw_target = (desired_next - (1.0 - arm_ma) * current_arm_targets) / arm_ma
                raw_target = torch.clamp(raw_target, arm_lower, arm_upper)
                delta = raw_target - arm_center
                target_action = torch.where(delta >= 0.0, delta / positive_span, delta / negative_span)
            else:
                effective_gain = max(
                    float(self.cfg.arm_action_scale) * float(self.cfg.arm_moving_average) * self.dt,
                    1.0e-6,
                )
                target_action = (target_arm_pos - current_arm_targets) / (effective_gain * remaining_steps)
            return torch.clamp(target_action, -1.0, 1.0)

        pregrasp_stop_step = start_step + max(
            int(getattr(self.cfg, "scripted_tabletop_pregrasp_prior_steps", 0)), 0
        )
        if prior_control_mode == "target_track" and pregrasp_stop_step > start_step:
            pregrasp_action = arm_target_track_action(desired_arm_pos, pregrasp_stop_step)
            pregrasp_unlocked = pregrasp_unlocked & (self.episode_length_buf < pregrasp_stop_step)
        else:
            pregrasp_action = arm_pos_to_action(desired_arm_pos)

        if bool(getattr(self.cfg, "scripted_tabletop_lift_target_prior_enabled", False)):
            lift_start_step = int(getattr(self.cfg, "scripted_action_prior_lift_start_step", 0))
            lift_steps = int(getattr(self.cfg, "scripted_action_prior_lift_steps", 0))
            lift_stop_step = lift_start_step + max(lift_steps, 0)
            lift_unlocked = (self.episode_length_buf >= lift_start_step) & (
                (lift_steps <= 0) | (self.episode_length_buf < lift_stop_step)
            )
            if bool(getattr(self.cfg, "scripted_action_prior_lift_requires_grasp", False)):
                grasp_unlock = prior_true_grasp
                if bool(getattr(self.cfg, "scripted_action_prior_lift_uses_proximity", False)):
                    proximity_dist = float(getattr(self.cfg, "scripted_action_prior_lift_proximity_distance", 0.0))
                    proximity_unlock = torch.zeros_like(grasp_unlock)
                    if proximity_dist > 0.0 and hasattr(self, "_surface_dist"):
                        proximity_unlock = proximity_unlock | (
                            self._surface_dist.min(dim=-1).values < proximity_dist
                        )
                    min_contacts = float(getattr(self.cfg, "scripted_action_prior_lift_proximity_min_contacts", 0.0))
                    if min_contacts > 0.0 and hasattr(self, "_finger_contact_count"):
                        proximity_unlock = proximity_unlock | (self._finger_contact_count >= min_contacts)
                    grasp_unlock = grasp_unlock | proximity_unlock
                if bool(getattr(self.cfg, "scripted_action_prior_lift_uses_grasp_memory", False)):
                    min_memory_steps = int(
                        getattr(self.cfg, "scripted_action_prior_lift_grasp_memory_min_steps", 0)
                    )
                    grasp_unlock = grasp_unlock | lift_grasp_memory(min_memory_steps)
                lift_unlocked = lift_unlocked & grasp_unlock
            if torch.any(lift_unlocked):
                lift_target_pos_cfg = getattr(self.cfg, "scripted_tabletop_lift_target_arm_pos", None)
                if lift_target_pos_cfg is not None:
                    lift_target_pos = torch.tensor(lift_target_pos_cfg, dtype=torch.float32, device=self.device).view(1, -1)
                else:
                    lift_delta = torch.tensor(
                        getattr(self.cfg, "scripted_tabletop_lift_target_arm_delta", (0.0,) * len(self._arm_joint_ids)),
                        dtype=torch.float32,
                        device=self.device,
                    ).view(1, -1)
                    lift_target_pos = desired_arm_pos + lift_delta
                if lift_target_pos.shape[-1] != len(self._arm_joint_ids):
                    raise ValueError(
                        "scripted_tabletop_lift_target arm target must contain "
                        f"{len(self._arm_joint_ids)} values, got {lift_target_pos.shape[-1]}"
                    )
                if prior_control_mode == "target_track" and lift_stop_step > lift_start_step:
                    lift_action = arm_target_track_action(lift_target_pos, lift_stop_step)
                    action_prior[lift_unlocked, :7] = lift_action[lift_unlocked]
                else:
                    lift_action = arm_pos_to_action(lift_target_pos)
                    lift_ramp_steps = max(
                        int(getattr(self.cfg, "scripted_tabletop_lift_target_prior_ramp_steps", 1)), 1
                    )
                    lift_age = self.episode_length_buf[lift_unlocked] - lift_start_step
                    lift_alpha = torch.clamp((lift_age + 1).float() / float(lift_ramp_steps), 0.0, 1.0).unsqueeze(-1)
                    action_prior[lift_unlocked, :7] = (
                        pregrasp_action[lift_unlocked]
                        + lift_alpha * (lift_action[lift_unlocked] - pregrasp_action[lift_unlocked])
                    )

        arm_has_lift_prior = torch.any(torch.abs(action_prior[:, :7]) > 1.0e-6, dim=-1)
        write_mask = pregrasp_unlocked & (~arm_has_lift_prior)
        if torch.any(write_mask):
            age = self.episode_length_buf[write_mask] - start_step
            alpha = torch.clamp((age + 1).float() / float(ramp_steps), 0.0, 1.0).unsqueeze(-1)
            action_prior[write_mask, :7] = alpha * pregrasp_action[write_mask]
        return action_prior

    def _apply_active_joint_target_action(self) -> None:
        arm_actions = self.actions[:, :7]
        hand_actions = self.actions[:, 7:]

        arm_lower = self._joint_lower_limits[self._arm_joint_ids].unsqueeze(0)
        arm_upper = self._joint_upper_limits[self._arm_joint_ids].unsqueeze(0)
        arm_center = torch.clamp(self._default_joint_pos[:, self._arm_joint_ids], arm_lower, arm_upper)
        positive_span = arm_upper - arm_center
        negative_span = arm_center - arm_lower
        raw_arm_targets = torch.where(
            arm_actions >= 0.0,
            arm_center + arm_actions * positive_span,
            arm_center + arm_actions * negative_span,
        )
        current_arm_targets = self._joint_targets[:, self._arm_joint_ids]
        arm_targets = (
            self.cfg.arm_moving_average * raw_arm_targets
            + (1.0 - self.cfg.arm_moving_average) * current_arm_targets
        )

        raw_hand_targets = self._active_hand_actions_to_sim_targets(hand_actions)
        current_hand_targets = self._joint_targets[:, self._control_hand_joint_ids]
        hand_targets = (
            self.cfg.hand_moving_average * raw_hand_targets
            + (1.0 - self.cfg.hand_moving_average) * current_hand_targets
        )
        arm_targets = self._rate_limit_joint_target_delta(
            arm_targets,
            current_arm_targets,
            float(getattr(self.cfg, "joint_target_arm_max_delta", 0.0)),
        )
        hand_targets = self._rate_limit_joint_target_delta(
            hand_targets,
            current_hand_targets,
            float(getattr(self.cfg, "joint_target_hand_max_delta", 0.0)),
        )

        self._prev_joint_targets[:] = self._joint_targets
        self._joint_targets[:, self._arm_joint_ids] = torch.clamp(arm_targets, arm_lower, arm_upper)
        hand_lower = self._joint_lower_limits[self._control_hand_joint_ids].unsqueeze(0)
        hand_upper = self._joint_upper_limits[self._control_hand_joint_ids].unsqueeze(0)
        self._joint_targets[:, self._control_hand_joint_ids] = torch.clamp(hand_targets, hand_lower, hand_upper)
        self._apply_initial_target_locks()
        self._apply_post_success_target_locks()
        self.robot.set_joint_position_target(
            self._joint_targets[:, self._controlled_joint_ids], joint_ids=self._controlled_joint_ids
        )

    def _active_hand_actions_to_sim_targets(self, hand_actions: torch.Tensor) -> torch.Tensor:
        if self._uses_revo2_active_hand_actions():
            return self._revo2_active_hand_actions_to_sim_targets(hand_actions)
        if self._uses_inspire_active_hand_actions():
            return self._inspire_active_hand_actions_to_sim_targets(hand_actions)
        raise RuntimeError(f"Unsupported active hand action contract: {self.cfg.action_contract}")

    def _revo2_active_hand_actions_to_sim_targets(self, hand_actions: torch.Tensor) -> torch.Tensor:
        hand_fractions = 0.5 * (hand_actions + 1.0) * self._reference_hand_fractions
        hand_fractions = torch.clamp(hand_fractions, 0.0, 1.0)
        return self._semantic_hand_fractions_to_joint_targets(
            hand_fractions,
            joint_ids=self._control_hand_joint_ids,
            joint_names=self._sim_hand_joint_names,
        )

    def _inspire_active_hand_actions_to_sim_targets(self, hand_actions: torch.Tensor) -> torch.Tensor:
        fractions = 0.5 * (hand_actions + 1.0) * self._reference_hand_fractions
        fractions = torch.clamp(fractions, 0.0, 1.0)
        targets = self._default_joint_pos[:, self._control_hand_joint_ids].clone()
        lower = self._joint_lower_limits
        upper = self._joint_upper_limits
        joint_index = {name: i for i, name in enumerate(self.robot.joint_names)}
        local_index = {name: i for i, name in enumerate(self._sim_hand_joint_names)}

        def write_joint(joint_name: str, fraction: torch.Tensor) -> None:
            if joint_name not in local_index:
                return
            local_id = local_index[joint_name]
            jid = joint_index[joint_name]
            open_target = torch.clamp(self._default_joint_pos[:, jid], lower[jid], upper[jid])
            if self._inspire_semantic_close_targets is None:
                close_target = upper[jid].expand_as(open_target)
            else:
                close_target = self._inspire_semantic_close_targets[local_id].expand_as(open_target)
            targets[:, local_id] = open_target + fraction * (close_target - open_target)

        write_joint("thumb_proximal_yaw_joint", fractions[:, 0])
        for joint_name in ("thumb_proximal_pitch_joint", "thumb_intermediate_joint", "thumb_distal_joint"):
            write_joint(joint_name, fractions[:, 1])

        for fraction_index, finger_name in enumerate(("index", "middle", "ring", "pinky"), start=2):
            write_joint(f"{finger_name}_proximal_joint", fractions[:, fraction_index])
            write_joint(f"{finger_name}_intermediate_joint", fractions[:, fraction_index])
        return targets

    def _apply_isaaclab_direct_action(self) -> None:
        arm_actions = self.actions[:, :7]
        hand_actions = self.actions[:, 7:]

        current_arm_targets = self._joint_targets[:, self._arm_joint_ids]
        raw_arm_targets = current_arm_targets + arm_actions * self.cfg.arm_action_scale * self.dt
        arm_lower = self._default_arm_pos - self._arm_clamp_delta
        arm_upper = self._default_arm_pos + self._arm_clamp_delta
        raw_arm_targets = torch.clamp(raw_arm_targets, arm_lower, arm_upper)
        arm_targets = (
            self.cfg.arm_moving_average * raw_arm_targets
            + (1.0 - self.cfg.arm_moving_average) * current_arm_targets
        )

        hand_lower = self._joint_lower_limits[self._control_hand_joint_ids].unsqueeze(0)
        hand_upper = self._joint_upper_limits[self._control_hand_joint_ids].unsqueeze(0)
        if self._uses_active_hand_actions():
            raw_hand_targets = self._active_hand_actions_to_sim_targets(hand_actions)
        else:
            raw_hand_targets = 0.5 * (hand_actions + 1.0) * (hand_upper - hand_lower) + hand_lower
        raw_hand_targets = torch.clamp(raw_hand_targets, hand_lower, hand_upper)
        current_hand_targets = self._joint_targets[:, self._control_hand_joint_ids]
        hand_targets = (
            self.cfg.hand_moving_average * raw_hand_targets
            + (1.0 - self.cfg.hand_moving_average) * current_hand_targets
        )

        self._prev_joint_targets[:] = self._joint_targets
        self._joint_targets[:, self._arm_joint_ids] = arm_targets
        self._joint_targets[:, self._control_hand_joint_ids] = hand_targets
        self._apply_initial_target_locks()
        self._apply_post_success_target_locks()
        self.robot.set_joint_position_target(
            self._joint_targets[:, self._controlled_joint_ids], joint_ids=self._controlled_joint_ids
        )

    def _apply_initial_target_locks(self) -> None:
        lock_arm = self.episode_length_buf < self.cfg.initial_arm_target_lock_steps
        if torch.any(lock_arm):
            self._joint_targets[lock_arm.nonzero(as_tuple=False).squeeze(-1)[:, None], self._arm_joint_ids] = (
                self._default_joint_pos[lock_arm][:, self._arm_joint_ids]
            )

        lock_hand = self.episode_length_buf < self.cfg.initial_hand_target_lock_steps
        if torch.any(lock_hand):
            self._joint_targets[lock_hand.nonzero(as_tuple=False).squeeze(-1)[:, None], self._control_hand_joint_ids] = (
                self._default_joint_pos[lock_hand][:, self._control_hand_joint_ids]
            )

    def _post_success_stability_enabled(self) -> bool:
        return self.cfg.task_family == "dynamic_tabletop_grasp" or (
            self.cfg.task_family == "falling_baton_grasp"
            and bool(getattr(self.cfg, "falling_post_success_stability_enabled", False))
        )

    def _apply_post_success_target_locks(self) -> None:
        if not self._post_success_stability_enabled():
            return
        if not bool(getattr(self.cfg, "tabletop_post_success_stability_latch_enabled", False)):
            return
        lock_mask = self._post_success_stability_latched
        if not torch.any(lock_mask):
            return
        lock_ids = lock_mask.nonzero(as_tuple=False).squeeze(-1)
        if bool(getattr(self.cfg, "tabletop_post_success_arm_target_lock_enabled", False)):
            blend = min(max(float(getattr(self.cfg, "tabletop_post_success_arm_target_lock_blend", 1.0)), 0.0), 1.0)
            current = self._joint_targets[lock_ids[:, None], self._arm_joint_ids]
            latched = self._post_success_arm_joint_target[lock_ids]
            self._joint_targets[lock_ids[:, None], self._arm_joint_ids] = (1.0 - blend) * current + blend * latched
        if bool(getattr(self.cfg, "tabletop_post_success_hand_target_lock_enabled", False)):
            blend = min(max(float(getattr(self.cfg, "tabletop_post_success_hand_target_lock_blend", 1.0)), 0.0), 1.0)
            current = self._joint_targets[lock_ids[:, None], self._control_hand_joint_ids]
            latched = self._post_success_hand_joint_target[lock_ids]
            self._joint_targets[lock_ids[:, None], self._control_hand_joint_ids] = (1.0 - blend) * current + blend * latched

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        self._update_object_contact_force_diagnostics()

        palm_distance = torch.norm(self._object_pos_w - self._palm_pos_w, dim=-1)
        palm_reach = torch.exp(-palm_distance / self.cfg.reach_distance_scale)
        fingertip_reach = torch.mean(torch.exp(-torch.relu(self._surface_dist) / self.cfg.fingertip_distance_scale), dim=-1)

        strict_reward_enabled = bool(getattr(self.cfg, "strict_reward_enabled", False))
        reward_contact_score = self._strict_contact_score if strict_reward_enabled else self._contact_score
        reward_true_grasp = self._strict_true_grasp if strict_reward_enabled else self._true_grasp
        reward_grasp_seen = self._strict_grasp_seen if strict_reward_enabled else self._grasp_seen
        if (
            self.cfg.task_family == "dynamic_tabletop_grasp"
            and bool(getattr(self.cfg, "tabletop_gate_boolean_grasp_rewards_by_clearance", False))
        ):
            reward_true_grasp = reward_true_grasp & self._tabletop_arm_clearance_ok
            reward_grasp_seen = reward_grasp_seen & self._tabletop_arm_clearance_ok
        reward_grasp_streak = (
            self._tabletop_strict_true_grasp_streak
            if strict_reward_enabled
            else self._tabletop_true_grasp_streak
        )
        reward_weighted_opposition_score = (
            self._strict_weighted_opposition_score if strict_reward_enabled else self._weighted_opposition_score
        )

        thumb_score = reward_contact_score[:, 0]
        non_thumb_score = torch.sum(reward_contact_score[:, 1:], dim=-1) / 4.0
        thumb_contact_weight = float(getattr(self.cfg, "thumb_contact_reward_weight", 0.45))
        thumb_contact_weight = min(max(thumb_contact_weight, 0.0), 1.0)
        thumb_true_grasp_weight = float(getattr(self.cfg, "thumb_true_grasp_score_weight", 0.50))
        thumb_true_grasp_weight = min(max(thumb_true_grasp_weight, 0.0), 1.0)
        thumb_pair_contact_score = torch.minimum(thumb_score, non_thumb_score)
        if bool(getattr(self.cfg, "contact_reward_requires_thumb_pair", False)):
            contact_rew = thumb_pair_contact_score
        else:
            contact_rew = thumb_contact_weight * thumb_score + (1.0 - thumb_contact_weight) * non_thumb_score
        opposition_reward_score = (
            reward_weighted_opposition_score
            if bool(getattr(self.cfg, "opposition_reward_uses_weighted_score", False))
            else reward_weighted_opposition_score if strict_reward_enabled else self._opposition_score
        )
        if bool(getattr(self.cfg, "contact_reward_uses_opposition_product", False)):
            min_contact_opposition = min(
                max(float(getattr(self.cfg, "contact_reward_opposition_min_multiplier", 0.0)), 0.0),
                1.0,
            )
            contact_opposition_multiplier = min_contact_opposition + (1.0 - min_contact_opposition) * torch.clamp(
                opposition_reward_score,
                0.0,
                1.0,
            )
            contact_rew = contact_rew * contact_opposition_multiplier
        if bool(getattr(self.cfg, "true_grasp_score_requires_thumb_pair", False)):
            true_grasp_base_score = thumb_pair_contact_score
        else:
            true_grasp_base_score = (
                thumb_true_grasp_weight * thumb_score + (1.0 - thumb_true_grasp_weight) * non_thumb_score
            )
        if bool(getattr(self.cfg, "true_grasp_score_uses_opposition_product", False)):
            min_opposition_multiplier = min(
                max(float(getattr(self.cfg, "true_grasp_score_opposition_min_multiplier", 0.0)), 0.0),
                1.0,
            )
            opposition_multiplier = min_opposition_multiplier + (1.0 - min_opposition_multiplier) * torch.clamp(
                opposition_reward_score,
                0.0,
                1.0,
            )
            true_grasp_score = torch.clamp(true_grasp_base_score * opposition_multiplier, 0.0, 1.0)
        else:
            true_grasp_score = torch.clamp(
                true_grasp_base_score + opposition_reward_score,
                0.0,
                1.0,
            )
        grasp_quality = self._strict_grasp_quality if strict_reward_enabled else self._grasp_quality
        lifted_true_grasp = lift_quality = torch.zeros_like(thumb_score)
        lift_progress = torch.zeros_like(thumb_score)

        object_pos_local = self._object_pos_w - self.scene.env_origins
        rel_vel_score = torch.exp(-self._object_palm_rel_vel / max(float(self.cfg.stable_object_palm_vel), 1.0e-3))
        z_gate = object_pos_local[:, 2] > self.cfg.catch_success_min_z
        stable_object = self._object_palm_rel_vel < self.cfg.stable_object_palm_vel
        stable_catch = reward_true_grasp & stable_object
        pregrasp_xy_rew = torch.zeros_like(palm_distance)
        pregrasp_height_rew = torch.zeros_like(palm_distance)
        pregrasp_low_palm_penalty = torch.zeros_like(palm_distance)
        pregrasp_contact_gate = torch.ones_like(palm_distance)
        side_contact_penalty = torch.zeros_like(palm_distance)
        non_thumb_without_thumb_penalty = torch.zeros_like(palm_distance)
        clearance_contact_gate = torch.ones_like(palm_distance)
        palm_pregrasp_xy_dist = torch.zeros_like(palm_distance)
        pregrasp_height_delta = self._palm_pos_w[:, 2] - self._object_pos_w[:, 2]
        object_speed = torch.norm(self._object_lin_vel_w, dim=-1)
        object_ang_speed = torch.norm(self._object_ang_vel_w, dim=-1)
        hover_xy_dist = torch.zeros_like(palm_distance)
        hover_z_error = torch.zeros_like(palm_distance)
        hover_target_rew = torch.zeros_like(palm_distance)
        hover_stability_rew = torch.zeros_like(palm_distance)
        hover_target_ok = torch.ones_like(stable_object)
        hover_speed_ok = torch.ones_like(stable_object)
        hover_latched = torch.zeros_like(palm_distance)
        hover_height_progress = torch.zeros_like(palm_distance)
        hover_goal_rew = torch.zeros_like(palm_distance)
        hover_linear_error = torch.zeros_like(palm_distance)
        hover_z_overshoot_ratio = torch.zeros_like(palm_distance)
        hover_under_height_ratio = torch.zeros_like(palm_distance)
        hover_z_vel = torch.zeros_like(palm_distance)
        hover_latch_phase = torch.zeros_like(palm_distance)
        hover_grasp_loss = torch.zeros_like(palm_distance)
        hover_post_latch_speed = torch.zeros_like(palm_distance)
        post_success_phase = torch.zeros_like(palm_distance)
        post_success_unstable = torch.zeros_like(palm_distance)
        post_success_grasp_loss = torch.zeros_like(palm_distance)
        post_success_under_height_ratio = torch.zeros_like(palm_distance)
        post_success_speed = torch.zeros_like(palm_distance)
        post_success_arm_joint_vel = torch.zeros_like(palm_distance)
        post_success_arm_target_drift = torch.zeros_like(palm_distance)
        post_success_palm_drift = torch.zeros_like(palm_distance)
        post_success_palm_drift_penalty = torch.zeros_like(palm_distance)
        no_lift_after_grasp_penalty = torch.zeros_like(palm_distance)
        grasped_palm_lift_rew = torch.zeros_like(palm_distance)
        tabletop_arm_lift_progress = torch.zeros_like(palm_distance)
        tabletop_arm_lift_rew = torch.zeros_like(palm_distance)
        tabletop_arm_object_lift_gap_penalty = torch.zeros_like(palm_distance)
        tabletop_lift_memory_grasp_gate = torch.zeros_like(palm_distance)
        tabletop_lift_action_prior = torch.zeros_like(palm_distance)
        tabletop_lift_action_prior_gate = torch.zeros_like(palm_distance)
        tabletop_lift_action_prior_rew = torch.zeros_like(palm_distance)
        tabletop_lift_without_object_penalty = torch.zeros_like(palm_distance)
        tabletop_lift_without_current_grasp_penalty = torch.zeros_like(palm_distance)
        tabletop_strict_hold_rew = torch.zeros_like(palm_distance)
        tabletop_strict_grasp_loss_penalty = torch.zeros_like(palm_distance)
        tabletop_object_up_vel_rew = torch.zeros_like(palm_distance)
        tabletop_object_carry_lift_rew = torch.zeros_like(palm_distance)
        tabletop_object_carry_stall_penalty = torch.zeros_like(palm_distance)
        tabletop_force_grasp_rew = torch.zeros_like(palm_distance)
        tabletop_force_grasp_streak_rew = torch.zeros_like(palm_distance)
        tabletop_force_stable_grasp_rew = torch.zeros_like(palm_distance)
        tabletop_force_grasp_loss_penalty = torch.zeros_like(palm_distance)
        tabletop_underwrap_rew = torch.zeros_like(palm_distance)
        tabletop_underwrap_thumb_score = torch.zeros_like(palm_distance)
        tabletop_underwrap_non_thumb_score = torch.zeros_like(palm_distance)
        tabletop_underwrap_pair_score = torch.zeros_like(palm_distance)
        tabletop_underwrap_progress_score = torch.zeros_like(palm_distance)
        tabletop_underwrap_mean_score = torch.zeros_like(palm_distance)
        tabletop_underwrap_min_tip_z_rel = torch.zeros_like(palm_distance)
        tabletop_underwrap_thumb_z_rel = torch.zeros_like(palm_distance)
        tabletop_underwrap_min_non_thumb_z_rel = torch.zeros_like(palm_distance)
        strict_approach_rew = torch.zeros_like(palm_distance)
        strict_multifinger_approach_rew = torch.zeros_like(palm_distance)
        strict_opposition_approach_rew = torch.zeros_like(palm_distance)
        strict_touch_rew = torch.zeros_like(palm_distance)
        strict_opposition_touch_rew = torch.zeros_like(palm_distance)
        falling_success_grasp_gate = torch.zeros_like(stable_object)
        falling_success_palm_gate = torch.ones_like(stable_object)
        falling_success_contact_gate = torch.ones_like(stable_object)
        tabletop_affordance = self._compute_tabletop_affordance_reward_terms()
        falling_affordance = self._compute_falling_baton_affordance_reward_terms()
        falling_affordance_positive_rew = falling_affordance["pos_score"]
        falling_affordance_thumb_geom_rew = falling_affordance["thumb_pos_geom_score"]
        falling_affordance_thumb_touch_rew = falling_affordance["thumb_pos_score"]
        falling_stable_grasp_rew = torch.zeros_like(palm_distance)
        falling_palm_gate_rew = torch.zeros_like(palm_distance)
        falling_positive_stable_rew = torch.zeros_like(palm_distance)
        falling_soft_success_progress_rew = torch.zeros_like(palm_distance)
        falling_opposed_stable_pinch_rew = torch.zeros_like(palm_distance)
        falling_pinched_rel_vel_penalty = torch.zeros_like(palm_distance)

        if self.cfg.task_family == "falling_baton_grasp":
            if bool(getattr(self.cfg, "falling_affordance_positive_requires_thumb_pair", False)):
                falling_affordance_positive_rew = falling_affordance_positive_rew * torch.minimum(
                    self._strict_thumb_touch_score,
                    self._strict_non_thumb_pair_touch_score,
                )
            if bool(getattr(self.cfg, "falling_affordance_positive_uses_opposition_product", False)):
                min_affordance_opposition = min(
                    max(float(getattr(self.cfg, "falling_affordance_positive_opposition_min_multiplier", 0.0)), 0.0),
                    1.0,
                )
                affordance_opposition_multiplier = min_affordance_opposition + (
                    1.0 - min_affordance_opposition
                ) * torch.clamp(self._strict_opposition_touch_score, 0.0, 1.0)
                falling_affordance_positive_rew = falling_affordance_positive_rew * affordance_opposition_multiplier

            if float(getattr(self.cfg, "falling_non_thumb_without_thumb_penalty_scale", 0.0)) > 0.0:
                non_thumb_gate_start = float(getattr(self.cfg, "falling_non_thumb_without_thumb_gate_start", 0.15))
                non_thumb_gate_ramp = max(
                    float(getattr(self.cfg, "falling_non_thumb_without_thumb_gate_ramp", 0.35)),
                    1.0e-6,
                )
                thumb_target = max(
                    float(getattr(self.cfg, "falling_non_thumb_without_thumb_thumb_target", 0.30)),
                    1.0e-6,
                )
                non_thumb_close = torch.clamp(
                    (self._strict_non_thumb_pair_touch_score - non_thumb_gate_start) / non_thumb_gate_ramp,
                    0.0,
                    1.0,
                )
                thumb_missing = torch.clamp((thumb_target - self._strict_thumb_touch_score) / thumb_target, 0.0, 1.0)
                non_thumb_without_thumb_penalty = non_thumb_close * thumb_missing

        if self.cfg.task_family == "dynamic_tabletop_grasp":
            lift_progress = torch.clamp(self._object_height_delta / self.cfg.tabletop_success_lift_height, 0.0, 1.0)
            lifted_true_grasp = lift_progress * reward_true_grasp.float()
            hover_latch_grasp = reward_true_grasp
            if bool(getattr(self.cfg, "tabletop_hover_latch_uses_grasp_seen", False)):
                hover_latch_grasp = hover_latch_grasp | reward_grasp_seen
            hover_reward_grasp_gate = reward_true_grasp.float()
            if bool(getattr(self.cfg, "tabletop_hover_reward_uses_grasp_seen", False)):
                hover_reward_grasp_gate = torch.maximum(hover_reward_grasp_gate, reward_grasp_seen.float())
            hover_success_grasp = reward_true_grasp
            if bool(getattr(self.cfg, "tabletop_success_uses_grasp_seen", False)):
                hover_success_grasp = hover_success_grasp | reward_grasp_seen
            tabletop_success_grasp = hover_success_grasp
            if bool(getattr(self.cfg, "strict_success_enabled", False)):
                tabletop_success_grasp = self._strict_true_grasp
            if bool(getattr(self.cfg, "tabletop_success_requires_force_grasp", False)):
                tabletop_success_grasp = tabletop_success_grasp & self._force_grasp
            if bool(getattr(self.cfg, "tabletop_lift_rewards_require_force_grasp", False)):
                force_lift_gate = self._force_grasp.float()
                lifted_true_grasp = lifted_true_grasp * force_lift_gate
                hover_reward_grasp_gate = hover_reward_grasp_gate * force_lift_gate
            unlatch_mask = ~self._object_hover_target_latched
            if torch.any(unlatch_mask):
                self._object_hover_target_pos_w[unlatch_mask, :2] = self._object_pos_w[unlatch_mask, :2]
                self._object_hover_target_pos_w[unlatch_mask, 2] = (
                    self.scene.env_origins[unlatch_mask, 2]
                    + self._active_object_start_z[unlatch_mask]
                    + float(getattr(self.cfg, "tabletop_hover_height_delta", 0.16))
                )
            latch_now = (
                unlatch_mask
                & hover_latch_grasp
                & (lift_progress >= float(getattr(self.cfg, "tabletop_hover_latch_lift_progress", 0.35)))
            )
            self._object_hover_target_latched = self._object_hover_target_latched | latch_now
            hover_latched = self._object_hover_target_latched.float()
            hover_target = self._object_hover_target_pos_w
            hover_xy_dist = torch.norm(self._object_pos_w[:, :2] - hover_target[:, :2], dim=-1)
            hover_z_error = torch.abs(self._object_pos_w[:, 2] - hover_target[:, 2])
            hover_height_delta = max(float(getattr(self.cfg, "tabletop_hover_height_delta", 0.16)), 1.0e-6)
            hover_height_progress = torch.clamp(self._object_height_delta / hover_height_delta, 0.0, 1.0)
            hover_xy_score = torch.exp(
                -hover_xy_dist / max(float(getattr(self.cfg, "tabletop_hover_xy_distance_scale", 0.18)), 1.0e-6)
            )
            hover_z_score = torch.exp(
                -hover_z_error / max(float(getattr(self.cfg, "tabletop_hover_z_distance_scale", 0.07)), 1.0e-6)
            )
            object_speed_score = torch.exp(
                -object_speed / max(float(getattr(self.cfg, "tabletop_hover_object_speed_scale", 0.25)), 1.0e-6)
            )
            object_ang_speed_score = torch.exp(
                -object_ang_speed / max(float(getattr(self.cfg, "tabletop_hover_ang_speed_scale", 8.0)), 1.0e-6)
            )
            hover_requires_xy = bool(getattr(self.cfg, "tabletop_hover_success_requires_xy", True))
            hover_target_rew = hover_xy_score * hover_z_score if hover_requires_xy else hover_z_score
            hover_stability_rew = (
                hover_target_rew
                * rel_vel_score
                * object_speed_score
                * object_ang_speed_score
                * hover_reward_grasp_gate
                * lift_progress
            )
            hover_target_ok = (
                hover_z_error <= float(getattr(self.cfg, "tabletop_hover_success_z_tolerance", 0.08))
            )
            if hover_requires_xy:
                hover_target_ok = hover_target_ok & (
                    hover_xy_dist <= float(getattr(self.cfg, "tabletop_hover_success_xy_tolerance", 0.18))
                )
            hover_speed_ok = (
                (object_speed <= float(getattr(self.cfg, "tabletop_hover_success_object_speed", 0.30)))
                & stable_object
            )
            hover_xy_tol = max(float(getattr(self.cfg, "tabletop_hover_success_xy_tolerance", 0.18)), 1.0e-6)
            hover_z_tol = max(float(getattr(self.cfg, "tabletop_hover_success_z_tolerance", 0.08)), 1.0e-6)
            hover_speed_tol = max(float(getattr(self.cfg, "tabletop_hover_success_object_speed", 0.30)), 1.0e-6)
            hover_ang_scale = max(float(getattr(self.cfg, "tabletop_hover_ang_speed_scale", 8.0)), 1.0e-6)
            hover_z_vel = torch.abs(self._object_lin_vel_w[:, 2])
            hover_z_overshoot_ratio = torch.clamp(
                torch.relu(self._object_pos_w[:, 2] - hover_target[:, 2] - hover_z_tol) / hover_z_tol,
                0.0,
                5.0,
            )
            hover_under_height_ratio = torch.clamp(
                torch.relu(hover_target[:, 2] - self._object_pos_w[:, 2] - hover_z_tol) / hover_z_tol,
                0.0,
                5.0,
            )
            hover_latch_phase = hover_latched * torch.clamp(
                0.35 + 0.65 * torch.maximum(lift_progress, hover_height_progress),
                0.0,
                1.0,
            )
            hover_xy_close = torch.clamp(1.0 - hover_xy_dist / (2.0 * hover_xy_tol), 0.0, 1.0)
            hover_z_close = torch.clamp(1.0 - hover_z_error / (2.0 * hover_z_tol), 0.0, 1.0)
            hover_speed_close = torch.clamp(1.0 - object_speed / (2.0 * hover_speed_tol), 0.0, 1.0)
            hover_goal_rew = (
                hover_latched
                * hover_reward_grasp_gate
                * torch.clamp(0.4 + 0.6 * hover_height_progress, 0.0, 1.0)
                * (
                    (
                        0.40 * hover_z_close
                        + 0.25 * hover_xy_close
                        + 0.25 * hover_speed_close
                        + 0.10 * rel_vel_score
                    )
                    if hover_requires_xy
                    else (
                        0.50 * hover_z_close
                        + 0.35 * hover_speed_close
                        + 0.15 * rel_vel_score
                    )
                )
            )
            xy_error_term = hover_xy_dist / hover_xy_tol if hover_requires_xy else 0.15 * hover_xy_dist / hover_xy_tol
            hover_linear_error = (
                hover_latched
                * hover_reward_grasp_gate
                * torch.clamp(0.35 + 0.65 * lift_progress, 0.0, 1.0)
                * torch.clamp(
                    xy_error_term
                    + hover_z_error / hover_z_tol
                    + object_speed / hover_speed_tol
                    + 0.05 * object_ang_speed / hover_ang_scale,
                    0.0,
                    4.0,
                )
            )
            hover_grasp_loss = hover_latch_phase * (1.0 - hover_reward_grasp_gate)
            hover_post_latch_speed = hover_latch_phase * (
                object_speed * object_speed
                + self._object_palm_rel_vel * self._object_palm_rel_vel
                + 0.02 * object_ang_speed * object_ang_speed
            )
            object_xy_vel = self._object_lin_vel_w[:, :2]
            object_xy_speed = torch.norm(object_xy_vel, dim=-1, keepdim=True)
            fallback_xy = self._object_pos_w[:, :2] - self._palm_pos_w[:, :2]
            fallback_xy_dir = fallback_xy / torch.clamp(torch.norm(fallback_xy, dim=-1, keepdim=True), min=1.0e-6)
            object_xy_dir = torch.where(
                object_xy_speed > 1.0e-4,
                object_xy_vel / torch.clamp(object_xy_speed, min=1.0e-6),
                fallback_xy_dir,
            )
            predicted_object_xy = (
                self._object_pos_w[:, :2]
                + object_xy_vel * float(self.cfg.dynamic_tabletop_pregrasp_lead_time)
            )
            pregrasp_target_xy = (
                predicted_object_xy
                + object_xy_dir * float(self.cfg.dynamic_tabletop_pregrasp_ahead_distance)
            )
            palm_pregrasp_xy_dist = torch.norm(self._palm_pos_w[:, :2] - pregrasp_target_xy, dim=-1)
            pregrasp_xy_rew = torch.exp(
                -palm_pregrasp_xy_dist / max(float(self.cfg.dynamic_tabletop_pregrasp_xy_distance_scale), 1.0e-6)
            )
            target_height = float(self.cfg.dynamic_tabletop_pregrasp_height_offset)
            pregrasp_height_rew = torch.exp(
                -torch.abs(pregrasp_height_delta - target_height)
                / max(float(self.cfg.dynamic_tabletop_pregrasp_height_scale), 1.0e-6)
            )
            low_palm = torch.clamp(
                (
                    float(self.cfg.dynamic_tabletop_min_palm_height_offset)
                    - pregrasp_height_delta
                )
                / max(float(self.cfg.dynamic_tabletop_low_palm_height_scale), 1.0e-6),
                min=0.0,
                max=float(self.cfg.dynamic_tabletop_low_palm_max_penalty),
            )
            pregrasp_low_palm_penalty = low_palm
            pregrasp_readiness = torch.clamp(0.55 * pregrasp_xy_rew + 0.45 * pregrasp_height_rew, 0.0, 1.0)
            if bool(self.cfg.dynamic_tabletop_gate_contact_rewards_by_pregrasp):
                min_multiplier = float(self.cfg.dynamic_tabletop_contact_pregrasp_gate_min)
                pregrasp_contact_gate = min_multiplier + (1.0 - min_multiplier) * pregrasp_readiness
                contact_rew = contact_rew * pregrasp_contact_gate
                true_grasp_score = true_grasp_score * pregrasp_contact_gate
            side_contact_penalty_scale = float(getattr(self.cfg, "dynamic_tabletop_side_contact_penalty_scale", 0.0))
            if side_contact_penalty_scale > 0.0:
                side_limit = float(getattr(self.cfg, "dynamic_tabletop_side_contact_xy_limit", 0.12))
                side_ramp = max(float(getattr(self.cfg, "dynamic_tabletop_side_contact_xy_ramp", 0.08)), 1.0e-6)
                side_over = torch.clamp((palm_pregrasp_xy_dist - side_limit) / side_ramp, 0.0, 3.0)
                side_contact_signal = torch.max(reward_contact_score, dim=-1).values
                side_contact_penalty = side_over * side_contact_signal
            non_thumb_without_thumb_penalty_scale = float(
                getattr(self.cfg, "tabletop_non_thumb_without_thumb_penalty_scale", 0.0)
            )
            if non_thumb_without_thumb_penalty_scale > 0.0:
                non_thumb_best_score = torch.max(reward_contact_score[:, 1:], dim=-1).values
                non_thumb_gate_start = float(getattr(self.cfg, "tabletop_non_thumb_without_thumb_gate_start", 0.08))
                non_thumb_gate_ramp = max(
                    float(getattr(self.cfg, "tabletop_non_thumb_without_thumb_gate_ramp", 0.30)),
                    1.0e-6,
                )
                thumb_target = max(float(getattr(self.cfg, "tabletop_non_thumb_without_thumb_thumb_target", 0.25)), 1.0e-6)
                non_thumb_close = torch.clamp(
                    (non_thumb_best_score - non_thumb_gate_start) / non_thumb_gate_ramp,
                    0.0,
                    1.0,
                )
                thumb_missing = torch.clamp((thumb_target - thumb_score) / thumb_target, 0.0, 1.0)
                non_thumb_without_thumb_penalty = non_thumb_close * thumb_missing
            if bool(getattr(self.cfg, "tabletop_gate_contact_rewards_by_clearance", False)):
                min_multiplier = float(getattr(self.cfg, "tabletop_contact_clearance_gate_min", 0.0))
                min_multiplier = min(max(min_multiplier, 0.0), 1.0)
                scale = max(float(getattr(self.cfg, "tabletop_contact_clearance_gate_scale", 0.50)), 1.0e-6)
                clearance_score = torch.exp(-self._tabletop_arm_clearance_penalty / scale)
                clearance_contact_gate = min_multiplier + (1.0 - min_multiplier) * clearance_score
                contact_rew = contact_rew * clearance_contact_gate
                true_grasp_score = true_grasp_score * clearance_contact_gate
                opposition_reward_score = opposition_reward_score * clearance_contact_gate
                grasp_quality = grasp_quality * clearance_contact_gate
            opposition_lift_gate_signal = torch.clamp(
                0.5 * self._weighted_opposition_score + 0.5 * self._opposing_contact.float(),
                0.0,
                1.0,
            )
            if bool(self.cfg.lift_reward_uses_grasp_quality_gate):
                min_multiplier = float(self.cfg.lift_reward_min_grasp_quality_multiplier)
                lift_quality = min_multiplier + (1.0 - min_multiplier) * grasp_quality
            else:
                lift_quality = torch.ones_like(lift_progress)
            if bool(getattr(self.cfg, "lift_reward_uses_opposition_gate", False)):
                min_multiplier = float(getattr(self.cfg, "lift_reward_min_opposition_multiplier", 1.0))
                lift_quality = lift_quality * (
                    min_multiplier + (1.0 - min_multiplier) * opposition_lift_gate_signal
                )
            if bool(getattr(self.cfg, "quality_lift_progress_uses_opposition_gate", False)):
                min_multiplier = float(getattr(self.cfg, "quality_lift_progress_min_opposition_multiplier", 1.0))
                quality_lift_gate = min_multiplier + (1.0 - min_multiplier) * opposition_lift_gate_signal
            else:
                quality_lift_gate = torch.ones_like(lift_progress)
            if bool(getattr(self.cfg, "tabletop_lift_rewards_require_force_grasp", False)):
                force_lift_gate = self._force_grasp.float()
                lift_quality = lift_quality * force_lift_gate
                quality_lift_gate = quality_lift_gate * force_lift_gate
            if bool(getattr(self.cfg, "tabletop_lift_rewards_require_current_strict_grasp", False)):
                strict_lift_gate = reward_true_grasp.float()
                lift_quality = lift_quality * strict_lift_gate
                quality_lift_gate = quality_lift_gate * strict_lift_gate
            if self._tabletop_arm_lift_baseline_mode in {"first_strict_grasp", "first_force_grasp"}:
                baseline_grasp = (
                    self._force_grasp
                    if self._tabletop_arm_lift_baseline_mode == "first_force_grasp"
                    else self._strict_true_grasp
                )
                self._tabletop_arm_lift_baseline_grasp_streak = torch.where(
                    baseline_grasp,
                    self._tabletop_arm_lift_baseline_grasp_streak + 1,
                    torch.zeros_like(self._tabletop_arm_lift_baseline_grasp_streak),
                )
                baseline_min_streak = max(
                    int(getattr(self.cfg, "tabletop_arm_lift_progress_baseline_grasp_streak", 1)),
                    1,
                )
                baseline_latch_now = (
                    self._tabletop_arm_lift_baseline_grasp_streak >= baseline_min_streak
                ) & (~self._tabletop_arm_lift_baseline_latched)
                if torch.any(baseline_latch_now):
                    baseline_latch_ids = baseline_latch_now.nonzero(as_tuple=False).squeeze(-1)
                    self._tabletop_arm_lift_baseline_pos[baseline_latch_ids] = self.robot.data.joint_pos[
                        baseline_latch_ids
                    ][:, self._arm_joint_ids]
                    self._tabletop_arm_lift_baseline_latched[baseline_latch_ids] = True
            arm_delta = self.robot.data.joint_pos[:, self._arm_joint_ids] - self._tabletop_arm_lift_baseline_pos
            tabletop_arm_lift_progress = torch.clamp(
                torch.sum(arm_delta * self._lift_arm_delta, dim=-1) / self._lift_delta_norm_sq,
                0.0,
                1.0,
            )
            tabletop_arm_lift_progress = (
                tabletop_arm_lift_progress * self._tabletop_arm_lift_baseline_latched.float()
            )
            if self.cfg.policy_action_interface == "joint_target":
                arm_lower = self._joint_lower_limits[self._arm_joint_ids].unsqueeze(0)
                arm_upper = self._joint_upper_limits[self._arm_joint_ids].unsqueeze(0)
                arm_center = torch.clamp(
                    self._default_joint_pos[:, self._arm_joint_ids], arm_lower, arm_upper
                )
                arm_actions = self.actions[:, :7]
                commanded_arm_target = torch.where(
                    arm_actions >= 0.0,
                    arm_center + arm_actions * (arm_upper - arm_center),
                    arm_center + arm_actions * (arm_center - arm_lower),
                )
                commanded_lift_delta = commanded_arm_target - self._tabletop_arm_lift_baseline_pos
                tabletop_lift_action_prior = torch.clamp(
                    torch.sum(commanded_lift_delta * self._lift_arm_delta, dim=-1)
                    / self._lift_delta_norm_sq,
                    0.0,
                    1.0,
                ) * self._tabletop_arm_lift_baseline_latched.float()
            else:
                tabletop_lift_action_prior = torch.clamp(
                    torch.sum(self.actions[:, :7] * self._lift_action_prior, dim=-1)
                    / self._lift_action_prior_den,
                    0.0,
                    1.0,
                )
            current_lift_grasp_gate = torch.clamp(
                0.55 * reward_true_grasp.float() + 0.45 * true_grasp_score,
                0.0,
                1.0,
            )
            if bool(getattr(self.cfg, "tabletop_lift_gate_requires_current_strict_grasp", False)):
                current_lift_grasp_gate = current_lift_grasp_gate * self._strict_true_grasp.float()
            if bool(getattr(self.cfg, "tabletop_lift_gate_requires_force_grasp", False)):
                current_lift_grasp_gate = current_lift_grasp_gate * self._force_grasp.float()
            tabletop_lift_memory_grasp_gate = current_lift_grasp_gate
            if bool(getattr(self.cfg, "tabletop_lift_use_grasp_seen_gate", False)):
                seen_gate = reward_grasp_seen.float() * float(
                    getattr(self.cfg, "tabletop_lift_grasp_seen_gate", 0.35)
                )
                tabletop_lift_memory_grasp_gate = torch.maximum(
                    tabletop_lift_memory_grasp_gate,
                    seen_gate,
                )
            lift_schedule_unlocked = (
                self.episode_length_buf >= int(getattr(self.cfg, "scripted_action_prior_lift_start_step", 0))
            ).float()
            lift_prior_gate_min = torch.clamp(
                torch.tensor(
                    float(getattr(self.cfg, "tabletop_lift_action_prior_gate_min", 0.0)),
                    dtype=torch.float32,
                    device=self.device,
                ),
                0.0,
                1.0,
            )
            tabletop_lift_action_prior_gate = lift_schedule_unlocked * (
                lift_prior_gate_min + (1.0 - lift_prior_gate_min) * current_lift_grasp_gate
            )
            remaining_lift = torch.clamp(1.0 - lift_progress, 0.0, 1.0)
            pre_lift_hold_gate = remaining_lift * lift_schedule_unlocked
            tabletop_strict_hold_rew = reward_true_grasp.float() * rel_vel_score * pre_lift_hold_gate
            if bool(getattr(self.cfg, "tabletop_strict_grasp_loss_on_transition_only", False)):
                strict_grasp_loss_gate = self._strict_reward_grasp_prev.float()
            else:
                strict_grasp_loss_gate = reward_grasp_seen.float()
            if bool(getattr(self.cfg, "tabletop_strict_grasp_loss_requires_lift_baseline", False)):
                strict_grasp_loss_gate = (
                    strict_grasp_loss_gate * self._tabletop_arm_lift_baseline_latched.float()
                )
            tabletop_strict_grasp_loss_penalty = (
                strict_grasp_loss_gate
                * (1.0 - reward_true_grasp.float())
                * pre_lift_hold_gate
            )
            if float(getattr(self.cfg, "tabletop_underwrap_rew_scale", 0.0)) > 0.0:
                active_radius = getattr(self, "_active_object_radius_tensor", None)
                if torch.is_tensor(active_radius):
                    object_radius = torch.clamp(active_radius, min=0.010).view(-1, 1)
                else:
                    object_radius = torch.full(
                        (self.num_envs, 1),
                        max(float(getattr(self.cfg, "object_radius", 0.030)), 0.010),
                        dtype=torch.float32,
                        device=self.device,
                    )
                tip_rel = self._fingertip_pos_w - self._object_pos_w.unsqueeze(1)
                tip_z_rel = tip_rel[..., 2]
                tabletop_underwrap_min_tip_z_rel = torch.min(tip_z_rel, dim=-1).values
                tabletop_underwrap_thumb_z_rel = tip_z_rel[:, 0]
                tabletop_underwrap_min_non_thumb_z_rel = torch.min(tip_z_rel[:, 1:], dim=-1).values
                tip_xy_radius = torch.norm(tip_rel[..., :2], dim=-1)
                below_center_fraction = float(getattr(self.cfg, "tabletop_underwrap_below_center_fraction", 0.20))
                target_tip_z = self._object_pos_w[:, 2:3] - below_center_fraction * object_radius
                height_scale = max(float(getattr(self.cfg, "tabletop_underwrap_height_scale", 0.012)), 1.0e-6)
                under_height_score = torch.clamp(
                    (target_tip_z - self._fingertip_pos_w[..., 2] + height_scale) / height_scale,
                    0.0,
                    1.0,
                )
                radial_target = float(getattr(self.cfg, "tabletop_underwrap_radial_fraction", 0.95)) * object_radius
                radial_scale = max(float(getattr(self.cfg, "tabletop_underwrap_radial_scale", 0.020)), 1.0e-6)
                radial_score = torch.exp(-torch.abs(tip_xy_radius - radial_target) / radial_scale)
                contact_scale = max(float(getattr(self.cfg, "tabletop_underwrap_contact_scale", 0.018)), 1.0e-6)
                contact_margin = max(float(getattr(self.cfg, "tabletop_underwrap_contact_margin", 0.0)), 0.0)
                near_surface_score = torch.exp(-torch.relu(self._surface_dist - contact_margin) / contact_scale)
                underwrap_scores = under_height_score * radial_score * near_surface_score
                tabletop_underwrap_mean_score = torch.mean(underwrap_scores, dim=-1)
                tabletop_underwrap_thumb_score = underwrap_scores[:, 0]
                underwrap_non_thumb = underwrap_scores[:, 1:]
                underwrap_k = min(
                    max(int(getattr(self.cfg, "tabletop_underwrap_min_non_thumb_contacts", 1)), 1),
                    underwrap_non_thumb.shape[-1],
                )
                tabletop_underwrap_non_thumb_score = torch.topk(
                    underwrap_non_thumb,
                    k=underwrap_k,
                    dim=-1,
                ).values.mean(dim=-1)
                tabletop_underwrap_pair_score = torch.minimum(
                    tabletop_underwrap_thumb_score,
                    tabletop_underwrap_non_thumb_score,
                )
                tabletop_underwrap_progress_score = torch.clamp(
                    0.5 * tabletop_underwrap_thumb_score + 0.5 * tabletop_underwrap_non_thumb_score,
                    0.0,
                    1.0,
                )
                if bool(getattr(self.cfg, "tabletop_underwrap_uses_opposition", True)):
                    min_opposition = min(
                        max(float(getattr(self.cfg, "tabletop_underwrap_opposition_min_multiplier", 0.10)), 0.0),
                        1.0,
                    )
                    tabletop_underwrap_pair_score = tabletop_underwrap_pair_score * (
                        min_opposition + (1.0 - min_opposition) * opposition_lift_gate_signal
                    )
                underwrap_progress_weight = max(
                    float(getattr(self.cfg, "tabletop_underwrap_progress_weight", 0.0)),
                    0.0,
                )
                underwrap_pair_weight = max(float(getattr(self.cfg, "tabletop_underwrap_pair_weight", 1.0)), 0.0)
                underwrap_weight_den = max(underwrap_progress_weight + underwrap_pair_weight, 1.0e-6)
                underwrap_reward_score = (
                    underwrap_progress_weight * tabletop_underwrap_progress_score
                    + underwrap_pair_weight * tabletop_underwrap_pair_score
                ) / underwrap_weight_den
                underwrap_gate = clearance_contact_gate
                if bool(getattr(self.cfg, "tabletop_underwrap_uses_pregrasp_gate", True)):
                    underwrap_gate = underwrap_gate * pregrasp_contact_gate
                tabletop_underwrap_rew = (
                    underwrap_reward_score
                    * underwrap_gate
                    * remaining_lift
                )
            arm_lift_reward_margin = float(getattr(self.cfg, "tabletop_arm_lift_reward_object_margin", 1.0))
            object_coupled_arm_lift = torch.minimum(
                tabletop_arm_lift_progress,
                torch.clamp(lift_progress + arm_lift_reward_margin, 0.0, 1.0),
            )
            tabletop_arm_lift_rew = object_coupled_arm_lift * current_lift_grasp_gate * remaining_lift
            tabletop_lift_action_prior_rew = tabletop_lift_action_prior * tabletop_lift_action_prior_gate * remaining_lift
            arm_object_lift_gap_margin = float(getattr(self.cfg, "tabletop_arm_object_lift_gap_margin", 1.0))
            arm_object_lift_gap = torch.clamp(
                (tabletop_arm_lift_progress - lift_progress - arm_object_lift_gap_margin)
                / max(1.0 - arm_object_lift_gap_margin, 1.0e-6),
                0.0,
                1.0,
            )
            tabletop_arm_object_lift_gap_penalty = (
                arm_object_lift_gap * tabletop_lift_memory_grasp_gate * lift_schedule_unlocked * remaining_lift
            )
            empty_lift_start = float(getattr(self.cfg, "tabletop_lift_without_object_min_arm_progress", 0.20))
            empty_lift_progress = torch.clamp(
                (tabletop_arm_lift_progress - empty_lift_start) / max(1.0 - empty_lift_start, 1.0e-6),
                0.0,
                1.0,
            )
            tabletop_lift_without_object_penalty = (
                empty_lift_progress
                * (1.0 - current_lift_grasp_gate)
                * remaining_lift
                * lift_schedule_unlocked
            )
            lift_without_grasp_start = float(
                getattr(self.cfg, "tabletop_lift_without_current_grasp_min_progress", 0.25)
            )
            lift_without_grasp_ramp = max(
                float(getattr(self.cfg, "tabletop_lift_without_current_grasp_ramp", 0.50)),
                1.0e-6,
            )
            tabletop_lift_without_current_grasp_penalty = (
                torch.clamp((lift_progress - lift_without_grasp_start) / lift_without_grasp_ramp, 0.0, 1.0)
                * (1.0 - current_lift_grasp_gate)
                * lift_schedule_unlocked
            )
            no_lift_min_progress = max(float(getattr(self.cfg, "tabletop_no_lift_min_progress", 0.15)), 1.0e-6)
            no_lift_grasp = self._true_grasp & (lift_progress < no_lift_min_progress)
            strict_no_lift_grasp = self._strict_true_grasp & (lift_progress < no_lift_min_progress)
            if bool(getattr(self.cfg, "tabletop_no_lift_uses_force_grasp_gate", False)):
                no_lift_grasp = self._force_grasp & (lift_progress < no_lift_min_progress)
                strict_no_lift_grasp = no_lift_grasp
            if bool(getattr(self.cfg, "tabletop_no_lift_uses_soft_grasp_gate", False)):
                soft_gate = float(getattr(self.cfg, "tabletop_no_lift_soft_grasp_gate", 0.10))
                soft_no_lift_grasp = (current_lift_grasp_gate >= soft_gate) & (
                    lift_progress < no_lift_min_progress
                )
                no_lift_grasp = soft_no_lift_grasp
                strict_no_lift_grasp = soft_no_lift_grasp
            self._tabletop_true_grasp_streak = torch.where(
                no_lift_grasp,
                self._tabletop_true_grasp_streak + 1,
                torch.zeros_like(self._tabletop_true_grasp_streak),
            )
            self._tabletop_strict_true_grasp_streak = torch.where(
                strict_no_lift_grasp,
                self._tabletop_strict_true_grasp_streak + 1,
                torch.zeros_like(self._tabletop_strict_true_grasp_streak),
            )
            reward_grasp_streak = (
                self._tabletop_strict_true_grasp_streak
                if strict_reward_enabled
                else self._tabletop_true_grasp_streak
            )
            if bool(getattr(self.cfg, "tabletop_no_lift_uses_force_grasp_gate", False)):
                reward_grasp_streak = self._force_grasp_streak
            no_lift_steps = torch.relu(
                reward_grasp_streak.float()
                - float(getattr(self.cfg, "tabletop_no_lift_after_grasp_grace_steps", 30))
            )
            no_lift_ramp = max(float(getattr(self.cfg, "tabletop_no_lift_after_grasp_ramp_steps", 80)), 1.0)
            no_lift_after_grasp_penalty = torch.clamp(
                no_lift_steps / no_lift_ramp,
                0.0,
                float(getattr(self.cfg, "tabletop_no_lift_after_grasp_max_penalty", 4.0)),
            ) * torch.clamp((no_lift_min_progress - lift_progress) / no_lift_min_progress, 0.0, 1.0)
            palm_lift_target_z = (
                self.scene.env_origins[:, 2]
                + self._active_object_start_z
                + float(getattr(self.cfg, "tabletop_grasped_palm_lift_height", 0.08))
            )
            palm_lift_progress = torch.clamp(
                (self._palm_pos_w[:, 2] - palm_lift_target_z)
                / max(float(getattr(self.cfg, "tabletop_grasped_palm_lift_scale", 0.05)), 1.0e-6),
                0.0,
                1.0,
            )
            grasped_palm_lift_rew = (
                reward_true_grasp.float()
                * rel_vel_score
                * palm_lift_progress
                * torch.clamp(1.0 - lift_progress, 0.0, 1.0)
            )
            carry_streak_gate = torch.ones_like(lift_progress)
            carry_min_streak = float(getattr(self.cfg, "tabletop_object_carry_min_grasp_streak", 0.0))
            if carry_min_streak > 0.0:
                carry_streak_ramp = max(float(getattr(self.cfg, "tabletop_object_carry_streak_ramp_steps", 1.0)), 1.0)
                carry_grasp_streak = reward_grasp_streak
                if bool(
                    getattr(
                        self.cfg,
                        "tabletop_object_carry_uses_lift_baseline_grasp_streak",
                        False,
                    )
                ):
                    # The no-lift streak intentionally resets once lift starts.
                    # Carry shaping instead needs the current strict-grasp streak,
                    # which is tracked continuously by the strict lift baseline.
                    carry_grasp_streak = self._tabletop_arm_lift_baseline_grasp_streak
                carry_streak_gate = torch.clamp(
                    (carry_grasp_streak.float() - carry_min_streak + 1.0) / carry_streak_ramp,
                    0.0,
                    1.0,
                )
                if bool(getattr(self.cfg, "tabletop_object_carry_uses_grasp_seen", False)):
                    seen_gate = reward_grasp_seen.float() * float(
                        getattr(self.cfg, "tabletop_object_carry_grasp_seen_gate", 0.25)
                    )
                    carry_streak_gate = torch.maximum(carry_streak_gate, seen_gate)
            object_up_vel = self._object_lin_vel_w[:, 2]
            object_up_vel_progress = torch.clamp(
                object_up_vel / max(float(getattr(self.cfg, "tabletop_object_up_vel_scale", 0.10)), 1.0e-6),
                0.0,
                1.0,
            )
            object_follow_arm_progress = torch.clamp(0.25 + 0.75 * tabletop_arm_lift_progress, 0.0, 1.0)
            object_carry_gate = tabletop_lift_memory_grasp_gate * carry_streak_gate * lift_schedule_unlocked
            tabletop_object_up_vel_rew = object_carry_gate * object_up_vel_progress * remaining_lift
            tabletop_object_carry_lift_rew = (
                object_carry_gate
                * lift_progress
                * rel_vel_score
                * object_follow_arm_progress
            )
            carry_stall_arm_start = float(getattr(self.cfg, "tabletop_object_carry_stall_min_arm_progress", 0.12))
            carry_stall_arm = torch.clamp(
                (tabletop_arm_lift_progress - carry_stall_arm_start) / max(1.0 - carry_stall_arm_start, 1.0e-6),
                0.0,
                1.0,
            )
            carry_stall_min_z_vel = max(float(getattr(self.cfg, "tabletop_object_carry_stall_min_z_vel", 0.015)), 1.0e-6)
            carry_stall_vel = torch.clamp((carry_stall_min_z_vel - object_up_vel) / carry_stall_min_z_vel, 0.0, 1.0)
            tabletop_object_carry_stall_penalty = (
                object_carry_gate
                * carry_stall_arm
                * carry_stall_vel
                * remaining_lift
            )
            force_grasp_streak_target = max(
                float(getattr(self.cfg, "tabletop_force_grasp_streak_target", 8)),
                1.0,
            )
            tabletop_force_grasp_rew = self._force_grasp.float() * remaining_lift
            tabletop_force_grasp_streak_rew = torch.clamp(
                self._force_grasp_streak.float() / force_grasp_streak_target,
                0.0,
                1.0,
            ) * remaining_lift
            tabletop_force_stable_grasp_rew = (
                self._force_grasp.float() * rel_vel_score * remaining_lift
            )
            tabletop_force_grasp_loss_penalty = (
                self._tabletop_arm_lift_baseline_latched.float()
                * self._force_grasp_prev.float()
                * (1.0 - self._force_grasp.float())
                * remaining_lift
            )
            success_now = tabletop_success_grasp & stable_object & (lift_progress > 0.98)
            if bool(getattr(self.cfg, "tabletop_success_requires_hover_target", False)):
                success_now = (
                    success_now
                    & self._object_hover_target_latched
                    & hover_target_ok
                    & hover_speed_ok
                )
            if bool(getattr(self.cfg, "tabletop_success_requires_arm_clearance", False)):
                success_now = success_now & self._tabletop_arm_clearance_ok
            catch_progress = (
                true_grasp_score
                * rel_vel_score
                * torch.clamp(0.25 + 0.75 * lift_progress, 0.0, 1.0)
                * torch.clamp(0.5 + 0.5 * opposition_lift_gate_signal, 0.0, 1.0)
            )
        else:
            lift_progress = torch.zeros_like(palm_distance)
            opposition_lift_gate_signal = torch.ones_like(lift_progress)
            quality_lift_gate = torch.ones_like(lift_progress)
            falling_success_palm_gate_score = torch.ones_like(palm_distance)
            falling_success_contact_gate_score = torch.ones_like(palm_distance)
            falling_success_positive_gate_score = torch.ones_like(palm_distance)
            if bool(getattr(self.cfg, "falling_success_uses_grasp_seen", True)):
                falling_success_grasp_gate = self._grasp_seen
            else:
                falling_success_grasp_gate = self._true_grasp
            if bool(getattr(self.cfg, "falling_success_uses_strict_grasp", False)):
                falling_success_grasp_gate = self._strict_true_grasp
            max_success_palm_dist = float(getattr(self.cfg, "falling_success_max_palm_distance", 0.0))
            if max_success_palm_dist > 0.0:
                falling_success_palm_gate = palm_distance < max_success_palm_dist
                palm_gate_scale = max(
                    float(getattr(self.cfg, "falling_success_palm_gate_soft_scale", 0.055)),
                    1.0e-6,
                )
                falling_success_palm_gate_score = torch.exp(
                    -torch.relu(palm_distance - max_success_palm_dist) / palm_gate_scale
                )
            min_success_finger_contacts = float(getattr(self.cfg, "falling_success_min_finger_contacts", 0.0))
            if min_success_finger_contacts > 0.0:
                falling_success_contact_gate = self._finger_contact_count >= min_success_finger_contacts
                falling_success_contact_gate_score = torch.clamp(
                    self._finger_contact_count.float() / max(min_success_finger_contacts, 1.0),
                    0.0,
                    1.0,
                )
            catch_hold = (
                falling_success_grasp_gate
                & stable_object
                & z_gate
                & falling_success_palm_gate
                & falling_success_contact_gate
            )
            if bool(getattr(self.cfg, "falling_success_requires_positive_affordance", False)):
                catch_hold = catch_hold & (falling_affordance["pos_contact"] > 0.5)
                falling_success_positive_gate_score = torch.clamp(
                    0.55 * falling_affordance["pos_contact"] + 0.45 * falling_affordance["pos_score"],
                    0.0,
                    1.0,
                )
            success_now = catch_hold
            catch_progress = torch.clamp(
                (0.65 * true_grasp_score + 0.35 * self._grasp_seen.float()) * rel_vel_score * z_gate.float(),
                0.0,
                1.0,
            )
            falling_stable_grasp_rew = reward_true_grasp.float() * rel_vel_score * z_gate.float()
            falling_palm_gate_rew = (
                true_grasp_score
                * rel_vel_score
                * z_gate.float()
                * falling_success_palm_gate_score
            )
            falling_positive_stable_rew = (
                falling_stable_grasp_rew
                * falling_success_palm_gate_score
                * falling_success_positive_gate_score
            )
            falling_soft_success_progress_rew = (
                true_grasp_score
                * rel_vel_score
                * z_gate.float()
                * falling_success_palm_gate_score
                * falling_success_contact_gate_score
                * falling_success_positive_gate_score
            )
            strict_pair_touch_score = torch.minimum(
                self._strict_thumb_touch_score,
                self._strict_non_thumb_pair_touch_score,
            )
            pinch_opposition_multiplier = torch.clamp(
                0.05 + 0.95 * self._strict_opposition_touch_score,
                0.0,
                1.0,
            )
            falling_opposed_stable_pinch_rew = (
                strict_pair_touch_score
                * pinch_opposition_multiplier
                * rel_vel_score
                * z_gate.float()
                * falling_success_palm_gate_score
                * falling_success_contact_gate_score
                * falling_success_positive_gate_score
            )
            rel_vel_over_stable = torch.relu(
                self._object_palm_rel_vel - float(self.cfg.stable_object_palm_vel)
            ) / max(float(self.cfg.stable_object_palm_vel), 1.0e-6)
            falling_pinched_rel_vel_penalty = (
                strict_pair_touch_score
                * falling_success_positive_gate_score
                * falling_success_palm_gate_score
                * z_gate.float()
                * torch.clamp(rel_vel_over_stable, 0.0, 4.0)
            )

        self._success_streak = torch.where(
            success_now,
            self._success_streak + 1,
            torch.zeros_like(self._success_streak),
        )
        hold_progress = torch.clamp(
            self._success_streak.float() / max(float(self.cfg.dynamic_success_hold_steps), 1.0),
            0.0,
            1.0,
        )
        success = self._success_streak >= self.cfg.dynamic_success_hold_steps
        if self._post_success_stability_enabled() and bool(
            getattr(self.cfg, "tabletop_post_success_stability_latch_enabled", False)
        ):
            latch_min_streak = max(
                int(getattr(self.cfg, "stability_target_latch_min_success_streak", 0)),
                0,
            )
            latch_ready = (
                success
                if latch_min_streak <= 0
                else self._success_streak >= latch_min_streak
            )
            latch_now = latch_ready & (~self._post_success_stability_latched)
            if torch.any(latch_now):
                latch_ids = latch_now.nonzero(as_tuple=False).squeeze(-1)
                self._post_success_arm_joint_target[latch_ids] = self.robot.data.joint_pos[latch_ids][
                    :, self._arm_joint_ids
                ]
                if bool(
                    getattr(
                        self.cfg,
                        "tabletop_post_success_hand_lock_uses_actual_joint_pos",
                        False,
                    )
                ):
                    latched_hand_target = self.robot.data.joint_pos[latch_ids][
                        :, self._control_hand_joint_ids
                    ]
                else:
                    latched_hand_target = self._joint_targets[latch_ids][
                        :, self._control_hand_joint_ids
                    ]
                close_fraction = min(
                    max(float(getattr(self.cfg, "tabletop_post_success_hand_close_fraction", 0.0)), 0.0),
                    1.0,
                )
                if close_fraction > 0.0:
                    if self._uses_active_hand_actions():
                        hand_dim = self.actions.shape[-1] - len(self._arm_joint_ids)
                        close_actions = torch.ones(
                            (self.num_envs, hand_dim), dtype=torch.float32, device=self.device
                        )
                        close_target = self._active_hand_actions_to_sim_targets(close_actions)[latch_ids]
                    else:
                        close_target = self._joint_upper_limits[
                            self._control_hand_joint_ids
                        ].unsqueeze(0).expand_as(latched_hand_target)
                    latched_hand_target = latched_hand_target + close_fraction * (
                        close_target - latched_hand_target
                    )
                self._post_success_hand_joint_target[latch_ids] = latched_hand_target
                self._post_success_palm_pos_w[latch_ids] = self._palm_pos_w[latch_ids]
                self._post_success_stability_latched[latch_ids] = True
        success_seen = self._success_seen | success
        if self._post_success_stability_enabled():
            post_success_phase = success_seen.float()
            post_success_unstable = post_success_phase * (1.0 - success_now.float())
            post_success_grasp_for_loss = self._true_grasp
            if bool(getattr(self.cfg, "strict_success_enabled", False)):
                post_success_grasp_for_loss = self._strict_true_grasp
            post_success_grasp_loss = post_success_phase * (1.0 - post_success_grasp_for_loss.float())
            post_success_under_height_ratio = post_success_phase * hover_under_height_ratio
            post_success_speed = post_success_phase * (
                object_speed * object_speed
                + self._object_palm_rel_vel * self._object_palm_rel_vel
                + 0.02 * object_ang_speed * object_ang_speed
            )
            stability_latch_phase = self._post_success_stability_latched.float()
            arm_joint_vel = self.robot.data.joint_vel[:, self._arm_joint_ids]
            post_success_arm_joint_vel = stability_latch_phase * torch.sum(arm_joint_vel * arm_joint_vel, dim=-1)
            arm_target_drift = torch.norm(
                self._joint_targets[:, self._arm_joint_ids] - self._post_success_arm_joint_target,
                dim=-1,
            )
            arm_target_drift_tol = float(getattr(self.cfg, "tabletop_post_success_arm_target_drift_tolerance", 0.10))
            arm_target_drift_scale = max(
                float(getattr(self.cfg, "tabletop_post_success_arm_target_drift_scale", 0.30)),
                1.0e-6,
            )
            post_success_arm_target_drift = stability_latch_phase * torch.clamp(
                torch.relu(arm_target_drift - arm_target_drift_tol) / arm_target_drift_scale,
                0.0,
                4.0,
            )
            palm_drift = torch.norm(self._palm_pos_w - self._post_success_palm_pos_w, dim=-1)
            palm_drift_tol = float(getattr(self.cfg, "tabletop_post_success_palm_drift_tolerance", 0.035))
            palm_drift_scale = max(float(getattr(self.cfg, "tabletop_post_success_palm_drift_scale", 0.08)), 1.0e-6)
            post_success_palm_drift = stability_latch_phase * palm_drift
            post_success_palm_drift_penalty = stability_latch_phase * torch.clamp(
                torch.relu(palm_drift - palm_drift_tol) / palm_drift_scale,
                0.0,
                4.0,
            )
        self._success_seen = success_seen
        contact_like = torch.clamp(0.5 * thumb_score + 0.5 * non_thumb_score, 0.0, 1.0)
        self._update_dynamic_speed_curriculum(
            success=success,
            true_grasp=self._true_grasp,
            strict_true_grasp=self._strict_true_grasp,
            stable_catch=stable_catch,
            catch_hold=success_now,
            contact_like=contact_like,
        )

        if strict_reward_enabled:
            strict_approach_rew = self._strict_approach_score * (1.0 - lift_progress)
            strict_multifinger_approach_rew = self._strict_multifinger_approach_score * (1.0 - lift_progress)
            strict_opposition_approach_rew = self._strict_opposition_approach_score * (1.0 - lift_progress)
            if bool(getattr(self.cfg, "strict_touch_reward_requires_thumb_pair", False)):
                strict_touch_rew = (
                    torch.minimum(self._strict_thumb_touch_score, self._strict_non_thumb_pair_touch_score)
                    * (1.0 - lift_progress)
                )
            else:
                strict_touch_rew = self._strict_multifinger_touch_score * (1.0 - lift_progress)
            if bool(getattr(self.cfg, "strict_touch_reward_uses_opposition_product", False)):
                min_touch_opposition = min(
                    max(float(getattr(self.cfg, "strict_touch_reward_opposition_min_multiplier", 0.0)), 0.0),
                    1.0,
                )
                touch_opposition_multiplier = min_touch_opposition + (
                    1.0 - min_touch_opposition
                ) * torch.clamp(self._strict_opposition_touch_score, 0.0, 1.0)
                strict_touch_rew = strict_touch_rew * touch_opposition_multiplier
            strict_opposition_touch_rew = self._strict_opposition_touch_score * (1.0 - lift_progress)

        policy_action_penalty = torch.sum(self.policy_actions * self.policy_actions, dim=-1)
        target_delta = self._joint_targets[:, self._controlled_joint_ids] - self._prev_joint_targets[
            :, self._controlled_joint_ids
        ]
        target_delta_penalty = torch.sum(target_delta * target_delta, dim=-1)
        dropped = object_pos_local[:, 2] < (
            self.cfg.table_top_z - 0.04
            if self.cfg.task_family == "dynamic_tabletop_grasp"
            else self.cfg.falling_drop_z
        )

        reward = (
            self.cfg.palm_reach_rew_scale * palm_reach
            + self.cfg.fingertip_reach_rew_scale * fingertip_reach
            + float(getattr(self.cfg, "strict_approach_rew_scale", 0.0)) * strict_approach_rew
            + float(getattr(self.cfg, "strict_multifinger_approach_rew_scale", 0.0))
            * strict_multifinger_approach_rew
            + float(getattr(self.cfg, "strict_opposition_approach_rew_scale", 0.0))
            * strict_opposition_approach_rew
            + float(getattr(self.cfg, "strict_touch_rew_scale", 0.0)) * strict_touch_rew
            + float(getattr(self.cfg, "strict_opposition_touch_rew_scale", 0.0)) * strict_opposition_touch_rew
            + self.cfg.dynamic_tabletop_pregrasp_xy_rew_scale * pregrasp_xy_rew * (1.0 - lift_progress)
            + self.cfg.dynamic_tabletop_pregrasp_height_rew_scale * pregrasp_height_rew * (1.0 - lift_progress)
            + self.cfg.contact_rew_scale * contact_rew
            + self.cfg.true_grasp_rew_scale * true_grasp_score
            + self.cfg.opposition_rew_scale * opposition_reward_score
            + self.cfg.catch_progress_rew_scale * catch_progress
            + self.cfg.grasp_quality_rew_scale * grasp_quality
            + float(getattr(self.cfg, "falling_stable_grasp_rew_scale", 0.0)) * falling_stable_grasp_rew
            + float(getattr(self.cfg, "falling_palm_gate_rew_scale", 0.0)) * falling_palm_gate_rew
            + float(getattr(self.cfg, "falling_positive_stable_rew_scale", 0.0)) * falling_positive_stable_rew
            + float(getattr(self.cfg, "falling_soft_success_progress_rew_scale", 0.0))
            * falling_soft_success_progress_rew
            + float(getattr(self.cfg, "falling_opposed_stable_pinch_rew_scale", 0.0))
            * falling_opposed_stable_pinch_rew
            + self.cfg.quality_lift_progress_rew_scale * lift_progress * grasp_quality * quality_lift_gate
            + self.cfg.lifted_true_grasp_rew_scale * lifted_true_grasp
            + float(getattr(self.cfg, "tabletop_stable_catch_rew_scale", 0.0))
            * stable_catch.float()
            * torch.clamp(
                float(getattr(self.cfg, "tabletop_stable_catch_min_lift_multiplier", 0.20))
                + (1.0 - float(getattr(self.cfg, "tabletop_stable_catch_min_lift_multiplier", 0.20)))
                * lift_progress,
                0.0,
                1.0,
            )
            + self.cfg.lift_progress_rew_scale
            * lift_progress
            * lift_quality
            * (self.cfg.task_family == "dynamic_tabletop_grasp")
            + self.cfg.stable_hold_rew_scale * success_now.float()
            + self.cfg.hold_progress_rew_scale * hold_progress * hold_progress
            + self.cfg.success_bonus * success.float()
            + self.cfg.tabletop_post_success_hold_rew_scale * post_success_phase * success_now.float()
            + self.cfg.tabletop_hover_height_progress_rew_scale
            * hover_height_progress
            * reward_true_grasp.float()
            * quality_lift_gate
            + self.cfg.tabletop_hover_target_rew_scale * hover_target_rew * reward_true_grasp.float() * lift_progress
            + self.cfg.tabletop_hover_goal_rew_scale * hover_goal_rew
            + self.cfg.tabletop_hover_stable_rew_scale * hover_stability_rew
            + self.cfg.tabletop_grasped_palm_lift_rew_scale * grasped_palm_lift_rew
            + self.cfg.tabletop_grasped_arm_lift_rew_scale * tabletop_arm_lift_rew
            + self.cfg.tabletop_lift_action_prior_rew_scale * tabletop_lift_action_prior_rew
            + float(getattr(self.cfg, "tabletop_strict_hold_rew_scale", 0.0)) * tabletop_strict_hold_rew
            + float(getattr(self.cfg, "tabletop_underwrap_rew_scale", 0.0)) * tabletop_underwrap_rew
            + float(getattr(self.cfg, "tabletop_object_up_vel_rew_scale", 0.0))
            * tabletop_object_up_vel_rew
            + float(getattr(self.cfg, "tabletop_object_carry_lift_rew_scale", 0.0))
            * tabletop_object_carry_lift_rew
            + float(getattr(self.cfg, "tabletop_force_grasp_rew_scale", 0.0))
            * tabletop_force_grasp_rew
            + float(getattr(self.cfg, "tabletop_force_grasp_streak_rew_scale", 0.0))
            * tabletop_force_grasp_streak_rew
            + float(getattr(self.cfg, "tabletop_force_stable_grasp_rew_scale", 0.0))
            * tabletop_force_stable_grasp_rew
            + float(getattr(self.cfg, "tabletop_affordance_positive_rew_scale", 0.0))
            * tabletop_affordance["pos_score"]
            + float(getattr(self.cfg, "tabletop_affordance_lift_rew_scale", 0.0))
            * tabletop_affordance["pos_contact"]
            * lift_progress
            * reward_true_grasp.float()
            + float(getattr(self.cfg, "falling_affordance_positive_rew_scale", 0.0))
            * falling_affordance_positive_rew
            + float(getattr(self.cfg, "falling_affordance_thumb_geom_rew_scale", 0.0))
            * falling_affordance_thumb_geom_rew
            + float(getattr(self.cfg, "falling_affordance_thumb_touch_rew_scale", 0.0))
            * falling_affordance_thumb_touch_rew
            - self.cfg.action_penalty_scale * policy_action_penalty
            - self.cfg.arm_target_delta_penalty_scale * target_delta_penalty
            - float(getattr(self.cfg, "tabletop_affordance_negative_penalty_scale", 0.0))
            * tabletop_affordance["neg_score"]
            - float(getattr(self.cfg, "falling_affordance_negative_penalty_scale", 0.0))
            * falling_affordance["neg_score"]
            - float(getattr(self.cfg, "tabletop_arm_clearance_penalty_scale", 0.0))
            * self._tabletop_arm_clearance_penalty
            - self.cfg.tabletop_no_lift_after_grasp_penalty_scale * no_lift_after_grasp_penalty
            - self.cfg.tabletop_lift_without_object_penalty_scale * tabletop_lift_without_object_penalty
            - float(getattr(self.cfg, "tabletop_lift_without_current_grasp_penalty_scale", 0.0))
            * tabletop_lift_without_current_grasp_penalty
            - float(getattr(self.cfg, "tabletop_strict_grasp_loss_penalty_scale", 0.0))
            * tabletop_strict_grasp_loss_penalty
            - float(getattr(self.cfg, "tabletop_arm_object_lift_gap_penalty_scale", 0.0))
            * tabletop_arm_object_lift_gap_penalty
            - float(getattr(self.cfg, "tabletop_object_carry_stall_penalty_scale", 0.0))
            * tabletop_object_carry_stall_penalty
            - float(getattr(self.cfg, "tabletop_force_grasp_loss_penalty_scale", 0.0))
            * tabletop_force_grasp_loss_penalty
            - self.cfg.tabletop_hover_linear_penalty_scale * hover_linear_error
            - self.cfg.tabletop_hover_overshoot_penalty_scale
            * hover_z_overshoot_ratio
            * hover_z_overshoot_ratio
            * hover_latched
            * reward_true_grasp.float()
            - self.cfg.tabletop_hover_z_vel_penalty_scale
            * hover_z_vel
            * hover_z_vel
            * hover_latched
            * reward_true_grasp.float()
            * torch.clamp(0.35 + 0.65 * lift_progress, 0.0, 1.0)
            - self.cfg.tabletop_hover_vel_penalty_scale
            * (object_speed * object_speed + 0.02 * object_ang_speed * object_ang_speed)
            * reward_true_grasp.float()
            * lift_progress
            - self.cfg.tabletop_hover_target_drift_penalty_scale
            * (hover_xy_dist * hover_xy_dist + hover_z_error * hover_z_error)
            * hover_latched
            * torch.clamp(0.35 + 0.65 * lift_progress, 0.0, 1.0)
            - self.cfg.tabletop_hover_grasp_loss_penalty_scale * hover_grasp_loss
            - self.cfg.tabletop_hover_under_height_penalty_scale
            * hover_under_height_ratio
            * hover_under_height_ratio
            * hover_latch_phase
            - self.cfg.tabletop_hover_post_latch_speed_penalty_scale * hover_post_latch_speed
            - self.cfg.tabletop_hover_post_latch_action_penalty_scale * policy_action_penalty * hover_latch_phase
            - self.cfg.tabletop_hover_post_latch_target_delta_penalty_scale
            * target_delta_penalty
            * hover_latch_phase
            - self.cfg.tabletop_post_success_unstable_penalty_scale * post_success_unstable
            - self.cfg.tabletop_post_success_grasp_loss_penalty_scale * post_success_grasp_loss
            - self.cfg.tabletop_post_success_under_height_penalty_scale
            * post_success_under_height_ratio
            * post_success_under_height_ratio
            - self.cfg.tabletop_post_success_speed_penalty_scale * post_success_speed
            - self.cfg.tabletop_post_success_action_penalty_scale * policy_action_penalty * post_success_phase
            - self.cfg.tabletop_post_success_target_delta_penalty_scale * target_delta_penalty * post_success_phase
            - float(getattr(self.cfg, "tabletop_post_success_arm_joint_vel_penalty_scale", 0.0))
            * post_success_arm_joint_vel
            - float(getattr(self.cfg, "tabletop_post_success_arm_target_drift_penalty_scale", 0.0))
            * post_success_arm_target_drift
            * post_success_arm_target_drift
            - float(getattr(self.cfg, "tabletop_post_success_palm_drift_penalty_scale", 0.0))
            * post_success_palm_drift_penalty
            * post_success_palm_drift_penalty
            - self.cfg.dynamic_tabletop_low_palm_penalty_scale * pregrasp_low_palm_penalty * (1.0 - lift_progress)
            - float(getattr(self.cfg, "dynamic_tabletop_side_contact_penalty_scale", 0.0))
            * side_contact_penalty
            * (1.0 - lift_progress)
            - float(getattr(self.cfg, "tabletop_non_thumb_without_thumb_penalty_scale", 0.0))
            * non_thumb_without_thumb_penalty
            * torch.clamp(
                torch.maximum(
                    1.0 - lift_progress,
                    torch.full_like(
                        lift_progress,
                        float(getattr(self.cfg, "tabletop_non_thumb_without_thumb_penalty_lift_gate_min", 0.0)),
                    ),
                ),
                0.0,
                1.0,
            )
            - float(getattr(self.cfg, "falling_non_thumb_without_thumb_penalty_scale", 0.0))
            * non_thumb_without_thumb_penalty
            * torch.clamp(
                torch.maximum(
                    1.0 - lift_progress,
                    torch.full_like(
                        lift_progress,
                        float(getattr(self.cfg, "falling_non_thumb_without_thumb_penalty_lift_gate_min", 0.0)),
                    ),
                ),
                0.0,
                1.0,
            )
            - float(getattr(self.cfg, "falling_pinched_rel_vel_penalty_scale", 0.0))
            * falling_pinched_rel_vel_penalty
            - self.cfg.scoop_lift_penalty_scale * self._scoop_lift.float() * lift_progress
            - self.cfg.palm_only_lift_penalty_scale * self._palm_only_lift.float() * lift_progress
            - self.cfg.drop_penalty * dropped.float()
        )

        self.extras["log"] = {
            "palm_distance": palm_distance.mean(),
            "min_surface_dist": self._surface_dist.min(dim=-1).values.mean(),
            "mean_surface_dist": self._surface_dist.mean(),
            "thumb_contact": self._thumb_contact.float().mean(),
            "finger_contacts": self._finger_contact_count.float().mean(),
            "non_thumb_contacts": self._non_thumb_contact_count.float().mean(),
            "strict_thumb_contact": self._strict_thumb_contact.float().mean(),
            "strict_finger_contacts": self._strict_finger_contact_count.float().mean(),
            "strict_non_thumb_contacts": self._strict_non_thumb_contact_count.float().mean(),
            "strict_opposing_contact": self._strict_opposing_contact.float().mean(),
            "strict_true_grasp": self._strict_true_grasp.float().mean(),
            "strict_grasp_seen": self._strict_grasp_seen.float().mean(),
            "opposition_score": self._opposition_score.mean(),
            "weighted_opposition_score": self._weighted_opposition_score.mean(),
            "strict_weighted_opposition_score": self._strict_weighted_opposition_score.mean(),
            "strict_approach_score": self._strict_approach_score.mean(),
            "strict_thumb_approach_score": self._strict_thumb_approach_score.mean(),
            "strict_non_thumb_pair_approach_score": self._strict_non_thumb_pair_approach_score.mean(),
            "strict_multifinger_approach_score": self._strict_multifinger_approach_score.mean(),
            "strict_thumb_touch_score": self._strict_thumb_touch_score.mean(),
            "strict_non_thumb_pair_touch_score": self._strict_non_thumb_pair_touch_score.mean(),
            "strict_multifinger_touch_score": self._strict_multifinger_touch_score.mean(),
            "strict_opposition_approach_score": self._strict_opposition_approach_score.mean(),
            "strict_opposition_touch_score": self._strict_opposition_touch_score.mean(),
            "strict_approach_rew": strict_approach_rew.mean(),
            "strict_multifinger_approach_rew": strict_multifinger_approach_rew.mean(),
            "strict_opposition_approach_rew": strict_opposition_approach_rew.mean(),
            "strict_touch_rew": strict_touch_rew.mean(),
            "strict_opposition_touch_rew": strict_opposition_touch_rew.mean(),
            "thumb_pair_contact_score": thumb_pair_contact_score.mean(),
            "opposition_reward_score": opposition_reward_score.mean(),
            "opposing_contact": self._opposing_contact.float().mean(),
            "finger_count_quality": self._finger_count_quality.mean(),
            "non_thumb_quality": self._non_thumb_quality.mean(),
            "grasp_quality": grasp_quality.mean(),
            "strict_grasp_quality": self._strict_grasp_quality.mean(),
            "reward_true_grasp": reward_true_grasp.float().mean(),
            "reward_grasp_seen": reward_grasp_seen.float().mean(),
            "opposition_lift_gate": opposition_lift_gate_signal.mean(),
            "quality_lift_gate": quality_lift_gate.mean(),
            "pregrasp_xy_rew": pregrasp_xy_rew.mean(),
            "pregrasp_height_rew": pregrasp_height_rew.mean(),
            "pregrasp_low_palm_penalty": pregrasp_low_palm_penalty.mean(),
            "side_contact_penalty": side_contact_penalty.mean(),
            "non_thumb_without_thumb_penalty": non_thumb_without_thumb_penalty.mean(),
            "falling_affordance_positive_rew": falling_affordance_positive_rew.mean(),
            "falling_affordance_thumb_geom_rew": falling_affordance_thumb_geom_rew.mean(),
            "falling_affordance_thumb_touch_rew": falling_affordance_thumb_touch_rew.mean(),
            "falling_affordance_thumb_pos_contact": falling_affordance["thumb_pos_contact"].mean(),
            "falling_stable_grasp_rew": falling_stable_grasp_rew.mean(),
            "falling_palm_gate_rew": falling_palm_gate_rew.mean(),
            "falling_positive_stable_rew": falling_positive_stable_rew.mean(),
            "falling_soft_success_progress_rew": falling_soft_success_progress_rew.mean(),
            "falling_opposed_stable_pinch_rew": falling_opposed_stable_pinch_rew.mean(),
            "falling_pinched_rel_vel_penalty": falling_pinched_rel_vel_penalty.mean(),
            "pregrasp_contact_gate": pregrasp_contact_gate.mean(),
            "tabletop_clearance_contact_gate": clearance_contact_gate.mean(),
            "palm_pregrasp_xy_dist": palm_pregrasp_xy_dist.mean(),
            "pregrasp_height_delta": pregrasp_height_delta.mean(),
            "palm_surface_dist": self._palm_surface_dist.mean(),
            "palm_contact": self._palm_contact.float().mean(),
            "scoop_lift": self._scoop_lift.float().mean(),
            "palm_only_lift": self._palm_only_lift.float().mean(),
            "true_grasp": self._true_grasp.float().mean(),
            "grasp_seen": self._grasp_seen.float().mean(),
            "stable_catch": stable_catch.float().mean(),
            "catch_hold": success_now.float().mean(),
            "success": success.float().mean(),
            "success_now": success_now.float().mean(),
            "success_seen": self._success_seen.float().mean(),
            "success_streak": self._success_streak.float().mean(),
            "catch_progress": catch_progress.mean(),
            "hold_progress": hold_progress.mean(),
            "lift_progress": lift_progress.mean(),
            "hover_latched": hover_latched.mean(),
            "hover_height_progress": hover_height_progress.mean(),
            "hover_xy_dist": hover_xy_dist.mean(),
            "hover_z_error": hover_z_error.mean(),
            "hover_goal_rew": hover_goal_rew.mean(),
            "hover_linear_error": hover_linear_error.mean(),
            "hover_z_overshoot_ratio": hover_z_overshoot_ratio.mean(),
            "hover_under_height_ratio": hover_under_height_ratio.mean(),
            "hover_z_vel": hover_z_vel.mean(),
            "hover_latch_phase": hover_latch_phase.mean(),
            "hover_grasp_loss": hover_grasp_loss.mean(),
            "hover_post_latch_speed": hover_post_latch_speed.mean(),
            "post_success_phase": post_success_phase.mean(),
            "post_success_unstable": post_success_unstable.mean(),
            "post_success_grasp_loss": post_success_grasp_loss.mean(),
            "post_success_under_height_ratio": post_success_under_height_ratio.mean(),
            "post_success_speed": post_success_speed.mean(),
            "post_success_stability_latched": self._post_success_stability_latched.float().mean(),
            "post_success_arm_joint_vel": post_success_arm_joint_vel.mean(),
            "post_success_arm_target_drift": post_success_arm_target_drift.mean(),
            "post_success_palm_drift": post_success_palm_drift.mean(),
            "post_success_palm_drift_penalty": post_success_palm_drift_penalty.mean(),
            "hover_target_rew": hover_target_rew.mean(),
            "hover_stability_rew": hover_stability_rew.mean(),
            "hover_object_speed": object_speed.mean(),
            "hover_object_ang_speed": object_ang_speed.mean(),
            "no_lift_after_grasp_penalty": no_lift_after_grasp_penalty.mean(),
            "tabletop_true_grasp_streak": self._tabletop_true_grasp_streak.float().mean(),
            "tabletop_strict_true_grasp_streak": self._tabletop_strict_true_grasp_streak.float().mean(),
            "grasped_palm_lift_rew": grasped_palm_lift_rew.mean(),
            "tabletop_arm_lift_progress": tabletop_arm_lift_progress.mean(),
            "tabletop_arm_lift_baseline_latched": self._tabletop_arm_lift_baseline_latched.float().mean(),
            "tabletop_arm_lift_baseline_grasp_streak": (
                self._tabletop_arm_lift_baseline_grasp_streak.float().mean()
            ),
            "tabletop_arm_lift_rew": tabletop_arm_lift_rew.mean(),
            "tabletop_arm_object_lift_gap_penalty": tabletop_arm_object_lift_gap_penalty.mean(),
            "tabletop_lift_memory_grasp_gate": tabletop_lift_memory_grasp_gate.mean(),
            "tabletop_lift_action_prior": tabletop_lift_action_prior.mean(),
            "tabletop_lift_action_prior_gate": tabletop_lift_action_prior_gate.mean(),
            "tabletop_lift_action_prior_rew": tabletop_lift_action_prior_rew.mean(),
            "tabletop_strict_hold_rew": tabletop_strict_hold_rew.mean(),
            "tabletop_strict_grasp_loss_penalty": tabletop_strict_grasp_loss_penalty.mean(),
            "tabletop_lift_without_object_penalty": tabletop_lift_without_object_penalty.mean(),
            "tabletop_lift_without_current_grasp_penalty": tabletop_lift_without_current_grasp_penalty.mean(),
            "tabletop_object_up_vel_rew": tabletop_object_up_vel_rew.mean(),
            "tabletop_object_carry_lift_rew": tabletop_object_carry_lift_rew.mean(),
            "tabletop_object_carry_stall_penalty": tabletop_object_carry_stall_penalty.mean(),
            "tabletop_force_grasp_rew": tabletop_force_grasp_rew.mean(),
            "tabletop_force_grasp_streak_rew": tabletop_force_grasp_streak_rew.mean(),
            "tabletop_force_stable_grasp_rew": tabletop_force_stable_grasp_rew.mean(),
            "tabletop_force_grasp_loss_penalty": tabletop_force_grasp_loss_penalty.mean(),
            "tabletop_force_grasp_seen": self._force_grasp_seen.float().mean(),
            "tabletop_underwrap_rew": tabletop_underwrap_rew.mean(),
            "tabletop_underwrap_thumb_score": tabletop_underwrap_thumb_score.mean(),
            "tabletop_underwrap_non_thumb_score": tabletop_underwrap_non_thumb_score.mean(),
            "tabletop_underwrap_pair_score": tabletop_underwrap_pair_score.mean(),
            "tabletop_underwrap_progress_score": tabletop_underwrap_progress_score.mean(),
            "tabletop_underwrap_mean_score": tabletop_underwrap_mean_score.mean(),
            "tabletop_underwrap_min_tip_z_rel": tabletop_underwrap_min_tip_z_rel.mean(),
            "tabletop_underwrap_thumb_z_rel": tabletop_underwrap_thumb_z_rel.mean(),
            "tabletop_underwrap_min_non_thumb_z_rel": tabletop_underwrap_min_non_thumb_z_rel.mean(),
            "tabletop_arm_clearance_penalty": self._tabletop_arm_clearance_penalty.mean(),
            "tabletop_arm_clearance_min_margin": self._tabletop_arm_clearance_min_margin.mean(),
            "tabletop_arm_clearance_active_fraction": self._tabletop_arm_clearance_active_fraction.mean(),
            "tabletop_arm_clearance_ok": self._tabletop_arm_clearance_ok.float().mean(),
            "object_height_delta": self._object_height_delta.mean(),
            "object_z": object_pos_local[:, 2].mean(),
            "object_palm_rel_vel": self._object_palm_rel_vel.mean(),
            "effective_action_norm": torch.norm(self.actions, dim=-1).mean(),
            "policy_action_norm": torch.norm(self.policy_actions, dim=-1).mean(),
            "scripted_action_prior_norm": torch.norm(self._action_prior, dim=-1).mean(),
            "target_delta_penalty": target_delta_penalty.mean(),
            "dropped": dropped.float().mean(),
            "dynamic_speed_curriculum_alpha": torch.tensor(
                self._dynamic_speed_curriculum_alpha(), dtype=torch.float32, device=self.device
            ),
            "dynamic_speed_curriculum_metric": torch.tensor(
                self._dynamic_speed_curriculum_metric_value, dtype=torch.float32, device=self.device
            ),
            "dynamic_speed_curriculum_metric_ema": torch.tensor(
                self._dynamic_speed_curriculum_metric_ema, dtype=torch.float32, device=self.device
            ),
            "robot_material_bind_count": torch.tensor(
                self._robot_material_bind_count, dtype=torch.float32, device=self.device
            ),
            "robot_material_bind_fail_count": torch.tensor(
                self._robot_material_bind_fail_count, dtype=torch.float32, device=self.device
            ),
            "robot_self_collision_filter_pair_count": torch.tensor(
                self._robot_self_collision_filter_pair_count, dtype=torch.float32, device=self.device
            ),
            "robot_self_collision_filter_fail_count": torch.tensor(
                self._robot_self_collision_filter_fail_count, dtype=torch.float32, device=self.device
            ),
        }
        if self.cfg.task_family == "dynamic_tabletop_grasp":
            tabletop_speed_min, tabletop_speed_max = self._tabletop_current_speed_range()
            tabletop_yaw_min, tabletop_yaw_max = self._tabletop_current_yaw_rate_range()
            tabletop_logs = {
                "tabletop_cmd_speed": torch.norm(self._tabletop_cmd_lin_vel_w[:, :2], dim=-1).mean(),
                "tabletop_speed_min": tabletop_speed_min.mean(),
                "tabletop_speed_max": tabletop_speed_max.mean(),
                "tabletop_yaw_rate_min": tabletop_yaw_min.mean(),
                "tabletop_yaw_rate_max": tabletop_yaw_max.mean(),
            }
            if self._uses_tabletop_asset_set():
                tabletop_logs.update(
                    {
                        "tabletop_asset_curriculum_alpha": torch.tensor(
                            self._tabletop_asset_curriculum_alpha(), dtype=torch.float32, device=self.device
                        ),
                        "tabletop_asset_count": torch.tensor(
                            self._tabletop_current_asset_count(), dtype=torch.float32, device=self.device
                        ),
                        "tabletop_motion_mode_curriculum_alpha": torch.tensor(
                            self._tabletop_motion_mode_curriculum_alpha(), dtype=torch.float32, device=self.device
                        ),
                        "tabletop_motion_mode_count": torch.tensor(
                            self._tabletop_current_motion_mode_count(), dtype=torch.float32, device=self.device
                        ),
                        "tabletop_active_asset_id_mean": self._tabletop_active_asset_ids.float().mean(),
                        "tabletop_active_shape_code_mean": self._active_object_shape_codes_tensor.float().mean(),
                        "tabletop_affordance_positive_fraction": self._active_asset_positive_fraction.mean(),
                        "tabletop_affordance_negative_fraction": self._active_asset_negative_fraction.mean(),
                        "tabletop_affordance_pos_score": tabletop_affordance["pos_score"].mean(),
                        "tabletop_affordance_neg_score": tabletop_affordance["neg_score"].mean(),
                        "tabletop_affordance_pos_contact": tabletop_affordance["pos_contact"].mean(),
                        "tabletop_affordance_neg_contact": tabletop_affordance["neg_contact"].mean(),
                        "tabletop_affordance_pos_min_dist": tabletop_affordance["pos_min_dist"].mean(),
                        "tabletop_affordance_neg_min_dist": tabletop_affordance["neg_min_dist"].mean(),
                        "tabletop_persistent_motion": torch.tensor(
                            float(bool(self.cfg.dynamic_tabletop_persistent_motion)),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                    }
                )
            self.extras["log"].update(tabletop_logs)
        elif self.cfg.task_family == "falling_baton_grasp":
            self.extras["log"].update(
                {
                    "falling_baton_xy_speed": torch.norm(self._object_lin_vel_w[:, :2], dim=-1).mean(),
                    "falling_baton_z_down_speed": torch.clamp(-self._object_lin_vel_w[:, 2], min=0.0).mean(),
                    "falling_baton_ang_speed": torch.norm(self._object_ang_vel_w, dim=-1).mean(),
                    "falling_success_grasp_gate": falling_success_grasp_gate.float().mean(),
                    "falling_success_palm_gate": falling_success_palm_gate.float().mean(),
                    "falling_success_contact_gate": falling_success_contact_gate.float().mean(),
                    "falling_affordance_pos_score": falling_affordance["pos_score"].mean(),
                    "falling_affordance_neg_score": falling_affordance["neg_score"].mean(),
                    "falling_affordance_pos_contact": falling_affordance["pos_contact"].mean(),
                    "falling_affordance_neg_contact": falling_affordance["neg_contact"].mean(),
                    "falling_affordance_pos_min_dist": falling_affordance["pos_min_dist"].mean(),
                    "falling_affordance_neg_min_dist": falling_affordance["neg_min_dist"].mean(),
                }
            )
        success_true_grasp_env = self._true_grasp
        if bool(getattr(self.cfg, "strict_success_enabled", False)):
            success_true_grasp_env = self._strict_true_grasp
        self.extras["success_env"] = success
        self.extras["true_grasp_env"] = success_true_grasp_env
        self.extras["legacy_true_grasp_env"] = self._true_grasp
        self.extras["strict_true_grasp_env"] = self._strict_true_grasp
        self.extras["strict_grasp_seen_env"] = self._strict_grasp_seen
        self.extras["thumb_contact_env"] = self._thumb_contact
        self.extras["non_thumb_contacts_env"] = self._non_thumb_contact_count
        self.extras["finger_contacts_env"] = self._finger_contact_count
        self.extras["strict_thumb_contact_env"] = self._strict_thumb_contact
        self.extras["strict_non_thumb_contacts_env"] = self._strict_non_thumb_contact_count
        self.extras["strict_finger_contacts_env"] = self._strict_finger_contact_count
        self.extras["strict_opposing_contact_env"] = self._strict_opposing_contact
        self.extras["min_surface_dist_env"] = self._surface_dist.min(dim=-1).values
        self.extras["palm_reach_env"] = palm_distance < float(
            getattr(self.cfg, "diagnostic_palm_reach_distance", 0.25)
        )
        self.extras["palm_distance_env"] = palm_distance
        self.extras["object_palm_rel_vel_env"] = self._object_palm_rel_vel
        self.extras["success_streak_env"] = self._success_streak
        self.extras["positive_affordance_contact_env"] = (
            tabletop_affordance["pos_contact"] > 0.5
            if self.cfg.task_family == "dynamic_tabletop_grasp"
            else falling_affordance["pos_contact"] > 0.5
        )
        self.extras["negative_affordance_contact_env"] = (
            tabletop_affordance["neg_contact"] > 0.5
            if self.cfg.task_family == "dynamic_tabletop_grasp"
            else falling_affordance["neg_contact"] > 0.5
        )
        self.extras["force_thumb_contact_env"] = self._force_thumb_contact
        self.extras["force_multifinger_contact_env"] = self._force_multifinger_contact
        self.extras["force_grasp_env"] = self._force_grasp
        self.extras["force_stable_grasp_env"] = self._force_grasp & (
            self._object_palm_rel_vel < self.cfg.stable_object_palm_vel
        )
        self.extras["force_grasp_streak_env"] = self._force_grasp_streak
        self.extras["object_fingertip_force_sum_env"] = self._object_fingertip_contact_forces.sum(dim=-1)
        self.extras["object_fingertip_force_max_env"] = self._object_fingertip_contact_forces.max(dim=-1).values
        self.extras["grasp_seen_env"] = self._grasp_seen
        self.extras["stable_hold_env"] = success_now
        self.extras["strict_lifted_env"] = self._lifted & self._strict_true_grasp
        self.extras["lifted_low_rel_vel_env"] = self._lifted & (
            self._object_palm_rel_vel < self.cfg.stable_object_palm_vel
        )
        self.extras["strict_stable_hold_env"] = self._strict_stable_hold
        self.extras["hover_latched_env"] = self._object_hover_target_latched
        self.extras["strict_stable_hover_latched_env"] = (
            self._strict_stable_hold & self._object_hover_target_latched
        )
        self.extras["strict_stable_hover_target_ok_env"] = self._strict_stable_hold & hover_target_ok
        self.extras["strict_stable_hover_speed_ok_env"] = self._strict_stable_hold & hover_speed_ok
        self.extras["strict_stable_clearance_ok_env"] = (
            self._strict_stable_hold & self._tabletop_arm_clearance_ok
        )
        self.extras["strict_lift_height_delta_env"] = torch.where(
            self._strict_true_grasp,
            self._object_height_delta,
            torch.zeros_like(self._object_height_delta),
        )
        self.extras["strict_stable_height_delta_env"] = torch.where(
            self._strict_stable_hold,
            self._object_height_delta,
            torch.zeros_like(self._object_height_delta),
        )
        self.extras["success_seen_env"] = self._success_seen
        self.extras["lifted_env"] = lift_progress > 0.98
        self.extras[
            "tabletop_arm_lift_baseline_latched_env"
        ] = self._tabletop_arm_lift_baseline_latched
        self.extras["tabletop_arm_lift_progress_env"] = tabletop_arm_lift_progress
        self.extras["tabletop_lift_action_prior_env"] = tabletop_lift_action_prior
        self.extras["tabletop_lift_action_prior_gate_env"] = tabletop_lift_action_prior_gate
        self.extras["tabletop_lift_action_prior_rew_env"] = tabletop_lift_action_prior_rew
        self.extras["tabletop_object_up_vel_rew_env"] = tabletop_object_up_vel_rew
        self.extras["tabletop_object_carry_lift_rew_env"] = tabletop_object_carry_lift_rew
        self.extras["tabletop_true_grasp_streak_env"] = self._tabletop_true_grasp_streak
        self.extras["tabletop_strict_true_grasp_streak_env"] = self._tabletop_strict_true_grasp_streak
        self.extras["strict_grasp_hold_env"] = self._tabletop_strict_true_grasp_streak >= int(
            getattr(self.cfg, "tabletop_strict_grasp_hold_steps", 20)
        )
        self.extras["object_height_delta_env"] = self._object_height_delta
        self.extras["tabletop_arm_clearance_ok_env"] = self._tabletop_arm_clearance_ok
        self.extras["force_grasp_clearance_ok_env"] = self._force_grasp & self._tabletop_arm_clearance_ok
        self.extras["tabletop_arm_clearance_penalty_env"] = self._tabletop_arm_clearance_penalty
        self.extras["tabletop_arm_clearance_min_margin_env"] = self._tabletop_arm_clearance_min_margin
        self.extras["tabletop_arm_clearance_active_fraction_env"] = self._tabletop_arm_clearance_active_fraction
        self.extras["tabletop_underwrap_thumb_score_env"] = tabletop_underwrap_thumb_score
        self.extras["tabletop_underwrap_non_thumb_score_env"] = tabletop_underwrap_non_thumb_score
        self.extras["tabletop_underwrap_pair_score_env"] = tabletop_underwrap_pair_score
        self.extras["tabletop_underwrap_progress_score_env"] = tabletop_underwrap_progress_score
        self.extras["tabletop_underwrap_mean_score_env"] = tabletop_underwrap_mean_score
        self.extras["tabletop_underwrap_min_tip_z_rel_env"] = tabletop_underwrap_min_tip_z_rel
        self.extras["tabletop_underwrap_thumb_z_rel_env"] = tabletop_underwrap_thumb_z_rel
        self.extras["tabletop_underwrap_min_non_thumb_z_rel_env"] = tabletop_underwrap_min_non_thumb_z_rel
        self._strict_reward_grasp_prev.copy_(reward_true_grasp)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.task_family == "dynamic_tabletop_grasp":
            self._apply_tabletop_persistent_motion()
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        object_pos_local = self._object_pos_w - self.scene.env_origins
        if self.cfg.task_family == "dynamic_tabletop_grasp":
            dropped = object_pos_local[:, 2] < self.cfg.table_top_z - 0.04
        else:
            dropped = object_pos_local[:, 2] < self.cfg.falling_drop_z
        out_xy = torch.any(torch.abs(object_pos_local[:, :2]) > self.cfg.workspace_xy_limit, dim=-1)
        success = self._success_streak >= self.cfg.dynamic_success_hold_steps
        terminated = dropped | out_xy
        clearance_violation = torch.zeros_like(terminated)
        if (
            self.cfg.task_family == "dynamic_tabletop_grasp"
            and bool(getattr(self.cfg, "tabletop_terminate_on_arm_clearance_violation", False))
        ):
            start_step = int(getattr(self.cfg, "tabletop_arm_clearance_violation_terminate_start_step", 0))
            terminate_threshold = max(
                float(
                    getattr(
                        self.cfg,
                        "tabletop_arm_clearance_terminate_penalty_threshold",
                        getattr(self.cfg, "tabletop_arm_clearance_ok_penalty_threshold", 1.0e-4),
                    )
                ),
                0.0,
            )
            clearance_violation = self._tabletop_arm_clearance_penalty > terminate_threshold
            if start_step > 0:
                clearance_violation = clearance_violation & (self.episode_length_buf >= start_step)
            terminated = terminated | clearance_violation
        if self.cfg.terminate_on_success:
            terminated = terminated | success
        self.extras["dropped_env"] = dropped
        self.extras["out_xy_env"] = out_xy
        self.extras["tabletop_arm_clearance_violation_env"] = clearance_violation
        self.extras["success_env"] = success
        self.extras["time_out_env"] = time_out
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if self._uses_tabletop_asset_set():
            self._reset_tabletop_asset_objects(env_ids)
        else:
            object_state = self.object.data.default_root_state[env_ids].clone()
            if self.cfg.task_family == "falling_baton_grasp":
                object_state[:, 0:3] = self._sample_falling_baton_root_pos(env_ids)
            else:
                if self._object_start_pos.shape[0] == self.num_envs:
                    object_start_pos = self._object_start_pos[env_ids]
                else:
                    object_start_pos = self._object_start_pos.expand(len(env_ids), -1)
                object_state[:, 0:3] = object_start_pos + self.scene.env_origins[env_ids]
                reset_noise = torch.tensor(self.cfg.reset_object_pos_noise, device=self.device).unsqueeze(0)
                object_state[:, 0:3] += sample_uniform(-1.0, 1.0, (len(env_ids), 3), self.device) * reset_noise
                if self.cfg.task_family == "dynamic_tabletop_grasp":
                    env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
                    self._active_object_start_z[env_ids_tensor] = object_state[:, 2] - self.scene.env_origins[env_ids_tensor, 2]
                    self._set_tabletop_hover_targets(env_ids_tensor, object_state[:, 0:3])
            object_state[:, 3:7] = self._object_start_rot
            if self.cfg.task_family == "dynamic_tabletop_grasp" and bool(self.cfg.dynamic_tabletop_randomize_yaw):
                yaw = sample_uniform(-torch.pi, torch.pi, (len(env_ids), 1), self.device)
                object_state[:, 3:7] = torch.cat(
                    (
                        torch.cos(0.5 * yaw),
                        torch.zeros_like(yaw),
                        torch.zeros_like(yaw),
                        torch.sin(0.5 * yaw),
                    ),
                    dim=-1,
                )
            elif self.cfg.task_family == "falling_baton_grasp" and bool(
                getattr(self.cfg, "falling_baton_randomize_orientation", False)
            ):
                (
                    (roll_min, roll_max),
                    (pitch_min, pitch_max),
                    (yaw_min, yaw_max),
                ) = self._falling_baton_current_orientation_ranges()
                roll = sample_uniform(float(roll_min), float(roll_max), (len(env_ids), 1), self.device)
                pitch = sample_uniform(float(pitch_min), float(pitch_max), (len(env_ids), 1), self.device)
                yaw = sample_uniform(float(yaw_min), float(yaw_max), (len(env_ids), 1), self.device)
                object_state[:, 3:7] = _quat_from_euler_xyz_wxyz(roll, pitch, yaw)

            lin_min = torch.tensor(self.cfg.object_lin_vel_min, device=self.device).unsqueeze(0)
            lin_max = torch.tensor(self.cfg.object_lin_vel_max, device=self.device).unsqueeze(0)
            ang_min = torch.tensor(self.cfg.object_ang_vel_min, device=self.device).unsqueeze(0)
            ang_max = torch.tensor(self.cfg.object_ang_vel_max, device=self.device).unsqueeze(0)
            if self.cfg.task_family == "falling_baton_grasp":
                xy_speed_min, xy_speed_max = self._falling_baton_current_xy_speed_range(lin_min, lin_max)
                xy_speeds = xy_speed_min + sample_uniform(0.0, 1.0, (len(env_ids), 1), self.device) * (
                    xy_speed_max - xy_speed_min
                )
                headings = sample_uniform(-torch.pi, torch.pi, (len(env_ids), 1), self.device)
                object_state[:, 7:9] = torch.cat((torch.cos(headings) * xy_speeds, torch.sin(headings) * xy_speeds), dim=-1)
                z_down_min, z_down_max = self._falling_baton_current_z_down_speed_range(lin_min, lin_max)
                z_down_speed = z_down_min + sample_uniform(0.0, 1.0, (len(env_ids), 1), self.device) * (
                    z_down_max - z_down_min
                )
                object_state[:, 9:10] = -z_down_speed
                ang_min, ang_max = self._falling_baton_current_ang_vel_range(ang_min, ang_max)
            else:
                object_state[:, 7:10] = lin_min + sample_uniform(0.0, 1.0, (len(env_ids), 3), self.device) * (
                    lin_max - lin_min
                )
                if self.cfg.task_family == "dynamic_tabletop_grasp":
                    cmd_lin_vel, cmd_yaw_rate = self._sample_tabletop_motion_command(len(env_ids))
                    object_state[:, 7:10] = cmd_lin_vel
                    object_state[:, 10:12] = 0.0
                    object_state[:, 12] = cmd_yaw_rate
                    self._tabletop_cmd_lin_vel_w[env_ids] = cmd_lin_vel
                    self._tabletop_cmd_yaw_rate[env_ids] = cmd_yaw_rate
            object_state[:, 10:13] = ang_min + sample_uniform(0.0, 1.0, (len(env_ids), 3), self.device) * (ang_max - ang_min)
            if self.cfg.task_family == "dynamic_tabletop_grasp":
                object_state[:, 10:12] = 0.0
                object_state[:, 12] = self._tabletop_cmd_yaw_rate[env_ids]
            self.object.write_root_pose_to_sim(object_state[:, :7], env_ids)
            self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids)

        self.actions[env_ids] = 0.0
        self.policy_actions[env_ids] = 0.0
        self._action_prior[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._success_streak[env_ids] = 0
        self._success_seen[env_ids] = False
        self._post_success_stability_latched[env_ids] = False
        self._force_thumb_contact[env_ids] = False
        self._force_multifinger_contact[env_ids] = False
        self._force_grasp[env_ids] = False
        self._force_grasp_prev[env_ids] = False
        self._force_grasp_streak[env_ids] = 0
        self._force_grasp_seen[env_ids] = False
        self._scripted_relative_lift_target_latched[env_ids] = False
        self._scripted_relative_lift_target_arm_pos[env_ids] = 0.0
        self._grasp_seen[env_ids] = False
        self._strict_grasp_seen[env_ids] = False
        self._tabletop_true_grasp_streak[env_ids] = 0
        self._tabletop_strict_true_grasp_streak[env_ids] = 0
        self._strict_reward_grasp_prev[env_ids] = False
        self._tabletop_arm_lift_baseline_grasp_streak[env_ids] = 0
        if self._tabletop_arm_lift_baseline_mode == "fixed":
            self._tabletop_arm_lift_baseline_pos[env_ids] = self._tabletop_arm_lift_fixed_baseline_pos
            self._tabletop_arm_lift_baseline_latched[env_ids] = True
        else:
            self._tabletop_arm_lift_baseline_pos[env_ids] = self.robot.data.joint_pos[env_ids][
                :, self._arm_joint_ids
            ]
            self._tabletop_arm_lift_baseline_latched[env_ids] = False
        self._scripted_lift_grasp_recent_steps[env_ids] = 0
        self._compute_intermediate_values()
        self._post_success_arm_joint_target[env_ids] = self._joint_targets[env_ids][:, self._arm_joint_ids]
        self._post_success_hand_joint_target[env_ids] = self._joint_targets[env_ids][:, self._control_hand_joint_ids]
        self._post_success_palm_pos_w[env_ids] = self._palm_pos_w[env_ids]
        self._palm_start_z[env_ids] = self._palm_pos_w[env_ids, 2]
        self._palm_min_z[env_ids] = self._palm_pos_w[env_ids, 2]

    def _reset_tabletop_asset_objects(self, env_ids: Sequence[int]) -> None:
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        count = len(env_ids_tensor)
        origins = self.scene.env_origins[env_ids_tensor]

        active_asset_ids = self._sample_tabletop_asset_ids(count)
        motion_mode_ids = self._sample_tabletop_motion_mode_ids(count)
        self._tabletop_active_asset_ids[env_ids_tensor] = active_asset_ids
        self._tabletop_motion_mode_ids[env_ids_tensor] = motion_mode_ids
        self._update_active_tabletop_asset_tensors(env_ids_tensor)

        local_pos = torch.tensor(self.cfg.object_start_pos, dtype=torch.float32, device=self.device).view(1, 3).expand(
            count, -1
        ).clone()
        reset_noise = torch.tensor(self.cfg.reset_object_pos_noise, dtype=torch.float32, device=self.device).view(1, 3)
        local_pos += sample_uniform(-1.0, 1.0, (count, 3), self.device) * reset_noise

        shape_codes = self._tabletop_asset_shape_codes[active_asset_ids]
        support_heights = self._tabletop_asset_support_heights[active_asset_ids]
        free_rolling = not bool(self.cfg.dynamic_tabletop_persistent_motion)
        if free_rolling:
            horizontal = (shape_codes == 2) | (shape_codes == 3)
            support_heights = torch.where(horizontal, self._tabletop_asset_radii[active_asset_ids], support_heights)
        start_z = float(self.cfg.table_top_z) + support_heights + 0.002
        self._active_object_start_z[env_ids_tensor] = start_z
        local_pos[:, 2] = start_z

        turntable_mask = self._motion_mode_mask(motion_mode_ids, "turntable")
        if torch.any(turntable_mask):
            center = torch.tensor(self.cfg.tabletop_turntable_center, dtype=torch.float32, device=self.device).view(1, 2)
            r_min, r_max = self.cfg.tabletop_turntable_radius_range
            theta = sample_uniform(-torch.pi, torch.pi, (int(turntable_mask.sum().item()), 1), self.device)
            radius = sample_uniform(float(r_min), float(r_max), (int(turntable_mask.sum().item()), 1), self.device)
            local_pos[turntable_mask, 0:2] = center + torch.cat((torch.cos(theta), torch.sin(theta)), dim=-1) * radius

        object_state = self._tabletop_objects[0].data.default_root_state[env_ids_tensor].clone()
        object_state[:, 0:3] = local_pos + origins
        self._set_tabletop_hover_targets(env_ids_tensor, object_state[:, 0:3])
        object_state[:, 3:7] = self._sample_tabletop_asset_orientation(active_asset_ids, free_rolling=free_rolling)
        cmd_lin_vel, cmd_yaw_rate = self._sample_tabletop_motion_command(
            count,
            motion_mode_ids=motion_mode_ids,
            object_pos_local=local_pos,
        )
        object_state[:, 7:10] = cmd_lin_vel
        if free_rolling:
            object_state[:, 10:13] = self._sample_tabletop_free_roll_ang_vel(cmd_lin_vel, cmd_yaw_rate, active_asset_ids)
        else:
            object_state[:, 10:12] = 0.0
            object_state[:, 12] = cmd_yaw_rate
        self._tabletop_cmd_lin_vel_w[env_ids_tensor] = cmd_lin_vel
        self._tabletop_cmd_yaw_rate[env_ids_tensor] = cmd_yaw_rate

        for asset_index, obj in enumerate(self._tabletop_objects):
            state = obj.data.default_root_state[env_ids_tensor].clone()
            state[:, 0:3] = origins + self._tabletop_inactive_asset_parking_local_pos(asset_index, count)
            state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
            state[:, 7:] = 0.0
            active = active_asset_ids == asset_index
            if torch.any(active):
                state[active] = object_state[active]
            obj.write_root_pose_to_sim(state[:, :7], env_ids_tensor)
            obj.write_root_velocity_to_sim(state[:, 7:], env_ids_tensor)

    def _tabletop_inactive_asset_parking_local_pos(self, asset_index: int, count: int) -> torch.Tensor:
        parking = torch.tensor(
            (100.0 + 2.0 * float(asset_index), 100.0, 10.0 + 0.5 * float(asset_index)),
            dtype=torch.float32,
            device=self.device,
        )
        return parking.view(1, 3).expand(count, -1)

    def _park_inactive_tabletop_asset_objects(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if len(self._tabletop_objects) <= 1:
            return
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids_tensor) == 0:
            return

        origins = self.scene.env_origins[env_ids_tensor]
        active_asset_ids = self._tabletop_active_asset_ids[env_ids_tensor]
        for asset_index, obj in enumerate(self._tabletop_objects):
            inactive = active_asset_ids != asset_index
            if not torch.any(inactive):
                continue
            inactive_env_ids = env_ids_tensor[inactive]
            state = obj.data.default_root_state[inactive_env_ids].clone()
            state[:, 0:3] = origins[inactive] + self._tabletop_inactive_asset_parking_local_pos(
                asset_index, int(inactive.sum().item())
            )
            state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
            state[:, 7:] = 0.0
            obj.write_root_pose_to_sim(state[:, :7], inactive_env_ids)
            obj.write_root_velocity_to_sim(state[:, 7:], inactive_env_ids)

    def _motion_mode_mask(self, motion_mode_ids: torch.Tensor, mode_name: str) -> torch.Tensor:
        mode_names = tuple(str(name).lower() for name in getattr(self.cfg, "tabletop_motion_modes", ("linear",)))
        mask = torch.zeros_like(motion_mode_ids, dtype=torch.bool)
        for index, name in enumerate(mode_names):
            if name == mode_name:
                mask = mask | (motion_mode_ids == index)
        return mask

    def _sample_tabletop_asset_orientation(self, asset_ids: torch.Tensor, *, free_rolling: bool) -> torch.Tensor:
        yaw = sample_uniform(-torch.pi, torch.pi, (len(asset_ids), 1), self.device)
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
        if free_rolling:
            shape_codes = self._tabletop_asset_shape_codes[asset_ids]
            horizontal = ((shape_codes == 2) | (shape_codes == 3)).unsqueeze(-1)
            pitch = torch.where(horizontal, torch.full_like(pitch, 0.5 * torch.pi), pitch)
        return _quat_from_euler_xyz_wxyz(roll, pitch, yaw)

    def _sample_tabletop_free_roll_ang_vel(
        self,
        lin_vel: torch.Tensor,
        yaw_rate: torch.Tensor,
        asset_ids: torch.Tensor,
    ) -> torch.Tensor:
        radius = torch.clamp(self._tabletop_asset_radii[asset_ids], min=0.012).unsqueeze(-1)
        ang_vel = torch.zeros((len(asset_ids), 3), dtype=torch.float32, device=self.device)
        ang_vel[:, 0:1] = -lin_vel[:, 1:2] / radius
        ang_vel[:, 1:2] = lin_vel[:, 0:1] / radius
        ang_vel[:, 2] = yaw_rate
        return ang_vel

    def _dynamic_speed_curriculum_alpha(self) -> float:
        override_alpha = getattr(self.cfg, "dynamic_grasp_speed_curriculum_override_alpha", None)
        if override_alpha is not None:
            return min(max(float(override_alpha), 0.0), 1.0)
        if not bool(self.cfg.dynamic_grasp_speed_curriculum):
            return 1.0
        mode = str(self.cfg.dynamic_grasp_speed_curriculum_mode).lower()
        if mode in {"success", "success_gate", "performance"}:
            return float(self._dynamic_speed_curriculum_success_alpha)
        steps = int(self.cfg.dynamic_grasp_speed_curriculum_steps)
        if steps <= 0:
            return 1.0
        return self._frame_curriculum_alpha(steps)

    def _update_dynamic_speed_curriculum(
        self,
        *,
        success: torch.Tensor,
        true_grasp: torch.Tensor,
        strict_true_grasp: torch.Tensor,
        stable_catch: torch.Tensor,
        catch_hold: torch.Tensor,
        contact_like: torch.Tensor,
    ) -> None:
        if not bool(self.cfg.dynamic_grasp_speed_curriculum):
            return
        mode = str(self.cfg.dynamic_grasp_speed_curriculum_mode).lower()
        if mode not in {"success", "success_gate", "performance"}:
            return

        metric_tensors = {
            "success": success.float(),
            "true_grasp": true_grasp.float(),
            "strict_true_grasp": strict_true_grasp.float(),
            "stable_catch": stable_catch.float(),
            "catch_hold": catch_hold.float(),
            "contact": contact_like.float(),
            "contact_like": contact_like.float(),
        }
        metric = str(self.cfg.dynamic_grasp_speed_curriculum_metric).lower()
        metric_value = float(metric_tensors.get(metric, metric_tensors["success"]).mean().item())
        ema_alpha = min(max(float(self.cfg.dynamic_grasp_speed_curriculum_ema_alpha), 0.0), 1.0)
        if not self._dynamic_speed_curriculum_metric_initialized:
            metric_ema = metric_value
            self._dynamic_speed_curriculum_metric_initialized = True
        else:
            metric_ema = (
                (1.0 - ema_alpha) * self._dynamic_speed_curriculum_metric_ema
                + ema_alpha * metric_value
            )

        start = float(self.cfg.dynamic_grasp_speed_curriculum_start_success)
        full = float(self.cfg.dynamic_grasp_speed_curriculum_full_success)
        if full <= start:
            target_alpha = 1.0 if metric_ema >= start else 0.0
        else:
            target_alpha = min(max((metric_ema - start) / (full - start), 0.0), 1.0)

        current_alpha = float(self._dynamic_speed_curriculum_success_alpha)
        rise = max(float(self.cfg.dynamic_grasp_speed_curriculum_alpha_rise), 0.0)
        if target_alpha > current_alpha and rise > 0.0:
            next_alpha = min(target_alpha, current_alpha + rise)
        elif target_alpha > current_alpha:
            next_alpha = target_alpha
        elif bool(self.cfg.dynamic_grasp_speed_curriculum_allow_decrease):
            next_alpha = target_alpha
        else:
            next_alpha = current_alpha

        self._dynamic_speed_curriculum_metric_value = metric_value
        self._dynamic_speed_curriculum_metric_ema = metric_ema
        self._dynamic_speed_curriculum_success_alpha = next_alpha

    def _lerp_scalar_pair(
        self, start_pair: tuple[float, float], target_min: torch.Tensor, target_max: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self._dynamic_speed_curriculum_alpha()
        start = torch.tensor(start_pair, device=self.device, dtype=target_min.dtype).view(1, 2)
        start_min = torch.minimum(start[:, 0:1], start[:, 1:2])
        start_max = torch.maximum(start[:, 0:1], start[:, 1:2])
        return (
            start_min + (target_min - start_min) * alpha,
            start_max + (target_max - start_max) * alpha,
        )

    def _falling_baton_current_xy_speed_range(
        self, lin_min: torch.Tensor, lin_max: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_min = torch.minimum(lin_min[:, 0:1], lin_max[:, 0:1])
        target_max = torch.maximum(lin_min[:, 0:1], lin_max[:, 0:1])
        return self._lerp_scalar_pair(self.cfg.falling_baton_start_initial_xy_speed_range, target_min, target_max)

    def _falling_baton_current_z_down_speed_range(
        self, lin_min: torch.Tensor, lin_max: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_min = torch.minimum(torch.abs(lin_min[:, 2:3]), torch.abs(lin_max[:, 2:3]))
        target_max = torch.maximum(torch.abs(lin_min[:, 2:3]), torch.abs(lin_max[:, 2:3]))
        return self._lerp_scalar_pair(self.cfg.falling_baton_start_initial_z_speed_range, target_min, target_max)

    def _falling_baton_current_ang_vel_range(
        self, ang_min: torch.Tensor, ang_max: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_min = torch.minimum(ang_min[:, 0:1], ang_max[:, 0:1])
        target_max = torch.maximum(ang_min[:, 0:1], ang_max[:, 0:1])
        current_min, current_max = self._lerp_scalar_pair(
            self.cfg.falling_baton_start_initial_ang_vel_range,
            target_min,
            target_max,
        )
        return current_min.expand_as(ang_min), current_max.expand_as(ang_max)

    def _lerp_float_pair(self, start_pair: tuple[float, float], target_pair: tuple[float, float]) -> tuple[float, float]:
        alpha = self._dynamic_speed_curriculum_alpha()
        start_min = min(float(start_pair[0]), float(start_pair[1]))
        start_max = max(float(start_pair[0]), float(start_pair[1]))
        target_min = min(float(target_pair[0]), float(target_pair[1]))
        target_max = max(float(target_pair[0]), float(target_pair[1]))
        return (
            start_min + (target_min - start_min) * alpha,
            start_max + (target_max - start_max) * alpha,
        )

    def _falling_baton_current_spawn_above_palm_range(self) -> tuple[float, float]:
        target_pair = self.cfg.falling_baton_spawn_above_palm_range
        if not bool(getattr(self.cfg, "falling_baton_spawn_height_curriculum", False)):
            return float(target_pair[0]), float(target_pair[1])
        start_pair = getattr(self.cfg, "falling_baton_start_spawn_above_palm_range", target_pair)
        return self._lerp_float_pair(start_pair, target_pair)

    def _falling_baton_current_orientation_ranges(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        roll_pair = self.cfg.falling_baton_roll_range
        pitch_pair = self.cfg.falling_baton_pitch_range
        yaw_pair = self.cfg.falling_baton_yaw_range
        if not bool(getattr(self.cfg, "falling_baton_orientation_curriculum", False)):
            return (
                (float(roll_pair[0]), float(roll_pair[1])),
                (float(pitch_pair[0]), float(pitch_pair[1])),
                (float(yaw_pair[0]), float(yaw_pair[1])),
            )
        return (
            self._lerp_float_pair(getattr(self.cfg, "falling_baton_start_roll_range", roll_pair), roll_pair),
            self._lerp_float_pair(getattr(self.cfg, "falling_baton_start_pitch_range", pitch_pair), pitch_pair),
            self._lerp_float_pair(getattr(self.cfg, "falling_baton_start_yaw_range", yaw_pair), yaw_pair),
        )

    def _tabletop_current_speed_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.tensor(self.cfg.dynamic_tabletop_initial_speed_range, device=self.device).view(1, 2)
        return self._lerp_scalar_pair(self.cfg.dynamic_tabletop_start_speed_range, target[:, 0:1], target[:, 1:2])

    def _tabletop_current_yaw_rate_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.tensor(self.cfg.dynamic_tabletop_initial_yaw_rate_range, device=self.device).view(1, 2)
        return self._lerp_scalar_pair(self.cfg.dynamic_tabletop_start_yaw_rate_range, target[:, 0:1], target[:, 1:2])

    def _sample_tabletop_motion_command(
        self,
        count: int,
        motion_mode_ids: torch.Tensor | None = None,
        object_pos_local: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        speed_min, speed_max = self._tabletop_current_speed_range()
        yaw_min, yaw_max = self._tabletop_current_yaw_rate_range()
        if bool(getattr(self.cfg, "dynamic_tabletop_speed_alpha_sample_enabled", False)):
            curriculum_alpha = float(self._dynamic_speed_curriculum_alpha())
            alpha_min = min(max(float(getattr(self.cfg, "dynamic_tabletop_speed_alpha_sample_min", 0.0)), 0.0), 1.0)
            alpha_max = min(max(float(getattr(self.cfg, "dynamic_tabletop_speed_alpha_sample_max", 1.0)), 0.0), 1.0)
            if bool(getattr(self.cfg, "dynamic_tabletop_speed_alpha_sample_curriculum_cap", True)):
                alpha_max = min(alpha_max, curriculum_alpha)
            alpha_min = min(alpha_min, alpha_max)
            alpha = alpha_min + sample_uniform(0.0, 1.0, (count, 1), self.device) * (alpha_max - alpha_min)
            full_fraction = min(
                max(float(getattr(self.cfg, "dynamic_tabletop_speed_alpha_sample_full_fraction", 0.0)), 0.0),
                1.0,
            )
            if full_fraction > 0.0 and alpha_max > alpha_min:
                full_mask = sample_uniform(0.0, 1.0, (count, 1), self.device) < full_fraction
                alpha = torch.where(full_mask, torch.full_like(alpha, alpha_max), alpha)

            speed_start = torch.tensor(
                self.cfg.dynamic_tabletop_start_speed_range,
                dtype=torch.float32,
                device=self.device,
            ).view(1, 2)
            speed_target = torch.tensor(
                self.cfg.dynamic_tabletop_initial_speed_range,
                dtype=torch.float32,
                device=self.device,
            ).view(1, 2)
            speed_start_min = torch.minimum(speed_start[:, 0:1], speed_start[:, 1:2])
            speed_start_max = torch.maximum(speed_start[:, 0:1], speed_start[:, 1:2])
            speed_target_min = torch.minimum(speed_target[:, 0:1], speed_target[:, 1:2])
            speed_target_max = torch.maximum(speed_target[:, 0:1], speed_target[:, 1:2])
            speed_min = speed_start_min + (speed_target_min - speed_start_min) * alpha
            speed_max = speed_start_max + (speed_target_max - speed_start_max) * alpha

            yaw_start = torch.tensor(
                self.cfg.dynamic_tabletop_start_yaw_rate_range,
                dtype=torch.float32,
                device=self.device,
            ).view(1, 2)
            yaw_target = torch.tensor(
                self.cfg.dynamic_tabletop_initial_yaw_rate_range,
                dtype=torch.float32,
                device=self.device,
            ).view(1, 2)
            yaw_start_min = torch.minimum(yaw_start[:, 0:1], yaw_start[:, 1:2])
            yaw_start_max = torch.maximum(yaw_start[:, 0:1], yaw_start[:, 1:2])
            yaw_target_min = torch.minimum(yaw_target[:, 0:1], yaw_target[:, 1:2])
            yaw_target_max = torch.maximum(yaw_target[:, 0:1], yaw_target[:, 1:2])
            yaw_min = yaw_start_min + (yaw_target_min - yaw_start_min) * alpha
            yaw_max = yaw_start_max + (yaw_target_max - yaw_start_max) * alpha
        speed = speed_min + sample_uniform(0.0, 1.0, (count, 1), self.device) * (speed_max - speed_min)
        heading_min, heading_max = self.cfg.dynamic_tabletop_heading_range
        heading = sample_uniform(float(heading_min), float(heading_max), (count, 1), self.device)
        lin_vel = torch.cat((torch.sin(heading) * speed, torch.cos(heading) * speed, torch.zeros_like(speed)), dim=-1)

        yaw_rate = yaw_min + sample_uniform(0.0, 1.0, (count, 1), self.device) * (yaw_max - yaw_min)
        yaw_rate = yaw_rate.squeeze(-1)
        if motion_mode_ids is None or not self._uses_tabletop_asset_set():
            return lin_vel, yaw_rate

        linear_mask = self._motion_mode_mask(motion_mode_ids, "linear") | self._motion_mode_mask(
            motion_mode_ids, "free"
        )
        curved_mask = self._motion_mode_mask(motion_mode_ids, "curved")
        turntable_mask = self._motion_mode_mask(motion_mode_ids, "turntable")
        yaw_rate = torch.where(linear_mask, torch.zeros_like(yaw_rate), yaw_rate)
        if torch.any(turntable_mask):
            if object_pos_local is None:
                object_pos_local = torch.tensor(
                    self.cfg.object_start_pos, dtype=torch.float32, device=self.device
                ).view(1, 3).expand(count, -1)
            center = torch.tensor(self.cfg.tabletop_turntable_center, dtype=torch.float32, device=self.device).view(1, 2)
            radial = object_pos_local[:, 0:2] - center
            radius = torch.clamp(torch.norm(radial, dim=-1, keepdim=True), min=0.03)
            tangent = torch.cat((-radial[:, 1:2], radial[:, 0:1]), dim=-1) / radius
            sign = torch.where(
                sample_uniform(0.0, 1.0, (count, 1), self.device) > 0.5,
                torch.ones((count, 1), dtype=torch.float32, device=self.device),
                -torch.ones((count, 1), dtype=torch.float32, device=self.device),
            )
            turntable_lin = tangent * speed * sign
            lin_vel[turntable_mask, 0:2] = turntable_lin[turntable_mask]
            yaw_rate[turntable_mask] = (speed.squeeze(-1) / radius.squeeze(-1) * sign.squeeze(-1))[turntable_mask]
        yaw_rate = torch.where(curved_mask | turntable_mask, yaw_rate, yaw_rate)
        return lin_vel, yaw_rate

    def _apply_tabletop_persistent_motion(self) -> None:
        if not bool(self.cfg.dynamic_tabletop_persistent_motion):
            return
        if self._uses_tabletop_asset_set():
            self._apply_tabletop_asset_persistent_motion()
            return
        self._compute_intermediate_values()
        object_pos_local = self._object_pos_w - self.scene.env_origins
        near_table = object_pos_local[:, 2] < (float(self.cfg.table_top_z) + 0.12)
        release_contact = torch.zeros_like(self._grasp_seen)
        if bool(getattr(self.cfg, "dynamic_tabletop_release_motion_on_contact", False)):
            min_contacts = max(float(getattr(self.cfg, "dynamic_tabletop_release_motion_contact_count", 1)), 1.0)
            release_contact = self._finger_contact_count >= min_contacts
        active = near_table & (~self._grasp_seen) & (~release_contact)
        if not torch.any(active):
            return

        cmd_vel = self._tabletop_cmd_lin_vel_w.clone()
        if bool(self.cfg.dynamic_tabletop_bounce_at_workspace):
            x_min, x_max = self.cfg.dynamic_tabletop_workspace_x
            y_min, y_max = self.cfg.dynamic_tabletop_workspace_y
            hit_x_min = (object_pos_local[:, 0] < float(x_min)) & (cmd_vel[:, 0] < 0.0)
            hit_x_max = (object_pos_local[:, 0] > float(x_max)) & (cmd_vel[:, 0] > 0.0)
            hit_y_min = (object_pos_local[:, 1] < float(y_min)) & (cmd_vel[:, 1] < 0.0)
            hit_y_max = (object_pos_local[:, 1] > float(y_max)) & (cmd_vel[:, 1] > 0.0)
            flip_x = hit_x_min | hit_x_max
            flip_y = hit_y_min | hit_y_max
            self._tabletop_cmd_lin_vel_w[flip_x, 0] *= -1.0
            self._tabletop_cmd_lin_vel_w[flip_y, 1] *= -1.0
            cmd_vel = self._tabletop_cmd_lin_vel_w.clone()

        root_vel = self.object.data.root_vel_w.clone()
        root_vel[active, 0:2] = cmd_vel[active, 0:2]
        root_vel[active, 2] = 0.0
        root_vel[active, 3:5] = 0.0
        root_vel[active, 5] = self._tabletop_cmd_yaw_rate[active]
        env_ids = active.nonzero(as_tuple=False).squeeze(-1)
        self.object.write_root_velocity_to_sim(root_vel[env_ids], env_ids)

    def _apply_tabletop_asset_persistent_motion(self) -> None:
        self._compute_intermediate_values()
        object_pos_local = self._object_pos_w - self.scene.env_origins
        near_table = object_pos_local[:, 2] < (float(self.cfg.table_top_z) + 0.12)
        release_contact = torch.zeros_like(self._grasp_seen)
        if bool(getattr(self.cfg, "dynamic_tabletop_release_motion_on_contact", False)):
            min_contacts = max(float(getattr(self.cfg, "dynamic_tabletop_release_motion_contact_count", 1)), 1.0)
            release_contact = self._finger_contact_count >= min_contacts
        active = near_table & (~self._grasp_seen) & (~release_contact)
        if not torch.any(active):
            return

        curved = active & self._motion_mode_mask(self._tabletop_motion_mode_ids, "curved")
        if torch.any(curved):
            angle = self._tabletop_cmd_yaw_rate[curved] * self.dt
            c = torch.cos(angle)
            s = torch.sin(angle)
            vx = self._tabletop_cmd_lin_vel_w[curved, 0].clone()
            vy = self._tabletop_cmd_lin_vel_w[curved, 1].clone()
            self._tabletop_cmd_lin_vel_w[curved, 0] = c * vx - s * vy
            self._tabletop_cmd_lin_vel_w[curved, 1] = s * vx + c * vy

        turntable = active & self._motion_mode_mask(self._tabletop_motion_mode_ids, "turntable")
        if torch.any(turntable):
            center = torch.tensor(self.cfg.tabletop_turntable_center, dtype=torch.float32, device=self.device).view(1, 2)
            radial = object_pos_local[turntable, 0:2] - center
            radius = torch.clamp(torch.norm(radial, dim=-1, keepdim=True), min=0.03)
            tangent = torch.cat((-radial[:, 1:2], radial[:, 0:1]), dim=-1) / radius
            omega = self._tabletop_cmd_yaw_rate[turntable].unsqueeze(-1)
            self._tabletop_cmd_lin_vel_w[turntable, 0:2] = tangent * omega * radius

        cmd_vel = self._tabletop_cmd_lin_vel_w.clone()
        if bool(self.cfg.dynamic_tabletop_bounce_at_workspace):
            bounce_active = active & (~turntable)
            x_min, x_max = self.cfg.dynamic_tabletop_workspace_x
            y_min, y_max = self.cfg.dynamic_tabletop_workspace_y
            hit_x_min = bounce_active & (object_pos_local[:, 0] < float(x_min)) & (cmd_vel[:, 0] < 0.0)
            hit_x_max = bounce_active & (object_pos_local[:, 0] > float(x_max)) & (cmd_vel[:, 0] > 0.0)
            hit_y_min = bounce_active & (object_pos_local[:, 1] < float(y_min)) & (cmd_vel[:, 1] < 0.0)
            hit_y_max = bounce_active & (object_pos_local[:, 1] > float(y_max)) & (cmd_vel[:, 1] > 0.0)
            flip_x = hit_x_min | hit_x_max
            flip_y = hit_y_min | hit_y_max
            self._tabletop_cmd_lin_vel_w[flip_x, 0] *= -1.0
            self._tabletop_cmd_lin_vel_w[flip_y, 1] *= -1.0
            cmd_vel = self._tabletop_cmd_lin_vel_w.clone()

        for asset_index, obj in enumerate(self._tabletop_objects):
            asset_active = active & (self._tabletop_active_asset_ids == asset_index)
            if not torch.any(asset_active):
                continue
            env_ids = asset_active.nonzero(as_tuple=False).squeeze(-1)
            root_vel = obj.data.root_vel_w.clone()
            root_vel[asset_active, 0:2] = cmd_vel[asset_active, 0:2]
            root_vel[asset_active, 2] = 0.0
            root_vel[asset_active, 3:5] = 0.0
            root_vel[asset_active, 5] = self._tabletop_cmd_yaw_rate[asset_active]
            obj.write_root_velocity_to_sim(root_vel[env_ids], env_ids)

    def _sample_falling_baton_root_pos(self, env_ids: Sequence[int]) -> torch.Tensor:
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        num_resets = len(env_ids_tensor)
        origins = self.scene.env_origins[env_ids_tensor]

        x_min, x_max = self.cfg.falling_baton_spawn_x_range
        y_min, y_max = self.cfg.falling_baton_spawn_y_range
        z_min, z_max = self.cfg.falling_baton_spawn_z_range
        local_pos = torch.cat(
            (
                sample_uniform(float(x_min), float(x_max), (num_resets, 1), self.device),
                sample_uniform(float(y_min), float(y_max), (num_resets, 1), self.device),
                sample_uniform(float(z_min), float(z_max), (num_resets, 1), self.device),
            ),
            dim=-1,
        )

        if self.cfg.falling_baton_palm_relative_spawn_enabled:
            catch_ref = self._falling_baton_catch_reference_pos(env_ids_tensor) - origins
            dx_min, dx_max = self.cfg.falling_baton_palm_relative_start_x_range
            dy_min, dy_max = self.cfg.falling_baton_palm_relative_start_y_range
            local_xy = catch_ref[:, 0:2] + torch.cat(
                (
                    sample_uniform(float(dx_min), float(dx_max), (num_resets, 1), self.device),
                    sample_uniform(float(dy_min), float(dy_max), (num_resets, 1), self.device),
                ),
                dim=-1,
            )
            if self.cfg.falling_baton_palm_relative_clamp_to_workspace:
                local_xy[:, 0] = torch.clamp(local_xy[:, 0], min=float(x_min), max=float(x_max))
                local_xy[:, 1] = torch.clamp(local_xy[:, 1], min=float(y_min), max=float(y_max))
            local_pos[:, 0:2] = local_xy

            if self.cfg.falling_baton_spawn_above_palm_enabled:
                above_min, above_max = self._falling_baton_current_spawn_above_palm_range()
                palm_relative_z = catch_ref[:, 2:3] + sample_uniform(
                    float(above_min), float(above_max), (num_resets, 1), self.device
                )
                local_pos[:, 2:3] = torch.maximum(local_pos[:, 2:3], palm_relative_z)

        return local_pos + origins

    def _falling_baton_catch_reference_pos(self, env_ids: torch.Tensor) -> torch.Tensor:
        palm_pos = self._palm_pos_w[env_ids]
        fingertip_center = self._fingertip_pos_w[env_ids].mean(dim=1)
        palm_to_fingers = fingertip_center - palm_pos
        ref_pos = palm_pos + float(self.cfg.falling_baton_catch_center_finger_weight) * palm_to_fingers

        palm_to_fingers_xy = palm_to_fingers[:, 0:2]
        forward_norm = torch.norm(palm_to_fingers_xy, dim=-1, keepdim=True)
        forward_xy = torch.where(
            forward_norm > 1.0e-6,
            palm_to_fingers_xy / torch.clamp(forward_norm, min=1.0e-6),
            torch.zeros_like(palm_to_fingers_xy),
        )
        ref_pos[:, 0:2] = ref_pos[:, 0:2] + float(self.cfg.falling_baton_catch_center_forward_offset) * forward_xy
        world_offset = torch.tensor(self.cfg.falling_baton_catch_center_world_offset, device=self.device).unsqueeze(0)
        return ref_pos + world_offset

    def _get_observations(self) -> dict:
        observations = super()._get_observations()
        policy_obs = observations["policy"]
        if self._uses_tabletop_asset_set() and bool(getattr(self.cfg, "tabletop_asset_obs_enabled", False)):
            policy_obs = torch.cat((policy_obs, self._tabletop_asset_obs), dim=-1)
            observations["policy"] = policy_obs
        if self._uses_tabletop_asset_set() and bool(getattr(self.cfg, "tabletop_hover_target_obs_enabled", False)):
            hover_obs_scale = max(float(getattr(self.cfg, "tabletop_hover_target_obs_scale", 0.16)), 1.0e-6)
            hover_target_delta = (self._object_hover_target_pos_w - self._object_pos_w) / hover_obs_scale
            hover_state_obs = torch.cat(
                (
                    hover_target_delta,
                    self._object_hover_target_latched.float().unsqueeze(-1),
                    self._success_seen.float().unsqueeze(-1),
                ),
                dim=-1,
            )
            policy_obs = torch.cat((policy_obs, hover_state_obs), dim=-1)
            observations["policy"] = policy_obs
        if policy_obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"Observation size mismatch: cfg={self.cfg.observation_space}, actual={policy_obs.shape[-1]}"
            )
        return observations


@torch.jit.script
def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 2.0 * (x - lower) / torch.clamp(upper - lower, min=1.0e-6) - 1.0
