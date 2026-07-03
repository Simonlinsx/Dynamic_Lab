"""IsaacLab DirectRLEnv for dynamic dexterous grasp privileged teachers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
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
            self._tabletop_arm_lift_baseline_pos = self._default_arm_pos
        else:
            self._tabletop_arm_lift_baseline_pos = torch.tensor(
                tabletop_lift_baseline,
                dtype=torch.float32,
                device=self.device,
            ).view(1, -1)
        self._tabletop_true_grasp_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._post_success_stability_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._post_success_arm_joint_target = torch.zeros(
            (self.num_envs, len(self._arm_joint_ids)), dtype=torch.float32, device=self.device
        )
        self._post_success_hand_joint_target = torch.zeros(
            (self.num_envs, len(self._control_hand_joint_ids)), dtype=torch.float32, device=self.device
        )
        self._post_success_palm_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
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
        return torch.randint(0, current_count, (count,), dtype=torch.long, device=self.device)

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

    def _compute_intermediate_values(self) -> None:
        if not self._uses_tabletop_asset_set():
            super()._compute_intermediate_values()
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
        if self.cfg.task_family == "dynamic_tabletop_grasp" and self._tabletop_arm_clearance_body_ids:
            clearance_body_pos = self.robot.data.body_pos_w[:, self._tabletop_arm_clearance_body_ids]
            clearance_body_z = clearance_body_pos[:, :, 2]
            clearance_body_xy = clearance_body_pos[:, :, :2] - self.scene.env_origins[:, None, :2]
            active_xy = torch.all(
                torch.abs(clearance_body_xy - self._tabletop_arm_clearance_xy_center)
                <= (self._tabletop_arm_clearance_xy_half_extent + self._tabletop_arm_clearance_xy_padding),
                dim=-1,
            )
            min_z = (
                self.scene.env_origins[:, 2:3]
                + float(self.cfg.table_top_z)
                + self._tabletop_arm_clearance_body_margins
            )
            clearance_margin = clearance_body_z - min_z
            deficit = torch.where(active_xy, torch.relu(-clearance_margin), torch.zeros_like(clearance_margin))
            scale = max(float(getattr(self.cfg, "tabletop_arm_clearance_scale", 0.060)), 1.0e-6)
            max_penalty = float(getattr(self.cfg, "tabletop_arm_clearance_max_penalty", 2.0))
            self._tabletop_arm_clearance_penalty = torch.clamp(deficit / scale, 0.0, max_penalty).max(dim=-1).values
            masked_margin = torch.where(active_xy, clearance_margin, torch.full_like(clearance_margin, float("inf")))
            min_margin = masked_margin.min(dim=-1).values
            self._tabletop_arm_clearance_min_margin = torch.where(active_xy.any(dim=-1), min_margin, torch.zeros_like(min_margin))
            self._tabletop_arm_clearance_active_fraction = active_xy.float().mean(dim=-1)
            self._tabletop_arm_clearance_ok = self._tabletop_arm_clearance_penalty <= 1.0e-4
        else:
            self._tabletop_arm_clearance_penalty.zero_()
            self._tabletop_arm_clearance_min_margin.zero_()
            self._tabletop_arm_clearance_active_fraction.zero_()
            self._tabletop_arm_clearance_ok.fill_(True)

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
        self._grasp_seen = self._grasp_seen | self._true_grasp
        self._object_height_delta = self._object_pos_w[:, 2] - self._active_object_start_z
        self._lifted = self._object_height_delta > self.cfg.lift_success_height
        rel_vel = self._object_lin_vel_w - self._palm_lin_vel_w
        self._object_palm_rel_vel = torch.norm(rel_vel, dim=-1)
        self._stable_hold = self._lifted & self._true_grasp & (self._object_palm_rel_vel < self.cfg.stable_object_palm_vel)
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
        pos_tip_score = torch.exp(-pos_dist / dist_scale) * contact_weight
        neg_tip_score = torch.exp(-neg_dist / dist_scale) * contact_weight
        return {
            "pos_score": pos_tip_score.max(dim=-1).values,
            "neg_score": neg_tip_score.max(dim=-1).values,
            "pos_contact": ((pos_dist < contact_dist) & (contact_weight > 0.30)).float().max(dim=-1).values,
            "neg_contact": ((neg_dist < contact_dist) & (contact_weight > 0.30)).float().max(dim=-1).values,
            "pos_min_dist": pos_dist.min(dim=-1).values,
            "neg_min_dist": neg_dist.min(dim=-1).values,
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
        smoothed = torch.clamp(smoothed, lower, upper)

        self._prev_joint_targets[:] = self._joint_targets
        self._joint_targets[:, control_ids] = smoothed
        self._apply_initial_target_locks()
        self._apply_post_success_target_locks()
        self.robot.set_joint_position_target(self._joint_targets[:, control_ids], joint_ids=control_ids)

    def _compute_scripted_action_prior(self) -> torch.Tensor:
        action_prior = super()._compute_scripted_action_prior()
        if self.cfg.task_family != "dynamic_tabletop_grasp":
            return action_prior
        if not bool(getattr(self.cfg, "scripted_tabletop_pregrasp_prior_enabled", False)):
            if bool(getattr(self.cfg, "scripted_action_prior_lift_uses_grasp_memory", False)):
                lift_start_step = int(getattr(self.cfg, "scripted_action_prior_lift_start_step", 0))
                lift_steps = int(getattr(self.cfg, "scripted_action_prior_lift_steps", 0))
                lift_stop_step = lift_start_step + max(lift_steps, 0)
                lift_unlocked = (self.episode_length_buf >= lift_start_step) & (
                    (lift_steps <= 0) | (self.episode_length_buf < lift_stop_step)
                )
                min_memory_steps = int(getattr(self.cfg, "scripted_action_prior_lift_grasp_memory_min_steps", 0))
                if (
                    bool(getattr(self.cfg, "scripted_action_prior_lift_memory_requires_streak", False))
                    and min_memory_steps > 0
                    and hasattr(self, "_tabletop_true_grasp_streak")
                ):
                    grasp_memory = self._tabletop_true_grasp_streak >= min_memory_steps
                elif min_memory_steps > 0 and hasattr(self, "_tabletop_true_grasp_streak"):
                    grasp_memory = self._grasp_seen | (self._tabletop_true_grasp_streak >= min_memory_steps)
                else:
                    grasp_memory = self._grasp_seen
                lift_unlocked = lift_unlocked & grasp_memory
                if torch.any(lift_unlocked):
                    if self._scripted_lift_action.shape[0] == self.num_envs:
                        action_prior[lift_unlocked, :7] = self._scripted_lift_action[lift_unlocked]
                    else:
                        action_prior[lift_unlocked, :7] = self._scripted_lift_action
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
                    grasp_memory = self._grasp_seen | (self._tabletop_true_grasp_streak >= min_memory_steps)
                else:
                    grasp_memory = self._grasp_seen
                hand_unlocked = hand_unlocked & grasp_memory
                if torch.any(hand_unlocked):
                    hand_action = float(getattr(self.cfg, "scripted_tabletop_hand_grasp_memory_action", 1.0))
                    hand_ramp_steps = int(
                        getattr(self.cfg, "scripted_tabletop_hand_grasp_memory_ramp_steps", 0)
                    )
                    if hand_ramp_steps > 0:
                        hand_age = self.episode_length_buf[hand_unlocked] - hand_start_step
                        hand_alpha = torch.clamp((hand_age + 1).float() / float(hand_ramp_steps), 0.0, 1.0)
                        ramped_hand_action = -1.0 + hand_alpha * (hand_action + 1.0)
                        action_prior[hand_unlocked, 7:] = ramped_hand_action.unsqueeze(-1)
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

        def arm_pos_to_action(target_arm_pos: torch.Tensor) -> torch.Tensor:
            target_arm_pos = torch.clamp(target_arm_pos.expand_as(arm_center), arm_lower, arm_upper)
            delta = target_arm_pos - arm_center
            target_action = torch.where(delta >= 0.0, delta / positive_span, delta / negative_span)
            return torch.clamp(target_action, -1.0, 1.0)

        pregrasp_action = arm_pos_to_action(desired_arm_pos)

        if bool(getattr(self.cfg, "scripted_tabletop_lift_target_prior_enabled", False)):
            lift_start_step = int(getattr(self.cfg, "scripted_action_prior_lift_start_step", 0))
            lift_steps = int(getattr(self.cfg, "scripted_action_prior_lift_steps", 0))
            lift_stop_step = lift_start_step + max(lift_steps, 0)
            lift_unlocked = (self.episode_length_buf >= lift_start_step) & (
                (lift_steps <= 0) | (self.episode_length_buf < lift_stop_step)
            )
            if bool(getattr(self.cfg, "scripted_action_prior_lift_requires_grasp", False)):
                grasp_unlock = self._true_grasp
                if bool(getattr(self.cfg, "scripted_action_prior_lift_uses_grasp_memory", False)):
                    min_memory_steps = int(
                        getattr(self.cfg, "scripted_action_prior_lift_grasp_memory_min_steps", 0)
                    )
                    if min_memory_steps > 0 and hasattr(self, "_tabletop_true_grasp_streak"):
                        grasp_memory = self._tabletop_true_grasp_streak >= min_memory_steps
                    else:
                        grasp_memory = self._grasp_seen
                    grasp_unlock = grasp_unlock | grasp_memory
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

    def _apply_post_success_target_locks(self) -> None:
        if self.cfg.task_family != "dynamic_tabletop_grasp":
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

        palm_distance = torch.norm(self._object_pos_w - self._palm_pos_w, dim=-1)
        palm_reach = torch.exp(-palm_distance / self.cfg.reach_distance_scale)
        fingertip_reach = torch.mean(torch.exp(-torch.relu(self._surface_dist) / self.cfg.fingertip_distance_scale), dim=-1)

        thumb_score = self._contact_score[:, 0]
        non_thumb_score = torch.sum(self._contact_score[:, 1:], dim=-1) / 4.0
        thumb_contact_weight = float(getattr(self.cfg, "thumb_contact_reward_weight", 0.45))
        thumb_contact_weight = min(max(thumb_contact_weight, 0.0), 1.0)
        thumb_true_grasp_weight = float(getattr(self.cfg, "thumb_true_grasp_score_weight", 0.50))
        thumb_true_grasp_weight = min(max(thumb_true_grasp_weight, 0.0), 1.0)
        contact_rew = thumb_contact_weight * thumb_score + (1.0 - thumb_contact_weight) * non_thumb_score
        opposition_reward_score = (
            self._weighted_opposition_score
            if bool(getattr(self.cfg, "opposition_reward_uses_weighted_score", False))
            else self._opposition_score
        )
        true_grasp_score = torch.clamp(
            thumb_true_grasp_weight * thumb_score
            + (1.0 - thumb_true_grasp_weight) * non_thumb_score
            + opposition_reward_score,
            0.0,
            1.0,
        )
        grasp_quality = self._grasp_quality
        lifted_true_grasp = lift_quality = torch.zeros_like(thumb_score)

        object_pos_local = self._object_pos_w - self.scene.env_origins
        rel_vel_score = torch.exp(-self._object_palm_rel_vel / max(float(self.cfg.stable_object_palm_vel), 1.0e-3))
        z_gate = object_pos_local[:, 2] > self.cfg.catch_success_min_z
        stable_object = self._object_palm_rel_vel < self.cfg.stable_object_palm_vel
        stable_catch = self._true_grasp & stable_object
        pregrasp_xy_rew = torch.zeros_like(palm_distance)
        pregrasp_height_rew = torch.zeros_like(palm_distance)
        pregrasp_low_palm_penalty = torch.zeros_like(palm_distance)
        pregrasp_contact_gate = torch.ones_like(palm_distance)
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
        tabletop_lift_action_prior = torch.zeros_like(palm_distance)
        tabletop_lift_action_prior_gate = torch.zeros_like(palm_distance)
        tabletop_lift_action_prior_rew = torch.zeros_like(palm_distance)
        tabletop_lift_without_object_penalty = torch.zeros_like(palm_distance)
        falling_success_grasp_gate = torch.zeros_like(stable_object)
        falling_success_palm_gate = torch.ones_like(stable_object)
        falling_success_contact_gate = torch.ones_like(stable_object)
        tabletop_affordance = self._compute_tabletop_affordance_reward_terms()
        falling_affordance = self._compute_falling_baton_affordance_reward_terms()

        if self.cfg.task_family == "dynamic_tabletop_grasp":
            lift_progress = torch.clamp(self._object_height_delta / self.cfg.tabletop_success_lift_height, 0.0, 1.0)
            lifted_true_grasp = lift_progress * self._true_grasp.float()
            hover_latch_grasp = self._true_grasp
            if bool(getattr(self.cfg, "tabletop_hover_latch_uses_grasp_seen", False)):
                hover_latch_grasp = hover_latch_grasp | self._grasp_seen
            hover_reward_grasp_gate = self._true_grasp.float()
            if bool(getattr(self.cfg, "tabletop_hover_reward_uses_grasp_seen", False)):
                hover_reward_grasp_gate = torch.maximum(hover_reward_grasp_gate, self._grasp_seen.float())
            hover_success_grasp = self._true_grasp
            if bool(getattr(self.cfg, "tabletop_success_uses_grasp_seen", False)):
                hover_success_grasp = hover_success_grasp | self._grasp_seen
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
            arm_delta = self.robot.data.joint_pos[:, self._arm_joint_ids] - self._tabletop_arm_lift_baseline_pos
            tabletop_arm_lift_progress = torch.clamp(
                torch.sum(arm_delta * self._lift_arm_delta, dim=-1) / self._lift_delta_norm_sq,
                0.0,
                1.0,
            )
            tabletop_lift_action_prior = torch.clamp(
                torch.sum(self.actions[:, :7] * self._lift_action_prior, dim=-1) / self._lift_action_prior_den,
                0.0,
                1.0,
            )
            current_lift_grasp_gate = torch.clamp(
                0.55 * self._true_grasp.float() + 0.45 * true_grasp_score,
                0.0,
                1.0,
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
                arm_object_lift_gap * current_lift_grasp_gate * lift_schedule_unlocked * remaining_lift
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
            no_lift_min_progress = max(float(getattr(self.cfg, "tabletop_no_lift_min_progress", 0.15)), 1.0e-6)
            no_lift_grasp = self._true_grasp & (lift_progress < no_lift_min_progress)
            self._tabletop_true_grasp_streak = torch.where(
                no_lift_grasp,
                self._tabletop_true_grasp_streak + 1,
                torch.zeros_like(self._tabletop_true_grasp_streak),
            )
            no_lift_steps = torch.relu(
                self._tabletop_true_grasp_streak.float()
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
                self._true_grasp.float()
                * rel_vel_score
                * palm_lift_progress
                * torch.clamp(1.0 - lift_progress, 0.0, 1.0)
            )
            success_now = hover_success_grasp & stable_object & (lift_progress > 0.98)
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
            if bool(getattr(self.cfg, "falling_success_uses_grasp_seen", True)):
                falling_success_grasp_gate = self._grasp_seen
            else:
                falling_success_grasp_gate = self._true_grasp
            max_success_palm_dist = float(getattr(self.cfg, "falling_success_max_palm_distance", 0.0))
            if max_success_palm_dist > 0.0:
                falling_success_palm_gate = palm_distance < max_success_palm_dist
            min_success_finger_contacts = float(getattr(self.cfg, "falling_success_min_finger_contacts", 0.0))
            if min_success_finger_contacts > 0.0:
                falling_success_contact_gate = self._finger_contact_count >= min_success_finger_contacts
            catch_hold = (
                falling_success_grasp_gate
                & stable_object
                & z_gate
                & falling_success_palm_gate
                & falling_success_contact_gate
            )
            if bool(getattr(self.cfg, "falling_success_requires_positive_affordance", False)):
                catch_hold = catch_hold & (falling_affordance["pos_contact"] > 0.5)
            success_now = catch_hold
            catch_progress = torch.clamp(
                (0.65 * true_grasp_score + 0.35 * self._grasp_seen.float()) * rel_vel_score * z_gate.float(),
                0.0,
                1.0,
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
        if (
            self.cfg.task_family == "dynamic_tabletop_grasp"
            and bool(getattr(self.cfg, "tabletop_post_success_stability_latch_enabled", False))
        ):
            latch_now = success & (~self._post_success_stability_latched)
            if torch.any(latch_now):
                latch_ids = latch_now.nonzero(as_tuple=False).squeeze(-1)
                self._post_success_arm_joint_target[latch_ids] = self.robot.data.joint_pos[latch_ids][
                    :, self._arm_joint_ids
                ]
                self._post_success_hand_joint_target[latch_ids] = self._joint_targets[latch_ids][
                    :, self._control_hand_joint_ids
                ]
                self._post_success_palm_pos_w[latch_ids] = self._palm_pos_w[latch_ids]
                self._post_success_stability_latched[latch_ids] = True
        success_seen = self._success_seen | success
        if self.cfg.task_family == "dynamic_tabletop_grasp":
            post_success_phase = success_seen.float()
            post_success_unstable = post_success_phase * (1.0 - success_now.float())
            post_success_grasp_loss = post_success_phase * (1.0 - self._true_grasp.float())
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
            stable_catch=stable_catch,
            catch_hold=success_now,
            contact_like=contact_like,
        )

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
            + self.cfg.dynamic_tabletop_pregrasp_xy_rew_scale * pregrasp_xy_rew * (1.0 - lift_progress)
            + self.cfg.dynamic_tabletop_pregrasp_height_rew_scale * pregrasp_height_rew * (1.0 - lift_progress)
            + self.cfg.contact_rew_scale * contact_rew
            + self.cfg.true_grasp_rew_scale * true_grasp_score
            + self.cfg.opposition_rew_scale * opposition_reward_score
            + self.cfg.catch_progress_rew_scale * catch_progress
            + self.cfg.grasp_quality_rew_scale * grasp_quality
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
            * self._true_grasp.float()
            * quality_lift_gate
            + self.cfg.tabletop_hover_target_rew_scale * hover_target_rew * self._true_grasp.float() * lift_progress
            + self.cfg.tabletop_hover_goal_rew_scale * hover_goal_rew
            + self.cfg.tabletop_hover_stable_rew_scale * hover_stability_rew
            + self.cfg.tabletop_grasped_palm_lift_rew_scale * grasped_palm_lift_rew
            + self.cfg.tabletop_grasped_arm_lift_rew_scale * tabletop_arm_lift_rew
            + self.cfg.tabletop_lift_action_prior_rew_scale * tabletop_lift_action_prior_rew
            + float(getattr(self.cfg, "tabletop_affordance_positive_rew_scale", 0.0))
            * tabletop_affordance["pos_score"]
            + float(getattr(self.cfg, "tabletop_affordance_lift_rew_scale", 0.0))
            * tabletop_affordance["pos_contact"]
            * lift_progress
            * self._true_grasp.float()
            + float(getattr(self.cfg, "falling_affordance_positive_rew_scale", 0.0))
            * falling_affordance["pos_score"]
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
            - float(getattr(self.cfg, "tabletop_arm_object_lift_gap_penalty_scale", 0.0))
            * tabletop_arm_object_lift_gap_penalty
            - self.cfg.tabletop_hover_linear_penalty_scale * hover_linear_error
            - self.cfg.tabletop_hover_overshoot_penalty_scale
            * hover_z_overshoot_ratio
            * hover_z_overshoot_ratio
            * hover_latched
            * self._true_grasp.float()
            - self.cfg.tabletop_hover_z_vel_penalty_scale
            * hover_z_vel
            * hover_z_vel
            * hover_latched
            * self._true_grasp.float()
            * torch.clamp(0.35 + 0.65 * lift_progress, 0.0, 1.0)
            - self.cfg.tabletop_hover_vel_penalty_scale
            * (object_speed * object_speed + 0.02 * object_ang_speed * object_ang_speed)
            * self._true_grasp.float()
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
            "opposition_score": self._opposition_score.mean(),
            "weighted_opposition_score": self._weighted_opposition_score.mean(),
            "opposition_reward_score": opposition_reward_score.mean(),
            "opposing_contact": self._opposing_contact.float().mean(),
            "finger_count_quality": self._finger_count_quality.mean(),
            "non_thumb_quality": self._non_thumb_quality.mean(),
            "grasp_quality": grasp_quality.mean(),
            "opposition_lift_gate": opposition_lift_gate_signal.mean(),
            "quality_lift_gate": quality_lift_gate.mean(),
            "pregrasp_xy_rew": pregrasp_xy_rew.mean(),
            "pregrasp_height_rew": pregrasp_height_rew.mean(),
            "pregrasp_low_palm_penalty": pregrasp_low_palm_penalty.mean(),
            "pregrasp_contact_gate": pregrasp_contact_gate.mean(),
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
            "grasped_palm_lift_rew": grasped_palm_lift_rew.mean(),
            "tabletop_arm_lift_progress": tabletop_arm_lift_progress.mean(),
            "tabletop_arm_lift_rew": tabletop_arm_lift_rew.mean(),
            "tabletop_arm_object_lift_gap_penalty": tabletop_arm_object_lift_gap_penalty.mean(),
            "tabletop_lift_action_prior": tabletop_lift_action_prior.mean(),
            "tabletop_lift_action_prior_gate": tabletop_lift_action_prior_gate.mean(),
            "tabletop_lift_action_prior_rew": tabletop_lift_action_prior_rew.mean(),
            "tabletop_lift_without_object_penalty": tabletop_lift_without_object_penalty.mean(),
            "tabletop_arm_clearance_penalty": self._tabletop_arm_clearance_penalty.mean(),
            "tabletop_arm_clearance_min_margin": self._tabletop_arm_clearance_min_margin.mean(),
            "tabletop_arm_clearance_active_fraction": self._tabletop_arm_clearance_active_fraction.mean(),
            "tabletop_arm_clearance_ok": self._tabletop_arm_clearance_ok.float().mean(),
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
        self.extras["success_env"] = success
        self.extras["true_grasp_env"] = self._true_grasp
        self.extras["grasp_seen_env"] = self._grasp_seen
        self.extras["stable_hold_env"] = success_now
        self.extras["success_seen_env"] = self._success_seen
        self.extras["lifted_env"] = lift_progress > 0.98
        self.extras["tabletop_arm_clearance_ok_env"] = self._tabletop_arm_clearance_ok
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
        if self.cfg.terminate_on_success:
            terminated = terminated | success
        self.extras["dropped_env"] = dropped
        self.extras["out_xy_env"] = out_xy
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
                roll_min, roll_max = self.cfg.falling_baton_roll_range
                pitch_min, pitch_max = self.cfg.falling_baton_pitch_range
                yaw_min, yaw_max = self.cfg.falling_baton_yaw_range
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
        self._grasp_seen[env_ids] = False
        self._tabletop_true_grasp_streak[env_ids] = 0
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
        speed = speed_min + sample_uniform(0.0, 1.0, (count, 1), self.device) * (speed_max - speed_min)
        heading_min, heading_max = self.cfg.dynamic_tabletop_heading_range
        heading = sample_uniform(float(heading_min), float(heading_max), (count, 1), self.device)
        lin_vel = torch.cat((torch.sin(heading) * speed, torch.cos(heading) * speed, torch.zeros_like(speed)), dim=-1)

        yaw_min, yaw_max = self._tabletop_current_yaw_rate_range()
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
                above_min, above_max = self.cfg.falling_baton_spawn_above_palm_range
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
