"""DirectRLEnv for Franka + BrainCo Revo2 static primitive grasping."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse, sample_uniform
from pxr import Usd, UsdPhysics, UsdShade

from .revo2_static_grasp_env_cfg import Revo2StaticGraspEnvCfg


FRANKA_ADJACENT_SELF_COLLISION_FILTER_PAIRS = (
    ("panda_link0", "panda_link1"),
    ("panda_link1", "panda_link2"),
    ("panda_link2", "panda_link3"),
    ("panda_link3", "panda_link4"),
    ("panda_link4", "panda_link5"),
    ("panda_link5", "panda_link6"),
    ("panda_link6", "panda_link7"),
    ("panda_link7", "panda_link8"),
    ("panda_link8", "panda_hand"),
)

REVO2_ADJACENT_SELF_COLLISION_FILTER_PAIRS = (
    ("panda_hand", "revo2_right_base_link"),
    ("revo2_right_base_link", "revo2_right_thumb_metacarpal_link"),
    ("revo2_right_thumb_metacarpal_link", "revo2_right_thumb_proximal_link"),
    ("revo2_right_thumb_proximal_link", "revo2_right_thumb_distal_link"),
    ("revo2_right_thumb_distal_link", "revo2_right_thumb_tip_link"),
    ("revo2_right_thumb_distal_link", "revo2_right_thumb_touch_link"),
    ("revo2_right_base_link", "revo2_right_index_proximal_link"),
    ("revo2_right_index_proximal_link", "revo2_right_index_distal_link"),
    ("revo2_right_index_distal_link", "revo2_right_index_tip_link"),
    ("revo2_right_index_distal_link", "revo2_right_index_touch_link"),
    ("revo2_right_base_link", "revo2_right_middle_proximal_link"),
    ("revo2_right_middle_proximal_link", "revo2_right_middle_distal_link"),
    ("revo2_right_middle_distal_link", "revo2_right_middle_tip_link"),
    ("revo2_right_middle_distal_link", "revo2_right_middle_touch_link"),
    ("revo2_right_base_link", "revo2_right_ring_proximal_link"),
    ("revo2_right_ring_proximal_link", "revo2_right_ring_distal_link"),
    ("revo2_right_ring_distal_link", "revo2_right_ring_tip_link"),
    ("revo2_right_ring_distal_link", "revo2_right_ring_touch_link"),
    ("revo2_right_base_link", "revo2_right_pinky_proximal_link"),
    ("revo2_right_pinky_proximal_link", "revo2_right_pinky_distal_link"),
    ("revo2_right_pinky_distal_link", "revo2_right_pinky_tip_link"),
    ("revo2_right_pinky_distal_link", "revo2_right_pinky_touch_link"),
)

INSPIRE_ADJACENT_SELF_COLLISION_FILTER_PAIRS = (
    ("panda_hand", "hand_base_link"),
    ("hand_base_link", "thumb_proximal_base"),
    ("thumb_proximal_base", "thumb_proximal"),
    ("thumb_proximal", "thumb_intermediate"),
    ("thumb_intermediate", "thumb_distal"),
    ("thumb_distal", "thumb_tip"),
    ("hand_base_link", "index_proximal"),
    ("index_proximal", "index_intermediate"),
    ("index_intermediate", "index_tip"),
    ("hand_base_link", "middle_proximal"),
    ("middle_proximal", "middle_intermediate"),
    ("middle_intermediate", "middle_tip"),
    ("hand_base_link", "ring_proximal"),
    ("ring_proximal", "ring_intermediate"),
    ("ring_intermediate", "ring_tip"),
    ("hand_base_link", "pinky_proximal"),
    ("pinky_proximal", "pinky_intermediate"),
    ("pinky_intermediate", "pinky_tip"),
)

INSPIRE_HAND_INTERNAL_SELF_COLLISION_FILTER_PAIRS = (
    # Same-finger ancestor/descendant pairs that can otherwise fight official
    # coupled close postures after URDF import into PhysX.
    ("thumb_proximal_base", "thumb_intermediate"),
    ("thumb_proximal_base", "thumb_distal"),
    ("thumb_proximal_base", "thumb_tip"),
    ("thumb_proximal", "thumb_distal"),
    ("thumb_proximal", "thumb_tip"),
    ("thumb_intermediate", "thumb_tip"),
    ("index_proximal", "index_tip"),
    ("middle_proximal", "middle_tip"),
    ("ring_proximal", "ring_tip"),
    ("pinky_proximal", "pinky_tip"),
    # Adjacent fingers are intentionally close in Inspire grasp tables.  Filter
    # those internal finger-finger pairs while preserving hand-object contact.
    ("index_proximal", "middle_proximal"),
    ("index_proximal", "middle_intermediate"),
    ("index_intermediate", "middle_proximal"),
    ("index_intermediate", "middle_intermediate"),
    ("index_tip", "middle_tip"),
    ("middle_proximal", "ring_proximal"),
    ("middle_proximal", "ring_intermediate"),
    ("middle_intermediate", "ring_proximal"),
    ("middle_intermediate", "ring_intermediate"),
    ("middle_tip", "ring_tip"),
    ("ring_proximal", "pinky_proximal"),
    ("ring_proximal", "pinky_intermediate"),
    ("ring_intermediate", "pinky_proximal"),
    ("ring_intermediate", "pinky_intermediate"),
    ("ring_tip", "pinky_tip"),
)


class Revo2StaticGraspEnv(DirectRLEnv):
    """Static primitive lift-and-hold task for Franka + BrainCo Revo2."""

    cfg: Revo2StaticGraspEnvCfg

    def __init__(self, cfg: Revo2StaticGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.policy_actions = torch.zeros_like(self.actions)
        self._action_prior = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        self._arm_joint_ids, self._arm_joint_names = self.robot.find_joints(
            list(self.cfg.arm_joint_names), preserve_order=True
        )
        self._hand_joint_ids, self._hand_joint_names = self.robot.find_joints(
            list(self.cfg.hand_joint_names), preserve_order=True
        )
        self._controlled_joint_ids = self._arm_joint_ids + self._hand_joint_ids

        self._palm_body_id = self.robot.body_names.index(self.cfg.palm_body_name)
        self._fingertip_body_ids = [self.robot.body_names.index(name) for name in self.cfg.fingertip_body_names]
        self._palm_body_offset = torch.as_tensor(
            getattr(self.cfg, "palm_offset", (0.0, 0.0, 0.0)),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3)
        fingertip_body_offsets = torch.as_tensor(
            getattr(
                self.cfg,
                "fingertip_body_offsets",
                ((0.0, 0.0, 0.0),) * len(self._fingertip_body_ids),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if fingertip_body_offsets.shape != (len(self._fingertip_body_ids), 3):
            raise ValueError(
                "fingertip_body_offsets must have shape "
                f"({len(self._fingertip_body_ids)}, 3), got {tuple(fingertip_body_offsets.shape)}"
            )
        self._fingertip_body_offsets = fingertip_body_offsets.unsqueeze(0)

        joint_limits = self.robot.data.soft_joint_pos_limits[0].to(device=self.device)
        self._joint_lower_limits = joint_limits[:, 0]
        self._joint_upper_limits = joint_limits[:, 1]

        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._prev_joint_targets = self._joint_targets.clone()
        self._default_joint_pos = self.robot.data.default_joint_pos.clone()
        self._default_arm_pos = torch.tensor(self.cfg.default_arm_pos, device=self.device).unsqueeze(0)
        self._arm_clamp_delta = torch.tensor(self.cfg.arm_target_clamp_delta, device=self.device).unsqueeze(0)
        self._reference_hand_fractions = torch.tensor(
            self.cfg.reference_hand_fractions, device=self.device
        ).unsqueeze(0)
        self._pregrasp_target_rel_palm = torch.tensor(
            self.cfg.pregrasp_target_rel_palm, device=self.device
        ).unsqueeze(0)
        self._pregrasp_target_scale = torch.tensor(self.cfg.pregrasp_target_scale, device=self.device).unsqueeze(0)
        self._lift_arm_delta = torch.tensor(self.cfg.lift_arm_delta, device=self.device).unsqueeze(0)
        self._lift_action_prior = torch.tensor(self.cfg.lift_action_prior, device=self.device).unsqueeze(0)
        self._scripted_lift_action = torch.tensor(
            self.cfg.scripted_action_prior_lift_action, device=self.device
        ).unsqueeze(0)
        self._lift_action_prior_den = torch.clamp(
            torch.sum(torch.abs(self._lift_action_prior), dim=-1), min=1.0e-6
        )
        self._lift_delta_norm_sq = torch.clamp(torch.sum(self._lift_arm_delta * self._lift_arm_delta), min=1.0e-6)
        self._object_start_pos = torch.tensor(self.cfg.object_start_pos, device=self.device).unsqueeze(0)
        self._object_start_rot = torch.tensor(self.cfg.object_start_rot, device=self.device).unsqueeze(0)

        self._success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._true_grasp = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grasp_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stable_hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._surface_dist = torch.zeros((self.num_envs, len(self._fingertip_body_ids)), device=self.device)
        self._contact_score = torch.zeros_like(self._surface_dist)
        self._opposition_score = torch.zeros(self.num_envs, device=self.device)
        self._weighted_opposition_score = torch.zeros(self.num_envs, device=self.device)
        self._opposing_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._finger_contact_count = torch.zeros(self.num_envs, device=self.device)
        self._finger_count_quality = torch.zeros(self.num_envs, device=self.device)
        self._non_thumb_quality = torch.zeros(self.num_envs, device=self.device)
        self._grasp_quality = torch.zeros(self.num_envs, device=self.device)
        self._scoop_lift = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._palm_only_lift = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._palm_surface_dist = torch.zeros(self.num_envs, device=self.device)
        self._palm_contact_score = torch.zeros(self.num_envs, device=self.device)
        self._palm_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._object_height_delta = torch.zeros(self.num_envs, device=self.device)
        self._object_palm_rel_vel = torch.zeros(self.num_envs, device=self.device)
        self._palm_start_z = torch.zeros(self.num_envs, device=self.device)
        self._palm_min_z = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._robot_material_bind_count = getattr(self, "_robot_material_bind_count", 0)
        self._robot_material_bind_fail_count = getattr(self, "_robot_material_bind_fail_count", 0)
        self._robot_self_collision_filter_pair_count = getattr(self, "_robot_self_collision_filter_pair_count", 0)
        self._robot_self_collision_filter_fail_count = getattr(self, "_robot_self_collision_filter_fail_count", 0)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.table = RigidObject(self.cfg.table_cfg) if self.cfg.create_table else None
        if self.cfg.video_camera_enabled:
            self._video_camera = TiledCamera(self.cfg.video_camera)
        if self.cfg.student_camera_enabled:
            self._student_camera = TiledCamera(self.cfg.student_camera)
        self._bind_robot_physics_material()
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self._apply_robot_self_collision_filters()
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        if self.table is not None:
            self.scene.rigid_objects["table"] = self.table
        if self.cfg.video_camera_enabled:
            self.scene.sensors["video_camera"] = self._video_camera
        if self.cfg.student_camera_enabled:
            self.scene.sensors["student_camera"] = self._student_camera
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.78, 0.78, 0.76))
        light_cfg.func("/World/Light", light_cfg)

    def _bind_robot_physics_material(self) -> None:
        material_path = "/World/PhysicsMaterials/revo2_robot_high_friction"
        self.cfg.robot_physics_material.func(material_path, self.cfg.robot_physics_material)

        source_robot_path = f"{self.scene.env_prim_paths[0]}/Robot"
        root_prim = self.scene.stage.GetPrimAtPath(source_robot_path)
        if not root_prim.IsValid():
            self._robot_material_bind_count = 0
            self._robot_material_bind_fail_count = 0
            return

        material = UsdShade.Material(self.scene.stage.GetPrimAtPath(material_path))
        bind_count = 0
        bind_fail_count = 0

        def bind_collision_prim(prim) -> int:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                return 0
            try:
                binding_api = UsdShade.MaterialBindingAPI(prim)
                if not binding_api:
                    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                binding_api.Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                return 1
            except Exception:
                return -1

        for prim in Usd.PrimRange(root_prim):
            bind_result = bind_collision_prim(prim)
            if bind_result > 0:
                bind_count += bind_result
            elif bind_result < 0:
                bind_fail_count += 1
        self._robot_material_bind_count = bind_count
        self._robot_material_bind_fail_count = bind_fail_count

    def _robot_self_collision_filter_link_pairs(self) -> tuple[tuple[str, str], ...]:
        pairs = list(FRANKA_ADJACENT_SELF_COLLISION_FILTER_PAIRS)
        hand_embodiment = str(getattr(self.cfg, "hand_embodiment", "")).lower()
        palm_body_name = str(getattr(self.cfg, "palm_body_name", ""))
        if hand_embodiment == "inspire" or palm_body_name == "hand_base_link":
            pairs.extend(INSPIRE_ADJACENT_SELF_COLLISION_FILTER_PAIRS)
            pairs.extend(INSPIRE_HAND_INTERNAL_SELF_COLLISION_FILTER_PAIRS)
        else:
            pairs.extend(REVO2_ADJACENT_SELF_COLLISION_FILTER_PAIRS)
        return tuple(pairs)

    def _collision_filter_prims_for_link(self, link_prim) -> list:
        collision_prims = [prim for prim in Usd.PrimRange(link_prim) if prim.HasAPI(UsdPhysics.CollisionAPI)]
        return collision_prims if collision_prims else [link_prim]

    def _apply_robot_self_collision_filters(self) -> None:
        pair_count = 0
        fail_count = 0
        filter_pairs = self._robot_self_collision_filter_link_pairs()
        stage = self.scene.stage
        for env_path in self.scene.env_prim_paths:
            robot_path = f"{env_path}/Robot"
            for link_a, link_b in filter_pairs:
                prim_a = stage.GetPrimAtPath(f"{robot_path}/{link_a}")
                prim_b = stage.GetPrimAtPath(f"{robot_path}/{link_b}")
                if not prim_a.IsValid() or not prim_b.IsValid():
                    fail_count += 1
                    continue
                filter_prims_a = self._collision_filter_prims_for_link(prim_a)
                filter_prims_b = self._collision_filter_prims_for_link(prim_b)
                for filter_prim_a in filter_prims_a:
                    for filter_prim_b in filter_prims_b:
                        filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(filter_prim_a)
                        if not filtered_pairs_api:
                            fail_count += 1
                            continue
                        filtered_pairs_api.CreateFilteredPairsRel().AddTarget(filter_prim_b.GetPath())
                        pair_count += 1
        self._robot_self_collision_filter_pair_count = pair_count
        self._robot_self_collision_filter_fail_count = fail_count

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        self.policy_actions = torch.clamp(actions, -1.0, 1.0)
        if self.cfg.scripted_action_prior_enabled:
            self._action_prior[:] = self._compute_scripted_action_prior()
            if bool(getattr(self.cfg, "scripted_action_prior_zero_passthrough_enabled", False)):
                prior_active = torch.any(torch.abs(self._action_prior) > 1.0e-6, dim=-1, keepdim=True)
                active_residual_scale = float(
                    getattr(
                        self.cfg,
                        "scripted_action_prior_active_residual_scale",
                        self.cfg.scripted_action_prior_residual_scale,
                    )
                )
                inactive_residual_scale = float(
                    getattr(self.cfg, "scripted_action_prior_inactive_residual_scale", 1.0)
                )
                residual_scale = torch.where(
                    prior_active,
                    torch.full_like(self.policy_actions[:, :1], active_residual_scale),
                    torch.full_like(self.policy_actions[:, :1], inactive_residual_scale),
                )
                self.actions = torch.clamp(self._action_prior + residual_scale * self.policy_actions, -1.0, 1.0)
            else:
                residual_scale = float(self.cfg.scripted_action_prior_residual_scale)
                self.actions = torch.clamp(self._action_prior + residual_scale * self.policy_actions, -1.0, 1.0)
        else:
            self._action_prior.zero_()
            self.actions = self.policy_actions

    def _compute_scripted_action_prior(self) -> torch.Tensor:
        action_prior = torch.zeros_like(self.actions)
        hand_unlocked = self.episode_length_buf >= self.cfg.scripted_action_prior_hand_start_step
        if torch.any(hand_unlocked):
            hand_action = float(self.cfg.scripted_action_prior_hand_action)
            hand_ramp_steps = int(self.cfg.scripted_action_prior_hand_ramp_steps)
            if hand_ramp_steps > 0:
                hand_age = self.episode_length_buf[hand_unlocked] - self.cfg.scripted_action_prior_hand_start_step
                hand_alpha = torch.clamp((hand_age + 1).float() / float(hand_ramp_steps), 0.0, 1.0)
                ramped_hand_action = -1.0 + hand_alpha * (hand_action + 1.0)
                action_prior[hand_unlocked, 7:] = ramped_hand_action.unsqueeze(-1)
            else:
                action_prior[hand_unlocked, 7:] = hand_action

        lift_start_step = self.cfg.scripted_action_prior_lift_start_step
        lift_stop_step = lift_start_step + self.cfg.scripted_action_prior_lift_steps
        lift_unlocked = (self.episode_length_buf >= lift_start_step) & (self.episode_length_buf < lift_stop_step)
        if self.cfg.scripted_action_prior_lift_requires_grasp:
            lift_unlocked = lift_unlocked & self._true_grasp
        if torch.any(lift_unlocked):
            if self._scripted_lift_action.shape[0] == self.num_envs:
                action_prior[lift_unlocked, :7] = self._scripted_lift_action[lift_unlocked]
            else:
                action_prior[lift_unlocked, :7] = self._scripted_lift_action
        return action_prior

    def _apply_action(self) -> None:
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

        hand_fractions = 0.5 * (hand_actions + 1.0) * self._reference_hand_fractions
        hand_fractions = torch.clamp(hand_fractions, 0.0, 1.0)
        raw_hand_targets = self._semantic_hand_fractions_to_joint_targets(hand_fractions)
        current_hand_targets = self._joint_targets[:, self._hand_joint_ids]
        hand_targets = (
            self.cfg.hand_moving_average * raw_hand_targets
            + (1.0 - self.cfg.hand_moving_average) * current_hand_targets
        )

        self._prev_joint_targets[:] = self._joint_targets
        self._joint_targets[:, self._arm_joint_ids] = arm_targets
        self._joint_targets[:, self._hand_joint_ids] = hand_targets

        lock_arm = self.episode_length_buf < self.cfg.initial_arm_target_lock_steps
        if torch.any(lock_arm):
            self._joint_targets[lock_arm.nonzero(as_tuple=False).squeeze(-1)[:, None], self._arm_joint_ids] = (
                self._default_joint_pos[lock_arm][:, self._arm_joint_ids]
            )

        lock_hand = self.episode_length_buf < self.cfg.initial_hand_target_lock_steps
        if torch.any(lock_hand):
            self._joint_targets[lock_hand.nonzero(as_tuple=False).squeeze(-1)[:, None], self._hand_joint_ids] = (
                self._default_joint_pos[lock_hand][:, self._hand_joint_ids]
            )

        self.robot.set_joint_position_target(
            self._joint_targets[:, self._controlled_joint_ids], joint_ids=self._controlled_joint_ids
        )

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        arm_pos = self.robot.data.joint_pos[:, self._arm_joint_ids] - self._default_arm_pos
        arm_vel = 0.1 * self.robot.data.joint_vel[:, self._arm_joint_ids]

        hand_pos = self.robot.data.joint_pos[:, self._hand_joint_ids]
        hand_vel = 0.1 * self.robot.data.joint_vel[:, self._hand_joint_ids]
        hand_lower = self._joint_lower_limits[self._hand_joint_ids]
        hand_upper = self._joint_upper_limits[self._hand_joint_ids]
        hand_pos_scaled = unscale(hand_pos, hand_lower, hand_upper)

        object_pos_local = self._object_pos_w - self.scene.env_origins
        palm_to_object_w = self._object_pos_w - self._palm_pos_w
        palm_to_object = quat_rotate_inverse(self._palm_quat_w, palm_to_object_w)
        fingertip_to_object = self._fingertip_pos_w - self._object_pos_w.unsqueeze(1)

        obs = torch.cat(
            (
                arm_pos,
                arm_vel,
                hand_pos_scaled,
                hand_vel,
                object_pos_local,
                self._object_quat_w,
                self._object_lin_vel_w,
                self._object_ang_vel_w,
                palm_to_object,
                fingertip_to_object.reshape(self.num_envs, -1),
                self._surface_dist,
                self._object_height_delta.unsqueeze(-1),
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        palm_distance = torch.norm(self._object_pos_w - self._palm_pos_w, dim=-1)
        palm_reach = torch.exp(-palm_distance / 0.16)
        palm_distance_excess = torch.relu(palm_distance - self.cfg.palm_distance_penalty_margin)
        fingertip_reach = torch.mean(torch.exp(-torch.relu(self._surface_dist) / 0.045), dim=-1)

        rel_palm_w = self._object_pos_w - self._palm_pos_w
        rel_palm = quat_rotate_inverse(self._palm_quat_w, rel_palm_w)
        pregrasp_error = (rel_palm - self._pregrasp_target_rel_palm) / self._pregrasp_target_scale
        pregrasp_rew = torch.exp(-torch.sum(pregrasp_error * pregrasp_error, dim=-1))

        thumb_score = self._contact_score[:, 0]
        non_thumb_score = torch.sum(self._contact_score[:, 1:], dim=-1) / 4.0
        contact_rew = 0.45 * thumb_score + 0.55 * non_thumb_score
        true_grasp_score = torch.clamp(0.5 * thumb_score + 0.5 * non_thumb_score + self._opposition_score, 0.0, 1.0)

        lift_progress = torch.clamp(self._object_height_delta / self.cfg.lift_success_height, 0.0, 1.0)
        lift_progress_shaped = lift_progress * lift_progress
        lifted_true_grasp = self._lifted.float() * self._true_grasp.float()
        stable_hold = self._stable_hold.float()
        grasp_gate = torch.clamp(0.5 * true_grasp_score + 0.5 * self._true_grasp.float(), 0.0, 1.0)
        lift_grasp_gate = grasp_gate
        if self.cfg.lift_gate_uses_grasp_memory:
            lift_grasp_gate = torch.maximum(lift_grasp_gate, self._grasp_seen.float())
        lift_reward_gate = lift_grasp_gate
        if not self.cfg.lift_reward_uses_grasp_memory:
            lift_reward_gate = grasp_gate
        palm_lift_baseline_z = self._palm_start_z
        if self.cfg.palm_lift_baseline_mode == "min":
            palm_lift_baseline_z = self._palm_min_z
        palm_lift_delta = torch.clamp(
            (self._palm_pos_w[:, 2] - palm_lift_baseline_z) / self.cfg.palm_lift_target_height, 0.0, 1.0
        )
        arm_delta = self.robot.data.joint_pos[:, self._arm_joint_ids] - self._default_arm_pos
        arm_lift_progress = torch.clamp(
            torch.sum(arm_delta * self._lift_arm_delta, dim=-1) / self._lift_delta_norm_sq,
            0.0,
            1.0,
        )
        lift_action_prior = torch.clamp(
            torch.sum(self.actions[:, :7] * self._lift_action_prior, dim=-1) / self._lift_action_prior_den,
            0.0,
            1.0,
        )
        arm_unlocked = (self.episode_length_buf >= self.cfg.initial_arm_target_lock_steps).float()
        lift_action_prior_gate = lift_grasp_gate * arm_unlocked
        if self.cfg.lift_action_prior_lift_gate_scale > 0.0:
            lift_gate = self.cfg.lift_action_prior_lift_gate_min + (
                1.0 - self.cfg.lift_action_prior_lift_gate_min
            ) * lift_progress
            lift_action_prior_gate = lift_action_prior_gate * torch.clamp(lift_gate, 0.0, 1.0)
        lift_progress_gate = self.cfg.lift_progress_min_grasp_gate + (
            1.0 - self.cfg.lift_progress_min_grasp_gate
        ) * true_grasp_score
        lift_coupling = torch.minimum(palm_lift_delta, lift_progress) * lift_reward_gate
        palm_lift_gap = torch.relu(palm_lift_delta - lift_progress)
        premature_lift_gate = (1.0 - lift_grasp_gate) * arm_unlocked
        self._success_streak = torch.where(
            self._stable_hold, self._success_streak + 1, torch.zeros_like(self._success_streak)
        )
        hold_progress = torch.clamp(
            self._success_streak.float() / max(float(self.cfg.success_hold_steps), 1.0),
            0.0,
            1.0,
        )

        policy_action_penalty = torch.sum(self.policy_actions * self.policy_actions, dim=-1)
        action_penalty = torch.sum(self.actions * self.actions, dim=-1)
        target_delta = self._joint_targets[:, self._controlled_joint_ids] - self._prev_joint_targets[
            :, self._controlled_joint_ids
        ]
        target_delta_penalty = torch.sum(target_delta * target_delta, dim=-1)
        dropped = self._object_pos_w[:, 2] < self.cfg.table_top_z - 0.04

        reward = (
            self.cfg.palm_reach_rew_scale * palm_reach
            + self.cfg.fingertip_reach_rew_scale * fingertip_reach
            + self.cfg.pregrasp_rew_scale * pregrasp_rew
            + self.cfg.contact_rew_scale * contact_rew
            + self.cfg.true_grasp_rew_scale * true_grasp_score
            + self.cfg.opposition_rew_scale * self._opposition_score
            + self.cfg.palm_lift_rew_scale * palm_lift_delta * lift_reward_gate
            + self.cfg.arm_lift_progress_rew_scale * arm_lift_progress * lift_reward_gate
            + self.cfg.lift_action_prior_rew_scale * lift_action_prior * lift_action_prior_gate
            + self.cfg.lift_progress_linear_rew_scale * lift_progress * lift_reward_gate
            + self.cfg.lift_progress_rew_scale * lift_progress_shaped * lift_progress_gate
            + self.cfg.lift_coupling_rew_scale * lift_coupling
            + self.cfg.lifted_true_grasp_rew_scale * lifted_true_grasp
            + self.cfg.stable_hold_rew_scale * stable_hold
            + self.cfg.hold_progress_rew_scale * hold_progress * hold_progress
            + self.cfg.success_bonus * (self._success_streak >= self.cfg.success_hold_steps).float()
            - self.cfg.action_penalty_scale * policy_action_penalty
            - self.cfg.arm_target_delta_penalty_scale * target_delta_penalty
            - self.cfg.palm_lift_gap_penalty_scale * palm_lift_gap * palm_lift_gap * lift_grasp_gate * arm_unlocked
            - self.cfg.palm_distance_penalty_scale * palm_distance_excess * palm_distance_excess
            - self.cfg.premature_arm_lift_penalty_scale
            * arm_lift_progress
            * arm_lift_progress
            * premature_lift_gate
            - self.cfg.premature_palm_lift_penalty_scale
            * palm_lift_delta
            * palm_lift_delta
            * premature_lift_gate
            - self.cfg.drop_penalty * dropped.float()
        )

        success = self._success_streak >= self.cfg.success_hold_steps
        self.extras["log"] = {
            "palm_distance": palm_distance.mean(),
            "palm_distance_excess": palm_distance_excess.mean(),
            "min_surface_dist": self._surface_dist.min(dim=-1).values.mean(),
            "mean_surface_dist": self._surface_dist.mean(),
            "thumb_contact": self._thumb_contact.float().mean(),
            "non_thumb_contacts": self._non_thumb_contact_count.float().mean(),
            "opposition_score": self._opposition_score.mean(),
            "true_grasp": self._true_grasp.float().mean(),
            "grasp_seen": self._grasp_seen.float().mean(),
            "lifted": self._lifted.float().mean(),
            "stable_hold": self._stable_hold.float().mean(),
            "hold_progress": hold_progress.mean(),
            "success": success.float().mean(),
            "object_height_delta": self._object_height_delta.mean(),
            "palm_lift_delta": palm_lift_delta.mean(),
            "arm_lift_progress": arm_lift_progress.mean(),
            "lift_action_prior": lift_action_prior.mean(),
            "lift_action_prior_gate": lift_action_prior_gate.mean(),
            "lift_grasp_gate": lift_grasp_gate.mean(),
            "lift_reward_gate": lift_reward_gate.mean(),
            "lift_coupling": lift_coupling.mean(),
            "palm_lift_gap": palm_lift_gap.mean(),
            "premature_lift_gate": premature_lift_gate.mean(),
            "effective_action_norm": torch.norm(self.actions, dim=-1).mean(),
            "policy_action_norm": torch.norm(self.policy_actions, dim=-1).mean(),
            "scripted_action_prior_norm": torch.norm(self._action_prior, dim=-1).mean(),
            "effective_action_penalty": action_penalty.mean(),
            "object_palm_rel_vel": self._object_palm_rel_vel.mean(),
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
        self.extras["success_env"] = success
        self.extras["true_grasp_env"] = self._true_grasp
        self.extras["grasp_seen_env"] = self._grasp_seen
        self.extras["lifted_env"] = self._lifted
        self.extras["stable_hold_env"] = self._stable_hold
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        object_pos_local = self._object_pos_w - self.scene.env_origins
        dropped = object_pos_local[:, 2] < self.cfg.table_top_z - 0.04
        out_xy = torch.any(torch.abs(object_pos_local[:, :2]) > self.cfg.workspace_xy_limit, dim=-1)
        success = self._success_streak >= self.cfg.success_hold_steps
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

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        if self.cfg.reset_arm_pos_noise > 0.0:
            joint_pos[:, self._arm_joint_ids] += sample_uniform(
                -self.cfg.reset_arm_pos_noise,
                self.cfg.reset_arm_pos_noise,
                (len(env_ids), len(self._arm_joint_ids)),
                self.device,
            )
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._joint_targets[env_ids] = joint_pos
        self._prev_joint_targets[env_ids] = joint_pos

        object_state = self.object.data.default_root_state[env_ids].clone()
        if self._object_start_pos.shape[0] == self.num_envs:
            object_start_pos = self._object_start_pos[env_ids]
        else:
            object_start_pos = self._object_start_pos.expand(len(env_ids), -1)
        object_state[:, 0:3] = object_start_pos + self.scene.env_origins[env_ids]
        reset_noise = torch.tensor(self.cfg.reset_object_pos_noise, device=self.device).unsqueeze(0)
        object_state[:, 0:3] += sample_uniform(-1.0, 1.0, (len(env_ids), 3), self.device) * reset_noise
        object_state[:, 3:7] = self._object_start_rot
        object_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids)

        self.actions[env_ids] = 0.0
        self.policy_actions[env_ids] = 0.0
        self._action_prior[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._success_streak[env_ids] = 0
        self._grasp_seen[env_ids] = False
        self._compute_intermediate_values()
        self._palm_start_z[env_ids] = self._palm_pos_w[env_ids, 2]
        self._palm_min_z[env_ids] = self._palm_pos_w[env_ids, 2]

    def _semantic_hand_fractions_to_joint_targets(
        self,
        fractions: torch.Tensor,
        joint_ids: Sequence[int] | None = None,
        joint_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        joint_ids = list(self._hand_joint_ids if joint_ids is None else joint_ids)
        joint_names = list(self._hand_joint_names if joint_names is None else joint_names)
        targets = self._default_joint_pos[:, joint_ids].clone()
        lower = self._joint_lower_limits
        upper = self._joint_upper_limits
        joint_index = {name: i for i, name in enumerate(self.robot.joint_names)}
        local_index = {name: i for i, name in enumerate(joint_names)}

        if not all(name.startswith("revo2_right_") for name in joint_names):
            if fractions.shape[-1] != len(joint_names):
                raise ValueError(f"Expected {len(joint_names)} hand fractions, got {fractions.shape[-1]}.")
            for index, joint_name in enumerate(joint_names):
                jid = joint_index[joint_name]
                targets[:, index] = lower[jid] + fractions[:, index] * (upper[jid] - lower[jid])
            return targets

        def scale_joint(joint_name: str, fraction: torch.Tensor) -> torch.Tensor:
            jid = joint_index[joint_name]
            return lower[jid] + fraction * (upper[jid] - lower[jid])

        def write_joint(joint_name: str, target: torch.Tensor) -> None:
            if joint_name not in local_index:
                return
            jid = joint_index[joint_name]
            targets[:, local_index[joint_name]] = torch.clamp(target, lower[jid], upper[jid])

        def write_mimic_joint(joint_name: str, driver_target: torch.Tensor, multiplier: float) -> None:
            write_joint(joint_name, driver_target * multiplier)

        thumb_meta = "revo2_right_thumb_metacarpal_joint"
        thumb_prox = "revo2_right_thumb_proximal_joint"
        thumb_meta_target = scale_joint(thumb_meta, fractions[:, 0])
        thumb_prox_target = scale_joint(thumb_prox, fractions[:, 1])
        write_joint(thumb_meta, thumb_meta_target)
        write_joint(thumb_prox, thumb_prox_target)
        write_mimic_joint("revo2_right_thumb_distal_joint", thumb_prox_target, 1.0)

        finger_names = ("index", "middle", "ring", "pinky")
        for finger_i, finger_name in enumerate(finger_names, start=2):
            prox = f"revo2_right_{finger_name}_proximal_joint"
            distal = f"revo2_right_{finger_name}_distal_joint"
            prox_target = scale_joint(prox, fractions[:, finger_i])
            write_joint(prox, prox_target)
            write_mimic_joint(distal, prox_target, 1.155)
        return targets

    def _active_object_shape_code(self) -> torch.Tensor:
        shape_codes = getattr(self, "_active_object_shape_codes_tensor", None)
        if shape_codes is not None:
            return shape_codes
        shape = str(self.cfg.object_shape).lower()
        code = {"sphere": 0, "ball": 0, "box": 1, "cube": 1, "cylinder": 2, "cone": 3}.get(shape, 1)
        return torch.full((self.num_envs,), code, dtype=torch.long, device=self.device)

    def _active_object_size(self) -> torch.Tensor:
        size = getattr(self, "_active_object_size_tensor", None)
        if size is not None:
            return size
        return torch.tensor(self.cfg.object_size, dtype=torch.float32, device=self.device).view(1, 3).expand(
            self.num_envs, -1
        )

    def _active_object_radius(self) -> torch.Tensor:
        radius = getattr(self, "_active_object_radius_tensor", None)
        if radius is not None:
            return radius
        return torch.full((self.num_envs,), float(self.cfg.object_radius), dtype=torch.float32, device=self.device)

    def _active_object_height(self) -> torch.Tensor:
        height = getattr(self, "_active_object_height_tensor", None)
        if height is not None:
            return height
        return self._active_object_size()[:, 2]

    def _object_surface_distance(self, rel_w: torch.Tensor) -> torch.Tensor:
        squeeze_points = rel_w.dim() == 2
        if squeeze_points:
            rel_w = rel_w.unsqueeze(1)

        point_count = rel_w.shape[1]
        quat = self._object_quat_w.unsqueeze(1).expand(-1, point_count, -1).reshape(-1, 4)
        local_points = quat_rotate_inverse(quat, rel_w.reshape(-1, 3)).reshape(self.num_envs, point_count, 3)
        shape_codes = self._active_object_shape_code().view(self.num_envs, 1)
        size = self._active_object_size().view(self.num_envs, 1, 3)
        radius = self._active_object_radius().view(self.num_envs, 1)
        height = self._active_object_height().view(self.num_envs, 1)

        sphere_dist = torch.norm(local_points, dim=-1) - radius

        half_size = size * 0.5
        q = torch.abs(local_points) - half_size
        box_outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)
        box_inside = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
        box_dist = box_outside + box_inside

        radial = torch.norm(local_points[..., :2], dim=-1)
        cyl_d = torch.stack((radial - radius, torch.abs(local_points[..., 2]) - 0.5 * height), dim=-1)
        cyl_dist = torch.norm(torch.clamp(cyl_d, min=0.0), dim=-1) + torch.clamp(
            torch.max(cyl_d, dim=-1).values, max=0.0
        )

        z = local_points[..., 2]
        radius_at_z = radius * torch.clamp((0.5 * height - z) / torch.clamp(height, min=1.0e-6), 0.0, 1.0)
        cone_d = torch.stack((radial - radius_at_z, torch.abs(z) - 0.5 * height), dim=-1)
        cone_dist = torch.norm(torch.clamp(cone_d, min=0.0), dim=-1) + torch.clamp(
            torch.max(cone_d, dim=-1).values, max=0.0
        )

        dist = torch.where(
            shape_codes == 0,
            sphere_dist,
            torch.where(shape_codes == 2, cyl_dist, torch.where(shape_codes == 3, cone_dist, box_dist)),
        )
        return dist.squeeze(1) if squeeze_points else dist

    def _compute_intermediate_values(self) -> None:
        self._object_pos_w = self.object.data.root_pos_w
        self._object_quat_w = self.object.data.root_quat_w
        self._object_lin_vel_w = self.object.data.root_lin_vel_w
        self._object_ang_vel_w = self.object.data.root_ang_vel_w
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
        self._finger_count_quality = torch.clamp(
            torch.sum(self._contact_score, dim=-1) / min_finger_contacts,
            0.0,
            1.0,
        )
        self._non_thumb_quality = torch.clamp(
            torch.sum(non_thumb_scores, dim=-1) / min_non_thumb_contacts,
            0.0,
            1.0,
        )

        thumb_vec = rel[:, 0]
        non_thumb_vec = rel[:, 1:]
        denom = torch.clamp(
            torch.norm(thumb_vec, dim=-1, keepdim=True) * torch.norm(non_thumb_vec, dim=-1),
            min=1.0e-6,
        )
        cos = torch.sum(thumb_vec.unsqueeze(1) * non_thumb_vec, dim=-1) / denom
        opposing = torch.relu(-cos)
        opposing = opposing * non_thumb_contacts.float()
        self._opposition_score = torch.max(opposing, dim=-1).values
        threshold = float(self.cfg.opposition_cos_threshold)
        opposition_den = max(threshold + 1.0, 1.0e-6)
        opposition_progress = torch.clamp((threshold - cos) / opposition_den, 0.0, 1.0)
        self._weighted_opposition_score = torch.max(
            opposition_progress * non_thumb_scores * thumb_score.unsqueeze(-1),
            dim=-1,
        ).values
        self._opposing_contact = (
            (cos < threshold) & non_thumb_contacts & self._thumb_contact.unsqueeze(-1)
        ).any(dim=-1)

        quality_finger_count_weight = float(getattr(self.cfg, "grasp_quality_finger_count_weight", 0.30))
        quality_non_thumb_weight = float(getattr(self.cfg, "grasp_quality_non_thumb_weight", 0.25))
        quality_thumb_weight = float(getattr(self.cfg, "grasp_quality_thumb_weight", 0.25))
        quality_opposition_weight = float(getattr(self.cfg, "grasp_quality_opposition_weight", 0.20))
        self._grasp_quality = torch.clamp(
            quality_finger_count_weight * self._finger_count_quality
            + quality_non_thumb_weight * self._non_thumb_quality
            + quality_thumb_weight * thumb_score
            + quality_opposition_weight * self._weighted_opposition_score,
            0.0,
            1.0,
        )

        palm_rel = self._palm_pos_w - self._object_pos_w
        self._palm_surface_dist = self._object_surface_distance(palm_rel)
        palm_contact_distance = max(float(getattr(self.cfg, "palm_contact_distance", 0.10)), 1.0e-6)
        self._palm_contact_score = torch.exp(-torch.relu(self._palm_surface_dist) / palm_contact_distance)
        self._palm_contact = self._palm_surface_dist < palm_contact_distance

        opposition_mode = str(getattr(self.cfg, "true_grasp_opposition_mode", "score")).lower()
        if opposition_mode == "dot":
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
        if self._object_start_pos.shape[0] == self.num_envs:
            object_start_z = self._object_start_pos[:, 2]
        else:
            object_start_z = self._object_start_pos[0, 2]
        self._object_height_delta = self._object_pos_w[:, 2] - object_start_z
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


@torch.jit.script
def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 2.0 * (x - lower) / torch.clamp(upper - lower, min=1.0e-6) - 1.0
