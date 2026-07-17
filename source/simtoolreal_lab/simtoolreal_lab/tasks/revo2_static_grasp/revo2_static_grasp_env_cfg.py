"""Configuration for Franka + BrainCo Revo2 static grasp DirectRLEnv tasks."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

SIMTOOLREAL_LAB_ROOT = Path(
    os.environ.get("SIMTOOLREAL_LAB_ROOT", Path(__file__).resolve().parents[5])
).expanduser().resolve()
SIMTOOLREAL_ROOT = Path(os.environ.get("SIMTOOLREAL_ROOT", "/data1/linsixu/simtoolreal")).expanduser().resolve()
REVO2_URDF = (
    SIMTOOLREAL_ROOT
    / "assets/generated/franka_brainco_revo2_right"
    / "franka_brainco_revo2_right.urdf"
)

FRANKA_ARM_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)

REVO2_HAND_JOINT_NAMES = (
    "revo2_right_thumb_metacarpal_joint",
    "revo2_right_thumb_proximal_joint",
    "revo2_right_index_proximal_joint",
    "revo2_right_middle_proximal_joint",
    "revo2_right_ring_proximal_joint",
    "revo2_right_pinky_proximal_joint",
)

REVO2_FINGERTIP_BODY_NAMES = (
    "revo2_right_thumb_tip_link",
    "revo2_right_index_tip_link",
    "revo2_right_middle_tip_link",
    "revo2_right_ring_tip_link",
    "revo2_right_pinky_tip_link",
)

REVO2_TOUCH_BODY_NAMES = (
    "revo2_right_thumb_touch_link",
    "revo2_right_index_touch_link",
    "revo2_right_middle_touch_link",
    "revo2_right_ring_touch_link",
    "revo2_right_pinky_touch_link",
)

DEFAULT_HAND_OPEN_POS = {
    "revo2_right_thumb_metacarpal_joint": 0.0,
    "revo2_right_thumb_proximal_joint": 0.0,
    "revo2_right_thumb_distal_joint": 0.03958540037274361,
    "revo2_right_index_proximal_joint": 0.0,
    "revo2_right_index_distal_joint": 0.0,
    "revo2_right_middle_proximal_joint": 0.0,
    "revo2_right_middle_distal_joint": 0.0,
    "revo2_right_ring_proximal_joint": 0.0,
    "revo2_right_ring_distal_joint": 0.0,
    "revo2_right_pinky_proximal_joint": 0.0,
    "revo2_right_pinky_distal_joint": 0.0,
}

V325_DEFAULT_ARM_POS = (
    -1.132532000541687,
    1.0449830293655396,
    1.2880635261535645,
    -1.2525746822357178,
    0.582234263420105,
    1.5055428743362427,
    -0.8549104332923889,
)

V327_PREGRASP_ARM_POS = (
    -1.1273080110549927,
    1.1112152338027954,
    1.1573164463043213,
    -1.3805291652679443,
    0.5772636532783508,
    1.5564199686050415,
    -1.061887264251709,
)

V327_LIFT_ARM_DELTA = (
    -0.005224,
    -0.066232,
    0.130747,
    0.127955,
    0.004971,
    -0.050877,
    0.206977,
)

V327_LIFT_ACTION_PRIOR = (
    -0.016,
    -0.197,
    0.389,
    0.381,
    0.015,
    -0.151,
    0.616,
)

V325_VERIFIED_LIFT_ARM_DELTA = (
    -0.020303,
    -0.013322,
    0.180019,
    0.140338,
    -0.056260,
    -0.043493,
    0.236498,
)

V325_VERIFIED_LIFT_ACTION_PRIOR_120 = (
    -0.0705,
    -0.0463,
    0.6251,
    0.4873,
    -0.1953,
    -0.1510,
    0.8212,
)

FRANKA_PLANNER_HOME_ARM_POS = (
    0.0,
    0.19634954084936207,
    0.0,
    -2.617993877991494,
    0.0,
    2.941592653589793,
    0.7853981633974483,
)

FRANKA_SRDF_HOME_ARM_POS = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    3.1416,
    1.5708,
)

FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS = (
    0.0,
    -0.569,
    0.0,
    -2.810,
    0.0,
    3.037,
    0.741,
)

ISAACLAB_BALL_START_POS = (0.580, -0.050, 0.326)
ISAACLAB_BALL_TABLE_POS = (0.580, -0.050, 0.2735)
ISAACLAB_BALL_PREGRASP_TARGET_REL_PALM = (0.015, 0.062, 0.069)
ISAACLAB_BALL_PREGRASP_TARGET_SCALE = (0.035, 0.035, 0.035)

V326_DEFAULT_ARM_POS = (
    -1.3145033121109009,
    1.190942406654358,
    1.564030408859253,
    -1.5763579607009888,
    0.3814008831977844,
    1.742279291152954,
    -0.7931675314903259,
)


def _robot_cfg(default_arm_pos: tuple[float, ...]) -> ArticulationCfg:
    joint_pos = {joint_name: value for joint_name, value in zip(FRANKA_ARM_JOINT_NAMES, default_arm_pos)}
    joint_pos.update(DEFAULT_HAND_OPEN_POS)
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(REVO2_URDF),
            fix_base=True,
            root_link_name="panda_link0",
            merge_fixed_joints=False,
            make_instanceable=False,
            convert_mimic_joints_to_normal_joints=False,
            self_collision=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "franka_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.2,
                stiffness=600.0,
                damping=60.0,
            ),
            "franka_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.7,
                stiffness=600.0,
                damping=60.0,
            ),
            "revo2_hand": ImplicitActuatorCfg(
                joint_names_expr=["revo2_right_.*_joint"],
                effort_limit_sim=180.0,
                velocity_limit_sim=8.0,
                stiffness=320.0,
                damping=32.0,
            ),
        },
    )


def _object_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.75,
        dynamic_friction=0.75,
        restitution=0.0,
    )


def _sphere_object_cfg(radius: float, mass: float, pos: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.SphereCfg(
            radius=radius,
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=4.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=_object_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.12, 0.04), roughness=0.55),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _cube_object_cfg(size: tuple[float, float, float], mass: float, pos: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=size,
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=4.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=_object_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.58, 0.10), roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _table_cfg(
    pos: tuple[float, float, float],
    size: tuple[float, float, float] = (0.82, 0.72, 0.045),
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.25,
                dynamic_friction=0.25,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.36, 0.38, 0.40), roughness=0.7),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class Revo2StaticGraspEnvCfg(DirectRLEnvCfg):
    """Base config shared by the static primitive Revo2 grasp tasks."""

    # env
    decimation = 2
    episode_length_s = 6.0
    action_space = 13
    observation_space = 76
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    robot_physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=3.0,
        dynamic_friction=2.5,
        restitution=0.0,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=1.35, replicate_physics=True)
    video_camera_enabled = False
    video_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/VideoCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.26, 0.78, 0.98), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
        width=640,
        height=360,
    )
    student_camera_enabled = False
    student_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/StudentCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.78, -0.58, 0.72), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 4.0),
        ),
        width=160,
        height=120,
        return_latest_camera_pose=True,
    )

    # robot/object assets; subclasses override robot_cfg/object_cfg and constants.
    robot_cfg: ArticulationCfg = _robot_cfg(V327_PREGRASP_ARM_POS)
    object_cfg: RigidObjectCfg = _sphere_object_cfg(radius=0.030, mass=0.025, pos=ISAACLAB_BALL_START_POS)
    table_cfg: RigidObjectCfg = _table_cfg(pos=ISAACLAB_BALL_TABLE_POS)
    create_table = True

    # named bodies/joints
    arm_joint_names = FRANKA_ARM_JOINT_NAMES
    hand_joint_names = REVO2_HAND_JOINT_NAMES
    fingertip_body_names = REVO2_TOUCH_BODY_NAMES
    touch_body_names = REVO2_TOUCH_BODY_NAMES
    robot_collision_disabled_body_names: tuple[str, ...] = ()
    robot_extra_self_collision_filter_pairs: tuple[tuple[str, str], ...] = ()
    robot_mimic_natural_frequency: float | None = None
    robot_mimic_damping_ratio: float | None = None
    robot_mimic_offset_overrides_deg: tuple[tuple[str, float], ...] = ()
    palm_body_name = "revo2_right_base_link"
    palm_offset = (0.0, 0.0, 0.0)
    fingertip_body_offsets = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    # task constants
    reference_name = "revo2_v327_static_ball_canonical_mount_isaaclab_window_rl"
    object_shape = "sphere"
    object_radius = 0.030
    object_size = (0.060, 0.060, 0.060)
    object_start_pos = ISAACLAB_BALL_START_POS
    object_start_rot = (1.0, 0.0, 0.0, 0.0)
    table_top_z = 0.296
    default_arm_pos = V327_PREGRASP_ARM_POS
    reference_hand_fractions = (0.45, 0.55, 0.30, 0.30, 0.30, 0.30)
    pregrasp_target_rel_palm = ISAACLAB_BALL_PREGRASP_TARGET_REL_PALM
    pregrasp_target_scale = ISAACLAB_BALL_PREGRASP_TARGET_SCALE
    lift_arm_delta = V327_LIFT_ARM_DELTA
    lift_action_prior = V327_LIFT_ACTION_PRIOR
    palm_lift_target_height = 0.055
    palm_lift_baseline_mode = "reset"
    lift_gate_uses_grasp_memory = False
    lift_reward_uses_grasp_memory = True
    lift_progress_min_grasp_gate = 0.25

    # control
    arm_action_scale = 0.45
    arm_moving_average = 0.32
    hand_moving_average = 0.50
    arm_target_clamp_delta = (0.16, 0.28, 0.50, 0.32, 0.36, 0.20, 0.54)
    initial_arm_target_lock_steps = 96
    initial_hand_target_lock_steps = 96
    initial_no_contact_steps = 96
    scripted_action_prior_enabled = False
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_active_residual_scale: float | None = None
    scripted_action_prior_active_arm_residual_scale: float | None = None
    scripted_action_prior_active_hand_residual_scale: float | None = None
    scripted_action_prior_post_grasp_arm_residual_scale: float | None = None
    scripted_action_prior_hand_start_step = 100
    scripted_action_prior_lift_start_step = 150
    scripted_action_prior_lift_steps = 140
    scripted_action_prior_hand_action = 1.0
    scripted_action_prior_hand_action_vector: tuple[float, ...] | None = None
    scripted_action_prior_hand_ramp_steps = 0
    scripted_action_prior_lift_action = V327_LIFT_ACTION_PRIOR
    scripted_action_prior_lift_requires_grasp = False

    # reset noise
    reset_arm_pos_noise = 0.0001
    reset_object_pos_noise = (0.00025, 0.00025, 0.00005)

    # grasp/lift gates
    contact_distance = 0.040
    contact_score_scale = 0.040
    lift_success_height = 0.018
    stable_object_palm_vel = 0.42
    min_non_thumb_contacts = 1
    opposition_cos_threshold = 0.10
    success_hold_steps = 1
    terminate_on_success = True
    workspace_xy_limit = 1.05

    # reward scales
    palm_reach_rew_scale = 2.0
    fingertip_reach_rew_scale = 3.0
    pregrasp_rew_scale = 4.0
    contact_rew_scale = 3.0
    true_grasp_rew_scale = 5.0
    opposition_rew_scale = 4.0
    palm_lift_rew_scale = 8.0
    arm_lift_progress_rew_scale = 8.0
    lift_action_prior_rew_scale = 45.0
    lift_action_prior_lift_gate_min = 1.0
    lift_action_prior_lift_gate_scale = 0.0
    lift_progress_linear_rew_scale = 0.0
    lift_progress_rew_scale = 420.0
    lift_coupling_rew_scale = 180.0
    palm_lift_gap_penalty_scale = 90.0
    premature_arm_lift_penalty_scale = 0.0
    premature_palm_lift_penalty_scale = 0.0
    palm_distance_penalty_scale = 0.0
    palm_distance_penalty_margin = 0.22
    lifted_true_grasp_rew_scale = 700.0
    stable_hold_rew_scale = 1200.0
    hold_progress_rew_scale = 0.0
    success_bonus = 3500.0
    action_penalty_scale = 0.010
    arm_target_delta_penalty_scale = 0.020
    drop_penalty = 25.0


@configclass
class Revo2StaticBallGraspEnvCfg(Revo2StaticGraspEnvCfg):
    """V325-aligned static sphere task."""

    robot_cfg: ArticulationCfg = _robot_cfg(V327_PREGRASP_ARM_POS)
    object_cfg: RigidObjectCfg = _sphere_object_cfg(radius=0.030, mass=0.025, pos=ISAACLAB_BALL_START_POS)
    table_cfg: RigidObjectCfg = _table_cfg(pos=ISAACLAB_BALL_TABLE_POS)
    reference_name = "revo2_v327_static_ball_canonical_mount_isaaclab_window_rl"
    object_shape = "sphere"
    object_radius = 0.030
    object_size = (0.060, 0.060, 0.060)
    object_start_pos = ISAACLAB_BALL_START_POS
    table_top_z = 0.296
    default_arm_pos = V327_PREGRASP_ARM_POS
    reference_hand_fractions = (0.45, 0.55, 0.30, 0.30, 0.30, 0.30)
    pregrasp_target_rel_palm = ISAACLAB_BALL_PREGRASP_TARGET_REL_PALM
    pregrasp_target_scale = ISAACLAB_BALL_PREGRASP_TARGET_SCALE


@configclass
class Revo2StaticBallInitialPoseGraspEnvCfg(Revo2StaticBallGraspEnvCfg):
    """Static sphere task reset from the V325 initial arm pose, with no scripted prior."""

    robot_cfg: ArticulationCfg = _robot_cfg(V325_DEFAULT_ARM_POS)
    reference_name = "revo2_v325_initial_pose_static_ball_canonical_mount_from_scratch_rl"
    episode_length_s = 8.0
    default_arm_pos = V325_DEFAULT_ARM_POS
    lift_arm_delta = V325_VERIFIED_LIFT_ARM_DELTA
    lift_action_prior = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120


@configclass
class Revo2StaticBallInitialPoseLiftGraspEnvCfg(Revo2StaticBallInitialPoseGraspEnvCfg):
    """Initial-pose sphere task with reward terms gated by actual object lift."""

    reference_name = "revo2_v325_initial_pose_static_ball_lift_gated_from_scratch_rl"
    lift_action_prior_rew_scale = 8.0
    lift_action_prior_lift_gate_min = 0.05
    lift_action_prior_lift_gate_scale = 1.0
    lift_progress_linear_rew_scale = 650.0
    lift_progress_rew_scale = 950.0
    lift_coupling_rew_scale = 260.0
    arm_lift_progress_rew_scale = 2.0
    palm_lift_rew_scale = 0.0
    palm_lift_gap_penalty_scale = 0.0
    lifted_true_grasp_rew_scale = 1800.0
    stable_hold_rew_scale = 2600.0
    success_bonus = 9000.0
    action_penalty_scale = 0.020


@configclass
class Revo2StaticBallInitialPoseStrictLiftGraspEnvCfg(Revo2StaticBallInitialPoseLiftGraspEnvCfg):
    """Initial-pose sphere task that requires a visibly sustained lift before success."""

    reference_name = "revo2_v325_initial_pose_static_ball_strict_lift_hold_from_scratch_rl"
    lift_success_height = 0.040
    stable_object_palm_vel = 0.24
    success_hold_steps = 30
    lift_progress_linear_rew_scale = 850.0
    lift_progress_rew_scale = 1300.0
    lift_coupling_rew_scale = 320.0
    lifted_true_grasp_rew_scale = 2600.0
    stable_hold_rew_scale = 4200.0
    hold_progress_rew_scale = 6500.0
    success_bonus = 14000.0
    drop_penalty = 70.0


@configclass
class Revo2StaticBallFrankaPlannerHomeStrictLiftGraspEnvCfg(Revo2StaticBallInitialPoseStrictLiftGraspEnvCfg):
    """Static sphere task reset from the Franka planner home pose in the generated asset config."""

    robot_cfg: ArticulationCfg = _robot_cfg(FRANKA_PLANNER_HOME_ARM_POS)
    reference_name = "revo2_franka_planner_home_static_ball_strict_lift_hold_from_scratch_rl"
    default_arm_pos = FRANKA_PLANNER_HOME_ARM_POS
    initial_arm_target_lock_steps = 0
    initial_hand_target_lock_steps = 0
    initial_no_contact_steps = 0
    reset_arm_pos_noise = 0.0
    reset_object_pos_noise = (0.0005, 0.0005, 0.00005)
    arm_action_scale = 0.55
    arm_moving_average = 0.40
    arm_target_clamp_delta = (2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25)
    episode_length_s = 10.0
    palm_reach_rew_scale = 8.0
    fingertip_reach_rew_scale = 6.0
    pregrasp_rew_scale = 0.0
    lift_action_prior_rew_scale = 0.0
    arm_lift_progress_rew_scale = 0.0


@configclass
class Revo2StaticBallFrankaSrdfHomeStrictLiftGraspEnvCfg(Revo2StaticBallFrankaPlannerHomeStrictLiftGraspEnvCfg):
    """Static sphere task reset from the Franka SRDF home group_state."""

    robot_cfg: ArticulationCfg = _robot_cfg(FRANKA_SRDF_HOME_ARM_POS)
    reference_name = "revo2_franka_srdf_home_static_ball_strict_lift_hold_from_scratch_rl"
    default_arm_pos = FRANKA_SRDF_HOME_ARM_POS


@configclass
class Revo2StaticBallFrankaDefaultHomeStrictLiftGraspEnvCfg(Revo2StaticBallFrankaPlannerHomeStrictLiftGraspEnvCfg):
    """Static sphere task reset from IsaacLab's default Franka Panda joint pose."""

    robot_cfg: ArticulationCfg = _robot_cfg(FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS)
    reference_name = "revo2_franka_isaaclab_default_home_static_ball_strict_lift_hold_from_scratch_rl"
    default_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS


@configclass
class Revo2StaticBallFrankaDefaultHomeLiftShapedStrictLiftGraspEnvCfg(
    Revo2StaticBallFrankaDefaultHomeStrictLiftGraspEnvCfg
):
    """Default-home sphere task with staged lift shaping, but no scripted action prior."""

    reference_name = "revo2_franka_isaaclab_default_home_static_ball_lift_shaped_strict_lift_hold_from_scratch_rl"
    palm_lift_baseline_mode = "min"
    contact_rew_scale = 4.0
    true_grasp_rew_scale = 8.0
    opposition_rew_scale = 5.0
    lift_action_prior_rew_scale = 8.0
    lift_action_prior_lift_gate_min = 0.05
    lift_action_prior_lift_gate_scale = 1.0
    arm_lift_progress_rew_scale = 2.0
    palm_lift_rew_scale = 2.0


@configclass
class Revo2StaticBallFrankaDefaultHomeReachThenLiftStrictLiftGraspEnvCfg(
    Revo2StaticBallFrankaDefaultHomeStrictLiftGraspEnvCfg
):
    """Default-home sphere task that strongly separates reach/contact from lift."""

    reference_name = "revo2_franka_isaaclab_default_home_static_ball_reach_then_lift_strict_lift_hold_from_scratch_rl"
    palm_lift_baseline_mode = "min"
    palm_reach_rew_scale = 14.0
    fingertip_reach_rew_scale = 12.0
    pregrasp_rew_scale = 6.0
    contact_rew_scale = 10.0
    true_grasp_rew_scale = 24.0
    opposition_rew_scale = 12.0
    lift_action_prior_rew_scale = 2.0
    lift_action_prior_lift_gate_min = 0.12
    lift_action_prior_lift_gate_scale = 1.0
    lift_gate_uses_grasp_memory = True
    lift_reward_uses_grasp_memory = False
    lift_progress_min_grasp_gate = 0.0
    arm_lift_progress_rew_scale = 0.0
    palm_lift_rew_scale = 0.0
    lift_action_prior_rew_scale = 0.5
    lift_progress_linear_rew_scale = 2200.0
    lift_progress_rew_scale = 3200.0
    lift_coupling_rew_scale = 800.0
    palm_lift_gap_penalty_scale = 2500.0
    lifted_true_grasp_rew_scale = 6000.0
    stable_hold_rew_scale = 6000.0
    hold_progress_rew_scale = 9000.0
    success_bonus = 25000.0
    premature_arm_lift_penalty_scale = 18.0
    premature_palm_lift_penalty_scale = 22.0
    action_penalty_scale = 0.016


@configclass
class Revo2StaticBallFrankaDefaultHomeHoldPriorityStrictLiftGraspEnvCfg(
    Revo2StaticBallFrankaDefaultHomeReachThenLiftStrictLiftGraspEnvCfg
):
    """Default-home sphere task tuned to reward sustained grasped lift over transient lift spikes."""

    reference_name = "revo2_franka_isaaclab_default_home_static_ball_hold_priority_strict_lift_from_scratch_rl"
    contact_rew_scale = 12.0
    true_grasp_rew_scale = 28.0
    opposition_rew_scale = 14.0
    lift_action_prior_rew_scale = 0.2
    lift_progress_linear_rew_scale = 1200.0
    lift_progress_rew_scale = 1800.0
    lift_coupling_rew_scale = 500.0
    palm_lift_gap_penalty_scale = 600.0
    lifted_true_grasp_rew_scale = 1500.0
    stable_hold_rew_scale = 1000.0
    hold_progress_rew_scale = 12000.0
    success_bonus = 30000.0
    success_hold_steps = 20
    premature_arm_lift_penalty_scale = 10.0
    premature_palm_lift_penalty_scale = 12.0


@configclass
class Revo2StaticBallFrankaDefaultHomeHoldPriorityV2StrictLiftGraspEnvCfg(
    Revo2StaticBallFrankaDefaultHomeHoldPriorityStrictLiftGraspEnvCfg
):
    """Default-home hold-priority task with stronger grasp retention and gentler PPO-friendly lift shaping."""

    reference_name = "revo2_franka_isaaclab_default_home_static_ball_hold_priority_v2_strict_lift_from_scratch_rl"
    palm_reach_rew_scale = 18.0
    fingertip_reach_rew_scale = 16.0
    pregrasp_rew_scale = 8.0
    contact_rew_scale = 18.0
    true_grasp_rew_scale = 48.0
    opposition_rew_scale = 22.0
    lift_action_prior_rew_scale = 0.1
    lift_progress_linear_rew_scale = 900.0
    lift_progress_rew_scale = 1400.0
    lift_coupling_rew_scale = 650.0
    palm_lift_gap_penalty_scale = 1000.0
    lifted_true_grasp_rew_scale = 1600.0
    stable_hold_rew_scale = 1600.0
    hold_progress_rew_scale = 16000.0
    success_bonus = 42000.0
    premature_arm_lift_penalty_scale = 18.0
    premature_palm_lift_penalty_scale = 40.0
    action_penalty_scale = 0.014


@configclass
class Revo2StaticBallFrankaDefaultHomeHoldPriorityV3StrictLiftGraspEnvCfg(
    Revo2StaticBallFrankaDefaultHomeHoldPriorityStrictLiftGraspEnvCfg
):
    """Default-home hold-priority task with an explicit penalty for moving the palm away from the object."""

    reference_name = "revo2_franka_isaaclab_default_home_static_ball_hold_priority_v3_distance_penalty_strict_lift_from_scratch_rl"
    contact_rew_scale = 14.0
    true_grasp_rew_scale = 34.0
    opposition_rew_scale = 16.0
    lift_action_prior_rew_scale = 0.15
    lift_progress_linear_rew_scale = 1100.0
    lift_progress_rew_scale = 1700.0
    lift_coupling_rew_scale = 550.0
    palm_lift_gap_penalty_scale = 700.0
    palm_distance_penalty_scale = 160.0
    palm_distance_penalty_margin = 0.18
    lifted_true_grasp_rew_scale = 1700.0
    stable_hold_rew_scale = 1800.0
    hold_progress_rew_scale = 16000.0
    success_bonus = 42000.0
    premature_arm_lift_penalty_scale = 18.0
    premature_palm_lift_penalty_scale = 32.0
    action_penalty_scale = 0.014


@configclass
class Revo2StaticBallFrankaDefaultHomeHoldPriorityV4StrictLiftGraspEnvCfg(
    Revo2StaticBallFrankaDefaultHomeHoldPriorityV3StrictLiftGraspEnvCfg
):
    """Default-home hold-priority task with stronger pre-grasp lift suppression."""

    reference_name = "revo2_franka_isaaclab_default_home_static_ball_hold_priority_v4_pregrasp_lift_penalty_strict_lift_from_scratch_rl"
    lift_action_prior_rew_scale = 0.05
    palm_lift_gap_penalty_scale = 1000.0
    premature_arm_lift_penalty_scale = 45.0
    premature_palm_lift_penalty_scale = 220.0


@configclass
class Revo2StaticBallFrankaDefaultHomeHoldPriorityV5StrictLiftGraspEnvCfg(
    Revo2StaticBallFrankaDefaultHomeHoldPriorityV3StrictLiftGraspEnvCfg
):
    """Default-home task that preserves grasp while re-opening the path to object lift."""

    reference_name = "revo2_franka_isaaclab_default_home_static_ball_hold_priority_v5_lift_unlock_strict_lift_from_scratch_rl"
    contact_rew_scale = 18.0
    true_grasp_rew_scale = 48.0
    opposition_rew_scale = 22.0
    lift_gate_uses_grasp_memory = False
    lift_reward_uses_grasp_memory = False
    lift_progress_min_grasp_gate = 0.35
    lift_action_prior_rew_scale = 0.0
    palm_lift_rew_scale = 0.35
    arm_lift_progress_rew_scale = 0.0
    lift_progress_linear_rew_scale = 2200.0
    lift_progress_rew_scale = 3600.0
    lift_coupling_rew_scale = 180.0
    palm_lift_gap_penalty_scale = 260.0
    palm_distance_penalty_scale = 90.0
    palm_distance_penalty_margin = 0.20
    lifted_true_grasp_rew_scale = 5200.0
    stable_hold_rew_scale = 9000.0
    hold_progress_rew_scale = 18000.0
    success_bonus = 52000.0
    success_hold_steps = 12
    stable_object_palm_vel = 0.32
    premature_arm_lift_penalty_scale = 12.0
    premature_palm_lift_penalty_scale = 45.0
    action_penalty_scale = 0.012


@configclass
class Revo2StaticBallResidualGraspEnvCfg(Revo2StaticBallGraspEnvCfg):
    """Residual-RL sphere task with a scripted close/lift action prior."""

    reference_name = "revo2_v327_static_ball_canonical_mount_residual_lift_prior_rl"
    episode_length_s = 12.0
    lift_arm_delta = V325_VERIFIED_LIFT_ARM_DELTA
    lift_action_prior = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    initial_hand_target_lock_steps = 0
    scripted_action_prior_enabled = True
    scripted_action_prior_residual_scale = 0.50
    scripted_action_prior_hand_start_step = 0
    scripted_action_prior_hand_ramp_steps = 120
    scripted_action_prior_lift_start_step = 150
    scripted_action_prior_lift_steps = 120
    scripted_action_prior_hand_action = 1.0
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_requires_grasp = True
    action_penalty_scale = 0.05


@configclass
class Revo2StaticCube3cmGraspEnvCfg(Revo2StaticGraspEnvCfg):
    """V326-aligned static 3 cm cube task."""

    robot_cfg: ArticulationCfg = _robot_cfg(V326_DEFAULT_ARM_POS)
    object_cfg: RigidObjectCfg = _cube_object_cfg(size=(0.030, 0.030, 0.030), mass=0.015, pos=(0.0, 0.0, 0.551))
    table_cfg: RigidObjectCfg = _table_cfg(pos=(0.0, 0.0, 0.5055))
    reference_name = "revo2_v326_static_cube3cm_current_verified_window_rl"
    object_shape = "cube"
    object_radius = 0.015
    object_size = (0.030, 0.030, 0.030)
    object_start_pos = (0.0, 0.0, 0.551)
    default_arm_pos = V326_DEFAULT_ARM_POS
    reference_hand_fractions = (0.45, 0.75, 0.50, 0.50, 0.50, 0.50)
    pregrasp_target_rel_palm = (0.045, 0.034, -0.007)
    pregrasp_target_scale = (0.025, 0.025, 0.014)
