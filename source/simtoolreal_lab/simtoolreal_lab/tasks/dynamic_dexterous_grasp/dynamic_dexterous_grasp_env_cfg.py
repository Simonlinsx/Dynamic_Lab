"""Configuration for IsaacLab dynamic dexterous grasp teacher tasks."""

from __future__ import annotations

import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass

from simtoolreal_lab.tasks.revo2_static_grasp.revo2_static_grasp_env_cfg import (
    DEFAULT_HAND_OPEN_POS,
    FRANKA_ARM_JOINT_NAMES,
    FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS,
    REVO2_HAND_JOINT_NAMES,
    REVO2_TOUCH_BODY_NAMES,
    Revo2StaticGraspEnvCfg,
    SIMTOOLREAL_LAB_ROOT,
    SIMTOOLREAL_ROOT,
    V325_VERIFIED_LIFT_ARM_DELTA,
    V325_VERIFIED_LIFT_ACTION_PRIOR_120,
    V327_PREGRASP_ARM_POS,
    V327_LIFT_ARM_DELTA,
    _cube_object_cfg,
    _object_material,
    _sphere_object_cfg,
    _table_cfg,
)

V699_REVO2_URDF = (
    SIMTOOLREAL_ROOT
    / "assets/generated/franka_brainco_revo2_right_v699"
    / "franka_brainco_revo2_right.urdf"
)
INSPIRE_Z180_URDF = (
    SIMTOOLREAL_ROOT
    / "assets/embodiments/franka-inspire-z180"
    / "franka_inspire_z180.urdf"
)

# The IsaacGym V699 YAML stores panda_joint4 as +1.376, but the IsaacLab
# Franka URDF joint limit is negative for joint4. Use the legal IsaacLab-side
# pose here and keep exact V699 replay conversion separate from env reset.
V699_FRONT110_ARM_POS = (-1.571, 1.571, -0.000, -1.376, -0.000, 1.485, 2.358)
ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS = (
    0.0,
    -1.20,
    0.0,
    -2.20,
    0.0,
    1.40,
    0.7853981633974483,
)

REVO2_V699_PHYSICAL_HAND_JOINT_NAMES = (
    "revo2_right_index_proximal_joint",
    "revo2_right_index_distal_joint",
    "revo2_right_middle_proximal_joint",
    "revo2_right_middle_distal_joint",
    "revo2_right_pinky_proximal_joint",
    "revo2_right_pinky_distal_joint",
    "revo2_right_ring_proximal_joint",
    "revo2_right_ring_distal_joint",
    "revo2_right_thumb_metacarpal_joint",
    "revo2_right_thumb_proximal_joint",
    "revo2_right_thumb_distal_joint",
)
INSPIRE_HAND_JOINT_NAMES = (
    "index_proximal_joint",
    "index_intermediate_joint",
    "middle_proximal_joint",
    "middle_intermediate_joint",
    "pinky_proximal_joint",
    "pinky_intermediate_joint",
    "ring_proximal_joint",
    "ring_intermediate_joint",
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
)
INSPIRE_ACTIVE_HAND_JOINT_NAMES = (
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)
INSPIRE_TABLETOP_CLEARANCE_BODY_NAMES = (
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_link8",
    "panda_hand",
    "hand_base_link",
    "index_proximal",
    "index_intermediate",
    "index_tip",
    "middle_proximal",
    "middle_intermediate",
    "middle_tip",
    "ring_proximal",
    "ring_intermediate",
    "ring_tip",
    "pinky_proximal",
    "pinky_intermediate",
    "pinky_tip",
    "thumb_proximal_base",
    "thumb_proximal",
    "thumb_intermediate",
    "thumb_distal",
    "thumb_tip",
)
INSPIRE_TABLETOP_CLEARANCE_BODY_MARGINS = (
    0.075,
    0.085,
    0.075,
    0.060,
    0.045,
    0.032,
    0.024,
    0.018,
    0.070,
    0.052,
    0.040,
    0.030,
    0.052,
    0.040,
    0.030,
    0.052,
    0.040,
    0.030,
    0.052,
    0.040,
    0.030,
    0.060,
    0.050,
    0.040,
    0.034,
    0.030,
)
INSPIRE_TABLETOP_STRICT_CLEARANCE_BODY_MARGINS = (
    0.075,
    0.085,
    0.075,
    0.060,
    0.045,
    0.032,
    0.030,
    0.040,
    0.095,
    0.075,
    0.060,
    0.050,
    0.075,
    0.060,
    0.050,
    0.075,
    0.060,
    0.050,
    0.075,
    0.060,
    0.050,
    0.085,
    0.075,
    0.060,
    0.050,
    0.045,
)
INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_NAMES = INSPIRE_TABLETOP_CLEARANCE_BODY_NAMES[7:]
INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_MARGINS = INSPIRE_TABLETOP_CLEARANCE_BODY_MARGINS[7:]
INSPIRE_TABLETOP_HAND_STRICT_CLEARANCE_BODY_MARGINS = INSPIRE_TABLETOP_STRICT_CLEARANCE_BODY_MARGINS[7:]
INSPIRE_TABLETOP_HAND_CAUTION_CLEARANCE_BODY_MARGINS = tuple(
    margin + 0.010 for margin in INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_MARGINS
)
# RH56BFX official speeds from Inspire Robots' product page, converted from
# deg/s to rad/s.  The published grip forces are fingertip forces, so keep the
# URDF torque effort unless a calibrated transmission model is available.
INSPIRE_RH56BFX_THUMB_YAW_VEL = 4.10  # 235 deg/s
INSPIRE_RH56BFX_THUMB_FLEX_VEL = 2.62  # 150 deg/s
INSPIRE_RH56BFX_FINGER_FLEX_VEL = 9.95  # 570 deg/s
# The official RH56BFX spec reports fingertip force, not URDF joint torque.
# Use a stronger simulated servo cap so light object/table contacts do not
# push the follower joints far away from the 6-active-DOF command manifold.
INSPIRE_RH56BFX_HAND_EFFORT = 12.0
INSPIRE_RH56BFX_HAND_STIFFNESS = 140.0
INSPIRE_RH56BFX_HAND_DAMPING = 14.0
# Conservative 6-active-DOF Inspire close envelope derived from AnyDex's
# width_12Dangle_6Dangle table, converted to the Isaac/URDF joint order above.
# This keeps the simulated follower joints inside the real hand's coupled
# motion range instead of driving every revolute joint to its URDF limit.
INSPIRE_ANYDEX_P80_CLOSE_TARGETS = (
    0.541,  # index_proximal_joint
    0.478,  # index_intermediate_joint
    0.617,  # middle_proximal_joint
    0.558,  # middle_intermediate_joint
    0.705,  # pinky_proximal_joint
    0.647,  # pinky_intermediate_joint
    0.719,  # ring_proximal_joint
    0.660,  # ring_intermediate_joint
    1.300,  # thumb_proximal_yaw_joint
    0.272,  # thumb_proximal_pitch_joint
    0.180,  # thumb_intermediate_joint
    0.443,  # thumb_distal_joint
)
# AnyDex Sphere_3_Finger width=4.0 cm target, converted to the Isaac/URDF
# joint order.  This is still an official coupled Inspire posture, but it
# closes the ring/pinky side enough for the 4.4 cm rolling ball.
INSPIRE_ANYDEX_SPHERE_4CM_CLOSE_TARGETS = (
    0.429,  # index_proximal_joint
    0.351,  # index_intermediate_joint
    0.543,  # middle_proximal_joint
    0.480,  # middle_intermediate_joint
    1.534,  # pinky_proximal_joint
    1.306,  # pinky_intermediate_joint
    0.909,  # ring_proximal_joint
    0.837,  # ring_intermediate_joint
    1.083,  # thumb_proximal_yaw_joint
    0.180,  # thumb_proximal_pitch_joint
    0.092,  # thumb_intermediate_joint
    0.383,  # thumb_distal_joint
)
# Safer blend between P80 and Sphere_3_Finger width=4.0 cm.  The full sphere
# posture over-flexes the pinky side in IsaacLab with self-collision enabled,
# which can make the solver crawl.  This keeps the useful wrap bias without
# pushing the coupled joints to their most aggressive AnyDex angle.
INSPIRE_ANYDEX_SPHERE_SAFE_CLOSE_TARGETS = (
    0.500,  # index_proximal_joint
    0.420,  # index_intermediate_joint
    0.585,  # middle_proximal_joint
    0.520,  # middle_intermediate_joint
    0.980,  # pinky_proximal_joint
    0.860,  # pinky_intermediate_joint
    0.860,  # ring_proximal_joint
    0.780,  # ring_intermediate_joint
    1.120,  # thumb_proximal_yaw_joint
    0.200,  # thumb_proximal_pitch_joint
    0.110,  # thumb_intermediate_joint
    0.400,  # thumb_distal_joint
)
INSPIRE_ANYDEX_P80_THUMB_WRAP_CLOSE_TARGETS = (
    0.541,  # index_proximal_joint
    0.478,  # index_intermediate_joint
    0.617,  # middle_proximal_joint
    0.558,  # middle_intermediate_joint
    0.705,  # pinky_proximal_joint
    0.647,  # pinky_intermediate_joint
    0.719,  # ring_proximal_joint
    0.660,  # ring_intermediate_joint
    1.580,  # thumb_proximal_yaw_joint
    0.420,  # thumb_proximal_pitch_joint
    0.260,  # thumb_intermediate_joint
    0.560,  # thumb_distal_joint
)
# Keep the sphere-safe non-thumb posture and move only the thumb halfway toward
# the P80 thumb-wrap target.  This probes whether lift failures are caused by a
# too-open thumb without changing the learned grasp family.
INSPIRE_ANYDEX_SPHERE_MILD_THUMB_WRAP_CLOSE_TARGETS = (
    0.500,  # index_proximal_joint
    0.420,  # index_intermediate_joint
    0.585,  # middle_proximal_joint
    0.520,  # middle_intermediate_joint
    0.980,  # pinky_proximal_joint
    0.860,  # pinky_intermediate_joint
    0.860,  # ring_proximal_joint
    0.780,  # ring_intermediate_joint
    1.350,  # thumb_proximal_yaw_joint
    0.310,  # thumb_proximal_pitch_joint
    0.185,  # thumb_intermediate_joint
    0.480,  # thumb_distal_joint
)
INSPIRE_OFFICIAL_ZERO_HAND_OPEN_POS = {joint_name: 0.0 for joint_name in INSPIRE_HAND_JOINT_NAMES}
INSPIRE_V340_KNOWN_GOOD_ARM_POS = (
    0.0,
    -0.35,
    0.0,
    -2.20,
    0.0,
    2.35,
    0.7853981633974483,
)
INSPIRE_V341_CLEAR_ARM_POS = (
    0.0,
    -0.35,
    0.0,
    -2.20,
    0.0,
    2.39,
    0.7853981633974483,
)
# V340 clean-reset lift direction, measured in IsaacLab with
# scripts/probe_arm_action_directions.py.  The old V325 lift prior drives the
# V340 wrist far sideways; j6:+ is the cleanest table-clear upward motion.
INSPIRE_V340_LIFT_ARM_DELTA = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.18,
    0.0,
)
INSPIRE_V340_LIFT_ACTION_PRIOR = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.75,
    0.0,
)
INSPIRE_P80_HOME_SEED_ARM_POS = (
    0.0,
    -0.5690000057,
    0.0,
    -2.8099999428,
    0.0,
    3.0369999409,
    1.6660000086,
)
INSPIRE_P80_HOME_SEED_HAND_FRACTIONS = (0.70, 1.0, 0.80, 0.80, 1.0, 1.0)
INSPIRE_P80_HOME_SEED_LIFT_DELTA = (
    -0.005224,
    -0.066232,
    0.130747,
    0.127955,
    0.004971,
    -0.050877,
    0.206977,
)
INSPIRE_V340_LIFT_CANDIDATE_LABELS = (
    "j6p075_current",
    "j6p050",
    "j6p035",
    "j6p050_j2m025",
    "j6p050_j2p025",
    "j6p050_j4p030",
    "j6p050_j7p030",
    "j6p050_j2m020_j4p030_j7p030",
    "p80_shape_medium",
    "p80_shape_strong",
    "p80_shape_no_j7",
    "p80_shape_j6_zero",
    "v325_prior",
    "j3p050_j4p050_j7p050",
    "j2m030_j3p055_j4p055_j6m020_j7p070",
    "j2m045_j3p075_j4p075_j6m030_j7p090",
)
INSPIRE_V340_LIFT_CANDIDATE_ACTIONS = (
    INSPIRE_V340_LIFT_ACTION_PRIOR,
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.50, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.0),
    (0.0, -0.25, 0.0, 0.0, 0.0, 0.50, 0.0),
    (0.0, 0.25, 0.0, 0.0, 0.0, 0.50, 0.0),
    (0.0, 0.0, 0.0, 0.30, 0.0, 0.50, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.50, 0.30),
    (0.0, -0.20, 0.0, 0.30, 0.0, 0.50, 0.30),
    (0.0, -0.25, 0.50, 0.50, 0.0, -0.20, 0.70),
    (0.0, -0.40, 0.80, 0.80, 0.0, -0.35, 1.00),
    (0.0, -0.35, 0.75, 0.75, 0.0, -0.30, 0.0),
    (0.0, -0.35, 0.75, 0.75, 0.0, 0.0, 0.85),
    V325_VERIFIED_LIFT_ACTION_PRIOR_120,
    (0.0, 0.0, 0.50, 0.50, 0.0, 0.0, 0.50),
    (0.0, -0.30, 0.55, 0.55, 0.0, -0.20, 0.70),
    (0.0, -0.45, 0.75, 0.75, 0.0, -0.30, 0.90),
)
INSPIRE_V340_HAND_MEMORY_CANDIDATE_LABELS = (
    "scalar095_baseline",
    "all100",
    "thumb100_finger095",
    "thumb100_indexmid095_ring085",
    "thumb100_indexmid085_ring095",
    "thumb100_finger085",
    "yaw090_flex100_finger095",
    "yaw100_flex090_finger095",
)
INSPIRE_V340_HAND_MEMORY_CANDIDATE_ACTIONS = (
    (0.95, 0.95, 0.95, 0.95, 0.95, 0.95),
    (1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
    (1.00, 1.00, 0.95, 0.95, 0.95, 0.95),
    (1.00, 1.00, 0.95, 0.95, 0.85, 0.85),
    (1.00, 1.00, 0.85, 0.85, 0.95, 0.95),
    (1.00, 1.00, 0.85, 0.85, 0.85, 0.85),
    (0.90, 1.00, 0.95, 0.95, 0.95, 0.95),
    (1.00, 0.90, 0.95, 0.95, 0.95, 0.95),
)
INSPIRE_V340_RELATIVE_LIFT_TARGET_LABELS = (
    "p80_delta_025",
    "p80_delta_050",
    "p80_delta_075",
    "p80_delta_100",
    "p80_delta_125",
    "j6_delta_008",
    "j6_delta_012",
    "j6_delta_016",
    "j6_delta_010_j7_delta_008",
    "mild_p80_shape",
    "mid_p80_shape",
    "elbow_j6pos_shape",
)
INSPIRE_V340_RELATIVE_LIFT_TARGET_DELTAS = (
    (-0.001306, -0.016558, 0.032687, 0.031989, 0.001243, -0.012719, 0.051744),
    (-0.002612, -0.033116, 0.065374, 0.063978, 0.002486, -0.025439, 0.103489),
    (-0.003918, -0.049674, 0.098060, 0.095966, 0.003728, -0.038158, 0.155233),
    INSPIRE_P80_HOME_SEED_LIFT_DELTA,
    (-0.006530, -0.082790, 0.163434, 0.159944, 0.006214, -0.063596, 0.258721),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.080000, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.120000, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.160000, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.100000, 0.080000),
    (0.0, -0.030000, 0.060000, 0.060000, 0.0, -0.020000, 0.100000),
    (0.0, -0.040000, 0.080000, 0.080000, 0.0, -0.030000, 0.140000),
    (0.0, -0.020000, 0.040000, 0.040000, 0.0, 0.080000, 0.080000),
)
INSPIRE_V340_RELATIVE_LIFT_SCALE_PROBE_FACTORS = (0.65, 0.85, 1.00, 1.15)
INSPIRE_V340_RELATIVE_LIFT_SCALE_PROBE_LABELS = tuple(
    f"relative_lift_scale_{scale:.2f}" for scale in INSPIRE_V340_RELATIVE_LIFT_SCALE_PROBE_FACTORS
)
INSPIRE_V340_RELATIVE_LIFT_SCALE_PROBE_DELTAS = tuple(
    tuple(scale * value for value in INSPIRE_V340_RELATIVE_LIFT_TARGET_DELTAS[2])
    for scale in INSPIRE_V340_RELATIVE_LIFT_SCALE_PROBE_FACTORS
)
INSPIRE_V340_RELATIVE_LIFT_HIGH_SCALE_PROBE_FACTORS = (1.15, 1.30, 1.45, 1.60)
INSPIRE_V340_RELATIVE_LIFT_HIGH_SCALE_PROBE_LABELS = tuple(
    f"relative_lift_scale_{scale:.2f}" for scale in INSPIRE_V340_RELATIVE_LIFT_HIGH_SCALE_PROBE_FACTORS
)
INSPIRE_V340_RELATIVE_LIFT_HIGH_SCALE_PROBE_DELTAS = tuple(
    tuple(scale * value for value in INSPIRE_V340_RELATIVE_LIFT_TARGET_DELTAS[2])
    for scale in INSPIRE_V340_RELATIVE_LIFT_HIGH_SCALE_PROBE_FACTORS
)
INSPIRE_FINGERTIP_BODY_NAMES = (
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
)
INSPIRE_PALM_OFFSET = (0.0, 0.0, 0.02)
INSPIRE_FINGERTIP_BODY_OFFSETS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)
INSPIRE_DEFAULT_HAND_OPEN_POS = {joint_name: 0.0 for joint_name in INSPIRE_HAND_JOINT_NAMES}
# With self-collision enabled, the mounted RH56 hand's neutral/open thumb yaw
# settles near 1.1 rad in IsaacLab and DOMINO uses the same yaw during lift.
# Treat that as the open semantic target so "open hand" does not fight the
# hand's own geometry before contact.
INSPIRE_DEFAULT_HAND_OPEN_POS["thumb_proximal_yaw_joint"] = 1.10

TABLETOP_AFFORDANCE_ROOT = SIMTOOLREAL_LAB_ROOT / "assets/affordance_labels"
TABLETOP_GENERATED_MESH_ROOT = SIMTOOLREAL_LAB_ROOT / "assets/generated/tabletop_affordance_meshes"
TABLETOP_AFFORDANCE_ANALYSIS = TABLETOP_AFFORDANCE_ROOT / "analysis/all_grasp_affordance_refined_v2_analysis.json"
TABLETOP_LARGE_TABLE_POS = (0.58, -0.08, 0.2735)
TABLETOP_LARGE_TABLE_SIZE = (1.00, 0.80, 0.045)
TABLETOP_LARGE_WORKSPACE_X = (0.26, 0.90)
TABLETOP_LARGE_WORKSPACE_Y = (-0.36, 0.24)
TABLETOP_LARGE_TURNTABLE_CENTER = (0.58, -0.08)
TABLETOP_LARGE_TURNTABLE_RADIUS_RANGE = (0.10, 0.26)
TABLETOP_SMALL_BALL_RADIUS = 0.022
TABLETOP_SMALL_BALL_SIZE = (0.044, 0.044, 0.044)
TABLETOP_SMALL_BALL_START_Z = 0.320
TABLETOP_FULL_HEADING_RANGE = (-3.141592653589793, 3.141592653589793)
FALLING_BATON_AFFORDANCE_URDF = (
    SIMTOOLREAL_LAB_ROOT / "assets/generated/falling_baton_affordance/baton_red_green.urdf"
)
FALLING_BATON_AFFORDANCE_SIZE = (0.016, 0.016, 0.165)
FALLING_BATON_AFFORDANCE_MASS = 0.015
FALLING_BATON_PHYSICS_SIZE = (0.018, 0.018, 0.180)
FALLING_BATON_MARKER_WIDTH = 0.020
FALLING_BATON_POSITIVE_MARKER_SIZE = (FALLING_BATON_MARKER_WIDTH, FALLING_BATON_MARKER_WIDTH, 0.0627)
FALLING_BATON_NEUTRAL_MARKER_SIZE = (FALLING_BATON_MARKER_WIDTH, FALLING_BATON_MARKER_WIDTH, 0.02805)
FALLING_BATON_NEGATIVE_MARKER_SIZE = (FALLING_BATON_MARKER_WIDTH, FALLING_BATON_MARKER_WIDTH, 0.07425)
FALLING_BATON_POSITIVE_MARKER_OFFSET = (0.0, 0.0, -0.05115)
FALLING_BATON_NEUTRAL_MARKER_OFFSET = (0.0, 0.0, -0.005775)
FALLING_BATON_NEGATIVE_MARKER_OFFSET = (0.0, 0.0, 0.045375)


def _falling_baton_physics_object_cfg(
    *, size: tuple[float, float, float], mass: float, pos: tuple[float, float, float]
) -> RigidObjectCfg:
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.52, 0.54), roughness=0.65),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _falling_baton_marker_cfg(
    *, prim_name: str, size: tuple[float, float, float], color: tuple[float, float, float], pos: tuple[float, float, float]
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            activate_contact_sensors=False,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0e-5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=0.001,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.45),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _load_affordance_asset_stats() -> dict[str, dict]:
    if not TABLETOP_AFFORDANCE_ANALYSIS.exists():
        return {}
    with TABLETOP_AFFORDANCE_ANALYSIS.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {asset["asset_id"]: asset for asset in data.get("assets", [])}


_AFFORDANCE_ASSET_STATS = _load_affordance_asset_stats()


def _asset_stats(asset_id: str) -> dict:
    stats = _AFFORDANCE_ASSET_STATS.get(asset_id, {})
    annotation_path = stats.get("annotation_path")
    return {
        "asset_id": asset_id,
        "label": stats.get("label", asset_id.rsplit("/", 1)[-1]),
        "annotation_path": annotation_path or "",
        "positive_fraction": float(stats.get("positive_fraction", 0.0)),
        "negative_fraction": float(stats.get("negative_fraction", 0.0)),
        "ignore_fraction": float(stats.get("ignore_fraction", 0.0)),
    }


def _generated_tabletop_mesh_urdf(asset_id: str) -> Path:
    return TABLETOP_GENERATED_MESH_ROOT.joinpath(*asset_id.split("/")) / f"{asset_id.rsplit('/', 1)[-1]}.urdf"


def _affordance_baton_object_cfg(*, pos: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(FALLING_BATON_AFFORDANCE_URDF),
            fix_base=False,
            joint_drive=None,
            collision_from_visuals=False,
            collider_type="convex_hull",
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=FALLING_BATON_AFFORDANCE_MASS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=4.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _tabletop_asset_spec(
    *,
    asset_id: str,
    category: str,
    proxy_shape: str,
    size: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    radius: float | None = None,
    height: float | None = None,
    axis: str = "Z",
    affordance_mode: str = "side_grasp",
    mesh_asset_path: str | Path | None = None,
    static_friction: float = 0.75,
    dynamic_friction: float = 0.75,
    restitution: float = 0.0,
    friction_combine_mode: str = "multiply",
    restitution_combine_mode: str = "multiply",
    contact_offset: float = 0.002,
    rest_offset: float = 0.0,
) -> dict:
    stats = _asset_stats(asset_id)
    sx, sy, sz = size
    if radius is None:
        radius = 0.5 * max(sx, sy)
    if height is None:
        height = sz
    spec = {
        **stats,
        "category": category,
        "proxy_shape": proxy_shape,
        "size": tuple(float(v) for v in size),
        "radius": float(radius),
        "height": float(height),
        "axis": axis,
        "mass": float(mass),
        "color": tuple(float(v) for v in color),
        "affordance_mode": affordance_mode,
        "static_friction": float(static_friction),
        "dynamic_friction": float(dynamic_friction),
        "restitution": float(restitution),
        "friction_combine_mode": str(friction_combine_mode),
        "restitution_combine_mode": str(restitution_combine_mode),
        "contact_offset": float(contact_offset),
        "rest_offset": float(rest_offset),
    }
    if mesh_asset_path is not None:
        spec["mesh_asset_path"] = str(mesh_asset_path)
    return spec


def _tabletop_object_material(spec: dict) -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode=str(spec.get("friction_combine_mode", "multiply")),
        restitution_combine_mode=str(spec.get("restitution_combine_mode", "multiply")),
        static_friction=float(spec.get("static_friction", 0.75)),
        dynamic_friction=float(spec.get("dynamic_friction", 0.75)),
        restitution=float(spec.get("restitution", 0.0)),
    )


def _with_tabletop_friction(
    specs: tuple[dict, ...],
    *,
    static_friction: float,
    dynamic_friction: float,
    friction_combine_mode: str | None = None,
) -> tuple[dict, ...]:
    updated_specs = []
    for spec in specs:
        updated = {
            **spec,
            "static_friction": float(static_friction),
            "dynamic_friction": float(dynamic_friction),
        }
        if friction_combine_mode is not None:
            updated["friction_combine_mode"] = str(friction_combine_mode)
        updated_specs.append(updated)
    return tuple(updated_specs)


TABLETOP_ROLLING_OBJECT_SPECS = (
    _tabletop_asset_spec(
        asset_id="domino20/apple",
        category="small_ball",
        proxy_shape="sphere",
        size=TABLETOP_SMALL_BALL_SIZE,
        radius=TABLETOP_SMALL_BALL_RADIUS,
        height=TABLETOP_SMALL_BALL_SIZE[2],
        mass=0.018,
        color=(0.92, 0.12, 0.08),
        affordance_mode="omni_grasp",
    ),
    _tabletop_asset_spec(
        asset_id="domino20/can",
        category="cylinder",
        proxy_shape="cylinder",
        size=(0.056, 0.056, 0.082),
        radius=0.028,
        height=0.082,
        mass=0.040,
        color=(0.92, 0.64, 0.10),
    ),
    _tabletop_asset_spec(
        asset_id="domino20/bottle",
        category="bottle",
        proxy_shape="cylinder",
        size=(0.048, 0.048, 0.120),
        radius=0.024,
        height=0.120,
        mass=0.045,
        color=(0.10, 0.48, 0.88),
    ),
    _tabletop_asset_spec(
        asset_id="primitive/tabletop_cone",
        category="cone",
        proxy_shape="cone",
        size=(0.060, 0.060, 0.080),
        radius=0.030,
        height=0.080,
        mass=0.030,
        color=(0.20, 0.72, 0.38),
    ),
    _tabletop_asset_spec(
        asset_id="domino20/pill_bottle",
        category="small_bottle",
        proxy_shape="cylinder",
        size=(0.040, 0.040, 0.088),
        radius=0.020,
        height=0.088,
        mass=0.026,
        color=(0.80, 0.20, 0.62),
    ),
)

TABLETOP_ROLLING_START_SPEC = TABLETOP_ROLLING_OBJECT_SPECS[0]
TABLETOP_ROLLING_START_Z = 0.296 + 0.5 * float(TABLETOP_ROLLING_START_SPEC["height"]) + 0.002

# Shared task contract used for cross-hand rolling comparisons.  Keep these
# values embodiment-agnostic; hand geometry, joint coupling, and clearance
# margins remain in the Revo2/Inspire parent configs.
UNIFIED_ROLLING_BENCHMARK_NAME = "rolling_multishape_v1"
UNIFIED_ROLLING_OBJECT_SPECS = TABLETOP_ROLLING_OBJECT_SPECS
UNIFIED_ROLLING_START_POS = (0.58, 0.0, TABLETOP_ROLLING_START_Z)
UNIFIED_ROLLING_RESET_OBJECT_POS_NOISE = (0.040, 0.030, 0.0015)
UNIFIED_ROLLING_START_SPEED_RANGE = (0.0, 0.0)
UNIFIED_ROLLING_TARGET_SPEED_RANGE = (0.10, 0.40)
UNIFIED_ROLLING_START_YAW_RATE_RANGE = (0.0, 0.0)
UNIFIED_ROLLING_TARGET_YAW_RATE_RANGE = (-2.40, 2.40)
UNIFIED_ROLLING_ASSET_CURRICULUM_STEPS = 2_400_000
UNIFIED_ROLLING_SPEED_CURRICULUM_STEPS = 3_200_000
UNIFIED_ROLLING_CURRICULUM_METRIC = "catch_hold"
UNIFIED_ROLLING_CURRICULUM_START_SUCCESS = 0.08
UNIFIED_ROLLING_CURRICULUM_FULL_SUCCESS = 0.45
UNIFIED_ROLLING_CURRICULUM_EMA_ALPHA = 0.035
UNIFIED_ROLLING_CURRICULUM_ALPHA_RISE = 0.00035
UNIFIED_ROLLING_SUCCESS_LIFT_HEIGHT = 0.040
UNIFIED_ROLLING_SUCCESS_HOLD_STEPS = 8
UNIFIED_ROLLING_STABLE_OBJECT_PALM_VEL = 0.42
UNIFIED_ROLLING_EPISODE_LENGTH_S = 8.0
UNIFIED_ROLLING_HOVER_HEIGHT_DELTA = 0.075
UNIFIED_ROLLING_HOVER_LATCH_LIFT_PROGRESS = 0.44
UNIFIED_ROLLING_HOVER_XY_DISTANCE_SCALE = 0.16
UNIFIED_ROLLING_HOVER_Z_DISTANCE_SCALE = 0.05
UNIFIED_ROLLING_HOVER_OBJECT_SPEED_SCALE = 0.20
UNIFIED_ROLLING_HOVER_ANG_SPEED_SCALE = 7.0
UNIFIED_ROLLING_HOVER_SUCCESS_XY_TOLERANCE = 0.20
UNIFIED_ROLLING_HOVER_SUCCESS_Z_TOLERANCE = 0.055
UNIFIED_ROLLING_HOVER_SUCCESS_OBJECT_SPEED = 0.30
UNIFIED_ROLLING_CONTACT_DISTANCE = 0.014
UNIFIED_ROLLING_CONTACT_SCORE_SCALE = 0.016
UNIFIED_ROLLING_STRICT_CONTACT_DISTANCE = 0.010
UNIFIED_ROLLING_STRICT_MIN_FINGER_CONTACTS = 3
UNIFIED_ROLLING_STRICT_MIN_NON_THUMB_CONTACTS = 2
UNIFIED_ROLLING_LIFT_ARM_DELTA = V325_VERIFIED_LIFT_ARM_DELTA
UNIFIED_ROLLING_LIFT_ACTION_PRIOR = V325_VERIFIED_LIFT_ACTION_PRIOR_120


class _UnifiedRollingRewardContract:
    """Embodiment-independent reward contract for rolling comparisons."""

    # Shared table-clearance semantics. Palm/fingertip locations come from each
    # embodiment adapter, while both hands use the same samples, normalized
    # penalty, and 3 mm geometric tolerance. This explicit override prevents
    # legacy Inspire "caution" experiments from making contact infeasible.
    tabletop_arm_clearance_xy_padding = 0.05
    tabletop_arm_clearance_scale = 0.060
    tabletop_arm_clearance_max_penalty = 2.0
    tabletop_arm_clearance_ok_penalty_threshold = 0.05
    tabletop_arm_clearance_include_fingertip_points = True
    tabletop_arm_clearance_fingertip_point_margin = 0.006
    tabletop_arm_clearance_include_palm_point = True
    tabletop_arm_clearance_palm_point_margin = 0.012

    action_penalty_scale = 0.003
    arm_lift_progress_rew_scale = 0.0
    arm_target_delta_penalty_scale = 0.004
    catch_progress_rew_scale = 160.0
    contact_rew_scale = 300.0
    drop_penalty = 25.0
    dynamic_tabletop_low_palm_penalty_scale = 25.0
    dynamic_tabletop_pregrasp_height_rew_scale = 20.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 80.0
    dynamic_tabletop_side_contact_penalty_scale = 0.0
    falling_affordance_negative_penalty_scale = 0.0
    falling_affordance_positive_rew_scale = 0.0
    falling_opposed_stable_pinch_rew_scale = 0.0
    falling_palm_gate_rew_scale = 0.0
    falling_pinched_rel_vel_penalty_scale = 0.0
    falling_positive_stable_rew_scale = 0.0
    falling_soft_success_progress_rew_scale = 0.0
    falling_stable_grasp_rew_scale = 0.0
    fingertip_reach_rew_scale = 16.0
    grasp_quality_rew_scale = 1500.0
    hold_progress_rew_scale = 10800.0
    lift_action_prior_rew_scale = 0.0
    lift_coupling_rew_scale = 0.0
    lift_progress_linear_rew_scale = 0.0
    lift_progress_rew_scale = 1100.0
    lifted_true_grasp_rew_scale = 3800.0
    opposition_rew_scale = 1000.0
    palm_distance_penalty_scale = 0.0
    palm_lift_gap_penalty_scale = 90.0
    palm_lift_rew_scale = 8.0
    palm_only_lift_penalty_scale = 340.0
    palm_reach_rew_scale = 12.0
    pregrasp_rew_scale = 1.0
    premature_arm_lift_penalty_scale = 0.0
    premature_palm_lift_penalty_scale = 0.0
    quality_lift_progress_rew_scale = 1500.0
    scoop_lift_penalty_scale = 280.0
    stable_hold_rew_scale = 7800.0
    strict_approach_rew_scale = 20.0
    strict_multifinger_approach_rew_scale = 40.0
    strict_opposition_approach_rew_scale = 5000.0
    strict_opposition_touch_rew_scale = 12000.0
    strict_touch_rew_scale = 3000.0
    success_bonus = 32000.0
    tabletop_affordance_lift_rew_scale = 0.0
    tabletop_affordance_negative_penalty_scale = 0.0
    tabletop_affordance_positive_rew_scale = 0.0
    tabletop_arm_clearance_penalty_scale = 0.0
    tabletop_arm_object_lift_gap_penalty_scale = 0.0
    tabletop_grasped_arm_lift_rew_scale = 0.0
    tabletop_grasped_palm_lift_rew_scale = 700.0
    tabletop_hover_goal_rew_scale = 360.0
    tabletop_hover_grasp_loss_penalty_scale = 260.0
    tabletop_hover_height_progress_rew_scale = 220.0
    tabletop_hover_linear_penalty_scale = 80.0
    tabletop_hover_overshoot_penalty_scale = 120.0
    tabletop_hover_post_latch_action_penalty_scale = 0.0
    tabletop_hover_post_latch_speed_penalty_scale = 180.0
    tabletop_hover_post_latch_target_delta_penalty_scale = 0.0
    tabletop_hover_stable_rew_scale = 420.0
    tabletop_hover_target_drift_penalty_scale = 120.0
    tabletop_hover_target_rew_scale = 180.0
    tabletop_hover_under_height_penalty_scale = 160.0
    tabletop_hover_vel_penalty_scale = 80.0
    tabletop_hover_z_vel_penalty_scale = 120.0
    tabletop_lift_action_prior_rew_scale = 0.0
    tabletop_lift_without_current_grasp_penalty_scale = 0.0
    tabletop_lift_without_object_penalty_scale = 0.0
    tabletop_no_lift_after_grasp_penalty_scale = 180.0
    tabletop_non_thumb_without_thumb_penalty_scale = 800.0
    tabletop_object_carry_lift_rew_scale = 0.0
    tabletop_object_carry_stall_penalty_scale = 0.0
    tabletop_object_up_vel_rew_scale = 0.0
    tabletop_post_success_action_penalty_scale = 0.05
    tabletop_post_success_arm_joint_vel_penalty_scale = 100.0
    tabletop_post_success_arm_target_drift_penalty_scale = 1800.0
    tabletop_post_success_grasp_loss_penalty_scale = 15000.0
    tabletop_post_success_hold_rew_scale = 20000.0
    tabletop_post_success_palm_drift_penalty_scale = 2800.0
    tabletop_post_success_speed_penalty_scale = 1400.0
    tabletop_post_success_target_delta_penalty_scale = 0.08
    tabletop_post_success_under_height_penalty_scale = 2200.0
    tabletop_post_success_unstable_penalty_scale = 16000.0
    tabletop_stable_catch_rew_scale = 1650.0
    tabletop_underwrap_rew_scale = 0.0
    true_grasp_rew_scale = 3000.0

    # Reward-component semantics. These values match the working Revo2
    # multi-shape teacher and are shared so the two hands optimize the same
    # mathematical objective even when their low-level controllers differ.
    contact_reward_requires_thumb_pair = True
    contact_reward_uses_opposition_product = True
    contact_reward_opposition_min_multiplier = 0.05
    opposition_reward_uses_weighted_score = True
    true_grasp_score_requires_thumb_pair = True
    true_grasp_score_uses_opposition_product = True
    true_grasp_score_opposition_min_multiplier = 0.05
    thumb_contact_reward_weight = 0.55
    thumb_true_grasp_score_weight = 0.58
    grasp_quality_finger_count_weight = 0.30
    grasp_quality_non_thumb_weight = 0.25
    grasp_quality_thumb_weight = 0.25
    grasp_quality_opposition_weight = 0.20

    reach_distance_scale = 0.22
    fingertip_distance_scale = 0.055
    palm_contact_distance = 0.10
    palm_only_lift_dist = 0.12
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = False
    dynamic_tabletop_contact_pregrasp_gate_min = 0.45
    dynamic_tabletop_low_palm_height_scale = 0.040
    dynamic_tabletop_low_palm_max_penalty = 3.0
    dynamic_tabletop_min_palm_height_offset = 0.012
    dynamic_tabletop_pregrasp_height_offset = 0.105
    dynamic_tabletop_pregrasp_height_scale = 0.065
    dynamic_tabletop_pregrasp_xy_distance_scale = 0.16
    dynamic_tabletop_speed_alpha_sample_full_fraction = 0.0

    lift_reward_min_grasp_quality_multiplier = 0.10
    lift_reward_min_opposition_multiplier = 0.15
    lift_reward_uses_opposition_gate = False
    quality_lift_progress_min_opposition_multiplier = 0.15
    quality_lift_progress_uses_opposition_gate = False
    tabletop_arm_lift_reward_object_margin = 0.14
    tabletop_arm_object_lift_gap_margin = 0.12
    tabletop_grasped_palm_lift_height = 0.08
    tabletop_grasped_palm_lift_scale = 0.05
    tabletop_lift_action_prior_gate_min = 0.24
    tabletop_lift_gate_requires_current_strict_grasp = False
    tabletop_lift_use_grasp_seen_gate = False
    tabletop_lift_without_current_grasp_min_progress = 0.0
    tabletop_lift_without_current_grasp_ramp = 1.0
    tabletop_lift_without_object_min_arm_progress = 0.16
    tabletop_no_lift_after_grasp_grace_steps = 20
    tabletop_no_lift_after_grasp_max_penalty = 3.0
    tabletop_no_lift_after_grasp_ramp_steps = 80
    tabletop_no_lift_min_progress = 0.15
    tabletop_no_lift_soft_grasp_gate = 0.0
    tabletop_no_lift_uses_soft_grasp_gate = False
    tabletop_non_thumb_without_thumb_gate_start = 0.06
    tabletop_non_thumb_without_thumb_gate_ramp = 0.25
    tabletop_non_thumb_without_thumb_thumb_target = 0.30
    tabletop_non_thumb_without_thumb_penalty_lift_gate_min = 0.0

    tabletop_gate_boolean_grasp_rewards_by_clearance = False
    tabletop_gate_contact_rewards_by_clearance = False
    tabletop_contact_clearance_gate_min = 1.0
    tabletop_contact_clearance_gate_scale = 0.50
    tabletop_hover_latch_uses_grasp_seen = True
    tabletop_hover_reward_uses_grasp_seen = True
    tabletop_success_uses_grasp_seen = True
    tabletop_object_carry_grasp_seen_gate = 0.25
    tabletop_object_carry_min_grasp_streak = 0
    tabletop_object_carry_stall_min_arm_progress = 0.12
    tabletop_object_carry_stall_min_z_vel = 0.015
    tabletop_object_carry_streak_ramp_steps = 1
    tabletop_object_up_vel_scale = 0.10
    tabletop_stable_catch_min_lift_multiplier = 0.05

    tabletop_post_success_arm_target_drift_scale = 0.28
    tabletop_post_success_arm_target_drift_tolerance = 0.09
    tabletop_post_success_palm_drift_scale = 0.085
    tabletop_post_success_palm_drift_tolerance = 0.035
    tabletop_underwrap_below_center_fraction = 0.20
    tabletop_underwrap_contact_margin = 0.0
    tabletop_underwrap_contact_scale = 0.018
    tabletop_underwrap_height_scale = 0.012
    tabletop_underwrap_opposition_min_multiplier = 0.10
    tabletop_underwrap_pair_weight = 1.0
    tabletop_underwrap_progress_weight = 0.0
    tabletop_underwrap_radial_fraction = 0.95
    tabletop_underwrap_radial_scale = 0.020
    tabletop_underwrap_uses_pregrasp_gate = True

    strict_approach_score_scale = 0.08
    strict_reward_contact_score_scale = 0.025
    strict_touch_reward_opposition_min_multiplier = 0.05
    strict_touch_reward_requires_thumb_pair = True
    strict_touch_reward_uses_opposition_product = True
    strict_touch_score_scale = 0.012

    default_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    policy_action_interface = "joint_target"
    arm_action_scale = 0.50
    arm_moving_average = 0.96
    initial_arm_target_lock_steps = 12
    initial_hand_target_lock_steps = 12
    tabletop_arm_lift_progress_baseline_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    tabletop_arm_lift_progress_baseline_mode = "first_strict_grasp"
    tabletop_arm_lift_progress_baseline_grasp_streak = 3
    lift_arm_delta = UNIFIED_ROLLING_LIFT_ARM_DELTA
    lift_action_prior = UNIFIED_ROLLING_LIFT_ACTION_PRIOR
    scripted_tabletop_lift_target_arm_delta = UNIFIED_ROLLING_LIFT_ARM_DELTA
    scripted_tabletop_relative_lift_target_arm_delta = UNIFIED_ROLLING_LIFT_ARM_DELTA
    scripted_action_prior_lift_action = UNIFIED_ROLLING_LIFT_ACTION_PRIOR

    # The official comparison is direct RL, not a different scripted residual
    # controller per hand. Geometry-specific target scaling remains in the
    # embodiment adapter, but no scripted reach, close, or lift action is added.
    scripted_action_prior_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_relative_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False


class _UnifiedRollingLiftHoldStage3Contract:
    """Shared continuation objective that turns an acquired grasp into a stable lift."""

    # Keep enough acquisition shaping to recover after a miss, while making a
    # stationary grasp less valuable than a strict, object-coupled lift.
    contact_rew_scale = 120.0
    grasp_quality_rew_scale = 700.0
    opposition_rew_scale = 400.0
    true_grasp_rew_scale = 1500.0
    strict_approach_rew_scale = 8.0
    strict_multifinger_approach_rew_scale = 16.0
    strict_opposition_approach_rew_scale = 120.0
    strict_touch_rew_scale = 1000.0
    strict_opposition_touch_rew_scale = 3000.0

    lift_progress_rew_scale = 5000.0
    quality_lift_progress_rew_scale = 7000.0
    lifted_true_grasp_rew_scale = 14000.0
    tabletop_stable_catch_rew_scale = 5000.0
    tabletop_grasped_palm_lift_rew_scale = 4500.0
    tabletop_grasped_arm_lift_rew_scale = 9000.0
    # The direct policy must discover a coordinated Franka lift after closing.
    # Reward an absolute joint target above the latched grasp pose only while
    # the current strict grasp is intact; no scripted action is injected.
    tabletop_lift_action_prior_rew_scale = 9000.0
    tabletop_lift_action_prior_gate_min = 0.0
    tabletop_object_up_vel_rew_scale = 3000.0
    tabletop_object_carry_lift_rew_scale = 16000.0
    stable_hold_rew_scale = 16000.0
    hold_progress_rew_scale = 22000.0
    success_bonus = 48000.0

    tabletop_no_lift_after_grasp_penalty_scale = 12000.0
    tabletop_no_lift_after_grasp_grace_steps = 8
    tabletop_no_lift_after_grasp_ramp_steps = 24
    tabletop_lift_without_object_penalty_scale = 6000.0
    tabletop_lift_without_current_grasp_penalty_scale = 10000.0
    tabletop_arm_object_lift_gap_penalty_scale = 7000.0
    tabletop_object_carry_stall_penalty_scale = 5000.0
    tabletop_strict_grasp_loss_penalty_scale = 8000.0
    tabletop_arm_clearance_penalty_scale = 6000.0
    # The filtered fingertip-force signal does not include every load-bearing
    # distal/proximal finger-link contact. Keep it as a diagnostic, but do not
    # let an incomplete sensor projection override a physically lifted object,
    # strict fingertip enclosure, and stable object-palm motion.
    tabletop_force_grasp_rew_scale = 0.0
    tabletop_force_grasp_streak_rew_scale = 0.0
    tabletop_force_stable_grasp_rew_scale = 0.0
    tabletop_force_grasp_loss_penalty_scale = 0.0

    object_contact_force_diagnostics_enabled = True
    object_contact_force_threshold = 0.05
    tabletop_arm_lift_progress_baseline_mode = "first_strict_grasp"
    # A one-frame strict grasp already requires thumb-plus-two-finger
    # opposition. Latch immediately so contact jitter cannot starve the direct
    # policy of every coordinated lift-target reward sample.
    tabletop_arm_lift_progress_baseline_grasp_streak = 1
    tabletop_force_grasp_streak_target = 8
    tabletop_lift_rewards_require_force_grasp = False
    lift_reward_uses_grasp_quality_gate = True
    lift_reward_uses_opposition_gate = True
    lift_reward_min_opposition_multiplier = 0.0
    quality_lift_progress_uses_opposition_gate = True
    quality_lift_progress_min_opposition_multiplier = 0.0
    tabletop_lift_gate_requires_current_strict_grasp = True
    tabletop_lift_gate_requires_force_grasp = False
    tabletop_lift_rewards_require_current_strict_grasp = True
    tabletop_lift_use_grasp_seen_gate = False
    tabletop_object_carry_uses_grasp_seen = False
    # Stage 3 must keep a current strict grasp. Memory-only hover gates make
    # grasp loss invisible after a one-frame contact event.
    tabletop_hover_latch_uses_grasp_seen = False
    tabletop_hover_reward_uses_grasp_seen = False
    tabletop_success_uses_grasp_seen = False
    tabletop_object_carry_min_grasp_streak = 3
    tabletop_object_carry_streak_ramp_steps = 4
    tabletop_object_carry_uses_lift_baseline_grasp_streak = True
    tabletop_no_lift_uses_force_grasp_gate = False
    tabletop_success_requires_force_grasp = False
    tabletop_lift_without_object_min_arm_progress = 0.08
    tabletop_arm_lift_reward_object_margin = 0.08
    tabletop_arm_object_lift_gap_margin = 0.08


class _UnifiedRollingGraspHoldStage2Contract:
    """Shared continuation objective for a sustained load-bearing grasp."""

    # Stage 2 already reaches strict opposition in nearly every static episode,
    # but often for only one frame. Reward a quiet strict grasp every step and
    # make losing it expensive before asking the policy to coordinate a lift.
    tabletop_strict_hold_rew_scale = 28000.0
    tabletop_strict_grasp_loss_penalty_scale = 12000.0
    tabletop_strict_grasp_hold_steps = 20

    # Keep this curriculum stage focused on stationary load-bearing contact.
    # Lift and hover shaping are restored by the shared Stage 3 contract.
    palm_lift_rew_scale = 0.0
    lift_progress_rew_scale = 0.0
    quality_lift_progress_rew_scale = 0.0
    lifted_true_grasp_rew_scale = 0.0
    tabletop_grasped_palm_lift_rew_scale = 0.0
    tabletop_stable_catch_rew_scale = 0.0
    stable_hold_rew_scale = 0.0
    hold_progress_rew_scale = 0.0
    tabletop_hover_goal_rew_scale = 0.0
    tabletop_hover_height_progress_rew_scale = 0.0
    tabletop_hover_stable_rew_scale = 0.0
    tabletop_hover_target_rew_scale = 0.0
    success_bonus = 0.0


UNIFIED_FALLING_BENCHMARK_NAME = "falling_baton_affordance_v1"
UNIFIED_FALLING_OBJECT_SIZE = (0.018, 0.018, 0.165)
UNIFIED_FALLING_OBJECT_MASS = 0.014
UNIFIED_FALLING_START_POS = (0.0, 0.23, 1.12)
UNIFIED_FALLING_SPAWN_X_RANGE = (-0.04, 0.40)
UNIFIED_FALLING_SPAWN_Y_RANGE = (0.08, 0.36)
UNIFIED_FALLING_SPAWN_Z_RANGE = (0.98, 1.34)
UNIFIED_FALLING_START_ABOVE_PALM_RANGE = (0.28, 0.42)
UNIFIED_FALLING_TARGET_ABOVE_PALM_RANGE = (0.34, 0.66)
UNIFIED_FALLING_START_ROLL_RANGE = (-0.45, 0.45)
UNIFIED_FALLING_START_PITCH_RANGE = (-0.35, 0.35)
UNIFIED_FALLING_START_YAW_RANGE = (-0.90, 0.90)
UNIFIED_FALLING_TARGET_ROLL_RANGE = (-3.141592653589793, 3.141592653589793)
UNIFIED_FALLING_TARGET_PITCH_RANGE = (-1.35, 1.35)
UNIFIED_FALLING_TARGET_YAW_RANGE = (-3.141592653589793, 3.141592653589793)
UNIFIED_FALLING_START_XY_SPEED_RANGE = (0.0, 0.005)
UNIFIED_FALLING_START_Z_SPEED_RANGE = (0.0, 0.015)
UNIFIED_FALLING_START_ANG_VEL_RANGE = (-0.08, 0.08)
UNIFIED_FALLING_TARGET_LIN_VEL_MIN = (0.0, 0.0, -0.26)
UNIFIED_FALLING_TARGET_LIN_VEL_MAX = (0.06, 0.06, -0.04)
UNIFIED_FALLING_TARGET_ANG_VEL_MIN = (-1.0, -1.0, -1.0)
UNIFIED_FALLING_TARGET_ANG_VEL_MAX = (1.0, 1.0, 1.0)
UNIFIED_FALLING_CURRICULUM_METRIC = "success"
UNIFIED_FALLING_CURRICULUM_START_SUCCESS = 0.02
UNIFIED_FALLING_CURRICULUM_FULL_SUCCESS = 0.30
UNIFIED_FALLING_CURRICULUM_EMA_ALPHA = 0.035
UNIFIED_FALLING_CURRICULUM_ALPHA_RISE = 0.00035
UNIFIED_FALLING_SUCCESS_HOLD_STEPS = 20
UNIFIED_FALLING_STABLE_OBJECT_PALM_VEL = 0.30
UNIFIED_FALLING_EPISODE_LENGTH_S = 200.0 / 60.0


def _tabletop_start_z_from_spec(spec: dict, table_top_z: float = 0.296) -> float:
    shape = str(spec.get("proxy_shape", "box")).lower()
    if shape == "sphere":
        support_height = float(spec.get("radius", 0.03))
    elif shape in {"cylinder", "cone"}:
        support_height = 0.5 * float(spec.get("height", spec.get("size", (0.04, 0.04, 0.08))[2]))
    else:
        support_height = 0.5 * float(spec.get("size", (0.04, 0.04, 0.08))[2])
    return float(table_top_z) + support_height + 0.002


def _tabletop_sphere_spec(
    *,
    asset_id: str,
    radius: float,
    mass: float,
    color: tuple[float, float, float],
    static_friction: float = 0.75,
    dynamic_friction: float = 0.75,
    restitution: float = 0.0,
    contact_offset: float = 0.002,
    rest_offset: float = 0.0,
) -> dict:
    diameter = 2.0 * float(radius)
    return _tabletop_asset_spec(
        asset_id=asset_id,
        category="small_ball",
        proxy_shape="sphere",
        size=(diameter, diameter, diameter),
        radius=float(radius),
        height=diameter,
        mass=float(mass),
        color=color,
        affordance_mode="omni_grasp",
        static_friction=float(static_friction),
        dynamic_friction=float(dynamic_friction),
        restitution=float(restitution),
        contact_offset=float(contact_offset),
        rest_offset=float(rest_offset),
    )


TABLETOP_INSPIRE_SPHERE_50MM_SPEC = _tabletop_sphere_spec(
    asset_id="primitive/inspire_sphere_50mm",
    radius=0.025,
    mass=0.026,
    color=(0.10, 0.55, 0.92),
)
TABLETOP_INSPIRE_SPHERE_60MM_SPEC = _tabletop_sphere_spec(
    asset_id="primitive/inspire_sphere_60mm",
    radius=0.030,
    mass=0.046,
    color=(0.20, 0.72, 0.38),
)
TABLETOP_INSPIRE_SPHERE_60MM_HIGH_FRICTION_SPEC = _tabletop_sphere_spec(
    asset_id="primitive/inspire_sphere_60mm_high_friction",
    radius=0.030,
    mass=0.046,
    color=(0.10, 0.62, 0.28),
    static_friction=1.05,
    dynamic_friction=0.90,
)
TABLETOP_INSPIRE_SPHERE_50MM_HIGH_FRICTION_SPEC = _tabletop_sphere_spec(
    asset_id="primitive/inspire_sphere_50mm_high_friction",
    radius=0.025,
    mass=0.026,
    color=(0.88, 0.20, 0.16),
    static_friction=1.05,
    dynamic_friction=0.90,
)
TABLETOP_INSPIRE_SPHERE_50MM_SOFT_CONTACT_SPEC = _tabletop_sphere_spec(
    asset_id="primitive/inspire_sphere_50mm_soft_contact",
    radius=0.025,
    mass=0.026,
    color=(0.72, 0.34, 0.84),
    static_friction=0.85,
    dynamic_friction=0.75,
    contact_offset=0.004,
)
TABLETOP_INSPIRE_CAN_HIGH_FRICTION_SPEC = _with_tabletop_friction(
    (TABLETOP_ROLLING_OBJECT_SPECS[1],),
    static_friction=1.05,
    dynamic_friction=0.90,
    friction_combine_mode="multiply",
)[0]

TABLETOP_TRANSPORT_OBJECT_SPECS = (
    _tabletop_asset_spec(
        asset_id="domino20/cup",
        category="cup",
        proxy_shape="cylinder",
        size=(0.064, 0.064, 0.078),
        radius=0.032,
        height=0.078,
        mass=0.040,
        color=(0.18, 0.54, 0.78),
        mesh_asset_path=_generated_tabletop_mesh_urdf("domino20/cup"),
    ),
    _tabletop_asset_spec(
        asset_id="domino20/mug",
        category="cup",
        proxy_shape="cylinder",
        size=(0.070, 0.070, 0.080),
        radius=0.035,
        height=0.080,
        mass=0.050,
        color=(0.74, 0.74, 0.70),
        mesh_asset_path=_generated_tabletop_mesh_urdf("domino20/mug"),
    ),
    _tabletop_asset_spec(
        asset_id="domino20/kettle",
        category="teapot",
        proxy_shape="box",
        size=(0.095, 0.075, 0.082),
        mass=0.070,
        color=(0.36, 0.42, 0.46),
        mesh_asset_path=_generated_tabletop_mesh_urdf("domino20/kettle"),
    ),
    _tabletop_asset_spec(
        asset_id="dextoolbench/hammer/claw_hammer",
        category="hammer",
        proxy_shape="box",
        size=(0.195, 0.086, 0.032),
        mass=0.100,
        color=(0.76, 0.36, 0.16),
        affordance_mode="handle_tool_negative",
        mesh_asset_path=_generated_tabletop_mesh_urdf("dextoolbench/hammer/claw_hammer"),
    ),
    _tabletop_asset_spec(
        asset_id="dextoolbench/hammer/mallet_hammer",
        category="hammer",
        proxy_shape="box",
        size=(0.170, 0.070, 0.045),
        mass=0.090,
        color=(0.70, 0.40, 0.20),
        affordance_mode="handle_tool_negative",
        mesh_asset_path=_generated_tabletop_mesh_urdf("dextoolbench/hammer/mallet_hammer"),
    ),
    _tabletop_asset_spec(
        asset_id="dextoolbench/brush/blue_brush",
        category="brush",
        proxy_shape="box",
        size=(0.210, 0.070, 0.040),
        mass=0.070,
        color=(0.12, 0.36, 0.82),
        affordance_mode="handle_tool_negative",
        mesh_asset_path=_generated_tabletop_mesh_urdf("dextoolbench/brush/blue_brush"),
    ),
    _tabletop_asset_spec(
        asset_id="dextoolbench/brush/red_brush",
        category="brush",
        proxy_shape="box",
        size=(0.210, 0.065, 0.045),
        mass=0.070,
        color=(0.82, 0.20, 0.18),
        affordance_mode="handle_tool_negative",
        mesh_asset_path=_generated_tabletop_mesh_urdf("dextoolbench/brush/red_brush"),
    ),
)


def _object_cfg_from_tabletop_spec(spec: dict, pos: tuple[float, float, float], prim_suffix: str = "") -> RigidObjectCfg:
    proxy_shape = str(spec["proxy_shape"]).lower()
    prim_path = f"/World/envs/env_.*/Object{prim_suffix}"
    base_common = dict(
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(mass=float(spec["mass"])),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=4.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=float(spec.get("contact_offset", 0.002)),
            rest_offset=float(spec.get("rest_offset", 0.0)),
        ),
    )
    mesh_asset_path = str(spec.get("mesh_asset_path", "") or "")
    if mesh_asset_path:
        spawn = sim_utils.UrdfFileCfg(
            asset_path=mesh_asset_path,
            fix_base=False,
            joint_drive=None,
            collision_from_visuals=False,
            collider_type=str(spec.get("mesh_collider_type", "convex_decomposition")),
            **base_common,
        )
    else:
        common = dict(
            **base_common,
            physics_material=_tabletop_object_material(spec),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(spec["color"]),
                roughness=0.58,
            ),
        )
        if proxy_shape == "sphere":
            spawn = sim_utils.SphereCfg(radius=float(spec["radius"]), **common)
        elif proxy_shape == "cylinder":
            spawn = sim_utils.CylinderCfg(
                radius=float(spec["radius"]),
                height=float(spec["height"]),
                axis=str(spec.get("axis", "Z")),
                **common,
            )
        elif proxy_shape == "cone":
            spawn = sim_utils.ConeCfg(
                radius=float(spec["radius"]),
                height=float(spec["height"]),
                axis=str(spec.get("axis", "Z")),
                **common,
            )
        else:
            spawn = sim_utils.CuboidCfg(size=tuple(spec["size"]), **common)
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _v699_revo2_robot_cfg(default_arm_pos: tuple[float, ...]) -> ArticulationCfg:
    joint_pos = {joint_name: value for joint_name, value in zip(FRANKA_ARM_JOINT_NAMES, default_arm_pos)}
    joint_pos.update(DEFAULT_HAND_OPEN_POS)
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(V699_REVO2_URDF),
            fix_base=True,
            root_link_name="panda_link0",
            merge_fixed_joints=False,
            make_instanceable=False,
            convert_mimic_joints_to_normal_joints=True,
            self_collision=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
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
                effort_limit_sim=115.0,
                velocity_limit_sim=8.5,
                stiffness=230.0,
                damping=20.0,
            ),
        },
    )


def _inspire_z180_robot_cfg(
    default_arm_pos: tuple[float, ...], *, default_hand_pos: dict[str, float] | None = None
) -> ArticulationCfg:
    joint_pos = {joint_name: value for joint_name, value in zip(FRANKA_ARM_JOINT_NAMES, default_arm_pos)}
    joint_pos.update(INSPIRE_DEFAULT_HAND_OPEN_POS if default_hand_pos is None else default_hand_pos)
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(INSPIRE_Z180_URDF),
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
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
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
            "inspire_thumb_yaw": ImplicitActuatorCfg(
                joint_names_expr=["thumb_proximal_yaw_joint"],
                effort_limit_sim=INSPIRE_RH56BFX_HAND_EFFORT,
                velocity_limit_sim=INSPIRE_RH56BFX_THUMB_YAW_VEL,
                stiffness=INSPIRE_RH56BFX_HAND_STIFFNESS,
                damping=INSPIRE_RH56BFX_HAND_DAMPING,
            ),
            "inspire_thumb_flex": ImplicitActuatorCfg(
                joint_names_expr=[
                    "thumb_proximal_pitch_joint",
                    "thumb_intermediate_joint",
                    "thumb_distal_joint",
                ],
                effort_limit_sim=INSPIRE_RH56BFX_HAND_EFFORT,
                velocity_limit_sim=INSPIRE_RH56BFX_THUMB_FLEX_VEL,
                stiffness=INSPIRE_RH56BFX_HAND_STIFFNESS,
                damping=INSPIRE_RH56BFX_HAND_DAMPING,
            ),
            "inspire_fingers": ImplicitActuatorCfg(
                joint_names_expr=[
                    "index_.*_joint",
                    "middle_.*_joint",
                    "ring_.*_joint",
                    "pinky_.*_joint",
                ],
                effort_limit_sim=INSPIRE_RH56BFX_HAND_EFFORT,
                velocity_limit_sim=INSPIRE_RH56BFX_FINGER_FLEX_VEL,
                stiffness=INSPIRE_RH56BFX_HAND_STIFFNESS,
                damping=INSPIRE_RH56BFX_HAND_DAMPING,
            ),
        },
    )


def _inspire_z180_legacy_robot_cfg(default_arm_pos: tuple[float, ...]) -> ArticulationCfg:
    """Earlier Inspire import used by the first matched-friction positive control."""

    joint_pos = {joint_name: value for joint_name, value in zip(FRANKA_ARM_JOINT_NAMES, default_arm_pos)}
    joint_pos.update(INSPIRE_OFFICIAL_ZERO_HAND_OPEN_POS)
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(INSPIRE_Z180_URDF),
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
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
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
            "inspire_hand": ImplicitActuatorCfg(
                joint_names_expr=list(INSPIRE_HAND_JOINT_NAMES),
                effort_limit_sim=1.0,
                velocity_limit_sim=0.8,
                stiffness=60.0,
                damping=6.0,
            ),
        },
    )


@configclass
class Revo2DynamicDexterousTeacherEnvCfg(Revo2StaticGraspEnvCfg):
    """Base config for dynamic privileged teacher tasks."""

    action_space = 13
    observation_space = 76
    episode_length_s = 5.0
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 120,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**20,
        ),
    )

    robot_cfg: ArticulationCfg = _v699_revo2_robot_cfg(V699_FRONT110_ARM_POS)
    table_cfg: RigidObjectCfg = _table_cfg(pos=(0.58, -0.05, 0.2735))
    object_cfg: RigidObjectCfg = _falling_baton_physics_object_cfg(
        size=FALLING_BATON_PHYSICS_SIZE,
        mass=FALLING_BATON_AFFORDANCE_MASS,
        pos=(0.58, -0.05, 1.18),
    )

    task_family = "falling_baton_grasp"
    hand_embodiment = "revo2"
    action_contract = "revo2_semantic_13d"
    policy_action_interface = "joint_target"
    reference_name = "revo2_v699_dynamic_teacher_base"

    # Real Revo2 exposes six active hand commands.  The env observes/actions
    # these active joints, while sim_hand_joint_names lets the dynamic action
    # adapter expand them to the URDF mimic joints for contact-rich simulation.
    hand_joint_names = REVO2_HAND_JOINT_NAMES
    sim_hand_joint_names = REVO2_V699_PHYSICAL_HAND_JOINT_NAMES
    fingertip_body_names = REVO2_TOUCH_BODY_NAMES
    touch_body_names = REVO2_TOUCH_BODY_NAMES
    palm_body_name = "revo2_right_base_link"

    object_shape = "box"
    object_radius = 0.011
    object_size = FALLING_BATON_PHYSICS_SIZE
    object_start_pos = (0.58, -0.05, 1.18)
    object_start_rot = (1.0, 0.0, 0.0, 0.0)
    affordance_label_mode = "handle_blade"
    affordance_positive_fraction = 0.38
    affordance_negative_fraction = 0.45
    affordance_positive_end = "negative"
    falling_baton_affordance_markers_enabled = True
    falling_baton_positive_marker_cfg = _falling_baton_marker_cfg(
        prim_name="ObjectAffordancePositive",
        size=FALLING_BATON_POSITIVE_MARKER_SIZE,
        color=(0.05, 0.78, 0.25),
        pos=(0.58, -0.05, 1.18),
    )
    falling_baton_neutral_marker_cfg = _falling_baton_marker_cfg(
        prim_name="ObjectAffordanceNeutral",
        size=FALLING_BATON_NEUTRAL_MARKER_SIZE,
        color=(0.62, 0.64, 0.66),
        pos=(0.58, -0.05, 1.18),
    )
    falling_baton_negative_marker_cfg = _falling_baton_marker_cfg(
        prim_name="ObjectAffordanceNegative",
        size=FALLING_BATON_NEGATIVE_MARKER_SIZE,
        color=(0.92, 0.07, 0.05),
        pos=(0.58, -0.05, 1.18),
    )
    falling_baton_positive_marker_local_offset = FALLING_BATON_POSITIVE_MARKER_OFFSET
    falling_baton_neutral_marker_local_offset = FALLING_BATON_NEUTRAL_MARKER_OFFSET
    falling_baton_negative_marker_local_offset = FALLING_BATON_NEGATIVE_MARKER_OFFSET
    table_top_z = 0.296
    create_table = True
    default_arm_pos = V699_FRONT110_ARM_POS

    # Joint target control. Revo2 policies use 6 active hand commands matching
    # the real hand; the env expands them to the 11 simulated hand joints.
    arm_moving_average = 0.96
    hand_moving_average = 0.74
    arm_action_scale = 0.50
    arm_target_clamp_delta = (2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30)
    initial_arm_target_lock_steps = 0
    initial_hand_target_lock_steps = 0
    initial_no_contact_steps = 0
    reset_arm_pos_noise = 0.0
    reset_object_pos_noise = (0.035, 0.045, 0.015)
    reference_hand_fractions = (1.0,) * 6
    inspire_semantic_close_targets: tuple[float, ...] | None = None
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_pregrasp_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    scripted_tabletop_pregrasp_prior_start_step = 0
    scripted_tabletop_pregrasp_prior_ramp_steps = 180
    scripted_tabletop_pregrasp_prior_steps = 0
    scripted_tabletop_pregrasp_prior_control_mode = "normalized_action"

    # Falling-baton aerial spawn, matching the IsaacGym reset structure.  The
    # object samples in a front workspace, then optionally snaps near a
    # palm/catch-center reference and is raised above that reference.
    falling_baton_palm_relative_spawn_enabled = False
    falling_baton_palm_relative_clamp_to_workspace = True
    falling_baton_spawn_x_range = (-0.24, 0.24)
    falling_baton_spawn_y_range = (-0.12, 0.24)
    falling_baton_spawn_z_range = (1.05, 1.42)
    falling_baton_spawn_above_palm_enabled = True
    falling_baton_spawn_above_palm_range = (0.40, 0.76)
    falling_baton_catch_center_finger_weight = 1.05
    falling_baton_catch_center_forward_offset = 0.13
    falling_baton_catch_center_world_offset = (0.0, 0.065, 0.0)
    falling_baton_palm_relative_start_x_range = (-0.060, 0.060)
    falling_baton_palm_relative_start_y_range = (0.030, 0.150)
    falling_baton_randomize_orientation = True
    falling_baton_roll_range = (-3.141592653589793, 3.141592653589793)
    falling_baton_pitch_range = (-1.35, 1.35)
    falling_baton_yaw_range = (-3.141592653589793, 3.141592653589793)

    # Dynamic reset.
    object_lin_vel_min = (0.0, 0.0, -0.45)
    object_lin_vel_max = (0.0, 0.0, -0.05)
    object_ang_vel_min = (-1.5, -1.5, -1.5)
    object_ang_vel_max = (1.5, 1.5, 1.5)
    falling_baton_start_initial_xy_speed_range = (0.0, 0.0)
    falling_baton_start_initial_z_speed_range = (0.0, 0.0)
    falling_baton_start_initial_ang_vel_range = (0.0, 0.0)
    dynamic_grasp_speed_curriculum = False
    dynamic_grasp_speed_curriculum_mode = "steps"
    dynamic_grasp_speed_curriculum_metric = "success"
    dynamic_grasp_speed_curriculum_start_success = 0.2
    dynamic_grasp_speed_curriculum_full_success = 0.6
    dynamic_grasp_speed_curriculum_ema_alpha = 0.02
    dynamic_grasp_speed_curriculum_alpha_rise = 1.0
    dynamic_grasp_speed_curriculum_allow_decrease = False
    dynamic_grasp_speed_curriculum_steps = 0
    dynamic_grasp_speed_curriculum_override_alpha = None
    falling_baton_spawn_height_curriculum = False
    falling_baton_start_spawn_above_palm_range = (0.40, 0.76)
    falling_baton_orientation_curriculum = False
    falling_baton_start_roll_range = (-3.141592653589793, 3.141592653589793)
    falling_baton_start_pitch_range = (-1.35, 1.35)
    falling_baton_start_yaw_range = (-3.141592653589793, 3.141592653589793)
    dynamic_tabletop_persistent_motion = False
    dynamic_tabletop_bounce_at_workspace = False
    dynamic_tabletop_release_motion_on_contact = False
    dynamic_tabletop_release_motion_contact_count = 1
    dynamic_tabletop_workspace_x = TABLETOP_LARGE_WORKSPACE_X
    dynamic_tabletop_workspace_y = TABLETOP_LARGE_WORKSPACE_Y
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.0, 0.0)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_heading_range = (-0.45, 0.45)
    dynamic_tabletop_randomize_yaw = False
    dynamic_tabletop_speed_alpha_sample_enabled = False
    dynamic_tabletop_speed_alpha_sample_min = 0.0
    dynamic_tabletop_speed_alpha_sample_max = 1.0
    dynamic_tabletop_speed_alpha_sample_curriculum_cap = True
    dynamic_tabletop_speed_alpha_sample_full_fraction = 0.0
    dynamic_tabletop_pregrasp_lead_time = 0.20
    dynamic_tabletop_pregrasp_ahead_distance = 0.04
    dynamic_tabletop_pregrasp_xy_distance_scale = 0.16
    dynamic_tabletop_pregrasp_ready_distance = 0.14
    dynamic_tabletop_pregrasp_xy_rew_scale = 0.0
    dynamic_tabletop_pregrasp_height_rew_scale = 0.0
    dynamic_tabletop_pregrasp_height_offset = 0.10
    dynamic_tabletop_pregrasp_height_scale = 0.05
    dynamic_tabletop_low_palm_penalty_scale = 0.0
    dynamic_tabletop_min_palm_height_offset = 0.05
    dynamic_tabletop_low_palm_height_scale = 0.04
    dynamic_tabletop_low_palm_max_penalty = 3.0
    tabletop_arm_clearance_body_names = (
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_link8",
        "panda_hand",
    )
    tabletop_arm_clearance_body_margins = ()
    tabletop_arm_clearance_xy_center = (TABLETOP_LARGE_TABLE_POS[0], TABLETOP_LARGE_TABLE_POS[1])
    tabletop_arm_clearance_xy_half_extent = (0.5 * TABLETOP_LARGE_TABLE_SIZE[0], 0.5 * TABLETOP_LARGE_TABLE_SIZE[1])
    tabletop_arm_clearance_xy_padding = 0.05
    tabletop_arm_clearance_margin = 0.040
    tabletop_arm_clearance_scale = 0.060
    tabletop_arm_clearance_max_penalty = 2.0
    tabletop_arm_clearance_penalty_scale = 0.0
    tabletop_arm_clearance_ok_penalty_threshold = 1.0e-4
    tabletop_success_requires_arm_clearance = False
    tabletop_terminate_on_arm_clearance_violation = False
    tabletop_arm_clearance_terminate_penalty_threshold = 1.0e-4
    tabletop_arm_clearance_violation_terminate_start_step = 0
    scripted_action_prior_lift_uses_proximity = False
    scripted_action_prior_lift_proximity_distance = 0.0
    scripted_action_prior_lift_proximity_min_contacts = 0.0
    scripted_action_prior_lift_grasp_recent_steps = 0
    tabletop_gate_contact_rewards_by_clearance = False
    tabletop_contact_clearance_gate_min = 1.0
    tabletop_contact_clearance_gate_scale = 0.50
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = False
    dynamic_tabletop_contact_pregrasp_gate_min = 1.0
    tabletop_object_asset_specs = ()
    tabletop_asset_set_enabled = False
    tabletop_asset_obs_enabled = False
    tabletop_asset_curriculum = False
    tabletop_asset_curriculum_mode = "steps"
    tabletop_asset_curriculum_start_count = 1
    tabletop_asset_curriculum_steps = 2_000_000
    tabletop_asset_curriculum_override_alpha = None
    tabletop_asset_sampling_weights = None
    tabletop_motion_modes = ("linear",)
    tabletop_motion_mode_curriculum = True
    tabletop_motion_mode_curriculum_start_count = 1
    tabletop_motion_mode_curriculum_steps = 2_000_000
    tabletop_motion_mode_curriculum_override_alpha = None
    tabletop_turntable_center = TABLETOP_LARGE_TURNTABLE_CENTER
    tabletop_turntable_radius_range = TABLETOP_LARGE_TURNTABLE_RADIUS_RANGE

    # Dynamic reward and termination gates.
    reach_distance_scale = 0.22
    fingertip_distance_scale = 0.055
    catch_success_min_z = 0.42
    falling_drop_z = 0.20
    tabletop_success_lift_height = 0.045
    dynamic_success_hold_steps = 8
    terminate_on_success = True
    workspace_xy_limit = 1.25
    stable_object_palm_vel = 0.36
    diagnostic_palm_reach_distance = 0.25
    object_contact_force_diagnostics_enabled = False
    object_contact_force_threshold = 0.05
    falling_success_uses_grasp_seen = True
    falling_success_uses_strict_grasp = False
    falling_success_max_palm_distance = 0.0
    falling_success_palm_gate_soft_scale = 0.055
    falling_success_min_finger_contacts = 0.0
    falling_success_requires_positive_affordance = False
    tabletop_success_requires_hover_target = False
    tabletop_hover_height_delta = 0.16
    tabletop_hover_latch_lift_progress = 0.35
    tabletop_hover_xy_distance_scale = 0.18
    tabletop_hover_z_distance_scale = 0.07
    tabletop_hover_object_speed_scale = 0.25
    tabletop_hover_ang_speed_scale = 8.0
    tabletop_hover_success_requires_xy = True
    tabletop_hover_latch_uses_grasp_seen = False
    tabletop_hover_reward_uses_grasp_seen = False
    tabletop_success_uses_grasp_seen = False
    tabletop_hover_success_xy_tolerance = 0.18
    tabletop_hover_success_z_tolerance = 0.08
    tabletop_hover_success_object_speed = 0.30
    contact_distance = 0.045
    contact_score_scale = 0.045
    min_finger_contacts = 2
    min_non_thumb_contacts = 1
    opposition_cos_threshold = 0.08
    true_grasp_opposition_mode = "score"
    palm_contact_distance = 0.10
    palm_only_lift_dist = 0.12
    strict_success_enabled = False
    strict_success_contact_distance = 0.010
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    strict_reward_enabled = False
    strict_reward_contact_score_scale = None
    strict_approach_score_scale = 0.08
    strict_approach_rew_scale = 0.0
    strict_multifinger_approach_rew_scale = 0.0
    strict_touch_score_scale = 0.008
    strict_touch_rew_scale = 0.0
    grasp_quality_finger_count_weight = 0.30
    grasp_quality_non_thumb_weight = 0.25
    grasp_quality_thumb_weight = 0.25
    grasp_quality_opposition_weight = 0.20
    tabletop_affordance_reward_enabled = False
    tabletop_affordance_label_root = TABLETOP_AFFORDANCE_ROOT
    tabletop_affordance_reward_points_per_region = 64
    tabletop_affordance_distance_scale = 0.035
    tabletop_affordance_contact_distance = 0.045
    tabletop_affordance_positive_rew_scale = 0.0
    tabletop_affordance_negative_penalty_scale = 0.0
    tabletop_affordance_lift_rew_scale = 0.0
    falling_affordance_reward_enabled = False
    falling_affordance_distance_scale = 0.025
    falling_affordance_contact_distance = 0.040
    falling_affordance_radial_margin = 0.020
    falling_affordance_positive_center_z = FALLING_BATON_POSITIVE_MARKER_OFFSET[2]
    falling_affordance_positive_half_z = 0.5 * FALLING_BATON_POSITIVE_MARKER_SIZE[2]
    falling_affordance_negative_center_z = FALLING_BATON_NEGATIVE_MARKER_OFFSET[2]
    falling_affordance_negative_half_z = 0.5 * FALLING_BATON_NEGATIVE_MARKER_SIZE[2]
    falling_affordance_positive_rew_scale = 0.0
    falling_affordance_negative_penalty_scale = 0.0

    # Reward scales.
    palm_reach_rew_scale = 4.0
    fingertip_reach_rew_scale = 5.0
    contact_rew_scale = 7.0
    true_grasp_rew_scale = 12.0
    opposition_rew_scale = 5.0
    catch_progress_rew_scale = 30.0
    falling_stable_grasp_rew_scale = 0.0
    falling_palm_gate_rew_scale = 0.0
    falling_positive_stable_rew_scale = 0.0
    falling_soft_success_progress_rew_scale = 0.0
    falling_opposed_stable_pinch_rew_scale = 0.0
    falling_pinched_rel_vel_penalty_scale = 0.0
    stable_hold_rew_scale = 80.0
    hold_progress_rew_scale = 140.0
    lift_progress_rew_scale = 45.0
    lift_reward_uses_grasp_quality_gate = False
    lift_reward_min_grasp_quality_multiplier = 1.0
    lift_reward_uses_opposition_gate = False
    lift_reward_min_opposition_multiplier = 1.0
    quality_lift_progress_uses_opposition_gate = False
    quality_lift_progress_min_opposition_multiplier = 1.0
    grasp_quality_rew_scale = 0.0
    quality_lift_progress_rew_scale = 0.0
    lifted_true_grasp_rew_scale = 0.0
    scoop_lift_penalty_scale = 0.0
    palm_only_lift_penalty_scale = 0.0
    success_bonus = 450.0
    tabletop_hover_target_rew_scale = 0.0
    tabletop_hover_height_progress_rew_scale = 0.0
    tabletop_hover_goal_rew_scale = 0.0
    tabletop_hover_stable_rew_scale = 0.0
    tabletop_hover_linear_penalty_scale = 0.0
    tabletop_hover_overshoot_penalty_scale = 0.0
    tabletop_hover_z_vel_penalty_scale = 0.0
    tabletop_hover_vel_penalty_scale = 0.0
    tabletop_hover_target_drift_penalty_scale = 0.0
    tabletop_hover_grasp_loss_penalty_scale = 0.0
    tabletop_hover_under_height_penalty_scale = 0.0
    tabletop_hover_post_latch_speed_penalty_scale = 0.0
    tabletop_hover_post_latch_action_penalty_scale = 0.0
    tabletop_hover_post_latch_target_delta_penalty_scale = 0.0
    tabletop_hover_target_obs_enabled = False
    tabletop_hover_target_obs_scale = 0.16
    tabletop_no_lift_after_grasp_penalty_scale = 0.0
    tabletop_no_lift_after_grasp_grace_steps = 30
    tabletop_no_lift_after_grasp_ramp_steps = 80
    tabletop_no_lift_after_grasp_max_penalty = 4.0
    tabletop_no_lift_min_progress = 0.15
    tabletop_grasped_palm_lift_rew_scale = 0.0
    tabletop_grasped_palm_lift_height = 0.08
    tabletop_grasped_palm_lift_scale = 0.05
    tabletop_grasped_arm_lift_rew_scale = 0.0
    tabletop_lift_action_prior_rew_scale = 0.0
    tabletop_lift_action_prior_gate_min = 0.10
    tabletop_lift_without_object_penalty_scale = 0.0
    tabletop_lift_without_object_min_arm_progress = 0.20
    tabletop_object_up_vel_rew_scale = 0.0
    tabletop_object_up_vel_scale = 0.10
    tabletop_object_carry_lift_rew_scale = 0.0
    tabletop_object_carry_min_grasp_streak = 0
    tabletop_object_carry_streak_ramp_steps = 1
    tabletop_object_carry_uses_grasp_seen = False
    tabletop_object_carry_grasp_seen_gate = 0.25
    tabletop_object_carry_stall_penalty_scale = 0.0
    tabletop_object_carry_stall_min_arm_progress = 0.12
    tabletop_object_carry_stall_min_z_vel = 0.015
    tabletop_underwrap_rew_scale = 0.0
    tabletop_underwrap_below_center_fraction = 0.20
    tabletop_underwrap_height_scale = 0.012
    tabletop_underwrap_radial_fraction = 0.95
    tabletop_underwrap_radial_scale = 0.020
    tabletop_underwrap_contact_scale = 0.018
    tabletop_underwrap_contact_margin = 0.0
    tabletop_underwrap_min_non_thumb_contacts = 1
    tabletop_underwrap_uses_opposition = True
    tabletop_underwrap_opposition_min_multiplier = 0.10
    tabletop_underwrap_progress_weight = 0.0
    tabletop_underwrap_pair_weight = 1.0
    tabletop_underwrap_uses_pregrasp_gate = True
    tabletop_post_success_hold_rew_scale = 0.0
    tabletop_post_success_unstable_penalty_scale = 0.0
    tabletop_post_success_grasp_loss_penalty_scale = 0.0
    tabletop_post_success_under_height_penalty_scale = 0.0
    tabletop_post_success_speed_penalty_scale = 0.0
    tabletop_post_success_action_penalty_scale = 0.0
    tabletop_post_success_target_delta_penalty_scale = 0.0
    tabletop_post_success_stability_latch_enabled = False
    tabletop_post_success_arm_target_lock_enabled = False
    tabletop_post_success_arm_target_lock_blend = 1.0
    tabletop_post_success_hand_target_lock_enabled = False
    tabletop_post_success_hand_target_lock_blend = 1.0
    tabletop_post_success_hand_lock_uses_actual_joint_pos = False
    stability_target_latch_min_success_streak = 0
    tabletop_post_success_arm_joint_vel_penalty_scale = 0.0
    tabletop_post_success_arm_target_drift_penalty_scale = 0.0
    tabletop_post_success_arm_target_drift_tolerance = 0.10
    tabletop_post_success_arm_target_drift_scale = 0.30
    tabletop_post_success_palm_drift_penalty_scale = 0.0
    tabletop_post_success_palm_drift_tolerance = 0.035
    tabletop_post_success_palm_drift_scale = 0.08
    falling_post_success_stability_enabled = False
    action_penalty_scale = 0.006
    arm_target_delta_penalty_scale = 0.010
    drop_penalty = 35.0


@configclass
class Revo2FallingBatonTeacherEnvCfg(Revo2DynamicDexterousTeacherEnvCfg):
    """Revo2 privileged teacher task for catching a falling baton."""

    task_family = "falling_baton_grasp"
    reference_name = "revo2_v699_falling_baton_privileged_teacher_joint_target"
    episode_length_s = 200.0 / 60.0
    robot_cfg: ArticulationCfg = _v699_revo2_robot_cfg(ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS)
    default_arm_pos = ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS
    create_table = False
    object_cfg: RigidObjectCfg = _falling_baton_physics_object_cfg(
        size=FALLING_BATON_PHYSICS_SIZE,
        mass=FALLING_BATON_AFFORDANCE_MASS,
        pos=(0.0, 0.24, 1.18),
    )
    object_size = FALLING_BATON_PHYSICS_SIZE
    object_start_pos = (0.0, 0.24, 1.18)
    reset_object_pos_noise = (0.0, 0.0, 0.0)
    falling_baton_palm_relative_spawn_enabled = True
    falling_baton_spawn_x_range = (-0.24, 0.24)
    falling_baton_spawn_y_range = (0.10, 0.42)
    falling_baton_spawn_z_range = (1.05, 1.42)
    falling_baton_spawn_above_palm_range = (0.40, 0.76)
    falling_baton_catch_center_finger_weight = 1.05
    falling_baton_catch_center_forward_offset = 0.13
    falling_baton_catch_center_world_offset = (0.0, 0.065, 0.0)
    falling_baton_palm_relative_start_x_range = (-0.060, 0.060)
    falling_baton_palm_relative_start_y_range = (0.030, 0.150)
    object_lin_vel_min = (0.00, 0.00, -0.26)
    object_lin_vel_max = (0.06, 0.06, -0.04)
    object_ang_vel_min = (-1.0, -1.0, -1.0)
    object_ang_vel_max = (1.0, 1.0, 1.0)
    falling_baton_start_initial_xy_speed_range = (0.00, 0.002)
    falling_baton_start_initial_z_speed_range = (0.00, 0.006)
    falling_baton_start_initial_ang_vel_range = (-0.03, 0.03)
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "contact_like"
    dynamic_grasp_speed_curriculum_start_success = 0.18
    dynamic_grasp_speed_curriculum_full_success = 0.62
    dynamic_grasp_speed_curriculum_ema_alpha = 0.03
    dynamic_grasp_speed_curriculum_alpha_rise = 0.001
    dynamic_grasp_speed_curriculum_allow_decrease = False
    dynamic_grasp_speed_curriculum_steps = 20_000_000
    catch_success_min_z = 0.43
    falling_drop_z = 0.05
    dynamic_success_hold_steps = 4
    stable_object_palm_vel = 0.60


@configclass
class Revo2FallingBatonFullSpeedEvalEnvCfg(Revo2FallingBatonTeacherEnvCfg):
    """Full V88-like falling-baton randomization without speed curriculum."""

    reference_name = "revo2_v699_falling_baton_full_speed_eval_teacher_joint_target"
    dynamic_grasp_speed_curriculum = False
    falling_baton_start_initial_xy_speed_range = (0.00, 0.06)
    falling_baton_start_initial_z_speed_range = (0.04, 0.26)
    falling_baton_start_initial_ang_vel_range = (-1.0, 1.0)


@configclass
class Revo2FallingBatonStableTeacherEnvCfg(Revo2FallingBatonTeacherEnvCfg):
    """Falling-baton teacher with strict stable-in-hand success semantics."""

    reference_name = "revo2_v699_falling_baton_stable_success_teacher"
    falling_success_uses_grasp_seen = False
    falling_success_uses_strict_grasp = True
    falling_success_max_palm_distance = 0.22
    falling_success_min_finger_contacts = 3.0
    strict_reward_enabled = True
    strict_success_enabled = True
    strict_success_contact_distance = 0.010
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    dynamic_success_hold_steps = 8
    stable_object_palm_vel = 0.38
    catch_success_min_z = 0.48
    dynamic_grasp_speed_curriculum_metric = "success"
    dynamic_grasp_speed_curriculum_start_success = 0.04
    dynamic_grasp_speed_curriculum_full_success = 0.24
    dynamic_grasp_speed_curriculum_ema_alpha = 0.035
    dynamic_grasp_speed_curriculum_alpha_rise = 0.0007
    dynamic_grasp_speed_curriculum_allow_decrease = True
    contact_rew_scale = 8.0
    true_grasp_rew_scale = 18.0
    opposition_rew_scale = 8.0
    catch_progress_rew_scale = 36.0
    stable_hold_rew_scale = 180.0
    hold_progress_rew_scale = 320.0
    success_bonus = 900.0


@configclass
class Revo2FallingBatonStableFullSpeedEvalEnvCfg(Revo2FallingBatonStableTeacherEnvCfg):
    """Strict stable-grasp eval with full falling-baton speed/orientation randomization."""

    reference_name = "revo2_v699_falling_baton_stable_full_speed_eval_teacher"
    dynamic_grasp_speed_curriculum = False
    falling_baton_start_initial_xy_speed_range = (0.00, 0.06)
    falling_baton_start_initial_z_speed_range = (0.04, 0.26)
    falling_baton_start_initial_ang_vel_range = (-1.0, 1.0)


@configclass
class Revo2FallingBatonStableAffordanceTeacherEnvCfg(Revo2FallingBatonStableTeacherEnvCfg):
    """Strict falling-baton task that rewards green-handle and penalizes red-region contact."""

    reference_name = "revo2_v699_falling_baton_stable_affordance_teacher"
    falling_affordance_reward_enabled = True
    falling_success_requires_positive_affordance = True
    falling_affordance_positive_rew_scale = 85.0
    falling_affordance_negative_penalty_scale = 140.0
    falling_affordance_distance_scale = 0.022
    falling_affordance_contact_distance = 0.038
    falling_affordance_radial_margin = 0.022


@configclass
class Revo2FallingBatonStableAffordanceFullSpeedEvalEnvCfg(Revo2FallingBatonStableAffordanceTeacherEnvCfg):
    """Strict green-region falling-baton eval with full randomization."""

    reference_name = "revo2_v699_falling_baton_stable_affordance_full_speed_eval_teacher"
    dynamic_grasp_speed_curriculum = False
    falling_baton_start_initial_xy_speed_range = (0.00, 0.06)
    falling_baton_start_initial_z_speed_range = (0.04, 0.26)
    falling_baton_start_initial_ang_vel_range = (-1.0, 1.0)


@configclass
class Revo2FallingBatonEasyTeacherEnvCfg(Revo2FallingBatonTeacherEnvCfg):
    """Early curriculum task for learning reach/contact before full aerial randomization."""

    reference_name = "revo2_v699_falling_baton_easy_curriculum_teacher_joint_target"
    episode_length_s = 200.0 / 60.0
    object_cfg: RigidObjectCfg = _falling_baton_physics_object_cfg(
        size=(0.018, 0.018, 0.165),
        mass=0.014,
        pos=(0.0, 0.23, 1.10),
    )
    object_size = (0.018, 0.018, 0.165)
    object_start_pos = (0.0, 0.23, 1.10)
    reset_object_pos_noise = (0.0, 0.0, 0.0)
    falling_baton_spawn_x_range = (-0.18, 0.18)
    falling_baton_spawn_y_range = (0.08, 0.36)
    falling_baton_spawn_z_range = (0.98, 1.30)
    falling_baton_spawn_above_palm_range = (0.32, 0.60)
    falling_baton_catch_center_finger_weight = 0.75
    falling_baton_catch_center_forward_offset = 0.055
    falling_baton_catch_center_world_offset = (0.0, 0.020, 0.0)
    falling_baton_palm_relative_start_x_range = (-0.035, 0.035)
    falling_baton_palm_relative_start_y_range = (0.015, 0.085)
    object_lin_vel_min = (0.00, 0.00, -0.015)
    object_lin_vel_max = (0.005, 0.005, 0.00)
    object_ang_vel_min = (-0.08, -0.08, -0.08)
    object_ang_vel_max = (0.08, 0.08, 0.08)
    falling_baton_start_initial_xy_speed_range = (0.00, 0.005)
    falling_baton_start_initial_z_speed_range = (0.00, 0.015)
    falling_baton_start_initial_ang_vel_range = (-0.08, 0.08)
    dynamic_grasp_speed_curriculum_metric = "true_grasp"
    dynamic_grasp_speed_curriculum_start_success = 0.02
    dynamic_grasp_speed_curriculum_full_success = 0.20
    dynamic_grasp_speed_curriculum_ema_alpha = 0.02
    dynamic_grasp_speed_curriculum_alpha_rise = 0.003
    dynamic_grasp_speed_curriculum_steps = 18_000_000
    catch_success_min_z = 0.46
    falling_drop_z = 0.22
    dynamic_success_hold_steps = 6
    contact_distance = 0.060
    contact_score_scale = 0.060
    stable_object_palm_vel = 0.55
    palm_reach_rew_scale = 10.0
    fingertip_reach_rew_scale = 10.0
    contact_rew_scale = 14.0
    true_grasp_rew_scale = 30.0
    opposition_rew_scale = 10.0
    catch_progress_rew_scale = 120.0
    stable_hold_rew_scale = 300.0
    hold_progress_rew_scale = 800.0
    success_bonus = 2500.0
    action_penalty_scale = 0.003
    arm_target_delta_penalty_scale = 0.005
    drop_penalty = 12.0


@configclass
class Revo2FallingBatonEasyStrictAffordanceTeacherEnvCfg(Revo2FallingBatonEasyTeacherEnvCfg):
    """Easy short-baton curriculum with strict load-bearing green-region grasp rewards."""

    reference_name = "revo2_v699_falling_baton_easy_strict_affordance_teacher"

    falling_success_uses_grasp_seen = False
    falling_success_uses_strict_grasp = True
    falling_success_requires_positive_affordance = True
    falling_success_max_palm_distance = 0.20
    falling_success_min_finger_contacts = 3.0

    contact_distance = 0.018
    contact_score_scale = 0.018
    palm_contact_distance = 0.050
    strict_reward_enabled = True
    strict_success_enabled = True
    strict_success_contact_distance = 0.010
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    strict_approach_rew_scale = 12.0
    strict_multifinger_approach_rew_scale = 8.0
    strict_opposition_approach_rew_scale = 180.0
    strict_touch_reward_requires_thumb_pair = True
    strict_touch_reward_uses_opposition_product = True
    strict_touch_reward_opposition_min_multiplier = 0.02
    strict_touch_rew_scale = 1000.0
    strict_opposition_touch_rew_scale = 3600.0

    falling_affordance_reward_enabled = True
    falling_affordance_positive_requires_thumb_pair = True
    falling_affordance_positive_uses_opposition_product = True
    falling_affordance_positive_opposition_min_multiplier = 0.02
    falling_affordance_positive_rew_scale = 80.0
    falling_affordance_thumb_geom_rew_scale = 10.0
    falling_affordance_thumb_touch_rew_scale = 260.0
    falling_affordance_negative_penalty_scale = 260.0
    falling_affordance_distance_scale = 0.018
    falling_affordance_contact_distance = 0.016
    falling_affordance_radial_margin = 0.010

    falling_non_thumb_without_thumb_penalty_scale = 120.0
    falling_non_thumb_without_thumb_gate_start = 0.16
    falling_non_thumb_without_thumb_gate_ramp = 0.32
    falling_non_thumb_without_thumb_thumb_target = 0.32

    dynamic_grasp_speed_curriculum_metric = "success"
    dynamic_grasp_speed_curriculum_start_success = 0.006
    dynamic_grasp_speed_curriculum_full_success = 0.16
    dynamic_grasp_speed_curriculum_ema_alpha = 0.035
    dynamic_grasp_speed_curriculum_alpha_rise = 0.001
    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_success_hold_steps = 10
    stable_object_palm_vel = 0.38

    contact_rew_scale = 8.0
    true_grasp_rew_scale = 80.0
    opposition_rew_scale = 18.0
    catch_progress_rew_scale = 24.0
    falling_stable_grasp_rew_scale = 650.0
    falling_palm_gate_rew_scale = 450.0
    falling_positive_stable_rew_scale = 1700.0
    falling_soft_success_progress_rew_scale = 2800.0
    falling_opposed_stable_pinch_rew_scale = 4000.0
    falling_pinched_rel_vel_penalty_scale = 700.0
    stable_hold_rew_scale = 12000.0
    hold_progress_rew_scale = 20000.0
    success_bonus = 40000.0


@configclass
class Revo2FallingBatonEasyStrictAffordancePostHoldTeacherEnvCfg(
    Revo2FallingBatonEasyStrictAffordanceTeacherEnvCfg
):
    """Strict Revo2 catch task that keeps a successful grasp stable afterward."""

    reference_name = "revo2_v699_falling_baton_easy_strict_affordance_posthold_teacher"

    dynamic_success_hold_steps = 20
    stable_object_palm_vel = 0.30
    terminate_on_success = False
    falling_post_success_stability_enabled = True
    tabletop_post_success_stability_latch_enabled = True
    tabletop_post_success_arm_target_lock_enabled = True
    tabletop_post_success_arm_target_lock_blend = 1.0
    tabletop_post_success_hand_target_lock_enabled = True
    tabletop_post_success_hand_target_lock_blend = 1.0
    tabletop_post_success_hand_close_fraction = 0.06

    tabletop_post_success_hold_rew_scale = 18000.0
    tabletop_post_success_unstable_penalty_scale = 15000.0
    tabletop_post_success_grasp_loss_penalty_scale = 12000.0
    tabletop_post_success_under_height_penalty_scale = 0.0
    tabletop_post_success_speed_penalty_scale = 2600.0
    tabletop_post_success_action_penalty_scale = 0.040
    tabletop_post_success_target_delta_penalty_scale = 0.060
    tabletop_post_success_arm_joint_vel_penalty_scale = 60.0
    tabletop_post_success_arm_target_drift_penalty_scale = 1100.0
    tabletop_post_success_arm_target_drift_tolerance = 0.12
    tabletop_post_success_arm_target_drift_scale = 0.30
    tabletop_post_success_palm_drift_penalty_scale = 1400.0
    tabletop_post_success_palm_drift_tolerance = 0.050
    tabletop_post_success_palm_drift_scale = 0.10

    stable_hold_rew_scale = 22000.0
    hold_progress_rew_scale = 40000.0
    success_bonus = 70000.0


@configclass
class Revo2UnifiedFallingBatonBenchmarkTeacherEnvCfg(
    Revo2FallingBatonEasyStrictAffordancePostHoldTeacherEnvCfg
):
    """Revo2 adapter for the shared red/green falling-baton benchmark."""

    reference_name = "revo2_unified_falling_baton_affordance_v1_teacher"
    benchmark_protocol = UNIFIED_FALLING_BENCHMARK_NAME
    action_space = 13
    observation_space = 76
    task_family = "falling_baton_grasp"
    create_table = False
    episode_length_s = UNIFIED_FALLING_EPISODE_LENGTH_S

    object_cfg: RigidObjectCfg = _falling_baton_physics_object_cfg(
        size=UNIFIED_FALLING_OBJECT_SIZE,
        mass=UNIFIED_FALLING_OBJECT_MASS,
        pos=UNIFIED_FALLING_START_POS,
    )
    object_shape = "box"
    object_radius = 0.010
    object_size = UNIFIED_FALLING_OBJECT_SIZE
    object_start_pos = UNIFIED_FALLING_START_POS
    object_start_rot = (1.0, 0.0, 0.0, 0.0)
    reset_object_pos_noise = (0.0, 0.0, 0.0)
    affordance_label_mode = "handle_blade"
    affordance_positive_fraction = 0.38
    affordance_negative_fraction = 0.45
    affordance_positive_end = "negative"
    falling_baton_affordance_markers_enabled = True

    falling_baton_palm_relative_spawn_enabled = True
    falling_baton_palm_relative_clamp_to_workspace = True
    falling_baton_spawn_x_range = UNIFIED_FALLING_SPAWN_X_RANGE
    falling_baton_spawn_y_range = UNIFIED_FALLING_SPAWN_Y_RANGE
    falling_baton_spawn_z_range = UNIFIED_FALLING_SPAWN_Z_RANGE
    falling_baton_spawn_above_palm_enabled = True
    falling_baton_spawn_height_curriculum = True
    falling_baton_start_spawn_above_palm_range = UNIFIED_FALLING_START_ABOVE_PALM_RANGE
    falling_baton_spawn_above_palm_range = UNIFIED_FALLING_TARGET_ABOVE_PALM_RANGE
    falling_baton_catch_center_finger_weight = 0.78
    falling_baton_catch_center_forward_offset = 0.060
    falling_baton_catch_center_world_offset = (0.0, 0.020, 0.0)
    falling_baton_palm_relative_start_x_range = (-0.040, 0.040)
    falling_baton_palm_relative_start_y_range = (0.015, 0.090)

    falling_baton_randomize_orientation = True
    falling_baton_orientation_curriculum = True
    falling_baton_start_roll_range = UNIFIED_FALLING_START_ROLL_RANGE
    falling_baton_start_pitch_range = UNIFIED_FALLING_START_PITCH_RANGE
    falling_baton_start_yaw_range = UNIFIED_FALLING_START_YAW_RANGE
    falling_baton_roll_range = UNIFIED_FALLING_TARGET_ROLL_RANGE
    falling_baton_pitch_range = UNIFIED_FALLING_TARGET_PITCH_RANGE
    falling_baton_yaw_range = UNIFIED_FALLING_TARGET_YAW_RANGE

    object_lin_vel_min = UNIFIED_FALLING_TARGET_LIN_VEL_MIN
    object_lin_vel_max = UNIFIED_FALLING_TARGET_LIN_VEL_MAX
    object_ang_vel_min = UNIFIED_FALLING_TARGET_ANG_VEL_MIN
    object_ang_vel_max = UNIFIED_FALLING_TARGET_ANG_VEL_MAX
    falling_baton_start_initial_xy_speed_range = UNIFIED_FALLING_START_XY_SPEED_RANGE
    falling_baton_start_initial_z_speed_range = UNIFIED_FALLING_START_Z_SPEED_RANGE
    falling_baton_start_initial_ang_vel_range = UNIFIED_FALLING_START_ANG_VEL_RANGE
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = UNIFIED_FALLING_CURRICULUM_METRIC
    dynamic_grasp_speed_curriculum_start_success = UNIFIED_FALLING_CURRICULUM_START_SUCCESS
    dynamic_grasp_speed_curriculum_full_success = UNIFIED_FALLING_CURRICULUM_FULL_SUCCESS
    dynamic_grasp_speed_curriculum_ema_alpha = UNIFIED_FALLING_CURRICULUM_EMA_ALPHA
    dynamic_grasp_speed_curriculum_alpha_rise = UNIFIED_FALLING_CURRICULUM_ALPHA_RISE
    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_grasp_speed_curriculum_override_alpha = None

    falling_success_uses_grasp_seen = False
    falling_success_uses_strict_grasp = True
    falling_success_requires_positive_affordance = True
    falling_success_max_palm_distance = 0.20
    falling_success_min_finger_contacts = 3.0
    catch_success_min_z = 0.46
    falling_drop_z = 0.22
    contact_distance = 0.018
    contact_score_scale = 0.018
    palm_contact_distance = 0.050
    strict_reward_enabled = True
    strict_success_enabled = True
    strict_success_contact_distance = 0.010
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    falling_affordance_reward_enabled = True
    falling_affordance_positive_requires_thumb_pair = True
    falling_affordance_positive_uses_opposition_product = True
    falling_affordance_positive_opposition_min_multiplier = 0.02
    falling_affordance_distance_scale = 0.018
    falling_affordance_contact_distance = 0.016
    falling_affordance_radial_margin = 0.010

    dynamic_success_hold_steps = UNIFIED_FALLING_SUCCESS_HOLD_STEPS
    stable_object_palm_vel = UNIFIED_FALLING_STABLE_OBJECT_PALM_VEL
    terminate_on_success = False
    falling_post_success_stability_enabled = True
    tabletop_post_success_stability_latch_enabled = True
    tabletop_post_success_arm_target_lock_enabled = True
    tabletop_post_success_arm_target_lock_blend = 1.0
    tabletop_post_success_hand_target_lock_enabled = True
    tabletop_post_success_hand_target_lock_blend = 1.0
    tabletop_post_success_hand_lock_uses_actual_joint_pos = False
    tabletop_post_success_hand_close_fraction = 0.06
    scripted_action_prior_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_relative_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False


@configclass
class Revo2DynamicTabletopTeacherEnvCfg(Revo2DynamicDexterousTeacherEnvCfg):
    """Revo2 privileged teacher task for grasping a moving tabletop object."""

    task_family = "dynamic_tabletop_grasp"
    reference_name = "revo2_dynamic_tabletop_privileged_teacher_joint_target"
    episode_length_s = 6.0
    robot_cfg: ArticulationCfg = _v699_revo2_robot_cfg(FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS)
    table_cfg: RigidObjectCfg = _table_cfg(pos=TABLETOP_LARGE_TABLE_POS, size=TABLETOP_LARGE_TABLE_SIZE)
    default_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    object_cfg: RigidObjectCfg = _cube_object_cfg(
        size=(0.040, 0.040, 0.080),
        mass=0.030,
        pos=(0.58, -0.16, 0.345),
    )
    object_size = (0.040, 0.040, 0.080)
    object_start_pos = (0.58, -0.16, 0.345)
    reset_object_pos_noise = (0.060, 0.030, 0.004)
    object_lin_vel_min = (-0.10, 0.03, 0.0)
    object_lin_vel_max = (0.10, 0.24, 0.0)
    object_ang_vel_min = (0.0, 0.0, -1.0)
    object_ang_vel_max = (0.0, 0.0, 1.0)
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "steps"
    dynamic_grasp_speed_curriculum_steps = 2_000_000
    dynamic_tabletop_persistent_motion = True
    dynamic_tabletop_bounce_at_workspace = True
    dynamic_tabletop_workspace_x = TABLETOP_LARGE_WORKSPACE_X
    dynamic_tabletop_workspace_y = TABLETOP_LARGE_WORKSPACE_Y
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.10, 0.24)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (-1.5, 1.5)
    dynamic_tabletop_heading_range = (-0.45, 0.45)
    dynamic_tabletop_randomize_yaw = True
    affordance_label_mode = "side_grasp"
    tabletop_success_lift_height = 0.045
    dynamic_success_hold_steps = 10
    catch_success_min_z = 0.30
    falling_drop_z = 0.22
    palm_reach_rew_scale = 6.0
    fingertip_reach_rew_scale = 6.0
    lift_progress_rew_scale = 180.0
    stable_hold_rew_scale = 140.0
    hold_progress_rew_scale = 220.0


@configclass
class Revo2DynamicTabletopLegacyFullHandTeacherEnvCfg(Revo2DynamicTabletopTeacherEnvCfg):
    """Legacy tabletop teacher matching old 18-action Revo2 simulation checkpoints."""

    reference_name = "revo2_dynamic_tabletop_legacy_full_hand_privileged_teacher"
    action_space = 18
    observation_space = 91
    action_contract = "revo2_legacy_full_hand_18d"
    hand_joint_names = REVO2_V699_PHYSICAL_HAND_JOINT_NAMES
    sim_hand_joint_names = REVO2_V699_PHYSICAL_HAND_JOINT_NAMES
    reference_hand_fractions = (1.0,) * len(REVO2_V699_PHYSICAL_HAND_JOINT_NAMES)


@configclass
class Revo2DynamicTabletopRollingTeacherEnvCfg(Revo2DynamicTabletopTeacherEnvCfg):
    """Rolling-object tabletop teacher: ball, bottle, cylinder and cone proxies."""

    reference_name = "revo2_dynamic_tabletop_rolling_assets_teacher"
    observation_space = 86
    tabletop_object_asset_specs = TABLETOP_ROLLING_OBJECT_SPECS
    tabletop_asset_set_enabled = True
    tabletop_asset_obs_enabled = True
    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_start_count = 1
    tabletop_asset_curriculum_steps = 2_000_000
    tabletop_motion_modes = ("free",)
    tabletop_motion_mode_curriculum = False
    tabletop_motion_mode_curriculum_start_count = 1
    tabletop_motion_mode_curriculum_steps = 2_000_000
    dynamic_tabletop_persistent_motion = False
    dynamic_tabletop_bounce_at_workspace = False
    dynamic_tabletop_release_motion_on_contact = False
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=(0.58, -0.16, TABLETOP_ROLLING_START_Z),
    )
    object_shape = str(TABLETOP_ROLLING_START_SPEC["proxy_shape"])
    object_radius = float(TABLETOP_ROLLING_START_SPEC["radius"])
    object_size = tuple(TABLETOP_ROLLING_START_SPEC["size"])
    object_start_pos = (0.58, -0.16, TABLETOP_ROLLING_START_Z)
    reset_object_pos_noise = (0.070, 0.035, 0.002)
    dynamic_tabletop_initial_speed_range = (0.04, 0.22)
    dynamic_tabletop_initial_yaw_rate_range = (-2.2, 2.2)
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "steps"
    dynamic_grasp_speed_curriculum_steps = 2_000_000
    affordance_label_mode = "tabletop_rolling_assets"


@configclass
class Revo2DynamicTabletopTransportTeacherEnvCfg(Revo2DynamicTabletopTeacherEnvCfg):
    """Transport tabletop teacher: cup/kettle/hammer/brush proxies on conveyor or turntable motion."""

    reference_name = "revo2_dynamic_tabletop_transport_assets_teacher"
    observation_space = 86
    tabletop_object_asset_specs = TABLETOP_TRANSPORT_OBJECT_SPECS
    tabletop_asset_set_enabled = True
    tabletop_asset_obs_enabled = True
    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_start_count = 2
    tabletop_asset_curriculum_steps = 2_500_000
    tabletop_motion_modes = ("linear", "curved", "turntable")
    tabletop_motion_mode_curriculum = True
    tabletop_motion_mode_curriculum_start_count = 1
    tabletop_motion_mode_curriculum_steps = 2_500_000
    tabletop_turntable_center = TABLETOP_LARGE_TURNTABLE_CENTER
    tabletop_turntable_radius_range = TABLETOP_LARGE_TURNTABLE_RADIUS_RANGE
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_TRANSPORT_OBJECT_SPECS[0],
        pos=(0.58, -0.16, 0.335),
    )
    object_shape = "cylinder"
    object_radius = 0.032
    object_size = (0.064, 0.064, 0.078)
    object_start_pos = (0.58, -0.16, 0.335)
    reset_object_pos_noise = (0.075, 0.040, 0.002)
    dynamic_tabletop_initial_speed_range = (0.03, 0.18)
    dynamic_tabletop_initial_yaw_rate_range = (-1.2, 1.2)
    dynamic_tabletop_heading_range = (-0.75, 0.75)
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "steps"
    dynamic_grasp_speed_curriculum_steps = 2_500_000
    affordance_label_mode = "tabletop_transport_assets"


@configclass
class Revo2DynamicTabletopRollingStrongRewardTeacherEnvCfg(
    Revo2DynamicTabletopRollingTeacherEnvCfg
):
    """Reward ablation B for Revo2 rolling tabletop grasping."""

    reference_name = "revo2_dynamic_tabletop_rolling_assets_strong_reward_teacher"
    episode_length_s = 7.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 24.0
    dynamic_tabletop_pregrasp_height_rew_scale = 30.0
    dynamic_tabletop_pregrasp_height_offset = 0.105
    dynamic_tabletop_pregrasp_height_scale = 0.055
    dynamic_tabletop_low_palm_penalty_scale = 75.0
    dynamic_tabletop_min_palm_height_offset = 0.055
    dynamic_tabletop_low_palm_height_scale = 0.040
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = True
    dynamic_tabletop_contact_pregrasp_gate_min = 0.20

    contact_rew_scale = 12.0
    true_grasp_rew_scale = 80.0
    opposition_rew_scale = 42.0
    catch_progress_rew_scale = 110.0
    lift_progress_rew_scale = 80.0
    stable_hold_rew_scale = 1150.0
    hold_progress_rew_scale = 2200.0
    success_bonus = 4200.0
    dynamic_success_hold_steps = 12
    terminate_on_success = False
    tabletop_success_requires_hover_target = True
    tabletop_hover_success_requires_xy = True
    tabletop_hover_height_delta = 0.10
    tabletop_hover_latch_lift_progress = 0.18
    tabletop_hover_xy_distance_scale = 0.16
    tabletop_hover_z_distance_scale = 0.050
    tabletop_hover_object_speed_scale = 0.20
    tabletop_hover_ang_speed_scale = 7.0
    tabletop_hover_success_xy_tolerance = 0.16
    tabletop_hover_success_z_tolerance = 0.055
    tabletop_hover_success_object_speed = 0.26
    tabletop_hover_target_rew_scale = 1100.0
    tabletop_hover_height_progress_rew_scale = 650.0
    tabletop_hover_goal_rew_scale = 5200.0
    tabletop_hover_stable_rew_scale = 5600.0
    tabletop_hover_linear_penalty_scale = 1200.0
    tabletop_hover_overshoot_penalty_scale = 1800.0
    tabletop_hover_z_vel_penalty_scale = 2300.0
    tabletop_hover_vel_penalty_scale = 950.0
    tabletop_hover_target_drift_penalty_scale = 3200.0
    tabletop_hover_grasp_loss_penalty_scale = 1800.0
    tabletop_hover_under_height_penalty_scale = 1200.0
    tabletop_hover_post_latch_speed_penalty_scale = 2200.0
    tabletop_hover_post_latch_action_penalty_scale = 0.045
    tabletop_hover_post_latch_target_delta_penalty_scale = 0.070
    tabletop_post_success_hold_rew_scale = 3500.0
    tabletop_post_success_unstable_penalty_scale = 6000.0
    tabletop_post_success_grasp_loss_penalty_scale = 3500.0
    tabletop_post_success_under_height_penalty_scale = 1600.0
    tabletop_post_success_speed_penalty_scale = 1000.0
    tabletop_post_success_action_penalty_scale = 0.050
    tabletop_post_success_target_delta_penalty_scale = 0.080

    lift_reward_uses_grasp_quality_gate = True
    lift_reward_min_grasp_quality_multiplier = 0.10
    lift_reward_uses_opposition_gate = True
    lift_reward_min_opposition_multiplier = 0.15
    quality_lift_progress_uses_opposition_gate = True
    quality_lift_progress_min_opposition_multiplier = 0.15
    grasp_quality_rew_scale = 70.0
    quality_lift_progress_rew_scale = 180.0
    lifted_true_grasp_rew_scale = 360.0
    scoop_lift_penalty_scale = 260.0
    palm_only_lift_penalty_scale = 240.0

    action_penalty_scale = 0.004
    arm_target_delta_penalty_scale = 0.006
    drop_penalty = 25.0


@configclass
class Revo2DynamicTabletopTransportStrongRewardTeacherEnvCfg(
    Revo2DynamicTabletopTransportTeacherEnvCfg
):
    """Reward ablation B for Revo2 transported tabletop objects."""

    reference_name = "revo2_dynamic_tabletop_transport_assets_strong_reward_teacher"
    episode_length_s = 7.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 14.0
    dynamic_tabletop_pregrasp_height_rew_scale = 10.0
    dynamic_tabletop_pregrasp_height_offset = 0.060
    dynamic_tabletop_pregrasp_height_scale = 0.045
    dynamic_tabletop_low_palm_penalty_scale = 36.0
    dynamic_tabletop_min_palm_height_offset = 0.018
    dynamic_tabletop_low_palm_height_scale = 0.030
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = True
    dynamic_tabletop_contact_pregrasp_gate_min = 0.45

    contact_rew_scale = 22.0
    true_grasp_rew_scale = 170.0
    opposition_rew_scale = 72.0
    catch_progress_rew_scale = 210.0
    lift_progress_rew_scale = 65.0
    stable_hold_rew_scale = 1850.0
    hold_progress_rew_scale = 3800.0
    success_bonus = 8000.0
    dynamic_success_hold_steps = 12
    terminate_on_success = False
    tabletop_success_requires_hover_target = True
    tabletop_hover_success_requires_xy = True
    tabletop_hover_height_delta = 0.08
    tabletop_hover_latch_lift_progress = 0.18
    tabletop_hover_xy_distance_scale = 0.14
    tabletop_hover_z_distance_scale = 0.045
    tabletop_hover_object_speed_scale = 0.18
    tabletop_hover_ang_speed_scale = 6.5
    tabletop_hover_success_xy_tolerance = 0.18
    tabletop_hover_success_z_tolerance = 0.050
    tabletop_hover_success_object_speed = 0.24
    tabletop_hover_target_rew_scale = 1700.0
    tabletop_hover_height_progress_rew_scale = 700.0
    tabletop_hover_goal_rew_scale = 6500.0
    tabletop_hover_stable_rew_scale = 6800.0
    tabletop_hover_linear_penalty_scale = 1500.0
    tabletop_hover_overshoot_penalty_scale = 2200.0
    tabletop_hover_z_vel_penalty_scale = 2700.0
    tabletop_hover_vel_penalty_scale = 1100.0
    tabletop_hover_target_drift_penalty_scale = 4200.0
    tabletop_hover_grasp_loss_penalty_scale = 2600.0
    tabletop_hover_under_height_penalty_scale = 1500.0
    tabletop_hover_post_latch_speed_penalty_scale = 2800.0
    tabletop_hover_post_latch_action_penalty_scale = 0.065
    tabletop_hover_post_latch_target_delta_penalty_scale = 0.095
    tabletop_post_success_hold_rew_scale = 5500.0
    tabletop_post_success_unstable_penalty_scale = 8000.0
    tabletop_post_success_grasp_loss_penalty_scale = 5200.0
    tabletop_post_success_under_height_penalty_scale = 2200.0
    tabletop_post_success_speed_penalty_scale = 1500.0
    tabletop_post_success_action_penalty_scale = 0.080
    tabletop_post_success_target_delta_penalty_scale = 0.120

    lift_reward_uses_grasp_quality_gate = True
    lift_reward_min_grasp_quality_multiplier = 0.10
    lift_reward_uses_opposition_gate = True
    lift_reward_min_opposition_multiplier = 0.15
    quality_lift_progress_uses_opposition_gate = True
    quality_lift_progress_min_opposition_multiplier = 0.15
    grasp_quality_rew_scale = 130.0
    quality_lift_progress_rew_scale = 260.0
    lifted_true_grasp_rew_scale = 720.0
    scoop_lift_penalty_scale = 260.0
    palm_only_lift_penalty_scale = 240.0

    action_penalty_scale = 0.004
    arm_target_delta_penalty_scale = 0.006
    drop_penalty = 25.0


@configclass
class Revo2DynamicTabletopTransportHoldRampTeacherEnvCfg(
    Revo2DynamicTabletopTransportStrongRewardTeacherEnvCfg
):
    """86-observation transport stage that preserves lift and learns stable holds."""

    reference_name = "revo2_dynamic_tabletop_transport_assets_hold_ramp_teacher"

    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_start_count = 2
    tabletop_asset_curriculum_steps = 3_500_000
    tabletop_motion_modes = ("linear", "curved", "turntable")
    tabletop_motion_mode_curriculum = True
    tabletop_motion_mode_curriculum_start_count = 1
    tabletop_motion_mode_curriculum_steps = 3_000_000

    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.02
    dynamic_grasp_speed_curriculum_full_success = 0.24
    dynamic_grasp_speed_curriculum_ema_alpha = 0.035
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00045
    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.010, 0.105)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (-0.55, 0.55)
    dynamic_tabletop_heading_range = (-0.80, 0.80)

    # Stage target: first make "grasped, lifted, stable in air" reliable.
    # Hover-target rewards remain active, but exact hover alignment is not a
    # hard success gate until a later target-observation stage.
    tabletop_success_requires_hover_target = False
    tabletop_success_uses_grasp_seen = True
    tabletop_hover_latch_uses_grasp_seen = True
    tabletop_hover_reward_uses_grasp_seen = True
    tabletop_success_lift_height = 0.045
    dynamic_success_hold_steps = 8
    stable_object_palm_vel = 0.40

    tabletop_hover_height_delta = 0.060
    tabletop_hover_latch_lift_progress = 0.14
    tabletop_hover_xy_distance_scale = 0.18
    tabletop_hover_z_distance_scale = 0.060
    tabletop_hover_object_speed_scale = 0.24
    tabletop_hover_ang_speed_scale = 7.0
    tabletop_hover_success_xy_tolerance = 0.22
    tabletop_hover_success_z_tolerance = 0.070
    tabletop_hover_success_object_speed = 0.30

    contact_rew_scale = 24.0
    true_grasp_rew_scale = 180.0
    opposition_rew_scale = 76.0
    catch_progress_rew_scale = 240.0
    lift_progress_rew_scale = 360.0
    stable_hold_rew_scale = 4200.0
    hold_progress_rew_scale = 6200.0
    success_bonus = 12500.0

    lift_reward_uses_grasp_quality_gate = False
    lift_reward_uses_opposition_gate = False
    quality_lift_progress_uses_opposition_gate = False
    quality_lift_progress_rew_scale = 900.0
    lifted_true_grasp_rew_scale = 2600.0
    grasp_quality_rew_scale = 145.0

    tabletop_hover_target_rew_scale = 950.0
    tabletop_hover_height_progress_rew_scale = 1500.0
    tabletop_hover_goal_rew_scale = 3600.0
    tabletop_hover_stable_rew_scale = 4600.0
    tabletop_hover_linear_penalty_scale = 520.0
    tabletop_hover_overshoot_penalty_scale = 900.0
    tabletop_hover_z_vel_penalty_scale = 1250.0
    tabletop_hover_vel_penalty_scale = 650.0
    tabletop_hover_target_drift_penalty_scale = 1100.0
    tabletop_hover_grasp_loss_penalty_scale = 1700.0
    tabletop_hover_under_height_penalty_scale = 950.0
    tabletop_hover_post_latch_speed_penalty_scale = 1200.0
    tabletop_hover_post_latch_action_penalty_scale = 0.035
    tabletop_hover_post_latch_target_delta_penalty_scale = 0.055

    tabletop_post_success_hold_rew_scale = 6200.0
    tabletop_post_success_unstable_penalty_scale = 5200.0
    tabletop_post_success_grasp_loss_penalty_scale = 3600.0
    tabletop_post_success_under_height_penalty_scale = 1200.0
    tabletop_post_success_speed_penalty_scale = 850.0
    tabletop_post_success_action_penalty_scale = 0.040
    tabletop_post_success_target_delta_penalty_scale = 0.060

    tabletop_grasped_palm_lift_rew_scale = 520.0
    tabletop_grasped_palm_lift_height = 0.050
    tabletop_grasped_palm_lift_scale = 0.045
    tabletop_arm_lift_reward_object_margin = 0.18
    tabletop_arm_object_lift_gap_margin = 0.12
    tabletop_arm_object_lift_gap_penalty_scale = 140.0
    tabletop_lift_without_object_min_arm_progress = 0.10
    tabletop_lift_without_object_penalty_scale = 80.0
    tabletop_no_lift_after_grasp_penalty_scale = 130.0
    tabletop_no_lift_after_grasp_grace_steps = 14
    tabletop_no_lift_after_grasp_ramp_steps = 70
    tabletop_no_lift_after_grasp_max_penalty = 4.0
    palm_only_lift_penalty_scale = 360.0
    scoop_lift_penalty_scale = 320.0

    action_penalty_scale = 0.004
    arm_target_delta_penalty_scale = 0.006
    drop_penalty = 28.0


@configclass
class Revo2DynamicTabletopTransportConservativeLiftTeacherEnvCfg(
    Revo2DynamicTabletopTransportHoldRampTeacherEnvCfg
):
    """Transport continuation that protects object lift success from contact-only reward hacking."""

    reference_name = "revo2_dynamic_tabletop_transport_conservative_lift_teacher"

    dynamic_grasp_speed_curriculum_allow_decrease = False
    dynamic_grasp_speed_curriculum_start_success = 0.08
    dynamic_grasp_speed_curriculum_full_success = 0.28
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00035

    dynamic_tabletop_pregrasp_xy_rew_scale = 9.0
    dynamic_tabletop_pregrasp_height_rew_scale = 11.0
    contact_rew_scale = 7.0
    true_grasp_rew_scale = 55.0
    opposition_rew_scale = 26.0
    grasp_quality_rew_scale = 24.0

    lift_reward_uses_grasp_quality_gate = True
    lift_reward_min_grasp_quality_multiplier = 0.10
    lift_reward_uses_opposition_gate = True
    lift_reward_min_opposition_multiplier = 0.10
    quality_lift_progress_uses_opposition_gate = True
    quality_lift_progress_min_opposition_multiplier = 0.12
    catch_progress_rew_scale = 420.0
    lift_progress_rew_scale = 980.0
    quality_lift_progress_rew_scale = 2200.0
    lifted_true_grasp_rew_scale = 5400.0
    tabletop_stable_catch_rew_scale = 2400.0
    stable_hold_rew_scale = 7600.0
    hold_progress_rew_scale = 12000.0
    success_bonus = 26000.0

    tabletop_grasped_palm_lift_rew_scale = 1200.0
    tabletop_grasped_arm_lift_rew_scale = 1500.0
    tabletop_arm_lift_reward_object_margin = 0.025
    tabletop_arm_object_lift_gap_margin = 0.025
    tabletop_arm_object_lift_gap_penalty_scale = 4200.0
    tabletop_lift_action_prior_rew_scale = 170.0
    tabletop_lift_action_prior_gate_min = 0.10
    tabletop_lift_without_object_min_arm_progress = 0.045
    tabletop_lift_without_object_penalty_scale = 2600.0
    tabletop_no_lift_after_grasp_penalty_scale = 1900.0
    tabletop_no_lift_after_grasp_grace_steps = 6
    tabletop_no_lift_after_grasp_ramp_steps = 36
    tabletop_no_lift_after_grasp_max_penalty = 6.0
    palm_only_lift_penalty_scale = 720.0
    scoop_lift_penalty_scale = 620.0
    tabletop_arm_clearance_body_margins = (
        0.155,
        0.165,
        0.150,
        0.130,
        0.105,
        0.080,
        0.060,
        0.035,
    )
    tabletop_arm_clearance_margin = 0.055
    tabletop_arm_clearance_scale = 0.045
    tabletop_arm_clearance_max_penalty = 4.0
    tabletop_arm_clearance_penalty_scale = 4200.0
    tabletop_arm_clearance_xy_padding = 0.18
    tabletop_success_requires_arm_clearance = True

    tabletop_hover_height_progress_rew_scale = 950.0
    tabletop_hover_target_rew_scale = 360.0
    tabletop_hover_goal_rew_scale = 1100.0
    tabletop_hover_stable_rew_scale = 1800.0
    tabletop_hover_linear_penalty_scale = 780.0
    tabletop_hover_grasp_loss_penalty_scale = 2600.0
    tabletop_hover_under_height_penalty_scale = 1600.0
    tabletop_hover_post_latch_speed_penalty_scale = 1800.0
    tabletop_post_success_hold_rew_scale = 8200.0
    tabletop_post_success_unstable_penalty_scale = 7800.0
    tabletop_post_success_grasp_loss_penalty_scale = 6200.0
    tabletop_post_success_speed_penalty_scale = 1700.0

    action_penalty_scale = 0.005
    arm_target_delta_penalty_scale = 0.008
    drop_penalty = 42.0


@configclass
class Revo2DynamicTabletopTransportClearanceWarmupTeacherEnvCfg(
    Revo2DynamicTabletopTransportHoldRampTeacherEnvCfg
):
    """Transport continuation with soft table-arm clearance shaping.

    This keeps the original HoldRamp success contract alive while nudging the
    arm links away from the tabletop.  A later strict stage can re-enable the
    hard clearance gate after grasp and hover behavior recover.
    """

    reference_name = "revo2_dynamic_tabletop_transport_clearance_warmup_teacher"

    tabletop_arm_clearance_body_margins = (
        0.090,
        0.100,
        0.085,
        0.070,
        0.052,
        0.038,
        0.028,
        0.018,
    )
    tabletop_arm_clearance_margin = 0.030
    tabletop_arm_clearance_scale = 0.085
    tabletop_arm_clearance_max_penalty = 2.0
    tabletop_arm_clearance_penalty_scale = 650.0
    tabletop_arm_clearance_xy_padding = 0.10
    tabletop_success_requires_arm_clearance = False

    tabletop_arm_object_lift_gap_margin = 0.085
    tabletop_arm_object_lift_gap_penalty_scale = 260.0
    tabletop_lift_without_object_min_arm_progress = 0.08
    tabletop_lift_without_object_penalty_scale = 180.0
    tabletop_no_lift_after_grasp_penalty_scale = 180.0
    tabletop_no_lift_after_grasp_grace_steps = 12
    tabletop_no_lift_after_grasp_ramp_steps = 64


@configclass
class Revo2DynamicTabletopTransportSkillPreserveTeacherEnvCfg(
    Revo2DynamicTabletopTransportHoldRampTeacherEnvCfg
):
    """Transport continuation that keeps the learned grasp skill intact.

    The prior clearance warmup was still too disruptive for the 6-active Revo2
    policy.  This stage uses the original HoldRamp success contract with only a
    small table-clearance shaping term, so PPO can keep the existing lift/hold
    behavior while gradually avoiding arm-table contacts.
    """

    reference_name = "revo2_dynamic_tabletop_transport_skill_preserve_teacher"

    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_grasp_speed_curriculum_start_success = 0.015
    dynamic_grasp_speed_curriculum_full_success = 0.22
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00025

    stable_hold_rew_scale = 4600.0
    hold_progress_rew_scale = 6800.0
    success_bonus = 14000.0

    tabletop_hover_stable_rew_scale = 5000.0
    tabletop_hover_post_latch_speed_penalty_scale = 1350.0
    tabletop_post_success_hold_rew_scale = 7000.0
    tabletop_post_success_unstable_penalty_scale = 5700.0
    tabletop_post_success_grasp_loss_penalty_scale = 4000.0
    tabletop_post_success_speed_penalty_scale = 950.0

    tabletop_arm_clearance_body_margins = (
        0.055,
        0.065,
        0.055,
        0.044,
        0.032,
        0.022,
        0.016,
        0.010,
    )
    tabletop_arm_clearance_margin = 0.015
    tabletop_arm_clearance_scale = 0.160
    tabletop_arm_clearance_max_penalty = 0.75
    tabletop_arm_clearance_penalty_scale = 65.0
    tabletop_arm_clearance_xy_padding = 0.03
    tabletop_success_requires_arm_clearance = False


@configclass
class Revo2DynamicTabletopTransportSkillPreserveAffordanceTeacherEnvCfg(
    Revo2DynamicTabletopTransportSkillPreserveTeacherEnvCfg
):
    """Transport teacher ablation with positive/negative affordance-region reward."""

    reference_name = "revo2_dynamic_tabletop_transport_skill_preserve_affordance_teacher"
    tabletop_affordance_reward_enabled = True
    tabletop_affordance_reward_points_per_region = 96
    tabletop_affordance_distance_scale = 0.035
    tabletop_affordance_contact_distance = 0.046
    tabletop_affordance_positive_rew_scale = 260.0
    tabletop_affordance_negative_penalty_scale = 420.0
    tabletop_affordance_lift_rew_scale = 520.0


@configclass
class Revo2DynamicTabletopTransportFromScratchTeacherEnvCfg(
    Revo2DynamicTabletopTransportHoldRampTeacherEnvCfg
):
    """Transport teacher tuned for no-checkpoint training from the Franka home pose.

    The continuation-oriented SkillPreserve task keeps a learned policy alive,
    but from-scratch PPO was exploiting pregrasp/lift shaping without turning it
    into stable in-hand holds.  This variant slows the curricula down and gates
    lift/hover reward more tightly on current true-grasp evidence.
    """

    reference_name = "revo2_dynamic_tabletop_transport_from_scratch_teacher"

    reset_object_pos_noise = (0.040, 0.025, 0.002)
    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_start_count = 1
    tabletop_asset_curriculum_steps = 6_000_000
    tabletop_motion_modes = ("linear", "curved", "turntable")
    tabletop_motion_mode_curriculum = True
    tabletop_motion_mode_curriculum_start_count = 1
    tabletop_motion_mode_curriculum_steps = 6_000_000

    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.10
    dynamic_grasp_speed_curriculum_full_success = 0.32
    dynamic_grasp_speed_curriculum_ema_alpha = 0.025
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00020
    dynamic_grasp_speed_curriculum_allow_decrease = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.008, 0.085)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (-0.35, 0.35)
    dynamic_tabletop_heading_range = (-0.45, 0.45)

    tabletop_success_requires_hover_target = False
    tabletop_success_uses_grasp_seen = False
    tabletop_hover_latch_uses_grasp_seen = False
    tabletop_hover_reward_uses_grasp_seen = False
    tabletop_success_lift_height = 0.050
    dynamic_success_hold_steps = 10
    stable_object_palm_vel = 0.34

    dynamic_tabletop_pregrasp_xy_rew_scale = 9.0
    dynamic_tabletop_pregrasp_height_rew_scale = 11.0
    dynamic_tabletop_pregrasp_height_offset = 0.070
    dynamic_tabletop_pregrasp_height_scale = 0.050
    dynamic_tabletop_low_palm_penalty_scale = 54.0
    dynamic_tabletop_min_palm_height_offset = 0.026
    dynamic_tabletop_low_palm_height_scale = 0.032
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = True
    dynamic_tabletop_contact_pregrasp_gate_min = 0.35

    contact_rew_scale = 8.0
    true_grasp_rew_scale = 240.0
    opposition_rew_scale = 115.0
    catch_progress_rew_scale = 320.0
    grasp_quality_rew_scale = 220.0
    lift_progress_rew_scale = 280.0
    lifted_true_grasp_rew_scale = 5600.0
    quality_lift_progress_rew_scale = 2100.0
    tabletop_stable_catch_rew_scale = 3200.0
    tabletop_stable_catch_min_lift_multiplier = 0.05
    stable_hold_rew_scale = 9000.0
    hold_progress_rew_scale = 14500.0
    success_bonus = 32000.0

    lift_reward_uses_grasp_quality_gate = True
    lift_reward_min_grasp_quality_multiplier = 0.05
    lift_reward_uses_opposition_gate = True
    lift_reward_min_opposition_multiplier = 0.05
    quality_lift_progress_uses_opposition_gate = True
    quality_lift_progress_min_opposition_multiplier = 0.08
    true_grasp_opposition_mode = "contact"

    tabletop_hover_height_delta = 0.065
    tabletop_hover_latch_lift_progress = 0.22
    tabletop_hover_xy_distance_scale = 0.18
    tabletop_hover_z_distance_scale = 0.055
    tabletop_hover_object_speed_scale = 0.20
    tabletop_hover_ang_speed_scale = 6.5
    tabletop_hover_success_xy_tolerance = 0.22
    tabletop_hover_success_z_tolerance = 0.065
    tabletop_hover_success_object_speed = 0.26
    tabletop_hover_target_rew_scale = 850.0
    tabletop_hover_height_progress_rew_scale = 1200.0
    tabletop_hover_goal_rew_scale = 4200.0
    tabletop_hover_stable_rew_scale = 5200.0
    tabletop_hover_linear_penalty_scale = 720.0
    tabletop_hover_overshoot_penalty_scale = 980.0
    tabletop_hover_z_vel_penalty_scale = 1450.0
    tabletop_hover_vel_penalty_scale = 760.0
    tabletop_hover_target_drift_penalty_scale = 1500.0
    tabletop_hover_grasp_loss_penalty_scale = 3600.0
    tabletop_hover_under_height_penalty_scale = 1500.0
    tabletop_hover_post_latch_speed_penalty_scale = 1800.0
    tabletop_hover_post_latch_action_penalty_scale = 0.045
    tabletop_hover_post_latch_target_delta_penalty_scale = 0.070

    tabletop_post_success_hold_rew_scale = 10500.0
    tabletop_post_success_unstable_penalty_scale = 9500.0
    tabletop_post_success_grasp_loss_penalty_scale = 8500.0
    tabletop_post_success_under_height_penalty_scale = 1800.0
    tabletop_post_success_speed_penalty_scale = 2300.0
    tabletop_post_success_action_penalty_scale = 0.060
    tabletop_post_success_target_delta_penalty_scale = 0.090

    tabletop_grasped_palm_lift_rew_scale = 850.0
    tabletop_grasped_palm_lift_height = 0.045
    tabletop_grasped_palm_lift_scale = 0.040
    tabletop_grasped_arm_lift_rew_scale = 950.0
    tabletop_arm_lift_reward_object_margin = 0.030
    tabletop_arm_object_lift_gap_margin = 0.030
    tabletop_arm_object_lift_gap_penalty_scale = 5200.0
    tabletop_lift_action_prior_rew_scale = 70.0
    tabletop_lift_action_prior_gate_min = 0.08
    tabletop_lift_without_object_min_arm_progress = 0.040
    tabletop_lift_without_object_penalty_scale = 3600.0
    tabletop_no_lift_after_grasp_penalty_scale = 1800.0
    tabletop_no_lift_after_grasp_grace_steps = 6
    tabletop_no_lift_after_grasp_ramp_steps = 38
    tabletop_no_lift_after_grasp_max_penalty = 6.0
    palm_only_lift_dist = 0.11
    palm_only_lift_penalty_scale = 1250.0
    scoop_lift_penalty_scale = 1050.0

    tabletop_arm_clearance_body_margins = (
        0.070,
        0.080,
        0.068,
        0.054,
        0.040,
        0.028,
        0.020,
        0.012,
    )
    tabletop_arm_clearance_margin = 0.020
    tabletop_arm_clearance_scale = 0.110
    tabletop_arm_clearance_max_penalty = 1.0
    tabletop_arm_clearance_penalty_scale = 120.0
    tabletop_arm_clearance_xy_padding = 0.045
    tabletop_success_requires_arm_clearance = False

    action_penalty_scale = 0.004
    arm_target_delta_penalty_scale = 0.006
    drop_penalty = 42.0


@configclass
class Revo2DynamicTabletopTransportFromScratchStrictOppositionTeacherEnvCfg(
    Revo2DynamicTabletopTransportFromScratchTeacherEnvCfg
):
    """From-scratch transport teacher that makes thumb-opposition a required skill.

    The first from-scratch hard-gate run learned to collect non-thumb contact
    reward while leaving thumb/opposing-contact at zero.  This variant keeps the
    same slow curriculum, but changes reward shaping so opposition reward and
    true-grasp score require thumb proximity instead of raw non-thumb geometry.
    """

    reference_name = "revo2_dynamic_tabletop_transport_from_scratch_strict_opposition_teacher"

    true_grasp_opposition_mode = "dot"
    opposition_reward_uses_weighted_score = True
    thumb_contact_reward_weight = 0.88
    thumb_true_grasp_score_weight = 0.90
    grasp_quality_finger_count_weight = 0.08
    grasp_quality_non_thumb_weight = 0.17
    grasp_quality_thumb_weight = 0.35
    grasp_quality_opposition_weight = 0.40

    contact_rew_scale = 3.0
    true_grasp_rew_scale = 320.0
    opposition_rew_scale = 260.0
    grasp_quality_rew_scale = 360.0
    catch_progress_rew_scale = 260.0
    lift_progress_rew_scale = 180.0
    quality_lift_progress_rew_scale = 2800.0
    lifted_true_grasp_rew_scale = 7600.0
    tabletop_stable_catch_rew_scale = 4200.0
    stable_hold_rew_scale = 10800.0
    hold_progress_rew_scale = 16500.0
    success_bonus = 36000.0

    lift_reward_uses_grasp_quality_gate = True
    lift_reward_min_grasp_quality_multiplier = 0.02
    lift_reward_uses_opposition_gate = True
    lift_reward_min_opposition_multiplier = 0.0
    quality_lift_progress_uses_opposition_gate = True
    quality_lift_progress_min_opposition_multiplier = 0.0

    tabletop_grasped_palm_lift_rew_scale = 1100.0
    tabletop_grasped_arm_lift_rew_scale = 1250.0
    tabletop_lift_action_prior_rew_scale = 45.0
    tabletop_lift_without_object_penalty_scale = 5200.0
    tabletop_no_lift_after_grasp_penalty_scale = 2400.0
    palm_only_lift_penalty_scale = 1650.0
    scoop_lift_penalty_scale = 1450.0


@configclass
class Revo2DynamicTabletopRollingCompatTeacherEnvCfg(
    Revo2DynamicTabletopRollingStrongRewardTeacherEnvCfg
):
    """6-active rolling curriculum that can continue from the 13-action/76-obs tabletop baseline."""

    reference_name = "revo2_dynamic_tabletop_rolling_assets_6active_compat_teacher"
    observation_space = 76
    tabletop_asset_obs_enabled = False
    tabletop_hover_target_obs_enabled = False

    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_start_count = 1
    tabletop_asset_curriculum_steps = 3_000_000
    tabletop_motion_modes = ("free",)
    tabletop_motion_mode_curriculum = False

    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.12
    dynamic_grasp_speed_curriculum_full_success = 0.45
    dynamic_grasp_speed_curriculum_ema_alpha = 0.03
    dynamic_grasp_speed_curriculum_alpha_rise = 0.0012
    dynamic_grasp_speed_curriculum_allow_decrease = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.04, 0.18)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (-1.4, 1.4)
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE

    reference_hand_fractions = (0.55, 0.82, 0.62, 0.62, 0.62, 0.62)
    scripted_action_prior_enabled = True
    lift_arm_delta = V325_VERIFIED_LIFT_ARM_DELTA
    lift_action_prior = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    tabletop_arm_lift_progress_baseline_pos = V327_PREGRASP_ARM_POS
    scripted_action_prior_residual_scale = 0.24
    scripted_tabletop_pregrasp_prior_enabled = True
    scripted_tabletop_pregrasp_arm_pos = V327_PREGRASP_ARM_POS
    scripted_tabletop_pregrasp_prior_start_step = 0
    scripted_tabletop_pregrasp_prior_ramp_steps = 150
    scripted_tabletop_lift_target_prior_enabled = True
    scripted_tabletop_lift_target_arm_delta = V325_VERIFIED_LIFT_ARM_DELTA
    scripted_tabletop_lift_target_prior_ramp_steps = 60
    scripted_action_prior_hand_start_step = 72
    scripted_action_prior_hand_ramp_steps = 110
    scripted_action_prior_hand_action = 1.0
    scripted_action_prior_lift_start_step = 88
    scripted_action_prior_lift_steps = 280
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 0

    palm_reach_rew_scale = 24.0
    fingertip_reach_rew_scale = 24.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 40.0
    dynamic_tabletop_pregrasp_height_rew_scale = 44.0
    dynamic_tabletop_pregrasp_height_offset = 0.080
    dynamic_tabletop_pregrasp_height_scale = 0.055
    dynamic_tabletop_low_palm_penalty_scale = 46.0
    dynamic_tabletop_min_palm_height_offset = 0.020
    dynamic_tabletop_contact_pregrasp_gate_min = 0.45
    contact_rew_scale = 14.0
    thumb_contact_reward_weight = 0.55
    thumb_true_grasp_score_weight = 0.58
    true_grasp_rew_scale = 115.0
    true_grasp_opposition_mode = "contact"
    opposition_rew_scale = 46.0
    catch_progress_rew_scale = 150.0
    grasp_quality_rew_scale = 160.0
    lift_reward_uses_opposition_gate = False
    quality_lift_progress_uses_opposition_gate = False
    quality_lift_progress_rew_scale = 1000.0
    lifted_true_grasp_rew_scale = 2400.0
    lift_progress_rew_scale = 700.0

    dynamic_success_hold_steps = 8
    stable_hold_rew_scale = 3800.0
    hold_progress_rew_scale = 5800.0
    success_bonus = 12000.0
    tabletop_hover_height_delta = 0.080
    tabletop_hover_latch_uses_grasp_seen = True
    tabletop_hover_reward_uses_grasp_seen = True
    tabletop_success_uses_grasp_seen = True
    tabletop_hover_success_xy_tolerance = 0.20
    tabletop_hover_success_z_tolerance = 0.055
    tabletop_hover_success_object_speed = 0.26
    tabletop_hover_target_rew_scale = 1600.0
    tabletop_hover_height_progress_rew_scale = 1600.0
    tabletop_hover_goal_rew_scale = 6900.0
    tabletop_hover_stable_rew_scale = 7400.0
    tabletop_hover_linear_penalty_scale = 900.0
    tabletop_hover_grasp_loss_penalty_scale = 1800.0
    tabletop_post_success_hold_rew_scale = 6000.0
    tabletop_post_success_unstable_penalty_scale = 6200.0
    tabletop_post_success_grasp_loss_penalty_scale = 4200.0
    tabletop_post_success_speed_penalty_scale = 1000.0
    tabletop_grasped_arm_lift_rew_scale = 650.0
    tabletop_arm_lift_reward_object_margin = 0.16
    tabletop_arm_object_lift_gap_margin = 0.12
    tabletop_arm_object_lift_gap_penalty_scale = 300.0
    tabletop_lift_action_prior_rew_scale = 140.0
    tabletop_lift_action_prior_gate_min = 0.24
    tabletop_lift_without_object_min_arm_progress = 0.08
    tabletop_lift_without_object_penalty_scale = 140.0
    tabletop_no_lift_after_grasp_penalty_scale = 180.0
    tabletop_no_lift_after_grasp_grace_steps = 12
    tabletop_no_lift_after_grasp_ramp_steps = 70
    tabletop_no_lift_after_grasp_max_penalty = 4.0
    palm_only_lift_penalty_scale = 420.0
    scoop_lift_penalty_scale = 360.0


@configclass
class Revo2DynamicTabletopRollingSmallBallStage1TeacherEnvCfg(
    Revo2DynamicTabletopRollingCompatTeacherEnvCfg
):
    """Stage-1 6-active small-ball teacher warm-started from the static tabletop baseline.

    The compat prior changes the baseline policy distribution too much for the tiny
    ball.  This stage keeps the 76-dim observation/action contract, starts with a
    single static small ball, and lets the policy re-learn a stable pinch/lift
    before the rolling speed curriculum opens up.
    """

    reference_name = "revo2_dynamic_tabletop_rolling_small_ball_stage1_6active_teacher"
    tabletop_object_asset_specs = (TABLETOP_ROLLING_START_SPEC,)
    tabletop_asset_curriculum = False
    tabletop_asset_curriculum_start_count = 1
    reset_object_pos_noise = (0.030, 0.025, 0.001)

    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.08
    dynamic_grasp_speed_curriculum_full_success = 0.35
    dynamic_grasp_speed_curriculum_ema_alpha = 0.04
    dynamic_grasp_speed_curriculum_alpha_rise = 0.0010
    dynamic_grasp_speed_curriculum_allow_decrease = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.02, 0.11)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (-0.8, 0.8)
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE

    scripted_action_prior_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    tabletop_lift_action_prior_rew_scale = 0.0

    reference_hand_fractions = (0.58, 0.86, 0.64, 0.64, 0.64, 0.64)
    dynamic_tabletop_pregrasp_height_offset = 0.105
    dynamic_tabletop_pregrasp_height_scale = 0.065
    dynamic_tabletop_low_palm_penalty_scale = 25.0
    dynamic_tabletop_min_palm_height_offset = 0.012
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = False
    thumb_contact_reward_weight = 0.55
    thumb_true_grasp_score_weight = 0.58
    true_grasp_opposition_mode = "contact"

    tabletop_success_requires_hover_target = False
    tabletop_success_lift_height = 0.040
    dynamic_success_hold_steps = 6
    stable_object_palm_vel = 0.40
    tabletop_hover_height_delta = 0.070
    tabletop_hover_latch_lift_progress = 0.58

    contact_rew_scale = 18.0
    true_grasp_rew_scale = 120.0
    opposition_rew_scale = 42.0
    catch_progress_rew_scale = 160.0
    grasp_quality_rew_scale = 170.0
    quality_lift_progress_rew_scale = 1350.0
    lifted_true_grasp_rew_scale = 3000.0
    lift_progress_rew_scale = 950.0
    tabletop_stable_catch_rew_scale = 1150.0
    tabletop_stable_catch_min_lift_multiplier = 0.05
    stable_hold_rew_scale = 4300.0
    hold_progress_rew_scale = 6400.0
    success_bonus = 15000.0
    tabletop_hover_target_rew_scale = 0.0
    tabletop_hover_height_progress_rew_scale = 0.0
    tabletop_hover_goal_rew_scale = 0.0
    tabletop_hover_stable_rew_scale = 0.0
    tabletop_hover_linear_penalty_scale = 0.0
    tabletop_hover_overshoot_penalty_scale = 0.0
    tabletop_hover_z_vel_penalty_scale = 0.0
    tabletop_hover_vel_penalty_scale = 0.0
    tabletop_hover_target_drift_penalty_scale = 0.0
    tabletop_hover_grasp_loss_penalty_scale = 0.0
    tabletop_hover_under_height_penalty_scale = 0.0
    tabletop_hover_post_latch_speed_penalty_scale = 0.0
    tabletop_hover_post_latch_action_penalty_scale = 0.0
    tabletop_hover_post_latch_target_delta_penalty_scale = 0.0

    tabletop_grasped_palm_lift_rew_scale = 450.0
    tabletop_grasped_arm_lift_rew_scale = 520.0
    tabletop_arm_lift_reward_object_margin = 0.14
    tabletop_arm_object_lift_gap_margin = 0.12
    tabletop_arm_object_lift_gap_penalty_scale = 180.0
    tabletop_lift_without_object_min_arm_progress = 0.16
    tabletop_lift_without_object_penalty_scale = 80.0
    tabletop_no_lift_after_grasp_penalty_scale = 90.0
    tabletop_no_lift_after_grasp_grace_steps = 20
    tabletop_no_lift_after_grasp_ramp_steps = 80
    tabletop_no_lift_after_grasp_max_penalty = 3.0
    palm_only_lift_penalty_scale = 260.0
    scoop_lift_penalty_scale = 220.0

    action_penalty_scale = 0.003
    arm_target_delta_penalty_scale = 0.004


@configclass
class Revo2DynamicTabletopRollingSmallBallStage2TeacherEnvCfg(
    Revo2DynamicTabletopRollingSmallBallStage1TeacherEnvCfg
):
    """Stage-2 small-ball teacher: convert transient lifts into stable hovered holds."""

    reference_name = "revo2_dynamic_tabletop_rolling_small_ball_stage2_stable_hover_teacher"

    dynamic_grasp_speed_curriculum_metric = "success"
    dynamic_grasp_speed_curriculum_start_success = 0.04
    dynamic_grasp_speed_curriculum_full_success = 0.28
    dynamic_grasp_speed_curriculum_ema_alpha = 0.03
    dynamic_grasp_speed_curriculum_alpha_rise = 0.0008
    dynamic_tabletop_initial_speed_range = (0.015, 0.095)
    dynamic_tabletop_initial_yaw_rate_range = (-0.7, 0.7)

    tabletop_success_requires_hover_target = True
    tabletop_success_uses_grasp_seen = False
    tabletop_hover_reward_uses_grasp_seen = False
    tabletop_hover_latch_uses_grasp_seen = False
    tabletop_hover_success_requires_xy = True
    tabletop_success_lift_height = 0.040
    tabletop_hover_height_delta = 0.075
    tabletop_hover_latch_lift_progress = 0.46
    tabletop_hover_xy_distance_scale = 0.15
    tabletop_hover_z_distance_scale = 0.045
    tabletop_hover_object_speed_scale = 0.16
    tabletop_hover_ang_speed_scale = 6.0
    tabletop_hover_success_xy_tolerance = 0.15
    tabletop_hover_success_z_tolerance = 0.050
    tabletop_hover_success_object_speed = 0.18
    stable_object_palm_vel = 0.22
    dynamic_success_hold_steps = 24

    tabletop_hover_target_rew_scale = 1500.0
    tabletop_hover_height_progress_rew_scale = 900.0
    tabletop_hover_goal_rew_scale = 5200.0
    tabletop_hover_stable_rew_scale = 6200.0
    tabletop_hover_linear_penalty_scale = 520.0
    tabletop_hover_overshoot_penalty_scale = 620.0
    tabletop_hover_z_vel_penalty_scale = 780.0
    tabletop_hover_vel_penalty_scale = 480.0
    tabletop_hover_target_drift_penalty_scale = 900.0
    tabletop_hover_grasp_loss_penalty_scale = 1800.0
    tabletop_hover_under_height_penalty_scale = 650.0
    tabletop_hover_post_latch_speed_penalty_scale = 900.0
    tabletop_hover_post_latch_action_penalty_scale = 0.018
    tabletop_hover_post_latch_target_delta_penalty_scale = 0.028

    stable_hold_rew_scale = 5200.0
    hold_progress_rew_scale = 9000.0
    success_bonus = 22000.0
    tabletop_post_success_hold_rew_scale = 7500.0
    tabletop_post_success_unstable_penalty_scale = 10500.0
    tabletop_post_success_grasp_loss_penalty_scale = 7200.0
    tabletop_post_success_under_height_penalty_scale = 3200.0
    tabletop_post_success_speed_penalty_scale = 2100.0
    tabletop_post_success_action_penalty_scale = 0.045
    tabletop_post_success_target_delta_penalty_scale = 0.065

    quality_lift_progress_rew_scale = 1200.0
    lifted_true_grasp_rew_scale = 2400.0
    lift_progress_rew_scale = 750.0
    tabletop_stable_catch_rew_scale = 1050.0
    tabletop_grasped_palm_lift_rew_scale = 520.0
    tabletop_grasped_arm_lift_rew_scale = 580.0
    tabletop_arm_object_lift_gap_penalty_scale = 220.0
    tabletop_lift_without_object_penalty_scale = 120.0
    tabletop_no_lift_after_grasp_penalty_scale = 120.0
    palm_only_lift_penalty_scale = 320.0
    scoop_lift_penalty_scale = 280.0


@configclass
class Revo2DynamicTabletopRollingSmallBallStage2MotionRampTeacherEnvCfg(
    Revo2DynamicTabletopRollingSmallBallStage1TeacherEnvCfg
):
    """Gentle Stage-2 curriculum that preserves Stage-1 lift success while adding motion."""

    reference_name = "revo2_dynamic_tabletop_rolling_small_ball_stage2_motion_ramp_teacher"

    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.05
    dynamic_grasp_speed_curriculum_full_success = 0.22
    dynamic_grasp_speed_curriculum_ema_alpha = 0.035
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00045
    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_tabletop_initial_speed_range = (0.010, 0.085)
    dynamic_tabletop_initial_yaw_rate_range = (-0.55, 0.55)

    tabletop_success_requires_hover_target = False
    tabletop_success_lift_height = 0.042
    dynamic_success_hold_steps = 8
    stable_object_palm_vel = 0.34

    tabletop_hover_height_delta = 0.075
    tabletop_hover_latch_lift_progress = 0.44
    tabletop_hover_success_object_speed = 0.30
    tabletop_hover_target_rew_scale = 180.0
    tabletop_hover_height_progress_rew_scale = 220.0
    tabletop_hover_goal_rew_scale = 360.0
    tabletop_hover_stable_rew_scale = 420.0
    tabletop_hover_linear_penalty_scale = 80.0
    tabletop_hover_overshoot_penalty_scale = 120.0
    tabletop_hover_z_vel_penalty_scale = 120.0
    tabletop_hover_vel_penalty_scale = 80.0
    tabletop_hover_target_drift_penalty_scale = 120.0
    tabletop_hover_grasp_loss_penalty_scale = 260.0
    tabletop_hover_under_height_penalty_scale = 160.0
    tabletop_hover_post_latch_speed_penalty_scale = 180.0

    quality_lift_progress_rew_scale = 1250.0
    lifted_true_grasp_rew_scale = 3300.0
    lift_progress_rew_scale = 900.0
    tabletop_stable_catch_rew_scale = 1250.0
    stable_hold_rew_scale = 4700.0
    hold_progress_rew_scale = 7000.0
    success_bonus = 17000.0

    tabletop_grasped_palm_lift_rew_scale = 520.0
    tabletop_grasped_arm_lift_rew_scale = 620.0
    tabletop_arm_object_lift_gap_penalty_scale = 240.0
    tabletop_lift_without_object_penalty_scale = 110.0
    tabletop_no_lift_after_grasp_penalty_scale = 140.0
    palm_only_lift_penalty_scale = 300.0
    scoop_lift_penalty_scale = 250.0


@configclass
class Revo2DynamicTabletopRollingAssetsStage2MotionRampTeacherEnvCfg(
    Revo2DynamicTabletopRollingSmallBallStage2MotionRampTeacherEnvCfg
):
    """Transfer the successful small-ball motion-ramp teacher to the full rolling asset set.

    This intentionally keeps the 13-action / 76-observation real Revo2 contract
    used by the small-ball policy, then opens the ball/cylinder/bottle/cone
    asset curriculum gradually.
    """

    reference_name = "revo2_dynamic_tabletop_rolling_assets_stage2_motion_ramp_teacher"
    tabletop_object_asset_specs = TABLETOP_ROLLING_OBJECT_SPECS
    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_start_count = 1
    tabletop_asset_curriculum_steps = 2_400_000
    reset_object_pos_noise = (0.050, 0.032, 0.0015)

    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.04
    dynamic_grasp_speed_curriculum_full_success = 0.20
    dynamic_grasp_speed_curriculum_ema_alpha = 0.035
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00035
    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_tabletop_initial_speed_range = (0.010, 0.075)
    dynamic_tabletop_initial_yaw_rate_range = (-0.45, 0.45)
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE

    tabletop_success_requires_hover_target = False
    tabletop_success_lift_height = 0.040
    dynamic_success_hold_steps = 8
    stable_object_palm_vel = 0.36

    contact_rew_scale = 20.0
    true_grasp_rew_scale = 145.0
    grasp_quality_rew_scale = 190.0
    quality_lift_progress_rew_scale = 1400.0
    lifted_true_grasp_rew_scale = 3600.0
    lift_progress_rew_scale = 980.0
    tabletop_stable_catch_rew_scale = 1350.0
    stable_hold_rew_scale = 5200.0
    hold_progress_rew_scale = 7600.0
    success_bonus = 19000.0

    tabletop_grasped_palm_lift_rew_scale = 620.0
    tabletop_grasped_arm_lift_rew_scale = 720.0
    tabletop_arm_object_lift_gap_penalty_scale = 260.0
    tabletop_lift_without_object_penalty_scale = 130.0
    tabletop_no_lift_after_grasp_penalty_scale = 150.0
    palm_only_lift_penalty_scale = 320.0
    scoop_lift_penalty_scale = 260.0


@configclass
class Revo2DynamicTabletopRollingAssetsFastSpeedTeacherEnvCfg(
    Revo2DynamicTabletopRollingAssetsStage2MotionRampTeacherEnvCfg
):
    """Fast free-rolling tabletop continuation targeting 0.10-0.40 m/s.

    The curriculum starts from the validated Stage-2 rolling speed range and
    gradually opens the full high-speed target range.  This keeps the existing
    6-active-DOF Revo2 grasp behavior alive while increasing object motion.
    """

    reference_name = "revo2_dynamic_tabletop_rolling_assets_fast_speed_teacher"
    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_start_count = 5
    tabletop_asset_curriculum_steps = 1

    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.20
    dynamic_grasp_speed_curriculum_full_success = 0.55
    dynamic_grasp_speed_curriculum_ema_alpha = 0.035
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00025
    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_tabletop_start_speed_range = (0.010, 0.075)
    dynamic_tabletop_initial_speed_range = (0.100, 0.400)
    dynamic_tabletop_start_yaw_rate_range = (-0.45, 0.45)
    dynamic_tabletop_initial_yaw_rate_range = (-2.40, 2.40)
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE

    dynamic_tabletop_pregrasp_lead_time = 0.36
    dynamic_tabletop_pregrasp_ahead_distance = 0.10
    dynamic_tabletop_pregrasp_ready_distance = 0.20
    reset_object_pos_noise = (0.040, 0.030, 0.0015)

    stable_object_palm_vel = 0.42
    tabletop_success_lift_height = 0.040
    dynamic_success_hold_steps = 8

    contact_rew_scale = 18.0
    true_grasp_rew_scale = 150.0
    grasp_quality_rew_scale = 210.0
    quality_lift_progress_rew_scale = 1500.0
    lifted_true_grasp_rew_scale = 3800.0
    lift_progress_rew_scale = 1100.0
    tabletop_stable_catch_rew_scale = 1650.0
    stable_hold_rew_scale = 7800.0
    hold_progress_rew_scale = 10800.0
    success_bonus = 32000.0

    dynamic_tabletop_pregrasp_xy_rew_scale = 420.0
    tabletop_grasped_palm_lift_rew_scale = 700.0
    tabletop_grasped_arm_lift_rew_scale = 820.0
    tabletop_arm_object_lift_gap_penalty_scale = 280.0
    tabletop_lift_without_object_penalty_scale = 150.0
    tabletop_no_lift_after_grasp_penalty_scale = 180.0
    palm_only_lift_penalty_scale = 340.0
    scoop_lift_penalty_scale = 280.0

    tabletop_post_success_stability_latch_enabled = True
    tabletop_post_success_arm_target_lock_enabled = True
    tabletop_post_success_arm_target_lock_blend = 1.0
    tabletop_post_success_hand_target_lock_enabled = True
    tabletop_post_success_hand_target_lock_blend = 1.0
    tabletop_post_success_hand_close_fraction = 0.08
    tabletop_post_success_hold_rew_scale = 9800.0
    tabletop_post_success_unstable_penalty_scale = 5200.0
    tabletop_post_success_grasp_loss_penalty_scale = 3600.0
    tabletop_post_success_under_height_penalty_scale = 1600.0
    tabletop_post_success_speed_penalty_scale = 950.0
    tabletop_post_success_action_penalty_scale = 0.035
    tabletop_post_success_target_delta_penalty_scale = 0.060
    tabletop_post_success_arm_joint_vel_penalty_scale = 80.0
    tabletop_post_success_arm_target_drift_penalty_scale = 1500.0
    tabletop_post_success_arm_target_drift_tolerance = 0.09
    tabletop_post_success_arm_target_drift_scale = 0.28
    tabletop_post_success_palm_drift_penalty_scale = 2400.0
    tabletop_post_success_palm_drift_tolerance = 0.035
    tabletop_post_success_palm_drift_scale = 0.085


@configclass
class Revo2DynamicTabletopRollingAssetsFastSpeedPostHoldTeacherEnvCfg(
    Revo2DynamicTabletopRollingAssetsFastSpeedTeacherEnvCfg
):
    """Fast rolling continuation that trains the state after a successful lift."""

    reference_name = "revo2_dynamic_tabletop_rolling_assets_fast_speed_posthold_teacher"

    # The base fast-speed task terminates as soon as its eight-step success
    # streak completes.  Keep the episode alive so PPO receives the actual
    # post-grasp contact, lift, and stability returns used by video evaluation.
    terminate_on_success = False
    episode_length_s = 8.0

    # Train against the same load-bearing contact contract used by strict
    # evaluation: thumb + two non-thumb fingers in physical opposition.
    strict_success_enabled = True
    strict_reward_enabled = True
    strict_success_contact_distance = 0.010
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    strict_reward_contact_score_scale = 0.025

    tabletop_post_success_hand_lock_uses_actual_joint_pos = True

    tabletop_post_success_hold_rew_scale = 20000.0
    tabletop_post_success_unstable_penalty_scale = 16000.0
    tabletop_post_success_grasp_loss_penalty_scale = 15000.0
    tabletop_post_success_under_height_penalty_scale = 2200.0
    tabletop_post_success_speed_penalty_scale = 1400.0
    tabletop_post_success_action_penalty_scale = 0.050
    tabletop_post_success_target_delta_penalty_scale = 0.080
    tabletop_post_success_arm_joint_vel_penalty_scale = 100.0
    tabletop_post_success_arm_target_drift_penalty_scale = 1800.0
    tabletop_post_success_palm_drift_penalty_scale = 2800.0


@configclass
class Revo2DynamicTabletopRollingAssetsFastSpeedStrictAcquisitionTeacherEnvCfg(
    Revo2DynamicTabletopRollingAssetsFastSpeedPostHoldTeacherEnvCfg
):
    """Post-hold task with dense shaping for physical thumb-pair acquisition."""

    reference_name = "revo2_dynamic_tabletop_rolling_assets_fast_speed_strict_acquisition_teacher"

    # PostHold enables strict reward selection, but its inherited dense strict
    # approach/touch scales are zero. Use the validated Revo2 falling-hand
    # contact shaping so elongated rolling assets cannot earn lift returns from
    # a loose or transient enclosure alone.
    strict_approach_score_scale = 0.030
    strict_touch_score_scale = 0.008
    strict_approach_rew_scale = 12.0
    strict_multifinger_approach_rew_scale = 8.0
    strict_opposition_approach_rew_scale = 180.0
    strict_touch_reward_requires_thumb_pair = True
    strict_touch_reward_uses_opposition_product = True
    strict_touch_reward_opposition_min_multiplier = 0.02
    strict_touch_rew_scale = 1000.0
    strict_opposition_touch_rew_scale = 3600.0


@configclass
class Revo2DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedTeacherEnvCfg(
    Revo2DynamicTabletopRollingAssetsFastSpeedPostHoldTeacherEnvCfg
):
    """Fast rolling teacher with privileged shape, size, and affordance asset state."""

    reference_name = "revo2_dynamic_tabletop_rolling_assets_fast_speed_asset_privileged_teacher"
    observation_space = 86
    tabletop_asset_obs_enabled = True


@configclass
class Revo2DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedTargetHandLockTeacherEnvCfg(
    Revo2DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedTeacherEnvCfg
):
    """Keep the successful commanded hand target so post-hold contact stays loaded."""

    reference_name = "revo2_dynamic_tabletop_rolling_assets_fast_speed_asset_privileged_target_hand_lock_teacher"
    tabletop_post_success_hand_lock_uses_actual_joint_pos = False


@configclass
class Revo2UnifiedRollingBenchmarkTeacherEnvCfg(
    _UnifiedRollingRewardContract,
    Revo2DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedTargetHandLockTeacherEnvCfg
):
    """Revo2 adapter for the shared multi-shape rolling benchmark."""

    reference_name = "revo2_unified_rolling_multishape_v1_teacher"
    benchmark_protocol = UNIFIED_ROLLING_BENCHMARK_NAME
    robot_cfg: ArticulationCfg = _v699_revo2_robot_cfg(FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS)

    # Revo2 uses the Franka link origins plus explicit palm/fingertip samples.
    # Finger-link geometry is covered by the simulator collisions and the
    # shared fingertip samples rather than guessed per-link origin radii.
    tabletop_arm_clearance_body_names = (
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_link8",
        "panda_hand",
    )
    tabletop_arm_clearance_body_margins = ()

    observation_space = 86
    tabletop_object_asset_specs = UNIFIED_ROLLING_OBJECT_SPECS
    tabletop_asset_set_enabled = True
    tabletop_asset_obs_enabled = True
    tabletop_asset_sampling_weights = None
    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_mode = "dynamic_speed"
    tabletop_asset_curriculum_start_count = 1
    tabletop_asset_curriculum_steps = UNIFIED_ROLLING_ASSET_CURRICULUM_STEPS
    tabletop_asset_curriculum_override_alpha = None

    object_start_pos = UNIFIED_ROLLING_START_POS
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=object_start_pos,
    )
    object_shape = str(TABLETOP_ROLLING_START_SPEC["proxy_shape"])
    object_radius = float(TABLETOP_ROLLING_START_SPEC["radius"])
    object_size = tuple(TABLETOP_ROLLING_START_SPEC["size"])
    reset_object_pos_noise = UNIFIED_ROLLING_RESET_OBJECT_POS_NOISE

    tabletop_motion_modes = ("free",)
    tabletop_motion_mode_curriculum = False
    dynamic_tabletop_persistent_motion = False
    dynamic_tabletop_bounce_at_workspace = False
    dynamic_tabletop_release_motion_on_contact = False
    dynamic_tabletop_randomize_yaw = True
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = UNIFIED_ROLLING_CURRICULUM_METRIC
    dynamic_grasp_speed_curriculum_start_success = UNIFIED_ROLLING_CURRICULUM_START_SUCCESS
    dynamic_grasp_speed_curriculum_full_success = UNIFIED_ROLLING_CURRICULUM_FULL_SUCCESS
    dynamic_grasp_speed_curriculum_ema_alpha = UNIFIED_ROLLING_CURRICULUM_EMA_ALPHA
    dynamic_grasp_speed_curriculum_alpha_rise = UNIFIED_ROLLING_CURRICULUM_ALPHA_RISE
    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_grasp_speed_curriculum_steps = UNIFIED_ROLLING_SPEED_CURRICULUM_STEPS
    dynamic_grasp_speed_curriculum_override_alpha = None
    dynamic_tabletop_speed_alpha_sample_enabled = False
    dynamic_tabletop_start_speed_range = UNIFIED_ROLLING_START_SPEED_RANGE
    dynamic_tabletop_initial_speed_range = UNIFIED_ROLLING_TARGET_SPEED_RANGE
    dynamic_tabletop_start_yaw_rate_range = UNIFIED_ROLLING_START_YAW_RATE_RANGE
    dynamic_tabletop_initial_yaw_rate_range = UNIFIED_ROLLING_TARGET_YAW_RATE_RANGE
    dynamic_tabletop_pregrasp_lead_time = 0.36
    dynamic_tabletop_pregrasp_ahead_distance = 0.10
    dynamic_tabletop_pregrasp_ready_distance = 0.20

    strict_success_enabled = True
    strict_reward_enabled = True
    strict_success_contact_distance = UNIFIED_ROLLING_STRICT_CONTACT_DISTANCE
    strict_success_min_finger_contacts = UNIFIED_ROLLING_STRICT_MIN_FINGER_CONTACTS
    strict_success_min_non_thumb_contacts = UNIFIED_ROLLING_STRICT_MIN_NON_THUMB_CONTACTS
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    contact_distance = UNIFIED_ROLLING_CONTACT_DISTANCE
    contact_score_scale = UNIFIED_ROLLING_CONTACT_SCORE_SCALE
    min_finger_contacts = UNIFIED_ROLLING_STRICT_MIN_FINGER_CONTACTS
    min_non_thumb_contacts = UNIFIED_ROLLING_STRICT_MIN_NON_THUMB_CONTACTS
    opposition_cos_threshold = 0.0
    tabletop_success_requires_hover_target = False
    tabletop_success_lift_height = UNIFIED_ROLLING_SUCCESS_LIFT_HEIGHT
    dynamic_success_hold_steps = UNIFIED_ROLLING_SUCCESS_HOLD_STEPS
    stable_object_palm_vel = UNIFIED_ROLLING_STABLE_OBJECT_PALM_VEL
    tabletop_hover_height_delta = UNIFIED_ROLLING_HOVER_HEIGHT_DELTA
    tabletop_hover_latch_lift_progress = UNIFIED_ROLLING_HOVER_LATCH_LIFT_PROGRESS
    tabletop_hover_xy_distance_scale = UNIFIED_ROLLING_HOVER_XY_DISTANCE_SCALE
    tabletop_hover_z_distance_scale = UNIFIED_ROLLING_HOVER_Z_DISTANCE_SCALE
    tabletop_hover_object_speed_scale = UNIFIED_ROLLING_HOVER_OBJECT_SPEED_SCALE
    tabletop_hover_ang_speed_scale = UNIFIED_ROLLING_HOVER_ANG_SPEED_SCALE
    tabletop_hover_success_requires_xy = True
    tabletop_hover_success_xy_tolerance = UNIFIED_ROLLING_HOVER_SUCCESS_XY_TOLERANCE
    tabletop_hover_success_z_tolerance = UNIFIED_ROLLING_HOVER_SUCCESS_Z_TOLERANCE
    tabletop_hover_success_object_speed = UNIFIED_ROLLING_HOVER_SUCCESS_OBJECT_SPEED
    tabletop_success_requires_arm_clearance = True
    tabletop_terminate_on_arm_clearance_violation = False

    terminate_on_success = False
    episode_length_s = UNIFIED_ROLLING_EPISODE_LENGTH_S
    tabletop_post_success_stability_latch_enabled = True
    tabletop_post_success_arm_target_lock_enabled = True
    tabletop_post_success_arm_target_lock_blend = 1.0
    tabletop_post_success_hand_target_lock_enabled = True
    tabletop_post_success_hand_target_lock_blend = 1.0
    tabletop_post_success_hand_lock_uses_actual_joint_pos = False
    tabletop_post_success_hand_close_fraction = 0.08
    affordance_label_mode = "tabletop_rolling_assets"


@configclass
class Revo2UnifiedRollingStage1TeacherEnvCfg(Revo2UnifiedRollingBenchmarkTeacherEnvCfg):
    """Home-to-pregrasp bootstrap stage for the shared rolling curriculum."""

    reference_name = "revo2_unified_rolling_multishape_v1_stage1_teacher"
    contact_rew_scale = 80.0
    dynamic_tabletop_pregrasp_height_rew_scale = 44.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 420.0
    fingertip_reach_rew_scale = 24.0
    grasp_quality_rew_scale = 500.0
    opposition_rew_scale = 200.0
    palm_reach_rew_scale = 24.0
    pregrasp_rew_scale = 4.0
    strict_approach_rew_scale = 80.0
    strict_multifinger_approach_rew_scale = 160.0
    strict_opposition_approach_rew_scale = 300.0
    strict_opposition_touch_rew_scale = 1200.0
    strict_touch_rew_scale = 400.0
    tabletop_non_thumb_without_thumb_penalty_scale = 300.0
    true_grasp_rew_scale = 500.0
    strict_touch_score_scale = 0.008


@configclass
class Revo2UnifiedRollingStage2HoldTeacherEnvCfg(
    _UnifiedRollingGraspHoldStage2Contract,
    Revo2UnifiedRollingBenchmarkTeacherEnvCfg,
):
    """Revo2 continuation stage for a sustained strict grasp before lift."""

    reference_name = "revo2_unified_rolling_multishape_v1_stage2_hold_teacher"


@configclass
class Revo2UnifiedRollingStage3TeacherEnvCfg(
    _UnifiedRollingLiftHoldStage3Contract,
    Revo2UnifiedRollingBenchmarkTeacherEnvCfg,
):
    """Revo2 adapter for the shared strict lift-and-hold continuation stage."""

    reference_name = "revo2_unified_rolling_multishape_v1_stage3_lift_hold_teacher"


@configclass
class Revo2DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedHardReplayTeacherEnvCfg(
    Revo2DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedTeacherEnvCfg
):
    """Training-only replay distribution focused on the three hardest rolling assets."""

    reference_name = "revo2_dynamic_tabletop_rolling_assets_fast_speed_asset_privileged_hard_replay_teacher"
    # Asset order: apple, can, bottle, cone, pill bottle. Evaluation keeps the
    # parent task's uniform distribution.
    tabletop_asset_sampling_weights = (0.05, 0.25, 0.35, 0.30, 0.05)


@configclass
class Revo2DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedStrictAcquisitionHardReplayTeacherEnvCfg(
    Revo2DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedTargetHandLockTeacherEnvCfg
):
    """Asset-aware hard replay with dense physical thumb-opposition shaping."""

    reference_name = (
        "revo2_dynamic_tabletop_rolling_assets_fast_speed_asset_privileged_"
        "strict_acquisition_hard_replay_teacher"
    )

    # Keep the final post-success target lock while replaying bottle, cone, and
    # cylinder failures more often. The scales match the existing validated
    # strict-acquisition task; only their missing asset-privileged combination
    # is introduced here.
    tabletop_asset_sampling_weights = (0.05, 0.25, 0.35, 0.30, 0.05)
    strict_approach_score_scale = 0.030
    strict_touch_score_scale = 0.008
    strict_approach_rew_scale = 12.0
    strict_multifinger_approach_rew_scale = 8.0
    strict_opposition_approach_rew_scale = 180.0
    strict_touch_reward_requires_thumb_pair = True
    strict_touch_reward_uses_opposition_product = True
    strict_touch_reward_opposition_min_multiplier = 0.02
    strict_touch_rew_scale = 1000.0
    strict_opposition_touch_rew_scale = 3600.0


@configclass
class Revo2DynamicTabletopRollingAssetsStage2StableHoldTeacherEnvCfg(
    Revo2DynamicTabletopRollingAssetsStage2MotionRampTeacherEnvCfg
):
    """Stage-2 continuation that preserves the learned 6-active grasp and rewards stable lift holds.

    This branch intentionally does not add scripted lift or hand residual priors.
    It is meant to continue from the best Stage-2 checkpoint and test whether the
    observed late-training drift is mainly reward selection/optimizer pressure
    rather than a missing scripted prior.
    """

    reference_name = "revo2_dynamic_tabletop_rolling_assets_stage2_stable_hold_teacher"

    tabletop_success_requires_hover_target = False
    tabletop_success_lift_height = 0.042
    dynamic_success_hold_steps = 12
    stable_object_palm_vel = 0.30

    contact_rew_scale = 16.0
    true_grasp_rew_scale = 120.0
    grasp_quality_rew_scale = 165.0
    quality_lift_progress_rew_scale = 1650.0
    lifted_true_grasp_rew_scale = 4300.0
    lift_progress_rew_scale = 1180.0
    tabletop_stable_catch_rew_scale = 1750.0
    stable_hold_rew_scale = 7000.0
    hold_progress_rew_scale = 9800.0
    success_bonus = 24000.0

    tabletop_hover_height_delta = 0.078
    tabletop_hover_latch_lift_progress = 0.40
    tabletop_hover_success_object_speed = 0.24
    tabletop_hover_target_rew_scale = 300.0
    tabletop_hover_height_progress_rew_scale = 420.0
    tabletop_hover_goal_rew_scale = 760.0
    tabletop_hover_stable_rew_scale = 960.0
    tabletop_hover_linear_penalty_scale = 100.0
    tabletop_hover_overshoot_penalty_scale = 180.0
    tabletop_hover_z_vel_penalty_scale = 180.0
    tabletop_hover_vel_penalty_scale = 125.0
    tabletop_hover_target_drift_penalty_scale = 180.0
    tabletop_hover_grasp_loss_penalty_scale = 520.0
    tabletop_hover_under_height_penalty_scale = 280.0
    tabletop_hover_post_latch_speed_penalty_scale = 320.0

    tabletop_post_success_hold_rew_scale = 7800.0
    tabletop_post_success_unstable_penalty_scale = 8200.0
    tabletop_post_success_grasp_loss_penalty_scale = 5200.0
    tabletop_post_success_under_height_penalty_scale = 2400.0
    tabletop_post_success_speed_penalty_scale = 1800.0
    tabletop_post_success_action_penalty_scale = 0.055
    tabletop_post_success_target_delta_penalty_scale = 0.080

    tabletop_grasped_palm_lift_rew_scale = 860.0
    tabletop_grasped_arm_lift_rew_scale = 1020.0
    tabletop_arm_lift_reward_object_margin = 0.13
    tabletop_arm_object_lift_gap_margin = 0.11
    tabletop_arm_object_lift_gap_penalty_scale = 360.0
    tabletop_lift_without_object_min_arm_progress = 0.11
    tabletop_lift_without_object_penalty_scale = 180.0
    tabletop_no_lift_min_progress = 0.18
    tabletop_no_lift_after_grasp_penalty_scale = 320.0
    tabletop_no_lift_after_grasp_grace_steps = 12
    tabletop_no_lift_after_grasp_ramp_steps = 58
    tabletop_no_lift_after_grasp_max_penalty = 4.0
    palm_only_lift_penalty_scale = 360.0
    scoop_lift_penalty_scale = 300.0


@configclass
class Revo2DynamicTabletopRollingAssetsStage3ResidualLiftHoverTeacherEnvCfg(
    Revo2DynamicTabletopRollingAssetsStage2MotionRampTeacherEnvCfg
):
    """Full rolling asset continuation that pushes grasped objects to a stable hover.

    The Stage-2 full-asset run reaches reliable true grasps but later checkpoints
    drift toward a contact/no-lift optimum.  This variant keeps the same
    6-active Revo2 action contract and asset curriculum, then adds only a weak
    residual lift prior after grasp memory plus stronger lift/hover rewards.
    """

    reference_name = "revo2_dynamic_tabletop_rolling_assets_stage3_residual_lift_hover_teacher"

    tabletop_success_requires_hover_target = True
    tabletop_success_uses_grasp_seen = False
    tabletop_hover_reward_uses_grasp_seen = False
    tabletop_hover_latch_uses_grasp_seen = True
    tabletop_hover_success_requires_xy = True
    tabletop_success_lift_height = 0.044
    dynamic_success_hold_steps = 14
    stable_object_palm_vel = 0.26
    tabletop_hover_height_delta = 0.080
    tabletop_hover_latch_lift_progress = 0.34
    tabletop_hover_xy_distance_scale = 0.14
    tabletop_hover_z_distance_scale = 0.045
    tabletop_hover_object_speed_scale = 0.16
    tabletop_hover_ang_speed_scale = 6.0
    tabletop_hover_success_xy_tolerance = 0.15
    tabletop_hover_success_z_tolerance = 0.055
    tabletop_hover_success_object_speed = 0.22

    scripted_action_prior_enabled = True
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.22
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_action_prior_hand_start_step = 9999
    scripted_action_prior_hand_ramp_steps = 0
    scripted_action_prior_hand_action = 0.0
    scripted_action_prior_lift_start_step = 72
    scripted_action_prior_lift_steps = 320
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 2
    scripted_tabletop_hand_grasp_memory_prior_enabled = True
    scripted_tabletop_hand_grasp_memory_prior_start_step = 64
    scripted_tabletop_hand_grasp_memory_prior_steps = 320
    scripted_tabletop_hand_grasp_memory_min_steps = 1
    scripted_tabletop_hand_grasp_memory_action = 0.88
    scripted_tabletop_hand_grasp_memory_ramp_steps = 22

    contact_rew_scale = 12.0
    true_grasp_rew_scale = 115.0
    grasp_quality_rew_scale = 140.0
    quality_lift_progress_rew_scale = 2300.0
    lifted_true_grasp_rew_scale = 6200.0
    lift_progress_rew_scale = 1700.0
    tabletop_stable_catch_rew_scale = 950.0
    stable_hold_rew_scale = 7000.0
    hold_progress_rew_scale = 11000.0
    success_bonus = 28000.0

    tabletop_hover_target_rew_scale = 900.0
    tabletop_hover_height_progress_rew_scale = 900.0
    tabletop_hover_goal_rew_scale = 3000.0
    tabletop_hover_stable_rew_scale = 4200.0
    tabletop_hover_linear_penalty_scale = 360.0
    tabletop_hover_overshoot_penalty_scale = 480.0
    tabletop_hover_z_vel_penalty_scale = 520.0
    tabletop_hover_vel_penalty_scale = 320.0
    tabletop_hover_target_drift_penalty_scale = 520.0
    tabletop_hover_grasp_loss_penalty_scale = 2500.0
    tabletop_hover_under_height_penalty_scale = 1200.0
    tabletop_hover_post_latch_speed_penalty_scale = 720.0
    tabletop_hover_post_latch_action_penalty_scale = 0.022
    tabletop_hover_post_latch_target_delta_penalty_scale = 0.035

    tabletop_post_success_hold_rew_scale = 6500.0
    tabletop_post_success_unstable_penalty_scale = 9000.0
    tabletop_post_success_grasp_loss_penalty_scale = 6200.0
    tabletop_post_success_under_height_penalty_scale = 4200.0
    tabletop_post_success_speed_penalty_scale = 2200.0
    tabletop_post_success_action_penalty_scale = 0.04
    tabletop_post_success_target_delta_penalty_scale = 0.06

    tabletop_grasped_palm_lift_rew_scale = 1400.0
    tabletop_grasped_arm_lift_rew_scale = 1700.0
    tabletop_arm_lift_reward_object_margin = 0.13
    tabletop_arm_object_lift_gap_margin = 0.11
    tabletop_arm_object_lift_gap_penalty_scale = 900.0
    tabletop_lift_action_prior_rew_scale = 180.0
    tabletop_lift_action_prior_gate_min = 0.22
    tabletop_lift_without_object_min_arm_progress = 0.09
    tabletop_lift_without_object_penalty_scale = 360.0
    tabletop_no_lift_min_progress = 0.22
    tabletop_no_lift_after_grasp_penalty_scale = 720.0
    tabletop_no_lift_after_grasp_grace_steps = 8
    tabletop_no_lift_after_grasp_ramp_steps = 44
    tabletop_no_lift_after_grasp_max_penalty = 5.0
    palm_only_lift_penalty_scale = 380.0
    scoop_lift_penalty_scale = 320.0


@configclass
class Revo2DynamicTabletopRollingAssetsStage3ResidualLiftHoldTeacherEnvCfg(
    Revo2DynamicTabletopRollingAssetsStage2MotionRampTeacherEnvCfg
):
    """Conservative full-asset continuation for lifting after a verified grasp.

    Unlike the hover-gated variant, this keeps the Stage-2 success definition so
    the warm-start policy does not lose its reach/contact behavior.  It only
    adds a weak residual lift and hand-hold prior after grasp memory and makes
    the no-lift-after-grasp failure mode more expensive.
    """

    reference_name = "revo2_dynamic_tabletop_rolling_assets_stage3_residual_lift_hold_teacher"

    tabletop_success_requires_hover_target = False
    tabletop_success_lift_height = 0.042
    dynamic_success_hold_steps = 10
    stable_object_palm_vel = 0.32
    tabletop_hover_height_delta = 0.078
    tabletop_hover_latch_lift_progress = 0.40
    tabletop_hover_success_object_speed = 0.26

    scripted_action_prior_enabled = True
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.14
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_action_prior_hand_start_step = 9999
    scripted_action_prior_hand_ramp_steps = 0
    scripted_action_prior_hand_action = 0.0
    scripted_action_prior_lift_start_step = 78
    scripted_action_prior_lift_steps = 280
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 1
    scripted_tabletop_hand_grasp_memory_prior_enabled = True
    scripted_tabletop_hand_grasp_memory_prior_start_step = 64
    scripted_tabletop_hand_grasp_memory_prior_steps = 280
    scripted_tabletop_hand_grasp_memory_min_steps = 1
    scripted_tabletop_hand_grasp_memory_action = 0.84
    scripted_tabletop_hand_grasp_memory_ramp_steps = 28

    quality_lift_progress_rew_scale = 1600.0
    lifted_true_grasp_rew_scale = 4300.0
    lift_progress_rew_scale = 1180.0
    tabletop_stable_catch_rew_scale = 1450.0
    stable_hold_rew_scale = 6100.0
    hold_progress_rew_scale = 8600.0
    success_bonus = 22000.0

    tabletop_hover_target_rew_scale = 260.0
    tabletop_hover_height_progress_rew_scale = 360.0
    tabletop_hover_goal_rew_scale = 620.0
    tabletop_hover_stable_rew_scale = 760.0
    tabletop_hover_linear_penalty_scale = 90.0
    tabletop_hover_overshoot_penalty_scale = 140.0
    tabletop_hover_z_vel_penalty_scale = 140.0
    tabletop_hover_vel_penalty_scale = 95.0
    tabletop_hover_target_drift_penalty_scale = 140.0
    tabletop_hover_grasp_loss_penalty_scale = 360.0
    tabletop_hover_under_height_penalty_scale = 220.0
    tabletop_hover_post_latch_speed_penalty_scale = 240.0

    tabletop_grasped_palm_lift_rew_scale = 820.0
    tabletop_grasped_arm_lift_rew_scale = 1050.0
    tabletop_arm_lift_reward_object_margin = 0.13
    tabletop_arm_object_lift_gap_margin = 0.11
    tabletop_arm_object_lift_gap_penalty_scale = 420.0
    tabletop_lift_action_prior_rew_scale = 110.0
    tabletop_lift_action_prior_gate_min = 0.18
    tabletop_lift_without_object_min_arm_progress = 0.10
    tabletop_lift_without_object_penalty_scale = 220.0
    tabletop_no_lift_min_progress = 0.18
    tabletop_no_lift_after_grasp_penalty_scale = 380.0
    tabletop_no_lift_after_grasp_grace_steps = 12
    tabletop_no_lift_after_grasp_ramp_steps = 52
    tabletop_no_lift_after_grasp_max_penalty = 4.0
    palm_only_lift_penalty_scale = 340.0
    scoop_lift_penalty_scale = 280.0


@configclass
class Revo2DynamicTabletopRollingSmallBallStage2LiftGuideTeacherEnvCfg(
    Revo2DynamicTabletopRollingSmallBallStage2TeacherEnvCfg
):
    """Stage-2 lift-guide variant to escape the stable-contact/no-lift local optimum."""

    reference_name = "revo2_dynamic_tabletop_rolling_small_ball_stage2_liftguide_teacher"

    dynamic_success_hold_steps = 18
    dynamic_grasp_speed_curriculum_start_success = 0.03
    dynamic_grasp_speed_curriculum_full_success = 0.24

    quality_lift_progress_rew_scale = 2600.0
    lifted_true_grasp_rew_scale = 6200.0
    lift_progress_rew_scale = 1900.0
    tabletop_stable_catch_rew_scale = 650.0
    tabletop_stable_catch_min_lift_multiplier = 0.12
    stable_hold_rew_scale = 5200.0
    hold_progress_rew_scale = 8600.0
    success_bonus = 24000.0

    tabletop_grasped_palm_lift_rew_scale = 1600.0
    tabletop_grasped_arm_lift_rew_scale = 2600.0
    tabletop_grasped_palm_lift_height = 0.075
    tabletop_grasped_palm_lift_scale = 0.045
    tabletop_arm_lift_reward_object_margin = 0.08
    tabletop_arm_object_lift_gap_margin = 0.07
    tabletop_arm_object_lift_gap_penalty_scale = 850.0
    tabletop_lift_without_object_min_arm_progress = 0.10
    tabletop_lift_without_object_penalty_scale = 420.0
    tabletop_no_lift_min_progress = 0.22
    tabletop_no_lift_after_grasp_penalty_scale = 1050.0
    tabletop_no_lift_after_grasp_grace_steps = 8
    tabletop_no_lift_after_grasp_ramp_steps = 36
    tabletop_no_lift_after_grasp_max_penalty = 6.0

    tabletop_hover_latch_lift_progress = 0.38
    tabletop_hover_height_progress_rew_scale = 1700.0
    tabletop_hover_target_rew_scale = 1500.0
    tabletop_hover_goal_rew_scale = 4700.0
    tabletop_hover_stable_rew_scale = 5600.0
    tabletop_hover_linear_penalty_scale = 420.0
    tabletop_hover_overshoot_penalty_scale = 520.0
    tabletop_hover_z_vel_penalty_scale = 620.0
    tabletop_hover_vel_penalty_scale = 360.0
    tabletop_hover_target_drift_penalty_scale = 780.0
    tabletop_hover_post_latch_speed_penalty_scale = 700.0


@configclass
class Revo2DynamicTabletopRollingSmallBallStage3PriorLiftTeacherEnvCfg(
    Revo2DynamicTabletopRollingSmallBallStage2LiftGuideTeacherEnvCfg
):
    """Small-ball stage that adds a weak lift action prior after verified contact.

    Stage-2 learned reliable hand contact but collapsed to a no-lift local optimum.
    This keeps the real 13D / 6-active Revo2 action contract and only injects a
    modest residual prior so the teacher sees stable lifted trajectories.
    """

    reference_name = "revo2_dynamic_tabletop_rolling_small_ball_stage3_prior_lift_teacher"

    scripted_action_prior_enabled = True
    scripted_action_prior_residual_scale = 0.16
    lift_arm_delta = V325_VERIFIED_LIFT_ARM_DELTA
    lift_action_prior = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    tabletop_arm_lift_progress_baseline_pos = V327_PREGRASP_ARM_POS
    scripted_tabletop_pregrasp_prior_enabled = True
    scripted_tabletop_pregrasp_arm_pos = V327_PREGRASP_ARM_POS
    scripted_tabletop_pregrasp_prior_start_step = 0
    scripted_tabletop_pregrasp_prior_ramp_steps = 145
    scripted_tabletop_lift_target_prior_enabled = True
    scripted_tabletop_lift_target_arm_delta = V325_VERIFIED_LIFT_ARM_DELTA
    scripted_tabletop_lift_target_prior_ramp_steps = 70
    scripted_action_prior_hand_start_step = 56
    scripted_action_prior_hand_ramp_steps = 100
    scripted_action_prior_hand_action = 1.0
    scripted_action_prior_lift_start_step = 96
    scripted_action_prior_lift_steps = 260
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 3

    dynamic_success_hold_steps = 12
    tabletop_hover_latch_lift_progress = 0.28
    tabletop_hover_success_object_speed = 0.22
    stable_object_palm_vel = 0.24

    quality_lift_progress_rew_scale = 2100.0
    lifted_true_grasp_rew_scale = 4600.0
    lift_progress_rew_scale = 1350.0
    tabletop_stable_catch_rew_scale = 900.0
    stable_hold_rew_scale = 5000.0
    hold_progress_rew_scale = 7600.0
    success_bonus = 18000.0

    tabletop_grasped_palm_lift_rew_scale = 900.0
    tabletop_grasped_arm_lift_rew_scale = 1500.0
    tabletop_arm_lift_reward_object_margin = 0.12
    tabletop_arm_object_lift_gap_margin = 0.11
    tabletop_arm_object_lift_gap_penalty_scale = 640.0
    tabletop_lift_action_prior_rew_scale = 120.0
    tabletop_lift_action_prior_gate_min = 0.16
    tabletop_lift_without_object_min_arm_progress = 0.08
    tabletop_lift_without_object_penalty_scale = 260.0
    tabletop_no_lift_min_progress = 0.18
    tabletop_no_lift_after_grasp_penalty_scale = 420.0
    tabletop_no_lift_after_grasp_grace_steps = 10
    tabletop_no_lift_after_grasp_ramp_steps = 50
    tabletop_no_lift_after_grasp_max_penalty = 5.0
    palm_only_lift_penalty_scale = 360.0
    scoop_lift_penalty_scale = 320.0


@configclass
class Revo2DynamicTabletopRollingSmallBallStage3ResidualLiftOnlyTeacherEnvCfg(
    Revo2DynamicTabletopRollingSmallBallStage2LiftGuideTeacherEnvCfg
):
    """Small-ball stage that preserves the learned reach/grasp policy and adds lift only after contact."""

    reference_name = "revo2_dynamic_tabletop_rolling_small_ball_stage3_residual_lift_only_teacher"

    scripted_action_prior_enabled = True
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.18
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_action_prior_hand_start_step = 9999
    scripted_action_prior_hand_ramp_steps = 0
    scripted_action_prior_hand_action = 0.0
    scripted_action_prior_lift_start_step = 72
    scripted_action_prior_lift_steps = 300
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 0

    dynamic_success_hold_steps = 14
    tabletop_hover_latch_lift_progress = 0.32
    stable_object_palm_vel = 0.24
    tabletop_lift_action_prior_rew_scale = 160.0
    tabletop_lift_action_prior_gate_min = 0.20
    tabletop_no_lift_after_grasp_penalty_scale = 620.0
    tabletop_no_lift_after_grasp_grace_steps = 8
    tabletop_no_lift_after_grasp_ramp_steps = 42
    tabletop_grasped_arm_lift_rew_scale = 1900.0


@configclass
class Revo2DynamicTabletopRollingSmallBallStage3ResidualLiftHoldTeacherEnvCfg(
    Revo2DynamicTabletopRollingSmallBallStage3ResidualLiftOnlyTeacherEnvCfg
):
    """Lift-only stage plus a grasp-memory hand hold prior for the 6-active Revo2 hand."""

    reference_name = "revo2_dynamic_tabletop_rolling_small_ball_stage3_residual_lift_hold_teacher"

    scripted_action_prior_active_residual_scale = 0.25
    scripted_tabletop_hand_grasp_memory_prior_enabled = True
    scripted_tabletop_hand_grasp_memory_prior_start_step = 72
    scripted_tabletop_hand_grasp_memory_prior_steps = 300
    scripted_tabletop_hand_grasp_memory_min_steps = 1
    scripted_tabletop_hand_grasp_memory_action = 0.85
    scripted_tabletop_hand_grasp_memory_ramp_steps = 24

    true_grasp_rew_scale = 240.0
    lifted_true_grasp_rew_scale = 7600.0
    stable_hold_rew_scale = 6800.0
    hold_progress_rew_scale = 10400.0
    tabletop_hover_grasp_loss_penalty_scale = 3600.0
    tabletop_post_success_grasp_loss_penalty_scale = 6400.0
    tabletop_lift_without_object_penalty_scale = 620.0
    tabletop_arm_object_lift_gap_penalty_scale = 1150.0
    tabletop_grasped_arm_lift_rew_scale = 1600.0
    tabletop_grasped_palm_lift_rew_scale = 2100.0


@configclass
class Revo2DynamicTabletopRollingSmallBallStage3ResidualLiftHoldStrictGateTeacherEnvCfg(
    Revo2DynamicTabletopRollingSmallBallStage3ResidualLiftHoldTeacherEnvCfg
):
    """Hand hold prior with lift gated by a sustained true-grasp streak."""

    reference_name = "revo2_dynamic_tabletop_rolling_small_ball_stage3_residual_lift_hold_strict_gate_teacher"

    scripted_action_prior_active_residual_scale = 0.32
    scripted_action_prior_lift_start_step = 96
    scripted_action_prior_lift_grasp_memory_min_steps = 8
    scripted_action_prior_lift_memory_requires_streak = True
    scripted_tabletop_hand_grasp_memory_prior_start_step = 60
    scripted_tabletop_hand_grasp_memory_prior_steps = 320
    scripted_tabletop_hand_grasp_memory_action = 0.95
    scripted_tabletop_hand_grasp_memory_ramp_steps = 18

    tabletop_lift_action_prior_gate_min = 0.35
    tabletop_no_lift_after_grasp_grace_steps = 16
    tabletop_no_lift_after_grasp_ramp_steps = 54
    tabletop_no_lift_after_grasp_penalty_scale = 760.0
    tabletop_lift_without_object_penalty_scale = 900.0
    tabletop_arm_object_lift_gap_penalty_scale = 1500.0
    tabletop_grasped_arm_lift_rew_scale = 1350.0
    tabletop_grasped_palm_lift_rew_scale = 2400.0
    lifted_true_grasp_rew_scale = 9000.0


@configclass
class Revo2DynamicTabletopRollingSmallBallFastDirectionTeacherEnvCfg(
    Revo2DynamicTabletopRollingSmallBallStage3ResidualLiftHoldStrictGateTeacherEnvCfg
):
    """Small-ball continuation with faster, full-heading tabletop rolling commands."""

    reference_name = "revo2_dynamic_tabletop_rolling_small_ball_fast_direction_teacher"
    reset_object_pos_noise = (0.055, 0.045, 0.0015)
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "success"
    dynamic_grasp_speed_curriculum_start_success = 0.06
    dynamic_grasp_speed_curriculum_full_success = 0.24
    dynamic_grasp_speed_curriculum_ema_alpha = 0.035
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00065
    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_tabletop_start_speed_range = (0.015, 0.050)
    dynamic_tabletop_initial_speed_range = (0.045, 0.180)
    dynamic_tabletop_start_yaw_rate_range = (-0.30, 0.30)
    dynamic_tabletop_initial_yaw_rate_range = (-1.80, 1.80)
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE
    stable_object_palm_vel = 0.30
    tabletop_hover_success_object_speed = 0.28
    dynamic_success_hold_steps = 12


@configclass
class Revo2DynamicTabletopRollingSmallBallLowSpeedEvalEnvCfg(
    Revo2DynamicTabletopRollingSmallBallFastDirectionTeacherEnvCfg
):
    """Low-speed held-out eval for small-ball rolling."""

    reference_name = "revo2_dynamic_tabletop_rolling_small_ball_low_speed_eval"
    dynamic_grasp_speed_curriculum = False
    dynamic_tabletop_initial_speed_range = (0.025, 0.085)
    dynamic_tabletop_initial_yaw_rate_range = (-0.75, 0.75)
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE


@configclass
class Revo2DynamicTabletopRollingSmallBallHighSpeedEvalEnvCfg(
    Revo2DynamicTabletopRollingSmallBallFastDirectionTeacherEnvCfg
):
    """High-speed held-out eval for small-ball rolling."""

    reference_name = "revo2_dynamic_tabletop_rolling_small_ball_high_speed_eval"
    dynamic_grasp_speed_curriculum = False
    dynamic_tabletop_initial_speed_range = (0.130, 0.260)
    dynamic_tabletop_initial_yaw_rate_range = (-2.40, 2.40)
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE


@configclass
class Revo2DynamicTabletopRollingSmallBallStage3TargetLiftOnlyTeacherEnvCfg(
    Revo2DynamicTabletopRollingSmallBallStage2LiftGuideTeacherEnvCfg
):
    """Small-ball stage with a target-space lift prior that stays off before contact."""

    reference_name = "revo2_dynamic_tabletop_rolling_small_ball_stage3_target_lift_only_teacher"

    scripted_action_prior_enabled = True
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.18
    scripted_tabletop_pregrasp_prior_enabled = True
    scripted_tabletop_pregrasp_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    scripted_tabletop_pregrasp_prior_start_step = 0
    scripted_tabletop_pregrasp_prior_ramp_steps = 1
    scripted_tabletop_lift_target_prior_enabled = True
    scripted_tabletop_lift_target_arm_pos = (
        -1.1476110110549927,
        1.0978932338027954,
        1.3373354463043213,
        -1.2401911652679443,
        0.5210036532783508,
        1.5129269686050415,
        -0.825389264251709,
    )
    scripted_tabletop_lift_target_prior_ramp_steps = 46
    scripted_action_prior_hand_start_step = 9999
    scripted_action_prior_hand_ramp_steps = 0
    scripted_action_prior_hand_action = 0.0
    scripted_action_prior_lift_start_step = 72
    scripted_action_prior_lift_steps = 300
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 0

    dynamic_success_hold_steps = 14
    tabletop_hover_latch_lift_progress = 0.32
    stable_object_palm_vel = 0.24
    tabletop_lift_action_prior_rew_scale = 160.0
    tabletop_lift_action_prior_gate_min = 0.20
    tabletop_no_lift_after_grasp_penalty_scale = 620.0
    tabletop_no_lift_after_grasp_grace_steps = 8
    tabletop_no_lift_after_grasp_ramp_steps = 42
    tabletop_grasped_arm_lift_rew_scale = 1900.0


@configclass
class Revo2DynamicTabletopTransportCompatTeacherEnvCfg(
    Revo2DynamicTabletopTransportStrongRewardTeacherEnvCfg
):
    """6-active transport curriculum that can continue from the 13-action/76-obs tabletop baseline."""

    reference_name = "revo2_dynamic_tabletop_transport_assets_6active_compat_teacher"
    observation_space = 76
    tabletop_asset_obs_enabled = False
    tabletop_hover_target_obs_enabled = False

    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_start_count = 2
    tabletop_asset_curriculum_steps = 3_500_000
    tabletop_motion_modes = ("linear", "curved", "turntable")
    tabletop_motion_mode_curriculum = True
    tabletop_motion_mode_curriculum_start_count = 1
    tabletop_motion_mode_curriculum_steps = 3_000_000

    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.10
    dynamic_grasp_speed_curriculum_full_success = 0.42
    dynamic_grasp_speed_curriculum_ema_alpha = 0.03
    dynamic_grasp_speed_curriculum_alpha_rise = 0.0010
    dynamic_grasp_speed_curriculum_allow_decrease = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.025, 0.14)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (-0.8, 0.8)
    dynamic_tabletop_heading_range = (-0.85, 0.85)

    lift_arm_delta = V325_VERIFIED_LIFT_ARM_DELTA
    lift_action_prior = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    tabletop_arm_lift_progress_baseline_pos = V327_PREGRASP_ARM_POS
    scripted_action_prior_residual_scale = 0.24
    scripted_tabletop_pregrasp_prior_enabled = True
    scripted_tabletop_pregrasp_arm_pos = V327_PREGRASP_ARM_POS
    scripted_tabletop_pregrasp_prior_start_step = 0
    scripted_tabletop_pregrasp_prior_ramp_steps = 145
    scripted_tabletop_lift_target_prior_enabled = True
    scripted_tabletop_lift_target_arm_delta = V325_VERIFIED_LIFT_ARM_DELTA
    scripted_tabletop_lift_target_prior_ramp_steps = 80
    scripted_action_prior_hand_start_step = 72
    scripted_action_prior_hand_ramp_steps = 110
    scripted_action_prior_hand_action = 1.0
    scripted_action_prior_lift_start_step = 104
    scripted_action_prior_lift_steps = 280
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 0

    dynamic_success_hold_steps = 10
    stable_hold_rew_scale = 1250.0
    hold_progress_rew_scale = 2600.0
    success_bonus = 6500.0
    tabletop_hover_height_delta = 0.085
    tabletop_hover_latch_uses_grasp_seen = True
    tabletop_hover_reward_uses_grasp_seen = True
    tabletop_success_uses_grasp_seen = True
    tabletop_hover_success_xy_tolerance = 0.22
    tabletop_hover_success_z_tolerance = 0.060
    tabletop_hover_success_object_speed = 0.28
    tabletop_hover_target_rew_scale = 1100.0
    tabletop_hover_height_progress_rew_scale = 1000.0
    tabletop_hover_goal_rew_scale = 4800.0
    tabletop_hover_stable_rew_scale = 5200.0
    tabletop_hover_linear_penalty_scale = 1050.0
    tabletop_hover_grasp_loss_penalty_scale = 2100.0
    tabletop_post_success_hold_rew_scale = 4500.0
    tabletop_post_success_unstable_penalty_scale = 6200.0
    tabletop_post_success_grasp_loss_penalty_scale = 4200.0
    tabletop_post_success_speed_penalty_scale = 1200.0
    lift_reward_uses_opposition_gate = False
    quality_lift_progress_uses_opposition_gate = False
    true_grasp_opposition_mode = "contact"
    thumb_contact_reward_weight = 0.68
    thumb_true_grasp_score_weight = 0.70
    quality_lift_progress_rew_scale = 700.0
    lifted_true_grasp_rew_scale = 1800.0
    lift_progress_rew_scale = 480.0
    tabletop_grasped_arm_lift_rew_scale = 480.0
    tabletop_arm_lift_reward_object_margin = 0.16
    tabletop_arm_object_lift_gap_margin = 0.12
    tabletop_arm_object_lift_gap_penalty_scale = 250.0
    tabletop_lift_action_prior_rew_scale = 90.0
    tabletop_lift_action_prior_gate_min = 0.20
    tabletop_lift_without_object_min_arm_progress = 0.08
    tabletop_lift_without_object_penalty_scale = 100.0
    tabletop_no_lift_after_grasp_penalty_scale = 100.0
    tabletop_no_lift_after_grasp_grace_steps = 12
    tabletop_no_lift_after_grasp_ramp_steps = 70
    tabletop_no_lift_after_grasp_max_penalty = 4.0
    palm_only_lift_penalty_scale = 430.0
    scoop_lift_penalty_scale = 360.0


@configclass
class Revo2DynamicTabletopRollingStrongRewardTargetObsTeacherEnvCfg(
    Revo2DynamicTabletopRollingStrongRewardTeacherEnvCfg
):
    """Rolling tabletop teacher with privileged hover-target hold state."""

    reference_name = "revo2_dynamic_tabletop_rolling_assets_strong_reward_target_obs_teacher"
    observation_space = 91
    tabletop_hover_target_obs_enabled = True


@configclass
class Revo2DynamicTabletopRollingRecoveryTargetObsTeacherEnvCfg(
    Revo2DynamicTabletopRollingStrongRewardTargetObsTeacherEnvCfg
):
    """Stage-1 rolling recovery teacher for learning Revo2 reach/opposition from home pose."""

    reference_name = "revo2_dynamic_tabletop_rolling_recovery_target_obs_teacher"
    episode_length_s = 8.0
    tabletop_object_asset_specs = (TABLETOP_ROLLING_START_SPEC,)
    tabletop_asset_curriculum = False
    tabletop_asset_curriculum_start_count = 1
    tabletop_asset_curriculum_steps = 10_000_000
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=(0.58, -0.05, TABLETOP_ROLLING_START_Z),
    )
    object_shape = str(TABLETOP_ROLLING_START_SPEC["proxy_shape"])
    object_radius = float(TABLETOP_ROLLING_START_SPEC["radius"])
    object_size = tuple(TABLETOP_ROLLING_START_SPEC["size"])
    object_start_pos = (0.58, -0.05, TABLETOP_ROLLING_START_Z)
    reset_object_pos_noise = (0.020, 0.020, 0.001)
    dynamic_tabletop_initial_speed_range = (0.0, 0.10)
    dynamic_tabletop_initial_yaw_rate_range = (-0.6, 0.6)
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "stable_catch"
    dynamic_grasp_speed_curriculum_start_success = 0.18
    dynamic_grasp_speed_curriculum_full_success = 0.45
    dynamic_grasp_speed_curriculum_ema_alpha = 0.03
    dynamic_grasp_speed_curriculum_alpha_rise = 0.0015
    dynamic_grasp_speed_curriculum_allow_decrease = False
    dynamic_grasp_speed_curriculum_steps = 10_000_000

    reference_hand_fractions = (0.45, 0.75, 0.50, 0.50, 0.50, 0.50)
    scripted_action_prior_enabled = True
    scripted_action_prior_residual_scale = 0.22
    scripted_tabletop_pregrasp_prior_enabled = True
    scripted_tabletop_pregrasp_arm_pos = V327_PREGRASP_ARM_POS
    scripted_tabletop_pregrasp_prior_start_step = 0
    scripted_tabletop_pregrasp_prior_ramp_steps = 150
    scripted_tabletop_lift_target_prior_enabled = True
    scripted_tabletop_lift_target_arm_delta = V327_LIFT_ARM_DELTA
    scripted_tabletop_lift_target_prior_ramp_steps = 110
    scripted_action_prior_hand_start_step = 72
    scripted_action_prior_hand_ramp_steps = 110
    scripted_action_prior_hand_action = 1.0
    scripted_action_prior_lift_start_step = 96
    scripted_action_prior_lift_steps = 220
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = False

    palm_reach_rew_scale = 22.0
    fingertip_reach_rew_scale = 22.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 38.0
    dynamic_tabletop_pregrasp_height_rew_scale = 42.0
    dynamic_tabletop_pregrasp_height_offset = 0.080
    dynamic_tabletop_pregrasp_height_scale = 0.055
    dynamic_tabletop_low_palm_penalty_scale = 45.0
    dynamic_tabletop_min_palm_height_offset = 0.020
    dynamic_tabletop_contact_pregrasp_gate_min = 0.45
    contact_rew_scale = 14.0
    true_grasp_rew_scale = 80.0
    true_grasp_opposition_mode = "contact"
    opposition_rew_scale = 45.0
    catch_progress_rew_scale = 110.0
    grasp_quality_rew_scale = 120.0
    quality_lift_progress_rew_scale = 700.0
    lifted_true_grasp_rew_scale = 1800.0
    lift_progress_rew_scale = 420.0
    tabletop_stable_catch_rew_scale = 1350.0
    stable_hold_rew_scale = 3600.0
    hold_progress_rew_scale = 5600.0
    success_bonus = 12000.0
    dynamic_success_hold_steps = 8
    tabletop_hover_height_delta = 0.075
    tabletop_hover_latch_lift_progress = 0.62
    tabletop_hover_success_z_tolerance = 0.055
    tabletop_hover_success_object_speed = 0.24
    tabletop_hover_target_rew_scale = 1600.0
    tabletop_hover_height_progress_rew_scale = 1600.0
    tabletop_hover_goal_rew_scale = 6800.0
    tabletop_hover_stable_rew_scale = 7200.0
    tabletop_hover_linear_penalty_scale = 260.0
    tabletop_hover_overshoot_penalty_scale = 390.0
    tabletop_hover_z_vel_penalty_scale = 500.0
    tabletop_hover_vel_penalty_scale = 280.0
    tabletop_hover_target_drift_penalty_scale = 620.0
    tabletop_hover_grasp_loss_penalty_scale = 850.0
    tabletop_hover_under_height_penalty_scale = 360.0
    tabletop_hover_post_latch_speed_penalty_scale = 500.0
    tabletop_no_lift_after_grasp_penalty_scale = 85.0
    tabletop_grasped_palm_lift_rew_scale = 0.0
    tabletop_grasped_arm_lift_rew_scale = 240.0
    tabletop_lift_action_prior_rew_scale = 70.0
    tabletop_lift_without_object_penalty_scale = 130.0
    tabletop_post_success_hold_rew_scale = 5000.0
    tabletop_post_success_unstable_penalty_scale = 7500.0
    tabletop_post_success_grasp_loss_penalty_scale = 5000.0
    tabletop_post_success_speed_penalty_scale = 1400.0
    action_penalty_scale = 0.003
    arm_target_delta_penalty_scale = 0.004


@configclass
class Revo2DynamicTabletopTransportStrongRewardTargetObsTeacherEnvCfg(
    Revo2DynamicTabletopTransportStrongRewardTeacherEnvCfg
):
    """Transport tabletop teacher with privileged hover-target hold state."""

    reference_name = "revo2_dynamic_tabletop_transport_assets_strong_reward_target_obs_teacher"
    observation_space = 91
    tabletop_hover_target_obs_enabled = True


@configclass
class Revo2DynamicTabletopRollingRichPriorTargetObsTeacherEnvCfg(
    Revo2DynamicTabletopRollingStrongRewardTargetObsTeacherEnvCfg
):
    """91-observation rolling teacher with asset/hover privileges and a stronger lift prior."""

    reference_name = "revo2_dynamic_tabletop_rolling_assets_rich_prior_target_obs_teacher"
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=(0.58, -0.05, TABLETOP_ROLLING_START_Z),
    )
    object_shape = str(TABLETOP_ROLLING_START_SPEC["proxy_shape"])
    object_radius = float(TABLETOP_ROLLING_START_SPEC["radius"])
    object_size = tuple(TABLETOP_ROLLING_START_SPEC["size"])
    object_start_pos = (0.58, -0.05, TABLETOP_ROLLING_START_Z)
    reset_object_pos_noise = (0.020, 0.020, 0.001)
    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_start_count = 1
    tabletop_asset_curriculum_steps = 3_000_000
    tabletop_motion_modes = ("free",)
    tabletop_motion_mode_curriculum = False

    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.03
    dynamic_grasp_speed_curriculum_full_success = 0.30
    dynamic_grasp_speed_curriculum_ema_alpha = 0.03
    dynamic_grasp_speed_curriculum_alpha_rise = 0.0012
    dynamic_grasp_speed_curriculum_allow_decrease = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.02, 0.16)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (-1.2, 1.2)
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE

    reference_hand_fractions = (0.62, 0.90, 0.70, 0.70, 0.70, 0.68)
    scripted_action_prior_enabled = True
    lift_arm_delta = V325_VERIFIED_LIFT_ARM_DELTA
    lift_action_prior = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    tabletop_arm_lift_progress_baseline_pos = V327_PREGRASP_ARM_POS
    scripted_action_prior_residual_scale = 0.10
    scripted_tabletop_pregrasp_prior_enabled = True
    scripted_tabletop_pregrasp_arm_pos = V327_PREGRASP_ARM_POS
    scripted_tabletop_pregrasp_prior_start_step = 0
    scripted_tabletop_pregrasp_prior_ramp_steps = 150
    scripted_tabletop_lift_target_prior_enabled = True
    scripted_tabletop_lift_target_arm_delta = V325_VERIFIED_LIFT_ARM_DELTA
    scripted_tabletop_lift_target_prior_ramp_steps = 60
    scripted_action_prior_hand_start_step = 44
    scripted_action_prior_hand_ramp_steps = 92
    scripted_action_prior_hand_action = 1.0
    scripted_action_prior_lift_start_step = 112
    scripted_action_prior_lift_steps = 300
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 0

    true_grasp_opposition_mode = "contact"
    lift_reward_uses_opposition_gate = False
    quality_lift_progress_uses_opposition_gate = False
    palm_reach_rew_scale = 24.0
    fingertip_reach_rew_scale = 24.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 40.0
    dynamic_tabletop_pregrasp_height_rew_scale = 44.0
    dynamic_tabletop_pregrasp_height_offset = 0.080
    dynamic_tabletop_pregrasp_height_scale = 0.055
    dynamic_tabletop_low_palm_penalty_scale = 46.0
    dynamic_tabletop_min_palm_height_offset = 0.020
    dynamic_tabletop_contact_pregrasp_gate_min = 0.45
    contact_rew_scale = 14.0
    true_grasp_rew_scale = 115.0
    opposition_rew_scale = 35.0
    catch_progress_rew_scale = 130.0
    grasp_quality_rew_scale = 185.0
    quality_lift_progress_rew_scale = 2400.0
    lifted_true_grasp_rew_scale = 4300.0
    lift_progress_rew_scale = 1450.0
    tabletop_stable_catch_rew_scale = 700.0
    tabletop_stable_catch_min_lift_multiplier = 0.02
    stable_hold_rew_scale = 4800.0
    hold_progress_rew_scale = 7200.0
    success_bonus = 14000.0
    dynamic_success_hold_steps = 8
    tabletop_hover_height_delta = 0.075
    tabletop_hover_latch_lift_progress = 0.50
    tabletop_hover_success_xy_tolerance = 0.22
    tabletop_hover_success_z_tolerance = 0.060
    tabletop_hover_success_object_speed = 0.28
    tabletop_hover_target_rew_scale = 1700.0
    tabletop_hover_height_progress_rew_scale = 1700.0
    tabletop_hover_goal_rew_scale = 7200.0
    tabletop_hover_stable_rew_scale = 7800.0
    tabletop_hover_linear_penalty_scale = 280.0
    tabletop_hover_overshoot_penalty_scale = 420.0
    tabletop_hover_z_vel_penalty_scale = 520.0
    tabletop_hover_vel_penalty_scale = 300.0
    tabletop_hover_target_drift_penalty_scale = 650.0
    tabletop_hover_grasp_loss_penalty_scale = 900.0
    tabletop_hover_under_height_penalty_scale = 380.0
    tabletop_hover_post_latch_speed_penalty_scale = 520.0
    tabletop_grasped_arm_lift_rew_scale = 700.0
    tabletop_arm_lift_reward_object_margin = 0.012
    tabletop_arm_object_lift_gap_margin = 0.018
    tabletop_arm_object_lift_gap_penalty_scale = 3600.0
    tabletop_lift_action_prior_rew_scale = 140.0
    tabletop_lift_action_prior_gate_min = 0.18
    tabletop_lift_without_object_min_arm_progress = 0.035
    tabletop_lift_without_object_penalty_scale = 2200.0
    tabletop_no_lift_after_grasp_penalty_scale = 1100.0
    tabletop_no_lift_after_grasp_grace_steps = 8
    tabletop_no_lift_after_grasp_ramp_steps = 50
    tabletop_no_lift_after_grasp_max_penalty = 6.0
    tabletop_post_success_hold_rew_scale = 6200.0
    tabletop_post_success_unstable_penalty_scale = 6500.0
    tabletop_post_success_grasp_loss_penalty_scale = 4500.0
    tabletop_post_success_speed_penalty_scale = 1100.0
    palm_only_lift_penalty_scale = 420.0
    scoop_lift_penalty_scale = 360.0


@configclass
class Revo2DynamicTabletopTransportRichPriorTargetObsTeacherEnvCfg(
    Revo2DynamicTabletopTransportStrongRewardTargetObsTeacherEnvCfg
):
    """91-observation transport teacher with asset/hover privileges and a lift prior."""

    reference_name = "revo2_dynamic_tabletop_transport_assets_rich_prior_target_obs_teacher"
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_TRANSPORT_OBJECT_SPECS[0],
        pos=(0.58, -0.05, 0.335),
    )
    object_start_pos = (0.58, -0.05, 0.335)
    reset_object_pos_noise = (0.035, 0.025, 0.002)
    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_start_count = 2
    tabletop_asset_curriculum_steps = 3_500_000
    tabletop_motion_modes = ("linear", "curved", "turntable")
    tabletop_motion_mode_curriculum = True
    tabletop_motion_mode_curriculum_start_count = 1
    tabletop_motion_mode_curriculum_steps = 3_000_000

    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.08
    dynamic_grasp_speed_curriculum_full_success = 0.22
    dynamic_grasp_speed_curriculum_ema_alpha = 0.03
    dynamic_grasp_speed_curriculum_alpha_rise = 0.0010
    dynamic_grasp_speed_curriculum_allow_decrease = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.025, 0.14)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (-0.8, 0.8)
    dynamic_tabletop_heading_range = (-0.85, 0.85)

    reference_hand_fractions = (0.55, 0.82, 0.62, 0.62, 0.62, 0.62)
    scripted_action_prior_enabled = True
    lift_arm_delta = V325_VERIFIED_LIFT_ARM_DELTA
    lift_action_prior = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    tabletop_arm_lift_progress_baseline_pos = V327_PREGRASP_ARM_POS
    scripted_action_prior_residual_scale = 0.16
    scripted_tabletop_pregrasp_prior_enabled = True
    scripted_tabletop_pregrasp_arm_pos = V327_PREGRASP_ARM_POS
    scripted_tabletop_pregrasp_prior_start_step = 0
    scripted_tabletop_pregrasp_prior_ramp_steps = 145
    scripted_tabletop_lift_target_prior_enabled = True
    scripted_tabletop_lift_target_arm_delta = V325_VERIFIED_LIFT_ARM_DELTA
    scripted_tabletop_lift_target_prior_ramp_steps = 80
    scripted_action_prior_hand_start_step = 72
    scripted_action_prior_hand_ramp_steps = 110
    scripted_action_prior_hand_action = 1.0
    scripted_action_prior_lift_start_step = 104
    scripted_action_prior_lift_steps = 280
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 4

    true_grasp_opposition_mode = "contact"
    lift_reward_uses_opposition_gate = False
    quality_lift_progress_uses_opposition_gate = False
    quality_lift_progress_rew_scale = 1300.0
    lifted_true_grasp_rew_scale = 2800.0
    lift_progress_rew_scale = 850.0
    tabletop_stable_catch_rew_scale = 500.0
    tabletop_stable_catch_min_lift_multiplier = 0.02
    stable_hold_rew_scale = 2600.0
    hold_progress_rew_scale = 4600.0
    success_bonus = 9500.0
    dynamic_success_hold_steps = 4
    tabletop_hover_height_delta = 0.050
    tabletop_hover_latch_lift_progress = 0.18
    tabletop_hover_latch_uses_grasp_seen = True
    tabletop_hover_reward_uses_grasp_seen = True
    tabletop_success_uses_grasp_seen = True
    tabletop_hover_success_xy_tolerance = 0.26
    tabletop_hover_success_z_tolerance = 0.10
    tabletop_hover_success_object_speed = 0.36
    tabletop_hover_target_rew_scale = 1700.0
    tabletop_hover_height_progress_rew_scale = 1500.0
    tabletop_hover_goal_rew_scale = 7600.0
    tabletop_hover_stable_rew_scale = 8200.0
    tabletop_hover_linear_penalty_scale = 420.0
    tabletop_hover_overshoot_penalty_scale = 620.0
    tabletop_hover_z_vel_penalty_scale = 760.0
    tabletop_hover_vel_penalty_scale = 420.0
    tabletop_hover_target_drift_penalty_scale = 900.0
    tabletop_hover_grasp_loss_penalty_scale = 1300.0
    tabletop_hover_under_height_penalty_scale = 620.0
    tabletop_hover_post_latch_speed_penalty_scale = 820.0
    tabletop_grasped_arm_lift_rew_scale = 1400.0
    tabletop_arm_lift_reward_object_margin = 0.12
    tabletop_arm_object_lift_gap_margin = 0.12
    tabletop_arm_object_lift_gap_penalty_scale = 900.0
    tabletop_lift_action_prior_rew_scale = 80.0
    tabletop_lift_action_prior_gate_min = 0.0
    tabletop_lift_without_object_penalty_scale = 260.0
    tabletop_no_lift_after_grasp_penalty_scale = 520.0
    tabletop_no_lift_after_grasp_grace_steps = 8
    tabletop_no_lift_after_grasp_ramp_steps = 50
    tabletop_no_lift_after_grasp_max_penalty = 6.0
    tabletop_post_success_hold_rew_scale = 6200.0
    tabletop_post_success_unstable_penalty_scale = 7200.0
    tabletop_post_success_grasp_loss_penalty_scale = 5200.0
    tabletop_post_success_speed_penalty_scale = 1500.0
    palm_only_lift_penalty_scale = 430.0
    scoop_lift_penalty_scale = 360.0


@configclass
class InspireDynamicDexterousTeacherEnvCfg(Revo2DynamicDexterousTeacherEnvCfg):
    """Base config for Franka + Inspire privileged teacher tasks."""

    action_space = 13
    observation_space = 76
    hand_embodiment = "inspire"
    action_contract = "inspire_semantic_13d"
    reference_name = "inspire_z180_dynamic_teacher_base"
    robot_cfg: ArticulationCfg = _inspire_z180_robot_cfg(ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS)
    default_arm_pos = ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS
    hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    sim_hand_joint_names = INSPIRE_HAND_JOINT_NAMES
    fingertip_body_names = INSPIRE_FINGERTIP_BODY_NAMES
    touch_body_names = INSPIRE_FINGERTIP_BODY_NAMES
    palm_body_name = "hand_base_link"
    palm_offset = INSPIRE_PALM_OFFSET
    fingertip_body_offsets = INSPIRE_FINGERTIP_BODY_OFFSETS
    reference_hand_fractions = (1.0,) * 6
    inspire_semantic_close_targets = INSPIRE_ANYDEX_P80_CLOSE_TARGETS
    hand_moving_average = 0.78
    contact_distance = 0.030
    contact_score_scale = 0.030
    palm_contact_distance = 0.060


@configclass
class InspireDynamicTabletopTeacherEnvCfg(Revo2DynamicTabletopTeacherEnvCfg):
    """Franka + Inspire privileged teacher task for moving tabletop grasping."""

    action_space = 13
    observation_space = 76
    task_family = "dynamic_tabletop_grasp"
    hand_embodiment = "inspire"
    action_contract = "inspire_semantic_13d"
    reference_name = "inspire_z180_dynamic_tabletop_privileged_teacher_joint_target"
    episode_length_s = 6.0
    robot_cfg: ArticulationCfg = _inspire_z180_robot_cfg(ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS)
    default_arm_pos = ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS
    hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    sim_hand_joint_names = INSPIRE_HAND_JOINT_NAMES
    fingertip_body_names = INSPIRE_FINGERTIP_BODY_NAMES
    touch_body_names = INSPIRE_FINGERTIP_BODY_NAMES
    palm_body_name = "hand_base_link"
    palm_offset = INSPIRE_PALM_OFFSET
    fingertip_body_offsets = INSPIRE_FINGERTIP_BODY_OFFSETS
    reference_hand_fractions = (1.0,) * 6
    inspire_semantic_close_targets = INSPIRE_ANYDEX_P80_CLOSE_TARGETS
    hand_moving_average = 0.78
    contact_distance = 0.030
    contact_score_scale = 0.030
    palm_contact_distance = 0.060
    true_grasp_opposition_mode = "dot"
    opposition_cos_threshold = 0.0
    stable_object_palm_vel = 0.42
    dynamic_success_hold_steps = 6
    dynamic_tabletop_pregrasp_xy_rew_scale = 45.0
    dynamic_tabletop_pregrasp_height_rew_scale = 55.0
    dynamic_tabletop_pregrasp_height_offset = 0.105
    dynamic_tabletop_pregrasp_height_scale = 0.055
    dynamic_tabletop_low_palm_penalty_scale = 120.0
    dynamic_tabletop_min_palm_height_offset = 0.055
    dynamic_tabletop_low_palm_height_scale = 0.040
    dynamic_tabletop_low_palm_max_penalty = 3.0
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = True
    dynamic_tabletop_contact_pregrasp_gate_min = 0.15
    lift_progress_rew_scale = 90.0
    contact_rew_scale = 12.0
    true_grasp_rew_scale = 140.0
    opposition_rew_scale = 85.0
    catch_progress_rew_scale = 150.0
    stable_hold_rew_scale = 900.0
    hold_progress_rew_scale = 1600.0
    success_bonus = 6500.0
    lift_reward_uses_grasp_quality_gate = True
    lift_reward_min_grasp_quality_multiplier = 0.0
    lift_reward_uses_opposition_gate = True
    lift_reward_min_opposition_multiplier = 0.10
    quality_lift_progress_uses_opposition_gate = True
    quality_lift_progress_min_opposition_multiplier = 0.10
    grasp_quality_finger_count_weight = 0.18
    grasp_quality_non_thumb_weight = 0.17
    grasp_quality_thumb_weight = 0.20
    grasp_quality_opposition_weight = 0.45
    grasp_quality_rew_scale = 120.0
    quality_lift_progress_rew_scale = 420.0
    lifted_true_grasp_rew_scale = 2200.0
    scoop_lift_penalty_scale = 460.0
    palm_only_lift_penalty_scale = 320.0
    palm_only_lift_dist = 0.14


@configclass
class InspireDynamicTabletopDirectResidualTeacherEnvCfg(InspireDynamicTabletopTeacherEnvCfg):
    """Franka + Inspire tabletop teacher with IsaacGym-style legacy/direct control and residual hand prior."""

    reference_name = "inspire_z180_dynamic_tabletop_direct_residual_privileged_teacher"
    policy_action_interface = "isaaclab_direct"
    arm_action_scale = 1.60
    arm_moving_average = 0.38
    hand_moving_average = 0.36
    arm_target_clamp_delta = (0.70, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95)
    initial_arm_target_lock_steps = 0
    initial_hand_target_lock_steps = 24

    scripted_action_prior_enabled = True
    scripted_action_prior_residual_scale = 0.85
    scripted_action_prior_hand_start_step = 48
    scripted_action_prior_hand_ramp_steps = 96
    scripted_action_prior_hand_action = 1.0
    scripted_action_prior_lift_start_step = 150
    scripted_action_prior_lift_steps = 120
    scripted_action_prior_lift_requires_grasp = True

    dynamic_tabletop_release_motion_on_contact = True
    dynamic_tabletop_release_motion_contact_count = 1
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.04, 0.15)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (-0.70, 0.70)
    dynamic_grasp_speed_curriculum_steps = 2_000_000

    dynamic_tabletop_pregrasp_xy_rew_scale = 7.0
    dynamic_tabletop_pregrasp_height_rew_scale = 8.0
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = False
    dynamic_tabletop_low_palm_penalty_scale = 35.0

    lift_progress_rew_scale = 160.0
    contact_rew_scale = 16.0
    true_grasp_rew_scale = 60.0
    opposition_rew_scale = 24.0
    catch_progress_rew_scale = 70.0
    grasp_quality_rew_scale = 28.0
    quality_lift_progress_rew_scale = 420.0
    lifted_true_grasp_rew_scale = 900.0
    stable_hold_rew_scale = 900.0
    hold_progress_rew_scale = 2600.0
    success_bonus = 9500.0
    lift_reward_uses_grasp_quality_gate = True
    lift_reward_min_grasp_quality_multiplier = 0.15
    lift_reward_uses_opposition_gate = True
    lift_reward_min_opposition_multiplier = 0.20
    quality_lift_progress_uses_opposition_gate = False
    scoop_lift_penalty_scale = 220.0
    palm_only_lift_penalty_scale = 260.0
    action_penalty_scale = 0.004
    arm_target_delta_penalty_scale = 0.006


@configclass
class InspireDynamicTabletopDirectResidualFullSpeedTeacherEnvCfg(
    InspireDynamicTabletopDirectResidualTeacherEnvCfg
):
    """Full-speed fine-tuning config for the direct-residual Inspire tabletop teacher."""

    reference_name = "inspire_z180_dynamic_tabletop_direct_residual_full_speed_privileged_teacher"
    dynamic_grasp_speed_curriculum = False


@configclass
class InspireDynamicTabletopRollingDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopDirectResidualTeacherEnvCfg
):
    """Inspire direct-residual rolling-object tabletop teacher."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_assets_direct_residual_teacher"
    observation_space = 86
    tabletop_object_asset_specs = TABLETOP_ROLLING_OBJECT_SPECS
    tabletop_asset_set_enabled = True
    tabletop_asset_obs_enabled = True
    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_start_count = 1
    tabletop_asset_curriculum_steps = 2_000_000
    tabletop_motion_modes = ("free",)
    tabletop_motion_mode_curriculum = False
    tabletop_motion_mode_curriculum_start_count = 1
    tabletop_motion_mode_curriculum_steps = 2_000_000
    dynamic_tabletop_persistent_motion = False
    dynamic_tabletop_bounce_at_workspace = False
    dynamic_tabletop_release_motion_on_contact = False
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=(0.58, -0.16, TABLETOP_ROLLING_START_Z),
    )
    object_shape = str(TABLETOP_ROLLING_START_SPEC["proxy_shape"])
    object_radius = float(TABLETOP_ROLLING_START_SPEC["radius"])
    object_size = tuple(TABLETOP_ROLLING_START_SPEC["size"])
    object_start_pos = (0.58, -0.16, TABLETOP_ROLLING_START_Z)
    reset_object_pos_noise = (0.070, 0.035, 0.002)
    dynamic_tabletop_initial_speed_range = (0.03, 0.16)
    dynamic_tabletop_initial_yaw_rate_range = (-1.2, 1.2)
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "steps"
    dynamic_grasp_speed_curriculum_steps = 2_000_000
    affordance_label_mode = "tabletop_rolling_assets"


@configclass
class InspireDynamicTabletopRollingFastSpeedDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingDirectResidualTeacherEnvCfg
):
    """Inspire rolling-object teacher with the same 0.10-0.40 m/s target band as the Revo2 fast eval."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_assets_fast_speed_direct_residual_teacher"
    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_start_count = 1
    tabletop_asset_curriculum_steps = 1_500_000

    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.18
    dynamic_grasp_speed_curriculum_full_success = 0.52
    dynamic_grasp_speed_curriculum_ema_alpha = 0.035
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00025
    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_tabletop_start_speed_range = (0.03, 0.16)
    dynamic_tabletop_initial_speed_range = (0.10, 0.40)
    dynamic_tabletop_start_yaw_rate_range = (-0.60, 0.60)
    dynamic_tabletop_initial_yaw_rate_range = (-2.40, 2.40)
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE

    dynamic_tabletop_pregrasp_lead_time = 0.34
    dynamic_tabletop_pregrasp_ahead_distance = 0.10
    dynamic_tabletop_pregrasp_ready_distance = 0.22
    stable_object_palm_vel = 0.42
    tabletop_success_lift_height = 0.040
    dynamic_success_hold_steps = 8

    stable_hold_rew_scale = 3000.0
    hold_progress_rew_scale = 5200.0
    success_bonus = 16000.0
    tabletop_post_success_stability_latch_enabled = True
    tabletop_post_success_arm_target_lock_enabled = True
    tabletop_post_success_arm_target_lock_blend = 1.0
    tabletop_post_success_hand_target_lock_enabled = True
    tabletop_post_success_hand_target_lock_blend = 1.0
    tabletop_post_success_hold_rew_scale = 4200.0
    tabletop_post_success_unstable_penalty_scale = 2800.0
    tabletop_post_success_grasp_loss_penalty_scale = 2400.0
    tabletop_post_success_under_height_penalty_scale = 1100.0
    tabletop_post_success_speed_penalty_scale = 650.0
    tabletop_post_success_action_penalty_scale = 0.018
    tabletop_post_success_target_delta_penalty_scale = 0.035


@configclass
class InspireDynamicTabletopRollingLiftFocusedDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingFastSpeedDirectResidualTeacherEnvCfg
):
    """Continuation config that pushes the Inspire rolling teacher from contact into lift/hold."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_assets_lift_focused_direct_residual_teacher"

    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.08
    dynamic_grasp_speed_curriculum_full_success = 0.40
    dynamic_grasp_speed_curriculum_ema_alpha = 0.035
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00015
    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_tabletop_start_speed_range = (0.02, 0.12)
    dynamic_tabletop_initial_speed_range = (0.10, 0.40)
    dynamic_tabletop_start_yaw_rate_range = (-0.45, 0.45)
    dynamic_tabletop_initial_yaw_rate_range = (-2.40, 2.40)

    arm_moving_average = 0.34
    scripted_action_prior_residual_scale = 0.95
    scripted_action_prior_lift_start_step = 120
    scripted_action_prior_lift_steps = 140
    scripted_action_prior_lift_requires_grasp = True

    dynamic_tabletop_pregrasp_xy_rew_scale = 140.0
    dynamic_tabletop_pregrasp_height_rew_scale = 80.0
    dynamic_tabletop_low_palm_penalty_scale = 80.0

    contact_rew_scale = 18.0
    true_grasp_rew_scale = 120.0
    opposition_rew_scale = 75.0
    catch_progress_rew_scale = 170.0
    grasp_quality_rew_scale = 150.0
    quality_lift_progress_rew_scale = 1400.0
    lifted_true_grasp_rew_scale = 3200.0
    lift_progress_rew_scale = 950.0
    tabletop_stable_catch_rew_scale = 1500.0
    stable_hold_rew_scale = 5200.0
    hold_progress_rew_scale = 9000.0
    success_bonus = 24000.0

    lift_reward_uses_grasp_quality_gate = True
    lift_reward_min_grasp_quality_multiplier = 0.20
    lift_reward_uses_opposition_gate = True
    lift_reward_min_opposition_multiplier = 0.20
    quality_lift_progress_uses_opposition_gate = False

    tabletop_grasped_palm_lift_rew_scale = 1400.0
    tabletop_grasped_palm_lift_height = 0.035
    tabletop_grasped_palm_lift_scale = 0.040
    tabletop_grasped_arm_lift_rew_scale = 1400.0
    tabletop_arm_lift_reward_object_margin = 0.12
    tabletop_arm_object_lift_gap_penalty_scale = 180.0
    tabletop_lift_without_object_penalty_scale = 95.0
    tabletop_lift_without_object_min_arm_progress = 0.22
    tabletop_no_lift_after_grasp_penalty_scale = 320.0
    tabletop_no_lift_after_grasp_grace_steps = 14
    tabletop_no_lift_after_grasp_ramp_steps = 46
    tabletop_no_lift_after_grasp_max_penalty = 5.0
    tabletop_no_lift_min_progress = 0.10
    tabletop_lift_action_prior_rew_scale = 180.0
    tabletop_lift_action_prior_gate_min = 0.20

    palm_only_lift_penalty_scale = 340.0
    scoop_lift_penalty_scale = 280.0

    dynamic_success_hold_steps = 6
    tabletop_success_lift_height = 0.040
    stable_object_palm_vel = 0.46
    tabletop_post_success_hold_rew_scale = 6400.0
    tabletop_post_success_unstable_penalty_scale = 4200.0
    tabletop_post_success_grasp_loss_penalty_scale = 3400.0
    tabletop_post_success_under_height_penalty_scale = 1700.0
    tabletop_post_success_speed_penalty_scale = 900.0
    tabletop_post_success_action_penalty_scale = 0.025
    tabletop_post_success_target_delta_penalty_scale = 0.050
    tabletop_post_success_arm_joint_vel_penalty_scale = 60.0
    tabletop_post_success_arm_target_drift_penalty_scale = 1100.0
    tabletop_post_success_arm_target_drift_tolerance = 0.10
    tabletop_post_success_arm_target_drift_scale = 0.30
    tabletop_post_success_palm_drift_penalty_scale = 1700.0
    tabletop_post_success_palm_drift_tolerance = 0.040
    tabletop_post_success_palm_drift_scale = 0.090


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftFastDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingLiftFocusedDirectResidualTeacherEnvCfg
):
    """Inspire rolling teacher with AnyDex sphere close target and earlier lift guidance."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_fast_direct_residual_teacher"
    inspire_semantic_close_targets = INSPIRE_ANYDEX_P80_CLOSE_TARGETS

    scripted_action_prior_hand_start_step = 54
    scripted_action_prior_hand_ramp_steps = 78
    scripted_action_prior_lift_start_step = 80
    scripted_action_prior_lift_steps = 150
    scripted_tabletop_lift_target_prior_ramp_steps = 48
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 0

    tabletop_lift_action_prior_gate_min = 0.42
    tabletop_lift_action_prior_rew_scale = 260.0
    tabletop_arm_lift_reward_object_margin = 0.18
    tabletop_arm_object_lift_gap_margin = 0.18
    tabletop_lift_without_object_min_arm_progress = 0.32
    tabletop_lift_without_object_penalty_scale = 70.0
    tabletop_no_lift_after_grasp_grace_steps = 8
    tabletop_no_lift_after_grasp_ramp_steps = 30
    tabletop_no_lift_min_progress = 0.055
    quality_lift_progress_rew_scale = 1900.0
    lifted_true_grasp_rew_scale = 4200.0
    lift_progress_rew_scale = 1250.0
    tabletop_grasped_palm_lift_rew_scale = 1900.0
    tabletop_grasped_arm_lift_rew_scale = 1900.0

    tabletop_success_lift_height = 0.035
    dynamic_success_hold_steps = 4
    stable_object_palm_vel = 0.54
    tabletop_hover_latch_lift_progress = 0.12
    tabletop_hover_success_z_tolerance = 0.075
    tabletop_hover_success_object_speed = 0.34


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftHoldBootstrapDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftFastDirectResidualTeacherEnvCfg
):
    """Low-speed Inspire continuation that turns frequent true-grasps into load-bearing lift/hold."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_hold_bootstrap_direct_residual_teacher"

    dynamic_tabletop_start_speed_range = (0.015, 0.080)
    dynamic_tabletop_initial_speed_range = (0.020, 0.140)
    dynamic_tabletop_start_yaw_rate_range = (-0.35, 0.35)
    dynamic_tabletop_initial_yaw_rate_range = (-0.85, 0.85)
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.03
    dynamic_grasp_speed_curriculum_full_success = 0.20
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00008
    dynamic_grasp_speed_curriculum_allow_decrease = True

    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.10
    scripted_action_prior_hand_start_step = 42
    scripted_action_prior_hand_ramp_steps = 70
    scripted_action_prior_lift_start_step = 86
    scripted_action_prior_lift_steps = 260
    scripted_tabletop_lift_target_prior_ramp_steps = 76
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 2
    scripted_action_prior_lift_memory_requires_streak = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = True
    scripted_tabletop_hand_grasp_memory_prior_start_step = 64
    scripted_tabletop_hand_grasp_memory_prior_steps = 260
    scripted_tabletop_hand_grasp_memory_min_steps = 1
    scripted_tabletop_hand_grasp_memory_action = 1.0
    scripted_tabletop_hand_grasp_memory_ramp_steps = 20

    tabletop_lift_action_prior_gate_min = 0.62
    tabletop_lift_action_prior_rew_scale = 420.0
    tabletop_arm_lift_reward_object_margin = 0.10
    tabletop_arm_object_lift_gap_margin = 0.10
    tabletop_arm_object_lift_gap_penalty_scale = 320.0
    tabletop_lift_without_object_min_arm_progress = 0.20
    tabletop_lift_without_object_penalty_scale = 180.0
    tabletop_no_lift_after_grasp_penalty_scale = 640.0
    tabletop_no_lift_after_grasp_grace_steps = 4
    tabletop_no_lift_after_grasp_ramp_steps = 22
    tabletop_no_lift_after_grasp_max_penalty = 8.0
    tabletop_no_lift_min_progress = 0.035

    quality_lift_progress_rew_scale = 2800.0
    lifted_true_grasp_rew_scale = 6800.0
    lift_progress_rew_scale = 2300.0
    tabletop_stable_catch_rew_scale = 2600.0
    tabletop_stable_catch_min_lift_multiplier = 0.35
    tabletop_grasped_palm_lift_rew_scale = 2600.0
    tabletop_grasped_arm_lift_rew_scale = 3000.0
    tabletop_hover_height_progress_rew_scale = 2200.0
    tabletop_hover_target_rew_scale = 1200.0
    tabletop_hover_goal_rew_scale = 2400.0
    tabletop_hover_stable_rew_scale = 3600.0
    stable_hold_rew_scale = 8200.0
    hold_progress_rew_scale = 13000.0
    success_bonus = 36000.0

    lift_reward_uses_grasp_quality_gate = True
    lift_reward_min_grasp_quality_multiplier = 0.45
    lift_reward_uses_opposition_gate = True
    lift_reward_min_opposition_multiplier = 0.45
    quality_lift_progress_uses_opposition_gate = True
    quality_lift_progress_min_opposition_multiplier = 0.35

    tabletop_success_lift_height = 0.030
    dynamic_success_hold_steps = 3
    stable_object_palm_vel = 0.62
    tabletop_hover_latch_lift_progress = 0.08
    tabletop_hover_height_delta = 0.115
    tabletop_hover_success_z_tolerance = 0.065
    tabletop_hover_success_object_speed = 0.32
    tabletop_success_requires_hover_target = False


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarrySoftGateDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftFastDirectResidualTeacherEnvCfg
):
    """Low-speed Inspire sphere teacher that first learns load-bearing lift before strict opposition."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_softgate_direct_residual_teacher"
    # The full 4 cm AnyDex sphere posture over-flexes the coupled fingers under
    # ball contact in IsaacLab.  Keep the RH56 motion on the stable close
    # manifold; reward/pregrasp can still learn tighter opposition.
    inspire_semantic_close_targets = INSPIRE_ANYDEX_SPHERE_SAFE_CLOSE_TARGETS

    dynamic_tabletop_start_speed_range = (0.010, 0.070)
    dynamic_tabletop_initial_speed_range = (0.020, 0.120)
    dynamic_tabletop_start_yaw_rate_range = (-0.30, 0.30)
    dynamic_tabletop_initial_yaw_rate_range = (-0.75, 0.75)
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.04
    dynamic_grasp_speed_curriculum_full_success = 0.24
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00008
    dynamic_grasp_speed_curriculum_allow_decrease = True

    true_grasp_opposition_mode = "contact"
    opposition_rew_scale = 120.0
    grasp_quality_opposition_weight = 0.30
    true_grasp_rew_scale = 150.0
    contact_rew_scale = 20.0
    catch_progress_rew_scale = 220.0
    grasp_quality_rew_scale = 180.0

    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.16
    scripted_action_prior_hand_start_step = 38
    scripted_action_prior_hand_ramp_steps = 64
    scripted_action_prior_lift_start_step = 80
    scripted_action_prior_lift_steps = 280
    scripted_tabletop_lift_target_prior_ramp_steps = 72
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 1
    scripted_action_prior_lift_memory_requires_streak = False
    scripted_action_prior_lift_uses_proximity = True
    scripted_action_prior_lift_proximity_distance = 0.055
    scripted_action_prior_lift_proximity_min_contacts = 1.0
    scripted_tabletop_hand_grasp_memory_prior_enabled = True
    scripted_tabletop_hand_grasp_memory_prior_start_step = 56
    scripted_tabletop_hand_grasp_memory_prior_steps = 300
    scripted_tabletop_hand_grasp_memory_min_steps = 1
    scripted_tabletop_hand_grasp_memory_action = 1.0
    scripted_tabletop_hand_grasp_memory_ramp_steps = 14

    lift_reward_uses_grasp_quality_gate = True
    lift_reward_min_grasp_quality_multiplier = 0.25
    lift_reward_uses_opposition_gate = False
    quality_lift_progress_uses_opposition_gate = False
    tabletop_lift_action_prior_gate_min = 0.50
    tabletop_lift_action_prior_rew_scale = 520.0
    tabletop_arm_lift_reward_object_margin = 0.08
    tabletop_arm_object_lift_gap_margin = 0.08
    tabletop_arm_object_lift_gap_penalty_scale = 460.0
    tabletop_lift_without_object_min_arm_progress = 0.18
    tabletop_lift_without_object_penalty_scale = 260.0
    tabletop_no_lift_after_grasp_penalty_scale = 760.0
    tabletop_no_lift_after_grasp_grace_steps = 5
    tabletop_no_lift_after_grasp_ramp_steps = 26
    tabletop_no_lift_after_grasp_max_penalty = 7.0
    tabletop_no_lift_min_progress = 0.050

    quality_lift_progress_rew_scale = 3200.0
    lifted_true_grasp_rew_scale = 7600.0
    lift_progress_rew_scale = 2900.0
    tabletop_stable_catch_rew_scale = 3400.0
    tabletop_stable_catch_min_lift_multiplier = 0.25
    tabletop_grasped_palm_lift_rew_scale = 3400.0
    tabletop_grasped_palm_lift_height = 0.030
    tabletop_grasped_palm_lift_scale = 0.035
    tabletop_grasped_arm_lift_rew_scale = 3600.0
    tabletop_hover_height_progress_rew_scale = 3000.0
    tabletop_hover_target_rew_scale = 1200.0
    tabletop_hover_goal_rew_scale = 2600.0
    tabletop_hover_stable_rew_scale = 3800.0
    stable_hold_rew_scale = 9000.0
    hold_progress_rew_scale = 15000.0
    success_bonus = 40000.0

    tabletop_success_lift_height = 0.030
    dynamic_success_hold_steps = 3
    stable_object_palm_vel = 0.62
    tabletop_hover_latch_lift_progress = 0.08
    tabletop_hover_height_delta = 0.110
    tabletop_hover_success_z_tolerance = 0.065
    tabletop_hover_success_object_speed = 0.32
    tabletop_success_requires_hover_target = False


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryStreakGateDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarrySoftGateDirectResidualTeacherEnvCfg
):
    """Require sustained grasp before the scripted arm lift prior can pull away from the object."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_streakgate_direct_residual_teacher"

    scripted_action_prior_lift_start_step = 92
    scripted_action_prior_lift_grasp_memory_min_steps = 8
    scripted_action_prior_lift_memory_requires_streak = True
    scripted_tabletop_lift_target_prior_ramp_steps = 96
    scripted_tabletop_hand_grasp_memory_prior_start_step = 54
    scripted_tabletop_hand_grasp_memory_min_steps = 1
    scripted_tabletop_hand_grasp_memory_ramp_steps = 12

    tabletop_lift_action_prior_gate_min = 0.05
    tabletop_lift_action_prior_rew_scale = 360.0
    tabletop_lift_without_object_min_arm_progress = 0.14
    tabletop_lift_without_object_penalty_scale = 420.0
    tabletop_arm_object_lift_gap_penalty_scale = 680.0
    tabletop_no_lift_after_grasp_grace_steps = 10
    tabletop_no_lift_after_grasp_ramp_steps = 32
    tabletop_no_lift_after_grasp_penalty_scale = 560.0
    tabletop_no_lift_min_progress = 0.045

    dynamic_tabletop_pregrasp_height_offset = 0.085
    dynamic_tabletop_pregrasp_height_scale = 0.045
    dynamic_tabletop_min_palm_height_offset = 0.040
    dynamic_tabletop_low_palm_penalty_scale = 70.0


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryObjectFollowDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryStreakGateDirectResidualTeacherEnvCfg
):
    """Reward the first load-bearing lift explicitly: object upward velocity and palm-object carry."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_object_follow_direct_residual_teacher"

    scripted_action_prior_lift_start_step = 88
    scripted_action_prior_lift_grasp_memory_min_steps = 3
    scripted_action_prior_lift_memory_requires_streak = True
    scripted_tabletop_lift_target_prior_ramp_steps = 84

    tabletop_lift_action_prior_gate_min = 0.12
    tabletop_lift_action_prior_rew_scale = 440.0
    tabletop_arm_lift_reward_object_margin = 0.06
    tabletop_arm_object_lift_gap_margin = 0.055
    tabletop_arm_object_lift_gap_penalty_scale = 900.0
    tabletop_lift_without_object_min_arm_progress = 0.10
    tabletop_lift_without_object_penalty_scale = 620.0

    tabletop_no_lift_after_grasp_grace_steps = 8
    tabletop_no_lift_after_grasp_ramp_steps = 34
    tabletop_no_lift_after_grasp_penalty_scale = 430.0
    tabletop_no_lift_min_progress = 0.035

    tabletop_object_up_vel_rew_scale = 2800.0
    tabletop_object_up_vel_scale = 0.080
    tabletop_object_carry_lift_rew_scale = 4200.0
    tabletop_object_carry_min_grasp_streak = 2
    tabletop_object_carry_streak_ramp_steps = 5
    tabletop_object_carry_uses_grasp_seen = True
    tabletop_object_carry_grasp_seen_gate = 0.30
    tabletop_object_carry_stall_penalty_scale = 1400.0
    tabletop_object_carry_stall_min_arm_progress = 0.07
    tabletop_object_carry_stall_min_z_vel = 0.018

    quality_lift_progress_rew_scale = 3600.0
    lifted_true_grasp_rew_scale = 8200.0
    lift_progress_rew_scale = 3300.0
    tabletop_grasped_palm_lift_rew_scale = 2800.0
    tabletop_grasped_arm_lift_rew_scale = 2400.0


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryObjectFollowGentleDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryObjectFollowDirectResidualTeacherEnvCfg
):
    """Gentler object-follow shaping that preserves contact before forcing load-bearing lift."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_object_follow_gentle_direct_residual_teacher"
    )

    scripted_action_prior_lift_start_step = 84
    scripted_action_prior_lift_grasp_memory_min_steps = 2
    scripted_tabletop_lift_target_prior_ramp_steps = 96

    tabletop_lift_action_prior_gate_min = 0.18
    tabletop_lift_action_prior_rew_scale = 520.0
    tabletop_arm_object_lift_gap_margin = 0.11
    tabletop_arm_object_lift_gap_penalty_scale = 260.0
    tabletop_lift_without_object_min_arm_progress = 0.22
    tabletop_lift_without_object_penalty_scale = 140.0

    tabletop_no_lift_after_grasp_grace_steps = 14
    tabletop_no_lift_after_grasp_ramp_steps = 60
    tabletop_no_lift_after_grasp_penalty_scale = 160.0
    tabletop_no_lift_min_progress = 0.035

    tabletop_object_up_vel_rew_scale = 4200.0
    tabletop_object_up_vel_scale = 0.070
    tabletop_object_carry_lift_rew_scale = 5200.0
    tabletop_object_carry_min_grasp_streak = 1
    tabletop_object_carry_streak_ramp_steps = 8
    tabletop_object_carry_uses_grasp_seen = False
    tabletop_object_carry_stall_penalty_scale = 180.0
    tabletop_object_carry_stall_min_arm_progress = 0.20
    tabletop_object_carry_stall_min_z_vel = 0.012

    quality_lift_progress_rew_scale = 4200.0
    lifted_true_grasp_rew_scale = 9200.0
    lift_progress_rew_scale = 4200.0
    tabletop_grasped_palm_lift_rew_scale = 3600.0
    tabletop_grasped_arm_lift_rew_scale = 3200.0


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryObjectFollowGentleDirectResidualTeacherEnvCfg
):
    """From-scratch Inspire rolling teacher with only the RH56 tabletop height corrected."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_heightfix_direct_residual_teacher"
    )

    # Keep the Revo2-style teacher/reward schedule intact, but raise the
    # RH56BFX approach band so the larger fingertips do not scrape the table.
    dynamic_tabletop_pregrasp_height_offset = 0.120
    dynamic_tabletop_pregrasp_height_scale = 0.055
    dynamic_tabletop_min_palm_height_offset = 0.075
    dynamic_tabletop_low_palm_height_scale = 0.035
    dynamic_tabletop_low_palm_max_penalty = 4.0
    dynamic_tabletop_pregrasp_height_rew_scale = 120.0
    dynamic_tabletop_low_palm_penalty_scale = 160.0


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMediumFrictionDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixDirectResidualTeacherEnvCfg
):
    """A/B task with a moderate tabletop object friction increase."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_heightfix_mediumfriction_direct_residual_teacher"
    )

    tabletop_object_asset_specs = _with_tabletop_friction(
        (TABLETOP_ROLLING_START_SPEC,),
        static_friction=1.00,
        dynamic_friction=0.90,
    )
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        tabletop_object_asset_specs[0],
        pos=(0.58, -0.16, TABLETOP_ROLLING_START_Z),
    )


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixDirectResidualTeacherEnvCfg
):
    """A/B task with tabletop object friction matched to the table material."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_heightfix_matchedfriction_direct_residual_teacher"
    )

    tabletop_object_asset_specs = _with_tabletop_friction(
        (TABLETOP_ROLLING_START_SPEC,),
        static_friction=1.20,
        dynamic_friction=1.00,
    )
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        tabletop_object_asset_specs[0],
        pos=(0.58, -0.16, TABLETOP_ROLLING_START_Z),
    )


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionStrictSuccessDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionDirectResidualTeacherEnvCfg
):
    """Matched-friction diagnostic with strict contact-based success only."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_heightfix_matchedfriction_strictsuccess_direct_residual_teacher"
    )

    strict_success_enabled = True
    strict_success_contact_distance = 0.008
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    strict_reward_enabled = True
    strict_reward_contact_score_scale = 0.025
    strict_approach_score_scale = 0.095
    strict_approach_rew_scale = 90.0
    strict_multifinger_approach_rew_scale = 900.0
    strict_touch_score_scale = 0.0075
    strict_touch_rew_scale = 1600.0
    tabletop_lift_action_prior_gate_min = 0.0
    dynamic_tabletop_pregrasp_height_offset = 0.105
    dynamic_tabletop_min_palm_height_offset = 0.060
    dynamic_tabletop_pregrasp_height_rew_scale = 80.0
    dynamic_tabletop_low_palm_penalty_scale = 110.0
    stable_object_palm_vel = 0.12
    tabletop_hover_success_object_speed = 0.12
    dynamic_success_hold_steps = 12


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionStrictMetricsDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionDirectResidualTeacherEnvCfg
):
    """Matched-friction eval with strict success metrics but unchanged policy priors."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_heightfix_"
        "matchedfriction_strictmetrics_direct_residual_teacher"
    )

    strict_success_enabled = True
    strict_success_contact_distance = 0.008
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    scripted_action_prior_uses_strict_grasp = False


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionNoPriorDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionDirectResidualTeacherEnvCfg
):
    """Diagnostic: old matched-friction task with all scripted action priors disabled."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_heightfix_"
        "matchedfriction_noprior_direct_residual_teacher"
    )

    scripted_action_prior_enabled = False
    scripted_action_prior_zero_passthrough_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_relative_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False
    tabletop_lift_action_prior_rew_scale = 0.0


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionLegacyHandDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionDirectResidualTeacherEnvCfg
):
    """Diagnostic: old matched-friction task with the earlier Inspire hand actuator/open pose."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_heightfix_"
        "matchedfriction_legacyhand_direct_residual_teacher"
    )

    robot_cfg: ArticulationCfg = _inspire_z180_legacy_robot_cfg(ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS)


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionLegacyHandNoPriorDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionLegacyHandDirectResidualTeacherEnvCfg
):
    """Diagnostic: legacy hand physics plus no scripted action prior."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_heightfix_"
        "matchedfriction_legacyhand_noprior_direct_residual_teacher"
    )

    scripted_action_prior_enabled = False
    scripted_action_prior_zero_passthrough_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_relative_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False
    tabletop_lift_action_prior_rew_scale = 0.0


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixConservativeFrictionStrictSuccessDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixDirectResidualTeacherEnvCfg
):
    """Conservative-friction diagnostic with strict contact-based success only."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_heightfix_conservativefriction_strictsuccess_direct_residual_teacher"
    )

    tabletop_object_asset_specs = _with_tabletop_friction(
        (TABLETOP_ROLLING_START_SPEC,),
        static_friction=0.80,
        dynamic_friction=0.60,
        friction_combine_mode="average",
    )
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        tabletop_object_asset_specs[0],
        pos=(0.58, -0.16, TABLETOP_ROLLING_START_Z),
    )

    strict_success_enabled = True
    strict_success_contact_distance = 0.008
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    strict_reward_enabled = True
    strict_reward_contact_score_scale = 0.025
    strict_approach_score_scale = 0.095
    strict_approach_rew_scale = 90.0
    strict_multifinger_approach_rew_scale = 900.0
    strict_touch_score_scale = 0.0075
    strict_touch_rew_scale = 1600.0
    tabletop_lift_action_prior_gate_min = 0.0
    dynamic_tabletop_pregrasp_height_offset = 0.105
    dynamic_tabletop_min_palm_height_offset = 0.060
    dynamic_tabletop_pregrasp_height_rew_scale = 80.0
    dynamic_tabletop_low_palm_penalty_scale = 110.0
    stable_object_palm_vel = 0.12
    tabletop_hover_success_object_speed = 0.12
    dynamic_success_hold_steps = 12


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixHighFrictionDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixDirectResidualTeacherEnvCfg
):
    """A/B task that only raises tabletop object friction for lift diagnostics."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_heightfix_highfriction_direct_residual_teacher"
    )

    tabletop_object_asset_specs = _with_tabletop_friction(
        (TABLETOP_ROLLING_START_SPEC,),
        static_friction=1.45,
        dynamic_friction=1.20,
    )
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        tabletop_object_asset_specs[0],
        pos=(0.58, -0.16, TABLETOP_ROLLING_START_Z),
    )


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixLoadBearingDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixDirectResidualTeacherEnvCfg
):
    """Height-fixed Inspire teacher with stronger load-bearing lift after first grasp."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_heightfix_load_bearing_direct_residual_teacher"
    )

    # Keep the working height-fixed approach, and only change the lift phase so
    # the policy cannot score by raising the hand while the ball stays on table.
    scripted_action_prior_lift_start_step = 112
    scripted_action_prior_lift_steps = 360
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_memory_requires_streak = True
    scripted_action_prior_lift_grasp_memory_min_steps = 8
    scripted_tabletop_lift_target_prior_ramp_steps = 170

    tabletop_lift_use_grasp_seen_gate = True
    tabletop_lift_grasp_seen_gate = 0.74
    tabletop_lift_action_prior_gate_min = 0.10
    tabletop_arm_lift_reward_object_margin = 0.040
    tabletop_arm_object_lift_gap_margin = 0.035
    tabletop_arm_object_lift_gap_penalty_scale = 3300.0
    tabletop_lift_without_object_min_arm_progress = 0.055
    tabletop_lift_without_object_penalty_scale = 2500.0

    tabletop_no_lift_after_grasp_grace_steps = 10
    tabletop_no_lift_after_grasp_ramp_steps = 40
    tabletop_no_lift_after_grasp_penalty_scale = 900.0
    tabletop_no_lift_min_progress = 0.030

    tabletop_object_up_vel_rew_scale = 5600.0
    tabletop_object_up_vel_scale = 0.050
    tabletop_object_carry_lift_rew_scale = 10400.0
    tabletop_object_carry_min_grasp_streak = 2
    tabletop_object_carry_streak_ramp_steps = 10
    tabletop_object_carry_uses_grasp_seen = True
    tabletop_object_carry_grasp_seen_gate = 0.70
    tabletop_object_carry_stall_penalty_scale = 2600.0
    tabletop_object_carry_stall_min_arm_progress = 0.055
    tabletop_object_carry_stall_min_z_vel = 0.010

    quality_lift_progress_rew_scale = 5200.0
    lifted_true_grasp_rew_scale = 11200.0
    lift_progress_rew_scale = 5200.0
    tabletop_grasped_palm_lift_rew_scale = 4400.0
    tabletop_grasped_arm_lift_rew_scale = 3800.0
    tabletop_stable_catch_rew_scale = 4200.0
    stable_hold_rew_scale = 12500.0
    hold_progress_rew_scale = 19000.0
    success_bonus = 45000.0

    tabletop_success_lift_height = 0.035
    dynamic_success_hold_steps = 10
    stable_object_palm_vel = 0.30
    tabletop_hover_success_object_speed = 0.20


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryCleanStartDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixDirectResidualTeacherEnvCfg
):
    """From-scratch Inspire rolling teacher with a non-penetrating Franka reset."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_cleanstart_direct_residual_teacher"
    )

    robot_cfg: ArticulationCfg = _inspire_z180_robot_cfg(ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS)
    default_arm_pos = ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS
    arm_target_clamp_delta = (2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30)


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryCleanStartSettledLiftDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryCleanStartDirectResidualTeacherEnvCfg
):
    """Clean-start Inspire rolling teacher that separates close/settle from load-bearing lift."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_cleanstart_settledlift_direct_residual_teacher"
    )

    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.010, 0.080)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (-0.55, 0.55)
    dynamic_grasp_speed_curriculum_start_success = 0.04
    dynamic_grasp_speed_curriculum_full_success = 0.28
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00010

    scripted_action_prior_enabled = True
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_residual_scale = 0.20
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.16
    tabletop_arm_lift_progress_baseline_pos = V327_PREGRASP_ARM_POS
    scripted_tabletop_pregrasp_prior_enabled = True
    scripted_tabletop_pregrasp_arm_pos = V327_PREGRASP_ARM_POS
    scripted_tabletop_pregrasp_prior_start_step = 0
    scripted_tabletop_pregrasp_prior_ramp_steps = 150
    scripted_tabletop_lift_target_prior_enabled = True
    scripted_tabletop_lift_target_arm_delta = V325_VERIFIED_LIFT_ARM_DELTA
    scripted_tabletop_lift_target_prior_ramp_steps = 140
    scripted_action_prior_hand_start_step = 110
    scripted_action_prior_hand_ramp_steps = 80
    scripted_action_prior_hand_action = 1.0
    scripted_action_prior_lift_start_step = 170
    scripted_action_prior_lift_steps = 340
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 8
    scripted_action_prior_lift_memory_requires_streak = True

    tabletop_lift_action_prior_gate_min = 0.08
    tabletop_lift_action_prior_rew_scale = 520.0
    tabletop_arm_lift_reward_object_margin = 0.060
    tabletop_arm_object_lift_gap_margin = 0.055
    tabletop_arm_object_lift_gap_penalty_scale = 1100.0
    tabletop_lift_without_object_min_arm_progress = 0.08
    tabletop_lift_without_object_penalty_scale = 0.0
    tabletop_no_lift_after_grasp_grace_steps = 14
    tabletop_no_lift_after_grasp_ramp_steps = 42
    tabletop_no_lift_after_grasp_penalty_scale = 320.0
    tabletop_no_lift_min_progress = 0.035

    tabletop_object_up_vel_rew_scale = 5600.0
    tabletop_object_up_vel_scale = 0.055
    tabletop_object_carry_lift_rew_scale = 9000.0
    tabletop_object_carry_min_grasp_streak = 3
    tabletop_object_carry_streak_ramp_steps = 8
    tabletop_object_carry_uses_grasp_seen = True
    tabletop_object_carry_grasp_seen_gate = 0.55
    tabletop_object_carry_stall_penalty_scale = 1200.0
    tabletop_object_carry_stall_min_arm_progress = 0.08
    tabletop_object_carry_stall_min_z_vel = 0.012

    quality_lift_progress_rew_scale = 4600.0
    lifted_true_grasp_rew_scale = 9800.0
    lift_progress_rew_scale = 4600.0
    tabletop_grasped_palm_lift_rew_scale = 4200.0
    tabletop_grasped_arm_lift_rew_scale = 3600.0
    tabletop_stable_catch_rew_scale = 3800.0
    stable_hold_rew_scale = 11000.0
    hold_progress_rew_scale = 17000.0
    success_bonus = 42000.0

    tabletop_success_lift_height = 0.035
    dynamic_success_hold_steps = 8
    stable_object_palm_vel = 0.36
    tabletop_hover_success_object_speed = 0.24


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryLoadBearingLiftDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryCleanStartSettledLiftDirectResidualTeacherEnvCfg
):
    """Clean-start Inspire teacher that keeps lift pressure active after the first grasp contact."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_load_bearing_lift_direct_residual_teacher"
    )

    scripted_action_prior_lift_start_step = 190
    scripted_action_prior_lift_steps = 420
    scripted_tabletop_lift_target_prior_ramp_steps = 210
    scripted_action_prior_lift_grasp_memory_min_steps = 10

    tabletop_lift_use_grasp_seen_gate = True
    tabletop_lift_grasp_seen_gate = 0.72
    tabletop_lift_action_prior_gate_min = 0.04
    tabletop_arm_lift_reward_object_margin = 0.035
    tabletop_arm_object_lift_gap_margin = 0.030
    tabletop_arm_object_lift_gap_penalty_scale = 2600.0

    tabletop_object_carry_grasp_seen_gate = 0.68
    tabletop_object_carry_min_grasp_streak = 2
    tabletop_object_carry_streak_ramp_steps = 10
    tabletop_object_carry_stall_penalty_scale = 2400.0
    tabletop_object_carry_stall_min_arm_progress = 0.05
    tabletop_object_carry_stall_min_z_vel = 0.010

    tabletop_no_lift_after_grasp_grace_steps = 18
    tabletop_no_lift_after_grasp_ramp_steps = 52
    tabletop_no_lift_after_grasp_penalty_scale = 240.0
    tabletop_object_up_vel_rew_scale = 5200.0
    tabletop_object_up_vel_scale = 0.045
    tabletop_object_carry_lift_rew_scale = 9800.0


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryClearanceDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryObjectFollowGentleDirectResidualTeacherEnvCfg
):
    """Inspire rolling teacher with explicit hand/table clearance before contact reward."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_clearance_direct_residual_teacher"
    )

    # RH56BFX is larger than Revo2.  A palm-only height target lets the fingers
    # scrape the table, so keep the palm higher and use clearance as a soft
    # table-contact penalty.  The clearance estimate is intentionally not a hard
    # success/contact gate because link origins are conservative for the coupled
    # Inspire fingers and otherwise prevent the lift phase from unlocking.
    dynamic_tabletop_pregrasp_height_offset = 0.135
    dynamic_tabletop_pregrasp_height_scale = 0.055
    dynamic_tabletop_min_palm_height_offset = 0.090
    dynamic_tabletop_low_palm_height_scale = 0.035
    dynamic_tabletop_low_palm_max_penalty = 4.0
    dynamic_tabletop_pregrasp_height_rew_scale = 180.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 90.0
    dynamic_tabletop_low_palm_penalty_scale = 260.0
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = True
    dynamic_tabletop_contact_pregrasp_gate_min = 0.22

    tabletop_arm_clearance_body_names = INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_NAMES
    tabletop_arm_clearance_body_margins = INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_MARGINS
    tabletop_arm_clearance_xy_padding = 0.20
    tabletop_arm_clearance_margin = 0.030
    tabletop_arm_clearance_scale = 0.045
    tabletop_arm_clearance_max_penalty = 5.0
    tabletop_arm_clearance_penalty_scale = 0.0
    tabletop_success_requires_arm_clearance = False
    tabletop_gate_contact_rewards_by_clearance = False
    tabletop_contact_clearance_gate_min = 0.35
    tabletop_contact_clearance_gate_scale = 0.70

    # Once a grasp has been seen, pull upward decisively enough that the policy
    # experiences load-bearing lift instead of settling into contact-only grasp.
    scripted_action_prior_lift_start_step = 54
    scripted_action_prior_lift_requires_grasp = False
    scripted_action_prior_lift_grasp_memory_min_steps = 0
    scripted_action_prior_lift_memory_requires_streak = False
    scripted_tabletop_lift_target_prior_ramp_steps = 56
    scripted_action_prior_active_residual_scale = 0.08
    tabletop_lift_action_prior_gate_min = 0.56
    tabletop_lift_action_prior_rew_scale = 620.0
    tabletop_lift_without_object_min_arm_progress = 0.18
    tabletop_lift_without_object_penalty_scale = 220.0
    tabletop_arm_object_lift_gap_margin = 0.09
    tabletop_arm_object_lift_gap_penalty_scale = 280.0
    tabletop_no_lift_after_grasp_grace_steps = 5
    tabletop_no_lift_after_grasp_ramp_steps = 28
    tabletop_no_lift_after_grasp_penalty_scale = 760.0


@configclass
class InspireDynamicTabletopRollingSphereCloseLiftCarryCleanStartConservativeFrictionStrictSuccessDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryCleanStartSettledLiftDirectResidualTeacherEnvCfg
):
    """Clean-start Inspire rolling teacher with conservative friction and strict physical success."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_close_lift_carry_cleanstart_"
        "conservativefriction_strictsuccess_direct_residual_teacher"
    )

    tabletop_object_asset_specs = _with_tabletop_friction(
        (TABLETOP_ROLLING_START_SPEC,),
        static_friction=0.80,
        dynamic_friction=0.60,
        friction_combine_mode="average",
    )
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        tabletop_object_asset_specs[0],
        pos=(0.58, -0.16, TABLETOP_ROLLING_START_Z),
    )

    strict_success_enabled = True
    strict_success_contact_distance = 0.008
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    strict_reward_enabled = True
    strict_reward_contact_score_scale = 0.025
    strict_approach_score_scale = 0.095
    strict_approach_rew_scale = 90.0
    strict_multifinger_approach_rew_scale = 900.0
    strict_touch_score_scale = 0.0075
    strict_touch_rew_scale = 1600.0

    # The reset is clean-start, but the RH56BFX fingers are large enough that
    # approach rewards also need a palm height band with real table clearance.
    dynamic_tabletop_pregrasp_height_offset = 0.135
    dynamic_tabletop_pregrasp_height_scale = 0.055
    dynamic_tabletop_min_palm_height_offset = 0.090
    dynamic_tabletop_low_palm_height_scale = 0.035
    dynamic_tabletop_low_palm_max_penalty = 4.0
    dynamic_tabletop_pregrasp_height_rew_scale = 180.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 90.0
    dynamic_tabletop_low_palm_penalty_scale = 260.0
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = True
    dynamic_tabletop_contact_pregrasp_gate_min = 0.22

    tabletop_arm_clearance_body_names = INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_NAMES
    tabletop_arm_clearance_body_margins = INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_MARGINS
    tabletop_arm_clearance_xy_padding = 0.20
    tabletop_arm_clearance_margin = 0.030
    tabletop_arm_clearance_scale = 0.045
    tabletop_arm_clearance_max_penalty = 5.0
    tabletop_arm_clearance_penalty_scale = 0.0
    tabletop_success_requires_arm_clearance = False
    tabletop_gate_contact_rewards_by_clearance = False
    tabletop_contact_clearance_gate_min = 0.35
    tabletop_contact_clearance_gate_scale = 0.70

    tabletop_lift_action_prior_gate_min = 0.0
    stable_object_palm_vel = 0.12
    tabletop_hover_success_object_speed = 0.12
    dynamic_success_hold_steps = 12


@configclass
class InspireDynamicTabletopRollingSphereApproachBootstrapTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereCloseLiftCarryCleanStartDirectResidualTeacherEnvCfg
):
    """Clean-start Inspire rolling task that first validates approach from scratch."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_approach_bootstrap_teacher"

    # This task is intentionally not residual.  The previous clean-start strict
    # run was anchored to a Revo2-tuned pregrasp prior, which pulled the Inspire
    # hand away from the sphere before the policy could get contact signal.
    scripted_action_prior_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False

    policy_action_interface = "isaaclab_direct"
    arm_action_scale = 1.60
    arm_moving_average = 0.38
    hand_moving_average = 0.36
    arm_target_clamp_delta = (2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30)
    initial_arm_target_lock_steps = 0
    initial_hand_target_lock_steps = 12

    tabletop_object_asset_specs = _with_tabletop_friction(
        (TABLETOP_ROLLING_START_SPEC,),
        static_friction=0.80,
        dynamic_friction=0.60,
        friction_combine_mode="average",
    )
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        tabletop_object_asset_specs[0],
        pos=(0.58, -0.16, TABLETOP_ROLLING_START_Z),
    )
    object_shape = str(TABLETOP_ROLLING_START_SPEC["proxy_shape"])
    object_radius = float(TABLETOP_ROLLING_START_SPEC["radius"])
    object_size = tuple(TABLETOP_ROLLING_START_SPEC["size"])
    object_start_pos = (0.58, -0.16, TABLETOP_ROLLING_START_Z)

    dynamic_tabletop_persistent_motion = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.0, 0.0)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (0.0, 0.0)
    dynamic_grasp_speed_curriculum = False
    reset_object_pos_noise = (0.020, 0.020, 0.001)

    strict_success_enabled = False
    strict_reward_enabled = False
    contact_distance = 0.045
    contact_score_scale = 0.045
    true_grasp_opposition_mode = "contact"

    reach_distance_scale = 0.24
    fingertip_distance_scale = 0.070
    dynamic_tabletop_pregrasp_lead_time = 0.0
    dynamic_tabletop_pregrasp_ahead_distance = 0.0
    dynamic_tabletop_pregrasp_xy_distance_scale = 0.12
    dynamic_tabletop_pregrasp_height_offset = 0.135
    dynamic_tabletop_pregrasp_height_scale = 0.060
    dynamic_tabletop_min_palm_height_offset = 0.090
    dynamic_tabletop_low_palm_height_scale = 0.035
    dynamic_tabletop_low_palm_max_penalty = 4.0
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = False

    palm_reach_rew_scale = 34.0
    fingertip_reach_rew_scale = 30.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 620.0
    dynamic_tabletop_pregrasp_height_rew_scale = 260.0
    dynamic_tabletop_low_palm_penalty_scale = 180.0
    contact_rew_scale = 24.0
    true_grasp_rew_scale = 120.0
    opposition_rew_scale = 40.0
    catch_progress_rew_scale = 120.0
    grasp_quality_rew_scale = 80.0

    tabletop_arm_clearance_body_names = INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_NAMES
    tabletop_arm_clearance_body_margins = INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_MARGINS
    tabletop_arm_clearance_xy_padding = 0.20
    tabletop_arm_clearance_margin = 0.030
    tabletop_arm_clearance_scale = 0.045
    tabletop_arm_clearance_max_penalty = 5.0
    tabletop_arm_clearance_penalty_scale = 0.0
    tabletop_success_requires_arm_clearance = False
    tabletop_gate_contact_rewards_by_clearance = False

    quality_lift_progress_rew_scale = 0.0
    lifted_true_grasp_rew_scale = 0.0
    lift_progress_rew_scale = 120.0
    tabletop_grasped_palm_lift_rew_scale = 0.0
    tabletop_grasped_arm_lift_rew_scale = 0.0
    tabletop_lift_action_prior_rew_scale = 0.0
    tabletop_lift_without_object_penalty_scale = 0.0
    tabletop_no_lift_after_grasp_penalty_scale = 0.0
    tabletop_object_up_vel_rew_scale = 0.0
    tabletop_object_carry_lift_rew_scale = 0.0
    stable_hold_rew_scale = 900.0
    hold_progress_rew_scale = 1600.0
    success_bonus = 6000.0
    action_penalty_scale = 0.002
    arm_target_delta_penalty_scale = 0.004


@configclass
class InspireDynamicTabletopRollingSphereJ2ApproachBootstrapTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereApproachBootstrapTeacherEnvCfg
):
    """Clean-start Inspire rolling task with a minimal joint-2 approach warm start."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_j2_approach_bootstrap_teacher"

    scripted_action_prior_enabled = True
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.40

    scripted_tabletop_approach_action_prior_enabled = True
    scripted_tabletop_approach_action_prior = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    scripted_tabletop_approach_action_prior_start_step = 0
    scripted_tabletop_approach_action_prior_steps = 105
    scripted_tabletop_approach_action_prior_ramp_steps = 24

    tabletop_arm_lift_progress_baseline_pos = ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS
    scripted_action_prior_hand_start_step = 96
    scripted_action_prior_hand_ramp_steps = 88
    scripted_action_prior_hand_action = 1.0
    scripted_action_prior_lift_start_step = 170
    scripted_action_prior_lift_steps = 320
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_memory_requires_streak = True
    scripted_action_prior_lift_grasp_memory_min_steps = 4

    tabletop_lift_action_prior_gate_min = 0.0
    tabletop_lift_action_prior_rew_scale = 0.0
    tabletop_lift_without_object_penalty_scale = 0.0


@configclass
class InspireDynamicTabletopRollingSphereP80CleanStartTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereApproachBootstrapTeacherEnvCfg
):
    """From-scratch Inspire rolling teacher with clean reset and P80 hand closure."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_p80_cleanstart_teacher"

    robot_cfg: ArticulationCfg = _inspire_z180_robot_cfg(ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS)
    default_arm_pos = ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS
    episode_length_s = 8.0

    object_start_pos = (0.58, 0.0, TABLETOP_ROLLING_START_Z)
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=object_start_pos,
    )
    object_shape = str(TABLETOP_ROLLING_START_SPEC["proxy_shape"])
    object_radius = float(TABLETOP_ROLLING_START_SPEC["radius"])
    object_size = tuple(TABLETOP_ROLLING_START_SPEC["size"])
    tabletop_object_asset_specs = (TABLETOP_ROLLING_START_SPEC,)
    tabletop_asset_set_enabled = False
    tabletop_asset_obs_enabled = False
    tabletop_asset_curriculum = False
    reset_object_pos_noise = (0.0, 0.0, 0.0)

    inspire_semantic_close_targets = INSPIRE_ANYDEX_P80_CLOSE_TARGETS
    reference_hand_fractions = INSPIRE_P80_HOME_SEED_HAND_FRACTIONS

    scripted_action_prior_enabled = True
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.10
    scripted_action_prior_uses_strict_grasp = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False
    scripted_action_prior_lift_start_step = 90
    scripted_action_prior_lift_steps = 260
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_memory_requires_streak = True
    scripted_action_prior_lift_grasp_memory_min_steps = 8

    arm_action_scale = 1.85
    arm_moving_average = 0.45
    hand_moving_average = 0.42
    initial_hand_target_lock_steps = 8

    strict_success_enabled = True
    strict_success_contact_distance = 0.010
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    strict_reward_enabled = False
    contact_distance = 0.040
    contact_score_scale = 0.035
    min_finger_contacts = 3
    min_non_thumb_contacts = 2
    true_grasp_opposition_mode = "contact"
    thumb_contact_reward_weight = 0.42
    thumb_true_grasp_score_weight = 0.38
    opposition_reward_uses_weighted_score = True
    grasp_quality_finger_count_weight = 0.30
    grasp_quality_non_thumb_weight = 0.30
    grasp_quality_thumb_weight = 0.20
    grasp_quality_opposition_weight = 0.20
    strict_reward_contact_score_scale = 0.024
    strict_approach_score_scale = 0.090
    strict_approach_rew_scale = 65.0
    strict_multifinger_approach_rew_scale = 520.0
    strict_touch_score_scale = 0.010
    strict_touch_rew_scale = 900.0

    dynamic_tabletop_gate_contact_rewards_by_pregrasp = True
    dynamic_tabletop_contact_pregrasp_gate_min = 0.25
    dynamic_tabletop_pregrasp_xy_rew_scale = 300.0
    dynamic_tabletop_pregrasp_height_offset = 0.125
    dynamic_tabletop_pregrasp_height_scale = 0.045
    dynamic_tabletop_pregrasp_height_rew_scale = 170.0
    dynamic_tabletop_min_palm_height_offset = 0.080
    dynamic_tabletop_low_palm_height_scale = 0.035
    dynamic_tabletop_low_palm_penalty_scale = 200.0
    dynamic_tabletop_side_contact_xy_limit = 0.150
    dynamic_tabletop_side_contact_xy_ramp = 0.090
    dynamic_tabletop_side_contact_penalty_scale = 220.0
    contact_rew_scale = 85.0
    true_grasp_rew_scale = 240.0
    opposition_rew_scale = 140.0
    catch_progress_rew_scale = 300.0
    grasp_quality_rew_scale = 900.0

    lift_reward_uses_grasp_quality_gate = True
    lift_reward_min_grasp_quality_multiplier = 0.0
    lift_reward_uses_opposition_gate = False
    lift_reward_min_opposition_multiplier = 1.0
    quality_lift_progress_uses_opposition_gate = False
    quality_lift_progress_min_opposition_multiplier = 1.0

    tabletop_lift_action_prior_gate_min = 0.0
    tabletop_lift_action_prior_rew_scale = 320.0
    tabletop_arm_lift_reward_object_margin = 0.080
    quality_lift_progress_rew_scale = 3000.0
    lifted_true_grasp_rew_scale = 7200.0
    lift_progress_rew_scale = 3000.0
    tabletop_grasped_palm_lift_rew_scale = 3000.0
    tabletop_grasped_arm_lift_rew_scale = 2800.0
    tabletop_stable_catch_rew_scale = 3200.0
    tabletop_object_up_vel_rew_scale = 3400.0
    tabletop_object_up_vel_scale = 0.055
    tabletop_object_carry_lift_rew_scale = 6200.0
    tabletop_object_carry_uses_grasp_seen = True
    tabletop_object_carry_grasp_seen_gate = 0.45
    tabletop_object_carry_stall_penalty_scale = 900.0
    tabletop_object_carry_stall_min_arm_progress = 0.08
    tabletop_object_carry_stall_min_z_vel = 0.012
    stable_hold_rew_scale = 9000.0
    hold_progress_rew_scale = 13000.0
    success_bonus = 36000.0

    tabletop_lift_without_object_min_arm_progress = 0.10
    tabletop_lift_without_object_penalty_scale = 1500.0
    tabletop_arm_object_lift_gap_margin = 0.055
    tabletop_arm_object_lift_gap_penalty_scale = 1500.0
    tabletop_no_lift_after_grasp_grace_steps = 8
    tabletop_no_lift_after_grasp_ramp_steps = 36
    tabletop_no_lift_after_grasp_penalty_scale = 760.0
    tabletop_no_lift_min_progress = 0.035

    tabletop_success_lift_height = 0.030
    dynamic_success_hold_steps = 6
    stable_object_palm_vel = 0.40
    tabletop_hover_success_object_speed = 0.24


@configclass
class InspireDynamicTabletopRollingSphereP80CleanStartStrictRewardTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80CleanStartTeacherEnvCfg
):
    """P80 clean-start teacher with reward gated by strict, load-bearing grasp evidence."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_p80_cleanstart_strictreward_teacher"

    strict_reward_enabled = True
    contact_distance = 0.012
    contact_score_scale = 0.018
    strict_success_contact_distance = 0.008
    strict_reward_contact_score_scale = 0.014

    contact_rew_scale = 30.0
    true_grasp_rew_scale = 75.0
    opposition_rew_scale = 80.0
    catch_progress_rew_scale = 120.0
    grasp_quality_rew_scale = 260.0
    strict_approach_rew_scale = 45.0
    strict_multifinger_approach_rew_scale = 220.0
    strict_touch_rew_scale = 420.0

    lift_reward_uses_grasp_quality_gate = True
    lift_reward_min_grasp_quality_multiplier = 0.20
    lift_reward_uses_opposition_gate = True
    lift_reward_min_opposition_multiplier = 0.20
    quality_lift_progress_uses_opposition_gate = True
    quality_lift_progress_min_opposition_multiplier = 0.20

    tabletop_lift_use_grasp_seen_gate = False
    tabletop_object_carry_uses_grasp_seen = False
    tabletop_lift_action_prior_rew_scale = 120.0
    tabletop_lift_without_object_penalty_scale = 2600.0
    tabletop_arm_object_lift_gap_penalty_scale = 2600.0
    tabletop_no_lift_after_grasp_penalty_scale = 1300.0
    tabletop_no_lift_after_grasp_grace_steps = 5
    tabletop_no_lift_after_grasp_ramp_steps = 28
    tabletop_no_lift_min_progress = 0.035

    lifted_true_grasp_rew_scale = 7800.0
    quality_lift_progress_rew_scale = 3400.0
    lift_progress_rew_scale = 3000.0
    tabletop_grasped_palm_lift_rew_scale = 2600.0
    tabletop_grasped_arm_lift_rew_scale = 2400.0
    tabletop_object_carry_lift_rew_scale = 7600.0
    tabletop_stable_catch_rew_scale = 4200.0
    stable_hold_rew_scale = 10500.0
    hold_progress_rew_scale = 15000.0
    success_bonus = 42000.0


@configclass
class InspireDynamicTabletopRollingSphereP80CleanStartStrictRewardFromScratchTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80CleanStartStrictRewardTeacherEnvCfg
):
    """Strict-reward P80 clean-start task with no scripted action prior."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_p80_cleanstart_strictreward_fromscratch_teacher"

    scripted_action_prior_enabled = False
    scripted_action_prior_zero_passthrough_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False
    tabletop_lift_action_prior_rew_scale = 0.0


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumStrictRewardFromScratchTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80CleanStartStrictRewardFromScratchTeacherEnvCfg
):
    """Strict P80 from-scratch rolling task with a static-to-fast speed curriculum."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "strictreward_fromscratch_teacher"
    )

    # Start as the validated static ball task, then ramp to the requested fast
    # rolling range.  The previous "nearly static" 0-0.03 m/s start plus
    # pregrasp lead target made from-scratch exploration optimize an ahead point
    # while staying far from the ball, so alpha=0 should be truly static.
    dynamic_tabletop_persistent_motion = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.100, 0.400)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (-1.20, 1.20)
    dynamic_tabletop_heading_range = (-3.141592653589793, 3.141592653589793)
    dynamic_tabletop_randomize_yaw = True
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success"
    dynamic_grasp_speed_curriculum_metric = "true_grasp"
    dynamic_grasp_speed_curriculum_start_success = 0.04
    dynamic_grasp_speed_curriculum_full_success = 0.25
    dynamic_grasp_speed_curriculum_ema_alpha = 0.03
    dynamic_grasp_speed_curriculum_alpha_rise = 0.02
    dynamic_grasp_speed_curriculum_allow_decrease = False

    dynamic_tabletop_pregrasp_lead_time = 0.0
    dynamic_tabletop_pregrasp_ahead_distance = 0.0
    reset_object_pos_noise = (0.0, 0.0, 0.0)


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardFromScratchTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80CleanStartTeacherEnvCfg
):
    """Loose-reward P80 from-scratch rolling task with static-to-fast curriculum."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_fromscratch_teacher"
    )

    scripted_action_prior_enabled = False
    scripted_action_prior_zero_passthrough_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False
    tabletop_lift_action_prior_rew_scale = 0.0

    dynamic_tabletop_persistent_motion = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.100, 0.400)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (-1.20, 1.20)
    dynamic_tabletop_heading_range = (-3.141592653589793, 3.141592653589793)
    dynamic_tabletop_randomize_yaw = True
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success"
    dynamic_grasp_speed_curriculum_metric = "true_grasp"
    dynamic_grasp_speed_curriculum_start_success = 0.04
    dynamic_grasp_speed_curriculum_full_success = 0.25
    dynamic_grasp_speed_curriculum_ema_alpha = 0.03
    dynamic_grasp_speed_curriculum_alpha_rise = 0.02
    dynamic_grasp_speed_curriculum_allow_decrease = False

    dynamic_tabletop_pregrasp_lead_time = 0.0
    dynamic_tabletop_pregrasp_ahead_distance = 0.0
    reset_object_pos_noise = (0.0, 0.0, 0.0)


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardFromScratchTeacherEnvCfg
):
    """Continue the learned P80 rolling grasp with reward-only lift-and-hold guidance."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_teacher"
    )

    # The source policy already reaches and wraps the ball at full curriculum
    # speed.  Keep the rollout distribution fixed and bias only the reward
    # toward load-bearing lift.
    dynamic_grasp_speed_curriculum_override_alpha = 1.0

    scripted_action_prior_enabled = False
    scripted_action_prior_zero_passthrough_enabled = False
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 1.0
    scripted_action_prior_uses_strict_grasp = False

    scripted_action_prior_hand_start_step = 9999
    scripted_action_prior_hand_ramp_steps = 0
    scripted_action_prior_hand_action = 0.0
    scripted_action_prior_lift_start_step = 84
    scripted_action_prior_lift_steps = 340
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_memory_requires_streak = False
    scripted_action_prior_lift_grasp_memory_min_steps = 1

    scripted_tabletop_hand_grasp_memory_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_start_step = 56
    scripted_tabletop_hand_grasp_memory_prior_steps = 320
    scripted_tabletop_hand_grasp_memory_min_steps = 1
    scripted_tabletop_hand_grasp_memory_action = 0.95
    scripted_tabletop_hand_grasp_memory_ramp_steps = 16

    tabletop_lift_action_prior_gate_min = 0.0
    tabletop_lift_action_prior_rew_scale = 0.0
    tabletop_arm_lift_reward_object_margin = 0.070
    tabletop_arm_object_lift_gap_margin = 0.060
    tabletop_arm_object_lift_gap_penalty_scale = 1100.0
    tabletop_lift_without_object_min_arm_progress = 0.12
    tabletop_lift_without_object_penalty_scale = 900.0
    tabletop_no_lift_after_grasp_grace_steps = 6
    tabletop_no_lift_after_grasp_ramp_steps = 30
    tabletop_no_lift_after_grasp_penalty_scale = 980.0
    tabletop_no_lift_min_progress = 0.040

    tabletop_object_up_vel_rew_scale = 5200.0
    tabletop_object_up_vel_scale = 0.055
    tabletop_object_carry_lift_rew_scale = 9800.0
    tabletop_object_carry_min_grasp_streak = 1
    tabletop_object_carry_streak_ramp_steps = 6
    tabletop_object_carry_uses_grasp_seen = True
    tabletop_object_carry_grasp_seen_gate = 0.35
    tabletop_object_carry_stall_penalty_scale = 1400.0
    tabletop_object_carry_stall_min_arm_progress = 0.08
    tabletop_object_carry_stall_min_z_vel = 0.012

    quality_lift_progress_rew_scale = 4200.0
    lifted_true_grasp_rew_scale = 9800.0
    lift_progress_rew_scale = 4400.0
    tabletop_grasped_palm_lift_rew_scale = 3800.0
    tabletop_grasped_arm_lift_rew_scale = 3600.0
    tabletop_stable_catch_rew_scale = 5200.0
    stable_hold_rew_scale = 12000.0
    hold_progress_rew_scale = 18000.0
    success_bonus = 48000.0

    tabletop_success_lift_height = 0.030
    dynamic_success_hold_steps = 6
    stable_object_palm_vel = 0.34
    tabletop_hover_success_object_speed = 0.22


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideClearanceTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideTeacherEnvCfg
):
    """Lift-guide continuation that rejects hand/table contact and press-down grasps."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_clearance_teacher"
    )

    # The previous lift-guide policy pressed the RH56BFX fingertips and palm
    # into the table, then received grasp/lift reward from object motion caused
    # by scraping.  Keep this as reward-only, but make table clearance a real
    # constraint for contact, lift, and success rewards.
    dynamic_tabletop_pregrasp_height_offset = 0.155
    dynamic_tabletop_pregrasp_height_scale = 0.050
    dynamic_tabletop_pregrasp_height_rew_scale = 260.0
    dynamic_tabletop_min_palm_height_offset = 0.115
    dynamic_tabletop_low_palm_height_scale = 0.028
    dynamic_tabletop_low_palm_max_penalty = 6.0
    dynamic_tabletop_low_palm_penalty_scale = 1400.0
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = True
    dynamic_tabletop_contact_pregrasp_gate_min = 0.10
    dynamic_tabletop_side_contact_penalty_scale = 900.0

    tabletop_arm_clearance_body_names = INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_NAMES
    tabletop_arm_clearance_body_margins = INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_MARGINS
    tabletop_arm_clearance_xy_padding = 0.20
    tabletop_arm_clearance_margin = 0.030
    tabletop_arm_clearance_scale = 0.045
    tabletop_arm_clearance_max_penalty = 5.0
    tabletop_arm_clearance_penalty_scale = 12000.0
    tabletop_gate_contact_rewards_by_clearance = True
    tabletop_gate_boolean_grasp_rewards_by_clearance = True
    tabletop_contact_clearance_gate_min = 0.0
    tabletop_contact_clearance_gate_scale = 0.25
    tabletop_success_requires_arm_clearance = True

    tabletop_lift_use_grasp_seen_gate = False
    tabletop_object_carry_uses_grasp_seen = False
    tabletop_hover_latch_uses_grasp_seen = False
    tabletop_hover_reward_uses_grasp_seen = False
    tabletop_success_uses_grasp_seen = False

    # With table-contact shortcuts removed, keep the lift pressure slightly
    # gentler so the policy does not compensate by sweeping the ball away.
    tabletop_arm_lift_reward_object_margin = 0.050
    tabletop_arm_object_lift_gap_margin = 0.045
    tabletop_lift_without_object_min_arm_progress = 0.10
    tabletop_object_carry_min_grasp_streak = 3
    tabletop_object_carry_streak_ramp_steps = 10


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideClearanceTeacherEnvCfg
):
    """Inspire rolling task that starts from a table-clear V340 pose before learning approach/lift."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_teacher"
    )
    observation_space = 76

    # The previous Inspire rolling attempts often started from a low Revo2-tuned
    # reset and scraped the RH56BFX fingers on the tabletop before learning any
    # useful approach.  Keep the same rolling/lift curriculum, but use the older
    # table-facing Inspire V340 arm pose as the clean reset/home.
    robot_cfg: ArticulationCfg = _inspire_z180_robot_cfg(INSPIRE_V341_CLEAR_ARM_POS)
    default_arm_pos = INSPIRE_V341_CLEAR_ARM_POS
    scripted_tabletop_pregrasp_arm_pos = INSPIRE_V341_CLEAR_ARM_POS
    tabletop_arm_lift_progress_baseline_pos = INSPIRE_V341_CLEAR_ARM_POS
    scripted_tabletop_lift_target_arm_delta = INSPIRE_V340_LIFT_ARM_DELTA
    lift_arm_delta = INSPIRE_V340_LIFT_ARM_DELTA
    lift_action_prior = INSPIRE_V340_LIFT_ACTION_PRIOR
    scripted_action_prior_lift_action = INSPIRE_V340_LIFT_ACTION_PRIOR
    reset_arm_pos_noise = 0.0
    initial_arm_target_lock_steps = 12
    initial_hand_target_lock_steps = 12
    dynamic_grasp_speed_curriculum_override_alpha = 0.0

    object_start_pos = (0.58, 0.0, TABLETOP_ROLLING_START_Z)
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=object_start_pos,
    )
    reset_object_pos_noise = (0.0, 0.0, 0.0)

    strict_reward_enabled = True
    dynamic_tabletop_pregrasp_xy_distance_scale = 0.18
    dynamic_tabletop_pregrasp_xy_rew_scale = 260.0
    dynamic_tabletop_pregrasp_height_offset = 0.125
    dynamic_tabletop_pregrasp_height_scale = 0.035
    dynamic_tabletop_pregrasp_height_rew_scale = 480.0
    dynamic_tabletop_min_palm_height_offset = 0.100
    dynamic_tabletop_low_palm_height_scale = 0.035
    contact_distance = 0.014
    contact_score_scale = 0.016
    strict_success_contact_distance = 0.0125
    strict_reward_contact_score_scale = 0.014
    strict_approach_score_scale = 0.050
    strict_approach_rew_scale = 60.0
    strict_multifinger_approach_rew_scale = 120.0
    strict_opposition_approach_rew_scale = 9000.0
    strict_touch_score_scale = 0.006
    strict_touch_reward_requires_thumb_pair = True
    strict_touch_reward_uses_opposition_product = True
    strict_touch_reward_opposition_min_multiplier = 0.03
    strict_touch_rew_scale = 500.0
    strict_opposition_touch_rew_scale = 32000.0
    contact_reward_requires_thumb_pair = True
    contact_reward_uses_opposition_product = True
    contact_reward_opposition_min_multiplier = 0.08
    true_grasp_score_requires_thumb_pair = True
    true_grasp_score_uses_opposition_product = True
    true_grasp_score_opposition_min_multiplier = 0.18
    thumb_contact_reward_weight = 1.0
    thumb_true_grasp_score_weight = 1.0
    grasp_quality_finger_count_weight = 0.0
    grasp_quality_non_thumb_weight = 0.0
    grasp_quality_thumb_weight = 0.0
    grasp_quality_opposition_weight = 1.0
    contact_rew_scale = 900.0
    true_grasp_rew_scale = 5200.0
    opposition_rew_scale = 22000.0
    grasp_quality_rew_scale = 18000.0
    lift_reward_uses_opposition_gate = True
    lift_reward_min_opposition_multiplier = 0.0
    quality_lift_progress_uses_opposition_gate = True
    quality_lift_progress_min_opposition_multiplier = 0.0
    dynamic_tabletop_low_palm_penalty_scale = 650.0
    dynamic_tabletop_side_contact_penalty_scale = 350.0
    tabletop_non_thumb_without_thumb_penalty_scale = 7200.0
    tabletop_non_thumb_without_thumb_gate_start = 0.08
    tabletop_non_thumb_without_thumb_gate_ramp = 0.30
    tabletop_non_thumb_without_thumb_thumb_target = 0.25
    tabletop_lift_action_prior_gate_min = 0.0
    tabletop_lift_action_prior_rew_scale = 15000.0
    tabletop_arm_lift_reward_object_margin = 0.100
    tabletop_arm_object_lift_gap_margin = 0.090
    tabletop_grasped_arm_lift_rew_scale = 6000.0
    tabletop_no_lift_uses_soft_grasp_gate = True
    tabletop_no_lift_soft_grasp_gate = 0.020
    tabletop_no_lift_after_grasp_grace_steps = 8
    tabletop_no_lift_after_grasp_ramp_steps = 24
    tabletop_no_lift_after_grasp_penalty_scale = 90.0
    tabletop_lift_without_object_min_arm_progress = 0.140
    tabletop_lift_without_object_penalty_scale = 3200.0
    tabletop_arm_clearance_body_margins = INSPIRE_TABLETOP_HAND_CAUTION_CLEARANCE_BODY_MARGINS
    tabletop_arm_clearance_fingertip_point_margin = 0.012
    tabletop_arm_clearance_palm_point_margin = 0.024
    tabletop_arm_clearance_scale = 0.035
    tabletop_arm_clearance_penalty_scale = 18000.0
    tabletop_arm_clearance_ok_penalty_threshold = 0.020
    tabletop_contact_clearance_gate_scale = 0.08
    tabletop_success_requires_arm_clearance = True
    tabletop_terminate_on_arm_clearance_violation = True
    tabletop_arm_clearance_terminate_penalty_threshold = 0.050
    tabletop_arm_clearance_violation_terminate_start_step = 12


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceTeacherEnvCfg
):
    """86-observation compatibility task for older clean-reset Inspire rolling checkpoints."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_compat86_teacher"
    )
    observation_space = 86
    tabletop_asset_set_enabled = True
    tabletop_asset_obs_enabled = True
    tabletop_asset_curriculum = False
    tabletop_asset_curriculum_start_count = 1


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceThumbWrapCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceCompat86TeacherEnvCfg
):
    """86-observation Inspire rolling task with a stronger thumb wrap close envelope."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_thumbwrap_compat86_teacher"
    )

    inspire_semantic_close_targets = INSPIRE_ANYDEX_P80_THUMB_WRAP_CLOSE_TARGETS
    reference_hand_fractions = (1.0, 1.0, 0.80, 0.80, 1.0, 1.0)


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceTeacherEnvCfg
):
    """Clean-reset Inspire rolling task with a 5 cm sphere for size/material diagnosis."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_teacher"
    )

    object_start_pos = (0.58, 0.0, _tabletop_start_z_from_spec(TABLETOP_INSPIRE_SPHERE_50MM_SPEC))
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_INSPIRE_SPHERE_50MM_SPEC,
        pos=object_start_pos,
    )
    object_shape = "sphere"
    object_radius = float(TABLETOP_INSPIRE_SPHERE_50MM_SPEC["radius"])
    object_size = tuple(TABLETOP_INSPIRE_SPHERE_50MM_SPEC["size"])
    tabletop_object_asset_specs = (TABLETOP_INSPIRE_SPHERE_50MM_SPEC,)
    tabletop_asset_set_enabled = False
    tabletop_asset_obs_enabled = False
    tabletop_asset_curriculum = False


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere60mmTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceTeacherEnvCfg
):
    """Clean-reset Inspire rolling task with a 6 cm sphere for size/material diagnosis."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere60mm_teacher"
    )

    object_start_pos = (0.58, 0.0, _tabletop_start_z_from_spec(TABLETOP_INSPIRE_SPHERE_60MM_SPEC))
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_INSPIRE_SPHERE_60MM_SPEC,
        pos=object_start_pos,
    )
    object_shape = "sphere"
    object_radius = float(TABLETOP_INSPIRE_SPHERE_60MM_SPEC["radius"])
    object_size = tuple(TABLETOP_INSPIRE_SPHERE_60MM_SPEC["size"])
    tabletop_object_asset_specs = (TABLETOP_INSPIRE_SPHERE_60MM_SPEC,)
    tabletop_asset_set_enabled = False
    tabletop_asset_obs_enabled = False
    tabletop_asset_curriculum = False


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceTeacherEnvCfg
):
    """Clean-reset Inspire rolling task with a 5 cm higher-friction sphere."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_teacher"
    )

    object_start_pos = (0.58, 0.0, _tabletop_start_z_from_spec(TABLETOP_INSPIRE_SPHERE_50MM_HIGH_FRICTION_SPEC))
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_INSPIRE_SPHERE_50MM_HIGH_FRICTION_SPEC,
        pos=object_start_pos,
    )
    object_shape = "sphere"
    object_radius = float(TABLETOP_INSPIRE_SPHERE_50MM_HIGH_FRICTION_SPEC["radius"])
    object_size = tuple(TABLETOP_INSPIRE_SPHERE_50MM_HIGH_FRICTION_SPEC["size"])
    tabletop_object_asset_specs = (TABLETOP_INSPIRE_SPHERE_50MM_HIGH_FRICTION_SPEC,)
    tabletop_asset_set_enabled = False
    tabletop_asset_obs_enabled = False
    tabletop_asset_curriculum = False


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionUnderwrapTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionTeacherEnvCfg
):
    """50 mm high-friction sphere teacher with explicit lower-half fingertip support shaping."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_underwrap_teacher"
    )

    tabletop_underwrap_rew_scale = 9000.0
    tabletop_underwrap_below_center_fraction = 0.18
    tabletop_underwrap_height_scale = 0.010
    tabletop_underwrap_radial_fraction = 0.90
    tabletop_underwrap_radial_scale = 0.018
    tabletop_underwrap_contact_scale = 0.016
    tabletop_underwrap_min_non_thumb_contacts = 1
    tabletop_underwrap_uses_opposition = True
    tabletop_underwrap_opposition_min_multiplier = 0.12


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionTeacherEnvCfg
):
    """50 mm high-friction sphere teacher with reachable lower-support shaping."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_softunderwrap_teacher"
    )

    tabletop_underwrap_rew_scale = 8500.0
    tabletop_underwrap_below_center_fraction = 0.05
    tabletop_underwrap_height_scale = 0.040
    tabletop_underwrap_radial_fraction = 0.78
    tabletop_underwrap_radial_scale = 0.045
    tabletop_underwrap_contact_scale = 0.035
    tabletop_underwrap_contact_margin = 0.018
    tabletop_underwrap_min_non_thumb_contacts = 1
    tabletop_underwrap_uses_opposition = True
    tabletop_underwrap_opposition_min_multiplier = 0.22
    tabletop_underwrap_progress_weight = 0.65
    tabletop_underwrap_pair_weight = 0.35
    tabletop_underwrap_uses_pregrasp_gate = False


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapTeacherEnvCfg
):
    """86-observation compatibility task for existing v344/v349 Inspire rolling checkpoints."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_softunderwrap_compat86_teacher"
    )
    observation_space = 86
    tabletop_asset_set_enabled = True
    tabletop_asset_obs_enabled = True
    tabletop_asset_curriculum = False
    tabletop_asset_curriculum_start_count = 1


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapTeacherEnvCfg
):
    """Soft-underwrap continuation that makes lifting mandatory after stable contact."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_softunderwrap_liftpush_teacher"
    )

    tabletop_underwrap_rew_scale = 4200.0
    tabletop_underwrap_below_center_fraction = 0.28
    tabletop_underwrap_height_scale = 0.030
    tabletop_underwrap_radial_fraction = 0.90
    tabletop_underwrap_radial_scale = 0.035
    tabletop_underwrap_contact_scale = 0.030
    tabletop_underwrap_contact_margin = 0.012
    tabletop_underwrap_opposition_min_multiplier = 0.35
    tabletop_underwrap_progress_weight = 0.30
    tabletop_underwrap_pair_weight = 0.70

    # Do not pay for lifting the palm alone; the object has to move up.
    tabletop_grasped_palm_lift_rew_scale = 0.0
    tabletop_lift_action_prior_rew_scale = 30000.0
    tabletop_grasped_arm_lift_rew_scale = 12000.0
    tabletop_object_up_vel_rew_scale = 11000.0
    tabletop_object_up_vel_scale = 0.035
    tabletop_object_carry_lift_rew_scale = 18000.0
    quality_lift_progress_rew_scale = 7600.0
    lifted_true_grasp_rew_scale = 14000.0
    lift_progress_rew_scale = 7200.0

    tabletop_no_lift_uses_soft_grasp_gate = True
    tabletop_no_lift_soft_grasp_gate = 0.020
    tabletop_no_lift_after_grasp_grace_steps = 3
    tabletop_no_lift_after_grasp_ramp_steps = 12
    tabletop_no_lift_after_grasp_max_penalty = 8.0
    tabletop_no_lift_after_grasp_penalty_scale = 3200.0
    tabletop_no_lift_min_progress = 0.025

    tabletop_lift_without_object_min_arm_progress = 0.025
    tabletop_lift_without_object_penalty_scale = 4500.0
    tabletop_arm_object_lift_gap_margin = 0.020
    tabletop_arm_object_lift_gap_penalty_scale = 4200.0
    tabletop_object_carry_stall_penalty_scale = 3000.0
    tabletop_object_carry_stall_min_arm_progress = 0.030
    tabletop_object_carry_stall_min_z_vel = 0.010

    # Keep contact quality useful, but stop it from dominating no-lift pressure.
    strict_opposition_touch_rew_scale = 18000.0
    opposition_rew_scale = 14000.0
    grasp_quality_rew_scale = 12000.0


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushTeacherEnvCfg
):
    """86-observation lift-push continuation task for existing v349 Inspire rolling checkpoints."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_softunderwrap_liftpush_compat86_teacher"
    )
    observation_space = 86
    tabletop_asset_set_enabled = True
    tabletop_asset_obs_enabled = True
    tabletop_asset_curriculum = False
    tabletop_asset_curriculum_start_count = 1


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushFixedPriorCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushCompat86TeacherEnvCfg
):
    """Lift-push continuation with the lift prior routed through the pregrasp-compatible target path."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_fixedprior_compat86_teacher"
    )

    scripted_action_prior_enabled = True
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.05
    scripted_action_prior_uses_strict_grasp = True
    scripted_tabletop_lift_target_prior_enabled = True
    scripted_tabletop_lift_target_prior_ramp_steps = 12
    scripted_tabletop_lift_target_arm_delta = INSPIRE_V340_LIFT_ARM_DELTA
    scripted_tabletop_pregrasp_prior_control_mode = "target_track"
    scripted_action_prior_lift_start_step = 84
    scripted_action_prior_lift_steps = 360
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_memory_requires_streak = False
    scripted_action_prior_lift_grasp_memory_min_steps = 4


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative050Compat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushCompat86TeacherEnvCfg
):
    """Lift-push continuation using the best relative lift candidate from the load-bearing probe."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative050_compat86_teacher"
    )

    scripted_action_prior_enabled = True
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.18
    scripted_action_prior_uses_strict_grasp = True
    scripted_tabletop_relative_lift_target_prior_enabled = True
    scripted_tabletop_relative_lift_target_arm_delta = INSPIRE_V340_RELATIVE_LIFT_TARGET_DELTAS[1]
    scripted_action_prior_lift_start_step = 84
    scripted_action_prior_lift_steps = 260
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_memory_requires_streak = False
    scripted_action_prior_lift_grasp_memory_min_steps = 4

    scripted_tabletop_hand_grasp_memory_prior_enabled = True
    scripted_tabletop_hand_grasp_memory_prior_start_step = 84
    scripted_tabletop_hand_grasp_memory_prior_steps = 360
    scripted_tabletop_hand_grasp_memory_min_steps = 4
    scripted_tabletop_hand_grasp_memory_action = 0.98
    scripted_tabletop_hand_grasp_memory_ramp_steps = 4

    tabletop_terminate_on_arm_clearance_violation = False
    tabletop_arm_clearance_ok_penalty_threshold = 0.060
    tabletop_lift_action_prior_rew_scale = 22000.0
    tabletop_object_up_vel_rew_scale = 18000.0
    tabletop_object_up_vel_scale = 0.035
    tabletop_object_carry_lift_rew_scale = 26000.0
    quality_lift_progress_rew_scale = 12000.0
    lifted_true_grasp_rew_scale = 26000.0
    lift_progress_rew_scale = 12000.0
    tabletop_no_lift_after_grasp_penalty_scale = 5200.0
    tabletop_no_lift_min_progress = 0.022
    tabletop_lift_without_object_min_arm_progress = 0.018
    tabletop_lift_without_object_penalty_scale = 5200.0
    tabletop_arm_object_lift_gap_margin = 0.018
    tabletop_arm_object_lift_gap_penalty_scale = 5200.0
    stable_hold_rew_scale = 18000.0
    hold_progress_rew_scale = 28000.0
    success_bonus = 70000.0


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StabilityCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative050Compat86TeacherEnvCfg
):
    """Relative-lift continuation that pushes higher lift while preserving strict grasp."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_stability_compat86_teacher"
    )

    scripted_tabletop_relative_lift_target_arm_delta = INSPIRE_V340_RELATIVE_LIFT_TARGET_DELTAS[2]
    scripted_action_prior_active_residual_scale = 0.12
    scripted_action_prior_lift_steps = 300
    scripted_tabletop_hand_grasp_memory_action = 1.0
    scripted_tabletop_hand_grasp_memory_ramp_steps = 8

    tabletop_terminate_on_arm_clearance_violation = True
    tabletop_arm_clearance_terminate_penalty_threshold = 0.65
    tabletop_arm_clearance_violation_terminate_start_step = 72
    tabletop_arm_clearance_ok_penalty_threshold = 0.080
    tabletop_arm_clearance_penalty_scale = 22000.0

    strict_touch_rew_scale = 1200.0
    strict_opposition_touch_rew_scale = 42000.0
    opposition_rew_scale = 24000.0
    grasp_quality_rew_scale = 22000.0
    tabletop_stable_catch_rew_scale = 12000.0
    tabletop_object_carry_min_grasp_streak = 2
    tabletop_object_carry_streak_ramp_steps = 5
    tabletop_object_carry_lift_rew_scale = 36000.0
    quality_lift_progress_rew_scale = 18000.0
    lifted_true_grasp_rew_scale = 42000.0
    lift_progress_rew_scale = 16000.0
    stable_hold_rew_scale = 36000.0
    hold_progress_rew_scale = 54000.0
    success_bonus = 110000.0
    stable_object_palm_vel = 0.26
    tabletop_hover_success_object_speed = 0.18


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictCarryCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StabilityCompat86TeacherEnvCfg
):
    """Relative-lift continuation that only unlocks lift from sustained strict grasp."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strictcarry_compat86_teacher"
    )

    # The previous relative075 run often touched the sphere for a few frames,
    # unlocked the lift prior, then lifted after thumb/opposition contact was
    # already gone.  Require a real consecutive strict-grasp streak instead.
    scripted_action_prior_lift_memory_requires_streak = True
    scripted_action_prior_lift_grasp_memory_min_steps = 12
    scripted_action_prior_lift_steps = 320
    scripted_action_prior_active_residual_scale = 0.10

    scripted_tabletop_hand_grasp_memory_min_steps = 6
    scripted_tabletop_hand_grasp_memory_action = 1.0
    scripted_tabletop_hand_grasp_memory_ramp_steps = 4

    tabletop_lift_use_grasp_seen_gate = False
    tabletop_object_carry_uses_grasp_seen = False
    tabletop_object_carry_min_grasp_streak = 0
    tabletop_object_carry_streak_ramp_steps = 1
    tabletop_object_carry_lift_rew_scale = 22000.0
    tabletop_object_up_vel_rew_scale = 9000.0
    tabletop_lift_action_prior_rew_scale = 16000.0
    tabletop_grasped_arm_lift_rew_scale = 4500.0
    quality_lift_progress_rew_scale = 9000.0
    lifted_true_grasp_rew_scale = 52000.0
    lift_progress_rew_scale = 6500.0

    tabletop_lift_without_object_min_arm_progress = 0.055
    tabletop_lift_without_object_penalty_scale = 11000.0
    tabletop_arm_object_lift_gap_margin = 0.045
    tabletop_arm_object_lift_gap_penalty_scale = 9000.0
    scoop_lift_penalty_scale = 5200.0

    strict_opposition_touch_rew_scale = 52000.0
    opposition_rew_scale = 30000.0
    grasp_quality_rew_scale = 26000.0
    tabletop_non_thumb_without_thumb_penalty_scale = 9800.0
    tabletop_underwrap_opposition_min_multiplier = 0.55

    tabletop_arm_clearance_ok_penalty_threshold = 0.020
    tabletop_arm_clearance_terminate_penalty_threshold = 0.12
    tabletop_arm_clearance_violation_terminate_start_step = 72
    tabletop_success_requires_arm_clearance = True
    tabletop_gate_contact_rewards_by_clearance = True
    tabletop_gate_boolean_grasp_rewards_by_clearance = True
    tabletop_contact_clearance_gate_scale = 0.08


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075Streak6CarryCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictCarryCompat86TeacherEnvCfg
):
    """Moderate strict-carry continuation: require sustained grasp, but keep lift reachable."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_streak6carry_compat86_teacher"
    )

    scripted_action_prior_lift_grasp_memory_min_steps = 6
    scripted_action_prior_lift_steps = 300
    scripted_action_prior_active_residual_scale = 0.12
    scripted_tabletop_hand_grasp_memory_min_steps = 3
    scripted_tabletop_hand_grasp_memory_ramp_steps = 4

    tabletop_lift_action_prior_rew_scale = 24000.0
    tabletop_grasped_arm_lift_rew_scale = 7000.0
    tabletop_object_up_vel_rew_scale = 14000.0
    tabletop_object_carry_lift_rew_scale = 28000.0
    quality_lift_progress_rew_scale = 11000.0
    lifted_true_grasp_rew_scale = 56000.0
    lift_progress_rew_scale = 8000.0

    tabletop_lift_without_object_min_arm_progress = 0.050
    tabletop_lift_without_object_penalty_scale = 9500.0
    tabletop_arm_object_lift_gap_margin = 0.050
    tabletop_arm_object_lift_gap_penalty_scale = 7600.0
    scoop_lift_penalty_scale = 4200.0


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075HybridLiftRecoveryCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StabilityCompat86TeacherEnvCfg
):
    """Lift-recovery continuation for the strict ep120 Inspire rolling policy."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_hybrid_lift_recovery_compat86_teacher"
    )

    # StrictCarry ep120 preserved good thumb/opposition contact, but the stricter
    # lift gate removed the continuous lift signal.  Keep the reachable lift
    # prior from Relative075 while increasing the value of carrying the object.
    scripted_action_prior_lift_memory_requires_streak = False
    scripted_action_prior_lift_grasp_memory_min_steps = 4
    scripted_action_prior_lift_steps = 300
    scripted_action_prior_active_residual_scale = 0.11

    scripted_tabletop_hand_grasp_memory_min_steps = 4
    scripted_tabletop_hand_grasp_memory_action = 1.0
    scripted_tabletop_hand_grasp_memory_ramp_steps = 6

    tabletop_lift_use_grasp_seen_gate = True
    tabletop_lift_grasp_seen_gate = 0.25
    tabletop_object_carry_uses_grasp_seen = True
    tabletop_object_carry_grasp_seen_gate = 0.25
    tabletop_object_carry_min_grasp_streak = 1
    tabletop_object_carry_streak_ramp_steps = 4

    tabletop_lift_action_prior_rew_scale = 30000.0
    tabletop_object_up_vel_rew_scale = 22000.0
    tabletop_object_carry_lift_rew_scale = 44000.0
    tabletop_grasped_arm_lift_rew_scale = 9000.0
    quality_lift_progress_rew_scale = 22000.0
    lifted_true_grasp_rew_scale = 52000.0
    lift_progress_rew_scale = 18000.0
    stable_hold_rew_scale = 52000.0
    hold_progress_rew_scale = 72000.0
    success_bonus = 140000.0

    tabletop_no_lift_uses_soft_grasp_gate = True
    tabletop_no_lift_soft_grasp_gate = 0.08
    tabletop_no_lift_after_grasp_grace_steps = 4
    tabletop_no_lift_after_grasp_ramp_steps = 18
    tabletop_no_lift_after_grasp_penalty_scale = 7200.0
    tabletop_no_lift_min_progress = 0.024

    tabletop_lift_without_object_min_arm_progress = 0.035
    tabletop_lift_without_object_penalty_scale = 7600.0
    tabletop_arm_object_lift_gap_margin = 0.032
    tabletop_arm_object_lift_gap_penalty_scale = 7000.0
    tabletop_object_carry_stall_penalty_scale = 5200.0
    tabletop_object_carry_stall_min_arm_progress = 0.045
    tabletop_object_carry_stall_min_z_vel = 0.012
    scoop_lift_penalty_scale = 3600.0

    strict_opposition_touch_rew_scale = 46000.0
    opposition_rew_scale = 28000.0
    grasp_quality_rew_scale = 26000.0
    tabletop_non_thumb_without_thumb_penalty_scale = 8200.0
    tabletop_underwrap_opposition_min_multiplier = 0.42

    tabletop_arm_clearance_ok_penalty_threshold = 0.040
    tabletop_arm_clearance_terminate_penalty_threshold = 0.22
    tabletop_success_requires_arm_clearance = True
    tabletop_gate_contact_rewards_by_clearance = True
    tabletop_gate_boolean_grasp_rewards_by_clearance = True
    tabletop_contact_clearance_gate_scale = 0.10


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075LiftOpenGateCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075HybridLiftRecoveryCompat86TeacherEnvCfg
):
    """Blend a lifting checkpoint with the strict hand anchor by opening the lift prior gate."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_lift_open_gate_compat86_teacher"
    )

    tabletop_lift_action_prior_gate_min = 0.35
    scripted_action_prior_lift_start_step = 84
    scripted_action_prior_lift_steps = 320
    scripted_action_prior_active_residual_scale = 0.10

    tabletop_lift_action_prior_rew_scale = 28000.0
    tabletop_object_up_vel_rew_scale = 26000.0
    tabletop_object_carry_lift_rew_scale = 52000.0
    tabletop_grasped_arm_lift_rew_scale = 10500.0
    quality_lift_progress_rew_scale = 24000.0
    lifted_true_grasp_rew_scale = 62000.0
    lift_progress_rew_scale = 20000.0
    stable_hold_rew_scale = 62000.0
    hold_progress_rew_scale = 84000.0
    success_bonus = 160000.0

    tabletop_no_lift_after_grasp_penalty_scale = 8200.0
    tabletop_lift_without_object_min_arm_progress = 0.060
    tabletop_lift_without_object_penalty_scale = 9800.0
    tabletop_arm_object_lift_gap_margin = 0.040
    tabletop_arm_object_lift_gap_penalty_scale = 9200.0
    tabletop_object_carry_stall_penalty_scale = 6800.0
    tabletop_object_carry_stall_min_arm_progress = 0.055

    strict_opposition_touch_rew_scale = 52000.0
    opposition_rew_scale = 30000.0
    grasp_quality_rew_scale = 30000.0
    tabletop_underwrap_opposition_min_multiplier = 0.50


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StableHoldRewardOnlyCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StabilityCompat86TeacherEnvCfg
):
    """Reward-only continuation that preserves the working Relative075 lift controller."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_stablehold_rewardonly_compat86_teacher"
    )

    strict_opposition_touch_rew_scale = 52000.0
    opposition_rew_scale = 30000.0
    grasp_quality_rew_scale = 30000.0
    tabletop_stable_catch_rew_scale = 18000.0
    tabletop_object_carry_lift_rew_scale = 44000.0
    quality_lift_progress_rew_scale = 22000.0
    lifted_true_grasp_rew_scale = 60000.0
    lift_progress_rew_scale = 18000.0
    stable_hold_rew_scale = 62000.0
    hold_progress_rew_scale = 84000.0
    success_bonus = 160000.0
    stable_object_palm_vel = 0.22

    tabletop_no_lift_after_grasp_penalty_scale = 6500.0
    tabletop_no_lift_min_progress = 0.024
    tabletop_object_carry_stall_penalty_scale = 4800.0
    tabletop_object_carry_stall_min_arm_progress = 0.040
    tabletop_object_carry_stall_min_z_vel = 0.012
    tabletop_non_thumb_without_thumb_penalty_scale = 7600.0
    tabletop_underwrap_opposition_min_multiplier = 0.42


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbHoldCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StableHoldRewardOnlyCompat86TeacherEnvCfg
):
    """Strict-thumb continuation: reject non-thumb-only lifts and preserve opposition during carry."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strictthumb_hold_compat86_teacher"
    )

    strict_success_enabled = True
    strict_reward_enabled = True
    tabletop_lift_gate_requires_current_strict_grasp = True
    tabletop_lift_use_grasp_seen_gate = False
    tabletop_object_carry_uses_grasp_seen = False
    tabletop_object_carry_min_grasp_streak = 2
    tabletop_object_carry_streak_ramp_steps = 4

    lift_reward_uses_opposition_gate = True
    lift_reward_min_opposition_multiplier = 0.0
    quality_lift_progress_uses_opposition_gate = True
    quality_lift_progress_min_opposition_multiplier = 0.0
    tabletop_underwrap_opposition_min_multiplier = 0.62

    strict_opposition_approach_rew_scale = 18000.0
    strict_opposition_touch_rew_scale = 76000.0
    strict_touch_rew_scale = 2200.0
    opposition_rew_scale = 42000.0
    grasp_quality_rew_scale = 42000.0
    tabletop_stable_catch_rew_scale = 22000.0
    tabletop_object_carry_lift_rew_scale = 30000.0
    quality_lift_progress_rew_scale = 18000.0
    lifted_true_grasp_rew_scale = 76000.0
    lift_progress_rew_scale = 11000.0
    stable_hold_rew_scale = 72000.0
    hold_progress_rew_scale = 96000.0
    success_bonus = 190000.0

    tabletop_non_thumb_without_thumb_penalty_scale = 26000.0
    tabletop_non_thumb_without_thumb_penalty_lift_gate_min = 1.0
    tabletop_non_thumb_without_thumb_gate_start = 0.04
    tabletop_non_thumb_without_thumb_gate_ramp = 0.24
    tabletop_non_thumb_without_thumb_thumb_target = 0.34
    scoop_lift_penalty_scale = 9000.0
    palm_only_lift_penalty_scale = 9000.0

    tabletop_lift_without_object_penalty_scale = 9000.0
    tabletop_arm_object_lift_gap_penalty_scale = 9000.0
    tabletop_object_carry_stall_penalty_scale = 7000.0
    tabletop_no_lift_after_grasp_penalty_scale = 5200.0
    tabletop_no_lift_min_progress = 0.022

    dynamic_success_hold_steps = 6
    stable_object_palm_vel = 0.28
    tabletop_success_lift_height = 0.045
    tabletop_hover_success_object_speed = 0.20


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictLiftHoldCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbHoldCompat86TeacherEnvCfg
):
    """Strict lift-hold continuation: discourage lifting after the strict grasp is lost."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strictlift_hold_compat86_teacher"
    )

    lift_progress_rew_scale = 2500.0
    quality_lift_progress_rew_scale = 9000.0
    lifted_true_grasp_rew_scale = 125000.0
    tabletop_stable_catch_rew_scale = 52000.0
    tabletop_object_carry_lift_rew_scale = 58000.0
    stable_hold_rew_scale = 105000.0
    hold_progress_rew_scale = 150000.0
    success_bonus = 260000.0

    tabletop_lift_without_current_grasp_penalty_scale = 62000.0
    tabletop_lift_without_current_grasp_min_progress = 0.20
    tabletop_lift_without_current_grasp_ramp = 0.45
    tabletop_lift_without_object_penalty_scale = 14000.0
    tabletop_object_carry_stall_penalty_scale = 9000.0
    tabletop_arm_object_lift_gap_penalty_scale = 12000.0

    dynamic_success_hold_steps = 6
    stable_object_palm_vel = 0.30


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictPreLiftHoldCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbHoldCompat86TeacherEnvCfg
):
    """Keep the strict grasp alive during the close/settle phase before lift starts."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_pre_lift_hold_compat86_teacher"
    )

    tabletop_strict_hold_rew_scale = 52000.0
    tabletop_strict_grasp_loss_penalty_scale = 26000.0

    tabletop_object_carry_lift_rew_scale = 36000.0
    lifted_true_grasp_rew_scale = 90000.0
    tabletop_stable_catch_rew_scale = 34000.0
    stable_hold_rew_scale = 90000.0
    hold_progress_rew_scale = 125000.0
    success_bonus = 230000.0
    lift_progress_rew_scale = 8500.0
    quality_lift_progress_rew_scale = 16000.0

    tabletop_no_lift_after_grasp_penalty_scale = 6200.0
    tabletop_no_lift_after_grasp_grace_steps = 26
    tabletop_no_lift_after_grasp_ramp_steps = 70
    tabletop_object_carry_stall_penalty_scale = 7200.0
    stable_object_palm_vel = 0.28


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictTimedLiftCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbHoldCompat86TeacherEnvCfg
):
    """Strict-thumb continuation that asks the hand to lift soon after a stable grasp appears."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_timed_lift_compat86_teacher"
    )

    tabletop_object_carry_lift_rew_scale = 42000.0
    lifted_true_grasp_rew_scale = 88000.0
    tabletop_stable_catch_rew_scale = 30000.0
    stable_hold_rew_scale = 84000.0
    hold_progress_rew_scale = 112000.0
    success_bonus = 220000.0
    lift_progress_rew_scale = 14000.0
    quality_lift_progress_rew_scale = 22000.0

    tabletop_no_lift_after_grasp_penalty_scale = 7600.0
    tabletop_no_lift_after_grasp_grace_steps = 3
    tabletop_no_lift_after_grasp_ramp_steps = 12
    tabletop_no_lift_after_grasp_max_penalty = 8.0
    tabletop_no_lift_min_progress = 0.026

    tabletop_object_carry_stall_penalty_scale = 8200.0
    tabletop_object_carry_stall_min_arm_progress = 0.038
    tabletop_object_carry_stall_min_z_vel = 0.014
    stable_object_palm_vel = 0.28


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictEarlyLiftCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbHoldCompat86TeacherEnvCfg
):
    """Strict-thumb checkpoint probe with a faster relative lift target."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_early_lift_compat86_teacher"
    )

    scripted_action_prior_lift_steps = 180
    scripted_action_prior_active_residual_scale = 0.10
    tabletop_object_carry_lift_rew_scale = 38000.0
    lifted_true_grasp_rew_scale = 84000.0
    lift_progress_rew_scale = 13000.0
    quality_lift_progress_rew_scale = 20000.0
    stable_hold_rew_scale = 82000.0
    hold_progress_rew_scale = 108000.0
    success_bonus = 220000.0


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairBoostCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbHoldCompat86TeacherEnvCfg
):
    """Strict-thumb continuation that emphasizes thumb/non-thumb opposition before and during lift."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_boost_compat86_teacher"
    )

    contact_reward_requires_thumb_pair = True
    true_grasp_score_requires_thumb_pair = True
    thumb_contact_reward_weight = 0.95
    grasp_quality_thumb_weight = 0.55

    strict_touch_reward_requires_thumb_pair = True
    strict_touch_reward_uses_opposition_product = True
    strict_touch_reward_opposition_min_multiplier = 0.18
    strict_touch_rew_scale = 18000.0
    strict_opposition_approach_rew_scale = 30000.0
    strict_opposition_touch_rew_scale = 112000.0
    opposition_rew_scale = 52000.0
    grasp_quality_rew_scale = 52000.0

    tabletop_underwrap_opposition_min_multiplier = 0.78
    tabletop_object_carry_lift_rew_scale = 34000.0
    lifted_true_grasp_rew_scale = 90000.0
    tabletop_stable_catch_rew_scale = 32000.0
    stable_hold_rew_scale = 92000.0
    hold_progress_rew_scale = 124000.0
    success_bonus = 230000.0


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairCurrentLiftCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairBoostCompat86TeacherEnvCfg
):
    """PairBoost reward whose lift prior remains active only while the current strict grasp is alive."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_current_lift_compat86_teacher"
    )

    scripted_action_prior_lift_memory_requires_streak = True
    scripted_action_prior_lift_grasp_memory_min_steps = 1


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairCurrentLiftCompat86TeacherEnvCfg
):
    """Current-lift variant with a short strict-grasp latch to survive contact chatter during lift."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_recent_lift_compat86_teacher"
    )

    scripted_action_prior_lift_grasp_memory_min_steps = 1
    scripted_action_prior_lift_grasp_recent_steps = 12
    tabletop_lift_without_current_grasp_penalty_scale = 8000.0
    tabletop_lift_without_current_grasp_min_progress = 0.12
    tabletop_lift_without_current_grasp_ramp = 0.45


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLateLiftCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftCompat86TeacherEnvCfg
):
    """Recent-lift variant that delays arm lift so the Inspire thumb can settle around the ball first."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_recent_late_lift_compat86_teacher"
    )

    scripted_action_prior_lift_start_step = 124
    tabletop_no_lift_after_grasp_grace_steps = 36


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftCandidateProbeCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftCompat86TeacherEnvCfg
):
    """Current best Inspire rolling task, with one scripted lift candidate assigned per env."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_recent_liftcandidate_probe_compat86_teacher"
    )

    scripted_action_prior_lift_candidate_labels = INSPIRE_V340_LIFT_CANDIDATE_LABELS
    scripted_action_prior_lift_candidate_actions = INSPIRE_V340_LIFT_CANDIDATE_ACTIONS


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftHandMemoryCandidateProbeCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftCompat86TeacherEnvCfg
):
    """Current best Inspire rolling task, with one hand-memory action vector assigned per env."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_recent_lift_handmemory_probe_compat86_teacher"
    )

    scripted_tabletop_hand_grasp_memory_action_candidate_labels = (
        INSPIRE_V340_HAND_MEMORY_CANDIDATE_LABELS
    )
    scripted_tabletop_hand_grasp_memory_action_candidates = INSPIRE_V340_HAND_MEMORY_CANDIDATE_ACTIONS


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftHandAll100Compat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftCompat86TeacherEnvCfg
):
    """Current best Inspire rolling task with fully closed 6-DOF hand-memory hold."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_recent_lift_handall100_compat86_teacher"
    )

    scripted_tabletop_hand_grasp_memory_action_vector = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftCompat86TeacherEnvCfg
):
    """Recent-lift probe that holds the 6-DOF hand as soon as the first strict grasp appears."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_recent_lift_earlyhand_compat86_teacher"
    )

    scripted_tabletop_hand_grasp_memory_prior_start_step = 56
    scripted_tabletop_hand_grasp_memory_prior_steps = 360
    scripted_tabletop_hand_grasp_memory_min_steps = 1
    scripted_tabletop_hand_grasp_memory_action_vector = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    scripted_tabletop_hand_grasp_memory_ramp_steps = 8


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandMixedSpeedInterceptCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandCompat86TeacherEnvCfg
):
    """Early-hand Inspire rolling task with mixed-speed reset sampling and a longer interception lead."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_recent_lift_"
        "earlyhand_mixedspeed_intercept_compat86_teacher"
    )

    dynamic_grasp_speed_curriculum_override_alpha = 1.0
    dynamic_tabletop_speed_alpha_sample_enabled = True
    dynamic_tabletop_speed_alpha_sample_min = 0.0
    dynamic_tabletop_speed_alpha_sample_max = 1.0
    dynamic_tabletop_speed_alpha_sample_curriculum_cap = True
    dynamic_tabletop_speed_alpha_sample_full_fraction = 0.35

    dynamic_tabletop_pregrasp_lead_time = 0.34
    dynamic_tabletop_pregrasp_ahead_distance = 0.09
    dynamic_tabletop_pregrasp_xy_distance_scale = 0.22
    dynamic_tabletop_pregrasp_xy_rew_scale = 420.0
    dynamic_tabletop_pregrasp_height_offset = 0.130
    dynamic_tabletop_pregrasp_height_scale = 0.045
    dynamic_tabletop_pregrasp_height_rew_scale = 540.0


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandMixedSpeedInterceptEvalCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandMixedSpeedInterceptCompat86TeacherEnvCfg
):
    """Matched-intercept evaluation task with an exact CLI-selected speed band."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_recent_lift_"
        "earlyhand_mixedspeed_intercept_eval_compat86_teacher"
    )

    dynamic_tabletop_speed_alpha_sample_enabled = False


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandRelativeLiftUnlockedCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandMixedSpeedInterceptCompat86TeacherEnvCfg
):
    """Use the latched relative lift target after the first strict grasp."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_recent_lift_"
        "earlyhand_relative_lift_unlocked_compat86_teacher"
    )

    # The one-step strict contact can disappear before the streak cache is
    # updated.  Keep the existing recent-grasp latch, but let grasp_seen unlock
    # the verified relative lift target instead of falling back to joint-6-only.
    scripted_action_prior_lift_memory_requires_streak = False
    scripted_action_prior_lift_grasp_memory_min_steps = 1

    tabletop_post_success_stability_latch_enabled = True
    tabletop_post_success_arm_target_lock_enabled = True
    tabletop_post_success_arm_target_lock_blend = 1.0
    tabletop_post_success_hand_target_lock_enabled = True
    tabletop_post_success_hand_target_lock_blend = 1.0
    tabletop_post_success_hand_close_fraction = 0.08


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandRelativeLiftUnlockedEvalCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandRelativeLiftUnlockedCompat86TeacherEnvCfg
):
    """Exact-speed evaluation variant of the relative-lift-unlocked task."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_recent_lift_"
        "earlyhand_relative_lift_unlocked_eval_compat86_teacher"
    )

    dynamic_tabletop_speed_alpha_sample_enabled = False


@configclass
class InspireRollingRelativeLiftScaleProbeTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandRelativeLiftUnlockedEvalCompat86TeacherEnvCfg
):
    """Assign four relative-lift amplitudes across environments for one-variable screening."""

    reference_name = "inspire_rolling_relative_lift_scale_probe_teacher"
    scripted_tabletop_relative_lift_target_candidate_labels = INSPIRE_V340_RELATIVE_LIFT_SCALE_PROBE_LABELS
    scripted_tabletop_relative_lift_target_candidate_deltas = INSPIRE_V340_RELATIVE_LIFT_SCALE_PROBE_DELTAS


@configclass
class InspireRollingRelativeLiftHighScaleProbeTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandRelativeLiftUnlockedEvalCompat86TeacherEnvCfg
):
    """Screen larger relative-lift amplitudes after the first probe's monotonic gain."""

    reference_name = "inspire_rolling_relative_lift_high_scale_probe_teacher"
    scripted_tabletop_relative_lift_target_candidate_labels = INSPIRE_V340_RELATIVE_LIFT_HIGH_SCALE_PROBE_LABELS
    scripted_tabletop_relative_lift_target_candidate_deltas = INSPIRE_V340_RELATIVE_LIFT_HIGH_SCALE_PROBE_DELTAS


@configclass
class InspireRollingRelativeLiftScale160TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandRelativeLiftUnlockedEvalCompat86TeacherEnvCfg
):
    """Full-speed Inspire rolling continuation with the verified 1.60x lift amplitude."""

    reference_name = "inspire_rolling_relative_lift_scale160_teacher"
    scripted_tabletop_relative_lift_target_arm_delta = tuple(
        1.60 * value for value in INSPIRE_V340_RELATIVE_LIFT_TARGET_DELTAS[2]
    )


@configclass
class InspireRollingRelativeLiftScale200PostHoldTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandRelativeLiftUnlockedEvalCompat86TeacherEnvCfg
):
    """Full-speed Inspire continuation that trains the load-bearing state after lift success."""

    reference_name = "inspire_rolling_relative_lift_scale200_posthold_teacher"
    scripted_tabletop_relative_lift_target_arm_delta = tuple(
        2.00 * value for value in INSPIRE_V340_RELATIVE_LIFT_TARGET_DELTAS[2]
    )

    # The parent task terminates on the first six-step success streak. Keeping
    # the episode alive exposes the same sustained grasp contract used by video
    # evaluation, while target locks prevent post-success arm thrashing.
    terminate_on_success = False
    episode_length_s = 8.0
    tabletop_post_success_stability_latch_enabled = True
    tabletop_post_success_arm_target_lock_enabled = True
    tabletop_post_success_arm_target_lock_blend = 1.0
    tabletop_post_success_hand_target_lock_enabled = True
    tabletop_post_success_hand_target_lock_blend = 1.0
    tabletop_post_success_hand_lock_uses_actual_joint_pos = True
    tabletop_post_success_hand_close_fraction = 0.08

    tabletop_post_success_hold_rew_scale = 180000.0
    tabletop_post_success_unstable_penalty_scale = 150000.0
    tabletop_post_success_grasp_loss_penalty_scale = 140000.0
    tabletop_post_success_under_height_penalty_scale = 26000.0
    tabletop_post_success_speed_penalty_scale = 12000.0
    tabletop_post_success_action_penalty_scale = 0.050
    tabletop_post_success_target_delta_penalty_scale = 0.080
    tabletop_post_success_arm_joint_vel_penalty_scale = 500.0
    tabletop_post_success_arm_target_drift_penalty_scale = 18000.0
    tabletop_post_success_arm_target_drift_tolerance = 0.10
    tabletop_post_success_arm_target_drift_scale = 0.30
    tabletop_post_success_palm_drift_penalty_scale = 28000.0
    tabletop_post_success_palm_drift_tolerance = 0.040
    tabletop_post_success_palm_drift_scale = 0.090


@configclass
class InspireRollingRelativeLiftScale200PostHoldTargetHandLockTeacherEnvCfg(
    InspireRollingRelativeLiftScale200PostHoldTeacherEnvCfg
):
    """Hold the successful commanded hand target instead of its contact-deflected pose."""

    reference_name = "inspire_rolling_relative_lift_scale200_posthold_target_hand_lock_teacher"
    tabletop_post_success_hand_lock_uses_actual_joint_pos = False


@configclass
class InspireRollingRelativeLiftScale200PostHoldSphere60TeacherEnvCfg(
    InspireRollingRelativeLiftScale200PostHoldTeacherEnvCfg
):
    """Matched 86-D diagnostic that changes only the rolling sphere to 60 mm."""

    reference_name = "inspire_rolling_relative_lift_scale200_posthold_sphere60_teacher"
    object_start_pos = (
        0.58,
        0.0,
        _tabletop_start_z_from_spec(TABLETOP_INSPIRE_SPHERE_60MM_HIGH_FRICTION_SPEC),
    )
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_INSPIRE_SPHERE_60MM_HIGH_FRICTION_SPEC,
        pos=object_start_pos,
    )
    object_shape = "sphere"
    object_radius = float(TABLETOP_INSPIRE_SPHERE_60MM_HIGH_FRICTION_SPEC["radius"])
    object_size = tuple(TABLETOP_INSPIRE_SPHERE_60MM_HIGH_FRICTION_SPEC["size"])
    tabletop_object_asset_specs = (TABLETOP_INSPIRE_SPHERE_60MM_HIGH_FRICTION_SPEC,)


@configclass
class InspireRollingRelativeLiftScale200PostHoldSphereSafeCloseTeacherEnvCfg(
    InspireRollingRelativeLiftScale200PostHoldTeacherEnvCfg
):
    """Matched diagnostic using the official sphere-safe coupled close posture."""

    reference_name = "inspire_rolling_relative_lift_scale200_posthold_sphere_safe_close_teacher"
    inspire_semantic_close_targets = INSPIRE_ANYDEX_SPHERE_SAFE_CLOSE_TARGETS


@configclass
class InspireRollingRelativeLiftScale200PostHoldMildThumbWrapCloseTeacherEnvCfg(
    InspireRollingRelativeLiftScale200PostHoldTeacherEnvCfg
):
    """Matched diagnostic using a modestly stronger load-bearing thumb wrap."""

    reference_name = "inspire_rolling_relative_lift_scale200_posthold_mild_thumb_wrap_close_teacher"
    inspire_semantic_close_targets = INSPIRE_ANYDEX_SPHERE_MILD_THUMB_WRAP_CLOSE_TARGETS


@configclass
class InspireRollingRelativeLiftScale200PostHoldMildThumbWrapTargetHandLockTeacherEnvCfg(
    InspireRollingRelativeLiftScale200PostHoldTargetHandLockTeacherEnvCfg
):
    """Combine the verified thumb wrap with command-target post-success hand locking."""

    reference_name = (
        "inspire_rolling_relative_lift_scale200_posthold_"
        "mild_thumb_wrap_target_hand_lock_teacher"
    )
    inspire_semantic_close_targets = INSPIRE_ANYDEX_SPHERE_MILD_THUMB_WRAP_CLOSE_TARGETS


@configclass
class InspireUnifiedRollingBenchmarkTeacherEnvCfg(
    _UnifiedRollingRewardContract,
    InspireRollingRelativeLiftScale200PostHoldMildThumbWrapTargetHandLockTeacherEnvCfg
):
    """Inspire RH56 adapter for the shared multi-shape rolling benchmark."""

    reference_name = "inspire_unified_rolling_multishape_v1_teacher"
    benchmark_protocol = UNIFIED_ROLLING_BENCHMARK_NAME
    # The RH56 hand is longer than Revo2. At the Franka default home pose its
    # open fingertips penetrate the tabletop and PhysX drives several hand
    # joints far outside their limits before the first policy action. Keep the
    # same upright reset family, but use the previously validated table-clear
    # Inspire pose and its calibrated upward Franka-j6 motion.
    robot_cfg: ArticulationCfg = _inspire_z180_robot_cfg(INSPIRE_V341_CLEAR_ARM_POS)
    default_arm_pos = INSPIRE_V341_CLEAR_ARM_POS
    scripted_tabletop_pregrasp_arm_pos = INSPIRE_V341_CLEAR_ARM_POS
    tabletop_arm_lift_progress_baseline_pos = INSPIRE_V341_CLEAR_ARM_POS
    lift_arm_delta = INSPIRE_V340_LIFT_ARM_DELTA
    lift_action_prior = INSPIRE_V340_LIFT_ACTION_PRIOR
    scripted_tabletop_lift_target_arm_delta = INSPIRE_V340_LIFT_ARM_DELTA
    scripted_tabletop_relative_lift_target_arm_delta = INSPIRE_V340_LIFT_ARM_DELTA
    scripted_action_prior_lift_action = INSPIRE_V340_LIFT_ACTION_PRIOR

    # Use the same safety samples as Revo2: Franka link origins plus the
    # embodiment's actual palm and fingertip contact points. A scalar margin at
    # every RH56 finger-link origin treats link length as vertical thickness and
    # falsely rejects valid horizontal wraps; PhysX still enforces every real
    # finger/table collision shape.
    tabletop_arm_clearance_body_names = (
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_link8",
        "panda_hand",
    )
    tabletop_arm_clearance_body_margins = ()

    observation_space = 86
    tabletop_object_asset_specs = UNIFIED_ROLLING_OBJECT_SPECS
    tabletop_asset_set_enabled = True
    tabletop_asset_obs_enabled = True
    tabletop_asset_sampling_weights = None
    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_mode = "dynamic_speed"
    tabletop_asset_curriculum_start_count = 1
    tabletop_asset_curriculum_steps = UNIFIED_ROLLING_ASSET_CURRICULUM_STEPS
    tabletop_asset_curriculum_override_alpha = None

    object_start_pos = UNIFIED_ROLLING_START_POS
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=object_start_pos,
    )
    object_shape = str(TABLETOP_ROLLING_START_SPEC["proxy_shape"])
    object_radius = float(TABLETOP_ROLLING_START_SPEC["radius"])
    object_size = tuple(TABLETOP_ROLLING_START_SPEC["size"])
    reset_object_pos_noise = UNIFIED_ROLLING_RESET_OBJECT_POS_NOISE

    tabletop_motion_modes = ("free",)
    tabletop_motion_mode_curriculum = False
    dynamic_tabletop_persistent_motion = False
    dynamic_tabletop_bounce_at_workspace = False
    dynamic_tabletop_release_motion_on_contact = False
    dynamic_tabletop_randomize_yaw = True
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = UNIFIED_ROLLING_CURRICULUM_METRIC
    dynamic_grasp_speed_curriculum_start_success = UNIFIED_ROLLING_CURRICULUM_START_SUCCESS
    dynamic_grasp_speed_curriculum_full_success = UNIFIED_ROLLING_CURRICULUM_FULL_SUCCESS
    dynamic_grasp_speed_curriculum_ema_alpha = UNIFIED_ROLLING_CURRICULUM_EMA_ALPHA
    dynamic_grasp_speed_curriculum_alpha_rise = UNIFIED_ROLLING_CURRICULUM_ALPHA_RISE
    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_grasp_speed_curriculum_steps = UNIFIED_ROLLING_SPEED_CURRICULUM_STEPS
    dynamic_grasp_speed_curriculum_override_alpha = None
    dynamic_tabletop_speed_alpha_sample_enabled = False
    dynamic_tabletop_start_speed_range = UNIFIED_ROLLING_START_SPEED_RANGE
    dynamic_tabletop_initial_speed_range = UNIFIED_ROLLING_TARGET_SPEED_RANGE
    dynamic_tabletop_start_yaw_rate_range = UNIFIED_ROLLING_START_YAW_RATE_RANGE
    dynamic_tabletop_initial_yaw_rate_range = UNIFIED_ROLLING_TARGET_YAW_RATE_RANGE
    dynamic_tabletop_pregrasp_lead_time = 0.36
    dynamic_tabletop_pregrasp_ahead_distance = 0.10
    dynamic_tabletop_pregrasp_ready_distance = 0.20

    strict_success_enabled = True
    strict_reward_enabled = True
    strict_success_contact_distance = UNIFIED_ROLLING_STRICT_CONTACT_DISTANCE
    strict_success_min_finger_contacts = UNIFIED_ROLLING_STRICT_MIN_FINGER_CONTACTS
    strict_success_min_non_thumb_contacts = UNIFIED_ROLLING_STRICT_MIN_NON_THUMB_CONTACTS
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    contact_distance = UNIFIED_ROLLING_CONTACT_DISTANCE
    contact_score_scale = UNIFIED_ROLLING_CONTACT_SCORE_SCALE
    min_finger_contacts = UNIFIED_ROLLING_STRICT_MIN_FINGER_CONTACTS
    min_non_thumb_contacts = UNIFIED_ROLLING_STRICT_MIN_NON_THUMB_CONTACTS
    opposition_cos_threshold = 0.0
    tabletop_success_requires_hover_target = False
    tabletop_success_lift_height = UNIFIED_ROLLING_SUCCESS_LIFT_HEIGHT
    dynamic_success_hold_steps = UNIFIED_ROLLING_SUCCESS_HOLD_STEPS
    stable_object_palm_vel = UNIFIED_ROLLING_STABLE_OBJECT_PALM_VEL
    tabletop_hover_height_delta = UNIFIED_ROLLING_HOVER_HEIGHT_DELTA
    tabletop_hover_latch_lift_progress = UNIFIED_ROLLING_HOVER_LATCH_LIFT_PROGRESS
    tabletop_hover_xy_distance_scale = UNIFIED_ROLLING_HOVER_XY_DISTANCE_SCALE
    tabletop_hover_z_distance_scale = UNIFIED_ROLLING_HOVER_Z_DISTANCE_SCALE
    tabletop_hover_object_speed_scale = UNIFIED_ROLLING_HOVER_OBJECT_SPEED_SCALE
    tabletop_hover_ang_speed_scale = UNIFIED_ROLLING_HOVER_ANG_SPEED_SCALE
    tabletop_hover_success_requires_xy = True
    tabletop_hover_success_xy_tolerance = UNIFIED_ROLLING_HOVER_SUCCESS_XY_TOLERANCE
    tabletop_hover_success_z_tolerance = UNIFIED_ROLLING_HOVER_SUCCESS_Z_TOLERANCE
    tabletop_hover_success_object_speed = UNIFIED_ROLLING_HOVER_SUCCESS_OBJECT_SPEED
    tabletop_success_requires_arm_clearance = True
    tabletop_terminate_on_arm_clearance_violation = False

    terminate_on_success = False
    episode_length_s = UNIFIED_ROLLING_EPISODE_LENGTH_S
    tabletop_post_success_stability_latch_enabled = True
    tabletop_post_success_arm_target_lock_enabled = True
    tabletop_post_success_arm_target_lock_blend = 1.0
    tabletop_post_success_hand_target_lock_enabled = True
    tabletop_post_success_hand_target_lock_blend = 1.0
    tabletop_post_success_hand_lock_uses_actual_joint_pos = False
    tabletop_post_success_hand_close_fraction = 0.08
    affordance_label_mode = "tabletop_rolling_assets"


@configclass
class InspireUnifiedRollingStage1TeacherEnvCfg(InspireUnifiedRollingBenchmarkTeacherEnvCfg):
    """Inspire adapter for the shared home-to-pregrasp bootstrap stage."""

    reference_name = "inspire_unified_rolling_multishape_v1_stage1_teacher"
    contact_rew_scale = 80.0
    dynamic_tabletop_pregrasp_height_rew_scale = 44.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 420.0
    fingertip_reach_rew_scale = 24.0
    grasp_quality_rew_scale = 500.0
    opposition_rew_scale = 200.0
    palm_reach_rew_scale = 24.0
    pregrasp_rew_scale = 4.0
    strict_approach_rew_scale = 80.0
    strict_multifinger_approach_rew_scale = 160.0
    strict_opposition_approach_rew_scale = 300.0
    strict_opposition_touch_rew_scale = 1200.0
    strict_touch_rew_scale = 400.0
    tabletop_non_thumb_without_thumb_penalty_scale = 300.0
    true_grasp_rew_scale = 500.0
    strict_touch_score_scale = 0.008


@configclass
class InspireUnifiedRollingStage2HoldTeacherEnvCfg(
    _UnifiedRollingGraspHoldStage2Contract,
    InspireUnifiedRollingBenchmarkTeacherEnvCfg,
):
    """Inspire continuation stage for a sustained strict grasp before lift."""

    reference_name = "inspire_unified_rolling_multishape_v1_stage2_hold_teacher"


@configclass
class InspireUnifiedRollingStage3TeacherEnvCfg(
    _UnifiedRollingLiftHoldStage3Contract,
    InspireUnifiedRollingBenchmarkTeacherEnvCfg,
):
    """Inspire adapter for the shared strict lift-and-hold continuation stage."""

    reference_name = "inspire_unified_rolling_multishape_v1_stage3_lift_hold_teacher"


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandHighSpeedFocusInterceptCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandMixedSpeedInterceptCompat86TeacherEnvCfg
):
    """Matched-intercept continuation concentrated on the full-speed rolling band."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_recent_lift_"
        "earlyhand_highspeed_focus_intercept_compat86_teacher"
    )

    dynamic_tabletop_speed_alpha_sample_min = 0.75
    dynamic_tabletop_speed_alpha_sample_max = 1.0
    dynamic_tabletop_speed_alpha_sample_full_fraction = 0.75


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandFastTailInterceptCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandMixedSpeedInterceptCompat86TeacherEnvCfg
):
    """Matched-intercept continuation focused on the difficult 0.25-0.40 m/s tail."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_recent_lift_"
        "earlyhand_fasttail_intercept_compat86_teacher"
    )

    dynamic_tabletop_speed_alpha_sample_enabled = False
    dynamic_tabletop_start_speed_range = (0.25, 0.40)
    dynamic_tabletop_initial_speed_range = (0.25, 0.40)


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairLiftHoldCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairBoostCompat86TeacherEnvCfg
):
    """Strict thumb-pair continuation that penalizes losing thumb/opposition contact during lift."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_lifthold_compat86_teacher"
    )

    tabletop_strict_hold_rew_scale = 28000.0
    tabletop_strict_grasp_loss_penalty_scale = 12000.0

    tabletop_lift_without_current_grasp_penalty_scale = 34000.0
    tabletop_lift_without_current_grasp_min_progress = 0.08
    tabletop_lift_without_current_grasp_ramp = 0.42

    tabletop_object_carry_lift_rew_scale = 42000.0
    quality_lift_progress_rew_scale = 22000.0
    lifted_true_grasp_rew_scale = 108000.0
    tabletop_stable_catch_rew_scale = 38000.0
    stable_hold_rew_scale = 110000.0
    hold_progress_rew_scale = 150000.0
    success_bonus = 270000.0

    tabletop_object_carry_stall_penalty_scale = 7600.0
    tabletop_no_lift_after_grasp_penalty_scale = 5800.0
    stable_object_palm_vel = 0.30


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbWrapPairBoostCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairBoostCompat86TeacherEnvCfg
):
    """Strict thumb-pair continuation with a stronger official thumb-wrap close envelope."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbwrap_pairboost_compat86_teacher"
    )

    inspire_semantic_close_targets = INSPIRE_ANYDEX_P80_THUMB_WRAP_CLOSE_TARGETS
    reference_hand_fractions = (1.0, 1.0, 0.80, 0.80, 1.0, 1.0)

    tabletop_underwrap_opposition_min_multiplier = 0.70
    tabletop_object_carry_lift_rew_scale = 36000.0
    lifted_true_grasp_rew_scale = 98000.0
    stable_hold_rew_scale = 100000.0
    hold_progress_rew_scale = 138000.0
    success_bonus = 250000.0


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictMildThumbWrapPairBoostCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairBoostCompat86TeacherEnvCfg
):
    """Strict thumb-pair reward with a mild thumb-wrap close envelope for lift retention."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_mildthumbwrap_pairboost_compat86_teacher"
    )

    inspire_semantic_close_targets = INSPIRE_ANYDEX_SPHERE_MILD_THUMB_WRAP_CLOSE_TARGETS
    reference_hand_fractions = (1.0, 1.0, 0.82, 0.82, 0.98, 0.98)
    tabletop_underwrap_opposition_min_multiplier = 0.74


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairSlowLiftCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairBoostCompat86TeacherEnvCfg
):
    """PairBoost reward with a slower, lower relative lift prior to reduce ball slip during carry."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_"
        "softunderwrap_liftpush_relative075_strict_thumbpair_slowlift_compat86_teacher"
    )

    scripted_tabletop_relative_lift_target_arm_delta = INSPIRE_V340_RELATIVE_LIFT_TARGET_DELTAS[1]
    scripted_action_prior_lift_start_step = 96
    scripted_action_prior_lift_steps = 420
    scripted_action_prior_active_residual_scale = 0.07
    scripted_action_prior_lift_grasp_memory_min_steps = 4
    scripted_action_prior_lift_memory_requires_streak = False

    scripted_tabletop_hand_grasp_memory_prior_start_step = 72
    scripted_tabletop_hand_grasp_memory_prior_steps = 460
    scripted_tabletop_hand_grasp_memory_min_steps = 3
    scripted_tabletop_hand_grasp_memory_action = 1.0
    scripted_tabletop_hand_grasp_memory_ramp_steps = 10


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPriorProbeCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapCompat86TeacherEnvCfg
):
    """86-observation probe: learned soft-underwrap grasp, then scripted lift/hold."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_softunderwrap_liftprior_probe_compat86_teacher"
    )

    scripted_action_prior_enabled = True
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.0
    scripted_action_prior_uses_strict_grasp = True

    scripted_action_prior_hand_start_step = 9999
    scripted_action_prior_hand_ramp_steps = 0
    scripted_action_prior_hand_action = 0.0
    scripted_action_prior_lift_start_step = 84
    scripted_action_prior_lift_steps = 360
    scripted_action_prior_lift_action = INSPIRE_V340_LIFT_ACTION_PRIOR
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_memory_requires_streak = False
    scripted_action_prior_lift_grasp_memory_min_steps = 8

    scripted_tabletop_hand_grasp_memory_prior_enabled = True
    scripted_tabletop_hand_grasp_memory_prior_start_step = 84
    scripted_tabletop_hand_grasp_memory_prior_steps = 360
    scripted_tabletop_hand_grasp_memory_min_steps = 8
    scripted_tabletop_hand_grasp_memory_action = 0.95
    scripted_tabletop_hand_grasp_memory_ramp_steps = 0


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapRelativeLiftCandidateProbeCompat86TeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPriorProbeCompat86TeacherEnvCfg
):
    """86-observation probe: learned soft-underwrap grasp, then scan relative lift targets."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_softunderwrap_"
        "relative_liftcandidate_probe_compat86_teacher"
    )

    scripted_tabletop_relative_lift_target_prior_enabled = True
    scripted_tabletop_relative_lift_target_arm_delta = INSPIRE_P80_HOME_SEED_LIFT_DELTA
    scripted_tabletop_relative_lift_target_candidate_labels = INSPIRE_V340_RELATIVE_LIFT_TARGET_LABELS
    scripted_tabletop_relative_lift_target_candidate_deltas = INSPIRE_V340_RELATIVE_LIFT_TARGET_DELTAS
    scripted_action_prior_lift_steps = 260
    tabletop_terminate_on_arm_clearance_violation = False


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmSoftContactTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceTeacherEnvCfg
):
    """Clean-reset Inspire rolling task with a 5 cm sphere and wider contact offset."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_softcontact_teacher"
    )

    object_start_pos = (0.58, 0.0, _tabletop_start_z_from_spec(TABLETOP_INSPIRE_SPHERE_50MM_SOFT_CONTACT_SPEC))
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_INSPIRE_SPHERE_50MM_SOFT_CONTACT_SPEC,
        pos=object_start_pos,
    )
    object_shape = "sphere"
    object_radius = float(TABLETOP_INSPIRE_SPHERE_50MM_SOFT_CONTACT_SPEC["radius"])
    object_size = tuple(TABLETOP_INSPIRE_SPHERE_50MM_SOFT_CONTACT_SPEC["size"])
    tabletop_object_asset_specs = (TABLETOP_INSPIRE_SPHERE_50MM_SOFT_CONTACT_SPEC,)
    tabletop_asset_set_enabled = False
    tabletop_asset_obs_enabled = False
    tabletop_asset_curriculum = False


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionLiftPriorProbeTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionTeacherEnvCfg
):
    """Physics probe: let the learned policy grasp, then script only lift-and-hold."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_liftprior_probe_teacher"
    )

    scripted_action_prior_enabled = True
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_residual_scale = 1.0
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.0
    scripted_action_prior_uses_strict_grasp = True

    scripted_action_prior_hand_start_step = 9999
    scripted_action_prior_hand_ramp_steps = 0
    scripted_action_prior_hand_action = 0.0
    scripted_action_prior_lift_start_step = 84
    scripted_action_prior_lift_steps = 360
    scripted_action_prior_lift_action = INSPIRE_V340_LIFT_ACTION_PRIOR
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_memory_requires_streak = False
    scripted_action_prior_lift_grasp_memory_min_steps = 8

    scripted_tabletop_hand_grasp_memory_prior_enabled = True
    scripted_tabletop_hand_grasp_memory_prior_start_step = 84
    scripted_tabletop_hand_grasp_memory_prior_steps = 360
    scripted_tabletop_hand_grasp_memory_min_steps = 8
    scripted_tabletop_hand_grasp_memory_action = 0.95
    scripted_tabletop_hand_grasp_memory_ramp_steps = 0


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionLiftCandidateProbeTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionLiftPriorProbeTeacherEnvCfg
):
    """Physics probe: assign one scripted lift candidate per env after learned grasp."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_liftcandidate_probe_teacher"
    )

    scripted_action_prior_lift_candidate_labels = INSPIRE_V340_LIFT_CANDIDATE_LABELS
    scripted_action_prior_lift_candidate_actions = INSPIRE_V340_LIFT_CANDIDATE_ACTIONS


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionRelativeLiftCandidateProbeTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionLiftPriorProbeTeacherEnvCfg
):
    """Physics probe: learned grasp, then track a relative arm-joint lift target per env."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere50mm_highfriction_relative_liftcandidate_probe_teacher"
    )

    scripted_tabletop_relative_lift_target_prior_enabled = True
    scripted_tabletop_relative_lift_target_arm_delta = INSPIRE_P80_HOME_SEED_LIFT_DELTA
    scripted_tabletop_relative_lift_target_candidate_labels = INSPIRE_V340_RELATIVE_LIFT_TARGET_LABELS
    scripted_tabletop_relative_lift_target_candidate_deltas = INSPIRE_V340_RELATIVE_LIFT_TARGET_DELTAS
    scripted_action_prior_lift_steps = 260


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere60mmHighFrictionRelativeLiftCandidateProbeTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionRelativeLiftCandidateProbeTeacherEnvCfg
):
    """Relative lift probe with a larger 6 cm high-friction sphere."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_sphere60mm_highfriction_relative_liftcandidate_probe_teacher"
    )

    object_start_pos = (0.58, 0.0, _tabletop_start_z_from_spec(TABLETOP_INSPIRE_SPHERE_60MM_HIGH_FRICTION_SPEC))
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_INSPIRE_SPHERE_60MM_HIGH_FRICTION_SPEC,
        pos=object_start_pos,
    )
    object_shape = "sphere"
    object_radius = float(TABLETOP_INSPIRE_SPHERE_60MM_HIGH_FRICTION_SPEC["radius"])
    object_size = tuple(TABLETOP_INSPIRE_SPHERE_60MM_HIGH_FRICTION_SPEC["size"])
    tabletop_object_asset_specs = (TABLETOP_INSPIRE_SPHERE_60MM_HIGH_FRICTION_SPEC,)
    tabletop_asset_set_enabled = False
    tabletop_asset_obs_enabled = False
    tabletop_asset_curriculum = False


@configclass
class InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceCanHighFrictionRelativeLiftCandidateProbeTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionRelativeLiftCandidateProbeTeacherEnvCfg
):
    """Relative lift probe with the rolling can/cylinder asset."""

    reference_name = (
        "inspire_z180_dynamic_tabletop_rolling_sphere_p80_fast_curriculum_"
        "loosereward_liftguide_cleanreset_clearance_can_highfriction_relative_liftcandidate_probe_teacher"
    )

    object_start_pos = (0.58, 0.0, _tabletop_start_z_from_spec(TABLETOP_INSPIRE_CAN_HIGH_FRICTION_SPEC))
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_INSPIRE_CAN_HIGH_FRICTION_SPEC,
        pos=object_start_pos,
    )
    object_shape = str(TABLETOP_INSPIRE_CAN_HIGH_FRICTION_SPEC["proxy_shape"])
    object_radius = float(TABLETOP_INSPIRE_CAN_HIGH_FRICTION_SPEC["radius"])
    object_size = tuple(TABLETOP_INSPIRE_CAN_HIGH_FRICTION_SPEC["size"])
    tabletop_object_asset_specs = (TABLETOP_INSPIRE_CAN_HIGH_FRICTION_SPEC,)
    tabletop_asset_set_enabled = False
    tabletop_asset_obs_enabled = False
    tabletop_asset_curriculum = False


@configclass
class InspireDynamicTabletopRollingSphereP80HomeSeedBootstrapTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereApproachBootstrapTeacherEnvCfg
):
    """Inspire rolling teacher bootstrap from a verified P80 home-pose lift seed."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_p80_home_seed_bootstrap_teacher"
    observation_space = 76

    robot_cfg: ArticulationCfg = _inspire_z180_robot_cfg(ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS)
    default_arm_pos = ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS
    episode_length_s = 8.0

    object_start_pos = (0.58, 0.0, TABLETOP_ROLLING_START_Z)
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=object_start_pos,
    )
    object_shape = str(TABLETOP_ROLLING_START_SPEC["proxy_shape"])
    object_radius = float(TABLETOP_ROLLING_START_SPEC["radius"])
    object_size = tuple(TABLETOP_ROLLING_START_SPEC["size"])
    tabletop_object_asset_specs = (TABLETOP_ROLLING_START_SPEC,)
    tabletop_asset_set_enabled = False
    tabletop_asset_obs_enabled = False
    tabletop_asset_curriculum = False
    reset_object_pos_noise = (0.0, 0.0, 0.0)

    inspire_semantic_close_targets = INSPIRE_ANYDEX_P80_CLOSE_TARGETS
    reference_hand_fractions = INSPIRE_P80_HOME_SEED_HAND_FRACTIONS

    scripted_action_prior_enabled = True
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_residual_scale = 0.20
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.10

    scripted_tabletop_pregrasp_prior_enabled = True
    scripted_tabletop_pregrasp_prior_control_mode = "target_track"
    scripted_tabletop_pregrasp_arm_pos = INSPIRE_P80_HOME_SEED_ARM_POS
    scripted_tabletop_pregrasp_prior_start_step = 0
    scripted_tabletop_pregrasp_prior_steps = 260
    scripted_tabletop_pregrasp_prior_ramp_steps = 1
    tabletop_arm_lift_progress_baseline_pos = INSPIRE_P80_HOME_SEED_ARM_POS

    scripted_action_prior_hand_start_step = 150
    scripted_action_prior_hand_ramp_steps = 110
    scripted_action_prior_hand_action = 1.0

    scripted_tabletop_lift_target_prior_enabled = True
    scripted_tabletop_lift_target_arm_delta = INSPIRE_P80_HOME_SEED_LIFT_DELTA
    scripted_tabletop_lift_target_prior_ramp_steps = 1
    scripted_action_prior_lift_start_step = 260
    scripted_action_prior_lift_steps = 190
    scripted_action_prior_lift_requires_grasp = False
    scripted_action_prior_lift_uses_grasp_memory = False
    scripted_action_prior_lift_memory_requires_streak = False
    scripted_action_prior_lift_grasp_memory_min_steps = 0

    lift_arm_delta = INSPIRE_P80_HOME_SEED_LIFT_DELTA
    lift_action_prior = V325_VERIFIED_LIFT_ACTION_PRIOR_120
    scripted_action_prior_lift_action = V325_VERIFIED_LIFT_ACTION_PRIOR_120

    quality_lift_progress_rew_scale = 4600.0
    lifted_true_grasp_rew_scale = 9800.0
    lift_progress_rew_scale = 4600.0
    tabletop_grasped_palm_lift_rew_scale = 4200.0
    tabletop_grasped_arm_lift_rew_scale = 3600.0
    tabletop_stable_catch_rew_scale = 3800.0
    tabletop_object_up_vel_rew_scale = 5200.0
    tabletop_object_carry_lift_rew_scale = 9800.0
    stable_hold_rew_scale = 11000.0
    hold_progress_rew_scale = 17000.0
    success_bonus = 42000.0

    tabletop_success_lift_height = 0.035
    dynamic_success_hold_steps = 8
    stable_object_palm_vel = 0.36
    tabletop_hover_success_object_speed = 0.24


@configclass
class InspireDynamicTabletopRollingSphereP80HomeSeedClearanceTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereP80HomeSeedBootstrapTeacherEnvCfg
):
    """Home-seed bootstrap with hard hand/table clearance for Inspire rolling."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_p80_home_seed_clearance_teacher"

    inspire_semantic_close_targets = INSPIRE_ANYDEX_SPHERE_SAFE_CLOSE_TARGETS
    reference_hand_fractions = (0.35, 1.0, 1.0, 1.0, 0.85, 0.85)
    scripted_action_prior_active_residual_scale = 0.08
    arm_action_scale = 1.35
    arm_moving_average = 0.30
    hand_moving_average = 0.34

    scripted_action_prior_hand_start_step = 150
    scripted_action_prior_hand_ramp_steps = 35

    dynamic_tabletop_pregrasp_height_offset = 0.150
    dynamic_tabletop_pregrasp_height_scale = 0.050
    dynamic_tabletop_pregrasp_height_rew_scale = 320.0
    dynamic_tabletop_min_palm_height_offset = 0.110
    dynamic_tabletop_low_palm_height_scale = 0.030
    dynamic_tabletop_low_palm_max_penalty = 5.0
    dynamic_tabletop_low_palm_penalty_scale = 620.0
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = True
    dynamic_tabletop_contact_pregrasp_gate_min = 0.05
    dynamic_tabletop_side_contact_xy_limit = 0.150
    dynamic_tabletop_side_contact_xy_ramp = 0.080
    dynamic_tabletop_side_contact_penalty_scale = 420.0

    tabletop_arm_clearance_body_names = INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_NAMES
    tabletop_arm_clearance_body_margins = INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_MARGINS
    tabletop_arm_clearance_xy_padding = 0.20
    tabletop_arm_clearance_margin = 0.040
    tabletop_arm_clearance_scale = 0.035
    tabletop_arm_clearance_max_penalty = 5.0
    tabletop_arm_clearance_penalty_scale = 2400.0
    tabletop_arm_clearance_ok_penalty_threshold = 0.50
    tabletop_gate_contact_rewards_by_clearance = True
    tabletop_gate_boolean_grasp_rewards_by_clearance = True
    tabletop_contact_clearance_gate_min = 0.0
    tabletop_contact_clearance_gate_scale = 0.20
    tabletop_success_requires_arm_clearance = True
    tabletop_terminate_on_arm_clearance_violation = True
    tabletop_arm_clearance_terminate_penalty_threshold = 0.95
    tabletop_arm_clearance_violation_terminate_start_step = 4

    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 1
    scripted_action_prior_lift_memory_requires_streak = False
    scripted_action_prior_lift_uses_proximity = True
    scripted_action_prior_lift_proximity_distance = 0.055
    scripted_action_prior_lift_proximity_min_contacts = 1.0


@configclass
class InspireDynamicTabletopRollingSphereKnownGoodHomeStaticTeacherEnvCfg(
    InspireDynamicTabletopTeacherEnvCfg
):
    """Inspire positive-control sphere task aligned with the old V340 home pose."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_v340_home_static_teacher"

    # Match the earlier IsaacGym positive-control line before adding dynamic
    # motion back: known-good table-facing arm, zero-open RH56 hand, joint targets.
    policy_action_interface = "joint_target"
    robot_cfg: ArticulationCfg = _inspire_z180_robot_cfg(
        INSPIRE_V340_KNOWN_GOOD_ARM_POS,
        default_hand_pos=INSPIRE_OFFICIAL_ZERO_HAND_OPEN_POS,
    )
    default_arm_pos = INSPIRE_V340_KNOWN_GOOD_ARM_POS
    episode_length_s = 8.0

    object_start_pos = (0.58, -0.05, TABLETOP_ROLLING_START_Z)
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=object_start_pos,
    )
    object_shape = str(TABLETOP_ROLLING_START_SPEC["proxy_shape"])
    object_radius = float(TABLETOP_ROLLING_START_SPEC["radius"])
    object_size = tuple(TABLETOP_ROLLING_START_SPEC["size"])
    tabletop_object_asset_specs = (TABLETOP_ROLLING_START_SPEC,)
    tabletop_asset_set_enabled = False
    tabletop_asset_obs_enabled = False
    reset_object_pos_noise = (0.0, 0.0, 0.0)
    reset_arm_pos_noise = 0.0

    inspire_semantic_close_targets = INSPIRE_ANYDEX_SPHERE_SAFE_CLOSE_TARGETS
    reference_hand_fractions = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    hand_moving_average = 0.78

    scripted_action_prior_enabled = False
    scripted_action_prior_zero_passthrough_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_approach_action_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False
    tabletop_lift_action_prior_rew_scale = 0.0

    dynamic_tabletop_persistent_motion = False
    dynamic_tabletop_release_motion_on_contact = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.0, 0.0)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_heading_range = (0.0, 0.0)
    dynamic_tabletop_randomize_yaw = False
    dynamic_grasp_speed_curriculum = False
    dynamic_tabletop_pregrasp_lead_time = 0.0
    dynamic_tabletop_pregrasp_ahead_distance = 0.0

    contact_distance = 0.014
    contact_score_scale = 0.014
    min_finger_contacts = 3
    min_non_thumb_contacts = 2
    true_grasp_opposition_mode = "dot"
    opposition_cos_threshold = 0.0
    palm_contact_distance = 0.060
    strict_success_enabled = True
    strict_success_contact_distance = 0.008
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    strict_reward_enabled = True
    strict_touch_score_scale = 0.008
    strict_touch_rew_scale = 20.0
    strict_approach_score_scale = 0.030
    strict_approach_rew_scale = 6.0
    strict_multifinger_approach_rew_scale = 12.0

    tabletop_arm_clearance_body_names = INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_NAMES
    tabletop_arm_clearance_body_margins = INSPIRE_TABLETOP_HAND_CLEARANCE_BODY_MARGINS
    tabletop_arm_clearance_xy_padding = 0.20
    tabletop_arm_clearance_margin = 0.025
    tabletop_arm_clearance_scale = 0.040
    tabletop_arm_clearance_max_penalty = 5.0
    tabletop_arm_clearance_penalty_scale = 9000.0
    tabletop_gate_contact_rewards_by_clearance = True
    tabletop_gate_boolean_grasp_rewards_by_clearance = True
    tabletop_contact_clearance_gate_min = 0.0
    tabletop_contact_clearance_gate_scale = 0.25
    tabletop_success_requires_arm_clearance = True

    tabletop_success_lift_height = 0.030
    dynamic_success_hold_steps = 6
    stable_object_palm_vel = 0.30
    tabletop_success_uses_grasp_seen = True
    tabletop_hover_latch_uses_grasp_seen = True
    tabletop_hover_reward_uses_grasp_seen = True
    tabletop_hover_success_object_speed = 0.24


@configclass
class InspireDynamicTabletopRollingSphereKnownGoodHomeStaticPhysicalTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereKnownGoodHomeStaticTeacherEnvCfg
):
    """12-DoF physical-joint positive control matching the old IsaacGym Inspire line."""

    reference_name = "inspire_z180_dynamic_tabletop_rolling_sphere_v340_home_static_physical_teacher"

    action_space = 19
    observation_space = 94
    action_contract = "joint_target_19d"
    hand_joint_names = INSPIRE_HAND_JOINT_NAMES
    sim_hand_joint_names = INSPIRE_HAND_JOINT_NAMES
    reference_hand_fractions = (1.0,) * len(INSPIRE_HAND_JOINT_NAMES)


@configclass
class InspireDynamicTabletopTransportDirectResidualTeacherEnvCfg(
    InspireDynamicTabletopDirectResidualTeacherEnvCfg
):
    """Inspire direct-residual transport-object tabletop teacher."""

    reference_name = "inspire_z180_dynamic_tabletop_transport_assets_direct_residual_teacher"
    observation_space = 86
    tabletop_object_asset_specs = TABLETOP_TRANSPORT_OBJECT_SPECS
    tabletop_asset_set_enabled = True
    tabletop_asset_obs_enabled = True
    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_start_count = 2
    tabletop_asset_curriculum_steps = 2_500_000
    tabletop_motion_modes = ("linear", "curved", "turntable")
    tabletop_motion_mode_curriculum = True
    tabletop_motion_mode_curriculum_start_count = 1
    tabletop_motion_mode_curriculum_steps = 2_500_000
    tabletop_turntable_center = TABLETOP_LARGE_TURNTABLE_CENTER
    tabletop_turntable_radius_range = TABLETOP_LARGE_TURNTABLE_RADIUS_RANGE
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_TRANSPORT_OBJECT_SPECS[0],
        pos=(0.58, -0.16, 0.335),
    )
    object_shape = "cylinder"
    object_radius = 0.032
    object_size = (0.064, 0.064, 0.078)
    object_start_pos = (0.58, -0.16, 0.335)
    reset_object_pos_noise = (0.075, 0.040, 0.002)
    dynamic_tabletop_initial_speed_range = (0.03, 0.15)
    dynamic_tabletop_initial_yaw_rate_range = (-0.9, 0.9)
    dynamic_tabletop_heading_range = (-0.75, 0.75)
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "steps"
    dynamic_grasp_speed_curriculum_steps = 2_500_000
    affordance_label_mode = "tabletop_transport_assets"


@configclass
class InspireFallingBatonTeacherEnvCfg(Revo2FallingBatonTeacherEnvCfg):
    """Franka + Inspire privileged teacher task for catching a falling baton."""

    action_space = 13
    observation_space = 76
    task_family = "falling_baton_grasp"
    hand_embodiment = "inspire"
    action_contract = "inspire_semantic_13d"
    reference_name = "inspire_z180_falling_baton_privileged_teacher_joint_target"
    robot_cfg: ArticulationCfg = _inspire_z180_robot_cfg(ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS)
    default_arm_pos = ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS
    create_table = False
    hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    sim_hand_joint_names = INSPIRE_HAND_JOINT_NAMES
    fingertip_body_names = INSPIRE_FINGERTIP_BODY_NAMES
    touch_body_names = INSPIRE_FINGERTIP_BODY_NAMES
    palm_body_name = "hand_base_link"
    palm_offset = INSPIRE_PALM_OFFSET
    fingertip_body_offsets = INSPIRE_FINGERTIP_BODY_OFFSETS
    reference_hand_fractions = (1.0,) * 6
    inspire_semantic_close_targets = INSPIRE_ANYDEX_P80_CLOSE_TARGETS
    hand_moving_average = 0.78
    contact_distance = 0.030
    contact_score_scale = 0.030
    palm_contact_distance = 0.060


@configclass
class InspireFallingBatonFullSpeedEvalEnvCfg(InspireFallingBatonTeacherEnvCfg):
    """Full falling-baton randomization for Franka + Inspire eval."""

    reference_name = "inspire_z180_falling_baton_full_speed_eval_teacher_joint_target"
    dynamic_grasp_speed_curriculum = False
    falling_baton_start_initial_xy_speed_range = (0.00, 0.06)
    falling_baton_start_initial_z_speed_range = (0.04, 0.26)
    falling_baton_start_initial_ang_vel_range = (-1.0, 1.0)


@configclass
class InspireFallingBatonStableTeacherEnvCfg(InspireFallingBatonTeacherEnvCfg):
    """Franka + Inspire falling-baton teacher with strict stable-in-hand success semantics."""

    reference_name = "inspire_z180_falling_baton_stable_success_teacher"
    falling_success_uses_grasp_seen = False
    falling_success_uses_strict_grasp = True
    falling_success_max_palm_distance = 0.22
    falling_success_min_finger_contacts = 3.0
    strict_reward_enabled = True
    strict_success_enabled = True
    strict_success_contact_distance = 0.012
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    strict_approach_rew_scale = 8.0
    strict_multifinger_approach_rew_scale = 4.0
    strict_opposition_approach_rew_scale = 120.0
    strict_touch_reward_requires_thumb_pair = True
    strict_touch_reward_uses_opposition_product = True
    strict_touch_reward_opposition_min_multiplier = 0.02
    strict_touch_rew_scale = 1000.0
    strict_opposition_touch_rew_scale = 3600.0
    falling_non_thumb_without_thumb_penalty_scale = 120.0
    falling_non_thumb_without_thumb_gate_start = 0.16
    falling_non_thumb_without_thumb_gate_ramp = 0.32
    falling_non_thumb_without_thumb_thumb_target = 0.32
    dynamic_success_hold_steps = 8
    stable_object_palm_vel = 0.38
    catch_success_min_z = 0.48
    dynamic_grasp_speed_curriculum_metric = "success"
    dynamic_grasp_speed_curriculum_start_success = 0.04
    dynamic_grasp_speed_curriculum_full_success = 0.24
    dynamic_grasp_speed_curriculum_ema_alpha = 0.035
    dynamic_grasp_speed_curriculum_alpha_rise = 0.0007
    dynamic_grasp_speed_curriculum_allow_decrease = True
    contact_rew_scale = 8.0
    true_grasp_rew_scale = 80.0
    opposition_rew_scale = 18.0
    catch_progress_rew_scale = 24.0
    falling_stable_grasp_rew_scale = 650.0
    falling_palm_gate_rew_scale = 450.0
    falling_positive_stable_rew_scale = 1700.0
    falling_soft_success_progress_rew_scale = 2800.0
    falling_opposed_stable_pinch_rew_scale = 4000.0
    falling_pinched_rel_vel_penalty_scale = 700.0
    stable_hold_rew_scale = 12000.0
    hold_progress_rew_scale = 20000.0
    success_bonus = 40000.0


@configclass
class InspireFallingBatonStableAffordanceTeacherEnvCfg(InspireFallingBatonStableTeacherEnvCfg):
    """Strict Inspire falling-baton task that rewards green-handle and penalizes red-region contact."""

    reference_name = "inspire_z180_falling_baton_stable_affordance_teacher"
    falling_affordance_reward_enabled = True
    falling_success_requires_positive_affordance = True
    falling_affordance_positive_rew_scale = 55.0
    falling_affordance_positive_requires_thumb_pair = True
    falling_affordance_positive_uses_opposition_product = True
    falling_affordance_positive_opposition_min_multiplier = 0.02
    falling_affordance_thumb_geom_rew_scale = 10.0
    falling_affordance_thumb_touch_rew_scale = 260.0
    falling_affordance_negative_penalty_scale = 260.0
    falling_affordance_distance_scale = 0.022
    falling_affordance_contact_distance = 0.038
    falling_affordance_radial_margin = 0.022


@configclass
class InspireFallingBatonEasyStableAffordanceTeacherEnvCfg(InspireFallingBatonStableAffordanceTeacherEnvCfg):
    """Bootstrap Inspire falling-baton task with low drop height and strict green-handle success."""

    reference_name = "inspire_z180_falling_baton_easy_stable_affordance_teacher"
    falling_baton_spawn_x_range = (-0.16, 0.16)
    falling_baton_spawn_y_range = (0.14, 0.36)
    falling_baton_spawn_z_range = (0.72, 0.94)
    falling_baton_spawn_above_palm_range = (0.16, 0.50)
    falling_baton_start_spawn_above_palm_range = (0.08, 0.22)
    falling_baton_spawn_height_curriculum = True
    falling_baton_catch_center_finger_weight = 0.80
    falling_baton_catch_center_forward_offset = 0.060
    falling_baton_catch_center_world_offset = (0.0, 0.020, 0.0)
    falling_baton_palm_relative_start_x_range = (-0.040, 0.040)
    falling_baton_palm_relative_start_y_range = (0.015, 0.090)
    falling_baton_start_roll_range = (-0.45, 0.45)
    falling_baton_start_pitch_range = (-0.35, 0.35)
    falling_baton_start_yaw_range = (-0.90, 0.90)
    falling_baton_orientation_curriculum = True
    object_lin_vel_min = (0.00, 0.00, -0.12)
    object_lin_vel_max = (0.04, 0.04, -0.01)
    object_ang_vel_min = (-0.45, -0.45, -0.45)
    object_ang_vel_max = (0.45, 0.45, 0.45)
    falling_baton_start_initial_xy_speed_range = (0.00, 0.006)
    falling_baton_start_initial_z_speed_range = (0.00, 0.020)
    falling_baton_start_initial_ang_vel_range = (-0.06, 0.06)
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_metric = "true_grasp"
    dynamic_grasp_speed_curriculum_start_success = 0.015
    dynamic_grasp_speed_curriculum_full_success = 0.18
    dynamic_grasp_speed_curriculum_ema_alpha = 0.03
    dynamic_grasp_speed_curriculum_alpha_rise = 0.002
    dynamic_grasp_speed_curriculum_allow_decrease = False
    dynamic_success_hold_steps = 8
    stable_object_palm_vel = 0.40
    catch_success_min_z = 0.50
    falling_drop_z = 0.18


@configclass
class InspireFallingBatonEasyCatchHoldCurriculumTeacherEnvCfg(
    InspireFallingBatonEasyStableAffordanceTeacherEnvCfg
):
    """Bootstrap Inspire falling-baton task whose curriculum waits for stable catch-hold."""

    reference_name = "inspire_z180_falling_baton_easy_catchhold_curriculum_teacher"
    dynamic_grasp_speed_curriculum_metric = "catch_hold"
    dynamic_grasp_speed_curriculum_start_success = 0.006
    dynamic_grasp_speed_curriculum_full_success = 0.075
    dynamic_grasp_speed_curriculum_ema_alpha = 0.04
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00035
    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_success_hold_steps = 8
    stable_object_palm_vel = 0.34
    falling_stable_grasp_rew_scale = 900.0
    falling_positive_stable_rew_scale = 2600.0
    falling_soft_success_progress_rew_scale = 3600.0
    falling_opposed_stable_pinch_rew_scale = 5600.0
    falling_pinched_rel_vel_penalty_scale = 1400.0
    stable_hold_rew_scale = 18000.0
    hold_progress_rew_scale = 32000.0
    success_bonus = 60000.0


@configclass
class InspireFallingBatonEasyPostHoldCurriculumTeacherEnvCfg(
    InspireFallingBatonEasyCatchHoldCurriculumTeacherEnvCfg
):
    """Inspire falling-baton task that requires catch success to stay stable after acquisition."""

    reference_name = "inspire_z180_falling_baton_easy_posthold_curriculum_teacher"
    dynamic_grasp_speed_curriculum_metric = "success"
    dynamic_grasp_speed_curriculum_start_success = 0.008
    dynamic_grasp_speed_curriculum_full_success = 0.16
    dynamic_grasp_speed_curriculum_ema_alpha = 0.035
    dynamic_grasp_speed_curriculum_alpha_rise = 0.00025
    dynamic_grasp_speed_curriculum_allow_decrease = True
    dynamic_success_hold_steps = 20
    stable_object_palm_vel = 0.30
    terminate_on_success = False
    falling_post_success_stability_enabled = True
    tabletop_post_success_stability_latch_enabled = True
    tabletop_post_success_arm_target_lock_enabled = True
    tabletop_post_success_arm_target_lock_blend = 1.0
    tabletop_post_success_hand_target_lock_enabled = True
    tabletop_post_success_hand_target_lock_blend = 1.0
    tabletop_post_success_hold_rew_scale = 11000.0
    tabletop_post_success_unstable_penalty_scale = 12000.0
    tabletop_post_success_grasp_loss_penalty_scale = 9000.0
    tabletop_post_success_under_height_penalty_scale = 0.0
    tabletop_post_success_speed_penalty_scale = 2200.0
    tabletop_post_success_action_penalty_scale = 0.040
    tabletop_post_success_target_delta_penalty_scale = 0.060
    tabletop_post_success_arm_joint_vel_penalty_scale = 50.0
    tabletop_post_success_arm_target_drift_penalty_scale = 900.0
    tabletop_post_success_arm_target_drift_tolerance = 0.12
    tabletop_post_success_arm_target_drift_scale = 0.30
    tabletop_post_success_palm_drift_penalty_scale = 1200.0
    tabletop_post_success_palm_drift_tolerance = 0.050
    tabletop_post_success_palm_drift_scale = 0.10
    falling_stable_grasp_rew_scale = 1000.0
    falling_positive_stable_rew_scale = 3200.0
    falling_soft_success_progress_rew_scale = 4600.0
    falling_opposed_stable_pinch_rew_scale = 6600.0
    falling_pinched_rel_vel_penalty_scale = 1800.0
    stable_hold_rew_scale = 22000.0
    hold_progress_rew_scale = 40000.0
    success_bonus = 70000.0


@configclass
class InspireFallingBatonEasyPostHoldConversionTeacherEnvCfg(
    InspireFallingBatonEasyPostHoldCurriculumTeacherEnvCfg
):
    """Post-hold continuation that converts intermittent catches into a 20-step stable hold."""

    reference_name = "inspire_z180_falling_baton_easy_posthold_conversion_teacher"

    tabletop_post_success_hold_rew_scale = 30000.0
    tabletop_post_success_unstable_penalty_scale = 18000.0
    tabletop_post_success_grasp_loss_penalty_scale = 16000.0
    tabletop_post_success_speed_penalty_scale = 3000.0
    tabletop_post_success_arm_joint_vel_penalty_scale = 80.0
    tabletop_post_success_arm_target_drift_penalty_scale = 1400.0
    tabletop_post_success_palm_drift_penalty_scale = 1800.0

    falling_stable_grasp_rew_scale = 1400.0
    falling_positive_stable_rew_scale = 4600.0
    falling_soft_success_progress_rew_scale = 6200.0
    falling_opposed_stable_pinch_rew_scale = 9000.0
    falling_pinched_rel_vel_penalty_scale = 2400.0
    stable_hold_rew_scale = 32000.0
    hold_progress_rew_scale = 65000.0
    success_bonus = 110000.0


@configclass
class InspireUnifiedFallingBatonBenchmarkTeacherEnvCfg(
    Revo2UnifiedFallingBatonBenchmarkTeacherEnvCfg
):
    """Inspire RH56 adapter for the shared red/green falling-baton benchmark."""

    reference_name = "inspire_unified_falling_baton_affordance_v1_teacher"
    hand_embodiment = "inspire"
    action_contract = "inspire_semantic_13d"
    robot_cfg: ArticulationCfg = _inspire_z180_robot_cfg(ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS)
    default_arm_pos = ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS
    hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    sim_hand_joint_names = INSPIRE_HAND_JOINT_NAMES
    fingertip_body_names = INSPIRE_FINGERTIP_BODY_NAMES
    touch_body_names = INSPIRE_FINGERTIP_BODY_NAMES
    palm_body_name = "hand_base_link"
    palm_offset = INSPIRE_PALM_OFFSET
    fingertip_body_offsets = INSPIRE_FINGERTIP_BODY_OFFSETS
    reference_hand_fractions = (1.0,) * 6
    inspire_semantic_close_targets = INSPIRE_ANYDEX_P80_CLOSE_TARGETS
    hand_moving_average = 0.78


@configclass
class InspireFallingBatonStableAffordanceFullSpeedEvalEnvCfg(InspireFallingBatonStableAffordanceTeacherEnvCfg):
    """Strict green-region Inspire falling-baton eval with full randomization."""

    reference_name = "inspire_z180_falling_baton_stable_affordance_full_speed_eval_teacher"
    dynamic_grasp_speed_curriculum = False
    falling_baton_start_initial_xy_speed_range = (0.00, 0.06)
    falling_baton_start_initial_z_speed_range = (0.04, 0.26)
    falling_baton_start_initial_ang_vel_range = (-1.0, 1.0)
