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
INSPIRE_RH56BFX_MIMIC_URDF = (
    SIMTOOLREAL_LAB_ROOT
    / "assets/embodiments/franka-inspire-rh56bfx-mimic"
    / "franka_inspire_rh56bfx_mimic.urdf"
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
# The stock IsaacLab Franka home leaves the mounted Revo2 finger row inside
# the tabletop contact envelope. A 10% interpolation toward the established
# clear home raises the palm by 11 mm while remaining a genuine home reset.
REVO2_STATIC_CLEAR_HOME_ARM_POS = (
    0.0,
    -0.5471,
    0.0,
    -2.749,
    0.0,
    2.9723,
    0.7454398163397448,
)

# Collision-free object-aligned reset poses solved with the measured palm
# Jacobian while preserving the stock Franka home wrist orientation.  Both
# poses use the same tabletop object pose and an 80 mm robot-base elevation;
# the Revo2 wrist sits higher because its fingers are shorter than RH56BFX.
REVO2_CANONICAL_STATIC_PREGRASP_ARM_POS = (
    0.2987901270389557,
    -0.32552334666252136,
    -0.552698016166687,
    -2.498549699783325,
    0.27725693583488464,
    2.983125686645508,
    0.18584927916526794,
)
INSPIRE_CANONICAL_STATIC_PREGRASP_ARM_POS = (
    0.1166767030954361,
    -0.5617563724517822,
    -0.40365704894065857,
    -2.7356746196746826,
    0.12784895300865173,
    2.9982457160949707,
    0.301824688911438,
)
CANONICAL_STATIC_ROBOT_BASE_POS = (0.0, 0.0, 0.080)

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
REVO2_V699_MIMIC_OPEN_POS = {
    **DEFAULT_HAND_OPEN_POS,
    # The official V699 URDF authors thumb distal = 1.0 * thumb proximal.
    "revo2_right_thumb_distal_joint": 0.0,
}

# Geometric fingertip observations use the calibrated touch links. Contact
# diagnostics instead attach to the actuated distal collision bodies, which
# carry load when an enclosed object does not press the small touch shell.
REVO2_DISTAL_CONTACT_BODY_NAMES = (
    "revo2_right_thumb_distal_link",
    "revo2_right_index_distal_link",
    "revo2_right_middle_distal_link",
    "revo2_right_ring_distal_link",
    "revo2_right_pinky_distal_link",
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
INSPIRE_RH56BFX_MIMIC_CLOSE_TARGETS = (
    1.25,  # thumb_proximal_yaw_joint
    0.599,  # thumb_proximal_pitch_joint
    0.95,  # index_proximal_joint
    0.95,  # middle_proximal_joint
    1.05,  # ring_proximal_joint
    1.10,  # pinky_proximal_joint
)
# AnyDex Sphere_3_Finger width-table endpoint in the deployable six-motor
# order.  Ring, pinky, and thumb yaw are already at this endpoint in the
# preshape; thumb flex, index, and middle then close progressively.
INSPIRE_ANYDEX_SPHERE_ACTIVE_CLOSE_TARGETS = (
    1.052,  # thumb_proximal_yaw_joint
    0.580,  # thumb_proximal_pitch_joint
    1.450,  # index_proximal_joint
    1.450,  # middle_proximal_joint
    0.909,  # ring_proximal_joint
    1.450,  # pinky_proximal_joint
)
INSPIRE_ANYDEX_SPHERE_ACTIVE_PRESHAPE_FRACTIONS = (
    1.000,
    0.318,
    0.276,
    0.357,
    1.000,
    1.000,
)
# Position-only RH56BFX endpoint calibrated in Isaac Lab for a 60 mm sphere.
# This is a local physical positive control, not an official Inspire posture:
# unlike the old size-independent full-close prior, it preserves a stable
# three-finger enclosure without relying on a force-triggered stop.
INSPIRE_RH56BFX_SPHERE_60MM_CALIBRATED_CLOSE_TARGETS = (
    1.05200000,  # thumb_proximal_yaw_joint
    0.22564416,  # thumb_proximal_pitch_joint
    0.63203079,  # index_proximal_joint
    0.61476982,  # middle_proximal_joint
    0.90900000,  # ring_proximal_joint
    1.45000000,  # pinky_proximal_joint
)
INSPIRE_RH56BFX_SPHERE_60MM_CALIBRATED_ACTION = (
    1.00000000,
    -0.22191668,
    -0.12823337,
    -0.15204167,
    1.00000000,
    1.00000000,
)
# DOMINO AnyDex rank-0 Ring candidate after refitting its 100 mm network
# aperture to the 60 mm sphere. These values were replayed with the official
# six-motor mimic hand under gravity and carried the sphere for 156 stable
# control steps without attachment or object teleportation.
INSPIRE_ANYDEX_RING60_ACTIVE_CLOSE_TARGETS = (
    1.12292099,
    0.22656241,
    0.40493385,
    0.0,
    0.0,
    0.0,
)
INSPIRE_ANYDEX_RING60_GRASP_ARM_POS = (
    0.14827494,
    -0.49574527,
    -0.05175123,
    -2.49802971,
    -0.00550513,
    2.69533658,
    0.58177274,
)
INSPIRE_ANYDEX_RING60_LIFT_ARM_POS = (
    0.14338678,
    -0.52514476,
    -0.04261565,
    -2.28843093,
    -0.00349799,
    2.45613146,
    0.58443260,
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
INSPIRE_RH56BFX_MIMIC_OPEN_POS = dict(INSPIRE_OFFICIAL_ZERO_HAND_OPEN_POS)
for _follower_name in (
    "index_intermediate_joint",
    "middle_intermediate_joint",
    "ring_intermediate_joint",
    "pinky_intermediate_joint",
):
    INSPIRE_RH56BFX_MIMIC_OPEN_POS[_follower_name] = -0.0454
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
INSPIRE_RH56BFX_SPHERE_60MM_PREGRASP_ARM_POS = (
    0.0,
    -0.682,
    0.0,
    -2.81,
    0.0,
    3.037,
    1.666,
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
# Shape-shared vertical carry direction measured from the successful RH56BFX
# AnyDex + CuRobo sphere, box, and cylinder trajectories.  Unlike the V340
# reset-pose probe above, these deltas are measured at the actual grasp pose.
INSPIRE_RH56BFX_CARTESIAN_CARRY_ARM_DELTA = (
    -0.015,
    -0.007,
    0.010,
    0.230,
    -0.015,
    -0.195,
    0.130,
)
# A 64-episode same-seed sweep found that 0.30 of the measured carry motion
# maximizes strict stable lift without overshooting the 40 mm task target.
INSPIRE_RH56BFX_SHORT_CARRY_ARM_DELTA = tuple(
    0.30 * value for value in INSPIRE_RH56BFX_CARTESIAN_CARRY_ARM_DELTA
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
# The fixed ``*_tip`` links are kinematic sample frames.  Their URDF collision
# spheres extend 10 mm beyond the visible collision meshes and must not
# participate in the faithful RH56BFX contact model.
INSPIRE_COLLISION_TIP_BODY_NAMES = INSPIRE_FINGERTIP_BODY_NAMES
INSPIRE_MESH_CONTACT_BODY_NAMES = (
    "thumb_distal",
    "index_intermediate",
    "middle_intermediate",
    "ring_intermediate",
    "pinky_intermediate",
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
UNIFIED_ROLLING_V26_START_POS = (0.58, -0.16, TABLETOP_ROLLING_START_Z)
UNIFIED_ROLLING_V26_RESET_OBJECT_POS_NOISE = (0.025, 0.025, 0.0015)
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
UNIFIED_ROLLING_LOAD_BEARING_LIFT_HEIGHT = 0.012
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

# Closed-hand support center calibrated with the official Revo2 geometry in the
# IsaacGym V98 audit. Keep legacy tasks at their original zero-offset semantics;
# the action-interface ablation below opts into this corrected sample point.
REVO2_CALIBRATED_GRASP_CENTER_OFFSET = (0.025, -0.012, 0.044)
REVO2_STATIC_ACTION_ABLATION_START_POS = (0.58, 0.0, TABLETOP_ROLLING_START_Z)
REVO2_STATIC_ACTION_ABLATION_MAX_ARM_JOINT_DELTA = 0.04


class _UnifiedRollingRewardContract:
    """Embodiment-independent reward contract for rolling comparisons."""

    # Stage-specific safety adapters default off and are enabled explicitly by
    # the lift stage. Keeping the fields present makes every curriculum stage
    # auditable under the same protocol schema.
    joint_target_arm_max_delta = 0.0
    joint_target_hand_max_delta = 0.0
    joint_target_rate_limit_requires_lift_baseline = False
    tabletop_lift_hand_target_lock_enabled = False
    tabletop_lift_hand_target_lock_blend = 0.0
    tabletop_lift_hand_target_close_fraction = 0.0
    tabletop_lift_rewards_require_current_strict_grasp = False
    tabletop_lift_rewards_require_force_grasp = False

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
    dynamic_tabletop_palm_frame_pregrasp_rew_scale = 0.0
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
    strict_thumb_touch_rew_scale = 0.0
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
    tabletop_lift_target_rew_scale = 0.0
    tabletop_lift_target_error_scale = 0.25
    tabletop_lift_target_requires_current_grasp = True
    tabletop_lift_target_mode = "joint_delta"
    tabletop_lift_cartesian_target_height = 0.05
    tabletop_lift_cartesian_target_damping = 0.08
    tabletop_lift_cartesian_target_max_joint_delta = 0.01
    tabletop_privileged_lift_target_obs_enabled = False
    tabletop_lift_without_current_grasp_penalty_scale = 0.0
    tabletop_lift_without_object_penalty_scale = 0.0
    tabletop_no_lift_after_grasp_penalty_scale = 180.0
    tabletop_non_thumb_without_thumb_penalty_scale = 800.0
    tabletop_object_carry_lift_rew_scale = 0.0
    tabletop_palm_object_carry_rew_scale = 0.0
    tabletop_palm_object_up_vel_rew_scale = 0.0
    tabletop_vertical_palm_carry_rew_scale = 0.0
    tabletop_object_palm_drift_penalty_scale = 0.0
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
    tabletop_relative_palm_lift_target = 0.075
    tabletop_palm_object_up_vel_target = 0.08
    tabletop_success_requires_relative_palm_lift = False
    tabletop_success_min_relative_palm_lift = 0.0
    tabletop_success_max_object_palm_drift = 0.0
    tabletop_success_max_palm_xy_drift = 0.0
    tabletop_success_max_palm_orientation_drift = 0.0
    tabletop_vertical_palm_xy_scale = 0.025
    tabletop_vertical_palm_orientation_scale = 0.35
    tabletop_object_palm_drift_tolerance = 0.025
    tabletop_object_palm_drift_scale = 0.040
    tabletop_object_palm_drift_max_penalty = 4.0
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


class _UnifiedDynamicJointDeltaV26PolicyContract:
    """Shared deployable action contract selected by the static v2.6 audit."""

    global_action_selection_scope = "revo2_and_inspire"
    dynamic_action_protocol = "jointdelta_v2_6_six_motor_absolute"
    hand_action_semantics = "six_physical_motor_absolute_target"
    action_space = 13
    policy_action_interface = "joint_target"
    joint_target_arm_action_mode = "incremental"
    joint_target_arm_delta_scale = 0.015
    arm_moving_average = 0.20
    hand_moving_average = 0.20
    joint_target_arm_max_delta = 0.015
    joint_target_hand_max_delta = 0.05
    joint_target_rate_limit_requires_lift_baseline = False
    arm_target_tracking_error_limit = 0.05

    # Direct PPO only. These switches are repeated here so no legacy parent can
    # silently re-enable a reach, closure, lift, residual, or planning prior.
    scripted_action_prior_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_approach_action_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_relative_lift_target_prior_enabled = False
    scripted_tabletop_cartesian_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False
    tabletop_privileged_lift_target_obs_enabled = False

    # Stability must be produced by the policy and reward, not by freezing the
    # arm or hand command after a success signal.
    tabletop_lift_hand_target_lock_enabled = False
    tabletop_lift_hand_target_lock_blend = 0.0
    tabletop_lift_hand_target_close_fraction = 0.0
    tabletop_post_success_arm_target_lock_enabled = False
    tabletop_post_success_arm_target_lock_blend = 0.0
    tabletop_post_success_hand_target_lock_enabled = False
    tabletop_post_success_hand_target_lock_blend = 0.0
    tabletop_post_success_hand_close_fraction = 0.0


class _UnifiedRollingJointDeltaV26Contract(
    _UnifiedDynamicJointDeltaV26PolicyContract
):
    """Static-to-fast curriculum under the accepted cross-hand controller."""

    benchmark_protocol = "rolling_multishape_jointdelta_v2_6"
    canonical_reset_curriculum_enabled = True
    canonical_reset_curriculum_override_alpha = None
    canonical_reset_curriculum_start_frames = 8_000_000
    canonical_reset_curriculum_end_frames = 48_000_000
    canonical_reset_curriculum_mixed_distribution = True
    canonical_reset_curriculum_pregrasp_anchor_fraction = 0.10
    canonical_reset_curriculum_hard_anchor_fraction = 0.50
    canonical_reset_home_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    canonical_reset_arm_pos_noise = 0.020
    canonical_reset_hand_action = 0.0

    object_start_pos = UNIFIED_ROLLING_V26_START_POS
    reset_object_pos_noise = UNIFIED_ROLLING_V26_RESET_OBJECT_POS_NOISE
    dynamic_tabletop_start_speed_range = UNIFIED_ROLLING_START_SPEED_RANGE
    dynamic_tabletop_initial_speed_range = UNIFIED_ROLLING_TARGET_SPEED_RANGE
    dynamic_tabletop_start_yaw_rate_range = UNIFIED_ROLLING_START_YAW_RATE_RANGE
    dynamic_tabletop_initial_yaw_rate_range = UNIFIED_ROLLING_TARGET_YAW_RATE_RANGE
    dynamic_tabletop_heading_range = TABLETOP_FULL_HEADING_RANGE
    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_override_alpha = None

    # A training success is already a physical opposed grasp, lift, and half
    # second stable hold. Formal videos add the stricter 60-step post-hold.
    object_contact_force_diagnostics_enabled = True
    object_contact_force_threshold = 0.05
    tabletop_success_requires_force_grasp = True
    tabletop_success_requires_hover_target = True
    tabletop_success_lift_height = UNIFIED_ROLLING_SUCCESS_LIFT_HEIGHT
    tabletop_hover_height_delta = UNIFIED_ROLLING_HOVER_HEIGHT_DELTA
    tabletop_hover_success_requires_xy = True
    tabletop_hover_success_xy_tolerance = 0.12
    tabletop_hover_success_z_tolerance = 0.035
    tabletop_hover_success_object_speed = 0.20
    stable_object_palm_vel = 0.20
    dynamic_success_hold_steps = 30
    terminate_on_success = True
    episode_length_s = 12.0


class _UnifiedRollingJointDeltaV27Contract(
    _UnifiedRollingJointDeltaV26Contract
):
    """Performance-gated static-to-dynamic curriculum with lift-first reward."""

    benchmark_protocol = "rolling_multishape_jointdelta_v2_7_lift_first"

    # The pre-grasp-to-home reset progression is unlocked by a physical hover
    # hold. Dynamic speed remains zero until the home-pose curriculum is done.
    canonical_reset_curriculum_mode = "success_gate"
    canonical_reset_curriculum_metric = "catch_hold"
    canonical_reset_curriculum_start_success = 0.08
    canonical_reset_curriculum_full_success = 0.45
    canonical_reset_curriculum_ema_alpha = 0.02
    canonical_reset_curriculum_alpha_rise = 0.00035
    canonical_reset_curriculum_allow_decrease = False
    dynamic_speed_curriculum_min_canonical_reset_alpha = 0.98
    reset_object_pos_noise = (0.010, 0.010, 0.001)

    # Reuse the v2.6 static benchmark's progress-only acquisition objective.
    # This removes the old rolling reward's persistent geometric-contact local
    # optimum: a policy receives sustained return only by lifting and hovering.
    canonical_static_reward_enabled = True
    canonical_progress_only_approach = True
    contact_distance = 0.040
    contact_score_scale = 0.025
    min_finger_contacts = 2
    min_non_thumb_contacts = 1
    opposition_cos_threshold = 0.0
    strict_reward_enabled = False
    strict_success_enabled = True
    strict_success_contact_distance = 0.006
    strict_success_min_finger_contacts = 2
    strict_success_min_non_thumb_contacts = 1
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    object_force_grasp_min_non_thumb_contacts = 1

    tabletop_lift_requires_clean_grasp_latch = False
    tabletop_lift_gate_requires_current_strict_grasp = False
    tabletop_lift_gate_requires_force_grasp = False
    tabletop_lift_rewards_require_current_strict_grasp = False
    tabletop_lift_rewards_require_force_grasp = False
    tabletop_lift_use_grasp_seen_gate = False
    lift_reward_uses_grasp_quality_gate = False
    lift_reward_min_grasp_quality_multiplier = 1.0
    lift_reward_uses_opposition_gate = False
    lift_reward_min_opposition_multiplier = 1.0
    quality_lift_progress_uses_opposition_gate = False
    quality_lift_progress_min_opposition_multiplier = 1.0
    tabletop_hover_latch_lift_progress = 0.95
    tabletop_hover_follow_object_xy_until_latch = False
    tabletop_hover_latch_uses_grasp_seen = False
    tabletop_hover_reward_uses_grasp_seen = False
    tabletop_hover_latch_requires_force_grasp = False
    tabletop_success_uses_grasp_seen = False
    tabletop_hover_success_object_speed = 0.20
    stable_object_palm_vel = 0.20

    tabletop_arm_clearance_margin = 0.010
    tabletop_arm_clearance_scale = 0.040
    tabletop_arm_clearance_max_penalty = 3.0
    tabletop_success_requires_arm_clearance = True
    tabletop_terminate_on_arm_clearance_violation = False
    dynamic_tabletop_min_palm_height_offset = 0.020
    dynamic_tabletop_low_palm_height_scale = 0.030
    dynamic_tabletop_low_palm_max_penalty = 3.0
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = False

    canonical_palm_progress_rew_scale = 8.0
    canonical_fingertip_progress_rew_scale = 14.0
    canonical_lift_step_progress_rew_scale = 20.0
    canonical_palm_reach_rew_scale = 0.5
    canonical_fingertip_reach_rew_scale = 1.5
    canonical_contact_rew_scale = 0.0
    canonical_opposition_rew_scale = 0.0
    canonical_grasp_quality_rew_scale = 0.0
    canonical_force_thumb_rew_scale = 0.0
    canonical_force_grasp_rew_scale = 0.0
    canonical_force_stable_rew_scale = 0.0
    canonical_lift_support_floor = 0.30
    canonical_lift_progress_rew_scale = 45.0
    canonical_lift_milestone_bonus = 150.0
    canonical_lifted_grasp_rew_scale = 12.0
    canonical_hover_goal_rew_scale = 40.0
    canonical_hover_stable_rew_scale = 90.0
    canonical_success_now_rew_scale = 90.0
    canonical_hold_progress_rew_scale = 180.0
    canonical_success_bonus = 300.0
    canonical_arm_clearance_penalty_scale = 20.0
    canonical_low_palm_penalty_scale = 2.0
    canonical_scoop_penalty_scale = 40.0
    canonical_palm_only_penalty_scale = 40.0
    canonical_unsupported_lift_penalty_scale = 18.0
    canonical_action_penalty_scale = 0.002
    canonical_target_delta_penalty_scale = 0.004
    canonical_drop_penalty = 30.0


class _UnifiedRollingLiftHoldStage3Contract:
    """Shared continuation objective that turns an acquired grasp into a stable lift."""

    # Real-controller-compatible target rate limits prevent absolute normalized
    # actions from commanding joint-limit jumps when the policy enters lift.
    joint_target_arm_max_delta = 0.04
    joint_target_hand_max_delta = 0.05
    joint_target_rate_limit_requires_lift_baseline = True
    tabletop_lift_hand_target_lock_enabled = True
    tabletop_lift_hand_target_lock_blend = 1.0
    tabletop_lift_hand_target_close_fraction = 0.15

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
    # Stage 2 supplies the quiet-grasp initialization. Stage 3 uses the
    # consecutive-grasp lift gate below rather than paying for staying still.
    tabletop_strict_hold_rew_scale = 0.0

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
    tabletop_strict_grasp_loss_penalty_scale = 12000.0
    tabletop_strict_grasp_loss_requires_lift_baseline = True
    tabletop_strict_grasp_loss_on_transition_only = True
    tabletop_hover_post_latch_speed_penalty_scale = 180.0
    tabletop_arm_clearance_penalty_scale = 6000.0
    # The filtered fingertip-force signal does not include every load-bearing
    # distal/proximal finger-link contact. Keep it as a diagnostic, but do not
    # let an incomplete sensor projection override a physically lifted object,
    # strict fingertip enclosure, and stable object-palm motion.
    tabletop_force_thumb_contact_rew_scale = 0.0
    tabletop_force_grasp_rew_scale = 0.0
    tabletop_force_grasp_streak_rew_scale = 0.0
    tabletop_force_stable_grasp_rew_scale = 0.0
    tabletop_force_grasp_loss_penalty_scale = 0.0

    object_contact_force_diagnostics_enabled = True
    object_contact_force_threshold = 0.05
    tabletop_arm_lift_progress_baseline_mode = "first_strict_grasp"
    # Two consecutive strict-contact steps reject one-frame collisions while
    # still latching through the millimeter-scale contact jitter seen during a
    # physically enclosed grasp. The hand safety adapter then holds the pose.
    tabletop_arm_lift_progress_baseline_grasp_streak = 2
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
    """Shared continuation objective for a sustained, load-bearing micro-lift."""

    # A per-frame contact reward admits a press-and-release solution. Blend a
    # small acquisition floor with consecutive-contact progress: the floor lets
    # PPO cross the first strict-contact step, while a 20-step grasp is worth
    # about six times more than a one-step touch.
    tabletop_strict_hold_rew_scale = 40000.0
    tabletop_strict_hold_uses_streak_progress = True
    tabletop_strict_hold_min_streak_multiplier = 0.125
    tabletop_strict_grasp_loss_penalty_scale = 12000.0
    tabletop_strict_grasp_hold_steps = 20

    # A geometric/force grasp above the object center only presses the ball
    # into the table. Stage 2 must first learn reachable lower-half support so
    # the subsequent Stage 3 lift has a load-bearing initialization.
    tabletop_underwrap_rew_scale = 8500.0
    tabletop_underwrap_below_center_fraction = 0.05
    tabletop_underwrap_height_scale = 0.040
    tabletop_underwrap_radial_fraction = 0.78
    tabletop_underwrap_radial_scale = 0.045
    tabletop_underwrap_contact_scale = 0.035
    tabletop_underwrap_contact_margin = 0.018
    # Match the strict success contract. Requiring only one non-thumb finger
    # lets the policy keep a thumb-plus-pinky table trap while the other
    # fingers remain open, which cannot carry the object after lift-off.
    tabletop_underwrap_min_non_thumb_contacts = UNIFIED_ROLLING_STRICT_MIN_NON_THUMB_CONTACTS
    tabletop_underwrap_uses_opposition = True
    tabletop_underwrap_opposition_min_multiplier = 0.22
    tabletop_underwrap_progress_weight = 0.65
    tabletop_underwrap_pair_weight = 0.35
    tabletop_underwrap_uses_pregrasp_gate = False

    # Contact while the object rests on the table does not prove that the grasp
    # can carry load. Require a small but unambiguous 12 mm micro-lift before the
    # full 40 mm Stage-3 lift. All terms below depend on simulated physical
    # state; no action prior or scripted lift is enabled.
    lift_success_height = UNIFIED_ROLLING_LOAD_BEARING_LIFT_HEIGHT
    tabletop_success_lift_height = UNIFIED_ROLLING_LOAD_BEARING_LIFT_HEIGHT
    palm_lift_rew_scale = 0.0
    lift_progress_rew_scale = 5000.0
    quality_lift_progress_rew_scale = 7000.0
    lifted_true_grasp_rew_scale = 14000.0
    tabletop_grasped_palm_lift_rew_scale = 0.0
    tabletop_grasped_arm_lift_rew_scale = 60000.0
    # palm_z progress is normalized by the shared 75 mm full-lift target, so
    # 0.16 permits a 12 mm exploration bridge before the object must follow.
    tabletop_arm_lift_reward_object_margin = 0.16
    tabletop_stable_catch_rew_scale = 5000.0
    stable_hold_rew_scale = 16000.0
    hold_progress_rew_scale = 22000.0
    tabletop_hover_goal_rew_scale = 0.0
    tabletop_hover_height_progress_rew_scale = 0.0
    tabletop_hover_stable_rew_scale = 0.0
    tabletop_hover_target_rew_scale = 0.0
    success_bonus = 36000.0

    tabletop_object_up_vel_rew_scale = 3000.0
    tabletop_object_carry_lift_rew_scale = 12000.0
    tabletop_lift_rewards_require_current_strict_grasp = True
    tabletop_lift_gate_requires_current_strict_grasp = True
    tabletop_lift_use_grasp_seen_gate = False
    tabletop_object_carry_uses_grasp_seen = False
    tabletop_success_uses_grasp_seen = False
    tabletop_object_carry_min_grasp_streak = 2
    tabletop_object_carry_streak_ramp_steps = 4
    tabletop_object_carry_uses_lift_baseline_grasp_streak = True

    tabletop_no_lift_after_grasp_penalty_scale = 3000.0
    tabletop_no_lift_after_grasp_grace_steps = 20
    tabletop_no_lift_after_grasp_ramp_steps = 40
    tabletop_lift_without_object_penalty_scale = 2500.0
    tabletop_arm_object_lift_gap_penalty_scale = 2500.0
    tabletop_object_carry_stall_penalty_scale = 2000.0


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


class _UnifiedFallingJointDeltaV26Contract(
    _UnifiedDynamicJointDeltaV26PolicyContract
):
    """Shared direct-RL controller and physical catch acceptance contract."""

    benchmark_protocol = "falling_baton_affordance_jointdelta_v2_6"
    object_contact_force_diagnostics_enabled = True
    object_contact_force_threshold = 0.05
    falling_success_requires_force_grasp = True
    falling_success_uses_strict_grasp = True
    falling_success_requires_positive_affordance = True
    dynamic_success_hold_steps = UNIFIED_FALLING_SUCCESS_HOLD_STEPS
    terminate_on_success = False
    falling_post_success_stability_enabled = True
    tabletop_post_success_stability_latch_enabled = True


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
TABLETOP_INSPIRE_BOX_55MM_SPEC = _tabletop_asset_spec(
    asset_id="primitive/inspire_box_55mm",
    category="box",
    proxy_shape="box",
    size=(0.055, 0.055, 0.055),
    radius=0.0275,
    height=0.055,
    mass=0.050,
    color=(0.12, 0.48, 0.90),
    affordance_mode="omni_grasp",
)
TABLETOP_INSPIRE_CYLINDER_50X75MM_SPEC = _tabletop_asset_spec(
    asset_id="primitive/inspire_cylinder_50x75mm",
    category="cylinder",
    proxy_shape="cylinder",
    size=(0.050, 0.050, 0.075),
    radius=0.025,
    height=0.075,
    mass=0.045,
    color=(0.92, 0.58, 0.10),
)
TABLETOP_INSPIRE_PRIMITIVE3_SPECS = (
    TABLETOP_INSPIRE_SPHERE_60MM_SPEC,
    TABLETOP_INSPIRE_BOX_55MM_SPEC,
    TABLETOP_INSPIRE_CYLINDER_50X75MM_SPEC,
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
            disable_gravity=bool(spec.get("disable_gravity", False)),
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


def _v699_revo2_six_motor_robot_cfg(default_arm_pos: tuple[float, ...]) -> ArticulationCfg:
    """Build Revo2 with six commanded motors and explicit follower targets.

    PhysX's native mimic constraint is unstable for the mixed-axis Revo2
    finger joints even at the exact zero equilibrium. Convert the five mimic
    followers to ordinary simulation joints and drive all eleven simulated
    joints from six motor targets using the official URDF coupling ratios in
    the environment adapter. The policy and hardware contract therefore remain
    strictly six-dimensional.
    """

    robot_cfg = _v699_revo2_robot_cfg(default_arm_pos)
    # Isaac Sim 4.5's importer authors PhysxMimicJointAPI when parse_mimic is
    # enabled. Disable that native constraint because the environment already
    # expands the six hardware motor commands to all eleven physical joints.
    robot_cfg.spawn.convert_mimic_joints_to_normal_joints = False
    hand_joint_pos = dict(robot_cfg.init_state.joint_pos)
    hand_joint_pos.update(REVO2_V699_MIMIC_OPEN_POS)
    robot_cfg.init_state.joint_pos = hand_joint_pos
    revo2_hand = robot_cfg.actuators["revo2_hand"]
    revo2_hand.joint_names_expr = list(REVO2_V699_PHYSICAL_HAND_JOINT_NAMES)
    # Match the original IsaacGym Revo2 position-drive contract. The distal
    # targets are mechanically coupled to the six commands by the action adapter.
    revo2_hand.effort_limit_sim = 20.0
    revo2_hand.velocity_limit_sim = 2.62
    revo2_hand.stiffness = 40.0
    revo2_hand.damping = 4.0
    return robot_cfg


def _v699_revo2_native_mimic_low_drive_robot_cfg(
    default_arm_pos: tuple[float, ...],
) -> ArticulationCfg:
    """Keep the URDF mimic constraint while matching the low-drive audit."""

    robot_cfg = _v699_revo2_robot_cfg(default_arm_pos)
    revo2_hand = robot_cfg.actuators["revo2_hand"]
    revo2_hand.effort_limit_sim = 20.0
    revo2_hand.velocity_limit_sim = 2.62
    revo2_hand.stiffness = 40.0
    revo2_hand.damping = 4.0
    return robot_cfg


def _v699_revo2_native_six_active_robot_cfg(
    default_arm_pos: tuple[float, ...],
) -> ArticulationCfg:
    """Drive only the six hardware motors and leave followers to URDF mimic."""

    robot_cfg = _v699_revo2_robot_cfg(default_arm_pos)
    hand_joint_pos = dict(robot_cfg.init_state.joint_pos)
    hand_joint_pos.update(REVO2_V699_MIMIC_OPEN_POS)
    robot_cfg.init_state.joint_pos = hand_joint_pos
    robot_cfg.actuators["revo2_hand"].joint_names_expr = list(REVO2_HAND_JOINT_NAMES)
    return robot_cfg


def _v699_revo2_official_six_active_robot_cfg(
    default_arm_pos: tuple[float, ...],
) -> ArticulationCfg:
    """Use the official URDF effort/velocity limits for all six hand motors."""

    robot_cfg = _v699_revo2_native_six_active_robot_cfg(default_arm_pos)
    robot_cfg.actuators.pop("revo2_hand")
    robot_cfg.actuators.update(
        {
            "revo2_thumb_yaw": ImplicitActuatorCfg(
                joint_names_expr=["revo2_right_thumb_metacarpal_joint"],
                effort_limit_sim=0.5,
                velocity_limit_sim=2.6175,
                stiffness=40.0,
                damping=4.0,
            ),
            "revo2_thumb_flex": ImplicitActuatorCfg(
                joint_names_expr=["revo2_right_thumb_proximal_joint"],
                effort_limit_sim=1.1,
                velocity_limit_sim=2.5303,
                stiffness=40.0,
                damping=4.0,
            ),
            "revo2_fingers": ImplicitActuatorCfg(
                joint_names_expr=[
                    "revo2_right_index_proximal_joint",
                    "revo2_right_middle_proximal_joint",
                    "revo2_right_ring_proximal_joint",
                    "revo2_right_pinky_proximal_joint",
                ],
                effort_limit_sim=2.0,
                velocity_limit_sim=2.2685,
                stiffness=40.0,
                damping=4.0,
            ),
        }
    )
    return robot_cfg


def _v699_revo2_explicit_follower_high_drive_robot_cfg(
    default_arm_pos: tuple[float, ...],
) -> ArticulationCfg:
    """Remove the URDF mimic constraint while retaining baseline drive gains."""

    robot_cfg = _v699_revo2_six_motor_robot_cfg(default_arm_pos)
    revo2_hand = robot_cfg.actuators["revo2_hand"]
    revo2_hand.effort_limit_sim = 115.0
    revo2_hand.velocity_limit_sim = 8.5
    revo2_hand.stiffness = 230.0
    revo2_hand.damping = 20.0
    return robot_cfg


def _v699_revo2_impedance_robot_cfg(default_arm_pos: tuple[float, ...]) -> ArticulationCfg:
    """Build the six-motor Revo2 model with FCI-style Franka effort control."""

    robot_cfg = _v699_revo2_six_motor_robot_cfg(default_arm_pos)
    robot_cfg.actuators["franka_shoulder"].stiffness = 0.0
    robot_cfg.actuators["franka_shoulder"].damping = 0.0
    robot_cfg.actuators["franka_forearm"].effort_limit_sim = 12.0
    robot_cfg.actuators["franka_forearm"].stiffness = 0.0
    robot_cfg.actuators["franka_forearm"].damping = 0.0
    return robot_cfg


def _v699_revo2_official_six_active_impedance_robot_cfg(
    default_arm_pos: tuple[float, ...],
) -> ArticulationCfg:
    """Use the official six-motor hand with FCI-style Franka effort control."""

    robot_cfg = _v699_revo2_official_six_active_robot_cfg(default_arm_pos)
    robot_cfg.actuators["franka_shoulder"].stiffness = 0.0
    robot_cfg.actuators["franka_shoulder"].damping = 0.0
    robot_cfg.actuators["franka_forearm"].effort_limit_sim = 12.0
    robot_cfg.actuators["franka_forearm"].stiffness = 0.0
    robot_cfg.actuators["franka_forearm"].damping = 0.0
    return robot_cfg


def _inspire_z180_robot_cfg(
    default_arm_pos: tuple[float, ...],
    *,
    default_hand_pos: dict[str, float] | None = None,
    hand_effort: float = INSPIRE_RH56BFX_HAND_EFFORT,
    hand_stiffness: float = INSPIRE_RH56BFX_HAND_STIFFNESS,
    hand_damping: float = INSPIRE_RH56BFX_HAND_DAMPING,
    hand_armature: float | None = None,
    thumb_yaw_velocity: float = INSPIRE_RH56BFX_THUMB_YAW_VEL,
    thumb_flex_velocity: float = INSPIRE_RH56BFX_THUMB_FLEX_VEL,
    finger_flex_velocity: float = INSPIRE_RH56BFX_FINGER_FLEX_VEL,
    asset_path: Path = INSPIRE_Z180_URDF,
    actuate_mimic_followers: bool = True,
    preserve_mimic_constraints: bool = False,
) -> ArticulationCfg:
    joint_pos = {joint_name: value for joint_name, value in zip(FRANKA_ARM_JOINT_NAMES, default_arm_pos)}
    joint_pos.update(INSPIRE_DEFAULT_HAND_OPEN_POS if default_hand_pos is None else default_hand_pos)
    thumb_flex_joint_names = ["thumb_proximal_pitch_joint"]
    finger_joint_names = [
        "index_proximal_joint",
        "middle_proximal_joint",
        "ring_proximal_joint",
        "pinky_proximal_joint",
    ]
    if actuate_mimic_followers:
        thumb_flex_joint_names.extend(["thumb_intermediate_joint", "thumb_distal_joint"])
        finger_joint_names = ["index_.*_joint", "middle_.*_joint", "ring_.*_joint", "pinky_.*_joint"]
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(asset_path),
            fix_base=True,
            root_link_name="panda_link0",
            merge_fixed_joints=False,
            make_instanceable=False,
            # Isaac Lab 4.5 forwards this field directly to the importer's
            # ``parse_mimic`` flag despite the field's inverse-sounding name.
            convert_mimic_joints_to_normal_joints=preserve_mimic_constraints,
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
                effort_limit_sim=hand_effort,
                velocity_limit_sim=thumb_yaw_velocity,
                stiffness=hand_stiffness,
                damping=hand_damping,
                armature=hand_armature,
            ),
            "inspire_thumb_flex": ImplicitActuatorCfg(
                joint_names_expr=thumb_flex_joint_names,
                effort_limit_sim=hand_effort,
                velocity_limit_sim=thumb_flex_velocity,
                stiffness=hand_stiffness,
                damping=hand_damping,
                armature=hand_armature,
            ),
            "inspire_fingers": ImplicitActuatorCfg(
                joint_names_expr=finger_joint_names,
                effort_limit_sim=hand_effort,
                velocity_limit_sim=finger_flex_velocity,
                stiffness=hand_stiffness,
                damping=hand_damping,
                armature=hand_armature,
            ),
        },
    )


def _inspire_rh56bfx_mimic_robot_cfg(default_arm_pos: tuple[float, ...]) -> ArticulationCfg:
    """Build the faithful six-actuator RH56BFX articulation."""

    return _inspire_z180_robot_cfg(
        default_arm_pos,
        default_hand_pos=INSPIRE_RH56BFX_MIMIC_OPEN_POS,
        hand_effort=1.0,
        hand_stiffness=60.0,
        hand_damping=6.0,
        hand_armature=0.01,
        thumb_yaw_velocity=1.0,
        thumb_flex_velocity=1.0,
        finger_flex_velocity=1.0,
        asset_path=INSPIRE_RH56BFX_MIMIC_URDF,
        actuate_mimic_followers=False,
        preserve_mimic_constraints=True,
    )


def _inspire_rh56bfx_gravity_robot_cfg(default_arm_pos: tuple[float, ...]) -> ArticulationCfg:
    """Build RH56BFX with real-gravity Franka limits for controller comparisons."""

    robot_cfg = _inspire_rh56bfx_mimic_robot_cfg(default_arm_pos)
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.actuators["franka_shoulder"].effort_limit_sim = 87.0
    robot_cfg.actuators["franka_forearm"].effort_limit_sim = 12.0
    return robot_cfg


def _inspire_rh56bfx_impedance_robot_cfg(default_arm_pos: tuple[float, ...]) -> ArticulationCfg:
    """Build RH56BFX with FCI-like effort arm and position-controlled hand.

    Franka's external torque command excludes gravity and motor friction.  The
    control side compensates both, so zero user torque holds the robot.  Turning
    off articulation gravity mirrors that contract while preserving external
    object loads and contact dynamics.
    """

    robot_cfg = _inspire_rh56bfx_mimic_robot_cfg(default_arm_pos)
    robot_cfg.actuators["franka_shoulder"].effort_limit_sim = 87.0
    robot_cfg.actuators["franka_forearm"].effort_limit_sim = 12.0
    robot_cfg.actuators["franka_shoulder"].stiffness = 0.0
    robot_cfg.actuators["franka_shoulder"].damping = 0.0
    robot_cfg.actuators["franka_forearm"].stiffness = 0.0
    robot_cfg.actuators["franka_forearm"].damping = 0.0
    return robot_cfg


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
    scripted_tabletop_pregrasp_hold_after_track = False
    scripted_tabletop_object_relative_pregrasp_enabled = False
    scripted_tabletop_object_relative_pregrasp_mode = "linearized_offset"
    scripted_tabletop_object_relative_pregrasp_hand_base_pos: tuple[float, float, float] | None = None
    scripted_tabletop_object_relative_pregrasp_hand_base_quat: tuple[float, float, float, float] | None = None
    scripted_tabletop_object_relative_pregrasp_lead_time = 0.0
    scripted_tabletop_object_relative_pregrasp_track_axes = (True, True, False)
    scripted_tabletop_object_relative_pregrasp_damping = 0.08
    scripted_tabletop_object_relative_pregrasp_max_cartesian_offset = 0.25
    scripted_tabletop_object_relative_pregrasp_max_joint_correction = 0.45
    scripted_tabletop_object_relative_pregrasp_max_ik_joint_delta = 0.04
    scripted_action_prior_hand_proximity_trigger_enabled = False
    scripted_action_prior_hand_proximity_min_step = 0
    scripted_action_prior_hand_proximity_position_threshold = 0.025
    scripted_action_prior_hand_proximity_rotation_threshold = 0.20
    scripted_action_prior_hand_proximity_ramp_steps = 48
    scripted_action_prior_hand_proximity_speed_adaptive_enabled = False
    scripted_action_prior_hand_proximity_position_speed_gain = 0.0
    scripted_action_prior_hand_proximity_position_threshold_max = 0.025
    scripted_action_prior_hand_proximity_ramp_speed_gain = 0.0
    scripted_action_prior_hand_proximity_ramp_min_steps = 0

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
    dynamic_tabletop_pregrasp_direction_min_speed = 0.005
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
    scripted_action_prior_uses_force_grasp = False
    scripted_action_prior_lift_relative_to_grasp = False
    scripted_action_prior_lift_grasp_delay_steps = 0
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
    object_force_grasp_min_non_thumb_contacts = 2
    falling_success_uses_grasp_seen = True
    falling_success_uses_strict_grasp = False
    falling_success_max_palm_distance = 0.0
    falling_success_palm_gate_soft_scale = 0.055
    falling_success_min_finger_contacts = 0.0
    falling_success_requires_positive_affordance = False
    falling_success_requires_force_grasp = False
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
    strict_touch_reward_target_distance = None
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
class Revo2UnifiedFallingJointDeltaV26TeacherEnvCfg(
    _UnifiedFallingJointDeltaV26Contract,
    Revo2UnifiedFallingBatonBenchmarkTeacherEnvCfg,
):
    """Official six-active Revo2 falling teacher under the selected action."""

    reference_name = "revo2_unified_falling_jointdelta_v26_teacher"
    action_contract = "revo2_semantic_13d"
    sim_hand_joint_names = REVO2_HAND_JOINT_NAMES
    touch_body_names = REVO2_DISTAL_CONTACT_BODY_NAMES
    robot_cfg: ArticulationCfg = _v699_revo2_official_six_active_robot_cfg(
        ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS
    )


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
        pos=(0.58, -0.16, 0.336),
    )
    object_size = (0.040, 0.040, 0.080)
    object_start_pos = (0.58, -0.16, 0.336)
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
class Revo2DynamicTabletopSixMotorPhysicsAuditEnvCfg(Revo2DynamicTabletopTeacherEnvCfg):
    """Single-variable audit of explicit six-motor Revo2 hand physics."""

    reference_name = "revo2_dynamic_tabletop_six_motor_physics_audit"
    robot_cfg: ArticulationCfg = _v699_revo2_six_motor_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )
    robot_mimic_natural_frequency = None
    robot_mimic_damping_ratio = None


@configclass
class Revo2DynamicTabletopNativeMimicLowDrivePhysicsAuditEnvCfg(
    Revo2DynamicTabletopTeacherEnvCfg
):
    """Audit low drive bandwidth with the URDF mimic constraint retained."""

    reference_name = "revo2_dynamic_tabletop_native_mimic_low_drive_physics_audit"
    robot_cfg: ArticulationCfg = _v699_revo2_native_mimic_low_drive_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


@configclass
class Revo2DynamicTabletopExplicitFollowerHighDrivePhysicsAuditEnvCfg(
    Revo2DynamicTabletopTeacherEnvCfg
):
    """Audit explicit follower targets with the baseline high drive bandwidth."""

    reference_name = "revo2_dynamic_tabletop_explicit_follower_high_drive_physics_audit"
    robot_cfg: ArticulationCfg = _v699_revo2_explicit_follower_high_drive_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )
    robot_mimic_natural_frequency = None
    robot_mimic_damping_ratio = None


@configclass
class Revo2DynamicTabletopNativeSixActivePhysicsAuditEnvCfg(
    Revo2DynamicTabletopTeacherEnvCfg
):
    """Deployable six-active-motor action contract with native mimic followers."""

    reference_name = "revo2_dynamic_tabletop_native_six_active_physics_audit"
    sim_hand_joint_names = REVO2_HAND_JOINT_NAMES
    robot_cfg: ArticulationCfg = _v699_revo2_native_six_active_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


@configclass
class Revo2DynamicTabletopOfficialSixActivePhysicsAuditEnvCfg(
    Revo2DynamicTabletopTeacherEnvCfg
):
    """Six-active-motor native-mimic contract with official URDF limits."""

    reference_name = "revo2_dynamic_tabletop_official_six_active_physics_audit"
    sim_hand_joint_names = REVO2_HAND_JOINT_NAMES
    robot_cfg: ArticulationCfg = _v699_revo2_official_six_active_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


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
class Revo2UnifiedRollingJointDeltaV26TeacherEnvCfg(
    _UnifiedRollingJointDeltaV26Contract,
    Revo2UnifiedRollingBenchmarkTeacherEnvCfg,
):
    """Official six-active Revo2 rolling teacher under the selected action."""

    reference_name = "revo2_unified_rolling_jointdelta_v26_teacher"
    action_contract = "revo2_semantic_13d"
    canonical_reset_pregrasp_arm_pos = REVO2_CANONICAL_STATIC_PREGRASP_ARM_POS
    default_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    sim_hand_joint_names = REVO2_HAND_JOINT_NAMES
    touch_body_names = REVO2_DISTAL_CONTACT_BODY_NAMES
    robot_cfg: ArticulationCfg = _v699_revo2_official_six_active_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )
    robot_cfg.init_state.pos = CANONICAL_STATIC_ROBOT_BASE_POS
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=UNIFIED_ROLLING_V26_START_POS,
    )


@configclass
class Revo2UnifiedRollingJointDeltaV27TeacherEnvCfg(
    _UnifiedRollingJointDeltaV27Contract,
    Revo2UnifiedRollingBenchmarkTeacherEnvCfg,
):
    """Revo2 rolling teacher with performance-gated lift-first curriculum."""

    reference_name = "revo2_unified_rolling_jointdelta_v27_teacher"
    action_contract = "revo2_semantic_13d"
    canonical_reset_pregrasp_arm_pos = REVO2_CANONICAL_STATIC_PREGRASP_ARM_POS
    default_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    sim_hand_joint_names = REVO2_HAND_JOINT_NAMES
    touch_body_names = REVO2_DISTAL_CONTACT_BODY_NAMES
    robot_cfg: ArticulationCfg = _v699_revo2_official_six_active_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )
    robot_cfg.init_state.pos = CANONICAL_STATIC_ROBOT_BASE_POS
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=UNIFIED_ROLLING_V26_START_POS,
    )


class _CartesianWristDirectPolicyContract:
    """Legacy measured-pose wrist deltas plus six deployable hand motors."""

    action_space = 12
    policy_action_interface = "cartesian_wrist_delta"
    # Preserve the semantics used to train the existing rolling checkpoints.
    cartesian_wrist_target_mode = "measured_delta"
    cartesian_wrist_translation_scale = 0.015
    cartesian_wrist_rotation_scale = 0.08
    cartesian_wrist_damping = 0.08
    cartesian_wrist_max_joint_delta = 0.04
    cartesian_wrist_max_position_error = 0.12
    cartesian_wrist_max_rotation_error = 0.50
    cartesian_wrist_nullspace_gain = 0.05

    # This contract changes only the arm action coordinates. It never injects
    # a reach, close, lift, motion-planning, or residual action into the policy.
    scripted_action_prior_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_approach_action_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_relative_lift_target_prior_enabled = False
    scripted_tabletop_cartesian_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False


class _PersistentCartesianWristDirectPolicyContract(
    _CartesianWristDirectPolicyContract
):
    """Persistent wrist-command target used by corrected from-scratch runs."""

    cartesian_wrist_target_mode = "integrated_delta"


class _CartesianImpedanceDirectPolicyContract:
    """Base-frame wrist deltas with torque-level OSC and six hand targets."""

    action_space = 12
    policy_action_interface = "cartesian_impedance"
    cartesian_wrist_translation_scale = 0.015
    cartesian_wrist_rotation_scale = 0.08
    cartesian_impedance_target_mode = "integrated_delta"
    cartesian_impedance_max_position_error = 0.06
    cartesian_impedance_max_rotation_error = 0.35

    cartesian_impedance_motion_stiffness = (
        150.0,
        150.0,
        150.0,
        20.0,
        20.0,
        20.0,
    )
    cartesian_impedance_motion_damping_ratio = (1.0,) * 6
    cartesian_impedance_inertial_dynamics_decoupling = True
    cartesian_impedance_partial_inertial_dynamics_decoupling = False
    # libfranka external torques are gravity-compensated by the Control side.
    cartesian_impedance_gravity_compensation = False
    cartesian_impedance_nullspace_control = "position"
    cartesian_impedance_nullspace_stiffness = 10.0
    cartesian_impedance_nullspace_damping_ratio = 1.0
    cartesian_impedance_arm_effort_limits = (
        87.0,
        87.0,
        87.0,
        87.0,
        12.0,
        12.0,
        12.0,
    )

    scripted_action_prior_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_approach_action_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_relative_lift_target_prior_enabled = False
    scripted_tabletop_cartesian_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False


@configclass
class Revo2DynamicTabletopCartesianTeacherEnvCfg(
    _PersistentCartesianWristDirectPolicyContract,
    Revo2DynamicTabletopTeacherEnvCfg,
):
    """Simple tabletop PPO baseline with only the arm action coordinates changed."""

    reference_name = "revo2_dynamic_tabletop_privileged_teacher_cartesian_wrist_delta"
    action_contract = "revo2_cartesian_wrist_12d"
    observation_space = 75


class _CartesianWristAcquisitionStage1Contract:
    """Remove lift shortcuts while learning reach, opposition, and closure."""

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
    tabletop_no_lift_after_grasp_penalty_scale = 0.0
    tabletop_lift_without_current_grasp_penalty_scale = 2000.0
    tabletop_lift_without_current_grasp_min_progress = 0.10
    tabletop_lift_without_current_grasp_ramp = 0.30


class _Revo2StaticActionInterfaceAblationContract:
    """Shared static geometry and reward semantics for the controller A/B."""

    benchmark_protocol = "revo2_static_action_interface_ablation_v9_local_control_balanced_shaping"
    robot_cfg: ArticulationCfg = _v699_revo2_robot_cfg(REVO2_STATIC_CLEAR_HOME_ARM_POS)
    default_arm_pos = REVO2_STATIC_CLEAR_HOME_ARM_POS
    scripted_tabletop_pregrasp_arm_pos = REVO2_STATIC_CLEAR_HOME_ARM_POS
    tabletop_arm_lift_progress_baseline_pos = REVO2_STATIC_CLEAR_HOME_ARM_POS
    touch_body_names = REVO2_DISTAL_CONTACT_BODY_NAMES
    # Distal collision-force sensors are diagnostic only and are expensive to
    # replicate across the 512-env training scene. Evaluations enable them
    # explicitly when auditing load-bearing contact.
    object_contact_force_diagnostics_enabled = False
    # Freeze the number of controller applications per policy step. The
    # incremental joint controller integrates once on each physics substep.
    decimation = 2
    # The acquisition reward must depend only on physical state. Action norms
    # and lift-progress definitions live in different coordinate systems for
    # the two controllers and would otherwise confound this ablation.
    action_penalty_scale = 0.0
    arm_target_delta_penalty_scale = 0.0
    # PPO uses a fixed unit Gaussian. Bound absolute joint-target exploration
    # to the same per-step joint correction used by the Cartesian DLS adapter.
    joint_target_arm_max_delta = REVO2_STATIC_ACTION_ABLATION_MAX_ARM_JOINT_DELTA
    joint_target_rate_limit_requires_lift_baseline = False
    tabletop_arm_lift_progress_mode = "palm_z"
    tabletop_hover_post_latch_action_penalty_scale = 0.0
    tabletop_hover_post_latch_target_delta_penalty_scale = 0.0
    tabletop_lift_action_prior_rew_scale = 0.0
    tabletop_lift_target_rew_scale = 0.0
    tabletop_lift_without_current_grasp_penalty_scale = 0.0
    tabletop_post_success_arm_target_drift_penalty_scale = 0.0
    tabletop_post_success_action_penalty_scale = 0.0
    tabletop_post_success_target_delta_penalty_scale = 0.0
    tabletop_privileged_lift_target_obs_enabled = False
    # This is a physical-state target shared by both action interfaces. A
    # load-bearing Revo2 policy places the object roughly 75 mm into the palm
    # frame; without this envelope both controllers can maximize fingertip
    # proximity by pressing the sphere between the palm and the table.
    pregrasp_target_rel_palm = (0.030, 0.007, 0.075)
    pregrasp_target_scale = (0.030, 0.025, 0.030)
    # This is reach shaping, not the terminal grasp objective. At 12k, the v8
    # local-control run deliberately opened both thumb motors to preserve a
    # high palm-pose score. Keep the calibrated envelope useful while allowing
    # thumb/non-thumb opposition and physical touch to dominate acquisition.
    dynamic_tabletop_palm_frame_pregrasp_rew_scale = 1800.0
    scripted_action_prior_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_approach_action_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_relative_lift_target_prior_enabled = False
    scripted_tabletop_cartesian_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False
    palm_offset = REVO2_CALIBRATED_GRASP_CENTER_OFFSET
    object_start_pos = REVO2_STATIC_ACTION_ABLATION_START_POS
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=REVO2_STATIC_ACTION_ABLATION_START_POS,
    )
    reset_arm_pos_noise = 0.0
    reset_object_pos_noise = (0.0, 0.0, 0.0)
    tabletop_asset_curriculum = False
    tabletop_asset_curriculum_start_count = 1
    tabletop_asset_curriculum_override_alpha = 0.0
    dynamic_grasp_speed_curriculum = False
    dynamic_grasp_speed_curriculum_override_alpha = 0.0
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.0, 0.0)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_heading_range = (0.0, 0.0)
    dynamic_tabletop_randomize_yaw = False
    dynamic_tabletop_pregrasp_lead_time = 0.0
    dynamic_tabletop_pregrasp_ahead_distance = 0.0


class _Revo2StaticJointDeltaPolicyContract:
    """Incremental joint-position control matching IsaacGym and real execution."""

    action_contract = "revo2_semantic_13d"
    action_space = 13
    policy_action_interface = "isaaclab_direct"
    observation_space = 90
    tabletop_privileged_phase_obs_enabled = True

    # The direct controller integrates once on each of two physics substeps.
    # At dt=1/60 and EMA=0.96, scale=1.25 gives a total 0.04 rad maximum
    # target correction per policy step, matching the Cartesian runtime probe.
    arm_action_scale = 1.25
    arm_moving_average = 0.96
    # Keep the integrated target local to measured state. Without this bound,
    # repeated actions can accumulate a multi-radian target error and turn the
    # position actuator into an unintended force/trajectory prior.
    arm_target_tracking_error_limit = 0.04
    arm_target_clamp_delta = (2.30, 2.30, 2.30, 2.30, 2.30, 2.30, 2.30)


class _Revo2StaticGraspHoldStage2Contract:
    """Move from the open-hand reach envelope into physical contact."""

    benchmark_protocol = "revo2_static_action_interface_ablation_v20_opposition_balanced_contact"
    # Median object pose in the palm frame at the first strict-grasp frame of
    # the validated JointDelta 20-trial rollout. This controller-independent
    # state target closes the 1-2 cm gap left by the Stage-1 reach envelope;
    # it does not prescribe an arm joint pose, hand action, or lift motion.
    pregrasp_target_rel_palm = (0.015, 0.016, 0.058)
    pregrasp_target_scale = (0.030, 0.025, 0.030)
    dynamic_tabletop_palm_frame_pregrasp_rew_scale = 1800.0
    # Force-audited rollouts show that a 2 mm point-to-object distance tracks
    # distal-link collision contact, while the old 10 mm boundary admits a
    # visible physical gap. Keep the success geometry unchanged for checkpoint
    # compatibility, but make dense touch shaping continue through the final
    # millimetres instead of saturating early.
    strict_touch_reward_target_distance = 0.002
    # Retain the audited 10 mm anti-collapse gate. Raising this negative term
    # to the 2 mm equivalent made PPO release the non-thumb fingers to avoid
    # the penalty; the positive 2 mm term and force reward provide the final
    # closure pressure without making an already useful enclosure negative.
    tabletop_non_thumb_without_thumb_thumb_target = 0.67
    tabletop_non_thumb_without_thumb_penalty_scale = 6000.0
    # A first exploratory strict grasp must not make every later miss in the
    # episode negative. Penalize the actual grasp-loss transition once; the
    # persistent seen-latch otherwise overwhelms the sparse acquisition return.
    tabletop_strict_grasp_loss_on_transition_only = True
    # A direct thumb-only term helped reach the collision boundary, but at 8k
    # it outweighed the inherited 1.2k opposition term and rewarded thumb and
    # non-thumb contacts on the same side of the sphere. Keep a small monotonic
    # thumb bridge while restoring the opposition-balanced static-grasp scales.
    strict_thumb_touch_rew_scale = 2000.0
    strict_touch_rew_scale = 3000.0
    strict_opposition_approach_rew_scale = 5000.0
    strict_opposition_touch_rew_scale = 12000.0
    # The 2 mm distance reward brings the fingers to the collision boundary;
    # filtered PhysX object-contact forces then distinguish a real opposed
    # grasp from two fingers that merely stop near the object. These existing
    # state rewards are shared by both arm action interfaces.
    object_contact_force_diagnostics_enabled = True
    object_contact_force_threshold = 0.05
    # Start the physical-force curriculum with a valid thumb-plus-one-finger
    # pinch. The simultaneous strict geometric hold still requires two
    # non-thumb fingers, so this bridge cannot redefine the final grasp goal.
    object_force_grasp_min_non_thumb_contacts = 1
    tabletop_force_thumb_contact_rew_scale = 16000.0
    tabletop_force_grasp_rew_scale = 24000.0
    tabletop_force_grasp_streak_rew_scale = 32000.0
    tabletop_force_stable_grasp_rew_scale = 42000.0
    tabletop_force_grasp_loss_penalty_scale = 0.0
    tabletop_force_grasp_streak_target = 12


class _Revo2StaticFromScratchContactContract(_Revo2StaticGraspHoldStage2Contract):
    """Home-reset contact curriculum shared by the controller A/B."""

    benchmark_protocol = "revo2_static_action_interface_ablation_v22_from_scratch_contact"
    # The Stage-2 value is intentionally strong for a policy that already
    # reaches the object. From home it dominates the palm-frame reach reward
    # before the thumb can physically approach, suppressing useful finger
    # exploration. Use the established Stage-1 value until contact is learned.
    tabletop_non_thumb_without_thumb_penalty_scale = 300.0


class _Revo2StaticMicroLiftVelocityBridgeContract:
    """Controller-independent bridge from a static hold to vertical carry."""

    benchmark_protocol = "revo2_static_action_interface_ablation_v26_force_gated_micro_lift_balanced"
    # Keep the acquisition target at the force-audited contact pose. Returning
    # to the Stage-1 reach envelope during lift training pulls the wrist away
    # from a load-bearing hold, especially under accurate Cartesian control.
    pregrasp_target_rel_palm = (0.015, 0.016, 0.058)
    strict_touch_reward_target_distance = 0.002

    # Stage 1 and the first hold continuation intentionally pay strongly for
    # acquisition. Once a load-bearing grasp checkpoint exists, those dense
    # stationary terms must not outweigh physical lift. Use the established
    # Stage-3 acquisition floor for both action interfaces, while retaining a
    # small underwrap term for recovery after a miss.
    contact_rew_scale = 120.0
    grasp_quality_rew_scale = 700.0
    opposition_rew_scale = 400.0
    true_grasp_rew_scale = 1500.0
    strict_approach_rew_scale = 8.0
    strict_multifinger_approach_rew_scale = 16.0
    strict_opposition_approach_rew_scale = 120.0
    strict_touch_rew_scale = 1000.0
    strict_opposition_touch_rew_scale = 3000.0
    dynamic_tabletop_palm_frame_pregrasp_rew_scale = 600.0
    tabletop_underwrap_rew_scale = 850.0
    tabletop_strict_hold_rew_scale = 8000.0
    tabletop_no_lift_after_grasp_penalty_scale = 6000.0

    # Both scores require a current strict grasp and simultaneous positive
    # palm/object Z velocity. The second term additionally rejects lateral and
    # angular wrist sweeps. No desired joint action or trajectory is supplied.
    tabletop_palm_object_up_vel_rew_scale = 48000.0
    tabletop_vertical_palm_velocity_rew_scale = 96000.0
    tabletop_palm_object_up_vel_target = 0.08
    tabletop_vertical_palm_xy_vel_scale = 0.040
    tabletop_vertical_palm_ang_vel_scale = 0.40
    # The 20-step hold is a persistent curriculum milestone. Once completed,
    # contact flicker cannot reset the high acquisition return; retain a small
    # floor while grasp-loss and physical lift rewards drive the next phase.
    tabletop_strict_hold_reward_latches_at_target = True
    tabletop_strict_hold_post_target_multiplier = 0.05

    # Lift and success must preserve a current physical thumb/opposed-finger
    # force loop. A 10 mm geometric proximity signal remains useful shaping,
    # but it is not accepted as a load-bearing grasp or successful lift.
    object_contact_force_diagnostics_enabled = True
    object_contact_force_threshold = 0.05
    object_force_grasp_min_non_thumb_contacts = 1
    tabletop_arm_lift_progress_baseline_mode = "first_force_grasp"
    tabletop_arm_lift_progress_baseline_grasp_streak = 4
    tabletop_lift_gate_requires_force_grasp = True
    tabletop_lift_rewards_require_force_grasp = True
    tabletop_lift_rewards_require_current_strict_grasp = True
    tabletop_no_lift_uses_force_grasp_gate = True
    tabletop_success_requires_force_grasp = True
    # A stationary load-bearing grasp must remain less valuable than beginning
    # the lift.  The previous 8k/12k/16k terms paid roughly 21k per step at the
    # audited joint-delta checkpoint, while the fully-ramped no-lift penalty was
    # only about 4.7k.  That made holding still the correct optimum for PPO.
    # Keep a modest contact floor and a transition-only loss cost; the upward
    # velocity and no-lift terms now decide whether the policy makes progress.
    tabletop_force_grasp_rew_scale = 1000.0
    tabletop_force_grasp_streak_rew_scale = 1500.0
    tabletop_force_stable_grasp_rew_scale = 2000.0
    tabletop_force_grasp_loss_penalty_scale = 8000.0
    tabletop_force_grasp_streak_target = 12


class _StaticStrictFromScratchControlABContract:
    """Controller-neutral static sphere task with a physical success gate."""

    benchmark_protocol = "static_strict_from_scratch_control_ab_v1"
    global_action_selection_scope = "revo2_and_inspire"
    hand_action_semantics = "six_physical_motor_absolute_target"

    # One shared 44 mm sphere and deterministic reset isolate controller and
    # embodiment behavior before object or pose randomization is introduced.
    object_start_pos = UNIFIED_ROLLING_START_POS
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=UNIFIED_ROLLING_START_POS,
    )
    object_shape = str(TABLETOP_ROLLING_START_SPEC["proxy_shape"])
    object_radius = float(TABLETOP_ROLLING_START_SPEC["radius"])
    object_size = tuple(TABLETOP_ROLLING_START_SPEC["size"])
    reset_arm_pos_noise = 0.0
    reset_object_pos_noise = (0.0, 0.0, 0.0)
    tabletop_asset_set_enabled = False
    tabletop_asset_obs_enabled = False
    tabletop_asset_curriculum = False
    tabletop_motion_modes = ("free",)
    tabletop_motion_mode_curriculum = False
    dynamic_grasp_speed_curriculum = False
    dynamic_grasp_speed_curriculum_override_alpha = 0.0
    dynamic_tabletop_persistent_motion = False
    dynamic_tabletop_release_motion_on_contact = False
    dynamic_tabletop_randomize_yaw = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.0, 0.0)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_heading_range = (0.0, 0.0)
    dynamic_tabletop_pregrasp_lead_time = 0.0
    dynamic_tabletop_pregrasp_ahead_distance = 0.0

    # No action, trajectory, grasp, or lift prior is injected or supervised.
    scripted_action_prior_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_approach_action_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_relative_lift_target_prior_enabled = False
    scripted_tabletop_cartesian_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False
    tabletop_privileged_lift_target_obs_enabled = False
    lift_action_prior_rew_scale = 0.0
    tabletop_lift_action_prior_rew_scale = 0.0
    tabletop_lift_target_rew_scale = 0.0

    # Coordinate-dependent action costs are disabled for the control A/B.
    action_penalty_scale = 0.0
    arm_target_delta_penalty_scale = 0.0
    tabletop_post_success_action_penalty_scale = 0.0
    tabletop_post_success_target_delta_penalty_scale = 0.0
    tabletop_post_success_arm_target_drift_penalty_scale = 0.0
    tabletop_hover_post_latch_action_penalty_scale = 0.0
    tabletop_hover_post_latch_target_delta_penalty_scale = 0.0

    # Success requires a 4 cm physical lift, low relative velocity, and twelve
    # consecutive strict thumb/opposition frames.  The filtered distal-link
    # force projection remains diagnostic: a load can transfer through the
    # palm or proximal links while the object is visibly and physically held.
    contact_distance = 0.012
    contact_score_scale = 0.014
    min_finger_contacts = 3
    min_non_thumb_contacts = 2
    opposition_cos_threshold = 0.0
    strict_success_enabled = True
    strict_reward_enabled = True
    strict_success_contact_distance = 0.008
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    strict_reward_contact_score_scale = 0.012
    strict_touch_reward_target_distance = 0.002
    object_contact_force_diagnostics_enabled = True
    object_contact_force_threshold = 0.05
    object_force_grasp_min_non_thumb_contacts = 1
    tabletop_success_requires_force_grasp = False
    lift_success_height = 0.040
    tabletop_success_lift_height = 0.040
    stable_object_palm_vel = 0.25
    dynamic_success_hold_steps = 12
    tabletop_success_uses_grasp_seen = False
    tabletop_success_requires_hover_target = False
    tabletop_success_requires_arm_clearance = True
    tabletop_terminate_on_arm_clearance_violation = False
    terminate_on_success = True
    episode_length_s = 8.0

    # Keep the policy responsible for the complete grasp and hold. Target locks
    # are useful deployment adapters, but would inflate this controller test.
    joint_target_arm_max_delta = 0.04
    joint_target_hand_max_delta = 0.05
    # Absolute targets remain the policy semantics, but may not wind up more
    # than 40 mrad away from the measured Franka joints.
    arm_target_tracking_error_limit = 0.04
    joint_target_rate_limit_requires_lift_baseline = False
    tabletop_lift_hand_target_lock_enabled = False
    tabletop_lift_hand_target_lock_blend = 0.0
    tabletop_lift_hand_target_close_fraction = 0.0
    tabletop_post_success_stability_latch_enabled = False
    tabletop_post_success_arm_target_lock_enabled = False
    tabletop_post_success_hand_target_lock_enabled = False

    # A compact state-only reward: reach, opposed physical closure, object lift,
    # and stable hold. Arm-joint lift projections are excluded from the A/B.
    palm_reach_rew_scale = 24.0
    fingertip_reach_rew_scale = 24.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 120.0
    dynamic_tabletop_pregrasp_height_rew_scale = 30.0
    dynamic_tabletop_palm_frame_pregrasp_rew_scale = 0.0
    contact_rew_scale = 120.0
    grasp_quality_rew_scale = 700.0
    opposition_rew_scale = 400.0
    true_grasp_rew_scale = 1500.0
    strict_approach_rew_scale = 20.0
    strict_multifinger_approach_rew_scale = 40.0
    strict_opposition_approach_rew_scale = 240.0
    strict_touch_rew_scale = 1000.0
    strict_opposition_touch_rew_scale = 3000.0
    tabletop_non_thumb_without_thumb_penalty_scale = 300.0
    tabletop_underwrap_rew_scale = 2500.0
    tabletop_strict_hold_rew_scale = 6000.0
    tabletop_strict_hold_uses_streak_progress = True
    tabletop_strict_hold_min_streak_multiplier = 0.10
    tabletop_strict_grasp_hold_steps = 12
    tabletop_strict_grasp_loss_penalty_scale = 3000.0
    lift_progress_rew_scale = 5000.0
    quality_lift_progress_rew_scale = 7000.0
    lifted_true_grasp_rew_scale = 14000.0
    tabletop_stable_catch_rew_scale = 5000.0
    tabletop_grasped_palm_lift_rew_scale = 4500.0
    tabletop_grasped_arm_lift_rew_scale = 0.0
    arm_lift_progress_rew_scale = 0.0
    tabletop_object_up_vel_rew_scale = 3000.0
    tabletop_object_carry_lift_rew_scale = 16000.0
    stable_hold_rew_scale = 16000.0
    hold_progress_rew_scale = 22000.0
    success_bonus = 48000.0
    tabletop_no_lift_after_grasp_penalty_scale = 4000.0
    tabletop_lift_without_object_penalty_scale = 3000.0
    tabletop_lift_without_current_grasp_penalty_scale = 4000.0
    tabletop_arm_object_lift_gap_penalty_scale = 3000.0
    tabletop_arm_clearance_penalty_scale = 3000.0
    tabletop_lift_gate_requires_current_strict_grasp = True
    tabletop_lift_gate_requires_force_grasp = False
    tabletop_lift_rewards_require_current_strict_grasp = True
    tabletop_lift_rewards_require_force_grasp = False
    tabletop_lift_use_grasp_seen_gate = False
    tabletop_object_carry_uses_grasp_seen = False
    tabletop_hover_latch_uses_grasp_seen = False
    tabletop_hover_reward_uses_grasp_seen = False
    tabletop_arm_lift_progress_baseline_mode = "first_strict_grasp"
    tabletop_arm_lift_progress_baseline_grasp_streak = 2
    tabletop_force_thumb_contact_rew_scale = 500.0
    tabletop_force_grasp_rew_scale = 1000.0
    tabletop_force_grasp_streak_rew_scale = 1000.0
    tabletop_force_stable_grasp_rew_scale = 1500.0
    tabletop_force_grasp_loss_penalty_scale = 2000.0
    tabletop_force_grasp_streak_target = 12


class _StaticStableHoverControlABContract:
    """Clean pre-lift enclosure followed by a compact vertical hover."""

    benchmark_protocol = "static_stable_hover_from_scratch_control_ab_v12_markov_phase_scalar"

    # The privileged teacher receives a compact semantic contact state. The
    # same 15 channels are used for Revo2 and Inspire and later become student
    # auxiliary targets; no force channel can independently define success.
    tabletop_privileged_contact_obs_enabled = True
    tabletop_privileged_contact_obs_dim = 15
    tabletop_privileged_contact_force_scale = 5.0
    tabletop_privileged_contact_relative_speed_scale = 0.25

    # A valid episode must first close around the sphere while it is still on
    # the tabletop and every sampled hand/arm point remains clear of the table.
    # Lifting before this latch is a failed table-assisted scoop.
    tabletop_clean_grasp_latch_enabled = True
    tabletop_lift_requires_clean_grasp_latch = True
    tabletop_arm_lift_baseline_requires_clean_grasp_latch = True
    tabletop_clean_grasp_contact_distance = 0.003
    tabletop_clean_grasp_min_non_thumb_contacts = 2
    tabletop_clean_grasp_max_object_height_delta = 0.005
    tabletop_clean_grasp_max_object_speed = 0.08
    tabletop_clean_grasp_max_relative_speed = 0.12
    tabletop_clean_grasp_hold_steps = 6
    tabletop_clean_grasp_requires_force_grasp = False
    tabletop_clean_grasp_rew_scale = 12000.0
    tabletop_clean_grasp_latch_bonus = 200000.0
    tabletop_acquisition_reward_post_clean_latch_floor = 0.05
    tabletop_unclean_lift_height = 0.012
    tabletop_unclean_lift_penalty_scale = 60000.0
    tabletop_terminate_on_unclean_lift = True

    # Strict geometric enclosure is first allowed to settle into real opposed
    # pressure. Lift incentives remain disabled until the clean latch, so the
    # policy is never asked to move upward while it is still acquiring contact.
    tabletop_opposed_force_pressure_target = 1.0
    tabletop_opposed_force_pressure_rew_scale = 6000.0
    tabletop_opposed_force_pressure_requires_strict_opposition = True
    tabletop_opposed_force_pressure_requires_clearance = True
    tabletop_opposed_force_pressure_max_object_height_delta = 0.005

    # Coarse geometry shaping has one job: discover the first strict opposed
    # grasp. Once discovered, retain only a small reacquisition floor and move
    # the policy onto the stricter clean-settle objective.
    tabletop_strict_grasp_milestone_enabled = True
    tabletop_strict_grasp_milestone_bonus = 100000.0
    tabletop_acquisition_reward_post_strict_grasp_floor = 0.05
    tabletop_clean_settle_strict_grasp_rew_scale = 250.0
    tabletop_clean_grasp_readiness_rew_scale = 30000.0
    tabletop_clean_grasp_readiness_distance_scale = 0.006
    tabletop_clean_grasp_readiness_height_scale = 0.005
    tabletop_clean_grasp_readiness_object_speed_scale = 0.08
    tabletop_clean_grasp_readiness_relative_speed_scale = 0.12
    tabletop_clean_grasp_readiness_opposition_floor = 0.25
    tabletop_clean_grasp_readiness_potential_gamma = 0.99

    # The v8 policy drove geometric hold return above 1.2M without ever
    # latching or lifting. Strict hold is already represented by the stronger
    # clean-candidate streak, so it must not remain an independently farmable
    # objective in this task.
    tabletop_strict_hold_rew_scale = 0.0
    success_bonus = 1000000.0

    tabletop_gate_contact_rewards_by_clearance = True
    tabletop_gate_boolean_grasp_rewards_by_clearance = True
    tabletop_contact_clearance_gate_min = 0.0
    tabletop_contact_clearance_gate_scale = 0.10
    tabletop_arm_clearance_penalty_scale = 10000.0
    tabletop_arm_clearance_ok_penalty_threshold = 0.05
    tabletop_terminate_on_arm_clearance_violation = True
    tabletop_arm_clearance_terminate_penalty_threshold = 0.10
    tabletop_arm_clearance_violation_terminate_start_step = 4

    # The object must remain in a compact 8 cm hover envelope with low world
    # and palm-relative velocity for 0.3 s at the 60 Hz policy rate.
    stable_object_palm_vel = 0.12
    dynamic_success_hold_steps = 18
    tabletop_success_requires_hover_target = True
    tabletop_hover_height_delta = 0.080
    tabletop_hover_latch_lift_progress = 0.35
    tabletop_hover_success_requires_xy = True
    tabletop_hover_success_xy_tolerance = 0.060
    tabletop_hover_success_z_tolerance = 0.025
    tabletop_hover_success_object_speed = 0.12
    tabletop_hover_success_object_ang_speed = 2.0
    tabletop_hover_xy_distance_scale = 0.060
    tabletop_hover_z_distance_scale = 0.025
    tabletop_hover_object_speed_scale = 0.12
    tabletop_hover_ang_speed_scale = 2.0
    tabletop_hover_phase_out_lift_rewards = True
    tabletop_hover_post_latch_lift_reward_floor = 1.0
    tabletop_hover_potential_progress_rew_scale = 8000.0
    tabletop_hover_potential_progress_clip = 2.0

    # Distal force sensors remain diagnostics. They cannot define success
    # because valid Revo2 loads can transfer through proximal links or palm.
    tabletop_success_requires_force_grasp = False
    tabletop_lift_gate_requires_force_grasp = False
    tabletop_lift_rewards_require_force_grasp = False
    tabletop_force_lift_curriculum_enabled = False
    tabletop_force_lift_curriculum_start_frames = 0
    tabletop_force_lift_curriculum_end_frames = 3_000_000
    tabletop_force_lift_curriculum_override_alpha = 0.0
    tabletop_hover_latch_requires_force_grasp = False
    tabletop_arm_lift_progress_baseline_mode = "first_strict_grasp"
    tabletop_arm_lift_progress_baseline_grasp_streak = 1
    tabletop_force_reward_post_lift_floor = 0.0
    tabletop_force_thumb_contact_rew_scale = 0.0
    tabletop_force_grasp_rew_scale = 0.0
    tabletop_force_grasp_streak_rew_scale = 0.0
    tabletop_force_stable_grasp_rew_scale = 0.0
    tabletop_force_grasp_loss_penalty_scale = 0.0
    tabletop_force_grasp_streak_target = 12

    # After the clean latch, require a mostly vertical wrist trajectory and a
    # bounded object-to-palm transform instead of a large joint-space sweep.
    tabletop_success_requires_relative_palm_lift = True
    tabletop_success_min_relative_palm_lift = 0.040
    tabletop_success_max_object_palm_drift = 0.035
    tabletop_success_max_palm_xy_drift = 0.030
    tabletop_success_max_palm_orientation_drift = 0.35
    tabletop_palm_object_up_vel_rew_scale = 6000.0
    tabletop_vertical_palm_velocity_rew_scale = 12000.0
    tabletop_vertical_palm_carry_rew_scale = 8000.0
    tabletop_vertical_palm_xy_vel_scale = 0.040
    tabletop_vertical_palm_ang_vel_scale = 0.50
    tabletop_object_palm_drift_tolerance = 0.020
    tabletop_object_palm_drift_scale = 0.040
    tabletop_object_palm_drift_penalty_scale = 8000.0

    # Acquisition and initial lift remain dense, but after the hover target is
    # latched the dominant return comes from braking and holding the object.
    tabletop_object_up_vel_rew_scale = 1500.0
    tabletop_object_carry_lift_rew_scale = 8000.0
    tabletop_hover_height_progress_rew_scale = 3000.0
    tabletop_hover_target_rew_scale = 3000.0
    tabletop_hover_goal_rew_scale = 8000.0
    tabletop_hover_stable_rew_scale = 12000.0
    tabletop_hover_linear_penalty_scale = 500.0
    tabletop_hover_overshoot_penalty_scale = 800.0
    tabletop_hover_z_vel_penalty_scale = 800.0
    tabletop_hover_vel_penalty_scale = 600.0
    tabletop_hover_target_drift_penalty_scale = 800.0
    tabletop_hover_grasp_loss_penalty_scale = 6000.0
    tabletop_hover_under_height_penalty_scale = 1200.0
    tabletop_hover_post_latch_speed_penalty_scale = 1200.0


class _StaticCleanGraspCurriculumContract:
    """Train through reachable contact gates, then recover the strict v12 contract."""

    benchmark_protocol = (
        "static_stable_hover_from_scratch_v13_clean_contact_curriculum_strict_eval"
    )
    tabletop_clean_grasp_curriculum_enabled = True
    tabletop_clean_grasp_curriculum_override_alpha = None
    tabletop_clean_grasp_curriculum_start_frames = 1_000_000
    tabletop_clean_grasp_curriculum_end_frames = 5_000_000
    tabletop_clean_grasp_curriculum_start_contact_distance = 0.008
    tabletop_clean_grasp_curriculum_start_max_object_speed = 0.16
    tabletop_clean_grasp_curriculum_start_max_relative_speed = 0.24
    tabletop_clean_grasp_curriculum_start_hold_steps = 2


class _StaticSynchronousContactRewardContract:
    """Make dense contact return depend on every finger required by success."""

    benchmark_protocol = (
        "static_stable_hover_from_scratch_v14_synchronous_contact_reward_strict_eval"
    )
    strict_non_thumb_pair_score_reduction = "min"
    strict_opposition_touch_reward_requires_non_thumb_pair = True


class _StaticSynchronousContactCurriculumRewardContract:
    """Preserve early contact exploration while closing the one-finger loophole."""

    benchmark_protocol = (
        "static_stable_hover_from_scratch_v15_synchronous_reward_curriculum_strict_eval"
    )
    strict_non_thumb_pair_score_reduction = "curriculum_mean_to_min"
    strict_opposition_touch_pair_gate_mode = "curriculum"
    strict_opposition_touch_reward_requires_non_thumb_pair = False


class _StaticPostCleanLiftExplorationContract:
    """Bridge a valid clean enclosure to controller-independent upward carry."""

    benchmark_protocol = (
        "static_stable_hover_v16_post_clean_palm_lift_exploration_strict_eval"
    )
    tabletop_post_clean_palm_lift_exploration_rew_scale = 12000.0
    tabletop_post_clean_palm_lift_attachment_scale = 0.012


class _StaticPostCleanLiftVelocityContract:
    """Give a latched policy an immediate upward state-space exploration signal."""

    benchmark_protocol = (
        "static_stable_hover_v17_post_clean_palm_up_velocity_strict_eval"
    )
    tabletop_post_clean_palm_up_velocity_rew_scale = 3000.0
    tabletop_post_clean_palm_up_velocity_target = 0.08


class _StaticPostCleanGripRetentionContract:
    """Keep a physically opposed multi-finger enclosure while lift is explored."""

    benchmark_protocol = (
        "static_stable_hover_v18_post_clean_grip_retention_strict_eval"
    )
    tabletop_post_clean_grip_retention_rew_scale = 8000.0
    tabletop_post_clean_grip_loss_penalty_scale = 8000.0
    tabletop_post_clean_grip_retention_target = 0.50
    tabletop_post_clean_grip_stationary_reward_floor = 0.10


class _StaticPostCleanActionContinuityContract:
    """Retain strict grasp and prevent abrupt semantic hand opening after latch."""

    benchmark_protocol = (
        "static_stable_hover_v19_post_clean_hand_continuity_strict_eval"
    )
    tabletop_post_clean_grip_retention_target = 0.30
    tabletop_post_clean_strict_grasp_retention_rew_scale = 20000.0
    tabletop_post_clean_hand_opening_penalty_scale = 20000.0


class _StaticSmoothLiftPhaseObservationContract:
    """Expose the post-latch phase continuously so hand targets cannot jump at one bit."""

    benchmark_protocol = (
        "static_stable_hover_v20_smooth_lift_phase_observation_strict_eval"
    )
    tabletop_clean_lift_phase_ramp_enabled = True
    tabletop_clean_lift_phase_ramp_steps = 30


class _StaticForceBackedCleanLiftContract:
    """Unlock lift only after a sustained load-bearing opposed enclosure."""

    benchmark_protocol = (
        "static_stable_hover_v21_force_backed_clean_lift_strict_eval"
    )
    tabletop_clean_grasp_requires_force_grasp = True
    tabletop_clean_grasp_min_force_streak_steps = 4
    tabletop_success_requires_force_grasp = True
    tabletop_lift_gate_requires_force_grasp = True
    tabletop_lift_rewards_require_force_grasp = True
    tabletop_no_lift_uses_force_grasp_gate = True
    tabletop_arm_lift_progress_baseline_mode = "first_force_grasp"
    tabletop_arm_lift_progress_baseline_grasp_streak = 4

    # Keep the geometric acquisition landscape, then make sustained opposed
    # pressure the dominant transition signal before any upward reward unlocks.
    tabletop_acquisition_reward_post_strict_grasp_floor = 0.25
    tabletop_force_reward_post_lift_floor = 1.0
    tabletop_force_thumb_contact_rew_scale = 4000.0
    tabletop_force_grasp_rew_scale = 8000.0
    tabletop_force_grasp_streak_rew_scale = 12000.0
    tabletop_force_stable_grasp_rew_scale = 12000.0
    tabletop_force_grasp_loss_penalty_scale = 24000.0
    tabletop_force_grasp_streak_target = 12
    tabletop_opposed_force_pressure_rew_scale = 12000.0


class _StaticForceBackedPinchLiftContract:
    """Bootstrap lift from a pinch without weakening final three-finger success."""

    benchmark_protocol = (
        "static_stable_hover_v22b_force_backed_pinch_lift_strict_eval"
    )
    tabletop_clean_grasp_min_non_thumb_contacts = 1
    object_force_grasp_min_non_thumb_contacts = 1
    tabletop_clean_grasp_min_force_streak_steps = 3
    tabletop_arm_lift_progress_baseline_grasp_streak = 1
    tabletop_lift_gate_requires_current_strict_grasp = False
    tabletop_lift_rewards_require_current_strict_grasp = False


class _StaticForceRetainedObjectLiftContract:
    """Require current physical grip for every dense post-latch lift signal."""

    benchmark_protocol = (
        "static_stable_hover_v23_force_retained_object_lift_strict_eval"
    )
    tabletop_force_lift_curriculum_override_alpha = 1.0
    tabletop_clean_grasp_latch_bonus = 100000.0
    tabletop_acquisition_reward_post_clean_latch_floor = 0.25
    tabletop_force_grasp_loss_penalty_scale = 100000.0

    # The v22 policy already explores upward wrist motion. Remove wrist-only
    # bridge credit so continuation can improve only by carrying the object.
    tabletop_post_clean_palm_lift_exploration_rew_scale = 0.0
    tabletop_post_clean_palm_up_velocity_rew_scale = 0.0
    tabletop_vertical_palm_velocity_rew_scale = 0.0
    tabletop_grasped_palm_lift_rew_scale = 0.0


class _StaticSustainedForceHoldBeforeLiftContract:
    """Require a settled force hold before exposing object-lift credit."""

    benchmark_protocol = (
        "static_stable_hover_v24_sustained_force_hold_before_lift_strict_eval"
    )
    # The v23 policy could collect a large latch bonus from a three-frame
    # thumb-index impact, then received a large loss penalty as that impact
    # ended. Keep transient contacts on the dense force curriculum and expose
    # the lift phase only after a genuinely sustained hold.
    tabletop_clean_grasp_min_force_streak_steps = 12
    tabletop_force_grasp_streak_target = 12
    tabletop_force_grasp_loss_penalty_scale = 30000.0


class _StaticForceCoupledMicroLiftContract:
    """Convert a settled force hold into the first object-coupled lift."""

    benchmark_protocol = (
        "static_stable_hover_v25_force_coupled_micro_lift_strict_eval"
    )
    # Once v24 has learned the twelve-step force hold, a stationary pinch must
    # no longer dominate the return. Retain enough pressure reward to preserve
    # contact while making a gentle, force-gated upward wrist velocity the
    # stronger local improvement.
    tabletop_clean_grasp_latch_bonus = 20000.0
    tabletop_acquisition_reward_post_clean_latch_floor = 0.05
    tabletop_force_thumb_contact_rew_scale = 1000.0
    tabletop_force_grasp_rew_scale = 2000.0
    tabletop_force_grasp_streak_rew_scale = 3000.0
    tabletop_force_stable_grasp_rew_scale = 4000.0
    tabletop_opposed_force_pressure_rew_scale = 4000.0

    tabletop_post_clean_palm_up_velocity_rew_scale = 48000.0
    tabletop_post_clean_palm_up_velocity_target = 0.04
    tabletop_palm_object_up_vel_rew_scale = 48000.0
    tabletop_vertical_palm_velocity_rew_scale = 96000.0
    tabletop_object_up_vel_rew_scale = 16000.0
    tabletop_object_carry_lift_rew_scale = 48000.0
    tabletop_no_lift_after_grasp_grace_steps = 12
    tabletop_no_lift_after_grasp_ramp_steps = 30
    tabletop_no_lift_after_grasp_penalty_scale = 16000.0


@configclass
class Revo2StaticActionInterfaceJointPhaseObsTeacherEnvCfg(
    _CartesianWristAcquisitionStage1Contract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage1TeacherEnvCfg,
):
    """Static from-scratch A/B arm using normalized joint-position targets."""

    reference_name = "revo2_static_action_ablation_joint_phase_obs_teacher"
    action_contract = "revo2_semantic_13d"
    action_space = 13
    policy_action_interface = "joint_target"
    observation_space = 90
    tabletop_privileged_phase_obs_enabled = True


@configclass
class Revo2StaticActionInterfaceJointDeltaPhaseObsTeacherEnvCfg(
    _Revo2StaticJointDeltaPolicyContract,
    _CartesianWristAcquisitionStage1Contract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage1TeacherEnvCfg,
):
    """Static from-scratch A/B arm using incremental joint-position targets."""

    reference_name = "revo2_static_action_ablation_joint_delta_phase_obs_teacher"


@configclass
class Revo2StaticActionInterfaceCartesianPhaseObsTeacherEnvCfg(
    _CartesianWristDirectPolicyContract,
    _CartesianWristAcquisitionStage1Contract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage1TeacherEnvCfg,
):
    """Static from-scratch A/B arm using incremental Cartesian wrist targets."""

    reference_name = "revo2_static_action_ablation_cartesian_phase_obs_teacher"
    action_contract = "revo2_cartesian_wrist_12d"
    observation_space = 89
    tabletop_privileged_phase_obs_enabled = True


@configclass
class Revo2StaticActionInterfaceCartesianImpedancePhaseObsTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    _CartesianWristAcquisitionStage1Contract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage1TeacherEnvCfg,
):
    """Static Revo2 A/B arm using FCI-style torque Cartesian impedance."""

    reference_name = "revo2_static_action_ablation_cartesian_impedance_phase_obs_teacher"
    action_contract = "revo2_cartesian_impedance_12d"
    observation_space = 89
    tabletop_privileged_phase_obs_enabled = True
    sim_hand_joint_names = REVO2_V699_PHYSICAL_HAND_JOINT_NAMES
    robot_mimic_natural_frequency = None
    robot_mimic_damping_ratio = None
    robot_cfg: ArticulationCfg = _v699_revo2_impedance_robot_cfg(
        REVO2_STATIC_CLEAR_HOME_ARM_POS
    )


@configclass
class Revo2StaticActionInterfaceJointStage2HoldPhaseObsTeacherEnvCfg(
    _Revo2StaticGraspHoldStage2Contract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage2HoldTeacherEnvCfg,
):
    """Static joint-target continuation for sustained physical closure."""

    reference_name = "revo2_static_action_ablation_joint_stage2_hold_phase_obs_teacher"
    action_contract = "revo2_semantic_13d"
    action_space = 13
    policy_action_interface = "joint_target"
    observation_space = 90
    tabletop_privileged_phase_obs_enabled = True


@configclass
class Revo2StaticActionInterfaceJointDeltaStage2HoldPhaseObsTeacherEnvCfg(
    _Revo2StaticGraspHoldStage2Contract,
    _Revo2StaticJointDeltaPolicyContract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage2HoldTeacherEnvCfg,
):
    """Static joint-delta continuation for sustained physical closure."""

    reference_name = "revo2_static_action_ablation_joint_delta_stage2_hold_phase_obs_teacher"


@configclass
class Revo2StaticActionInterfaceCartesianStage2HoldPhaseObsTeacherEnvCfg(
    _Revo2StaticGraspHoldStage2Contract,
    _CartesianWristDirectPolicyContract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage2HoldTeacherEnvCfg,
):
    """Static Cartesian continuation for sustained physical closure."""

    reference_name = "revo2_static_action_ablation_cartesian_stage2_hold_phase_obs_teacher"
    action_contract = "revo2_cartesian_wrist_12d"
    observation_space = 89
    tabletop_privileged_phase_obs_enabled = True


@configclass
class Revo2StaticActionInterfaceJointDeltaFromScratchContactPhaseObsTeacherEnvCfg(
    _Revo2StaticFromScratchContactContract,
    _Revo2StaticJointDeltaPolicyContract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage2HoldTeacherEnvCfg,
):
    """Home-reset JointDelta acquisition under the matched physical reward."""

    reference_name = "revo2_static_action_ablation_joint_delta_from_scratch_contact_phase_obs_teacher"


@configclass
class Revo2StaticActionInterfaceCartesianFromScratchContactPhaseObsTeacherEnvCfg(
    _Revo2StaticFromScratchContactContract,
    _CartesianWristDirectPolicyContract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage2HoldTeacherEnvCfg,
):
    """Home-reset Cartesian acquisition under the matched physical reward."""

    reference_name = "revo2_static_action_ablation_cartesian_from_scratch_contact_phase_obs_teacher"
    action_contract = "revo2_cartesian_wrist_12d"
    observation_space = 89
    tabletop_privileged_phase_obs_enabled = True


@configclass
class Revo2StaticActionInterfaceJointDeltaStage2LiftBridgePhaseObsTeacherEnvCfg(
    _Revo2StaticMicroLiftVelocityBridgeContract,
    _Revo2StaticJointDeltaPolicyContract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage2HoldTeacherEnvCfg,
):
    """Joint-delta continuation from a completed hold to physical micro-lift."""

    reference_name = "revo2_static_action_ablation_joint_delta_stage2_lift_bridge_phase_obs_teacher"


@configclass
class Revo2StaticActionInterfaceCartesianStage2LiftBridgePhaseObsTeacherEnvCfg(
    _Revo2StaticMicroLiftVelocityBridgeContract,
    _CartesianWristDirectPolicyContract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage2HoldTeacherEnvCfg,
):
    """Cartesian continuation from a completed hold to physical micro-lift."""

    reference_name = "revo2_static_action_ablation_cartesian_stage2_lift_bridge_phase_obs_teacher"
    action_contract = "revo2_cartesian_wrist_12d"
    observation_space = 89
    tabletop_privileged_phase_obs_enabled = True


@configclass
class Revo2StaticActionInterfaceJointStage3LiftPhaseObsTeacherEnvCfg(
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage3TeacherEnvCfg,
):
    """Static joint-target continuation for strict lift and stable hold."""

    reference_name = "revo2_static_action_ablation_joint_stage3_lift_phase_obs_teacher"
    action_contract = "revo2_semantic_13d"
    action_space = 13
    policy_action_interface = "joint_target"
    observation_space = 90
    tabletop_privileged_phase_obs_enabled = True


@configclass
class Revo2StaticActionInterfaceJointDeltaStage3LiftPhaseObsTeacherEnvCfg(
    _Revo2StaticJointDeltaPolicyContract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage3TeacherEnvCfg,
):
    """Static joint-delta continuation for strict lift and stable hold."""

    reference_name = "revo2_static_action_ablation_joint_delta_stage3_lift_phase_obs_teacher"


@configclass
class Revo2StaticActionInterfaceCartesianStage3LiftPhaseObsTeacherEnvCfg(
    _CartesianWristDirectPolicyContract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage3TeacherEnvCfg,
):
    """Static Cartesian continuation for strict lift and stable hold."""

    reference_name = "revo2_static_action_ablation_cartesian_stage3_lift_phase_obs_teacher"
    action_contract = "revo2_cartesian_wrist_12d"
    observation_space = 89
    tabletop_privileged_phase_obs_enabled = True


@configclass
class Revo2StaticStrictJointTargetABTeacherEnvCfg(
    _StaticStrictFromScratchControlABContract,
    _Revo2StaticActionInterfaceAblationContract,
    Revo2UnifiedRollingStage3TeacherEnvCfg,
):
    """Full from-scratch static benchmark with 7D Franka joint targets."""

    reference_name = "revo2_static_strict_joint_target_ab_teacher"
    action_contract = "revo2_semantic_13d"
    action_space = 13
    policy_action_interface = "joint_target"
    observation_space = 80
    tabletop_privileged_phase_obs_enabled = True
    sim_hand_joint_names = REVO2_V699_PHYSICAL_HAND_JOINT_NAMES
    robot_mimic_natural_frequency = None
    robot_mimic_damping_ratio = None
    robot_cfg: ArticulationCfg = _v699_revo2_six_motor_robot_cfg(
        REVO2_STATIC_CLEAR_HOME_ARM_POS
    )


@configclass
class Revo2StaticStrictCartesianImpedanceABTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    Revo2StaticStrictJointTargetABTeacherEnvCfg,
):
    """Matched static benchmark with measured base-frame EEF delta and torque OSC."""

    reference_name = "revo2_static_strict_cartesian_impedance_ab_teacher"
    action_contract = "revo2_cartesian_impedance_12d"
    cartesian_impedance_target_mode = "measured_delta"
    observation_space = 79
    robot_cfg: ArticulationCfg = _v699_revo2_impedance_robot_cfg(
        REVO2_STATIC_CLEAR_HOME_ARM_POS
    )


@configclass
class Revo2StaticStableHoverJointTargetABTeacherEnvCfg(
    _StaticStableHoverControlABContract,
    Revo2StaticStrictJointTargetABTeacherEnvCfg,
):
    """Revo2 joint-target A/B with a low-speed stable-hover success contract."""

    reference_name = "revo2_static_stable_hover_joint_target_ab_teacher"
    observation_space = 95


@configclass
class Revo2StaticStableHoverCartesianImpedanceABTeacherEnvCfg(
    _StaticStableHoverControlABContract,
    Revo2StaticStrictCartesianImpedanceABTeacherEnvCfg,
):
    """Revo2 measured-delta OSC A/B with the same stable-hover contract."""

    reference_name = "revo2_static_stable_hover_cartesian_impedance_ab_teacher"
    observation_space = 94


@configclass
class Revo2StaticStableHoverJointTargetCleanCurriculumTeacherEnvCfg(
    _StaticCleanGraspCurriculumContract,
    Revo2StaticStableHoverJointTargetABTeacherEnvCfg,
):
    """Revo2 v13 teacher with a training-only clean-contact curriculum."""

    reference_name = "revo2_static_stable_hover_joint_target_clean_curriculum_teacher"


@configclass
class Revo2StaticStableHoverJointTargetSynchronousContactTeacherEnvCfg(
    _StaticSynchronousContactRewardContract,
    Revo2StaticStableHoverJointTargetCleanCurriculumTeacherEnvCfg,
):
    """Revo2 v14 teacher with bottleneck multi-finger contact shaping."""

    reference_name = "revo2_static_stable_hover_joint_target_synchronous_contact_teacher"


@configclass
class Revo2StaticStableHoverJointTargetSynchronousCurriculumTeacherEnvCfg(
    _StaticSynchronousContactCurriculumRewardContract,
    Revo2StaticStableHoverJointTargetCleanCurriculumTeacherEnvCfg,
):
    """Revo2 v15 teacher with a smooth mean-to-bottleneck reward transition."""

    reference_name = (
        "revo2_static_stable_hover_joint_target_synchronous_curriculum_teacher"
    )


@configclass
class Revo2StaticStableHoverJointTargetPostCleanLiftTeacherEnvCfg(
    _StaticPostCleanLiftExplorationContract,
    Revo2StaticStableHoverJointTargetSynchronousCurriculumTeacherEnvCfg,
):
    """Revo2 v16 teacher with a clean-grasp-to-lift exploration bridge."""

    reference_name = "revo2_static_stable_hover_joint_target_post_clean_lift_teacher"


@configclass
class Revo2StaticStableHoverJointTargetPostCleanLiftVelocityTeacherEnvCfg(
    _StaticPostCleanLiftVelocityContract,
    Revo2StaticStableHoverJointTargetPostCleanLiftTeacherEnvCfg,
):
    """Revo2 v17 teacher with an attachment-gated upward palm velocity bridge."""

    reference_name = (
        "revo2_static_stable_hover_joint_target_post_clean_lift_velocity_teacher"
    )


@configclass
class Revo2StaticStableHoverJointTargetPostCleanGripRetentionTeacherEnvCfg(
    _StaticPostCleanGripRetentionContract,
    Revo2StaticStableHoverJointTargetPostCleanLiftVelocityTeacherEnvCfg,
):
    """Revo2 v18 teacher retaining real opposed contact during lift exploration."""

    reference_name = (
        "revo2_static_stable_hover_joint_target_post_clean_grip_retention_teacher"
    )


@configclass
class Revo2StaticStableHoverJointTargetPostCleanActionContinuityTeacherEnvCfg(
    _StaticPostCleanActionContinuityContract,
    Revo2StaticStableHoverJointTargetPostCleanGripRetentionTeacherEnvCfg,
):
    """Revo2 v19 teacher with reward-only post-latch hand continuity."""

    reference_name = (
        "revo2_static_stable_hover_joint_target_post_clean_action_continuity_teacher"
    )


@configclass
class Revo2StaticStableHoverJointTargetSmoothLiftPhaseTeacherEnvCfg(
    _StaticSmoothLiftPhaseObservationContract,
    Revo2StaticStableHoverJointTargetPostCleanActionContinuityTeacherEnvCfg,
):
    """Revo2 v20 teacher with a 30-step observable post-latch transition."""

    reference_name = (
        "revo2_static_stable_hover_joint_target_smooth_lift_phase_teacher"
    )


@configclass
class Revo2StaticStableHoverJointTargetForceBackedCleanLiftTeacherEnvCfg(
    _StaticForceBackedCleanLiftContract,
    Revo2StaticStableHoverJointTargetSmoothLiftPhaseTeacherEnvCfg,
):
    """Revo2 v21 teacher whose lift phase starts from physical contact."""

    reference_name = (
        "revo2_static_stable_hover_joint_target_force_backed_clean_lift_teacher"
    )


@configclass
class Revo2StaticStableHoverJointTargetForceBackedPinchLiftTeacherEnvCfg(
    _StaticForceBackedPinchLiftContract,
    Revo2StaticStableHoverJointTargetForceBackedCleanLiftTeacherEnvCfg,
):
    """Revo2 v22 teacher using a physical thumb-finger sphere pinch."""

    reference_name = (
        "revo2_static_stable_hover_joint_target_force_backed_pinch_lift_teacher"
    )


@configclass
class Revo2StaticStableHoverJointTargetForceRetainedObjectLiftTeacherEnvCfg(
    _StaticForceRetainedObjectLiftContract,
    Revo2StaticStableHoverJointTargetForceBackedPinchLiftTeacherEnvCfg,
):
    """Revo2 v23 teacher rewarded only for force-retained object lift."""

    reference_name = (
        "revo2_static_stable_hover_joint_target_force_retained_object_lift_teacher"
    )


@configclass
class Revo2StaticStableHoverJointTargetSustainedForceLiftTeacherEnvCfg(
    _StaticSustainedForceHoldBeforeLiftContract,
    Revo2StaticStableHoverJointTargetForceRetainedObjectLiftTeacherEnvCfg,
):
    """Revo2 v24 teacher that settles a load-bearing hold before lifting."""

    reference_name = (
        "revo2_static_stable_hover_joint_target_sustained_force_lift_teacher"
    )


@configclass
class Revo2StaticStableHoverJointTargetForceCoupledMicroLiftTeacherEnvCfg(
    _StaticForceCoupledMicroLiftContract,
    Revo2StaticStableHoverJointTargetSustainedForceLiftTeacherEnvCfg,
):
    """Revo2 v25 teacher converting its settled pinch into object lift."""

    reference_name = (
        "revo2_static_stable_hover_joint_target_force_coupled_micro_lift_teacher"
    )


class _StaticOfficialSixActiveFromScratchContract:
    """Simple physical static benchmark shared by both Franka controllers."""

    benchmark_protocol = "revo2_static_official_six_active_from_scratch_v2_measured_bounded"

    # Start from the original learnable tabletop cube and freeze all motion and
    # reset randomization. Controller and hand physics are the only A/B change.
    episode_length_s = 8.0
    object_cfg: RigidObjectCfg = _cube_object_cfg(
        size=(0.040, 0.040, 0.080),
        mass=0.030,
        pos=(0.58, -0.16, 0.336),
    )
    object_shape = "box"
    object_size = (0.040, 0.040, 0.080)
    # Table top z (0.296 m) plus the cube half-height (0.040 m).
    # This keeps the reset, lift baseline, and physically settled pose equal.
    object_start_pos = (0.58, -0.16, 0.336)
    reset_arm_pos_noise = 0.0
    reset_object_pos_noise = (0.0, 0.0, 0.0)
    object_lin_vel_min = (0.0, 0.0, 0.0)
    object_lin_vel_max = (0.0, 0.0, 0.0)
    object_ang_vel_min = (0.0, 0.0, 0.0)
    object_ang_vel_max = (0.0, 0.0, 0.0)
    dynamic_grasp_speed_curriculum = False
    dynamic_grasp_speed_curriculum_override_alpha = 0.0
    dynamic_tabletop_persistent_motion = False
    dynamic_tabletop_release_motion_on_contact = False
    dynamic_tabletop_randomize_yaw = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.0, 0.0)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_heading_range = (0.0, 0.0)
    dynamic_tabletop_pregrasp_lead_time = 0.0
    dynamic_tabletop_pregrasp_ahead_distance = 0.0

    # No policy prior, residual, target action, post-success lock, or
    # controller-coordinate reward is allowed in this from-scratch benchmark.
    scripted_action_prior_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_approach_action_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_relative_lift_target_prior_enabled = False
    scripted_tabletop_cartesian_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False
    tabletop_privileged_lift_target_obs_enabled = False
    lift_action_prior_rew_scale = 0.0
    tabletop_lift_action_prior_rew_scale = 0.0
    tabletop_lift_target_rew_scale = 0.0
    action_penalty_scale = 0.0
    arm_target_delta_penalty_scale = 0.0
    tabletop_post_success_stability_latch_enabled = False
    tabletop_post_success_arm_target_lock_enabled = False
    tabletop_post_success_hand_target_lock_enabled = False
    tabletop_lift_hand_target_lock_enabled = False

    # Both controllers see the same physical state objective. The policy must
    # maintain a current opposed enclosure and real PhysX contact while the
    # object remains near an 8 cm hover target at low linear/angular velocity.
    joint_target_arm_max_delta = 0.04
    joint_target_hand_max_delta = 0.05
    joint_target_rate_limit_requires_lift_baseline = False
    arm_target_tracking_error_limit = 0.04
    tabletop_arm_lift_progress_mode = "palm_z"
    contact_distance = 0.012
    contact_score_scale = 0.014
    min_finger_contacts = 3
    min_non_thumb_contacts = 2
    opposition_cos_threshold = 0.0
    strict_success_enabled = True
    strict_reward_enabled = True
    strict_success_contact_distance = 0.008
    strict_success_min_finger_contacts = 3
    strict_success_min_non_thumb_contacts = 2
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    strict_reward_contact_score_scale = 0.012
    strict_approach_score_scale = 0.060
    strict_touch_score_scale = 0.008
    strict_touch_reward_target_distance = 0.002
    object_contact_force_diagnostics_enabled = True
    object_contact_force_threshold = 0.05
    object_force_grasp_min_non_thumb_contacts = 1
    tabletop_success_requires_force_grasp = True
    lift_success_height = 0.040
    tabletop_success_lift_height = 0.040
    stable_object_palm_vel = 0.15
    dynamic_success_hold_steps = 12
    tabletop_success_uses_grasp_seen = False
    tabletop_success_requires_hover_target = True
    tabletop_success_requires_arm_clearance = True
    tabletop_terminate_on_arm_clearance_violation = False
    terminate_on_success = True
    tabletop_hover_latch_uses_grasp_seen = False
    tabletop_hover_reward_uses_grasp_seen = False
    tabletop_hover_latch_requires_force_grasp = True
    tabletop_hover_height_delta = 0.080
    tabletop_hover_latch_lift_progress = 0.25
    tabletop_hover_success_requires_xy = True
    tabletop_hover_success_xy_tolerance = 0.070
    tabletop_hover_success_z_tolerance = 0.030
    tabletop_hover_success_object_speed = 0.15
    tabletop_hover_success_object_ang_speed = 2.5
    tabletop_hover_xy_distance_scale = 0.070
    tabletop_hover_z_distance_scale = 0.030
    tabletop_hover_object_speed_scale = 0.15
    tabletop_hover_ang_speed_scale = 2.5
    tabletop_lift_rewards_require_current_strict_grasp = True
    tabletop_lift_rewards_require_force_grasp = False
    tabletop_lift_gate_requires_current_strict_grasp = True
    tabletop_lift_gate_requires_force_grasp = False
    tabletop_lift_use_grasp_seen_gate = False

    # Five compact state-reward groups: reach, opposed closure, real contact,
    # physical lift, and stable hover. Values stay close to the simple baseline
    # and intentionally avoid the old multi-stage 10k-scale shaping stack.
    palm_reach_rew_scale = 8.0
    fingertip_reach_rew_scale = 8.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 12.0
    dynamic_tabletop_pregrasp_height_rew_scale = 8.0
    dynamic_tabletop_palm_frame_pregrasp_rew_scale = 0.0
    contact_rew_scale = 30.0
    true_grasp_rew_scale = 60.0
    opposition_rew_scale = 30.0
    grasp_quality_rew_scale = 40.0
    strict_approach_rew_scale = 8.0
    strict_multifinger_approach_rew_scale = 12.0
    strict_opposition_approach_rew_scale = 25.0
    strict_thumb_touch_rew_scale = 25.0
    strict_touch_rew_scale = 40.0
    strict_opposition_touch_rew_scale = 80.0
    catch_progress_rew_scale = 30.0
    lift_progress_rew_scale = 200.0
    quality_lift_progress_rew_scale = 0.0
    lifted_true_grasp_rew_scale = 250.0
    tabletop_stable_catch_rew_scale = 250.0
    stable_hold_rew_scale = 300.0
    hold_progress_rew_scale = 400.0
    success_bonus = 1000.0
    tabletop_force_thumb_contact_rew_scale = 20.0
    tabletop_force_grasp_rew_scale = 50.0
    tabletop_force_grasp_streak_rew_scale = 80.0
    tabletop_force_stable_grasp_rew_scale = 100.0
    tabletop_force_grasp_loss_penalty_scale = 100.0
    tabletop_force_grasp_streak_target = 12
    tabletop_hover_height_progress_rew_scale = 120.0
    tabletop_hover_target_rew_scale = 150.0
    tabletop_hover_goal_rew_scale = 250.0
    tabletop_hover_stable_rew_scale = 350.0
    tabletop_hover_linear_penalty_scale = 25.0
    tabletop_hover_overshoot_penalty_scale = 40.0
    tabletop_hover_z_vel_penalty_scale = 30.0
    tabletop_hover_vel_penalty_scale = 25.0
    tabletop_hover_target_drift_penalty_scale = 30.0
    tabletop_hover_grasp_loss_penalty_scale = 100.0
    tabletop_hover_under_height_penalty_scale = 40.0
    tabletop_hover_post_latch_speed_penalty_scale = 40.0
    tabletop_arm_clearance_penalty_scale = 40.0


@configclass
class Revo2StaticOfficialJointTargetTeacherEnvCfg(
    _StaticOfficialSixActiveFromScratchContract,
    Revo2DynamicTabletopTeacherEnvCfg,
):
    """Official Revo2 hand with 7-D Franka joint targets, trained from scratch."""

    reference_name = "revo2_static_official_six_active_joint_target_teacher"
    action_contract = "revo2_semantic_13d"
    action_space = 13
    observation_space = 76
    policy_action_interface = "joint_target"
    sim_hand_joint_names = REVO2_HAND_JOINT_NAMES
    touch_body_names = REVO2_DISTAL_CONTACT_BODY_NAMES
    robot_cfg: ArticulationCfg = _v699_revo2_official_six_active_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


@configclass
class Revo2StaticOfficialCartesianImpedanceTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    Revo2StaticOfficialJointTargetTeacherEnvCfg,
):
    """Matched base-frame EEF-delta and torque-impedance static benchmark."""

    reference_name = "revo2_static_official_six_active_cartesian_impedance_teacher"
    action_contract = "revo2_cartesian_impedance_12d"
    observation_space = 75
    cartesian_impedance_target_mode = "measured_delta"
    robot_cfg: ArticulationCfg = _v699_revo2_official_six_active_impedance_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


class _StaticCanonicalFromScratchContract:
    """Progress-lift-target static grasp curriculum shared by both hands."""

    benchmark_protocol = "static_canonical_progress_lift_target_v2_5"
    global_action_selection_scope = "revo2_and_inspire"
    hand_action_semantics = "six_physical_motor_absolute_target"
    canonical_static_reward_enabled = True
    canonical_progress_only_approach = True

    # Random policy weights are trained from an object-aligned, collision-free
    # reset that continuously moves back to the upright Franka home.  This is a
    # reset distribution, not an action prior or residual controller.
    canonical_reset_curriculum_enabled = True
    canonical_reset_curriculum_override_alpha = None
    canonical_reset_curriculum_start_frames = 10_000_000
    canonical_reset_curriculum_end_frames = 50_000_000
    canonical_reset_curriculum_mixed_distribution = True
    canonical_reset_curriculum_pregrasp_anchor_fraction = 0.20
    canonical_reset_curriculum_hard_anchor_fraction = 0.20
    canonical_reset_home_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    canonical_reset_pregrasp_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    canonical_reset_arm_pos_noise = 0.01
    canonical_reset_hand_action = 0.0

    episode_length_s = 10.0
    reset_arm_pos_noise = 0.0
    reset_object_pos_noise = (0.0, 0.0, 0.0)
    object_cfg: RigidObjectCfg = _cube_object_cfg(
        size=(0.040, 0.040, 0.080),
        mass=0.030,
        pos=(0.58, -0.16, 0.336),
    )
    object_shape = "box"
    object_size = (0.040, 0.040, 0.080)
    object_start_pos = (0.58, -0.16, 0.336)
    object_lin_vel_min = (0.0, 0.0, 0.0)
    object_lin_vel_max = (0.0, 0.0, 0.0)
    object_ang_vel_min = (0.0, 0.0, 0.0)
    object_ang_vel_max = (0.0, 0.0, 0.0)
    dynamic_grasp_speed_curriculum = False
    dynamic_grasp_speed_curriculum_override_alpha = 0.0
    dynamic_tabletop_persistent_motion = False
    dynamic_tabletop_release_motion_on_contact = False
    dynamic_tabletop_randomize_yaw = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = (0.0, 0.0)
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = (0.0, 0.0)

    # SimToolReal-style arm target increments and absolute six-motor hand
    # targets. A zero arm action holds the current reset pose.
    joint_target_arm_action_mode = "incremental"
    joint_target_arm_delta_scale = 0.025
    arm_moving_average = 0.20
    hand_moving_average = 0.20
    joint_target_arm_max_delta = 0.025
    joint_target_hand_max_delta = 0.05
    joint_target_rate_limit_requires_lift_baseline = False
    arm_target_tracking_error_limit = 0.05

    # No scripted reach, closure, lift, motion planner, residual, or post-hoc
    # target lock is part of this benchmark.
    scripted_action_prior_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_approach_action_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_relative_lift_target_prior_enabled = False
    scripted_tabletop_cartesian_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False
    tabletop_privileged_lift_target_obs_enabled = False
    tabletop_lift_hand_target_lock_enabled = False
    tabletop_post_success_stability_latch_enabled = False
    tabletop_post_success_arm_target_lock_enabled = False
    tabletop_post_success_hand_target_lock_enabled = False

    # Training uses smooth geometric and physical terms. Strict contact and
    # force are required only by terminal success, never to unlock lift credit.
    contact_distance = 0.040
    contact_score_scale = 0.025
    min_finger_contacts = 2
    min_non_thumb_contacts = 1
    opposition_cos_threshold = 0.0
    strict_reward_enabled = False
    strict_success_enabled = True
    strict_success_contact_distance = 0.006
    strict_success_min_finger_contacts = 2
    strict_success_min_non_thumb_contacts = 1
    strict_success_opposition_mode = "dot"
    strict_success_opposition_cos_threshold = 0.0
    object_contact_force_diagnostics_enabled = True
    object_contact_force_threshold = 0.05
    object_force_grasp_min_non_thumb_contacts = 1
    tabletop_success_requires_force_grasp = True
    tabletop_lift_requires_clean_grasp_latch = False
    tabletop_lift_gate_requires_current_strict_grasp = False
    tabletop_lift_gate_requires_force_grasp = False
    tabletop_lift_rewards_require_current_strict_grasp = False
    tabletop_lift_rewards_require_force_grasp = False
    tabletop_lift_use_grasp_seen_gate = False
    lift_reward_uses_grasp_quality_gate = False
    lift_reward_min_grasp_quality_multiplier = 1.0
    lift_reward_uses_opposition_gate = False
    lift_reward_min_opposition_multiplier = 1.0
    quality_lift_progress_uses_opposition_gate = False
    quality_lift_progress_min_opposition_multiplier = 1.0

    tabletop_success_lift_height = 0.10
    lift_success_height = 0.10
    tabletop_hover_height_delta = 0.10
    tabletop_hover_latch_lift_progress = 0.95
    tabletop_hover_follow_object_xy_until_latch = False
    tabletop_hover_latch_uses_grasp_seen = False
    tabletop_hover_reward_uses_grasp_seen = False
    tabletop_hover_latch_requires_force_grasp = False
    tabletop_success_uses_grasp_seen = False
    tabletop_success_requires_hover_target = True
    tabletop_hover_success_requires_xy = True
    tabletop_hover_success_xy_tolerance = 0.05
    tabletop_hover_success_z_tolerance = 0.025
    tabletop_hover_success_object_speed = 0.12
    tabletop_hover_success_object_ang_speed = 3.0
    stable_object_palm_vel = 0.12
    dynamic_success_hold_steps = 180
    terminate_on_success = True

    # RobustDexGrasp-style soft table safety during training. Strict success
    # additionally rejects any current clearance violation.
    tabletop_arm_clearance_body_margins = (
        0.060,
        0.060,
        0.050,
        0.040,
        0.030,
        0.020,
        0.015,
        0.010,
    )
    tabletop_arm_clearance_margin = 0.010
    tabletop_arm_clearance_scale = 0.040
    tabletop_arm_clearance_max_penalty = 3.0
    tabletop_arm_clearance_penalty_scale = 1.0
    tabletop_success_requires_arm_clearance = True
    tabletop_terminate_on_arm_clearance_violation = False
    dynamic_tabletop_min_palm_height_offset = 0.020
    dynamic_tabletop_low_palm_height_scale = 0.030
    dynamic_tabletop_low_palm_max_penalty = 3.0
    dynamic_tabletop_gate_contact_rewards_by_pregrasp = False

    # Match the mature SimToolReal/RobustDexGrasp reward topology: approach is
    # paid only when a new closest distance is reached. Persistent return then
    # comes from real object lift and convergence to the hover target, so a
    # policy cannot profit indefinitely by merely touching the object.
    canonical_palm_progress_rew_scale = 8.0
    canonical_fingertip_progress_rew_scale = 14.0
    canonical_lift_step_progress_rew_scale = 20.0
    # A weak reach anchor prevents an incremental arm policy from random
    # walking out of the workspace. It is deliberately much smaller than the
    # lift return and contains no contact or closure reward.
    canonical_palm_reach_rew_scale = 0.5
    canonical_fingertip_reach_rew_scale = 1.5
    canonical_contact_rew_scale = 0.0
    canonical_opposition_rew_scale = 0.0
    canonical_grasp_quality_rew_scale = 0.0
    canonical_force_thumb_rew_scale = 0.0
    canonical_force_grasp_rew_scale = 0.0
    canonical_force_stable_rew_scale = 0.0
    canonical_lift_support_floor = 0.30
    canonical_lift_progress_rew_scale = 45.0
    canonical_lift_milestone_bonus = 150.0
    canonical_lifted_grasp_rew_scale = 12.0
    canonical_hover_goal_rew_scale = 24.0
    canonical_hover_stable_rew_scale = 36.0
    canonical_success_now_rew_scale = 45.0
    canonical_hold_progress_rew_scale = 100.0
    canonical_success_bonus = 300.0
    canonical_arm_clearance_penalty_scale = 20.0
    canonical_low_palm_penalty_scale = 2.0
    canonical_scoop_penalty_scale = 40.0
    canonical_palm_only_penalty_scale = 40.0
    canonical_unsupported_lift_penalty_scale = 18.0
    canonical_action_penalty_scale = 0.002
    canonical_target_delta_penalty_scale = 0.004
    canonical_drop_penalty = 30.0


@configclass
class Revo2StaticCanonicalJointDeltaTeacherEnvCfg(
    _StaticCanonicalFromScratchContract,
    Revo2DynamicTabletopTeacherEnvCfg,
):
    """Official six-active Revo2 canonical static teacher."""

    reference_name = "revo2_static_canonical_joint_delta_teacher"
    action_contract = "revo2_semantic_13d"
    action_space = 13
    observation_space = 76
    policy_action_interface = "joint_target"
    canonical_reset_pregrasp_arm_pos = REVO2_CANONICAL_STATIC_PREGRASP_ARM_POS
    sim_hand_joint_names = REVO2_HAND_JOINT_NAMES
    touch_body_names = REVO2_DISTAL_CONTACT_BODY_NAMES
    robot_cfg: ArticulationCfg = _v699_revo2_official_six_active_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )
    robot_cfg.init_state.pos = CANONICAL_STATIC_ROBOT_BASE_POS


@configclass
class Revo2StaticCanonicalCartesianImpedanceTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    Revo2StaticCanonicalJointDeltaTeacherEnvCfg,
):
    """Matched Revo2 base-frame Cartesian impedance ablation."""

    reference_name = "revo2_static_canonical_cartesian_impedance_teacher"
    action_contract = "revo2_cartesian_impedance_12d"
    observation_space = 75
    cartesian_impedance_target_mode = "measured_delta"
    robot_cfg: ArticulationCfg = _v699_revo2_official_six_active_impedance_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )
    robot_cfg.init_state.pos = CANONICAL_STATIC_ROBOT_BASE_POS


class _StaticCanonicalRobustHomeContract:
    """Robust-home refinement of the canonical from-scratch benchmark."""

    benchmark_protocol = "static_canonical_robust_home_stable_v2_6"

    # Keep easy anchors for exploration, but make the official upright home
    # pose the majority endpoint once the curriculum has matured.
    canonical_reset_curriculum_pregrasp_anchor_fraction = 0.10
    canonical_reset_curriculum_hard_anchor_fraction = 0.50
    canonical_reset_arm_pos_noise = 0.020

    # The v2.5 policy often drove near-maximal increments throughout hover.
    # A smaller physical target step preserves reachability while making the
    # contact-rich lift and settle phase less sensitive to solver perturbations.
    joint_target_arm_delta_scale = 0.015
    joint_target_arm_max_delta = 0.015

    # Reweight the existing state objective toward low-motion hover and a
    # continuous three-second hold. No new reward signal or controller prior is
    # introduced here.
    canonical_hover_goal_rew_scale = 40.0
    canonical_hover_stable_rew_scale = 90.0
    canonical_success_now_rew_scale = 90.0
    canonical_hold_progress_rew_scale = 180.0


@configclass
class Revo2StaticCanonicalRobustJointDeltaTeacherEnvCfg(
    _StaticCanonicalRobustHomeContract,
    Revo2StaticCanonicalJointDeltaTeacherEnvCfg,
):
    """Revo2 v2.6 robust-home JointDelta teacher."""

    reference_name = "revo2_static_canonical_robust_joint_delta_teacher"


@configclass
class Revo2StaticCanonicalRobustCartesianImpedanceTeacherEnvCfg(
    _StaticCanonicalRobustHomeContract,
    Revo2StaticCanonicalCartesianImpedanceTeacherEnvCfg,
):
    """Revo2 v2.6 robust-home Cartesian impedance teacher."""

    reference_name = "revo2_static_canonical_robust_cartesian_impedance_teacher"


class _StaticOfficialSphereLiftControlABContract(
    _StaticStrictFromScratchControlABContract
):
    """Learnable static sphere lift shared by both official hand models."""

    benchmark_protocol = "static_official_sphere_lift_global_action_ab_v1_acquisition"
    global_action_selection_scope = "revo2_and_inspire"
    hand_action_semantics = "six_physical_motor_absolute_target"

    # The archived 44 mm sphere task establishes that the scene and state
    # objective can produce a physical lift. Reuse that simple acquisition
    # problem while retaining the current deployable controller bounds. Keep
    # the common object outside the longer Inspire fingers' reset-settling
    # sweep; using one shared pose avoids an embodiment-specific reset prior.
    object_start_pos = (0.58, -0.16, TABLETOP_ROLLING_START_Z)
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=object_start_pos,
    )
    object_shape = str(TABLETOP_ROLLING_START_SPEC["proxy_shape"])
    object_radius = float(TABLETOP_ROLLING_START_SPEC["radius"])
    object_size = tuple(TABLETOP_ROLLING_START_SPEC["size"])

    # Absolute joint targets must remain local to measured state. This removes
    # the multi-radian position-servo windup used by the archived policy while
    # matching the local correction enforced by Cartesian impedance.
    arm_target_tracking_error_limit = 0.04
    joint_target_arm_max_delta = 0.04
    joint_target_hand_max_delta = 0.05


class _StaticOfficialSphereForcePalmLiftStage2Contract:
    """Shared Stage-2 lift curriculum applied after acquisition training."""

    benchmark_protocol = "static_official_sphere_global_action_ab_v3_stage2_force_palm_lift"
    curriculum_stage = "stage2_force_palm_lift"
    global_action_selection_scope = "revo2_and_inspire"
    hand_action_semantics = "six_physical_motor_absolute_target"

    # Stage 1 supplies a deterministic acquisition mean. Stage 2 keeps its
    # dense geometric shaping, but only real filtered fingertip contact may
    # unlock lift reward or terminal success. Palm-z is a shared physical state
    # metric and therefore does not favor either arm action parameterization.
    tabletop_success_requires_force_grasp = True
    tabletop_lift_gate_requires_current_strict_grasp = False
    tabletop_lift_gate_requires_force_grasp = True
    tabletop_lift_rewards_require_current_strict_grasp = False
    tabletop_lift_rewards_require_force_grasp = True
    tabletop_no_lift_uses_force_grasp_gate = True
    tabletop_arm_lift_progress_mode = "palm_z"
    tabletop_arm_lift_progress_baseline_mode = "first_force_grasp"
    tabletop_arm_lift_progress_baseline_grasp_streak = 6
    tabletop_arm_lift_reward_object_margin = 1.0
    tabletop_grasped_arm_lift_rew_scale = 50000.0

    # Preserve the Stage-1 contact landscape instead of letting rare sampled
    # contacts dominate PPO. The only large new term is physical palm lift
    # while current force contact and low object/palm relative velocity hold.
    tabletop_force_reward_post_lift_floor = 1.0
    tabletop_force_thumb_contact_rew_scale = 500.0
    tabletop_force_grasp_rew_scale = 1000.0
    tabletop_force_grasp_streak_rew_scale = 1000.0
    tabletop_force_stable_grasp_rew_scale = 1500.0
    tabletop_force_grasp_loss_penalty_scale = 6000.0
    tabletop_force_grasp_streak_target = 12


class _StaticOfficialSphereForceHoldStage2Contract:
    """Shared Stage 2 that converts geometric acquisition into load-bearing contact."""

    benchmark_protocol = "static_official_sphere_global_action_ab_v5_stage2_force_hold"
    curriculum_stage = "stage2_force_hold"
    global_action_selection_scope = "revo2_and_inspire"
    hand_action_semantics = "six_physical_motor_absolute_target"

    # The Stage-1 mean can enter the contact envelope, but an 8 mm geometric
    # shell remains rewarding without PhysX contact. Keep a smooth millimeter-
    # scale approach landscape while making the discrete enclosure test match
    # the physical contact scale.
    strict_success_contact_distance = 0.002
    strict_reward_contact_score_scale = 0.004
    strict_touch_score_scale = 0.004
    strict_touch_reward_target_distance = 0.001

    # This stage deliberately contains no lift objective. A first real opposed
    # contact latches the hold phase, then consecutive force contact and low
    # object/palm relative velocity dominate the continuation objective.
    tabletop_arm_lift_progress_mode = "palm_z"
    tabletop_arm_lift_progress_baseline_mode = "first_force_grasp"
    tabletop_arm_lift_progress_baseline_grasp_streak = 1
    tabletop_success_requires_force_grasp = True
    tabletop_lift_gate_requires_current_strict_grasp = False
    tabletop_lift_gate_requires_force_grasp = True
    tabletop_lift_rewards_require_current_strict_grasp = False
    tabletop_lift_rewards_require_force_grasp = True
    tabletop_no_lift_uses_force_grasp_gate = True

    # Spell out the complete acquisition-to-contact transition. In particular,
    # do not inherit the Stage-1 under-wrap or geometric hold bonuses: they can
    # grow while the filtered PhysX contact streak remains zero.
    palm_reach_rew_scale = 8.0
    fingertip_reach_rew_scale = 8.0
    dynamic_tabletop_pregrasp_xy_rew_scale = 12.0
    dynamic_tabletop_pregrasp_height_rew_scale = 8.0
    dynamic_tabletop_palm_frame_pregrasp_rew_scale = 0.0
    contact_rew_scale = 0.0
    true_grasp_rew_scale = 0.0
    opposition_rew_scale = 0.0
    grasp_quality_rew_scale = 0.0
    catch_progress_rew_scale = 0.0
    strict_approach_rew_scale = 8.0
    strict_multifinger_approach_rew_scale = 12.0
    strict_opposition_approach_rew_scale = 25.0
    strict_thumb_touch_rew_scale = 100.0
    strict_touch_rew_scale = 200.0
    strict_opposition_touch_rew_scale = 400.0
    tabletop_underwrap_rew_scale = 0.0
    tabletop_strict_hold_rew_scale = 0.0
    tabletop_strict_grasp_loss_penalty_scale = 0.0
    tabletop_non_thumb_without_thumb_penalty_scale = 0.0

    lift_progress_rew_scale = 0.0
    quality_lift_progress_rew_scale = 0.0
    lifted_true_grasp_rew_scale = 0.0
    tabletop_stable_catch_rew_scale = 0.0
    stable_hold_rew_scale = 0.0
    hold_progress_rew_scale = 0.0
    success_bonus = 0.0
    tabletop_grasped_palm_lift_rew_scale = 0.0
    tabletop_grasped_arm_lift_rew_scale = 0.0
    tabletop_object_up_vel_rew_scale = 0.0
    tabletop_object_carry_lift_rew_scale = 0.0
    tabletop_palm_object_carry_rew_scale = 0.0
    tabletop_palm_object_up_vel_rew_scale = 0.0
    tabletop_vertical_palm_carry_rew_scale = 0.0
    tabletop_vertical_palm_velocity_rew_scale = 0.0
    tabletop_hover_height_progress_rew_scale = 0.0
    tabletop_hover_target_rew_scale = 0.0
    tabletop_hover_goal_rew_scale = 0.0
    tabletop_hover_stable_rew_scale = 0.0
    tabletop_no_lift_after_grasp_penalty_scale = 0.0
    tabletop_lift_without_object_penalty_scale = 1000.0
    tabletop_lift_without_current_grasp_penalty_scale = 0.0
    tabletop_arm_object_lift_gap_penalty_scale = 0.0
    tabletop_object_carry_stall_penalty_scale = 0.0
    tabletop_object_palm_drift_penalty_scale = 0.0
    tabletop_hover_linear_penalty_scale = 0.0
    tabletop_hover_overshoot_penalty_scale = 0.0
    tabletop_hover_z_vel_penalty_scale = 0.0
    tabletop_hover_vel_penalty_scale = 0.0
    tabletop_hover_target_drift_penalty_scale = 0.0
    tabletop_hover_grasp_loss_penalty_scale = 0.0
    tabletop_hover_under_height_penalty_scale = 0.0
    tabletop_hover_post_latch_speed_penalty_scale = 0.0
    tabletop_arm_clearance_penalty_scale = 40.0
    terminate_on_success = False

    # Thumb-only contact is useful as an entry signal but cannot dominate. The
    # large terms require a thumb/non-thumb force pair and reward maintaining it
    # for twenty consecutive control steps.
    tabletop_force_reward_post_lift_floor = 1.0
    tabletop_force_thumb_contact_rew_scale = 500.0
    tabletop_force_grasp_rew_scale = 12000.0
    tabletop_force_grasp_streak_rew_scale = 24000.0
    tabletop_force_stable_grasp_rew_scale = 16000.0
    tabletop_force_grasp_loss_penalty_scale = 12000.0
    tabletop_force_grasp_streak_target = 20


class _StaticOfficialSphereOpposedPressureHoldStage2Contract(
    _StaticOfficialSphereForceHoldStage2Contract
):
    """Shared Stage 2 that preserves acquisition while learning opposed pressure."""

    benchmark_protocol = "static_official_sphere_global_action_ab_v6_opposed_pressure_hold"
    curriculum_stage = "stage2_opposed_pressure_hold"
    global_action_selection_scope = "revo2_and_inspire"
    hand_action_semantics = "six_physical_motor_absolute_target"

    # Keep the Stage-1 coordinated enclosure landscape. The hard contact gate
    # remains at 2 mm, while reward shaping reaches farther so PPO does not trade
    # away thumb acquisition for a sparse one-frame force event.
    strict_reward_contact_score_scale = 0.012
    strict_touch_score_scale = 0.008
    strict_touch_reward_target_distance = 0.002
    contact_rew_scale = 30.0
    true_grasp_rew_scale = 60.0
    opposition_rew_scale = 30.0
    grasp_quality_rew_scale = 40.0
    catch_progress_rew_scale = 30.0
    strict_thumb_touch_rew_scale = 25.0
    strict_touch_rew_scale = 40.0
    strict_opposition_touch_rew_scale = 80.0

    # A single shared physical term rewards the weaker side of the opposed
    # contact pair, softly saturated at 5 N and discounted by relative motion.
    # Hard force-grasp metrics and success remain threshold based.
    tabletop_opposed_force_pressure_target = 5.0
    tabletop_opposed_force_pressure_rew_scale = 1000.0
    tabletop_force_thumb_contact_rew_scale = 20.0
    tabletop_force_grasp_rew_scale = 50.0
    tabletop_force_grasp_streak_rew_scale = 80.0
    tabletop_force_stable_grasp_rew_scale = 100.0
    tabletop_force_grasp_loss_penalty_scale = 0.0


@configclass
class Revo2StaticOfficialSphereJointTargetTeacherEnvCfg(
    _StaticOfficialSphereLiftControlABContract,
    Revo2StaticOfficialJointTargetTeacherEnvCfg,
):
    """Official Revo2 static sphere lift with bounded joint-absolute actions."""

    reference_name = "revo2_static_official_sphere_joint_target_teacher"
    observation_space = 80
    tabletop_privileged_phase_obs_enabled = True


@configclass
class Revo2StaticOfficialSphereCartesianImpedanceTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    Revo2StaticOfficialSphereJointTargetTeacherEnvCfg,
):
    """Matched Revo2 static sphere lift with measured EEF-delta actions."""

    reference_name = "revo2_static_official_sphere_cartesian_impedance_teacher"
    action_contract = "revo2_cartesian_impedance_12d"
    observation_space = 79
    cartesian_impedance_target_mode = "measured_delta"
    robot_cfg: ArticulationCfg = _v699_revo2_official_six_active_impedance_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


@configclass
class Revo2StaticOfficialSphereLiftStage2JointTargetTeacherEnvCfg(
    _StaticOfficialSphereForcePalmLiftStage2Contract,
    Revo2StaticOfficialSphereJointTargetTeacherEnvCfg,
):
    """Revo2 Stage 2 with the shared force-gated palm-lift curriculum."""

    reference_name = "revo2_static_official_sphere_lift_stage2_joint_target_teacher"


@configclass
class Revo2StaticOfficialSphereLiftStage2CartesianImpedanceTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    Revo2StaticOfficialSphereLiftStage2JointTargetTeacherEnvCfg,
):
    """Matched Revo2 Cartesian Stage 2 under the same lift curriculum."""

    reference_name = "revo2_static_official_sphere_lift_stage2_cartesian_impedance_teacher"
    action_contract = "revo2_cartesian_impedance_12d"
    observation_space = 79
    cartesian_impedance_target_mode = "measured_delta"
    robot_cfg: ArticulationCfg = _v699_revo2_official_six_active_impedance_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


@configclass
class Revo2StaticOfficialSphereForceHoldStage2JointTargetTeacherEnvCfg(
    _StaticOfficialSphereForceHoldStage2Contract,
    Revo2StaticOfficialSphereJointTargetTeacherEnvCfg,
):
    """Revo2 Stage 2 for sustained real contact with joint-absolute actions."""

    reference_name = "revo2_static_official_sphere_force_hold_stage2_joint_target_teacher"


@configclass
class Revo2StaticOfficialSphereForceHoldStage2CartesianImpedanceTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    Revo2StaticOfficialSphereForceHoldStage2JointTargetTeacherEnvCfg,
):
    """Matched Revo2 force-hold stage with measured EEF-delta actions."""

    reference_name = "revo2_static_official_sphere_force_hold_stage2_cartesian_impedance_teacher"
    action_contract = "revo2_cartesian_impedance_12d"
    observation_space = 79
    cartesian_impedance_target_mode = "measured_delta"
    robot_cfg: ArticulationCfg = _v699_revo2_official_six_active_impedance_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


@configclass
class Revo2StaticOfficialSphereOpposedPressureHoldStage2JointTargetTeacherEnvCfg(
    _StaticOfficialSphereOpposedPressureHoldStage2Contract,
    Revo2StaticOfficialSphereJointTargetTeacherEnvCfg,
):
    """Revo2 opposed-pressure hold stage with joint-absolute actions."""

    reference_name = "revo2_static_official_sphere_opposed_pressure_hold_stage2_joint_target_teacher"


@configclass
class Revo2StaticOfficialSphereOpposedPressureHoldStage2CartesianImpedanceTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    Revo2StaticOfficialSphereOpposedPressureHoldStage2JointTargetTeacherEnvCfg,
):
    """Matched Revo2 opposed-pressure hold stage with EEF-delta actions."""

    reference_name = (
        "revo2_static_official_sphere_opposed_pressure_hold_stage2_cartesian_impedance_teacher"
    )
    action_contract = "revo2_cartesian_impedance_12d"
    observation_space = 79
    cartesian_impedance_target_mode = "measured_delta"
    robot_cfg: ArticulationCfg = _v699_revo2_official_six_active_impedance_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


@configclass
class Revo2UnifiedRollingStage1CartesianPhaseObsTeacherEnvCfg(
    _CartesianWristDirectPolicyContract,
    Revo2UnifiedRollingStage1TeacherEnvCfg,
):
    """Revo2 Stage 1 with direct Cartesian wrist and six-motor hand actions."""

    reference_name = "revo2_unified_rolling_stage1_cartesian_phase_obs_teacher"
    action_contract = "revo2_cartesian_wrist_12d"
    observation_space = 89
    tabletop_privileged_phase_obs_enabled = True


@configclass
class Revo2UnifiedRollingStage1AcquisitionCartesianPhaseObsTeacherEnvCfg(
    _CartesianWristAcquisitionStage1Contract,
    Revo2UnifiedRollingStage1CartesianPhaseObsTeacherEnvCfg,
):
    """Revo2 Cartesian Stage 1 without object-lift reward shortcuts."""

    reference_name = "revo2_unified_rolling_stage1_acquisition_cartesian_phase_obs_teacher"


@configclass
class Revo2UnifiedRollingStage1AcquisitionCartesianPalmFramePhaseObsTeacherEnvCfg(
    Revo2UnifiedRollingStage1AcquisitionCartesianPhaseObsTeacherEnvCfg,
):
    """Revo2 Cartesian acquisition with an explicit palm-frame grasp envelope."""

    reference_name = "revo2_unified_rolling_stage1_acquisition_cartesian_palm_frame_phase_obs_teacher"
    # Median object position at the first strict-grasp frame of the matched
    # 13D Revo2 static baseline (seed 642, 20 trials). The broad tolerance
    # shapes wrist orientation without prescribing a joint-space pose.
    pregrasp_target_rel_palm = (0.030, 0.007, 0.075)
    pregrasp_target_scale = (0.030, 0.025, 0.030)
    dynamic_tabletop_palm_frame_pregrasp_rew_scale = 1800.0


@configclass
class Revo2UnifiedRollingStage2HoldCartesianPhaseObsTeacherEnvCfg(
    _CartesianWristDirectPolicyContract,
    Revo2UnifiedRollingStage2HoldTeacherEnvCfg,
):
    """Revo2 Stage 2 hold objective under the same 12D action contract."""

    reference_name = "revo2_unified_rolling_stage2_hold_cartesian_phase_obs_teacher"
    action_contract = "revo2_cartesian_wrist_12d"
    observation_space = 89
    tabletop_privileged_phase_obs_enabled = True


@configclass
class Revo2UnifiedRollingStage3CartesianPhaseObsTeacherEnvCfg(
    _CartesianWristDirectPolicyContract,
    Revo2UnifiedRollingStage3TeacherEnvCfg,
):
    """Revo2 Stage 3 lift objective under the same 12D action contract."""

    reference_name = "revo2_unified_rolling_stage3_cartesian_phase_obs_teacher"
    action_contract = "revo2_cartesian_wrist_12d"
    observation_space = 89
    tabletop_privileged_phase_obs_enabled = True


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
class InspireStaticOfficialJointTargetTeacherEnvCfg(
    _StaticOfficialSixActiveFromScratchContract,
    InspireDynamicTabletopTeacherEnvCfg,
):
    """Official RH56BFX mimic hand with the shared joint-absolute benchmark."""

    reference_name = "inspire_static_official_six_active_joint_target_teacher"
    action_contract = "inspire_semantic_13d"
    action_space = 13
    observation_space = 76
    policy_action_interface = "joint_target"
    default_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    sim_hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    fingertip_body_names = INSPIRE_FINGERTIP_BODY_NAMES
    touch_body_names = INSPIRE_MESH_CONTACT_BODY_NAMES
    palm_body_name = "hand_base_link"
    palm_offset = INSPIRE_PALM_OFFSET
    fingertip_body_offsets = INSPIRE_FINGERTIP_BODY_OFFSETS
    reference_hand_fractions = (1.0,) * 6
    inspire_semantic_close_targets = INSPIRE_RH56BFX_MIMIC_CLOSE_TARGETS
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_mimic_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )
    robot_collision_disabled_body_names = ()
    robot_mimic_natural_frequency = 0.0
    robot_mimic_damping_ratio = 0.0
    robot_mimic_offset_overrides_deg = (
        ("index_intermediate_joint", 2.604352333),
        ("middle_intermediate_joint", 2.604352333),
        ("ring_intermediate_joint", 2.604352333),
        ("pinky_intermediate_joint", 2.604352333),
    )


@configclass
class InspireStaticOfficialCartesianImpedanceTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    InspireStaticOfficialJointTargetTeacherEnvCfg,
):
    """Matched RH56BFX base-frame EEF-delta torque-impedance benchmark."""

    reference_name = "inspire_static_official_six_active_cartesian_impedance_teacher"
    action_contract = "inspire_cartesian_impedance_12d"
    observation_space = 75
    cartesian_impedance_target_mode = "measured_delta"
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_impedance_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


@configclass
class InspireStaticCanonicalJointDeltaTeacherEnvCfg(
    _StaticCanonicalFromScratchContract,
    InspireDynamicTabletopTeacherEnvCfg,
):
    """Official six-active RH56BFX canonical static teacher."""

    reference_name = "inspire_static_canonical_joint_delta_teacher"
    action_contract = "inspire_semantic_13d"
    action_space = 13
    observation_space = 76
    policy_action_interface = "joint_target"
    default_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    canonical_reset_pregrasp_arm_pos = INSPIRE_CANONICAL_STATIC_PREGRASP_ARM_POS
    hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    sim_hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    fingertip_body_names = INSPIRE_FINGERTIP_BODY_NAMES
    touch_body_names = INSPIRE_MESH_CONTACT_BODY_NAMES
    palm_body_name = "hand_base_link"
    palm_offset = INSPIRE_PALM_OFFSET
    fingertip_body_offsets = INSPIRE_FINGERTIP_BODY_OFFSETS
    reference_hand_fractions = (1.0,) * 6
    inspire_semantic_close_targets = INSPIRE_RH56BFX_MIMIC_CLOSE_TARGETS
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_mimic_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )
    robot_cfg.init_state.pos = CANONICAL_STATIC_ROBOT_BASE_POS
    robot_collision_disabled_body_names = ()
    robot_mimic_natural_frequency = 0.0
    robot_mimic_damping_ratio = 0.0
    robot_mimic_offset_overrides_deg = (
        ("index_intermediate_joint", 2.604352333),
        ("middle_intermediate_joint", 2.604352333),
        ("ring_intermediate_joint", 2.604352333),
        ("pinky_intermediate_joint", 2.604352333),
    )


@configclass
class InspireStaticCanonicalCartesianImpedanceTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    InspireStaticCanonicalJointDeltaTeacherEnvCfg,
):
    """Matched RH56BFX base-frame Cartesian impedance ablation."""

    reference_name = "inspire_static_canonical_cartesian_impedance_teacher"
    action_contract = "inspire_cartesian_impedance_12d"
    observation_space = 75
    cartesian_impedance_target_mode = "measured_delta"
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_impedance_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )
    robot_cfg.init_state.pos = CANONICAL_STATIC_ROBOT_BASE_POS


@configclass
class InspireStaticCanonicalRobustJointDeltaTeacherEnvCfg(
    _StaticCanonicalRobustHomeContract,
    InspireStaticCanonicalJointDeltaTeacherEnvCfg,
):
    """Inspire v2.6 robust-home JointDelta teacher."""

    reference_name = "inspire_static_canonical_robust_joint_delta_teacher"


@configclass
class InspireStaticCanonicalRobustCartesianImpedanceTeacherEnvCfg(
    _StaticCanonicalRobustHomeContract,
    InspireStaticCanonicalCartesianImpedanceTeacherEnvCfg,
):
    """Inspire v2.6 robust-home Cartesian impedance teacher."""

    reference_name = "inspire_static_canonical_robust_cartesian_impedance_teacher"


@configclass
class InspireStaticOfficialSphereJointTargetTeacherEnvCfg(
    _StaticOfficialSphereLiftControlABContract,
    InspireStaticOfficialJointTargetTeacherEnvCfg,
):
    """Official RH56BFX static sphere lift with bounded joint-absolute actions."""

    reference_name = "inspire_static_official_sphere_joint_target_teacher"
    observation_space = 80
    tabletop_privileged_phase_obs_enabled = True


@configclass
class InspireStaticOfficialSphereCartesianImpedanceTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    InspireStaticOfficialSphereJointTargetTeacherEnvCfg,
):
    """Matched RH56BFX static sphere lift with measured EEF-delta actions."""

    reference_name = "inspire_static_official_sphere_cartesian_impedance_teacher"
    action_contract = "inspire_cartesian_impedance_12d"
    observation_space = 79
    cartesian_impedance_target_mode = "measured_delta"
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_impedance_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


@configclass
class InspireStaticOfficialSphereLiftStage2JointTargetTeacherEnvCfg(
    _StaticOfficialSphereForcePalmLiftStage2Contract,
    InspireStaticOfficialSphereJointTargetTeacherEnvCfg,
):
    """RH56BFX Stage 2 with the shared force-gated palm-lift curriculum."""

    reference_name = "inspire_static_official_sphere_lift_stage2_joint_target_teacher"


@configclass
class InspireStaticOfficialSphereLiftStage2CartesianImpedanceTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    InspireStaticOfficialSphereLiftStage2JointTargetTeacherEnvCfg,
):
    """Matched RH56BFX Cartesian Stage 2 under the same lift curriculum."""

    reference_name = "inspire_static_official_sphere_lift_stage2_cartesian_impedance_teacher"
    action_contract = "inspire_cartesian_impedance_12d"
    observation_space = 79
    cartesian_impedance_target_mode = "measured_delta"
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_impedance_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


@configclass
class InspireStaticOfficialSphereForceHoldStage2JointTargetTeacherEnvCfg(
    _StaticOfficialSphereForceHoldStage2Contract,
    InspireStaticOfficialSphereJointTargetTeacherEnvCfg,
):
    """RH56BFX Stage 2 for sustained real contact with joint-absolute actions."""

    reference_name = "inspire_static_official_sphere_force_hold_stage2_joint_target_teacher"


@configclass
class InspireStaticOfficialSphereForceHoldStage2CartesianImpedanceTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    InspireStaticOfficialSphereForceHoldStage2JointTargetTeacherEnvCfg,
):
    """Matched RH56BFX force-hold stage with measured EEF-delta actions."""

    reference_name = "inspire_static_official_sphere_force_hold_stage2_cartesian_impedance_teacher"
    action_contract = "inspire_cartesian_impedance_12d"
    observation_space = 79
    cartesian_impedance_target_mode = "measured_delta"
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_impedance_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


@configclass
class InspireStaticOfficialSphereOpposedPressureHoldStage2JointTargetTeacherEnvCfg(
    _StaticOfficialSphereOpposedPressureHoldStage2Contract,
    InspireStaticOfficialSphereJointTargetTeacherEnvCfg,
):
    """RH56BFX opposed-pressure hold stage with joint-absolute actions."""

    reference_name = "inspire_static_official_sphere_opposed_pressure_hold_stage2_joint_target_teacher"


@configclass
class InspireStaticOfficialSphereOpposedPressureHoldStage2CartesianImpedanceTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    InspireStaticOfficialSphereOpposedPressureHoldStage2JointTargetTeacherEnvCfg,
):
    """Matched RH56BFX opposed-pressure hold stage with EEF-delta actions."""

    reference_name = (
        "inspire_static_official_sphere_opposed_pressure_hold_stage2_cartesian_impedance_teacher"
    )
    action_contract = "inspire_cartesian_impedance_12d"
    observation_space = 79
    cartesian_impedance_target_mode = "measured_delta"
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_impedance_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )


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
class InspirePhysicalAuditPhantomTipsTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereKnownGoodHomeStaticTeacherEnvCfg
):
    """Unmodified tip-sphere baseline with force diagnostics enabled."""

    reference_name = "inspire_rh56bfx_physical_audit_phantom_tips_teacher"
    object_contact_force_diagnostics_enabled = True


@configclass
class InspirePhysicalAuditMeshOnlyTeacherEnvCfg(
    InspireDynamicTabletopRollingSphereKnownGoodHomeStaticTeacherEnvCfg
):
    """RH56BFX positive control using only the visible finger collision meshes."""

    reference_name = "inspire_rh56bfx_physical_audit_mesh_only_teacher"
    robot_collision_disabled_body_names = INSPIRE_COLLISION_TIP_BODY_NAMES
    touch_body_names = INSPIRE_MESH_CONTACT_BODY_NAMES
    object_contact_force_diagnostics_enabled = True


@configclass
class InspirePhysicalAuditMeshOnlyLegacyActuationTeacherEnvCfg(
    InspirePhysicalAuditMeshOnlyTeacherEnvCfg
):
    """Mesh-only positive control with the URDF/old IsaacGym hand drive scale."""

    reference_name = "inspire_rh56bfx_physical_audit_mesh_only_legacy_actuation_teacher"
    robot_cfg: ArticulationCfg = _inspire_z180_legacy_robot_cfg(INSPIRE_V340_KNOWN_GOOD_ARM_POS)


@configclass
class InspirePhysicalAuditMimicTeacherEnvCfg(InspirePhysicalAuditMeshOnlyTeacherEnvCfg):
    """Faithful RH56BFX six-actuator model using the source URDF mimic constraints."""

    reference_name = "inspire_rh56bfx_physical_audit_mimic_teacher"
    robot_cfg: ArticulationCfg = _inspire_z180_robot_cfg(
        INSPIRE_V340_KNOWN_GOOD_ARM_POS,
        default_hand_pos=INSPIRE_RH56BFX_MIMIC_OPEN_POS,
        hand_effort=1.0,
        hand_stiffness=60.0,
        hand_damping=6.0,
        hand_armature=0.01,
        thumb_yaw_velocity=1.0,
        thumb_flex_velocity=1.0,
        finger_flex_velocity=1.0,
        asset_path=INSPIRE_RH56BFX_MIMIC_URDF,
        actuate_mimic_followers=False,
        preserve_mimic_constraints=True,
    )
    sim_hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    inspire_semantic_close_targets = INSPIRE_RH56BFX_MIMIC_CLOSE_TARGETS
    robot_collision_disabled_body_names = ()
    robot_mimic_natural_frequency = 0.0
    robot_mimic_damping_ratio = 0.0
    # PhysX angular mimic offsets are authored in degrees and use the opposite
    # sign in q_follower + gearing*q_reference + offset = 0.
    robot_mimic_offset_overrides_deg = (
        ("index_intermediate_joint", 2.604352333),
        ("middle_intermediate_joint", 2.604352333),
        ("ring_intermediate_joint", 2.604352333),
        ("pinky_intermediate_joint", 2.604352333),
    )


@configclass
class InspireRH56BFXStaticJointTargetABTeacherEnvCfg(
    InspirePhysicalAuditMimicTeacherEnvCfg
):
    """Static physical baseline paired with the torque-level OSC task."""

    reference_name = "inspire_rh56bfx_static_joint_target_ab_teacher"
    benchmark_protocol = "inspire_static_joint_vs_torque_cartesian_impedance_v1"
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_mimic_robot_cfg(
        INSPIRE_V340_KNOWN_GOOD_ARM_POS
    )
    tabletop_arm_lift_progress_mode = "palm_z"
    tabletop_post_success_stability_latch_enabled = True
    tabletop_post_success_arm_target_lock_enabled = True
    tabletop_post_success_hand_target_lock_enabled = True
    tabletop_post_success_hand_lock_uses_actual_joint_pos = True


@configclass
class InspireRH56BFXStaticCartesianImpedanceABTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    InspireRH56BFXStaticJointTargetABTeacherEnvCfg,
):
    """Static RH56BFX task with base-frame delta pose and torque-level OSC."""

    reference_name = "inspire_rh56bfx_static_cartesian_impedance_ab_teacher"
    action_contract = "inspire_cartesian_impedance_12d"
    observation_space = 75
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_impedance_robot_cfg(
        INSPIRE_V340_KNOWN_GOOD_ARM_POS
    )


@configclass
class InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg(
    InspireUnifiedRollingBenchmarkTeacherEnvCfg
):
    """Unified rolling teacher using the deployable RH56BFX physical contract."""

    reference_name = "inspire_rh56bfx_faithful_unified_rolling_teacher"
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_mimic_robot_cfg(INSPIRE_V341_CLEAR_ARM_POS)
    default_arm_pos = INSPIRE_V341_CLEAR_ARM_POS
    scripted_tabletop_pregrasp_arm_pos = INSPIRE_V341_CLEAR_ARM_POS
    tabletop_arm_lift_progress_baseline_pos = INSPIRE_V341_CLEAR_ARM_POS

    hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    sim_hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    reference_hand_fractions = (1.0,) * 6
    inspire_semantic_close_targets = INSPIRE_ANYDEX_SPHERE_ACTIVE_CLOSE_TARGETS
    touch_body_names = INSPIRE_MESH_CONTACT_BODY_NAMES
    robot_collision_disabled_body_names = ()
    robot_mimic_natural_frequency = 0.0
    robot_mimic_damping_ratio = 0.0
    robot_mimic_offset_overrides_deg = (
        ("index_intermediate_joint", 2.604352333),
        ("middle_intermediate_joint", 2.604352333),
        ("ring_intermediate_joint", 2.604352333),
        ("pinky_intermediate_joint", 2.604352333),
    )

    object_contact_force_diagnostics_enabled = True
    object_contact_force_threshold = 0.05
    tabletop_success_requires_force_grasp = True
    tabletop_force_grasp_rew_scale = 1200.0
    tabletop_force_grasp_streak_rew_scale = 800.0
    tabletop_force_stable_grasp_rew_scale = 1800.0
    tabletop_force_grasp_loss_penalty_scale = 1200.0
    tabletop_force_grasp_streak_target = 8
    tabletop_post_success_hand_close_fraction = 0.0
    tabletop_arm_lift_baseline_sync_target_to_measured = True
    tabletop_lift_hand_target_latch_uses_measured_pos = True
    # The shared benchmark defaults to the Revo2 V325 lift direction.  At the
    # RH56BFX grasp pose that direction sweeps the wrist sideways.  Use the
    # vertical carry delta measured from successful RH56BFX motion-planning
    # trajectories for reward projection and any optional target diagnostics.
    lift_arm_delta = INSPIRE_RH56BFX_CARTESIAN_CARRY_ARM_DELTA
    scripted_tabletop_lift_target_arm_delta = INSPIRE_RH56BFX_CARTESIAN_CARRY_ARM_DELTA
    scripted_tabletop_relative_lift_target_arm_delta = INSPIRE_RH56BFX_CARTESIAN_CARRY_ARM_DELTA


@configclass
class InspireRH56BFXFaithfulUnifiedRollingStage1TeacherEnvCfg(
    InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg
):
    """Faithful RH56BFX home-to-pregrasp bootstrap for the unified benchmark."""

    reference_name = "inspire_rh56bfx_faithful_unified_rolling_stage1_teacher"
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
class InspireRH56BFXFaithfulUnifiedRollingStage2HoldTeacherEnvCfg(
    _UnifiedRollingGraspHoldStage2Contract,
    InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg,
):
    """Faithful RH56BFX continuation that learns a sustained opposed grasp."""

    reference_name = "inspire_rh56bfx_faithful_unified_rolling_stage2_hold_teacher"


@configclass
class InspireRH56BFXFaithfulUnifiedRollingStage3TeacherEnvCfg(
    _UnifiedRollingLiftHoldStage3Contract,
    InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg,
):
    """Faithful RH56BFX continuation that learns strict lift and stable hold."""

    reference_name = "inspire_rh56bfx_faithful_unified_rolling_stage3_teacher"
    # Preserve the load-bearing enclosure learned in Stage 2 while the policy
    # discovers a coordinated lift.  A lifted strict grasp still earns much
    # more through the Stage 3 carry, lift, hold, and success terms.
    tabletop_strict_hold_rew_scale = 12000.0
    tabletop_underwrap_rew_scale = 4000.0
    tabletop_lift_hand_target_close_fraction = 0.15


@configclass
class InspireUnifiedRollingJointDeltaV26TeacherEnvCfg(
    _UnifiedRollingJointDeltaV26Contract,
    InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg,
):
    """Official six-active RH56BFX rolling teacher under the selected action."""

    reference_name = "inspire_unified_rolling_jointdelta_v26_teacher"
    action_contract = "inspire_semantic_13d"
    canonical_reset_pregrasp_arm_pos = INSPIRE_CANONICAL_STATIC_PREGRASP_ARM_POS
    default_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    scripted_tabletop_pregrasp_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    tabletop_arm_lift_progress_baseline_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    sim_hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    reference_hand_fractions = (1.0,) * 6
    inspire_semantic_close_targets = INSPIRE_RH56BFX_MIMIC_CLOSE_TARGETS
    touch_body_names = INSPIRE_MESH_CONTACT_BODY_NAMES
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_mimic_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )
    robot_cfg.init_state.pos = CANONICAL_STATIC_ROBOT_BASE_POS
    robot_collision_disabled_body_names = ()
    robot_mimic_natural_frequency = 0.0
    robot_mimic_damping_ratio = 0.0
    robot_mimic_offset_overrides_deg = (
        ("index_intermediate_joint", 2.604352333),
        ("middle_intermediate_joint", 2.604352333),
        ("ring_intermediate_joint", 2.604352333),
        ("pinky_intermediate_joint", 2.604352333),
    )
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=UNIFIED_ROLLING_V26_START_POS,
    )


@configclass
class InspireUnifiedRollingJointDeltaV27TeacherEnvCfg(
    _UnifiedRollingJointDeltaV27Contract,
    InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg,
):
    """RH56BFX rolling teacher with performance-gated lift-first curriculum."""

    reference_name = "inspire_unified_rolling_jointdelta_v27_teacher"
    action_contract = "inspire_semantic_13d"
    canonical_reset_pregrasp_arm_pos = INSPIRE_CANONICAL_STATIC_PREGRASP_ARM_POS
    default_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    scripted_tabletop_pregrasp_arm_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    tabletop_arm_lift_progress_baseline_pos = FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    sim_hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    reference_hand_fractions = (1.0,) * 6
    inspire_semantic_close_targets = INSPIRE_RH56BFX_MIMIC_CLOSE_TARGETS
    touch_body_names = INSPIRE_MESH_CONTACT_BODY_NAMES
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_mimic_robot_cfg(
        FRANKA_ISAACLAB_DEFAULT_HOME_ARM_POS
    )
    robot_cfg.init_state.pos = CANONICAL_STATIC_ROBOT_BASE_POS
    robot_collision_disabled_body_names = ()
    robot_mimic_natural_frequency = 0.0
    robot_mimic_damping_ratio = 0.0
    robot_mimic_offset_overrides_deg = (
        ("index_intermediate_joint", 2.604352333),
        ("middle_intermediate_joint", 2.604352333),
        ("ring_intermediate_joint", 2.604352333),
        ("pinky_intermediate_joint", 2.604352333),
    )
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_ROLLING_START_SPEC,
        pos=UNIFIED_ROLLING_V26_START_POS,
    )


@configclass
class InspireRH56BFXStaticStrictJointTargetABTeacherEnvCfg(
    _StaticStrictFromScratchControlABContract,
    InspireRH56BFXFaithfulUnifiedRollingStage3TeacherEnvCfg,
):
    """Full from-scratch RH56BFX benchmark with Franka joint targets."""

    reference_name = "inspire_rh56bfx_static_strict_joint_target_ab_teacher"
    action_contract = "inspire_semantic_13d"
    action_space = 13
    policy_action_interface = "joint_target"
    observation_space = 80
    tabletop_privileged_phase_obs_enabled = True
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_mimic_robot_cfg(
        INSPIRE_V341_CLEAR_ARM_POS
    )
    default_arm_pos = INSPIRE_V341_CLEAR_ARM_POS
    scripted_tabletop_pregrasp_arm_pos = INSPIRE_V341_CLEAR_ARM_POS
    tabletop_arm_lift_progress_baseline_pos = INSPIRE_V341_CLEAR_ARM_POS


@configclass
class InspireRH56BFXStaticStrictCartesianImpedanceABTeacherEnvCfg(
    _CartesianImpedanceDirectPolicyContract,
    InspireRH56BFXStaticStrictJointTargetABTeacherEnvCfg,
):
    """Matched RH56BFX benchmark with measured base-frame EEF delta and torque OSC."""

    reference_name = "inspire_rh56bfx_static_strict_cartesian_impedance_ab_teacher"
    action_contract = "inspire_cartesian_impedance_12d"
    cartesian_impedance_target_mode = "measured_delta"
    observation_space = 79
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_impedance_robot_cfg(
        INSPIRE_V341_CLEAR_ARM_POS
    )


@configclass
class InspireRH56BFXStaticStableHoverJointTargetABTeacherEnvCfg(
    _StaticStableHoverControlABContract,
    InspireRH56BFXStaticStrictJointTargetABTeacherEnvCfg,
):
    """RH56BFX joint-target A/B with a low-speed stable-hover success contract."""

    reference_name = "inspire_rh56bfx_static_stable_hover_joint_target_ab_teacher"
    observation_space = 95


@configclass
class InspireRH56BFXStaticStableHoverCartesianImpedanceABTeacherEnvCfg(
    _StaticStableHoverControlABContract,
    InspireRH56BFXStaticStrictCartesianImpedanceABTeacherEnvCfg,
):
    """RH56BFX measured-delta OSC A/B with the same stable-hover contract."""

    reference_name = "inspire_rh56bfx_static_stable_hover_cartesian_impedance_ab_teacher"
    observation_space = 94


@configclass
class InspireRH56BFXStaticStableHoverJointTargetCleanCurriculumTeacherEnvCfg(
    _StaticCleanGraspCurriculumContract,
    InspireRH56BFXStaticStableHoverJointTargetABTeacherEnvCfg,
):
    """RH56BFX v13 teacher with the same training-only contact curriculum."""

    reference_name = (
        "inspire_rh56bfx_static_stable_hover_joint_target_clean_curriculum_teacher"
    )


@configclass
class InspireRH56BFXStaticStableHoverJointTargetSynchronousContactTeacherEnvCfg(
    _StaticSynchronousContactRewardContract,
    InspireRH56BFXStaticStableHoverJointTargetCleanCurriculumTeacherEnvCfg,
):
    """RH56BFX v14 teacher with the same bottleneck contact shaping."""

    reference_name = (
        "inspire_rh56bfx_static_stable_hover_joint_target_synchronous_contact_teacher"
    )


@configclass
class InspireRH56BFXStaticStableHoverJointTargetSynchronousCurriculumTeacherEnvCfg(
    _StaticSynchronousContactCurriculumRewardContract,
    InspireRH56BFXStaticStableHoverJointTargetCleanCurriculumTeacherEnvCfg,
):
    """RH56BFX v15 teacher with the same smooth reward transition."""

    reference_name = (
        "inspire_rh56bfx_static_stable_hover_joint_target_synchronous_curriculum_teacher"
    )


@configclass
class InspireRH56BFXStaticStableHoverJointTargetPostCleanLiftTeacherEnvCfg(
    _StaticPostCleanLiftExplorationContract,
    InspireRH56BFXStaticStableHoverJointTargetSynchronousCurriculumTeacherEnvCfg,
):
    """RH56BFX v16 teacher with the same clean-grasp-to-lift bridge."""

    reference_name = (
        "inspire_rh56bfx_static_stable_hover_joint_target_post_clean_lift_teacher"
    )


@configclass
class InspireRH56BFXStaticStableHoverJointTargetPostCleanLiftVelocityTeacherEnvCfg(
    _StaticPostCleanLiftVelocityContract,
    InspireRH56BFXStaticStableHoverJointTargetPostCleanLiftTeacherEnvCfg,
):
    """RH56BFX v17 teacher with the same upward palm velocity bridge."""

    reference_name = (
        "inspire_rh56bfx_static_stable_hover_joint_target_post_clean_lift_velocity_teacher"
    )


@configclass
class InspireRH56BFXStaticStableHoverJointTargetPostCleanGripRetentionTeacherEnvCfg(
    _StaticPostCleanGripRetentionContract,
    InspireRH56BFXStaticStableHoverJointTargetPostCleanLiftVelocityTeacherEnvCfg,
):
    """RH56BFX v18 teacher retaining real opposed contact during lift exploration."""

    reference_name = (
        "inspire_rh56bfx_static_stable_hover_joint_target_post_clean_grip_retention_teacher"
    )


@configclass
class InspireRH56BFXStaticStableHoverJointTargetPostCleanActionContinuityTeacherEnvCfg(
    _StaticPostCleanActionContinuityContract,
    InspireRH56BFXStaticStableHoverJointTargetPostCleanGripRetentionTeacherEnvCfg,
):
    """RH56BFX v19 teacher with the same reward-only hand continuity."""

    reference_name = (
        "inspire_rh56bfx_static_stable_hover_joint_target_post_clean_action_continuity_teacher"
    )


@configclass
class InspireRH56BFXStaticStableHoverJointTargetSmoothLiftPhaseTeacherEnvCfg(
    _StaticSmoothLiftPhaseObservationContract,
    InspireRH56BFXStaticStableHoverJointTargetPostCleanActionContinuityTeacherEnvCfg,
):
    """RH56BFX v20 teacher with the same smooth phase observation."""

    reference_name = (
        "inspire_rh56bfx_static_stable_hover_joint_target_smooth_lift_phase_teacher"
    )


@configclass
class InspireRH56BFXStaticStableHoverJointTargetForceBackedCleanLiftTeacherEnvCfg(
    _StaticForceBackedCleanLiftContract,
    InspireRH56BFXStaticStableHoverJointTargetSmoothLiftPhaseTeacherEnvCfg,
):
    """RH56BFX v21 teacher using the same physical phase boundary as Revo2."""

    reference_name = (
        "inspire_rh56bfx_static_stable_hover_joint_target_force_backed_clean_lift_teacher"
    )


@configclass
class InspireRH56BFXStaticStableHoverJointTargetForceBackedPinchLiftTeacherEnvCfg(
    _StaticForceBackedPinchLiftContract,
    InspireRH56BFXStaticStableHoverJointTargetForceBackedCleanLiftTeacherEnvCfg,
):
    """RH56BFX v22 teacher using the same physical sphere pinch contract."""

    reference_name = (
        "inspire_rh56bfx_static_stable_hover_joint_target_force_backed_pinch_lift_teacher"
    )


@configclass
class InspireRH56BFXStaticStableHoverJointTargetForceRetainedObjectLiftTeacherEnvCfg(
    _StaticForceRetainedObjectLiftContract,
    InspireRH56BFXStaticStableHoverJointTargetForceBackedPinchLiftTeacherEnvCfg,
):
    """RH56BFX v23 teacher with the same force-retained object lift contract."""

    reference_name = (
        "inspire_rh56bfx_static_stable_hover_joint_target_force_retained_object_lift_teacher"
    )


@configclass
class InspireRH56BFXStaticStableHoverJointTargetSustainedForceLiftTeacherEnvCfg(
    _StaticSustainedForceHoldBeforeLiftContract,
    InspireRH56BFXStaticStableHoverJointTargetForceRetainedObjectLiftTeacherEnvCfg,
):
    """RH56BFX v24 teacher with the same sustained-force lift boundary."""

    reference_name = (
        "inspire_rh56bfx_static_stable_hover_joint_target_sustained_force_lift_teacher"
    )


@configclass
class InspireRH56BFXStaticStableHoverJointTargetForceCoupledMicroLiftTeacherEnvCfg(
    _StaticForceCoupledMicroLiftContract,
    InspireRH56BFXStaticStableHoverJointTargetSustainedForceLiftTeacherEnvCfg,
):
    """RH56BFX v25 teacher using the same force-coupled lift bridge."""

    reference_name = (
        "inspire_rh56bfx_static_stable_hover_joint_target_force_coupled_micro_lift_teacher"
    )


@configclass
class InspireRH56BFXFaithfulUnifiedRollingStage1PhaseObsTeacherEnvCfg(
    InspireRH56BFXFaithfulUnifiedRollingStage1TeacherEnvCfg
):
    """Stage 1 with the shared privileged grasp-to-lift phase observation."""

    reference_name = "inspire_rh56bfx_faithful_unified_rolling_stage1_phase_obs_teacher"
    observation_space = 90
    tabletop_privileged_phase_obs_enabled = True


@configclass
class InspireRH56BFXFaithfulUnifiedRollingStage2HoldPhaseObsTeacherEnvCfg(
    InspireRH56BFXFaithfulUnifiedRollingStage2HoldTeacherEnvCfg
):
    """Stage 2 with the same observation contract used by the lift policy."""

    reference_name = "inspire_rh56bfx_faithful_unified_rolling_stage2_hold_phase_obs_teacher"
    observation_space = 90
    tabletop_privileged_phase_obs_enabled = True


@configclass
class InspireRH56BFXFaithfulUnifiedRollingStage3PhaseObsTeacherEnvCfg(
    InspireRH56BFXFaithfulUnifiedRollingStage3TeacherEnvCfg
):
    """Stage 3 with explicit privileged grasp and lift phase state."""

    reference_name = "inspire_rh56bfx_faithful_unified_rolling_stage3_phase_obs_teacher"
    observation_space = 90
    tabletop_privileged_phase_obs_enabled = True
    lift_arm_delta = INSPIRE_RH56BFX_SHORT_CARRY_ARM_DELTA
    scripted_tabletop_lift_target_arm_delta = INSPIRE_RH56BFX_SHORT_CARRY_ARM_DELTA
    scripted_tabletop_relative_lift_target_arm_delta = INSPIRE_RH56BFX_SHORT_CARRY_ARM_DELTA


@configclass
class InspireRH56BFXFaithfulUnifiedRollingStage1CartesianPhaseObsTeacherEnvCfg(
    _CartesianWristDirectPolicyContract,
    InspireRH56BFXFaithfulUnifiedRollingStage1TeacherEnvCfg,
):
    """RH56BFX Stage 1 with direct Cartesian wrist and six real motors."""

    reference_name = "inspire_rh56bfx_unified_rolling_stage1_cartesian_phase_obs_teacher"
    action_contract = "inspire_cartesian_wrist_12d"
    observation_space = 89
    tabletop_privileged_phase_obs_enabled = True


@configclass
class InspireRH56BFXFaithfulUnifiedRollingStage1AcquisitionCartesianPhaseObsTeacherEnvCfg(
    _CartesianWristAcquisitionStage1Contract,
    InspireRH56BFXFaithfulUnifiedRollingStage1CartesianPhaseObsTeacherEnvCfg,
):
    """RH56BFX Cartesian Stage 1 without object-lift reward shortcuts."""

    reference_name = "inspire_rh56bfx_stage1_acquisition_cartesian_phase_obs_teacher"


@configclass
class InspireRH56BFXFaithfulUnifiedRollingStage2HoldCartesianPhaseObsTeacherEnvCfg(
    _CartesianWristDirectPolicyContract,
    InspireRH56BFXFaithfulUnifiedRollingStage2HoldTeacherEnvCfg,
):
    """RH56BFX Stage 2 hold objective under the same 12D action contract."""

    reference_name = "inspire_rh56bfx_unified_rolling_stage2_hold_cartesian_phase_obs_teacher"
    action_contract = "inspire_cartesian_wrist_12d"
    observation_space = 89
    tabletop_privileged_phase_obs_enabled = True


@configclass
class InspireRH56BFXFaithfulUnifiedRollingStage3CartesianPhaseObsTeacherEnvCfg(
    _CartesianWristDirectPolicyContract,
    InspireRH56BFXFaithfulUnifiedRollingStage3TeacherEnvCfg,
):
    """RH56BFX Stage 3 lift objective under the same 12D action contract."""

    reference_name = "inspire_rh56bfx_unified_rolling_stage3_cartesian_phase_obs_teacher"
    action_contract = "inspire_cartesian_wrist_12d"
    observation_space = 89
    tabletop_privileged_phase_obs_enabled = True


@configclass
class InspireRH56BFXFaithfulUnifiedRollingStage3TargetPhaseObsTeacherEnvCfg(
    InspireRH56BFXFaithfulUnifiedRollingStage3PhaseObsTeacherEnvCfg
):
    """Direct Stage 3 with dense supervision of the complete 7-DoF lift target."""

    reference_name = "inspire_rh56bfx_faithful_unified_rolling_stage3_target_phase_obs_teacher"
    # A scalar projection leaves six unconstrained arm directions. Reward the
    # complete latched-pose + calibrated-delta target while retaining direct
    # policy actions and the strict physical lift/success criteria.
    tabletop_lift_action_prior_rew_scale = 0.0
    tabletop_lift_target_rew_scale = 60000.0
    tabletop_lift_target_error_scale = 0.25
    tabletop_lift_target_requires_current_grasp = False
    tabletop_lift_target_mode = "cartesian_ik"
    tabletop_lift_cartesian_target_height = 0.05
    tabletop_lift_cartesian_target_damping = 0.08
    tabletop_lift_cartesian_target_max_joint_delta = 0.01
    observation_space = 97
    tabletop_privileged_lift_target_obs_enabled = True
    # Keep the proven Stage 2 enclosure valuable while the arm learns the
    # phase switch. Delayed no-lift pressure avoids destroying it immediately.
    tabletop_strict_hold_rew_scale = 28000.0
    tabletop_underwrap_rew_scale = 8500.0
    tabletop_strict_grasp_loss_penalty_scale = 24000.0
    tabletop_no_lift_after_grasp_penalty_scale = 2500.0
    tabletop_no_lift_after_grasp_grace_steps = 24
    tabletop_no_lift_after_grasp_ramp_steps = 60
    tabletop_lift_without_object_penalty_scale = 4000.0
    tabletop_arm_object_lift_gap_penalty_scale = 4000.0
    tabletop_object_carry_stall_penalty_scale = 3000.0


@configclass
class InspireRH56BFXFaithfulUnifiedRollingStage3CartesianPriorTargetPhaseObsTeacherEnvCfg(
    InspireRH56BFXFaithfulUnifiedRollingStage3TargetPhaseObsTeacherEnvCfg
):
    """Stage 3 with a strict-grasp-gated Cartesian lift prior and PPO residual."""

    reference_name = (
        "inspire_rh56bfx_faithful_unified_rolling_stage3_cartesian_prior_"
        "target_phase_obs_teacher"
    )
    scripted_action_prior_enabled = True
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_residual_scale = 0.0
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_active_residual_scale = 0.0
    scripted_action_prior_active_arm_residual_scale = None
    scripted_action_prior_active_hand_residual_scale = 1.0
    scripted_action_prior_post_grasp_arm_residual_scale = None

    # Preserve the direct Stage-2 approach and hand policy. Only after two
    # consecutive strict-grasp steps latch the measured palm pose does the arm
    # receive a 5 cm vertical, orientation-preserving DLS IK target.
    scripted_tabletop_cartesian_lift_target_prior_enabled = True
    scripted_action_prior_hand_start_step = 9999
    scripted_action_prior_lift_start_step = 9999


@configclass
class InspireRH56BFXFaithfulSphere60RollingCurriculumTeacherEnvCfg(
    InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg
):
    """From-scratch RH56BFX bootstrap from a static 60 mm sphere to 0.4 m/s."""

    reference_name = "inspire_rh56bfx_faithful_sphere60_rolling_curriculum_teacher"
    object_start_pos = (
        0.535,
        0.075,
        _tabletop_start_z_from_spec(TABLETOP_INSPIRE_SPHERE_60MM_SPEC),
    )
    object_cfg: RigidObjectCfg = _object_cfg_from_tabletop_spec(
        TABLETOP_INSPIRE_SPHERE_60MM_SPEC,
        pos=object_start_pos,
    )
    object_shape = "sphere"
    object_radius = float(TABLETOP_INSPIRE_SPHERE_60MM_SPEC["radius"])
    object_size = tuple(TABLETOP_INSPIRE_SPHERE_60MM_SPEC["size"])
    tabletop_object_asset_specs = (TABLETOP_INSPIRE_SPHERE_60MM_SPEC,)
    tabletop_asset_sampling_weights = (1.0,)
    tabletop_asset_curriculum = False
    tabletop_asset_curriculum_start_count = 1
    reset_object_pos_noise = (0.0, 0.0, 0.0)

    scripted_tabletop_pregrasp_arm_pos = INSPIRE_RH56BFX_SPHERE_60MM_PREGRASP_ARM_POS
    scripted_action_prior_hand_action_vector = INSPIRE_RH56BFX_SPHERE_60MM_CALIBRATED_ACTION
    scripted_tabletop_hand_grasp_memory_action_vector = (
        INSPIRE_RH56BFX_SPHERE_60MM_CALIBRATED_ACTION
    )

    dynamic_grasp_speed_curriculum = True
    dynamic_grasp_speed_curriculum_mode = "success_gate"
    dynamic_grasp_speed_curriculum_override_alpha = None
    dynamic_tabletop_speed_alpha_sample_enabled = False
    dynamic_tabletop_start_speed_range = (0.0, 0.0)
    dynamic_tabletop_initial_speed_range = UNIFIED_ROLLING_TARGET_SPEED_RANGE
    dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    dynamic_tabletop_initial_yaw_rate_range = UNIFIED_ROLLING_TARGET_YAW_RATE_RANGE


@configclass
class InspireRH56BFXFaithfulSphere60ForceHoldTeacherEnvCfg(
    _UnifiedRollingGraspHoldStage2Contract,
    InspireRH56BFXFaithfulSphere60RollingCurriculumTeacherEnvCfg,
):
    """Direct-RL curriculum stage for a sustained physical sphere pinch."""

    reference_name = "inspire_rh56bfx_faithful_sphere60_force_hold_teacher"

    # A thumb plus one opposed digit is a physically valid pinch, as verified
    # independently by AnyDex+CuRobo. AnyDex is not used by this policy or
    # environment; this only makes the success contract match physical grasping.
    min_finger_contacts = 2
    min_non_thumb_contacts = 1
    strict_success_min_finger_contacts = 2
    strict_success_min_non_thumb_contacts = 1
    object_force_grasp_min_non_thumb_contacts = 1

    # The previous from-scratch stage reaches geometric enclosure but almost
    # never carries load. Make simultaneous thumb/opposed force and quiet
    # retention dominate this curriculum stage.
    tabletop_force_grasp_rew_scale = 24000.0
    tabletop_force_grasp_streak_rew_scale = 32000.0
    tabletop_force_stable_grasp_rew_scale = 42000.0
    tabletop_force_grasp_loss_penalty_scale = 30000.0
    tabletop_force_grasp_streak_target = 12
    tabletop_strict_hold_rew_scale = 4000.0
    tabletop_strict_grasp_loss_penalty_scale = 6000.0
    tabletop_arm_lift_progress_baseline_mode = "first_force_grasp"
    tabletop_arm_lift_progress_baseline_grasp_streak = 4
    tabletop_success_requires_force_grasp = True

    # This stage learns contact from the policy's own actions. No scripted
    # reach, close, lift, AnyDex pose, or residual controller is active.
    scripted_action_prior_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_relative_lift_target_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False


@configclass
class InspireRH56BFXFaithfulSphere60ForceLiftTeacherEnvCfg(
    _UnifiedRollingLiftHoldStage3Contract,
    InspireRH56BFXFaithfulSphere60RollingCurriculumTeacherEnvCfg,
):
    """Direct-RL curriculum stage that converts a physical pinch into a lift."""

    reference_name = "inspire_rh56bfx_faithful_sphere60_force_lift_teacher"

    min_finger_contacts = 2
    min_non_thumb_contacts = 1
    strict_success_min_finger_contacts = 2
    strict_success_min_non_thumb_contacts = 1
    object_force_grasp_min_non_thumb_contacts = 1
    tabletop_success_requires_force_grasp = True

    tabletop_arm_lift_progress_baseline_mode = "first_force_grasp"
    tabletop_arm_lift_progress_baseline_grasp_streak = 4
    tabletop_lift_gate_requires_force_grasp = True
    tabletop_lift_rewards_require_force_grasp = True
    tabletop_lift_rewards_require_current_strict_grasp = True
    tabletop_no_lift_uses_force_grasp_gate = True
    tabletop_lift_hand_target_close_fraction = 0.03

    tabletop_force_grasp_rew_scale = 8000.0
    tabletop_force_grasp_streak_rew_scale = 12000.0
    tabletop_force_stable_grasp_rew_scale = 16000.0
    tabletop_force_grasp_loss_penalty_scale = 24000.0
    tabletop_force_grasp_streak_target = 12
    tabletop_strict_hold_rew_scale = 3000.0

    scripted_action_prior_enabled = False
    scripted_tabletop_pregrasp_prior_enabled = False
    scripted_tabletop_relative_lift_target_prior_enabled = False
    scripted_tabletop_lift_target_prior_enabled = False
    scripted_tabletop_hand_grasp_memory_prior_enabled = False


@configclass
class InspireRH56BFXFaithfulSphere60LiftCommitTeacherEnvCfg(
    InspireRH56BFXFaithfulSphere60ForceLiftTeacherEnvCfg
):
    """Direct-RL continuation that favors carrying the sphere over table contact."""

    reference_name = "inspire_rh56bfx_faithful_sphere60_lift_commit_teacher"

    # The force-lift policy already reaches a load-bearing pinch in 63/64
    # static episodes, but 40/64 episodes stop before a full lift. Preserve the
    # force gate and grasp-loss cost while making stationary table contact much
    # less valuable than object-coupled upward motion.
    tabletop_force_grasp_rew_scale = 2500.0
    tabletop_force_grasp_streak_rew_scale = 3500.0
    tabletop_force_stable_grasp_rew_scale = 5000.0
    tabletop_force_grasp_loss_penalty_scale = 24000.0
    tabletop_strict_hold_rew_scale = 1500.0

    lift_progress_rew_scale = 10000.0
    quality_lift_progress_rew_scale = 14000.0
    lifted_true_grasp_rew_scale = 28000.0
    tabletop_stable_catch_rew_scale = 10000.0
    tabletop_grasped_arm_lift_rew_scale = 18000.0
    tabletop_lift_action_prior_rew_scale = 18000.0
    tabletop_object_up_vel_rew_scale = 8000.0
    tabletop_object_carry_lift_rew_scale = 32000.0
    stable_hold_rew_scale = 32000.0
    hold_progress_rew_scale = 44000.0
    success_bonus = 96000.0

    tabletop_no_lift_after_grasp_penalty_scale = 24000.0
    tabletop_no_lift_after_grasp_grace_steps = 6
    tabletop_no_lift_after_grasp_ramp_steps = 18
    tabletop_object_carry_stall_penalty_scale = 10000.0


@configclass
class InspireRH56BFXFaithfulPrimitive3LiftCommitTeacherEnvCfg(
    InspireRH56BFXFaithfulSphere60LiftCommitTeacherEnvCfg
):
    """Shape-only bridge from the sphere policy to box and cylinder grasping."""

    reference_name = "inspire_rh56bfx_faithful_primitive3_lift_commit_teacher"
    tabletop_object_asset_specs = TABLETOP_INSPIRE_PRIMITIVE3_SPECS
    tabletop_asset_set_enabled = True
    tabletop_asset_obs_enabled = True
    tabletop_asset_sampling_weights = None
    tabletop_asset_curriculum = True
    tabletop_asset_curriculum_mode = "dynamic_speed"
    tabletop_asset_curriculum_start_count = 1
    tabletop_asset_curriculum_override_alpha = None

    # Keep the sphere curriculum's validated interception point and remove
    # position randomization here so this bridge changes object geometry only.
    reset_object_pos_noise = (0.0, 0.0, 0.0)


@configclass
class InspireRH56BFXFaithfulPrimitive3LoadBearingTeacherEnvCfg(
    InspireRH56BFXFaithfulPrimitive3LiftCommitTeacherEnvCfg
):
    """Static curriculum that teaches shape-normalized support before lift."""

    reference_name = "inspire_rh56bfx_faithful_primitive3_load_bearing_teacher"

    # Strict contact is already common in the source policy, but roughly half
    # of those contacts cannot carry load. Reward a thumb/opposed-finger pair
    # near the lower object half before the lift begins. The score uses each
    # active primitive's radius, so the target is shared across all three
    # shapes rather than encoding an object-specific grasp pose.
    tabletop_underwrap_rew_scale = 16000.0
    tabletop_underwrap_below_center_fraction = 0.05
    tabletop_underwrap_height_scale = 0.040
    tabletop_underwrap_radial_fraction = 0.80
    tabletop_underwrap_radial_scale = 0.040
    tabletop_underwrap_contact_scale = 0.030
    tabletop_underwrap_contact_margin = 0.010
    tabletop_underwrap_min_non_thumb_contacts = 1
    tabletop_underwrap_uses_opposition = True
    tabletop_underwrap_opposition_min_multiplier = 0.15
    tabletop_underwrap_progress_weight = 0.70
    tabletop_underwrap_pair_weight = 0.30
    tabletop_underwrap_uses_pregrasp_gate = False

    # Do not force an immediate lift from the first light contact. The policy
    # still controls every arm and hand action, and all scripted priors remain
    # disabled by the parent direct-RL contract.
    tabletop_no_lift_after_grasp_grace_steps = 18
    tabletop_no_lift_after_grasp_ramp_steps = 30


@configclass
class InspireRH56BFXFaithfulPrimitive3CartesianCarryTeacherEnvCfg(
    InspireRH56BFXFaithfulPrimitive3LoadBearingTeacherEnvCfg
):
    """Direct-RL continuation that lifts the palm instead of flipping the wrist."""

    reference_name = "inspire_rh56bfx_faithful_primitive3_cartesian_carry_teacher"

    # CuRobo's physical sphere/box/cylinder lifts agree on this local vertical
    # direction at the grasp pose. It is only a reward direction; the policy
    # continues to output every Franka and RH56BFX action itself.
    lift_arm_delta = INSPIRE_RH56BFX_CARTESIAN_CARRY_ARM_DELTA
    scripted_tabletop_lift_target_arm_delta = INSPIRE_RH56BFX_CARTESIAN_CARRY_ARM_DELTA
    scripted_tabletop_relative_lift_target_arm_delta = INSPIRE_RH56BFX_CARTESIAN_CARRY_ARM_DELTA

    # The load-bearing stage has already learned lower-half support. Reduce its
    # stationary reward and favor object-coupled palm translation with little
    # object-to-palm drift. This rejects the wrist-flip/toss shortcut observed
    # in the ep30 trace while preserving a dense direct-RL lift objective.
    tabletop_underwrap_rew_scale = 4000.0
    tabletop_grasped_arm_lift_rew_scale = 12000.0
    tabletop_lift_action_prior_rew_scale = 24000.0
    tabletop_palm_object_carry_rew_scale = 36000.0
    tabletop_object_palm_drift_penalty_scale = 42000.0
    tabletop_object_palm_drift_tolerance = 0.025
    tabletop_object_palm_drift_scale = 0.040
    tabletop_object_palm_drift_max_penalty = 4.0
    tabletop_relative_palm_lift_target = 0.075
    tabletop_success_requires_relative_palm_lift = True
    tabletop_success_min_relative_palm_lift = 0.045
    tabletop_success_max_object_palm_drift = 0.045

    tabletop_no_lift_after_grasp_penalty_scale = 32000.0
    tabletop_no_lift_after_grasp_grace_steps = 10
    tabletop_no_lift_after_grasp_ramp_steps = 20
    tabletop_object_carry_lift_rew_scale = 40000.0
    tabletop_object_carry_stall_penalty_scale = 16000.0
    joint_target_arm_max_delta = 0.012


@configclass
class InspireRH56BFXFaithfulPrimitive3PalmCoupledCarryTeacherEnvCfg(
    InspireRH56BFXFaithfulPrimitive3CartesianCarryTeacherEnvCfg
):
    """Static curriculum that rewards only palm-coupled vertical transport."""

    reference_name = "inspire_rh56bfx_faithful_primitive3_palm_coupled_carry_teacher"

    # The preceding stage can lift objects by flipping the wrist. Remove every
    # dense term that can be earned by object height alone. Final success and
    # hold bonuses remain behind the strict palm-lift and relative-drift gates.
    lift_progress_rew_scale = 0.0
    quality_lift_progress_rew_scale = 0.0
    lifted_true_grasp_rew_scale = 0.0
    tabletop_stable_catch_rew_scale = 0.0
    tabletop_grasped_palm_lift_rew_scale = 0.0
    tabletop_hover_height_progress_rew_scale = 0.0
    tabletop_hover_target_rew_scale = 0.0
    tabletop_hover_goal_rew_scale = 0.0
    tabletop_object_up_vel_rew_scale = 0.0

    # Dense lift credit now requires current force-backed enclosure, matching
    # palm/object upward velocity, low relative velocity, and matched progress.
    tabletop_grasped_arm_lift_rew_scale = 16000.0
    tabletop_lift_action_prior_rew_scale = 28000.0
    tabletop_object_carry_lift_rew_scale = 8000.0
    tabletop_palm_object_up_vel_rew_scale = 48000.0
    tabletop_palm_object_carry_rew_scale = 72000.0
    tabletop_palm_object_up_vel_target = 0.08

    tabletop_object_palm_drift_penalty_scale = 52000.0
    tabletop_object_palm_drift_tolerance = 0.015
    tabletop_object_palm_drift_scale = 0.030
    tabletop_object_palm_drift_max_penalty = 4.0
    tabletop_arm_object_lift_gap_penalty_scale = 30000.0
    tabletop_lift_without_object_penalty_scale = 30000.0
    tabletop_lift_without_current_grasp_penalty_scale = 30000.0
    tabletop_strict_grasp_loss_penalty_scale = 30000.0
    tabletop_force_grasp_loss_penalty_scale = 30000.0
    tabletop_object_carry_stall_penalty_scale = 14000.0

    # Let the six active fingers settle before defining the carry frame. The
    # policy still controls all 13 actions; this only chooses reward baselines.
    tabletop_arm_lift_progress_baseline_grasp_streak = 12
    # Remove absolute-target windup at the grasp-to-carry transition. The
    # policy still controls every subsequent action, now rate-limited from the
    # measured grasp pose instead of a stale joint-limit command.
    tabletop_arm_lift_baseline_sync_target_to_measured = True
    tabletop_lift_hand_target_latch_uses_measured_pos = True
    tabletop_lift_hand_target_close_fraction = 0.15
    tabletop_underwrap_rew_scale = 3500.0
    tabletop_no_lift_after_grasp_penalty_scale = 18000.0
    tabletop_no_lift_after_grasp_grace_steps = 24
    tabletop_no_lift_after_grasp_ramp_steps = 30
    joint_target_arm_max_delta = 0.008

    tabletop_success_min_relative_palm_lift = 0.045
    tabletop_success_max_object_palm_drift = 0.035


@configclass
class InspireRH56BFXFaithfulPrimitive3PalmCoupledCarry15mmTeacherEnvCfg(
    InspireRH56BFXFaithfulPrimitive3PalmCoupledCarryTeacherEnvCfg
):
    """First carry curriculum stage: retain a grasp through a short vertical lift."""

    reference_name = "inspire_rh56bfx_faithful_primitive3_palm_coupled_carry_15mm_teacher"

    # A short, force-backed lift supplies reachable positive examples before
    # asking PPO for the full 45 mm carry. Final evaluation still uses the
    # strict parent contract above.
    tabletop_success_lift_height = 0.015
    tabletop_relative_palm_lift_target = 0.020
    tabletop_success_min_relative_palm_lift = 0.012
    tabletop_success_max_object_palm_drift = 0.050
    dynamic_success_hold_steps = 4

    # Keep palm/object coupling profitable while avoiding the stationary
    # no-lift solution induced by the full-stage drift penalty.
    tabletop_object_palm_drift_tolerance = 0.035
    tabletop_object_palm_drift_scale = 0.045
    tabletop_object_palm_drift_max_penalty = 2.0
    tabletop_object_palm_drift_penalty_scale = 12000.0
    tabletop_arm_object_lift_gap_penalty_scale = 8000.0
    tabletop_lift_without_object_penalty_scale = 8000.0
    tabletop_lift_without_current_grasp_penalty_scale = 12000.0
    tabletop_strict_grasp_loss_penalty_scale = 12000.0
    tabletop_force_grasp_loss_penalty_scale = 12000.0
    tabletop_no_lift_after_grasp_penalty_scale = 24000.0


@configclass
class InspireRH56BFXFaithfulPrimitive3PalmCoupledCarry30mmTeacherEnvCfg(
    InspireRH56BFXFaithfulPrimitive3PalmCoupledCarryTeacherEnvCfg
):
    """Second carry curriculum stage: extend the physical carry to 30 mm."""

    reference_name = "inspire_rh56bfx_faithful_primitive3_palm_coupled_carry_30mm_teacher"

    tabletop_success_lift_height = 0.030
    tabletop_relative_palm_lift_target = 0.040
    tabletop_success_min_relative_palm_lift = 0.025
    tabletop_success_max_object_palm_drift = 0.040
    dynamic_success_hold_steps = 6

    tabletop_object_palm_drift_tolerance = 0.025
    tabletop_object_palm_drift_scale = 0.035
    tabletop_object_palm_drift_max_penalty = 3.0
    tabletop_object_palm_drift_penalty_scale = 30000.0
    tabletop_arm_object_lift_gap_penalty_scale = 18000.0
    tabletop_lift_without_object_penalty_scale = 18000.0
    tabletop_lift_without_current_grasp_penalty_scale = 22000.0
    tabletop_strict_grasp_loss_penalty_scale = 22000.0
    tabletop_force_grasp_loss_penalty_scale = 22000.0
    tabletop_no_lift_after_grasp_penalty_scale = 22000.0


@configclass
class InspireRH56BFXFaithfulPrimitive3VerticalCarry15mmTeacherEnvCfg(
    InspireRH56BFXFaithfulPrimitive3PalmCoupledCarry15mmTeacherEnvCfg
):
    """Sphere-only short carry with an explicit vertical palm-pose contract."""

    reference_name = "inspire_rh56bfx_faithful_primitive3_vertical_carry_15mm_teacher"

    # Stage one uses only the sphere. Box/cylinder are restored after the
    # physical carry itself is reliable, so they need not be spawned here.
    tabletop_object_asset_specs = (TABLETOP_INSPIRE_SPHERE_60MM_SPEC,)
    tabletop_asset_curriculum = False

    # Reward the same palm/object progress only when the palm translates in Z
    # without the wrist-flip shortcut. No additional stationary penalty is
    # used. The shaping bandwidth is deliberately broader than the final
    # success gate so PPO receives credit for incremental pose improvement.
    tabletop_vertical_palm_carry_rew_scale = 128000.0
    tabletop_vertical_palm_xy_scale = 0.050
    tabletop_vertical_palm_orientation_scale = 0.50
    tabletop_vertical_palm_velocity_rew_scale = 96000.0
    tabletop_vertical_palm_xy_vel_scale = 0.040
    tabletop_vertical_palm_ang_vel_scale = 0.40

    # A fixed joint-space lift direction is only locally valid around the
    # AnyDex calibration pose.  At policy-discovered grasps it rewards the
    # sideways wrist sweep seen in strict ep60 evaluation.  This stage learns
    # only from pose-preserving measured palm/object motion; direction-agnostic
    # upward and carry terms otherwise still pay for moving the grasp sideways.
    tabletop_grasped_arm_lift_rew_scale = 0.0
    tabletop_lift_action_prior_rew_scale = 0.0
    tabletop_object_carry_lift_rew_scale = 0.0
    tabletop_palm_object_carry_rew_scale = 0.0
    tabletop_palm_object_up_vel_rew_scale = 0.0
    tabletop_object_palm_drift_penalty_scale = 24000.0

    tabletop_success_max_palm_xy_drift = 0.025
    tabletop_success_max_palm_orientation_drift = 0.40


@configclass
class InspireRH56BFXFaithfulPrimitive3SphereBalancedLiftCommitTeacherEnvCfg(
    InspireRH56BFXFaithfulPrimitive3LiftCommitTeacherEnvCfg
):
    """Training-only sampler that compensates for the harder moving sphere."""

    reference_name = "inspire_rh56bfx_faithful_primitive3_sphere_balanced_lift_commit_teacher"
    tabletop_asset_sampling_weights = (0.60, 0.30, 0.10)


@configclass
class InspireRH56BFXAnyDexRing60BootstrapTeacherEnvCfg(
    InspireRH56BFXFaithfulSphere60RollingCurriculumTeacherEnvCfg
):
    """Static-to-dynamic teacher bootstrapped by a physically verified AnyDex Ring grasp."""

    reference_name = "inspire_rh56bfx_anydex_ring60_bootstrap_teacher"
    episode_length_s = 10.0
    dynamic_grasp_speed_curriculum_allow_decrease = False

    inspire_semantic_close_targets = INSPIRE_ANYDEX_RING60_ACTIVE_CLOSE_TARGETS
    reference_hand_fractions = (1.0,) * 6

    scripted_action_prior_enabled = True
    scripted_action_prior_zero_passthrough_enabled = True
    scripted_action_prior_residual_scale = 0.15
    scripted_action_prior_active_residual_scale = 0.15
    scripted_action_prior_inactive_residual_scale = 1.0
    scripted_action_prior_uses_strict_grasp = True

    scripted_tabletop_pregrasp_prior_enabled = True
    scripted_tabletop_pregrasp_prior_control_mode = "target_track"
    scripted_tabletop_pregrasp_arm_pos = INSPIRE_ANYDEX_RING60_GRASP_ARM_POS
    scripted_tabletop_pregrasp_prior_start_step = 0
    scripted_tabletop_pregrasp_prior_steps = 260
    scripted_tabletop_pregrasp_prior_ramp_steps = 1
    scripted_tabletop_pregrasp_hold_after_track = True
    tabletop_arm_lift_progress_baseline_pos = INSPIRE_ANYDEX_RING60_GRASP_ARM_POS

    scripted_action_prior_hand_start_step = 260
    scripted_action_prior_hand_ramp_steps = 120
    scripted_action_prior_hand_action = 1.0
    scripted_action_prior_hand_action_vector = (1.0,) * 6
    scripted_tabletop_hand_grasp_memory_action_vector = (1.0,) * 6

    scripted_tabletop_lift_target_prior_enabled = True
    scripted_tabletop_lift_target_arm_pos = INSPIRE_ANYDEX_RING60_LIFT_ARM_POS
    scripted_tabletop_lift_target_prior_ramp_steps = 1
    scripted_action_prior_lift_start_step = 380
    scripted_action_prior_lift_steps = 180
    scripted_action_prior_lift_requires_grasp = True
    scripted_action_prior_lift_uses_grasp_memory = True
    scripted_action_prior_lift_grasp_memory_min_steps = 2
    scripted_action_prior_lift_memory_requires_streak = False

    # A Ring grasp is a force-verified thumb/index pinch. Keep the original
    # unified three-finger task untouched and scope the two-digit contract to
    # this explicit AnyDex bootstrap task.
    min_finger_contacts = 2
    min_non_thumb_contacts = 1
    strict_success_min_finger_contacts = 2
    strict_success_min_non_thumb_contacts = 1
    object_force_grasp_min_non_thumb_contacts = 1
    tabletop_success_requires_force_grasp = True

    # The verified pose has 5.4 mm measured clearance while PhysX still
    # enforces all real hand/table collision meshes.
    tabletop_arm_clearance_margin = 0.003
    tabletop_success_requires_arm_clearance = True
    tabletop_terminate_on_arm_clearance_violation = False


@configclass
class InspireRH56BFXAnyDexRing60ArmResidualTeacherEnvCfg(
    InspireRH56BFXAnyDexRing60BootstrapTeacherEnvCfg
):
    """Learn moving-object interception while preserving AnyDex closure and lift."""

    reference_name = "inspire_rh56bfx_anydex_ring60_arm_residual_teacher"
    scripted_action_prior_active_residual_scale = 0.005
    scripted_action_prior_active_arm_residual_scale = 0.05
    scripted_action_prior_active_hand_residual_scale = 0.0
    scripted_action_prior_post_grasp_arm_residual_scale = 0.0


@configclass
class InspireRH56BFXAnyDexRing60ObjectRelativeTeacherEnvCfg(
    InspireRH56BFXAnyDexRing60ArmResidualTeacherEnvCfg
):
    """Track the moving sphere while preserving the verified AnyDex grasp manifold."""

    reference_name = "inspire_rh56bfx_anydex_ring60_object_relative_teacher"
    scripted_tabletop_object_relative_pregrasp_enabled = True
    scripted_tabletop_object_relative_pregrasp_mode = "iterative_pose_ik"
    scripted_tabletop_pregrasp_prior_control_mode = "object_relative_ik"
    scripted_tabletop_object_relative_pregrasp_hand_base_pos = (
        0.4768927508,
        0.0452512911,
        0.4932073997,
    )
    scripted_tabletop_object_relative_pregrasp_hand_base_quat = (
        0.6628679856,
        0.6522524054,
        -0.1400856442,
        0.3399247645,
    )
    scripted_tabletop_object_relative_pregrasp_lead_time = 0.08
    scripted_tabletop_object_relative_pregrasp_track_axes = (True, True, False)
    scripted_tabletop_object_relative_pregrasp_damping = 0.08
    scripted_tabletop_object_relative_pregrasp_max_cartesian_offset = 0.25
    scripted_tabletop_object_relative_pregrasp_max_joint_correction = 0.80
    scripted_tabletop_object_relative_pregrasp_max_ik_joint_delta = 0.04
    # Start each hand from the calibrated open pose and close independently
    # once the moving AnyDex target has actually been reached.
    scripted_action_prior_hand_start_step = 9999
    scripted_action_prior_hand_ramp_steps = 0
    scripted_action_prior_hand_proximity_trigger_enabled = True
    scripted_action_prior_hand_proximity_min_step = 20
    scripted_action_prior_hand_proximity_position_threshold = 0.010
    scripted_action_prior_hand_proximity_rotation_threshold = 0.20
    scripted_action_prior_hand_proximity_ramp_steps = 120
    scripted_action_prior_hand_proximity_speed_adaptive_enabled = True
    scripted_action_prior_hand_proximity_position_speed_gain = 0.05
    scripted_action_prior_hand_proximity_position_threshold_max = 0.015
    scripted_action_prior_hand_proximity_ramp_speed_gain = 300.0
    scripted_action_prior_hand_proximity_ramp_min_steps = 90
    scripted_action_prior_uses_force_grasp = True
    scripted_action_prior_lift_relative_to_grasp = True
    scripted_action_prior_lift_grasp_delay_steps = 0
    scripted_action_prior_lift_steps = 180


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
class InspireRH56BFXFaithfulUnifiedFallingTeacherEnvCfg(
    InspireUnifiedFallingBatonBenchmarkTeacherEnvCfg
):
    """Red/green falling teacher using the deployable RH56BFX physical contract."""

    reference_name = "inspire_rh56bfx_faithful_unified_falling_teacher"
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_mimic_robot_cfg(
        ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS
    )
    hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    sim_hand_joint_names = INSPIRE_ACTIVE_HAND_JOINT_NAMES
    reference_hand_fractions = (1.0,) * 6
    inspire_semantic_close_targets = INSPIRE_ANYDEX_SPHERE_ACTIVE_CLOSE_TARGETS
    touch_body_names = INSPIRE_MESH_CONTACT_BODY_NAMES
    robot_collision_disabled_body_names = ()
    robot_mimic_natural_frequency = 0.0
    robot_mimic_damping_ratio = 0.0
    robot_mimic_offset_overrides_deg = (
        ("index_intermediate_joint", 2.604352333),
        ("middle_intermediate_joint", 2.604352333),
        ("ring_intermediate_joint", 2.604352333),
        ("pinky_intermediate_joint", 2.604352333),
    )

    object_contact_force_diagnostics_enabled = True
    object_contact_force_threshold = 0.05
    falling_success_requires_force_grasp = True
    tabletop_post_success_hand_close_fraction = 0.0


@configclass
class InspireUnifiedFallingJointDeltaV26TeacherEnvCfg(
    _UnifiedFallingJointDeltaV26Contract,
    InspireRH56BFXFaithfulUnifiedFallingTeacherEnvCfg,
):
    """Official six-active RH56BFX falling teacher under the selected action."""

    reference_name = "inspire_unified_falling_jointdelta_v26_teacher"
    action_contract = "inspire_semantic_13d"
    robot_cfg: ArticulationCfg = _inspire_rh56bfx_mimic_robot_cfg(
        ISAACGYM_DYNAMIC_REVO2_LOWER_SAFE_ARM_POS
    )


@configclass
class InspireRH56BFXFaithfulUnifiedFallingCartesianTeacherEnvCfg(
    _PersistentCartesianWristDirectPolicyContract,
    InspireRH56BFXFaithfulUnifiedFallingTeacherEnvCfg,
):
    """RH56BFX red/green falling teacher with persistent Cartesian wrist control."""

    reference_name = "inspire_rh56bfx_faithful_unified_falling_cartesian_teacher"
    action_contract = "inspire_cartesian_wrist_12d"
    observation_space = 75


@configclass
class InspireRH56BFXFaithfulUnifiedFallingMeasuredCartesianTeacherEnvCfg(
    _CartesianWristDirectPolicyContract,
    InspireRH56BFXFaithfulUnifiedFallingTeacherEnvCfg,
):
    """RH56BFX falling teacher with the memoryless Cartesian contract used by rolling."""

    reference_name = "inspire_rh56bfx_faithful_unified_falling_measured_cartesian_teacher"
    action_contract = "inspire_cartesian_wrist_12d"
    observation_space = 75


@configclass
class InspireFallingBatonStableAffordanceFullSpeedEvalEnvCfg(InspireFallingBatonStableAffordanceTeacherEnvCfg):
    """Strict green-region Inspire falling-baton eval with full randomization."""

    reference_name = "inspire_z180_falling_baton_stable_affordance_full_speed_eval_teacher"
    dynamic_grasp_speed_curriculum = False
    falling_baton_start_initial_xy_speed_range = (0.00, 0.06)
    falling_baton_start_initial_z_speed_range = (0.04, 0.26)
    falling_baton_start_initial_ang_vel_range = (-1.0, 1.0)
