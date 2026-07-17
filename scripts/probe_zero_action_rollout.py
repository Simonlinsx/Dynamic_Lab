"""Run a zero-policy rollout to verify environment-side priors and reset geometry."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
EXT_SOURCE = REPO_ROOT / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80HomeSeedBootstrap-Teacher-Direct-v0",
)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument(
    "--warmup-steps",
    type=int,
    default=0,
    help="Settle under the same constant action before recording the measurement baseline.",
)
parser.add_argument("--steps", type=int, default=480)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--default-arm-pos",
    type=float,
    nargs=7,
    default=None,
    metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    help="Override the seven-joint reset/home pose for a clearance diagnostic.",
)
parser.add_argument(
    "--canonical-reset-alpha",
    type=float,
    default=None,
    help="Override canonical reset alpha (0=pregrasp, 1=official upright home).",
)
parser.add_argument(
    "--canonical-reset-pregrasp-arm-pos",
    type=float,
    nargs=7,
    default=None,
    metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    help="Diagnostic-only override for the canonical alpha=0 arm pose.",
)
parser.add_argument(
    "--canonical-reset-alpha-sweep",
    type=float,
    nargs="+",
    default=None,
    help="Probe several canonical reset interpolation points in one simulator launch.",
)
parser.add_argument("--canonical-reset-sweep-settle-steps", type=int, default=12)
parser.add_argument(
    "--canonical-reset-sweep-hand-action",
    type=float,
    default=-1.0,
    help="Hand action used while settling each reset-sweep pose.",
)
parser.add_argument(
    "--arm-action",
    type=float,
    default=0.0,
    help="Constant normalized action applied to every arm-control dimension.",
)
parser.add_argument(
    "--hand-action",
    type=float,
    default=0.0,
    help="Constant normalized action applied to every active hand dimension (-1=open, +1=close).",
)
parser.add_argument(
    "--teleport-object-z-offset",
    type=float,
    default=0.0,
    help="Diagnostic-only object Z teleport; never used by task training.",
)
parser.add_argument(
    "--teleport-object-at-step",
    type=int,
    default=-1,
    help="Zero-based rollout step at which to apply the diagnostic object teleport.",
)
parser.add_argument("--output-json", required=True)
parser.add_argument("--video-path", default=None)
parser.add_argument("--video-stride", type=int, default=2)
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--video-camera-eye", type=float, nargs=3, default=(1.35, -1.10, 0.84))
parser.add_argument("--video-camera-target", type=float, nargs=3, default=(0.58, 0.00, 0.38))
parser.add_argument("--video-camera-focal-length", type=float, default=22.0)
parser.add_argument("--video-camera-resolution", type=int, nargs=2, default=(1280, 720))
parser.add_argument("--debug-interval", type=int, default=60)
parser.add_argument(
    "--initial-target-lock-steps",
    type=int,
    default=None,
    help="Override both arm and hand initialization lock durations before environment creation.",
)
parser.add_argument("--initial-arm-target-lock-steps", type=int, default=None)
parser.add_argument("--initial-hand-target-lock-steps", type=int, default=None)
parser.add_argument(
    "--disable-self-collision",
    action="store_true",
    help="Diagnostic A/B override; task defaults remain unchanged.",
)
parser.add_argument(
    "--isolate-robot",
    action="store_true",
    help="Remove the table and zero scene gravity to isolate controller/robot drift.",
)
parser.add_argument(
    "--remove-table",
    action="store_true",
    help="Diagnostic A/B override that removes only the table.",
)
parser.add_argument(
    "--zero-scene-gravity",
    action="store_true",
    help="Diagnostic A/B override that zeroes only global scene gravity.",
)
parser.add_argument(
    "--freeze-object",
    action="store_true",
    help="Disable gravity only for task objects so diagnostics cannot auto-reset on drop.",
)
parser.add_argument(
    "--disable-robot-body-collision",
    action="append",
    default=[],
    help="Disable collision geometry for one robot body; repeat for a diagnostic group.",
)
parser.add_argument(
    "--report-external-contacts",
    action="store_true",
    help="Record env-0 robot contacts with the table, object, or ground.",
)
parser.add_argument("--hand-effort-limit", type=float, default=None)
parser.add_argument("--hand-velocity-limit", type=float, default=None)
parser.add_argument("--hand-stiffness", type=float, default=None)
parser.add_argument("--hand-damping", type=float, default=None)
parser.add_argument(
    "--osc-inertial-decoupling",
    choices=("on", "off"),
    default=None,
    help="Override full operational-space inertial decoupling before environment creation.",
)
parser.add_argument(
    "--osc-partial-inertial-decoupling",
    choices=("on", "off"),
    default=None,
    help="Override translation/rotation-only inertial decoupling before environment creation.",
)
parser.add_argument(
    "--osc-nullspace-control",
    choices=("none", "position"),
    default=None,
    help="Override operational-space nullspace control before environment creation.",
)
parser.add_argument(
    "--osc-nullspace-stiffness",
    type=float,
    default=None,
    help="Override nullspace position stiffness before environment creation.",
)
parser.add_argument(
    "--osc-motion-stiffness",
    type=float,
    nargs=6,
    default=None,
    metavar=("KX", "KY", "KZ", "KRX", "KRY", "KRZ"),
    help="Override six task-space stiffness values before environment creation.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video_path and hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402
from debug_video_artifacts import write_debug_video_artifacts  # noqa: E402


def _trace(message: str) -> None:
    print(f"[probe_zero] {message}", flush=True)


def _force_camera_update(camera, dt: float) -> None:
    if hasattr(camera, "update"):
        try:
            camera.update(dt=dt, force_recompute=True)
        except TypeError:
            camera.update(dt)


def _set_video_camera_pose(unwrapped_env) -> bool:
    camera = getattr(unwrapped_env, "_video_camera", None)
    if camera is None:
        return False
    origins = unwrapped_env.scene.env_origins
    eye = torch.tensor(args_cli.video_camera_eye, device=origins.device, dtype=origins.dtype).unsqueeze(0)
    target = torch.tensor(args_cli.video_camera_target, device=origins.device, dtype=origins.dtype).unsqueeze(0)
    camera.set_world_poses_from_view(origins[:1] + eye, origins[:1] + target)
    _force_camera_update(camera, float(getattr(unwrapped_env, "dt", unwrapped_env.step_dt)))
    return True


def _record_frame(env, frames: list[np.ndarray]) -> None:
    base = env.unwrapped
    camera = getattr(base, "_video_camera", None)
    if camera is None:
        frame = base.render(recompute=True)
        if frame is not None and frame.size > 0:
            frames.append(frame.copy())
        return

    _force_camera_update(camera, float(getattr(base, "dt", base.step_dt)))
    rgb = camera.data.output.get("rgb")
    if rgb is None or rgb.numel() == 0:
        return
    frame = rgb[0, ..., :3].detach().cpu().numpy()
    if frame.dtype != np.uint8:
        max_value = float(frame.max()) if frame.size else 0.0
        frame = np.clip(frame * 255.0 if max_value <= 1.0 else frame, 0, 255).astype(np.uint8)
    frames.append(frame.copy())


def _mean_log(extras: dict) -> dict[str, float]:
    return {key: float(value.item()) for key, value in extras.get("log", {}).items() if hasattr(value, "item")}


def _usd_debug_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _usd_hand_joint_debug(env) -> list[dict]:
    """Export the live drive and mimic properties authored on hand joints."""

    base = env.unwrapped
    hand_joint_names = {
        joint_name
        for joint_name in base.robot.joint_names
        if joint_name not in set(base.cfg.arm_joint_names)
    }
    robot_prefix = "/World/envs/env_0/Robot"
    debug_tokens = (
        "drive",
        "mimic",
        "target",
        "stiffness",
        "damping",
        "frequency",
        "gearing",
        "armature",
        "friction",
        "limit",
    )
    rows = []
    for prim in base.scene.stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(robot_prefix) or str(prim.GetName()) not in hand_joint_names:
            continue
        attributes = {
            attr.GetName(): _usd_debug_value(attr.Get())
            for attr in prim.GetAttributes()
            if any(token in attr.GetName().lower() for token in debug_tokens)
        }
        relationships = {
            rel.GetName(): [str(target) for target in rel.GetTargets()]
            for rel in prim.GetRelationships()
            if any(token in rel.GetName().lower() for token in ("mimic", "reference", "body"))
        }
        rows.append(
            {
                "path": path,
                "applied_schemas": [str(schema) for schema in prim.GetAppliedSchemas()],
                "attributes": attributes,
                "relationships": relationships,
            }
        )
    return rows


def _usd_robot_collision_bounds(env) -> list[dict]:
    """Return env-0 world AABBs for enabled robot collision geometry."""

    base = env.unwrapped
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=True,
    )
    rows = []
    robot_prefix = "/World/envs/env_0/Robot/"
    for prim in base.scene.stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(robot_prefix) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_api = UsdPhysics.CollisionAPI(prim)
        enabled_attr = collision_api.GetCollisionEnabledAttr()
        if enabled_attr.IsValid() and enabled_attr.Get() is False:
            continue
        aligned_box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        minimum = aligned_box.GetMin()
        maximum = aligned_box.GetMax()
        rows.append(
            {
                "path": path,
                "min_xyz": [float(value) for value in minimum],
                "max_xyz": [float(value) for value in maximum],
            }
        )
    rows.sort(key=lambda row: row["min_xyz"][2])
    return rows


def _enable_external_contact_report(env):
    base = env.unwrapped
    stage = base.scene.stage
    robot_prefix = "/World/envs/env_0/Robot/"
    rows: dict[tuple[str, str, str, str], dict] = {}
    for body_name in base.robot.body_names:
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{body_name}")
        if not prim.IsValid():
            continue
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report_api.CreateThresholdAttr().Set(0.0)

    def on_contact_report(contact_headers, contact_data):
        for header in contact_headers:
            actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            actor0_is_robot = actor0.startswith(robot_prefix)
            actor1_is_robot = actor1.startswith(robot_prefix)
            if actor0_is_robot == actor1_is_robot:
                continue
            collider0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            collider1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            if actor1_is_robot:
                actor0, actor1 = actor1, actor0
                collider0, collider1 = collider1, collider0
            key = (actor0, actor1, collider0, collider1)
            row = rows.setdefault(
                key,
                {
                    "robot_actor": actor0,
                    "other_actor": actor1,
                    "robot_collider": collider0,
                    "other_collider": collider1,
                    "event_count": 0,
                    "contact_point_count": 0,
                    "min_separation_m": float("inf"),
                    "max_impulse_norm": 0.0,
                },
            )
            row["event_count"] += 1
            start = int(header.contact_data_offset)
            stop = start + int(header.num_contact_data)
            for index in range(start, stop):
                datum = contact_data[index]
                impulse = datum.impulse
                impulse_norm = sum(float(component) ** 2 for component in impulse) ** 0.5
                row["contact_point_count"] += 1
                row["min_separation_m"] = min(
                    row["min_separation_m"], float(datum.separation)
                )
                row["max_impulse_norm"] = max(
                    row["max_impulse_norm"], impulse_norm
                )

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(
        on_contact_report
    )
    return subscription, rows


def _external_contact_rows(rows: dict) -> list[dict]:
    output = []
    for row in rows.values():
        item = dict(row)
        if item["min_separation_m"] == float("inf"):
            item["min_separation_m"] = None
        output.append(item)
    output.sort(
        key=lambda item: (
            item["max_impulse_norm"],
            item["contact_point_count"],
            item["event_count"],
        ),
        reverse=True,
    )
    return output


def _geometry_debug(env) -> dict:
    base = env.unwrapped
    base._compute_intermediate_values()
    origin = base.scene.env_origins[0]
    out = {}
    arm_ids = base._arm_joint_ids
    arm_pos = base.robot.data.joint_pos[0, arm_ids]
    arm_vel = base.robot.data.joint_vel[0, arm_ids]
    arm_target = base._joint_targets[0, arm_ids]
    out["arm_joint_pos"] = [float(v) for v in arm_pos.detach().cpu()]
    out["arm_joint_vel"] = [float(v) for v in arm_vel.detach().cpu()]
    out["arm_joint_target"] = [float(v) for v in arm_target.detach().cpu()]
    out["arm_target_tracking_error_max"] = float(
        torch.max(torch.abs(arm_target - arm_pos)).detach().cpu()
    )
    hand_ids = base._control_hand_joint_ids
    hand_pos = base.robot.data.joint_pos[0, hand_ids]
    hand_target = base._joint_targets[0, hand_ids]
    out["hand_joint_names"] = [base.robot.joint_names[joint_id] for joint_id in hand_ids]
    out["hand_joint_pos"] = [float(v) for v in hand_pos.detach().cpu()]
    out["hand_joint_target"] = [float(v) for v in hand_target.detach().cpu()]
    out["hand_target_tracking_error_max"] = float(
        torch.max(torch.abs(hand_target - hand_pos)).detach().cpu()
    )
    physical_hand_ids = [
        joint_id for joint_id in range(len(base.robot.joint_names)) if joint_id not in arm_ids
    ]
    physical_hand_pos = base.robot.data.joint_pos[0, physical_hand_ids]
    out["physical_hand_joint_names"] = [
        base.robot.joint_names[joint_id] for joint_id in physical_hand_ids
    ]
    out["physical_hand_joint_pos"] = [
        float(v) for v in physical_hand_pos.detach().cpu()
    ]
    physical_hand_vel = base.robot.data.joint_vel[0, physical_hand_ids]
    out["physical_hand_joint_vel"] = [
        float(v) for v in physical_hand_vel.detach().cpu()
    ]
    runtime_properties = {
        "physical_hand_stiffness": base.robot.root_physx_view.get_dof_stiffnesses(),
        "physical_hand_damping": base.robot.root_physx_view.get_dof_dampings(),
        "physical_hand_effort_limit": base.robot.root_physx_view.get_dof_max_forces(),
        "physical_hand_velocity_limit": base.robot.root_physx_view.get_dof_max_velocities(),
    }
    for key, values in runtime_properties.items():
        out[key] = [float(v) for v in values[0, physical_hand_ids].detach().cpu()]
    applied_torque = getattr(base.robot.data, "applied_torque", None)
    if applied_torque is not None:
        out["physical_hand_applied_torque"] = [
            float(v) for v in applied_torque[0, physical_hand_ids].detach().cpu()
        ]
    impedance_efforts = getattr(base, "_cartesian_impedance_joint_efforts", None)
    if impedance_efforts is not None:
        out["cartesian_impedance_joint_efforts"] = [
            float(v) for v in impedance_efforts[0].detach().cpu()
        ]
        out["cartesian_impedance_effort_norm"] = float(
            torch.linalg.norm(impedance_efforts[0]).detach().cpu()
        )
    impedance_saturation = getattr(
        base, "_cartesian_impedance_effort_saturation", None
    )
    if impedance_saturation is not None:
        out["cartesian_impedance_effort_saturation"] = float(
            impedance_saturation[0].detach().cpu()
        )
    impedance_position_error = getattr(
        base, "_cartesian_wrist_policy_position_error", None
    )
    if impedance_position_error is not None:
        out["cartesian_wrist_position_error"] = float(
            impedance_position_error[0].detach().cpu()
        )
    impedance_rotation_error = getattr(
        base, "_cartesian_wrist_policy_rotation_error", None
    )
    if impedance_rotation_error is not None:
        out["cartesian_wrist_rotation_error"] = float(
            impedance_rotation_error[0].detach().cpu()
        )
    for attr_name, key in (
        ("_cartesian_impedance_target_pose_b", "cartesian_impedance_target_pose_b"),
        ("_cartesian_impedance_desired_pose_b", "cartesian_impedance_desired_pose_b"),
    ):
        value = getattr(base, attr_name, None)
        if value is not None:
            out[key] = [float(v) for v in value[0].detach().cpu()]
    gravity = base.robot.root_physx_view.get_gravity_compensation_forces()[0, arm_ids]
    out["arm_gravity_compensation"] = [float(v) for v in gravity.detach().cpu()]
    for attr_name, key in (
        ("applied_torque", "arm_applied_torque"),
        ("computed_torque", "arm_computed_torque"),
    ):
        value = getattr(base.robot.data, attr_name, None)
        if value is not None:
            out[key] = [float(v) for v in value[0, arm_ids].detach().cpu()]
    palm_pos_w, palm_quat_w = base._current_palm_point_pose_w()
    out["palm_quat"] = [float(v) for v in palm_quat_w[0].detach().cpu()]
    palm_jacobian = base._compute_palm_jacobian_b(palm_quat_w)
    jacobian_singular_values = torch.linalg.svdvals(palm_jacobian[0])
    out["palm_jacobian_singular_values"] = [
        float(v) for v in jacobian_singular_values.detach().cpu()
    ]
    for attr_name, key in (
        ("_object_pos_w", "object_pos"),
        ("_palm_pos_w", "palm_pos"),
    ):
        value = getattr(base, attr_name, None)
        if value is not None:
            out[key] = [float(v) for v in (value[0] - origin).detach().cpu()]
    fingertip_pos_w = getattr(base, "_fingertip_pos_w", None)
    object_pos_w = getattr(base, "_object_pos_w", None)
    if fingertip_pos_w is not None:
        out["fingertip_positions"] = [
            [float(v) for v in point]
            for point in (fingertip_pos_w[0] - origin).detach().cpu().tolist()
        ]
        if object_pos_w is not None:
            out["fingertip_positions_object_relative"] = [
                [float(v) for v in point]
                for point in (
                    fingertip_pos_w[0] - object_pos_w[0].unsqueeze(0)
                ).detach().cpu().tolist()
            ]
    surface_dist = getattr(base, "_surface_dist", None)
    if surface_dist is not None:
        out["surface_distances"] = [float(v) for v in surface_dist[0].detach().cpu()]
        out["min_surface_distance"] = float(surface_dist[0].min().detach().cpu())
    for attr_name, key in (
        ("_true_grasp", "true_grasp"),
        ("_lifted", "lifted"),
        ("_stable_hold", "stable_hold"),
        ("_success", "success"),
        ("_strict_true_grasp", "strict_true_grasp"),
        ("_tabletop_arm_clearance_ok", "tabletop_arm_clearance_ok"),
    ):
        value = getattr(base, attr_name, None)
        if value is not None:
            out[key] = bool(value[0].detach().cpu())
    for attr_name, key in (
        ("_tabletop_arm_clearance_penalty", "tabletop_arm_clearance_penalty"),
        ("_tabletop_arm_clearance_min_margin", "tabletop_arm_clearance_min_margin"),
        ("_tabletop_arm_clearance_active_fraction", "tabletop_arm_clearance_active_fraction"),
    ):
        value = getattr(base, attr_name, None)
        if value is not None:
            out[key] = float(value[0].detach().cpu())
    return out


def main() -> None:
    settings = carb.settings.get_settings()
    settings.set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.canonical_reset_alpha is not None and hasattr(
        env_cfg, "canonical_reset_curriculum_override_alpha"
    ):
        env_cfg.canonical_reset_curriculum_override_alpha = float(
            args_cli.canonical_reset_alpha
        )
    if args_cli.canonical_reset_pregrasp_arm_pos is not None:
        if not hasattr(env_cfg, "canonical_reset_pregrasp_arm_pos"):
            raise ValueError("Task does not expose a canonical pregrasp pose.")
        env_cfg.canonical_reset_pregrasp_arm_pos = tuple(
            float(value)
            for value in args_cli.canonical_reset_pregrasp_arm_pos
        )
    if args_cli.default_arm_pos is not None:
        default_arm_pos = tuple(float(value) for value in args_cli.default_arm_pos)
        env_cfg.default_arm_pos = default_arm_pos
        for joint_name, joint_pos in zip(env_cfg.arm_joint_names, default_arm_pos):
            env_cfg.robot_cfg.init_state.joint_pos[joint_name] = joint_pos
        if hasattr(env_cfg, "tabletop_arm_lift_progress_baseline_pos"):
            env_cfg.tabletop_arm_lift_progress_baseline_pos = default_arm_pos
    if args_cli.isolate_robot or args_cli.remove_table:
        env_cfg.create_table = False
    if args_cli.isolate_robot or args_cli.zero_scene_gravity:
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    if args_cli.freeze_object:
        if bool(getattr(env_cfg, "tabletop_asset_set_enabled", False)):
            env_cfg.tabletop_object_asset_specs = tuple(
                {**spec, "disable_gravity": True}
                for spec in env_cfg.tabletop_object_asset_specs
            )
        elif getattr(getattr(env_cfg, "object_cfg", None), "spawn", None) is not None:
            env_cfg.object_cfg.spawn.rigid_props.disable_gravity = True
    if args_cli.disable_robot_body_collision:
        env_cfg.robot_collision_disabled_body_names = tuple(
            dict.fromkeys(
                tuple(getattr(env_cfg, "robot_collision_disabled_body_names", ()))
                + tuple(args_cli.disable_robot_body_collision)
            )
        )
    if args_cli.disable_self_collision:
        env_cfg.robot_cfg.spawn.self_collision = False
        env_cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions = False
    for actuator_name, actuator_cfg in env_cfg.robot_cfg.actuators.items():
        if actuator_name.startswith("franka_"):
            continue
        if args_cli.hand_effort_limit is not None:
            actuator_cfg.effort_limit_sim = float(args_cli.hand_effort_limit)
        if args_cli.hand_velocity_limit is not None:
            actuator_cfg.velocity_limit_sim = float(args_cli.hand_velocity_limit)
        if args_cli.hand_stiffness is not None:
            actuator_cfg.stiffness = float(args_cli.hand_stiffness)
        if args_cli.hand_damping is not None:
            actuator_cfg.damping = float(args_cli.hand_damping)
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.initial_target_lock_steps is not None:
        lock_steps = max(int(args_cli.initial_target_lock_steps), 0)
        if hasattr(env_cfg, "initial_arm_target_lock_steps"):
            env_cfg.initial_arm_target_lock_steps = lock_steps
        if hasattr(env_cfg, "initial_hand_target_lock_steps"):
            env_cfg.initial_hand_target_lock_steps = lock_steps
    if args_cli.initial_arm_target_lock_steps is not None and hasattr(
        env_cfg, "initial_arm_target_lock_steps"
    ):
        env_cfg.initial_arm_target_lock_steps = max(
            int(args_cli.initial_arm_target_lock_steps), 0
        )
    if args_cli.initial_hand_target_lock_steps is not None and hasattr(
        env_cfg, "initial_hand_target_lock_steps"
    ):
        env_cfg.initial_hand_target_lock_steps = max(
            int(args_cli.initial_hand_target_lock_steps), 0
        )
    if args_cli.osc_inertial_decoupling is not None:
        env_cfg.cartesian_impedance_inertial_dynamics_decoupling = (
            args_cli.osc_inertial_decoupling == "on"
        )
    if args_cli.osc_partial_inertial_decoupling is not None:
        env_cfg.cartesian_impedance_partial_inertial_dynamics_decoupling = (
            args_cli.osc_partial_inertial_decoupling == "on"
        )
    if args_cli.osc_nullspace_control is not None:
        env_cfg.cartesian_impedance_nullspace_control = args_cli.osc_nullspace_control
    if args_cli.osc_nullspace_stiffness is not None:
        env_cfg.cartesian_impedance_nullspace_stiffness = float(
            args_cli.osc_nullspace_stiffness
        )
    if args_cli.osc_motion_stiffness is not None:
        env_cfg.cartesian_impedance_motion_stiffness = tuple(
            float(value) for value in args_cli.osc_motion_stiffness
        )
    if args_cli.video_path and hasattr(env_cfg, "video_camera_enabled"):
        env_cfg.video_camera_enabled = True
        env_cfg.video_camera.data_types = ["rgb"]
        env_cfg.video_camera.spawn.focal_length = float(args_cli.video_camera_focal_length)
        env_cfg.video_camera.width = int(args_cli.video_camera_resolution[0])
        env_cfg.video_camera.height = int(args_cli.video_camera_resolution[1])
        if hasattr(env_cfg, "terminate_on_success"):
            env_cfg.terminate_on_success = False

    render_mode = "rgb_array" if args_cli.video_path else None
    _trace("creating env")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    _trace("env created")
    env.unwrapped.sim._app_control_on_stop_handle = None
    contact_subscription = None
    external_contact_rows = {}
    try:
        if args_cli.report_external_contacts:
            contact_subscription, external_contact_rows = _enable_external_contact_report(env)
        if args_cli.video_path:
            _set_video_camera_pose(env.unwrapped)
        _trace("reset")
        env.reset()
        _trace("reset done")
        if args_cli.video_path:
            _set_video_camera_pose(env.unwrapped)

        base = env.unwrapped
        device = base.device
        action_dim = int(base.cfg.action_space)
        arm_dim = int(base._policy_arm_action_dim())
        actions = torch.full(
            (args_cli.num_envs, action_dim), float(args_cli.arm_action), device=device
        )
        actions[:, arm_dim:] = float(args_cli.hand_action)
        canonical_reset_sweep = []
        if args_cli.canonical_reset_alpha_sweep:
            if not hasattr(base.cfg, "canonical_reset_curriculum_override_alpha"):
                raise ValueError("The selected task has no canonical reset curriculum.")
            final_reset_alpha = base.cfg.canonical_reset_curriculum_override_alpha
            sweep_actions = torch.zeros_like(actions)
            sweep_actions[:, arm_dim:] = float(
                args_cli.canonical_reset_sweep_hand_action
            )
            for alpha in args_cli.canonical_reset_alpha_sweep:
                if not 0.0 <= float(alpha) <= 1.0:
                    raise ValueError("Canonical reset sweep alpha must be in [0, 1].")
                base.cfg.canonical_reset_curriculum_override_alpha = float(alpha)
                env.reset()
                before_settle = _geometry_debug(env)
                for _ in range(max(int(args_cli.canonical_reset_sweep_settle_steps), 0)):
                    env.step(sweep_actions)
                after_settle = _geometry_debug(env)
                canonical_reset_sweep.append(
                    {
                        "alpha": float(alpha),
                        "before_settle": before_settle,
                        "after_settle": after_settle,
                    }
                )
            base.cfg.canonical_reset_curriculum_override_alpha = final_reset_alpha
            env.reset()
        reset_geometry = _geometry_debug(env)
        reset_robot_collision_bounds = _usd_robot_collision_bounds(env)
        for _ in range(max(int(args_cli.warmup_steps), 0)):
            env.step(actions)
        initial_geometry = _geometry_debug(env)
        _trace(
            f"constant action: arm_dim={arm_dim} arm={args_cli.arm_action:.3f} "
            f"hand_dim={action_dim - arm_dim} hand={args_cli.hand_action:.3f}"
        )
        metric_max = {
            "success": torch.zeros(args_cli.num_envs, device=device),
            "true_grasp": torch.zeros(args_cli.num_envs, device=device),
            "lifted": torch.zeros(args_cli.num_envs, device=device),
            "stable_hold": torch.zeros(args_cli.num_envs, device=device),
        }
        frames: list[np.ndarray] = []
        last_extras = {}
        terminated_any = False
        truncated_any = False
        clearance_violation_any = False
        unclean_lift_any = False
        unclean_lift_expected_before_step_any = False
        termination_steps: list[int] = []
        teleport_diagnostic = None
        dropped_any = False
        out_xy_any = False
        max_clearance_penalty = torch.zeros(args_cli.num_envs, device=device)
        min_clearance_margin = torch.full((args_cli.num_envs,), float("inf"), device=device)
        max_arm_target_tracking_error = torch.zeros(args_cli.num_envs, device=device)
        reward_finite_all = True
        reward_min = float("inf")
        reward_max = float("-inf")
        reward_mean_final = float("nan")
        debug_snapshots = []
        for step in range(int(args_cli.steps)):
            if (
                step == int(args_cli.teleport_object_at_step)
                and abs(float(args_cli.teleport_object_z_offset)) > 0.0
            ):
                object_asset = getattr(env.unwrapped, "object", None)
                if object_asset is None:
                    raise RuntimeError("Task has no object asset for the teleport diagnostic.")
                object_state = object_asset.data.root_state_w.clone()
                object_state[:, 2] += float(args_cli.teleport_object_z_offset)
                object_asset.write_root_pose_to_sim(object_state[:, :7])
                object_asset.write_root_velocity_to_sim(object_state[:, 7:13])
                env.unwrapped._compute_intermediate_values()
                teleport_height_delta = env.unwrapped._object_height_delta.detach().clone()
                teleport_latched = (
                    env.unwrapped._tabletop_clean_grasp_latched.detach().clone()
                )
                unclean_height = float(
                    getattr(env.unwrapped.cfg, "tabletop_unclean_lift_height", 0.012)
                )
                teleport_unclean_expected = (
                    (teleport_height_delta >= unclean_height) & (~teleport_latched)
                )
                unclean_lift_expected_before_step_any = (
                    unclean_lift_expected_before_step_any
                    or bool(torch.any(teleport_unclean_expected).detach().cpu())
                )
                teleport_diagnostic = {
                    "step": step,
                    "object_height_delta": [
                        float(value) for value in teleport_height_delta.cpu().tolist()
                    ],
                    "clean_grasp_latched": [
                        bool(value) for value in teleport_latched.cpu().tolist()
                    ],
                    "unclean_lift_expected": [
                        bool(value) for value in teleport_unclean_expected.cpu().tolist()
                    ],
                }
                _trace(
                    f"diagnostic object teleport: step={step} "
                    f"z_offset={args_cli.teleport_object_z_offset:.4f}"
                )
            if args_cli.video_path and step % max(args_cli.video_stride, 1) == 0:
                _record_frame(env, frames)
            _, rewards, terminated, truncated, last_extras = env.step(actions)
            reward_finite_all = reward_finite_all and bool(
                torch.all(torch.isfinite(rewards)).detach().cpu()
            )
            reward_min = min(reward_min, float(rewards.min().detach().cpu()))
            reward_max = max(reward_max, float(rewards.max().detach().cpu()))
            reward_mean_final = float(rewards.mean().detach().cpu())
            arm_target_tracking_error = torch.abs(
                env.unwrapped._joint_targets[:, env.unwrapped._arm_joint_ids]
                - env.unwrapped.robot.data.joint_pos[:, env.unwrapped._arm_joint_ids]
            ).amax(dim=-1)
            max_arm_target_tracking_error = torch.maximum(
                max_arm_target_tracking_error,
                arm_target_tracking_error,
            )
            for key in metric_max:
                tensor = last_extras.get(f"{key}_env")
                if tensor is not None:
                    metric_max[key] = torch.maximum(metric_max[key], tensor.float())
            clearance_penalty = last_extras.get("tabletop_arm_clearance_penalty_env")
            if clearance_penalty is not None:
                max_clearance_penalty = torch.maximum(max_clearance_penalty, clearance_penalty.float())
            clearance_margin = last_extras.get("tabletop_arm_clearance_min_margin_env")
            if clearance_margin is not None:
                min_clearance_margin = torch.minimum(min_clearance_margin, clearance_margin.float())
            clearance_violation = last_extras.get("tabletop_arm_clearance_violation_env")
            if clearance_violation is not None:
                clearance_violation_any = clearance_violation_any or bool(torch.any(clearance_violation).detach().cpu())
            unclean_lift = last_extras.get("tabletop_unclean_lift_env")
            if unclean_lift is not None:
                unclean_lift_any = unclean_lift_any or bool(
                    torch.any(unclean_lift).detach().cpu()
                )
            dropped = last_extras.get("dropped_env")
            if dropped is not None:
                dropped_any = dropped_any or bool(torch.any(dropped).detach().cpu())
            out_xy = last_extras.get("out_xy_env")
            if out_xy is not None:
                out_xy_any = out_xy_any or bool(torch.any(out_xy).detach().cpu())
            terminated_any = terminated_any or bool(torch.any(terminated).detach().cpu())
            if bool(torch.any(terminated).detach().cpu()):
                termination_steps.append(step)
            truncated_any = truncated_any or bool(torch.any(truncated).detach().cpu())
            if args_cli.debug_interval > 0 and (step + 1) % int(args_cli.debug_interval) == 0:
                position_error = getattr(
                    env.unwrapped, "_cartesian_wrist_policy_position_error", None
                )
                rotation_error = getattr(
                    env.unwrapped, "_cartesian_wrist_policy_rotation_error", None
                )
                impedance_effort = getattr(
                    env.unwrapped, "_cartesian_impedance_joint_efforts", None
                )
                effort_saturation = getattr(
                    env.unwrapped,
                    "_cartesian_impedance_effort_saturation",
                    None,
                )
                _trace(
                    "step="
                    f"{step + 1:04d} success={metric_max['success'].mean().item():.3f} "
                    f"lifted={metric_max['lifted'].mean().item():.3f} "
                    f"true={metric_max['true_grasp'].mean().item():.3f} "
                    f"reward={reward_mean_final:.3f} "
                    f"pos_err={float(position_error.mean()) if position_error is not None else float('nan'):.4f} "
                    f"rot_err={float(rotation_error.mean()) if rotation_error is not None else float('nan'):.4f} "
                    f"tau={float(torch.linalg.norm(impedance_effort, dim=-1).mean()) if impedance_effort is not None else float('nan'):.2f} "
                    f"sat={float(effort_saturation.mean()) if effort_saturation is not None else float('nan'):.3f}"
                )
                snapshot = _geometry_debug(env)
                snapshot["step"] = step + 1
                debug_snapshots.append(snapshot)

        if args_cli.video_path:
            _record_frame(env, frames)
            video_path = Path(args_cli.video_path).expanduser().resolve()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_path, frames, fps=args_cli.video_fps, macro_block_size=16)
            _trace(f"video={video_path}")
        else:
            video_path = None

        summary = {
            "task": args_cli.task,
            "num_envs": int(args_cli.num_envs),
            "warmup_steps": int(args_cli.warmup_steps),
            "steps": int(args_cli.steps),
            "arm_action": float(args_cli.arm_action),
            "default_arm_pos_override": (
                [float(value) for value in args_cli.default_arm_pos]
                if args_cli.default_arm_pos is not None
                else None
            ),
            "hand_action": float(args_cli.hand_action),
            "canonical_reset_alpha": args_cli.canonical_reset_alpha,
            "canonical_reset_pregrasp_arm_pos_override": (
                [
                    float(value)
                    for value in args_cli.canonical_reset_pregrasp_arm_pos
                ]
                if args_cli.canonical_reset_pregrasp_arm_pos is not None
                else None
            ),
            "canonical_reset_sweep": canonical_reset_sweep,
            "teleport_object_z_offset": float(args_cli.teleport_object_z_offset),
            "teleport_object_at_step": int(args_cli.teleport_object_at_step),
            "isolate_robot": bool(args_cli.isolate_robot),
            "remove_table": bool(args_cli.remove_table),
            "zero_scene_gravity": bool(args_cli.zero_scene_gravity),
            "freeze_object": bool(args_cli.freeze_object),
            "disabled_robot_body_collisions": list(args_cli.disable_robot_body_collision),
            "external_contacts": _external_contact_rows(external_contact_rows),
            "arm_action_dim": arm_dim,
            "hand_action_dim": action_dim - arm_dim,
            "initial_arm_target_lock_steps": int(
                getattr(env.unwrapped.cfg, "initial_arm_target_lock_steps", 0)
            ),
            "initial_hand_target_lock_steps": int(
                getattr(env.unwrapped.cfg, "initial_hand_target_lock_steps", 0)
            ),
            "hand_actuator_overrides": {
                "effort_limit": args_cli.hand_effort_limit,
                "velocity_limit": args_cli.hand_velocity_limit,
                "stiffness": args_cli.hand_stiffness,
                "damping": args_cli.hand_damping,
            },
            "osc_config": {
                "inertial_dynamics_decoupling": bool(
                    getattr(
                        env.unwrapped.cfg,
                        "cartesian_impedance_inertial_dynamics_decoupling",
                        False,
                    )
                ),
                "partial_inertial_dynamics_decoupling": bool(
                    getattr(
                        env.unwrapped.cfg,
                        "cartesian_impedance_partial_inertial_dynamics_decoupling",
                        False,
                    )
                ),
                "nullspace_control": str(
                    getattr(
                        env.unwrapped.cfg,
                        "cartesian_impedance_nullspace_control",
                        "none",
                    )
                ),
                "nullspace_stiffness": float(
                    getattr(
                        env.unwrapped.cfg,
                        "cartesian_impedance_nullspace_stiffness",
                        0.0,
                    )
                ),
                "motion_stiffness": [
                    float(value)
                    for value in getattr(
                        env.unwrapped.cfg,
                        "cartesian_impedance_motion_stiffness",
                        (),
                    )
                ],
            },
            "max_success_rate": float(metric_max["success"].mean().detach().cpu()),
            "max_true_grasp_rate": float(metric_max["true_grasp"].mean().detach().cpu()),
            "max_lifted_rate": float(metric_max["lifted"].mean().detach().cpu()),
            "max_stable_hold_rate": float(metric_max["stable_hold"].mean().detach().cpu()),
            "terminated_any": terminated_any,
            "truncated_any": truncated_any,
            "clearance_violation_any": clearance_violation_any,
            "unclean_lift_any": unclean_lift_any,
            "unclean_lift_expected_before_step_any": unclean_lift_expected_before_step_any,
            "termination_steps": termination_steps,
            "teleport_diagnostic": teleport_diagnostic,
            "dropped_any": dropped_any,
            "out_xy_any": out_xy_any,
            "max_clearance_penalty": float(max_clearance_penalty.max().detach().cpu()),
            "min_clearance_margin": float(min_clearance_margin.min().detach().cpu()),
            "max_arm_target_tracking_error": float(
                max_arm_target_tracking_error.max().detach().cpu()
            ),
            "reward_finite_all": reward_finite_all,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "reward_mean_final": reward_mean_final,
            "reset_geometry": reset_geometry,
            "reset_robot_collision_bounds": reset_robot_collision_bounds,
            "initial_geometry": initial_geometry,
            "final_log": _mean_log(last_extras),
            "final_geometry": _geometry_debug(env),
            "usd_hand_joint_debug": _usd_hand_joint_debug(env),
            "debug_snapshots": debug_snapshots,
            "video_path": str(video_path) if video_path is not None else None,
        }
        output_path = Path(args_cli.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if video_path is not None:
            summary["debug_video_sidecars"] = write_debug_video_artifacts(
                video_path,
                method="constant_action_physics_probe",
                args=args_cli,
                env_cfg=env.unwrapped.cfg,
                metrics=summary,
                extra={"primary_metrics_json": str(output_path)},
            )
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _trace(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    if exit_code:
        raise SystemExit(exit_code)
