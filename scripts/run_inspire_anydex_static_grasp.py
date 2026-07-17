#!/usr/bin/env python3
"""Execute DOMINO AnyDex candidates with the faithful six-motor RH56BFX hand."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from isaaclab.app import AppLauncher


DEFAULT_TASK = "SimToolReal-Inspire-Franka-RH56BFXFaithfulSphere60RollingCurriculum-Teacher-Direct-v0"
DEFAULT_OUTPUT_ROOT = Path("outputs/anydex_inspire/20260713_sphere60_physical")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--candidate-ranks", type=int, nargs="+", default=(0, 1, 2))
parser.add_argument("--object-pos", type=float, nargs=3, default=(0.535, 0.075, 0.328))
parser.add_argument("--object-shape", choices=("sphere", "box", "cylinder"), default="sphere")
parser.add_argument("--object-size", type=float, nargs=3, default=(0.060, 0.060, 0.060))
parser.add_argument("--object-radius", type=float, default=0.030)
parser.add_argument("--object-mass", type=float, default=0.046)
parser.add_argument("--table-top-z", type=float, default=0.298)
parser.add_argument(
    "--table-pos-xy",
    type=float,
    nargs=2,
    default=(0.58, -0.08),
    help="Fixed large-table center in the environment frame.",
)
parser.add_argument(
    "--stage-object-away",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Keep the object clear of the reset hand, then restore it after reaching pregrasp.",
)
parser.add_argument("--object-staging-offset", type=float, nargs=3, default=(0.28, -0.20, 0.0))
parser.add_argument("--object-placement-settle-steps", type=int, default=12)
parser.add_argument(
    "--initial-arm-pos",
    type=float,
    nargs=7,
    default=None,
    metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    help="Optional collision-free Franka reset posture for AnyDex motion execution.",
)
parser.add_argument("--lift-height", type=float, default=0.10)
parser.add_argument("--settle-steps", type=int, default=40)
parser.add_argument("--safe-waypoint-height", type=float, default=0.60)
parser.add_argument("--safe-lift-steps", type=int, default=100)
parser.add_argument("--safe-align-steps", type=int, default=180)
parser.add_argument("--pregrasp-steps", type=int, default=120)
parser.add_argument("--approach-steps", type=int, default=150)
parser.add_argument("--close-steps", type=int, default=140)
parser.add_argument("--post-close-steps", type=int, default=40)
parser.add_argument(
    "--force-limited-close",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Stop each flexion motor after sustained physical contact with the object.",
)
parser.add_argument("--finger-stop-force", type=float, default=2.0)
parser.add_argument("--finger-stop-consecutive-steps", type=int, default=3)
parser.add_argument("--lift-steps", type=int, default=180)
parser.add_argument("--hold-steps", type=int, default=120)
parser.add_argument("--strict-hold-steps", type=int, default=20)
parser.add_argument("--ik-command-type", choices=("pose", "position"), default="pose")
parser.add_argument(
    "--arm-execution",
    choices=("differential_ik", "curobo_joint_trajectory"),
    default="differential_ik",
    help="Use the legacy Cartesian IK diagnostic or execute a DOMINO CuRobo Franka trajectory.",
)
parser.add_argument(
    "--max-ik-joint-delta",
    type=float,
    default=0.0,
    help="Optional per-control-step clamp around measured arm joint position; zero disables it.",
)
parser.add_argument(
    "--safe-lift-only",
    action="store_true",
    help="Stop after the first vertical waypoint for a minimal IK/actuation diagnostic.",
)
parser.add_argument(
    "--curobo-plan",
    type=Path,
    default=None,
    help="Optional precomputed .npz from plan_anydex_inspire_curobo.py.",
)
parser.add_argument(
    "--curobo-python",
    type=Path,
    default=Path("/data1/linsixu/miniconda3/envs/domino/bin/python"),
)
parser.add_argument(
    "--curobo-planner-script",
    type=Path,
    default=Path(__file__).resolve().with_name("plan_anydex_inspire_curobo.py"),
)
parser.add_argument("--curobo-path-time-scale", type=float, default=3.0)
parser.add_argument("--curobo-phase-settle-steps", type=int, default=20)
parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
parser.add_argument(
    "--anydex-result",
    type=Path,
    default=Path("outputs/anydex_inspire/20260713_sphere60/sphere60_result.json"),
)
parser.add_argument("--run-predictor", action="store_true")
parser.add_argument("--anydex-input", type=Path, default=None)
parser.add_argument("--anydex-root", type=Path, default=Path("/data1/linsixu/AnyDexGrasp"))
parser.add_argument("--domino-root", type=Path, default=Path("/data1/linsixu/DOMINO"))
parser.add_argument(
    "--anydex-python",
    type=Path,
    default=Path("/data1/linsixu/miniconda3/envs/anydex-torch/bin/python"),
)
parser.add_argument("--pregrasp-distance", type=float, default=0.075)
parser.add_argument("--depth-base-offset", type=float, default=0.010)
parser.add_argument("--frame-offset-xyz", type=float, nargs=3, default=(-0.00902886, 0.00683512, 0.02743894))
parser.add_argument("--frame-offset-rpy", type=float, nargs=3, default=(0.10579421, 0.04518228, 1.4874773))
parser.add_argument(
    "--grasp-type-override",
    type=int,
    choices=tuple(range(1, 9)),
    default=None,
    help="Optional object-conditioned AnyDex grasp type (7 is Sphere_3_Finger).",
)
parser.add_argument(
    "--width-override",
    type=float,
    default=None,
    help="Optional object-conditioned AnyDex aperture in meters.",
)
parser.add_argument(
    "--active-hand-targets",
    type=float,
    nargs=6,
    default=None,
    metavar=("THUMB_YAW", "THUMB_FLEX", "INDEX", "MIDDLE", "RING", "PINKY"),
    help="Optional six-motor physical close endpoint; the AnyDex pose is unchanged.",
)
parser.add_argument("--video-path", type=Path, default=None)
parser.add_argument("--video-env-id", type=int, default=0)
parser.add_argument("--video-stride", type=int, default=2)
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--video-camera-eye", type=float, nargs=3, default=(1.25, -1.05, 0.72))
parser.add_argument("--video-camera-target", type=float, nargs=3, default=(0.54, 0.05, 0.40))
parser.add_argument("--video-resolution", type=int, nargs=2, default=(960, 544))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video_path and hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    compute_pose_error,
    matrix_from_quat,
    quat_from_matrix,
    quat_inv,
    quat_slerp,
    subtract_frame_transforms,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402
from simtoolreal_lab.anydex import (  # noqa: E402
    ACTIVE_JOINT_NAMES,
    ANYDEX_REQUIRED_NON_THUMB_CONTACTS,
    AnyDexInspirePaths,
    load_anydex_candidates,
    make_primitive_predictor_input,
    run_predictor,
)
from simtoolreal_lab.anydex.inspire_adapter import ACTIVE_JOINT_UPPER  # noqa: E402
from simtoolreal_lab.tasks.dynamic_dexterous_grasp.dynamic_dexterous_grasp_env_cfg import (  # noqa: E402
    _object_cfg_from_tabletop_spec,
)


def _trace(message: str) -> None:
    print(f"[INSPIRE-ANYDEX] {message}", flush=True)


def _resolve_output_path(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _prepare_anydex_result(paths: AnyDexInspirePaths, output_root: Path) -> Path:
    result_path = _resolve_output_path(args_cli.anydex_result)
    if not args_cli.run_predictor:
        if not result_path.exists():
            raise FileNotFoundError(f"AnyDex result does not exist: {result_path}")
        return result_path
    input_path = (
        _resolve_output_path(args_cli.anydex_input)
        if args_cli.anydex_input is not None
        else output_root / "anydex_input.npz"
    )
    make_primitive_predictor_input(
        input_path,
        object_center_world=args_cli.object_pos,
        shape=str(args_cli.object_shape),
        size=args_cli.object_size,
    )
    run_predictor(
        input_path,
        result_path,
        paths=paths,
        candidate_limit=max(max(args_cli.candidate_ranks) + 1, 10),
        visualization_dir=output_root / "anydex_visualization",
    )
    return result_path


def _diagnostic_object_spec() -> dict:
    size = tuple(float(value) for value in args_cli.object_size)
    if any(value <= 0.0 for value in size):
        raise ValueError("--object-size dimensions must be positive")
    if args_cli.object_shape in {"sphere", "cylinder"} and float(args_cli.object_radius) <= 0.0:
        raise ValueError("--object-radius must be positive")
    if args_cli.object_shape == "sphere" and not np.allclose(
        np.asarray(size),
        2.0 * float(args_cli.object_radius),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("sphere --object-size must equal 2 * --object-radius on all axes")
    if args_cli.object_shape == "cylinder" and not np.allclose(
        np.asarray(size[:2]),
        2.0 * float(args_cli.object_radius),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("cylinder x/y --object-size must equal 2 * --object-radius")
    colors = {
        "sphere": (0.20, 0.72, 0.38),
        "box": (0.90, 0.32, 0.16),
        "cylinder": (0.12, 0.52, 0.90),
    }
    return {
        "asset_id": f"diagnostic/anydex_{args_cli.object_shape}",
        "category": str(args_cli.object_shape),
        "proxy_shape": str(args_cli.object_shape),
        "size": size,
        "radius": float(args_cli.object_radius),
        "height": float(size[2]),
        "axis": "Z",
        "mass": float(args_cli.object_mass),
        "color": colors[str(args_cli.object_shape)],
        "affordance_mode": "omni_grasp",
        "static_friction": 0.75,
        "dynamic_friction": 0.75,
        "restitution": 0.0,
        "friction_combine_mode": "multiply",
        "restitution_combine_mode": "multiply",
        "contact_offset": 0.002,
        "rest_offset": 0.0,
        "positive_fraction": 1.0,
        "negative_fraction": 0.0,
    }


def _configure_env(env_cfg, num_envs: int) -> None:
    env_cfg.seed = int(args_cli.seed)
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.episode_length_s = 60.0
    target_object_pos = tuple(float(value) for value in args_cli.object_pos)
    if args_cli.stage_object_away:
        spawn_object_pos = tuple(
            target_object_pos[index] + float(args_cli.object_staging_offset[index])
            for index in range(3)
        )
    else:
        spawn_object_pos = target_object_pos
    object_spec = _diagnostic_object_spec()
    env_cfg.object_start_pos = spawn_object_pos
    env_cfg.object_cfg = _object_cfg_from_tabletop_spec(object_spec, pos=env_cfg.object_start_pos)
    env_cfg.tabletop_object_asset_specs = (object_spec,)
    env_cfg.tabletop_asset_set_enabled = True
    env_cfg.tabletop_asset_sampling_weights = (1.0,)
    env_cfg.table_top_z = float(args_cli.table_top_z)
    env_cfg.table_cfg.init_state.pos = (
        float(args_cli.table_pos_xy[0]),
        float(args_cli.table_pos_xy[1]),
        float(args_cli.table_top_z) - 0.0225,
    )
    env_cfg.reset_object_pos_noise = (0.0, 0.0, 0.0)
    env_cfg.reset_arm_pos_noise = 0.0
    if args_cli.initial_arm_pos is not None:
        initial_arm_pos = tuple(float(value) for value in args_cli.initial_arm_pos)
        env_cfg.default_arm_pos = initial_arm_pos
        env_cfg.robot_cfg.init_state.joint_pos.update(
            {f"panda_joint{index + 1}": value for index, value in enumerate(initial_arm_pos)}
        )
    env_cfg.inspire_semantic_close_targets = tuple(float(value) for value in ACTIVE_JOINT_UPPER)
    env_cfg.reference_hand_fractions = (1.0,) * 6
    env_cfg.initial_arm_target_lock_steps = 0
    env_cfg.initial_hand_target_lock_steps = 0
    env_cfg.terminate_on_success = False
    env_cfg.workspace_xy_limit = 2.0
    env_cfg.tabletop_terminate_on_arm_clearance_violation = False
    env_cfg.tabletop_success_requires_arm_clearance = False
    env_cfg.tabletop_lift_hand_target_lock_enabled = False
    env_cfg.tabletop_post_success_hand_close_fraction = 0.0
    env_cfg.dynamic_tabletop_persistent_motion = False
    env_cfg.dynamic_tabletop_release_motion_on_contact = False
    env_cfg.dynamic_grasp_speed_curriculum = False
    env_cfg.tabletop_asset_curriculum = False
    env_cfg.dynamic_tabletop_start_speed_range = (0.0, 0.0)
    env_cfg.dynamic_tabletop_initial_speed_range = (0.0, 0.0)
    env_cfg.dynamic_tabletop_start_yaw_rate_range = (0.0, 0.0)
    env_cfg.dynamic_tabletop_initial_yaw_rate_range = (0.0, 0.0)
    for name in (
        "scripted_action_prior_enabled",
        "scripted_tabletop_pregrasp_prior_enabled",
        "scripted_tabletop_approach_action_prior_enabled",
        "scripted_tabletop_relative_lift_target_prior_enabled",
        "scripted_tabletop_lift_target_prior_enabled",
        "scripted_tabletop_hand_grasp_memory_prior_enabled",
    ):
        if hasattr(env_cfg, name):
            setattr(env_cfg, name, False)
    env_cfg.object_shape = str(args_cli.object_shape)
    env_cfg.object_radius = float(args_cli.object_radius)
    env_cfg.object_size = tuple(float(value) for value in args_cli.object_size)
    if args_cli.video_path:
        env_cfg.video_camera_enabled = True
        env_cfg.video_camera.data_types = ["rgb"]
        env_cfg.video_camera.width = int(args_cli.video_resolution[0])
        env_cfg.video_camera.height = int(args_cli.video_resolution[1])
        env_cfg.video_camera.spawn.focal_length = 24.0


def _place_object_at_target(base) -> None:
    env_ids = torch.arange(base.num_envs, dtype=torch.long, device=base.device)
    local_pos = torch.tensor(args_cli.object_pos, dtype=torch.float32, device=base.device).view(1, 3)
    local_pos = local_pos.expand(base.num_envs, -1).clone()
    state = base.object.data.default_root_state.clone()
    state[:, :3] = base.scene.env_origins + local_pos
    state[:, 7:] = 0.0
    base.object.write_root_pose_to_sim(state[:, :7], env_ids)
    base.object.write_root_velocity_to_sim(state[:, 7:], env_ids)
    if hasattr(base, "_object_start_pos"):
        base._object_start_pos = local_pos.clone()
    if hasattr(base, "_active_object_start_z"):
        base._active_object_start_z[:] = local_pos[:, 2]
    if hasattr(base, "_tabletop_cmd_lin_vel_w"):
        base._tabletop_cmd_lin_vel_w.zero_()
    if hasattr(base, "_tabletop_cmd_yaw_rate"):
        base._tabletop_cmd_yaw_rate.zero_()
    if hasattr(base, "_set_tabletop_hover_targets"):
        base._set_tabletop_hover_targets(env_ids, state[:, :3])
    base.scene.write_data_to_sim()
    base.sim.forward()
    base._compute_intermediate_values()


def _set_camera(base) -> None:
    camera = getattr(base, "_video_camera", None)
    if camera is None:
        return
    eye = torch.tensor(args_cli.video_camera_eye, dtype=torch.float32, device=base.device).view(1, 3)
    target = torch.tensor(args_cli.video_camera_target, dtype=torch.float32, device=base.device).view(1, 3)
    camera.set_world_poses_from_view(base.scene.env_origins + eye, base.scene.env_origins + target)
    if hasattr(camera, "_update_poses") and hasattr(camera, "_ALL_INDICES"):
        camera._update_poses(camera._ALL_INDICES)


def _record_frame(base, frames: list[np.ndarray]) -> None:
    camera = getattr(base, "_video_camera", None)
    if camera is None:
        return
    camera.update(float(base.dt), force_recompute=True)
    rgb = camera.data.output.get("rgb")
    if rgb is None or rgb.numel() == 0:
        return
    frame = rgb[int(args_cli.video_env_id), ..., :3].detach().cpu().numpy()
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0 if float(frame.max()) <= 1.0 else frame, 0, 255).astype(np.uint8)
    frames.append(frame.copy())


def _target_to_action(base, target: torch.Tensor, joint_ids: list[int], moving_average: float) -> torch.Tensor:
    current = base._joint_targets[:, joint_ids]
    lower = base._joint_lower_limits[joint_ids].unsqueeze(0)
    upper = base._joint_upper_limits[joint_ids].unsqueeze(0)
    center = torch.clamp(base._default_joint_pos[:, joint_ids], lower, upper)
    target = torch.clamp(target, lower, upper)
    blend = max(float(moving_average), 1.0e-6)
    raw_target = torch.clamp((target - (1.0 - blend) * current) / blend, lower, upper)
    positive_span = torch.clamp(upper - center, min=1.0e-6)
    negative_span = torch.clamp(center - lower, min=1.0e-6)
    delta = raw_target - center
    return torch.clamp(torch.where(delta >= 0.0, delta / positive_span, delta / negative_span), -1.0, 1.0)


def _active_hand_target_to_action(base, target: torch.Tensor, joint_ids: list[int]) -> torch.Tensor:
    """Invert Inspire's semantic -1=open/+1=close action mapping."""
    current = base._joint_targets[:, joint_ids]
    lower = base._joint_lower_limits[joint_ids].unsqueeze(0)
    upper = base._joint_upper_limits[joint_ids].unsqueeze(0)
    open_target = torch.clamp(base._default_joint_pos[:, joint_ids], lower, upper)
    close_target = torch.tensor(
        base.cfg.inspire_semantic_close_targets,
        dtype=target.dtype,
        device=target.device,
    ).view(1, -1)
    close_target = torch.clamp(close_target, lower, upper)
    target = torch.clamp(target, torch.minimum(open_target, close_target), torch.maximum(open_target, close_target))
    blend = max(float(base.cfg.hand_moving_average), 1.0e-6)
    raw_target = (target - (1.0 - blend) * current) / blend
    span = close_target - open_target
    fraction = torch.where(
        torch.abs(span) > 1.0e-6,
        (raw_target - open_target) / span,
        torch.zeros_like(raw_target),
    )
    reference = torch.tensor(
        base.cfg.reference_hand_fractions,
        dtype=target.dtype,
        device=target.device,
    ).view(1, -1)
    action = 2.0 * torch.clamp(fraction, 0.0, 1.0) / torch.clamp(reference, min=1.0e-6) - 1.0
    return torch.clamp(action, -1.0, 1.0)


class PhysicalStats:
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        required_non_thumb_contacts: torch.Tensor,
    ):
        self.required_non_thumb_contacts = required_non_thumb_contacts.to(
            device=device, dtype=torch.long
        ).reshape(num_envs)
        self.max_height_delta = torch.zeros(num_envs, device=device)
        self.max_force_sum = torch.zeros(num_envs, device=device)
        self.max_force_grasp = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.max_strict_grasp = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.hold_streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.max_hold_streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.max_anydex_force_grasp = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.anydex_hold_streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.max_anydex_hold_streak = torch.zeros(num_envs, dtype=torch.long, device=device)

    def update(self, base) -> None:
        base._compute_intermediate_values()
        force_grasp = getattr(
            base, "_force_grasp", torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        )
        strict_grasp = getattr(base, "_strict_true_grasp", base._true_grasp)
        height_delta = base._object_height_delta
        rel_vel = getattr(base, "_object_palm_rel_vel", torch.full_like(height_delta, float("inf")))
        stable = rel_vel < float(base.cfg.stable_object_palm_vel)
        strict_now = force_grasp & strict_grasp & (height_delta >= 0.045) & stable
        self.hold_streak = torch.where(strict_now, self.hold_streak + 1, torch.zeros_like(self.hold_streak))
        self.max_hold_streak = torch.maximum(self.max_hold_streak, self.hold_streak)
        self.max_height_delta = torch.maximum(self.max_height_delta, height_delta)
        self.max_force_grasp |= force_grasp
        self.max_strict_grasp |= strict_grasp
        forces = getattr(base, "_object_fingertip_contact_forces", None)
        if isinstance(forces, torch.Tensor):
            self.max_force_sum = torch.maximum(self.max_force_sum, forces.sum(dim=-1))
            threshold = max(
                float(getattr(base.cfg, "object_contact_force_threshold", 0.05)), 0.0
            )
            force_contacts = forces > threshold
            anydex_force_grasp = force_contacts[:, 0] & (
                force_contacts[:, 1:].sum(dim=-1) >= self.required_non_thumb_contacts
            )
            self.max_anydex_force_grasp |= anydex_force_grasp
            anydex_strict_now = anydex_force_grasp & (height_delta >= 0.045) & stable
            self.anydex_hold_streak = torch.where(
                anydex_strict_now,
                self.anydex_hold_streak + 1,
                torch.zeros_like(self.anydex_hold_streak),
            )
            self.max_anydex_hold_streak = torch.maximum(
                self.max_anydex_hold_streak, self.anydex_hold_streak
            )


class AnyDexExecutor:
    def __init__(
        self,
        env,
        candidate_matrices: torch.Tensor,
        hand_targets: torch.Tensor,
        required_non_thumb_contacts: torch.Tensor,
    ):
        self.env = env
        self.base = env.unwrapped
        self.candidate_matrices = candidate_matrices
        self.hand_targets = hand_targets
        self.frames: list[np.ndarray] = []
        self.frame_step = 0
        body_ids = self.base.robot.find_bodies("hand_base_link")[0]
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one hand_base_link, found {body_ids}")
        self.ee_body_id = int(body_ids[0])
        self.ee_jacobian_id = self.ee_body_id - 1
        self.arm_joint_ids = list(self.base._arm_joint_ids)
        self.hand_joint_ids = list(self.base._control_hand_joint_ids)
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type=str(args_cli.ik_command_type),
                use_relative_mode=False,
                ik_method="dls",
            ),
            num_envs=self.base.num_envs,
            device=self.base.device,
        )
        self.required_non_thumb_contacts = required_non_thumb_contacts
        self.stats = PhysicalStats(
            self.base.num_envs,
            self.base.device,
            self.required_non_thumb_contacts,
        )
        self.last_arm_target = self.base.robot.data.joint_pos[:, self.arm_joint_ids].clone()

    def _desired_world_pose(self, matrices_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position = matrices_local[:, :3, 3] + self.base.scene.env_origins
        quaternion = quat_from_matrix(matrices_local[:, :3, :3])
        return position, quaternion

    def _ik_joint_target(self, desired_pos_w: torch.Tensor, desired_quat_w: torch.Tensor) -> torch.Tensor:
        robot = self.base.robot
        root_pose_w = robot.data.root_state_w[:, :7]
        desired_pos_b, desired_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], desired_pos_w, desired_quat_w
        )
        if args_cli.ik_command_type == "position":
            self.ik.set_command(desired_pos_b, ee_quat=desired_quat_b)
        else:
            self.ik.set_command(torch.cat((desired_pos_b, desired_quat_b), dim=-1))
        ee_pose_w = robot.data.body_state_w[:, self.ee_body_id, :7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
        )
        jacobian = robot.root_physx_view.get_jacobians()[:, self.ee_jacobian_id, :, self.arm_joint_ids].clone()
        # PhysX reports each link Jacobian at its center of mass, while
        # body_state_w stores the link-frame origin pose.  RH56BFX's palm COM
        # is offset by several centimeters, so shift the linear Jacobian to
        # hand_base_link before solving IK.
        link_rotation_w = matrix_from_quat(ee_pose_w[:, 3:7])
        com_offset_link = robot.data.com_pos_b[:, self.ee_body_id]
        link_from_com_w = -torch.bmm(link_rotation_w, com_offset_link.unsqueeze(-1)).squeeze(-1)
        angular_columns_w = jacobian[:, 3:, :].transpose(1, 2)
        link_velocity_shift_w = torch.cross(
            angular_columns_w,
            link_from_com_w.unsqueeze(1).expand_as(angular_columns_w),
            dim=-1,
        ).transpose(1, 2)
        jacobian[:, :3, :] += link_velocity_shift_w
        root_rotation_inv = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
        jacobian[:, :3, :] = torch.bmm(root_rotation_inv, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(root_rotation_inv, jacobian[:, 3:, :])
        joint_pos = robot.data.joint_pos[:, self.arm_joint_ids]
        target = self.ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        max_delta = max(float(args_cli.max_ik_joint_delta), 0.0)
        if max_delta > 0.0:
            target = joint_pos + torch.clamp(target - joint_pos, -max_delta, max_delta)
        return target

    def step(self, desired_pos_w: torch.Tensor, desired_quat_w: torch.Tensor, desired_hand: torch.Tensor) -> None:
        arm_target = self._ik_joint_target(desired_pos_w, desired_quat_w)
        if not torch.all(torch.isfinite(arm_target)):
            raise FloatingPointError(f"Differential IK produced non-finite arm target: {arm_target}")
        self.step_joint_target(arm_target, desired_hand)

    def step_joint_target(self, arm_target: torch.Tensor, desired_hand: torch.Tensor) -> None:
        """Execute one planned Franka target while preserving physical hand control."""

        if arm_target.ndim == 1:
            arm_target = arm_target.unsqueeze(0).expand(self.base.num_envs, -1)
        if arm_target.shape != (self.base.num_envs, len(self.arm_joint_ids)):
            raise ValueError(
                "arm_target must have shape "
                f"({self.base.num_envs}, {len(self.arm_joint_ids)}), got {tuple(arm_target.shape)}"
            )
        if not torch.all(torch.isfinite(arm_target)):
            raise FloatingPointError(f"Planned arm target is non-finite: {arm_target}")
        self.last_arm_target = arm_target.clone()
        arm_action = _target_to_action(
            self.base, arm_target, self.arm_joint_ids, float(self.base.cfg.arm_moving_average)
        )
        hand_action = _active_hand_target_to_action(self.base, desired_hand, self.hand_joint_ids)
        self.env.step(torch.cat((arm_action, hand_action), dim=-1))
        self.stats.update(self.base)
        self.frame_step += 1
        if args_cli.video_path and self.frame_step % max(int(args_cli.video_stride), 1) == 0:
            _record_frame(self.base, self.frames)

    def execute_joint_path(
        self,
        path: np.ndarray,
        hand_target: torch.Tensor,
        *,
        time_scale: float,
    ) -> torch.Tensor:
        """Time-dilate and execute a single-environment CuRobo joint path."""

        if self.base.num_envs != 1:
            raise RuntimeError("CuRobo joint-path execution currently requires exactly one environment.")
        source = torch.as_tensor(path, dtype=self.last_arm_target.dtype, device=self.base.device).reshape(-1, 7)
        if source.shape[0] < 1:
            raise ValueError("CuRobo path is empty")
        scale = max(float(time_scale), 1.0)
        target_steps = max(int(round((source.shape[0] - 1) * scale)) + 1, source.shape[0])
        sample = torch.linspace(0.0, float(source.shape[0] - 1), target_steps, device=self.base.device)
        lower = torch.floor(sample).long()
        upper = torch.clamp(lower + 1, max=source.shape[0] - 1)
        alpha = (sample - lower.float()).unsqueeze(-1)
        resampled = source[lower] + alpha * (source[upper] - source[lower])
        for target in resampled:
            self.step_joint_target(target.unsqueeze(0), hand_target)
        return resampled[-1].unsqueeze(0)

    def hold_joint_target(
        self,
        arm_target: torch.Tensor,
        hand_target: torch.Tensor,
        steps: int,
    ) -> None:
        for _ in range(max(int(steps), 0)):
            self.step_joint_target(arm_target, hand_target)

    def move_cartesian(self, matrices_local: torch.Tensor, hand_target: torch.Tensor, steps: int) -> None:
        robot = self.base.robot
        start_pose_w = robot.data.body_state_w[:, self.ee_body_id, :7].clone()
        target_pos_w, target_quat_w = self._desired_world_pose(matrices_local)
        for step in range(max(int(steps), 1)):
            alpha = float(step + 1) / float(max(int(steps), 1))
            desired_pos_w = (1.0 - alpha) * start_pose_w[:, :3] + alpha * target_pos_w
            desired_quat_w = torch.stack(
                [
                    quat_slerp(start_pose_w[env_id, 3:7], target_quat_w[env_id], alpha)
                    for env_id in range(self.base.num_envs)
                ],
                dim=0,
            )
            self.step(desired_pos_w, desired_quat_w, hand_target)
            if (step + 1) % 25 == 0 or step + 1 == max(int(steps), 1):
                actual = robot.data.body_state_w[:, self.ee_body_id, :7]
                pos_error, rot_error = compute_pose_error(
                    actual[:, :3], actual[:, 3:7], desired_pos_w, desired_quat_w
                )
                _trace(
                    f"cartesian_step={step + 1}/{max(int(steps), 1)} "
                    f"pos_err_max={float(torch.linalg.norm(pos_error, dim=-1).max()):.5f} "
                    f"rot_err_max={float(torch.linalg.norm(rot_error, dim=-1).max()):.5f} "
                    f"actual_z={float(actual[0, 2]):.5f} desired_z={float(desired_pos_w[0, 2]):.5f} "
                    f"joint_track_err_max={float(torch.abs(self.last_arm_target - robot.data.joint_pos[:, self.arm_joint_ids]).max()):.5f}"
                )

    def hold_pose(self, matrices_local: torch.Tensor, hand_target: torch.Tensor, steps: int) -> None:
        desired_pos_w, desired_quat_w = self._desired_world_pose(matrices_local)
        for _ in range(max(int(steps), 0)):
            self.step(desired_pos_w, desired_quat_w, hand_target)

    def pose_debug(self, matrices_local: torch.Tensor) -> dict:
        target_pos_w, target_quat_w = self._desired_world_pose(matrices_local)
        actual = self.base.robot.data.body_state_w[:, self.ee_body_id, :7]
        pos_error, rot_error = compute_pose_error(
            actual[:, :3], actual[:, 3:7], target_pos_w, target_quat_w
        )
        self.base._compute_intermediate_values()
        debug = {
            "position_error_m": torch.linalg.norm(pos_error, dim=-1).detach().cpu().tolist(),
            "rotation_error_rad": torch.linalg.norm(rot_error, dim=-1).detach().cpu().tolist(),
            "actual_hand_base_pos_world": actual[:, :3].detach().cpu().tolist(),
            "target_hand_base_pos_world": target_pos_w.detach().cpu().tolist(),
            "object_pos_local": (self.base._object_pos_w - self.base.scene.env_origins).detach().cpu().tolist(),
            "object_height_delta": self.base._object_height_delta.detach().cpu().tolist(),
            "force_grasp": getattr(self.base, "_force_grasp").detach().cpu().tolist(),
            "strict_true_grasp": self.base._strict_true_grasp.detach().cpu().tolist(),
            "fingertip_force_n": self.base._object_fingertip_contact_forces.detach().cpu().tolist(),
            "fingertip_pos_local": (
                self.base._fingertip_pos_w - self.base.scene.env_origins.unsqueeze(1)
            ).detach().cpu().tolist(),
            "fingertip_surface_dist_m": self.base._surface_dist.detach().cpu().tolist(),
            "palm_pos_local": (
                self.base._palm_pos_w - self.base.scene.env_origins
            ).detach().cpu().tolist(),
            "active_hand_joint_pos": self.base.robot.data.joint_pos[:, self.hand_joint_ids].detach().cpu().tolist(),
            "active_hand_joint_target": self.base._joint_targets[:, self.hand_joint_ids].detach().cpu().tolist(),
            "arm_joint_pos": self.base.robot.data.joint_pos[:, self.arm_joint_ids].detach().cpu().tolist(),
            "arm_joint_target": self.base._joint_targets[:, self.arm_joint_ids].detach().cpu().tolist(),
            "last_ik_arm_target": self.last_arm_target.detach().cpu().tolist(),
        }
        for name in ("applied_torque", "computed_torque"):
            value = getattr(self.base.robot.data, name, None)
            if isinstance(value, torch.Tensor):
                debug[f"arm_{name}"] = value[:, self.arm_joint_ids].detach().cpu().tolist()
        clearance = getattr(self.base, "_tabletop_arm_clearance_min_margin", None)
        if isinstance(clearance, torch.Tensor):
            debug["tabletop_arm_clearance_min_margin_m"] = clearance.detach().cpu().tolist()
        clearance_penalty = getattr(self.base, "_tabletop_arm_clearance_penalty", None)
        if isinstance(clearance_penalty, torch.Tensor):
            debug["tabletop_arm_clearance_penalty"] = clearance_penalty.detach().cpu().tolist()
        return debug


def _prepare_curobo_plan(
    output_root: Path,
    result_path: Path,
    ranks: list[int],
    start_q: torch.Tensor,
) -> Path:
    """Create or validate the external DOMINO CuRobo plan used for execution."""

    if len(ranks) != 1:
        raise ValueError("--arm-execution curobo_joint_trajectory currently requires one candidate rank")
    if float(args_cli.curobo_path_time_scale) < 1.0:
        raise ValueError("--curobo-path-time-scale must be at least 1.0")
    if args_cli.curobo_plan is not None:
        plan_path = _resolve_output_path(args_cli.curobo_plan)
        if not plan_path.exists():
            raise FileNotFoundError(plan_path)
    else:
        plan_path = output_root / "curobo_plan.npz"
        command = [
            str(args_cli.curobo_python.expanduser().resolve()),
            str(args_cli.curobo_planner_script.expanduser().resolve()),
            "--anydex-result",
            str(result_path),
            "--candidate-ranks",
            str(ranks[0]),
            "--start-q",
            *[f"{float(value):.10g}" for value in start_q.detach().cpu().reshape(-1)],
            "--lift-height",
            f"{float(args_cli.lift_height):.10g}",
            "--pregrasp-distance",
            f"{float(args_cli.pregrasp_distance):.10g}",
            "--depth-base-offset",
            f"{float(args_cli.depth_base_offset):.10g}",
            "--frame-offset-xyz",
            *[f"{float(value):.10g}" for value in args_cli.frame_offset_xyz],
            "--frame-offset-rpy",
            *[f"{float(value):.10g}" for value in args_cli.frame_offset_rpy],
            "--object-pos",
            *[f"{float(value):.10g}" for value in args_cli.object_pos],
            "--object-shape",
            str(args_cli.object_shape),
            "--object-size",
            *[f"{float(value):.10g}" for value in args_cli.object_size],
            "--object-radius",
            f"{float(args_cli.object_radius):.10g}",
            "--table-center",
            f"{float(args_cli.table_pos_xy[0]):.10g}",
            f"{float(args_cli.table_pos_xy[1]):.10g}",
            f"{float(args_cli.table_top_z) - 0.5 * 0.045:.10g}",
            "--table-dims",
            "1.0",
            "0.8",
            "0.045",
            "--seed",
            str(int(args_cli.seed)),
            "--output",
            str(plan_path),
        ]
        if args_cli.grasp_type_override is not None:
            command.extend(("--grasp-type-override", str(int(args_cli.grasp_type_override))))
        if args_cli.width_override is not None:
            command.extend(("--width-override", f"{float(args_cli.width_override):.10g}"))
        _trace("curobo_command=" + json.dumps(command))
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"CuRobo planner failed with exit code {completed.returncode}")
    required_keys = [f"rank_{ranks[0]}_{phase}" for phase in ("pregrasp", "grasp", "lift")]
    with np.load(plan_path, allow_pickle=False) as plan:
        missing = [key for key in required_keys if key not in plan]
    if missing:
        raise KeyError(f"CuRobo plan {plan_path} is missing {missing}")
    return plan_path


def main() -> None:
    carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    output_root = _resolve_output_path(args_cli.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = AnyDexInspirePaths(
        anydex_root=args_cli.anydex_root.expanduser().resolve(),
        domino_root=args_cli.domino_root.expanduser().resolve(),
        python=args_cli.anydex_python.expanduser().resolve(),
    )
    result_path = _prepare_anydex_result(paths, output_root)
    all_candidates = load_anydex_candidates(
        result_path,
        width_angle_table_path=paths.width_angle_table,
        pregrasp_distance=float(args_cli.pregrasp_distance),
        depth_base_offset=float(args_cli.depth_base_offset),
        frame_offset_xyz=args_cli.frame_offset_xyz,
        frame_offset_rpy=args_cli.frame_offset_rpy,
        grasp_type_override=args_cli.grasp_type_override,
        width_override=args_cli.width_override,
    )
    ranks = [int(rank) for rank in args_cli.candidate_ranks]
    if any(rank < 0 or rank >= len(all_candidates) for rank in ranks):
        raise ValueError(f"Candidate ranks {ranks} outside available range [0, {len(all_candidates) - 1}]")
    candidates = [all_candidates[rank] for rank in ranks]
    _trace(
        "candidates="
        + json.dumps(
            [
                {
                    "rank": candidate.rank,
                    "type": candidate.grasp_type_name,
                    "score": candidate.score,
                    "width": candidate.width,
                    "active_targets": candidate.active_joint_targets.tolist(),
                }
                for candidate in candidates
            ]
        )
    )

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=len(candidates))
    _configure_env(env_cfg, len(candidates))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video_path else None)
    try:
        env.reset(seed=int(args_cli.seed))
        base = env.unwrapped
        _set_camera(base)
        if not 0 <= int(args_cli.video_env_id) < base.num_envs:
            raise ValueError(f"--video-env-id must be in [0, {base.num_envs - 1}]")
        device = base.device
        dtype = base.robot.data.joint_pos.dtype
        pregrasp_matrices = torch.tensor(
            np.stack([candidate.pregrasp_hand_base_matrix_world for candidate in candidates]),
            dtype=dtype,
            device=device,
        )
        grasp_matrices = torch.tensor(
            np.stack([candidate.grasp_hand_base_matrix_world for candidate in candidates]),
            dtype=dtype,
            device=device,
        )
        lift_matrices = grasp_matrices.clone()
        lift_matrices[:, 2, 3] += float(args_cli.lift_height)
        open_hand = torch.zeros((base.num_envs, 6), dtype=dtype, device=device)
        close_hand = torch.tensor(
            np.stack([candidate.active_joint_targets for candidate in candidates]),
            dtype=dtype,
            device=device,
        )
        if args_cli.active_hand_targets is not None:
            close_hand = torch.tensor(
                args_cli.active_hand_targets,
                dtype=dtype,
                device=device,
            ).view(1, 6).repeat(base.num_envs, 1)
        required_non_thumb_contacts = torch.tensor(
            [ANYDEX_REQUIRED_NON_THUMB_CONTACTS[candidate.grasp_type] for candidate in candidates],
            dtype=torch.long,
            device=device,
        )
        executor = AnyDexExecutor(
            env,
            grasp_matrices,
            close_hand,
            required_non_thumb_contacts,
        )
        latched_fingers = torch.zeros(
            (base.num_envs, close_hand.shape[-1]), dtype=torch.bool, device=device
        )
        latch_streaks = torch.zeros((base.num_envs, 5), dtype=torch.long, device=device)
        final_close_hand = close_hand.clone()
        if args_cli.force_limited_close:
            if float(args_cli.finger_stop_force) < 0.0:
                raise ValueError("--finger-stop-force must be non-negative")
            if int(args_cli.finger_stop_consecutive_steps) < 1:
                raise ValueError("--finger-stop-consecutive-steps must be at least one")
            final_close_hand = open_hand.clone()
        if args_cli.video_path:
            _record_frame(base, executor.frames)

        phase_debug = {}
        _trace("phase=settle")
        current = base.robot.data.body_state_w[:, executor.ee_body_id, :7]
        for _ in range(max(int(args_cli.settle_steps), 0)):
            executor.step(current[:, :3], current[:, 3:7], open_hand)

        curobo_plan_path = None
        if args_cli.arm_execution == "curobo_joint_trajectory":
            if args_cli.safe_lift_only:
                raise ValueError("--safe-lift-only is only defined for differential_ik execution")
            start_q = base.robot.data.joint_pos[0, executor.arm_joint_ids].clone()
            curobo_plan_path = _prepare_curobo_plan(output_root, result_path, ranks, start_q)
            rank = ranks[0]
            with np.load(curobo_plan_path, allow_pickle=False) as plan:
                pregrasp_path = np.asarray(plan[f"rank_{rank}_pregrasp"], dtype=np.float32)
                grasp_path = np.asarray(plan[f"rank_{rank}_grasp"], dtype=np.float32)
                lift_path = np.asarray(plan[f"rank_{rank}_lift"], dtype=np.float32)

            _trace("phase=curobo_pregrasp")
            pregrasp_q = executor.execute_joint_path(
                pregrasp_path,
                open_hand,
                time_scale=float(args_cli.curobo_path_time_scale),
            )
            executor.hold_joint_target(
                pregrasp_q, open_hand, int(args_cli.curobo_phase_settle_steps)
            )
            phase_debug["pregrasp"] = executor.pose_debug(pregrasp_matrices)

            if args_cli.stage_object_away:
                _trace("phase=restore_object")
                _place_object_at_target(base)
                executor.stats = PhysicalStats(
                    base.num_envs,
                    base.device,
                    executor.required_non_thumb_contacts,
                )
                executor.hold_joint_target(
                    pregrasp_q, open_hand, int(args_cli.object_placement_settle_steps)
                )
                phase_debug["object_restored"] = executor.pose_debug(pregrasp_matrices)

            _trace("phase=curobo_approach")
            grasp_q = executor.execute_joint_path(
                grasp_path,
                open_hand,
                time_scale=float(args_cli.curobo_path_time_scale),
            )
            executor.hold_joint_target(
                grasp_q, open_hand, int(args_cli.curobo_phase_settle_steps)
            )
            phase_debug["approach"] = executor.pose_debug(grasp_matrices)

            _trace("phase=close")
            for step in range(max(int(args_cli.close_steps), 1)):
                alpha = float(step + 1) / float(max(int(args_cli.close_steps), 1))
                proposed_close_hand = alpha * close_hand
                if args_cli.force_limited_close:
                    final_close_hand = torch.where(
                        latched_fingers,
                        final_close_hand,
                        proposed_close_hand,
                    )
                else:
                    final_close_hand = proposed_close_hand
                executor.step_joint_target(grasp_q, final_close_hand)
                if args_cli.force_limited_close:
                    force_contacts = (
                        base._object_fingertip_contact_forces >= float(args_cli.finger_stop_force)
                    )
                    latch_streaks = torch.where(
                        force_contacts,
                        latch_streaks + 1,
                        torch.zeros_like(latch_streaks),
                    )
                    closing_dims = torch.abs(close_hand - open_hand) > 1.0e-6
                    for touch_index, action_index in enumerate((1, 2, 3, 4, 5)):
                        newly_latched = (
                            (
                                latch_streaks[:, touch_index]
                                >= int(args_cli.finger_stop_consecutive_steps)
                            )
                            & closing_dims[:, action_index]
                            & ~latched_fingers[:, action_index]
                        )
                        latched_fingers[newly_latched, action_index] = True
            executor.hold_joint_target(grasp_q, final_close_hand, int(args_cli.post_close_steps))
            phase_debug["close"] = executor.pose_debug(grasp_matrices)

            _trace("phase=curobo_lift")
            lift_q = executor.execute_joint_path(
                lift_path,
                final_close_hand,
                time_scale=float(args_cli.curobo_path_time_scale),
            )
            phase_debug["lift"] = executor.pose_debug(lift_matrices)

            _trace("phase=hold")
            executor.hold_joint_target(lift_q, final_close_hand, int(args_cli.hold_steps))
            phase_debug["hold"] = executor.pose_debug(lift_matrices)
        else:
            current = base.robot.data.body_state_w[:, executor.ee_body_id, :7].clone()
            safe_lift_matrices = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(base.num_envs, 1, 1)
            safe_lift_matrices[:, :3, :3] = matrix_from_quat(current[:, 3:7])
            safe_lift_matrices[:, :3, 3] = current[:, :3] - base.scene.env_origins
            safe_lift_matrices[:, 2, 3] = torch.maximum(
                safe_lift_matrices[:, 2, 3],
                torch.full((base.num_envs,), float(args_cli.safe_waypoint_height), dtype=dtype, device=device),
            )
            safe_align_matrices = pregrasp_matrices.clone()
            safe_align_matrices[:, 2, 3] = torch.maximum(
                safe_align_matrices[:, 2, 3],
                torch.full((base.num_envs,), float(args_cli.safe_waypoint_height), dtype=dtype, device=device),
            )

            _trace("phase=safe_lift")
            executor.move_cartesian(safe_lift_matrices, open_hand, int(args_cli.safe_lift_steps))
            phase_debug["safe_lift"] = executor.pose_debug(safe_lift_matrices)

            if not args_cli.safe_lift_only:
                _trace("phase=safe_align")
                executor.move_cartesian(safe_align_matrices, open_hand, int(args_cli.safe_align_steps))
                phase_debug["safe_align"] = executor.pose_debug(safe_align_matrices)

                _trace("phase=pregrasp_descend")
                executor.move_cartesian(pregrasp_matrices, open_hand, int(args_cli.pregrasp_steps))
                phase_debug["pregrasp"] = executor.pose_debug(pregrasp_matrices)

                if args_cli.stage_object_away:
                    _trace("phase=restore_object")
                    _place_object_at_target(base)
                    executor.stats = PhysicalStats(
                        base.num_envs,
                        base.device,
                        executor.required_non_thumb_contacts,
                    )
                    executor.hold_pose(
                        pregrasp_matrices,
                        open_hand,
                        int(args_cli.object_placement_settle_steps),
                    )
                    phase_debug["object_restored"] = executor.pose_debug(pregrasp_matrices)

                _trace("phase=approach")
                executor.move_cartesian(grasp_matrices, open_hand, int(args_cli.approach_steps))
                phase_debug["approach"] = executor.pose_debug(grasp_matrices)

                _trace("phase=close")
                grasp_pos_w, grasp_quat_w = executor._desired_world_pose(grasp_matrices)
                for step in range(max(int(args_cli.close_steps), 1)):
                    alpha = float(step + 1) / float(max(int(args_cli.close_steps), 1))
                    proposed_close_hand = alpha * close_hand
                    if args_cli.force_limited_close:
                        final_close_hand = torch.where(
                            latched_fingers,
                            final_close_hand,
                            proposed_close_hand,
                        )
                    else:
                        final_close_hand = proposed_close_hand
                    executor.step(grasp_pos_w, grasp_quat_w, final_close_hand)
                    if args_cli.force_limited_close:
                        force_contacts = (
                            base._object_fingertip_contact_forces >= float(args_cli.finger_stop_force)
                        )
                        latch_streaks = torch.where(
                            force_contacts,
                            latch_streaks + 1,
                            torch.zeros_like(latch_streaks),
                        )
                        closing_dims = torch.abs(close_hand - open_hand) > 1.0e-6
                        for touch_index, action_index in enumerate((1, 2, 3, 4, 5)):
                            newly_latched = (
                                (
                                    latch_streaks[:, touch_index]
                                    >= int(args_cli.finger_stop_consecutive_steps)
                                )
                                & closing_dims[:, action_index]
                                & ~latched_fingers[:, action_index]
                            )
                            latched_fingers[newly_latched, action_index] = True
                executor.hold_pose(grasp_matrices, final_close_hand, int(args_cli.post_close_steps))
                phase_debug["close"] = executor.pose_debug(grasp_matrices)

                _trace("phase=lift")
                executor.move_cartesian(lift_matrices, final_close_hand, int(args_cli.lift_steps))
                phase_debug["lift"] = executor.pose_debug(lift_matrices)

                _trace("phase=hold")
                executor.hold_pose(lift_matrices, final_close_hand, int(args_cli.hold_steps))
                phase_debug["hold"] = executor.pose_debug(lift_matrices)

        if args_cli.video_path:
            _record_frame(base, executor.frames)
            video_path = _resolve_output_path(args_cli.video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(
                video_path,
                executor.frames,
                fps=int(args_cli.video_fps),
                macro_block_size=16,
            )
            _trace(f"video={video_path}")
        else:
            video_path = None

        base._compute_intermediate_values()
        rows = []
        for env_id, candidate in enumerate(candidates):
            anydex_success = (
                int(executor.stats.max_anydex_hold_streak[env_id])
                >= int(args_cli.strict_hold_steps)
            )
            benchmark_success = (
                int(executor.stats.max_hold_streak[env_id]) >= int(args_cli.strict_hold_steps)
            )
            rows.append(
                {
                    "env_id": env_id,
                    "candidate": candidate.as_dict(),
                    "strict_success": bool(anydex_success),
                    "benchmark_three_finger_strict_success": bool(benchmark_success),
                    "required_non_thumb_contacts": int(
                        executor.stats.required_non_thumb_contacts[env_id].detach().cpu()
                    ),
                    "max_anydex_strict_hold_streak": int(
                        executor.stats.max_anydex_hold_streak[env_id].detach().cpu()
                    ),
                    "max_benchmark_strict_hold_streak": int(
                        executor.stats.max_hold_streak[env_id].detach().cpu()
                    ),
                    "max_object_height_delta": float(executor.stats.max_height_delta[env_id].detach().cpu()),
                    "max_force_sum_n": float(executor.stats.max_force_sum[env_id].detach().cpu()),
                    "ever_anydex_force_grasp": bool(
                        executor.stats.max_anydex_force_grasp[env_id].detach().cpu()
                    ),
                    "ever_force_grasp": bool(executor.stats.max_force_grasp[env_id].detach().cpu()),
                    "ever_strict_true_grasp": bool(executor.stats.max_strict_grasp[env_id].detach().cpu()),
                    "final_object_pos_local": (
                        base._object_pos_w[env_id] - base.scene.env_origins[env_id]
                    ).detach().cpu().tolist(),
                    "final_active_hand_joint_pos": base.robot.data.joint_pos[
                        env_id, executor.hand_joint_ids
                    ].detach().cpu().tolist(),
                }
            )
        summary = {
            "task": args_cli.task,
            "seed": int(args_cli.seed),
            "anydex_result": str(result_path),
            "candidate_ranks": ranks,
            "object_pos": [float(value) for value in args_cli.object_pos],
            "object_shape": str(args_cli.object_shape),
            "object_size": [float(value) for value in args_cli.object_size],
            "object_radius": float(args_cli.object_radius),
            "object_mass": float(args_cli.object_mass),
            "table_pos_xy": [float(value) for value in args_cli.table_pos_xy],
            "table_top_z": float(args_cli.table_top_z),
            "stage_object_away": bool(args_cli.stage_object_away),
            "object_staging_offset": [float(value) for value in args_cli.object_staging_offset],
            "object_placement_settle_steps": int(args_cli.object_placement_settle_steps),
            "initial_arm_pos": (
                [float(value) for value in args_cli.initial_arm_pos]
                if args_cli.initial_arm_pos is not None
                else None
            ),
            "lift_height": float(args_cli.lift_height),
            "arm_execution": str(args_cli.arm_execution),
            "curobo_plan": str(curobo_plan_path) if curobo_plan_path is not None else None,
            "curobo_path_time_scale": float(args_cli.curobo_path_time_scale),
            "curobo_phase_settle_steps": int(args_cli.curobo_phase_settle_steps),
            "ik_command_type": str(args_cli.ik_command_type),
            "max_ik_joint_delta": float(args_cli.max_ik_joint_delta),
            "grasp_type_override": args_cli.grasp_type_override,
            "width_override": args_cli.width_override,
            "frame_offset_xyz": [float(value) for value in args_cli.frame_offset_xyz],
            "frame_offset_rpy": [float(value) for value in args_cli.frame_offset_rpy],
            "active_hand_targets": (
                [float(value) for value in args_cli.active_hand_targets]
                if args_cli.active_hand_targets is not None
                else None
            ),
            "force_limited_close": bool(args_cli.force_limited_close),
            "finger_stop_force_n": float(args_cli.finger_stop_force),
            "finger_stop_consecutive_steps": int(args_cli.finger_stop_consecutive_steps),
            "latched_fingers": latched_fingers.detach().cpu().tolist(),
            "latch_streaks": latch_streaks.detach().cpu().tolist(),
            "final_close_hand_targets": final_close_hand.detach().cpu().tolist(),
            "safe_lift_only": bool(args_cli.safe_lift_only),
            "strict_definition": {
                "contract": "anydex_grasp_type_physical_contact",
                "thumb_force_contact": True,
                "required_non_thumb_force_contacts_by_grasp_type": {
                    str(key): int(value)
                    for key, value in ANYDEX_REQUIRED_NON_THUMB_CONTACTS.items()
                },
                "force_contact_threshold_n": float(base.cfg.object_contact_force_threshold),
                "object_height_delta_m": 0.045,
                "stable_object_palm_velocity": float(base.cfg.stable_object_palm_vel),
                "consecutive_hold_steps": int(args_cli.strict_hold_steps),
            },
            "benchmark_three_finger_definition_unchanged": {
                "force_grasp": "thumb plus at least two non-thumb force contacts",
                "strict_true_grasp": True,
                "consecutive_hold_steps": int(args_cli.strict_hold_steps),
            },
            "self_collision_enabled": bool(base.cfg.robot_cfg.spawn.self_collision),
            "object_collision_disabled_during_approach": False,
            "video_path": str(video_path) if video_path is not None else None,
            "phase_debug": phase_debug,
            "results": rows,
            "strict_success_count": sum(int(row["strict_success"]) for row in rows),
        }
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _trace(
            json.dumps(
                {
                    "strict_success_count": summary["strict_success_count"],
                    "results": [
                        {
                            key: row[key]
                            for key in (
                                "env_id",
                                "strict_success",
                                "benchmark_three_finger_strict_success",
                                "max_anydex_strict_hold_streak",
                                "max_benchmark_strict_hold_streak",
                                "max_object_height_delta",
                                "max_force_sum_n",
                                "ever_anydex_force_grasp",
                                "ever_force_grasp",
                            )
                        }
                        for row in rows
                    ],
                    "summary": str(summary_path),
                },
                indent=2,
            )
        )
    except BaseException as exc:
        failure_path = output_root / "failure.json"
        failure_path.write_text(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _trace(f"failure={failure_path} {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
