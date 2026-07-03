#!/usr/bin/env python3
"""Collect a small teacher-student dataset from an IsaacLab dynamic teacher env."""

from __future__ import annotations

import argparse
import copy
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToolReal-Revo2-Franka-FallingBaton-Teacher-Direct-v0")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=256)
parser.add_argument("--history", type=int, default=4)
parser.add_argument("--object-points", type=int, default=128)
parser.add_argument(
    "--pointcloud-source",
    choices=("clean", "rgbd_projected_mask"),
    default="clean",
    help="clean uses simulator object geometry; rgbd_projected_mask uses a student camera depth image plus an oracle object mask.",
)
parser.add_argument("--rgbd-width", type=int, default=160)
parser.add_argument("--rgbd-height", type=int, default=120)
parser.add_argument("--rgbd-mask-points", type=int, default=768)
parser.add_argument("--rgbd-mask-dilation", type=int, default=1)
parser.add_argument("--rgbd-depth-tolerance", type=float, default=0.045)
parser.add_argument("--rgbd-min-valid-points", type=int, default=16)
parser.add_argument("--rgbd-clean-fallback", action="store_true", default=True)
parser.add_argument("--no-rgbd-clean-fallback", action="store_false", dest="rgbd_clean_fallback")
parser.add_argument("--store-rgbd-frames", action="store_true")
parser.add_argument(
    "--student-camera-track-object",
    action="store_true",
    default=False,
    help="Aim the student RGB-D camera at the live object. By default the camera is fixed in a stable third view.",
)
parser.add_argument("--student-camera-track-offset", type=float, nargs=3, default=(0.65, 0.95, 0.65))
parser.add_argument("--student-camera-track-target-offset", type=float, nargs=3, default=(0.0, 0.0, 0.05))
parser.add_argument("--student-camera-focal-length", type=float, default=24.0)
parser.add_argument("--action-source", choices=("teacher", "student_dagger", "random", "zeros"), default="teacher")
parser.add_argument("--checkpoint", default="", help="rl_games teacher checkpoint when --action-source=teacher.")
parser.add_argument(
    "--student-checkpoint",
    default="",
    help="PointTemporalStudent checkpoint used to execute actions when --action-source=student_dagger.",
)
parser.add_argument("--student-device", default=None, help="Torch device for student policy execution. Defaults to --device.")
parser.add_argument("--student-action-clamp", type=float, default=1.0)
parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic teacher actions.")
parser.add_argument("--stochastic", action="store_false", dest="deterministic", help="Sample teacher actions.")
parser.add_argument("--rl-device", default=None, help="Device for teacher policy inference. Defaults to --device.")
parser.add_argument(
    "--sample-timing",
    choices=("pre_action", "post_step"),
    default="pre_action",
    help=(
        "pre_action stores the current observation history with the action chosen from that observation. "
        "post_step preserves the older one-step-lag export behavior."
    ),
)
parser.add_argument(
    "--hold-label-source",
    choices=("true_grasp_rel_vel", "grasp_seen_rel_vel", "success", "none"),
    default="true_grasp_rel_vel",
    help="Label samples used to supervise the student's deployable hold/reflex hand target head.",
)
parser.add_argument(
    "--hold-label-rel-vel-threshold",
    type=float,
    default=0.2,
    help="object-palm relative velocity threshold for *_rel_vel hold labels.",
)
parser.add_argument(
    "--dynamic-curriculum-alpha",
    type=float,
    default=None,
    help="Override dynamic speed curriculum alpha when the task cfg exposes it. Use 1.0 for full-speed collection.",
)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.pointcloud_source == "rgbd_projected_mask":
    setattr(args_cli, "enable_cameras", True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper  # noqa: E402
from rl_games.algos_torch.players import PpoPlayerContinuous  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402
from simtoolreal_lab.teacher_student import (  # noqa: E402
    PointTemporalStudent,
    default_dataset_spec,
    masked_rgbd_object_points_in_palm_frame,
    object_points_in_palm_frame,
    object_points_in_world_frame,
    quat_rotate_inverse,
    rigid_object_point_flow_in_palm_frame,
    sample_box_surface_points,
    validate_student_batch,
)


def _trace(message: str) -> None:
    print(f"[COLLECT] {message}", flush=True)


def _policy_tensor_from_wrapped_obs(obs) -> torch.Tensor:
    if isinstance(obs, dict):
        if "obs" in obs:
            return obs["obs"]
        if "policy" in obs:
            return obs["policy"]
    return obs


def _affordance_kwargs(cfg) -> dict:
    return {
        "affordance_mode": str(getattr(cfg, "affordance_label_mode", "side_grasp")),
        "affordance_positive_fraction": float(getattr(cfg, "affordance_positive_fraction", 0.38)),
        "affordance_negative_fraction": float(getattr(cfg, "affordance_negative_fraction", 0.45)),
        "affordance_positive_end": str(getattr(cfg, "affordance_positive_end", "negative")),
    }


def _set_student_camera_poses(unwrapped_env) -> bool:
    camera = getattr(unwrapped_env, "_student_camera", None)
    if camera is None:
        return False
    origins = unwrapped_env.scene.env_origins
    target_offset = torch.tensor(
        args_cli.student_camera_track_target_offset, device=origins.device, dtype=origins.dtype
    ).unsqueeze(0)
    camera_offset = torch.tensor(
        args_cli.student_camera_track_offset, device=origins.device, dtype=origins.dtype
    ).unsqueeze(0)

    object_root_pos = None
    if bool(args_cli.student_camera_track_object):
        object_asset = getattr(unwrapped_env, "object", None)
        object_data = getattr(object_asset, "data", None)
        object_root_pos = getattr(object_data, "root_pos_w", None)
    if object_root_pos is not None:
        targets = object_root_pos[:, :3] + target_offset
    else:
        object_init_pos = torch.tensor(
            unwrapped_env.cfg.object_cfg.init_state.pos, device=origins.device, dtype=origins.dtype
        ).unsqueeze(0)
        targets = origins + object_init_pos + target_offset
    camera.set_world_poses_from_view(targets + camera_offset, targets)
    if hasattr(camera, "_update_poses") and hasattr(camera, "_ALL_INDICES"):
        camera._update_poses(camera._ALL_INDICES)
    return True


def _student_camera_rgbd(unwrapped_env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    camera = getattr(unwrapped_env, "_student_camera", None)
    if camera is None:
        raise RuntimeError("pointcloud-source=rgbd_projected_mask requires cfg.student_camera_enabled=True")
    data = camera.data
    rgb = data.output.get("rgb")
    depth = data.output.get("distance_to_image_plane")
    if depth is None:
        depth = data.output.get("depth")
    if rgb is None or depth is None:
        raise RuntimeError(f"student camera output missing rgb/depth; available keys={list(data.output.keys())}")
    return rgb, depth, data.pos_w, data.quat_w_ros, data.intrinsic_matrices


def _make_teacher_player(task: str, wrapped_env: RlGamesVecEnvWrapper, checkpoint: Path) -> PpoPlayerContinuous:
    agent_cfg = copy.deepcopy(load_cfg_from_registry(task, "rl_games_cfg_entry_point"))
    params = agent_cfg["params"]
    config = params["config"]
    rl_device = args_cli.rl_device or args_cli.device
    config["device"] = rl_device
    config["device_name"] = rl_device
    config["num_actors"] = wrapped_env.num_envs
    config["env_info"] = wrapped_env.get_env_info()
    config["vec_env"] = wrapped_env
    config.setdefault("player", {})
    config["player"]["deterministic"] = bool(args_cli.deterministic)
    config["player"]["print_stats"] = False

    _trace(f"building rl_games teacher player device={rl_device} actors={wrapped_env.num_envs}")
    player = PpoPlayerContinuous(params)
    player.has_batch_dimension = True
    player.batch_size = wrapped_env.num_envs
    _trace(f"restoring teacher checkpoint: {checkpoint}")
    player.restore(str(checkpoint))
    player.reset()
    _trace("teacher checkpoint restored")
    return player


def _load_student_policy(
    checkpoint: Path,
    device: torch.device,
    *,
    expected_history: int,
    expected_points: int,
    expected_proprio_dim: int,
    expected_action_dim: int,
) -> PointTemporalStudent:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"Student checkpoint {checkpoint} is not a PointTemporalStudent checkpoint.")
    metadata = dict(payload.get("metadata", {}))
    spec = dict(payload.get("spec", {}))
    history = int(spec.get("history", metadata.get("history", expected_history)))
    num_points = int(spec.get("num_object_points", metadata.get("object_points", expected_points)))
    proprio_dim = int(spec.get("proprio_dim", metadata.get("proprio_dim", expected_proprio_dim)))
    action_dim = int(spec.get("action_dim", metadata.get("action_dim", expected_action_dim)))
    privileged_dim = int(spec.get("compact_privileged_dim", metadata.get("compact_privileged_dim", 32)))
    mismatches = []
    if history != expected_history:
        mismatches.append(f"history checkpoint={history} args={expected_history}")
    if num_points != expected_points:
        mismatches.append(f"num_points checkpoint={num_points} args={expected_points}")
    if proprio_dim != expected_proprio_dim:
        mismatches.append(f"proprio_dim checkpoint={proprio_dim} env={expected_proprio_dim}")
    if action_dim != expected_action_dim:
        mismatches.append(f"action_dim checkpoint={action_dim} env={expected_action_dim}")
    if mismatches:
        raise RuntimeError("Student checkpoint shape mismatch: " + "; ".join(mismatches))
    model = PointTemporalStudent(
        history=history,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        privileged_dim=privileged_dim,
    ).to(device)
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    allowed_missing_prefixes = ("hold_head.", "hold_gate_head.")
    bad_missing = [name for name in missing if not name.startswith(allowed_missing_prefixes)]
    if bad_missing or unexpected:
        raise RuntimeError(
            "Student checkpoint state mismatch: "
            f"missing={bad_missing}, unexpected={list(unexpected)}"
        )
    if missing:
        _trace(f"student checkpoint missing optional hold head weights; initialized {list(missing)}")
    model.eval()
    return model


def _wrap_for_teacher(task: str, env) -> RlGamesVecEnvWrapper:
    agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    env_block = agent_cfg["params"]["env"]
    return RlGamesVecEnvWrapper(
        env,
        args_cli.rl_device or args_cli.device,
        float(env_block.get("clip_observations", 100.0)),
        float(env_block.get("clip_actions", 1.0)),
    )


def _compact_privileged(unwrapped_env, success_env: torch.Tensor) -> torch.Tensor:
    rel_obj_palm = quat_rotate_inverse(
        unwrapped_env._palm_quat_w,
        unwrapped_env._object_pos_w - unwrapped_env._palm_pos_w,
    )
    rel_vel_palm = quat_rotate_inverse(
        unwrapped_env._palm_quat_w,
        unwrapped_env._object_lin_vel_w - unwrapped_env._palm_lin_vel_w,
    )
    ang_vel_palm = quat_rotate_inverse(unwrapped_env._palm_quat_w, unwrapped_env._object_ang_vel_w)
    values = [
        rel_obj_palm,
        rel_vel_palm,
        ang_vel_palm,
        unwrapped_env._surface_dist,
        unwrapped_env._object_palm_rel_vel.unsqueeze(-1),
        unwrapped_env._true_grasp.float().unsqueeze(-1),
        unwrapped_env._grasp_seen.float().unsqueeze(-1),
        success_env.float().unsqueeze(-1),
        unwrapped_env.episode_length_buf.float().unsqueeze(-1) / float(unwrapped_env.max_episode_length),
    ]
    compact = torch.cat(values, dim=-1)
    if compact.shape[-1] < 32:
        compact = torch.cat(
            [compact, torch.zeros((compact.shape[0], 32 - compact.shape[-1]), device=compact.device)],
            dim=-1,
        )
    return compact[:, :32]


def _hold_mask_from_compact(compact: torch.Tensor) -> torch.Tensor:
    source = str(args_cli.hold_label_source)
    if source == "none":
        return torch.zeros(compact.shape[0], dtype=compact.dtype, device=compact.device)
    if source == "success":
        if compact.shape[-1] <= 17:
            return torch.zeros(compact.shape[0], dtype=compact.dtype, device=compact.device)
        return (compact[:, 17] > 0.5).to(dtype=compact.dtype)

    if compact.shape[-1] <= 16:
        return torch.zeros(compact.shape[0], dtype=compact.dtype, device=compact.device)
    rel_vel = compact[:, 14]
    if source == "true_grasp_rel_vel":
        grasp_label = compact[:, 15] > 0.5
    else:
        grasp_label = compact[:, 16] > 0.5
    stable_label = rel_vel < float(args_cli.hold_label_rel_vel_threshold)
    return (grasp_label & stable_label).to(dtype=compact.dtype)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve() if args_cli.checkpoint else None
    if args_cli.action_source in {"teacher", "student_dagger"} and (checkpoint is None or not checkpoint.exists()):
        raise FileNotFoundError("--checkpoint is required when --action-source=teacher/student_dagger")
    student_checkpoint = (
        Path(args_cli.student_checkpoint).expanduser().resolve() if args_cli.student_checkpoint else None
    )
    if args_cli.action_source == "student_dagger" and (
        student_checkpoint is None or not student_checkpoint.exists()
    ):
        raise FileNotFoundError("--student-checkpoint is required when --action-source=student_dagger")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.dynamic_curriculum_alpha is not None and hasattr(
        env_cfg, "dynamic_grasp_speed_curriculum_override_alpha"
    ):
        env_cfg.dynamic_grasp_speed_curriculum_override_alpha = float(args_cli.dynamic_curriculum_alpha)
    if args_cli.pointcloud_source == "rgbd_projected_mask":
        if not hasattr(env_cfg, "student_camera_enabled"):
            raise RuntimeError(f"Task {args_cli.task} does not expose student_camera_enabled.")
        env_cfg.student_camera_enabled = True
        env_cfg.student_camera.data_types = ["rgb", "distance_to_image_plane"]
        env_cfg.student_camera.width = int(args_cli.rgbd_width)
        env_cfg.student_camera.height = int(args_cli.rgbd_height)
        env_cfg.student_camera.spawn.focal_length = float(args_cli.student_camera_focal_length)
        env_cfg.student_camera.return_latest_camera_pose = True

    _trace(f"making env task={args_cli.task} num_envs={args_cli.num_envs} device={args_cli.device}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    wrapped_env = None
    teacher_player = None
    if args_cli.action_source in {"teacher", "student_dagger"}:
        _trace("wrapping env for rl_games teacher")
        wrapped_env = _wrap_for_teacher(args_cli.task, env)
        assert checkpoint is not None
        teacher_player = _make_teacher_player(args_cli.task, wrapped_env, checkpoint)
        _trace("resetting wrapped teacher env")
        teacher_obs = wrapped_env.reset()
        policy_obs = _policy_tensor_from_wrapped_obs(teacher_obs)
        _trace(f"teacher env reset complete obs_shape={tuple(policy_obs.shape)}")
    else:
        _trace(f"resetting raw env action_source={args_cli.action_source}")
        obs, _ = env.reset()
        teacher_obs = None
        policy_obs = obs["policy"]
        _trace(f"raw env reset complete obs_shape={tuple(policy_obs.shape)}")
    if args_cli.pointcloud_source == "rgbd_projected_mask":
        _set_student_camera_poses(unwrapped)
    obs_dim = int(policy_obs.shape[-1])
    student_policy = None
    student_device = torch.device(args_cli.student_device or args_cli.device)
    if args_cli.action_source == "student_dagger":
        assert student_checkpoint is not None
        _trace(f"loading student execution policy: {student_checkpoint}")
        student_policy = _load_student_policy(
            student_checkpoint,
            student_device,
            expected_history=int(args_cli.history),
            expected_points=int(args_cli.object_points),
            expected_proprio_dim=obs_dim,
            expected_action_dim=int(unwrapped.cfg.action_space),
        )
        _trace(f"student execution policy loaded device={student_device}")

    affordance_kwargs = _affordance_kwargs(unwrapped.cfg)
    local_points, affordance_labels = sample_box_surface_points(
        tuple(float(v) for v in unwrapped.cfg.object_size),
        args_cli.object_points,
        unwrapped.device,
        **affordance_kwargs,
    )
    rgbd_mask_local_points = None
    if args_cli.pointcloud_source == "rgbd_projected_mask":
        rgbd_mask_local_points, _ = sample_box_surface_points(
            tuple(float(v) for v in unwrapped.cfg.object_size),
            max(int(args_cli.rgbd_mask_points), int(args_cli.object_points)),
            unwrapped.device,
            **affordance_kwargs,
        )
    point_hist = torch.zeros(
        (unwrapped.num_envs, args_cli.history, args_cli.object_points, 3),
        device=unwrapped.device,
    )
    valid_hist = torch.zeros((unwrapped.num_envs, args_cli.history, args_cli.object_points), device=unwrapped.device)
    proprio_hist = torch.zeros((unwrapped.num_envs, args_cli.history, obs_dim), device=unwrapped.device)
    prev_points = None

    samples: dict[str, list[torch.Tensor]] = {
        "pointcloud_seq": [],
        "pointcloud_valid_seq": [],
        "proprio_seq": [],
        "target": [],
        "point_flow_delta": [],
        "point_flow_velocity": [],
        "affordance_region_labels": [],
        "compact_privileged": [],
        "hold_target": [],
        "hold_mask": [],
        "phase": [],
        "episode_success": [],
    }
    if args_cli.action_source == "student_dagger":
        samples["executed_action"] = []
    if args_cli.store_rgbd_frames and args_cli.pointcloud_source == "rgbd_projected_mask":
        samples["camera_rgb"] = []
        samples["camera_depth"] = []
        samples["camera_object_mask"] = []
    rgbd_fallback_env_frames = 0
    rgbd_valid_point_sum = 0.0
    rgbd_valid_frame_count = 0
    last_frame_ready = False

    def append_current_sample(
        action_targets: torch.Tensor,
        success_env: torch.Tensor,
        phase_step: int,
        executed_actions: torch.Tensor | None = None,
    ) -> None:
        samples["pointcloud_seq"].append(point_hist.detach().cpu())
        samples["pointcloud_valid_seq"].append(valid_hist.detach().cpu())
        samples["proprio_seq"].append(proprio_hist.detach().cpu())
        samples["target"].append(action_targets.detach().cpu())
        if "executed_action" in samples:
            action_value = executed_actions if executed_actions is not None else action_targets
            samples["executed_action"].append(action_value.detach().cpu())
        samples["point_flow_delta"].append(flow_delta.detach().cpu())
        samples["point_flow_velocity"].append(flow_velocity.detach().cpu())
        samples["affordance_region_labels"].append(current_affordance_labels.detach().cpu())
        compact = _compact_privileged(unwrapped, success_env)
        samples["compact_privileged"].append(compact.detach().cpu())
        samples["hold_target"].append(action_targets[:, 7:].detach().cpu())
        samples["hold_mask"].append(_hold_mask_from_compact(compact).detach().cpu())
        phase_id = min(4, int(5 * phase_step / max(args_cli.steps, 1)))
        samples["phase"].append(torch.full((unwrapped.num_envs,), phase_id, dtype=torch.long))
        samples["episode_success"].append(success_env.detach().cpu().float())
        if args_cli.store_rgbd_frames and args_cli.pointcloud_source == "rgbd_projected_mask":
            assert rgbd_rgb is not None and rgbd_depth is not None and rgbd_mask is not None
            samples["camera_rgb"].append(rgbd_rgb.detach().cpu())
            samples["camera_depth"].append(rgbd_depth.detach().cpu().to(torch.float16))
            samples["camera_object_mask"].append(rgbd_mask.detach().cpu())

    for step in range(args_cli.steps):
        if args_cli.action_source == "zeros":
            actions = torch.zeros((unwrapped.num_envs, unwrapped.cfg.action_space), device=unwrapped.device)
            if args_cli.sample_timing == "pre_action" and last_frame_ready and step >= args_cli.history:
                success_env = unwrapped.extras.get(
                    "success_env",
                    torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device),
                )
                append_current_sample(actions, success_env, step)
            obs, _rew, terminated, truncated, extras = env.step(actions)
            policy_obs = obs["policy"]
        elif args_cli.action_source == "teacher":
            assert wrapped_env is not None and teacher_player is not None and teacher_obs is not None
            with torch.inference_mode():
                actions = teacher_player.get_action(teacher_obs, is_deterministic=args_cli.deterministic)
            if args_cli.sample_timing == "pre_action" and last_frame_ready and step >= args_cli.history:
                success_env = unwrapped.extras.get(
                    "success_env",
                    torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device),
                )
                append_current_sample(actions, success_env, step)
            teacher_obs, _rew, dones, extras = wrapped_env.step(actions)
            terminated = dones
            truncated = torch.zeros_like(dones)
            policy_obs = _policy_tensor_from_wrapped_obs(teacher_obs)
        elif args_cli.action_source == "student_dagger":
            assert wrapped_env is not None and teacher_player is not None and teacher_obs is not None
            assert student_policy is not None
            with torch.inference_mode():
                teacher_actions = teacher_player.get_action(teacher_obs, is_deterministic=args_cli.deterministic)
                if last_frame_ready and step >= args_cli.history:
                    student_actions = student_policy(
                        point_hist.to(student_device),
                        valid_hist.to(student_device),
                        proprio_hist.to(student_device),
                    )["action"].to(unwrapped.device)
                    if args_cli.student_action_clamp > 0.0:
                        student_actions = torch.clamp(
                            student_actions,
                            -float(args_cli.student_action_clamp),
                            float(args_cli.student_action_clamp),
                        )
                    actions = student_actions
                else:
                    actions = teacher_actions
            if args_cli.sample_timing == "pre_action" and last_frame_ready and step >= args_cli.history:
                success_env = unwrapped.extras.get(
                    "success_env",
                    torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device),
                )
                append_current_sample(teacher_actions, success_env, step, executed_actions=actions)
            teacher_obs, _rew, dones, extras = wrapped_env.step(actions)
            terminated = dones
            truncated = torch.zeros_like(dones)
            policy_obs = _policy_tensor_from_wrapped_obs(teacher_obs)
        else:
            actions = 2.0 * torch.rand((unwrapped.num_envs, unwrapped.cfg.action_space), device=unwrapped.device) - 1.0
            if args_cli.sample_timing == "pre_action" and last_frame_ready and step >= args_cli.history:
                success_env = unwrapped.extras.get(
                    "success_env",
                    torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device),
                )
                append_current_sample(actions, success_env, step)
            obs, _rew, terminated, truncated, extras = env.step(actions)
            policy_obs = obs["policy"]

        unwrapped._compute_intermediate_values()
        clean_points_w = object_points_in_world_frame(
            local_points,
            unwrapped._object_pos_w,
            unwrapped._object_quat_w,
        )
        clean_points = object_points_in_palm_frame(
            local_points,
            unwrapped._object_pos_w,
            unwrapped._object_quat_w,
            unwrapped._palm_pos_w,
            unwrapped._palm_quat_w,
        )
        clean_labels = affordance_labels.unsqueeze(0).expand(unwrapped.num_envs, -1)
        rgbd_rgb = None
        rgbd_depth = None
        rgbd_mask = None
        if args_cli.pointcloud_source == "rgbd_projected_mask":
            assert rgbd_mask_local_points is not None
            _set_student_camera_poses(unwrapped)
            rgbd_rgb, rgbd_depth, camera_pos_w, camera_quat_w_ros, camera_intrinsics = _student_camera_rgbd(unwrapped)
            rgbd_points = masked_rgbd_object_points_in_palm_frame(
                rgbd_mask_local_points,
                tuple(float(v) for v in unwrapped.cfg.object_size),
                rgbd_depth,
                camera_pos_w,
                camera_quat_w_ros,
                camera_intrinsics,
                unwrapped._object_pos_w,
                unwrapped._object_quat_w,
                unwrapped._palm_pos_w,
                unwrapped._palm_quat_w,
                num_points=args_cli.object_points,
                mask_dilation=args_cli.rgbd_mask_dilation,
                depth_tolerance=args_cli.rgbd_depth_tolerance,
                **affordance_kwargs,
            )
            current_points = rgbd_points["points_palm"]
            current_points_w = rgbd_points["points_w"]
            valid = rgbd_points["valid"]
            current_affordance_labels = rgbd_points["affordance_labels"]
            rgbd_mask = rgbd_points["object_mask"]
            valid_counts = torch.sum(valid > 0.0, dim=-1)
            rgbd_valid_point_sum += float(valid_counts.detach().sum().cpu())
            rgbd_valid_frame_count += int(valid_counts.numel())
            low_valid = valid_counts < int(args_cli.rgbd_min_valid_points)
            if bool(low_valid.any()) and bool(args_cli.rgbd_clean_fallback):
                rgbd_fallback_env_frames += int(low_valid.sum().detach().cpu())
                current_points[low_valid] = clean_points[low_valid]
                current_points_w[low_valid] = clean_points_w[low_valid]
                valid[low_valid] = 1.0
                current_affordance_labels[low_valid] = clean_labels[low_valid]
            flow_velocity = rigid_object_point_flow_in_palm_frame(
                current_points_w,
                unwrapped._object_pos_w,
                unwrapped._object_lin_vel_w,
                unwrapped._object_ang_vel_w,
                unwrapped._palm_quat_w,
                unwrapped._palm_lin_vel_w,
            )
            flow_velocity = flow_velocity * valid.unsqueeze(-1)
            flow_delta = flow_velocity * max(float(unwrapped.dt), 1.0e-6)
        else:
            current_points = clean_points
            valid = torch.ones((unwrapped.num_envs, args_cli.object_points), device=unwrapped.device)
            current_affordance_labels = clean_labels
            flow_delta = torch.zeros_like(current_points) if prev_points is None else current_points - prev_points
            flow_velocity = flow_delta / max(float(unwrapped.dt), 1.0e-6)
            prev_points = current_points.detach()

        done = terminated | truncated
        if bool(done.any()):
            flow_delta[done] = 0.0
            flow_velocity[done] = 0.0

        point_hist = torch.roll(point_hist, shifts=-1, dims=1)
        valid_hist = torch.roll(valid_hist, shifts=-1, dims=1)
        proprio_hist = torch.roll(proprio_hist, shifts=-1, dims=1)
        if bool(done.any()):
            point_hist[done] = 0.0
            valid_hist[done] = 0.0
            proprio_hist[done] = 0.0
        point_hist[:, -1] = current_points
        valid_hist[:, -1] = valid
        proprio_hist[:, -1] = policy_obs
        last_frame_ready = True

        if args_cli.sample_timing == "post_step" and step >= args_cli.history - 1:
            success_env = extras.get("success_env", torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device))
            append_current_sample(actions, success_env, step)

        if step == 0 or step == args_cli.steps - 1 or (step + 1) % 16 == 0:
            valid_msg = ""
            if args_cli.pointcloud_source == "rgbd_projected_mask" and rgbd_valid_frame_count > 0:
                valid_mean = rgbd_valid_point_sum / max(rgbd_valid_frame_count, 1)
                valid_msg = f" rgbd_valid_mean={valid_mean:.1f} rgbd_fallback={rgbd_fallback_env_frames}"
            _trace(
                f"step={step:04d} done={int(done.sum())} "
                f"samples={sum(item.shape[0] for item in samples['target'])}{valid_msg}"
            )

    payload = {key: torch.cat(value, dim=0) for key, value in samples.items()}
    payload["metadata"] = {
        "task": args_cli.task,
        "action_source": args_cli.action_source,
        "history": args_cli.history,
        "object_points": args_cli.object_points,
        "proprio_dim": obs_dim,
        "action_dim": int(unwrapped.cfg.action_space),
        "compact_privileged_dim": 32,
        "task_family": str(unwrapped.cfg.task_family),
        "hand_embodiment": str(getattr(unwrapped.cfg, "hand_embodiment", "revo2")),
        "action_contract": str(unwrapped.cfg.action_contract),
        "checkpoint": str(checkpoint) if checkpoint is not None else "",
        "deterministic_teacher": (
            bool(args_cli.deterministic) if args_cli.action_source in {"teacher", "student_dagger"} else None
        ),
        "student_checkpoint": str(student_checkpoint) if student_checkpoint is not None else "",
        "student_action_clamp": float(args_cli.student_action_clamp) if args_cli.action_source == "student_dagger" else None,
        "sample_timing": args_cli.sample_timing,
        "hold_label_source": args_cli.hold_label_source,
        "hold_label_rel_vel_threshold": float(args_cli.hold_label_rel_vel_threshold),
        "dynamic_curriculum_alpha": args_cli.dynamic_curriculum_alpha,
        "source": f"isaaclab_{args_cli.pointcloud_source}_teacher_export_v1",
        "pointcloud_source": args_cli.pointcloud_source,
        "rgbd_camera": {
            "enabled": args_cli.pointcloud_source == "rgbd_projected_mask",
            "width": int(args_cli.rgbd_width),
            "height": int(args_cli.rgbd_height),
            "focal_length": float(args_cli.student_camera_focal_length),
            "track_object": bool(args_cli.student_camera_track_object),
            "track_offset": list(args_cli.student_camera_track_offset),
            "track_target_offset": list(args_cli.student_camera_track_target_offset),
            "mask_points": int(args_cli.rgbd_mask_points),
            "mask_dilation": int(args_cli.rgbd_mask_dilation),
            "depth_tolerance": float(args_cli.rgbd_depth_tolerance),
            "clean_fallback_enabled": bool(args_cli.rgbd_clean_fallback),
            "fallback_env_frames": int(rgbd_fallback_env_frames),
            "valid_points_mean_before_fallback": (
                float(rgbd_valid_point_sum / max(rgbd_valid_frame_count, 1))
                if rgbd_valid_frame_count > 0
                else None
            ),
        },
        "stored_rgbd_frames": bool(args_cli.store_rgbd_frames and args_cli.pointcloud_source == "rgbd_projected_mask"),
    }

    spec = default_dataset_spec(
        task_family=unwrapped.cfg.task_family,
        hand=getattr(unwrapped.cfg, "hand_embodiment", "revo2"),
        action_contract=unwrapped.cfg.action_contract,
        history=args_cli.history,
        num_object_points=args_cli.object_points,
        proprio_dim=obs_dim,
        compact_privileged_dim=32,
    )
    errors = validate_student_batch(payload, spec)
    if errors:
        raise RuntimeError("Exported dataset failed schema validation: " + "; ".join(errors))

    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args_cli.out)
    _trace(f"saved {payload['target'].shape[0]} samples to {args_cli.out}")
    if wrapped_env is not None:
        wrapped_env.close()
    else:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
