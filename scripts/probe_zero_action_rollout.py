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
parser.add_argument("--steps", type=int, default=480)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--output-json", required=True)
parser.add_argument("--video-path", default=None)
parser.add_argument("--video-stride", type=int, default=2)
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--video-camera-eye", type=float, nargs=3, default=(1.35, -1.10, 0.84))
parser.add_argument("--video-camera-target", type=float, nargs=3, default=(0.58, 0.00, 0.38))
parser.add_argument("--video-camera-focal-length", type=float, default=22.0)
parser.add_argument("--video-camera-resolution", type=int, nargs=2, default=(1280, 720))
parser.add_argument("--debug-interval", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video_path and hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


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


def _geometry_debug(env) -> dict:
    base = env.unwrapped
    base._compute_intermediate_values()
    origin = base.scene.env_origins[0]
    out = {}
    for attr_name, key in (
        ("_object_pos_w", "object_pos"),
        ("_palm_pos_w", "palm_pos"),
    ):
        value = getattr(base, attr_name, None)
        if value is not None:
            out[key] = [float(v) for v in (value[0] - origin).detach().cpu()]
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
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
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
    try:
        if args_cli.video_path:
            _set_video_camera_pose(env.unwrapped)
        _trace("reset")
        env.reset()
        _trace("reset done")
        if args_cli.video_path:
            _set_video_camera_pose(env.unwrapped)

        device = env.unwrapped.device
        actions = torch.zeros((args_cli.num_envs, env.unwrapped.cfg.action_space), device=device)
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
        dropped_any = False
        out_xy_any = False
        max_clearance_penalty = torch.zeros(args_cli.num_envs, device=device)
        min_clearance_margin = torch.full((args_cli.num_envs,), float("inf"), device=device)
        for step in range(int(args_cli.steps)):
            if args_cli.video_path and step % max(args_cli.video_stride, 1) == 0:
                _record_frame(env, frames)
            _, _, terminated, truncated, last_extras = env.step(actions)
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
            dropped = last_extras.get("dropped_env")
            if dropped is not None:
                dropped_any = dropped_any or bool(torch.any(dropped).detach().cpu())
            out_xy = last_extras.get("out_xy_env")
            if out_xy is not None:
                out_xy_any = out_xy_any or bool(torch.any(out_xy).detach().cpu())
            terminated_any = terminated_any or bool(torch.any(terminated).detach().cpu())
            truncated_any = truncated_any or bool(torch.any(truncated).detach().cpu())
            if args_cli.debug_interval > 0 and (step + 1) % int(args_cli.debug_interval) == 0:
                log = _mean_log(last_extras)
                _trace(
                    "step="
                    f"{step + 1:04d} success={metric_max['success'].mean().item():.3f} "
                    f"lifted={metric_max['lifted'].mean().item():.3f} "
                    f"true={metric_max['true_grasp'].mean().item():.3f} "
                    f"reward={log.get('total_reward', float('nan')):.3f}"
                )

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
            "steps": int(args_cli.steps),
            "max_success_rate": float(metric_max["success"].mean().detach().cpu()),
            "max_true_grasp_rate": float(metric_max["true_grasp"].mean().detach().cpu()),
            "max_lifted_rate": float(metric_max["lifted"].mean().detach().cpu()),
            "max_stable_hold_rate": float(metric_max["stable_hold"].mean().detach().cpu()),
            "terminated_any": terminated_any,
            "truncated_any": truncated_any,
            "clearance_violation_any": clearance_violation_any,
            "dropped_any": dropped_any,
            "out_xy_any": out_xy_any,
            "max_clearance_penalty": float(max_clearance_penalty.max().detach().cpu()),
            "min_clearance_margin": float(min_clearance_margin.min().detach().cpu()),
            "final_log": _mean_log(last_extras),
            "final_geometry": _geometry_debug(env),
            "video_path": str(video_path) if video_path is not None else None,
        }
        output_path = Path(args_cli.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
