#!/usr/bin/env python3
"""Render a fixed-camera preview rollout for dynamic dexterous grasp envs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="SimToolReal-Revo2-Franka-FallingBaton-Teacher-Direct-v0")
parser.add_argument("--out", required=True, help="Output mp4 path.")
parser.add_argument("--summary-out", default=None, help="Optional JSON summary path.")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--video-stride", type=int, default=2)
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--video-camera-track-object", action="store_true", default=False)
parser.add_argument(
    "--video-camera-scene-eye",
    type=float,
    nargs=3,
    default=(1.85, 1.55, 1.35),
    help="Fixed camera eye in each env's local frame when not tracking the object.",
)
parser.add_argument(
    "--video-camera-scene-target",
    type=float,
    nargs=3,
    default=(0.18, 0.04, 0.56),
    help="Fixed camera target in each env's local frame when not tracking the object.",
)
parser.add_argument("--video-camera-track-offset", type=float, nargs=3, default=(0.65, 0.95, 0.65))
parser.add_argument("--video-camera-track-target-offset", type=float, nargs=3, default=(0.0, 0.0, 0.05))
parser.add_argument("--video-camera-focal-length", type=float, default=20.0)
parser.add_argument("--video-camera-resolution", type=int, nargs=2, default=(960, 540))
parser.add_argument("--dynamic-curriculum-alpha", type=float, default=None)
parser.add_argument("--action-mode", choices=("zeros", "random", "close"), default="zeros")
parser.add_argument("--close-start-step", type=int, default=30)
parser.add_argument("--show-affordance-colors", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdShade  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


def _trace(message: str) -> None:
    print(f"[PREVIEW] {message}", flush=True)


def _set_video_camera_poses(unwrapped_env) -> bool:
    camera = getattr(unwrapped_env, "_video_camera", None)
    if camera is None:
        return False
    origins = unwrapped_env.scene.env_origins
    if bool(args_cli.video_camera_track_object):
        target_offset = torch.tensor(
            args_cli.video_camera_track_target_offset, device=origins.device, dtype=origins.dtype
        ).unsqueeze(0)
        camera_offset = torch.tensor(
            args_cli.video_camera_track_offset, device=origins.device, dtype=origins.dtype
        ).unsqueeze(0)
        object_root_pos = getattr(getattr(getattr(unwrapped_env, "object", None), "data", None), "root_pos_w", None)
        if object_root_pos is not None:
            targets = object_root_pos[:, :3] + target_offset
        else:
            object_init_pos = torch.tensor(
                unwrapped_env.cfg.object_cfg.init_state.pos, device=origins.device, dtype=origins.dtype
            ).unsqueeze(0)
            targets = origins + object_init_pos + target_offset
        eyes = targets + camera_offset
    else:
        scene_eye = torch.tensor(args_cli.video_camera_scene_eye, device=origins.device, dtype=origins.dtype).unsqueeze(0)
        scene_target = torch.tensor(
            args_cli.video_camera_scene_target, device=origins.device, dtype=origins.dtype
        ).unsqueeze(0)
        eyes = origins + scene_eye
        targets = origins + scene_target

    camera.set_world_poses_from_view(eyes, targets)
    if hasattr(camera, "_update_poses") and hasattr(camera, "_ALL_INDICES"):
        camera._update_poses(camera._ALL_INDICES)
    return True


def _make_preview_material(stage, path: str, color: tuple[float, float, float]):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _add_cube_segment(stage, path: str, size: tuple[float, float, float], center_z: float, material) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.8, 0.8)])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(center_z)))
    xform.AddScaleOp().Set(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)


def _add_affordance_visualization(unwrapped_env) -> bool:
    if not bool(args_cli.show_affordance_colors):
        return False
    if str(getattr(unwrapped_env.cfg, "task_family", "")) != "falling_baton_grasp":
        return False

    stage = unwrapped_env.scene.stage
    materials_root = "/World/Looks/AffordancePreview"
    positive_mat = _make_preview_material(stage, f"{materials_root}/positive_handle_green", (0.05, 0.80, 0.25))
    negative_mat = _make_preview_material(stage, f"{materials_root}/negative_blade_red", (0.95, 0.08, 0.06))
    neutral_mat = _make_preview_material(stage, f"{materials_root}/neutral_ignore_gray", (0.78, 0.78, 0.72))

    sx, sy, sz = (float(v) for v in unwrapped_env.cfg.object_size)
    pos_frac = min(max(float(getattr(unwrapped_env.cfg, "affordance_positive_fraction", 0.38)), 0.0), 1.0)
    neg_frac = min(max(float(getattr(unwrapped_env.cfg, "affordance_negative_fraction", 0.45)), 0.0), 1.0)
    pos_len = sz * pos_frac
    neg_len = sz * neg_frac
    neutral_len = max(sz - pos_len - neg_len, 0.0)
    visual_xy = 1.18 * max(sx, sy)
    positive_at_negative_end = str(getattr(unwrapped_env.cfg, "affordance_positive_end", "negative")).lower() not in {
        "positive",
        "+",
        "pos",
        "tip",
    }

    if positive_at_negative_end:
        pos_center = -0.5 * sz + 0.5 * pos_len
        neg_center = 0.5 * sz - 0.5 * neg_len
        neutral_center = -0.5 * sz + pos_len + 0.5 * neutral_len
    else:
        pos_center = 0.5 * sz - 0.5 * pos_len
        neg_center = -0.5 * sz + 0.5 * neg_len
        neutral_center = -0.5 * sz + neg_len + 0.5 * neutral_len

    for env_id in range(unwrapped_env.num_envs):
        root = f"/World/envs/env_{env_id}/Object/AffordancePreview"
        UsdGeom.Xform.Define(stage, root)
        _add_cube_segment(stage, f"{root}/positive_handle", (visual_xy, visual_xy, pos_len), pos_center, positive_mat)
        if neutral_len > 1.0e-5:
            _add_cube_segment(stage, f"{root}/neutral_ignore", (visual_xy, visual_xy, neutral_len), neutral_center, neutral_mat)
        _add_cube_segment(stage, f"{root}/negative_blade", (visual_xy, visual_xy, neg_len), neg_center, negative_mat)
    return True


def _force_camera_update(camera, dt: float) -> None:
    if camera is not None and hasattr(camera, "update"):
        camera.update(dt, force_recompute=True)


def _capture_frame(env, frames: list[np.ndarray]) -> None:
    unwrapped = env.unwrapped
    camera = getattr(unwrapped, "_video_camera", None)
    if camera is None:
        frame = unwrapped.render(recompute=True)
        if frame is not None and frame.size > 0:
            frames.append(frame.copy())
        return

    if bool(args_cli.video_camera_track_object):
        _set_video_camera_poses(unwrapped)
    _force_camera_update(camera, float(unwrapped.dt))
    rgb = camera.data.output.get("rgb")
    if rgb is None or rgb.numel() == 0:
        return
    frame = rgb[0, ..., :3].detach().cpu().numpy()
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0 if frame.max() <= 1.0 else frame, 0, 255).astype(np.uint8)
    frames.append(frame.copy())


def _make_action(unwrapped, step: int) -> torch.Tensor:
    actions = torch.zeros((unwrapped.num_envs, int(unwrapped.cfg.action_space)), device=unwrapped.device)
    if args_cli.action_mode == "random":
        return 2.0 * torch.rand_like(actions) - 1.0
    if args_cli.action_mode == "close" and step >= int(args_cli.close_start_step) and actions.shape[-1] > 7:
        actions[:, 7:] = 1.0
    return actions


def _float_list(tensor: torch.Tensor) -> list:
    return [float(v) for v in tensor.detach().cpu().flatten()]


def main() -> None:
    carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    torch.manual_seed(int(args_cli.seed))
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    if args_cli.dynamic_curriculum_alpha is not None and hasattr(
        env_cfg, "dynamic_grasp_speed_curriculum_override_alpha"
    ):
        env_cfg.dynamic_grasp_speed_curriculum_override_alpha = float(args_cli.dynamic_curriculum_alpha)
    if hasattr(env_cfg, "video_camera_enabled"):
        env_cfg.video_camera_enabled = True
        env_cfg.video_camera.data_types = ["rgb"]
        env_cfg.video_camera.spawn.focal_length = float(args_cli.video_camera_focal_length)
        env_cfg.video_camera.width = int(args_cli.video_camera_resolution[0])
        env_cfg.video_camera.height = int(args_cli.video_camera_resolution[1])
    if hasattr(env_cfg, "terminate_on_success"):
        env_cfg.terminate_on_success = False

    _trace(f"making env task={args_cli.task} num_envs={args_cli.num_envs} device={args_cli.device}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.unwrapped.sim._app_control_on_stop_handle = None
    frames: list[np.ndarray] = []
    try:
        obs, _ = env.reset(seed=int(args_cli.seed))
        del obs
        affordance_visualization = _add_affordance_visualization(env.unwrapped)
        _set_video_camera_poses(env.unwrapped)
        env.unwrapped._compute_intermediate_values()
        initial_object_pos_local = env.unwrapped._object_pos_w - env.unwrapped.scene.env_origins
        for step in range(int(args_cli.steps)):
            actions = _make_action(env.unwrapped, step)
            env.step(actions)
            if step % max(int(args_cli.video_stride), 1) == 0:
                _capture_frame(env, frames)
        _capture_frame(env, frames)

        out_path = Path(args_cli.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_path, frames, fps=int(args_cli.video_fps), macro_block_size=16)

        unwrapped = env.unwrapped
        unwrapped._compute_intermediate_values()
        object_pos_local = unwrapped._object_pos_w - unwrapped.scene.env_origins
        palm_pos_local = unwrapped._palm_pos_w - unwrapped.scene.env_origins
        table_paths = []
        stage = unwrapped.scene.stage
        for env_id in range(min(unwrapped.num_envs, 4)):
            table_path = f"/World/envs/env_{env_id}/Table"
            if stage.GetPrimAtPath(table_path).IsValid():
                table_paths.append(table_path)
        summary = {
            "task": args_cli.task,
            "video_path": str(out_path),
            "frames": len(frames),
            "action_mode": args_cli.action_mode,
            "camera_track_object": bool(args_cli.video_camera_track_object),
            "camera_scene_eye": list(args_cli.video_camera_scene_eye),
            "camera_scene_target": list(args_cli.video_camera_scene_target),
            "camera_track_offset": list(args_cli.video_camera_track_offset),
            "camera_track_target_offset": list(args_cli.video_camera_track_target_offset),
            "show_affordance_colors": bool(affordance_visualization),
            "object_size": list(getattr(unwrapped.cfg, "object_size", ())),
            "create_table_cfg": bool(getattr(unwrapped.cfg, "create_table", False)),
            "valid_table_paths": table_paths,
            "initial_object_pos_local_env0": _float_list(initial_object_pos_local[0]),
            "object_pos_local_env0": _float_list(object_pos_local[0]),
            "palm_pos_local_env0": _float_list(palm_pos_local[0]),
            "action_space": int(unwrapped.cfg.action_space),
        }
        summary_path = (
            Path(args_cli.summary_out).expanduser().resolve()
            if args_cli.summary_out
            else out_path.with_suffix(".json")
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _trace(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
