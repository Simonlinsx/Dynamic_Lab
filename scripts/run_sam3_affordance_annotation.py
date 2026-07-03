#!/usr/bin/env python
"""Run SAM3-based multiview affordance annotation for object assets.

The script renders object meshes from several orthographic views, asks SAM3 for
text-prompt masks on each rendered image, and projects those 2D masks back to
mesh vertices by visible-face voting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "assets" / "affordance_labels" / "asset_manifest.json"

AFFORDANCE_IDS = {
    "background": 0,
    "positive_grasp": 1,
    "negative_grasp": 2,
    "tool_use_surface": 3,
    "safe_contact": 4,
}

AFFORDANCE_COLORS = {
    "positive_grasp": (35, 145, 230),
    "negative_grasp": (230, 55, 45),
    "tool_use_surface": (35, 185, 90),
    "safe_contact": (240, 170, 45),
}


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    texcoords: np.ndarray | None
    face_texcoords: np.ndarray | None
    texture: np.ndarray | None
    face_colors: np.ndarray | None


@dataclass
class RenderData:
    image: Image.Image
    face_id: np.ndarray
    view_name: str


def _resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _parse_mtl(mtl_path: Path) -> Dict[str, Dict[str, Any]]:
    materials: Dict[str, Dict[str, Any]] = {}
    current: Dict[str, Any] | None = None
    if not mtl_path.exists():
        return materials

    with mtl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            key = fields[0]
            if key == "newmtl" and len(fields) >= 2:
                current = {}
                materials[fields[1]] = current
            elif current is not None and key == "Kd" and len(fields) >= 4:
                current["kd"] = tuple(float(x) for x in fields[1:4])
            elif current is not None and key == "map_Kd" and len(fields) >= 2:
                current["map_Kd"] = " ".join(fields[1:])
    return materials


def _load_obj(path: Path) -> MeshData:
    vertices: List[Tuple[float, float, float]] = []
    texcoords: List[Tuple[float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    face_texcoords: List[Tuple[int, int, int]] = []
    face_materials: List[str | None] = []
    mtllibs: List[Path] = []
    current_material: str | None = None

    def parse_face_token(token: str) -> Tuple[int, int]:
        parts = token.split("/")
        vertex_idx = int(parts[0])
        tex_idx = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if vertex_idx < 0:
            vertex_idx = len(vertices) + vertex_idx + 1
        if tex_idx < 0:
            tex_idx = len(texcoords) + tex_idx + 1
        return vertex_idx - 1, tex_idx - 1

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if raw.startswith("v "):
                fields = raw.split()
                if len(fields) >= 4:
                    vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
            elif raw.startswith("vt "):
                fields = raw.split()
                if len(fields) >= 3:
                    texcoords.append((float(fields[1]), float(fields[2])))
            elif raw.startswith("mtllib "):
                mtllibs.append(path.parent / raw.split(maxsplit=1)[1].strip())
            elif raw.startswith("usemtl "):
                current_material = raw.split(maxsplit=1)[1].strip()
            elif raw.startswith("f "):
                tokens = raw.split()[1:]
                if len(tokens) < 3:
                    continue
                parsed = [parse_face_token(token) for token in tokens]
                for idx in range(1, len(parsed) - 1):
                    tri = (parsed[0], parsed[idx], parsed[idx + 1])
                    faces.append((tri[0][0], tri[1][0], tri[2][0]))
                    face_texcoords.append((tri[0][1], tri[1][1], tri[2][1]))
                    face_materials.append(current_material)

    vertices_np = np.asarray(vertices, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int32)
    tex_np = np.asarray(texcoords, dtype=np.float32) if texcoords else None
    face_tex_np = np.asarray(face_texcoords, dtype=np.int32) if texcoords else None

    materials: Dict[str, Dict[str, Any]] = {}
    for mtl_path in mtllibs:
        materials.update(_parse_mtl(mtl_path))

    texture = None
    if face_materials:
        for material_name in face_materials:
            if not material_name or material_name not in materials:
                continue
            texture_name = materials[material_name].get("map_Kd")
            if texture_name:
                texture_path = path.parent / texture_name
                if texture_path.exists():
                    texture = np.asarray(Image.open(texture_path).convert("RGB"))
                    break

    face_colors = None
    if face_materials:
        colors = []
        for material_name in face_materials:
            kd = materials.get(material_name or "", {}).get("kd", (0.72, 0.72, 0.72))
            colors.append(kd)
        face_colors = np.asarray(colors, dtype=np.float32)

    return MeshData(
        vertices=vertices_np,
        faces=faces_np,
        texcoords=tex_np,
        face_texcoords=face_tex_np,
        texture=texture,
        face_colors=face_colors,
    )


def _load_trimesh_mesh(path: Path) -> MeshData:
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            f"{path.suffix} assets need trimesh. Install it in the SAM3 env or use OBJ assets first."
        ) from exc

    loaded = trimesh.load(path, force="scene")
    if hasattr(loaded, "geometry"):
        if not loaded.geometry:
            raise ValueError(f"No geometry found in {path}")
        if hasattr(loaded, "to_geometry"):
            mesh = loaded.to_geometry()
        else:
            mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if mesh is None:
        raise ValueError(f"No geometry found in {path}")
    colors = None
    if getattr(mesh.visual, "kind", None) == "face" and len(mesh.visual.face_colors):
        colors = np.asarray(mesh.visual.face_colors[:, :3], dtype=np.float32) / 255.0
    else:
        try:
            color_visual = mesh.visual.to_color()
            face_colors = getattr(color_visual, "face_colors", None)
            vertex_colors = getattr(color_visual, "vertex_colors", None)
            if face_colors is not None and len(face_colors) == len(mesh.faces):
                colors = np.asarray(face_colors[:, :3], dtype=np.float32) / 255.0
            elif vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
                vertex_rgb = np.asarray(vertex_colors[:, :3], dtype=np.float32) / 255.0
                colors = vertex_rgb[np.asarray(mesh.faces, dtype=np.int64)].mean(axis=1)
        except Exception:
            colors = None
    return MeshData(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        texcoords=None,
        face_texcoords=None,
        texture=None,
        face_colors=colors,
    )


def load_mesh(path: Path) -> MeshData:
    if path.suffix.lower() == ".obj":
        mesh = _load_obj(path)
    else:
        mesh = _load_trimesh_mesh(path)
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Mesh has no vertices/faces: {path}")
    return mesh


def _view_axes(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def orthographic_axes(
        depth_value: Tuple[float, float, float],
        up_value: Tuple[float, float, float] = (0, 0, 1),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth = np.asarray(depth_value, dtype=np.float32)
        depth = depth / max(float(np.linalg.norm(depth)), 1e-8)
        up = np.asarray(up_value, dtype=np.float32)
        up = up / max(float(np.linalg.norm(up)), 1e-8)
        if abs(float(np.dot(up, depth))) > 0.92:
            up = np.asarray((0, 1, 0), dtype=np.float32)
        u_axis = np.cross(up, depth)
        u_axis = u_axis / max(float(np.linalg.norm(u_axis)), 1e-8)
        v_axis = np.cross(depth, u_axis)
        v_axis = v_axis / max(float(np.linalg.norm(v_axis)), 1e-8)
        return u_axis, v_axis, depth

    base = {
        "front": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        "back": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        "left": ((0, 1, 0), (0, 0, 1), (-1, 0, 0)),
        "right": ((0, -1, 0), (0, 0, 1), (1, 0, 0)),
        "top": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "bottom": ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
        "iso1": ((1, -1, 0), (0.5, 0.5, 1.0), (1, 1, 1)),
        "iso2": ((-1, -1, 0), (-0.5, 0.5, 1.0), (-1, 1, 1)),
    }
    generated = {
        "iso3": (1, -1, -1),
        "iso4": (-1, -1, -1),
        "front_top": (0, -1, 1),
        "back_top": (0, 1, 1),
        "left_top": (-1, 0, 1),
        "right_top": (1, 0, 1),
        "front_bottom": (0, -1, -1),
        "back_bottom": (0, 1, -1),
        "left_bottom": (-1, 0, -1),
        "right_bottom": (1, 0, -1),
    }
    if name in generated:
        return orthographic_axes(generated[name])
    if name not in base:
        raise ValueError(f"Unknown view {name!r}")
    axes = []
    for axis in base[name]:
        arr = np.asarray(axis, dtype=np.float32)
        axes.append(arr / max(float(np.linalg.norm(arr)), 1e-8))
    return axes[0], axes[1], axes[2]


def _triangle_barycentric(
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    tri: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0, y0 = tri[0]
    x1, y1 = tri[1]
    x2, y2 = tri[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denom)) < 1e-8:
        empty = np.zeros_like(pixel_x, dtype=np.float32)
        return empty, empty, empty, np.zeros_like(pixel_x, dtype=bool)
    w0 = ((y1 - y2) * (pixel_x - x2) + (x2 - x1) * (pixel_y - y2)) / denom
    w1 = ((y2 - y0) * (pixel_x - x2) + (x0 - x2) * (pixel_y - y2)) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
    return w0, w1, w2, inside


def render_mesh(mesh: MeshData, view_name: str, image_size: int, margin: float) -> RenderData:
    u_axis, v_axis, depth_axis = _view_axes(view_name)
    vertices = mesh.vertices
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    centered = vertices - center
    x = centered @ u_axis
    y = centered @ v_axis
    z = centered @ depth_axis

    extent = max(float(x.max() - x.min()), float(y.max() - y.min()), 1e-6)
    scale = (image_size * (1.0 - 2.0 * margin)) / extent
    px = (x - (x.min() + x.max()) / 2.0) * scale + image_size / 2.0
    py = image_size / 2.0 - (y - (y.min() + y.max()) / 2.0) * scale
    screen = np.stack([px, py], axis=1).astype(np.float32)

    rgb = np.full((image_size, image_size, 3), 246, dtype=np.uint8)
    depth = np.full((image_size, image_size), -np.inf, dtype=np.float32)
    face_id = np.full((image_size, image_size), -1, dtype=np.int32)

    texture = mesh.texture
    tex_h = tex_w = 0
    if texture is not None:
        tex_h, tex_w = texture.shape[:2]

    for fid, face in enumerate(mesh.faces):
        tri = screen[face]
        min_x = max(int(math.floor(float(tri[:, 0].min()))), 0)
        max_x = min(int(math.ceil(float(tri[:, 0].max()))), image_size - 1)
        min_y = max(int(math.floor(float(tri[:, 1].min()))), 0)
        max_y = min(int(math.ceil(float(tri[:, 1].max()))), image_size - 1)
        if min_x > max_x or min_y > max_y:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        w0, w1, w2, inside = _triangle_barycentric(grid_x, grid_y, tri)
        if not inside.any():
            continue

        face_depth = w0 * z[face[0]] + w1 * z[face[1]] + w2 * z[face[2]]
        patch_depth = depth[min_y : max_y + 1, min_x : max_x + 1]
        visible = inside & (face_depth > patch_depth)
        if not visible.any():
            continue

        if (
            texture is not None
            and mesh.texcoords is not None
            and mesh.face_texcoords is not None
            and np.all(mesh.face_texcoords[fid] >= 0)
        ):
            uv_tri = mesh.texcoords[mesh.face_texcoords[fid]]
            uu = w0 * uv_tri[0, 0] + w1 * uv_tri[1, 0] + w2 * uv_tri[2, 0]
            vv = w0 * uv_tri[0, 1] + w1 * uv_tri[1, 1] + w2 * uv_tri[2, 1]
            tex_x = np.clip((uu * (tex_w - 1)).astype(np.int32), 0, tex_w - 1)
            tex_y = np.clip(((1.0 - vv) * (tex_h - 1)).astype(np.int32), 0, tex_h - 1)
            color = texture[tex_y, tex_x]
        else:
            face_vertices = vertices[face]
            normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0])
            normal_norm = max(float(np.linalg.norm(normal)), 1e-8)
            shade = 0.55 + 0.45 * abs(float(np.dot(normal / normal_norm, depth_axis)))
            base_color = (
                mesh.face_colors[fid]
                if mesh.face_colors is not None and fid < len(mesh.face_colors)
                else np.asarray([0.72, 0.72, 0.72], dtype=np.float32)
            )
            color = np.clip(base_color * shade * 255.0, 0, 255).astype(np.uint8)

        patch_rgb = rgb[min_y : max_y + 1, min_x : max_x + 1]
        patch_face = face_id[min_y : max_y + 1, min_x : max_x + 1]
        if isinstance(color, np.ndarray) and color.ndim == 3:
            patch_rgb[visible] = color[visible]
        else:
            patch_rgb[visible] = color
        patch_depth[visible] = face_depth[visible]
        patch_face[visible] = fid

    return RenderData(image=Image.fromarray(rgb), face_id=face_id, view_name=view_name)


def _prompt_map(entry: Mapping[str, Any]) -> Dict[str, str]:
    prompts = dict(entry.get("suggested_prompts") or {})
    label = str(entry.get("label", entry.get("asset_id", "object"))).replace("_", " ")
    part_vocab = [str(part).replace("_", " ") for part in entry.get("part_vocab", [])]
    functional_priority = [
        "tip",
        "blade",
        "bristles",
        "striking face",
        "head",
        "claw",
        "rim",
        "spout",
        "chuck",
        "trigger",
        "cap",
        "lid",
        "top",
        "neck",
        "inner surface",
        "center",
        "buttons",
        "scroll wheel",
        "metal tip",
        "pages",
        "stem",
        "weights",
    ]
    if "positive_grasp" not in prompts:
        if "handle" in part_vocab:
            grasp_part = "handle"
        elif "body" in part_vocab:
            grasp_part = "body"
        elif part_vocab:
            grasp_part = part_vocab[0]
        else:
            grasp_part = "body"
        prompts["positive_grasp"] = f"{label} {grasp_part}"
    if "negative_grasp" not in prompts and part_vocab:
        grasp_text = prompts["positive_grasp"].removeprefix(f"{label} ").strip()
        negative_parts = [
            part
            for part in functional_priority
            if part in part_vocab and part not in grasp_text
        ]
        if not negative_parts:
            negative_parts = [part for part in part_vocab if part not in grasp_text][:2]
        prompts["negative_grasp"] = f"{label} {' or '.join(negative_parts[:2])}"
    if "tool_use_surface" not in prompts:
        use_parts = [part for part in functional_priority if part in part_vocab]
        prompts["tool_use_surface"] = f"{label} {(use_parts[0] if use_parts else part_vocab[-1] if part_vocab else 'functional part')}"
    return {k: v for k, v in prompts.items() if k in AFFORDANCE_IDS}


def _prompt_variants(entry: Mapping[str, Any], prompts: Mapping[str, str]) -> Dict[str, List[str]]:
    label_words = set(str(entry.get("label", "")).replace("_", " ").split())
    variants: Dict[str, List[str]] = {}
    for affordance, prompt in prompts.items():
        ordered: List[str] = []
        for candidate in [prompt, *prompt.replace("/", " or ").split(" or ")]:
            clean = " ".join(candidate.replace("_", " ").split())
            if clean and clean not in ordered:
                ordered.append(clean)
            part_only = " ".join(word for word in clean.split() if word not in label_words)
            if part_only and part_only not in ordered:
                ordered.append(part_only)
        variants[affordance] = ordered
    return variants


def _load_sam3_model(args: argparse.Namespace):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The official SAM3 image model requires CUDA here, but torch.cuda.is_available() is False. "
            "Run this script on a node with /dev/nvidia* devices."
        )
    if args.device != "cuda":
        raise RuntimeError("The official SAM3 image model is CUDA-only in this setup; use --device cuda.")

    _patch_torch_batched_gemm_for_sam3(torch)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    checkpoint_path = str(args.checkpoint_path) if args.checkpoint_path else None
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=checkpoint_path is None and not args.no_hf,
        device=args.device,
        eval_mode=True,
        enable_inst_interactivity=False,
    )
    return Sam3Processor(model, device=args.device, confidence_threshold=args.confidence_threshold)


def _patch_torch_batched_gemm_for_sam3(torch_module: Any) -> None:
    """Avoid cuBLAS strided-batched GEMM failures in this CUDA/PyTorch stack.

    On the local L40 + torch 2.10/cu128 setup, even a small
    torch.bmm([8, 201, 32], [8, 32, 201]) can fail with
    CUBLAS_STATUS_INVALID_VALUE. SAM3's decoder uses exactly these batched GEMMs.
    The loop fallback is slower but reliable for offline annotation.
    """

    if getattr(torch_module, "_simtoolreal_sam3_batched_gemm_patch", False):
        return

    original_bmm = torch_module.bmm
    original_baddbmm = torch_module.baddbmm
    original_matmul = torch_module.matmul

    def loop_bmm(a, b, *args, **kwargs):
        if a.is_cuda and a.dim() == 3 and b.dim() == 3 and a.shape[0] == b.shape[0]:
            return torch_module.stack(
                [
                    original_matmul(a[idx].contiguous(), b[idx].contiguous())
                    for idx in range(a.shape[0])
                ],
                dim=0,
            )
        return original_bmm(a, b, *args, **kwargs)

    def loop_baddbmm(input_tensor, batch1, batch2, *, beta=1, alpha=1, out=None):
        if (
            batch1.is_cuda
            and batch1.dim() == 3
            and batch2.dim() == 3
            and batch1.shape[0] == batch2.shape[0]
        ):
            result = input_tensor * beta + loop_bmm(batch1, batch2) * alpha
            if out is not None:
                out.copy_(result)
                return out
            return result
        return original_baddbmm(
            input_tensor, batch1, batch2, beta=beta, alpha=alpha, out=out
        )

    def safe_matmul(a, b, *args, **kwargs):
        if torch_module.is_tensor(a) and torch_module.is_tensor(b) and a.is_cuda:
            if b.dim() >= 2 and b.shape[-1] == 1 and a.shape[-1] == b.shape[-2]:
                return (a * b.squeeze(-1)).sum(dim=-1, keepdim=True)
            if a.dim() == 3 and b.dim() == 3 and a.shape[0] == b.shape[0]:
                return torch_module.stack(
                    [
                        original_matmul(a[idx].contiguous(), b[idx].contiguous())
                        for idx in range(a.shape[0])
                    ],
                    dim=0,
                )
        return original_matmul(a, b, *args, **kwargs)

    torch_module.bmm = loop_bmm
    torch_module.baddbmm = loop_baddbmm
    torch_module.matmul = safe_matmul
    torch_module._simtoolreal_sam3_batched_gemm_patch = True


def _to_float32_nested(value: Any):
    import torch

    if torch.is_tensor(value) and torch.is_floating_point(value):
        return value.float()
    if isinstance(value, dict):
        return {key: _to_float32_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_float32_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_float32_nested(item) for item in value)
    return value


def _sam3_mask_for_prompt(
    processor: Any,
    image: Image.Image,
    amp_dtype: str,
) -> Dict[str, Any]:
    import torch

    use_amp = processor.device == "cuda" and amp_dtype != "none"
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
        state = processor.set_image(image)
    state["backbone_out"] = _to_float32_nested(state["backbone_out"])
    state["_simtoolreal_image_size"] = image.size
    return state


def _sam3_prompt_on_state(
    processor: Any,
    state: Dict[str, Any],
    prompt: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    processor.reset_all_prompts(state)
    output = processor.set_text_prompt(prompt=prompt, state=state)
    masks = output.get("masks")
    scores = output.get("scores")
    boxes = output.get("boxes")
    width, height = state["_simtoolreal_image_size"]
    if masks is None or len(masks) == 0:
        return np.zeros((height, width), dtype=bool), {"num_masks": 0, "scores": []}

    masks_np = masks.detach().cpu().numpy().astype(bool)
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    scores_np = scores.detach().cpu().numpy() if scores is not None else np.ones((len(masks_np),), dtype=np.float32)
    keep = scores_np >= processor.confidence_threshold
    if not keep.any():
        return np.zeros((height, width), dtype=bool), {
            "num_masks": int(len(masks_np)),
            "kept_masks": 0,
            "scores": [float(x) for x in scores_np],
        }
    union = masks_np[keep].any(axis=0)
    info: Dict[str, Any] = {
        "num_masks": int(len(masks_np)),
        "kept_masks": int(keep.sum()),
        "scores": [float(x) for x in scores_np],
    }
    if boxes is not None:
        info["boxes_xyxy"] = boxes.detach().cpu().numpy().tolist()
    return union, info


def _sam3_mask_for_variants(
    processor: Any,
    state: Dict[str, Any],
    variants: Sequence[str],
    prompt_cache: Dict[str, Tuple[np.ndarray, Dict[str, Any]]],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    combined = np.zeros((image_size[1], image_size[0]), dtype=bool)
    infos = []
    for prompt in variants:
        if prompt in prompt_cache:
            mask, info = prompt_cache[prompt]
        else:
            mask, info = _sam3_prompt_on_state(processor, state, prompt)
            prompt_cache[prompt] = (mask, info)
        combined |= mask
        prompt_info = dict(info)
        prompt_info["prompt"] = prompt
        prompt_info["mask_pixels"] = int(mask.sum())
        infos.append(prompt_info)
    best_scores = [
        max(info.get("scores", []) or [0.0])
        for info in infos
    ]
    return combined, {
        "variants": infos,
        "best_score": float(max(best_scores) if best_scores else 0.0),
        "mask_pixels": int(combined.sum()),
    }


def _overlay_mask(image: Image.Image, masks: Mapping[str, np.ndarray]) -> Image.Image:
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    for affordance, mask in masks.items():
        color = AFFORDANCE_COLORS.get(affordance, (255, 255, 255))
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        rgba = Image.new("RGBA", image.size, color + (0,))
        alpha = np.zeros((image.height, image.width), dtype=np.uint8)
        alpha[ys, xs] = 92
        rgba.putalpha(Image.fromarray(alpha))
        overlay = Image.alpha_composite(overlay, rgba)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        draw = ImageDraw.Draw(overlay, "RGBA")
        draw.rectangle(bbox, outline=color + (220,), width=2)
    return overlay.convert("RGB")


def _vote_vertices(
    votes: np.ndarray,
    faces: np.ndarray,
    face_id: np.ndarray,
    mask: np.ndarray,
    affordance_idx: int,
) -> int:
    visible_faces = face_id[mask & (face_id >= 0)]
    if len(visible_faces) == 0:
        return 0
    unique_faces, counts = np.unique(visible_faces, return_counts=True)
    for fid, count in zip(unique_faces, counts):
        votes[faces[int(fid)], affordance_idx] += float(count)
    return int(len(unique_faces))


def _save_label_npz(
    output_path: Path,
    mesh: MeshData,
    votes: np.ndarray,
    prompts: Mapping[str, str],
    view_names: Sequence[str],
    confidence_threshold: float,
    min_vote_pixels: float,
) -> Dict[str, Any]:
    class_names = ["background"] + list(AFFORDANCE_IDS.keys())[1:]
    label_scores = votes.copy()
    total = label_scores.sum(axis=1, keepdims=True)
    normalized = np.divide(label_scores, np.maximum(total, 1.0), out=np.zeros_like(label_scores), where=total > 0)
    affordance_multihot = label_scores >= min_vote_pixels
    best_idx = label_scores.argmax(axis=1) + 1
    best_score = label_scores.max(axis=1)
    affordance_id = np.where(best_score >= min_vote_pixels, best_idx, 0).astype(np.int32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        vertices_obj=mesh.vertices.astype(np.float32),
        faces=mesh.faces.astype(np.int32),
        affordance_id=affordance_id,
        affordance_multihot=affordance_multihot.astype(np.uint8),
        affordance_scores=label_scores.astype(np.float32),
        affordance_scores_normalized=normalized.astype(np.float32),
        affordance_vocab=np.asarray(class_names, dtype=object),
        affordance_multihot_vocab=np.asarray(class_names[1:], dtype=object),
        prompts_json=json.dumps(dict(prompts), indent=2),
        view_names=np.asarray(list(view_names), dtype=object),
        sam3_confidence_threshold=np.asarray([confidence_threshold], dtype=np.float32),
        min_vote_pixels=np.asarray([min_vote_pixels], dtype=np.float32),
        source=np.asarray(["sam3_multiview_text_projection"], dtype=object),
    )
    return {
        "annotation_path": str(output_path),
        "num_vertices": int(len(mesh.vertices)),
        "labeled_vertices": int((affordance_id > 0).sum()),
        "class_vertex_counts": {
            class_names[idx]: int((affordance_id == idx).sum()) for idx in range(len(class_names))
        },
        "multi_label_vertex_counts": {
            class_names[idx + 1]: int(affordance_multihot[:, idx].sum())
            for idx in range(affordance_multihot.shape[1])
        },
        "overlap_vertices": int((affordance_multihot.sum(axis=1) > 1).sum()),
    }


def annotate_entry(
    entry: Mapping[str, Any],
    repo_root: Path,
    processor: Any | None,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    mesh_path = _resolve_path(str(entry["visual_mesh_path"]), repo_root)
    annotation_dir = _resolve_path(str(entry["annotation_dir"]), repo_root)
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    label_path = annotation_dir / "affordance.npz"
    if label_path.exists() and not args.overwrite and not args.render_only:
        return {
            "asset_id": entry["asset_id"],
            "status": "skipped_existing",
            "annotation_path": str(label_path),
        }

    mesh = load_mesh(mesh_path)
    prompts = _prompt_map(entry)
    prompt_variants = _prompt_variants(entry, prompts)
    view_names = [name.strip() for name in args.views.split(",") if name.strip()]
    votes = np.zeros((len(mesh.vertices), len(AFFORDANCE_IDS) - 1), dtype=np.float32)

    render_dir = annotation_dir / "sam3_views"
    render_dir.mkdir(parents=True, exist_ok=True)
    prompt_infos: Dict[str, Any] = {}
    for view_name in view_names:
        render = render_mesh(mesh, view_name=view_name, image_size=args.image_size, margin=args.margin)
        image_path = render_dir / f"{view_name}.png"
        render.image.save(image_path)

        view_masks: Dict[str, np.ndarray] = {}
        prompt_infos[view_name] = {}
        if processor is not None:
            state = _sam3_mask_for_prompt(
                processor,
                render.image,
                amp_dtype=args.amp_dtype,
            )
            prompt_cache: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
            for affordance, prompt in prompts.items():
                mask, info = _sam3_mask_for_variants(
                    processor,
                    state,
                    prompt_variants[affordance],
                    prompt_cache,
                    image_size=render.image.size,
                )
                view_masks[affordance] = mask
                mask_path = render_dir / f"{view_name}_{affordance}_mask.png"
                Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
                voted_faces = _vote_vertices(
                    votes=votes,
                    faces=mesh.faces,
                    face_id=render.face_id,
                    mask=mask,
                    affordance_idx=AFFORDANCE_IDS[affordance] - 1,
                )
                info.update(
                    {
                        "prompt": prompt,
                        "prompt_variants": prompt_variants[affordance],
                        "mask_path": str(mask_path),
                        "voted_faces": voted_faces,
                        "mask_pixels": int(mask.sum()),
                    }
                )
                prompt_infos[view_name][affordance] = info
            _overlay_mask(render.image, view_masks).save(render_dir / f"{view_name}_overlay.png")

    summary: Dict[str, Any] = {
        "asset_id": entry["asset_id"],
        "label": entry.get("label"),
        "mesh_path": str(mesh_path),
        "annotation_dir": str(annotation_dir),
        "prompts": prompts,
        "views": view_names,
        "render_dir": str(render_dir),
        "render_only": bool(args.render_only),
        "prompt_infos": prompt_infos,
    }
    if not args.render_only:
        summary.update(
            _save_label_npz(
                output_path=label_path,
                mesh=mesh,
                votes=votes,
                prompts=prompts,
                view_names=view_names,
                confidence_threshold=args.confidence_threshold,
                min_vote_pixels=args.min_vote_pixels,
            )
        )

    summary_path = annotation_dir / "sam3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _entries_for_args(manifest: Mapping[str, Any], stage: str, asset_ids: Sequence[str]) -> List[Mapping[str, Any]]:
    entries = [
        entry
        for entry in manifest["assets"]
        if stage == "all" or str(entry.get("stage")) == stage
    ]
    if asset_ids:
        wanted = set(asset_ids)
        entries = [entry for entry in entries if entry.get("asset_id") in wanted or entry.get("label") in wanted]
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", choices=["dextoolbench12", "domino20", "all"], default="dextoolbench12")
    parser.add_argument("--asset-id", action="append", default=[], help="Filter by asset_id or label. Can be repeated.")
    parser.add_argument("--max-assets", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument(
        "--views",
        default="front,back,left,right,top,bottom,iso1,iso2",
        help="Comma-separated view names.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument("--min-vote-pixels", type=float, default=20.0)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16", "none"])
    parser.add_argument("--no-hf", action="store_true", help="Do not auto-download SAM3 weights from Hugging Face.")
    parser.add_argument("--render-only", action="store_true", help="Only render multiview images; do not load or run SAM3.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Write the full run summary to disk without printing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = Path(manifest.get("repo_root", REPO_ROOT)).resolve()
    entries = _entries_for_args(manifest, args.stage, args.asset_id)
    if args.max_assets > 0:
        entries = entries[: args.max_assets]
    if not entries:
        raise ValueError(f"No manifest entries matched stage={args.stage!r} asset_id={args.asset_id!r}")

    processor = None if args.render_only else _load_sam3_model(args)
    summaries = []
    failures = []
    for entry in entries:
        try:
            summaries.append(annotate_entry(entry, repo_root=repo_root, processor=processor, args=args))
        except Exception as exc:
            failures.append({"asset_id": entry.get("asset_id"), "error": repr(exc)})
            if not args.render_only:
                raise

    output_dir = repo_root / "assets" / "affordance_labels" / "sam3_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "manifest_path": str(manifest_path),
        "stage": args.stage,
        "num_requested": len(entries),
        "num_completed": len(summaries),
        "num_failed": len(failures),
        "render_only": bool(args.render_only),
        "summaries": summaries,
        "failures": failures,
    }
    run_summary_path = output_dir / f"{args.stage}_sam3_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    if args.quiet:
        print(
            json.dumps(
                {
                    "summary_path": str(run_summary_path),
                    "stage": args.stage,
                    "num_requested": len(entries),
                    "num_completed": len(summaries),
                    "num_failed": len(failures),
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
