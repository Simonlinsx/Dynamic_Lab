#!/usr/bin/env python
"""Run positive-only grasp affordance annotation and 3D cleanup.

This keeps SAM3 focused on the graspable part, then post-processes the projected
mesh labels with connected components and light graph dilation. The resulting
labels are intended for RL priors or auxiliary supervision, not as a hard task
success signal.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from run_sam3_affordance_annotation import (
    DEFAULT_MANIFEST,
    _load_sam3_model,
    _overlay_mask,
    _prompt_map,
    _prompt_variants,
    _resolve_path,
    _sam3_mask_for_prompt,
    _vote_vertices,
    load_mesh,
    render_mesh,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _entries_for_args(
    manifest: Mapping[str, Any],
    stage: str,
    asset_ids: Sequence[str],
) -> List[Mapping[str, Any]]:
    entries = [
        entry
        for entry in manifest["assets"]
        if stage == "all" or str(entry.get("stage")) == stage
    ]
    if asset_ids:
        wanted = set(asset_ids)
        entries = [entry for entry in entries if entry.get("asset_id") in wanted or entry.get("label") in wanted]
    return entries


def _build_adjacency(num_vertices: int, faces: np.ndarray) -> List[np.ndarray]:
    neighbors = [set() for _ in range(num_vertices)]
    for a, b, c in np.asarray(faces, dtype=np.int64):
        neighbors[a].update((int(b), int(c)))
        neighbors[b].update((int(a), int(c)))
        neighbors[c].update((int(a), int(b)))
    return [np.asarray(sorted(items), dtype=np.int32) for items in neighbors]


def _connected_components(mask: np.ndarray, adjacency: Sequence[np.ndarray]) -> List[np.ndarray]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: List[np.ndarray] = []
    for start in np.flatnonzero(mask):
        if visited[start]:
            continue
        queue: deque[int] = deque([int(start)])
        visited[start] = True
        component = []
        while queue:
            idx = queue.popleft()
            component.append(idx)
            for nb in adjacency[idx]:
                nb_int = int(nb)
                if mask[nb_int] and not visited[nb_int]:
                    visited[nb_int] = True
                    queue.append(nb_int)
        components.append(np.asarray(component, dtype=np.int32))
    components.sort(key=len, reverse=True)
    return components


def _component_filter(
    mask: np.ndarray,
    adjacency: Sequence[np.ndarray],
    min_component_vertices: int,
    keep_component_ratio: float,
) -> Tuple[np.ndarray, List[int]]:
    components = _connected_components(mask, adjacency)
    if not components:
        return np.zeros_like(mask), []
    largest = len(components[0])
    min_keep = max(min_component_vertices, int(round(largest * keep_component_ratio)))
    filtered = np.zeros_like(mask)
    kept_sizes = []
    for component in components:
        if len(component) >= min_keep:
            filtered[component] = True
            kept_sizes.append(int(len(component)))
    return filtered, kept_sizes


def _dilate(mask: np.ndarray, adjacency: Sequence[np.ndarray], steps: int) -> np.ndarray:
    current = mask.copy()
    for _ in range(max(0, steps)):
        expanded = current.copy()
        for idx in np.flatnonzero(current):
            expanded[adjacency[int(idx)]] = True
        current = expanded
    return current


def _fill_small_holes(mask: np.ndarray, adjacency: Sequence[np.ndarray], steps: int) -> np.ndarray:
    current = mask.copy()
    for _ in range(max(0, steps)):
        updated = current.copy()
        for idx in np.flatnonzero(~current):
            neigh = adjacency[int(idx)]
            if len(neigh) == 0:
                continue
            inside = int(current[neigh].sum())
            if inside >= max(3, int(np.ceil(0.70 * len(neigh)))):
                updated[int(idx)] = True
        current = updated
    return current


def _cleanup_positive_mask(
    raw_mask: np.ndarray,
    mesh_faces: np.ndarray,
    min_component_vertices: int,
    keep_component_ratio: float,
    dilation_steps: int,
    hole_fill_steps: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    adjacency = _build_adjacency(len(raw_mask), mesh_faces)
    connected, kept_sizes = _component_filter(
        raw_mask,
        adjacency,
        min_component_vertices=min_component_vertices,
        keep_component_ratio=keep_component_ratio,
    )
    if raw_mask.any() and not connected.any():
        connected = raw_mask.copy()
        kept_sizes = [int(raw_mask.sum())]
    filled = _dilate(connected, adjacency, steps=dilation_steps)
    filled = _fill_small_holes(filled, adjacency, steps=hole_fill_steps)
    info = {
        "raw_positive_vertices": int(raw_mask.sum()),
        "connected_positive_vertices": int(connected.sum()),
        "filled_positive_vertices": int(filled.sum()),
        "kept_component_sizes": kept_sizes,
        "dilation_steps": int(dilation_steps),
        "hole_fill_steps": int(hole_fill_steps),
    }
    return connected, filled, info


def _positive_mask_for_variants(
    processor: Any,
    state: Dict[str, Any],
    variants: Sequence[str],
    image_size: Tuple[int, int],
    selection: str,
    reject_slender: bool,
    max_slender_aspect: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    width, height = image_size
    candidates = []
    variant_infos = []
    for prompt in variants:
        processor.reset_all_prompts(state)
        output = processor.set_text_prompt(prompt=prompt, state=state)
        masks = output.get("masks")
        scores = output.get("scores")
        boxes = output.get("boxes")
        if masks is None or len(masks) == 0:
            variant_infos.append({"prompt": prompt, "num_masks": 0, "scores": [], "mask_pixels": 0})
            continue
        masks_np = masks.detach().cpu().numpy().astype(bool)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]
        scores_np = scores.detach().cpu().numpy() if scores is not None else np.ones((len(masks_np),), dtype=np.float32)
        boxes_np = boxes.detach().cpu().numpy().tolist() if boxes is not None else None
        keep = scores_np >= processor.confidence_threshold
        variant_infos.append(
            {
                "prompt": prompt,
                "num_masks": int(len(masks_np)),
                "kept_masks": int(keep.sum()),
                "scores": [float(x) for x in scores_np],
                "boxes_xyxy": boxes_np,
                "mask_pixels": int(masks_np[keep].any(axis=0).sum()) if keep.any() else 0,
            }
        )
        for idx, kept in enumerate(keep):
            if not kept:
                continue
            candidates.append(
                {
                    "prompt": prompt,
                    "score": float(scores_np[idx]),
                    "mask": masks_np[idx],
                    "mask_pixels": int(masks_np[idx].sum()),
                    "box_xyxy": boxes_np[idx] if boxes_np is not None else None,
                }
            )
    if not candidates:
        return np.zeros((height, width), dtype=bool), {
            "variants": variant_infos,
            "selection": selection,
            "selected": [],
            "best_score": 0.0,
            "mask_pixels": 0,
        }

    filtered_candidates = candidates
    if reject_slender:
        nonslender = []
        for item in candidates:
            box = item.get("box_xyxy")
            if box is None:
                nonslender.append(item)
                continue
            width_box = max(float(box[2]) - float(box[0]), 1.0)
            height_box = max(float(box[3]) - float(box[1]), 1.0)
            aspect = max(width_box, height_box) / max(min(width_box, height_box), 1.0)
            if aspect <= max_slender_aspect:
                nonslender.append(item)
        if nonslender:
            filtered_candidates = nonslender

    if selection == "union":
        selected = filtered_candidates
    elif selection == "best_per_prompt":
        selected = []
        for prompt in variants:
            prompt_candidates = [item for item in filtered_candidates if item["prompt"] == prompt]
            if prompt_candidates:
                selected.append(max(prompt_candidates, key=lambda item: item["score"]))
    else:
        selected = [max(filtered_candidates, key=lambda item: item["score"])]
    combined = np.zeros((height, width), dtype=bool)
    for item in selected:
        combined |= item["mask"]
    selected_info = [
        {
            "prompt": item["prompt"],
            "score": item["score"],
            "mask_pixels": item["mask_pixels"],
            "box_xyxy": item["box_xyxy"],
        }
        for item in selected
    ]
    return combined, {
        "variants": variant_infos,
        "selection": selection,
        "reject_slender": bool(reject_slender),
        "max_slender_aspect": float(max_slender_aspect),
        "selected": selected_info,
        "best_score": float(max(item["score"] for item in candidates)),
        "mask_pixels": int(combined.sum()),
    }


def _save_positive_npz(
    output_path: Path,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    positive_score: np.ndarray,
    raw_mask: np.ndarray,
    connected_mask: np.ndarray,
    filled_mask: np.ndarray,
    prompt: str,
    prompt_variants: Sequence[str],
    view_names: Sequence[str],
    args: argparse.Namespace,
) -> None:
    grasp_label = np.full((len(mesh_vertices),), -1, dtype=np.int8)
    grasp_label[filled_mask] = 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        vertices_obj=mesh_vertices.astype(np.float32),
        faces=mesh_faces.astype(np.int32),
        grasp_label=grasp_label,
        positive_score=positive_score.astype(np.float32),
        positive_mask_raw=raw_mask.astype(np.uint8),
        positive_mask_connected=connected_mask.astype(np.uint8),
        positive_mask_filled=filled_mask.astype(np.uint8),
        prompt=np.asarray([prompt], dtype=object),
        prompt_variants=np.asarray(list(prompt_variants), dtype=object),
        view_names=np.asarray(list(view_names), dtype=object),
        sam3_confidence_threshold=np.asarray([args.confidence_threshold], dtype=np.float32),
        min_vote_pixels=np.asarray([args.min_vote_pixels], dtype=np.float32),
        image_size=np.asarray([args.image_size], dtype=np.int32),
        source=np.asarray(["sam3_positive_only_multiview_graph_cleanup"], dtype=object),
    )


def _save_clean_binary_v2(
    output_path: Path,
    annotation_dir: Path,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    positive_mask: np.ndarray,
    positive_score: np.ndarray,
    min_vote_pixels: float,
) -> Dict[str, int]:
    raw_path = annotation_dir / "affordance.npz"
    negative_mask = np.zeros_like(positive_mask, dtype=bool)
    negative_score = np.zeros_like(positive_score, dtype=np.float32)
    if raw_path.exists():
        raw = np.load(raw_path, allow_pickle=True)
        scores = np.asarray(raw["affordance_scores"], dtype=np.float32)
        if scores.shape[1] >= 2:
            negative_score = scores[:, 1]
            negative_mask = negative_score >= min_vote_pixels
    negative_mask = negative_mask & ~positive_mask
    grasp_label = np.full((len(mesh_vertices),), -1, dtype=np.int8)
    grasp_label[negative_mask] = 0
    grasp_label[positive_mask] = 1
    np.savez_compressed(
        output_path,
        vertices_obj=mesh_vertices.astype(np.float32),
        faces=mesh_faces.astype(np.int32),
        grasp_label=grasp_label,
        positive_mask=positive_mask.astype(np.uint8),
        negative_mask=negative_mask.astype(np.uint8),
        ignore_mask=(grasp_label < 0).astype(np.uint8),
        positive_score=positive_score.astype(np.float32),
        negative_score=negative_score.astype(np.float32),
        label_vocab=np.asarray(["ignore", "negative_grasp", "positive_grasp"], dtype=object),
        source=np.asarray(["positive_only_clean_v2_with_conservative_existing_negative"], dtype=object),
    )
    return {
        "clean_v2_positive_vertices": int(positive_mask.sum()),
        "clean_v2_negative_vertices": int(negative_mask.sum()),
        "clean_v2_ignore_vertices": int((grasp_label < 0).sum()),
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

    positive_path = annotation_dir / args.output_name
    if positive_path.exists() and not args.overwrite and not args.render_only:
        return {
            "asset_id": entry["asset_id"],
            "status": "skipped_existing",
            "annotation_path": str(positive_path),
        }

    mesh = load_mesh(mesh_path)
    prompts = _prompt_map(entry)
    positive_prompt = prompts["positive_grasp"]
    positive_variants = _prompt_variants(entry, {"positive_grasp": positive_prompt})["positive_grasp"]
    view_names = [name.strip() for name in args.views.split(",") if name.strip()]
    votes = np.zeros((len(mesh.vertices),), dtype=np.float32)

    render_dir = annotation_dir / "positive_only_views"
    render_dir.mkdir(parents=True, exist_ok=True)
    prompt_infos: Dict[str, Any] = {}
    for view_name in view_names:
        render = render_mesh(mesh, view_name=view_name, image_size=args.image_size, margin=args.margin)
        image_path = render_dir / f"{view_name}.png"
        render.image.save(image_path)
        if processor is None:
            continue

        state = _sam3_mask_for_prompt(processor, render.image, amp_dtype=args.amp_dtype)
        mask, info = _positive_mask_for_variants(
            processor,
            state,
            positive_variants,
            image_size=render.image.size,
            selection=args.mask_selection,
            reject_slender=str(entry.get("label", "")).replace("_", " ") in args.reject_slender_labels_set,
            max_slender_aspect=args.max_slender_aspect,
        )
        mask_path = render_dir / f"{view_name}_positive_grasp_mask.png"
        Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
        face_votes = np.zeros((len(mesh.vertices), 1), dtype=np.float32)
        voted_faces = _vote_vertices(
            votes=face_votes,
            faces=mesh.faces,
            face_id=render.face_id,
            mask=mask,
            affordance_idx=0,
        )
        votes += face_votes[:, 0]
        _overlay_mask(render.image, {"positive_grasp": mask}).save(render_dir / f"{view_name}_overlay.png")
        info.update(
            {
                "prompt": positive_prompt,
                "prompt_variants": positive_variants,
                "mask_path": str(mask_path),
                "voted_faces": voted_faces,
                "mask_pixels": int(mask.sum()),
            }
        )
        prompt_infos[view_name] = info

    raw_mask = votes >= args.min_vote_pixels
    connected_mask, filled_mask, cleanup_info = _cleanup_positive_mask(
        raw_mask,
        mesh.faces,
        min_component_vertices=args.min_component_vertices,
        keep_component_ratio=args.keep_component_ratio,
        dilation_steps=args.dilation_steps,
        hole_fill_steps=args.hole_fill_steps,
    )

    summary: Dict[str, Any] = {
        "asset_id": entry["asset_id"],
        "label": entry.get("label"),
        "mesh_path": str(mesh_path),
        "annotation_dir": str(annotation_dir),
        "positive_prompt": positive_prompt,
        "positive_prompt_variants": positive_variants,
        "views": view_names,
        "render_dir": str(render_dir),
        "render_only": bool(args.render_only),
        "prompt_infos": prompt_infos,
        "num_vertices": int(len(mesh.vertices)),
        **cleanup_info,
    }
    if not args.render_only:
        _save_positive_npz(
            positive_path,
            mesh.vertices,
            mesh.faces,
            votes,
            raw_mask,
            connected_mask,
            filled_mask,
            positive_prompt,
            positive_variants,
            view_names,
            args,
        )
        summary["annotation_path"] = str(positive_path)
        if args.write_clean_binary:
            clean_binary_path = annotation_dir / args.clean_binary_name
            summary.update(
                _save_clean_binary_v2(
                    clean_binary_path,
                    annotation_dir,
                    mesh.vertices,
                    mesh.faces,
                    filled_mask,
                    votes,
                    min_vote_pixels=args.min_vote_pixels,
                )
            )
            summary["clean_binary_path"] = str(clean_binary_path)

    summary_path = annotation_dir / "positive_grasp_only_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", choices=["dextoolbench12", "domino20", "all"], default="dextoolbench12")
    parser.add_argument("--asset-id", action="append", default=[], help="Filter by asset_id or label. Can be repeated.")
    parser.add_argument("--max-assets", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument(
        "--views",
        default="front,back,left,right,top,bottom,iso1,iso2,iso3,iso4,front_top,back_top",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.20)
    parser.add_argument("--min-vote-pixels", type=float, default=8.0)
    parser.add_argument("--min-component-vertices", type=int, default=20)
    parser.add_argument("--keep-component-ratio", type=float, default=0.08)
    parser.add_argument("--dilation-steps", type=int, default=1)
    parser.add_argument("--hole-fill-steps", type=int, default=2)
    parser.add_argument(
        "--mask-selection",
        choices=["best_per_view", "best_per_prompt", "union"],
        default="best_per_view",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp-dtype", default="bfloat16", choices=["bfloat16", "float16", "none"])
    parser.add_argument("--no-hf", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-name", default="positive_grasp_only.npz")
    parser.add_argument("--write-clean-binary", action="store_true")
    parser.add_argument("--clean-binary-name", default="grasp_affordance_clean_v2.npz")
    parser.add_argument("--reject-slender-labels", default="long screwdriver,short screwdriver,screwdriver,knife")
    parser.add_argument("--max-slender-aspect", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reject_slender_labels_set = {
        item.strip().replace("_", " ")
        for item in args.reject_slender_labels.split(",")
        if item.strip()
    }
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
    run_summary_path = output_dir / f"{args.stage}_positive_grasp_only_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    short = {
        "summary_path": str(run_summary_path),
        "stage": args.stage,
        "num_requested": len(entries),
        "num_completed": len(summaries),
        "num_failed": len(failures),
    }
    print(json.dumps(short if args.quiet else run_summary, indent=2))


if __name__ == "__main__":
    main()
