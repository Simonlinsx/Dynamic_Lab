"""Point-cloud helpers for teacher-student dataset export."""

from __future__ import annotations

import torch


def quat_rotate(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in wxyz convention."""

    qw = quat_wxyz[..., :1]
    qvec = quat_wxyz[..., 1:]
    uv = torch.cross(qvec.expand_as(vec), vec, dim=-1)
    uuv = torch.cross(qvec.expand_as(vec), uv, dim=-1)
    return vec + 2.0 * (qw * uv + uuv)


def quat_rotate_inverse(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Inverse-rotate vectors by quaternions in wxyz convention."""

    conj = torch.cat([quat_wxyz[..., :1], -quat_wxyz[..., 1:]], dim=-1)
    return quat_rotate(conj, vec)


def sample_box_surface_points(
    size: tuple[float, float, float],
    num_points: int,
    device: torch.device | str,
    *,
    affordance_mode: str = "side_grasp",
    affordance_positive_fraction: float = 0.38,
    affordance_negative_fraction: float = 0.45,
    affordance_positive_end: str = "negative",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample local-frame box surface points and affordance labels."""

    points = torch.rand((num_points, 3), device=device) - 0.5
    face_ids = torch.arange(num_points, device=device) % 6
    axes = torch.div(face_ids, 2, rounding_mode="floor")
    signs = torch.where(face_ids % 2 == 0, -1.0, 1.0)
    row_ids = torch.arange(num_points, device=device)
    points[row_ids, axes] = 0.5 * signs

    size_tensor = torch.tensor(size, dtype=points.dtype, device=device)
    points = points * size_tensor.unsqueeze(0)

    labels = box_affordance_labels_from_local_points(
        size,
        points,
        affordance_mode=affordance_mode,
        affordance_positive_fraction=affordance_positive_fraction,
        affordance_negative_fraction=affordance_negative_fraction,
        affordance_positive_end=affordance_positive_end,
    )
    return points, labels


def sample_unit_primitive_surface_points(
    num_points: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample normalized sphere, box, cylinder, and cone surfaces.

    The returned tensor has shape ``[4, N, 3]`` and follows the environment's
    shape-code contract: sphere=0, box=1, cylinder=2, cone=3. Every primitive
    fits inside a unit-size bounding box centered at the origin.
    """

    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    dtype = torch.float32
    sphere = torch.randn((num_points, 3), dtype=dtype, device=device)
    sphere = 0.5 * sphere / torch.linalg.vector_norm(sphere, dim=-1, keepdim=True).clamp_min(1.0e-6)

    box, _ = sample_box_surface_points((1.0, 1.0, 1.0), num_points, device)

    theta = 2.0 * torch.pi * torch.rand(num_points, dtype=dtype, device=device)
    cylinder = torch.empty((num_points, 3), dtype=dtype, device=device)
    cap_count = max(num_points // 4, 1)
    side_count = num_points - cap_count
    cylinder[:side_count, 0] = 0.5 * torch.cos(theta[:side_count])
    cylinder[:side_count, 1] = 0.5 * torch.sin(theta[:side_count])
    cylinder[:side_count, 2] = torch.rand(side_count, dtype=dtype, device=device) - 0.5
    cap_radius = 0.5 * torch.sqrt(torch.rand(cap_count, dtype=dtype, device=device))
    cylinder[side_count:, 0] = cap_radius * torch.cos(theta[side_count:])
    cylinder[side_count:, 1] = cap_radius * torch.sin(theta[side_count:])
    cylinder[side_count:, 2] = torch.where(
        torch.arange(cap_count, device=device) % 2 == 0,
        torch.full((cap_count,), -0.5, dtype=dtype, device=device),
        torch.full((cap_count,), 0.5, dtype=dtype, device=device),
    )

    cone = torch.empty((num_points, 3), dtype=dtype, device=device)
    cone_side_count = num_points - cap_count
    cone_z = torch.rand(cone_side_count, dtype=dtype, device=device) - 0.5
    cone_radius = 0.5 * (0.5 - cone_z)
    cone[:cone_side_count, 0] = cone_radius * torch.cos(theta[:cone_side_count])
    cone[:cone_side_count, 1] = cone_radius * torch.sin(theta[:cone_side_count])
    cone[:cone_side_count, 2] = cone_z
    cone_cap_radius = 0.5 * torch.sqrt(torch.rand(cap_count, dtype=dtype, device=device))
    cone[cone_side_count:, 0] = cone_cap_radius * torch.cos(theta[cone_side_count:])
    cone[cone_side_count:, 1] = cone_cap_radius * torch.sin(theta[cone_side_count:])
    cone[cone_side_count:, 2] = -0.5

    return torch.stack((sphere, box, cylinder, cone), dim=0)


def primitive_surface_points_for_envs(
    unit_points_by_shape: torch.Tensor,
    shape_codes: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    """Scale normalized primitive surfaces for each active environment asset."""

    if unit_points_by_shape.ndim != 3 or unit_points_by_shape.shape[0] != 4:
        raise ValueError(
            "unit_points_by_shape must have shape [4,N,3], "
            f"got {tuple(unit_points_by_shape.shape)}"
        )
    if shape_codes.ndim != 1 or sizes.shape != (shape_codes.shape[0], 3):
        raise ValueError(
            "shape_codes/sizes must have shapes [E] and [E,3], "
            f"got {tuple(shape_codes.shape)} and {tuple(sizes.shape)}"
        )
    selected = unit_points_by_shape[shape_codes.long().clamp(min=0, max=3)]
    return selected * sizes.to(dtype=selected.dtype).unsqueeze(1)


def box_affordance_labels_from_local_points(
    size: tuple[float, float, float] | torch.Tensor,
    local_points: torch.Tensor,
    *,
    affordance_mode: str = "side_grasp",
    affordance_positive_fraction: float = 0.38,
    affordance_negative_fraction: float = 0.45,
    affordance_positive_end: str = "negative",
) -> torch.Tensor:
    """Assign affordance labels to box-surface points in object frame.

    Labels use the student dataset convention: 1 means positive/graspable,
    0 means explicitly negative, and -1 means unlabeled/ignored.
    """

    size_tensor = torch.as_tensor(size, dtype=local_points.dtype, device=local_points.device)
    half_size = torch.clamp(0.5 * size_tensor, min=1.0e-6)
    long_axis = int(torch.argmax(size_tensor).item())
    mode = str(affordance_mode).lower()

    if mode in {"handle_blade", "knife_handle_blade", "baton_handle_blade"}:
        long_coord = local_points[..., long_axis]
        long_size = torch.clamp(size_tensor[long_axis], min=1.0e-6)
        long_half = torch.clamp(half_size[long_axis], min=1.0e-6)
        positive_fraction = min(max(float(affordance_positive_fraction), 0.0), 1.0)
        negative_fraction = min(max(float(affordance_negative_fraction), 0.0), 1.0)
        positive_width = positive_fraction * long_size
        negative_width = negative_fraction * long_size
        labels = torch.full_like(long_coord, -1.0)
        if str(affordance_positive_end).lower() in {"positive", "+", "pos", "tip"}:
            positive_mask = long_coord >= (long_half - positive_width)
            negative_mask = long_coord <= (-long_half + negative_width)
        else:
            positive_mask = long_coord <= (-long_half + positive_width)
            negative_mask = long_coord >= (long_half - negative_width)
        labels[negative_mask] = 0.0
        labels[positive_mask] = 1.0
        return labels

    if mode in {"all_positive", "positive"}:
        return torch.ones(local_points.shape[:-1], dtype=local_points.dtype, device=local_points.device)
    if mode in {"none", "unlabeled"}:
        return torch.full(local_points.shape[:-1], -1.0, dtype=local_points.dtype, device=local_points.device)

    normalized = torch.abs(local_points) / half_size.view(*((1,) * (local_points.ndim - 1)), 3)
    face_axes = torch.argmax(normalized, dim=-1)
    long_half = torch.clamp(half_size[long_axis], min=1.0e-6)
    side_faces = face_axes != long_axis
    centered_on_long_axis = torch.abs(local_points[..., long_axis]) < 0.82 * long_half
    return (side_faces & centered_on_long_axis).to(dtype=local_points.dtype)


def object_points_in_world_frame(
    object_local_points: torch.Tensor,
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Transform object-local points into world frame for each environment."""

    if object_local_points.ndim == 2:
        local = object_local_points.unsqueeze(0).expand(object_pos_w.shape[0], -1, -1)
    elif object_local_points.ndim == 3 and object_local_points.shape[0] == object_pos_w.shape[0]:
        local = object_local_points
    else:
        raise ValueError(
            "object_local_points must have shape [N,3] or [E,N,3], "
            f"got {tuple(object_local_points.shape)}"
        )
    return quat_rotate(object_quat_w.unsqueeze(1), local) + object_pos_w.unsqueeze(1)


def object_points_in_palm_frame(
    object_local_points: torch.Tensor,
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
    palm_pos_w: torch.Tensor,
    palm_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Transform object-local points into each environment's palm frame."""

    points_w = object_points_in_world_frame(object_local_points, object_pos_w, object_quat_w)
    return quat_rotate_inverse(palm_quat_w.unsqueeze(1), points_w - palm_pos_w.unsqueeze(1))


def rigid_object_point_flow_in_palm_frame(
    points_w: torch.Tensor,
    object_pos_w: torch.Tensor,
    object_lin_vel_w: torch.Tensor,
    object_ang_vel_w: torch.Tensor,
    palm_quat_w: torch.Tensor,
    palm_lin_vel_w: torch.Tensor,
) -> torch.Tensor:
    """Approximate point flow from object rigid velocity, expressed in palm frame."""

    rel_point = points_w - object_pos_w.unsqueeze(1)
    point_vel_w = object_lin_vel_w.unsqueeze(1) + torch.cross(object_ang_vel_w.unsqueeze(1), rel_point, dim=-1)
    rel_vel_w = point_vel_w - palm_lin_vel_w.unsqueeze(1)
    return quat_rotate_inverse(palm_quat_w.unsqueeze(1), rel_vel_w)


def masked_rgbd_object_points_in_palm_frame(
    object_local_points: torch.Tensor,
    object_size: tuple[float, float, float] | torch.Tensor,
    depth: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_w_ros: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
    palm_pos_w: torch.Tensor,
    palm_quat_w: torch.Tensor,
    *,
    num_points: int,
    rgb: torch.Tensor | None = None,
    mask_dilation: int = 1,
    depth_tolerance: float = 0.035,
    affordance_mode: str = "side_grasp",
    affordance_positive_fraction: float = 0.38,
    affordance_negative_fraction: float = 0.45,
    affordance_positive_end: str = "negative",
) -> dict[str, torch.Tensor]:
    """Create masked RGB-D object point clouds using projected object geometry as an oracle mask.

    The returned point cloud is expressed in the palm frame and keeps a fixed
    number of slots per environment. Invalid padded slots have valid=0 and
    affordance label -1.
    """

    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth_image = depth[..., 0]
    elif depth.ndim == 3:
        depth_image = depth
    else:
        raise ValueError(f"depth must have shape [E,H,W] or [E,H,W,1], got {tuple(depth.shape)}")

    num_envs, height, width = depth_image.shape
    device = depth_image.device
    dtype = depth_image.dtype
    projected_points_w = object_points_in_world_frame(object_local_points, object_pos_w, object_quat_w)
    projected_points_c = quat_rotate_inverse(
        camera_quat_w_ros.unsqueeze(1),
        projected_points_w - camera_pos_w.unsqueeze(1),
    )

    if rgb is not None:
        if rgb.ndim != 4 or rgb.shape[:3] != (num_envs, height, width) or rgb.shape[-1] < 3:
            raise ValueError(
                "rgb must have shape [E,H,W,C>=3] matching depth, "
                f"got {tuple(rgb.shape)}"
            )
        rgb_image = rgb[..., :3]
    else:
        rgb_image = None

    points_palm = torch.zeros((num_envs, num_points, 3), dtype=dtype, device=device)
    points_w = torch.zeros_like(points_palm)
    colors = torch.zeros_like(points_palm)
    valid = torch.zeros((num_envs, num_points), dtype=dtype, device=device)
    labels = torch.full((num_envs, num_points), -1.0, dtype=dtype, device=device)
    masks = torch.zeros((num_envs, height, width), dtype=torch.bool, device=device)

    for env_id in range(num_envs):
        intr = camera_intrinsics[env_id]
        fx = torch.clamp(intr[0, 0], min=1.0e-6)
        fy = torch.clamp(intr[1, 1], min=1.0e-6)
        cx = intr[0, 2]
        cy = intr[1, 2]

        points_c = projected_points_c[env_id]
        z = points_c[:, 2]
        u = torch.round(points_c[:, 0] * fx / torch.clamp(z, min=1.0e-6) + cx).long()
        v = torch.round(points_c[:, 1] * fy / torch.clamp(z, min=1.0e-6) + cy).long()
        in_view = (z > 1.0e-4) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not bool(in_view.any()):
            continue

        u_valid = u[in_view]
        v_valid = v[in_view]
        z_valid = z[in_view].to(dtype=dtype)
        z_buffer = torch.full((height * width,), float("inf"), dtype=dtype, device=device)
        radius = max(int(mask_dilation), 0)
        flat_indices: list[torch.Tensor] = []
        flat_depths: list[torch.Tensor] = []
        for dv in range(-radius, radius + 1):
            for du in range(-radius, radius + 1):
                shifted_u = torch.clamp(u_valid + du, min=0, max=width - 1)
                shifted_v = torch.clamp(v_valid + dv, min=0, max=height - 1)
                flat_indices.append(shifted_v * width + shifted_u)
                flat_depths.append(z_valid)
        flat_index = torch.cat(flat_indices, dim=0)
        flat_depth = torch.cat(flat_depths, dim=0)
        if hasattr(z_buffer, "scatter_reduce_"):
            z_buffer.scatter_reduce_(0, flat_index, flat_depth, reduce="amin", include_self=True)
        else:
            for index, value in zip(flat_index.tolist(), flat_depth.tolist()):
                if value < float(z_buffer[index]):
                    z_buffer[index] = value

        mask = torch.isfinite(z_buffer).view(height, width)
        masks[env_id] = mask
        pixel_vu = torch.nonzero(mask, as_tuple=False)
        if pixel_vu.numel() == 0:
            continue

        pix_v = pixel_vu[:, 0]
        pix_u = pixel_vu[:, 1]
        rendered_depth = depth_image[env_id, pix_v, pix_u]
        expected_depth = z_buffer[pix_v * width + pix_u]
        finite_depth = torch.isfinite(rendered_depth) & (rendered_depth > 1.0e-4)
        visible = finite_depth & (torch.abs(rendered_depth - expected_depth) <= float(depth_tolerance))
        if not bool(visible.any()):
            continue

        pix_v = pix_v[visible]
        pix_u = pix_u[visible]
        rendered_depth = rendered_depth[visible]
        count = int(rendered_depth.shape[0])
        if count >= num_points:
            sample_ids = torch.linspace(0, count - 1, num_points, device=device).round().long()
            out_count = num_points
        else:
            sample_ids = torch.arange(count, device=device)
            out_count = count

        sample_u = pix_u[sample_ids].to(dtype=dtype)
        sample_v = pix_v[sample_ids].to(dtype=dtype)
        sample_depth = rendered_depth[sample_ids].to(dtype=dtype)
        x = (sample_u - cx) * sample_depth / fx
        y = (sample_v - cy) * sample_depth / fy
        z_sample = sample_depth
        sampled_c = torch.stack((x, y, z_sample), dim=-1)
        sampled_w = quat_rotate(camera_quat_w_ros[env_id].unsqueeze(0), sampled_c) + camera_pos_w[env_id].unsqueeze(0)
        sampled_palm = quat_rotate_inverse(
            palm_quat_w[env_id].unsqueeze(0),
            sampled_w - palm_pos_w[env_id].unsqueeze(0),
        )
        sampled_local = quat_rotate_inverse(
            object_quat_w[env_id].unsqueeze(0),
            sampled_w - object_pos_w[env_id].unsqueeze(0),
        )
        if rgb_image is not None:
            sampled_rgb = rgb_image[env_id, pix_v[sample_ids], pix_u[sample_ids], :3].to(dtype=dtype)
            if sampled_rgb.numel() > 0 and float(sampled_rgb.max().detach().cpu()) > 1.0:
                sampled_rgb = sampled_rgb / 255.0
            colors[env_id, :out_count] = sampled_rgb[:out_count]

        points_palm[env_id, :out_count] = sampled_palm[:out_count]
        points_w[env_id, :out_count] = sampled_w[:out_count]
        valid[env_id, :out_count] = 1.0
        env_object_size = (
            object_size[env_id]
            if torch.is_tensor(object_size) and object_size.ndim == 2
            else object_size
        )
        labels[env_id, :out_count] = box_affordance_labels_from_local_points(
            env_object_size,
            sampled_local[:out_count],
            affordance_mode=affordance_mode,
            affordance_positive_fraction=affordance_positive_fraction,
            affordance_negative_fraction=affordance_negative_fraction,
            affordance_positive_end=affordance_positive_end,
        )

    return {
        "points_palm": points_palm,
        "points_w": points_w,
        "colors": colors,
        "valid": valid,
        "affordance_labels": labels,
        "object_mask": masks,
    }
