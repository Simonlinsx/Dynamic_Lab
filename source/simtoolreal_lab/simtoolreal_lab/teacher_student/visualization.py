"""Lightweight visualizations for deployable student observations."""

from __future__ import annotations

import numpy as np


_BACKGROUND = np.asarray((17, 21, 27), dtype=np.uint8)
_BORDER = np.asarray((184, 194, 204), dtype=np.uint8)
_AXIS_COLORS = (
    np.asarray((238, 92, 88), dtype=np.uint8),
    np.asarray((84, 205, 126), dtype=np.uint8),
    np.asarray((78, 149, 239), dtype=np.uint8),
)


def _draw_line(
    image: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    color: np.ndarray,
    width: int = 1,
) -> None:
    length = max(int(np.ceil(np.linalg.norm(np.subtract(end, start)))), 1)
    xs = np.rint(np.linspace(start[0], end[0], length + 1)).astype(np.int64)
    ys = np.rint(np.linspace(start[1], end[1], length + 1)).astype(np.int64)
    radius = max(int(width) // 2, 0)
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            px = xs + offset_x
            py = ys + offset_y
            inside = (
                (px >= 0)
                & (px < image.shape[1])
                & (py >= 0)
                & (py < image.shape[0])
            )
            image[py[inside], px[inside]] = color


def _view_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eye = np.asarray((0.34, -0.34, 0.24), dtype=np.float32)
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0), dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up, forward


def _project_points(
    points: np.ndarray,
    width: int,
    height: int,
    view_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    right, up, forward = _view_basis()
    scale = 0.44 * float(min(width, height)) / max(float(view_range), 1.0e-6)
    projected = np.empty((points.shape[0], 2), dtype=np.float32)
    projected[:, 0] = float(width) * 0.5 + points @ right * scale
    projected[:, 1] = float(height) * 0.5 - points @ up * scale
    depth = points @ forward
    return projected, depth


def _point_colors(features: np.ndarray) -> np.ndarray:
    if features.shape[1] < 6:
        return np.tile(np.asarray((66, 211, 238), dtype=np.uint8), (features.shape[0], 1))
    rgb = np.nan_to_num(features[:, 3:6], nan=0.0, posinf=1.0, neginf=0.0)
    if rgb.size > 0 and float(np.max(rgb)) > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return (32.0 + 223.0 * rgb).astype(np.uint8)


def render_pointcloud_panel(
    features: np.ndarray,
    valid: np.ndarray,
    *,
    resolution: tuple[int, int] = (320, 192),
    view_range: float = 0.30,
    point_radius: int = 2,
) -> np.ndarray:
    """Render a palm-frame point cloud with fixed axes and camera geometry."""

    width, height = (int(resolution[0]), int(resolution[1]))
    if width <= 0 or height <= 0:
        raise ValueError("Point-cloud panel resolution must be positive.")
    if float(view_range) <= 0.0:
        raise ValueError("Point-cloud view range must be positive.")
    features = np.asarray(features, dtype=np.float32)
    valid = np.asarray(valid).reshape(-1) > 0.5
    if features.ndim != 2 or features.shape[1] < 3:
        raise ValueError(f"Expected point features shaped [N, >=3], got {features.shape}.")
    if valid.shape[0] != features.shape[0]:
        raise ValueError("Point features and valid mask must contain the same number of points.")

    panel = np.empty((height, width, 3), dtype=np.uint8)
    panel[:] = _BACKGROUND
    panel[[0, -1], :] = _BORDER
    panel[:, [0, -1]] = _BORDER

    axis_length = min(float(view_range) * 0.34, 0.10)
    axis_points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (axis_length, 0.0, 0.0),
            (0.0, axis_length, 0.0),
            (0.0, 0.0, axis_length),
        ),
        dtype=np.float32,
    )
    projected_axes, _ = _project_points(axis_points, width, height, float(view_range))
    origin = tuple(projected_axes[0])
    for axis_index, color in enumerate(_AXIS_COLORS, start=1):
        _draw_line(panel, origin, tuple(projected_axes[axis_index]), color, width=2)

    finite = np.isfinite(features[:, :3]).all(axis=1)
    selected = valid & finite
    selected_features = features[selected]
    if selected_features.shape[0] > 0:
        projected, depth = _project_points(
            selected_features[:, :3], width, height, float(view_range)
        )
        colors = _point_colors(selected_features)
        visible = (
            np.isfinite(projected).all(axis=1)
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < float(width))
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < float(height))
        )
        order = np.argsort(depth)[::-1]
        radius = max(int(point_radius), 1)
        for point_index in order:
            if not visible[point_index]:
                continue
            px, py = np.rint(projected[point_index]).astype(np.int64)
            x0 = max(int(px) - radius, 1)
            x1 = min(int(px) + radius + 1, width - 1)
            y0 = max(int(py) - radius, 1)
            y1 = min(int(py) + radius + 1, height - 1)
            panel[y0:y1, x0:x1] = colors[point_index]

    ratio = float(selected.sum()) / max(int(valid.shape[0]), 1)
    bar_x0 = max(width - 7, 1)
    bar_x1 = max(width - 3, bar_x0 + 1)
    bar_y0 = 5
    bar_y1 = max(height - 5, bar_y0 + 1)
    panel[bar_y0:bar_y1, bar_x0:bar_x1] = np.asarray((52, 59, 68), dtype=np.uint8)
    fill = int(round((bar_y1 - bar_y0) * ratio))
    if fill > 0:
        panel[bar_y1 - fill : bar_y1, bar_x0:bar_x1] = np.asarray(
            (73, 210, 132), dtype=np.uint8
        )
    return panel


def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    y_indices = np.minimum(
        np.floor(np.linspace(0, image.shape[0], height, endpoint=False)).astype(np.int64),
        image.shape[0] - 1,
    )
    x_indices = np.minimum(
        np.floor(np.linspace(0, image.shape[1], width, endpoint=False)).astype(np.int64),
        image.shape[1] - 1,
    )
    return image[y_indices[:, None], x_indices[None, :]]


def add_pointcloud_inset(
    frame: np.ndarray,
    panel: np.ndarray,
    *,
    margin: int = 12,
) -> np.ndarray:
    """Place a point-cloud panel in the top-right corner without mutating inputs."""

    frame = np.asarray(frame)
    panel = np.asarray(panel)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame shaped [H, W, 3], got {frame.shape}.")
    if panel.ndim != 3 or panel.shape[2] != 3:
        raise ValueError(f"Expected RGB panel shaped [H, W, 3], got {panel.shape}.")
    margin = max(int(margin), 0)
    max_width = max(frame.shape[1] - 2 * margin, 1)
    max_height = max(frame.shape[0] - 2 * margin, 1)
    scale = min(1.0, max_width / panel.shape[1], max_height / panel.shape[0])
    target_width = max(int(round(panel.shape[1] * scale)), 1)
    target_height = max(int(round(panel.shape[0] * scale)), 1)
    if target_width != panel.shape[1] or target_height != panel.shape[0]:
        panel = _resize_nearest(panel, target_width, target_height)
    x0 = max(frame.shape[1] - margin - target_width, 0)
    y0 = min(margin, max(frame.shape[0] - target_height, 0))
    output = frame.copy()
    output[y0 : y0 + target_height, x0 : x0 + target_width] = panel
    return output
