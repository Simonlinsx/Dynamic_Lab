import numpy as np

from simtoolreal_lab.teacher_student.visualization import (
    add_pointcloud_inset,
    render_pointcloud_panel,
)


def test_render_pointcloud_panel_uses_valid_rgb_points():
    features = np.asarray(
        [
            [0.05, 0.00, 0.02, 1.0, 0.0, 0.0],
            [0.00, 0.05, 0.03, 0.0, 1.0, 0.0],
            [-0.04, 0.00, 0.01, 0.0, 0.0, 1.0],
            [0.00, -0.04, 0.02, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    valid = np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32)

    panel = render_pointcloud_panel(
        features,
        valid,
        resolution=(160, 96),
        view_range=0.20,
        point_radius=2,
    )

    assert panel.shape == (96, 160, 3)
    assert panel.dtype == np.uint8
    assert float(panel.std()) > 5.0
    assert np.any((panel[..., 0] > 200) & (panel[..., 1] < 120))
    assert np.any((panel[..., 1] > 180) & (panel[..., 0] < 150))


def test_add_pointcloud_inset_places_panel_top_right_without_mutating_frame():
    frame = np.full((120, 200, 3), 127, dtype=np.uint8)
    panel = np.full((40, 60, 3), (12, 34, 56), dtype=np.uint8)

    output = add_pointcloud_inset(frame, panel, margin=8)

    assert np.all(frame == 127)
    assert output.shape == frame.shape
    assert np.array_equal(output[8:48, 132:192], panel)
    assert np.all(output[:8] == 127)
