import numpy as np
from scipy.spatial.transform import Rotation as R

from thesis_robot import wrist_servo as ws

K = np.array([[1297.7, 0, 620.9], [0, 1298.6, 238.3], [0, 0, 1]])


def camera_over(flange_xy, flange_z):
    """Wrist camera pose for a flange pointing straight down at (x, y, z): the
    optical axis is the tool axis (down), mounted 56 mm off-axis (URDF camera_module)."""
    R_ee = R.from_euler('x', 180, degrees=True)                 # tool z points down
    R_cam = R_ee * R.from_euler('z', 180, degrees=True)         # camera_module rpy 0 0 pi
    t = np.array([flange_xy[0], flange_xy[1], flange_z]) + R_ee.apply([0, 0.05639, 0.01305])
    return R_cam.as_matrix(), t


def project(P, Rm, t):
    c = Rm.T @ (np.asarray(P) - t)
    return K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]


def test_recovers_object_position_on_the_plane():
    Rm, t = camera_over((0.45, 0.14), 0.40)
    plane = ws.plane_height('cup', -0.0125)
    for obj in [(0.42, 0.12), (0.47, 0.17), (0.40, 0.10)]:
        u, v = project([obj[0], obj[1], plane], Rm, t)
        x, y = ws.pixel_to_plane_xy(u, v, K, Rm, t, plane)
        assert abs(x - obj[0]) < 1e-6 and abs(y - obj[1]) < 1e-6


def test_wrong_height_assumption_gives_a_small_error_when_looking_down():
    Rm, t = camera_over((0.42, 0.12), 0.40)
    u, v = project([0.45, 0.14, 0.04], Rm, t)                   # object top really at z=0.04
    x, y = ws.pixel_to_plane_xy(u, v, K, Rm, t, 0.0775)          # we assumed 0.0775
    assert np.hypot(x - 0.45, y - 0.14) < 0.012


def test_ray_that_misses_the_plane_is_rejected():
    Rm, t = camera_over((0.42, 0.12), 0.40)
    assert ws.pixel_to_plane_xy(620 + 1297 * 20, 238, K, Rm, t, 0.05) is None   # almost parallel to the table
    assert ws.pixel_to_plane_xy(620, 238, K, Rm, t, 0.9) is None          # plane behind the camera


def test_step_is_clipped_and_reports_the_true_distance():
    dx, dy, dist = ws.clipped_step((0.5, 0.2), (0.4, 0.2), 0.06)
    assert abs(dx - 0.06) < 1e-9 and dy == 0 and abs(dist - 0.10) < 1e-9
    dx, dy, dist = ws.clipped_step((0.41, 0.2), (0.4, 0.2), 0.06)
    assert abs(dx - 0.01) < 1e-9


def test_detection_choice():
    dets = [{'label': 'cup', 'confidence': 0.9, 'u': 100, 'v': 100},
            {'label': 'cup', 'confidence': 0.5, 'u': 640, 'v': 360},
            {'label': 'mouse', 'confidence': 0.99, 'u': 0, 'v': 0}]
    assert ws.pick_detection(dets, 'cup')['confidence'] == 0.9
    assert ws.pick_detection(dets, 'cup', expected_uv=(650, 350))['confidence'] == 0.5
    assert ws.pick_detection(dets, 'bottle') is None


def test_search_pattern_is_small_and_covers_all_sides():
    offs = ws.search_offsets(0.04)
    assert all(max(abs(a), abs(b)) <= 0.04 for a, b in offs)
    assert {(0.04, 0), (-0.04, 0), (0, 0.04), (0, -0.04)} <= set(offs)


def test_handle_makes_the_box_wider_than_tall():
    assert ws.needs_quarter_turn((447, 232, 990, 646))        # the mug seen from above, handle on the left
    assert not ws.needs_quarter_turn((300, 200, 700, 600))     # round object
    assert not ws.needs_quarter_turn((300, 200, 700, 800))     # long along y: already fine


def test_quarter_turn_stays_inside_the_joint_limit():
    import math
    assert abs(ws.quarter_turn_target(0.0, 2.7)) == math.pi / 2
    assert ws.quarter_turn_target(1.57, 2.7) == 1.57 - 3.141592653589793 / 2     # +90 would hit 3.14
    assert ws.quarter_turn_target(2.6, 2.7) < 2.6
    assert ws.quarter_turn_target(2.0, 0.3) is None
