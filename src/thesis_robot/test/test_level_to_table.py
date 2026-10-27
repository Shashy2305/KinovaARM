"""calibration/level_to_table.py: a camera whose pose is wrong in height and tilt is levelled by the table plane, x/y/yaw are left alone."""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'calibration'))
import level_to_table as lt  # noqa: E402

TOP = -0.0125
K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
T_TRUE = np.array([1.03, -0.10, 0.77])


def look_at(position, target):
    f = (np.asarray(target, float) - position)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    return np.column_stack([r, np.cross(f, r), f])                   # camera x right, y down, z forward


R_TRUE = look_at(T_TRUE, [0.45, 0.0, TOP])


def table_depth(R, t, noise=0.002, seed=0):
    """depth image (m) of the horizontal table z = TOP seen by a camera at pose (R, t)"""
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:480, 0:640]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], axis=-1)
    d_base = rays @ R.T
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (TOP - t[2]) / d_base[..., 2]
    depth = np.where((s > 0.3) & (s < 4), s, 0.0)
    return (depth + rng.normal(0, noise, depth.shape)).astype(np.float32)


def test_plane_fit_of_the_true_pose_is_horizontal_at_the_table_height():
    depth = table_depth(R_TRUE, T_TRUE)
    n, c, inl, _ = lt.fit_plane(lt.table_points(depth, K, R_TRUE, T_TRUE))
    assert np.degrees(np.arccos(n[2])) < 0.3 and abs(c[2] - TOP) < 0.005 and inl > 1000


def test_a_tilted_and_low_camera_is_levelled_and_xy_yaw_are_kept():
    depth = table_depth(R_TRUE, T_TRUE)                                    # what the camera really sees
    R_bad = (Rotation.from_euler('xy', [3.0, -2.5], degrees=True) * Rotation.from_matrix(R_TRUE)).as_matrix()      # the calibration is tilted ...
    t_bad = T_TRUE + np.array([0.0, 0.0, -0.028])                           # ... and 2.8 cm off in height
    n, c, _, _ = lt.fit_plane(lt.table_points(depth, K, R_bad, t_bad))
    assert np.degrees(np.arccos(n[2])) > 2.5 and abs(c[2] - TOP) > 0.01     # the plane test sees the mistake
    R2, t2 = lt.level_correction(R_bad, t_bad, n, c, TOP)
    n2, c2, _, _ = lt.fit_plane(lt.table_points(depth, K, R2, t2))
    assert np.degrees(np.arccos(n2[2])) < 0.3 and abs(c2[2] - TOP) < 0.004
    # the correction is a rotation about a point of the plane plus a lift: the rotation about the vertical barely changes
    yaw = lambda R: Rotation.from_matrix(R).as_euler('zyx', degrees=True)[0]      # noqa: E731
    assert abs(yaw(R2) - yaw(R_bad)) < 4.0
    # and the camera ends up close to the truth in height / tilt
    assert abs(t2[2] - T_TRUE[2]) < 0.06


def test_a_level_camera_is_not_moved():
    depth = table_depth(R_TRUE, T_TRUE)
    n, c, _, _ = lt.fit_plane(lt.table_points(depth, K, R_TRUE, T_TRUE))
    R2, t2 = lt.level_correction(R_TRUE, T_TRUE, n, c, TOP)
    assert np.linalg.norm(t2 - T_TRUE) < 0.012 and np.degrees((Rotation.from_matrix(R2).inv() * Rotation.from_matrix(R_TRUE)).magnitude()) < 0.3
