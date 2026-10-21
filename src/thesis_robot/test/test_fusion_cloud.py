"""The fused point cloud of the 3D Fusion tab (dashboard/backend/app/fusion_cloud.py): synthetic cameras looking at a table with a box on it."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dashboard', 'backend'))
from app import fusion_cloud as fc  # noqa: E402

W, H = 640, 480
K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def pose_looking_down(x, y, z):
    """base <- camera for a camera at (x, y, z) looking straight down (optical z = -base z)."""
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    T[:3, 3] = [x, y, z]
    return T


def render_depth(T_base_cam, table_z=0.0, box=None, noise_mm=0.0, rng=None):
    """16UC1 depth image (mm) of the table plane and an optional box (xmin, xmax, ymin, ymax, top_z) as the camera sees them."""
    v, u = np.mgrid[0:H, 0:W]
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, dtype=float)], axis=-1)
    d_base = d_cam @ T_base_cam[:3, :3].T
    o = T_base_cam[:3, 3]
    t = (table_z - o[2]) / d_base[..., 2]
    if box is not None:
        tb = (box[4] - o[2]) / d_base[..., 2]
        hit = o[None, None, :] + tb[..., None] * d_base
        inside = (hit[..., 0] >= box[0]) & (hit[..., 0] <= box[1]) & (hit[..., 1] >= box[2]) & (hit[..., 1] <= box[3]) & (tb > 0)
        t = np.where(inside, tb, t)
    z_cam = t                                              # d_cam has z = 1, so the parameter is the camera-frame depth
    depth = (z_cam * 1000.0 + (rng.normal(0, noise_mm, z_cam.shape) if noise_mm else 0.0))
    return np.clip(depth, 0, 65000).astype(np.uint16)


COLOR = np.full((H, W, 3), 128, np.uint8)


def test_the_same_input_gives_the_same_cloud_no_randomness():
    T = pose_looking_down(0.4, 0.0, 1.0)
    d = render_depth(T)
    a = fc.camera_points([d, d, d], COLOR, K, T)
    b = fc.camera_points([d, d, d], COLOR, K, T)
    assert len(a[0]) > 10000
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    ma = fc.voxel_merge({'realsense': a})
    mb = fc.voxel_merge({'realsense': b})
    assert np.array_equal(ma[0], mb[0])


def test_the_table_comes_out_flat_and_at_the_right_height_in_base_link():
    T = pose_looking_down(0.4, 0.0, 1.0)
    p, _ = fc.camera_points([render_depth(T)] * 3, COLOR, K, T)
    assert abs(float(np.median(p[:, 2]))) < 0.003 and float(np.std(p[:, 2])) < 0.004


def test_temporal_median_removes_a_single_noisy_frame():
    T = pose_looking_down(0.4, 0.0, 1.0)
    good = render_depth(T)
    bad = good.copy()
    bad[::2, ::2] = 2500                                        # one wild frame
    p, _ = fc.camera_points([good, bad, good], COLOR, K, T)
    assert float(np.max(p[:, 2])) < 0.01


def test_flying_pixels_at_a_depth_edge_are_dropped():
    T = pose_looking_down(0.4, 0.0, 1.0)
    d = render_depth(T, box=(0.3, 0.5, -0.1, 0.1, 0.30))
    p, _ = fc.camera_points([d] * 3, COLOR, K, T)
    z = p[:, 2]
    # nothing between the table and the box top except the box's own surface: the (0.30..0.0) gap holds no points
    assert np.sum((z > 0.02) & (z < 0.28)) == 0
    assert np.sum(z > 0.28) > 100


def test_two_aligned_cameras_merge_into_shared_voxels_and_a_shifted_one_shows_its_error():
    Ta, Tb = pose_looking_down(0.35, -0.15, 1.0), pose_looking_down(0.45, 0.15, 1.1)
    pa, ca = fc.camera_points([render_depth(Ta, table_z=0.006)] * 3, COLOR, K, Ta)   # mid-voxel: a table exactly on a voxel boundary splits by float noise
    pb, cb = fc.camera_points([render_depth(Tb, table_z=0.006)] * 3, COLOR, K, Tb)
    p, c, mask = fc.voxel_merge({'realsense': (pa, ca), 'realsense2': (pb, cb)})
    assert (mask == 3).sum() > 0.2 * len(mask)                  # a lot of the table is seen by both
    st = fc.pair_stats({'realsense': (pa, ca), 'realsense2': (pb, cb)})['realsense|realsense2']
    assert st['median_nn_cm'] < 1.5 and st['overlap_pct'] > 85
    # the second camera's calibration is wrong by 6 cm in x: the pair statistic must say so
    Tbad = Tb.copy()
    Tbad[0, 3] += 0.06
    pb2, cb2 = fc.camera_points([render_depth(Tb)] * 3, COLOR, K, Tbad)                          # rendered with the right pose, placed with the wrong one
    # a shift along a flat table is invisible to a nearest-point test, so tilt it instead: 3 degrees about y
    th = np.radians(3.0)
    Rt = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    Ttilt = Tb.copy()
    Ttilt[:3, :3] = Rt @ Tb[:3, :3]
    pb3, cb3 = fc.camera_points([render_depth(Tb)] * 3, COLOR, K, Ttilt)
    st_bad = fc.pair_stats({'realsense': (pa, ca), 'realsense2': (pb3, cb3)})['realsense|realsense2']
    assert st_bad['median_nn_cm'] > 3 * st['median_nn_cm'] or st_bad['overlap_pct'] < st['overlap_pct'] - 20


def test_colour_modes_mark_the_cameras():
    mask = np.array([1, 2, 4, 3, 7], dtype=np.int32)
    rgb = np.zeros((5, 3), np.float32)
    cam = fc.colorize(rgb, mask, 'camera')
    assert np.allclose(cam[3], (1.0, 1.0, 1.0)) and np.allclose(cam[0], fc.CAMERA_COLORS['realsense'])
    ov = fc.colorize(rgb, mask, 'overlap')
    assert np.allclose(ov[0], 0.35) and np.allclose(ov[4], (0.2, 0.95, 0.35)) and np.allclose(ov[3], (1.0, 0.85, 0.2))
    assert fc.colorize(rgb, mask, 'rgb') is rgb
