#!/usr/bin/env python3
"""
level_to_table.py -- use the TABLE as a ruler for a static camera's height and tilt. READ-ONLY; writes a candidate file, installs nothing.

    source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
    python3 calibration/level_to_table.py --camera realsense2 [--file ~/.ros/realsense2_calibration.yaml] [--out CANDIDATE.yaml]

The table top is flat, horizontal in base_link, and its height was measured independently (~/.ros/table_geometry.yaml, the finger pads touching it). A camera's depth image of it, put through the
camera's extrinsic, must therefore be a plane at that height with zero tilt. The fit measures the plane (RANSAC on depth points in the table region) and corrects exactly three of the six extrinsic
degrees of freedom: the camera's height and its roll/pitch (a rotation about the point of the plane under the table centre). x, y and the rotation about the vertical stay as they were: the table
cannot see them.
Found 2026-10-26: RS2 2.8 cm too low, RS1 tilted 3.9 deg (about 5 cm over a metre) although RS1 passed the
arm-ruler check, which only measures distance along the view direction.
Pure functions (tested): fit_plane, level_correction. Check the result with calibration/check_arm_depth.py and by comparing the same objects seen by two cameras.
"""
import argparse
import os
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

TABLE_REGION = ((0.15, 0.75), (-0.45, 0.45))        # x, y range in base_link (m): in front of the robot, where the table is clear
Z_RANGE = (-0.12, 0.04)
CENTRE_XY = (0.45, 0.0)

TOPICS = {
    'realsense': '/global_camera/global_camera',
    'realsense2': '/global_camera_2/global_camera_2',
    'realsense3': '/global_camera_3/global_camera_3',
}
FILES = {'realsense': '~/.ros/realsense_calibration.yaml', 'realsense2': '~/.ros/realsense2_calibration.yaml',
         'realsense3': '~/.ros/realsense3_calibration.yaml'}


def fit_plane(points, tol=0.01, iterations=300, seed=0):
    """RANSAC plane through points (n,3): returns (normal pointing up (3,), point on the plane at CENTRE_XY (3,), inlier count, inlier points)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        s = points[rng.choice(len(points), 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(nrm)
        if norm < 1e-9:
            continue
        nrm = nrm / norm
        if nrm[2] < 0:
            nrm = -nrm
        inl = int((np.abs((points - s[0]) @ nrm) < tol).sum())
        if best is None or inl > best[0]:
            best = (inl, s[0], nrm)
    inl, p0, nrm = best
    M = points[np.abs((points - p0) @ nrm) < 1.2 * tol]
    A = np.c_[M[:, 0], M[:, 1], np.ones(len(M))]
    coef, *_ = np.linalg.lstsq(A, M[:, 2], rcond=None)                 # z = a x + b y + c over the inliers
    n = np.array([-coef[0], -coef[1], 1.0])
    n /= np.linalg.norm(n)
    c = np.array([CENTRE_XY[0], CENTRE_XY[1], coef[0] * CENTRE_XY[0] + coef[1] * CENTRE_XY[1] + coef[2]])
    return n, c, len(M), M


def level_correction(R_bc, t_bc, normal, centre, table_top_z):
    """New (R, t) of the camera so that the plane (normal, through centre) becomes horizontal at z = table_top_z: rotate about `centre` (normal to +z), then lift."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        Rc = np.eye(3)
    else:
        Rc = Rotation.from_rotvec(axis / s * np.arctan2(s, float(normal @ z))).as_matrix()
    dz = np.array([0.0, 0.0, table_top_z - centre[2]])
    R_new = Rc @ R_bc
    t_new = Rc @ (np.asarray(t_bc) - centre) + centre + dz
    return R_new, t_new


def load_file(path):
    y = yaml.safe_load(open(os.path.expanduser(path)))
    return Rotation.from_quat([y['rotation_quat'][k] for k in 'xyzw']).as_matrix(), np.array([y['translation'][k] for k in 'xyz']), y


def table_points(depth_m, K, R, t, step=3):
    h, w = depth_m.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth_m[v, u]
    ok = z > 0.3
    x = (u[ok] - K[0, 2]) * z[ok] / K[0, 0]
    y = (v[ok] - K[1, 2]) * z[ok] / K[1, 1]
    P = (R @ np.vstack([x, y, z[ok]])).T + t
    (x0, x1), (y0, y1) = TABLE_REGION
    box = (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1) & (P[:, 2] > Z_RANGE[0]) & (P[:, 2] < Z_RANGE[1])
    return P[box]


def grab(camera, timeout=10.0):
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    rclpy.init()
    node = Node('level_to_table')
    got = {}
    ns = TOPICS[camera]
    node.create_subscription(Image, ns + '/aligned_depth_to_color/image_raw', lambda m: got.__setitem__('d', m), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, ns + '/aligned_depth_to_color/camera_info', lambda m: got.__setitem__('k', m), qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < timeout and not ('d' in got and 'k' in got):
        rclpy.spin_once(node, timeout_sec=0.1)
    rclpy.shutdown()
    if not ('d' in got and 'k' in got):
        raise SystemExit(f'no depth from {camera}')
    return CvBridge().imgmsg_to_cv2(got['d'], '16UC1').astype(np.float32) / 1000.0, np.array(got['k'].k).reshape(3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', required=True, choices=list(TOPICS))
    ap.add_argument('--file', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--table-top', type=float, default=None, help='table top height in base_link (default: ~/.ros/table_geometry.yaml)')
    a = ap.parse_args()
    top = a.table_top
    if top is None:
        top = float(yaml.safe_load(open(os.path.expanduser('~/.ros/table_geometry.yaml')))['table_top_z'])
    R, t, y = load_file(a.file or FILES[a.camera])
    depth, K = grab(a.camera)
    P = table_points(depth, K, R, t)
    if len(P) < 1500:
        raise SystemExit(f'only {len(P)} depth points on the table region: the table must be clear and in view')
    n, c, inl, _ = fit_plane(P)
    tilt = np.degrees(np.arccos(n[2]))
    print(f'{a.camera}: {len(P)} points in the table region, {inl} on the plane: height at ({CENTRE_XY[0]}, {CENTRE_XY[1]}) = {c[2]:.4f} m '
          f'(measured table top {top:.4f}: {100 * (c[2] - top):+.1f} cm), tilt {tilt:.2f} deg')
    R2, t2 = level_correction(R, t, n, c, top)
    shift = np.linalg.norm(t2 - t) * 100
    turn = np.degrees((Rotation.from_matrix(R).inv() * Rotation.from_matrix(R2)).magnitude())
    print(f'correction: camera moves {shift:.1f} cm ({np.round(100 * (t2 - t), 1).tolist()} cm in x, y, z) and turns {turn:.2f} deg')
    n2, c2, _, _ = fit_plane(table_points(depth, K, R2, t2))
    print(f'check on the same depth after the correction: height {100 * (c2[2] - top):+.2f} cm vs the table, tilt {np.degrees(np.arccos(n2[2])):.2f} deg')
    if a.out:
        y['translation'] = {k: float(v) for k, v in zip('xyz', t2)}
        q = Rotation.from_matrix(R2).as_quat()
        y['rotation_quat'] = {k: float(v) for k, v in zip('xyzw', q)}
        y['note'] = (f'{y.get("note", "")} Levelled to the table plane ({time.strftime("%Y-%m-%d %H:%M")}, calibration/level_to_table.py): height {100 * (c[2] - top):+.1f} cm and tilt {tilt:.2f} deg '
                     f'removed. NOT verified until it passes check_arm_depth.py and the shared-object comparison.').strip()
        with open(os.path.expanduser(a.out), 'w') as f:
            yaml.safe_dump(y, f, default_flow_style=False)
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
