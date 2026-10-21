#!/usr/bin/env python3
"""
refine_from_clouds.py -- fix (or check) a static camera's extrinsic by registering its depth cloud onto the clouds of cameras that are already
calibrated. Needs no clicking and no board: every camera sees the same table, arm and room, so the right pose is the one that makes the clouds overlap.

    source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
    python3 calibration/refine_from_clouds.py --target realsense3 [--refs realsense realsense2] [--init <yaml> ...] [--save]
    python3 calibration/refine_from_clouds.py --target realsense2 --refs realsense          # how well do two calibrated cameras agree? (a yardstick)

Default --min-z 0.05 registers the ARM and objects only: with the table included the answer slid 6 cm between runs; arm-only converged to 92-94% overlap.
Read-only unless --save, which writes ~/.ros/calibration_candidates/<target>_calibration.icp.candidate.yaml (never the live file).
The arm must stay still and the table should hold the usual clutter (more structure helps). People walking through are ignored by the robust cost.
The result is only as good as the reference cameras' own calibration (about 2-3 cm here): it makes the cameras AGREE, which is what the fusion needs.
"""
import argparse
import time

import numpy as np
import open3d as o3d
import rclpy
import tf2_ros
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

TOPICS = {
    'realsense': ('/global_camera/global_camera/aligned_depth_to_color/image_raw', '/global_camera/global_camera/aligned_depth_to_color/camera_info', 'global_camera_color_optical_frame'),
    'realsense2': ('/global_camera_2/global_camera_2/aligned_depth_to_color/image_raw', '/global_camera_2/global_camera_2/aligned_depth_to_color/camera_info', 'global_camera_2_color_optical_frame'),
    'realsense3': ('/global_camera_3/global_camera_3/aligned_depth_to_color/image_raw', '/global_camera_3/global_camera_3/aligned_depth_to_color/camera_info', 'global_camera_3_color_optical_frame'),
}
BOX_MIN, BOX_MAX = np.array([-0.35, -1.0, -0.10]), np.array([1.05, 1.0, 1.25])     # base_link: table, arm, the room around them
STRIDE_PX = 4


class Grab(Node):
    def __init__(self, cams):
        super().__init__('refine_from_clouds')
        self.bridge = CvBridge()
        self.tf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf, self)
        self.frames, self.K = {c: [] for c in cams}, {}
        for c in cams:
            d, i, _ = TOPICS[c]
            self.create_subscription(Image, d, lambda m, c=c: self._depth(c, m), 1)
            self.create_subscription(CameraInfo, i, lambda m, c=c: self.K.__setitem__(c, np.array(m.k).reshape(3, 3)), 1)

    def _depth(self, c, m):
        if len(self.frames[c]) < 9:
            self.frames[c].append(self.bridge.imgmsg_to_cv2(m, '16UC1'))


def median_depth(frames):
    st = np.stack(frames).astype(np.float32)
    st[st == 0] = np.nan
    with np.errstate(all='ignore'):
        med = np.nanmedian(st, axis=0)
    return np.nan_to_num(med, nan=0.0) / 1000.0


def cloud(depth_m, K):
    h, w = depth_m.shape
    v, u = np.mgrid[0:h:STRIDE_PX, 0:w:STRIDE_PX]
    z = depth_m[v, u]
    ok = (z > 0.3) & (z < 3.5)
    x = (u[ok] - K[0, 2]) * z[ok] / K[0, 0]
    y = (v[ok] - K[1, 2]) * z[ok] / K[1, 1]
    return np.column_stack([x, y, z[ok]])


def T_from_tf(node, cam, wait_s=8.0):
    """base <- camera frame; waits (spinning) for the TF, because a fresh node needs a moment to discover the broadcasters."""
    end = time.time() + wait_s
    while not node.tf.can_transform('base_link', TOPICS[cam][2], rclpy.time.Time()) and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    t = node.tf.lookup_transform('base_link', TOPICS[cam][2], rclpy.time.Time())
    q, p = t.transform.rotation, t.transform.translation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def T_from_yaml(path):
    d = yaml.safe_load(open(path))
    q, t = d['rotation_quat'], d['translation']
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
    T[:3, 3] = [t['x'], t['y'], t['z']]
    return T


def to_pcd(points):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(points)
    return p


def crop(points_base):
    m = np.all((points_base >= BOX_MIN) & (points_base <= BOX_MAX), axis=1)
    return points_base[m]


def register(src_cam_pts, ref_pts_base, T_init):
    """ICP with a shrinking correspondence distance and a robust (Tukey) cost; the source starts at T_init (base <- camera)."""
    ref = to_pcd(ref_pts_base)
    ref = ref.voxel_down_sample(0.02)
    ref.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    T = T_init.copy()
    res = None
    for dist in (0.25, 0.12, 0.06, 0.03):
        src_b = crop(src_cam_pts @ T[:3, :3].T + T[:3, 3])
        src = to_pcd(src_b).voxel_down_sample(0.02)
        if len(src.points) < 500:
            return None
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane(o3d.pipelines.registration.TukeyLoss(k=dist * 0.5))
        res = o3d.pipelines.registration.registration_icp(
            src, ref, dist, np.eye(4), est, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T = res.transformation @ T
    src_b = crop(src_cam_pts @ T[:3, :3].T + T[:3, 3])
    final = o3d.pipelines.registration.evaluate_registration(to_pcd(src_b).voxel_down_sample(0.02), ref, 0.03, np.eye(4))
    return T, final.fitness, final.inlier_rmse


def delta(T_a, T_b):
    dR = Rotation.from_matrix(T_a[:3, :3] @ T_b[:3, :3].T)
    return float(np.degrees(dR.magnitude())), float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='realsense3')
    ap.add_argument('--refs', nargs='+', default=['realsense', 'realsense2'])
    ap.add_argument('--init', nargs='*', default=[], help='yaml files with starting poses for the target (default: its live TF)')
    ap.add_argument('--min-z', type=float, default=0.05, help='keep only points this high above base_link z=0 (default 0.05: the arm and objects, NOT the table; a table-dominated cloud lets ICP slide along the plane)')
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--dump', help='write the captured clouds to this .npz (to iterate offline)')
    args = ap.parse_args()
    global BOX_MIN
    BOX_MIN = np.array([-0.35, -1.0, args.min_z])
    rclpy.init()
    cams = args.refs + [args.target]
    n = Grab(cams)
    end = time.time() + 12
    while time.time() < end and not all(len(n.frames[c]) >= 9 and c in n.K for c in cams):
        rclpy.spin_once(n, timeout_sec=0.1)
    clouds = {}
    for c in cams:
        if len(n.frames[c]) < 3 or c not in n.K:
            raise SystemExit(f'{c}: no depth frames / camera info')
        clouds[c] = cloud(median_depth(n.frames[c]), n.K[c])
    ref_base = []
    for c in args.refs:
        T = T_from_tf(n, c)
        ref_base.append(crop(clouds[c] @ T[:3, :3].T + T[:3, 3]))
        print(f'reference {c}: {len(ref_base[-1])} points inside the workspace box')
    ref_pts = np.vstack(ref_base)
    inits = [T_from_yaml(p) for p in args.init]
    live = None
    if not args.init:                                          # only needed as the start when no yaml was given
        try:
            live = T_from_tf(n, args.target)
        except Exception:
            pass
    if live is not None:
        inits.append(live)
    if not inits:
        raise SystemExit('no starting pose: pass --init <yaml> or start the target camera TF broadcaster')
    if args.dump:
        np.savez(args.dump, ref=ref_pts, tgt=clouds[args.target], inits=np.stack(inits))
    best = None
    for i, T0 in enumerate(inits):
        pre = o3d.pipelines.registration.evaluate_registration(to_pcd(crop(clouds[args.target] @ T0[:3, :3].T + T0[:3, 3])).voxel_down_sample(0.02), to_pcd(ref_pts).voxel_down_sample(0.02), 0.03, np.eye(4))
        print(f'start {i}: overlap before ICP {pre.fitness:.3f}')
        out = register(clouds[args.target], ref_pts, T0)
        if out is None:
            print(f'start {i}: too few points overlap'); continue
        T, fit, rmse = out
        rot, tr = delta(T, T0)
        print(f'start {i}: fitness {fit:.3f} (share of points within 3 cm), inlier RMSE {rmse * 100:.2f} cm; moved {rot:.1f} deg, {tr:.1f} cm from its start')
        if best is None or fit > best[1]:
            best = (T, fit, rmse, i)
    if best is None:
        raise SystemExit('registration failed')
    T, fit, rmse, i = best
    print(f'best start: {i}; camera at {np.round(T[:3, 3], 3)}, rpy {np.round(Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True), 1)} deg')
    for j, T0 in enumerate(inits):
        if j != i:
            print(f'  differs from start {j}: {delta(T, T0)[0]:.1f} deg, {delta(T, T0)[1]:.1f} cm')
    if args.save:
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        d = {'parent_frame': 'base_link', 'child_frame': TOPICS[args.target][2],
             'rotation_quat': {'x': float(q[0]), 'y': float(q[1]), 'z': float(q[2]), 'w': float(q[3])},
             'translation': {'x': float(T[0, 3]), 'y': float(T[1, 3]), 'z': float(T[2, 3])},
             'note': f'Registered onto {args.refs} depth clouds (calibration/refine_from_clouds.py), {time.strftime("%Y-%m-%d %H:%M")}, fitness {fit:.2f}, RMSE {rmse * 100:.1f} cm. NOT active.'}
        import os
        path = os.path.expanduser(f'~/.ros/calibration_candidates/{args.target}_calibration.icp.candidate.yaml')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        yaml.safe_dump(d, open(path, 'w'), default_flow_style=False)
        print('saved', path)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
