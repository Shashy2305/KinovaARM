#!/usr/bin/env python3
"""
check_extrinsics.py -- are the static cameras' calibrations consistent with each
other and with the real world?  Read-only; moves nothing.

    source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
    python3 /mnt/ros_workspace/Shashproject/calibration/check_extrinsics.py [--label cup]

Two checks that need no ground truth:

 1. TABLE PLANE. The table is one flat level surface and the robot stands on it.
    For each static camera the 'dining table' segmentation mask is back-projected
    into base_link and a plane is fitted. A correctly calibrated camera gives a
    LEVEL plane (tilt ~0 deg) at the SAME height as the other cameras (and as the
    recorded table top). Large tilt or height disagreement = that camera's
    extrinsic rotation/height is wrong.
 2. OBJECT AGREEMENT. One physical object must land at (nearly) the same base_link
    position from every camera that sees it. A spread of tens of cm means the
    cameras disagree and the scene graph will invent duplicate objects.

Verdict thresholds: tilt > 3 deg, plane-height spread > 3 cm, object spread > 5 cm.
Needs the camera drivers, the three calibration TFs and (for segmentation)
/mnt/ros_workspace/models/yolov8m-seg.pt.
"""
import argparse
import os
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO

CAMS = {
    'oakd': ('/global_camera/color/image_raw', '/global_camera/depth/image_raw',
             '/global_camera/color/camera_info'),
    'realsense': ('/global_camera/global_camera/color/image_raw',
                  '/global_camera/global_camera/aligned_depth_to_color/image_raw',
                  '/global_camera/global_camera/color/camera_info'),
    'realsense2': ('/global_camera_2/global_camera_2/color/image_raw',
                   '/global_camera_2/global_camera_2/aligned_depth_to_color/image_raw',
                   '/global_camera_2/global_camera_2/color/camera_info'),
}
SEG_MODEL = '/mnt/ros_workspace/models/yolov8m-seg.pt'
TILT_LIMIT_DEG, HEIGHT_LIMIT_M, OBJECT_LIMIT_M = 3.0, 0.03, 0.05
rng = np.random.default_rng(0)


def ransac_plane(P, iters=500, thr=0.012):
    best = None
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        L = np.linalg.norm(n)
        if L < 1e-9:
            continue
        n /= L
        inl = np.abs((P - s[0]) @ n) < thr
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), inl)
    if best is None:
        return None
    Q = P[best[1]]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][2]
    if n[2] < 0:
        n = -n
    return n, c, int(best[0]), len(P)


def mask_cloud(mask, depth, K, scale):
    H, W = mask.shape
    dh, dw = depth.shape
    if (dh, dw) != (H, W):
        depth = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(mask)
    d = depth[ys, xs].astype(np.float32) * scale
    ok = (d > 0.15) & (d < 4.0)
    xs, ys, d = xs[ok], ys[ok], d[ok]
    return np.column_stack([(xs - K[0, 2]) * d / K[0, 0], (ys - K[1, 2]) * d / K[1, 1], d]), (xs, ys, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='cup', help='object class to compare across cameras')
    ap.add_argument('--model', default=SEG_MODEL)
    ap.add_argument('--table-z', type=float, default=None,
                    help='recorded table top z in base_link (default: read ~/.ros/table_geometry.yaml if present)')
    args = ap.parse_args()

    rclpy.init()
    node = Node('check_extrinsics')
    br = CvBridge()
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)
    got = {c: {} for c in CAMS}
    for c, (ti, td, tk) in CAMS.items():
        node.create_subscription(Image, ti, lambda m, c=c: got[c].__setitem__('rgb', m), 1)
        node.create_subscription(Image, td, lambda m, c=c: got[c].__setitem__('depth', m), 1)
        node.create_subscription(CameraInfo, tk, lambda m, c=c: got[c].__setitem__('info', m), 1)
    t0 = time.time()
    while time.time() - t0 < 12 and not all(len(v) == 3 for v in got.values()):
        rclpy.spin_once(node, timeout_sec=0.2)
    for _ in range(25):
        rclpy.spin_once(node, timeout_sec=0.1)

    table_z = args.table_z
    if table_z is None:
        try:
            from thesis_robot import safety_geometry as sg
            g, _ = sg.load_geometry()
            table_z = g['table_top_z'] if g else None
        except Exception:
            pass

    model = YOLO(args.model)
    table_ids = [i for i, n in model.names.items() if n == 'dining table']
    planes, objects = {}, {}
    for c in CAMS:
        g = got[c]
        if len(g) < 3:
            print(f'{c:11s}: no data (driver down?)')
            continue
        try:
            tf = buf.lookup_transform('base_link', g['rgb'].header.frame_id, rclpy.time.Time())
        except Exception as e:
            print(f'{c:11s}: no TF base_link <- {g["rgb"].header.frame_id}: {e}')
            continue
        q, t = tf.transform.rotation, tf.transform.translation
        Rm = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T = np.array([t.x, t.y, t.z])
        bgr = br.imgmsg_to_cv2(g['rgb'], 'bgr8')
        depth = br.imgmsg_to_cv2(g['depth'], 'passthrough')
        K = np.array(g['info'].k).reshape(3, 3)
        scale = 0.001 if depth.dtype == np.uint16 else 1.0
        H, W = bgr.shape[:2]
        r = model(bgr, verbose=False)[0]
        if r.masks is None:
            print(f'{c:11s}: model produced no masks')
            continue

        # 1. table plane from the 'dining table' mask
        tab = np.zeros((H, W), np.uint8)
        for i, b in enumerate(r.boxes):
            if int(b.cls) in table_ids:
                m = cv2.resize(r.masks.data[i].cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                tab |= m
        tab = cv2.erode(tab, np.ones((9, 9), np.uint8))
        if tab.sum() > 2000:
            P, _ = mask_cloud(tab, depth, K, scale)
            B = (Rm @ P.T).T + T
            fit = ransac_plane(B[:: max(1, len(B) // 20000)])
            if fit:
                n, ctr, ninl, ntot = fit
                tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
                z0 = ctr[2] - (n[0] * (0.4 - ctr[0]) + n[1] * (0.0 - ctr[1])) / n[2]
                planes[c] = (tilt, float(z0), ninl / ntot)
                print(f'{c:11s}: table plane tilt {tilt:5.1f} deg | height at (0.4, 0) z={z0:+.3f} m | {ninl}/{ntot} points fit')
        else:
            print(f'{c:11s}: no usable "dining table" mask in this view')

        # 2. objects
        for i, b in enumerate(r.boxes):
            if r.names[int(b.cls)] != args.label:
                continue
            m = cv2.erode(cv2.resize(r.masks.data[i].cpu().numpy().astype(np.uint8), (W, H),
                                     interpolation=cv2.INTER_NEAREST), np.ones((5, 5), np.uint8))
            P, _ = mask_cloud(m, depth, K, scale)
            if len(P) < 30:
                continue
            med = np.median(P[:, 2])
            P = P[np.abs(P[:, 2] - med) < 0.05]
            objects.setdefault(c, []).append((Rm @ np.median(P, axis=0) + T, float(b.conf)))

    print('\n=== verdict ===')
    problems = []
    if planes:
        for c, (tilt, z0, frac) in planes.items():
            if tilt > TILT_LIMIT_DEG:
                problems.append(f'{c}: table appears tilted {tilt:.1f} deg -> its extrinsic rotation is wrong')
        zs = [z for _, z, _ in planes.values()]
        if len(zs) > 1 and max(zs) - min(zs) > HEIGHT_LIMIT_M:
            problems.append(f'cameras disagree on the table height by {(max(zs) - min(zs)) * 100:.0f} cm {dict((c, round(v[1], 3)) for c, v in planes.items())}')
        if table_z is not None:
            for c, (_, z0, _) in planes.items():
                if abs(z0 - table_z) > 0.05:
                    problems.append(f'{c}: table at z={z0:+.3f} but the recorded table top is {table_z:+.3f}')
    allp = [(c, p) for c, lst in objects.items() for p, _ in lst]
    for i, (c, p) in enumerate(allp):
        grp = [(c, p)] + [(c2, p2) for c2, p2 in allp[i + 1:] if c2 != c and np.linalg.norm(p2[:2] - p[:2]) < 0.6]
        if len(grp) > 1:
            pts = np.array([x[1] for x in grp])
            sp = max(np.linalg.norm(a[:2] - b[:2]) for a in pts for b in pts)
            print(f'{args.label} seen by {[x[0] for x in grp]}: xy spread {sp * 100:.0f} cm, z {np.round(pts[:, 2], 3).tolist()}')
            if sp > OBJECT_LIMIT_M:
                problems.append(f'{args.label} positions disagree by {sp * 100:.0f} cm between {[x[0] for x in grp]}')
            break
    if not problems:
        print('OK: cameras agree with each other and with a level table.')
    else:
        for p in problems:
            print('PROBLEM:', p)
        print('\n-> recalibrate (dashboard Calibration tab; needs the wrist camera working) and rerun this check.')
    rclpy.shutdown()
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
