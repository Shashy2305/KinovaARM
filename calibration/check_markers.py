#!/usr/bin/env python3
"""
check_markers.py -- do the static cameras agree on where a fixed ArUco board is? Read-only; moves nothing.

    source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
    python3 /mnt/ros_workspace/Shashproject/calibration/check_markers.py [--cams realsense realsense2 realsense3] [--seconds 6]

Each camera finds the markers in its colour image, reads the depth at each marker centre, and the camera's calibration (TF base_link <- camera
frame) turns that into a base_link position. A marker lying on the table must come out at (nearly) the same base_link position from every camera
and at the table height. Unlike check_extrinsics.py this needs no segmentation mask and no cup: any marker board in view of two or more cameras works.
Verdict per camera: its markers' distance from the mean of the OTHER cameras (cm). Good: under 3 cm. Marker ids are only compared when two cameras see the same id.
"""
import argparse
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

TOPICS = {
    'realsense': ('/global_camera/global_camera/color/image_raw', '/global_camera/global_camera/aligned_depth_to_color/image_raw',
                  '/global_camera/global_camera/color/camera_info', 'global_camera_color_optical_frame'),
    'realsense2': ('/global_camera_2/global_camera_2/color/image_raw', '/global_camera_2/global_camera_2/aligned_depth_to_color/image_raw',
                   '/global_camera_2/global_camera_2/color/camera_info', 'global_camera_2_color_optical_frame'),
    'realsense3': ('/global_camera_3/global_camera_3/color/image_raw', '/global_camera_3/global_camera_3/aligned_depth_to_color/image_raw',
                   '/global_camera_3/global_camera_3/color/camera_info', 'global_camera_3_color_optical_frame'),
}
DICTS = ['DICT_4X4_50', 'DICT_4X4_100',  # the plywood board of 2026-10-20 is DICT_4X4_50, ids 0-3
          'DICT_4X4_250', 'DICT_5X5_50', 'DICT_5X5_100', 'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250',
         'DICT_7X7_50', 'DICT_ARUCO_ORIGINAL', 'DICT_APRILTAG_36h11']


def detect(gray, name):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    if hasattr(cv2.aruco, 'ArucoDetector'):                   # OpenCV >= 4.7 (the ROS environment has 4.13)
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=cv2.aruco.DetectorParameters_create())
    return corners, (ids.flatten() if ids is not None else np.array([]))


class Grab(Node):
    def __init__(self, cams):
        super().__init__('check_markers')
        self.bridge = CvBridge()
        self.tf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf, self)
        self.color, self.depth, self.K = {}, {}, {}
        for c in cams:
            ci, di, ki, _ = TOPICS[c]
            self.create_subscription(Image, ci, lambda m, c=c: self.color.__setitem__(c, m), 1)
            self.create_subscription(Image, di, lambda m, c=c: self.depth.__setitem__(c, m), 1)
            self.create_subscription(CameraInfo, ki, lambda m, c=c: self.K.__setitem__(c, np.array(m.k).reshape(3, 3)), 1)


def to_base(node, cam, u, v, depth_m, K):
    frame = TOPICS[cam][3]
    t = node.tf.lookup_transform('base_link', frame, rclpy.time.Time())
    q, p = t.transform.rotation, t.transform.translation
    x, y, w, z = q.x, q.y, q.w, q.z
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    pc = np.array([(u - K[0, 2]) * depth_m / K[0, 0], (v - K[1, 2]) * depth_m / K[1, 1], depth_m])
    return R @ pc + np.array([p.x, p.y, p.z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cams', nargs='+', default=['realsense', 'realsense2', 'realsense3'])
    ap.add_argument('--seconds', type=float, default=6.0)
    args = ap.parse_args()
    rclpy.init()
    n = Grab(args.cams)
    end = time.time() + args.seconds
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
    found = {}                                                  # cam -> {id: base xyz}
    best_dict = None
    for cam in args.cams:
        if cam not in n.color or cam not in n.depth or cam not in n.K:
            print(f'{cam:11s}: no image / depth / camera info'); continue
        bgr = n.bridge.imgmsg_to_cv2(n.color[cam], 'bgr8')
        dep = n.bridge.imgmsg_to_cv2(n.depth[cam], '16UC1')
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        names = [best_dict] if best_dict else DICTS
        scored = [(len(detect(gray, d)[1]), d) for d in names]
        count, dname = max(scored)
        if count == 0:
            print(f'{cam:11s}: no ArUco markers found'); continue
        best_dict = best_dict or dname
        corners, ids = detect(gray, dname)
        found[cam] = {}
        for c, i in zip(corners, ids):
            u, v = c.reshape(4, 2).mean(axis=0)
            h, w = dep.shape
            sx, sy = w / bgr.shape[1], h / bgr.shape[0]
            patch = dep[max(0, int(v * sy) - 4):int(v * sy) + 5, max(0, int(u * sx) - 4):int(u * sx) + 5]
            ok = patch[patch > 0]
            if len(ok) < 5:
                continue
            try:
                found[cam][int(i)] = to_base(n, cam, u, v, float(np.median(ok)) / 1000.0, n.K[cam])
            except Exception as e:
                print(f'{cam:11s}: no TF base_link <- {TOPICS[cam][3]} ({str(e)[:50]})'); found.pop(cam); break
        if cam in found:
            print(f'{cam:11s}: {len(found[cam])} markers ({dname}): ' + ', '.join(f'id{i} ({p[0]:.2f},{p[1]:.2f},{p[2]:.3f})' for i, p in sorted(found[cam].items())))
    print()
    ids = sorted({i for d in found.values() for i in d})
    spreads = {c: [] for c in found}
    for i in ids:
        seen = {c: d[i] for c, d in found.items() if i in d}
        if len(seen) < 2:
            continue
        for c, p in seen.items():
            others = [q for k, q in seen.items() if k != c]
            spreads[c].append(float(np.linalg.norm(p - np.mean(others, axis=0))) * 100)
    worst = 0.0
    for c, s in spreads.items():
        if s:
            print(f'{c:11s}: {len(s)} shared markers, distance from the other cameras: mean {np.mean(s):.1f} cm, max {np.max(s):.1f} cm')
            worst = max(worst, np.mean(s))
    zs = [p[2] for d in found.values() for p in d.values()]
    if zs:
        print(f'marker heights: mean {np.mean(zs):.3f} m, spread {100 * (max(zs) - min(zs)):.1f} cm (the board lies on the table)')
    if not any(spreads.values()):
        print('VERDICT: no marker is seen by two cameras, nothing to compare')
    else:
        print('VERDICT:', 'OK, the cameras agree on the board' if worst < 3.0 else f'PROBLEM: a camera is off by {worst:.1f} cm on average')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
