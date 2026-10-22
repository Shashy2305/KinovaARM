#!/usr/bin/env python3
"""
check_arm_depth.py -- is a static camera where its calibration says? READ-ONLY; uses the arm as the ruler (no board, no objects, no clicking).

    source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
    python3 calibration/check_arm_depth.py [--cams realsense realsense2 realsense3]

Points are sampled along the arm's skeleton (base -> shoulder -> ... -> flange -> finger pads) from the live TF, projected into each camera with ITS calibration, and
the depth image is read at those pixels. The skeleton runs along the middle of the links, so the surface the camera measures is about one link radius NEARER than
the prediction: a well-placed camera shows a CONSTANT offset of about 4-7 cm with a spread of 1-2 cm; a camera that has moved shows a different median and a large spread
(or too few usable points because the arm lands on the background). Compare the cameras with each other: the one that stands out is the one that moved.
"""
import argparse
import sys
import time

import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, '/mnt/ros_workspace/Shashproject/src/thesis_robot')
from thesis_robot import safety_geometry as sg  # noqa: E402

TOPICS = {
    'realsense': ('/global_camera/global_camera/aligned_depth_to_color/image_raw', '/global_camera/global_camera/aligned_depth_to_color/camera_info', 'global_camera_color_optical_frame'),
    'realsense2': ('/global_camera_2/global_camera_2/aligned_depth_to_color/image_raw', '/global_camera_2/global_camera_2/aligned_depth_to_color/camera_info', 'global_camera_2_color_optical_frame'),
    'realsense3': ('/global_camera_3/global_camera_3/aligned_depth_to_color/image_raw', '/global_camera_3/global_camera_3/aligned_depth_to_color/camera_info', 'global_camera_3_color_optical_frame'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cams', nargs='+', default=['realsense', 'realsense2', 'realsense3'])
    args = ap.parse_args()
    rclpy.init()
    n = Node('check_arm_depth')
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, n)
    bridge = CvBridge()
    depth, K = {}, {}
    for c in args.cams:
        d, i, _ = TOPICS[c]
        n.create_subscription(Image, d, lambda m, c=c: depth.__setitem__(c, m), 1)
        n.create_subscription(CameraInfo, i, lambda m, c=c: K.__setitem__(c, np.array(m.k).reshape(3, 3)), 1)
    end = time.time() + 10
    while time.time() < end and not all(c in depth and c in K for c in args.cams):
        rclpy.spin_once(n, timeout_sec=0.1)
    end = time.time() + 3
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)

    def pos(frame):
        t = buf.lookup_transform('base_link', frame, rclpy.time.Time()).transform.translation
        return np.array([t.x, t.y, t.z])
    chain = [np.zeros(3)] + [pos(f) for f in sg.ARM_CHAIN_FRAMES]
    chain += [pos(f) for f in sg.FINGER_PAD_FRAMES]
    segs = list(zip(chain[:-2], chain[1:-1])) + [(chain[-3], chain[-2]), (chain[-3], chain[-1])]
    pts = []
    for a, b in segs:
        L = np.linalg.norm(b - a)
        for t in np.linspace(0, 1, max(2, int(L / 0.04))):
            pts.append(a + t * (b - a))
    pts = np.array(pts)
    print(f'{len(pts)} skeleton points along the arm')
    for c in args.cams:
        if c not in depth or c not in K:
            print(f'{c:11s}: no depth / camera info'); continue
        try:
            t = buf.lookup_transform('base_link', TOPICS[c][2], rclpy.time.Time()).transform
        except Exception as e:
            print(f'{c:11s}: no TF ({str(e)[:50]})'); continue
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_matrix()
        T = np.array([t.translation.x, t.translation.y, t.translation.z])
        cam = (R.T @ (pts - T).T).T
        d = bridge.imgmsg_to_cv2(depth[c], '16UC1').astype(np.float32) / 1000.0
        h, w = d.shape
        res = []
        for p in cam:
            if p[2] <= 0.2:
                continue
            u = int(round(K[c][0, 0] * p[0] / p[2] + K[c][0, 2]))
            v = int(round(K[c][1, 1] * p[1] / p[2] + K[c][1, 2]))
            if not (8 <= u < w - 8 and 8 <= v < h - 8):
                continue
            patch = d[v - 3:v + 4, u - 3:u + 4]
            ok = patch[patch > 0.2]
            if len(ok) < 10:
                continue
            res.append(float(np.linalg.norm(p)) * 0 + (p[2] - float(np.median(ok))))      # predicted depth minus measured depth (the surface is nearer: positive)
        res = np.array(res)
        if len(res) < 8:
            print(f'{c:11s}: only {len(res)} usable points'); continue
        med = np.median(res)
        mad = np.median(np.abs(res - med))
        inl = np.mean(np.abs(res - 0.055) < 0.05)
        print(f'{c:11s}: {len(res):3d} points, offset median {100 * med:5.1f} cm, spread (MAD) {100 * mad:4.1f} cm, {100 * inl:3.0f}% within 5 cm of the expected 5.5 cm')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
