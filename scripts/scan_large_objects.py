#!/usr/bin/env python3
"""Print every blob standing on the table as the depth cameras see it, WITHOUT subtracting what the scene already explains: position, footprint, height, per camera.
Use it to find a big thing the detectors name wrongly (a coffee machine read as a 'cup') and to get its numbers for config/fixed_obstacles.yaml.
    source scripts/ros_env.sh; ROS_DOMAIN_ID=42 python3 scripts/scan_large_objects.py"""
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

from thesis_robot import obstacle_guard as og
from thesis_robot import safety_geometry as sg
from thesis_robot.obstacle_guard_node import CAMERAS, BASE
import tf2_ros


def main():
    rclpy.init()
    n = Node('scan_large_objects')
    br = CvBridge()
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, n)
    depth, K = {}, {}
    for name, c in CAMERAS.items():
        n.create_subscription(Image, c['depth'], lambda m, k=name: depth.__setitem__(k, m), 1)
        n.create_subscription(CameraInfo, c['info'], lambda m, k=name: K.__setitem__(k, np.array(m.k).reshape(3, 3)), 1)
    t0 = time.time()
    while time.time() - t0 < 6:
        rclpy.spin_once(n, timeout_sec=0.1)
    geom, _ = sg.load_geometry()
    chain = [(0.0, 0.0, 0.0)]
    try:
        chain += [tuple(getattr(buf.lookup_transform(BASE, f, rclpy.time.Time()).transform.translation, a) for a in 'xyz') for f in sg.ARM_CHAIN_FRAMES]
        fingers = [(chain[-1], tuple(getattr(buf.lookup_transform(BASE, f, rclpy.time.Time()).transform.translation, a) for a in 'xyz')) for f in sg.FINGER_PAD_FRAMES]
    except Exception as e:
        print('no arm TF, the arm will show up as obstacles:', e)
        chain, fingers = None, None
    xr, yr = (0.0, 0.85), (-0.6, 0.6)
    for name in CAMERAS:
        if name not in depth or name not in K:
            print(f'{name}: no depth'); continue
        tf = buf.lookup_transform(BASE, CAMERAS[name]['frame'], rclpy.time.Time()).transform
        R = Rotation.from_quat([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]).as_matrix()
        t = [tf.translation.x, tf.translation.y, tf.translation.z]
        d = br.imgmsg_to_cv2(depth[name], desired_encoding='passthrough').astype(np.float32)
        d = d / 1000.0 if d.max() > 50 else d
        pts = og.backproject(d, K[name], R, t, stride=3)
        for b in og.unknown_obstacles(pts, geom['table_top_z'], xr, yr, [], chain, fingers):
            print(f"{name:11s} x={b['x']:.3f} y={b['y']:.3f} height={b['height']:.3f} footprint={b['w']:.3f}x{b['h']:.3f} m  area={b['area_m2'] * 1e4:.0f} cm2")
    rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
