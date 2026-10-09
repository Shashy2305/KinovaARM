#!/usr/bin/env python3
"""
Unknown-obstacle guard node: everything standing on the table that the object detectors do not know.

Subscribes: aligned depth + camera_info of the static RealSense cameras, /scene_snapshot (what IS explained), TF
Publishes:  /unknown_obstacles  (std_msgs/String JSON {"stamp", "obstacles": [{x, y, height, w, h, area_m2, seen_by}]}),
            2 Hz, only obstacles that persisted for at least `min_frames` frames.
The arm controller reads it (parameter use_unknown_obstacles, default OFF) to keep carries and finger sweeps clear of
things the cameras can see but cannot name. See obstacle_guard.py for the method.
"""
import json
import math
import os
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')           # the first version took 740% CPU: BLAS/OpenCV threads fighting the rest of the stack

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from thesis_robot import obstacle_guard as og
from thesis_robot import safety_geometry as sg
from thesis_robot import wrist_servo as ws

cv2.setNumThreads(1)

BASE = 'base_link'
CAMERAS = {
    'realsense': {'depth': '/global_camera/global_camera/aligned_depth_to_color/image_raw',
                  'info': '/global_camera/global_camera/aligned_depth_to_color/camera_info',
                  'frame': 'global_camera_color_optical_frame'},
    'realsense2': {'depth': '/global_camera_2/global_camera_2/aligned_depth_to_color/image_raw',
                   'info': '/global_camera_2/global_camera_2/aligned_depth_to_color/camera_info',
                   'frame': 'global_camera_2_color_optical_frame'},
}


class ObstacleGuardNode(Node):
    def __init__(self):
        super().__init__('obstacle_guard')
        self.declare_parameter('min_frames', 2)               # a blob must persist this many cycles
        self.declare_parameter('period_s', 1.0)
        self.declare_parameter('known_extra_m', 0.03)         # scene positions are +-5 cm between cameras
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.depth, self.K, self.scene = {}, {}, {}
        self.tracks = []                                      # [{'x','y','count','last', ...}]
        for name, c in CAMERAS.items():
            self.create_subscription(Image, c['depth'], lambda m, n=name: self._on_depth(n, m), 1)
            self.create_subscription(CameraInfo, c['info'], lambda m, n=name: self._on_info(n, m), 1)
        self.create_subscription(String, '/scene_snapshot', self._on_scene, 5)
        self.pub = self.create_publisher(String, '/unknown_obstacles', 5)
        self.create_timer(float(self.get_parameter('period_s').value), self._tick)
        self.get_logger().info('Obstacle guard ready: publishing /unknown_obstacles')

    # ── inputs ────────────────────────────────────────────────────────
    def _on_depth(self, name, msg):
        self.depth[name] = (time.monotonic(), msg)

    def _on_info(self, name, msg):
        self.K[name] = np.array(msg.k).reshape(3, 3)

    def _on_scene(self, msg):
        try:
            self.scene = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _pos(self, frame):
        t = self.tf_buffer.lookup_transform(BASE, frame, rclpy.time.Time()).transform.translation
        return (t.x, t.y, t.z)

    def _arm_body(self):
        try:
            chain = [(0.0, 0.0, 0.0)] + [self._pos(f) for f in sg.ARM_CHAIN_FRAMES]
            fingers = [(chain[-1], self._pos(f)) for f in sg.FINGER_PAD_FRAMES]
            return chain, fingers
        except Exception:
            return None, None

    def _known(self):
        extra = float(self.get_parameter('known_extra_m').value)
        out = []
        for o in self.scene.values():
            if isinstance(o, dict) and not o.get('stale') and o.get('x') is not None:
                out.append((float(o['x']), float(o['y']), ws.OBJECT_RADIUS_M.get(o.get('label'), ws.DEFAULT_RADIUS_M) + extra))
        return out

    # ── one cycle ─────────────────────────────────────────────────────
    def _tick(self):
        t_tick = time.monotonic()
        geom, _ = sg.load_geometry()
        if geom is None:
            return
        chain, fingers = self._arm_body()
        if chain is None:
            return                                            # cannot tell the arm from an obstacle: publish nothing
        x_range = (0.0, min(0.75, geom['x'][1]))
        y_range = (max(-0.55, geom['y'][0]), min(0.55, geom['y'][1]))
        known = self._known()
        per_cam = {}
        for name, c in CAMERAS.items():
            item, K = self.depth.get(name), self.K.get(name)
            if not item or K is None or time.monotonic() - item[0] > 1.0:
                continue
            try:
                tf = self.tf_buffer.lookup_transform(BASE, c['frame'], rclpy.time.Time()).transform
                R = Rotation.from_quat([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]).as_matrix()
                t = [tf.translation.x, tf.translation.y, tf.translation.z]
                d = self.bridge.imgmsg_to_cv2(item[1], desired_encoding='passthrough').astype(np.float32)
                d = d / 1000.0 if d.max() > 50 else d
                pts = og.backproject(d, K, R, t, stride=4)
                per_cam[name] = og.unknown_obstacles(pts, geom['table_top_z'], x_range, y_range, known, chain, fingers)
            except Exception as e:
                self.get_logger().warn(f'{name}: {e}', throttle_duration_sec=10.0)
        if not per_cam:
            return
        merged = og.merge_across_cameras(per_cam)
        now = time.monotonic()
        new_tracks = []
        for m in merged:
            prev = next((tr for tr in self.tracks if math.hypot(tr['x'] - m['x'], tr['y'] - m['y']) < 0.06), None)
            m['count'] = (prev['count'] + 1) if prev else 1
            m['last'] = now
            new_tracks.append(m)
        self.tracks = new_tracks
        need = int(self.get_parameter('min_frames').value)
        stable = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k not in ('count', 'last', 'angle')}
                  for m in self.tracks if m['count'] >= need]
        self.pub.publish(String(data=json.dumps({'stamp': time.time(), 'obstacles': stable})))
        dt = time.monotonic() - t_tick
        if dt > 0.3:
            self.get_logger().warn(f'obstacle cycle took {dt * 1000:.0f} ms', throttle_duration_sec=30.0)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleGuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
