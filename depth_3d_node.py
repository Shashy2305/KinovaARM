#!/usr/bin/env python3
"""
depth_3d_node.py  —  Kinova + OAK-D Pro Wide
Takes /detections (from object_detection.py) + depth image,
back-projects to 3D, transforms to robot base_link frame.

Publishes:
  /bottle_position               → PointStamped  (camera frame, legacy)
  /object_position_robot_frame   → PointStamped  (robot base_link frame)
  /detections_3d                 → String JSON   (enriched detections)

Fixes vs previous version:
  • Reads intrinsics from /global_camera/depth/camera_info (not hardcoded)
  • Removed wrong depth-resolution scaling (sy = depth_h/480 was wrong — depth is 400)
  • cx_px / cy_px from object_detection.py are already in 640×400 space
  • No double-scaling anywhere
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
import json
import numpy as np


class Depth3DNode(Node):
    def __init__(self):
        super().__init__('depth_3d_node')
        self.bridge = CvBridge()
        self.depth_image = None

        # ✅ FIX: intrinsics loaded from camera_info, not hardcoded
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_scale = 0.001   # mm → m

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            CameraInfo, '/global_camera/depth/camera_info',
            self.camera_info_cb, 10)
        self.create_subscription(
            Image, '/global_camera/depth/image_raw',
            self.depth_cb, 10)
        self.create_subscription(
            String, '/detections',
            self.det_cb, 10)

        self.point_pub       = self.create_publisher(
            PointStamped, '/bottle_position', 10)
        self.robot_point_pub = self.create_publisher(
            PointStamped, '/object_position_robot_frame', 10)
        self.det3d_pub       = self.create_publisher(
            String, '/detections_3d', 10)

        self.get_logger().info(
            'Depth 3D node ready — waiting for camera_info to load intrinsics...'
        )

    # ── Load intrinsics once from camera_info ────────────────────────────────
    def camera_info_cb(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(
                f'Intrinsics loaded: fx={self.fx:.2f}  fy={self.fy:.2f}  '
                f'cx={self.cx:.2f}  cy={self.cy:.2f}  '
                f'res={msg.width}×{msg.height}'
            )

    # ── Depth image ──────────────────────────────────────────────────────────
    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')

    # ── Main detection callback ──────────────────────────────────────────────
    def det_cb(self, msg):
        if self.depth_image is None or self.fx is None:
            return

        try:
            detections = json.loads(msg.data)
        except Exception:
            return

        if not isinstance(detections, list):
            detections = [detections]

        depth_h, depth_w = self.depth_image.shape
        enriched = []

        for det in detections:
            label = det.get('label') or det.get('class', 'unknown')
            conf  = float(det.get('confidence', 0.0))
            if conf < 0.45:
                continue

            # ── Get pixel centre ─────────────────────────────────────────────
            # object_detection.py publishes cx_px/cy_px in 640×400 space
            # depth image is also 640×400  →  NO scaling needed
            if 'cx_px' in det:
                cx_px = int(det['cx_px'])
                cy_px = int(det['cy_px'])
            elif 'center' in det:
                cx_px, cy_px = det['center']
            else:
                # fallback: use 3D coords from object_detection directly
                enriched.append({
                    'label': label, 'confidence': conf,
                    'cx_3d':    det.get('cx_3d', 0.0),
                    'cy_3d':    det.get('cy_3d', 0.0),
                    'cz_3d':    det.get('cz_3d', 0.0),
                    'x_robot':  det.get('cx_3d', 0.0),
                    'y_robot':  det.get('cy_3d', 0.0),
                    'z_robot':  det.get('cz_3d', 0.0),
                })
                continue

            # ── Sample depth with median patch ───────────────────────────────
            margin = 7
            # ✅ FIX: cx_px is already in depth image coords — clamp only
            dx = max(0, min(int(cx_px), depth_w - 1))
            dy = max(0, min(int(cy_px), depth_h - 1))

            region = self.depth_image[
                max(0, dy - margin): min(depth_h, dy + margin),
                max(0, dx - margin): min(depth_w, dx + margin),
            ]
            valid = region[region > 0]
            if len(valid) == 0:
                self.get_logger().warn(
                    f'{label}: no depth at ({dx},{dy}), skipping'
                )
                continue

            depth_mm = float(np.median(valid))
            depth_m  = depth_mm * self.depth_scale

            # ── Back-project to camera frame ──────────────────────────────────
            X_cam = (float(cx_px) - self.cx) * depth_m / self.fx
            Y_cam = (float(cy_px) - self.cy) * depth_m / self.fy
            Z_cam = depth_m

            # ── Publish in camera frame ───────────────────────────────────────
            pt_cam = PointStamped()
            pt_cam.header.stamp    = self.get_clock().now().to_msg()
            pt_cam.header.frame_id = 'global_camera_link'
            pt_cam.point.x = X_cam
            pt_cam.point.y = Y_cam
            pt_cam.point.z = Z_cam
            self.point_pub.publish(pt_cam)

            # ── Transform to robot base_link frame ───────────────────────────
            x_robot, y_robot, z_robot = X_cam, Y_cam, Z_cam   # fallback

            try:
                tf = self.tf_buffer.lookup_transform(
                    'base_link', 'global_camera_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1),
                )
                pt_robot = tf2_geometry_msgs.do_transform_point(pt_cam, tf)
                x_robot  = pt_robot.point.x
                y_robot  = pt_robot.point.y
                z_robot  = pt_robot.point.z
                self.robot_point_pub.publish(pt_robot)
                self.get_logger().info(
                    f'{label} ({conf:.0%})  '
                    f'cam=[{X_cam:.3f}, {Y_cam:.3f}, {Z_cam:.3f}]  '
                    f'robot=[{x_robot:.3f}, {y_robot:.3f}, {z_robot:.3f}]'
                )
            except Exception as e:
                self.get_logger().warn(
                    f'TF camera→base_link failed ({e}) — using camera frame'
                )

            enriched.append({
                'label':      label,
                'confidence': round(conf, 3),
                # camera optical frame
                'cx_3d':   round(X_cam, 4),
                'cy_3d':   round(Y_cam, 4),
                'cz_3d':   round(Z_cam, 4),
                # robot base_link frame
                'x_robot': round(x_robot, 4),
                'y_robot': round(y_robot, 4),
                'z_robot': round(z_robot, 4),
            })

        if enriched:
            self.det3d_pub.publish(String(data=json.dumps(enriched)))


def main():
    rclpy.init()
    rclpy.spin(Depth3DNode())


if __name__ == '__main__':
    main()
