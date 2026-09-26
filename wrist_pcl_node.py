#!/usr/bin/env python3
"""
wrist_pcl_node.py
Generates a PointCloud2 from the wrist camera's native depth image.

kinova_vision with depth_registration=false publishes the raw 480×270 depth
image but does NOT create a point cloud (that requires the RegisterNode +
PointCloudXyzrgbNode component which only loads under depth_registration=true).
This node fills that gap.

Subscribes:
  /camera/depth/image_raw      sensor_msgs/Image     16UC1  480×270  depth (mm)
  /camera/depth/camera_info    sensor_msgs/CameraInfo

Publishes:
  /camera/depth/color/points   sensor_msgs/PointCloud2  XYZ float32
  frame_id: camera_depth_frame
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from cv_bridge import CvBridge

_QOS_SENSOR = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# Kinova wrist camera depth intrinsics at 480×270 (from kinova_vision calibration).
# Updated automatically when /camera/depth/camera_info arrives.
_DEFAULT_FX = 548.5
_DEFAULT_FY = 548.5
_DEFAULT_CX = 240.0
_DEFAULT_CY = 135.0

MIN_Z_M = 0.05
MAX_Z_M = 4.0

OUTPUT_FRAME = "camera_depth_frame"


class WristPclNode(Node):
    def __init__(self):
        super().__init__("wrist_pcl_node")
        self._bridge = CvBridge()
        self._fx = _DEFAULT_FX
        self._fy = _DEFAULT_FY
        self._cx = _DEFAULT_CX
        self._cy = _DEFAULT_CY

        self.create_subscription(
            CameraInfo, "/camera/depth/camera_info",
            self._on_camera_info, _QOS_SENSOR,
        )
        self.create_subscription(
            Image, "/camera/depth/image_raw",
            self._on_depth, _QOS_SENSOR,
        )
        self._pub = self.create_publisher(
            PointCloud2, "/camera/depth/color/points", 10
        )
        self.get_logger().info(
            f"Wrist PCL node ready"
            f"\n  depth in : /camera/depth/image_raw"
            f"\n  cloud out: /camera/depth/color/points  frame={OUTPUT_FRAME}"
            f"\n  z range  : [{MIN_Z_M}, {MAX_Z_M}] m"
        )

    def _on_camera_info(self, msg: CameraInfo):
        if msg.k[0] > 0.0:
            self._fx, self._fy = msg.k[0], msg.k[4]
            self._cx, self._cy = msg.k[2], msg.k[5]

    def _on_depth(self, msg: Image):
        try:
            depth_mm = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}", throttle_duration_sec=5.0)
            return

        depth_mm = depth_mm.astype(np.float32)
        h, w = depth_mm.shape

        uu, vv = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )

        z = depth_mm * 0.001  # mm → m
        valid = (z > MIN_Z_M) & (z < MAX_Z_M)

        z_v  = z[valid]
        x_v  = (uu[valid] - self._cx) * z_v / self._fx
        y_v  = (vv[valid] - self._cy) * z_v / self._fy

        n   = len(x_v)
        pts = np.column_stack([
            x_v.astype(np.float32),
            y_v.astype(np.float32),
            z_v.astype(np.float32),
        ])

        cloud = PointCloud2()
        cloud.header.stamp    = msg.header.stamp
        cloud.header.frame_id = OUTPUT_FRAME
        cloud.height          = 1
        cloud.width           = n
        cloud.is_dense        = True
        cloud.is_bigendian    = False
        cloud.point_step      = 12
        cloud.row_step        = 12 * n
        cloud.fields          = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        cloud.data = pts.tobytes()
        self._pub.publish(cloud)


def main():
    rclpy.init()
    rclpy.spin(WristPclNode())


if __name__ == "__main__":
    main()
