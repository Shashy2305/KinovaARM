#!/usr/bin/env python3
"""
three_camera_subscriber.py
Subscribes to all 3 cameras simultaneously and prints per-topic health.

Camera topics:
  Global RealSense D435I  /global_camera/global_camera/color/image_raw
                          /global_camera/global_camera/depth/color/points
  Wrist camera (Kinova)   /camera/color/image_raw
                          /camera/depth/image_raw
  OAK-D (standalone)      /global_camera/color/image_raw
                          /global_camera/stereo/points

Run (from the repo root):
  source ~/workspace/ros2_kortex_ws/install/setup.bash
  python3 utils/three_camera_subscriber.py
"""

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2


class ThreeCameraSubscriber(Node):

    TOPICS = {
        "realsense_color": ("/global_camera/global_camera/color/image_raw",   Image),
        "realsense_pcl":   ("/global_camera/global_camera/depth/color/points", PointCloud2),
        "wrist_color":     ("/camera/color/image_raw",                         Image),
        "wrist_depth":     ("/camera/depth/image_raw",                         Image),
        "oak_color":       ("/global_camera/color/image_raw",                  Image),
        "oak_pcl":         ("/global_camera/stereo/points",                    PointCloud2),
    }

    def __init__(self):
        super().__init__("three_camera_subscriber")
        self._counts = {k: 0 for k in self.TOPICS}
        self._last   = {k: 0.0 for k in self.TOPICS}
        self._hz     = {k: 0.0 for k in self.TOPICS}

        for name, (topic, msg_type) in self.TOPICS.items():
            self.create_subscription(
                msg_type, topic,
                lambda msg, n=name: self._cb(n, msg),
                10,
            )

        self.create_timer(2.0, self._report)
        self.get_logger().info("3-camera subscriber started — status every 2 s")

    def _cb(self, name: str, msg):
        now = time.monotonic()
        dt = now - self._last[name]
        if dt > 0:
            self._hz[name] = 1.0 / dt
        self._last[name] = now
        self._counts[name] += 1

    def _report(self):
        now = time.monotonic()
        lines = ["── Camera health ─────────────────────────────────────────"]
        for name, (topic, _) in self.TOPICS.items():
            age  = now - self._last[name] if self._last[name] > 0 else float("inf")
            hz   = self._hz[name]
            total = self._counts[name]
            if total == 0:
                status = "NO DATA"
            elif age > 3.0:
                status = f"STALE  ({age:.1f}s ago)"
            else:
                status = f"OK  {hz:.1f} Hz  ({total} msgs)"
            lines.append(f"  {name:<18} {status}")
        self.get_logger().info("\n".join(lines))


def main():
    rclpy.init()
    node = ThreeCameraSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
