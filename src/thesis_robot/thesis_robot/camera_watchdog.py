#!/usr/bin/env python3
"""
Camera watchdog — enforces "block until recalibrated" for the
independently-mounted cameras (OAK-D, RealSense).

Their base_link -> camera TF comes from a one-shot calibration
(multi_camera_calibrate.py) that assumes the camera hasn't moved since.
If a camera's image topic goes stale and then comes back (unplugged/
replugged, USB dropout, physically bumped), that assumption is no longer
safe, so this node marks it as needing recalibration:
  ~/.ros/<camera>_needs_recalibration.flag   (presence = blocked)

and publishes /camera_calibration_status (JSON) so object_detection.py /
realsense_detection.py can stop publishing for that camera until an
operator reruns multi_camera_calibrate.py, which clears the flag. The
wrist camera never needs this — its extrinsic comes from the robot's own
kinematics and is never invalidated by a reconnect.
"""
import json
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

STALE_SEC = 6.0
CHECK_PERIOD_SEC = 0.5
FLAG_DIR = os.path.expanduser('~/.ros')

CAMERAS = {
    'oakd': '/global_camera/color/image_raw',
    'realsense': '/global_camera/global_camera/color/image_raw',
}


class CameraWatchdog(Node):
    def __init__(self):
        super().__init__('camera_watchdog')
        os.makedirs(FLAG_DIR, exist_ok=True)

        self._last_msg_time = {name: None for name in CAMERAS}
        self._was_stale = {name: False for name in CAMERAS}
        # Tracks whether a camera has ever been seen connected, so the
        # very first connection at startup isn't mistaken for a reconnect.
        self._ever_fresh = {name: False for name in CAMERAS}

        for name, topic in CAMERAS.items():
            self.create_subscription(
                Image, topic, lambda msg, n=name: self._on_image(n), 10)

        self.status_pub = self.create_publisher(String, '/camera_calibration_status', 10)
        self.create_timer(CHECK_PERIOD_SEC, self._tick)
        self.get_logger().info(
            'Camera watchdog ready — flags a camera for recalibration if its '
            'image topic drops and reconnects (bump/unplug), blocking its '
            'detections until multi_camera_calibrate.py is rerun.')

    def _on_image(self, name):
        self._last_msg_time[name] = time.monotonic()

    def _flag_path(self, name):
        return os.path.join(FLAG_DIR, f'{name}_needs_recalibration.flag')

    def _needs_recalibration(self, name):
        return os.path.exists(self._flag_path(name))

    def _flag_for_recalibration(self, name):
        with open(self._flag_path(name), 'w') as f:
            f.write(f'flagged at {time.time()}: image topic reconnected after a drop\n')
        self.get_logger().warn(
            f'{name}: reconnected after dropping out — flagged for '
            f'recalibration. Its detections are blocked until you rerun '
            f'calibration/multi_camera_calibrate.py.')

    def _tick(self):
        now = time.monotonic()
        for name in CAMERAS:
            last = self._last_msg_time[name]
            stale = last is None or (now - last) > STALE_SEC
            if not stale:
                if self._was_stale[name] and self._ever_fresh[name]:
                    # Was connected before, went stale, and just came back:
                    # a real reconnect, not the first-ever connection.
                    self._flag_for_recalibration(name)
                self._ever_fresh[name] = True
            self._was_stale[name] = stale

        status = {
            name: ('needs_recalibration' if self._needs_recalibration(name) else 'ok')
            for name in CAMERAS
        }
        status['wrist'] = 'ok'
        self.status_pub.publish(String(data=json.dumps(status)))


def main(args=None):
    rclpy.init(args=args)
    node = CameraWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
