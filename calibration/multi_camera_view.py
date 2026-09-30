#!/usr/bin/env python3
"""
multi_camera_view.py — one window showing all three camera feeds side by
side, with a live ChArUco-detected/not-detected border on each and the
camera_calibration_status badge from camera_watchdog.

Use this BEFORE running multi_camera_calibrate.py: put the board where you
want it, then watch this window and jog the arm / reposition OAK-D or
RealSense until all three panels show a green "BOARD DETECTED" border at
once. That's the moment multi_camera_calibrate.py will succeed.

Not a colcon node — run directly like the other calibration/ tools:
  python3 calibration/multi_camera_view.py
"""
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(__file__))
from markers.charuco_detector import detect_pose, make_board_and_detector  # noqa: E402

TILE_W, TILE_H = 480, 360
STALE_SEC = 2.0

CAMERAS = {
    'oakd':      {'image': '/global_camera/color/image_raw',
                  'info':  '/global_camera/color/camera_info'},
    'realsense': {'image': '/global_camera/global_camera/color/image_raw',
                  'info':  '/global_camera/global_camera/color/camera_info'},
    'wrist':     {'image': '/camera/color/image_raw',
                  'info':  '/camera/color/camera_info'},
}

GREEN = (60, 200, 60)
RED = (50, 50, 220)
AMBER = (30, 170, 220)
WHITE = (235, 235, 235)
BG = (25, 25, 25)


class CameraPanel:
    def __init__(self, node, name, topics, bridge, board, detector):
        self.name = name
        self.bridge = bridge
        self.board = board
        self.detector = detector
        self.bgr = None
        self.K = None
        self.dist = None
        self.last_frame_time = 0.0
        self.calib_status = 'unknown'
        node.create_subscription(Image, topics['image'], self._on_image, 10)
        node.create_subscription(CameraInfo, topics['info'], self._on_info, 10)

    def _on_image(self, msg):
        self.bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.last_frame_time = time.monotonic()

    def _on_info(self, msg):
        # Always update, not just on the first message: if the underlying
        # driver is restarted at a different resolution while this viewer
        # keeps running, stale intrinsics here would silently produce wrong
        # distances against the new image size.
        self.K = np.array(msg.k).reshape(3, 3)
        self.dist = np.array(msg.d)

    def render_tile(self):
        alive = (time.monotonic() - self.last_frame_time) < STALE_SEC if self.last_frame_time else False

        if not alive or self.bgr is None:
            tile = np.full((TILE_H, TILE_W, 3), BG, dtype=np.uint8)
            cv2.putText(tile, 'NO SIGNAL', (TILE_W // 2 - 90, TILE_H // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, RED, 2, cv2.LINE_AA)
            border = RED
        else:
            frame = cv2.resize(self.bgr, (TILE_W, TILE_H))
            border = (90, 90, 90)
            if self.K is not None:
                Ks = self.K.copy()
                sx = TILE_W / self.bgr.shape[1]
                sy = TILE_H / self.bgr.shape[0]
                Ks[0] *= sx
                Ks[1] *= sy
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                result = detect_pose(self.detector, self.board, gray, Ks, self.dist)
                if result is not None:
                    rvec, tvec = result
                    dist_cm = float(np.linalg.norm(tvec)) * 100
                    cv2.drawFrameAxes(frame, Ks, self.dist, rvec, tvec, 0.06)
                    cv2.putText(frame, f'BOARD  {dist_cm:.0f}cm', (10, 26),
                                cv2.FONT_HERSHEY_DUPLEX, 0.55, GREEN, 2, cv2.LINE_AA)
                    border = GREEN
                else:
                    cv2.putText(frame, 'no board', (10, 26),
                                cv2.FONT_HERSHEY_DUPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
            tile = frame

        cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), border, 4)

        status_color = {'ok': GREEN, 'needs_recalibration': RED}.get(self.calib_status, AMBER)
        label = f'{self.name}  [{self.calib_status}]'
        cv2.rectangle(tile, (0, TILE_H - 30), (TILE_W, TILE_H), (15, 15, 15), -1)
        cv2.putText(tile, label, (8, TILE_H - 9),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, status_color, 1, cv2.LINE_AA)
        return tile


class MultiCameraView(Node):
    def __init__(self):
        super().__init__('multi_camera_view')
        bridge = CvBridge()
        board, detector = make_board_and_detector()
        self.panels = {
            name: CameraPanel(self, name, topics, bridge, board, detector)
            for name, topics in CAMERAS.items()
        }
        self.create_subscription(
            String, '/camera_calibration_status', self._on_status, 10)

    def _on_status(self, msg):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        for name, panel in self.panels.items():
            if name in status:
                panel.calib_status = status[name]


def main():
    rclpy.init()
    node = MultiCameraView()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    cv2.namedWindow('Multi-Camera View', cv2.WINDOW_NORMAL)
    print('\nMulti-camera view — green border + axes = board detected in that '
          'camera right now. Press Q to quit.\n')

    try:
        while rclpy.ok():
            tiles = [node.panels[name].render_tile() for name in ('oakd', 'realsense', 'wrist')]
            canvas = np.hstack(tiles)
            cv2.imshow('Multi-Camera View', canvas)
            if cv2.waitKey(30) & 0xFF in (ord('q'), ord('Q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
