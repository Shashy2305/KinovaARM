"""
The dashboard's one long-lived rclpy context. `RosBridge` subscribes to
every status/scene topic the UI needs and to each camera's image topic
(re-encoded to JPEG on arrival for the MJPEG stream endpoints). Calibration
tools (multi_camera_calibrate.py's MultiCameraCalibrate,
define_workspace_boundary.py's DefineWorkspaceBoundary) are created lazily
as separate nodes in this SAME rclpy context when the calibration wizard
is first opened, each spun on its own background thread — this mirrors
exactly how those tools already avoid the threading bug hit during
development (see multi_camera_calibrate.py's own comments): never drive a
blocking ROS call from a thread nothing is spinning.
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
from sensor_msgs.msg import Image
from std_msgs.msg import String

from . import config

REPO_ROOT = os.path.expanduser('~/Shashproject')
sys.path.insert(0, os.path.join(REPO_ROOT, 'calibration'))

_ros_initialized = False


def init_ros():
    global _ros_initialized
    if not _ros_initialized:
        rclpy.init()
        _ros_initialized = True


class RosBridge(Node):
    def __init__(self):
        super().__init__('dashboard_bridge')
        self._lock = threading.Lock()
        self.bridge = CvBridge()

        self.latest = {
            'camera_calibration_status': None,
            'scene_snapshot': None,
            'planner_status': None,
            'pick_place_status': None,
            'arm_status': None,
        }
        self._jpeg = {name: None for name in config.CAMERA_TOPICS}
        self._frame_ts = {name: 0.0 for name in config.CAMERA_TOPICS}

        self.create_subscription(String, '/camera_calibration_status',
                                  self._mk_status_cb('camera_calibration_status'), 10)
        self.create_subscription(String, '/scene_snapshot',
                                  self._mk_status_cb('scene_snapshot'), 10)
        self.create_subscription(String, '/planner_status',
                                  self._mk_status_cb('planner_status'), 10)
        self.create_subscription(String, '/pick_place_status',
                                  self._mk_status_cb('pick_place_status'), 10)
        self.create_subscription(String, '/arm_status',
                                  self._mk_status_cb('arm_status'), 10)

        for name, topics in config.CAMERA_TOPICS.items():
            self.create_subscription(
                Image, topics['image'],
                lambda msg, n=name: self._on_image(n, msg), 10)

        self.voice_pub = self.create_publisher(String, '/voice_command', 10)

    def _mk_status_cb(self, key):
        def cb(msg):
            with self._lock:
                self.latest[key] = msg.data
        return cb

    def _on_image(self, name, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return
        with self._lock:
            self._jpeg[name] = buf.tobytes()
            self._frame_ts[name] = time.monotonic()

    # ── read side (called from FastAPI request handlers / other threads) ──
    def get_status(self):
        with self._lock:
            out = {}
            for key, raw in self.latest.items():
                if raw is None:
                    out[key] = None
                    continue
                try:
                    out[key] = json.loads(raw)
                except json.JSONDecodeError:
                    out[key] = raw
            return out

    def get_jpeg(self, camera_name):
        with self._lock:
            jpeg = self._jpeg.get(camera_name)
            ts = self._frame_ts.get(camera_name, 0.0)
        stale = (time.monotonic() - ts) > 3.0 if ts else True
        return jpeg, stale

    def publish_voice_command(self, text):
        self.voice_pub.publish(String(data=text))


_bridge = None
# One shared executor + one shared spin thread for every node this backend
# owns (the bridge, plus the calibration/boundary tools created lazily
# below). Discovered the hard way: rclpy.spin(node) implicitly builds its
# own executor per call, and running several of those concurrently in
# separate threads against nodes in the SAME context crashes with
# "ValueError: generator already executing" — they're not as independent
# as they look. One MultiThreadedExecutor, nodes added/removed from it
# dynamically (both supported while it's already spinning), avoids that
# entirely.
_executor = None
_executor_thread = None


def _ensure_executor():
    global _executor, _executor_thread
    init_ros()
    if _executor is None:
        _executor = rclpy.executors.MultiThreadedExecutor()
        _executor_thread = threading.Thread(target=_executor.spin, daemon=True)
        _executor_thread.start()
    return _executor


def start_bridge():
    global _bridge
    executor = _ensure_executor()
    if _bridge is None:
        _bridge = RosBridge()
        executor.add_node(_bridge)
    return _bridge


def get_bridge():
    if _bridge is None:
        raise RuntimeError('ROS bridge not started yet — call start_bridge() first')
    return _bridge


# ── lazily-created calibration-tool nodes, same rclpy context/executor ──
_calibrator = None
_boundary_tool = None


def get_or_create_calibrator():
    global _calibrator
    executor = _ensure_executor()
    if _calibrator is None:
        from multi_camera_calibrate import MultiCameraCalibrate  # local import: needs sys.path above
        _calibrator = MultiCameraCalibrate()
        executor.add_node(_calibrator)
    return _calibrator


def get_or_create_boundary_tool():
    global _boundary_tool
    executor = _ensure_executor()
    if _boundary_tool is None:
        from define_workspace_boundary import DefineWorkspaceBoundary  # noqa
        _boundary_tool = DefineWorkspaceBoundary('oakd')
        executor.add_node(_boundary_tool)
    return _boundary_tool
