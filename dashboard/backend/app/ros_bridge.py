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
import tf2_ros
from cv_bridge import CvBridge
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

from . import config

# camera name -> (point cloud topic, has_rgb, fixed display color if no rgb)
POINTCLOUD_TOPICS = {
    'oakd':      {'topic': '/global_camera/stereo/points', 'rgb': True},
    'realsense': {'topic': '/global_camera/global_camera/depth/color/points', 'rgb': True},
    'wrist':     {'topic': '/camera/depth/color/points', 'rgb': False, 'tint': (0.55, 0.75, 1.0)},
}
MAX_POINTS_PER_CAMERA = 3000
BASE_FRAME = 'base_link'

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

        # Point clouds for the 3D fusion view. Only the latest raw message
        # per camera is kept here (cheap) -- decoding/transforming/
        # downsampling happens lazily in get_fused_points(), rate-limited by
        # whoever calls it (the fusion WS loop), not by camera framerate.
        self._pc_lock = threading.Lock()
        self._latest_pc = {name: None for name in POINTCLOUD_TOPICS}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        for name, cfg in POINTCLOUD_TOPICS.items():
            self.create_subscription(
                PointCloud2, cfg['topic'],
                lambda msg, n=name: self._on_pointcloud(n, msg), 1)

    def _on_pointcloud(self, name, msg):
        with self._pc_lock:
            self._latest_pc[name] = msg

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

    def get_fused_points(self):
        """Decode, TF-transform into base_link, and downsample each
        camera's latest point cloud, returning one interleaved
        [x,y,z,r,g,b, x,y,z,r,g,b, ...] float32 array (colors in 0-1) --
        this IS the actual sensor fusion (positions registered into one
        common frame via the same calibration the rest of the pipeline
        uses), not just three images shown side by side like the
        MultiCameraView tiles are."""
        with self._pc_lock:
            snapshot = dict(self._latest_pc)

        chunks = []
        for name, cfg in POINTCLOUD_TOPICS.items():
            msg = snapshot.get(name)
            if msg is None:
                continue
            try:
                field_names = ['x', 'y', 'z', 'rgb'] if cfg['rgb'] else ['x', 'y', 'z']
                pts = point_cloud2.read_points(msg, field_names=field_names, skip_nans=True)
                if len(pts) == 0:
                    continue
                xyz = np.column_stack([pts['x'], pts['y'], pts['z']]).astype(np.float32)

                if len(xyz) > MAX_POINTS_PER_CAMERA:
                    idx = np.random.choice(len(xyz), MAX_POINTS_PER_CAMERA, replace=False)
                    xyz = xyz[idx]
                    pts = pts[idx]

                try:
                    tf = self.tf_buffer.lookup_transform(
                        BASE_FRAME, msg.header.frame_id, rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.2))
                except Exception:
                    continue  # no TF yet for this camera -- skip this round, not fatal
                q = tf.transform.rotation
                t = tf.transform.translation
                R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                xyz_base = (R @ xyz.T).T + np.array([t.x, t.y, t.z], dtype=np.float32)

                if cfg['rgb']:
                    rgb_u32 = pts['rgb'].view(np.uint32)
                    r = ((rgb_u32 >> 16) & 0xFF).astype(np.float32) / 255.0
                    g = ((rgb_u32 >> 8) & 0xFF).astype(np.float32) / 255.0
                    b = (rgb_u32 & 0xFF).astype(np.float32) / 255.0
                    rgb = np.column_stack([r, g, b])
                else:
                    rgb = np.tile(np.array(cfg['tint'], dtype=np.float32), (len(xyz_base), 1))

                interleaved = np.hstack([xyz_base, rgb]).astype(np.float32)
                chunks.append(interleaved)
            except Exception as e:
                self.get_logger().warn(f'get_fused_points: {name} failed: {e}', throttle_duration_sec=5.0)
                continue

        if not chunks:
            return np.zeros((0, 6), dtype=np.float32)
        return np.vstack(chunks)


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
        # Defaults to os.cpu_count() worker threads (32 on this machine) --
        # massive overkill for the handful of subscriptions this backend
        # has (bridge + up to 3 boundary tools + 1 calibrator). Capping it
        # keeps the process's baseline thread count sane.
        _executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
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
_boundary_tools = {}  # camera_name -> DefineWorkspaceBoundary instance


def get_or_create_calibrator():
    global _calibrator
    executor = _ensure_executor()
    if _calibrator is None:
        from multi_camera_calibrate import MultiCameraCalibrate  # local import: needs sys.path above
        _calibrator = MultiCameraCalibrate()
        executor.add_node(_calibrator)
    return _calibrator


def get_or_create_boundary_tool(camera_name='oakd'):
    """One DefineWorkspaceBoundary instance per camera, not just oakd --
    OAK-D is both far from the table and the camera with the most
    calibration uncertainty, so a pixel-click error there turns into a
    much bigger real-world error than the same click on, say, the wrist
    camera (which has zero calibration error, being driven by exact
    kinematics, and can be jogged right up to each corner). Switching
    camera_name starts a fresh point list for that camera's own tool --
    points aren't shared across cameras, so pick one camera for a whole
    boundary-definition session rather than mixing."""
    global _boundary_tools
    executor = _ensure_executor()
    if camera_name not in _boundary_tools:
        from define_workspace_boundary import DefineWorkspaceBoundary  # noqa
        tool = DefineWorkspaceBoundary(camera_name)
        executor.add_node(tool)
        _boundary_tools[camera_name] = tool
    return _boundary_tools[camera_name]
