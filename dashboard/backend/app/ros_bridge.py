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
import collections
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
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32, String

from . import config, fusion_cloud
from .joint_feed import JointFeed

sys.path.insert(0, os.path.join(config.REPO_ROOT, 'src', 'thesis_robot', 'thesis_robot'))
import camera_calibration_files as ccf  # noqa: E402

# camera name -> (point cloud topic, has_rgb, fixed display color if no rgb)
POINTCLOUD_TOPICS = {
    'oakd':       {'topic': '/global_camera/stereo/points', 'rgb': True},
    'realsense':  {'topic': '/global_camera/global_camera/depth/color/points', 'rgb': True},
    'realsense2': {'topic': '/global_camera_2/global_camera_2/depth/color/points', 'rgb': True},
    'realsense3': {'topic': '/global_camera_3/global_camera_3/depth/color/points', 'rgb': True},
    'wrist':      {'topic': '/camera/depth/color/points', 'rgb': False, 'tint': (0.55, 0.75, 1.0)},
}
MAX_POINTS_PER_CAMERA = 3000          # the wrist cloud only (a fixed stride, not random); the RealSense clouds are built from their depth images
# camera -> (aligned depth image topic, its camera_info); the fused cloud is built from these, not from the drivers' PointCloud2 (see fusion_cloud.py)
DEPTH_TOPICS = {
    'realsense':  ('/global_camera/global_camera/aligned_depth_to_color/image_raw', '/global_camera/global_camera/aligned_depth_to_color/camera_info'),
    'realsense2': ('/global_camera_2/global_camera_2/aligned_depth_to_color/image_raw', '/global_camera_2/global_camera_2/aligned_depth_to_color/camera_info'),
    'realsense3': ('/global_camera_3/global_camera_3/aligned_depth_to_color/image_raw', '/global_camera_3/global_camera_3/aligned_depth_to_color/camera_info'),
}
JPEG_MIN_PERIOD_S = 0.08              # the MJPEG stream sends 10 fps: encoding every frame of four cameras at full size was a major CPU cost
BASE_FRAME = 'base_link'

REPO_ROOT = config.REPO_ROOT
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
            'audio_status': None,
            'voice_transcript': None,
            'unknown_obstacles': None,
        }
        self._audio_level = (0.0, 0.0)   # (rms, monotonic time received)
        # Every status message, in order. Status topics flip in milliseconds
        # (planner: PLANNING -> EXECUTING -> READY) so sampling them at the
        # UI's 2 Hz misses states; the UI replays this log instead.
        self._events = collections.deque(maxlen=400)
        self._event_seq = 0
        self._last_event_text = {}
        self._jpeg = {name: None for name in config.CAMERA_TOPICS}
        self._frame_ts = {name: 0.0 for name in config.CAMERA_TOPICS}
        self._frame_ids = {}      # camera -> frame_id of its latest image (the optical frame the pipeline projects with)
        self._image_size = {}

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
        self.create_subscription(String, '/unknown_obstacles',
                                  self._mk_status_cb('unknown_obstacles'), 5)
        self.create_subscription(String, '/audio_status',
                                  self._mk_status_cb('audio_status'), 10)
        self.create_subscription(String, '/voice_transcript',
                                  self._mk_status_cb('voice_transcript'), 10)
        self.create_subscription(Float32, '/audio_level', self._on_audio_level, 10)
        self.audio_control_pub = self.create_publisher(String, '/audio_control', 10)

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
        self._raw_color = {}
        self._depth_hist = {n: collections.deque(maxlen=3) for n in DEPTH_TOPICS}      # (monotonic time, Image) of the last depth frames
        self._depth_K = {}
        self._fusion_stats = {}
        self._pair_cache = (0.0, {})
        for n, (dtopic, itopic) in DEPTH_TOPICS.items():
            self.create_subscription(Image, dtopic, lambda m, n=n: self._depth_hist[n].append((time.monotonic(), m)), 1)
            self.create_subscription(CameraInfo, itopic, lambda m, n=n: self._depth_K.__setitem__(n, np.array(m.k).reshape(3, 3)), 1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # facts for the health banner (health_issues.py): the joint feedback (and whether it is frozen), the driver's fault flag, whether the arm TF exists, and how late this node's own
        # timer runs (a starved executor shows up here first)
        self.joint_feed = JointFeed(self)
        self._joint_last = None
        self._joint_changed_t = time.monotonic()
        self._fault = None
        self._tf_ok = None
        self._health_tick_t = time.monotonic()
        self._started_t = time.monotonic()
        self.create_subscription(Bool, '/fault_controller/internal_fault', self._on_fault, 5)
        self.create_timer(0.5, self._health_tick)
        # Only the wrist's native cloud is read (get_fused_points); the static cameras are fused from their depth images. Subscribing to their full-resolution driver clouds
        # decoded megabytes per message at 30 Hz for nothing and starved the camera-image callbacks (no frames reached the dashboard, 2026-10-22).
        for name, cfg in POINTCLOUD_TOPICS.items():
            if name != 'wrist':
                continue
            self.create_subscription(
                PointCloud2, cfg['topic'],
                lambda msg, n=name: self._on_pointcloud(n, msg), 1)

    def _on_pointcloud(self, name, msg):
        with self._pc_lock:
            self._latest_pc[name] = msg

    def _on_fault(self, msg):
        self._fault = bool(msg.data)

    def _health_tick(self):
        now = time.monotonic()
        self._health_tick_t = now
        pos, age = self.joint_feed.latest()
        vec = tuple(pos.get(f'joint_{i}') for i in range(1, 8))
        if age < 1.0 and None not in vec:
            if vec != self._joint_last:
                self._joint_last = vec
                self._joint_changed_t = now
        else:
            self._joint_changed_t = now                       # no feed: "missing" is reported, not "frozen"
        try:
            self._tf_ok = bool(self.tf_buffer.can_transform('base_link', 'end_effector_link', rclpy.time.Time()))
        except Exception:
            self._tf_ok = False

    def get_health_facts(self):
        """The raw facts health_issues.evaluate() needs (cheap: everything was gathered by _health_tick and the callbacks)."""
        now = time.monotonic()
        _pos, age = self.joint_feed.latest()
        return {'joint_age_s': None if age > 1e8 else age, 'frozen_for_s': now - self._joint_changed_t if age < 1.0 else 0.0,
                'fault': self._fault, 'tf_ok': self._tf_ok, 'executor_lag_s': max(0.0, now - self._health_tick_t - 0.5), 'uptime_s': now - self._started_t}

    def _on_audio_level(self, msg):
        self._audio_level = (float(msg.data), time.monotonic())

    def publish_audio_control(self, action):
        self.audio_control_pub.publish(String(data=action))

    def get_audio_state(self):
        """Small, cheap snapshot for the 10 Hz audio WebSocket."""
        with self._lock:
            status = self.latest.get('audio_status')
            raw = self.latest.get('voice_transcript')
        level, ts = self._audio_level
        try:
            transcript = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            transcript = None
        age = time.monotonic() - ts if ts else None
        return {
            'status': status,
            'level': level if (age is not None and age < 1.0) else 0.0,
            'mic_alive': age is not None and age < 1.0,
            'transcript': transcript,
        }

    _EVENT_KEYS = ('planner_status', 'arm_status', 'pick_place_status', 'audio_status')

    def _mk_status_cb(self, key):
        def cb(msg):
            with self._lock:
                self.latest[key] = msg.data
                if key in self._EVENT_KEYS and msg.data != self._last_event_text.get(key):
                    self._last_event_text[key] = msg.data
                    self._event_seq += 1
                    self._events.append({'seq': self._event_seq, 'key': key,
                                         'text': msg.data, 't': time.time()})
        return cb

    def current_event_seq(self):
        with self._lock:
            return self._event_seq

    def get_events_since(self, seq):
        with self._lock:
            return [e for e in self._events if e['seq'] > seq]

    def _on_image(self, name, msg):
        now = time.monotonic()
        with self._lock:
            self._raw_color[name] = msg                                     # the fused cloud takes its colours from this
            self._frame_ts[name] = now if self._jpeg.get(name) is None else self._frame_ts[name]
            if self._jpeg.get(name) is not None and now - self._frame_ts[name] < JPEG_MIN_PERIOD_S:
                return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return
        with self._lock:
            self._jpeg[name] = buf.tobytes()
            self._frame_ts[name] = time.monotonic()
            self._frame_ids[name] = msg.header.frame_id
            self._image_size[name] = (msg.width, msg.height)

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
        # Is anything publishing the arm's joint states? A cheap graph query (no 900 Hz subscription). On 2026-10-09 the
        # robot driver froze with the process alive, the dashboard said "bringup up" and every arm command timed out.
        try:
            out['joint_state_publishers'] = int(self.count_publishers('/joint_states'))
        except Exception:
            out['joint_state_publishers'] = None
        return out

    def get_image_meta(self, camera_name):
        with self._lock:
            return self._frame_ids.get(camera_name), self._image_size.get(camera_name)

    def get_jpeg(self, camera_name):
        with self._lock:
            jpeg = self._jpeg.get(camera_name)
            ts = self._frame_ts.get(camera_name, 0.0)
        stale = (time.monotonic() - ts) > 3.0 if ts else True
        return jpeg, stale

    def publish_voice_command(self, text):
        self.voice_pub.publish(String(data=text))

    def _tf_matrix(self, frame_id):
        tf = self.tf_buffer.lookup_transform(BASE_FRAME, frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.2))
        q, t = tf.transform.rotation, tf.transform.translation
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def get_fused_points(self, mode='rgb'):
        """One interleaved float32 [x,y,z,r,g,b, ...] array in base_link (colours 0-1) of ALL cameras merged: see fusion_cloud.py. Deterministic (a fixed
        pixel grid, a temporal median of the last depth frames, flying-pixel removal, a workspace crop and a voxel merge), so the picture is stable.
        mode: 'rgb' the camera colours, 'camera' one colour per camera (white where several agree), 'overlap' by number of cameras.
        Per-camera status and pair statistics are left in self._fusion_stats (get_fusion_stats())."""
        t0 = time.monotonic()
        clouds, cams = {}, {}
        for name in DEPTH_TOPICS:
            with self._lock:
                hist = list(self._depth_hist[name])
                color_msg = self._raw_color.get(name)
                K = self._depth_K.get(name)
            if not hist or color_msg is None or K is None:
                cams[name] = {'status': 'no data', 'points': 0}
                continue
            age = time.monotonic() - hist[-1][0]
            if age > 3.0:
                cams[name] = {'status': 'stale', 'points': 0, 'age_s': round(age, 1)}
                continue
            try:
                if not ccf.has_calibration(name):
                    raise RuntimeError('no calibration file')
                T = self._tf_matrix(hist[-1][1].header.frame_id)
            except Exception:
                cams[name] = {'status': 'no calibration (TF)', 'points': 0}
                continue
            try:
                frames = [self.bridge.imgmsg_to_cv2(m, '16UC1') for _, m in hist if (m.width, m.height) == (hist[-1][1].width, hist[-1][1].height)]
                rgb = self.bridge.imgmsg_to_cv2(color_msg, 'rgb8')
                pts, col = fusion_cloud.camera_points(frames, rgb, K, T)
            except Exception as e:
                self.get_logger().warn(f'get_fused_points: {name} failed: {e}', throttle_duration_sec=5.0)
                cams[name] = {'status': 'error', 'points': 0}
                continue
            clouds[name] = (pts, col)
            cams[name] = {'status': 'ok', 'points': int(len(pts)), 'frames': len(frames), 'age_s': round(age, 2), 'size': [hist[-1][1].width, hist[-1][1].height]}

        # the wrist camera: its native depth cloud (no colour), a fixed stride instead of a random sample
        with self._pc_lock:
            wmsg = self._latest_pc.get('wrist')
        if wmsg is not None:
            try:
                pts = point_cloud2.read_points(wmsg, field_names=['x', 'y', 'z'], skip_nans=True)
                xyz = np.column_stack([pts['x'], pts['y'], pts['z']]).astype(np.float32) if len(pts) else np.zeros((0, 3), np.float32)
                if len(xyz) > MAX_POINTS_PER_CAMERA:
                    xyz = xyz[:: len(xyz) // MAX_POINTS_PER_CAMERA + 1]
                T = self._tf_matrix(wmsg.header.frame_id)
                xb = xyz @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
                keep = np.all((xb >= fusion_cloud.CROP_MIN) & (xb <= fusion_cloud.CROP_MAX), axis=1)
                tint = np.tile(np.array(POINTCLOUD_TOPICS['wrist']['tint'], dtype=np.float32), (int(keep.sum()), 1))
                clouds['wrist'] = (xb[keep], tint)
                cams['wrist'] = {'status': 'ok', 'points': int(keep.sum())}
            except Exception:
                cams['wrist'] = {'status': 'no calibration (TF)', 'points': 0}

        if not clouds:
            with self._lock:
                self._fusion_stats = {'time': time.time(), 'cameras': cams, 'pairs': {}, 'merged': 0, 'ms': 0}
            return np.zeros((0, 6), dtype=np.float32)
        p, c, mask = fusion_cloud.voxel_merge(clouds)
        out = np.hstack([p, fusion_cloud.colorize(c, mask, mode)]).astype(np.float32)
        # pair statistics are a few ms of KD-tree each: refresh them every 2 s, not every frame
        now = time.monotonic()
        if now - self._pair_cache[0] > 2.0:
            try:
                self._pair_cache = (now, fusion_cloud.pair_stats({k: v for k, v in clouds.items() if k != 'wrist'}))
            except Exception:
                self._pair_cache = (now, {})
        with self._lock:
            self._fusion_stats = {'time': time.time(), 'cameras': cams, 'pairs': self._pair_cache[1], 'merged': int(len(p)),
                                  'ms': round(1000 * (time.monotonic() - t0), 1), 'mode': mode}
        return out

    def get_fusion_stats(self):
        with self._lock:
            return dict(self._fusion_stats)


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
_table_recorder = None


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


def get_or_create_table_recorder():
    global _table_recorder
    executor = _ensure_executor()
    if _table_recorder is None:
        from .table_recorder import TableRecorder
        _table_recorder = TableRecorder()
        executor.add_node(_table_recorder)
    return _table_recorder


_arm_calib = None


_jogger = None


def get_or_create_jogger():
    global _jogger
    executor = _ensure_executor()
    if _jogger is None:
        from .jog import Jogger
        _jogger = Jogger()
        executor.add_node(_jogger)
    return _jogger


def get_or_create_arm_calib():
    global _arm_calib
    executor = _ensure_executor()
    if _arm_calib is None:
        from .arm_calib import ArmCalib
        _arm_calib = ArmCalib()
        executor.add_node(_arm_calib)
    return _arm_calib
