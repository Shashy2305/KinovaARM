#!/usr/bin/env python3
"""
sensor_fusion_node.py  —  Multi-camera point cloud fusion for Kinova Gen3
==========================================================================
Fuses point clouds from all 3 cameras into one unified cloud for MoveIt2.

Input topics (subscribes):
  /global_camera/stereo/points     ← OAK-D full scene cloud (oak_camera_node.py)
  /global_camera/global_camera/depth/color/points  ← RealSense D435I
  /camera/depth/color/points       ← Kinova wrist camera (kinova_vision_node)

Output topics (publishes):
  /fused/points                    ← merged XYZRGB cloud in base_link
  /fused/status                    ← std_msgs/String — which cameras are live

MoveIt2 octomap config (in robot.launch.py or moveit config):
  point_cloud_topic: /fused/points
  max_range: 2.5
  padding_scale: 1.0

How fusion works:
  1. Each camera cloud arrives in its own TF frame
  2. TF2 transforms each cloud to base_link
  3. Voxel-grid downsampling (leaf=1cm) reduces density
  4. Statistical outlier removal cleans noise
  5. All clouds merged and published at 10 Hz

Run:
  source ~/workspace/ros2_kortex_ws/install/setup.bash
  python3 sensor_fusion_node.py

Verify:
  ros2 topic hz /fused/points
  ros2 topic echo /fused/status
"""

import struct
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped

import tf2_ros
from tf2_ros import Buffer, TransformListener
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# ── Config ────────────────────────────────────────────────────────────────────

# Output frame — everything fused into base_link for MoveIt2
OUTPUT_FRAME = "base_link"

# Publish rate (Hz)
PUBLISH_HZ = 10.0

# Voxel grid leaf size (metres) — smaller = denser but slower
VOXEL_LEAF = 0.01   # 1 cm

# Statistical outlier removal
OUTLIER_K   = 20    # neighbours to consider
OUTLIER_STD = 2.0   # std dev multiplier

# Depth clip per camera (metres) — points outside discarded
CLIP = {
    "oak":        (0.15, 3.0),
    "realsense":  (0.20, 3.5),
    "wrist":      (0.0, 3.0),
}

# Stale cloud timeout (seconds) — camera considered offline if no data received
STALE_TIMEOUT = 8.0

# ── Topic names ───────────────────────────────────────────────────────────────
TOPIC_OAK        = "/global_camera/stereo/points"
TOPIC_REALSENSE  = "/global_camera/global_camera/depth/color/points"
TOPIC_WRIST      = "/camera/depth/color/points"
TOPIC_FUSED      = "/fused/points"
TOPIC_STATUS     = "/fused/status"


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def parse_pointcloud2(msg: PointCloud2) -> np.ndarray | None:
    """
    Parse a PointCloud2 message into an Nx4 float32 array [x, y, z, rgb].
    Handles both XYZ and XYZRGB clouds.
    Returns None if the cloud is empty or malformed.
    """
    if msg.width == 0 or msg.height == 0:
        return None

    # Find field offsets
    fields = {f.name: f.offset for f in msg.fields}
    if 'x' not in fields or 'y' not in fields or 'z' not in fields:
        return None

    ox = fields['x']
    oy = fields['y']
    oz = fields['z']
    orgb = fields.get('rgb', None)

    step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8)
    n    = msg.width * msg.height

    try:
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, step)
        xs  = raw[:, ox:ox+4].copy().view(np.float32).reshape(n)
        ys  = raw[:, oy:oy+4].copy().view(np.float32).reshape(n)
        zs  = raw[:, oz:oz+4].copy().view(np.float32).reshape(n)
    except Exception as e:
        return None

    if orgb is not None:
        try:
            rgbs = raw[:, orgb:orgb+4].copy().view(np.float32).reshape(n)
        except Exception:
            rgbs = np.zeros(n, dtype=np.float32)
    else:
        rgbs = np.zeros(n, dtype=np.float32)

    pts = np.column_stack([xs, ys, zs, rgbs]).astype(np.float32)

    # Remove NaN/Inf
    valid = np.isfinite(pts[:, :3]).all(axis=1)
    pts   = pts[valid]
    return pts if len(pts) > 0 else None


def transform_points(pts_xyz: np.ndarray, tf: TransformStamped) -> np.ndarray:
    """
    Apply a TransformStamped to Nx3 XYZ points.
    Returns Nx3 transformed points.
    """
    t  = tf.transform.translation
    q  = tf.transform.rotation
    tx, ty, tz = t.x, t.y, t.z
    qx, qy, qz, qw = q.x, q.y, q.z, q.w

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)

    return (R @ pts_xyz.T).T + np.array([tx, ty, tz])


def voxel_downsample(pts: np.ndarray, leaf: float) -> np.ndarray:
    """
    Simple voxel grid downsampling.
    pts: Nx4 [x,y,z,rgb]. Returns downsampled Nx4.
    """
    if len(pts) == 0:
        return pts

    xyz   = pts[:, :3]
    mn    = xyz.min(axis=0)
    voxel = ((xyz - mn) / leaf).astype(np.int32)

    # Use dict to keep one point per voxel (the first seen)
    seen  = {}
    keep  = []
    for i, key in enumerate(map(tuple, voxel)):
        if key not in seen:
            seen[key] = True
            keep.append(i)

    return pts[keep]


def statistical_outlier_removal(pts: np.ndarray, k: int, std_mult: float) -> np.ndarray:
    """
    Remove statistical outliers from Nx4 point array.
    Fast approximate version using grid-based mean distance.
    """
    if len(pts) < k + 1:
        return pts

    xyz = pts[:, :3].astype(np.float32)

    # Compute mean distance to k nearest neighbours via random sampling
    # Full kNN is too slow for real-time; use a fast approximation:
    # partition space into 8 octants, compute within-octant mean distance
    try:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(xyz)
        dists, _ = nbrs.kneighbors(xyz)
        mean_dists = dists[:, 1:].mean(axis=1)
        mu  = mean_dists.mean()
        sig = mean_dists.std()
        return pts[mean_dists < mu + std_mult * sig]
    except ImportError:
        # sklearn not available — skip outlier removal
        return pts


def make_pointcloud2(pts: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Build a PointCloud2 message from Nx4 [x,y,z,rgb] float32 array."""
    msg = PointCloud2()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.height          = 1
    msg.width           = len(pts)
    msg.is_dense        = False
    msg.is_bigendian    = False
    msg.point_step      = 16
    msg.row_step        = 16 * len(pts)
    msg.fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = pts.astype(np.float32).tobytes()
    return msg


# ═══════════════════════════════════════════════════════════════════════════════
#  Fusion node
# ═══════════════════════════════════════════════════════════════════════════════

class SensorFusionNode(Node):

    def __init__(self):
        super().__init__("sensor_fusion_node")
        self.cb_group = ReentrantCallbackGroup()
        self._lock    = threading.Lock()

        # TF2
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Latest clouds per camera: {name: (pts Nx4, frame_id, recv_time)}
        self._clouds: dict[str, tuple | None] = {
            "oak":       None,
            "realsense": None,
            "wrist":     None,
        }

        # QoS per camera — must match publisher exactly or callbacks never fire
        # Wrist (kinova point_cloud_xyzrgb): BEST_EFFORT, depth=5
        qos_wrist = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        # RealSense (realsense2_camera): RELIABLE, depth=10
        qos_realsense = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        # OAK-D (oak_camera_node): RELIABLE, depth=10
        qos_oak = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Subscribers — each matched to its publisher QoS
        self.create_subscription(
            PointCloud2, TOPIC_OAK,
            lambda m: self._cb(m, "oak"),
            qos_oak, callback_group=self.cb_group)

        self.create_subscription(
            PointCloud2, TOPIC_REALSENSE,
            lambda m: self._cb(m, "realsense"),
            qos_realsense, callback_group=self.cb_group)

        self.create_subscription(
            PointCloud2, TOPIC_WRIST,
            lambda m: self._cb(m, "wrist"),
            qos_wrist, callback_group=self.cb_group)

        # Publishers
        self._fused_pub  = self.create_publisher(PointCloud2, TOPIC_FUSED,  10)
        self._status_pub = self.create_publisher(String,      TOPIC_STATUS, 10)

        # Publish timer
        period = 1.0 / PUBLISH_HZ
        self.create_timer(period, self._publish_fused, callback_group=self.cb_group)

        self.get_logger().info(
            f"\n{'='*55}\n"
            f"  Sensor Fusion Node  →  {TOPIC_FUSED}\n"
            f"{'='*55}\n"
            f"  SUBSCRIBES:\n"
            f"    [OAK-D]      {TOPIC_OAK}\n"
            f"    [RealSense]  {TOPIC_REALSENSE}\n"
            f"    [Wrist]      {TOPIC_WRIST}\n"
            f"  PUBLISHES:\n"
            f"    {TOPIC_FUSED}   → MoveIt2 octomap\n"
            f"    {TOPIC_STATUS}  → camera status\n"
            f"  Voxel leaf : {VOXEL_LEAF*100:.0f} cm\n"
            f"  Rate       : {PUBLISH_HZ:.0f} Hz\n"
            f"  Output TF  : {OUTPUT_FRAME}\n"
            f"{'='*55}\n"
            f"  ⚠ Add to MoveIt2 sensors config:\n"
            f"    point_cloud_topic: {TOPIC_FUSED}\n"
            f"    max_range: 2.5\n"
        )

    # ── Callback ──────────────────────────────────────────────────────────────
    def _cb(self, msg: PointCloud2, name: str):
        pts = parse_pointcloud2(msg)
        if pts is None:
            return

        # Depth clip in camera frame (z = depth)
        z_min, z_max = CLIP[name]
        pts = pts[(pts[:, 2] > z_min) & (pts[:, 2] < z_max)]
        if len(pts) == 0:
            return

        with self._lock:
            self._clouds[name] = (pts, msg.header.frame_id, time.time())

    # ── Fuse and publish ──────────────────────────────────────────────────────
    def _publish_fused(self):
        now_ros  = self.get_clock().now()
        now_wall = time.time()
        stamp    = now_ros.to_msg()

        with self._lock:
            snapshot = dict(self._clouds)

        merged    = []
        live      = []
        missing   = []

        for name, entry in snapshot.items():
            if entry is None:
                missing.append(name)
                continue

            pts, frame_id, recv_t = entry

            # Stale check
            if now_wall - recv_t > STALE_TIMEOUT:
                missing.append(f"{name}(stale)")
                continue

            # Transform to base_link via TF2
            try:
                tf = self._tf_buffer.lookup_transform(
                    OUTPUT_FRAME, frame_id,
                    rclpy.time.Time(),           # latest available
                    timeout=rclpy.duration.Duration(seconds=0.2)
                )
            except Exception as e:
                missing.append(f"{name}(no_tf)")
                self.get_logger().warn(
                    f"TF lookup failed {frame_id}→{OUTPUT_FRAME}: {e}",
                    throttle_duration_sec=5.0)
                continue

            # Apply transform to XYZ only, keep RGB
            xyz_t = transform_points(pts[:, :3].astype(np.float64), tf).astype(np.float32)
            pts_t = np.column_stack([xyz_t, pts[:, 3]])

            # After transform, clip to robot workspace
            # Keep only points in front/sides of robot (reasonable workspace)
            in_ws = (
                (pts_t[:, 0] > -1.5) & (pts_t[:, 0] < 1.5) &  # X: ±1.5m
                (pts_t[:, 1] > -1.5) & (pts_t[:, 1] < 1.5) &  # Y: ±1.5m
                (pts_t[:, 2] >  0.0) & (pts_t[:, 2] < 2.0)    # Z: 0–2m above base
            )
            pts_t = pts_t[in_ws]
            if len(pts_t) == 0:
                continue

            merged.append(pts_t)
            live.append(name)

        # Status message
        status_msg = String()
        live_str   = ", ".join(live)   if live    else "none"
        miss_str   = ", ".join(missing) if missing else "none"
        status_msg.data = f"live=[{live_str}]  missing=[{miss_str}]"
        self._status_pub.publish(status_msg)

        if not merged:
            self.get_logger().warn("No camera data available for fusion.",
                                   throttle_duration_sec=5.0)
            return

        # Concatenate all clouds
        combined = np.vstack(merged)

        # Voxel downsample
        combined = voxel_downsample(combined, VOXEL_LEAF)

        # Statistical outlier removal disabled — too slow for 10Hz
        # if len(combined) > OUTLIER_K * 2:
        #     combined = statistical_outlier_removal(combined, OUTLIER_K, OUTLIER_STD)

        # Publish
        self._fused_pub.publish(
            make_pointcloud2(combined, OUTPUT_FRAME, stamp)
        )

        self.get_logger().info(
            f"Fused: {len(combined)} pts  "
            f"live=[{live_str}]  "
            f"missing=[{miss_str}]",
            throttle_duration_sec=2.0
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    node     = SensorFusionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
