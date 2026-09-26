#!/usr/bin/env python3
"""
pointcloud_fusion.py
====================
Fuses RealSense D435I + OAK-D + wrist camera point clouds into a single
/fused_pointcloud for MoveIt2 OctoMap obstacle avoidance.

Run standalone:
  source ~/workspace/ros2_kortex_ws/install/setup.bash
  python3 ~/workspace/ros2_kortex_ws/pointcloud_fusion.py

Or via robot.launch.py with launch_fusion:=true.

Camera topics
─────────────
  realsense  /global_camera/global_camera/depth/color/points  (frame: global_camera_depth_optical_frame)
  oak_d      /global_camera/stereo/points                     (frame: global_camera_link)
  wrist      /camera/depth/color/points                       (frame: camera_depth_optical_frame)

Output
──────
  /fused_pointcloud  (frame: world, XYZ float32, ~10 Hz)
"""

import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# ── Topic / frame config ──────────────────────────────────────────────────────
CAMERAS = {
    "realsense": "/global_camera/global_camera/depth/color/points",
    "oak_d":     "/global_camera/stereo/points",
    "wrist":     "/camera/depth/color/points",
}

# ── QoS profiles ─────────────────────────────────────────────────────────────
# realsense2_camera and kinova_vision both publish sensor data as BEST_EFFORT.
# A RELIABLE subscriber is incompatible with a BEST_EFFORT publisher in ROS2,
# so all point-cloud subscriptions use BEST_EFFORT.  This is safe for oak_d
# too (oak_camera_node.py publishes RELIABLE; BEST_EFFORT subscriber still
# receives every message — it just doesn't request retransmit on loss).
_QOS_SENSOR = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,   # latest only — prevents backlog when OAK-D floods the network
)
OUTPUT_TOPIC  = "/fused_pointcloud"
OUTPUT_FRAME  = "world"

# ── Tuning ────────────────────────────────────────────────────────────────────
PUBLISH_HZ      = 10.0    # fusion publication rate
STATUS_PERIOD_S = 5.0     # how often to log camera health
# Per-camera absence timeout. Realsense gets 10 s because MIPI errors
# cause brief drop-outs while the driver auto-reconnects.
CAMERA_TIMEOUT = {
    "realsense": 10.0,
    "oak_d":      2.0,
    "wrist":      2.0,
}
SYNC_SLOP       = 0.15    # ApproximateTimeSynchronizer tolerance (s)
SYNC_QUEUE      = 5       # ATSynchronizer queue depth per camera
TF_TIMEOUT      = 0.1     # seconds — max wait for TF lookup
MIN_Z_M         = 0.05    # drop points closer than this in the source frame
MAX_Z_M         = 4.0     # drop points farther than this in the source frame


class PointCloudFusionNode(Node):
    """
    Architecture
    ────────────
    Each camera has one message_filters.Subscriber.  Every arriving message
    updates a per-camera cache (latest msg + wall timestamp).

    An ApproximateTimeSynchronizer fires when all 3 have temporally-close
    messages.  Its callback atomically overwrites all three cache entries with
    the synced messages, giving the 10 Hz timer a temporally consistent
    snapshot when all cameras are running.

    The 10 Hz timer picks whichever cameras have a message younger than
    CAMERA_TIMEOUT, transforms each cloud to world, merges, and publishes.
    If only 1 or 2 cameras are alive the timer still publishes — it never
    blocks waiting for a missing camera.
    """

    def __init__(self) -> None:
        super().__init__("pointcloud_fusion")

        # TF listener — all lookups use the latest available transform
        self._tf = Buffer()
        TransformListener(self._tf, self)

        # Publisher
        self._pub = self.create_publisher(PointCloud2, OUTPUT_TOPIC, 10)

        # Cache: name → (PointCloud2, wall_time) | None
        self._lock: threading.Lock = threading.Lock()
        self._cache: Dict[str, Optional[Tuple[PointCloud2, float]]] = {
            name: None for name in CAMERAS
        }

        # Status counters — reset each STATUS_PERIOD_S
        self._fusions_this_period: int = 0
        self._last_contributors:   List[str] = []
        # Valid (z-filtered) point counts per camera from last fusion pass
        self._valid_pts: Dict[str, int] = {name: 0 for name in CAMERAS}
        # Log wrist cloud structure once on first arrival
        self._wrist_logged: bool = False
        # Watchdog: track last successful publish and last RealSense receipt
        self._last_pub_monotonic:   float = 0.0
        self._realsense_last_rx:    float = 0.0

        # ── message_filters subscriptions ────────────────────────────────────
        # All three use BEST_EFFORT so kinova_vision and realsense2_camera
        # publishers are compatible (they advertise BEST_EFFORT).
        subs = {
            name: message_filters.Subscriber(
                self, PointCloud2, topic, qos_profile=_QOS_SENSOR
            )
            for name, topic in CAMERAS.items()
        }

        # Per-camera cache callbacks (fire on every message from that camera)
        for name, sub in subs.items():
            sub.registerCallback(lambda msg, n=name: self._store(n, msg))

        # ApproximateTimeSynchronizer: fires when all 3 have synced messages.
        # Its callback atomically refreshes all three cache entries so the
        # timer always gets a temporally-consistent set when available.
        self._sync = message_filters.ApproximateTimeSynchronizer(
            list(subs.values()),
            queue_size=SYNC_QUEUE,
            slop=SYNC_SLOP,
        )
        self._sync.registerCallback(self._on_all_synced)

        # Timers
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_cb)
        self.create_timer(STATUS_PERIOD_S,   self._status_cb)

        self.get_logger().info(
            "\nPointCloud fusion node ready"
            f"\n  Inputs:"
            + "".join(f"\n    {n:<12} {t}" for n, t in CAMERAS.items()) +
            f"\n  Output:    {OUTPUT_TOPIC}  frame={OUTPUT_FRAME}"
            f"\n  Rate:      {PUBLISH_HZ:.0f} Hz"
            f"\n  Sync slop: {SYNC_SLOP} s   Camera timeout: {CAMERA_TIMEOUT} s"
        )

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _store(self, name: str, msg: PointCloud2) -> None:
        with self._lock:
            if name == "realsense":
                self._realsense_last_rx = time.monotonic()
            if name == "wrist" and not self._wrist_logged:
                self.get_logger().info(
                    f"wrist cloud: width={msg.width} height={msg.height} "
                    f"point_step={msg.point_step} "
                    f"fields={[f.name for f in msg.fields]}"
                )
                self._wrist_logged = True
            self._cache[name] = (msg, time.monotonic())

    def _on_all_synced(self, *msgs: PointCloud2) -> None:
        """Atomically update cache when ATSynchronizer delivers all 3 together."""
        now = time.monotonic()
        with self._lock:
            for name, msg in zip(CAMERAS.keys(), msgs):
                self._cache[name] = (msg, now)

    # ── 10 Hz publication ─────────────────────────────────────────────────────

    def _publish_cb(self) -> None:
        now_wall = time.monotonic()

        # Grab a snapshot of cameras that have data within the timeout window
        with self._lock:
            snapshot: Dict[str, PointCloud2] = {
                name: entry[0]
                for name, entry in self._cache.items()
                if entry is not None and now_wall - entry[1] <= CAMERA_TIMEOUT[name]
            }

        if not snapshot:
            self._pub.publish(_make_empty_cloud(self.get_clock().now().to_msg()))
            return

        parts:        List[np.ndarray] = []
        contributors: List[str]        = []

        for name, msg in snapshot.items():
            xyz = _extract_xyz(msg)
            if xyz is None or len(xyz) == 0:
                continue

            # Track valid (z-filtered) pts per camera for status display.
            # This reflects real geometry, not the raw cloud width×height.
            self._valid_pts[name] = len(xyz)

            # Sanity bounds for the wrist cloud.  Too sparse → depth not ready.
            # Too dense (>200k) → subsampling did not apply or unexpected cloud.
            if name == "wrist":
                if len(xyz) < 50:
                    self.get_logger().warn(
                        f"wrist cloud too sparse ({len(xyz)} valid pts) — skipping",
                        throttle_duration_sec=5.0,
                    )
                    continue
                if len(xyz) > 200_000:
                    self.get_logger().warn(
                        f"wrist cloud too dense ({len(xyz)} valid pts > 200000) — skipping",
                        throttle_duration_sec=5.0,
                    )
                    continue

            try:
                tf = self._tf.lookup_transform(
                    OUTPUT_FRAME,
                    msg.header.frame_id,
                    Time(),
                    rclpy.duration.Duration(seconds=TF_TIMEOUT),
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as exc:
                self.get_logger().warn(
                    f"TF missing: '{name}' "
                    f"({msg.header.frame_id} → {OUTPUT_FRAME}): {exc}",
                    throttle_duration_sec=5.0,
                )
                continue

            parts.append(_apply_transform(xyz, tf))
            contributors.append(name)

        if not parts:
            self._pub.publish(_make_empty_cloud(self.get_clock().now().to_msg()))
            return

        merged = np.vstack(parts).astype(np.float32)
        out = _make_cloud(merged, self.get_clock().now().to_msg(), OUTPUT_FRAME)
        self._pub.publish(out)

        self._fusions_this_period += 1
        self._last_contributors = contributors
        self._last_pub_monotonic = time.monotonic()

    # ── Status report ─────────────────────────────────────────────────────────

    def _status_cb(self) -> None:
        now_wall = time.monotonic()
        with self._lock:
            active   = [n for n, e in self._cache.items()
                        if e is not None and now_wall - e[1] <= CAMERA_TIMEOUT[n]]
            inactive = [n for n in CAMERAS if n not in active]

        count = self._fusions_this_period
        self._fusions_this_period = 0
        valid_pts_snap = dict(self._valid_pts)

        # Watchdog: if RealSense has been sending but nothing got published for
        # 10 s, something is blocking the fusion pipeline.  Reset the cache to
        # drop stale messages and force fresh synchronisation.
        now_w = time.monotonic()
        if (count == 0
                and now_w - self._realsense_last_rx < STATUS_PERIOD_S
                and now_w - self._last_pub_monotonic > 10.0):
            self.get_logger().error(
                "[fusion] WATCHDOG: RealSense active but no output for >10s "
                "— resetting camera cache"
            )
            with self._lock:
                self._cache = {name: None for name in CAMERAS}

        self.get_logger().info(
            f"[fusion] published={count} in last {STATUS_PERIOD_S:.0f}s  |  "
            f"contributing: {self._last_contributors}  |  "
            f"missing/timeout: {inactive if inactive else 'none'}  |  "
            f"valid pts (z-filtered): { {n: valid_pts_snap.get(n, 0) for n in CAMERAS} }"
        )


# ── Module-level pure functions ───────────────────────────────────────────────

def _extract_xyz(msg: PointCloud2) -> Optional[np.ndarray]:
    """
    Parse a PointCloud2 message and return a valid Nx3 float32 array.

    Handles any point_step and field layout by reading the byte offsets
    from msg.fields.  Drops NaN, inf, and points outside [MIN_Z_M, MAX_Z_M]
    in the source frame's Z axis.
    """
    n = msg.width * msg.height
    if n == 0 or len(msg.data) == 0:
        return None

    # Locate x/y/z byte offsets inside each point record
    offsets: Dict[str, int] = {}
    for field in msg.fields:
        if field.name in ("x", "y", "z"):
            offsets[field.name] = field.offset

    if not {"x", "y", "z"}.issubset(offsets):
        return None

    step = msg.point_step
    raw  = np.frombuffer(msg.data, dtype=np.uint8)

    # Guard against truncated buffers
    expected = n * step
    if len(raw) < expected:
        return None

    # Reshape into (n_points, bytes_per_point)
    pts = raw[:expected].reshape(n, step)

    # Subsample if this is the depth_registration upsampled wrist cloud
    # (kinova_vision with depth_registration=True fills 480×270 depth into the
    # full 1280×720 color frame, producing ~920K pts).  Taking every 8th point
    # in each axis reduces it to ~14K before z-filtering — sufficient for
    # OctoMap while avoiding the per-frame CPU spike from a near-megapoint cloud.
    if msg.width == 1280 and msg.height == 720:
        pts = (
            pts.reshape(msg.height, msg.width, step)[::8, ::8, :]
            .reshape(-1, step)
        )
        n = len(pts)

    def read_f32(off: int) -> np.ndarray:
        # Make a contiguous (N, 4) uint8 slice and reinterpret as float32
        return (
            np.ascontiguousarray(pts[:, off: off + 4])
            .view(np.float32)
            .reshape(n)
        )

    x = read_f32(offsets["x"])
    y = read_f32(offsets["y"])
    z = read_f32(offsets["z"])

    valid = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        & (z > MIN_Z_M) & (z < MAX_Z_M)
    )
    if not np.any(valid):
        return None

    return np.column_stack([x[valid], y[valid], z[valid]])


def _apply_transform(xyz: np.ndarray, tf) -> np.ndarray:
    """
    Apply a geometry_msgs/TransformStamped to an Nx3 float32 point array.
    Returns an Nx3 float64 array in the target frame.
    """
    tr = tf.transform.translation
    ro = tf.transform.rotation
    qx, qy, qz, qw = ro.x, ro.y, ro.z, ro.w

    # Quaternion → rotation matrix (right-hand, column-major)
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [  2*(qx*qy + qz*qw),   1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
        [  2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    T = np.array([tr.x, tr.y, tr.z], dtype=np.float64)

    return (R @ xyz.astype(np.float64).T).T + T


def _make_empty_cloud(stamp) -> PointCloud2:
    """Return a zero-point PointCloud2 so MoveIt2 clears its OctoMap."""
    msg = PointCloud2()
    msg.header.stamp    = stamp
    msg.header.frame_id = OUTPUT_FRAME
    msg.height          = 1
    msg.width           = 0
    msg.is_dense        = True
    msg.is_bigendian    = False
    msg.point_step      = 12
    msg.row_step        = 0
    msg.fields          = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = b""
    return msg


def _make_cloud(xyz: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    """
    Build a PointCloud2 message from an Nx3 float32 XYZ array.
    Uses a compact 12-byte-per-point layout (XYZ only) — sufficient for
    MoveIt2 OctoMap collision avoidance.
    """
    n   = len(xyz)
    msg = PointCloud2()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.height          = 1
    msg.width           = n
    msg.is_dense        = True
    msg.is_bigendian    = False
    msg.point_step      = 12        # 3 × float32
    msg.row_step        = 12 * n
    msg.fields          = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = xyz.astype(np.float32).tobytes()
    return msg


def main() -> None:
    rclpy.init()
    node = PointCloudFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
