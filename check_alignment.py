#!/usr/bin/env python3
"""
check_alignment.py
Alignment verification tool for hand-eye calibration diagnostics.

Prints every 5 s:
  - Joint angles from /joint_states
  - Point cloud frame_id
  - TF: base_link → global_camera_color_optical_frame  (calibration TF)
  - TF: base_link → end_effector_link                  (robot kinematics)
  - Nearest point-cloud point to end_effector_link      (alignment check)

Interpretation:
  distance < 0.1 m → WARNING: Robot arm visible in its own point cloud
                     (calibration error causes arm to occlude itself)
  distance > 0.5 m → OK: Robot arm not in its own point cloud

Run:
  source ~/workspace/ros2_kortex_ws/install/setup.bash
  python3 ~/workspace/ros2_kortex_ws/check_alignment.py
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import rclpy.duration
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

_QOS_SENSOR = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

CHECK_PERIOD_S = 5.0


class AlignmentChecker(Node):
    def __init__(self):
        super().__init__("alignment_checker")

        self._tf = Buffer()
        TransformListener(self._tf, self)

        self._joint_lock   = threading.Lock()
        self._joint_names  = []
        self._joint_pos    = []
        self._cloud_frame  = None
        self._cloud_pts    = None   # Nx3 float32 in cloud frame

        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        self.create_subscription(
            PointCloud2,
            "/global_camera/global_camera/depth/color/points",
            self._on_cloud, _QOS_SENSOR,
        )
        self.create_timer(CHECK_PERIOD_S, self._report)
        self.get_logger().info("Alignment checker started — reporting every 5 s")

    def _on_joints(self, msg: JointState):
        with self._joint_lock:
            self._joint_names = list(msg.name)
            self._joint_pos   = list(msg.position)

    def _on_cloud(self, msg: PointCloud2):
        self._cloud_frame = msg.header.frame_id
        pts = _extract_xyz(msg)
        self._cloud_pts = pts  # may be None if empty

    def _lookup_tf(self, parent: str, child: str):
        try:
            return self._tf.lookup_transform(
                parent, child, Time(),
                rclpy.duration.Duration(seconds=0.1),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            return str(e)

    def _report(self):
        print("\n" + "=" * 60)
        print(f"  Alignment check  [{time.strftime('%H:%M:%S')}]")
        print("=" * 60)

        # Joint states
        with self._joint_lock:
            if self._joint_names:
                print("\nJoint angles (deg):")
                for name, pos in zip(self._joint_names, self._joint_pos):
                    print(f"  {name:<30} {np.degrees(pos):+.2f}°")
            else:
                print("\nJoint states: not received yet")

        # Cloud frame
        print(f"\nPoint cloud frame_id: {self._cloud_frame or '(not received)'}")

        # Calibration TF
        tf_cam = self._lookup_tf("base_link", "global_camera_color_optical_frame")
        if isinstance(tf_cam, str):
            print(f"\nTF base_link → global_camera_color_optical_frame: MISSING ({tf_cam})")
        else:
            t = tf_cam.transform.translation
            r = tf_cam.transform.rotation
            print(f"\nTF base_link → global_camera_color_optical_frame:")
            print(f"  xyz=({t.x:.4f}, {t.y:.4f}, {t.z:.4f})")
            print(f"  quat=({r.x:.4f}, {r.y:.4f}, {r.z:.4f}, {r.w:.4f})")

        # End-effector TF
        tf_ee = self._lookup_tf("base_link", "end_effector_link")
        ee_xyz = None
        if isinstance(tf_ee, str):
            print(f"\nTF base_link → end_effector_link: MISSING ({tf_ee})")
        else:
            t = tf_ee.transform.translation
            ee_xyz = np.array([t.x, t.y, t.z])
            print(f"\nTF base_link → end_effector_link:")
            print(f"  xyz=({t.x:.4f}, {t.y:.4f}, {t.z:.4f})")

        # Nearest point
        if ee_xyz is not None and self._cloud_pts is not None and len(self._cloud_pts) > 0:
            # Transform cloud to world (use base_link → cloud frame TF if available)
            tf_cloud = self._lookup_tf("base_link", self._cloud_frame or "world")
            if isinstance(tf_cloud, str):
                print(f"\nDistance check: TF to cloud frame missing — skipping")
            else:
                tr = tf_cloud.transform.translation
                ro = tf_cloud.transform.rotation
                qx, qy, qz, qw = ro.x, ro.y, ro.z, ro.w
                R = np.array([
                    [1-2*(qy*qy+qz*qz),  2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
                    [2*(qx*qy+qz*qw),    1-2*(qx*qx+qz*qz),  2*(qy*qz-qx*qw)],
                    [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
                ], dtype=np.float64)
                T = np.array([tr.x, tr.y, tr.z])
                pts_world = (R @ self._cloud_pts.astype(np.float64).T).T + T

                dists = np.linalg.norm(pts_world - ee_xyz, axis=1)
                min_dist = float(np.min(dists))

                print(f"\nNearest cloud point to end_effector_link: {min_dist:.3f} m")
                if min_dist < 0.1:
                    print("  *** WARNING: Robot arm in point cloud — calibration is off ***")
                elif min_dist > 0.5:
                    print("  OK: Robot arm not in its own point cloud")
                else:
                    print("  MARGINAL: End-effector near cloud boundary (check visually in RViz)")
        else:
            print("\nDistance check: waiting for point cloud data")

        print()


def _extract_xyz(msg: PointCloud2):
    """Extract Nx3 float32 XYZ from a PointCloud2 (best-effort, returns None on error)."""
    n = msg.width * msg.height
    if n == 0 or not msg.data:
        return None
    offsets = {f.name: f.offset for f in msg.fields if f.name in ("x", "y", "z")}
    if not {"x", "y", "z"}.issubset(offsets):
        return None
    step = msg.point_step
    raw  = np.frombuffer(msg.data, dtype=np.uint8)
    if len(raw) < n * step:
        return None
    pts = raw[: n * step].reshape(n, step)

    def f32(off):
        return np.ascontiguousarray(pts[:, off:off+4]).view(np.float32).reshape(n)

    x, y, z = f32(offsets["x"]), f32(offsets["y"]), f32(offsets["z"])
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(valid):
        return None
    return np.column_stack([x[valid], y[valid], z[valid]])


def main():
    rclpy.init()
    rclpy.spin(AlignmentChecker())


if __name__ == "__main__":
    main()
