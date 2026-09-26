#!/usr/bin/env python3
"""
Pipeline:
  1. ArUco detects 4 markers (IDs 0,1,2,3) in camera frame
  2. Glass centre = average of 4 marker positions in camera frame
  3. TF: camera frame -> base_link (static transform)
  4. base_link == world (identity confirmed by tf2_echo)
  5. Small CALIB correction (measured from Kinova web interface)
  6. MoveIt2: plan only first (RViz), then execute on real robot
  7. InsertContainer action server: phases with feedback
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from aruco_interfaces.msg import ArucoMarkers
from kortex_bringup.action import InsertContainer
from pymoveit2 import MoveIt2
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import threading
import time

# ── Frames ────────────────────────────────────────────────────────
CAMERA_FRAME = "global_camera_color_optical_frame"
BASE_FRAME   = "base_link"

# ── ArUco ─────────────────────────────────────────────────────────
MARKER_IDS = [0, 1, 2, 3]

# ── Calibration ───────────────────────────────────────────────────
CALIB_DX = -0.017
CALIB_DY = -0.066
CALIB_DZ =  0.165

# ── Motion ────────────────────────────────────────────────────────
Z_ABOVE          = 0.15
MAX_VELOCITY     = 0.02
MAX_ACCELERATION = 0.02
CARTESIAN_STEP   = 0.005

# ── Home position (safe resting pose) ────────────────────────────
HOME_X = 0.30
HOME_Y = 0.00
HOME_Z = 0.40


class GlassInsertNode(Node):
    def __init__(self):
        super().__init__("glass_pick_node")
        self.cbg = ReentrantCallbackGroup()

        # ── TF ────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.marker_poses  = {}
        self.markers_ready = threading.Event()
        self.glass_base    = None   # (x, y, z) in base_link/world frame

        # ── MoveIt2 ───────────────────────────────────────────────
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=[f"joint_{i}" for i in range(1, 8)],
            base_link_name="base_link",
            end_effector_name="pen_tip",
            group_name="manipulator",
            callback_group=self.cbg,
            use_move_group_action=True,
        )
        self.moveit2.max_velocity     = MAX_VELOCITY
        self.moveit2.max_acceleration = MAX_ACCELERATION

        # ── ArUco subscriber ──────────────────────────────────────
        self.create_subscription(
            ArucoMarkers, "/aruco_markers",
            self.aruco_cb, 10, callback_group=self.cbg)

        # ── Action server ─────────────────────────────────────────
        self._action_server = ActionServer(
            self,
            InsertContainer,
            "insert_container",
            execute_callback=self.execute_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
            callback_group=self.cbg,
        )

        self.get_logger().info("InsertContainer action server ready.")
        self.get_logger().info("Waiting for ArUco markers [0,1,2,3]...")

    # ── Goal / Cancel callbacks ───────────────────────────────────
    def goal_cb(self, goal_request):
        self.get_logger().info("Goal received.")
        return GoalResponse.ACCEPT

    def cancel_cb(self, goal_handle):
        self.get_logger().info("Cancel requested.")
        return CancelResponse.ACCEPT

    # ── ArUco callback ────────────────────────────────────────────
    def aruco_cb(self, msg):
        if self.markers_ready.is_set():
            return

        for i, mid in enumerate(msg.marker_ids):
            if int(mid) in MARKER_IDS:
                self.marker_poses[int(mid)] = msg.poses[i]

        if not all(m in self.marker_poses for m in MARKER_IDS):
            return

        # Glass centre in camera frame
        cx = float(np.mean([self.marker_poses[m].position.x for m in MARKER_IDS]))
        cy = float(np.mean([self.marker_poses[m].position.y for m in MARKER_IDS]))
        cz = float(np.mean([self.marker_poses[m].position.z for m in MARKER_IDS]))
        self.get_logger().info(
            f"Glass centre CAMERA: x={cx:.4f} y={cy:.4f} z={cz:.4f}")

        # TF camera -> base_link
        ps = PoseStamped()
        ps.header.frame_id    = CAMERA_FRAME
        ps.header.stamp       = self.get_clock().now().to_msg()
        ps.pose.position.x    = cx
        ps.pose.position.y    = cy
        ps.pose.position.z    = cz
        ps.pose.orientation.w = 1.0

        try:
            ps_base = self.tf_buffer.transform(
                ps, BASE_FRAME,
                timeout=rclpy.duration.Duration(seconds=3.0))
        except Exception as e:
            self.get_logger().error(f"TF camera->base_link failed: {e}")
            return

        # Apply calibration
        wx = ps_base.pose.position.x + CALIB_DX
        wy = ps_base.pose.position.y + CALIB_DY
        wz = ps_base.pose.position.z + CALIB_DZ

        self.get_logger().info(
            f"Glass centre WORLD(calibrated): x={wx:.4f} y={wy:.4f} z={wz:.4f}")

        self.glass_base = (wx, wy, wz)
        self.markers_ready.set()

    # ── Helpers ───────────────────────────────────────────────────
    def make_pose(self, x, y, z):
        ps = PoseStamped()
        ps.header.frame_id    = BASE_FRAME
        ps.header.stamp       = self.get_clock().now().to_msg()
        ps.pose.position.x    = x
        ps.pose.position.y    = y
        ps.pose.position.z    = z
        ps.pose.orientation.x = 1.0   # 180° around X = pen_tip down
        ps.pose.orientation.y = 0.0
        ps.pose.orientation.z = 0.0
        ps.pose.orientation.w = 0.0
        return ps

    def move(self, x, y, z, plan_only=False):
        """Move or plan-only to (x, y, z). Returns True on success."""
        self.moveit2._MoveIt2__move_action_goal\
            .planning_options.plan_only = plan_only
        self.moveit2.move_to_pose(
            pose=self.make_pose(x, y, z).pose,
            frame_id=BASE_FRAME,
            cartesian=True,
            cartesian_max_step=CARTESIAN_STEP)
        return self.moveit2.wait_until_executed()

    def send_feedback(self, goal_handle, phase: str, progress: float):
        fb = InsertContainer.Feedback()
        fb.current_phase = phase
        fb.progress      = progress
        goal_handle.publish_feedback(fb)
        self.get_logger().info(f"[{phase}] progress={progress:.0%}")

    # ── Main action execute callback ──────────────────────────────
    def execute_cb(self, goal_handle):
        goal    = goal_handle.request
        dry_run = goal.dry_run
        hover   = goal.hover_above_top if goal.hover_above_top > 0.0 else Z_ABOVE

        result          = InsertContainer.Result()
        result.success  = False
        result.message  = ""

        self.get_logger().info(
            f"Executing InsertContainer | dry_run={dry_run} "
            f"skip_home={goal.skip_home_move} hover={hover:.3f}m")

        # Wait for markers
        self.get_logger().info("Waiting for ArUco markers...")
        self.markers_ready.wait()
        wx, wy, wz = self.glass_base

        # ── Override XY from goal if provided ─────────────────────
        # (non-zero goal values take priority over ArUco detection)
        if abs(goal.target_x) > 1e-6 or abs(goal.target_y) > 1e-6:
            wx = goal.target_x
            wy = goal.target_y
            self.get_logger().info(
                f"Using goal XY override: x={wx:.4f} y={wy:.4f}")

        self.get_logger().info(
            f"Target: x={wx:.4f} y={wy:.4f} z={wz:.4f} | "
            f"approach z={wz+hover:.4f}")

        # ── PHASE 0: home_move ────────────────────────────────────
        if not goal.skip_home_move:
            self.send_feedback(goal_handle, "home_move", 0.0)
            ok = self.move(HOME_X, HOME_Y, HOME_Z, plan_only=dry_run)
            if not ok:
                result.message = "home_move failed"
                goal_handle.abort()
                return result
        
        # ── PHASE 1: preflight ────────────────────────────────────
        self.send_feedback(goal_handle, "preflight", 0.15)
        self.get_logger().info(
            f"Preflight check: target x={wx:.4f} y={wy:.4f} z={wz:.4f}")
        time.sleep(0.5)   # brief pause for any last-moment checks

        # ── PHASE 2: approach (hover above container) ─────────────
        self.send_feedback(goal_handle, "approach", 0.30)
        ok = self.move(wx, wy, wz + hover, plan_only=dry_run)
        if not ok:
            result.message = "approach move failed"
            goal_handle.abort()
            return result

        # ── PHASE 3: descend (into container) ────────────────────
        self.send_feedback(goal_handle, "descend", 0.55)
        ok = self.move(wx, wy, wz, plan_only=dry_run)
        if not ok:
            result.message = "descend move failed"
            goal_handle.abort()
            return result

        # ── PHASE 4: hold (brief pause at insertion point) ────────
        self.send_feedback(goal_handle, "hold", 0.70)
        time.sleep(1.0)

        # ── PHASE 5: ascend (back up to hover height) ─────────────
        self.send_feedback(goal_handle, "ascend", 0.80)
        ok = self.move(wx, wy, wz + hover, plan_only=dry_run)
        if not ok:
            result.message = "ascend move failed"
            goal_handle.abort()
            return result

        # ── PHASE 6: return (back to home) ────────────────────────
        self.send_feedback(goal_handle, "return", 0.90)
        ok = self.move(HOME_X, HOME_Y, HOME_Z, plan_only=dry_run)
        if not ok:
            result.message = "return to home failed"
            goal_handle.abort()
            return result

        # ── PHASE 7: complete ─────────────────────────────────────
        self.send_feedback(goal_handle, "complete", 1.0)
        result.success = True
        result.message = (
            f"{'DRY RUN ' if dry_run else ''}InsertContainer SUCCESS "
            f"x={wx:.4f} y={wy:.4f} z={wz:.4f}")
        goal_handle.succeed()
        self.get_logger().info(result.message)
        return result


# ── Entry point ───────────────────────────────────────────────────
def main():
    rclpy.init()
    node = GlassInsertNode()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down.")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
