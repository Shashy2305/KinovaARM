#!/usr/bin/env python3
"""
robot_keepalive.py
Prevents BaseCyclicClient::Refresh timeouts when the arm is idle.

Three mechanisms:
  1. Every 10 s — log joint-state receipt to confirm DDS connection is live.
  2. Every 20 s — publish a zero-velocity hold trajectory to
     /joint_trajectory_controller/joint_trajectory, which forces the Kortex
     hardware interface to issue a network round-trip and resets the idle timer.
  3. Watch /fault_controller/internal_fault and auto-call its reset_fault
     service the instant a fault appears, instead of leaving the hardware
     interface to retry RefreshFeedback on its own - observed: that passive
     retry can itself keep timing out and loop indefinitely without ever
     clearing the fault.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from action_msgs.msg import GoalStatusArray, GoalStatus
from example_interfaces.msg import Bool
from example_interfaces.srv import Trigger

# Exactly the 7 arm joints the joint_trajectory_controller expects.
# finger_joint lives on a separate controller and must be excluded.
ARM_JOINTS = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6", "joint_7",
]


class Keepalive(Node):
    def __init__(self):
        super().__init__("robot_keepalive")

        self._joint_positions: dict = {}   # name → position (rad)
        self._last_joint_stamp = None
        self._goal_active = False  # a real MoveIt trajectory is currently executing

        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        # Publishing directly to the controller's command topic preempts
        # whatever it's currently executing - including an in-progress
        # MoveIt action goal (observed: keepalive's 20s hold fired ~0.5s
        # before a real move was reported stuck/timed out). Track the
        # action's own status so the hold is skipped while a real goal is
        # active instead of silently hijacking it.
        self.create_subscription(
            GoalStatusArray,
            "/joint_trajectory_controller/follow_joint_trajectory/_action/status",
            self._on_goal_status,
            10,
        )
        self._traj_pub = self.create_publisher(
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
            10,
        )

        self._reset_in_flight = False
        self.create_subscription(Bool, "/fault_controller/internal_fault", self._on_fault, 10)
        self._reset_client = self.create_client(Trigger, "/fault_controller/reset_fault")

        self.create_timer(10.0, self._log_status)
        self.create_timer(20.0, self._send_hold_trajectory)

        self.get_logger().info(
            "Robot keepalive active — status every 10s, hold trajectory every 20s"
        )

    def _on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._joint_positions[name] = pos
        self._last_joint_stamp = self.get_clock().now()

    def _on_goal_status(self, msg: GoalStatusArray):
        active_statuses = (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING)
        self._goal_active = any(s.status in active_statuses for s in msg.status_list)

    def _log_status(self):
        if self._last_joint_stamp is not None:
            self.get_logger().debug(
                f"keepalive: joints alive at "
                f"{self._last_joint_stamp.nanoseconds / 1e9:.1f}s"
            )
        else:
            self.get_logger().warn("keepalive: no /joint_states received yet")

    def _send_hold_trajectory(self):
        if not self._joint_positions:
            self.get_logger().debug("keepalive: waiting for joint states before sending hold")
            return
        if self._goal_active:
            self.get_logger().debug("keepalive: real trajectory in progress, skipping hold")
            return

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ARM_JOINTS

        point = JointTrajectoryPoint()
        point.positions     = [self._joint_positions.get(j, 0.0) for j in ARM_JOINTS]
        point.velocities    = [0.0] * len(ARM_JOINTS)
        point.accelerations = [0.0] * len(ARM_JOINTS)
        point.time_from_start = Duration(sec=0, nanosec=500_000_000)  # 0.5 s

        traj.points = [point]
        self._traj_pub.publish(traj)
        self.get_logger().debug("keepalive: published hold trajectory")

    def _on_fault(self, msg: Bool):
        # internal_fault publishes continuously (every controller update)
        # for as long as the fault persists, not just once on onset - the
        # in-flight guard stops that from spamming duplicate reset calls.
        if not msg.data or self._reset_in_flight:
            return
        self._reset_in_flight = True
        self.get_logger().warn("keepalive: hardware fault detected, auto-resetting")
        if not self._reset_client.wait_for_service(timeout_sec=0.0):
            self.get_logger().error("keepalive: reset_fault service not available")
            self._reset_in_flight = False
            return
        future = self._reset_client.call_async(Trigger.Request())
        future.add_done_callback(self._on_reset_done)

    def _on_reset_done(self, future):
        self._reset_in_flight = False
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f"keepalive: reset_fault call failed: {e}")
            return
        if result.success:
            self.get_logger().info("keepalive: fault reset succeeded")
        else:
            self.get_logger().error(
                "keepalive: fault reset failed - robot may need a power cycle or cable check"
            )


def main():
    rclpy.init()
    rclpy.spin(Keepalive())


if __name__ == "__main__":
    main()
