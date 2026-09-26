#!/usr/bin/env python3
"""
Two-phase InsertContainer client:
  Phase 1 → dry_run=True  (RViz simulation only, arm does NOT move)
  Phase 2 → dry_run=False (executes on real arm after confirmation)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from kortex_bringup.action import InsertContainer


class InsertContainerClient(Node):
    def __init__(self):
        super().__init__("insert_container_client")
        self._client = ActionClient(
            self, InsertContainer, "insert_container")

    def send_goal(self, dry_run: bool, skip_home: bool = False):
        self.get_logger().info(
            f"Waiting for action server...")
        self._client.wait_for_server()

        goal = InsertContainer.Goal()
        goal.target_x        = 0.0      # 0.0 = use ArUco detection
        goal.target_y        = 0.0
        goal.hover_above_top = 0.15
        goal.dry_run         = dry_run
        goal.skip_home_move  = skip_home

        self.get_logger().info(
            f"Sending goal | dry_run={dry_run} skip_home={skip_home}")

        send_goal_future = self._client.send_goal_async(
            goal,
            feedback_callback=self.feedback_cb)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected!")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self.get_logger().info(
            f"Result: success={result.success} | {result.message}")
        return result.success

    def feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        bar = int(fb.progress * 20) * "█"
        self.get_logger().info(
            f"  [{bar:<20}] {fb.progress:.0%}  phase={fb.current_phase}")


def main():
    rclpy.init()
    client = InsertContainerClient()

    # ── PHASE 1: Simulation (RViz only) ──────────────────────────
    print("\n" + "="*55)
    print("  PHASE 1: SIMULATION (RViz only — arm will NOT move)")
    print("="*55)

    ok = client.send_goal(dry_run=True)
    if not ok:
        print("Simulation failed. Aborting.")
        rclpy.shutdown()
        return

    # ── Confirmation ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Simulation complete. Check RViz trajectory.")
    print("  Press ENTER to execute on real arm")
    print("  Press Ctrl+C to abort")
    print("="*55)
    try:
        input(">>> ")
    except KeyboardInterrupt:
        print("\nAborted by user.")
        rclpy.shutdown()
        return

    # ── PHASE 2: Real arm execution ───────────────────────────────
    print("\n" + "="*55)
    print("  PHASE 2: EXECUTING ON REAL ARM")
    print("="*55)

    # Skip home move on real run since arm is already near target
    ok = client.send_goal(dry_run=False, skip_home=True)
    if ok:
        print("\n  SUCCESS: Arm completed InsertContainer!")
    else:
        print("\n  FAILED: Check logs above.")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
