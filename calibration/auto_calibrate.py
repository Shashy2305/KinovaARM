#!/usr/bin/env python3
"""
auto_calibrate.py — Moves the Kinova Gen3 through 15 diverse calibration poses.

At each pose it pauses and waits for you to click Take Sample in the MoveIt2
RViz calibration panel, then press Enter to continue.

Run (from the repo root):
  source ~/workspace/ros2_kortex_ws/install/setup.bash
  python3 calibration/auto_calibrate.py
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from pymoveit2 import MoveIt2

JOINT_NAMES  = [f"joint_{i}" for i in range(1, 8)]
GROUP_NAME   = "manipulator"
BASE_LINK    = "base_link"
EE_LINK      = "end_effector_link"
VELOCITY     = 0.20   # 20% — slow enough to be safe
ACCELERATION = 0.10

# (name, j1, j2, j3, j4, j5, j6, j7) — all radians
POSES = [
    # Translation diversity
    ("home",        0.000, -0.349,  3.141, -2.269,  0.000,  0.960,  1.571),
    ("high",        0.000, -0.524,  3.141, -1.745,  0.000,  0.524,  1.571),
    ("low",         0.000,  0.524,  3.141, -2.618,  0.000,  1.396,  1.571),
    ("extended",    0.000,  0.175,  3.141, -1.920,  0.000,  0.698,  1.571),
    ("retracted",   0.000,  0.436,  3.141, -2.793,  0.000,  1.658,  1.571),
    ("left",        0.611,  0.000,  3.141, -2.269,  0.000,  0.960,  1.571),
    ("right",      -0.611,  0.000,  3.141, -2.269,  0.000,  0.960,  1.571),
    # Rotation diversity — wrist changes only
    ("roll_left",   0.000, -0.349,  3.141, -2.269, -0.524,  0.960,  1.571),
    ("roll_right",  0.000, -0.349,  3.141, -2.269,  0.524,  0.960,  1.571),
    ("pitch_up",    0.000, -0.349,  3.141, -2.269,  0.000,  0.524,  1.571),
    ("pitch_down",  0.000, -0.349,  3.141, -2.269,  0.000,  1.396,  1.571),
    ("ee_rot_0",    0.000, -0.349,  3.141, -2.269,  0.000,  0.960,  0.000),
    ("ee_rot_180",  0.000, -0.349,  3.141, -2.269,  0.000,  0.960,  3.141),
    # Combined translation + rotation
    ("combined_1",  0.524, -0.175,  3.141, -1.920, -0.349,  0.698,  0.785),
    ("combined_2", -0.524,  0.785,  3.141, -2.618,  0.436,  1.309,  2.356),
]


class AutoCalibrate(Node):
    def __init__(self):
        super().__init__("auto_calibrate")
        cbg = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=JOINT_NAMES,
            base_link_name=BASE_LINK,
            end_effector_name=EE_LINK,
            group_name=GROUP_NAME,
            callback_group=cbg,
            use_move_group_action=True,
        )
        self.moveit2.max_velocity     = VELOCITY
        self.moveit2.max_acceleration = ACCELERATION

    def run(self):
        total = len(POSES)
        print(f"\n{'='*60}")
        print(f"  Auto-calibration: {total} poses")
        print(f"  At each pose:")
        print(f"    1. Robot stops and holds position")
        print(f"    2. Click  Take Sample  in the RViz panel")
        print(f"    3. Press  Enter  here to move to the next pose")
        print(f"{'='*60}\n")

        input("Press Enter when the MoveIt2 calibration panel is open and ready...")
        print()

        for idx, pose_def in enumerate(POSES, start=1):
            name   = pose_def[0]
            joints = list(pose_def[1:])

            print(f"[{idx:2d}/{total}]  Moving to pose: {name} ...")
            self.moveit2.move_to_configuration(joint_positions=joints)
            success = self.moveit2.wait_until_executed()

            if not success:
                print(f"  WARNING: motion to '{name}' may not have completed fully.")
                print(f"  Check RViz for collisions or joint limit violations.")

            print(f"\n{'*'*60}")
            print(f"  POSE {idx:2d}/{total}: {name}")
            print(f"  joints: {[f'{j:.3f}' for j in joints]}")
            print(f"  >>> CLICK  Take Sample  IN THE RVIZ PANEL NOW <<<")
            print(f"{'*'*60}")
            input("  Then press Enter to move to the next pose: ")
            print()

        print(f"{'='*60}")
        print(f"  All {total} poses done.")
        print(f"  >>> CLICK  Solve  IN THE RVIZ PANEL NOW <<<")
        print(f"  Target: rotation error < 0.05 rad, translation error < 0.01 m")
        print(f"{'='*60}\n")


def main():
    rclpy.init()
    node = AutoCalibrate()

    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
