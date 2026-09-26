#!/usr/bin/env python3
"""
Moves robot to predefined diverse poses for hand-eye calibration.
Press ENTER to move to next pose.
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math

# 30 diverse poses: [j1, j2, j3, j4, j5, j6, j7] in radians
# Varied wrist (j5,j6,j7) across all poses
POSES = [
    [0.0,  0.3, -1.0, -2.0,  0.0,  0.0,  0.0],
    [0.3,  0.3, -1.0, -2.0,  0.0,  0.0,  0.0],
    [-0.3, 0.3, -1.0, -2.0,  0.0,  0.0,  0.0],
    [0.0,  0.5, -1.2, -2.0,  0.0,  0.0,  0.0],
    [0.0,  0.1, -0.8, -2.0,  0.0,  0.0,  0.0],
    # Round B - j6 = +1.0 rad (~57deg)
    [0.0,  0.3, -1.0, -2.0,  0.0,  1.0,  0.0],
    [0.3,  0.3, -1.0, -2.0,  0.0,  1.0,  0.0],
    [-0.3, 0.3, -1.0, -2.0,  0.0,  1.0,  0.0],
    [0.0,  0.5, -1.2, -2.0,  0.0,  1.0,  0.0],
    [0.0,  0.1, -0.8, -2.0,  0.0,  1.0,  0.0],
    # Round C - j6 = -1.0 rad
    [0.0,  0.3, -1.0, -2.0,  0.0, -1.0,  0.0],
    [0.3,  0.3, -1.0, -2.0,  0.0, -1.0,  0.0],
    [-0.3, 0.3, -1.0, -2.0,  0.0, -1.0,  0.0],
    [0.0,  0.5, -1.2, -2.0,  0.0, -1.0,  0.0],
    [0.0,  0.1, -0.8, -2.0,  0.0, -1.0,  0.0],
    # Round D - j7 = +1.0 rad
    [0.0,  0.3, -1.0, -2.0,  0.0,  0.0,  1.0],
    [0.3,  0.3, -1.0, -2.0,  0.0,  0.0,  1.0],
    [-0.3, 0.3, -1.0, -2.0,  0.0,  0.0,  1.0],
    [0.0,  0.5, -1.2, -2.0,  0.0,  0.0,  1.0],
    [0.0,  0.1, -0.8, -2.0,  0.0,  0.0,  1.0],
    # Round E - j7 = -1.0 rad
    [0.0,  0.3, -1.0, -2.0,  0.0,  0.0, -1.0],
    [0.3,  0.3, -1.0, -2.0,  0.0,  0.0, -1.0],
    [-0.3, 0.3, -1.0, -2.0,  0.0,  0.0, -1.0],
    [0.0,  0.5, -1.2, -2.0,  0.0,  0.0, -1.0],
    [0.0,  0.1, -0.8, -2.0,  0.0,  0.0, -1.0],
    # Round F - j6=+0.8, j7=+0.8
    [0.0,  0.3, -1.0, -2.0,  0.0,  0.8,  0.8],
    [0.3,  0.3, -1.0, -2.0,  0.0,  0.8,  0.8],
    [-0.3, 0.3, -1.0, -2.0,  0.0,  0.8,  0.8],
    [0.0,  0.5, -1.2, -2.0,  0.0,  0.8,  0.8],
    [0.0,  0.1, -0.8, -2.0,  0.0,  0.8,  0.8],
]

JOINT_NAMES = ['joint_1','joint_2','joint_3','joint_4',
               'joint_5','joint_6','joint_7']

class PoseRunner(Node):
    def __init__(self):
        super().__init__('pose_runner')
        self._client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')
        self.get_logger().info('Waiting for action server...')
        self._client.wait_for_server()
        self.get_logger().info('Ready!')

    def go_to(self, joints):
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = joints
        pt.time_from_start = Duration(sec=4)
        traj.points = [pt]
        goal.trajectory = traj
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

def main():
    rclpy.init()
    node = PoseRunner()
    for i, pose in enumerate(POSES):
        print(f'\n--- Pose {i+1}/{len(POSES)} ---')
        print(f'j6={math.degrees(pose[5]):.0f}°  j7={math.degrees(pose[6]):.0f}°')
        input('Press ENTER to move robot, then SPACE in calibration GUI to capture...')
        node.go_to(pose)
        print('Robot moved. Now press SPACE in calibration GUI!')
    rclpy.shutdown()

if __name__ == '__main__':
    main()
