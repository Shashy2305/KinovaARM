#!/usr/bin/env python3
"""Feeds the episode bag (scripts/bag_recorder.sh) with tidy, decimated topics. It only LISTENS and publishes under /rec/: it never commands the arm.

    /rec/joint_states   sensor_msgs/JointState   the arm's joints and the gripper finger, at most 100 Hz (the driver publishes ~1 kHz); position, velocity, effort
    /rec/gripper        std_msgs/Float64MultiArray  [finger_joint position, velocity, effort]  (0 = open, ~0.35 closed on a cup, ~0.69 closed on air), 100 Hz
    /rec/ee_pose        geometry_msgs/PoseStamped   base_link -> end_effector_link (the flange), 50 Hz
    /rec/pad_midpoint   geometry_msgs/PointStamped  the middle between the two finger pads in base_link (where the fingers close), 50 Hz
    /rec/meta           std_msgs/String             JSON once a second: the code tag (commit, dirty, branch, settings hash), the arm's status text
"""
import json
import time

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import Buffer, TransformListener

from thesis_robot import code_version


class Helper(Node):
    def __init__(self):
        super().__init__('episode_bag_helper')
        self.latest, self.sent_stamp, self.status = None, None, ''
        self.create_subscription(JointState, '/joint_states', lambda m: setattr(self, 'latest', m), 20)
        self.create_subscription(String, '/arm_status', lambda m: setattr(self, 'status', m.data), 5)
        self.pub_js = self.create_publisher(JointState, '/rec/joint_states', 50)
        self.pub_grip = self.create_publisher(Float64MultiArray, '/rec/gripper', 50)
        self.pub_ee = self.create_publisher(PoseStamped, '/rec/ee_pose', 20)
        self.pub_pad = self.create_publisher(PointStamped, '/rec/pad_midpoint', 20)
        self.pub_meta = self.create_publisher(String, '/rec/meta', 5)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.01, self.on_joints)
        self.create_timer(0.02, self.on_pose)
        self.create_timer(1.0, self.on_meta)

    def on_joints(self):
        m = self.latest
        if m is None:
            return
        stamp = (m.header.stamp.sec, m.header.stamp.nanosec)
        if stamp == self.sent_stamp:
            return
        self.sent_stamp = stamp
        self.pub_js.publish(m)
        try:
            i = list(m.name).index('finger_joint')
            vals = [float(m.position[i]), float(m.velocity[i]) if len(m.velocity) > i else 0.0, float(m.effort[i]) if len(m.effort) > i else 0.0]
            self.pub_grip.publish(Float64MultiArray(data=vals))
        except (ValueError, IndexError):
            pass

    def on_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('base_link', 'end_effector_link', rclpy.time.Time())
            p = PoseStamped()
            p.header = t.header
            p.pose.position.x, p.pose.position.y, p.pose.position.z = t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
            p.pose.orientation = t.transform.rotation
            self.pub_ee.publish(p)
            a = self.tf_buffer.lookup_transform('base_link', 'left_inner_finger_pad', rclpy.time.Time()).transform.translation
            b = self.tf_buffer.lookup_transform('base_link', 'right_inner_finger_pad', rclpy.time.Time()).transform.translation
            q = PointStamped()
            q.header = t.header
            q.point.x, q.point.y, q.point.z = (a.x + b.x) / 2.0, (a.y + b.y) / 2.0, (a.z + b.z) / 2.0
            self.pub_pad.publish(q)
        except Exception:
            pass

    def on_meta(self):
        self.pub_meta.publish(String(data=json.dumps({'t': round(time.time(), 3), 'code': code_version.tag(), 'arm_status': self.status})))


def main():
    rclpy.init()
    node = Helper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
