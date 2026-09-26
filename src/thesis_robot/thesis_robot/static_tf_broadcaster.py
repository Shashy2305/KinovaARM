#!/usr/bin/env python3
"""
Static TF Broadcaster — publishes camera to robot base transform.
Reads from ~/.ros/handeye_calibration_corrected.yaml
Run handeye_calibration.py to update this transform.
"""
import rclpy, yaml, os
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros

class StaticTFBroadcaster(Node):
    def __init__(self):
        super().__init__('camera_tf_broadcaster')
        self.broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        calib_path = os.path.expanduser('~/.ros/handeye_calibration_corrected.yaml')
        if not os.path.exists(calib_path):
            # Fallback to placeholder
            self.get_logger().warn(
                f'No calibration found at {calib_path}. '
                f'Run handeye_calibration.py to generate it.'
            )
            t = [0.5, 0.05, 0.75]
            q = [-0.5, -0.5, 0.5, -0.5]
        else:
            with open(calib_path) as f:
                c = yaml.safe_load(f)
            tr = c['translation']
            ro = c['rotation']
            t  = [tr['x'], tr['y'], tr['z']]
            q  = [ro['x'], ro['y'], ro['z'], ro['w']]
            self.get_logger().info(
                f'Loaded calibration: t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]'
            )

        msg = TransformStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.child_frame_id  = 'global_camera_link'
        msg.transform.translation.x = t[0]
        msg.transform.translation.y = t[1]
        msg.transform.translation.z = t[2]
        msg.transform.rotation.x = q[0]
        msg.transform.rotation.y = q[1]
        msg.transform.rotation.z = q[2]
        msg.transform.rotation.w = q[3]

        self.broadcaster.sendTransform(msg)
        self.get_logger().info(
            f'Broadcasting base_link -> global_camera_link TF'
        )

def main(args=None):
    rclpy.init(args=args)
    node = StaticTFBroadcaster()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
