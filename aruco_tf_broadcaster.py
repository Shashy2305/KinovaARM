import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from aruco_interfaces.msg import ArucoMarkers
import tf2_ros

class ArucoTFBroadcaster(Node):
    def __init__(self):
        super().__init__('aruco_tf_broadcaster')
        self.br = tf2_ros.TransformBroadcaster(self)
        self.sub = self.create_subscription(
            ArucoMarkers, '/aruco/markers', self.callback, 10)
        self.get_logger().info('ArUco TF broadcaster started')

    def callback(self, msg):
        for i, pose in enumerate(msg.poses):
            t = TransformStamped()
            t.header.stamp = msg.header.stamp
            t.header.frame_id = 'global_camera_link'
            t.child_frame_id = f'aruco_marker_{msg.marker_ids[i]}'
            t.transform.translation.x = pose.position.x
            t.transform.translation.y = pose.position.y
            t.transform.translation.z = pose.position.z
            t.transform.rotation = pose.orientation
            self.br.sendTransform(t)

def main():
    rclpy.init()
    rclpy.spin(ArucoTFBroadcaster())

if __name__ == '__main__':
    main()
