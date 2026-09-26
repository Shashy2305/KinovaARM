#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from yolo_msgs.msg import DetectionArray
from geometry_msgs.msg import PoseArray, Pose

TARGET_CLASSES = ['cup', 'bottle']

class YoloToGrasp(Node):
    def __init__(self):
        super().__init__('yolo_to_grasp')
        self.pub = self.create_publisher(PoseArray, '/grasp_poses', 10)
        self.create_subscription(DetectionArray, '/yolo/detections_3d', self._cb, 10)
        self.get_logger().info('Yolo to grasp bridge started')

    def _cb(self, msg):
        pa = PoseArray()
        pa.header = msg.header
        pa.header.frame_id = 'base_link'
        for det in msg.detections:
            if det.class_name in TARGET_CLASSES:
                if det.bbox3d.center.position.z > 0:
                    p = Pose()
                    p.position = det.bbox3d.center.position
                    p.orientation.w = 1.0
                    pa.poses.append(p)
                    self.get_logger().info(
                        f'Sending {det.class_name} to grasp: '
                        f'x={p.position.x:.3f} y={p.position.y:.3f} z={p.position.z:.3f}')
        if pa.poses:
            self.pub.publish(pa)

def main():
    rclpy.init()
    rclpy.spin(YoloToGrasp())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
