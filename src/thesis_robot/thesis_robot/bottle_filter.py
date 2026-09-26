import rclpy
from rclpy.node import Node
from yolo_msgs.msg import DetectionArray

class BottleFilter(Node):
    def __init__(self):
        super().__init__('bottle_filter')
        self.sub = self.create_subscription(
            DetectionArray, '/yolo/detections', self.callback, 10)
        self.pub = self.create_publisher(
            DetectionArray, '/yolo/detections/bottle', 10)

    def callback(self, msg):
        filtered = DetectionArray()
        filtered.header = msg.header
        filtered.detections = [
            d for d in msg.detections
            if d.class_name == 'bottle' and d.score > 0.75
        ]
        if filtered.detections:
            self.get_logger().info(
                f'Bottle detected: {len(filtered.detections)} | '
                f'Score: {filtered.detections[0].score:.2f} | '
                f'BBox center: ({filtered.detections[0].bbox.center.position.x:.1f}, '
                f'{filtered.detections[0].bbox.center.position.y:.1f})'
            )
        self.pub.publish(filtered)

def main():
    rclpy.init()
    rclpy.spin(BottleFilter())

if __name__ == '__main__':
    main()
