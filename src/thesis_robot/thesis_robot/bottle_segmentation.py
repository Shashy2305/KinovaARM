import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from yolo_msgs.msg import DetectionArray
from cv_bridge import CvBridge
import numpy as np
import struct

# Camera intrinsics from camera_info
FX = 606.62353515625
FY = 606.5178833007812
CX = 324.76763916015625
CY = 254.594970703125

PADDING = 20  # pixels to expand bbox

class BottleSegmentation(Node):
    def __init__(self):
        super().__init__('bottle_segmentation')
        self.bridge = CvBridge()
        self.latest_detection = None

        self.create_subscription(
            DetectionArray, '/yolo/detections/bottle',
            self.detection_callback, 10)
        self.create_subscription(
            Image, '/camera/depth/image_raw',
            self.depth_callback, 10)

        self.pc_pub = self.create_publisher(
            PointCloud2, '/bottle_pointcloud', 10)

        self.get_logger().info('Bottle segmentation node started')

    def detection_callback(self, msg):
        if msg.detections:
            self.latest_detection = msg.detections[0]

    def depth_callback(self, msg):
        if self.latest_detection is None:
            return

        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        det = self.latest_detection
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        w = det.bbox.size.x
        h = det.bbox.size.y

        # Bounding box with padding
        x1 = max(0, int(cx - w/2) - PADDING)
        y1 = max(0, int(cy - h/2) - PADDING)
        x2 = min(640, int(cx + w/2) + PADDING)
        y2 = min(480, int(cy + h/2) + PADDING)

        points = []
        for v in range(y1, y2):
            for u in range(x1, x2):
                z = depth_image[v, u] * 0.001  # mm to meters
                if z <= 0.01 or z > 2.0:
                    continue
                x = (u - CX) * z / FX
                y = (v - CY) * z / FY
                points.append((x, y, z))

        if not points:
            self.get_logger().warn('No valid depth points in bottle bbox')
            return

        pc_msg = self.create_pointcloud2(msg.header, points)
        self.pc_pub.publish(pc_msg)
        self.get_logger().info(
            f'Published bottle cloud: {len(points)} points | '
            f'BBox: ({x1},{y1})->({x2},{y2})')

    def create_pointcloud2(self, header, points):
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        data = bytearray()
        for (x, y, z) in points:
            data += struct.pack('fff', x, y, z)

        pc = PointCloud2()
        pc.header = header
        pc.height = 1
        pc.width = len(points)
        pc.fields = fields
        pc.is_bigendian = False
        pc.point_step = 12
        pc.row_step = 12 * len(points)
        pc.data = bytes(data)
        pc.is_dense = True
        return pc

def main():
    rclpy.init()
    rclpy.spin(BottleSegmentation())

if __name__ == '__main__':
    main()
