import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import json
from std_msgs.msg import String

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector")
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")
        self.get_logger().info("YOLO model loaded")
        self.sub = self.create_subscription(
            Image, "/global_camera/color/image_raw",
            self.image_callback, 10)
        self.det_pub = self.create_publisher(String, "/detections", 10)
        self.vis_pub = self.create_publisher(Image, "/detections/image", 10)

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(img, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = self.model.names[cls_id]
            conf   = float(box.conf[0])
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
            cx = (x1+x2)//2
            cy = (y1+y2)//2
            detections.append({
                "label": label, "confidence": round(conf,2),
                "bbox": [x1,y1,x2,y2], "center": [cx,cy]
            })
            self.get_logger().info(f"Detected: {label} ({conf:.2f}) at [{cx},{cy}]")
        out = String()
        out.data = json.dumps(detections)
        self.det_pub.publish(out)
        ann = results.plot()
        vis_msg = self.bridge.cv2_to_imgmsg(ann, "bgr8")
        vis_msg.header = msg.header
        self.vis_pub.publish(vis_msg)

def main():
    rclpy.init()
    node = YoloDetectorNode()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
