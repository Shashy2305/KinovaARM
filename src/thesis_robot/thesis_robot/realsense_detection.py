#!/usr/bin/env python3
"""
RealSense D435I object detection node.
Uses YOLOv8m + HSV color fallback.
Publishes to /detections_side with robot-frame coords via TF.
"""
import rclpy, threading, json
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2, numpy as np
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

class RealSenseDetection(Node):
    def __init__(self):
        super().__init__('realsense_detection')

        self.bridge  = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.depth   = None
        self._lock   = threading.Lock()

        # TF for camera->robot transform
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # YOLO
        self.model = None
        if YOLO_AVAILABLE:
            model_path = '/home/lab/workspace/ros2_kortex_ws/yolov8m.pt'
            try:
                self.model = YOLO(model_path)
                self.get_logger().info(f'YOLOv8m loaded')
            except Exception as e:
                self.get_logger().warn(f'YOLO failed: {e} — using color detection only')

        self.target_classes = [
            'bottle', 'cup', 'bowl', 'cell phone',
            'remote', 'book', 'scissors', 'vase', 'mouse'
        ]

        # Subscriptions — use aligned depth (perfect alignment with RGB)
        self.create_subscription(
            Image, '/global_camera/global_camera/color/image_raw', self._rgb_cb, 10)
        self.create_subscription(
            Image, '/global_camera/global_camera/aligned_depth_to_color/image_raw',
            self._depth_cb, 10)
        self.create_subscription(
            CameraInfo, '/global_camera/global_camera/color/camera_info', self._info_cb, 10)

        self.det_pub = self.create_publisher(String, '/detections_side', 10)
        self.get_logger().info('RealSense detection node ready')

    def _info_cb(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(
                f'RealSense intrinsics: fx={self.fx:.1f} '
                f'cx={self.cx:.1f} cy={self.cy:.1f}')

    def _depth_cb(self, msg):
        with self._lock:
            self.depth = self.bridge.imgmsg_to_cv2(msg, '16UC1')

    def _get_depth(self, u, v):
        with self._lock:
            if self.depth is None:
                return 0.0
            h, w = self.depth.shape
            u = max(0, min(u, w-1))
            v = max(0, min(v, h-1))
            patch = self.depth[
                max(0,v-8):min(h,v+8),
                max(0,u-8):min(w,u+8)]
            valid = patch[patch > 0]
            if len(valid) == 0:
                return 0.0
            return float(np.median(valid)) / 1000.0

    def _backproject(self, u, v, depth_m):
        if depth_m < 0.1 or self.fx is None:
            return None
        cx3 = (u - self.cx) * depth_m / self.fx
        cy3 = (v - self.cy) * depth_m / self.fy
        cz3 = depth_m
        return cx3, cy3, cz3

    def _to_robot_frame(self, cx3, cy3, cz3):
        """
        Affine transform: RealSense camera frame -> robot base frame.
        Calibrated 5-point correspondence, avg error 1.0cm.
        The fit only models x and y, so z is returned as None (unknown)
        rather than a fabricated 0.0.
        """
        px = [-0.0442,  5.2755,  3.0851, -3.4580]
        py = [ 1.0619, -3.2731, -2.6431,  3.1335]
        rx = px[0]*cx3 + px[1]*cy3 + px[2]*cz3 + px[3]
        ry = py[0]*cx3 + py[1]*cy3 + py[2]*cz3 + py[3]
        return float(rx), float(ry), None

    def _color_detect(self, bgr):
        """HSV color fallback detection."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        detections = []
        color_ranges = {
            'red_obj':    [(0,120,70),(10,255,255),(170,120,70),(180,255,255)],
            'blue_obj':   [(100,100,70),(130,255,255),None,None],
            'green_obj':  [(40,100,70),(80,255,255),None,None],
            'yellow_obj': [(20,100,70),(35,255,255),None,None],
        }
        for name, (lo1,hi1,lo2,hi2) in color_ranges.items():
            m1 = cv2.inRange(hsv, np.array(lo1), np.array(hi1))
            if lo2:
                m2 = cv2.inRange(hsv, np.array(lo2), np.array(hi2))
                mask = cv2.bitwise_or(m1, m2)
            else:
                mask = m1
            k = np.ones((7,7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) < 500:
                continue
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            u = int(M['m10']/M['m00'])
            v = int(M['m01']/M['m00'])
            detections.append((name, 0.75, u, v))
        return detections

    def _rgb_cb(self, msg):
        if self.fx is None:
            return

        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return

        raw_detections = []

        # YOLO detection
        if self.model is not None:
            try:
                results = self.model(bgr, verbose=False)
                for r in results:
                    for box in r.boxes:
                        cls_name = self.model.names[int(box.cls)]
                        if cls_name not in self.target_classes:
                            continue
                        conf = float(box.conf)
                        if conf < 0.30:
                            continue
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        u = (x1+x2)//2
                        v = (y1+y2)//2
                        raw_detections.append((cls_name, conf, u, v))
            except Exception as e:
                self.get_logger().warn(f'YOLO error: {e}')

        # Color fallback if YOLO found nothing
        if not raw_detections:
            raw_detections = self._color_detect(bgr)

        if not raw_detections:
            return

        detections = []
        display = bgr.copy()

        for label, conf, u, v in raw_detections:
            depth_m = self._get_depth(u, v)
            if depth_m < 0.1:
                continue
            coords = self._backproject(u, v, depth_m)
            if coords is None:
                continue
            cx3, cy3, cz3 = coords
            rx, ry, rz = self._to_robot_frame(cx3, cy3, cz3)

            detections.append({
                'label':      label,
                'confidence': round(conf, 3),
                'cx_3d':      round(cx3, 4),
                'cy_3d':      round(cy3, 4),
                'cz_3d':      round(cz3, 4),
                'frame_id':   msg.header.frame_id,
                'x_robot':    round(rx, 4),
                'y_robot':    round(ry, 4),
                'z_robot':    None,
                'source':     'realsense',
            })

            cv2.circle(display, (u,v), 6, (0,255,0), -1)
            cv2.putText(display,
                f"{label} {conf:.0%} z={depth_m:.2f}m",
                (u+8, v), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 2)

        if detections:
            self.det_pub.publish(String(data=json.dumps(detections)))

        cv2.imshow('RealSense Detection', display)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
