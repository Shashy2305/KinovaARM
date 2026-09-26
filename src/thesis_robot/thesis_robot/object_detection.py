#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import json
import threading

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection')
        self.declare_parameter('desired_object', 'bottle')
        self.desired_object = self.get_parameter('desired_object').value
        self.target_classes = ['bottle', 'cup', 'bowl', 'cell phone', 'remote', 'book', 'scissors', 'vase', 'mouse']
        self.get_logger().info(f'Looking for: {self.desired_object}')

        model_path = '/home/lab/workspace/ros2_kortex_ws/yolov8m.pt'
        self.model = YOLO(model_path)
        self.get_logger().info(f'YOLOv8 loaded from {model_path}')

        self.fx = self.fy = self.cx = self.cy = None
        self.bridge = CvBridge()
        self.latest_depth = None
        self.latest_display_frame = None
        self.frame_lock = threading.Lock()

        self.create_subscription(CameraInfo,
            '/global_camera/color/camera_info',
            self.camera_info_cb, 10)
        self.create_subscription(Image,
            '/global_camera/depth/image_raw',
            self.depth_cb, 10)
        self.create_subscription(Image,
            '/global_camera/color/image_raw',
            self.rgb_cb, 10)

        self.bbox_pub = self.create_publisher(String, '/object_detection/bbox', 10)
        self.detections_pub = self.create_publisher(String, '/detections', 10)
        self.debug_img_pub = self.create_publisher(Image, '/object_detection/debug_image', 10)
        self.get_logger().info('Node ready. Press q in window to quit.')

    def camera_info_cb(self, msg):
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.get_logger().info(f'Intrinsics: fx={self.fx:.1f} fy={self.fy:.1f}')

    def depth_cb(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def detect_cap_fallback(self, bgr, depth_img):
        h_img, w_img = bgr.shape[:2]

        # --- Mask out right 25% of image (arm is always there) ---
        mask_x_max = int(w_img * 0.75)
        roi_bgr = bgr[:, :mask_x_max].copy()

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=80, param1=100, param2=55,
            minRadius=15, maxRadius=70)
        if circles is None:
            return None

        circles = np.uint16(np.around(circles))
        h_d, w_d = depth_img.shape
        best = None
        best_score = 0

        for c in circles[0]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])

            # Check depth - bottle cap on table should be 0.2m-1.0m
            u = max(0, min(cx, w_d-1))
            v = max(0, min(cy, h_d-1))
            d_mm = float(depth_img[v, u])
            if d_mm == 0.0:
                patch = depth_img[max(0,v-5):min(h_d,v+5), max(0,u-5):min(w_d,u+5)]
                nz = patch[patch > 0]
                d_mm = float(np.median(nz)) if len(nz) > 0 else 0.0
            d_m = d_mm / 1000.0
            if not (0.2 < d_m < 1.0):
                continue

            # Check the circle is filled (bottle cap) not hollow (arm ring)
            # Sample brightness inside the circle - should be relatively uniform
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), max(1, r-4), 255, -1)
            inner_pixels = gray[mask == 255]
            if len(inner_pixels) == 0:
                continue
            fill_std = float(np.std(inner_pixels))
            # High std = lots of variation = likely a ring/hollow object, skip it
            if fill_std > 50:
                continue

            # Score = radius (bigger = more likely a bottle cap vs small joint)
            if r > best_score:
                best_score = r
                best = (cx, cy, r)

        if best is None:
            return None
        cx, cy, r = best
        return int(cx-r), int(cy-r), int(cx+r), int(cy+r), int(cx), int(cy)

    def rgb_cb(self, msg):
        if self.fx is None or self.latest_depth is None:
            waiting = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(waiting, 'Waiting for camera...', (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            with self.frame_lock:
                self.latest_display_frame = waiting
            return

        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(bgr, verbose=False)

        best_box = None
        best_conf = 0.0
        for result in results:
            for box in result.boxes:
                cls_name = self.model.names[int(box.cls)]
                conf = float(box.conf)
                if cls_name in self.target_classes and conf > best_conf:
                    best_conf = conf
                    best_box = box

        detection_source = None
        x1 = y1 = x2 = y2 = u = v = 0

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            u = (x1 + x2) // 2
            v = (y1 + y2) // 2
            detection_source = f'YOLO {best_conf:.0%}'
        else:
            result = self.detect_cap_fallback(bgr, self.latest_depth)
            if result is not None:
                x1, y1, x2, y2, u, v = result
                best_conf = 0.0
                detection_source = 'CAP-FALLBACK'

        if detection_source is not None:
            h, w = self.latest_depth.shape
            # Scale pixel coords from RGB resolution to depth resolution
            rgb_h, rgb_w = 720, 1280  # OAK-D RGB output resolution
            du = int(u * w / rgb_w)
            dv = int(v * h / rgb_h)
            du = max(0, min(du, w-1))
            dv = max(0, min(dv, h-1))
            u = max(0, min(u, w-1))
            v = max(0, min(v, h-1))
            depth_mm = float(self.latest_depth[dv, du])
            if depth_mm == 0.0:
                patch = self.latest_depth[max(0,dv-5):min(h,dv+5), max(0,du-5):min(w,du+5)]
                nz = patch[patch > 0]
                depth_mm = float(np.median(nz)) if len(nz) > 0 else 0.0
            depth_m = depth_mm / 1000.0
            x_3d = (u - self.cx) * depth_m / self.fx
            y_3d = (v - self.cy) * depth_m / self.fy
            z_3d = depth_m

            payload = {
                'class': self.desired_object,
                'confidence': round(best_conf, 3),
                'x_min': x1, 'y_min': y1, 'x_max': x2, 'y_max': y2,
                'cx_3d': round(x_3d, 4),
                'cy_3d': round(y_3d, 4),
                'cz_3d': round(z_3d, 4),
            }
            self.bbox_pub.publish(String(data=json.dumps(payload)))

            # Also collect ALL detections for scene graph
            all_detections = []
            for r in results:
                for box in r.boxes:
                    cls_name = self.model.names[int(box.cls)]
                    if cls_name not in self.target_classes:
                        continue
                    conf_val = float(box.conf)
                    if conf_val < 0.30:
                        continue
                    bx1,by1,bx2,by2 = map(int, box.xyxy[0])
                    bu = (bx1 + bx2) // 2
                    bv = (by1 + by2) // 2
                    if self.latest_depth is not None:
                        bh, bw = self.latest_depth.shape
                        # Scale RGB coords (1280x720) to depth coords (640x400)
                        bdu = int(bu * bw / 1280)
                        bdv = int(bv * bh / 720)
                        bdu = max(0, min(bdu, bw-1))
                        bdv = max(0, min(bdv, bh-1))
                        bpatch = self.latest_depth[max(0,bdv-5):min(bh,bdv+5), max(0,bdu-5):min(bw,bdu+5)]
                        bnz = bpatch[bpatch > 0]
                        bdepth_mm = float(np.median(bnz)) if len(bnz) > 0 else 0.0
                    else:
                        bdepth_mm = 0.0
                    bdepth_m = bdepth_mm / 1000.0
                    # Use original RGB coords for back-projection with RGB intrinsics
                    bx3d = (bu - self.cx) * bdepth_m / self.fx if self.fx else 0.0
                    by3d = (bv - self.cy) * bdepth_m / self.fy if self.fy else 0.0
                    bz3d = bdepth_m
                    all_detections.append({
                        'label':      cls_name,
                        'confidence': round(conf_val, 3),
                        'cx_3d':      round(bx3d, 4),
                        'cy_3d':      round(by3d, 4),
                        'cz_3d':      round(bz3d, 4),
                    })
            if all_detections:
                self.detections_pub.publish(String(data=json.dumps(all_detections)))

            cv2.rectangle(bgr, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(bgr, (x1, y1-45), (x1+300, y1), (0,255,0), -1)
            cv2.putText(bgr, f'{self.desired_object.upper()} [{detection_source}]',
                (x1+4, y1-28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(bgr, f'X:{x_3d:.3f}m Y:{y_3d:.3f}m Z:{z_3d:.3f}m',
                (x1+4, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.circle(bgr, (u,v), 6, (0,0,255), -1)
            cv2.rectangle(bgr, (0,0), (380,35), (0,180,0), -1)
            cv2.putText(bgr, f'DETECTED [{detection_source}] z={z_3d:.2f}m',
                (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            cv2.rectangle(bgr, (0,0), (300,35), (0,0,180), -1)
            cv2.putText(bgr, f'Searching: {self.desired_object}',
                (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(bgr, 'Press Q to quit',
            (bgr.shape[1]-180, bgr.shape[0]-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        with self.frame_lock:
            self.latest_display_frame = bgr.copy()

        debug_msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_img_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Object Detection', 848, 480)
    print('\n[INFO] Window open. Press q to quit.\n')

    try:
        while rclpy.ok():
            with node.frame_lock:
                frame = node.latest_display_frame
            if frame is not None:
                cv2.imshow('Object Detection', frame)
            else:
                blank = np.zeros((480, 848, 3), dtype=np.uint8)
                cv2.putText(blank, 'Waiting...', (200,240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
                cv2.imshow('Object Detection', blank)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
