#!/usr/bin/env python3
"""
Kinova wrist camera object detection node.
Uses YOLOv8m + HSV color fallback, same pattern as realsense_detection.py.
Publishes to /detections_wrist: camera-frame coords (cx_3d/cy_3d/cz_3d) plus
the image frame_id. scene_graph_node converts them to base_link through TF
(base_link -> ... -> wrist frame_id), which comes from the robot's own
kinematics plus the camera_module hand-eye joint in the URDF (gen3_macro.xacro;
fixed on 2026-10-05, the old calibrated value was wrong).

Also publishes /wrist_pixel_detections: pixel-level boxes with a LOW confidence
floor and NO depth requirement. The arm controller uses these to centre the
gripper over an object (wrist_servo.py); the depth sensor returns nothing below
about 0.2-0.3 m, which is where that centering happens.

Unverified on hardware (flag before trusting positions from this node):
  - Topic names below match utils/three_camera_subscriber.py's documented
    kinova_vision topics, but haven't been re-confirmed here.
  - Whether /camera/depth/image_raw is pixel-aligned to /camera/color/image_raw.
    RealSense has an explicit aligned_depth_to_color topic for this;
    kinova_vision's alignment is not confirmed. If it isn't aligned, the
    backprojected z below will be wrong at object edges/occlusion boundaries.
"""
import os
import rclpy, threading, json, time
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String

from thesis_robot import wrist_servo as ws
from thesis_robot import shape_analysis as sa
from thesis_robot import training_samples as tsm
from cv_bridge import CvBridge
import cv2, numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

PIXEL_CONF_FLOOR = 0.10   # for /wrist_pixel_detections only; /detections_wrist keeps 0.30



def unrotate_cw90(points_xy, height):
    """Map points found in an image that was rotated 90 deg clockwise (cv2.ROTATE_90_CLOCKWISE) back to the original image,
    whose height is `height`. The rotation sends (x, y) to (height - 1 - y, x), so the inverse is (x', y') -> (y', height - 1 - x')."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    return np.stack([pts[:, 1], (height - 1) - pts[:, 0]], axis=1)


class WristDetection(Node):
    def __init__(self):
        super().__init__('wrist_detection')

        self.bridge  = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.depth   = None
        self._lock   = threading.Lock()

        self.model = None
        if YOLO_AVAILABLE:
            default_model_path = os.path.join(
                get_package_share_directory('thesis_robot'), 'models', 'yolov8m.pt')
            self.declare_parameter('model_path', default_model_path)
            # The segmentation model gives the same boxes AND a mask per object, from which the orientation
            # is computed (a mouse at 45 deg has a square box; the arm could not tell which way to turn the wrist).
            self.declare_parameter('seg_model_path', '/mnt/ros_workspace/models/yolov8m-seg.pt')
            model_path = self.get_parameter('model_path').value
            seg_path = self.get_parameter('seg_model_path').value
            if seg_path and os.path.exists(seg_path):
                model_path = seg_path
            try:
                self.model = YOLO(model_path)
                self.get_logger().info('YOLOv8m loaded')
            except Exception as e:
                self.get_logger().warn(f'YOLO failed: {e} — using color detection only')

        self.target_classes = [
            'bottle', 'cup', 'bowl', 'cell phone',
            'remote', 'book', 'scissors', 'vase', 'mouse'
        ]

        # Classical shape analysis of the best few detections (contour, minAreaRect, widths per angle, contact patches,
        # handle): published as det['shape'] on /wrist_pixel_detections. GrabCut refinement is too slow for the live
        # stream (tens of ms per object); it is available for offline analysis.
        self.declare_parameter('shape_analysis', True)
        self.declare_parameter('shape_grabcut', False)
        self.declare_parameter('shape_max_objects', 3)
        self.declare_parameter('shape_min_conf', 0.25)
        self.declare_parameter('rotated_fallback', True)     # look again with the image turned 90 deg when nothing confident is found
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('info_topic', '/camera/color/camera_info')

        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._rgb_cb, 10)
        self.create_subscription(
            Image, self.get_parameter('depth_topic').value, self._depth_cb, 10)
        self.create_subscription(
            CameraInfo, self.get_parameter('info_topic').value, self._info_cb, 10)

        # Self-labelled training data: the arm controller asks for the current frame to be saved before a descent
        # (/training_sample {action: save|commit|discard, id, label, u, v}); only picks that succeed are kept.
        self.declare_parameter('training_dir', tsm.DEFAULT_DIR)
        self._last_frame = None
        self.create_subscription(String, '/training_sample', self._on_training_sample, 10)
        self.det_pub = self.create_publisher(String, '/detections_wrist', 10)
        self.pix_pub = self.create_publisher(String, '/wrist_pixel_detections', 10)
        self.get_logger().info('Wrist camera detection node ready')

    def _info_cb(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(
                f'Wrist camera intrinsics: fx={self.fx:.1f} '
                f'cx={self.cx:.1f} cy={self.cy:.1f}')

    def _on_training_sample(self, msg):
        try:
            req = json.loads(msg.data)
            d = self.get_parameter('training_dir').value
            sid, action = str(req['id']), req.get('action')
            if action == 'save':
                if self._last_frame is None:
                    return
                stamp, bgr, dets, polys = self._last_frame
                tsm.save_pending(d, sid, bgr, {
                    'label': req.get('label'), 'u': req.get('u'), 'v': req.get('v'), 'stamp': stamp,
                    'dist_m': req.get('dist_m'), 'detections': dets, 'polygons': polys})
            elif action == 'commit':
                if tsm.commit(d, sid):
                    self.get_logger().info(f'training sample {sid} kept ({req.get("label")})')
            elif action == 'discard':
                tsm.discard(d, sid)
            tsm.purge_old_pending(d)
        except Exception as e:
            self.get_logger().warn(f'training sample request failed: {e}')

    def _depth_cb(self, msg):
        with self._lock:
            self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

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
            # depth may be 16UC1 (mm) or 32FC1 (m) depending on kinova_vision
            # config; treat values as millimetres unless already sub-10 range.
            med = float(np.median(valid))
            return med / 1000.0 if med > 10.0 else med

    def _backproject(self, u, v, depth_m):
        if depth_m < 0.05 or self.fx is None:
            return None
        cx3 = (u - self.cx) * depth_m / self.fx
        cy3 = (v - self.cy) * depth_m / self.fy
        cz3 = depth_m
        return cx3, cy3, cz3

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
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        raw_detections = []
        pixel_dets = []
        det_polys = []
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        shape_on = bool(self.get_parameter('shape_analysis').value)
        shape_left = int(self.get_parameter('shape_max_objects').value)
        shape_min_conf = float(self.get_parameter('shape_min_conf').value)

        if self.model is not None:
            try:
                H = bgr.shape[0]
                rotated = False
                while True:
                    src = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE) if rotated else bgr
                    results = self.model(src, verbose=False, conf=PIXEL_CONF_FLOOR)
                    for r in results:
                        for bi, box in enumerate(r.boxes):
                            cls_name = self.model.names[int(box.cls)]
                            conf = float(box.conf)
                            # /wrist_pixel_detections carries EVERY class: from straight above the detectors call a
                            # water bottle a "sports ball" or a "bowl", a mouse a "cup". The arm matches by position.
                            in_scene_classes = cls_name in self.target_classes
                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            poly = r.masks.xy[bi] if r.masks is not None else None
                            if rotated:       # the model saw the image turned 90 deg: bring the box and the mask back
                                q = unrotate_cw90([(x1, y1), (x2, y2)], H)
                                x1, y1, x2, y2 = (int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()), int(q[:, 1].max()))
                                poly = unrotate_cw90(poly, H) if poly is not None else None
                            u = (x1+x2)//2
                            v = (y1+y2)//2
                            det = {
                                'label': cls_name, 'confidence': round(conf, 3),
                                'u': u, 'v': v, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                            if rotated:
                                det['via_rotation'] = True
                            try:
                                if poly is not None:
                                    ang, elong = ws.polygon_orientation(poly)
                                    det['orient_deg'] = round(ang, 1)
                                    det['elong'] = round(elong, 2)
                            except Exception:
                                pass
                            if shape_on and shape_left > 0 and conf >= shape_min_conf and poly is not None:
                                try:
                                    t_sh = time.monotonic()
                                    an = sa.analyse(bgr, (x1, y1, x2, y2), polygon=poly,
                                                    use_grabcut=bool(self.get_parameter('shape_grabcut').value))
                                    if an is not None:
                                        det['shape'] = an
                                        shape_left -= 1
                                    if time.monotonic() - t_sh > 0.08:
                                        self.get_logger().warn(
                                            f'shape analysis of a {cls_name} took {1000 * (time.monotonic() - t_sh):.0f} ms')
                                except Exception as e:
                                    self.get_logger().warn(f'shape analysis failed: {e}')
                            pixel_dets.append(det)
                            try:
                                det_polys.append(poly.round(1).tolist() if poly is not None else None)
                            except Exception:
                                det_polys.append(None)
                            if conf < 0.30 or not in_scene_classes:
                                continue
                            raw_detections.append((cls_name, conf, u, v))
                    # The detector is orientation-sensitive: a mouse lying across the image (0 deg) was not found at all
                    # (best guess "traffic light" 0.08) but read "mouse" 0.94 with the image turned 90 deg (2026-10-09).
                    # When nothing confident was found, look once more with the image turned.
                    if rotated or raw_detections or not bool(self.get_parameter('rotated_fallback').value):
                        break
                    rotated = True
            except Exception as e:
                self.get_logger().warn(f'YOLO error: {e}')
        self._last_frame = (stamp, bgr, pixel_dets, det_polys)
        # always published (an empty list means "looked, saw nothing")
        self.pix_pub.publish(String(data=json.dumps({
            'frame_id': msg.header.frame_id, 'stamp': stamp,
            'width': bgr.shape[1], 'height': bgr.shape[0], 'detections': pixel_dets})))

        if not raw_detections:
            raw_detections = self._color_detect(bgr)

        if not raw_detections:
            return

        detections = []
        for label, conf, u, v in raw_detections:
            depth_m = self._get_depth(u, v)
            if depth_m < 0.05:
                continue
            coords = self._backproject(u, v, depth_m)
            if coords is None:
                continue
            cx3, cy3, cz3 = coords

            detections.append({
                'label':      label,
                'confidence': round(conf, 3),
                'cx_3d':      round(cx3, 4),
                'cy_3d':      round(cy3, 4),
                'cz_3d':      round(cz3, 4),
                'frame_id':   msg.header.frame_id,
                'stamp':   msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'source':     'wrist',
            })

        if detections:
            self.det_pub.publish(String(data=json.dumps(detections)))

def main(args=None):
    rclpy.init(args=args)
    node = WristDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
