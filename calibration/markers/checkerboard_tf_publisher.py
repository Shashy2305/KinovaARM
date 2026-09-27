#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
import cv2
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# ── Checkerboard parameters ───────────────────────────────────────────────────
ROWS        = 7          # internal corners rows
COLS        = 10         # internal corners columns
SQUARE_SIZE = 0.015      # 15mm in meters

CAMERA_FRAME = 'global_camera_link'
MARKER_FRAME = 'checkerboard_frame'
IMAGE_TOPIC  = '/global_camera/color/image_raw'
INFO_TOPIC   = '/global_camera/color/camera_info'
DEBUG_TOPIC  = '/global_camera/color/image_debug'
# ─────────────────────────────────────────────────────────────────────────────

class CheckerboardTFPublisher(Node):
    def __init__(self):
        super().__init__('checkerboard_tf_publisher')
        self.bridge      = CvBridge()
        self.K           = None
        self.dist        = None
        self.last_t      = None
        self.broadcaster = tf2_ros.TransformBroadcaster(self)

        # 3D object points for the checkerboard
        self.objp = np.zeros((ROWS * COLS, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)
        self.objp *= SQUARE_SIZE

        self.debug_pub = self.create_publisher(Image, DEBUG_TOPIC, 10)
        self.create_subscription(CameraInfo, INFO_TOPIC,  self._cb_info, 10)
        self.create_subscription(Image,      IMAGE_TOPIC, self._cb_img,  10)
        self.create_timer(0.05, self._timer_cb)
        self.get_logger().info(
            f'Checkerboard TF publisher started — {COLS}x{ROWS} internal corners, '
            f'{SQUARE_SIZE*1000:.0f}mm squares — debug on {DEBUG_TOPIC}')

    def _timer_cb(self):
        if self.last_t is not None:
            self.last_t.header.stamp = self.get_clock().now().to_msg()
            self.broadcaster.sendTransform(self.last_t)

    def _cb_info(self, msg):
        if self.K is None:
            self.K    = np.array(msg.k).reshape(3, 3)
            self.dist = np.array(msg.d)
            self.get_logger().info(
                f'Intrinsics: fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f} '
                f'cx={self.K[0,2]:.1f} cy={self.K[1,2]:.1f}')

    def _cb_img(self, msg):
        if self.K is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        debug = frame.copy()

        ret, corners = cv2.findChessboardCorners(
            gray, (COLS, ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FAST_CHECK)

        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

            # Draw corners
            cv2.drawChessboardCorners(debug, (COLS, ROWS), corners2, ret)

            # Pose estimation
            ret2, rvec, tvec = cv2.solvePnP(
                self.objp, corners2, self.K, self.dist)

            if ret2:
                # Draw axes
                cv2.drawFrameAxes(debug, self.K, self.dist,
                                  rvec, tvec, SQUARE_SIZE * 3)

                tvec = tvec.flatten()
                R, _ = cv2.Rodrigues(rvec)
                qx, qy, qz, qw = self._R_to_quat(R)

                dist_cm = np.linalg.norm(tvec) * 100
                cv2.putText(debug, f'Detected! {dist_cm:.1f}cm',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)

                # Publish TF
                t = TransformStamped()
                t.header.stamp    = self.get_clock().now().to_msg()
                t.header.frame_id = CAMERA_FRAME
                t.child_frame_id  = MARKER_FRAME
                t.transform.translation.x = float(tvec[0])
                t.transform.translation.y = float(tvec[1])
                t.transform.translation.z = float(tvec[2])
                t.transform.rotation.x = qx
                t.transform.rotation.y = qy
                t.transform.rotation.z = qz
                t.transform.rotation.w = qw
                self.last_t = t

                self.get_logger().info(
                    f'Checkerboard detected — dist={dist_cm:.1f}cm  '
                    f'x={tvec[0]:.3f}  y={tvec[1]:.3f}  z={tvec[2]:.3f}',
                    throttle_duration_sec=1.0)
        else:
            cv2.putText(debug, 'Checkerboard NOT detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)

        debug_msg = self.bridge.cv2_to_imgmsg(debug, 'bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

    def _R_to_quat(self, R):
        tr = R[0,0] + R[1,1] + R[2,2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            return (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s, 0.25/s
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            return 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            return (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            return (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s


def main():
    rclpy.init()
    rclpy.spin(CheckerboardTFPublisher())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
