#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
import cv2
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# ── Marker parameters ─────────────────────────────────────────────────────────
ARUCO_DICT  = cv2.aruco.DICT_APRILTAG_36h11
MARKER_ID   = 1
MARKER_SIZE = 0.100  # meters (100mm)

CAMERA_FRAME = 'global_camera_link'
MARKER_FRAME = 'apriltag_frame'
IMAGE_TOPIC  = '/global_camera/color/image_raw'
INFO_TOPIC   = '/global_camera/color/camera_info'
DEBUG_TOPIC  = '/global_camera/color/image_debug'   # ← new debug image topic
# ─────────────────────────────────────────────────────────────────────────────

class AprilTagTFPublisher(Node):
    def __init__(self):
        super().__init__('charuco_tf_publisher')
        self.bridge      = CvBridge()
        self.K           = None
        self.dist        = None
        self.last_t      = None
        self.broadcaster = tf2_ros.TransformBroadcaster(self)

        self.adict  = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.params = cv2.aruco.DetectorParameters_create()

        # Thresholding
        self.params.adaptiveThreshWinSizeMin  = 3
        self.params.adaptiveThreshWinSizeMax  = 23
        self.params.adaptiveThreshWinSizeStep = 10
        self.params.adaptiveThreshConstant    = 7

        # Contour filtering
        self.params.minMarkerPerimeterRate      = 0.01
        self.params.maxMarkerPerimeterRate      = 4.0
        self.params.polygonalApproxAccuracyRate = 0.05
        self.params.minCornerDistanceRate       = 0.05
        self.params.minMarkerDistanceRate       = 0.05
        self.params.minDistanceToBorder         = 3

        # Bits extraction
        self.params.markerBorderBits                      = 1
        self.params.minOtsuStdDev                         = 5.0
        self.params.perspectiveRemovePixelPerCell         = 4
        self.params.perspectiveRemoveIgnoredMarginPerCell = 0.13

        # Marker identification
        self.params.maxErroneousBitsInBorderRate = 0.35
        self.params.errorCorrectionRate          = 0.6

        # Corner refinement
        self.params.cornerRefinementMethod        = cv2.aruco.CORNER_REFINE_SUBPIX
        self.params.cornerRefinementWinSize       = 5
        self.params.cornerRefinementMaxIterations = 30
        self.params.cornerRefinementMinAccuracy   = 0.1

        # ── Publishers ────────────────────────────────────────────────────────
        self.debug_pub = self.create_publisher(Image, DEBUG_TOPIC, 10)

        self.create_subscription(CameraInfo, INFO_TOPIC,  self._cb_info, 10)
        self.create_subscription(Image,      IMAGE_TOPIC, self._cb_img,  10)
        self.create_timer(0.05, self._timer_cb)
        self.get_logger().info(
            f'AprilTag TF publisher started — debug image on {DEBUG_TOPIC}')

    def _timer_cb(self):
        if self.last_t is not None:
            self.last_t.header.stamp = self.get_clock().now().to_msg()
            self.broadcaster.sendTransform(self.last_t)

    def _cb_info(self, msg):
        if self.K is None:
            self.K    = np.array(msg.k).reshape(3, 3)
            self.dist = np.array(msg.d)
            self.get_logger().info(
                f'Camera intrinsics received  '
                f'fx={self.K[0,0]:.1f}  fy={self.K[1,1]:.1f}  '
                f'cx={self.K[0,2]:.1f}  cy={self.K[1,2]:.1f}'
            )

    def _cb_img(self, msg):
        if self.K is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        debug = frame.copy()

        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.adict, parameters=self.params)

        if ids is not None and len(ids) > 0:
            # Draw ALL detected markers in blue
            cv2.aruco.drawDetectedMarkers(debug, corners, ids,
                                          borderColor=(255, 0, 0))

            ids_flat = ids.flatten()
            if MARKER_ID in ids_flat:
                idx = np.where(ids_flat == MARKER_ID)[0][0]
                marker_corners = corners[idx]

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    marker_corners, MARKER_SIZE, self.K, self.dist)

                rvec = rvecs[0][0]
                tvec = tvecs[0][0]

                # Draw axes on the target marker (green/red/blue axes)
                cv2.drawFrameAxes(debug, self.K, self.dist,
                                  rvec, tvec, MARKER_SIZE * 0.5)

                # Draw thick green border on target marker
                pts = marker_corners[0].astype(int)
                cv2.polylines(debug, [pts], True, (0, 255, 0), 3)

                # Distance label
                dist_cm = np.linalg.norm(tvec) * 100
                label = f'ID:{MARKER_ID}  {dist_cm:.1f}cm'
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean()) - 15
                cv2.putText(debug, label, (cx - 60, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2, cv2.LINE_AA)

                # Publish TF
                R, _ = cv2.Rodrigues(rvec)
                qx, qy, qz, qw = self._R_to_quat(R)

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
                    f'Marker {MARKER_ID} detected — dist={dist_cm:.1f}cm  '
                    f'x={tvec[0]:.3f}  y={tvec[1]:.3f}  z={tvec[2]:.3f}',
                    throttle_duration_sec=1.0)
        else:
            # No marker — show red "NOT DETECTED" text
            cv2.putText(debug, 'AprilTag NOT detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)

        # Always publish debug image
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
    rclpy.spin(AprilTagTFPublisher())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
