#!/usr/bin/env python3
"""
multi_camera_calibrate.py — triggered, marker-based extrinsic calibration
for OAK-D and RealSense, anchored by the Kinova wrist camera.

Why this exists (vs. handeye_calibration.py / handeye_calibrationintel.py):
those scripts do a classic moving-target eye-on-base calibration — a
ChArUco board on the end effector, 15+ arm poses, cv2.calibrateHandEye.
That's accurate but manual and offline, and doesn't help when a camera
gets bumped or unplugged mid-session.

This script instead uses the one camera whose extrinsic is already exact
and needs no calibration at all: the Kinova wrist camera, whose pose is
known from the robot's own kinematic chain (base_link -> ... -> wrist
camera frame, published by the robot's own URDF / kortex_bringup TF —
no code here computes it). Put a ChArUco board somewhere near the robot
base where the wrist camera, OAK-D and RealSense can all see it, and this
script:

  1. looks up base_link -> wrist_camera_frame from TF (already published),
  2. detects the board from the wrist camera to get base_link -> marker,
  3. detects the board from OAK-D and/or RealSense to get camera -> marker,
  4. composes (2) and (3) to get base_link -> camera for each of them,
  5. writes each to a YAML file that `camera_tf_broadcaster`
     (static_tf_broadcaster.py) already knows how to load.

The board can then be removed — nothing at runtime depends on it being
visible. A camera that later drops and reconnects does NOT silently
re-trust its old calibration or re-solve on its own: camera_watchdog.py
flags it and detection nodes stop publishing for it (see README) until
this script is run again with the board back in view.

Usage:
  # 1. Put the ChArUco board near the robot base.
  # 2. Jog the arm (RViz / joystick) to a pose where the WRIST camera can
  #    see the board. This script does NOT move the arm by default —
  #    that's on you, since there's no safe generic "look at my own base"
  #    pose for every possible board placement.
  python3 calibration/multi_camera_calibrate.py

  # Optional: skip a camera that isn't connected right now.
  python3 calibration/multi_camera_calibrate.py --ros-args -p calibrate_oakd:=false

  # Optional: once you've found a joint configuration that reliably shows
  # the board to the wrist camera, you can let the script drive to it:
  python3 calibration/multi_camera_calibrate.py --ros-args \
      -p move_arm:=true -p calibration_view_joints:="[0.0,-0.35,3.14,-2.27,0.0,0.96,1.57]"
"""
import os
import sys
import time

import numpy as np
import rclpy
import tf2_ros
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, os.path.dirname(__file__))
from markers.charuco_detector import detect_pose, make_board_and_detector  # noqa: E402

CAPTURE_TIMEOUT_SEC = 10.0
FLAG_DIR = os.path.expanduser('~/.ros')


class CameraCapture:
    """Latest image + intrinsics + frame_id for one camera, by topic name."""

    def __init__(self, node, name, image_topic, info_topic):
        self.name = name
        self.image_topic = image_topic
        self.info_topic = info_topic
        self.bridge = CvBridge()
        self.gray = None
        self.K = None
        self.dist = None
        self.frame_id = None
        node.create_subscription(Image, image_topic, self._on_image, 10)
        node.create_subscription(CameraInfo, info_topic, self._on_info, 10)

    def _on_image(self, msg):
        import cv2
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self.frame_id = msg.header.frame_id

    def _on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.dist = np.array(msg.d)

    @property
    def ready(self):
        return self.gray is not None and self.K is not None


def rt_to_matrix(rvec, tvec):
    import cv2
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def tf_to_matrix(transform):
    tr = transform.transform.translation
    q = transform.transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [tr.x, tr.y, tr.z]
    return T


def matrix_to_translation_quat(T):
    t = T[:3, 3]
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x,y,z,w
    return t, q


def write_calibration_yaml(path, base_frame, child_frame, t, q, note):
    out = {
        'calibration_type': 'wrist_anchored_marker',
        'parent_frame': base_frame,
        'child_frame': child_frame,
        'translation': {'x': float(t[0]), 'y': float(t[1]), 'z': float(t[2])},
        'rotation_quat': {'x': float(q[0]), 'y': float(q[1]), 'z': float(q[2]), 'w': float(q[3])},
        'note': note,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(out, f, default_flow_style=False)


def clear_recalibration_flag(camera_name):
    flag = os.path.join(FLAG_DIR, f'{camera_name}_needs_recalibration.flag')
    if os.path.exists(flag):
        os.remove(flag)


class MultiCameraCalibrate(Node):
    def __init__(self):
        super().__init__('multi_camera_calibrate')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('move_arm', False)
        self.declare_parameter('calibration_view_joints', [0.0])
        self.declare_parameter('calibrate_oakd', True)
        self.declare_parameter('calibrate_realsense', True)
        self.declare_parameter('wrist_image_topic', '/camera/color/image_raw')
        self.declare_parameter('wrist_info_topic', '/camera/color/camera_info')
        self.declare_parameter('oakd_image_topic', '/global_camera/color/image_raw')
        self.declare_parameter('oakd_info_topic', '/global_camera/color/camera_info')
        self.declare_parameter('realsense_image_topic', '/global_camera/global_camera/color/image_raw')
        self.declare_parameter('realsense_info_topic', '/global_camera/global_camera/color/camera_info')
        self.declare_parameter('oakd_calib_file', '~/.ros/oakd_calibration.yaml')
        self.declare_parameter('realsense_calib_file', '~/.ros/realsense_calibration.yaml')

        self.base_frame = self.get_parameter('base_frame').value
        self.board, self.detector = make_board_and_detector()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.wrist = CameraCapture(
            self, 'wrist',
            self.get_parameter('wrist_image_topic').value,
            self.get_parameter('wrist_info_topic').value)
        self.cams = {}
        if self.get_parameter('calibrate_oakd').value:
            self.cams['oakd'] = CameraCapture(
                self, 'oakd',
                self.get_parameter('oakd_image_topic').value,
                self.get_parameter('oakd_info_topic').value)
        if self.get_parameter('calibrate_realsense').value:
            self.cams['realsense'] = CameraCapture(
                self, 'realsense',
                self.get_parameter('realsense_image_topic').value,
                self.get_parameter('realsense_info_topic').value)

    def _spin_until(self, predicate, timeout_sec):
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate():
                return True
        return False

    def _maybe_move_arm(self):
        if not self.get_parameter('move_arm').value:
            return
        joints = list(self.get_parameter('calibration_view_joints').value)
        if len(joints) != 7:
            self.get_logger().fatal(
                'move_arm:=true but calibration_view_joints does not have '
                '7 values — refusing to move. Set it explicitly.')
            raise SystemExit(1)
        from pymoveit2 import MoveIt2
        moveit2 = MoveIt2(
            node=self,
            joint_names=[f'joint_{i}' for i in range(1, 8)],
            base_link_name=self.base_frame,
            end_effector_name='end_effector_link',
            group_name='manipulator',
        )
        moveit2.max_velocity = 0.15
        moveit2.max_acceleration = 0.10
        self.get_logger().info(f'Moving to calibration_view_joints={joints} ...')
        moveit2.move_to_configuration(joint_positions=joints)
        moveit2.wait_until_executed()

    def run(self):
        self._maybe_move_arm()

        self.get_logger().info('Waiting for wrist camera frame + ChArUco detection...')
        found_wrist = self._spin_until(
            lambda: self.wrist.ready and detect_pose(
                self.detector, self.board, self.wrist.gray, self.wrist.K, self.wrist.dist
            ) is not None,
            CAPTURE_TIMEOUT_SEC)

        if not found_wrist:
            self.get_logger().fatal(
                'Could not see the ChArUco board from the wrist camera within '
                f'{CAPTURE_TIMEOUT_SEC:.0f}s. The wrist camera is the calibration '
                'anchor — nothing can be calibrated without it. Jog the arm so the '
                'wrist camera has a clear view of the board and try again.')
            return False

        wrist_rvec, wrist_tvec = detect_pose(
            self.detector, self.board, self.wrist.gray, self.wrist.K, self.wrist.dist)
        T_wrist_marker = rt_to_matrix(wrist_rvec, wrist_tvec)

        try:
            tf_base_wrist = self.tf_buffer.lookup_transform(
                self.base_frame, self.wrist.frame_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
        except Exception as e:
            self.get_logger().fatal(
                f'No TF {self.base_frame} -> {self.wrist.frame_id} — is the robot '
                f'bringup (URDF / kortex_bringup) running? ({e})')
            return False

        T_base_wrist = tf_to_matrix(tf_base_wrist)
        T_base_marker = T_base_wrist @ T_wrist_marker
        self.get_logger().info(
            f'Anchor OK — marker pose established in {self.base_frame} '
            f'via wrist camera (frame {self.wrist.frame_id}).')

        any_calibrated = False
        for name, cam in self.cams.items():
            calib_file = os.path.expanduser(
                self.get_parameter(f'{name}_calib_file').value)

            self.get_logger().info(f'Waiting for {name} to see the board...')
            found = self._spin_until(
                lambda c=cam: c.ready and detect_pose(
                    self.detector, self.board, c.gray, c.K, c.dist) is not None,
                CAPTURE_TIMEOUT_SEC)

            if not found:
                self.get_logger().warn(
                    f'{name}: no frames or board not visible within '
                    f'{CAPTURE_TIMEOUT_SEC:.0f}s — skipping. Its existing '
                    f'calibration (if any) is left untouched, but if it was '
                    f'flagged for recalibration it stays blocked.')
                continue

            rvec, tvec = detect_pose(self.detector, self.board, cam.gray, cam.K, cam.dist)
            T_cam_marker = rt_to_matrix(rvec, tvec)
            T_base_cam = T_base_marker @ np.linalg.inv(T_cam_marker)
            t, q = matrix_to_translation_quat(T_base_cam)

            write_calibration_yaml(
                calib_file, self.base_frame, cam.frame_id, t, q,
                note=f'Computed by multi_camera_calibrate.py, anchored via wrist camera.')
            clear_recalibration_flag(name)
            any_calibrated = True

            self.get_logger().info(
                f'{name}: wrote {calib_file}  '
                f't=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]  '
                f'({self.base_frame} -> {cam.frame_id})')
            self.get_logger().info(
                f'  >>> Restart its TF broadcaster to pick this up, e.g.:\n'
                f'      ros2 run thesis_robot camera_tf_broadcaster --ros-args '
                f'-r __node:={name}_tf_broadcaster '
                f'-p calibration_file:={calib_file} '
                f'-p parent_frame:={self.base_frame} -p child_frame:={cam.frame_id}')

        if any_calibrated:
            self.get_logger().info(
                'Done. You can remove the ChArUco board now — nothing at '
                'runtime depends on it being visible.')
        return any_calibrated


def main():
    rclpy.init()
    node = MultiCameraCalibrate()
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
