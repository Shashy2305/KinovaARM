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
no code here computes it).

It calibrates ONE camera at a time, pairing it with the wrist camera:
for each camera being calibrated, you're prompted to position the board
so BOTH the wrist camera and that one other camera can see it (jogging
the arm as needed). Only two cameras ever need to agree at once — the
wrist doesn't need OAK-D and RealSense to also be looking at the same
spot simultaneously, since it just follows the board to wherever you put
it for each one in turn. For each camera in turn, the script:

  1. looks up base_link -> wrist_camera_frame from TF (already published),
  2. detects the board from the wrist camera RIGHT NOW to get
     base_link -> marker (recomputed fresh each time, since the board
     moves between cameras),
  3. detects the board from that camera to get camera -> marker,
  4. composes (2) and (3) to get base_link -> camera,
  5. writes it to a YAML file that `camera_tf_broadcaster`
     (static_tf_broadcaster.py) already knows how to load.

The board can then be removed — nothing at runtime depends on it being
visible. A camera that later drops and reconnects does NOT silently
re-trust its old calibration or re-solve on its own: camera_watchdog.py
flags it and detection nodes stop publishing for it (see README) until
this script is run again with the board back in view.

Usage:
  # For each camera, jog the arm so the wrist camera sees the board
  # wherever you've placed/held it for that camera, then press Enter when
  # prompted. This script does NOT move the arm by default — that's on
  # you, since there's no safe generic "look at the board" pose that
  # works for every placement.
  python3 calibration/multi_camera_calibrate.py

  # Optional: skip a camera that isn't connected right now.
  python3 calibration/multi_camera_calibrate.py --ros-args -p calibrate_oakd:=false

  # Optional: calibrate_realsense2 is on by default (harmless no-op if that
  # camera isn't connected/launched) -- turn it off explicitly if you want
  # to skip it:
  python3 calibration/multi_camera_calibrate.py --ros-args -p calibrate_realsense2:=false

  # Optional: once you've found a joint configuration that reliably shows
  # the board to the wrist camera, you can let the script drive to it:
  python3 calibration/multi_camera_calibrate.py --ros-args \
      -p move_arm:=true -p calibration_view_joints:="[0.0,-0.35,3.14,-2.27,0.0,0.96,1.57]"
"""
import os
import sys
import threading
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
        # Always update (not just once) — if a camera driver is restarted
        # at a different resolution while this is running, stale
        # intrinsics would silently produce a wrong calibration.
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
        # Defaults True like calibrate_oakd/calibrate_realsense -- subscribing
        # to a topic nothing publishes yet (camera not plugged in, or
        # cameras.launch.py's launch_realsense_2:=false) is harmless, and
        # this way the dashboard's calibrator (which constructs this node
        # with no param overrides) picks realsense2 up automatically the
        # moment the camera and its driver are actually running.
        self.declare_parameter('calibrate_realsense2', True)
        self.declare_parameter('wrist_image_topic', '/camera/color/image_raw')
        self.declare_parameter('wrist_info_topic', '/camera/color/camera_info')
        self.declare_parameter('oakd_image_topic', '/global_camera/color/image_raw')
        self.declare_parameter('oakd_info_topic', '/global_camera/color/camera_info')
        self.declare_parameter('realsense_image_topic', '/global_camera/global_camera/color/image_raw')
        self.declare_parameter('realsense_info_topic', '/global_camera/global_camera/color/camera_info')
        self.declare_parameter('realsense2_image_topic', '/global_camera_2/global_camera_2/color/image_raw')
        self.declare_parameter('realsense2_info_topic', '/global_camera_2/global_camera_2/color/camera_info')
        self.declare_parameter('oakd_calib_file', '~/.ros/oakd_calibration.yaml')
        self.declare_parameter('realsense_calib_file', '~/.ros/realsense_calibration.yaml')
        self.declare_parameter('realsense2_calib_file', '~/.ros/realsense2_calibration.yaml')

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
        if self.get_parameter('calibrate_realsense2').value:
            self.cams['realsense2'] = CameraCapture(
                self, 'realsense2',
                self.get_parameter('realsense2_image_topic').value,
                self.get_parameter('realsense2_info_topic').value)

    def _spin_until(self, predicate, timeout_sec):
        # Spinning happens continuously on a background thread (started in
        # main()), including while input() blocks the main thread — so this
        # just polls, it does not spin itself. Calling rclpy.spin_once()
        # here too, on top of a concurrently-running spin thread, would
        # race on the same node. A lookup_transform(..., timeout=...) call
        # right after this returns relies on that background thread too:
        # without it, such a call blocks for its full timeout with nothing
        # processing the pending TF message and always fails.
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
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

    def _anchor_from_wrist(self):
        """Compute base_link -> marker from the wrist camera's CURRENT
        detection. Called fresh for each target camera (not once at the
        start) so the board only has to be visible to the wrist camera and
        ONE other camera at a time, not all cameras simultaneously — the
        wrist can be jogged to follow the board wherever it's placed."""
        wrist_rvec, wrist_tvec = detect_pose(
            self.detector, self.board, self.wrist.gray, self.wrist.K, self.wrist.dist)
        T_wrist_marker = rt_to_matrix(wrist_rvec, wrist_tvec)

        tf_base_wrist = self.tf_buffer.lookup_transform(
            self.base_frame, self.wrist.frame_id, rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=2.0))
        T_base_wrist = tf_to_matrix(tf_base_wrist)
        return T_base_wrist @ T_wrist_marker

    def board_visible(self, name):
        """True if the wrist camera AND camera `name` both see the board
        RIGHT NOW. Non-blocking — used for live "ready to capture?" status
        (e.g. the dashboard's calibration wizard), unlike capture_camera()
        which waits up to CAPTURE_TIMEOUT_SEC."""
        cam = self.cams.get(name)
        if cam is None:
            return False
        return (
            self.wrist.ready and cam.ready
            and detect_pose(self.detector, self.board, self.wrist.gray,
                             self.wrist.K, self.wrist.dist) is not None
            and detect_pose(self.detector, self.board, cam.gray,
                             cam.K, cam.dist) is not None
        )

    def capture_camera(self, name, timeout_sec=CAPTURE_TIMEOUT_SEC):
        """Detect the board from the wrist camera and camera `name` right
        now (or within timeout_sec), compute base_link -> camera, and write
        its calibration YAML. Returns (ok, message, result_dict_or_None).

        Non-interactive — this is the part run() below drives from a
        terminal prompt, and what a GUI would call directly from a
        "Capture" button instead, since neither needs the other's
        interaction style."""
        cam = self.cams.get(name)
        if cam is None:
            return False, f'{name} is not configured for calibration', None
        calib_file = os.path.expanduser(self.get_parameter(f'{name}_calib_file').value)

        found = self._spin_until(lambda: self.board_visible(name), timeout_sec)
        if not found:
            return False, (
                f'{name}: wrist camera and {name} did not both see the '
                f'board within {timeout_sec:.0f}s — skipping. Its existing '
                f'calibration (if any) is left untouched, but if it was '
                f'flagged for recalibration it stays blocked.'), None

        try:
            T_base_marker = self._anchor_from_wrist()
        except Exception as e:
            return False, (
                f'{name}: lost the wrist camera\'s TF/detection right at '
                f'capture time ({e}) — try again.'), None

        rvec, tvec = detect_pose(self.detector, self.board, cam.gray, cam.K, cam.dist)
        T_cam_marker = rt_to_matrix(rvec, tvec)
        T_base_cam = T_base_marker @ np.linalg.inv(T_cam_marker)
        t, q = matrix_to_translation_quat(T_base_cam)

        write_calibration_yaml(
            calib_file, self.base_frame, cam.frame_id, t, q,
            note='Computed by multi_camera_calibrate.py, anchored via wrist camera.')
        clear_recalibration_flag(name)

        result = {
            'camera': name,
            'calib_file': calib_file,
            'parent_frame': self.base_frame,
            'child_frame': cam.frame_id,
            'translation': [float(v) for v in t],
            'rotation_quat': [float(v) for v in q],
            'restart_cmd': (
                f'ros2 run thesis_robot camera_tf_broadcaster --ros-args '
                f'-r __node:={name}_tf_broadcaster '
                f'-p calibration_file:={calib_file} '
                f'-p parent_frame:={self.base_frame} -p child_frame:={cam.frame_id}'),
        }
        msg = (f'{name}: wrote {calib_file}  t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]  '
               f'({self.base_frame} -> {cam.frame_id})')
        return True, msg, result

    def run(self):
        self._maybe_move_arm()

        any_calibrated = False
        for name in self.cams:
            input(
                f'\nPosition the ChArUco board so BOTH the wrist camera and '
                f'{name} can see it (jog the arm as needed), then press '
                f'Enter — waiting up to {CAPTURE_TIMEOUT_SEC:.0f}s once you do...')

            ok, msg, result = self.capture_camera(name)
            if not ok:
                self.get_logger().warn(msg)
                continue

            any_calibrated = True
            self.get_logger().info(msg)
            self.get_logger().info(f'  >>> Restart its TF broadcaster to pick this up, e.g.:\n'
                                    f'      {result["restart_cmd"]}')

        if any_calibrated:
            self.get_logger().info(
                'Done. You can remove the ChArUco board now — nothing at '
                'runtime depends on it being visible.')
        return any_calibrated


def main():
    rclpy.init()
    node = MultiCameraCalibrate()
    # Spin continuously on a background thread for this node's whole
    # lifetime — including while input() blocks the main thread waiting
    # for you to position the board. Without this, TF/image/info messages
    # only get processed during the brief windows _spin_until polls, which
    # can be too short for tf_static's transient-local message to arrive,
    # or blocking calls like lookup_transform(..., timeout=...) just hang
    # for their full timeout with nothing servicing the pending message.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        ok = node.run()
    finally:
        # Shut down the context FIRST so the spin thread's rclpy.spin(node)
        # call returns on its own, then join it, THEN destroy the node —
        # destroying the node while the spin thread might still be inside
        # spin() racing on the same entities is what caused the "terminate
        # called without an active exception" abort on exit.
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
