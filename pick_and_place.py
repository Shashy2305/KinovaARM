#!/usr/bin/env python3
"""
pick_place_node.py
Detects the closest cup/wine-glass with YOLO + OAK-D stereo depth,
transforms it to the robot world frame, then executes a pick-and-place
with the Kinova Gen3 + Robotiq 140 via MoveIt2 (pymoveit2).

Usage:
    ros2 run <your_package> pick_place_node
  or directly:
    python3 pick_place_node.py
"""

import sys
import time
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Pose, Point, Quaternion
from control_msgs.action import GripperCommand

# pymoveit2  (pip install pymoveit2  or build from source)
from pymoveit2 import MoveIt2
from pymoveit2.robots import kinova_gen3 as robot

# OAK-D + YOLO
import depthai as dai
import sys, os
sys.path.insert(0, os.path.expanduser("~/workspace/YOLO-3D"))
from detection_model import ObjectDetector
from depth_model import DepthEstimator


# ── Constants ────────────────────────────────────────────────────────────────

# Classes to detect: 40 = wine glass, 41 = cup
DETECT_CLASSES = [40, 41]

# Camera → World transform matrix  (base_link → global_camera_link)
# Rotation from quaternion qx=-0.341494 qy=-0.888985 qz=0.299234 qw=0.059552
# Translation: x=0.48 y=0.72 z=1.0  (matches T_BASE_CAM in oak_camera_node.py)
T_WORLD_CAM = np.array([
    [-0.760,  0.572, -0.310,  0.480],
    [ 0.643,  0.588, -0.491,  0.720],
    [-0.098, -0.573, -0.814,  1.000],
    [ 0.000,  0.000,  0.000,  1.000]
], dtype=np.float64)

# Fixed place location in world frame  (adjust to your drop-off point)
PLACE_POSITION = [0.4, -0.3, 0.15]   # metres  [x, y, z]

# Grasp approach parameters
PRE_GRASP_Z_OFFSET = 0.15   # metres above object before descending
GRASP_Z_OFFSET     = 0.01   # extra clearance at grasp height
RETREAT_Z_OFFSET   = 0.20   # how high to lift after grasping

# Gripper values (Robotiq 140: 0.0 = fully open, 0.8 = fully closed)
GRIPPER_OPEN   = 0.0
GRIPPER_CLOSED = 0.6   # adjust for cup/glass size

# MoveIt2 planning parameters
PLAN_TIMEOUT  = 5.0   # seconds
MAX_VELOCITY  = 0.3   # fraction of max (0–1), keep slow for safety
MAX_ACCEL     = 0.2


# ── Helper functions ─────────────────────────────────────────────────────────

def pixel_to_camera_frame(u, v, depth_m, fx, fy, cx, cy):
    """Back-project a pixel + metric depth to a 3-D point in camera frame."""
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z, 1.0])   # homogeneous


def camera_to_world(point_cam_h):
    """Apply the camera→world transform (4×4 matrix)."""
    p_world = T_WORLD_CAM @ point_cam_h
    return p_world[:3]


def make_top_down_quaternion():
    """
    Gripper pointing straight down (tool z-axis → world -z).
    This is a 180° rotation about the world x-axis from identity.
    """
    # Quaternion for pointing down: rotate 180° about x
    return Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)


def get_oak_intrinsics():
    """Read fx, fy, cx, cy from the OAK-D calibration at 640×480."""
    with dai.Device() as dev:
        calib = dev.readCalibration()
        K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 480)
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    print(f"Camera intrinsics — fx:{fx:.1f}  fy:{fy:.1f}  cx:{cx:.1f}  cy:{cy:.1f}")
    return fx, fy, cx, cy


# ── ROS2 Node ────────────────────────────────────────────────────────────────

class PickPlaceNode(Node):

    def __init__(self):
        super().__init__("pick_place_node")
        self.get_logger().info("Initialising Pick & Place node...")

        # ── Camera intrinsics ────────────────────────────────────────────
        self.fx, self.fy, self.cx, self.cy = get_oak_intrinsics()

        # ── YOLO detector ────────────────────────────────────────────────
        self.get_logger().info("Loading YOLO detector...")
        self.detector = ObjectDetector(
            model_size="nano",
            conf_thres=0.35,
            iou_thres=0.45,
            classes=DETECT_CLASSES,
            device="cuda"
        )

        # ── OAK-D depth estimator ────────────────────────────────────────
        self.get_logger().info("Starting OAK-D pipeline...")
        self.depth_est = DepthEstimator()

        # ── MoveIt2 interface ────────────────────────────────────────────
        self.get_logger().info("Connecting to MoveIt2...")
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=robot.joint_names(),
            base_link_name=robot.base_link_name(),
            end_effector_name=robot.end_effector_name(),
            group_name=robot.MOVE_GROUP_ARM,
        )
        self.moveit2.max_velocity = MAX_VELOCITY
        self.moveit2.max_acceleration = MAX_ACCEL

        # ── Gripper action client ────────────────────────────────────────
        self._gripper_client = ActionClient(
            self,
            GripperCommand,
            "/robotiq_gripper_controller/gripper_cmd"
        )
        self.get_logger().info("Waiting for gripper action server...")
        self._gripper_client.wait_for_server(timeout_sec=10.0)

        self.get_logger().info("Ready — starting detection loop")

    # ── Gripper helpers ──────────────────────────────────────────────────────

    def _send_gripper(self, position, max_effort=50.0):
        """Send a gripper command and wait for result."""
        goal = GripperCommand.Goal()
        goal.command.position   = position
        goal.command.max_effort = max_effort
        future = self._gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)

    def open_gripper(self):
        self.get_logger().info("Opening gripper")
        self._send_gripper(GRIPPER_OPEN)

    def close_gripper(self):
        self.get_logger().info("Closing gripper")
        self._send_gripper(GRIPPER_CLOSED)

    # ── Motion helpers ───────────────────────────────────────────────────────

    def move_to_pose(self, position, quaternion, label=""):
        """Plan and execute a Cartesian pose goal."""
        pose = Pose(
            position=Point(x=position[0], y=position[1], z=position[2]),
            orientation=quaternion
        )
        self.get_logger().info(
            f"Moving to {label}: "
            f"[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]"
        )
        self.moveit2.move_to_pose(pose=pose, cartesian=False)
        self.moveit2.wait_until_executed()

    def move_to_pose_cartesian(self, position, quaternion, label=""):
        """Cartesian linear motion — safer for approach/retreat."""
        pose = Pose(
            position=Point(x=position[0], y=position[1], z=position[2]),
            orientation=quaternion
        )
        self.get_logger().info(f"Cartesian move to {label}")
        self.moveit2.move_to_pose(pose=pose, cartesian=True)
        self.moveit2.wait_until_executed()

    # ── Detection ────────────────────────────────────────────────────────────

    def get_closest_object(self):
        """
        Grab one frame, run YOLO, return (world_xyz, depth_mm, frame)
        for the closest detected cup/glass, or None if nothing found.
        """
        frame = self.depth_est.get_frame()
        depth_raw = self.depth_est.get_depth_mm()   # uint16, millimetres

        vis_frame = frame.copy()
        _, detections = self.detector.detect(vis_frame, track=False)

        if not detections:
            return None, None, frame

        best      = None
        best_depth_mm = float('inf')

        for (bbox, score, class_id, _) in detections:
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cx_px = (x1 + x2) // 2
            cy_px = (y1 + y2) // 2

            # Get raw depth at centre pixel (mm)
            if depth_raw is not None:
                h, w = depth_raw.shape
                cx_px = max(0, min(cx_px, w - 1))
                cy_px = max(0, min(cy_px, h - 1))
                d_mm = float(depth_raw[cy_px, cx_px])
            else:
                continue

            if d_mm <= 0 or d_mm > 3000:   # ignore invalid or far readings
                continue

            if d_mm < best_depth_mm:
                best_depth_mm = d_mm
                best = (bbox, score, class_id, cx_px, cy_px, d_mm)

        if best is None:
            return None, None, frame

        bbox, score, class_id, u, v, d_mm = best
        class_name = self.detector.get_class_names()[class_id]
        self.get_logger().info(
            f"Closest object: {class_name} (score={score:.2f}) "
            f"at pixel ({u},{v}), depth={d_mm:.0f} mm"
        )

        # Back-project to camera frame then world frame
        d_m = d_mm / 1000.0
        p_cam = pixel_to_camera_frame(u, v, d_m, self.fx, self.fy, self.cx, self.cy)
        p_world = camera_to_world(p_cam)
        self.get_logger().info(
            f"World position: [{p_world[0]:.3f}, {p_world[1]:.3f}, {p_world[2]:.3f}] m"
        )

        # Draw on frame for visualisation
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"{class_name} {d_mm/1000:.2f}m",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return p_world, d_mm, frame

    # ── Pick and Place sequence ──────────────────────────────────────────────

    def pick(self, obj_world_xyz):
        """Execute the pick sequence for the given world position."""
        q_down = make_top_down_quaternion()

        # 1. Open gripper
        self.open_gripper()
        time.sleep(0.5)

        # 2. Pre-grasp — above the object
        pre_grasp_pos = [
            obj_world_xyz[0],
            obj_world_xyz[1],
            obj_world_xyz[2] + PRE_GRASP_Z_OFFSET
        ]
        self.move_to_pose(pre_grasp_pos, q_down, label="pre-grasp")
        time.sleep(0.5)

        # 3. Descend to grasp height
        grasp_pos = [
            obj_world_xyz[0],
            obj_world_xyz[1],
            obj_world_xyz[2] + GRASP_Z_OFFSET
        ]
        self.move_to_pose_cartesian(grasp_pos, q_down, label="grasp")
        time.sleep(0.5)

        # 4. Close gripper
        self.close_gripper()
        time.sleep(1.0)

        # 5. Retreat upward
        retreat_pos = [
            obj_world_xyz[0],
            obj_world_xyz[1],
            obj_world_xyz[2] + RETREAT_Z_OFFSET
        ]
        self.move_to_pose_cartesian(retreat_pos, q_down, label="retreat")
        time.sleep(0.5)

    def place(self):
        """Execute the place sequence at the fixed drop-off location."""
        q_down = make_top_down_quaternion()

        # 1. Move above place location
        pre_place_pos = [
            PLACE_POSITION[0],
            PLACE_POSITION[1],
            PLACE_POSITION[2] + PRE_GRASP_Z_OFFSET
        ]
        self.move_to_pose(pre_place_pos, q_down, label="pre-place")
        time.sleep(0.5)

        # 2. Descend to place height
        self.move_to_pose_cartesian(PLACE_POSITION, q_down, label="place")
        time.sleep(0.5)

        # 3. Open gripper (release)
        self.open_gripper()
        time.sleep(0.8)

        # 4. Retreat
        retreat_pos = [
            PLACE_POSITION[0],
            PLACE_POSITION[1],
            PLACE_POSITION[2] + RETREAT_Z_OFFSET
        ]
        self.move_to_pose_cartesian(retreat_pos, q_down, label="place-retreat")

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        """
        Detection + pick-and-place loop.
        Keeps picking the closest object until no more are found.
        """
        cv2.namedWindow("Pick & Place Detection", cv2.WINDOW_NORMAL)

        while rclpy.ok():
            # Show live feed while waiting / searching
            frame = self.depth_est.get_frame()
            vis = frame.copy()
            _, detections = self.detector.detect(vis, track=False)
            cv2.putText(vis, "Press SPACE to pick closest | Q to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.imshow("Pick & Place Detection", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("Quit requested.")
                break

            if key == ord(' '):
                self.get_logger().info("SPACE pressed — detecting closest object...")
                obj_xyz, depth_mm, det_frame = self.get_closest_object()
                cv2.imshow("Pick & Place Detection", det_frame)
                cv2.waitKey(500)

                if obj_xyz is None:
                    self.get_logger().warn("No cup/glass detected. Try again.")
                    continue

                self.get_logger().info("Starting pick sequence...")
                self.pick(obj_xyz)

                self.get_logger().info("Starting place sequence...")
                self.place()

                self.get_logger().info("Pick-and-place complete!")

        cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = PickPlaceNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
