#!/usr/bin/env python3
"""
oak_launch.py  —  Single definitive OAK-D launch script
=========================================================
Launches oak_camera_node.py ONLY.
No depthai_ros_driver, no MobileNet, no extra components.

Usage:
  # USB 3.0 — full RGBD + pointcloud:
  ros2 launch ~/workspace/ros2_kortex_ws/oak_launch.py

  # USB 2.0 — RGB only, no crash:
  DEPTHAI_USB2_MODE=1 ros2 launch ~/workspace/ros2_kortex_ws/oak_launch.py

Topics published:
  /global_camera/color/image_raw       ← perception_module.py subscribes here
  /global_camera/color/camera_info
  /global_camera/depth/image_raw       ← perception_module.py subscribes here
  /global_camera/depth/camera_info
  /global_camera/stereo/points         ← MoveIt2 octomap subscribes here

TF published (by thesis_robot camera_tf_broadcaster, started here):
  base_link → global_camera_link       ← hand-eye calibration result

After calibration — write the result to
  ~/.ros/handeye_calibration_corrected.yaml
(translation {x,y,z} + rotation_quat {x,y,z,w}, the format
handeye_calibration.py writes) and relaunch. No code edits needed.
Requires the thesis_robot package to be built and sourced.

Verify after launch:
  ros2 topic hz /global_camera/color/image_raw
  ros2 topic hz /global_camera/depth/image_raw
  ros2 topic hz /global_camera/stereo/points
  ros2 run tf2_ros tf2_echo base_link global_camera_link
"""

import os
import sys

# Pass USB2 mode through to the node process
if os.environ.get("DEPTHAI_USB2_MODE") == "1":
    print("[oak_launch] USB2 mode ON — depth and pointcloud disabled")
else:
    print("[oak_launch] USB3 mode — full RGBD + pointcloud")

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    # oak_camera_node.py must be in the same directory as this launch file
    node_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "oak_camera_node.py"
    )

    if not os.path.exists(node_script):
        print(f"[oak_launch] ERROR: oak_camera_node.py not found at {node_script}")
        print(f"[oak_launch] Both files must be in the same directory.")
        sys.exit(1)

    env = dict(os.environ)  # inherit DEPTHAI_USB2_MODE and ROS env

    return LaunchDescription([
        ExecuteProcess(
            cmd=["python3", node_script],
            output="screen",
            emulate_tty=True,
            env=env,
        ),
        # Single publisher of base_link → global_camera_link
        Node(
            package="thesis_robot",
            executable="camera_tf_broadcaster",
            output="screen",
        ),
    ])
