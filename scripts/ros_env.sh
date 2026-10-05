#!/bin/bash
# Load the ROS + project + Python environment for THIS shell:
#
#     source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
#
# The "not found: .../local_setup.bash" lines it may print (handeye_target_detection,
# dai_ros_plugins) are harmless leftovers from removed packages.
source /opt/ros/humble/setup.bash
source /mnt/ros_workspace/ros2_kortex_ws/install/setup.bash
source /mnt/ros_workspace/Shashproject/install/setup.bash
source /mnt/ros_workspace/venv/bin/activate
export ROS_DOMAIN_ID=42
export KINOVA_ROBOT_IP=192.168.1.10
