"""All ROS tests run on a private DDS domain.

Several tests build the real ArmControllerNode / LLMPlannerNode and call real pick/place code with fake hardware. Without isolation, a shell with
ROS_DOMAIN_ID=42 (scripts/ros_env.sh, scripts/preflight.py --tests) lets those nodes talk to the LIVE stack: on 2026-10-09 the flow tests
published /training_sample save/commit requests that the real wrist detector turned into four fake training samples, and a second node called
`arm_controller` appeared in the live graph. rclpy reads the domain when the context is created, so set it before any test imports rclpy.init.
"""
import os

os.environ['ROS_DOMAIN_ID'] = '97'
