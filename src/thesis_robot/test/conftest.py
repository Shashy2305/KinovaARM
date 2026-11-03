"""All ROS tests run on a private DDS domain.

Several tests build the real ArmControllerNode / LLMPlannerNode and call real pick/place code with fake hardware. Without isolation, a shell with
ROS_DOMAIN_ID=42 (scripts/ros_env.sh, scripts/preflight.py --tests) lets those nodes talk to the LIVE stack: on 2026-10-09 the flow tests
published /training_sample save/commit requests that the real wrist detector turned into four fake training samples, and a second node called
`arm_controller` appeared in the live graph. rclpy reads the domain when the context is created, so set it before any test imports rclpy.init.
"""
import os

os.environ['ROS_DOMAIN_ID'] = '97'


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _private_coffee_learn_file(tmp_path, monkeypatch):
    """The coffee routine's learned offsets live in ~/.ros/coffee_learn.yaml: no test may read or change the real one."""
    from thesis_robot import coffee_learn
    monkeypatch.setattr(coffee_learn, 'LEARN_FILE', str(tmp_path / 'coffee_learn.yaml'))
