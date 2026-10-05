"""Regression: ArmCalib.click() once deadlocked against itself (it held the node lock while
calling arm_joint_vector(), which takes the same lock), which froze every later request."""
import os
import sys
import threading

import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('cv2')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
os.environ.setdefault('SHASHPROJECT_REPO_ROOT', ROOT)
sys.path.insert(0, os.path.join(ROOT, 'dashboard', 'backend'))


def _within(seconds, fn):
    done = threading.Event()
    t = threading.Thread(target=lambda: (fn(), done.set()), daemon=True)
    t.start()
    return done.wait(seconds)


def test_click_state_reset_do_not_deadlock():
    from app.arm_calib import ArmCalib
    rclpy.init()
    node = ArmCalib()
    try:
        def work():
            assert node.click('realsense', 'elbow', 100.0, 200.0) == (True, 'ok')
            node.click('realsense', 'flange', 300.0, 250.0)
            node.state('realsense', 'frame', (1280, 720))
            node.reset('realsense')
        assert _within(5, work), 'ArmCalib deadlocked'
        assert node.click('realsense', 'nope', 1, 1)[0] is False
    finally:
        node.destroy_node()
        rclpy.shutdown()
