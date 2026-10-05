import time
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
tf2_ros = pytest.importorskip('tf2_ros')
from geometry_msgs.msg import TransformStamped  # noqa: E402

from thesis_robot.scene_graph_node import SceneGraphNode  # noqa: E402


def tf_at(t, x, frame='wrist_cam'):
    m = TransformStamped()
    m.header.frame_id = 'base_link'
    m.child_frame_id = frame
    m.header.stamp.sec = int(t)
    m.header.stamp.nanosec = int((t % 1) * 1e9)
    m.transform.translation.x = x
    m.transform.rotation.w = 1.0
    return m


def lookup(buf, frame, stamp):
    return SceneGraphNode._lookup_tf(SimpleNamespace(tf_buffer=buf), frame, stamp)


@pytest.fixture
def moving_camera():
    """The wrist camera moves 1 m/s along x: x == (t - t0)."""
    buf = tf2_ros.Buffer()
    t0 = time.time() - 0.6
    for k in range(7):
        buf.set_transform(tf_at(t0 + 0.1 * k, 0.1 * k), 'test')
    return buf, t0


def test_uses_the_pose_at_capture_time_not_the_latest(moving_camera):
    buf, t0 = moving_camera
    tf = lookup(buf, 'wrist_cam', t0 + 0.2)          # image taken 0.4 s ago
    assert tf.transform.translation.x == pytest.approx(0.2, abs=0.01)
    latest = lookup(buf, 'wrist_cam', None)           # what the old code used
    assert latest.transform.translation.x == pytest.approx(0.6, abs=0.01)


def test_frames_older_than_1_5_seconds_are_rejected(moving_camera):
    buf, _ = moving_camera
    assert lookup(buf, 'wrist_cam', time.time() - 3.0) is False


def test_foreign_clock_stamp_is_ignored_and_latest_is_used(moving_camera):
    buf, _ = moving_camera
    tf = lookup(buf, 'wrist_cam', time.time() - 100000.0)   # device clock, not ROS time
    assert tf.transform.translation.x == pytest.approx(0.6, abs=0.01)


def test_unknown_frame_returns_none(moving_camera):
    buf, t0 = moving_camera
    assert lookup(buf, 'nope', t0 + 0.2) is None


def test_static_camera_needs_no_matching_time():
    buf = tf2_ros.Buffer()
    buf.set_transform_static(tf_at(1.0, 0.5, 'static_cam'), 'test')
    assert lookup(buf, 'static_cam', time.time() - 0.3).transform.translation.x == pytest.approx(0.5)
