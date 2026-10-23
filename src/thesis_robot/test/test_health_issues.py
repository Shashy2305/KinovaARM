"""The dashboard's health banner logic (dashboard/backend/app/health_issues.py): every failure seen on the real system turns into a plain-language issue with a hint."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dashboard', 'backend'))
from app import health_issues as hi  # noqa: E402


def healthy(**kw):
    s = {'robot_expected': True, 'joint_age_s': 0.01, 'frozen_for_s': 0.0, 'fault': False, 'tf_ok': True, 'executor_lag_s': 0.0, 'uptime_s': 600.0, 'arm_link_up': True,
         'log_hits': {}, 'cameras': {'realsense': {'has_frame': True, 'stale': False, 'label': 'RealSense 1'}}}
    s.update(kw)
    return s


def ids(s):
    return [i['id'] for i in hi.evaluate(s)]


def test_a_healthy_system_has_no_issues():
    assert hi.evaluate(healthy()) == []


def test_frozen_joint_feedback_is_reported_with_a_restart_hint():
    out = hi.evaluate(healthy(frozen_for_s=12.0))
    assert [i['id'] for i in out] == ['joints_frozen']
    assert 'FROZEN' in out[0]['title'] and 'Restart Robot bringup' in out[0]['hint'] and '12' in out[0]['detail']
    assert ids(healthy(frozen_for_s=1.0)) == []                      # a short pause of identical values is not a freeze


def test_missing_joint_states_and_missing_tf():
    assert ids(healthy(joint_age_s=None)) == ['joints_missing']
    assert ids(healthy(joint_age_s=9.0)) == ['joints_missing']
    assert ids(healthy(tf_ok=False)) == ['tf_missing']
    assert ids(healthy(robot_expected=False, joint_age_s=None, tf_ok=False)) == []        # bringup stopped on purpose: not a fault
    assert ids(healthy(joint_age_s=None, tf_ok=False)) == ['joints_missing']               # no joint states: do not also blame TF


def test_fault_flag_and_arm_link():
    assert ids(healthy(fault=True)) == ['fault_flag']
    assert ids(healthy(arm_link_up=False)) == ['arm_link_down']
    assert ids(healthy(arm_link_up=None)) == []                                              # interface unknown: say nothing


def test_backend_lag_is_an_error():
    out = hi.evaluate(healthy(executor_lag_s=7.0))
    assert out[0]['id'] == 'backend_lag' and out[0]['severity'] == 'error' and 'restart_dashboard_backend' in out[0]['hint']


def test_camera_without_frames_is_a_warning_sorted_after_errors():
    s = healthy(fault=True, cameras={'realsense3': {'has_frame': False, 'stale': True, 'label': 'RealSense 3'}})
    out = hi.evaluate(s)
    assert [i['id'] for i in out] == ['fault_flag', 'camera_realsense3'] and out[1]['severity'] == 'warn' and 'RealSense 3' in out[1]['title']
    assert ids(healthy(cameras={'realsense': {'has_frame': True, 'stale': True, 'label': 'x', 'age_s': 8}})) == ['camera_realsense']


def test_log_patterns_only_count_recent_lines():
    now = 1792732400.0
    lines = [
        '[ros2_control_node-1] [ERROR] [1792732390.5] [KortexMultiInterfaceHardware]: RefreshFeedback failed in fault recovery: std::future_error',
        '[ros2_control_node-1] [ERROR] [1792732391.5] [KortexMultiInterfaceHardware]: RefreshFeedback failed in fault recovery: std::future_error',
        '[ros2_control_node-1] There was a problem connecting to the server. Aborting',            # no stamp: counted
        '[spawner-3] [FATAL] [1792700000.0] [spawner_joint_state_broadcaster]: Failed loading controller',  # old: ignored
    ]
    hits = hi.log_hits(lines, now)
    assert hits == {'driver_fault_recovery': 2, 'arm_unreachable': 1}
    out = hi.evaluate(healthy(log_hits=hits))
    assert {'driver_fault_recovery', 'arm_unreachable'} <= {i['id'] for i in out}
    assert 'Xbox' in next(i for i in out if i['id'] == 'driver_fault_recovery')['detail']


def test_nothing_is_reported_while_the_backend_is_warming_up():
    assert hi.evaluate(healthy(uptime_s=3.0, joint_age_s=None, cameras={'a': {'has_frame': False, 'stale': True}})) == []
