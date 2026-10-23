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


def test_frozen_joint_feedback_is_an_error_only_when_the_driver_confirms_it():
    out = hi.evaluate(healthy(frozen_for_s=12.0, fault=True))
    assert 'joints_frozen' in [i['id'] for i in out]
    frozen = next(i for i in out if i['id'] == 'joints_frozen')
    assert frozen['severity'] == 'error' and 'FROZEN' in frozen['title'] and 'Restart Robot bringup' in frozen['hint'] and '12' in frozen['detail']
    out = hi.evaluate(healthy(frozen_for_s=12.0, log_hits={'driver_fault_recovery': 40}))
    assert 'joints_frozen' in [i['id'] for i in out]
    # identical values alone (a stationary arm reports exactly the same numbers) is only a warning with a way to check
    out = hi.evaluate(healthy(frozen_for_s=12.0))
    assert [i['id'] for i in out] == ['joints_unchanged'] and out[0]['severity'] == 'warn' and 'jog the arm' in out[0]['hint']
    assert ids(healthy(frozen_for_s=1.0)) == []                      # a short pause of identical values is nothing


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


def test_every_issue_with_a_remedy_offers_a_button_and_a_terminal_command():
    cases = {
        'driver_fault_recovery': healthy(log_hits={'driver_fault_recovery': 3}),
        'fault_flag': healthy(fault=True),
        'tf_missing': healthy(tf_ok=False),
        'joints_missing': healthy(joint_age_s=None),
        'backend_lag': healthy(executor_lag_s=9.0),
        'camera_realsense3': healthy(cameras={'realsense3': {'has_frame': False, 'stale': True, 'label': 'RealSense 3'}}),
        'controller_failed': healthy(log_hits={'controller_failed': 1}),
        'arm_unreachable': healthy(log_hits={'arm_unreachable': 1}),
    }
    for want, snap in cases.items():
        issue = next(i for i in hi.evaluate(snap) if i['id'] == want)
        assert issue['actions'], want
        for a in issue['actions']:
            assert a['id'] in hi.ACTIONS and a['label'] and a['confirm'] and a['command'], (want, a)
    # a physical problem has no button but a check command; a camera restart offers exactly the camera button
    link = next(i for i in hi.evaluate(healthy(arm_link_up=False)) if i['id'] == 'arm_link_down')
    assert link['actions'] == [] and 'ping' in link['command']
    cam = next(i for i in hi.evaluate(cases['camera_realsense3']) if i['id'] == 'camera_realsense3')
    assert [a['id'] for a in cam['actions']] == ['restart_cameras']
    # a broken controller start offers the cheap fix first, the full restart second
    ctl = next(i for i in hi.evaluate(cases['controller_failed']) if i['id'] == 'controller_failed')
    assert [a['id'] for a in ctl['actions']] == ['fix_controllers', 'restart_robot_bringup']


def test_fix_controllers_only_touches_the_state_publishers(monkeypatch):
    from app import config
    monkeypatch.setattr(config, 'REPO_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))      # an earlier test may have imported config with the default root
    from app.routers import health as h
    calls = []

    def fake_ros(cmd, timeout=40):
        calls.append(cmd)
        if 'list_controllers' in cmd:
            return 0, ('joint_trajectory_controller active\njoint_state_broadcaster unconfigured\nfault_controller active\n'
                       'twist_controller inactive')
        return 0, 'Successfully activated'
    monkeypatch.setattr(h, '_ros', fake_ros)
    ok, msg = h.fix_controllers()
    assert ok and 'joint_state_broadcaster activated' in msg
    sets = [c for c in calls if 'set_controller_state' in c]
    assert sets == ['ros2 control set_controller_state joint_state_broadcaster inactive', 'ros2 control set_controller_state joint_state_broadcaster active']
    assert not any('joint_trajectory' in c or 'twist' in c for c in sets)
    calls.clear()
    monkeypatch.setattr(h, '_ros', lambda cmd, timeout=40: (0, 'joint_state_broadcaster active\nfault_controller active'))
    ok, msg = h.fix_controllers()
    assert ok and 'nothing to fix' in msg
