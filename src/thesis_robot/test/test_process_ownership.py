"""Another account's process must never count as ours: not 'already running', not killable, and it is reported as holding hardware.
Uses a fake process table; no ROS, no real processes."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dashboard', 'backend'))

from app import config, process_manager as pm   # noqa: E402

ROBOT_SIG = config.PROCESSES['robot_bringup']['signature']


def manager(table):
    """table: {pid: (user, cmdline)}"""
    m = pm.ProcessManager()
    m._me = 'shash'
    cmdlines = {pid: cmd for pid, (_u, cmd) in table.items()}
    m._owners = {pid: u for pid, (u, _c) in table.items()}
    m._snapshot_cmdlines = lambda: cmdlines          # keep the injected owners
    return m, cmdlines


def test_other_users_driver_is_not_running_external():
    m, c = manager({10: ('kinova', f'/usr/bin/python3 ros2 launch {ROBOT_SIG} robot_ip:=192.168.1.10')})
    assert m.status('robot_bringup', c) == 'running_other_user'
    assert m.other_users('robot_bringup', c) == ['kinova']


def test_own_process_outside_dashboard_is_still_running_external():
    m, c = manager({10: ('shash', f'ros2 launch {ROBOT_SIG}')})
    assert m.status('robot_bringup', c) == 'running_external'


def test_ours_wins_over_theirs():
    m, c = manager({10: ('kinova', f'ros2 launch {ROBOT_SIG}'), 11: ('shash', f'ros2 launch {ROBOT_SIG}')})
    assert m.status('robot_bringup', c) == 'running_external'


def test_start_refuses_with_a_name_and_stop_does_not_touch_it(monkeypatch):
    m, c = manager({10: ('kinova', f'ros2 launch {ROBOT_SIG}')})
    killed = []
    monkeypatch.setattr(m, '_terminate_tree', lambda pid, timeout=8.0: killed.append(pid) or True)
    ok, msg = m.start('robot_bringup')
    assert not ok and 'kinova' in msg
    ok, msg = m.stop('robot_bringup')
    assert not ok and 'kinova' in msg and not killed


def test_hardware_holders_name_the_camera_and_skip_shells():
    m, c = manager({
        1: ('kinova', "tmux new -d -s front_cam bash -c 'ros2 launch realsense2_camera rs_launch.py serial_no:=_938422070760'"),
        2: ('kinova', 'bash -c ros2 launch realsense2_camera rs_launch.py serial_no:=_938422070760'),
        3: ('kinova', '/home/kinova/x/kinova_vision_node --ros-args -r __node:=kinova_vision_color'),
        4: ('kinova', '/home/kinova/x/ros2_control_node --params-file p.yaml'),
        5: ('shash', '/x/ros2_control_node --params-file ours.yaml'),
        6: ('kinova', '/opt/ros/humble/lib/realsense2_camera/realsense2_camera_node --ros-args -r __node:=front_cam'),
    })
    out = m.hardware_holders(c)
    whats = sorted(h['what'] for h in out)
    assert any('RealSense RS1' in w for w in whats)
    assert any('wrist camera' in w for w in whats)
    assert any('arm driver' in w for w in whats)
    assert all(h['user'] == 'kinova' for h in out)               # our own ros2_control_node is not listed
    assert not any(w == 'a RealSense camera' for w in whats)     # the generic row is dropped when a serial names the camera
