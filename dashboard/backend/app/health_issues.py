"""
What is wrong right now, in words an operator can act on (pure python, no ROS): the dashboard's health banner.

Everything here was seen on the real system: the Kortex driver in a fault-recovery loop with frozen joint feedback (2026-10-22, eight calibration poses recorded the same arm pose), the
arm's Ethernet link going down, a controller spawner timing out and leaving joint_state_broadcaster unconfigured (no joint states, no TF), a backend whose callbacks starved
(no camera frames, joint feed 5 s late). evaluate() turns a snapshot of the facts into a list of issues, each with a plain-language hint of what to do.
"""
import re

FROZEN_S = 3.0               # all 7 joint positions bit-for-bit identical this long while messages keep arriving: a live arm always shows sensor noise
STALE_JOINT_S = 1.5
LAG_S = 3.0                  # the backend's own 0.5 s timer is this late: its ROS callbacks are starved
LOG_WINDOW_S = 90.0
WARMUP_S = 15.0              # a freshly started backend has not discovered the topics yet

# (id, regex on a robot_bringup log line, severity, title, detail, hint)
LOG_PATTERNS = [
    ('driver_fault_recovery', r'RefreshFeedback failed in fault recovery', 'error',
     'The Kortex driver is stuck in fault recovery',
     'The driver cannot refresh the arm\'s feedback, so joint states freeze at the last value. Usual cause: a second control session on the arm (the Xbox pad over Kinova\'s own '
     'session, the Kinova web app) while robot bringup runs.',
     'Stop using the other controller, then restart Robot bringup (it does not move the arm).'),
    ('arm_unreachable', r'There was a problem connecting to the server', 'error',
     'The driver cannot connect to the arm (192.168.1.10)',
     'Robot bringup started but the arm did not answer.',
     'Check the arm is powered, the Ethernet cable is seated at both ends and the link below is up, then restart Robot bringup.'),
    ('controller_failed', r'Failed loading controller|Failed getting a result from calling /controller_manager', 'error',
     'A controller failed to start',
     'The controller manager was too slow to answer when the controllers were started; a controller (usually joint_state_broadcaster) may be left unconfigured, which means no joint '
     'states and no arm TF.',
     'Restart Robot bringup. If it repeats: ros2 control set_controller_state joint_state_broadcaster inactive, then active.'),
]


def _issue(id_, severity, title, detail, hint):
    return {'id': id_, 'severity': severity, 'title': title, 'detail': detail, 'hint': hint}


def log_hits(lines, now_epoch, window_s=LOG_WINDOW_S):
    """{pattern id: count} of robot_bringup log lines matching a known failure whose own ROS timestamp [seconds.nanoseconds] is within window_s of now."""
    out = {}
    stamp = re.compile(r'\[(\d{10})\.\d+\]')
    for ln in lines:
        m = stamp.search(ln)
        if m and now_epoch - float(m.group(1)) > window_s:
            continue
        for pid, rx, *_ in LOG_PATTERNS:
            if re.search(rx, ln):
                out[pid] = out.get(pid, 0) + 1
    return out


def evaluate(s):
    """s: {'robot_expected': bool, 'joint_age_s': float|None, 'frozen_for_s': float, 'fault': bool|None, 'tf_ok': bool|None, 'executor_lag_s': float,
    'cameras': {name: {'has_frame': bool, 'stale': bool, 'age_s': float|None, 'label': str}}, 'log_hits': {id: count}, 'arm_link_up': bool|None}.
    Returns a list of issues, most serious first."""
    issues = []
    if (s.get('uptime_s') if s.get('uptime_s') is not None else 1e9) < WARMUP_S:
        return issues                                    # the backend has only just started: topics have not matched yet, nothing is known
    robot = bool(s.get('robot_expected'))
    age = s.get('joint_age_s')
    lag = s.get('executor_lag_s') or 0.0
    if lag > LAG_S:
        issues.append(_issue('backend_lag', 'error', 'The dashboard backend is overloaded or hung',
                             f'Its own ROS timer is {lag:.1f} s late, so camera frames, joint feedback and status are late or missing here (the robot itself may be fine).',
                             'Restart it: bash scripts/restart_dashboard_backend.sh. If it keeps happening, say so: something is using too much CPU.'))
    if s.get('arm_link_up') is False:
        issues.append(_issue('arm_link_down', 'error', 'The Ethernet link to the arm is down',
                             'The PC network port for the arm (192.168.1.x) has no link: no cable, a loose connector, or the arm is off.',
                             'Check the arm is on and re-seat the cable at both ends. Then restart Robot bringup.'))
    hits = s.get('log_hits') or {}
    for pid, _rx, sev, title, detail, hint in LOG_PATTERNS:
        if hits.get(pid):
            issues.append(_issue(pid, sev, title, f'{detail} (seen {hits[pid]}x in the last {int(LOG_WINDOW_S)} s of the driver log)', hint))
    if s.get('fault'):
        issues.append(_issue('fault_flag', 'error', 'The driver reports a fault',
                             '/fault_controller/internal_fault is true: the arm is in a fault state and will not follow commands.',
                             'Clear the cause (check the arm\'s own display / E-stop), then restart Robot bringup.'))
    if robot and (age is None or age > STALE_JOINT_S):
        issues.append(_issue('joints_missing', 'error', 'No joint states from the arm',
                             'Robot bringup is running but nothing has published the arm\'s joint positions' + ('' if age is None else f' for {age:.0f} s') + '.',
                             'Restart Robot bringup. If it persists, check the cable and that nobody else (the kinova account) holds the arm.'))
    elif robot and (s.get('frozen_for_s') or 0) >= FROZEN_S:
        # identical values alone prove nothing: a stationary arm's encoders can also report exactly the same numbers (seen live). It is a FREEZE only when the driver says so too.
        corroborated = bool(s.get('fault')) or bool((s.get('log_hits') or {}).get('driver_fault_recovery'))
        if corroborated:
            issues.append(_issue('joints_frozen', 'error', 'The arm\'s joint feedback is FROZEN',
                                 f'Joint states keep arriving but every value has been identical for {s["frozen_for_s"]:.0f} s and the driver reports a fault. Nothing that uses the arm\'s '
                                 'position (calibration poses, planning, the jog) can be trusted.',
                                 'Restart Robot bringup (it does not move the arm). Do not click calibration poses until this is gone.'))
        else:
            issues.append(_issue('joints_unchanged', 'warn', f'Joint values have not changed for {s["frozen_for_s"]:.0f} s',
                                 'Normal when the arm is at rest. If you moved the arm (jog, hand-guiding) and the values stay exactly the same, the feedback is frozen '
                                 '(the driver in a fault loop): the banner turns red when the driver confirms it.',
                                 'Before clicking calibration poses, jog the arm a little and check the joint angles on the Jog tab change. If not, restart Robot bringup.'))
    if robot and age is not None and age <= STALE_JOINT_S and s.get('tf_ok') is False:
        issues.append(_issue('tf_missing', 'error', 'The arm has no TF (base_link is missing)',
                             'Joint states arrive but base_link -> end_effector_link cannot be looked up: robot_state_publisher is not publishing. Calibration from the arm, the scene '
                             'map and the safety checks need it.',
                             'Restart Robot bringup.'))
    for name, c in (s.get('cameras') or {}).items():
        if not c.get('has_frame') or c.get('stale'):
            when = '' if c.get('age_s') is None else f' (last frame {c["age_s"]:.0f} s ago)'
            issues.append(_issue(f'camera_{name}', 'warn', f'{c.get("label", name)} has no live image' + when,
                                 'The dashboard is not receiving frames from this camera.',
                                 'Check the USB cable / port (USB 3, direct on the PC), then restart Cameras bringup. Note: a restart flags the static cameras for recalibration check.'))
    order = {'error': 0, 'warn': 1}
    issues.sort(key=lambda i: order.get(i['severity'], 2))
    return issues
