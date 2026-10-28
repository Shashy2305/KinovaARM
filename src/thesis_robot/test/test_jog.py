"""Dashboard jog (dashboard/backend/app/kinematics.py, jog_math.py, jog.py): forward kinematics checked against a live TF reading, step sizes and caps, the table / joint-limit
checks, and the Jogger state machine (enable phrase, faults, busy arm, frozen feedback) with a stubbed publisher."""
import json
import os
import sys
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'dashboard', 'backend'))
from app import jog_math as jm  # noqa: E402
from app import kinematics as kin  # noqa: E402

URDF = open(os.path.join(HERE, 'data', 'gen3_robotiq_2f140.urdf')).read()
LIVE = json.load(open(os.path.join(HERE, 'data', 'live_pose_2026-10-22.json')))
Q = np.array(LIVE['q'])
GRIP = {'finger_joint': LIVE['finger_joint']}
GEOM = {'table_top_z': -0.0125, 'x': [-0.15, 0.784], 'y': [-0.899, 0.85], 'rear_wall_x': -0.2}


@pytest.fixture(scope='module')
def k():
    return kin.Kinematics(URDF)


def test_fk_matches_live_tf(k):
    for link, (x, y, z, qx, qy, qz, qw) in LIVE['links'].items():
        T = k.pose_for([link], Q, GRIP)[link]
        assert np.linalg.norm(T[:3, 3] - [x, y, z]) < 2e-4, link
        assert (Rotation.from_matrix(T[:3, :3]).inv() * Rotation.from_quat([qx, qy, qz, qw])).magnitude() < 1e-3, link


def test_limits_from_urdf(k):
    lo, hi = k.limits()
    assert np.isinf(lo[0]) and np.isinf(hi[2])
    assert hi[1] == pytest.approx(2.24) and lo[3] == pytest.approx(-2.57) and hi[5] == pytest.approx(2.09)


def test_jacobian_predicts_small_motion(k):
    rng = np.random.default_rng(1)
    dq = rng.normal(size=7) * 1e-3
    J = k.jacobian(Q)
    T0 = k.pose_for(['end_effector_link'], Q)['end_effector_link']
    T1 = k.pose_for(['end_effector_link'], Q + dq)['end_effector_link']
    assert np.allclose(J[:3] @ dq, T1[:3, 3] - T0[:3, 3], atol=2e-6)
    assert np.allclose(J[3:] @ dq, kin.rotvec(T1[:3, :3] @ T0[:3, :3].T), atol=2e-5)


def test_rotvec_roundtrip():
    for ang in (0.0, 0.3, 2.0, np.pi - 1e-3, np.pi):
        ax = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
        got = kin.rotvec(kin.rot_axis(ax, ang))
        assert np.allclose(np.abs(got), np.abs(ax * ang), atol=1e-4)


def test_profile_shape_and_coast_distance():
    v = np.array([0.1, -0.2, 0, 0, 0, 0, 0.0])
    q0 = np.ones(7)
    times, pos, vel = jm.velocity_profile(q0, v)
    assert times[0] == 0.0 and np.all(np.diff(times) > 0) and times[-1] == pytest.approx(jm.BACK_S + jm.HOLD_S + jm.RAMP_S)
    assert np.allclose(vel[-1], 0) and np.allclose(vel[0], v)
    # the line passes through q_ref exactly BACK_S after the (back-dated) start: this is what makes consecutive messages join
    assert np.allclose(pos[0] + v * jm.BACK_S, q0)
    assert np.allclose(pos[-1] - q0, jm.coast_displacement(v))
    # the positions are the integral of the velocities (trapezoid), so the controller sees a consistent spline
    est = [pos[0]]
    for i in range(1, len(times)):
        est.append(est[-1] + 0.5 * (vel[i] + vel[i - 1]) * (times[i] - times[i - 1]))
    assert np.allclose(est[-1], pos[-1], atol=2e-3)
    assert np.all(np.sign(np.diff(pos[:, 0])) >= 0)


def test_velocity_changes_gently():
    v = np.zeros(7)
    target = np.array([0.3, 0, 0, 0, 0, 0, 0.0])
    steps = 0
    while not np.allclose(v, target) and steps < 100:
        nv = jm.approach_velocity(v, target, 0.08)
        assert np.max(np.abs(nv - v)) <= jm.ACCEL_RAD_S2 * 0.08 + 1e-12
        v, steps = nv, steps + 1
    assert steps == 5                                                # 0.3 / (0.8 * 0.08) = 4.7 ticks
    # all joints are scaled together: the direction of a Cartesian move is kept while it speeds up
    nv = jm.approach_velocity(np.zeros(7), np.array([0.3, 0.15, 0, 0, 0, 0, 0.0]), 0.08)
    assert nv[1] / nv[0] == pytest.approx(0.5)


def test_coast_after_commands_stop_is_small():
    d = jm.coast_displacement(np.full(7, jm.JOINT_MAX_RAD_S))
    assert np.degrees(d.max()) < 5.0                                # at full speed the arm still stops within 5 degrees of a joint


def test_joint_step_is_capped_and_directional():
    dq = jm.joint_step(Q, 2, +1, 1.0)
    assert dq[2] == pytest.approx(jm.JOINT_MAX_RAD_S * jm.HORIZON_S) and np.count_nonzero(dq) == 1
    assert jm.joint_step(Q, 0, -1, 5.0)[0] == pytest.approx(-jm.JOINT_MAX_RAD_S * jm.HORIZON_S)   # speed > 1 is clamped
    assert not np.any(jm.joint_step(Q, 0, 0, 1.0))


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_cart_step_moves_flange_the_requested_way(k, axis):
    v = np.zeros(6)
    v[axis] = 1.0
    dq, bad = jm.cart_step(k, Q, v, 0.5)
    assert bad is None
    T0 = k.pose_for(['end_effector_link'], Q)['end_effector_link']
    T1 = k.pose_for(['end_effector_link'], Q + dq)['end_effector_link']
    d = T1[:3, 3] - T0[:3, 3]
    want = 0.5 * jm.LINEAR_MAX_M_S * jm.HORIZON_S
    assert d[axis] == pytest.approx(want, rel=0.15)
    assert np.linalg.norm(np.delete(d, axis)) < 0.15 * want
    assert np.max(np.abs(dq)) / jm.HORIZON_S <= jm.JOINT_MAX_RAD_S * 0.5 + 1e-9


def test_cart_rotation_and_tool_frame(k):
    dq, bad = jm.cart_step(k, Q, [0, 0, 0, 0, 0, 1.0], 0.5)
    assert bad is None
    T0 = k.pose_for(['end_effector_link'], Q)['end_effector_link']
    T1 = k.pose_for(['end_effector_link'], Q + dq)['end_effector_link']
    rv = kin.rotvec(T1[:3, :3] @ T0[:3, :3].T)
    want = 0.5 * jm.ANGULAR_MAX_RAD_S * jm.HORIZON_S
    # the joint-speed cap may slow a rotation down (never speed it up) and the rotation stays about the requested axis
    assert 0.4 * want < rv[2] <= 1.15 * want
    assert np.linalg.norm(rv[:2]) < 0.25 * rv[2]
    # tool-frame +z of the tool is the base-frame direction R[:,2]
    dq, _ = jm.cart_step(k, Q, [0, 0, 1.0, 0, 0, 0], 0.5, frame='tool')
    T1 = k.pose_for(['end_effector_link'], Q + dq)['end_effector_link']
    d = (T1[:3, 3] - T0[:3, 3])
    cosang = d @ T0[:3, 2] / np.linalg.norm(d)
    assert cosang > 0.97


def test_cart_zero_input_is_no_motion(k):
    dq, bad = jm.cart_step(k, Q, np.zeros(6), 1.0)
    assert bad is None and not np.any(dq)


def test_singularity_is_refused(k):
    q = np.array([0, 0.0, 0, 0.0, 0, 0.0, 0])           # arm stretched straight up
    dq, bad = jm.cart_step(k, q, [1, 0, 0, 0, 0, 0], 1.0)
    assert dq is None and 'singular' in bad


def test_joint_limit_margin(k):
    q = Q.copy()
    q[1] = 2.24 - jm.JOINT_LIMIT_MARGIN_RAD - 0.01
    ok, _ = jm.check_joint_limits(k, q, q + jm.joint_step(q, 1, +1, 1.0))
    assert not ok
    ok, _ = jm.check_joint_limits(k, q, q + jm.joint_step(q, 1, -1, 1.0))      # moving back out is allowed
    assert ok
    ok, _ = jm.check_joint_limits(k, Q, Q + jm.joint_step(Q, 2, +1, 1.0))      # continuous joint: no limit
    assert ok


def test_check_pose_needs_geometry(k):
    ok, why = jm.check_pose(k, Q, Q, None, GRIP)
    assert not ok and 'geometry' in why


def test_table_floor_blocks_going_down_but_not_up(k):
    z_pads = min(p[2] for p in jm.guarded_positions(k, Q, GRIP).values())
    geom = dict(GEOM, table_top_z=z_pads - sg_margin() + 0.01)   # table limit 1 cm ABOVE the lowest link: the arm starts "too low"
    down, _ = jm.cart_step(k, Q, [0, 0, -1, 0, 0, 0], 1.0)
    up, _ = jm.cart_step(k, Q, [0, 0, 1, 0, 0, 0], 1.0)
    ok_down, why = jm.check_pose(k, Q, Q + down, geom, GRIP)
    ok_up, _ = jm.check_pose(k, Q, Q + up, geom, GRIP)
    assert not ok_down and 'below' in why
    assert ok_up
    # a table far below allows both
    far = dict(GEOM, table_top_z=-0.5)
    assert jm.check_pose(k, Q, Q + down, far, GRIP)[0]


def sg_margin():
    from thesis_robot import safety_geometry as sg
    return sg.LINK_TABLE_CLEARANCE_M


def test_rear_limit_blocks_moving_further_back(k):
    # pose with the flange behind x = -0.10 cannot be jogged further back, but can come forward
    from thesis_robot import safety_geometry as sg
    q = np.array([np.pi, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    pos = jm.guarded_positions(k, q)
    if min(p[0] for p in pos.values()) >= sg.REAR_KEEP_OUT_X:
        pytest.skip('constructed pose is not behind the rear limit')
    back, _ = jm.cart_step(k, q, [-1, 0, 0, 0, 0, 0], 1.0)
    fwd, _ = jm.cart_step(k, q, [1, 0, 0, 0, 0, 0], 1.0)
    far = dict(GEOM, table_top_z=-5.0)
    if back is not None:
        assert not jm.check_pose(k, q, q + back, far)[0]
    if fwd is not None:
        assert jm.check_pose(k, q, q + fwd, far)[0]


# ── Jogger state machine (needs rclpy) ──────────────────────────────────────
rclpy = pytest.importorskip('rclpy')


class StubPub:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


@pytest.fixture(scope='module')
def ros():
    """One rclpy context for these tests, shut down afterwards (other test files call rclpy.init themselves)."""
    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    yield
    if owned:
        rclpy.shutdown()


@pytest.fixture()
def jogger(ros, monkeypatch):
    from app import jog
    from thesis_robot import safety_geometry as sg
    j = jog.Jogger(urdf_xml=URDF)
    j.pub = StubPub()
    monkeypatch.setattr(sg, 'load_geometry', lambda *a, **kw: (dict(GEOM, table_top_z=-0.5), None))
    j.set_q = lambda q=Q: j.feed.inject(_js(q))
    j.set_q()
    yield j
    j.destroy_node()


def _js(q):
    from sensor_msgs.msg import JointState
    m = JointState()
    m.name = list(kin.ARM_JOINTS) + ['finger_joint']
    m.position = [float(v) for v in q] + [0.0]
    return m


def test_enable_requires_phrase(jogger):
    assert not jogger.enable('yes')['ok']
    assert not jogger.enabled
    assert jogger.enable('Area clear, hands off ')['ok']
    assert jogger.enabled


def test_enable_refused_with_fault_busy_or_stale(jogger):
    from std_msgs.msg import Bool, String
    jogger._on_fault(Bool(data=True))
    assert 'fault' in jogger.enable('area clear, hands off')['error']
    jogger._on_fault(Bool(data=False))
    jogger._on_status(String(data='EXECUTING: pick the cup'))
    assert 'busy' in jogger.enable('area clear, hands off')['error']
    jogger._on_status(String(data='READY [DRY RUN]'))
    jogger.feed._t -= 5.0
    assert 'joint states' in jogger.enable('area clear, hands off')['error']


def test_command_when_off_does_nothing(jogger):
    st = jogger.command({'mode': 'joint', 'joint': 1, 'dir': 1, 'speed': 0.5})
    assert st['blocked'] == 'jog is off' and not jogger.pub.sent


def test_joint_command_publishes_a_continuous_velocity_profile(jogger):
    jogger.enable('area clear, hands off')
    st = jogger.command({'mode': 'joint', 'joint': 3, 'dir': -1, 'speed': 1.0})
    assert st['moved'] and len(jogger.pub.sent) == 1
    msg = jogger.pub.sent[0]
    pts = msg.points
    v = -jm.ACCEL_RAD_S2 * jm.NOMINAL_TICK_S                          # first tick of a move: only one acceleration step of the full speed
    assert all(len(p.velocities) == 7 for p in pts)                   # position-only points made the arm vibrate at the command rate
    assert pts[0].velocities[2] == pytest.approx(v)
    assert pts[0].positions[2] + v * jm.BACK_S == pytest.approx(Q[2])  # the line passes through the arm's position BACK_S after the stamp
    assert msg.header.stamp.sec > 0                                   # back-dated
    assert all(abs(x) < 1e-12 for x in pts[-1].velocities)            # ends at rest (the controller rejects a non-zero end velocity)
    tf = pts[-1].time_from_start
    assert tf.sec + tf.nanosec / 1e9 == pytest.approx(jm.BACK_S + jm.HOLD_S + jm.RAMP_S, abs=1e-6)


def test_consecutive_messages_lie_on_one_line_and_speed_up_gently(jogger):
    jogger.enable('area clear, hands off')
    jogger.command({'mode': 'joint', 'joint': 1, 'dir': 1, 'speed': 1.0})
    t1, ref1, v1 = jogger.last_tick_t, jogger._ref.copy(), jogger._v.copy()
    time.sleep(0.09)
    jogger.command({'mode': 'joint', 'joint': 1, 'dir': 1, 'speed': 1.0})       # the arm has not moved in this test: the reference must not be reset to it
    t2, ref2, v2 = jogger.last_tick_t, jogger._ref, jogger._v
    assert ref2[0] == pytest.approx(ref1[0] + v1[0] * (t2 - t1), abs=1e-9)     # message 2 starts exactly where message 1's line is at that moment
    assert v1[0] < v2[0] <= v1[0] + jm.ACCEL_RAD_S2 * (t2 - t1) + 1e-9
    msg = jogger.pub.sent[1].points[0]
    assert msg.positions[0] + msg.velocities[0] * jm.BACK_S == pytest.approx(ref2[0])


def test_bad_commands_are_rejected_without_motion(jogger):
    jogger.enable('area clear, hands off')
    for cmd in ({'mode': 'joint', 'joint': 9, 'dir': 1}, {'mode': 'cart', 'v': [1, 2]}, {'mode': 'cart', 'v': [float('nan')] * 6},
                {'mode': 'banana'}, {}):
        time.sleep(0.07)
        st = jogger.command(cmd)
        assert st['blocked'] and not st['moved']
    assert not jogger.pub.sent


def test_fault_during_jog_stops_and_disables(jogger):
    from std_msgs.msg import Bool
    jogger.enable('area clear, hands off')
    jogger._on_fault(Bool(data=True))
    st = jogger.command({'mode': 'joint', 'joint': 1, 'dir': 1, 'speed': 0.5})
    assert not jogger.enabled and 'fault' in (st['blocked'] or '')
    assert len(jogger.pub.sent) == 1                      # the hold-position message, not a move
    assert jogger.pub.sent[0].points[0].positions == pytest.approx(list(Q))


def test_frozen_feedback_trips_the_jog(jogger, monkeypatch):
    from app import jog
    monkeypatch.setattr(jog, 'FOLLOW_CHECK_S', 0.3)
    jogger.enable('area clear, hands off')
    t0 = time.monotonic()
    tripped = False
    while time.monotonic() - t0 < 1.5 and not tripped:
        jogger.set_q()                                     # joint states keep arriving but never change
        jogger.command({'mode': 'joint', 'joint': 1, 'dir': 1, 'speed': 1.0})
        tripped = not jogger.enabled
        time.sleep(0.07)
    assert tripped and 'not following' in jogger.message


def test_following_feedback_does_not_trip(jogger, monkeypatch):
    from app import jog
    monkeypatch.setattr(jog, 'FOLLOW_CHECK_S', 0.3)
    jogger.enable('area clear, hands off')
    q = Q.copy()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        st = jogger.command({'mode': 'joint', 'joint': 1, 'dir': 1, 'speed': 1.0})
        if st['moved']:
            q = q + jm.joint_step(q, 0, 1, 1.0)            # the arm does what it was told
            jogger.set_q(q)
        time.sleep(0.07)
    assert jogger.enabled


def test_idle_timeout_switches_off(jogger, monkeypatch):
    from app import jog
    jogger.enable('area clear, hands off')
    jogger.last_cmd_t -= jog.IDLE_DISABLE_S + 1
    jogger.tick_idle()
    assert not jogger.enabled


def test_stop_holds_position_and_disables(jogger):
    jogger.enable('area clear, hands off')
    jogger.stop('test')
    assert not jogger.enabled and len(jogger.pub.sent) == 1
    assert jogger.pub.sent[0].points[0].positions == pytest.approx(list(Q))


# ---- the gripper buttons --------------------------------------------------------------------------------------------------------------------
def test_the_gripper_is_refused_while_the_jog_is_off_faulted_or_the_arm_is_busy(jogger):
    from std_msgs.msg import Bool, String
    goals = []
    jogger._gripper_goal = lambda t: (goals.append(t) or (True, 'ok'))
    assert not jogger.gripper('open')['ok'] and goals == []                   # jog off
    assert jogger.enable('area clear, hands off')['ok']
    jogger._on_status(String(data='EXECUTING: pick the cup'))
    assert 'busy' in jogger.gripper('open')['message'] and goals == []
    jogger._on_status(String(data='READY'))
    jogger._on_fault(Bool(data=True))
    assert 'fault' in jogger.gripper('open')['message'] and goals == []


def test_open_close_and_set_send_the_right_finger_positions_and_clamp(jogger):
    goals = []
    jogger._gripper_goal = lambda t: (goals.append(t) or (True, 'ok'))
    assert jogger.enable('area clear, hands off')['ok']
    assert jogger.gripper('open')['ok'] and jogger.gripper('close')['ok']
    assert jogger.gripper('set', 0.3)['ok'] and jogger.gripper('set', 5.0)['ok'] and jogger.gripper('set', -1)['ok']
    assert goals == [0.0, 0.695, 0.3, 0.695, 0.0]
    for bad in (None, 'x', float('nan')):
        assert not jogger.gripper('set', bad)['ok']
    assert not jogger.gripper('squeeze')['ok']
    assert len(goals) == 5                                                    # nothing was sent for the bad requests


def test_the_state_reports_the_finger_position(jogger):
    assert jogger.state()['finger'] == 0.0
