"""2026-11-03: the held object is lowered onto its spot until the joint torques show it touching (it slammed onto the table from the inferred height)."""
import pytest

from thesis_robot import arm_controller_node as acn

from test_pick_place_flow import node  # noqa: F401

BASE = [3.9, -10.4, -2.2]


def world(node, monkeypatch, contact_z=None, fail_at=None):
    """Fake moves that track the flange z, and a torque signal that jumps once z is at or below contact_z."""
    node.dry_run, node.moveit2 = False, object()
    node._je = {'joint_2': BASE[0], 'joint_4': BASE[1], 'joint_6': BASE[2]}
    state = {'z': 0.5, 'moves': []}

    def move(x, y, z, label, **kw):
        state['moves'].append((round(z, 4), label))
        if fail_at is not None and z <= fail_at:
            return False
        state['z'] = z
        return True

    def effort(seconds=0.25):
        touching = contact_z is not None and state['z'] <= contact_z + 1e-9
        return [b + (0.8 if touching else 0.01) for b in BASE]
    monkeypatch.setattr(node, '_move_verified', move)
    monkeypatch.setattr(node, '_effort_vector', effort)
    monkeypatch.setattr(acn.time, 'sleep', lambda s: None)
    return state


def test_it_stops_at_the_first_step_that_feels_the_table(node, monkeypatch):
    t = 0.071
    st = world(node, monkeypatch, contact_z=t - 0.003)
    assert node._touch_down(0.4, 0.0, t, t - 0.05, 'lower') is True
    zs = [z for z, _ in st['moves']]
    assert zs[0] == pytest.approx(t + 0.010) and st['moves'][0][1] == 'lower (above the spot)'                          # starts 10 mm above the computed height
    assert zs[-1] == pytest.approx(t - 0.004)                                                                           # the first 2 mm step at or below the contact height
    assert all(b - a == pytest.approx(-0.002, abs=1e-6) for a, b in zip(zs[1:-1], zs[2:])) or len(zs) <= 3              # 2 mm steps
    assert min(zs) == zs[-1]                                                                                            # and not one step further


def test_without_contact_it_stops_6_mm_under_the_computed_height_never_below_the_floor(node, monkeypatch):
    t = 0.071
    st = world(node, monkeypatch, contact_z=None)
    assert node._touch_down(0.4, 0.0, t, t - 0.05, 'lower') is True and st['moves'][-1][0] == pytest.approx(t - 0.006)
    st = world(node, monkeypatch, contact_z=None)
    assert node._touch_down(0.4, 0.0, t, t - 0.002, 'lower') is True and st['moves'][-1][0] == pytest.approx(t - 0.002)    # the safety floor wins


def test_a_failed_step_fails_the_lowering_and_a_missing_torque_signal_means_the_plain_move(node, monkeypatch):
    t = 0.071
    world(node, monkeypatch, contact_z=None, fail_at=t - 0.002)
    assert node._touch_down(0.4, 0.0, t, t - 0.05, 'lower') is False
    st = world(node, monkeypatch, contact_z=None)
    node._je = None                                                                                                     # the arm reports no efforts
    assert node._touch_down(0.4, 0.0, t, t - 0.05, 'lower') is True and st['moves'] == [(t, 'lower')]                   # exactly the old behaviour
    node.dry_run = True
    st['moves'].clear()
    try:
        assert node._touch_down(0.4, 0.0, t, t - 0.05, 'lower') is True and st['moves'] == [(t, 'lower')]
    finally:
        node.dry_run = False


def test_a_small_torque_change_is_not_a_touch_and_joint_states_efforts_are_stored(node):
    from sensor_msgs.msg import JointState
    msg = JointState()
    msg.name = ['joint_2', 'joint_4', 'joint_6']
    msg.position = [0.0, 0.0, 0.0]
    msg.effort = [3.9, -10.4, -2.2]
    node._on_joint_state(msg)
    assert node._je == {'joint_2': 3.9, 'joint_4': -10.4, 'joint_6': -2.2}
    assert acn.ArmControllerNode.TOUCH_DELTA_NM > 10 * 0.02                                                            # well above the rest noise measured on 2026-11-03 (std 0.015-0.023 Nm)


def test_a_torque_blip_that_does_not_last_is_not_a_touch(node, monkeypatch):
    # 2026-11-05: 0.36 Nm right after the first 2 mm step was settling; the mug was released 1.2 cm above the table
    t = 0.071
    st = world(node, monkeypatch, contact_z=None)
    reads = {'n': 0}

    def blip(seconds=0.25):
        reads['n'] += 1
        # the first read after the first step is high (settling), every other read is quiet
        return [b + (0.9 if reads['n'] == 2 else 0.01) for b in BASE]
    monkeypatch.setattr(node, '_effort_vector', blip)
    assert node._touch_down(0.4, 0.0, t, t - 0.05, 'lower') is True
    assert st['moves'][-1][0] == pytest.approx(t - 0.006)                                                             # went all the way down: the blip was not taken for a touch
    st = world(node, monkeypatch, contact_z=t - 0.002)
    assert node._touch_down(0.4, 0.0, t, t - 0.05, 'lower') is True and st['moves'][-1][0] == pytest.approx(t - 0.002)   # a lasting change still stops it
    assert acn.ArmControllerNode.TOUCH_DELTA_NM >= 0.5 and acn.ArmControllerNode.TOUCH_CONFIRM_S > 0
