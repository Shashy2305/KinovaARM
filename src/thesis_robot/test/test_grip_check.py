"""2026-11-05: a cup pinched by its rim or handle (finger joint 0.58-0.65) counted as held, was lifted, fell during the carry and the run went on to the far side with an empty gripper."""
import pytest

from thesis_robot import arm_controller_node as acn

from test_pick_place_flow import node, side_world, tilt_world, world  # noqa: F401


def finger(node, f):
    node._js = {'finger_joint': f} if f is not None else {}


def test_a_cup_is_held_by_its_body_only_when_the_fingers_stop_where_a_cup_body_is(node):
    for f in (0.295, 0.327, 0.2, 0.4):                                                   # the blue mug measured 0.295 / 0.327
        finger(node, f)
        assert node._grip_plausible('cup') is True and node._is_holding() is True
    for f in (0.55, 0.62, 0.645):                                                         # a rim or handle pinch: "holding" for the plain test, not a cup body
        finger(node, f)
        assert node._is_holding() is True and node._grip_plausible('cup') is False
    finger(node, 0.1)
    assert node._grip_plausible('cup') is False                                           # open too wide to have a cup between the pads
    finger(node, 0.62)
    assert node._grip_plausible('bottle') is True and node._grip_plausible(None) is True    # nothing is known about other objects: the plain test applies
    finger(node, None)
    assert node._grip_plausible('cup') is True                                            # no reading: cannot judge


def test_the_pick_grip_check_rejects_a_pinch_opens_the_gripper_and_reports_it(node, monkeypatch):
    node.dry_run, node.moveit2 = False, object()
    opened = []
    monkeypatch.setattr(node, '_open_gripper', lambda: opened.append(1) or True)
    monkeypatch.setattr(acn.time, 'sleep', lambda s: None)
    statuses = []
    monkeypatch.setattr(node, '_publish_pp_status', lambda t: statuses.append(t))
    node._grip_label = 'cup'
    finger(node, 0.31)
    assert node._check_grip('check the grip') is True and not opened
    finger(node, 0.60)
    assert node._check_grip('check the grip') is False and opened and any('not holding' in t for t in statuses)
    opened.clear()
    finger(node, 0.69)                                                                     # closed on air: the old message and behaviour
    assert node._check_grip('check the grip') is False and opened


def test_a_side_held_cup_that_slipped_is_not_carried_across_the_table(node, side_world, monkeypatch):
    from test_pick_place_flow import run_cup_pick
    assert run_cup_pick(node) is True
    held = dict(node._held)
    side_world.verified.clear()
    monkeypatch.setattr(node, '_is_holding', lambda: True)
    monkeypatch.setattr(node, '_restore_held', lambda: None)
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (__import__('numpy').eye(3), [0.30, -0.35, 0.153]))
    monkeypatch.setattr(acn.time, 'sleep', lambda s: None)
    node._held = dict(held)
    node.latest_scene = {}
    finger(node, 0.63)                                                                     # hanging by the handle
    assert node._place(0.45, 0.0) is False
    assert not any(v[3] in ('carry', 'lower') for v in side_world.verified)                # it did not carry or lower anything
    node._held = dict(held)
    finger(node, 0.31)                                                                     # a proper grip: carried as before
    assert node._place(0.45, 0.0) is True
