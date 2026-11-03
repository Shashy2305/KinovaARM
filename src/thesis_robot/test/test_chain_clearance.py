"""2026-11-03: a side pick checks, while it is still choosing its approach, that the arm's links clear the table on the IK branch it is about to use (three live picks were refused at execution by 1 mm)."""
import numpy as np
import pytest

from thesis_robot import arm_controller_node as acn
from thesis_robot import safety_geometry as sg

from test_pick_place_flow import TABLE, node, run_cup_pick, side_world, tilt_world, world  # noqa: F401


def positions(forearm_z):
    """{link: (x, y, z)} with every guarded link comfortably high except the forearm."""
    out = {link: (0.3, 0.0, 0.5) for link in sg.GUARDED_LINKS}
    out['forearm_link'] = (0.3, 0.0, forearm_z)
    return out


def test_the_clearance_check_passes_a_branch_with_margin_and_refuses_one_a_millimetre_over_the_limit(node, monkeypatch):
    limit = TABLE['table_top_z'] + sg.LINK_CLEARANCE_M['forearm_link']
    q = [0.0] * 7
    monkeypatch.setattr(node, '_fk_link_positions', lambda names, q_: positions(limit + 0.010))
    assert node._chain_links_clear(q, q, TABLE) == (True, 'ok')
    monkeypatch.setattr(node, '_fk_link_positions', lambda names, q_: positions(limit + 0.001))               # 1 mm over: the margin is 3 mm
    ok, why = node._chain_links_clear(q, q, TABLE)
    assert not ok and 'forearm_link' in why
    monkeypatch.setattr(node, '_fk_link_positions', lambda names, q_: positions(limit - 0.002))
    assert node._chain_links_clear(q, q, TABLE)[0] is False


def test_the_midpoint_between_two_poses_is_checked_too(node, monkeypatch):
    limit = TABLE['table_top_z'] + sg.LINK_CLEARANCE_M['forearm_link']
    seen = []

    def fk(names, q):
        seen.append(q[0])
        return positions(limit + 0.02 if abs(q[0]) < 0.01 or abs(q[0] - 1.0) < 0.01 else limit - 0.01)      # fine at both ends, low in between
    monkeypatch.setattr(node, '_fk_link_positions', fk)
    ok, _ = node._chain_links_clear([0.0] * 7, [1.0] + [0.0] * 6, TABLE)
    assert not ok and any(abs(v - 0.5) < 1e-6 for v in seen)


def test_when_forward_kinematics_is_unavailable_the_check_passes_and_the_execution_check_still_stands(node, monkeypatch):
    def broken(names, q):
        raise RuntimeError('/compute_fk is not available')
    monkeypatch.setattr(node, '_fk_link_positions', broken)
    assert node._chain_links_clear([0.0] * 7, [0.1] * 7, TABLE) == (True, 'ok')


def test_a_side_pick_asks_the_ik_again_for_a_branch_that_clears_and_gives_up_when_none_does(node, side_world, monkeypatch):
    asked = {'n': 0}
    real_ik = node._ik

    def ik(seed, pos, quat, timeout_s=0.15):
        asked['n'] += 1
        return real_ik(seed, pos, quat, timeout_s)
    monkeypatch.setattr(node, '_ik', ik)
    verdicts = iter([(False, 'forearm_link would be at z=0.067'), (False, 'forearm_link would be at z=0.067'), (True, 'ok')] + [(True, 'ok')] * 50)
    monkeypatch.setattr(node, '_chain_links_clear', lambda a, b, g: next(verdicts))
    side_world.guarded.clear()
    assert run_cup_pick(node) is True and 'side: swing to the high pre-pose' in side_world.guarded
    assert asked['n'] >= 3 + 2                                                                                   # the first pose of the chain needed three asks: two low branches, then a clear one
    node._held = None
    side_world.guarded.clear()
    monkeypatch.setattr(node, '_chain_links_clear', lambda a, b, g: (False, 'forearm_link would be at z=0.067'))
    node._pick_side_az = 90.0
    try:
        assert run_cup_pick(node) is False
    finally:
        node._pick_side_az = None
    assert 'side: swing to the high pre-pose' not in side_world.guarded                                          # no branch clears: refused BEFORE the swing, nothing moved
