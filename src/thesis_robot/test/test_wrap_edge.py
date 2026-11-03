"""2026-11-03: joints 3 / 7 may now sit at 180 deg (limits 3.5 rad). The driver reports them wrapped to +-pi, so at the edge the reading flips sign; a plan must never turn a joint a full 2*pi because of it."""
import math

import pytest

from thesis_robot import arm_controller_node as acn
from thesis_robot import motion_utils as mu

from test_pick_place_flow import node  # noqa: F401


def fresh(node):
    node._cont_prev = {}
    return node


def test_a_continuous_joint_at_the_edge_stays_on_the_branch_it_was_on(node):
    fresh(node)
    base = [0.0, 0.3, 3.1, -1.0, 0.2, 0.5, 3.139]
    assert node._continue_branch(base)[2] == pytest.approx(3.1) and node._continue_branch(base)[6] == pytest.approx(3.139)      # first reading: as is
    flipped = [0.0, 0.3, -3.12, -1.0, 0.2, 0.5, -3.141]                                                                          # the driver wrapped joints 3 and 7
    out = node._continue_branch(flipped)
    assert out[2] == pytest.approx(-3.12 + 2 * math.pi) and out[6] == pytest.approx(-3.141 + 2 * math.pi)                          # still just past +pi, the same branch
    assert out[2] > math.pi and out[2] < 3.5 and out[6] < 3.5                                                                      # inside the new planner limit
    again = node._continue_branch(flipped)
    assert again[2] == pytest.approx(out[2])                                                                                       # and it stays there on every further reading


def test_leaving_the_edge_band_returns_the_plain_reading_and_other_joints_are_never_touched(node):
    fresh(node)
    node._continue_branch([0.0, 3.1, 3.13, 0.0, 3.1, 0.0, 0.0])
    out = node._continue_branch([0.0, -3.1, -3.14, 0.0, -3.1, 0.0, 0.0])
    assert out[2] > math.pi                                                                                                         # joint 3 continues
    assert out[1] == -3.1 and out[3] == 0.0                                                                                         # joint 2 is not continuous: its reading is never shifted
    assert out[4] == pytest.approx(-3.1 + 2 * math.pi)                                                                              # joint 5 too (it is continuous): a flip inside the band is continued
    out = node._continue_branch([0.0, 0.0, 2.0, 0.0, 0.5, 0.0, 0.0])
    assert out[2] == 2.0 and out[4] == 0.5                                                                                          # clearly away from the edge: as read


class FakeMoveIt:
    pipeline_id = planner_id = ''

    def __init__(self):
        self.calls = 0

    def plan(self, **kw):
        self.calls += 1
        return None


def test_a_plan_that_would_turn_a_joint_a_full_turn_is_refused_before_planning(node, monkeypatch):
    node.dry_run, node.moveit2 = False, FakeMoveIt()
    start = [0.1, 0.3, 3.10, -1.0, 0.2, 0.5, 3.14]
    monkeypatch.setattr(node, '_current_joint_vector', lambda: list(start))
    goal = list(start)
    goal[6] = -3.14                                                                                                                 # the same wrist angle, one turn away
    traj, why = node._plan_joint_goal(goal, {})
    assert traj is None and 'joint_7' in why and 'wrap-around' in why and node.moveit2.calls == 0
    near = list(start)
    near[6] = 3.3                                                                                                                   # a short step past 180 deg is fine
    node._plan_joint_goal(near, {})
    assert node.moveit2.calls > 0


def test_go_home_may_still_turn_joints_a_long_way(node, monkeypatch):
    node.dry_run, node.moveit2 = False, FakeMoveIt()
    monkeypatch.setattr(node, '_current_joint_vector', lambda: [-2.5, 0.3, -2.0, -1.0, -2.0, 0.5, -2.0])
    node._plan_joint_goal(list(acn.HOME_JOINTS), {})                                                                                # fixed joint angles: not refused by the long-turn check
    assert node.moveit2.calls > 0


def test_the_limits_agree_with_the_wider_joints(node):
    assert mu.joint_limit(2) > math.pi and mu.joint_limit(6) > math.pi and acn.ArmControllerNode.EDGE_BAND_RAD < 0.3
