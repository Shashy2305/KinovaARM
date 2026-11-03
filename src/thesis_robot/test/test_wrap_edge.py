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


# ---- equivalent seeds: the IK is not indifferent to how the same angle is written ----------------------------------------------------------------------------
def test_equivalent_seeds_only_appear_for_joints_3_and_7_near_the_edge():
    plain = [0.0, 0.3, 1.0, -1.0, 0.2, 0.5, 1.5]
    assert mu.equivalent_seeds(plain) == [plain]                                                                               # nothing near 180 deg: one seed, as before
    near7 = [0.0, 0.3, 1.0, -1.0, 0.2, 0.5, -2.98]
    seeds = mu.equivalent_seeds(near7)
    assert seeds[0] == near7 and len(seeds) == 2 and seeds[1][6] == pytest.approx(-2.98 + 2 * math.pi) and seeds[1][:6] == near7[:6]
    both = [0.0, 0.3, 3.0, -1.0, 0.2, 0.5, -3.0]
    seeds = mu.equivalent_seeds(both)
    assert len(seeds) == 4 and seeds[0] == both and all(abs(v[2]) <= 3.5 and abs(v[6]) <= 3.5 for v in seeds)
    assert mu.equivalent_seeds([3.0, 0, 0, 0, 3.0, 0, 0]) == [[3.0, 0, 0, 0, 3.0, 0, 0]]                                         # joints 1 and 5 keep the old limit: no variants
    out = mu.equivalent_seeds([0, 0, -3.3, 0, 0, 0, 0])
    assert len(out) == 2 and out[1][2] == pytest.approx(-3.3 + 2 * math.pi)                                                    # -3.3 is also 2.98: both writings are inside 3.5
    assert mu.equivalent_seeds([0, 0, 2.0, 0, 0, 0, 0]) == [[0, 0, 2.0, 0, 0, 0, 0]]                                           # well inside: no variant


def test_the_goal_solver_tries_the_other_writing_of_the_wrist_angle_and_finds_the_clean_solution(node, monkeypatch):
    node.dry_run, node.moveit2 = False, object()
    cur = [0.1, 0.3, 1.0, -1.0, 0.2, 0.5, -2.98]                                                                               # joint 7 right at the edge, written on the negative side
    monkeypatch.setattr(node, '_current_joint_vector', lambda: list(cur))
    asked = []
    good = [0.1, 0.3, 1.0, -1.0, 0.2, 0.5, 3.30]                                                                               # the clean answer, written on the positive side

    def ik(seed, position, quat, timeout_s=0.15):
        asked.append(round(seed[6], 2))
        return list(good) if seed[6] > 0 else [0.1, 0.3, 1.0, -1.0, 0.2, 0.5 + 3.5, 1.0]                                      # the negative writing comes back as a far-away branch
    monkeypatch.setattr(node, '_ik', ik)
    goal, _q = node._solve_goal_joints([0.4, 0.0, 0.1], (0, 0, 0, 1), False)
    assert goal[6] == pytest.approx(-2.98, abs=0.01) and max(abs(a - b) for a, b in zip(goal, cur)) < 0.1                       # unwrapped back next to where the arm is: a tiny move
    assert any(v > 0 for v in asked) and any(v < 0 for v in asked)                                                              # both writings were tried


# ---- 2026-11-03 (evening): moves across 180 deg are reported unsuccessful by the trajectory controller although the arm is at the goal -------------------------------
class Pt:
    def __init__(self, positions):
        self.positions = positions
        self.time_from_start = type('D', (), {'sec': 1, 'nanosec': 0})()


class FakeTraj:
    joint_names = list(acn.JOINT_NAMES)

    def __init__(self, goal):
        self.points = [Pt([0.0] * 7), Pt(list(goal))]


def set_state(node, q):
    node._js = dict(zip(acn.JOINT_NAMES, q))


def test_a_move_past_180_deg_counts_as_reached_when_the_joints_match_modulo_2pi(node):
    goal = [0.1, 0.3, 3.30, -1.0, 0.2, 0.5, 3.20]                                  # commanded past +pi on joints 3 and 7
    set_state(node, [0.1, 0.3, 3.30 - 2 * math.pi, -1.0, 0.2, 0.5, 3.20 - 2 * math.pi])      # the driver reports them wrapped
    assert node._reached_modulo_2pi(FakeTraj(goal)) is True
    set_state(node, [0.1, 0.3, 3.30 - 2 * math.pi + 0.02, -1.0, 0.2, 0.5, 3.20 - 2 * math.pi])
    assert node._reached_modulo_2pi(FakeTraj(goal)) is True                           # within 0.03 rad
    set_state(node, [0.1, 0.3, 3.30 - 2 * math.pi + 0.08, -1.0, 0.2, 0.5, 3.20 - 2 * math.pi])
    assert node._reached_modulo_2pi(FakeTraj(goal)) is False                          # a real shortfall is still a failure
    set_state(node, [0.1, 0.9, 3.30 - 2 * math.pi, -1.0, 0.2, 0.5, 3.20 - 2 * math.pi])
    assert node._reached_modulo_2pi(FakeTraj(goal)) is False                          # any other joint off: failure
    node._js = None
    assert node._reached_modulo_2pi(FakeTraj(goal)) is False                          # no joint state: failure


def test_guarded_move_accepts_an_unsuccessful_report_only_when_the_arm_is_at_the_goal(node, monkeypatch):
    node.dry_run = False

    class MoveIt:
        pipeline_id = planner_id = ''

        def execute(self, traj):
            pass

        def wait_until_executed(self):
            return False                                                              # the controller says: not executed
    node.moveit2 = MoveIt()
    goal = [0.1, 0.3, 3.30, -1.0, 0.2, 0.5, 1.0]
    monkeypatch.setattr(node, '_ensure_safety_scene', lambda geom: None)
    monkeypatch.setattr(node, '_plan_joint_goal', lambda g, geom, allow_long=False: (FakeTraj(g), 'ok'))
    monkeypatch.setattr(node, '_smoothed', lambda traj, label: traj)
    monkeypatch.setattr(node, '_actual_tool_quat', lambda default=None: default)
    monkeypatch.setattr(acn.time, 'sleep', lambda s: None)
    set_state(node, [0.1, 0.3, 3.30 - 2 * math.pi, -1.0, 0.2, 0.5, 1.0])
    assert node._guarded_move({'table_top_z': -0.0125}, 'swing', joint_positions=goal) is True
    set_state(node, [0.1, 0.3, 1.0, -1.0, 0.2, 0.5, 1.0])                            # stopped half way: still a failure
    assert node._guarded_move({'table_top_z': -0.0125}, 'swing', joint_positions=goal) is False


def test_go_home_aims_for_the_writing_of_the_home_angles_nearest_to_the_arm_and_is_exempt_from_the_long_turn_check(node, monkeypatch):
    seen = {}
    monkeypatch.setattr(node, '_current_joint_vector', lambda: [0.0, 0.3, -2.98, -1.0, 0.0, 0.5, 1.5])
    home = mu.unwrap_to_seed(acn.HOME_JOINTS, [0.0, 0.3, -2.98, -1.0, 0.0, 0.5, 1.5])
    assert home[2] == pytest.approx(3.14 - 2 * math.pi, abs=1e-6) and abs(home[2] - (-2.98)) < 0.2                      # -3.14, not +3.14 (6.1 rad away)
    node.dry_run, node.moveit2 = False, FakeMoveIt()
    far = [0.0, 0.3, 1.0, -0.2, 0.0, -2.0, 1.5]                                     # joints 4 and 6 far from home: go_home may turn them a long way
    monkeypatch.setattr(node, '_current_joint_vector', lambda: list(far))
    node._plan_joint_goal(list(acn.HOME_JOINTS), {}, allow_long=True)
    assert node.moveit2.calls > 0
    monkeypatch.setattr(node, '_current_joint_vector', lambda: list(far))
    traj, why = node._plan_joint_goal([0.0, 0.3, 1.0, -0.2, 0.0, 1.5, 1.5], {})       # an ordinary goal 3.5 rad away on joint 6 is refused
    assert traj is None and 'joint_6' in why
