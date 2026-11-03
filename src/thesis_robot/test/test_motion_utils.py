import math

import pytest

from thesis_robot import motion_utils as mu


def test_yaw_candidates_start_at_zero_and_cover_span():
    c = mu.yaw_candidates_deg(90, 15)
    assert c[0] == 0 and len(c) == 13 and set(c) == set(range(-90, 91, 15))
    assert [abs(x) for x in c] == sorted(abs(x) for x in c)


def test_rotate_about_tool_z_matches_scipy():
    R = pytest.importorskip('scipy.spatial.transform').Rotation
    q = (0.773, 0.635, -0.015, 0.019)
    for yaw in (-90, -30, 0, 15, 75):
        want = (R.from_quat(q) * R.from_euler('z', yaw, degrees=True)).as_quat()
        got = mu.rotate_about_tool_z(q, yaw)
        # same rotation up to quaternion sign
        dot = abs(sum(a * b for a, b in zip(got, want)))
        n = math.sqrt(sum(a * a for a in q))
        assert dot / n == pytest.approx(1.0, abs=1e-3)


def test_unwrap_picks_nearest_equivalent_for_continuous_joints_only():
    seed = [0.0, 0.0, 1.5, 0.5, -1.8, 0.8, 2.2]
    sol = [0.0, 1.0, 1.5 - 2 * math.pi, 0.5, -1.8 + 2 * math.pi, 0.8, 2.2]
    out = mu.unwrap_to_seed(sol, seed)
    assert out[2] == pytest.approx(1.5) and out[4] == pytest.approx(-1.8)
    assert out[1] == 1.0 and out[3] == 0.5          # non-continuous joints untouched


def test_unwrap_respects_planner_limit():
    # nearest equivalent (3.0) is outside +-2.88, the next one (3.0-2pi=-3.28) too
    assert mu.unwrap_to_seed([3.0, 0, 0, 0, 0, 0, 0], [2.5, 0, 0, 0, 0, 0, 0]) is None


def test_joints_3_and_7_may_sit_between_2_88_and_3_5_rad_the_others_keep_the_old_limit():
    # 2026-11-03: the side pick toward the coffee tray needs joint 3 / 7 near 180 deg; joints 1 and 5 never did
    seed = [0.0, 0.0, 3.0, 0.5, 0.0, 0.8, 3.05]
    out = mu.unwrap_to_seed([0.0, 0.0, 3.1, 0.5, 0.0, 0.8, 3.2], seed)
    assert out is not None and out[2] == pytest.approx(3.1) and out[6] == pytest.approx(3.2)            # no longer the long way round (3.1 - 2pi = -3.18)
    assert mu.unwrap_to_seed([3.0, 0, 0, 0, 0, 0, 0], [2.5, 0, 0, 0, 0, 0, 0]) is None                      # joint 1: unchanged
    assert mu.unwrap_to_seed([0, 0, 0, 0, 3.0, 0, 0], [0, 0, 0, 0, 2.5, 0, 0]) is None                      # joint 5: unchanged
    assert mu.unwrap_to_seed([0, 0, 3.7, 0, 0, 0, 0], [0, 0, 3.4, 0, 0, 0, 0])[2] == pytest.approx(3.7 - 2 * math.pi)   # beyond 3.5: the other equivalent
    assert mu.joint_limit(2) == mu.joint_limit(6) == 3.5 and mu.joint_limit(0) == mu.joint_limit(4) == mu.joint_limit(1) == mu.PLANNER_JOINT_LIMIT
    assert mu.unwrap_to_seed([0, 0, 3.1, 0, 0, 0, 0], [0, 0, 3.0, 0, 0, 0, 0], limit=2.88) is None          # an explicit limit is still uniform


def test_unwrap_removes_the_long_way_round():
    seed = [0.0, 0.0, 1.57, -2.4, -1.88, 0.8, 2.25]
    sol = [0.1, 0.2, 1.57 - 2 * math.pi, -2.3, -1.9, 0.9, 2.2]      # joint_3 returned 2pi away
    assert max(mu.joint_deltas(sol, seed)) > 6.0
    assert max(mu.joint_deltas(mu.unwrap_to_seed(sol, seed), seed)) < 0.3


def test_better_prefers_smaller_max_then_total():
    assert mu.better((1.0, 5.0), None)
    assert mu.better((0.9, 9.0), (1.0, 2.0))
    assert mu.better((1.0, 2.0), (1.0, 3.0))
    assert not mu.better((1.1, 0.1), (1.0, 9.0))
