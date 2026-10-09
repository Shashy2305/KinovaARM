import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from thesis_robot import angled_approach as aa

GRASP = [0.773, 0.635, -0.015, 0.019]            # the controller's vertical grasp orientation (tool z pointing down)


def test_the_base_orientation_points_the_tool_down():
    z = R.from_quat(GRASP).as_matrix()[:, 2]
    assert np.allclose(z, [0, 0, -1], atol=0.06)                              # (the stored quaternion is ~3 degrees off true vertical)


def test_tool_axis_for_known_tilts():
    assert np.allclose(aa.tool_axis(0, 123), [0, 0, -1])
    a = aa.tool_axis(30, 0)
    assert np.allclose(a, [0.5, 0.0, -math.cos(math.radians(30))])
    assert np.allclose(aa.tool_axis(90, 90), [0, 1, 0], atol=1e-9)           # fully horizontal, leading toward +y
    for tilt in (10, 25, 45):
        for az in (0, 70, 180, 300):
            assert abs(np.linalg.norm(aa.tool_axis(tilt, az)) - 1) < 1e-12


@pytest.mark.parametrize('tilt', [0, 15, 30, 45])
@pytest.mark.parametrize('az', [0, 90, 180, 270])
@pytest.mark.parametrize('yaw', [0, 45, 90])
def test_tool_quat_puts_the_tool_z_axis_on_the_requested_axis(tilt, az, yaw):
    q = aa.tool_quat(GRASP, tilt, az, yaw)
    assert abs(sum(v * v for v in q) - 1.0) < 1e-9
    z = R.from_quat(q).as_matrix()[:, 2]
    # the vertical GRASP quat is ~3 degrees off true vertical, so allow that slack
    assert np.allclose(z, aa.tool_axis(tilt, az), atol=0.06)


def test_yaw_turns_the_fingers_about_the_tool_axis_only():
    q0, q1 = aa.tool_quat(GRASP, 30, 0, 0), aa.tool_quat(GRASP, 30, 0, 90)
    m0, m1 = R.from_quat(q0).as_matrix(), R.from_quat(q1).as_matrix()
    assert np.allclose(m0[:, 2], m1[:, 2], atol=1e-9)                       # same tool axis
    assert abs(float(m0[:, 0] @ m1[:, 0])) < 1e-6                           # closing axis turned by 90 degrees


def test_flange_is_behind_the_fingertips_along_the_tool_axis():
    tip = np.array([0.4, 0.1, 0.02])
    a = aa.tool_axis(30, 0)
    fl = aa.flange_for_tip(tip, a)
    assert np.allclose(aa.tip_for_flange(fl, a), tip)
    assert fl[0] < tip[0] and fl[2] > tip[2]                                  # leads toward +x, so the flange is behind it
    assert math.isclose(np.linalg.norm(fl - tip), aa.TCP_REACH_M, rel_tol=1e-9)


def test_vertical_flange_height_is_the_known_rule_and_tilting_lowers_it():
    tip = np.array([0.4, 0.0, 0.02])
    v = aa.flange_for_tip(tip, aa.tool_axis(0, 0))
    assert v[2] == pytest.approx(0.02 + aa.TCP_REACH_M)
    t = aa.flange_for_tip(tip, aa.tool_axis(45, 0))
    assert t[2] == pytest.approx(0.02 + aa.TCP_REACH_M * math.cos(math.radians(45)))
    assert t[2] < v[2]


def test_the_approach_is_a_straight_line_down_the_tool_axis():
    pre, grasp, a = aa.approach_poses([0.4, 0.1, 0.02], 30, 90, standoff=0.12)
    d = grasp - pre
    assert np.linalg.norm(d) == pytest.approx(0.12)
    assert np.allclose(d / np.linalg.norm(d), a)
    assert pre[2] > grasp[2]                                                  # the pregrasp is higher


def test_lowest_tool_z_with_the_fingers_closing_along_the_tilt_is_below_the_tip():
    a = aa.tool_axis(30, 0)
    fl = aa.flange_for_tip([0.4, 0.0, 0.05], a)
    across = aa.lowest_tool_z(fl, a, (0.0, 1.0))                              # fingers spread sideways: both tips at the tip height
    along = aa.lowest_tool_z(fl, a, (1.0, 0.0))                               # fingers spread along the lean: one tip lower
    assert across == pytest.approx(0.05, abs=1e-9)
    assert along < across


def test_tilted_from_keeps_the_wrist_alignment_and_leans_the_axis():
    cur = aa.tool_quat(GRASP, 0, 0, 40.0)                       # a vertical tool already yawed by 40 degrees
    leaned = aa.tilted_from(cur, 20, 90)
    z = R.from_quat(leaned).as_matrix()[:, 2]
    assert np.allclose(z, aa.tool_axis(20, 90), atol=0.06)
    # the closing axis (tool x) turned only as much as the lean forces: its angle to the original stays small
    x0, x1 = R.from_quat(cur).as_matrix()[:, 0], R.from_quat(leaned).as_matrix()[:, 0]
    assert float(x0 @ x1) > math.cos(math.radians(25))
