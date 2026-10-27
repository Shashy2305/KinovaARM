"""Which way to grasp: the side-grasp geometry (from the 2026-10-26 hand-guided demonstration), the clearance against neighbours, the handle rules and the ranking. Pure numpy."""
import json
import math
import os

import numpy as np
import pytest

from thesis_robot import grasp_options as go

TABLE_TOP = -0.0125
DEMO = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'docs', 'demonstrations', 'cup_side_grasp_2026-10-26', 'pose.json')


def test_the_side_orientation_at_the_demonstrated_azimuth_is_the_demonstrated_one():
    q = go.side_quat(go.DEMO_AZ_DEG)
    assert np.allclose(q, go.DEMO_QUAT_XYZW, atol=2e-3) or np.allclose(q, -np.array(go.DEMO_QUAT_XYZW), atol=2e-3)


def test_the_constants_match_the_recorded_demonstration():
    if not os.path.exists(DEMO):
        pytest.skip('demonstration file not in this checkout')
    d = json.load(open(DEMO))
    assert d['flange_quat_xyzw'] == pytest.approx(list(go.DEMO_QUAT_XYZW), abs=1e-3)
    assert d['tool_azimuth_deg'] == pytest.approx(go.DEMO_AZ_DEG, abs=0.1)
    assert d['tool_below_horizontal_deg'] == pytest.approx(90.0 - go.SIDE_TILT_DEG, abs=0.1)


def test_turning_the_azimuth_turns_the_tool_axis_and_keeps_the_fingers_closing_sideways():
    from scipy.spatial.transform import Rotation as R
    for az in (0, 45, 90, 180, 270):
        m = R.from_quat(go.side_quat(az)).as_matrix()
        axis = m[:, 2]
        assert np.allclose(axis, go.side_axis(az), atol=0.02)
        closing = m[:, 0]                                    # the demonstration closed along the tool x axis
        assert abs(closing @ axis) < 0.02 and abs(closing[2]) < 0.1     # across the approach and (almost) horizontal


def test_the_flip_is_the_same_grasp_with_the_wrist_turned_half_a_turn():
    from scipy.spatial.transform import Rotation as R
    a, b = R.from_quat(go.side_quat(90)).as_matrix(), R.from_quat(go.side_quat(90, flip=True)).as_matrix()
    assert np.allclose(a[:, 2], b[:, 2], atol=1e-6) and np.allclose(a[:, 0], -b[:, 0], atol=1e-6)


def test_the_poses_put_the_pad_midpoint_on_the_object_and_back_off_along_the_axis():
    high, pre, grasp, a = go.side_poses((0.35, 0.0), 0.07, 270.0)
    assert np.allclose(grasp + go.PAD_REACH_M * a, [0.35, 0.0, 0.07], atol=1e-9)
    assert np.allclose(pre, grasp - go.STANDOFF_M * a) and np.allclose(high - pre, [0, 0, go.HIGH_CLEAR_M])
    assert grasp[1] > 0.15 and abs(grasp[0] - 0.35) < 0.02          # the flange sits behind the object, on the side the fingers lead away from


def test_the_demonstrated_grasp_is_reproduced_within_a_centimetre():
    if not os.path.exists(DEMO):
        pytest.skip('demonstration file not in this checkout')
    d = json.load(open(DEMO))
    pad = d['pad_mid']
    _h, _p, grasp, _a = go.side_poses((pad[0], pad[1]), pad[2], d['tool_azimuth_deg'])
    assert np.linalg.norm(grasp - np.array(d['flange_pos'])) < 0.015


def test_pad_height_is_just_under_the_rim_of_a_cup_and_in_the_lower_body_of_a_bottle():
    assert go.side_pad_z('cup', TABLE_TOP) == pytest.approx(TABLE_TOP + 0.09 - 0.008)
    assert go.side_pad_z('bottle', TABLE_TOP) == pytest.approx(TABLE_TOP + 0.30 * 0.22)


def test_only_tall_enough_objects_get_a_side_grasp():
    assert go.side_eligible('cup') and go.side_eligible('mug') and go.side_eligible('bottle')
    for flat in ('mouse', 'remote', 'cell phone', 'book', 'scissors', 'bowl'):
        assert not go.side_eligible(flat)


def test_a_clear_table_has_unlimited_clearance_in_every_direction():
    for az in go.AZIMUTHS:
        assert go.side_clearance((0.4, 0.0), az, 0.07, TABLE_TOP, [])[0] == math.inf


def test_a_neighbour_behind_the_object_blocks_the_approach_from_that_side_only():
    near = [('cup', 0.40, 0.15, 0.0)]                         # +y of the target at (0.40, 0)
    c270, what = go.side_clearance((0.40, 0.0), 270.0, 0.07, TABLE_TOP, near)     # fingertips toward -y: the gripper comes from +y
    c90, _ = go.side_clearance((0.40, 0.0), 90.0, 0.07, TABLE_TOP, near)          # fingertips toward +y: the neighbour is beyond the fingertips
    assert c270 < 0.0 and 'cup' in what
    assert c90 > 0.0


def test_a_neighbour_beside_the_open_fingers_blocks_but_one_further_out_does_not():
    assert go.side_clearance((0.40, 0.0), 270.0, 0.07, TABLE_TOP, [('cup', 0.40 + 0.10, 0.0)])[0] < 0.0
    assert go.side_clearance((0.40, 0.0), 270.0, 0.07, TABLE_TOP, [('cup', 0.40 + 0.20, 0.0)])[0] > 0.0


def test_the_target_itself_and_low_objects_out_of_reach_of_the_gripper_never_count():
    assert go.side_clearance((0.40, 0.0), 270.0, 0.07, TABLE_TOP, [('cup', 0.41, 0.01)])[0] == math.inf
    # a phone lies far below the fingers and the base even right at the pads
    assert go.side_clearance((0.40, 0.0), 270.0, 0.07, TABLE_TOP, [('cell phone', 0.40, 0.10)])[0] == math.inf


def test_a_tall_object_far_behind_still_hits_the_wrist():
    tall = [('bottle', 0.40, 0.38)]
    assert go.side_clearance((0.40, 0.0), 270.0, 0.07, TABLE_TOP, tall)[0] < 0.0


def test_handle_toward_the_fingertips_is_preferred_and_a_pad_on_the_handle_is_refused():
    assert go.handle_adjust(270.0, 265.0)[0] > 0 and go.handle_adjust(270.0, 265.0)[1] is None
    assert go.handle_adjust(270.0, 0.0)[1]                                        # handle to the side: a pad lands on it
    assert go.handle_adjust(270.0, 90.0)[0] < 0 and go.handle_adjust(270.0, 90.0)[1] is None    # handle toward the arm: allowed, disliked
    assert go.handle_adjust(270.0, None) == (0.0, None)


def test_rank_side_orders_by_score_and_lists_the_rejected_ones_with_their_reason():
    out = go.rank_side('cup', (0.40, 0.0), TABLE_TOP, [('cup', 0.40, 0.15, 0.0)])
    ok = [o for o in out if 'score' in o]
    bad = [o for o in out if 'rejected' in o]
    assert [o['score'] for o in ok] == sorted((o['score'] for o in ok), reverse=True)
    assert any(o['az'] == 270.0 for o in bad) and all('hit' in o['rejected'] for o in bad if o['az'] == 270.0)
    assert 270.0 not in [o['az'] for o in ok]


def test_a_handle_removes_the_directions_that_would_put_a_pad_on_it():
    out = go.rank_side('cup', (0.40, 0.0), TABLE_TOP, [], handle_bearing_deg=270.0)
    ok = {o['az'] for o in out if 'score' in o}
    assert 270.0 in ok and 0.0 not in ok and 180.0 not in ok
    assert out[0]['az'] == 270.0                                               # fingertips toward the handle first


def test_a_cup_on_an_empty_table_is_grasped_from_the_side_first_and_a_mouse_from_above():
    cup = go.rank_methods('cup', (0.4, 0.0), TABLE_TOP, [], 0.05, False)
    assert cup[0]['method'] == 'side' and cup[0]['az'] in (90.0, 270.0)
    assert [o['method'] for o in cup if o['method'] != 'side'] == ['top']
    mouse = go.rank_methods('mouse', (0.4, 0.0), TABLE_TOP, [], 0.05, False)
    assert [o['method'] for o in mouse] == ['top']


def test_with_the_side_grasp_off_there_is_never_a_side_option_and_with_only_nothing_else():
    assert 'side' not in {o['method'] for o in go.rank_methods('cup', (0.4, 0.0), TABLE_TOP, [], 0.05, True, side_mode='off')}
    assert {o['method'] for o in go.rank_methods('cup', (0.4, 0.0), TABLE_TOP, [], 0.05, True, side_mode='only')} == {'side'}
    assert go.rank_methods('mouse', (0.4, 0.0), TABLE_TOP, [], 0.05, True, side_mode='only') == []


def test_when_the_fingers_cannot_come_down_around_it_from_above_the_side_grasp_wins_and_top_is_last():
    out = go.rank_methods('cup', (0.4, 0.0), TABLE_TOP, [('cup', 0.4, 0.14, 0.0)], None, False)
    assert out[0]['method'] == 'side' and out[-1]['method'] == 'top'
    assert out[-1]['score'] < out[0]['score']


def test_a_crowded_table_leaves_the_side_with_the_most_room():
    crowd = [('cup', 0.40, 0.20, 0.0), ('cup', 0.40 - 0.20, 0.0, 0.0)]       # behind it on +y and toward the base on -x
    out = go.rank_side('cup', (0.40, 0.0), TABLE_TOP, crowd)
    assert out and out[0]['az'] not in (270.0, 180.0)
    assert out[0]['clearance_m'] is None or out[0]['clearance_m'] > 0


def test_a_method_that_just_failed_is_tried_last_on_the_retry():
    base = go.rank_methods('cup', (0.4, 0.0), TABLE_TOP, [], 0.05, False)
    retry = go.rank_methods('cup', (0.4, 0.0), TABLE_TOP, [], 0.05, False, avoid_methods=['side'])
    assert base[0]['method'] == 'side' and retry[0]['method'] == 'top'


def test_past_outcomes_move_the_ranking():
    fails = [{'kind': 'pick', 'label': 'cup', 'method': 'side', 'ok': False, 'live': True}] * 4
    wins = [{'kind': 'pick', 'label': 'cup', 'method': 'top', 'ok': True, 'live': True}] * 4
    assert go.method_factor('cup', 'side', []) == 1.0
    assert go.method_factor('cup', 'side', fails) < 0.7 and go.method_factor('cup', 'top', wins) > 1.2
    out = go.rank_methods('cup', (0.4, 0.0), TABLE_TOP, [], 0.05, False, records=fails + wins)
    assert out[0]['method'] == 'top'
    assert go.method_factor('cup', 'side', [{'kind': 'pick', 'label': 'cup', 'method': 'side', 'ok': False, 'live': False}]) == 1.0      # dry runs do not count
