"""Taught locations (locations.py) and the arm controller's go_to_location (2026-11-02: the coffee tray)."""
import numpy as np
import pytest
import yaml

from thesis_robot import locations as locs

TAUGHT = {'coffee_tray_grasp': {'recorded': 'x', 'pad_midpoint': [0.4413, 0.2913, 0.0898], 'approach_azimuth_deg': 79.6, 'tool_axis': [0.18, 0.98, -0.03], 'finger': 0.266, 'flange': [0.4066, 0.1033, 0.096]}}


@pytest.fixture
def files(tmp_path, monkeypatch):
    lp, op = tmp_path / 'locations.yaml', tmp_path / 'fixed_obstacles.yaml'
    lp.write_text(yaml.safe_dump(TAUGHT))
    op.write_text(yaml.safe_dump({'obstacles': [{'name': 'coffee machine', 'x': 0.485, 'y': 0.305, 'radius': 0.16, 'size_x': 0.18, 'size_y': 0.225}]}))
    monkeypatch.setattr(locs, 'LOCATIONS_FILE', str(lp))
    monkeypatch.setattr(locs, 'FIXED_OBSTACLES_FILE', str(op))
    return lp, op


def test_a_complete_location_loads_and_a_broken_one_does_not(files):
    assert locs.get('coffee_tray_grasp')['approach_azimuth_deg'] == pytest.approx(79.6)
    assert locs.get('nothing') is None
    lp, _ = files
    lp.write_text(yaml.safe_dump({'bad': {'pad_midpoint': [0.4, 'x', 0.1], 'approach_azimuth_deg': 80}, 'short': {'pad_midpoint': [0.4, 0.2], 'approach_azimuth_deg': 80}}))
    assert locs.get('bad') is None and locs.get('short') is None


def test_the_approach_pose_is_a_standoff_back_along_the_tool_axis_with_the_pads_short_of_the_object(files):
    from thesis_robot import grasp_options as go
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    assert np.linalg.norm((p['grasp'] - p['pre'])) == pytest.approx(go.STANDOFF_M, abs=1e-6)
    assert np.allclose(p['high'][:2], p['pre'][:2]) and p['high'][2] - p['pre'][2] == pytest.approx(go.HIGH_CLEAR_M)
    assert np.linalg.norm(p['pre_pad'][:2] - p['pad'][:2]) == pytest.approx(go.STANDOFF_M * np.linalg.norm(p['axis'][:2]), abs=1e-3)      # the pads stop STANDOFF short of the object


def test_a_location_can_ask_for_a_longer_standoff(files):
    loc = dict(locs.get('coffee_tray_grasp'), standoff_m=0.22)
    p = locs.approach_poses(loc)
    assert np.linalg.norm(p['grasp'] - p['pre']) == pytest.approx(0.22, abs=1e-6)
    gap, _ = locs.approach_clearance(p, locs.fixed_obstacle_rects())
    assert gap > locs.approach_clearance(locs.approach_poses(locs.get('coffee_tray_grasp')), locs.fixed_obstacle_rects())[0] + 0.04      # 5 cm further back = more room


def test_the_open_fingertips_at_the_approach_pose_are_clear_of_the_machine_front(files):
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    gap, what = locs.approach_clearance(p, locs.fixed_obstacle_rects())
    assert what == 'coffee machine' and 0.03 < gap < 0.10                  # about 4-7 cm in front of the head, as measured on 2026-11-02
    assert locs.rect_distance(0.485, 0.305, ('m', 0.395, 0.575, 0.195, 0.415)) == 0.0
    assert locs.approach_clearance(p, []) == (float('inf'), None)


# ---- the arm controller's go_to_location --------------------------------------------------------------------------------------------------------------------
from test_pick_place_flow import node, side_world, tilt_world, world  # noqa: E402,F401  (the fake world fixtures)


def test_go_to_location_swings_to_the_high_pre_pose_and_comes_down_in_front_of_the_object(node, side_world, files):
    side_world.guarded.clear()
    side_world.verified.clear()
    side_world.grippers.clear()
    assert node._go_to_location('coffee_tray_grasp') is True
    assert [g[1] for g in side_world.grippers] == ['open_gripper']                          # the gripper is opened, never closed
    assert side_world.guarded == ['go_home', 'coffee_tray_grasp: swing to the high pre-pose']       # home first (an odd parked pose has no swing), then the same route a side pick uses
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    v = side_world.verified[-1]
    assert np.allclose(v[:3], p['pre'], atol=2e-3) and v[3] == 'coffee_tray_grasp: down to the approach pose'


def test_go_to_location_refuses_without_moving_for_a_missing_location_a_held_object_or_a_close_obstacle(node, side_world, files, monkeypatch):
    lp, op = files
    side_world.guarded.clear()
    side_world.verified.clear()
    assert node._go_to_location('not_taught') is False
    monkeypatch.setattr(node, '_is_holding', lambda: True)
    assert node._go_to_location('coffee_tray_grasp') is False
    monkeypatch.setattr(node, '_is_holding', lambda: False)
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    tip = p['pre_pad'] + 0.03 * p['axis']
    op.write_text(yaml.safe_dump({'obstacles': [{'name': 'box', 'x': float(tip[0]), 'y': float(tip[1]) + 0.03, 'size_x': 0.1, 'size_y': 0.05}]}))      # a box 1.5 cm from the fingertips
    assert node._go_to_location('coffee_tray_grasp') is False
    assert node._go_to_location('coffee_tray_grasp', 'out') is False                        # only 'pre', 'in' and 'take' exist
    assert side_world.guarded == [] and side_world.verified == []


def test_go_to_location_in_a_dry_run_only_logs(node, side_world, files):
    side_world.guarded.clear()
    node.dry_run = True
    try:
        assert node._go_to_location('coffee_tray_grasp') is True
    finally:
        node.dry_run = False
    assert side_world.guarded == []


def test_stage_in_goes_straight_to_the_grasp_pose_only_from_the_approach_pose(node, side_world, files, monkeypatch):
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    side_world.verified.clear()
    side_world.guarded.clear()
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [0.0, 0.0, 0.40]))                 # the arm is somewhere else
    assert node._go_to_location('coffee_tray_grasp', 'in') is False
    assert side_world.verified == []
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [float(v) for v in p['pre']]))      # at the approach pose
    assert node._go_to_location('coffee_tray_grasp', 'in') is True
    (v,) = side_world.verified
    assert np.allclose(v[:3], p['grasp'], atol=2e-3) and v[3] == 'coffee_tray_grasp: in to the object'
    assert side_world.guarded == []                                                                                         # no swing, no home: just the straight run in


def test_stage_take_closes_lifts_draws_straight_out_and_records_what_is_held(node, side_world, files, monkeypatch):
    from thesis_robot import grasp_options as go
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    side_world.verified.clear()
    side_world.guarded.clear()
    side_world.grippers.clear()
    node._held = None
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [float(v) for v in p['grasp']]))
    monkeypatch.setattr(node, '_check_grip', lambda what: True)
    state = {'n': 0}
    monkeypatch.setattr(node, '_is_holding', lambda: state['n'] > 0 and False)           # nothing held before the close
    assert node._go_to_location('coffee_tray_grasp', 'take') is True
    assert [g[1] for g in side_world.grippers] == ['close_gripper']
    lift, out = side_world.verified
    assert lift[3].endswith(': lift') and lift[2] == pytest.approx(p['grasp'][2] + node.TAKE_LIFT_M, abs=2e-3) and np.allclose(lift[:2], p['grasp'][:2], atol=2e-3)
    assert out[3].endswith(': out of the machine') and np.allclose(out[:2], p['pre'][:2], atol=2e-3) and out[2] == pytest.approx(lift[2])      # straight back along the axis, level
    assert node._held['label'] == 'cup' and node._held['tilt'] == pytest.approx(go.SIDE_TILT_DEG) and node._held['az'] == pytest.approx(p['az'])
    assert node._held['grasp_z'] == pytest.approx(p['grasp'][2])


def test_stage_take_refuses_away_from_the_grasp_pose_or_when_a_grip_closes_on_air(node, side_world, files, monkeypatch):
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    side_world.verified.clear()
    side_world.grippers.clear()
    node._held = None
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [float(v) for v in p['pre']]))      # still at the approach pose
    assert node._go_to_location('coffee_tray_grasp', 'take') is False
    assert side_world.grippers == [] and side_world.verified == []                                                          # nothing was closed
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [float(v) for v in p['grasp']]))
    monkeypatch.setattr(node, '_check_grip', lambda what: False)                                                            # closed on air
    assert node._go_to_location('coffee_tray_grasp', 'take') is False
    assert [g[1] for g in side_world.grippers] == ['close_gripper'] and side_world.verified == [] and node._held is None   # no lift, nothing recorded
