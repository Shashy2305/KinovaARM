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


def test_the_pre_stage_goes_home_and_hovers_again_when_a_hover_branch_leaves_no_way_to_the_approach_pose(node, side_world, files, monkeypatch):
    # 2026-11-03: the hover came out in an odd joint branch (joints 1/3/5 at 51/116/-143 deg), the swing had no solution, the stage refused; a second go home -> hover is a fresh branch
    side_world.guarded.clear()
    side_world.verified.clear()
    real = node._solve_goal_joints
    calls = {'n': 0}

    def flaky(position, quat, allow_yaw):
        calls['n'] += 1
        if calls['n'] <= 2:                                                                 # both wrist orientations fail on the first hover
            raise RuntimeError('no collision-free IK solution for that pose')
        return real(position, quat, allow_yaw)
    monkeypatch.setattr(node, '_solve_goal_joints', flaky)
    assert node._go_to_location('coffee_tray_grasp') is True
    assert side_world.guarded.count('go_home') == 2 and side_world.guarded[-1] == 'coffee_tray_grasp: swing to the high pre-pose'


def test_the_pre_stage_still_refuses_after_its_tries_and_does_not_swing(node, side_world, files, monkeypatch):
    side_world.guarded.clear()
    side_world.verified.clear()

    def never(position, quat, allow_yaw):
        raise RuntimeError('no collision-free IK solution for that pose')
    monkeypatch.setattr(node, '_solve_goal_joints', never)
    assert node._go_to_location('coffee_tray_grasp') is False
    assert side_world.guarded.count('go_home') == node.PRE_HOVER_TRIES and not any('swing' in g for g in side_world.guarded)


def test_a_swing_the_safety_check_refuses_without_moving_is_tried_again_from_home_on_a_new_branch(node, side_world, files, monkeypatch):
    # 2026-11-03: forearm_link would have been at z=0.059 (limit 0.068) on one IK branch; the refusal comes before any motion, so another go home -> hover -> swing is safe
    side_world.guarded.clear()
    side_world.verified.clear()
    real = node._guarded_move
    state = {'swings': 0}
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [0.0, 0.0, 0.5]))        # the arm does not move

    def guarded(geom, label, **kw):
        if 'swing' in label:
            state['swings'] += 1
            if state['swings'] == 1:
                return False
        return real(geom, label, **kw)
    monkeypatch.setattr(node, '_guarded_move', guarded)
    assert node._go_to_location('coffee_tray_grasp') is True
    assert state['swings'] == 2 and side_world.guarded.count('go_home') == 2
    # a branch that never works: the stage gives up
    monkeypatch.setattr(node, '_guarded_move', lambda geom, label, **kw: False if 'swing' in label else real(geom, label, **kw))
    assert node._go_to_location('coffee_tray_grasp') is False


def test_a_swing_that_started_moving_and_failed_is_never_repeated(node, side_world, files, monkeypatch):
    side_world.guarded.clear()
    real = node._guarded_move
    pose = {'p': [0.0, 0.0, 0.5]}
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), list(pose['p'])))
    n = {'swings': 0}

    def guarded(geom, label, **kw):
        if 'swing' in label:
            n['swings'] += 1
            pose['p'] = [0.2, 0.2, 0.4]                                                      # it moved, then failed
            return False
        return real(geom, label, **kw)
    monkeypatch.setattr(node, '_guarded_move', guarded)
    assert node._go_to_location('coffee_tray_grasp') is False and n['swings'] == 1


def test_the_hover_move_is_asked_again_only_when_it_was_refused_without_moving(node, monkeypatch):
    pose = {'p': [0.0, 0.0, 0.5]}
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), list(pose['p'])))
    results = iter([False, False, True])                                                    # refused twice (2.7 rad > 2.6), then it works
    monkeypatch.setattr(node, '_move_to', lambda x, y, z, **kw: next(results))
    assert node._hover_over(0.44, -0.11, 0.38) is True
    monkeypatch.setattr(node, '_move_to', lambda x, y, z, **kw: False)
    assert node._hover_over(0.44, -0.11, 0.38) is False                                     # refused every time: gives up after HOVER_MOVE_TRIES
    calls = {'n': 0}

    def moved_then_failed(x, y, z, **kw):
        calls['n'] += 1
        pose['p'] = [0.1, 0.1, 0.45]                                                        # the arm moved before the failure: never repeat it
        return False
    monkeypatch.setattr(node, '_move_to', moved_then_failed)
    assert node._hover_over(0.44, -0.11, 0.38) is False and calls['n'] == 1


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
    assert np.allclose(v[:3], locs.take_pose(locs.get('coffee_tray_grasp'), p), atol=2e-3) and v[3] == 'coffee_tray_grasp: in to the object'
    assert side_world.guarded == []                                                                                         # no swing, no home: just the straight run in


def test_stage_take_closes_lifts_draws_straight_out_and_records_what_is_held(node, side_world, files, monkeypatch):
    from thesis_robot import grasp_options as go
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    t = locs.take_pose(locs.get('coffee_tray_grasp'), p)
    side_world.verified.clear()
    side_world.guarded.clear()
    side_world.grippers.clear()
    node._held = None
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [float(v) for v in t]))
    monkeypatch.setattr(node, '_check_grip', lambda what: True)
    state = {'n': 0}
    monkeypatch.setattr(node, '_is_holding', lambda: state['n'] > 0 and False)           # nothing held before the close
    assert node._go_to_location('coffee_tray_grasp', 'take') is True
    assert [g[1] for g in side_world.grippers] == ['close_gripper']
    lift, out = side_world.verified
    assert lift[3].endswith(': lift') and lift[2] == pytest.approx(t[2] + node.TAKE_LIFT_M, abs=2e-3) and np.allclose(lift[:2], t[:2], atol=2e-3)
    assert out[3].endswith(': out of the machine') and np.allclose(out[:2], p['pre'][:2], atol=2e-3) and out[2] == pytest.approx(lift[2])      # straight back along the axis, level
    assert node._held['label'] == 'cup' and node._held['tilt'] == pytest.approx(go.SIDE_TILT_DEG) and node._held['az'] == pytest.approx(p['az'])
    off = locs.surface_offset(locs.get('coffee_tray_grasp'), -0.0125)
    assert off == pytest.approx(0.0203, abs=2e-3)                                         # the tray surface is ~2 cm above the table
    assert node._held['grasp_z'] == pytest.approx(t[2] - off)                    # recorded for the cup standing on the TABLE, so the place does not release it 2 cm up


def test_the_take_grips_the_cup_lower_than_the_taught_pose_and_the_put_keeps_the_taught_height(node, side_world, files, monkeypatch):
    # 2026-11-03: the pick and the put hold the cup with the pads ~3 mm under the rim; closing at that height missed the cup (fingers above the rim, grip closed on air)
    loc = locs.get('coffee_tray_grasp')
    p = locs.approach_poses(loc)
    t = locs.take_pose(loc, p)
    assert t[2] == pytest.approx(p['grasp'][2] - locs.TAKE_DEPTH_M)
    ax = np.array([p['axis'][0], p['axis'][1]]) / np.linalg.norm(p['axis'][:2])
    assert np.allclose(t[:2] - p['grasp'][:2], locs.TAKE_IN_M * ax, atol=1e-9)                                  # and 1.2 cm further in along the tool axis (the take closed on the near edge of the cup)
    assert locs.take_pose(dict(loc, take_depth_m=0.035), p)[2] == pytest.approx(p['grasp'][2] - 0.035)       # per-location override
    assert locs.take_pose(dict(loc, take_depth_m=0.5), p)[2] == pytest.approx(p['grasp'][2] - 0.05)          # capped: never far below the taught pose
    assert locs.take_pose(dict(loc, take_depth_m=-1.0), p)[2] == pytest.approx(p['grasp'][2])                # never above it
    assert np.allclose(locs.take_pose(dict(loc, take_in_m=0.0), p)[:2], p['grasp'][:2])
    assert np.linalg.norm(locs.take_pose(dict(loc, take_in_m=0.5), p)[:2] - p['grasp'][:2]) == pytest.approx(0.03)   # capped at 3 cm


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


# ---- 2026-11-02: the place retreat after a side hold, the withdraw action, and the wrist angle recorded by take ------------------------------------------------
def _side_place(node, side_world, monkeypatch, fail_labels):
    """A cup held from the side is put down at (0.40, -0.1); `fail_labels(label)` says which verified moves have no solution."""
    from test_pick_place_flow import run_cup_pick
    assert run_cup_pick(node) is True
    held = dict(node._held)
    side_world.verified.clear()
    names = []
    real = node._move_verified

    def mv(x, y, z, label, **kw):
        names.append(label)
        if fail_labels(label):
            return False
        return real(x, y, z, label, **kw)
    monkeypatch.setattr(node, '_move_verified', mv)
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [0.40, 0.2, side_world.z_flange]))
    monkeypatch.setattr(node, '_is_holding', lambda: True)
    node._held = held
    node.latest_scene = {}
    return names


def test_the_retreat_after_a_side_place_tries_shorter_runs_when_the_full_one_has_no_solution(node, side_world, monkeypatch):
    names = _side_place(node, side_world, monkeypatch, lambda lab: lab.startswith('retreat (17'))
    assert node._place(0.40, -0.1) is True
    retreats = [n for n in names if n.startswith('retreat (')]
    assert retreats == ['retreat (17 cm back along the tool axis)', 'retreat (12 cm back along the tool axis)']      # the full run failed, the 12 cm one worked
    assert 'rise after the retreat' in names and 'rise (no way back along the axis)' not in names


def test_when_no_retreat_has_a_solution_the_open_gripper_is_lifted_straight_up(node, side_world, monkeypatch):
    names = _side_place(node, side_world, monkeypatch, lambda lab: lab.startswith('retreat ('))
    assert node._place(0.40, -0.1) is True
    assert names[-1] == 'rise (no way back along the axis)'


def test_withdraw_backs_out_along_the_tool_axis_then_up_and_never_while_holding(node, side_world, files, monkeypatch):
    side_world.verified.clear()
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [0.466, -0.288, 0.111]))
    monkeypatch.setattr(node, '_tool_axis_now', lambda: np.array([0.18, 0.98, -0.09]))
    monkeypatch.setattr(node, '_is_holding', lambda: True)
    assert node._withdraw() is False and side_world.verified == []
    monkeypatch.setattr(node, '_is_holding', lambda: False)
    assert node._withdraw() is True
    back, up = side_world.verified
    assert back[3].startswith('withdraw (21 cm') and back[1] == pytest.approx(-0.288 - 0.21 * 0.98, abs=2e-3) and back[2] == pytest.approx(0.111 + 0.21 * 0.09, abs=2e-3)      # straight back along the axis (toward -y), not up yet
    assert up[3] == 'withdraw: rise' and up[2] == pytest.approx(back[2] + node.WITHDRAW_RISE_M) and np.allclose(up[:2], back[:2])


def test_take_records_the_wrist_angle_after_the_draw_out(node, side_world, files, monkeypatch):
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    node._held = None
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [float(v) for v in locs.take_pose(locs.get('coffee_tray_grasp'), p)]))
    monkeypatch.setattr(node, '_check_grip', lambda what: True)
    angles = iter([1.47, 1.22])                                                            # joint 7 moves while the tool orientation is kept
    monkeypatch.setattr(node, '_wrist_angle_now', lambda: next(angles))
    assert node._go_to_location('coffee_tray_grasp', 'take') is True
    assert node._held['j7'] == pytest.approx(1.22)                                          # the later reading: the place check compares against where the carry starts


# ---- the four coffee pieces (2026-11-02): pick at one azimuth, put in the tray, wait, the "give me coffee" plan ---------------------------------------------------------
from thesis_robot import command_grammar as cg  # noqa: E402
from thesis_robot import grasp_options as go_  # noqa: E402

COFFEE_SCENE = {'cup_03': {'label': 'cup', 'x': 0.452, 'y': 0.01, 'z': 0.05, 'confidence': 0.9, 'reachable': True, 'stale': False, 'color': 'blue'},
                'mouse_03': {'label': 'mouse', 'x': 0.31, 'y': -0.11, 'z': 0.02, 'confidence': 0.9, 'reachable': True, 'stale': False}}


@pytest.fixture
def coffee_files(files, tmp_path, monkeypatch):
    cp = tmp_path / 'coffee.yaml'
    cp.write_text(yaml.safe_dump({'serve_xy': [0.45, 0.0], 'wait_s': 15, 'pick_az': 90, 'tray_location': 'coffee_tray_grasp'}))
    monkeypatch.setattr(locs, 'COFFEE_FILE', str(cp))
    return cp


def test_give_me_coffee_is_recognised_in_its_usual_forms():
    for text in ('give me coffee', 'Can you give me a coffee please', 'make me a cup of coffee', 'i want coffee', 'coffee', 'get me some coffee'):
        assert cg.parse_any(text)['kind'] == 'coffee', text
    assert cg.parse('pick up the cup')['kind'] == 'pick'


def test_the_coffee_plan_is_pick_put_wait_in_take_place_with_the_configured_spot_and_wait(coffee_files):
    intent = cg.parse('give me coffee')
    plan = cg.build_plan(intent, COFFEE_SCENE)
    assert [(s['action'], s.get('stage')) for s in plan] == [('go_home', None), ('pick', None), ('go_to_location', 'put'), ('wait', None), ('go_to_location', 'in'), ('go_to_location', 'take'), ('place', None), ('go_home', None)]
    assert plan[1]['object_id'] == 'cup_03' and plan[1]['side_az'] == 90.0
    assert plan[3]['seconds'] == 15.0 and (plan[6]['x'], plan[6]['y']) == (0.45, 0.0)
    assert cg.check(intent, plan, COFFEE_SCENE) == (True, '') and cg.no_reachable_reason(intent, COFFEE_SCENE) is None
    assert cg.check(intent, plan[:3], COFFEE_SCENE)[0] is False


def test_the_coffee_command_is_refused_before_anything_moves_without_a_cup_at_the_spot_or_a_taught_tray(coffee_files, tmp_path, monkeypatch):
    intent = cg.parse('give me coffee')
    no_cup = {k: v for k, v in COFFEE_SCENE.items() if k != 'cup_03'}
    assert 'no cup at the serving spot' in cg.no_reachable_reason(intent, no_cup)
    far = dict(COFFEE_SCENE, cup_03=dict(COFFEE_SCENE['cup_03'], x=0.30, y=-0.30))
    assert 'no cup at the serving spot' in cg.no_reachable_reason(intent, far)
    monkeypatch.setattr(locs, 'LOCATIONS_FILE', str(tmp_path / 'missing.yaml'))
    assert 'has not been taught' in cg.no_reachable_reason(intent, COFFEE_SCENE)


def test_the_coffee_settings_have_safe_limits_and_defaults(tmp_path):
    p = tmp_path / 'c.yaml'
    p.write_text(yaml.safe_dump({'wait_s': 9999, 'serve_xy': [0.4, 0.1]}))
    c = locs.coffee_config(str(p))
    assert c['wait_s'] == 120.0 and c['serve_xy'] == [0.4, 0.1] and c['pick_az'] == 90.0
    p.write_text('wait_s: oops')
    assert locs.coffee_config(str(p)) == locs.COFFEE_DEFAULTS
    assert locs.coffee_config(str(tmp_path / 'nope.yaml')) == locs.COFFEE_DEFAULTS


def test_a_side_pick_can_be_limited_to_one_azimuth_and_nothing_else():
    ranked = go_.rank_methods('cup', (0.45, 0.0), -0.0125, [], 0.05, False, side_mode='auto', only_az=90.0)
    assert [(o['method'], o['az']) for o in ranked] == [('side', 90.0)]
    blocked = go_.rank_methods('cup', (0.45, 0.0), -0.0125, [('mouse', 0.45, 0.13)], 0.05, False, side_mode='auto', only_az=90.0)         # a neighbour where the gripper would come in
    assert blocked == [] or blocked[0]['az'] == 90.0
    assert go_.rank_methods('mouse', (0.45, 0.0), -0.0125, [], 0.05, False, only_az=90.0) == []                                          # a flat object has no side grasp


def test_the_pick_in_the_coffee_plan_is_not_checked_against_the_final_place(node):
    plan = cg.build_plan(cg.parse('give me coffee'), COFFEE_SCENE) if False else [
        {'action': 'pick', 'object_id': 'cup_03'}, {'action': 'go_to_location', 'name': 't', 'stage': 'put'}, {'action': 'wait', 'seconds': 15}, {'action': 'place', 'x': 0.45, 'y': 0.0}]
    assert acn_place_after(plan, 0) is None                                               # the cup goes into the tray first: the later place is not this pick's carry


def acn_place_after(plan, i):
    from thesis_robot import arm_controller_node as acn
    return acn.ArmControllerNode._place_after(plan, i)


def test_wait_waits_and_refuses_a_bad_number(node, side_world, monkeypatch):
    import time as _t
    t0 = _t.monotonic()
    assert node._wait(0.6) is True and _t.monotonic() - t0 >= 0.55
    assert node._wait('soon') is False
    assert node._dispatch({'action': 'wait', 'seconds': 0}) is True


def _held_cup(node, az=90.0, tilt=84.7):
    node._held = {'label': 'cup', 'object_id': 'c', 'grasp_z': 0.107, 'tilt': tilt, 'az': az, 'j7': 0.0}


def test_put_carries_in_lowers_releases_and_backs_out_to_the_approach_pose(node, side_world, coffee_files, monkeypatch):
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    side_world.verified.clear()
    side_world.grippers.clear()
    _held_cup(node)
    monkeypatch.setattr(node, '_is_holding', lambda: True)
    monkeypatch.setattr(node, '_restore_held', lambda: None)
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [0.45, -0.185, 0.14]))                 # the arm after the pick, cup held toward 90
    monkeypatch.setattr(node, '_tray_carry_blocked', lambda a, b, h, held: None)
    assert node._go_to_location('coffee_tray_grasp', 'put') is True
    names = [v[3].split(': ', 1)[1] for v in side_world.verified]
    assert names == ['lift to carry height', 'carry to above the approach pose', 'down to the approach pose', 'in to the tray', 'lower onto the tray', 'back out to the approach pose']
    lift, carry, down, inn, lower, out = side_world.verified
    assert carry[2] == pytest.approx(0.107 + node.CARRY_CLEARANCE_M) and np.allclose(carry[:2], p['pre'][:2], atol=2e-3)
    assert inn[2] == pytest.approx(p['grasp'][2] + node.TAKE_LIFT_M, abs=2e-3) and np.allclose(inn[:2], p['grasp'][:2], atol=2e-3)
    assert lower[2] == pytest.approx(p['grasp'][2], abs=2e-3) and np.allclose(out[:3], p['pre'], atol=2e-3)
    assert [g[1] for g in side_world.grippers] == ['open_gripper'] and node._held is None                                      # released, the held record cleared


def test_put_refuses_before_moving_for_nothing_held_a_top_held_cup_or_the_wrong_azimuth(node, side_world, coffee_files, monkeypatch):
    side_world.verified.clear()
    monkeypatch.setattr(node, '_restore_held', lambda: None)
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [0.45, -0.185, 0.14]))
    monkeypatch.setattr(node, '_is_holding', lambda: False)
    node._held = None
    assert node._go_to_location('coffee_tray_grasp', 'put') is False
    monkeypatch.setattr(node, '_is_holding', lambda: True)
    _held_cup(node, tilt=0.0)
    assert node._go_to_location('coffee_tray_grasp', 'put') is False                         # held from above: only a side-held cup can go into the tray
    _held_cup(node, az=45.0)
    assert node._go_to_location('coffee_tray_grasp', 'put') is False                         # held toward 45, the tray is entered toward 90
    assert side_world.verified == []


def test_a_failed_step_of_the_put_leaves_the_cup_held_and_never_opens_the_gripper(node, side_world, coffee_files, monkeypatch):
    side_world.verified.clear()
    side_world.grippers.clear()
    _held_cup(node)
    monkeypatch.setattr(node, '_is_holding', lambda: True)
    monkeypatch.setattr(node, '_restore_held', lambda: None)
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [0.45, -0.185, 0.14]))
    monkeypatch.setattr(node, '_tray_carry_blocked', lambda a, b, h, held: None)
    real = node._move_verified
    monkeypatch.setattr(node, '_move_verified', lambda x, y, z, label, **kw: False if 'in to the tray' in label else real(x, y, z, label, **kw))
    assert node._go_to_location('coffee_tray_grasp', 'put') is False
    assert side_world.grippers == [] and node._held is not None


def test_a_pick_with_side_az_uses_that_side_grasp_only_and_never_falls_back_to_the_top(node, side_world, monkeypatch):
    from test_pick_place_flow import run_cup_pick
    node._pick_side_az = 90.0
    try:
        assert run_cup_pick(node) is True
        assert node._held['az'] == pytest.approx(90.0) and side_world.guarded == ['side: swing to the high pre-pose']
        # now no side grasp has a solution: the pick must fail, not come down from above
        side_world.guarded.clear()
        side_world.verified.clear()
        node._held = None
        monkeypatch.setattr(node, '_ik', lambda seed, pos, quat, timeout_s=0.15: None)
        assert run_cup_pick(node) is False
        assert side_world.guarded == [] and node._held is None
    finally:
        node._pick_side_az = None


def test_the_whole_coffee_plan_runs_through_the_executor_in_a_dry_run(node, coffee_files, monkeypatch):
    import json
    node.dry_run = True
    node.latest_scene = dict(COFFEE_SCENE)
    node._scene_time = __import__('time').monotonic()
    steps_seen = []
    real = node._dispatch
    monkeypatch.setattr(node, '_dispatch', lambda step: (steps_seen.append((step['action'], step.get('stage'))) or real(step)))
    try:
        plan = cg.build_plan(cg.parse('give me coffee'), COFFEE_SCENE)
        node.executing = True
        node._execute(json.dumps({'command': 'give me coffee', 'plan': plan, 'model': 'grammar'}))
    finally:
        node.dry_run = False
    assert steps_seen == [('go_home', None), ('pick', None), ('go_to_location', 'put'), ('wait', None), ('go_to_location', 'in'), ('go_to_location', 'take'), ('place', None), ('go_home', None)]


def test_the_carry_to_the_tray_ignores_the_carried_cup_and_the_machine_but_not_a_mouse_on_the_way(node, side_world, monkeypatch):
    held = {'label': 'cup', 'object_id': 'cup_03'}
    monkeypatch.setattr(node, '_pad_midpoint_xy', lambda default: (0.44, -0.05))                     # the pads (and the cup in them) are here
    node.latest_scene = {
        'cup_03': {'label': 'cup', 'x': 0.44, 'y': -0.03, 'z': 0.06, 'confidence': 0.95, 'reachable': True, 'stale': False},               # the carried cup's own (old) entry
        'cup_09': {'label': 'cup', 'x': 0.45, 'y': -0.04, 'z': 0.06, 'confidence': 0.9, 'reachable': True, 'stale': False},                # ... split into a second track
        'fixed_00': {'label': 'coffee machine', 'x': 0.485, 'y': 0.305, 'z': 0.31, 'confidence': 1.0, 'reachable': False, 'stale': False},
    }
    node._scene_time = __import__('time').monotonic()
    assert node._tray_carry_blocked((0.44, -0.19), (0.44, 0.05), 0.0, held) is None
    node.latest_scene['bottle_07'] = {'label': 'bottle', 'x': 0.55, 'y': 0.0, 'z': 0.20, 'confidence': 0.95, 'reachable': True, 'stale': False}          # a tall thing 11 cm beside the path
    assert node._tray_carry_blocked((0.44, -0.19), (0.44, 0.05), 0.0, held) == 'bottle'


def test_the_surface_offset_is_the_taught_pad_height_over_the_table_pad_height_never_negative(files):
    loc = locs.get('coffee_tray_grasp')
    assert locs.surface_offset(dict(loc, surface_z=0.03), -0.0125) == 0.03                 # stated explicitly
    assert locs.surface_offset(dict(loc, surface_z='x'), -0.0125) == pytest.approx(locs.surface_offset(loc, -0.0125))
    low = dict(loc, pad_midpoint=[0.4, 0.2, 0.01])                                        # taught lower than a table pick: no negative offset
    assert locs.surface_offset(low, -0.0125) == 0.0


def test_a_cup_taken_from_the_tray_is_set_down_at_table_height(node, side_world, files, monkeypatch):
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    node._held = None
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [float(v) for v in locs.take_pose(locs.get('coffee_tray_grasp'), p)]))
    monkeypatch.setattr(node, '_check_grip', lambda what: True)
    assert node._go_to_location('coffee_tray_grasp', 'take') is True
    held = dict(node._held)
    side_world.verified.clear()
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [float(v) for v in p['pre']]))
    monkeypatch.setattr(node, '_is_holding', lambda: True)
    monkeypatch.setattr(node, '_restore_held', lambda: None)
    node._held = held
    node.latest_scene = {}
    assert node._place(0.45, 0.0) is True
    lower = [v for v in side_world.verified if v[3] == 'lower'][0]
    assert lower[2] == pytest.approx(held['grasp_z'] + node.SET_DOWN_GAP_M, abs=2e-3)
    assert lower[2] < p['grasp'][2] - 0.015                                               # about 2 cm lower than it would have been (0.111): at the table, not above it


def test_the_model_padding_cleaners_would_strip_the_coffee_routines_home_steps_so_the_planner_skips_them_for_grammar_plans():
    from thesis_robot import llm_planner_node as lp
    plan = cg.build_plan(cg.parse('give me coffee'), COFFEE_SCENE)
    assert plan[0]['action'] == 'go_home' and plan[-1]['action'] == 'go_home'
    stripped = lp.trim_pick_extras(lp.drop_home_around_pick(list(plan), 'give me coffee'), 'give me coffee')
    assert [s['action'] for s in stripped].count('go_home') < 2                      # why the planner must not run these on a plan the grammar built itself
    src = open(lp.__file__).read()
    assert 'if not grammar_planned:\n                plan = drop_home_around_pick' in src and 'if not grammar_planned:\n                plan = trim_pick_extras' in src


# ---- 2026-11-03: deeper grip, gentle set-down -----------------------------------------------------------------------------------------------------------------
def _spy_moves(node, monkeypatch):
    """Record (label, speed scale) of every verified move and of the settle pause, then run the fake move."""
    seen = {'moves': [], 'settles': 0}
    real = node._move_verified

    def spy(x, y, z, label, **kw):
        seen['moves'].append((label, node._gentle_scale, (x, y, z)))
        return real(x, y, z, label, **kw)
    monkeypatch.setattr(node, '_move_verified', spy)
    monkeypatch.setattr(node, '_settle', lambda: seen.__setitem__('settles', seen['settles'] + 1))
    return seen


def test_the_set_down_is_slow_and_pauses_after_the_release_then_the_speed_is_back_to_normal(node, side_world, files, monkeypatch):
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    node._held = None
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [float(v) for v in locs.take_pose(locs.get('coffee_tray_grasp'), p)]))
    monkeypatch.setattr(node, '_check_grip', lambda what: True)
    assert node._go_to_location('coffee_tray_grasp', 'take') is True
    held = dict(node._held)
    seen = _spy_moves(node, monkeypatch)
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [float(v) for v in p['pre']]))
    monkeypatch.setattr(node, '_is_holding', lambda: True)
    monkeypatch.setattr(node, '_restore_held', lambda: None)
    node._held = held
    node.latest_scene = {}
    assert node._place(0.45, 0.0) is True
    slow = {label: scale for label, scale, _ in seen['moves']}
    assert slow['lower'] == pytest.approx(node.GENTLE_VEL_SCALE) and node.GENTLE_VEL_SCALE < 0.5
    assert [scale for label, scale, _ in seen['moves'] if label.startswith('retreat')][0] == pytest.approx(node.GENTLE_VEL_SCALE)
    assert slow['carry'] == 1.0                                                              # the carry itself is not slowed
    assert seen['settles'] == 1 and node._gentle_scale == 1.0                                # one pause after the fingers open; normal speed again afterwards


def test_a_cup_held_deeper_than_its_centre_is_set_down_with_the_pads_that_far_short_of_the_spot(node, side_world, files, monkeypatch):
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    node._held = None
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [float(v) for v in locs.take_pose(locs.get('coffee_tray_grasp'), p)]))
    monkeypatch.setattr(node, '_check_grip', lambda what: True)
    assert node._go_to_location('coffee_tray_grasp', 'take') is True
    held = dict(node._held)
    assert held['in_extra'] == pytest.approx(locs.TAKE_IN_M, abs=1e-6)                        # the take recorded how much deeper than the taught pad midpoint it closed
    seen = _spy_moves(node, monkeypatch)
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [float(v) for v in p['pre']]))
    monkeypatch.setattr(node, '_is_holding', lambda: True)
    monkeypatch.setattr(node, '_restore_held', lambda: None)
    node._held = dict(held, in_extra=0.0)
    node.latest_scene = {}
    assert node._place(0.45, 0.0) is True
    plain = [xyz for label, _s, xyz in seen['moves'] if label == 'lower'][0]
    seen['moves'].clear()
    node._held = dict(held, in_extra=0.012)
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [float(v) for v in p['pre']]))
    assert node._place(0.45, 0.0) is True
    deeper = [xyz for label, _s, xyz in seen['moves'] if label == 'lower'][0]
    from thesis_robot import grasp_options as go
    a = go.side_axis(held['az'])
    assert (deeper[0] - plain[0], deeper[1] - plain[1]) == pytest.approx((-0.012 * a[0], -0.012 * a[1]), abs=1e-6)     # its centre lands on the spot: the pads stop 1.2 cm short


def test_the_tray_put_shifts_the_pads_back_by_how_deep_the_cup_was_gripped(node, side_world, coffee_files, monkeypatch):
    from thesis_robot import grasp_options as go
    p = locs.approach_poses(locs.get('coffee_tray_grasp'))
    tilt = go.SIDE_TILT_DEG
    seen = _spy_moves(node, monkeypatch)
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [0.45, -0.185, 0.14]))
    monkeypatch.setattr(node, '_is_holding', lambda: True)
    monkeypatch.setattr(node, '_restore_held', lambda: None)
    monkeypatch.setattr(node, '_tray_carry_blocked', lambda a, b, h, held: None)
    ends = []
    for extra in (0.0, 0.012):
        seen['moves'].clear()
        node._held = {'label': 'cup', 'object_id': 'c', 'grasp_z': 0.107, 'tilt': tilt, 'az': p['az'], 'j7': 0.0, 'in_extra': extra}
        assert node._go_to_location('coffee_tray_grasp', 'put') is True
        ends.append([xyz for label, _s, xyz in seen['moves'] if label.endswith('lower onto the tray')][0])
    a = np.array(p['axis'][:2]) / np.linalg.norm(p['axis'][:2])
    assert np.allclose(np.array(ends[1][:2]) - np.array(ends[0][:2]), -0.012 * a, atol=1e-6)
    assert np.allclose(ends[0][:2], p['grasp'][:2], atol=1e-6)                               # a cup gripped at its centre goes to the taught pose unchanged
