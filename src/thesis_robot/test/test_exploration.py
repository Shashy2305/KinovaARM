"""2026-11-05: randomized-parameter mode for the exploration campaign: off by default, safe ranges only, restored after every command, logged in every record."""
import json
import random

import pytest
import rclpy

from thesis_robot import arm_controller_node as acn
from thesis_robot import exploration as ex
from thesis_robot import grasp_options as go
from thesis_robot import locations as locs

from test_pick_place_flow import node, set_params, world, tilt_world, side_world, run_cup_pick  # noqa: F401


def test_every_draw_is_inside_its_range_and_every_range_inside_the_hard_limits():
    rng = random.Random(1)
    methods_seen = set()
    for _ in range(300):
        d = ex.sample(rng, path='/nonexistent')
        assert set(d) == set(ex.DEFAULT_RANGES) | {'method', 'seed'}
        methods_seen.add(d['method'])
        for name, v in d.items():
            if name in ('method', 'seed'):
                continue
            lo, hi = ex.ranges('/nonexistent')[name]
            assert lo <= v <= hi and ex.HARD_LIMITS[name][0] <= v <= ex.HARD_LIMITS[name][1]
    assert methods_seen == {'top', 'side'}                                                # 300 draws: both show up
    for name, (lo, hi) in ex.DEFAULT_RANGES.items():
        assert ex.HARD_LIMITS[name][0] <= lo < hi <= ex.HARD_LIMITS[name][1]


def test_grasp_z_offset_and_hover_above_m_are_in_the_ranges_and_hard_limits_the_safety_sweep_found():
    # scripts/dev/grasp_param_safety_sweep.py plan-checked these HARD_LIMITS against the live move_group
    # on 18 real logged pick positions (2026-11-09) -- this just guards against someone loosening them
    # without re-running that sweep.
    assert ex.HARD_LIMITS['grasp_z_offset'] == (-0.03, 0.03)
    assert ex.HARD_LIMITS['hover_above_m'] == (0.04, 0.20)


def test_a_ranges_file_can_narrow_but_never_widen_past_the_hard_limits(tmp_path):
    p = tmp_path / 'r.yaml'
    p.write_text(json.dumps({'smooth_vel_scale': [0.25, 0.26], 'place_settle_s': [0.0, 99.0], 'take_in_m': [0.02, 0.005], 'nonsense': [1, 2], 'touch_step_m': 'x'}))
    r = ex.ranges(str(p))
    assert r['smooth_vel_scale'] == (0.25, 0.26)
    assert r['place_settle_s'] == ex.HARD_LIMITS['place_settle_s']                      # asked 0-99 s: clamped to 0.3-2.0
    assert r['take_in_m'] == (0.005, 0.02)                                                # an inverted pair is put in order
    assert r['touch_step_m'] == ex.DEFAULT_RANGES['touch_step_m']                         # unusable entries fall back
    assert 'nonsense' not in r


def test_a_seed_makes_the_campaign_reproducible():
    a = [ex.sample(random.Random(7), path='/nonexistent') for _ in range(1)]
    b = [ex.sample(random.Random(7), path='/nonexistent') for _ in range(1)]
    assert a == b and a != [ex.sample(random.Random(8), path='/nonexistent')]


def set_mode(node, on, seed=0, confirm=True):
    node.set_parameters([
        rclpy.parameter.Parameter('exploration_mode', rclpy.Parameter.Type.BOOL, on),
        rclpy.parameter.Parameter('exploration_seed', rclpy.Parameter.Type.INTEGER, seed),
        rclpy.parameter.Parameter('exploration_confirm_phrase', rclpy.Parameter.Type.STRING,
                                   acn.ArmControllerNode.EXPLORE_CONFIRM_PHRASE if confirm else ''),
        # explicit, not relying on the declared default: a preceding test that used the `world` fixture
        # (sets outcome_log=False) can otherwise leak into this one -- the same ROS2 parameter-service
        # timing issue test_pick_place_flow.py's own outcome-log test works around the same way.
        rclpy.parameter.Parameter('outcome_log', rclpy.Parameter.Type.BOOL, True),
    ])


def test_exploration_mode_alone_does_nothing_without_the_confirm_phrase(node):
    set_mode(node, True, confirm=False)
    try:
        before = (node.GENTLE_VEL_SCALE, node.hover_above_m)
        node._explore_begin()
        assert node._explore_sample is None and (node.GENTLE_VEL_SCALE, node.hover_above_m) == before
        node._explore_end()
    finally:
        set_mode(node, False)
        node._explore_end()


def test_it_is_off_by_default_and_changes_nothing(node):
    assert node.get_parameter('exploration_mode').value is False
    before = (node.GENTLE_VEL_SCALE, node.PLACE_SETTLE_S, node.TOUCH_STEP_M, node.TOUCH_RELIEVE_M, node.TAKE_LIFT_M, go.SIDE_IN_EXTRA_M['cup'], locs.TAKE_IN_M)
    node._explore_begin()
    assert node._explore_sample is None and (node.GENTLE_VEL_SCALE, node.PLACE_SETTLE_S, node.TOUCH_STEP_M, node.TOUCH_RELIEVE_M, node.TAKE_LIFT_M, go.SIDE_IN_EXTRA_M['cup'], locs.TAKE_IN_M) == before
    node._explore_end()


def test_a_command_runs_with_its_own_draw_and_the_defaults_come_back_afterwards(node):
    defaults = (acn.ArmControllerNode.GENTLE_VEL_SCALE, acn.ArmControllerNode.PLACE_SETTLE_S, acn.ArmControllerNode.TOUCH_STEP_M, acn.ArmControllerNode.TOUCH_RELIEVE_M,
                acn.ArmControllerNode.TAKE_LIFT_M, go.SIDE_IN_EXTRA_M['cup'], locs.TAKE_IN_M)
    grasp_hover_defaults = (node.grasp_z_offset, node.hover_above_m)
    set_mode(node, True, seed=11)
    try:
        node._explore_begin()
        d = node._explore_sample
        assert d and node.GENTLE_VEL_SCALE == d['gentle_vel_scale'] and node.PLACE_SETTLE_S == d['place_settle_s'] and node.TOUCH_STEP_M == d['touch_step_m']
        assert node.TOUCH_RELIEVE_M == d['touch_relieve_m'] and node.TAKE_LIFT_M == d['take_lift_m'] and go.SIDE_IN_EXTRA_M['cup'] == d['cup_in_extra_m'] and locs.TAKE_IN_M == d['take_in_m']
        assert node._explore_vel == d['smooth_vel_scale']
        assert node.grasp_z_offset == d['grasp_z_offset'] and node.hover_above_m == d['hover_above_m']
        assert d['method'] in ('top', 'side') and node._explore_method_draw == d['method']
        assert d['seed'] == 11
        node._explore_end()
        assert (node.GENTLE_VEL_SCALE, node.PLACE_SETTLE_S, node.TOUCH_STEP_M, node.TOUCH_RELIEVE_M, node.TAKE_LIFT_M, go.SIDE_IN_EXTRA_M['cup'], locs.TAKE_IN_M) == defaults
        assert (node.grasp_z_offset, node.hover_above_m) == grasp_hover_defaults
        assert node._explore_sample is None and 'GENTLE_VEL_SCALE' not in node.__dict__ and node._explore_vel is None and node._explore_method_draw is None
        node._explore_begin()
        second = node._explore_sample
        node._explore_end()
        assert second != d                                                                # the next command draws again
    finally:
        set_mode(node, False)
        node._explore_end()


def test_an_exploration_side_draw_takes_the_real_side_grasp_path(node, side_world):
    """Through the real _pick_once()/choose_method() path, not a reimplementation: side_grasp is left
    at 'auto' (so this isn't just testing the existing side_grasp:=only config), but an exploration
    'side' draw still forces the same swing side_world's own side-grasp test exercises."""
    node._explore_method_draw = 'side'
    try:
        assert run_cup_pick(node) is True
        assert side_world.guarded == ['side: swing to the high pre-pose']       # the real side-swing path ran
    finally:
        node._explore_method_draw = None


def test_an_exploration_top_draw_forces_straight_down_even_with_side_grasp_auto(node, side_world):
    """side_world's cup is geometrically side-gras-able (its own test proves that); a 'top' draw must
    still force straight-down through the real path, not just when side_grasp happens to be off."""
    node._explore_method_draw = 'top'
    try:
        assert run_cup_pick(node) is True
        assert side_world.guarded == [] and side_world.ik_calls == []           # no side swing attempted
    finally:
        node._explore_method_draw = None




def test_the_executor_draws_per_command_restores_in_the_end_and_the_command_record_carries_the_draw(node, monkeypatch):
    set_mode(node, True, seed=5)
    records = []

    class Sink:
        def append(self, r):
            records.append(r)
    monkeypatch.setattr(node, '_outcomes', Sink())
    node.dry_run = True
    try:
        node.executing = True
        node._execute(json.dumps({'command': 'go home', 'plan': [{'action': 'go_home'}], 'model': 'grammar'}))
    finally:
        node.dry_run = False
        set_mode(node, False)
    cmd = [r for r in records if r.get('kind') == 'command'][0]
    assert set(cmd['explore']) == set(ex.DEFAULT_RANGES) | {'method', 'seed'}
    assert cmd['explore']['seed'] == 5 and cmd['explore']['method'] in ('top', 'side')
    assert node._explore_sample is None and node.GENTLE_VEL_SCALE == acn.ArmControllerNode.GENTLE_VEL_SCALE
