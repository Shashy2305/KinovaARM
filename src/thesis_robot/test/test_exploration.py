"""2026-11-05: randomized-parameter mode for the exploration campaign: off by default, safe ranges only, restored after every command, logged in every record."""
import json
import random

import pytest
import rclpy

from thesis_robot import arm_controller_node as acn
from thesis_robot import exploration as ex
from thesis_robot import grasp_options as go
from thesis_robot import locations as locs

from test_pick_place_flow import node  # noqa: F401


def test_every_draw_is_inside_its_range_and_every_range_inside_the_hard_limits():
    rng = random.Random(1)
    for _ in range(300):
        d = ex.sample(rng, path='/nonexistent')
        assert set(d) == set(ex.DEFAULT_RANGES)
        for name, v in d.items():
            lo, hi = ex.ranges('/nonexistent')[name]
            assert lo <= v <= hi and ex.HARD_LIMITS[name][0] <= v <= ex.HARD_LIMITS[name][1]
    for name, (lo, hi) in ex.DEFAULT_RANGES.items():
        assert ex.HARD_LIMITS[name][0] <= lo < hi <= ex.HARD_LIMITS[name][1]


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


def set_mode(node, on, seed=0):
    node.set_parameters([rclpy.parameter.Parameter('exploration_mode', rclpy.Parameter.Type.BOOL, on), rclpy.parameter.Parameter('exploration_seed', rclpy.Parameter.Type.INTEGER, seed)])


def test_it_is_off_by_default_and_changes_nothing(node):
    assert node.get_parameter('exploration_mode').value is False
    before = (node.GENTLE_VEL_SCALE, node.PLACE_SETTLE_S, node.TOUCH_STEP_M, node.TOUCH_RELIEVE_M, node.TAKE_LIFT_M, go.SIDE_IN_EXTRA_M['cup'], locs.TAKE_IN_M)
    node._explore_begin()
    assert node._explore_sample is None and (node.GENTLE_VEL_SCALE, node.PLACE_SETTLE_S, node.TOUCH_STEP_M, node.TOUCH_RELIEVE_M, node.TAKE_LIFT_M, go.SIDE_IN_EXTRA_M['cup'], locs.TAKE_IN_M) == before
    node._explore_end()


def test_a_command_runs_with_its_own_draw_and_the_defaults_come_back_afterwards(node):
    defaults = (acn.ArmControllerNode.GENTLE_VEL_SCALE, acn.ArmControllerNode.PLACE_SETTLE_S, acn.ArmControllerNode.TOUCH_STEP_M, acn.ArmControllerNode.TOUCH_RELIEVE_M,
                acn.ArmControllerNode.TAKE_LIFT_M, go.SIDE_IN_EXTRA_M['cup'], locs.TAKE_IN_M)
    set_mode(node, True, seed=11)
    try:
        node._explore_begin()
        d = node._explore_sample
        assert d and node.GENTLE_VEL_SCALE == d['gentle_vel_scale'] and node.PLACE_SETTLE_S == d['place_settle_s'] and node.TOUCH_STEP_M == d['touch_step_m']
        assert node.TOUCH_RELIEVE_M == d['touch_relieve_m'] and node.TAKE_LIFT_M == d['take_lift_m'] and go.SIDE_IN_EXTRA_M['cup'] == d['cup_in_extra_m'] and locs.TAKE_IN_M == d['take_in_m']
        assert node._explore_vel == d['smooth_vel_scale']
        node._explore_end()
        assert (node.GENTLE_VEL_SCALE, node.PLACE_SETTLE_S, node.TOUCH_STEP_M, node.TOUCH_RELIEVE_M, node.TAKE_LIFT_M, go.SIDE_IN_EXTRA_M['cup'], locs.TAKE_IN_M) == defaults
        assert node._explore_sample is None and 'GENTLE_VEL_SCALE' not in node.__dict__ and node._explore_vel is None
        node._explore_begin()
        second = node._explore_sample
        node._explore_end()
        assert second != d                                                                # the next command draws again
    finally:
        set_mode(node, False)
        node._explore_end()


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
    assert set(cmd['explore']) == set(ex.DEFAULT_RANGES)
    assert node._explore_sample is None and node.GENTLE_VEL_SCALE == acn.ArmControllerNode.GENTLE_VEL_SCALE
