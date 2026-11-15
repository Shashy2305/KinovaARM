"""2026-11-13: the RL-project handoff schema fields on every new outcome record (session_id,
schema_version, grip_check_version, mode, params_source, policy_id, scene_id, pool, step_index,
pre_action_features, failure_category) -- the node-level half; the pure-Python half (pool_for_scene,
categorize_failure, Attempt.finish()) is covered in test_outcome_log.py."""
import json

import rclpy

from thesis_robot import outcome_log as ol

from test_pick_place_flow import node, world  # noqa: F401


def test_scene_id_and_policy_id_default_empty(node):
    node.set_parameters([rclpy.parameter.Parameter('scene_id', rclpy.Parameter.Type.STRING, ''),
                          rclpy.parameter.Parameter('policy_id', rclpy.Parameter.Type.STRING, '')])
    assert node.get_parameter('scene_id').value == '' and node.get_parameter('policy_id').value == ''
    fields = node._schema_fields()
    assert fields['scene_id'] == '' and fields['policy_id'] == '' and fields['pool'] == ''
    assert fields['mode'] == 'default' and fields['params_source'] == 'heuristic'


def test_schema_fields_follow_scene_id_and_policy_id(node):
    node.set_parameters([rclpy.parameter.Parameter('scene_id', rclpy.Parameter.Type.STRING, 'layout_3'),
                          rclpy.parameter.Parameter('policy_id', rclpy.Parameter.Type.STRING, 'ppo_v1')])
    fields = node._schema_fields()
    assert fields['scene_id'] == 'layout_3' and fields['policy_id'] == 'ppo_v1'
    assert fields['pool'] == ol.pool_for_scene('layout_3') and fields['pool'] in ('train_pool', 'test_pool')


def test_schema_fields_reflect_an_active_exploration_draw(node):
    node._explore_sample = {'method': 'top'}        # what _explore_begin() would have set
    try:
        fields = node._schema_fields()
        assert fields['mode'] == 'exploration' and fields['params_source'] == 'random'
    finally:
        node._explore_sample = None


def test_the_command_record_carries_step_index_none_but_its_picks_carry_the_real_one(node, monkeypatch):
    records = []

    class Sink:
        def append(self, r):
            records.append(r)
    monkeypatch.setattr(node, '_outcomes', Sink())
    # explicit, not relying on declared defaults: a preceding test can leak a parameter value into this
    # node of the same name before its own declare_parameter default takes effect (same ROS2
    # parameter-service timing issue worked around elsewhere, e.g. test_exploration.py's set_mode()).
    node.set_parameters([rclpy.parameter.Parameter('scene_id', rclpy.Parameter.Type.STRING, 'layout_9'),
                          rclpy.parameter.Parameter('policy_id', rclpy.Parameter.Type.STRING, '')])
    node.dry_run = True
    try:
        node.executing = True
        # go_home has no pick/place step, but still exercises _execute()'s per-step index tracking and
        # the command record's own schema fields without needing a fake world.
        node._execute(json.dumps({'command': 'go home', 'plan': [{'action': 'go_home'}], 'model': 'grammar'}))
    finally:
        node.dry_run = False
        node.set_parameters([rclpy.parameter.Parameter('scene_id', rclpy.Parameter.Type.STRING, '')])
    cmd = [r for r in records if r.get('kind') == 'command'][0]
    assert cmd['scene_id'] == 'layout_9' and cmd['pool'] == ol.pool_for_scene('layout_9')
    assert cmd['mode'] == 'default' and cmd['params_source'] == 'heuristic' and cmd['policy_id'] == ''
    assert node._current_step_index is None                     # reset once the command finishes


def test_a_pick_inside_a_plan_carries_its_own_step_index(node, world, monkeypatch):
    records = []

    class Sink:
        def append(self, r):
            records.append(r)
    monkeypatch.setattr(node, '_outcomes', Sink())
    node.set_parameters([rclpy.parameter.Parameter('outcome_log', rclpy.Parameter.Type.BOOL, True)])
    node.executing = True
    try:
        node._execute(json.dumps({
            'command': 'pick up the mouse', 'model': 'grammar',
            'plan': [{'action': 'pick', 'object_id': 'mouse_00'}]}))
    finally:
        node.executing = False
    picks = [r for r in records if r.get('kind') == 'pick']
    assert picks and picks[0]['step_index'] == 0
    assert 'pre_action_features' in picks[0] and picks[0]['pre_action_features']['label'] == 'mouse'
