import pytest

pytest.importorskip('rclpy')
pytest.importorskip('ollama')

from thesis_robot import safety_geometry as sg  # noqa: E402
from thesis_robot.llm_planner_node import fix_move_to_heights, fix_pick_heights, trim_pick_extras, validate_plan  # noqa: E402

GEOM = {'table_top_z': -0.06, 'x': [-0.15, 0.75], 'y': [-0.80, 0.85], 'rear_wall_x': -0.20}
SCENE = {'cup_00': {'label': 'cup', 'x': 0.159, 'y': 0.1327, 'z': -0.0897,
                    'reachable': True, 'stale': False}}


def _plan(z):
    return [{'action': 'move_to', 'x': 0.1586, 'y': 0.1327, 'z': z, 'speed': 0.2}]


@pytest.mark.parametrize('llm_z', [-0.091, 0.06, 0.12, 0.5])
def test_move_to_height_keeps_fingertips_above_table(llm_z):
    floor = sg.flange_floor_z(GEOM)
    plan = fix_move_to_heights(_plan(llm_z), SCENE, floor)
    z = plan[0]['z']
    assert z >= floor
    assert z - sg.TCP_REACH_M >= GEOM['table_top_z'] + sg.LINK_TABLE_CLEARANCE_M
    assert validate_plan(plan, SCENE, floor)[0]


def test_old_incident_target_would_now_be_rejected():
    # flange z=0.12 (the previous "safe" minimum) puts the pads below the table
    floor = sg.flange_floor_z(GEOM)
    ok, why, _ = validate_plan(_plan(0.12), SCENE, floor)
    assert not ok and 'floor' in why


def test_fallback_floor_without_geometry():
    floor = sg.flange_floor_z(None)
    plan = fix_move_to_heights(_plan(-0.091), SCENE, floor)
    assert plan[0]['z'] == floor


def test_pick_approach_height_is_derived_from_the_scene():
    floor = sg.flange_floor_z({'table_top_z': -0.0125})       # the real table: floor 0.2525
    scene = {'cup_00': {'label': 'cup', 'x': 0.4, 'y': 0.0, 'z': 0.05, 'reachable': True, 'stale': False}}
    plan = [{'action': 'pick', 'object_id': 'cup_00', 'approach_z': 0.25}]     # the LLM's value that used to be rejected
    assert not validate_plan([dict(plan[0])], scene, floor)[0]
    fixed = fix_pick_heights(plan, scene, floor)
    assert abs(fixed[0]['approach_z'] - (0.05 + sg.TCP_REACH_M + 0.12)) < 1e-6
    assert validate_plan(fixed, scene, floor)[0]


def test_pick_height_is_capped_for_tall_objects():
    floor = sg.flange_floor_z(GEOM)
    scene = {'b': {'label': 'bottle', 'x': 0.4, 'y': 0.0, 'z': 0.30, 'reachable': True, 'stale': False}}
    assert fix_pick_heights([{'action': 'pick', 'object_id': 'b', 'approach_z': 0.2}], scene, floor)[0]['approach_z'] == 0.5


def _llm_pick_plan():
    return [{'action': 'move_to', 'x': 0.4, 'y': 0.0, 'z': 0.37}, {'action': 'pick', 'object_id': 'cup_00', 'approach_z': 0.39},
            {'action': 'go_home'}, {'action': 'open_gripper'}]


def test_pick_up_does_not_drop_the_object():
    acts = [s['action'] for s in trim_pick_extras(_llm_pick_plan(), 'pick up the cup')]
    assert acts == ['pick']


def test_a_requested_release_or_return_is_kept():
    assert [s['action'] for s in trim_pick_extras(_llm_pick_plan(), 'pick up the cup and put it down')][-1] == 'open_gripper'
    acts = [s['action'] for s in trim_pick_extras(_llm_pick_plan(), 'grab the cup and go back home')]
    assert 'go_home' in acts and 'open_gripper' not in acts


def test_open_gripper_before_a_pick_is_untouched():
    plan = [{'action': 'open_gripper'}, {'action': 'pick', 'object_id': 'c', 'approach_z': 0.4}]
    assert trim_pick_extras(plan, 'pick up the cup') == plan


def test_stray_move_after_a_pick_is_removed():
    plan = [{'action': 'pick', 'object_id': 'c', 'approach_z': 0.4}, {'action': 'move_to', 'x': 0.26, 'y': -0.24, 'z': 0.33}]
    assert [x['action'] for x in trim_pick_extras(plan, 'pick up the cup')] == ['pick']
    assert [x['action'] for x in trim_pick_extras(plan, 'pick up the cup and put it next to the mouse')] == ['pick', 'move_to']


def test_move_to_before_a_pick_is_removed_because_pick_approaches_by_itself():
    plan = [{'action': 'move_to', 'x': 0.4035, 'y': 0.0279, 'z': 0.2779}, {'action': 'open_gripper'},
            {'action': 'pick', 'object_id': 'c', 'approach_z': 0.4}]
    assert [x['action'] for x in trim_pick_extras(plan, 'pick up the cup')] == ['open_gripper', 'pick']
    only_move = [{'action': 'move_to', 'x': 0.4, 'y': 0.0, 'z': 0.37}]
    assert trim_pick_extras(only_move, 'go near the cup') == only_move       # no pick: untouched
