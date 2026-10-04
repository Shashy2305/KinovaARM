import pytest

pytest.importorskip('rclpy')
pytest.importorskip('ollama')

from thesis_robot import safety_geometry as sg  # noqa: E402
from thesis_robot.llm_planner_node import fix_move_to_heights, validate_plan  # noqa: E402

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
