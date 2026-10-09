import pytest

from thesis_robot import command_grammar as cg

SCENE = {
    'cup_01': {'label': 'cup', 'x': 0.40, 'y': -0.30, 'z': 0.04, 'reachable': True, 'stale': False, 'confidence': 0.95},
    'cup_02': {'label': 'cup', 'x': 0.30, 'y': 0.20, 'z': 0.04, 'reachable': True, 'stale': False, 'confidence': 0.80},
    'bowl_00': {'label': 'bowl', 'x': 0.32, 'y': 0.28, 'z': 0.02, 'reachable': True, 'stale': False, 'confidence': 0.90},
    'mouse_01': {'label': 'mouse', 'x': 0.37, 'y': -0.09, 'z': 0.02, 'reachable': True, 'stale': False, 'confidence': 0.93},
    'bottle_00': {'label': 'bottle', 'x': 0.25, 'y': -0.21, 'z': 0.07, 'reachable': True, 'stale': False, 'confidence': 0.93},
    'bowl_stale': {'label': 'bowl', 'x': 0.5, 'y': 0.0, 'z': 0.02, 'reachable': True, 'stale': True, 'confidence': 0.99},
}


@pytest.mark.parametrize('text,kind,a,b', [
    ('go home', 'go_home', None, None),
    ('Please go back home', 'go_home', None, None),
    ('return to home position', 'go_home', None, None),
    ('home', 'go_home', None, None),
    ('put it down', 'put_down', None, None),
    ('set it down.', 'put_down', None, None),
    ('release it', 'put_down', None, None),
    ('pick up the cup', 'pick', 'cup', None),
    ('grab the mug', 'pick', 'cup', None),
    ('Can you pick up the blue bottle?', 'pick', 'bottle', None),
    ('pick up the cup and put it next to the bowl', 'pick_place_near', 'cup', 'bowl'),
    ('grab the bottle and set it beside the mouse', 'pick_place_near', 'bottle', 'mouse'),
    ('pick up the mouse, then place it near the cup', 'pick_place_near', 'mouse', 'cup'),
    ('move the bowl next to the bottle', 'pick_place_near', 'bowl', 'bottle'),
    ('put the mouse beside the bowl', 'pick_place_near', 'mouse', 'bowl'),
    ('pick up the cup and put it aside', 'pick_aside', 'cup', None),
    ('move the mouse out of the way', 'pick_aside', 'mouse', None),
    ('go near the cup', 'go_near', 'cup', None),
    ('move over the bowl', 'go_near', 'bowl', None),
    ('look at the bottle', 'go_near', 'bottle', None),
])
def test_parse(text, kind, a, b):
    i = cg.parse(text)
    assert i is not None, text
    assert (i['kind'], i['a'], i['b']) == (kind, a, b)


@pytest.mark.parametrize('text', ['', 'tell me a joke', 'rotate the wrist by thirty degrees', 'pick up everything',
                                  'calibrate the camera and then go home maybe later'])
def test_unrecognised_commands_are_left_to_the_model(text):
    assert cg.parse(text) is None


def test_check_accepts_the_right_plan_and_names_the_mistake_in_a_wrong_one():
    i = cg.parse('pick up the cup and put it next to the bowl')
    good = [{'action': 'pick', 'object_id': 'cup_01'}, {'action': 'place', 'near': 'bowl_00'}]
    assert cg.check(i, good, SCENE) == (True, '')
    assert not cg.check(i, [{'action': 'move_to', 'x': 0.3, 'y': 0.2, 'z': 0.4}] * 2, SCENE)[0]
    wrong_obj = [{'action': 'pick', 'object_id': 'bowl_00'}, {'action': 'place', 'near': 'bowl_00'}]
    ok, why = cg.check(i, wrong_obj, SCENE)
    assert not ok and 'bowl' in why
    far = [{'action': 'pick', 'object_id': 'cup_01'}, {'action': 'place', 'x': 0.5, 'y': -0.3}]
    assert not cg.check(i, far, SCENE)[0]                                    # not near any bowl
    near_xy = [{'action': 'pick', 'object_id': 'cup_01'}, {'action': 'place', 'x': 0.33, 'y': 0.15}]
    assert cg.check(i, near_xy, SCENE)[0]                                    # within 0.30 of the bowl


def test_check_for_the_simple_kinds():
    assert cg.check(cg.parse('go home'), [{'action': 'go_home'}], SCENE)[0]
    assert not cg.check(cg.parse('go home'), [{'action': 'move_to', 'x': 0.3, 'y': 0, 'z': 0.4}], SCENE)[0]
    assert cg.check(cg.parse('put it down'), [{'action': 'place', 'here': True}], SCENE)[0]
    assert not cg.check(cg.parse('put it down'), [], SCENE)[0]
    assert cg.check(cg.parse('pick up the cup'), [{'action': 'pick', 'object_id': 'cup_01'}], SCENE)[0]
    assert cg.check(cg.parse('pick up the cup'), [{'action': 'pick', 'object_id': 'cup_01'}, {'action': 'go_home'}], SCENE)[0]   # trimmed later
    assert not cg.check(cg.parse('pick up the cup'), [{'action': 'pick', 'object_id': 'cup_01'}, {'action': 'place', 'here': True}], SCENE)[0]
    assert cg.check(cg.parse('go near the bowl'), [{'action': 'move_to', 'x': 0.32, 'y': 0.28, 'z': 0.3}], SCENE)[0]


def test_build_plan_round_trips_through_check_for_every_kind():
    for text in ('go home', 'put it down', 'pick up the cup', 'pick up the mouse and put it next to the bottle',
                 'pick up the cup and put it aside', 'go near the bowl'):
        i = cg.parse(text)
        plan = cg.build_plan(i, SCENE)
        assert plan is not None, text
        ok, why = cg.check(i, plan, SCENE)
        assert ok, f'{text}: {why}'


def test_build_plan_picks_the_most_confident_fresh_object_and_a_different_reference():
    plan = cg.build_plan(cg.parse('pick up the cup and put it next to the cup'), SCENE)
    assert plan[0]['object_id'] == 'cup_01' and plan[1]['near'] == 'cup_02'      # the other cup, never itself
    assert cg.build_plan(cg.parse('pick up the bowl'), SCENE)[0]['object_id'] == 'bowl_00'   # the stale one is ignored
    assert cg.build_plan(cg.parse('pick up the scissors'), SCENE) is None                     # not in the scene


@pytest.mark.parametrize('text,side', [
    ('pick up the cup and put it to the left of the bowl', 'left'),
    ('grab the mouse and place it on the right side of the bottle', 'right'),
    ('pick up the bottle and put it in front of the bowl', 'front'),
    ('pick up the cup and set it behind the mouse', 'behind'),
    ('move the mouse to the left of the cup', 'left'),
    ('put the bowl in back of the bottle', 'behind'),
])
def test_side_relations_parse(text, side):
    i = cg.parse(text)
    assert i is not None and i['kind'] == 'pick_place_side' and i['side'] == side, text


def test_side_check_and_build():
    i = cg.parse('pick up the cup and put it to the left of the bowl')
    plan = cg.build_plan(i, SCENE)
    assert plan[1] == {'action': 'place', 'near': 'bowl_00', 'side': 'left'}
    assert cg.check(i, plan, SCENE)[0]
    assert not cg.check(i, [plan[0], {'action': 'place', 'near': 'bowl_00'}], SCENE)[0]            # missing side
    assert not cg.check(i, [plan[0], {'action': 'place', 'near': 'bowl_00', 'side': 'right'}], SCENE)[0]
    # coordinates are accepted when they really lie on that side: the bowl is at (0.32, 0.28), left is -y
    assert cg.check(i, [plan[0], {'action': 'place', 'x': 0.32, 'y': 0.12}], SCENE)[0]
    assert not cg.check(i, [plan[0], {'action': 'place', 'x': 0.32, 'y': 0.44}], SCENE)[0]
