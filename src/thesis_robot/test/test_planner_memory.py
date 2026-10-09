import json

from thesis_robot import planner_memory as pm

SCENE = {
    'cup_01': {'label': 'cup', 'reachable': True, 'stale': False, 'confidence': 0.95},
    'bowl_00': {'label': 'bowl', 'reachable': True, 'stale': False, 'confidence': 0.9},
    'mouse_02': {'label': 'mouse', 'reachable': True, 'stale': False, 'confidence': 0.92},
    'bottle_00': {'label': 'bottle', 'reachable': True, 'stale': False, 'confidence': 0.93},
}


def test_templatize_replaces_object_names_with_slots_in_order():
    t, labels = pm.templatize('Pick up the mug and put it next to the bowl')
    assert t == 'pick up the <A> and put it next to the <B>' and labels == ['cup', 'bowl']
    t2, l2 = pm.templatize('pick up the cup and put it next to the bowl')
    assert t2 == t and l2 == labels                       # mug == cup


def test_similar_commands_with_different_objects_share_a_template():
    t1, _ = pm.templatize('pick up the cup and put it next to the bowl')
    t2, _ = pm.templatize('pick up the mouse and put it next to the bottle')
    assert pm.similarity(t1, t2) == 1.0
    t3, _ = pm.templatize('go home')
    assert pm.similarity(t1, t3) < 0.3


def test_skeleton_roundtrip_to_another_object_pair():
    plan = [{'action': 'pick', 'object_id': 'cup_01', 'approach_z': 0.4}, {'action': 'place', 'near': 'bowl_00'}]
    _, labels = pm.templatize('pick up the cup and put it next to the bowl')
    sk = pm.skeletonize(plan, SCENE, labels)
    assert sk == [{'action': 'pick', 'slot': 'A'}, {'action': 'place', 'near': 'B'}]
    _, labels2 = pm.templatize('pick up the mouse and put it next to the bottle')
    steps = pm.instantiate(sk, labels2, SCENE)
    assert steps[0]['object_id'] == 'mouse_02' and steps[1] == {'action': 'place', 'near': 'bottle_00'}


def test_skeleton_rejects_a_plan_touching_an_object_the_command_did_not_name():
    plan = [{'action': 'pick', 'object_id': 'bowl_00'}]
    assert pm.skeletonize(plan, SCENE, ['cup']) is None


def test_instantiate_returns_none_when_an_object_is_missing_from_the_scene():
    sk = [{'action': 'pick', 'slot': 'A'}]
    assert pm.instantiate(sk, ['scissors'], SCENE) is None


def test_prompt_block_uses_a_seed_example_and_stays_short(tmp_path):
    mem = pm.PlannerMemory(str(tmp_path))
    block = mem.prompt_block('pick up the mouse and put it next to the bottle', SCENE)
    assert 'mouse_02' in block and 'bottle_00' in block and '"near":"bottle_00"' in block
    assert len(block) <= pm.MAX_BLOCK_CHARS
    json.loads(block.split('structure: ')[1])               # the example is complete JSON


def test_object_free_commands_get_their_fixed_structure_and_unknown_ones_get_nothing(tmp_path):
    mem = pm.PlannerMemory(str(tmp_path))
    assert '"go_home"' in mem.prompt_block('go home', SCENE)
    assert '"here":true' in mem.prompt_block('put it down', SCENE)
    assert mem.prompt_block('tell me a joke', SCENE) == ''


def test_outcome_attribution_and_warning_for_a_rejected_plan(tmp_path):
    mem = pm.PlannerMemory(str(tmp_path))
    cmd = 'pick up the cup and put it next to the bowl'
    mem.record_rejected(cmd, [{'action': 'move_to', 'x': 0.3, 'y': 0.0, 'z': 0.4}], SCENE, 'the plan has no pick step')
    block = mem.prompt_block('pick up the mouse and put it next to the bottle', {**SCENE})
    assert 'WENT WRONG' in block and 'no pick step' in block
    # a plan that completes is remembered as a worked example
    mem.begin(cmd, [{'action': 'pick', 'object_id': 'cup_01'}, {'action': 'place', 'near': 'bowl_00'}], SCENE)
    assert mem.finish('EXECUTING: foo') is None            # not finished yet: still in flight
    rec = mem.finish('COMPLETE', live=True)
    assert rec['result'] == 'ok' and rec['live'] is True
    assert len(mem.entries()) == 2


def test_failed_execution_is_recorded_and_never_offered_as_an_example(tmp_path):
    mem = pm.PlannerMemory(str(tmp_path))
    cmd = 'pick up the cup'
    mem.begin(cmd, [{'action': 'pick', 'object_id': 'cup_01'}], SCENE)
    mem.finish('FAILED at step 1: the carry path passes within 12 cm of a bottle')
    ents = mem.entries()
    assert ents[0]['result'] == 'failed'
    block = mem.prompt_block('pick up the cup', SCENE)
    assert 'WENT WRONG' in block and block.count('WORKED') == 1       # the seed still serves as the example, not the failure
