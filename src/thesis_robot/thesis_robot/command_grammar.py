"""
A small grammar for the common commands (pure Python, no ROS): what the operator MEANT, so a plan from the language
model can be VERIFIED against it and, only when the model keeps getting it wrong, REPAIRED.

Why: on a benchmark of eight command forms the 7B model answered "go home" with a move_to at (0.3, 0, 0.4), "put it
down" with an empty plan, and "move the bowl next to the bottle" with two move_to steps and no pick. A wrong plan for
a simple command is not a harmless miss on a real arm. The model stays the planner (it handles everything this grammar
does not recognise); the grammar is the check and the safety net.

Intents (dict): {'kind', 'a', 'b'} with kind one of
  go_home | put_down | pick | pick_place_near | pick_aside | go_near
and a, b = object labels (cup, bowl, bottle, mouse, ...), or None.
"""
import math
import re

SYNONYMS = {'cup': 'cup', 'mug': 'cup', 'bowl': 'bowl', 'bottle': 'bottle', 'mouse': 'mouse',
            'phone': 'cell phone', 'remote': 'remote', 'book': 'book', 'scissors': 'scissors', 'vase': 'vase'}
_NAMES = '|'.join(sorted(SYNONYMS, key=len, reverse=True))
OBJ = r'(?:the |a |an |that |this )?(?:[a-z]+ )??(?P<{g}>' + _NAMES + r')'
PICK_VERB = r'(?:pick up|pick|grab|take|get|lift|fetch|grasp)'
PUT_VERB = r'(?:put|place|set|move|bring|drop|shift)'
NEXT_TO = r'(?:next to|beside|besides|by|near|close to|alongside|adjacent to)'
ASIDE = r'(?:aside|away|out of the way|to the side|somewhere else|off to the side|over there)'
FILLER = re.compile(r'\b(please|can you|could you|would you|robot|now|kindly|just)\b')


def _clean(command):
    c = (command or '').lower()
    c = re.sub(r"[^a-z' ]+", ' ', c)
    c = FILLER.sub(' ', c)
    return re.sub(r'\s+', ' ', c).strip()


def _o(group):
    return OBJ.replace('{g}', group)


PATTERNS = [
    ('go_home', re.compile(r'^(?:(?:go|return|come|move|send it|bring it|get|take it)(?: it| the arm| back)*(?: to)?(?: the)? )?home(?: position)?$')),
    ('put_down', re.compile(r"^(?:put|set|place)(?: it| that| this| the object)? (?:down|back)$|^(?:release|let go of)(?: it| that)?$|^drop it$")),
    ('pick_place_near', re.compile(r'^' + PICK_VERB + r' ' + _o('a') + r'(?: and| then|,)? ' + PUT_VERB + r' (?:it|that) ' + NEXT_TO + r' ' + _o('b') + r'$')),
    ('pick_place_near', re.compile(r'^' + PUT_VERB + r' ' + _o('a') + r' ' + NEXT_TO + r' ' + _o('b') + r'$')),
    ('pick_aside', re.compile(r'^' + PICK_VERB + r' ' + _o('a') + r'(?: and| then|,)? ' + PUT_VERB + r' (?:it|that) ' + ASIDE + r'$')),
    ('pick_aside', re.compile(r'^' + PUT_VERB + r' ' + _o('a') + r' ' + ASIDE + r'$')),
    ('pick', re.compile(r'^' + PICK_VERB + r' ' + _o('a') + r'$')),
    ('go_near', re.compile(r'^(?:go|move|come|fly|hover)(?: to| near| over| above| towards| close to)? ' + _o('a') + r'$')),
    ('go_near', re.compile(r'^look at ' + _o('a') + r'$')),
]


def parse(command):
    """Intent dict for a recognised command, else None (the model plans it alone)."""
    c = _clean(command)
    for kind, rx in PATTERNS:
        m = rx.match(c)
        if m:
            g = m.groupdict()
            return {'kind': kind, 'a': SYNONYMS.get(g.get('a')) if g.get('a') else None,
                    'b': SYNONYMS.get(g.get('b')) if g.get('b') else None}
    return None


def _label(scene, oid):
    return (scene.get(oid) or {}).get('label')


def _objects(scene, label):
    return [(oid, o) for oid, o in scene.items()
            if isinstance(o, dict) and o.get('label') == label and o.get('reachable') and not o.get('stale')]


def best_object(scene, label, exclude=None):
    cands = [(oid, o) for oid, o in _objects(scene, label) if oid != exclude]
    if not cands:
        return None
    return max(cands, key=lambda c: float(c[1].get('confidence', 0) or 0))[0]


def _near(step, scene, label, tol=0.30):
    if step.get('near'):
        return _label(scene, step['near']) == label
    if 'x' in step and 'y' in step:
        return any(math.hypot(o['x'] - step['x'], o['y'] - step['y']) <= tol
                   for oid, o in _objects(scene, label) if o.get('x') is not None)
    return False


def check(intent, steps, scene):
    """(ok, why) for the model's raw plan `steps` against the intent, in the current `scene`. why is '' when ok."""
    def res(ok, why):
        return bool(ok), '' if ok else why
    k = intent['kind']
    acts = [s.get('action') for s in steps if isinstance(s, dict)]
    if k == 'go_home':
        return res(acts == ['go_home'], f'expected only go_home, got {acts}')
    if k == 'put_down':
        return res(acts == ['place'] and bool(steps[0].get('here')), f'expected a single place with here=true, got {acts}')
    picks = [s for s in steps if s.get('action') == 'pick']
    places = [s for s in steps if s.get('action') == 'place']
    if k == 'go_near':
        if acts != ['move_to']:
            return False, f'expected a single move_to, got {acts}'
        s = steps[0]
        over = 'x' in s and any(math.hypot(o['x'] - s['x'], o['y'] - s['y']) < 0.08 for _, o in _objects(scene, intent['a'])
                                if o.get('x') is not None)
        return res(over, f'the move_to is not over the {intent["a"]}')
    if len(picks) != 1:
        return False, f'expected one pick step, got {len(picks)}'
    if _label(scene, picks[0].get('object_id')) != intent['a']:
        return False, f'picked a {_label(scene, picks[0].get("object_id"))}, the command says {intent["a"]}'
    if k == 'pick':
        # extra go_home / open_gripper steps are trimmed later by the planner (trim_pick_extras); a place is not
        return res(not places, 'a plain pick must not include a place step')
    if len(places) != 1:
        return False, f'expected one place step, got {len(places)}'
    if k == 'pick_place_near':
        return res(_near(places[0], scene, intent['b']), f'the place is not next to the {intent["b"]}')
    return True, ''


def build_plan(intent, scene):
    """The raw plan the model should have produced, or None if the objects it needs are not in the scene."""
    k = intent['kind']
    if k == 'go_home':
        return [{'action': 'go_home'}]
    if k == 'put_down':
        return [{'action': 'place', 'here': True}]
    a = best_object(scene, intent['a'])
    if a is None:
        return None
    if k == 'pick':
        return [{'action': 'pick', 'object_id': a, 'approach_z': 0.35}]
    if k == 'pick_aside':
        return [{'action': 'pick', 'object_id': a, 'approach_z': 0.35}, {'action': 'place', 'x': 0.45, 'y': 0.0}]
    if k == 'pick_place_near':
        b = best_object(scene, intent['b'], exclude=a)
        if b is None:
            return None
        return [{'action': 'pick', 'object_id': a, 'approach_z': 0.35}, {'action': 'place', 'near': b}]
    if k == 'go_near':
        o = scene[a]
        return [{'action': 'move_to', 'x': o['x'], 'y': o['y'], 'z': 0.4, 'speed': 0.2}]
    return None
