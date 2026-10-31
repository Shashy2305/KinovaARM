"""
A small grammar for the common commands (pure Python, no ROS): what the operator MEANT, so a plan from the language
model can be VERIFIED against it and, only when the model keeps getting it wrong, REPAIRED.

Why: on a benchmark of eight command forms the 7B model answered "go home" with a move_to at (0.3, 0, 0.4), "put it
down" with an empty plan, and "move the bowl next to the bottle" with two move_to steps and no pick. A wrong plan for
a simple command is not a harmless miss on a real arm. The model stays the planner (it handles everything this grammar
does not recognise); the grammar is the check and the safety net.

Intents (dict): {'kind', 'a', 'b'} with kind one of
  go_home | put_down | pick | pick_place_near | pick_place_side | pick_place_base | pick_aside | go_near
plus a_color / b_color (white, blue, ...) when the operator named a colour, and a place 'near the robot base'
and a, b = object labels (cup, bowl, bottle, mouse, ...), or None.
"""
import math
import re

SYNONYMS = {'cup': 'cup', 'mug': 'cup', 'bowl': 'bowl', 'bottle': 'bottle', 'mouse': 'mouse',
            'phone': 'cell phone', 'remote': 'remote', 'book': 'book', 'scissors': 'scissors', 'vase': 'vase'}
_NAMES = '|'.join(sorted(SYNONYMS, key=len, reverse=True))
# colour words an operator may use for an object ("the white cup", "the blue mug"); the value is the name the cameras' classifier (object_colour.py) reports
COLOURS = {'white': 'white', 'black': 'black', 'grey': 'grey', 'gray': 'grey', 'silver': 'grey', 'blue': 'blue', 'teal': 'blue', 'navy': 'blue', 'red': 'red', 'green': 'green',
           'yellow': 'yellow', 'orange': 'orange', 'purple': 'purple', 'pink': 'pink', 'brown': 'brown'}
_COLOUR_WORDS = '|'.join(sorted(COLOURS, key=len, reverse=True))
OBJ = r'(?:the |a |an |that |this )?(?:(?:dark|light|bright|pale|small|big|large|little) )?(?:(?P<{g}_c>' + _COLOUR_WORDS + r') )?(?:[a-z]+ )??(?P<{g}>' + _NAMES + r')'
PICK_VERB = r'(?:pick up|pick|grab|take|get|lift|fetch|grasp)'
PUT_VERB = r'(?:put|place|set|move|bring|drop|shift)'
NEXT_TO = r'(?:next to|beside|besides|by|near|close to|alongside|adjacent to)'
SIDE = r'(?P<side>(?:to the |on the |at the )?(?:left|right)(?: side)?(?: of)?|in front of|in the front of|behind|in back of|at the back of|to the back of)'
ASIDE = r'(?:aside|away|out of the way|to the side|somewhere else|off to the side|over there)'
BASE = r'(?:the )?(?:robot(?:s|\'s)? |arm(?:s|\'s)? )?base(?: of the (?:robot|arm))?'
NEAR_BASE = r'(?:next to|beside|near|by|close to|at|to|towards|toward|around)'
# Corners of the table as seen from the robot base. 'left' is -y and 'front' is +x as in SIDE_VECTOR; "near"/"back" is toward the base. When no depth is said the FAR corner is meant.
# (x, y) inside the planner's workspace (sg.PLAN_X_RANGE 0.07..0.65, PLAN_Y_RANGE -0.50..0.50) with a margin for the fingers.
CORNER_XY = {('left', 'far'): (0.55, -0.38), ('right', 'far'): (0.55, 0.38), ('left', 'near'): (0.22, -0.38), ('right', 'near'): (0.22, 0.38)}
_DEPTH = {'far': 'far', 'front': 'far', 'near': 'near', 'back': 'near'}
CORNER_PUT = r'(?:put|place|set|move|bring|drop|shift|keep|leave|park|store)'
CORNER = (r'(?:(?:in|on|at|to) )?(?:the )?(?:(?P<cdepth0>near|far|front|back) )?(?P<cside>left|right)(?: (?P<cdepth>near|far|front|back))? (?:corner|edge)'
          r'(?: of the (?:table|workspace|desk))?(?: (?:from|of|by) the (?:robot )?(?:base|robot|arm))?')
FILLER = re.compile(r'\b(please|can you|could you|would you|robot|now|kindly|just)\b')


def _side_word(text):
    """'to the left of' -> 'left', 'in front of' -> 'front', 'behind'/'in back of' -> 'behind', else None."""
    t = text or ''
    for key, word in (('left', 'left'), ('right', 'right'), ('front', 'front'), ('behind', 'behind'), ('back', 'behind')):
        if key in t:
            return word
    return None


# Which way each side points from the reference object, as (dx, dy) in base_link: "left" is -y (the operator's left, as the
# scene graph's left_of), "front" +x (toward the operator), "behind" -x (toward the robot base). Same as the planner's SIDE_DIRECTION_DEG.
SIDE_VECTOR = {'left': (0.0, -1.0), 'right': (0.0, 1.0), 'front': (1.0, 0.0), 'behind': (-1.0, 0.0)}


def _clean(command):
    c = (command or '').lower()
    c = re.sub(r"[^a-z' ]+", ' ', c)
    c = FILLER.sub(' ', c)
    return re.sub(r'\s+', ' ', c).strip()


def _o(group):
    return OBJ.replace('{g}', group)


PATTERNS = [
    ('go_home', re.compile(r'^(?:(?:go|return|come|move|send it|bring it|get|take it)(?: it| the arm| back)*(?: to)?(?: the)? )?home(?: position)?$')),
    # "put it down", "put the cup back on the table" (2026-10-26: the second was not recognised, nothing checked the model's plan and a lone go_home was approved with the cup still in the gripper)
    ('put_down', re.compile(r"^(?:put|set|place)(?: it| that| this| the object| the [a-z]+(?: [a-z]+)?)? (?:down|back)(?: down)?(?: (?:on|onto|at|in) (?:the )?(?:table|desk|surface|same place|same spot))?$"
                            r"|^(?:release|let go of)(?: it| that)?$|^drop it$")),
    ('pick_place_side', re.compile(r'^' + PICK_VERB + r' ' + _o('a') + r'(?: and| then|,)? ' + PUT_VERB + r' (?:it|that) ' + SIDE + r' ' + _o('b') + r'$')),
    ('pick_place_side', re.compile(r'^' + PUT_VERB + r' ' + _o('a') + r' ' + SIDE + r' ' + _o('b') + r'$')),
    ('pick_place_base', re.compile(r'^' + PICK_VERB + r' ' + _o('a') + r'(?: and| then|,)? ' + PUT_VERB + r' (?:it|that) ' + NEAR_BASE + r' ' + BASE + r'$')),
    ('pick_place_base', re.compile(r'^' + PUT_VERB + r' ' + _o('a') + r' ' + NEAR_BASE + r' ' + BASE + r'$')),
    ('pick_place_near', re.compile(r'^' + PICK_VERB + r' ' + _o('a') + r'(?: and| then|,)? ' + PUT_VERB + r' (?:it|that) ' + NEXT_TO + r' ' + _o('b') + r'$')),
    ('pick_place_near', re.compile(r'^' + PUT_VERB + r' ' + _o('a') + r' ' + NEXT_TO + r' ' + _o('b') + r'$')),
    ('pick_place_corner', re.compile(r'^' + PICK_VERB + r' ' + _o('a') + r'(?: and| then|,)? ' + CORNER_PUT + r' (?:it|that) ' + CORNER + r'$')),
    ('pick_place_corner', re.compile(r'^' + CORNER_PUT + r' ' + _o('a') + r' ' + CORNER + r'$')),
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
                    'b': SYNONYMS.get(g.get('b')) if g.get('b') else None,
                    'a_color': COLOURS.get(g.get('a_c')) if g.get('a_c') else None,
                    'b_color': COLOURS.get(g.get('b_c')) if g.get('b_c') else None,
                    'side': _side_word(g.get('side')),
                    'corner': (g.get('cside'), _DEPTH.get(g.get('cdepth') or g.get('cdepth0') or 'far')) if g.get('cside') else None}
    return None


def _label(scene, oid):
    return (scene.get(oid) or {}).get('label')


def _objects(scene, label, colour=None):
    """Reachable, fresh objects of this class; with `colour` only those the cameras call that colour (an object whose colour is not known never matches a colour request)."""
    return [(oid, o) for oid, o in scene.items()
            if isinstance(o, dict) and o.get('label') == label and o.get('reachable') and not o.get('stale')
            and (colour is None or o.get('color') == colour)]


def _describe(label, colour):
    return f'{colour} {label}' if colour else label


def no_reachable_reason(intent, scene, x_range=(0.10, 0.60), y_range=(-0.35, 0.35)):
    """None when the command can go on to the model; else a plain-language reason it cannot be done: the object it names is not in the scene, is stale, or stands outside what the arm reaches.
    Without this the model returned an empty plan after three slow retries and the operator only saw "Plan is empty or not a list"."""
    label = (intent or {}).get('a')
    colour = (intent or {}).get('a_color')
    if not label or _objects(scene, label, colour):
        return None
    mine = [(oid, o) for oid, o in scene.items() if isinstance(o, dict) and o.get('label') == label]
    if not mine:
        return f'there is no {label} in the scene (the cameras do not see one)'
    if colour:
        others = [(oid, o) for oid, o in mine if o.get('reachable') and not o.get('stale')]
        if others:
            seen = ', '.join(f'{o.get("color") or "unknown colour"} ({oid})' for oid, o in others)
            return f'no {colour} {label} that the arm can reach: the {label}s I see are {seen}'
    parts = []
    for oid, o in mine:
        x, y = o.get('x'), o.get('y')
        if o.get('stale'):
            parts.append(f'{oid} was not seen recently (stale)')
        elif x is not None and y is not None and not (x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]):
            parts.append(f'{oid} at x={x:.2f} y={y:.2f} is outside the arm\'s range (x {x_range[0]:.2f} to {x_range[1]:.2f}, y {y_range[0]:.2f} to {y_range[1]:.2f}): move it closer')
        else:
            parts.append(f'{oid} is not reachable')
    return f'no {_describe(label, colour)} the arm can reach: ' + '; '.join(parts)


def best_object(scene, label, exclude=None, colour=None):
    cands = [(oid, o) for oid, o in _objects(scene, label, colour) if oid != exclude]
    if not cands:
        return None
    return max(cands, key=lambda c: float(c[1].get('confidence', 0) or 0))[0]


def _near(step, scene, label, tol=0.30, colour=None):
    if step.get('near'):
        return _label(scene, step['near']) == label and (colour is None or (scene.get(step['near']) or {}).get('color') == colour)
    if 'x' in step and 'y' in step:
        return any(math.hypot(o['x'] - step['x'], o['y'] - step['y']) <= tol
                   for oid, o in _objects(scene, label, colour) if o.get('x') is not None)
    return False


def _on_side(step, scene, label, side, reach=0.35):
    """A place with x/y lies on `side` of some object of `label` (within `reach`): its offset from that object points
    that way (within 55 degrees) and is not tiny."""
    vx, vy = SIDE_VECTOR[side]
    for oid, o in _objects(scene, label):
        dx, dy = step['x'] - o['x'], step['y'] - o['y']
        d = math.hypot(dx, dy)
        if 0.05 <= d <= reach and (dx * vx + dy * vy) / d >= math.cos(math.radians(55)):
            return True
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
    want = intent.get('a_color')
    if want and (scene.get(picks[0].get('object_id')) or {}).get('color') != want:
        return False, f'picked the {(scene.get(picks[0].get("object_id")) or {}).get("color") or "unknown colour"} {intent["a"]}, the command says the {want} one'
    if k == 'pick':
        # extra go_home / open_gripper steps are trimmed later by the planner (trim_pick_extras); a place is not
        return res(not places, 'a plain pick must not include a place step')
    if len(places) != 1:
        return False, f'expected one place step, got {len(places)}'
    if k == 'pick_place_corner':
        st = places[0]
        cx, cy = CORNER_XY[intent['corner']]
        ok = 'x' in st and 'y' in st and math.hypot(st['x'] - cx, st['y'] - cy) <= 0.08
        return res(ok, f'the place must be at the {intent["corner"][1]} {intent["corner"][0]} corner of the table: {{"action":"place","x":{cx},"y":{cy}}}')
    if k == 'pick_place_base':
        st = places[0]
        ok = str(st.get('near', '')).lower() == 'base' or ('x' in st and math.hypot(st['x'], st['y']) <= 0.45)
        return res(ok, 'the place must be near the robot base ({"action":"place","near":"base"})')
    if k == 'pick_place_near':
        return res(_near(places[0], scene, intent['b'], colour=intent.get('b_color')), f'the place is not next to the {_describe(intent["b"], intent.get("b_color"))}')
    if k == 'pick_place_side':
        st = places[0]
        if st.get('near'):
            ok = _label(scene, st['near']) == intent['b'] and str(st.get('side', '')).lower() == intent['side']
        else:
            ok = 'x' in st and _on_side(st, scene, intent['b'], intent['side'])
        return res(ok, f'the place must be on the {intent["side"]} of the {intent["b"]} (near that object, with side "{intent["side"]}")')
    return True, ''


def build_plan(intent, scene):
    """The raw plan the model should have produced, or None if the objects it needs are not in the scene."""
    k = intent['kind']
    if k == 'go_home':
        return [{'action': 'go_home'}]
    if k == 'put_down':
        return [{'action': 'place', 'here': True}]
    a = best_object(scene, intent['a'], colour=intent.get('a_color'))
    if a is None:
        return None
    if k == 'pick':
        return [{'action': 'pick', 'object_id': a, 'approach_z': 0.35}]
    if k == 'pick_aside':
        return [{'action': 'pick', 'object_id': a, 'approach_z': 0.35}, {'action': 'place', 'x': 0.45, 'y': 0.0}]
    if k == 'pick_place_base':
        return [{'action': 'pick', 'object_id': a, 'approach_z': 0.35}, {'action': 'place', 'near': 'base'}]
    if k == 'pick_place_corner':
        cx, cy = CORNER_XY[intent['corner']]
        return [{'action': 'pick', 'object_id': a, 'approach_z': 0.35}, {'action': 'place', 'x': cx, 'y': cy}]
    if k in ('pick_place_near', 'pick_place_side'):
        b = best_object(scene, intent['b'], exclude=a, colour=intent.get('b_color'))
        if b is None:
            return None
        place = {'action': 'place', 'near': b}
        if k == 'pick_place_side':
            place['side'] = intent['side']
        return [{'action': 'pick', 'object_id': a, 'approach_z': 0.35}, place]
    if k == 'go_near':
        o = scene[a]
        return [{'action': 'move_to', 'x': o['x'], 'y': o['y'], 'z': 0.4, 'speed': 0.2}]
    return None
