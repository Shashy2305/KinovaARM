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

from thesis_robot import locations as locs

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
# NAMED PLACES in the ROBOT's own frame. The base is at (0, 0); x runs forward along the table (front = away from the base, back = toward the base); the robot's LEFT is +y and its RIGHT is -y
# (the robot faces +x). "Corner" without a depth means the corner on the line of the base (back), "edge"/"side" without one means halfway (mid). Say "my left" / "the operator's right" to use the
# operator's side instead (the operator faces the robot, so it is mirrored). All points lie inside the planner's workspace (sg.PLAN_X_RANGE 0.07..0.65, PLAN_Y_RANGE -0.50..0.50) with a margin.
ZONE_X = {'front': 0.55, 'mid': 0.40, 'back': 0.22}
ZONE_Y = {'left': 0.38, 'centre': 0.0, 'right': -0.38}
_DEPTH = {'front': 'front', 'far': 'front', 'forward': 'front', 'back': 'back', 'near': 'back', 'rear': 'back'}
_SIDE = {'left': 'left', 'right': 'right', 'centre': 'centre', 'center': 'centre', 'middle': 'centre'}
CORNER_PUT = r'(?:put|place|set|move|bring|drop|shift|keep|leave|park|store)'
_ZD = r'(?:front|far|forward|back|near|rear)'
_ZS = r'(?:left|right|centre|center|middle)'
ZONE = (r"(?:(?:in|on|at|to) )?(?:the )?(?P<zframe>my |the operators |operators |operator's |robots |robot's |robot )?(?:(?P<zd0>" + _ZD + r")(?: |-))?(?P<zs>" + _ZS + r")(?:(?: |-)(?P<zd1>" + _ZD + r"))? (?P<znoun>corner|edge|side)"
        r"(?: of the (?:table|workspace|desk))?(?: (?:from|of|by|on|along|at) the (?:robot )?(?:base|robot|arm)(?: line)?)?")
ZONE_FB = r"(?:in|on|at|to) the (?P<zfb>front|back|far|near|rear) (?:of the (?:table|workspace|desk)|edge|side|area)"


def zone_xy(depth, side, operator=False):
    """(x, y) of a named place in the robot frame; operator=True mirrors left/right (the operator's own left)."""
    y = ZONE_Y[side]
    return ZONE_X[depth], (-y if operator else y)


def _zone_from_groups(g):
    """(x, y) for a matched ZONE / ZONE_FB, or None when the match has no zone."""
    if g.get('zs'):
        depth = _DEPTH.get(g.get('zd1') or g.get('zd0') or '') or ('back' if g.get('znoun') == 'corner' else 'mid')
        operator = (g.get('zframe') or '').strip().startswith(('my', 'the operator', 'operator'))
        return zone_xy(depth, _SIDE[g['zs']], operator)
    if g.get('zfb'):
        return zone_xy(_DEPTH[g['zfb']], 'centre')
    return None
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
    c = re.sub(r'\bthan\b', 'then', c)                    # "pick up A ... than pick up B" (a typo for "then"; nobody says "than" in a command)
    return re.sub(r'\s+', ' ', c).strip()


def _o(group):
    return OBJ.replace('{g}', group)


COFFEE_ASK = (r"^(?:(?:give|get|make|bring|fetch|pour|serve) me|(?:i )?(?:want|would like|need)|can i have|could i have|may i have|make)(?: a| an| some| my| the| another| one)?(?: cup of| mug of| fresh| hot)* ?coffee(?: (?:now|again))?$"
              r"|^coffee$|^(?:make|brew) (?:a |some |the )?coffee$")

# "give me coffee in the white cup (and put it back where it was)" / "use the white cup for coffee and bring it back": the coffee routine with a named cup, and optionally the cup is brought back to where it stood
# (2026-11-05, the professor's long plan: pick the white cup where it is, take it through the coffee machine, return it to the same spot).
_RETURN = (r"(?:(?:,| and| then| and then| after that| afterwards| after) ?)+(?:(?:put|bring|take|keep|place|send|carry|move|set)(?: it| the cup| the mug)? back|return(?: it| the cup| the mug)?(?: back)?)"
           r"(?: to where it was| where it was| to where it came from| to its (?:place|spot|original (?:place|spot|position|location))| to the (?:same|original) (?:place|spot|position)| where you found it| where you got it)?")
_COFFEE_BEG = r"(?:(?:give|get|make|bring|fetch|pour|serve) me|(?:i )?(?:want|would like|need)|make)(?: a| an| some| my| the| another| one)?(?: cup of| mug of| fresh| hot)* ?coffee"
_CUP_NAMED = r"(?:the |a |my |that |this )?(?:(?P<cc>" + _COLOUR_WORDS + r") )?(?:cup|mug)"
COFFEE_IN = r"^" + _COFFEE_BEG + r" (?:in|with|using|into) " + _CUP_NAMED + r"(?P<ret>" + _RETURN + r")?$"
COFFEE_USE = r"^(?:use|take|grab|get) " + _CUP_NAMED + r" (?:to make|for|to brew|to pour|to get) (?:me )?(?:a |some |the )?(?:cup of |mug of )?coffee(?P<ret>" + _RETURN + r")?$"

PATTERNS = [
    ('coffee', re.compile(COFFEE_IN)),
    ('coffee', re.compile(COFFEE_USE)),
    ('coffee', re.compile(COFFEE_ASK)),
    ('go_home', re.compile(r'^(?:(?:go|return|come|move|send it|bring it|get|take it)(?: it| the arm| back)*(?: to)?(?: the)? )?home(?: position)?$')),
    # "put it down", "put the cup back on the table" (2026-10-26: the second was not recognised, nothing checked the model's plan and a lone go_home was approved with the cup still in the gripper)
    ('put_down', re.compile(r"^(?:put|set|place)(?: it| that| this| the object| the [a-z]+(?: [a-z]+)?)? (?:down|back)(?: down)?(?: (?:on|onto|at|in) (?:the )?(?:table|desk|surface|same place|same spot))?$"
                            r"|^(?:release|let go of)(?: it| that)?$|^drop it$")),
    ('pick_place_side', re.compile(r'^' + PICK_VERB + r' ' + _o('a') + r'(?: and then| and| then|,)? ' + PUT_VERB + r' (?:it|that) ' + SIDE + r' ' + _o('b') + r'$')),
    ('pick_place_side', re.compile(r'^' + PUT_VERB + r' ' + _o('a') + r' ' + SIDE + r' ' + _o('b') + r'$')),
    ('pick_place_base', re.compile(r'^' + PICK_VERB + r' ' + _o('a') + r'(?: and then| and| then|,)? ' + PUT_VERB + r' (?:it|that) ' + NEAR_BASE + r' ' + BASE + r'$')),
    ('pick_place_base', re.compile(r'^' + PUT_VERB + r' ' + _o('a') + r' ' + NEAR_BASE + r' ' + BASE + r'$')),
    ('pick_place_near', re.compile(r'^' + PICK_VERB + r' ' + _o('a') + r'(?: and then| and| then|,)? ' + PUT_VERB + r' (?:it|that) ' + NEXT_TO + r' ' + _o('b') + r'$')),
    ('pick_place_near', re.compile(r'^' + PUT_VERB + r' ' + _o('a') + r' ' + NEXT_TO + r' ' + _o('b') + r'$')),
    ('pick_place_zone', re.compile(r'^' + PICK_VERB + r' ' + _o('a') + r'(?: and then| and| then|,)? ' + CORNER_PUT + r' (?:it|that) ' + ZONE + r'$')),
    ('pick_place_zone', re.compile(r'^' + CORNER_PUT + r' ' + _o('a') + r' ' + ZONE + r'$')),
    ('pick_place_zone', re.compile(r'^' + PICK_VERB + r' ' + _o('a') + r'(?: and then| and| then|,)? ' + CORNER_PUT + r' (?:it|that) ' + ZONE_FB + r'$')),
    ('pick_place_zone', re.compile(r'^' + CORNER_PUT + r' ' + _o('a') + r' ' + ZONE_FB + r'$')),
    ('pick_aside', re.compile(r'^' + PICK_VERB + r' ' + _o('a') + r'(?: and then| and| then|,)? ' + PUT_VERB + r' (?:it|that) ' + ASIDE + r'$')),
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
                    'zone': _zone_from_groups(g) if kind == 'pick_place_zone' else None,
                    'cup_colour': COLOURS.get(g.get('cc')) if g.get('cc') else None,
                    'return': bool(g.get('ret'))}
    return None


SEQ_SPLIT = re.compile(r'\b(?:and then|then|afterwards|after that)\b')


def parse_sequence(command):
    """{'kind': 'sequence', 'items': [intent, ...]} for a command that chains several recognised commands ("pick up the blue cup and put it next to the mouse, then pick up the mouse and put it next to
    the white cup"), else None (also when ANY part is not recognised: then the whole command goes to the model as before)."""
    parts = [p.strip() for p in SEQ_SPLIT.split(_clean(command)) if p.strip()]
    if len(parts) < 2:
        return None
    items = [parse(p) for p in parts]
    if any(i is None for i in items):
        return None
    return {'kind': 'sequence', 'items': items}


def parse_any(command):
    """A single recognised command, or a recognised sequence of them, or None."""
    return parse(command) or parse_sequence(command)


def _label(scene, oid):
    return (scene.get(oid) or {}).get('label')


def _objects(scene, label, colour=None):
    """Reachable, fresh objects of this class; with `colour` only those the cameras call that colour (an object whose colour is not known never matches a colour request)."""
    return [(oid, o) for oid, o in scene.items()
            if isinstance(o, dict) and o.get('label') == label and o.get('reachable') and not o.get('stale')
            and (colour is None or o.get('color') == colour)]


def _describe(label, colour):
    return f'{colour} {label}' if colour else label


def _why_unreachable(label, colour, scene, x_range, y_range, need=1):
    """None when at least `need` fresh, reachable objects of this class (and colour) exist, else the plain-language reason."""
    if not label or len(_objects(scene, label, colour)) >= need:
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


def no_reachable_reason(intent, scene, x_range=(0.10, 0.60), y_range=(-0.35, 0.35)):
    """None when the command can go on to the model; else a plain-language reason it cannot be done: an object it names (the one to pick, and the one to put it NEXT TO) is not in the scene, is stale,
    or stands outside what the arm reaches. Without this the model returned an empty plan after three slow retries; and when the reference object was missing it improvised (2026-11-01: "put it next
    to the mouse" with no mouse in view put the cup next to the white cup). A sequence is checked step by step."""
    if not intent:
        return None
    if intent.get('kind') == 'coffee':
        cfg = locs.coffee_config()
        if locs.get(cfg['tray_location']) is None:
            return f'the coffee tray location {cfg["tray_location"]!r} has not been taught (scripts/teach_location.py record {cfg["tray_location"]})'
        colour = intent.get('cup_colour')
        if _coffee_cup(scene, cfg, colour) is None:
            return (f'there is no {colour} cup the arm can reach on the table' if colour else 'there is no cup the arm can reach on the table') + ': put one within reach and try again'
        return None
    if intent.get('kind') == 'sequence':
        for i, item in enumerate(intent['items'], 1):
            why = no_reachable_reason(item, scene, x_range, y_range)
            if why:
                return f'step {i} of {len(intent["items"])}: {why}'
        return None
    why = _why_unreachable(intent.get('a'), intent.get('a_color'), scene, x_range, y_range)
    if why:
        return why
    if intent.get('kind') in ('pick_place_near', 'pick_place_side') and intent.get('b'):
        same = intent.get('b') == intent.get('a') and (intent.get('b_color') in (None, intent.get('a_color')))
        return _why_unreachable(intent['b'], intent.get('b_color'), scene, x_range, y_range, need=2 if same else 1)
    return None


def _coffee_cup(scene, cfg, colour=None):
    """(object_id, entry) of the fresh, reachable cup nearest to the serving spot, from anywhere on the table (2026-11-03: the cup no longer has to stand at one spot; one that is outside the window the
    side pick reaches is first brought to the spot), or None. With a colour only a cup the cameras call that colour qualifies."""
    sx, sy = cfg['serve_xy']
    near = sorted(((math.hypot(o['x'] - sx, o['y'] - sy), oid, o) for oid, o in _objects(scene, 'cup', colour) if o.get('x') is not None), key=lambda t: t[0])
    return (near[0][1], near[0][2]) if near else None


def in_coffee_window(entry, cfg):
    """True when the cup stands where a side pick toward the tray azimuth is reliable: within window_x / window_y of the serving spot (see locations.coffee_config)."""
    sx, sy = cfg['serve_xy']
    return abs(float(entry['x']) - sx) <= cfg['window_x_m'] and abs(float(entry['y']) - sy) <= cfg['window_y_m']


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


_STEPS = {'go_home': 1, 'put_down': 1, 'go_near': 1, 'pick': 1}        # steps a part of a sequence takes (everything else is a pick and a place)


def check(intent, steps, scene):
    """(ok, why) for the model's raw plan `steps` against the intent, in the current `scene`. why is '' when ok."""
    def res(ok, why):
        return bool(ok), '' if ok else why
    k = intent['kind']
    if k == 'coffee':
        good = build_plan(intent, scene)
        return res(good is not None and steps == good, 'expected the coffee routine: pick the cup at the serving spot (side grasp), put it in the tray, wait, take it out, put it down at the serving spot')
    if k == 'sequence':
        want = sum(_STEPS.get(i['kind'], 2) for i in intent['items'])
        if len(steps) != want:
            return False, f'expected {want} steps for the {len(intent["items"])} parts of the command, got {len(steps)}'
        n = 0
        for i, item in enumerate(intent['items'], 1):
            m = _STEPS.get(item['kind'], 2)
            ok, why = check(item, steps[n:n + m], scene)
            if not ok:
                return False, f'part {i} of the command: {why}'
            n += m
        return True, ''
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
    if k == 'pick_place_zone':
        st = places[0]
        cx, cy = intent['zone']
        ok = 'x' in st and 'y' in st and math.hypot(st['x'] - cx, st['y'] - cy) <= 0.08
        return res(ok, f'the place must be at the named spot x={cx}, y={cy}: {{"action":"place","x":{cx},"y":{cy}}}')
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
    if k == 'coffee':
        cfg = locs.coffee_config()
        colour = intent.get('cup_colour')
        cup = _coffee_cup(scene, cfg, colour)
        if cup is None or locs.get(cfg['tray_location']) is None:
            return None
        tray = cfg['tray_location']
        serve = {'x': cfg['serve_xy'][0], 'y': cfg['serve_xy'][1]}
        origin = [round(float(cup[1]['x']), 3), round(float(cup[1]['y']), 3)]          # where the cup stands now: with "put it back" it is returned there
        named = {'colour': colour} if colour else {}
        staging = []
        outside = not in_coffee_window(cup[1], cfg)
        if outside:
            # A cup anywhere else is brought to the serving spot first (any grasp that works), looked up again there, and picked from the side by position: its id changes when it is seen again.
            staging = [{'action': 'stage_object', 'object_id': cup[0], 'label': 'cup', **named, 'from': origin, **serve, 'tol_x': cfg['window_x_m'], 'tol_y': cfg['window_y_m'],
                        'tries': cfg['stage_tries'], 'approach_z': 0.35}]
        side_pick = {'action': 'pick', 'object_id': cup[0], 'approach_z': 0.35, 'side_az': cfg['pick_az']}
        if staging:
            side_pick.update({'near': [serve['x'], serve['y']], 'label': 'cup', **named})
        # Home first and home last: the routine's own moves leave the tool horizontal, and a pick that starts from there has no solution for its first straight rise (2026-11-02, second live run).
        plan = [{'action': 'go_home'}] + staging + [
                side_pick,
                {'action': 'go_to_location', 'name': tray, 'stage': 'put'},
                {'action': 'wait', 'seconds': cfg['wait_s']},
                {'action': 'go_to_location', 'name': tray, 'stage': 'in'},
                {'action': 'go_to_location', 'name': tray, 'stage': 'take'},
                {'action': 'place', 'x': cfg['serve_xy'][0], 'y': cfg['serve_xy'][1], 'learn': 'serve'},
                {'action': 'go_home'},
                {'action': 'check_placed', 'label': 'cup', **named, **serve, 'learn': 'serve'}]
        if intent.get('return') and outside:
            # ... and "put it back": the same stage step in the other direction, from the serving spot to where the cup stood (the side-held route cannot reach that far, so it is carried from above), then a check
            plan += [{'action': 'stage_object', 'object_id': cup[0], 'label': 'cup', **named, 'from': [serve['x'], serve['y']], 'x': origin[0], 'y': origin[1], 'tol_x': 0.04, 'tol_y': 0.04,
                      'tries': cfg['stage_tries'], 'approach_z': 0.35, 'to_origin': True},
                     {'action': 'go_home'},
                     {'action': 'check_placed', 'label': 'cup', **named, 'x': origin[0], 'y': origin[1]}]
        return plan
    if k == 'sequence':
        out = []
        for item in intent['items']:
            part = build_plan(item, scene)
            if not part:
                return None
            out += part
        return out
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
    if k == 'pick_place_zone':
        cx, cy = intent['zone']
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
