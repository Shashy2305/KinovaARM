#!/usr/bin/env python3
"""
LLM Planner Node — the brain of the Audio-to-Action pipeline.

Subscribes to:
  /scene_snapshot   (std_msgs/String — JSON world model)
  /voice_command    (std_msgs/String — transcribed text)

Publishes to:
  /action_plan      (std_msgs/String — validated JSON plan)
  /planner_status   (std_msgs/String — human-readable status)

Your thesis contribution: LLM-based task decomposition grounded
in a live scene graph, with pre-execution safety validation.
"""
import rclpy, json, math, time, threading, ollama
from rclpy.node import Node
from std_msgs.msg import String

from thesis_robot import safety_geometry as sg

# ── SYSTEM PROMPT ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a task planner for a Kinova Gen3 7-DOF robotic arm.

SAFETY RULES — every action must obey ALL of these without exception:
  RULE 1: ALL z values >= {Z_FLOOR} at minimum (z is the wrist flange height;
          the fingertips hang ~0.21 m below it, so this keeps them above the table)
  RULE 2: approach_z in pick >= {Z_FLOOR} (approach from well above object)
  RULE 3: place z >= {Z_FLOOR} (place gently, never at table level)
  RULE 4: Speed always exactly 0.20 (20% of maximum — safety requirement)
  RULE 5: x must be in [0.10, 0.60]  y must be in [-0.35, 0.35]
  RULE 6: Never command picking a stale object (stale=true in world state)

AVAILABLE ACTIONS — use only these, no others:
  {"action":"move_to",      "x":float, "y":float, "z":float, "speed":0.2}
  {"action":"pick",         "object_id":"str", "approach_z":float}
  {"action":"place",        "near":"object_id"}     (put the held object down next to that object)
  {"action":"place",        "x":float, "y":float}   (put the held object down at that table position)
  {"action":"place",        "here":true}            (put the held object down where the arm is now)
  {"action":"open_gripper"}
  {"action":"close_gripper"}
  {"action":"go_home"}
  {"action":"null_space_adjust", "objective":"clear_camera"}

WHICH ACTION FOR WHICH COMMAND:
  - "go to/near/over X", "move to X", "look at X" (no picking up) ->
    ONE move_to step. Use X's x/y from the world state, and
    z = max(X's z + 0.31, {Z_FLOOR}) (hover above it with the fingertips ~10 cm
    clear of its top, don't descend onto it).
    Do NOT use pick for these — pick closes the gripper on the object,
    which is not what "go near" means.
  - "pick up X", "grab X", "get X" -> a pick step (approach_z>=0.15),
    optionally followed by place/open_gripper if the command also says
    where to put it down.
  - "put/place it next to X", "pick up A and put it beside B" -> pick A, then
    {"action":"place","near":"<B's object_id>"}. Do not compute coordinates for a place
    next to an object, give the object_id. "put it down" -> {"action":"place","here":true}.
    Never add go_home or open_gripper unless the command asks for it.
  - If the named object isn't in the world state below, or every
    instance of it is stale, say so in "reasoning" and return an empty
    plan rather than guessing a position.

WORKSPACE BOUNDS: x=[0.10, 0.60]  y=[-0.35, 0.35]  z=[{Z_FLOOR}, 0.50]

RESPOND WITH EXACTLY THIS JSON STRUCTURE — no other format:
{
  "reasoning": "step by step thinking about which objects, sequence, safety checks",
  "plan": [
    {"action": "...", ...},
    {"action": "...", ...}
  ]
}"""

# ── MOVE_TO HEIGHT CORRECTION ────────────────────────────────────────
# The system prompt tells the LLM to set move_to's z to "object z + 0.15"
# (hover above, don't descend onto it), but a 7B local model doesn't
# follow that arithmetic reliably -- it sometimes copies the object's raw
# z straight through, which is at/below table height and gets rejected
# by the floor check below. Since the LLM's x/y *are* reliably copied
# from the target object, re-derive z from the matched object ourselves
# instead of trusting the LLM's addition, rather than just rejecting and
# hoping a retry does better.
# flange height above the object's centre: the fingertips hang TCP_REACH_M below the flange,
# and we want them ~10 cm clear of the object (a cup is ~9 cm tall)
MOVE_TO_HOVER_M = sg.TCP_REACH_M + 0.10
_XY_MATCH_TOL_M = 0.05
# z_floor comes from safety_geometry.flange_floor_z(): table height + the
# 0.215 m flange-to-fingertip reach + clearance. (An earlier fixed 0.12
# minimum here still put the fingertips at the table surface.)


def fix_move_to_heights(plan, scene, z_floor):
    """Mutates move_to steps in place: if a step's (x,y) matches a known
    scene object closely enough, override z to that object's z + hover
    offset (never below z_floor), regardless of what the LLM computed."""
    if not isinstance(plan, list):
        return plan
    for step in plan:
        if not isinstance(step, dict) or step.get('action') != 'move_to':
            continue
        if not (_is_number(step.get('x')) and _is_number(step.get('y'))):
            continue
        best_id, best_dist = None, _XY_MATCH_TOL_M
        for obj_id, obj in scene.items():
            if not isinstance(obj, dict) or obj.get('z') is None:
                continue
            dist = ((obj['x'] - step['x']) ** 2 + (obj['y'] - step['y']) ** 2) ** 0.5
            if dist < best_dist:
                best_id, best_dist = obj_id, dist
        if best_id is not None:
            step['z'] = min(0.50, max(scene[best_id]['z'] + MOVE_TO_HOVER_M, z_floor))
    return plan


PICK_HOVER_ABOVE_M = 0.12     # fingertips above the object's centre while the wrist camera centres on it


def fix_pick_heights(plan, scene, z_floor):
    """Set every pick's approach_z from the scene instead of trusting the LLM's arithmetic
    (it returned 0.250 against a floor of 0.253 and the plan was rejected). The arm controller
    uses approach_z only as a minimum; it derives the real hover height itself."""
    if not isinstance(plan, list):
        return plan
    for step in plan:
        if not isinstance(step, dict) or step.get('action') != 'pick':
            continue
        obj = scene.get(step.get('object_id')) if isinstance(scene, dict) else None
        if isinstance(obj, dict) and _is_number(obj.get('z')):
            step['approach_z'] = round(min(0.50, max(obj['z'] + sg.TCP_REACH_M + PICK_HOVER_ABOVE_M, z_floor)), 4)
    return plan


RELEASE_WORDS = ('place', 'put', 'drop', 'release', 'give', 'hand', 'set ', 'down', 'let go', 'throw',
                 'bring', 'move it', 'carry', 'deliver', 'pour', 'open', 'next to', 'beside', 'near it', 'move the')
HOME_WORDS = ('home', 'back', 'return', 'rest')


def trim_pick_extras(plan, command):
    """The 7B model likes to pad a pick with extra steps: go_home, open_gripper (drops what it
    just picked up, from wherever the arm is), or a stray move_to over another object. The
    command "pick up the cup" asked for none of that. After the first pick keep only what the
    command asks for: everything if it mentions putting/releasing/moving the object somewhere,
    only go_home if it mentions going home or back, nothing otherwise."""
    if not isinstance(plan, list):
        return plan
    cmd = (command or '').lower()
    wants_release = any(w in cmd for w in RELEASE_WORDS)
    wants_home = any(w in cmd for w in HOME_WORDS)
    # A pick approaches the object itself (hover, wrist-camera centring, straight descent). The
    # model's own move_to before it is redundant, and was once 5 cm off the object at fingertip
    # height, i.e. the open fingers beside the cup. Drop move_to steps that come before the pick.
    first_pick = next((i for i, st in enumerate(plan) if isinstance(st, dict) and st.get('action') == 'pick'), None)
    if first_pick is not None:
        plan = [st for i, st in enumerate(plan)
                if not (i < first_pick and isinstance(st, dict) and st.get('action') == 'move_to')]
    out, picked = [], False
    for step in plan:
        act = step.get('action') if isinstance(step, dict) else None
        if picked and not wants_release and not (wants_home and act == 'go_home'):
            continue
        if act == 'open_gripper' and any(isinstance(o, dict) and o.get('action') == 'place' for o in out):
            continue                                  # place releases the object itself
        if act == 'go_home' and picked and not wants_home:
            continue
        if act == 'pick':
            picked = True
        out.append(step)
    # A place carries, lowers and releases by itself. A move_to the model put between the pick and
    # the place (it once hovered the held mug 3 cm above the table, straight over the mouse) or after
    # the place is a collision risk, never needed.
    if any(isinstance(o, dict) and o.get('action') == 'place' for o in out):
        out = [o for o in out if isinstance(o, dict) and o.get('action') in ('pick', 'place', 'go_home')]
    return out


# ── PLACE TARGETS ────────────────────────────────────────────────────
PLACE_RINGS_M = (0.15, 0.18, 0.22)      # distance from the target object's centre to try (the open fingers swing ~9.5 cm
                                        # to each side of the released object: 13 cm left no room, 2026-10-06)
PLACE_MIN_CLEAR_M = 0.12                # centre-to-centre distance to any OTHER object (mug r~0.045 + object r~0.05 + margin)
PLACE_EDGE_MARGIN_M = 0.03              # stay this far inside the planner's x/y limits
PLACE_MIN_RADIUS_M = 0.25               # keep away from the robot's own base column


OBSTACLE_X_RANGE = (-0.05, 0.85)       # where the table is, in base_link: objects here are physically on it
OBSTACLE_Y_RANGE = (-0.95, 0.95)
OBSTACLE_Z_RANGE = (-0.03, 0.32)       # centre heights that can belong to an object standing on the table


def _scene_points(scene, skip=()):
    """Positions of everything that is really on the table, for free-spot search. NOT only what the arm can
    pick: a bottle at the table edge (beyond the planner's y limit) still blocks a place next to it."""
    out = []
    for oid, o in (scene or {}).items():
        if oid in skip or not isinstance(o, dict) or o.get('stale'):
            continue
        if _is_number(o.get('confidence')) and o['confidence'] < 0.65:
            continue                                  # phantoms read 0.5-0.62; real objects 0.9+
        if not (_is_number(o.get('x')) and _is_number(o.get('y')) and _is_number(o.get('z'))):
            continue
        if (OBSTACLE_X_RANGE[0] <= o['x'] <= OBSTACLE_X_RANGE[1] and OBSTACLE_Y_RANGE[0] <= o['y'] <= OBSTACLE_Y_RANGE[1]
                and OBSTACLE_Z_RANGE[0] <= o['z'] <= OBSTACLE_Z_RANGE[1]):
            out.append((oid, float(o['x']), float(o['y'])))
    return out


def find_free_spot(target_xy, scene, prefer_xy, skip=(), rings=PLACE_RINGS_M, min_clear=PLACE_MIN_CLEAR_M):
    """A table spot a little way from the target object that is inside the workspace, clear of
    every other known object and not under the robot's base column. Among the candidates the one
    closest to prefer_xy (where the held object is now) wins, so the carry is short.
    Returns (x, y) or None."""
    others = _scene_points(scene, skip)
    xlo, xhi = sg.PLAN_X_RANGE[0] + PLACE_EDGE_MARGIN_M, sg.PLAN_X_RANGE[1] - PLACE_EDGE_MARGIN_M
    ylo, yhi = sg.PLAN_Y_RANGE[0] + PLACE_EDGE_MARGIN_M, sg.PLAN_Y_RANGE[1] - PLACE_EDGE_MARGIN_M
    best = None
    for r in rings:
        for k in range(24):
            a = math.radians(15 * k)
            x, y = target_xy[0] + r * math.cos(a), target_xy[1] + r * math.sin(a)
            if not (xlo <= x <= xhi and ylo <= y <= yhi) or math.hypot(x, y) < PLACE_MIN_RADIUS_M:
                continue
            if any(math.hypot(x - ox, y - oy) < min_clear for _, ox, oy in others):
                continue
            d = math.hypot(x - prefer_xy[0], y - prefer_xy[1])
            if best is None or d < best[0]:
                best = (d, x, y)
        if best is not None:
            break                                    # the closest ring that has any free spot
    return None if best is None else (round(best[1], 4), round(best[2], 4))


def _resolve_object(name, scene):
    """Scene id for `name` (an id, or a label such as 'mouse' -> the reachable one)."""
    if not isinstance(scene, dict) or not name:
        return None
    if name in scene:
        return name
    cands = [oid for oid, o in scene.items() if isinstance(o, dict) and o.get('label') == name
             and o.get('reachable') and not o.get('stale')]
    return cands[0] if cands else None


def fix_place_targets(plan, scene, current_xy=None):
    """Turn {"place", "near": B} into a concrete free (x, y) next to B; a place with x/y is moved to
    the nearest free spot if it would land on another object; {"here": true} is left for the arm.
    A place that cannot be satisfied is left with an 'unplaceable' reason, which validate_plan rejects."""
    if not isinstance(plan, list):
        return plan
    picked = None
    for step in plan:
        if not isinstance(step, dict):
            continue
        if step.get('action') == 'pick':
            picked = step.get('object_id')
        if step.get('action') != 'place' or step.get('here'):
            continue
        prefer = current_xy
        pobj = (scene or {}).get(picked) if picked else None
        if isinstance(pobj, dict) and _is_number(pobj.get('x')):
            prefer = (pobj['x'], pobj['y'])
        prefer = prefer or (0.4, 0.0)
        near = step.pop('near', None)
        if near:
            tid = _resolve_object(near, scene)
            if tid is None:
                step['unplaceable'] = f"object '{near}' is not in the scene"
                continue
            spot = find_free_spot((scene[tid]['x'], scene[tid]['y']), scene, prefer, skip=(picked, tid))
            if spot is None:
                step['unplaceable'] = f"no free spot next to '{tid}'"
                continue
            step['x'], step['y'] = spot
        elif _is_number(step.get('x')) and _is_number(step.get('y')):
            clash = [oid for oid, ox, oy in _scene_points(scene, (picked,))
                     if math.hypot(step['x'] - ox, step['y'] - oy) < PLACE_MIN_CLEAR_M]
            if clash:
                spot = find_free_spot((step['x'], step['y']), scene, prefer, skip=(picked,), rings=(0.0,) + PLACE_RINGS_M)
                if spot is None:
                    step['unplaceable'] = f"that spot is taken by {clash[0]} and no free one is nearby"
                    continue
                step['x'], step['y'] = spot
        else:
            step['unplaceable'] = 'place needs a target (near an object, x/y, or here)'
        step.pop('z', None)                          # the arm derives the set-down height from how it is holding the object
    return plan


DESTINATION_WORDS = ('next to', 'beside', 'near', 'by the', 'left', 'right', 'front', 'behind', ' on ', ' at ',
                     'over', 'under', 'between', 'x ', 'y ', ' to the', 'other side', 'away')


PICK_WORDS = ('pick', 'grab', 'take', 'lift', 'fetch', 'get the', 'grasp')


PLACE_WORDS = (' put ', ' place ', ' set it', ' next to', ' beside', ' aside', ' away', ' near ')


def plan_covers_command(steps, command):
    """(ok, why): does the model's plan contain what the command asks for? A "pick up the cup and put it next to
    the bowl" that came back as a lone move_to (2026-10-09) was approved and ran, leaving the cup where it was."""
    cmd = f' {(command or "").lower()} '
    acts = [st.get('action') for st in steps if isinstance(st, dict)]
    wants_pick = any(w in cmd for w in PICK_WORDS)
    if wants_pick and 'pick' not in acts:
        return False, 'the plan has no pick step'
    if wants_pick and any(w in cmd for w in PLACE_WORDS) and 'place' not in acts:
        return False, 'the plan has no place step'
    return True, ''


def normalize_place_here(plan, command):
    """"put it down" / "set it down" / "put it back" with no destination word means: where it was
    picked up. The model tends to invent coordinates for it (it once moved the mug 10 cm), so turn
    such a place into {"place", "here": true}."""
    if not isinstance(plan, list):
        return plan
    cmd = f' {(command or "").lower()} '
    if not any(w in cmd for w in ('put it down', 'set it down', 'put it back', 'place it down', 'put down', 'set down')):
        return plan
    if any(w in cmd for w in DESTINATION_WORDS) or any(ch.isdigit() for ch in cmd):
        return plan
    for st in plan:
        if isinstance(st, dict) and st.get('action') == 'place':
            for k in ('x', 'y', 'z', 'near', 'unplaceable'):
                st.pop(k, None)
            st['here'] = True
    if not any(w in cmd for w in PICK_WORDS):
        # "put it down" on its own releases what is already in the gripper. The model once added a pick of some
        # other object in front of it (2026-10-07), which the controller refused only because it was holding.
        only = [st for st in plan if isinstance(st, dict) and st.get('action') == 'place']
        if only:
            return only
    return plan


ASIDE_WORDS = ('aside', 'to the side', 'out of the way', 'somewhere else', 'over there', 'move it away',
               'put it away', 'set it aside', 'off to the side')
ASIDE_RINGS_M = (0.15, 0.19, 0.23)
NAMED_DESTINATION_WORDS = ('next to', 'beside', 'near the', 'near it', 'by the', 'on the left', 'on the right',
                           'in front', 'behind', 'at x', 'x ')


def normalize_place_aside(plan, command, scene):
    """"put it aside" / "out of the way": a free table spot 15-23 cm from where the object was picked up
    (the model otherwise invents coordinates, sometimes under the robot's base or at the table edge).
    Not applied when the command names a destination ("next to the mouse", "at x 0.3")."""
    if not isinstance(plan, list):
        return plan
    cmd = f' {(command or "").lower()} '
    if not any(w in cmd for w in ASIDE_WORDS) or any(w in cmd for w in NAMED_DESTINATION_WORDS):
        return plan
    picked = next((st.get('object_id') for st in plan if isinstance(st, dict) and st.get('action') == 'pick'), None)
    obj = (scene or {}).get(picked) if picked else None
    if not (isinstance(obj, dict) and _is_number(obj.get('x')) and _is_number(obj.get('y'))):
        return plan
    here = (obj['x'], obj['y'])
    spot = find_free_spot(here, scene, here, skip=(picked,), rings=ASIDE_RINGS_M)
    for st in plan:
        if isinstance(st, dict) and st.get('action') == 'place':
            for k in ('x', 'y', 'z', 'near', 'here', 'unplaceable'):
                st.pop(k, None)
            if spot is None:
                st['unplaceable'] = 'no free spot to the side of the object'
            else:
                st['x'], st['y'] = spot
    return plan


def fix_pick_target(plan, scene, table_top):
    """The model takes the first "mouse"/"cup" in the world state, which is often a phantom (a mouse at
    z=-0.037, below the table; a cup at the wrong height). If the object it picked is implausible and another
    reachable, non-stale object of the same label is plausible, pick that one (most confident first)."""
    if not isinstance(plan, list) or not isinstance(scene, dict) or table_top is None:
        return plan

    def plausible(o):
        return (isinstance(o, dict) and o.get('reachable') and not o.get('stale') and _is_number(o.get('z'))
                and table_top - 0.02 <= o['z'] <= table_top + 0.30)

    for st in plan:
        if not isinstance(st, dict) or st.get('action') != 'pick':
            continue
        chosen = scene.get(st.get('object_id'))
        if plausible(chosen):
            continue
        label = chosen.get('label') if isinstance(chosen, dict) else None
        cands = [(oid, o) for oid, o in scene.items() if plausible(o) and (label is None or o.get('label') == label)]
        if cands:
            best = max(cands, key=lambda c: c[1].get('confidence', 0))
            st['object_id'] = best[0]
    return plan


def drop_home_around_pick(plan, command):
    """A pick/place command never needs go_home unless it says so (the model added one before the pick; the
    arm went home first and the pick then failed)."""
    if not isinstance(plan, list) or not any(isinstance(s, dict) and s.get('action') == 'pick' for s in plan):
        return plan
    if any(w in (command or '').lower() for w in HOME_WORDS):
        return plan
    return [s for s in plan if not (isinstance(s, dict) and s.get('action') == 'go_home')]


# ── SAFETY VALIDATOR ─────────────────────────────────────────────────
def validate_plan(plan, scene, z_floor=0.08):
    """
    Scan entire plan for safety violations before first move.
    Returns (is_safe: bool, reason: str, warnings: list[str]).
    Warnings do not block the plan (e.g. picking a stale object).
    """
    warnings = []
    if not isinstance(plan, list) or len(plan) == 0:
        return False, "Plan is empty or not a list", warnings

    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            return False, f"Step {i}: not an object ({step!r})", warnings
        act = step.get('action', '')

        # Every coordinate present must be a real number (null z is possible
        # now that the scene reports unknown heights as null)
        for k in ['x', 'y', 'z', 'approach_z']:
            if k in step and not _is_number(step[k]):
                return False, f"Step {i} ({act}): {k}={step[k]!r} is not a number", warnings

        # Z floor check
        for k in ['z', 'approach_z']:
            if k in step and step[k] < z_floor:
                return False, f"Step {i} ({act}): {k}={step[k]:.3f} violates floor {z_floor:.3f}", warnings

        # approach_z must be high enough
        if act == 'pick':
            if step.get('approach_z', 1.0) < max(0.15, z_floor):
                return False, f"Step {i}: approach_z={step.get('approach_z')} too low", warnings
            # Object must be named and exist in scene
            oid = step.get('object_id')
            if not oid:
                return False, f"Step {i}: pick has no object_id", warnings
            if oid not in scene:
                return False, f"Step {i}: object_id '{oid}' not in scene", warnings
            obj = scene[oid] if isinstance(scene[oid], dict) else {}
            if obj.get('z') is None:
                return False, f"Step {i}: '{oid}' has unknown height", warnings
            # Stale objects: warning only — arm_controller uses the last seen position
            if obj.get('stale', False):
                warnings.append(f"Step {i}: object '{oid}' is stale but proceeding")

        if act == 'place' and step.get('unplaceable'):
            return False, f"Step {i}: cannot place — {step['unplaceable']}", warnings
        if act == 'place' and not step.get('here') and not (_is_number(step.get('x')) and _is_number(step.get('y'))):
            return False, f"Step {i}: place has no target", warnings
        # Place Z check
        if act == 'place' and step.get('z', 1.0) < max(0.10, z_floor):
            return False, f"Step {i}: place z={step.get('z'):.3f} too low", warnings

        # Workspace bounds — x upper bound matches arm_controller_node.py's
        # clamp (0.60, widened 2026-09-30; see its comment for why)
        if 'x' in step and not (sg.PLAN_X_RANGE[0] <= step['x'] <= sg.PLAN_X_RANGE[1]):
            return False, f"Step {i}: x={step['x']:.3f} outside workspace", warnings
        if 'y' in step and not (sg.PLAN_Y_RANGE[0] <= step['y'] <= sg.PLAN_Y_RANGE[1]):
            return False, f"Step {i}: y={step['y']:.3f} outside workspace", warnings

    return True, f"Plan approved — {len(plan)} steps", warnings


def _is_number(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


class LLMPlannerNode(Node):

    def __init__(self):
        super().__init__('llm_planner')

        # Declare model as a ROS2 parameter — change without recompiling
        self.declare_parameter('model', 'qwen2.5:7b')
        # A hung Ollama must not leave is_planning stuck True (every later
        # command would be ignored as "already planning").
        self.declare_parameter('llm_timeout_s', 60.0)
        self.declare_parameter('llm_keep_alive', '6h')   # how long Ollama keeps the model loaded between commands
        self.declare_parameter('max_scene_age_s', 3.0)
        self.model = self.get_parameter('model').value
        self.max_scene_age = float(self.get_parameter('max_scene_age_s').value)
        self.llm = ollama.Client(timeout=float(self.get_parameter('llm_timeout_s').value))
        self._scene_time = 0.0

        # ── state ────────────────────────────────────────────────────
        self.latest_scene    = {}
        self.latest_scene_raw = '{}'
        self.is_planning     = False   # prevent overlapping calls
        self._lock           = threading.Lock()

        # ── subscribers ──────────────────────────────────────────────
        self.create_subscription(
            String, '/scene_snapshot',
            self.scene_callback, 10)
        self.create_subscription(
            String, '/voice_command',
            self.command_callback, 10)

        # ── publishers ───────────────────────────────────────────────
        self.plan_pub   = self.create_publisher(String, '/action_plan',    10)
        self.status_pub = self.create_publisher(String, '/planner_status', 10)

        self.get_logger().info(
            f'LLMPlannerNode ready — model: {self.model}'
        )
        self._publish_status('READY — waiting for voice command')

    # ── CALLBACKS ────────────────────────────────────────────────────
    def scene_callback(self, msg):
        """Cache latest scene — runs at 5Hz, never blocks."""
        try:
            self.latest_scene     = json.loads(msg.data)
            self.latest_scene_raw = msg.data
            self._scene_time      = time.time()
        except json.JSONDecodeError:
            pass

    def command_callback(self, msg):
        """Trigger planning on voice command — run in separate thread."""
        command = msg.data.strip()
        if not command:
            return

        with self._lock:
            if self.is_planning:
                self.get_logger().warn(
                    'Already planning — ignoring new command. '
                    'Wait for current plan to complete.'
                )
                return
            self.is_planning = True

        # Run LLM call in background thread — keeps spin() responsive
        t = threading.Thread(
            target=self._plan_and_publish,
            args=(command,),
            daemon=True
        )
        t.start()

    # ── CORE PLANNING LOGIC ──────────────────────────────────────────
    def _plan_and_publish(self, command):
        """Called in background thread. Calls LLM, validates, publishes."""
        try:
            self.get_logger().info(f'Command: "{command}"')
            self._publish_status(f'PLANNING: {command}')

            geom, geom_err = sg.load_geometry()
            z_floor = sg.flange_floor_z(geom)
            if geom is None:
                self.get_logger().warn(
                    f'No table geometry ({geom_err}) — using conservative flange floor '
                    f'z={z_floor:.2f}. Record the table in the dashboard.')
            system_prompt = SYSTEM_PROMPT.replace('{Z_FLOOR}', f'{z_floor:.2f}')

            scene_age = time.time() - self._scene_time
            if self._scene_time and scene_age > self.max_scene_age:
                self._publish_status(f'ERROR: scene data is {scene_age:.0f}s old — is scene_graph_node running?')
                self.get_logger().warn(
                    f'Refusing to plan on a {scene_age:.1f}s-old scene snapshot '
                    f'(limit {self.max_scene_age:.0f}s).')
                return
            if not self.latest_scene:
                self._publish_status('ERROR: no scene data yet')
                self.get_logger().warn('No scene data — publish to /scene_snapshot first')
                return

            # Build prompt — only send reachable objects. Unreachable
            # background clutter (false-positive YOLO/color detections
            # outside the workspace, common with this scene's noise) was
            # blowing the full scene up past 40KB / ~11k tokens, which
            # overflows Ollama's default context window and made the model
            # return an empty plan with empty reasoning every time rather
            # than actually failing loudly. Validation below still checks
            # against the FULL scene, not this filtered view, so the plan
            # is never less safe — the LLM just can't reference something
            # it was never told the arm can't reach anyway.
            prompt_scene = {
                obj_id: obj for obj_id, obj in self.latest_scene.items()
                if obj.get('reachable') and not obj.get('stale', False)
            }
            if not prompt_scene:
                self.get_logger().warn(
                    f'No reachable objects in scene ({len(self.latest_scene)} '
                    f'total, 0 reachable) — sending the LLM an empty world state.')
            user_msg = (
                f"CURRENT WORLD STATE:\n"
                f"{json.dumps(prompt_scene, indent=2)}\n\n"
                f"ENGINEER COMMAND: \"{command}\"\n\n"
                f"Reason step by step, then output the JSON."
            )

            # Call LLM. An empty or non-JSON reply happens right after Ollama (re)loads the model; retry rather
            # than abandon a command the operator already gave (2026-10-06: "JSON parse error", nothing happened).
            t0 = time.time()
            parsed = None
            for attempt in (1, 2, 3):
                response = self.llm.chat(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user',   'content': user_msg},
                    ],
                    format='json',
                    keep_alive=str(self.get_parameter('llm_keep_alive').value)   # a reload from the external disk took 78 s
                )
                raw = (response['message']['content'] or '').strip()
                why = 'an empty/invalid reply'
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        try:
                            steps = [json.loads(st) if isinstance(st, str) else st for st in parsed.get('plan', []) if st]
                        except (json.JSONDecodeError, TypeError):
                            steps = None
                        if steps is None:
                            why = 'a plan with unreadable steps'
                        else:
                            covered, why = plan_covers_command(steps, command)
                            if covered:
                                break
                except json.JSONDecodeError:
                    pass
                self.get_logger().warn(f'LLM returned {why} (attempt {attempt}/3) — retrying')
                parsed = None
                time.sleep(1.0)
            if parsed is None:
                raise json.JSONDecodeError('the language model returned no valid JSON after 3 attempts', '', 0)
            latency = time.time() - t0
            plan      = parsed.get('plan', [])
            # Sanitise — ensure each step is a dict not a string
            plan = [
                json.loads(s) if isinstance(s, str) else s
                for s in plan
                if s  # skip empty
            ]
            reasoning = parsed.get('reasoning', '')

            self.get_logger().info(
                f'LLM response in {latency:.1f}s — '
                f'{len(plan)} steps | Reasoning: {reasoning[:80]}...'
            )

            # Re-derive move_to heights from the matched scene object rather
            # than trusting the LLM's "z + 0.15" arithmetic (see comment on
            # fix_move_to_heights) — then validate as usual.
            plan = fix_pick_target(plan, self.latest_scene, geom['table_top_z'] if geom else None)
            plan = drop_home_around_pick(plan, command)
            plan = fix_move_to_heights(plan, self.latest_scene, z_floor)
            plan = fix_pick_heights(plan, self.latest_scene, z_floor)
            plan = trim_pick_extras(plan, command)
            plan = normalize_place_here(plan, command)
            plan = normalize_place_aside(plan, command, self.latest_scene)
            plan = fix_place_targets(plan, self.latest_scene)
            ok, reason, warnings = validate_plan(plan, self.latest_scene, z_floor)
            for w in warnings:
                self.get_logger().warn(w)

            if not ok:
                self.get_logger().warn(f'Plan REJECTED: {reason}')
                self._publish_status(f'REJECTED: {reason}')
                return

            # Publish approved plan
            self.get_logger().info(f'Plan APPROVED: {reason}')
            self._publish_status(f'EXECUTING: {len(plan)} steps')

            out = String()
            out.data = json.dumps({
                'command':   command,
                'reasoning': reasoning,
                'plan':      plan,
                'latency_s': round(latency, 2),
                'model':     self.model,
            })
            self.plan_pub.publish(out)

            # Log each step
            for i, step in enumerate(plan):
                self.get_logger().info(f'  Step {i+1}: {json.dumps(step)}')

        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON parse error: {e}')
            self._publish_status(f'ERROR: JSON parse failed')
        except Exception as e:
            self.get_logger().error(f'Planning failed: {e}')
            self._publish_status(f'ERROR: {str(e)[:80]}')
        finally:
            with self._lock:
                self.is_planning = False
            self._publish_status('READY — waiting for voice command')

    # ── HELPERS ──────────────────────────────────────────────────────
    def _publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LLMPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
