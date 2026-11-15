"""
Outcome log: one JSON line per pick, place and command, so the system can be measured and (later) learn from it.

Pure Python, no ROS. Every call is wrapped by the caller (arm_controller_node) so that a logging problem can never
stop or change a move. Files live in ~/.ros/outcomes/outcomes-YYYY-MM.jsonl (outside the git repo: they hold
this table's data, not code).

A record is {'ts', 'kind', 'live', ...}; kinds:
  pick     one attempt: label, position, neighbours, strategy (full_open | narrowed | turned_90), alignment, centring,
           grip finger readings, per-step durations, ok, failed_step, reason
  place    target, carry/set-down heights, strategy, ok, failed_step, reason
  command  the typed/spoken command, the plan, ok, failed step, duration
"""
import json
import math
import os
import threading
import time
import zlib

from thesis_robot import code_version

DEFAULT_DIR = os.path.expanduser('~/.ros/outcomes')
MAX_FILE_BYTES = 20 * 1024 * 1024

TEST_POOL_FRACTION = 5   # 1 in this many scenes goes to test_pool (an 80/20 split)


def pool_for_scene(scene_id):
    """'train_pool' or 'test_pool' for a scene_id, stable across every record of that scene (an empty
    scene_id, i.e. the operator has not tagged a scene yet, gets '' -- never silently defaulted to a
    pool, since that would mislabel it). A hash split, not a counter, so re-running the same scene_id
    later (e.g. a repeated layout) always lands in the same pool."""
    if not scene_id:
        return ''
    return 'test_pool' if zlib.crc32(scene_id.encode()) % TEST_POOL_FRACTION == 0 else 'train_pool'


# The known-before-acting fields for each kind (2026-11-05 label audit / Task 8 decision-time-fields
# review, reward_model_rlproject/analysis/task8_decision_time_fields/report.md): exactly the fields
# confirmed, by reading the code, to be computed and logged before the first physical motion of this
# attempt. Deliberately excludes 'strategy' (pick's can be overwritten mid-pick once the grasp-width /
# neighbour check runs) and anything decided in or after 'choose how to grasp it' (method, side_az,
# blocker -- all computed from a wrist-camera view only available after the arm has already hovered).
PRE_ACTION_FIELDS = {
    'pick': ('object_id', 'label', 'xy', 'z', 'attempt', 'neighbours', 'grasp_z', 'hover_z',
             'tip_clearance_m', 'scene_conf', 'unknown_near'),
    'place': ('object_id', 'label', 'requested', 'target', 'carry_z', 'set_z', 'neighbours', 'unknown_near'),
}

# Task 7 failure taxonomy (reward_model_rlproject/analysis/task7_failure_taxonomy/report.md), ported
# here so NEW records are categorized automatically instead of only in the offline analysis script.
# Kept deliberately small and close to the report's own rules -- a change to the taxonomy should change
# both; this is not a separate, drifting copy of the logic, it IS the logic (the offline script's
# EXACT_RULES/SUBSTRING_RULES predate this and are the source the categories below are copied from).
_FAILURE_EXACT = {
    ('pick', 'hover', ''): ('planning-or-kinematics', 'high'),
    ('pick', 'raise', ''): ('planning-or-kinematics', 'high'),
    ('pick', 'descend', ''): ('planning-or-kinematics', 'high'),
    ('pick', 'check the grip', ''): ('parameter-related', 'high'),
    ('pick', 'check the grip (3 cm up)', ''): ('parameter-related', 'high'),
    ('pick', 'check it is still held', ''): ('parameter-related', 'high'),
    ('pick', 'turn the fingers away from any handle', ''): ('planning-or-kinematics', 'low'),
    ('pick', 'centre on the object (wrist camera)', ''): ('perception-related', 'high'),
    ('place', 'lift', ''): ('planning-or-kinematics', 'high'),
    ('place', 'carry', ''): ('planning-or-kinematics', 'high'),
    ('place', 'lower', ''): ('planning-or-kinematics', 'high'),
    ('place', 'place', ''): ('other', 'low'),
}
_FAILURE_SUBSTRING = (
    ('too close to grip', 'parameter-related', 'high'),
    ('would hit the', 'parameter-related', 'high'),
    ('passes within', 'parameter-related', 'high'),
    ('no side grasp of the cup is reachable', 'planning-or-kinematics', 'high'),
    ('no way to grasp the cup here', 'planning-or-kinematics', 'high'),
    ('no way to reach the approach pose', 'planning-or-kinematics', 'high'),
    ('unsafe:', 'planning-or-kinematics', 'high'),
    ('planning scene not ready', 'hardware-or-system', 'high'),
    ('move_group did not accept', 'hardware-or-system', 'high'),
    ('already holding something', 'other', 'high'),
    ('do not know how the object is held', 'other', 'high'),
    ('not inside the window around', 'perception-related', 'low'),
    ('no fresh white cup in the scene', 'perception-related', 'high'),
)


def categorize_failure(kind, failed_step, reason):
    """(category, confidence) for a failed pick/place, or (None, None) when no rule matches (left for
    a human look, same as the offline taxonomy's own fallback) -- never guessed past what the Task 7
    report actually checked."""
    key = (kind, failed_step or '', reason or '')
    if key in _FAILURE_EXACT:
        return _FAILURE_EXACT[key]
    r = reason or ''
    for pat, cat, conf in _FAILURE_SUBSTRING:
        if pat in r:
            return cat, conf
    return None, None


class OutcomeLog:
    def __init__(self, directory=DEFAULT_DIR):
        self.dir = directory
        self._lock = threading.Lock()

    def _path(self):
        return os.path.join(self.dir, time.strftime('outcomes-%Y-%m.jsonl'))

    def append(self, record):
        rec = {'ts': round(time.time(), 3), 'iso': time.strftime('%Y-%m-%dT%H:%M:%S')}
        rec.update(record)
        if 'code' not in rec:
            try:
                rec['code'] = code_version.tag()                   # which commit and settings wrote this record (never allowed to stop a log)
            except Exception:
                pass
        # Schema fields every record carries regardless of kind (2026-11-13, RL-project handoff):
        # per-process/per-schema-version facts, not a per-record choice, so every caller gets them for free.
        rec.setdefault('schema_version', code_version.SCHEMA_VERSION)
        rec.setdefault('session_id', code_version.SESSION_ID)
        rec.setdefault('grip_check_version', code_version.GRIP_CHECK_VERSION)
        line = json.dumps(rec, default=_json_default, separators=(',', ':'))
        with self._lock:
            os.makedirs(self.dir, exist_ok=True)
            path = self._path()
            if os.path.exists(path) and os.path.getsize(path) > MAX_FILE_BYTES:
                os.replace(path, path + f'.{int(time.time())}')
            with open(path, 'a') as f:
                f.write(line + '\n')
        return rec


def _json_default(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


class Attempt:
    """Notes collected while one pick/place runs; finish() turns them into a record."""

    def __init__(self, kind, **fields):
        self.kind = kind
        self.t0 = time.monotonic()
        self.data = dict(fields)
        self.events = []
        self.steps = {}                 # step name -> seconds
        self.grip = []                  # finger_joint readings at each grip check

    def note(self, **kw):
        self.data.update(kw)

    def event(self, name, **kw):
        self.events.append({'t': round(time.monotonic() - self.t0, 2), 'what': name, **kw})

    def step_time(self, name, seconds):
        self.steps[name] = round(seconds, 2)

    def grip_reading(self, name, value):
        self.grip.append({'at': name, 'finger': None if value is None else round(float(value), 3)})

    def finish(self, ok, failed_step=None, reason=None):
        rec = {'kind': self.kind, **self.data, 'ok': bool(ok), 'failed_step': failed_step, 'reason': reason,
               'seconds': round(time.monotonic() - self.t0, 1), 'steps': self.steps, 'grip': self.grip,
               'events': self.events}
        pre_fields = PRE_ACTION_FIELDS.get(self.kind)
        if pre_fields:
            rec['pre_action_features'] = {k: self.data[k] for k in pre_fields if k in self.data}
        if not ok:
            cat, conf = categorize_failure(self.kind, failed_step, reason)
            if cat:
                rec['failure_category'], rec['failure_category_confidence'] = cat, conf
        return rec


def neighbours(obstacles, center, k=4):
    """[(label, x, y)] -> the k nearest to `center` as [{'label', 'dist_m', 'bearing_deg'}], nearest first."""
    out = []
    for label, x, y in obstacles:
        dx, dy = x - center[0], y - center[1]
        out.append({'label': label, 'dist_m': round(math.hypot(dx, dy), 3),
                    'bearing_deg': round(math.degrees(math.atan2(dy, dx)), 0)})
    out.sort(key=lambda n: n['dist_m'])
    return out[:k]


def read_records(directory=DEFAULT_DIR):
    """Every record in the directory, oldest first. Unreadable lines are skipped."""
    out = []
    if not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        if '.jsonl' not in name:
            continue
        try:
            with open(os.path.join(directory, name)) as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    out.sort(key=lambda r: r.get('ts', 0))
    return out
