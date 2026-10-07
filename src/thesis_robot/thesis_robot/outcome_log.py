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

DEFAULT_DIR = os.path.expanduser('~/.ros/outcomes')
MAX_FILE_BYTES = 20 * 1024 * 1024


class OutcomeLog:
    def __init__(self, directory=DEFAULT_DIR):
        self.dir = directory
        self._lock = threading.Lock()

    def _path(self):
        return os.path.join(self.dir, time.strftime('outcomes-%Y-%m.jsonl'))

    def append(self, record):
        rec = {'ts': round(time.time(), 3), 'iso': time.strftime('%Y-%m-%dT%H:%M:%S')}
        rec.update(record)
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
