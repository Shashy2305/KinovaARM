"""What the coffee routine learns from its own runs (2026-11-03).

After a run the cup is looked up where it was set down. The difference to where the arm was COMMANDED to put it is the systematic placement error (the mug landed 6 cm off its target in the first
full live run); it is averaged over the runs and subtracted from the next command, so the error shrinks run by run. Kept in ~/.ros/coffee_learn.yaml; a hand-edit or deleting the file resets it.

    offset   estimated displacement (observed - commanded), metres, in base_link x / y
    n        runs it is based on
    history  the last few measured offsets (for the report)
A run is ignored when the cup was not found near its spot (it may have tipped) or when the measurement is wildly off (a different cup, a bumped table): a wrong sample must not drag the correction.
"""
import os
import time

import yaml

LEARN_FILE = os.path.expanduser('~/.ros/coffee_learn.yaml')
ALPHA = 0.5                  # weight of a new measurement once there is a history (the first one is taken in full)
OFFSET_CAP_M = 0.08          # the correction never exceeds this per axis
OUTLIER_M = 0.15             # a measured offset beyond this is not used
HISTORY = 12


def _load(path=None):
    try:
        with open(path or LEARN_FILE) as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def offset(key, path=None):
    """(dx, dy) the arm's placement is estimated to be off by for this kind of placement (0, 0 when nothing is learned)."""
    rec = _load(path).get(key) or {}
    try:
        return float(rec['offset'][0]), float(rec['offset'][1])
    except (KeyError, TypeError, ValueError, IndexError):
        return 0.0, 0.0


def corrected(key, target, path=None):
    """The point to COMMAND so that the object ends on `target`: the target minus the learned offset."""
    dx, dy = offset(key, path)
    return float(target[0]) - dx, float(target[1]) - dy


def update(key, commanded, observed, path=None):
    """Fold one run in. Returns (used: bool, text). `commanded` and `observed` are (x, y) in base_link."""
    try:
        mx, my = float(observed[0]) - float(commanded[0]), float(observed[1]) - float(commanded[1])
    except (TypeError, ValueError, IndexError):
        return False, 'no usable measurement'
    if max(abs(mx), abs(my)) > OUTLIER_M:
        return False, f'measured offset ({mx * 100:+.1f}, {my * 100:+.1f}) cm is beyond {OUTLIER_M * 100:.0f} cm: not used'
    data = _load(path)
    rec = data.get(key) if isinstance(data.get(key), dict) else {}
    n = int(rec.get('n', 0) or 0)
    ox, oy = offset(key, path)
    a = 1.0 if n == 0 else ALPHA
    nx, ny = (1 - a) * ox + a * mx, (1 - a) * oy + a * my
    nx, ny = max(-OFFSET_CAP_M, min(OFFSET_CAP_M, nx)), max(-OFFSET_CAP_M, min(OFFSET_CAP_M, ny))
    hist = list(rec.get('history') or [])[-(HISTORY - 1):] + [[round(mx, 4), round(my, 4), time.strftime('%Y-%m-%d %H:%M')]]
    data[key] = {'offset': [round(nx, 4), round(ny, 4)], 'n': n + 1, 'history': hist}
    try:
        os.makedirs(os.path.dirname(path or LEARN_FILE), exist_ok=True)
        with open(path or LEARN_FILE, 'w') as f:
            yaml.safe_dump(data, f)
    except OSError as e:
        return False, f'could not save ({e})'
    return True, f'{key}: measured offset ({mx * 100:+.1f}, {my * 100:+.1f}) cm -> learned offset ({nx * 100:+.1f}, {ny * 100:+.1f}) cm from {n + 1} run(s)'
