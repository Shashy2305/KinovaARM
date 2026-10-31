"""
Mapping a 3Dconnexion SpaceMouse onto the dashboard jog's Cartesian command (pure numpy-free python, no ROS, no device access: see scripts/spacemouse_jog.py).

The jog takes {'mode': 'cart', 'v': [x, y, z, rot x, rot y, rot z] each in -1..1, 'speed': 0..1, 'frame': 'base'|'tool'}. The SpaceMouse gives six deflections (its own x, y, z, rx, ry, rz) as
absolute positions of the cap, about +-350 at full push. A mapping says, for each of the jog's six axes, which device axis drives it and with which sign.

Safety shaping done here, before anything reaches the arm:
  * dead zone: a resting cap is never exactly zero, small deflections are ignored;
  * a gentle curve (expo) so small pushes give very small speeds;
  * translation OR rotation, never both: the larger of the two groups wins and the other is zeroed (a push on the cap always has a little twist in it);
  * one dominant axis inside the winning group only when `dominant` is set (a fixed, easy-to-predict motion for the first tests).
"""
import json
import os

AXIS_NAMES = ('x', 'y', 'z', 'rx', 'ry', 'rz')
MAP_FILE = os.path.expanduser('~/.ros/spacemouse_map.json')
# jog axis -> (device axis index, sign). A first guess for a user sitting behind the robot looking along +x: the probe (scripts/spacemouse_jog.py --probe) replaces it with what the user's own
# gestures show.
DEFAULT_MAP = {'x': (1, -1), 'y': (0, 1), 'z': (2, -1), 'rx': (4, 1), 'ry': (3, -1), 'rz': (5, -1)}


def load_map(path=MAP_FILE):
    """The saved mapping, or DEFAULT_MAP. Anything malformed falls back to the default (never raises)."""
    try:
        with open(path) as f:
            raw = json.load(f)
        out = {}
        for k in AXIS_NAMES:
            idx, sign = raw[k]
            idx, sign = int(idx), int(sign)
            if not (0 <= idx <= 5 and sign in (-1, 1)):
                return dict(DEFAULT_MAP)
            out[k] = (idx, sign)
        if len({v[0] for v in out.values()}) != 6:
            return dict(DEFAULT_MAP)                         # two jog axes on one device axis: not a mapping
        return out
    except (OSError, ValueError, KeyError, TypeError):
        return dict(DEFAULT_MAP)


def save_map(mapping, path=MAP_FILE):
    with open(path, 'w') as f:
        json.dump({k: [int(v[0]), int(v[1])] for k, v in mapping.items()}, f, indent=1)


def normalise(raw, full_scale):
    """Device counts -> -1..1 (clipped)."""
    return max(-1.0, min(1.0, float(raw) / float(full_scale))) if full_scale else 0.0


def shape(x, deadzone=0.12, expo=2.0):
    """Dead zone with a smooth start, then a power curve: 0 inside the dead zone, 1 at full push, continuous in between."""
    x = max(-1.0, min(1.0, float(x)))
    a = abs(x)
    if a <= deadzone:
        return 0.0
    s = (a - deadzone) / (1.0 - deadzone)
    return (s ** expo) * (1.0 if x > 0 else -1.0)


def to_jog_vector(axes, mapping=None, deadzone=0.12, expo=2.0, dominant=False):
    """axes: six normalised device deflections (-1..1). Returns the jog's v (six numbers)."""
    mapping = mapping or DEFAULT_MAP
    v = []
    for name in AXIS_NAMES:
        idx, sign = mapping[name]
        v.append(sign * shape(axes[idx], deadzone, expo))
    trans, rot = v[:3], v[3:]
    if max(abs(t) for t in trans) >= max(abs(r) for r in rot):
        rot = [0.0, 0.0, 0.0]
    else:
        trans = [0.0, 0.0, 0.0]
    group = trans if any(trans) else rot
    if dominant and any(group):
        k = max(range(3), key=lambda i: abs(group[i]))
        group[:] = [g if i == k else 0.0 for i, g in enumerate(group)]
    return [float(c) for c in trans + rot]


def deadman_ok(buttons, mode='any'):
    """buttons: {code: bool}. mode 'any' (either button), 'left' (0x100), 'right' (0x101)."""
    left, right = bool(buttons.get(0x100)), bool(buttons.get(0x101))
    if mode == 'left':
        return left
    if mode == 'right':
        return right
    return left or right


def mapping_from_probe(results):
    """results: {jog axis name: (device axis index, sign)} from the guided gestures. Returns a valid mapping or raises ValueError (two gestures landing on one device axis, a missing gesture)."""
    out = {}
    for k in AXIS_NAMES:
        if k not in results or results[k] is None:
            raise ValueError(f'no clear gesture was seen for {k}')
        out[k] = (int(results[k][0]), int(results[k][1]))
    if len({v[0] for v in out.values()}) != 6:
        raise ValueError('two gestures moved the same device axis')
    return out
