"""Randomized-parameter mode for an exploration campaign (2026-11-05). OFF by default; safe ranges only.

With the arm controller parameter `exploration_mode:=true`, each COMMAND (episode) draws a fresh value for every parameter below, uniformly inside its range, applies it for that command only, writes
the drawn values into every record the command produces (`explore` dict in the pick / place / command records), and restores the defaults afterwards. Nothing else changes: the safety checks, the
workspace limits, the table and link clearances, the joint-delta limits and the grip checks are never touched.

Ranges: DEFAULT_RANGES below (centred on today's values), optionally narrowed by ~/.ros/exploration_ranges.yaml ({name: [low, high]}). Whatever the file says, every value is CLAMPED into HARD_LIMITS:
the file can narrow the ranges but cannot widen them past what was judged safe. `exploration_seed` (int, 0 = random) makes a campaign reproducible.

    name              what it changes                                              default   range           hard limits
    smooth_vel_scale  peak joint speed of every move, fraction of the joint limit   0.30      0.22 - 0.34     0.15 - 0.40
    gentle_vel_scale  extra slow-down factor while setting down / backing away      0.35      0.25 - 0.50     0.20 - 0.60
    place_settle_s    pause after the fingers open, before anything moves           0.8       0.6 - 1.2       0.3 - 2.0
    touch_step_m      step of the touch-down lowering                              0.002     0.0015 - 0.003  0.001 - 0.004
    touch_relieve_m   lift after the touch of a side-held cup                       0.0015    0.001 - 0.0025  0.0 - 0.003
    take_lift_m       lift of a cup taken out of the tray before it is drawn out     0.02      0.015 - 0.03    0.01 - 0.04
    cup_in_extra_m    how much deeper than the seen centre a cup is gripped          0.012     0.008 - 0.016   0.0 - 0.02
    take_in_m         how much further in than the taught pose the tray take goes    0.012     0.008 - 0.016   0.0 - 0.02
"""
import os
import random

import yaml

RANGES_FILE = os.path.expanduser('~/.ros/exploration_ranges.yaml')
DEFAULT_RANGES = {
    'smooth_vel_scale': (0.22, 0.34), 'gentle_vel_scale': (0.25, 0.50), 'place_settle_s': (0.6, 1.2), 'touch_step_m': (0.0015, 0.0030),
    'touch_relieve_m': (0.0010, 0.0025), 'take_lift_m': (0.015, 0.030), 'cup_in_extra_m': (0.008, 0.016), 'take_in_m': (0.008, 0.016),
}
HARD_LIMITS = {
    'smooth_vel_scale': (0.15, 0.40), 'gentle_vel_scale': (0.20, 0.60), 'place_settle_s': (0.3, 2.0), 'touch_step_m': (0.001, 0.004),
    'touch_relieve_m': (0.0, 0.003), 'take_lift_m': (0.010, 0.040), 'cup_in_extra_m': (0.0, 0.020), 'take_in_m': (0.0, 0.020),
}


def clamp(name, value):
    lo, hi = HARD_LIMITS[name]
    return max(lo, min(hi, float(value)))


def ranges(path=None):
    """The ranges in force: the defaults, narrowed by the file when there is one, every end clamped into HARD_LIMITS, never inverted."""
    out = {}
    try:
        with open(path or RANGES_FILE) as f:
            custom = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        custom = {}
    for name, (lo, hi) in DEFAULT_RANGES.items():
        try:
            c_lo, c_hi = custom[name]
            lo, hi = float(c_lo), float(c_hi)
        except (KeyError, TypeError, ValueError):
            pass
        lo, hi = clamp(name, lo), clamp(name, hi)
        out[name] = (min(lo, hi), max(lo, hi))
    return out


def sample(rng=None, path=None):
    """One draw: {name: value}, uniform inside each (clamped) range, rounded for the log."""
    rng = rng or random.Random()
    return {name: round(clamp(name, rng.uniform(lo, hi)), 5) for name, (lo, hi) in ranges(path).items()}
