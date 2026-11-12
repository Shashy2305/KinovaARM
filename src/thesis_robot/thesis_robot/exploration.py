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
    grasp_z_offset    added to the computed grasp height (ws.pick_heights)           0.0       -0.015 - 0.015  -0.03 - 0.03
    hover_above_m     fingertip clearance above the object while centring            0.12      0.08 - 0.16     0.04 - 0.20

Added 2026-11-09 (RL-project handoff Task 10, rebuilding this to match the doc's own spec -- the
original version above only randomized motion timing, never the grasp parameters the project needs):
`grasp_z_offset` and `hover_above_m` are the two config offsets that make the otherwise-deterministic
`ws.pick_heights()` formula tunable at all (Task 9 found grasp_z/hover_z have no free parameter
otherwise). Their HARD_LIMITS are not a judgement call like the ones above -- they are exactly the
envelope `scripts/dev/grasp_param_safety_sweep.py` plan-checked against the live `move_group` on 18
real logged pick positions (10 cup, 8 mouse): every value in that envelope produced a collision-free
plan on every sampled position; values outside it were never tested, not shown unsafe, so the hard
limit stops at the tested edge rather than guessing further (see
reward_model_rlproject/analysis/task9_parameter_ranges/report.md).

Grasp METHOD is drawn separately (not through `ranges()`/`clamp()`, which only handle numeric spans):
`sample()` also returns `'method'`, `rng.choice(['top', 'side'])` with equal probability. There is no
hard limit to clamp into -- method isn't numeric -- so "respecting any rule that makes one method
impossible for an object" is handled downstream, not here: `arm_controller_node.py` forces the drawn
method into the EXISTING `side_grasp` ranking ('only' for a 'side' draw, 'off' for a 'top' draw), and
the existing "no way to grasp it here" refusal fires exactly when the draw was geometrically
impossible -- that refusal, logged as a real failed attempt, *is* the data this campaign wants (an
infeasible side grasp is a legitimate negative example, not noise to filter out).
"""
import os
import random

import yaml

RANGES_FILE = os.path.expanduser('~/.ros/exploration_ranges.yaml')
DEFAULT_RANGES = {
    'smooth_vel_scale': (0.22, 0.34), 'gentle_vel_scale': (0.25, 0.50), 'place_settle_s': (0.6, 1.2), 'touch_step_m': (0.0015, 0.0030),
    'touch_relieve_m': (0.0010, 0.0025), 'take_lift_m': (0.015, 0.030), 'cup_in_extra_m': (0.008, 0.016), 'take_in_m': (0.008, 0.016),
    'grasp_z_offset': (-0.015, 0.015), 'hover_above_m': (0.08, 0.16),
}
HARD_LIMITS = {
    'smooth_vel_scale': (0.15, 0.40), 'gentle_vel_scale': (0.20, 0.60), 'place_settle_s': (0.3, 2.0), 'touch_step_m': (0.001, 0.004),
    'touch_relieve_m': (0.0, 0.003), 'take_lift_m': (0.010, 0.040), 'cup_in_extra_m': (0.0, 0.020), 'take_in_m': (0.0, 0.020),
    'grasp_z_offset': (-0.03, 0.03), 'hover_above_m': (0.04, 0.20),
}
METHOD_CHOICES = ('top', 'side')


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


def sample(rng=None, path=None, seed=None):
    """One draw: {name: value} for every numeric parameter (uniform inside each clamped range, rounded
    for the log), plus 'method' ('top' or 'side', equal probability) and 'seed' (the campaign seed, so
    it rides along in every record without a separate schema field)."""
    rng = rng or random.Random()
    draw = {name: round(clamp(name, rng.uniform(lo, hi)), 5) for name, (lo, hi) in ranges(path).items()}
    draw['method'] = rng.choice(METHOD_CHOICES)
    draw['seed'] = seed
    return draw
