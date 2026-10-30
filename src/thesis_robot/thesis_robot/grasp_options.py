"""
Which WAY to grasp an object (pure numpy/scipy, no ROS): from above, leaning, or from the SIDE with the tool horizontal.

The side grasp is the way a person holds a cup: the tool points horizontally at the object, the pads close across it just under the rim, the object stays upright. It was shown by hand on
2026-10-26 (docs/demonstrations/cup_side_grasp_2026-10-26/pose.json) and its orientation is taken from that demonstration, turned about the vertical to any azimuth.

Frames: azimuth az = the direction the fingertips lead (degrees from base +x, counter-clockwise from above); u = (cos az, sin az). t = distance along u from the grasp centre, l = distance
across it. The pad midpoint sits PAD_REACH_M from the flange along the tool axis.

What is decided here is GEOMETRY and PREFERENCE only. Whether the arm can reach a candidate is the IK's answer (arm_controller_node tries the ranked list in order).
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation as R

from thesis_robot import angled_approach as aa
from thesis_robot import wrist_servo as ws

# the demonstration (flange orientation x, y, z, w; azimuth and lean of its tool axis)
DEMO_QUAT_XYZW = (0.7387, -0.0282, 0.0167, 0.6733)
DEMO_AZ_DEG = 269.2
SIDE_TILT_DEG = 84.7                 # 5.3 deg below horizontal, as demonstrated (the fingers slope slightly down onto the object)
PAD_REACH_M = 0.185                  # flange -> pad midpoint along the tool axis (0.177 open, 0.193 closed on the cup)
STANDOFF_M = 0.17                    # the pregrasp sits this far back along the tool axis. 0.12 left ~3 cm between the fingertips and the cup rim: a 3 cm disagreement of the cup position knocked a cup over (2026-10-29)
HIGH_CLEAR_M = 0.12                  # the first pose is this much higher still, so the swing from the hover passes over the object
RIM_BELOW_M = 0.008                  # pads this far under the rim of a short object (demo: 7 mm)
MIN_SIDE_HEIGHT_M = 0.07             # shorter objects have no side to hold: the pads would dip under the table margin
PAD_MIN_ABOVE_TABLE_M = 0.06         # lowest pad-frame height over the table (the trajectory check wants 0.05)
AZIMUTHS = tuple(range(0, 360, 45))
# The side grasp is only used where it has been proven on the arm (2026-10-29: pick + put-down of a cup at x 0.37-0.45, y 0.0-0.10, six times). Outside this box the straight-down grasp is used:
# a cup at x = 0.58 got a side grasp that needed a turned wrist and a 28 s swing. Widen it only after testing a position.
SIDE_ZONE_X = (0.30, 0.52)
SIDE_ZONE_Y = (-0.22, 0.22)


def in_side_zone(centre_xy):
    return SIDE_ZONE_X[0] <= centre_xy[0] <= SIDE_ZONE_X[1] and SIDE_ZONE_Y[0] <= centre_xy[1] <= SIDE_ZONE_Y[1]
# With the tool horizontal the flange sits ~0.3 m behind the object, so its x/y can lie outside the planner's targets for the OBJECT (safety_geometry.PLAN_*_RANGE). The object must still be inside
# those; the flange only has to stay over the table side of the base. The trajectory check (links over the table, in front of the rear limit) still covers every path.
FLANGE_X_RANGE = (-0.05, 0.90)
FLANGE_Y_RANGE = (-0.70, 0.70)


def flange_inside(p):
    return FLANGE_X_RANGE[0] <= p[0] <= FLANGE_X_RANGE[1] and FLANGE_Y_RANGE[0] <= p[1] <= FLANGE_Y_RANGE[1]

# the gripper's footprint seen from above, per zone: (t_lo, t_hi, half width, how far below the pad height it reaches)
FINGER_HALF_SPAN_M = ws.FINGER_HALF_SPAN_M
ZONES = (
    ('fingers', -(STANDOFF_M + 0.045), 0.05, FINGER_HALF_SPAN_M, 0.03),
    ('gripper base', -(STANDOFF_M + 0.045 + 0.155), -(STANDOFF_M + 0.045), 0.085, 0.05),
    ('wrist', -(STANDOFF_M + 0.045 + 0.155 + 0.13), -(STANDOFF_M + 0.045 + 0.155), 0.06, 0.06),
)
CLEAR_MARGIN_M = 0.015
CLEAR_CAP_M = 0.10
SAME_OBJECT_M = ws.SAME_OBJECT_M

# base preference per class and method (1 = what we would normally do); the learned factor and the clearance move it
PREF = {
    'cup': {'side': 0.9, 'top': 0.6, 'lean': 0.5},
    'mug': {'side': 0.9, 'top': 0.6, 'lean': 0.5},
    'bottle': {'side': 0.6, 'top': 0.7, 'lean': 0.4},
}
DEFAULT_PREF = {'side': 0.5, 'top': 1.0, 'lean': 0.6}
AZ_PRIOR = {90.0: 0.15, 270.0: 0.15}            # sideways to the arm: the reach study found +-y the most reachable


def ang_diff(a, b):
    """Smallest difference between two directions in degrees, [0, 180]."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# ---- geometry of the side grasp -------------------------------------------------------------------------------------------------------------------
def side_axis(az_deg):
    return aa.tool_axis(SIDE_TILT_DEG, az_deg)


def side_quat(az_deg, flip=False):
    """Flange orientation (x, y, z, w) for the side grasp toward az: the demonstrated one turned about the vertical. flip = the same grasp with the wrist turned 180 deg (the gripper is symmetric)."""
    r = R.from_euler('z', az_deg - DEMO_AZ_DEG, degrees=True) * R.from_quat(DEMO_QUAT_XYZW)
    if flip:
        r = r * R.from_euler('z', 180.0, degrees=True)
    return [float(v) for v in r.as_quat()]


def side_eligible(label, table_top_z=0.0, width_m=None):
    """True when the object is tall enough to be held at its side and fits between the open fingers."""
    if ws.OBJECT_HEIGHT_M.get(label, ws.DEFAULT_HEIGHT_M) < MIN_SIDE_HEIGHT_M:
        return False
    return ws.too_wide_for_gripper(label, width_m) is None


def side_pad_z(label, table_top_z):
    """Height of the pad midpoint: just under the rim of a short object, in the straight lower body of a tall one."""
    h = ws.OBJECT_HEIGHT_M.get(label, ws.DEFAULT_HEIGHT_M)
    z = table_top_z + (ws.TALL_GRASP_FRACTION * h if h >= ws.TALL_OBJECT_M else h - RIM_BELOW_M)
    return max(z, table_top_z + PAD_MIN_ABOVE_TABLE_M)


def side_poses(centre_xy, pad_z, az_deg):
    """(high flange, pre flange, grasp flange, axis): the pad midpoint ends on the object's centre at pad_z. High = the pre pose lifted by HIGH_CLEAR_M (reached first, from the hover);
    pre = STANDOFF_M back along the axis; the approach is then straight down to pre and straight forward to the grasp."""
    a = side_axis(az_deg)
    pad = np.array([centre_xy[0], centre_xy[1], pad_z], float)
    grasp = pad - PAD_REACH_M * a
    pre = grasp - STANDOFF_M * a
    high = pre + np.array([0.0, 0.0, HIGH_CLEAR_M])
    return high, pre, grasp, a


# ---- what the open gripper would hit --------------------------------------------------------------------------------------------------------------
def zone_gap(centre_xy, az_deg, pad_z, table_top_z, obstacle, same_object_m=SAME_OBJECT_M):
    """(gap_m, zone name) for one obstacle (label, x, y): the smallest distance between its edge (+ margin) and the gripper's footprint for an approach toward az; negative = a hit.
    Obstacles lower than the part of the gripper that passes them never count. None when it is the target itself."""
    label, x, y = obstacle[0], obstacle[1], obstacle[2]
    dx, dy = x - centre_xy[0], y - centre_xy[1]
    if math.hypot(dx, dy) < same_object_m:
        return None
    top = table_top_z + ws.OBJECT_HEIGHT_M.get(label, ws.DEFAULT_HEIGHT_M)
    u = np.array([math.cos(math.radians(az_deg)), math.sin(math.radians(az_deg))])
    t = dx * u[0] + dy * u[1]
    lat = abs(-dx * u[1] + dy * u[0])
    r = ws.OBJECT_RADIUS_M.get(label, ws.DEFAULT_RADIUS_M)
    best = (math.inf, None)
    for name, t_lo, t_hi, hw, below in ZONES:
        if top < pad_z - below:
            continue                                    # passes over/under nothing: this zone of the gripper is above it
        d = math.hypot(max(t_lo - t, 0.0, t - t_hi), max(lat - hw, 0.0))
        gap = d - r - CLEAR_MARGIN_M
        if gap < best[0]:
            best = (gap, name)
    return best if best[1] is not None else (math.inf, None)


def side_clearance(centre_xy, az_deg, pad_z, table_top_z, obstacles, same_object_m=SAME_OBJECT_M):
    """(clearance_m, what): the smallest gap to any obstacle for an approach toward az (inf if none); negative = blocked by `what`."""
    worst, what = math.inf, None
    for o in obstacles:
        g = zone_gap(centre_xy, az_deg, pad_z, table_top_z, o, same_object_m)
        if g is None or g[1] is None:
            continue
        if g[0] < worst:
            worst, what = g[0], f'{o[0]} ({g[1]})'
    return worst, what


def handle_adjust(az_deg, handle_bearing_deg):
    """(score adjustment, reject reason or None) for an approach toward az when the object's handle sticks out toward handle_bearing_deg (base frame, from the centre).
    Fingertips leading toward the handle (as demonstrated) is best; coming from the handle's side puts it in the gripper; across it, a pad lands on the handle."""
    if handle_bearing_deg is None:
        return 0.0, None
    d = ang_diff(az_deg, handle_bearing_deg)
    if d <= 35.0:
        return 0.25, None
    if 55.0 <= d <= 125.0:
        return 0.0, 'a pad would land on the handle'
    if d >= 145.0:
        return -0.25, None
    return -0.1, None


def method_factor(label, method, records):
    """What past outcomes say about this method for this class: 1.0 with no data; (successes+1)/(trials+2)/0.5 clipped to [0.4, 1.5]. records: the outcome log (picks carry 'method')."""
    s = n = 0
    for r in records or ():
        if r.get('kind') != 'pick' or not r.get('live', True) or r.get('label') != label or r.get('method') != method:
            continue
        n += 1
        s += int(bool(r.get('ok')))
    return float(min(1.5, max(0.4, (s + 1.0) / (n + 2.0) / 0.5)))


# ---- the ranking ----------------------------------------------------------------------------------------------------------------------------------
def rank_side(label, centre_xy, table_top_z, obstacles, handle_bearing_deg=None, records=(), same_object_m=SAME_OBJECT_M,
              azimuths=AZIMUTHS, exclude_az=()):
    """Side-grasp options for every azimuth, best first: [{'method': 'side', 'az', 'score', 'clearance_m', 'pad_z'}], plus the rejected ones as {'method': 'side', 'az', 'rejected': why}."""
    pad_z = side_pad_z(label, table_top_z)
    base = PREF.get(label, DEFAULT_PREF)['side'] * method_factor(label, 'side', records)
    ok, rejected = [], []
    for az in azimuths:
        if any(ang_diff(az, e) < 1.0 for e in exclude_az):
            continue
        clear, what = side_clearance(centre_xy, az, pad_z, table_top_z, obstacles, same_object_m)
        if clear < 0.0:
            rejected.append({'method': 'side', 'az': float(az), 'rejected': f'the gripper would hit the {what}'})
            continue
        adj, why = handle_adjust(az, handle_bearing_deg)
        if why:
            rejected.append({'method': 'side', 'az': float(az), 'rejected': why})
            continue
        score = base + 0.5 * min(clear, CLEAR_CAP_M) / CLEAR_CAP_M + AZ_PRIOR.get(float(az), 0.0) + adj
        ok.append({'method': 'side', 'az': float(az), 'score': round(score, 3), 'clearance_m': None if math.isinf(clear) else round(clear, 3), 'pad_z': round(pad_z, 4)})
    ok.sort(key=lambda o: -o['score'])
    return ok + rejected


def rank_methods(label, centre_xy, table_top_z, obstacles, top_clear_m, lean_possible, side_mode='auto', handle_bearing_deg=None,
                 records=(), same_object_m=SAME_OBJECT_M, width_m=None, avoid_methods=()):
    """Every way to grasp this object, best first: [{'method': 'side'|'top'|'lean', ...}].
    top_clear_m: how much room the straight-down approach has (None = a neighbour is in the open fingers' sweep). lean_possible: the lean is switched on and allowed for this object.
    side_mode: 'off' (never), 'auto' (ranked with the others), 'only' (nothing else). avoid_methods: methods that already failed on this object (a retry should try another)."""
    pref = PREF.get(label, DEFAULT_PREF)
    out = []
    if side_mode != 'off' and side_eligible(label, table_top_z, width_m) and (in_side_zone(centre_xy) or side_mode == 'only'):
        out += [o for o in rank_side(label, centre_xy, table_top_z, obstacles, handle_bearing_deg, records, same_object_m) if 'score' in o]
    if side_mode != 'only':
        top = {'method': 'top', 'score': pref['top'] * method_factor(label, 'top', records) + (0.5 * min(top_clear_m, CLEAR_CAP_M) / CLEAR_CAP_M if top_clear_m is not None else -1.0)}
        out.append(top)
        if lean_possible:
            out.append({'method': 'lean', 'score': pref['lean'] * method_factor(label, 'lean', records) + (0.3 if top_clear_m is None else 0.0)})
    for o in out:
        if o['method'] in avoid_methods:
            o['score'] -= 1.0
            o['retry_penalty'] = True
        o['score'] = round(o['score'], 3)
    out.sort(key=lambda o: -o['score'])
    return out
