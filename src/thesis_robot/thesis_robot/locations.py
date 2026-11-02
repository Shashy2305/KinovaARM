"""
Named locations taught by hand (scripts/teach_location.py writes ~/.ros/locations.yaml) and the poses the arm needs to go to one. No ROS, no hardware.

A location records where the pad midpoint was with the gripper closed on the object (for the coffee tray: the cup standing in the tray), the way the tool pointed (approach azimuth) and the joints.
From it the side-grasp geometry of grasp_options gives the poses of the approach: HIGH (above and back), PRE (STANDOFF back along the tool axis, fingers open) and GRASP.
"""
import math
import os

import numpy as np
import yaml

from thesis_robot import grasp_options as go

LOCATIONS_FILE = os.path.expanduser('~/.ros/locations.yaml')
FIXED_OBSTACLES_FILE = os.path.expanduser('~/.ros/fixed_obstacles.yaml')
MIN_FRONT_CLEARANCE_M = 0.03          # the open fingers at the approach pose must stay this far from a fixed obstacle


def load_all(path=None):
    try:
        with open(path or LOCATIONS_FILE) as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def get(name, path=None):
    """The location called `name` if it is complete and sane, else None."""
    e = load_all(path).get(name)
    try:
        pad = [float(v) for v in e['pad_midpoint']]
        az = float(e['approach_azimuth_deg'])
    except (TypeError, KeyError, ValueError):
        return None
    if len(pad) != 3 or not all(math.isfinite(v) for v in pad + [az]):
        return None
    return dict(e, pad_midpoint=pad, approach_azimuth_deg=az)


def approach_poses(loc):
    """{'high', 'pre', 'grasp': flange positions, 'axis': tool axis, 'az', 'pad', 'pre_pad': pad midpoint at the approach pose} for a location."""
    pad = loc['pad_midpoint']
    az = loc['approach_azimuth_deg']
    high, pre, grasp, axis = go.side_poses((pad[0], pad[1]), pad[2], az)
    pre_pad = pre + go.PAD_REACH_M * axis
    return {'high': high, 'pre': pre, 'grasp': grasp, 'axis': axis, 'az': az, 'pad': np.array(pad), 'pre_pad': pre_pad}


def fixed_obstacle_rects(path=None):
    """[(name, x_min, x_max, y_min, y_max)] for the declared fixed obstacles: the rectangle when the file gives size_x / size_y, else the bounding square of its circle."""
    try:
        with open(path or FIXED_OBSTACLES_FILE) as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return []
    out = []
    for o in data.get('obstacles') or []:
        try:
            x, y = float(o['x']), float(o['y'])
            hx = float(o.get('size_x', 2.0 * float(o.get('radius', 0.12)))) / 2.0
            hy = float(o.get('size_y', 2.0 * float(o.get('radius', 0.12)))) / 2.0
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        out.append((str(o.get('name', 'obstacle')), x - hx, x + hx, y - hy, y + hy))
    return out


def rect_distance(px, py, rect):
    """Distance from a point to an axis-aligned rectangle (0 inside)."""
    _n, x0, x1, y0, y1 = rect
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def approach_clearance(poses, rects):
    """(smallest distance, name) from the open fingertips at the approach pose (a little beyond the pad midpoint) to the fixed obstacles; (inf, None) when there are none."""
    tip = poses['pre_pad'] + 0.03 * poses['axis']
    best = (float('inf'), None)
    for r in rects:
        d = rect_distance(float(tip[0]), float(tip[1]), r)
        if d < best[0]:
            best = (d, r[0])
    return best
