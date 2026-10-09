"""
Geometry of an ANGLED (tilted) tool approach (pure numpy/scipy, no ROS).

The arm has always come straight down: the tool z axis points at the floor. A tilted approach leans the tool by `tilt`
degrees from vertical, with the fingertips leading toward azimuth `az` (degrees from base +x, counter-clockwise seen
from above). The tool axis a (flange -> fingertips) is then
    a = (sin(tilt) cos(az), sin(tilt) sin(az), -cos(tilt))
and the flange sits TCP_REACH_M behind the fingertips along it. A pregrasp is the same pose slid back along a, so the
final approach is a straight line down the tool axis (an angled entry), not a vertical drop.

What a tilt buys: it moves where the gripper BODY is while the fingertips stay on the object. With the fingertips
leading away from a neighbour, the wrist and finger bases stay clear of it; with the object near the table's edge
or wall, the arm can come in from the open side. What it costs: the swing from the usual pose is bigger, and the wrist
camera no longer looks straight down.

Feasibility is NOT assumed here: scripts/angled_approach_study.py asks MoveIt's IK which tilts are reachable.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation as R

TCP_REACH_M = 0.215
DOWN = np.array([0.0, 0.0, -1.0])


def tool_axis(tilt_deg, az_deg):
    """Unit vector flange -> fingertips for a tilt from vertical and the azimuth the fingertips lead toward."""
    t, a = math.radians(tilt_deg), math.radians(az_deg)
    return np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), -math.cos(t)])


def _minimal_rotation(frm, to):
    """Smallest rotation taking unit vector frm to unit vector to."""
    frm, to = np.asarray(frm, float), np.asarray(to, float)
    c = float(np.clip(frm @ to, -1.0, 1.0))
    axis = np.cross(frm, to)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return R.identity() if c > 0 else R.from_rotvec([math.pi, 0, 0])
    return R.from_rotvec(axis / n * math.acos(c))


def tool_quat(base_quat_xyzw, tilt_deg, az_deg, yaw_deg=0.0):
    """Orientation (x, y, z, w) of the tool for the tilted approach: the base (vertical-down) orientation, tilted so its z
    axis is tool_axis(tilt, az), then turned by yaw_deg about that tool axis (which way the fingers close)."""
    r0 = R.from_quat(base_quat_xyzw)
    r = _minimal_rotation(r0.as_matrix()[:, 2], tool_axis(tilt_deg, az_deg)) * r0
    r = r * R.from_euler('z', yaw_deg, degrees=True)
    return [float(v) for v in r.as_quat()]


def tilted_from(current_quat_xyzw, tilt_deg, az_deg):
    """Orientation obtained by leaning the arm's CURRENT tool orientation (which carries the wrist alignment already done
    for this object) to the tilted tool axis, without changing which way the fingers close beyond what the lean forces.
    The current tool z axis is taken as 'vertical' and replaced by tool_axis(tilt, az)."""
    r0 = R.from_quat(current_quat_xyzw)
    r = _minimal_rotation(r0.as_matrix()[:, 2], tool_axis(tilt_deg, az_deg)) * r0
    return [float(v) for v in r.as_quat()]


def flange_for_tip(tip_xyz, axis, reach=TCP_REACH_M):
    """Flange position that puts the fingertips at tip_xyz with the tool along `axis`."""
    return np.asarray(tip_xyz, float) - reach * np.asarray(axis, float)


def tip_for_flange(flange_xyz, axis, reach=TCP_REACH_M):
    return np.asarray(flange_xyz, float) + reach * np.asarray(axis, float)


def approach_poses(tip_xyz, tilt_deg, az_deg, standoff=0.12, reach=TCP_REACH_M):
    """(pregrasp_flange, grasp_flange, axis): the straight-line entry down the tool axis."""
    a = tool_axis(tilt_deg, az_deg)
    tip = np.asarray(tip_xyz, float)
    return flange_for_tip(tip - standoff * a, a, reach), flange_for_tip(tip, a, reach), a


def lowest_tool_z(flange_xyz, axis, closing_axis_xy, half_gap=0.0675, finger_len=0.035):
    """Lowest z of the gripper's fingertips (both fingers) for a tool pose: each fingertip is half_gap to either side of
    the tool axis along the horizontal closing direction, `reach` along the axis from the flange; the tip face
    reaches finger_len further (the pad frame sits 3.5 cm above the tip)."""
    a = np.asarray(axis, float)
    c = np.array([closing_axis_xy[0], closing_axis_xy[1], 0.0])
    c -= (c @ a) * a                                   # keep the closing axis perpendicular to the tool axis
    n = np.linalg.norm(c)
    c = c / n if n > 1e-9 else c
    tip = tip_for_flange(flange_xyz, a)
    return float(min((tip + s * half_gap * c)[2] for s in (-1.0, 1.0)))
