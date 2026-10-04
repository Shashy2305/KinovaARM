"""
Table / rear-wall geometry and trajectory safety checks, shared by
arm_controller_node (enforces them) and llm_planner_node (height floor).

All coordinates are metres in base_link (+X forward into the table, +Y
right of the robot, +Z up). Pure python on purpose so it can be unit
tested without ROS.

Why this exists: `end_effector_link` is the WRIST FLANGE. The Robotiq
2F-140 pads reach ~0.21 m beyond it along the tool axis (URDF, gripper
open), so with the gripper pointing down a flange target of z=0.12 puts
the fingertips at about -0.09 -- the table surface. The old
`z >= 0.08` floors treated z as fingertip height. MoveIt also had no
table in its planning scene, so nothing stopped a planned path from
passing through it either.
"""
import os

import yaml

TABLE_GEOMETRY_FILE = os.path.expanduser('~/.ros/table_geometry.yaml')

# Flange -> lowest point of the finger pads along the tool axis (gripper
# open). URDF pad frame origins are at 0.177 but the 7 cm pad collision
# boxes extend to 0.212 (computed from the live URDF).
TCP_REACH_M = 0.215
# Extra hover margin for flange targets, on top of table_top + TCP_REACH_M.
TCP_CLEARANCE_M = 0.05
# The pad frame origin sits this far above the lowest point of the pad
# (0.212 - 0.177); used when turning recorded "pad tips touching the
# table" poses into a surface height.
PAD_FRAME_ABOVE_TIP_M = 0.035
# Trajectory check: every guarded link frame must stay at least this far
# above the table surface (pad frame origins are 3.5 cm above the pad's
# lowest point, plus a margin).
LINK_TABLE_CLEARANCE_M = 0.05
# Wall behind the robot (where the operator stands): planner must not path
# through it, and guarded links must stay in front of REAR_KEEP_OUT_X.
REAR_WALL_DEFAULT_X = -0.20
REAR_KEEP_OUT_X = -0.10
# The slab starts in front of the robot base so it can never overlap the
# base/shoulder collision meshes (which would make every start state
# "in collision"). The trajectory check still covers the strip near the base.
SLAB_X_MIN = 0.10
SLAB_THICKNESS_M = 1.0
# Used only when no geometry file exists: a flange height that keeps the
# fingertips above the base mounting plane (table top is at or below z=0).
FALLBACK_FLANGE_Z_MIN = 0.30

# Table footprint in base_link from tape measurements (table 183x90 cm,
# robot on the long edge inset ~15 cm, 80 cm / 85 cm to the ends). Recorded
# points can only widen it.
DEFAULT_FOOTPRINT_X = (-0.15, 0.75)
DEFAULT_FOOTPRINT_Y = (-0.80, 0.85)

GUARDED_LINKS = [
    'end_effector_link', 'bracelet_link', 'robotiq_140_base_link',
    'left_inner_finger_pad', 'right_inner_finger_pad',
    'left_outer_finger', 'right_outer_finger',
]

MAX_RECORDED_Z_SPREAD_M = 0.03


def load_geometry(path=TABLE_GEOMETRY_FILE):
    """Returns (geometry dict or None, error string or None)."""
    if not os.path.exists(path):
        return None, f'{path} does not exist'
    try:
        with open(path) as f:
            g = yaml.safe_load(f)
        top = float(g['table_top_z'])
        x0, x1 = (float(v) for v in g['x'])
        y0, y1 = (float(v) for v in g['y'])
        wall = float(g.get('rear_wall_x', REAR_WALL_DEFAULT_X))
    except Exception as e:  # malformed file must fail closed, not crash the node
        return None, f'{path} is invalid: {e}'
    if not (x0 < x1 and y0 < y1):
        return None, f'{path} has an empty footprint'
    if not (-1.0 < top < 0.5):
        return None, f'{path} table_top_z={top} is implausible for base_link'
    return {'table_top_z': top, 'x': [x0, x1], 'y': [y0, y1], 'rear_wall_x': wall}, None


def geometry_from_points(points):
    """points: list of (x, y, z) of the fingertip PAD FRAME while the pads
    touched the table. Returns (geometry dict, error string or None)."""
    if len(points) < 3:
        return None, 'need at least 3 recorded points'
    zs = sorted(p[2] for p in points)
    spread = zs[-1] - zs[0]
    if spread > MAX_RECORDED_Z_SPREAD_M:
        return None, (f'recorded heights differ by {spread * 100:.1f} cm '
                      f'(limit {MAX_RECORDED_Z_SPREAD_M * 100:.0f} cm) — the pads were not all '
                      f'touching the table; redo the recording')
    median = zs[len(zs) // 2] if len(zs) % 2 else 0.5 * (zs[len(zs) // 2 - 1] + zs[len(zs) // 2])
    x0 = min(DEFAULT_FOOTPRINT_X[0], min(p[0] for p in points))
    x1 = max(DEFAULT_FOOTPRINT_X[1], max(p[0] for p in points))
    y0 = min(DEFAULT_FOOTPRINT_Y[0], min(p[1] for p in points))
    y1 = max(DEFAULT_FOOTPRINT_Y[1], max(p[1] for p in points))
    return {
        'table_top_z': round(median - PAD_FRAME_ABOVE_TIP_M, 4),
        'x': [round(x0, 3), round(x1, 3)],
        'y': [round(y0, 3), round(y1, 3)],
        'rear_wall_x': REAR_WALL_DEFAULT_X,
        'recorded_points': len(points),
        'recorded_z_spread_m': round(spread, 4),
    }, None


def save_geometry(geom, path=TABLE_GEOMETRY_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(geom, f, default_flow_style=False)


def flange_floor_z(geom):
    """Lowest allowed z for an end_effector_link (flange) target."""
    if geom is None:
        return FALLBACK_FLANGE_Z_MIN
    return geom['table_top_z'] + TCP_REACH_M + TCP_CLEARANCE_M


def collision_boxes(geom):
    """[(id, (dx, dy, dz), (cx, cy, cz))] in base_link for MoveIt's scene."""
    x0 = max(SLAB_X_MIN, geom['x'][0])
    x1 = geom['x'][1]
    y0, y1 = geom['y']
    top = geom['table_top_z']
    wall_x = geom['rear_wall_x']
    return [
        ('table_slab',
         (x1 - x0, y1 - y0, SLAB_THICKNESS_M),
         ((x0 + x1) / 2.0, (y0 + y1) / 2.0, top - SLAB_THICKNESS_M / 2.0)),
        ('rear_safety_wall',
         (0.05, 4.0, 3.0),
         (wall_x - 0.025, 0.0, 1.0)),
    ]


def check_link_positions(positions, geom):
    """positions: {link: (x, y, z)} in base_link. Returns (ok, reason)."""
    min_z = geom['table_top_z'] + LINK_TABLE_CLEARANCE_M
    for link, (x, y, z) in positions.items():
        if z < min_z:
            return False, (f'{link} would be at z={z:.3f}, below the table limit '
                           f'{min_z:.3f} (table top {geom["table_top_z"]:.3f})')
        if x < REAR_KEEP_OUT_X:
            return False, (f'{link} would be at x={x:.3f}, behind the rear limit '
                           f'{REAR_KEEP_OUT_X:.2f}')
    return True, 'ok'


def sample_indices(n, max_samples=40):
    """Evenly spaced indices into a trajectory, always including first and last."""
    if n <= 0:
        return []
    if n <= max_samples:
        return list(range(n))
    step = (n - 1) / float(max_samples - 1)
    idx = sorted({int(round(i * step)) for i in range(max_samples)} | {0, n - 1})
    return idx
