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
import math
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

# Targets the planner will accept and arm_controller will not clamp. One
# definition, used by llm_planner (validation + prompt), arm_controller
# (clamp) and scene_graph_node (what counts as "reachable"), so the scene
# never offers the LLM an object the planner would then reject.
PLAN_X_RANGE = (0.10, 0.60)
PLAN_Y_RANGE = (-0.35, 0.35)
# Conservative horizontal reach limit for the 'reachable' flag (true max ~0.9 m).
MAX_REACH_XY_M = 0.85
# An object whose height is more than this far below the recorded table
# surface is a phantom (calibration error, reflection), not something on it.
BELOW_TABLE_TOL_M = 0.04

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


def start_floors(first_positions, geom):
    """Per-link minimum heights for a path that STARTS where first_positions says. A link already
    below the table margin at the start (the fingers close ~2 cm lower than they open, so a grasp
    of a low object ends there) may leave it, but never go lower than it began: otherwise every
    move away from such a pose, including the lift, is refused and the arm is stuck."""
    margin = geom['table_top_z'] + LINK_TABLE_CLEARANCE_M
    return {link: min(margin, z - 0.002) for link, (_x, _y, z) in first_positions.items()}


def check_link_positions(positions, geom, floors=None):
    """positions: {link: (x, y, z)} in base_link. Returns (ok, reason).
    floors: optional per-link minimum z (see start_floors) in place of the table margin."""
    min_z = geom['table_top_z'] + LINK_TABLE_CLEARANCE_M
    for link, (x, y, z) in positions.items():
        if z < (floors or {}).get(link, min_z):
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


def is_reachable(x, y, z, workspace, table_top_z=None):
    """True if an object at base_link (x, y, z) is something the arm may be
    sent to: inside the workspace box, inside the planner's x/y limits, inside
    the horizontal reach circle, and not below the recorded table surface."""
    if z is None:
        return False                      # height unknown -- cannot claim it is graspable
    if not (workspace['x'][0] <= x <= workspace['x'][1]
            and workspace['y'][0] <= y <= workspace['y'][1]
            and workspace['z'][0] <= z <= workspace['z'][1]):
        return False
    if not (PLAN_X_RANGE[0] <= x <= PLAN_X_RANGE[1] and PLAN_Y_RANGE[0] <= y <= PLAN_Y_RANGE[1]):
        return False
    if (x * x + y * y) ** 0.5 > MAX_REACH_XY_M:
        return False
    if table_top_z is not None and z < table_top_z - BELOW_TABLE_TOL_M:
        return False
    return True


# ── the robot's own body, for rejecting detections of the robot itself ──
# Arm links (frame origins, base to flange) and the two fingers (flange -> pad frame, 17.7 cm long).
ARM_CHAIN_FRAMES = ['shoulder_link', 'half_arm_1_link', 'half_arm_2_link', 'forearm_link',
                    'spherical_wrist_1_link', 'spherical_wrist_2_link', 'bracelet_link', 'end_effector_link']
FINGER_PAD_FRAMES = ['left_inner_finger_pad', 'right_inner_finger_pad']
ARM_BODY_RADIUS_M = 0.10        # a detection this close to the arm's links is the arm, not an object
FINGER_BODY_RADIUS_M = 0.045    # ... and this close to a finger. Smaller: an object held or grasped sits
                                # ~6 cm from each pad and must stay visible.


def _dist_point_segment3(p, a, b):
    ab = [b[i] - a[i] for i in range(3)]
    ap = [p[i] - a[i] for i in range(3)]
    L2 = sum(v * v for v in ab)
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, sum(ap[i] * ab[i] for i in range(3)) / L2))
    return math.sqrt(sum((p[i] - (a[i] + t * ab[i])) ** 2 for i in range(3)))


def point_on_robot(point, chain, fingers):
    """True if `point` (x, y, z in base_link) lies on the robot: within ARM_BODY_RADIUS_M of the arm chain
    (a list of 3D points from the base to the flange) or FINGER_BODY_RADIUS_M of a finger segment
    (fingers: list of (flange_xyz, pad_xyz)). The cameras label the dark gripper "bottle"/"cup"/"remote";
    such a phantom beside the real target made the neighbour check refuse a mouse pick."""
    for a, b in zip(chain, chain[1:]):
        if _dist_point_segment3(point, a, b) < ARM_BODY_RADIUS_M:
            return True
    for a, b in fingers:
        if _dist_point_segment3(point, a, b) < FINGER_BODY_RADIUS_M:
            return True
    return False


def in_table_region(geom, x, y, z, margin=0.05):
    """True if (x, y, z) in base_link is above the recorded table footprint (plus margin) and at a height an
    object standing on it can have. The cameras see the whole lab; detections behind the robot (x < the
    table's rear edge) or beyond its far edge are other desks, chairs and people, not objects to pick."""
    if not (geom['x'][0] - margin <= x <= geom['x'][1] + margin and geom['y'][0] - margin <= y <= geom['y'][1] + margin):
        return False
    return z is None or geom['table_top_z'] - 0.05 <= z <= geom['table_top_z'] + 0.45
