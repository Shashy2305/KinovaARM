"""
Geometry for centering the gripper over an object with the wrist camera.

No ROS here: the arm controller feeds in the detected pixel, the camera
intrinsics, the camera pose in base_link (from TF) and the height of the
object's top, and gets back where the object is on the table plane. Depth is
deliberately not used: the Kinova wrist depth sensor returns nothing below
about 0.2-0.3 m, which is exactly where the final centering happens.
"""
import math

import numpy as np

# Height of the top of the object above the table (m). The pixel we detect is
# the middle of the bounding box, which for an object seen from above is its
# axis at (roughly) the top surface.
OBJECT_HEIGHT_M = {
    'cup': 0.09, 'mug': 0.09, 'bowl': 0.06, 'bottle': 0.22, 'mouse': 0.035,
    'cell phone': 0.01, 'remote': 0.03, 'book': 0.03, 'scissors': 0.02, 'vase': 0.20,
}
DEFAULT_HEIGHT_M = 0.06
MIN_RAY_DOWN = 0.2          # |dir_z| below this: ray almost parallel to the table, reject


def plane_height(label, table_top_z):
    return table_top_z + OBJECT_HEIGHT_M.get(label, DEFAULT_HEIGHT_M)


def center_plane_height(label, table_top_z):
    """Plane height for turning the centre of the object's BOX into a table position.
    The box centre of a tall object seen from above (off-axis camera) is about its mid-height, not its
    top: using the top plane put the fingers 2.5-3 cm off a 20 cm bottle, sideways to the closing
    direction. Short objects (a mug, a mouse) keep the top plane, as verified on the arm."""
    height = OBJECT_HEIGHT_M.get(label, DEFAULT_HEIGHT_M)
    return table_top_z + (0.5 * height if height >= TALL_OBJECT_M else height)


def pixel_to_plane_xy(u, v, K, R_bc, t_bc, plane_z):
    """base_link (x, y) where the camera ray through pixel (u, v) meets the
    horizontal plane z = plane_z. K: 3x3 intrinsics. (R_bc, t_bc): camera pose in
    base_link, optical-frame convention (x right, y down, z forward).
    Returns None if the ray does not hit the plane in front of the camera."""
    K = np.asarray(K, float).reshape(3, 3)
    d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    d = np.asarray(R_bc, float) @ d_cam
    d /= np.linalg.norm(d)
    if abs(d[2]) < MIN_RAY_DOWN:
        return None
    s = (plane_z - t_bc[2]) / d[2]
    if s <= 0:
        return None
    p = np.asarray(t_bc, float) + s * d
    return float(p[0]), float(p[1])


def clipped_step(target_xy, gripper_xy, max_step):
    """(dx, dy, distance) to move the gripper onto the target, the step limited
    to max_step metres (the remaining error is handled by the next iteration)."""
    dx, dy = target_xy[0] - gripper_xy[0], target_xy[1] - gripper_xy[1]
    dist = math.hypot(dx, dy)
    if dist > max_step:
        k = max_step / dist
        dx, dy = dx * k, dy * k
    return dx, dy, dist


def search_offsets(radius=0.04):
    """Small moves tried, in order, when the object is not visible: the four
    compass points, then the diagonals (all relative to where the search began)."""
    r = radius
    return [(r, 0), (-r, 0), (0, r), (0, -r), (r, r), (-r, -r), (r, -r), (-r, r)]


def pick_detection(dets, label, expected_uv=None):
    """Best detection of `label` from a list of dicts with confidence/u/v.
    If expected_uv is given, prefer the one closest to it (several cups on the
    table); otherwise the most confident one."""
    cands = [d for d in dets if d.get('label') == label]
    if not cands:
        return None
    if expected_uv is None:
        return max(cands, key=lambda d: d['confidence'])
    return min(cands, key=lambda d: math.hypot(d['u'] - expected_uv[0], d['v'] - expected_uv[1]))


HANDLE_ASPECT = 1.15        # bbox this much longer along the closing axis -> a handle (or a long object) is in the way


def needs_quarter_turn(bbox, ratio=HANDLE_ASPECT):
    """True if the object's box is clearly longer along the image x axis than along y.
    The two fingers sit at the left and right of the wrist image, so the gripper closes
    along image x; a box stretched along x means the long part (a mug handle, a bottle
    lying down) would be hit by a finger. Turning the wrist 90 degrees puts the short
    side between the fingers. bbox = (x1, y1, x2, y2)."""
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return h > 0 and w / h > ratio


def quarter_turn_target(q7, limit=2.7):
    """Joint-7 angle after a +-90 degree turn: whichever stays inside +-limit and is
    closest to zero. None if neither fits."""
    cands = [q7 + sgn * (3.141592653589793 / 2) for sgn in (1, -1)]
    cands = [c for c in cands if abs(c) <= limit]
    return min(cands, key=abs) if cands else None


HOVER_MAX_Z = 0.58            # highest flange z a pick may hover at (tall objects)
Z_TRUST_M = 0.04              # how far a scene z may be from the object's expected centre height before it is ignored
TALL_OBJECT_M = 0.15          # objects at least this tall are grasped low on the body
TALL_GRASP_FRACTION = 0.30    # fingertips at this fraction of the object's height


def pick_heights(oz, label, table_top, tcp_reach, floor_z, tip_clear, default_tip_clear, approach_z=0.0,
                 grasp_offset=0.0, hover_above=0.12, pregrasp_clearance=0.10, top_clearance=0.07):
    """Flange heights for a pick: (grasp_z, hover_z, lift_z, low_floor).

    grasp: fingertips at the object's centre (flange = centre + reach), never below the floor. A LOW object
    (<= 6 cm) with a lowered tip clearance is grasped with the tips near the table, just inside its top.
    hover: the open fingertips must clear the object's TOP by top_clearance (a 26 cm bottle needs the
    hover well above where a 9 cm mug does), and be at least approach_z.
    lift: just far enough to carry (never above the hover)."""
    height = OBJECT_HEIGHT_M.get(label, DEFAULT_HEIGHT_M)
    top = table_top + height
    # The scene's z is a depth-camera surface point: 2-4 cm high for ordinary objects, and sometimes
    # wildly wrong (a "cup" at z=0.132 made the arm grasp 8 cm above the real mug and close on air).
    # Where it disagrees with the object's known size by more than 4 cm, use the size instead.
    prior_centre = table_top + 0.5 * height
    if abs(oz - prior_centre) > Z_TRUST_M:
        oz = prior_centre
    grasp_z = max(oz + tcp_reach + grasp_offset, floor_z)
    low_floor = table_top + tcp_reach + tip_clear
    if height >= TALL_OBJECT_M:
        # the scene's z reads the "centre" of a tall object 3-4 cm too high (a surface point), and the upper
        # part of a bottle is its tapered shoulder, which squeezes the pads off. Grasp the straight lower body.
        grasp_z = max(table_top + TALL_GRASP_FRACTION * height + tcp_reach + grasp_offset, floor_z)
    if height <= 0.06 and tip_clear < default_tip_clear:
        grasp_z = max(low_floor, min(oz, top - 0.025) + tcp_reach)
    hover_z = max(float(approach_z), oz + tcp_reach + hover_above, grasp_z + pregrasp_clearance,
                  top + tcp_reach + top_clearance)
    hover_z = min(HOVER_MAX_Z, hover_z)
    lift_z = min(hover_z, grasp_z + 0.15)
    return grasp_z, hover_z, lift_z, low_floor


SAME_LABEL_TOL_M = 0.15       # a detection of the expected class this close (m) to the expected spot is the object
OTHER_LABEL_TOL_M = 0.06      # a detection of ANY graspable class this close is accepted too


def match_detection(cands, label, expected_xy):
    """Pick the wrist detection that is the target object. cands: [(detection, (x, y) on the table plane)].
    The detector's class is unreliable from straight above (a bottle seen from above is "cup", a bowl is
    "mouse"), so besides the expected class we also accept ANY class whose position is within
    OTHER_LABEL_TOL_M of where the scene says the object is. Nothing else sits that close to the target.
    Returns (detection, xy) or None."""
    if not cands:
        return None
    if expected_xy is None:
        same = [c for c in cands if c[0].get('label') == label]
        return max(same, key=lambda c: c[0]['confidence']) if same else None

    def dist(c):
        return math.hypot(c[1][0] - expected_xy[0], c[1][1] - expected_xy[1])

    same = [c for c in cands if c[0].get('label') == label]
    if same:
        best = min(same, key=dist)
        if dist(best) <= SAME_LABEL_TOL_M:
            return best
    best = min(cands, key=dist)
    return best if dist(best) <= OTHER_LABEL_TOL_M else None
