"""
Grasp-angle planner (pure Python, no ROS): which direction should the fingers close along, and what to try next
when that failed.

Inputs come from shape_analysis (per closing direction phi in the wrist image: the span the fingers need and the length
of flat contact on each side). The two fingers sit left and right in the wrist image, so the gripper currently closes
along image-x (phi = 0). To close along the object direction phi the wrist is turned by delta = -phi (wrapped to
+-90 degrees, a parallel gripper is symmetric): the same convention as wrist_servo.rotation_to_align.

Why this exists: the pick used ONE strategy, the short side of the object, and a retry repeated it. A retry should come
from a NEW entry angle. Angles that were tried are remembered relative to the OBJECT (rho = phi - its long axis), not
to the image, because the wrist turns between attempts and an image angle would mean something else the next time.
"""
import math

GRIPPER_OPEN_M = 0.135        # pad-to-pad gap when fully open (measured)
EDGE_MARGIN_M = 0.012         # keep this much slack across the object when choosing a direction
MIN_WIDTH_M = 0.012           # narrower than this is not something the pads can hold reliably
CONTACT_FULL_M = 0.045        # flat contact this long on both sides counts as a fully secure hold
HANDLE_AVOID_DEG = 28.0       # a finger within this angle of the handle bearing risks the handle
DEFAULT_MIN_SEP_DEG = 30.0    # a retry must differ from every tried direction by at least this (mod 180)


def wrap90(deg):
    """Wrap an angle to (-90, 90]: a parallel gripper looks the same after a 180 degree turn."""
    d = (deg + 90.0) % 180.0 - 90.0
    return 90.0 if d == -90.0 else d


def angle_diff(a, b):
    """Smallest difference between two directions (mod 180), in [0, 90]."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def delta_for_phi(phi_deg):
    """Wrist image-plane rotation that puts the object direction phi between the fingers."""
    return wrap90(-phi_deg)


def rho_of(phi_deg, long_axis_deg):
    """Closing direction expressed relative to the object's long axis, in [0, 180)."""
    return (phi_deg - long_axis_deg) % 180.0


def phi_of(rho_deg, long_axis_deg):
    return (rho_deg + long_axis_deg) % 180.0


def rank_grasps(profile_m, contact_m, handle_bearing_deg=None, tried_rho=(), long_axis_deg=0.0,
                blocked=None, min_sep_deg=DEFAULT_MIN_SEP_DEG, max_turn_deg=90.0, gripper_open_m=GRIPPER_OPEN_M,
                prior=None):
    """Candidate closing directions, best first.

    profile_m   {phi_deg: width_m}   span needed per direction (keys may be str or int)
    contact_m   {phi_deg: contact_m} flat-contact length per direction (same keys)
    handle_bearing_deg  bearing of a handle/notch from the object centre (image degrees, y down) or None
    tried_rho   directions already tried, relative to the long axis (see rho_of)
    blocked     callable(phi_deg) -> bool: the open fingers would hit a neighbour along this direction
    prior       callable(rho_deg) -> multiplier in [0, 1]: what past outcomes say about closing along this direction
                (grasp_policy.GraspPolicy.prior); it scales the score, it never overrides a hard constraint
    Returns [{'phi', 'delta', 'width_m', 'contact_m', 'score', 'rho', 'reason'}]; directions that do not fit between
    the open fingers, that a neighbour blocks, or that repeat a tried one (within min_sep_deg) are left out."""
    out = []
    widths = {float(k): float(v) for k, v in profile_m.items()}
    contacts = {float(k): float(v) for k, v in (contact_m or {}).items()}
    for phi, w in sorted(widths.items()):
        if w > gripper_open_m - EDGE_MARGIN_M or w < MIN_WIDTH_M:
            continue
        if blocked is not None and blocked(phi):
            continue
        rho = rho_of(phi, long_axis_deg)
        if any(angle_diff(rho, t) < min_sep_deg for t in tried_rho):
            continue
        delta = delta_for_phi(phi)
        if abs(delta) > max_turn_deg:
            continue
        s_contact = min(1.0, contacts.get(phi, 0.0) / CONTACT_FULL_M)
        s_width = 1.0 - min(1.0, w / gripper_open_m)
        s_handle = 1.0
        if handle_bearing_deg is not None:
            near = min(angle_diff(phi, handle_bearing_deg), angle_diff(phi + 180.0, handle_bearing_deg))
            if near < HANDLE_AVOID_DEG:
                s_handle = 0.2
        s_turn = 1.0 - 0.25 * abs(delta) / 90.0
        score = 0.45 * s_contact + 0.25 * s_width + 0.20 * s_handle + 0.10 * s_turn
        if prior is not None:
            score *= float(prior(rho))
        out.append({'phi': phi, 'delta': delta, 'width_m': w, 'contact_m': contacts.get(phi, 0.0),
                    'score': round(score, 4), 'rho': round(rho, 1),
                    'reason': f'width {1000 * w:.0f} mm, contact {1000 * contacts.get(phi, 0.0):.0f} mm'})
    out.sort(key=lambda c: -c['score'])
    return out


def best_grasp(*args, **kwargs):
    ranked = rank_grasps(*args, **kwargs)
    return ranked[0] if ranked else None


def shape_to_metres(shape, distance_m, fx):
    """(profile_m, contact_m) in metres from an analyse() result (px) for a camera `distance_m` above the object."""
    k = distance_m / fx
    prof = {float(a): v * k for a, v in shape['profile'].items()}
    cont = {float(a): v * k for a, v in shape['contact'].items()}
    return prof, cont


def blocked_by_neighbours(phi_deg, image_axis_to_base, centre_xy, obstacles, sweep_blocker, **kw):
    """True if closing along image direction phi would sweep a neighbour. `image_axis_to_base(phi)` gives the unit
    (x, y) of that closing axis in base_link; `sweep_blocker` is wrist_servo.finger_sweep_blocker."""
    ux, uy = image_axis_to_base(phi_deg)
    return sweep_blocker(centre_xy, (ux, uy), obstacles, **kw) is not None
