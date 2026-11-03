"""
Pure-python helpers for choosing a SMALL, repeatable arm motion to a pose.

Why this exists: with the stock setup (MoveIt's default OMPL pipeline; the
'PTP' planner id was ignored because it is a Pilz planner) the same target
produced a different random path every run -- joints swinging up to 4.8 rad
and the arm sweeping 0.3-0.7 m behind its own base (where the operator
stands) in about half of the planning runs. Measured plan-only against the
live move_group from the home pose.

Fix: solve IK ourselves, seeded from the current joint state, trying a set
of tool yaws (a two-finger gripper is symmetric about its approach axis),
unwrap the continuous joints to the nearest equivalent angle, take the
solution that moves the joints least, and plan a joint-space move to it.
"""
import math

# joint_1/3/5/7 of the Gen3 are continuous: IK may return an angle that is
# equivalent modulo 2*pi, and a joint-space planner then takes the long way.
CONTINUOUS_JOINT_IDX = (0, 2, 4, 6)
# Pilz's joint_limits.yaml restricts every joint (including the continuous
# ones) to +-2.88 rad, so an unwrapped angle must stay inside it.
PLANNER_JOINT_LIMIT = 2.88
# 2026-11-03: joints 3 and 7 may go further. A side pick toward the coffee tray needs joint 3 / 7 between about 165 and 195 deg (the arm's home has joint 3 at 179.9 deg): both of their
# equivalents lay outside +-2.88, so the pick was refused for any cup more than ~1 cm off the one good spot. Measured: Pilz PTP plans these joints well past 3.5 rad (plan-only probe), the
# Kortex hardware interface reports angles wrapped to +-pi and wraps commands modulo 360 deg, and the installed joint_trajectory_controller normalises the position error of continuous joints
# (angle_wraparound). Joints 1 and 5 never needed more and keep the old limit.
JOINT_LIMIT_WIDE = 3.5
CONTINUOUS_JOINT_LIMITS = {0: PLANNER_JOINT_LIMIT, 2: JOINT_LIMIT_WIDE, 4: PLANNER_JOINT_LIMIT, 6: JOINT_LIMIT_WIDE}


def joint_limit(i):
    """The largest |angle| the planner is asked for on joint index i (0-based) when it is continuous; the old limit for any other joint."""
    return CONTINUOUS_JOINT_LIMITS.get(i, PLANNER_JOINT_LIMIT)
# A goal that needs more than this on any single joint is a big arm
# reconfiguration; refuse it instead of swinging the arm through it.
MAX_JOINT_DELTA_RAD = 2.6
# Stop searching once a solution this close to the current pose is found.
GOOD_ENOUGH_DELTA_RAD = 0.8


def yaw_candidates_deg(span_deg=90, step_deg=15):
    """0, +15, -15, +30, ... ordered by |yaw|, so the original grasp
    orientation is tried first."""
    out = [0]
    for k in range(step_deg, span_deg + 1, step_deg):
        out += [k, -k]
    return out


def quat_mul(a, b):
    """Hamilton product a*b for quaternions in (x, y, z, w) order."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate_about_tool_z(quat_xyzw, yaw_deg):
    """quat * Rz(yaw): an intrinsic rotation about the tool's own z axis."""
    h = math.radians(yaw_deg) / 2.0
    return quat_mul(tuple(quat_xyzw), (0.0, 0.0, math.sin(h), math.cos(h)))


def unwrap_to_seed(solution, seed, limit=None):
    """Return `solution` with each continuous joint replaced by its
    equivalent angle (k*2*pi apart) nearest the seed that is still inside
    +-limit (joint_limit(i) for each joint unless a limit is given); None if
    some continuous joint has no such equivalent."""
    out = []
    for i, (v, s) in enumerate(zip(solution, seed)):
        if i in CONTINUOUS_JOINT_IDX:
            lim = joint_limit(i) if limit is None else limit
            cands = [v + 2.0 * math.pi * k for k in (-2, -1, 0, 1, 2)]
            cands = [c for c in cands if abs(c) <= lim]
            if not cands:
                return None
            v = min(cands, key=lambda c: abs(c - s))
        out.append(v)
    return out


EDGE_SEED_BAND_RAD = 2.6      # a widened joint (3 / 7) this far from zero has an equivalent angle (+-2*pi) that is also inside its limit


def equivalent_seeds(seed):
    """The seed first, then the seeds that describe the SAME arm pose with joint 3 and/or joint 7 written 2*pi apart (only for a joint near +-pi, where the other writing is still inside
    joint_limit). The IK service is not indifferent to how the same angle is written (2026-11-03: the same wrist pose, seeded -2.98 rad, was solved cleanly 1 time in 10, seeded +3.30 rad 10 in 10)."""
    variants = [[float(v) for v in seed]]
    for i in CONTINUOUS_JOINT_IDX:
        if joint_limit(i) <= PLANNER_JOINT_LIMIT or abs(seed[i]) < EDGE_SEED_BAND_RAD:
            continue
        other = seed[i] - math.copysign(2.0 * math.pi, seed[i])
        if abs(other) > joint_limit(i):
            continue
        variants += [v[:i] + [float(other)] + v[i + 1:] for v in variants]
    return variants


def joint_deltas(solution, seed):
    return [abs(a - b) for a, b in zip(solution, seed)]


def better(candidate, best):
    """candidate/best: (max_delta, total_delta, ...) tuples; smaller max
    single-joint change wins, total change breaks ties."""
    return best is None or (candidate[0], candidate[1]) < (best[0], best[1])
