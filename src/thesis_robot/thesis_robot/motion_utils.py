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


def unwrap_to_seed(solution, seed, limit=PLANNER_JOINT_LIMIT):
    """Return `solution` with each continuous joint replaced by its
    equivalent angle (k*2*pi apart) nearest the seed that is still inside
    +-limit; None if some continuous joint has no such equivalent."""
    out = []
    for i, (v, s) in enumerate(zip(solution, seed)):
        if i in CONTINUOUS_JOINT_IDX:
            cands = [v + 2.0 * math.pi * k for k in (-2, -1, 0, 1, 2)]
            cands = [c for c in cands if abs(c) <= limit]
            if not cands:
                return None
            v = min(cands, key=lambda c: abs(c - s))
        out.append(v)
    return out


def joint_deltas(solution, seed):
    return [abs(a - b) for a, b in zip(solution, seed)]


def better(candidate, best):
    """candidate/best: (max_delta, total_delta, ...) tuples; smaller max
    single-joint change wins, total change breaks ties."""
    return best is None or (candidate[0], candidate[1]) < (best[0], best[1])
