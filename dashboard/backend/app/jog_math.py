"""
The numbers behind the dashboard jog (pure numpy, no ROS): how far one tick may move each joint, and whether the resulting pose is allowed.

Every limit lives HERE, on the server. The browser only sends "which way, how fast (a fraction)"; it can neither raise a cap nor skip a check.
"""
import numpy as np

from thesis_robot import safety_geometry as sg


JOINT_MAX_RAD_S = 0.30        # one joint at speed 1.0 (the Kinova limit is 1.2-1.4); the UI slider starts at 25 %
LINEAR_MAX_M_S = 0.08
ANGULAR_MAX_RAD_S = 0.30
HORIZON_S = 0.20              # each command asks the controller to reach (measured + velocity * HORIZON) in HORIZON seconds, so it stops by itself if commands stop
JOINT_LIMIT_MARGIN_RAD = 0.10
DLS_LAMBDA = 0.05             # damping of the Cartesian solve
MIN_SINGULAR = 0.02           # smallest singular value of the 6x7 Jacobian below which a Cartesian move is refused (near a singularity)


def clamp_speed(speed):
    return float(np.clip(speed, 0.0, 1.0))


def joint_step(q, joint_index, direction, speed):
    """dq (7,) for one tick moving a single joint. direction: +1 / -1 (anything else = 0)."""
    dq = np.zeros(len(q))
    d = 1.0 if direction > 0 else -1.0 if direction < 0 else 0.0
    dq[joint_index] = d * clamp_speed(speed) * JOINT_MAX_RAD_S * HORIZON_S
    return dq


def cart_step(kinematics, q, v, speed, frame='base', link='end_effector_link'):
    """dq (7,) for one tick of a Cartesian move. v: 6 numbers in [-1, 1] (x, y, z, rot x, rot y, rot z) in the base frame, or in the tool frame (frame='tool').
    Returns (dq, None) or (None, reason). Joint speeds are scaled down together (the direction of the move is kept) if any joint would exceed JOINT_MAX_RAD_S."""
    v = np.clip(np.asarray(v, float), -1.0, 1.0)
    if not np.any(v):
        return np.zeros(len(q)), None
    s = clamp_speed(speed)
    twist = np.concatenate([v[:3] * LINEAR_MAX_M_S * s, v[3:] * ANGULAR_MAX_RAD_S * s])
    if frame == 'tool':
        R = kinematics.pose_for([link], q)[link][:3, :3]
        twist = np.concatenate([R @ twist[:3], R @ twist[3:]])
    J = kinematics.jacobian(q, link)
    sv = np.linalg.svd(J, compute_uv=False)
    if sv[-1] < MIN_SINGULAR:
        return None, f'near a singularity (smallest singular value {sv[-1]:.3f}): move a joint instead'
    dq_dot = J.T @ np.linalg.solve(J @ J.T + DLS_LAMBDA ** 2 * np.eye(6), twist)
    peak = np.max(np.abs(dq_dot))
    cap = JOINT_MAX_RAD_S * max(s, 0.05)
    if peak > cap:
        dq_dot = dq_dot * (cap / peak)
    return dq_dot * HORIZON_S, None


def check_joint_limits(kinematics, q_now, q_new):
    """A joint inside the margin of its limit may move back out, never further in."""
    lo, hi = kinematics.limits()
    m = JOINT_LIMIT_MARGIN_RAD
    for i in range(len(q_new)):
        n = f'joint_{i + 1}'
        if np.isfinite(lo[i]) and q_new[i] < lo[i] + m and q_new[i] <= q_now[i]:
            return False, f'{n} would reach {np.degrees(q_new[i]):.0f} deg, inside {np.degrees(m):.0f} deg of its lower limit {np.degrees(lo[i]):.0f}'
        if np.isfinite(hi[i]) and q_new[i] > hi[i] - m and q_new[i] >= q_now[i]:
            return False, f'{n} would reach {np.degrees(q_new[i]):.0f} deg, inside {np.degrees(m):.0f} deg of its upper limit {np.degrees(hi[i]):.0f}'
    return True, 'ok'


def guarded_positions(kinematics, q, gripper=None):
    """{guarded link: (x, y, z)} in base_link for the arm configuration q (gripper: {joint: value} for the finger joint, default open)."""
    poses = kinematics.pose_for(sg.GUARDED_LINKS, q, gripper)
    return {name: tuple(float(v) for v in T[:3, 3]) for name, T in poses.items()}


def check_pose(kinematics, q_now, q_new, geom, gripper=None):
    """(ok, reason): the table / rear limits of safety_geometry applied to the pose after the step. A link that is ALREADY past a limit may move away from it but never
    further in (same rule as the planner's start_floors), so the arm can always be jogged out of a bad pose. geom None = refuse (no table known, no safe floor)."""
    if geom is None:
        return False, 'no table geometry recorded (Operations > table): the jog cannot know where the table is'
    ok, why = check_joint_limits(kinematics, q_now, q_new)
    if not ok:
        return False, why
    now = guarded_positions(kinematics, q_now, gripper)
    new = guarded_positions(kinematics, q_new, gripper)
    floors = sg.start_floors(now, geom)
    for link, (x, _y, z) in new.items():
        if z < floors[link]:
            return False, (f'{link} would be at z={z:.3f}, below its limit {floors[link]:.3f} '
                           f'(table top {geom["table_top_z"]:.3f} + {sg.LINK_TABLE_CLEARANCE_M:.2f})')
        if x < sg.REAR_KEEP_OUT_X and x <= now[link][0] + 1e-4:
            return False, f'{link} would be at x={x:.3f}, behind the rear limit {sg.REAR_KEEP_OUT_X:.2f}'
    return True, 'ok'
