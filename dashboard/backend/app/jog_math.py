"""
The numbers behind the dashboard jog (pure numpy, no ROS): how far one tick may move each joint, and whether the resulting pose is allowed.

Every limit lives HERE, on the server. The browser only sends "which way, how fast (a fraction)"; it can neither raise a cap nor skip a check.
"""
import numpy as np

from thesis_robot import safety_geometry as sg


JOINT_MAX_RAD_S = 0.30        # one joint at speed 1.0 (the Kinova limit is 1.2-1.4); the UI slider starts at 25 %
LINEAR_MAX_M_S = 0.08
ANGULAR_MAX_RAD_S = 0.30
HORIZON_S = 0.20              # the reference step of one command: dq = velocity * HORIZON_S (the caps above are speeds, so this only converts)
HOLD_S = 0.20                 # each command asks for constant velocity for this long ...
RAMP_S = 0.12                 # ... then a linear slow-down to a stop over this long: if commands stop the arm coasts about v * (HOLD_S + RAMP_S / 2) and halts
LEAD_MAX_S = 0.25             # the integrated reference may run at most this many seconds of motion ahead of the measured joints (a lagging arm cannot wind it up)
# each message is back-dated by BACK_S (header.stamp in the past, first waypoint on the reference line): the controller then samples the line itself, not the measured state,
# so consecutive messages join exactly (a message starting at the measured state snapped the desired position back by the tracking lag every 80 ms: a 12.5 Hz velocity ripple of
# 33 % of the speed, measured on the arm 2026-10-22)
BACK_S = 0.10
ACCEL_RAD_S2 = 0.8            # the commanded joint velocity changes by at most this per second (full speed is reached in 0.4 s)
CONT_S = 0.18                 # a command within this of the previous one continues its line (shorter than HOLD_S, so the previous message still covers it)
NOMINAL_TICK_S = 0.08
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


def approach_velocity(v_now, v_target, dt):
    """The commanded velocity one tick later: moves toward v_target by at most ACCEL_RAD_S2 * dt (all joints scaled together, so the direction of the move is kept)."""
    v_now, v_target = np.asarray(v_now, float), np.asarray(v_target, float)
    diff = v_target - v_now
    peak = np.max(np.abs(diff))
    limit = ACCEL_RAD_S2 * dt
    if peak <= limit or peak == 0:
        return v_target.copy()
    return v_now + diff * (limit / peak)


def velocity_profile(q_ref, v, back_s=BACK_S, hold_s=HOLD_S, ramp_s=RAMP_S):
    """Waypoints for ONE message of a continuous jog, times counted from the message's (back-dated) stamp: constant velocity v (7,) through the reference q_ref (reached at
    back_s) until back_s + hold_s, then a linear slow-down to zero over ramp_s. Every waypoint carries a velocity, so the controller's spline follows the line exactly.
    Returns (times (n,), positions (n,7), velocities (n,7))."""
    q_ref, v = np.asarray(q_ref, float), np.asarray(v, float)
    t_end_hold = back_s + hold_s
    times = [0.0, t_end_hold]
    pos = [q_ref - v * back_s, q_ref + v * hold_s]
    vel = [v, v]
    q_end_hold = pos[1]
    for f in (1 / 3, 2 / 3, 1.0):
        tau = ramp_s * f
        times.append(t_end_hold + tau)
        pos.append(q_end_hold + v * (tau - tau * tau / (2 * ramp_s)))
        vel.append(v * (1 - tau / ramp_s))
    return np.array(times), np.array(pos), np.array(vel)


def coast_displacement(v, hold_s=HOLD_S, ramp_s=RAMP_S):
    """How far each joint still travels after the last message if no more arrive."""
    return np.asarray(v, float) * (hold_s + ramp_s / 2.0)
