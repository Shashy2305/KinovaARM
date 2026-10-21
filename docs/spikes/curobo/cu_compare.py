import sys, yaml, json, numpy as np, torch
sys.path.insert(0, '/mnt/ros_workspace/Shashproject/src/thesis_robot')
from thesis_robot import trajectory_smoothing as ts
from scipy.spatial.transform import Rotation as R
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState
VMAX = np.array([1.3963] * 4 + [1.2218] * 3); AMAX = np.array([8.6] * 7)
VS, AS, JMAX = 0.45, 0.2, 8.0                       # the same caps for all three: speed 45% of the limit, acceleration 20%, jerk 8 rad/s^3

def scurve_T(d, v, a, j):
    """Shortest time of a rest-to-rest move of length d with speed/accel/jerk limits (jerk-limited S-curve)."""
    def Tv(vp):                                      # time to go from 0 to vp and the distance covered, accelerating at most a with jerk j
        return (vp / a + a / j) if vp * j >= a * a else 2 * np.sqrt(vp / j)
    if d <= 1e-9: return 0.0
    if d >= v * Tv(v): return d / v + Tv(v)
    lo, hi = 0.0, v
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mid * Tv(mid) < d else (lo, mid)
    vp = (lo + hi) / 2
    return 2 * (vp / 2 * Tv(vp)) / vp * 0 + (d / vp + Tv(vp))   # d/vp + Tv(vp)

home = np.array([0.0, -0.35, 3.14, -2.27, 0.0, 0.96, 1.57])
d = yaml.safe_load(open('gen3_robotiq_flange.yml')); cs = d['kinematics']['cspace']
cs['velocity_scale'] = [VS] * 7; cs['acceleration_scale'] = [AS * 8.6 / 10.0] * 7; cs['jerk_scale'] = [JMAX / 500.0] * 7
yaml.safe_dump(d, open('/tmp/s4/cu_cmp.yml', 'w'), default_flow_style=False)
scene = {'cuboid': {'table': {'dims': [2.0, 2.0, 0.05], 'pose': [0.5, 0.0, -0.0375, 1, 0, 0, 0]}}}
planner = MotionPlanner(MotionPlannerCfg.create(robot='/tmp/s4/cu_cmp.yml', scene_model=scene)); planner.warmup(enable_graph=True, num_warmup_iterations=3)
dt = planner.trajopt_solver.config.interpolation_dt
q0 = JointState.from_position(torch.tensor(home[None], device='cuda', dtype=torch.float32), joint_names=planner.joint_names)
rng = np.random.default_rng(2); rows = []
for i in range(24):
    x, y = rng.uniform(0.25, 0.55), rng.uniform(-0.30, 0.30); yaw = rng.choice([0, 45, 90, 135])
    q = (R.from_quat([0.773, 0.635, -0.015, 0.019]) * R.from_euler('z', yaw, degrees=True)).as_quat()
    goal = GoalToolPose(tool_frames=planner.tool_frames, position=torch.tensor([[[[[x, y, 0.37]]]]], device='cuda', dtype=torch.float32),
                        quaternion=torch.tensor([[[[[q[3], q[0], q[1], q[2]]]]]], device='cuda', dtype=torch.float32))
    res = planner.plan_pose(goal, q0)
    if res is None or not bool(res.success.any()): continue
    tr = res.get_interpolated_plan().position.squeeze().cpu().numpy()[:, :7]
    dq = np.abs(tr[-1] - home)
    ours = ts.smooth([list(home), list(tr[-1])], list(VMAX), list(AMAX), vel_scale=VS, acc_scale=AS, min_duration=0.6)
    sc = max(scurve_T(dq[k], VS * VMAX[k], AS * AMAX[k], JMAX) for k in range(7))
    rows.append((len(tr) * dt, ours['duration'], sc, float(dq.max())))
a = np.array(rows)
print(f'{len(a)} moves from home, same caps (speed {VS:.0%}, accel {AS:.0%}, jerk {JMAX:.0f} rad/s^3); largest joint move median {np.median(a[:, 3]):.2f} rad')
print(f'  cuRobo trajectory duration:            median {np.median(a[:, 0]):.2f} s  (p90 {np.percentile(a[:, 0], 90):.2f})')
print(f'  our minimum-jerk retimer (today):      median {np.median(a[:, 1]):.2f} s  (p90 {np.percentile(a[:, 1], 90):.2f})')
print(f'  jerk-limited S-curve on the same path: median {np.median(a[:, 2]):.2f} s  (p90 {np.percentile(a[:, 2], 90):.2f})  -> {100 * (1 - np.median(a[:, 2]) / np.median(a[:, 1])):.0f}% shorter than minimum-jerk')
json.dump(a.tolist(), open('/mnt/ros_workspace/Shashproject/docs/data/curobo_vs_minjerk_vs_scurve_2026-10-20.json', 'w'))
