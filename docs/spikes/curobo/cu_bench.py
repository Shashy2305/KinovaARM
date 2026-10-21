"""cuRobo vs the MoveIt baseline (docs/data/ik_baseline_moveit_2026-10-20.json): same targets, same seed pose."""
import json, sys, time
import numpy as np
import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose, JointState

base = json.load(open('/mnt/ros_workspace/Shashproject/docs/data/ik_baseline_moveit_2026-10-20.json'))
seed = np.array(base['seed'])
rows = {(round(r['x'], 6), round(r['y'], 6), r['yaw']) for r in base['rows']}
targets = sorted(rows)
print('targets', len(targets), 'seed', np.round(seed, 2))
GRASP_XYZW = np.array([0.773, 0.635, -0.015, 0.019])

def rot_tool_z(q_xyzw, yaw_deg):
    x, y, z, w = q_xyzw; h = np.radians(yaw_deg) / 2; zz, ww = np.sin(h), np.cos(h)
    # q * Rz(yaw)
    return np.array([w * 0 + w * 0 + x * ww + y * zz - 0, 0, 0, 0])  # placeholder replaced below

from scipy.spatial.transform import Rotation as R
def quat_wxyz(yaw):
    q = (R.from_quat(GRASP_XYZW) * R.from_euler('z', yaw, degrees=True)).as_quat()      # xyzw, intrinsic rotation about the tool z
    return np.array([q[3], q[0], q[1], q[2]])

N = len(targets)
pos = torch.tensor([[t[0], t[1], 0.25] for t in targets], device='cuda', dtype=torch.float32)
quat = torch.tensor(np.stack([quat_wxyz(t[2]) for t in targets]), device='cuda', dtype=torch.float32)
cfg = InverseKinematicsCfg.create(robot='/mnt/ros_workspace/curobo_spike/gen3_robotiq_flange.yml', num_seeds=32, max_batch_size=N,
                                  self_collision_check=True, position_tolerance=0.005, orientation_tolerance=0.05)
ik = InverseKinematics(cfg)
print('tool frames', ik.tool_frames, 'dof', len(ik.joint_names) if hasattr(ik, 'joint_names') else '?')
goal = GoalToolPose.from_poses({ik.tool_frames[0]: Pose(position=pos, quaternion=quat)}, num_goalset=1)
res = ik.solve_pose(goal)                    # warm-up (CUDA graph capture)
torch.cuda.synchronize()
times = []
for _ in range(10):
    t0 = time.time(); res = ik.solve_pose(goal); torch.cuda.synchronize(); times.append(1000 * (time.time() - t0))
ok = res.success.reshape(-1).cpu().numpy().astype(bool)
sol = res.js_solution.position.reshape(N, -1).cpu().numpy()[:, :7]
perr = res.position_error.reshape(-1).cpu().numpy() * 1000
print(f'batch of {N}: solved {100 * ok.mean():.0f}%, {np.median(times):.0f} ms per BATCH (median of 10) = {np.median(times) / N:.2f} ms per pose; position error mean {perr[ok].mean():.2f} mm')
dq = np.abs(sol[ok] - seed[None, :sol.shape[1]]).max(axis=1)
print(f'largest single-joint change from the current pose (best-cost solution): median {np.median(dq):.2f} rad, p90 {np.percentile(dq, 90):.2f}, max {dq.max():.2f}')
# several solutions per target: take the one nearest to the current pose
try:
    res8 = ik.solve_pose(goal, return_seeds=16)
    S = res8.js_solution.position.reshape(N, 16, -1).cpu().numpy()[:, :, :7]; OK = res8.success.reshape(N, 16).cpu().numpy().astype(bool)
    best = []
    for i in range(N):
        if OK[i].any():
            cand = S[i][OK[i]]
            best.append(np.abs(cand - seed[None, :cand.shape[1]]).max(axis=1).min())
    best = np.array(best)
    print(f'with 16 returned seeds, nearest successful solution: median {np.median(best):.2f} rad, p90 {np.percentile(best, 90):.2f}, max {best.max():.2f} ({len(best)}/{N} targets have a solution)')
except Exception as e:
    print('return_seeds variant failed:', str(e)[:200])
