"""cuRobo collision-aware motion planning on the Gen3 + Robotiq from the home pose to hover poses over the table (pick-style goals)."""
import json, time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState

TABLE_TOP_Z = -0.0125
scene = {'cuboid': {'table': {'dims': [2.0, 2.0, 0.05], 'pose': [0.5, 0.0, TABLE_TOP_Z - 0.025, 1, 0, 0, 0]},
                    'wall_rear': {'dims': [0.05, 2.0, 1.5], 'pose': [-0.40, 0.0, 0.7, 1, 0, 0, 0]}}}
cfg = MotionPlannerCfg.create(robot='/mnt/ros_workspace/curobo_spike/gen3_robotiq_flange.yml', scene_model=scene)
planner = MotionPlanner(cfg)
t0 = time.time(); planner.warmup(enable_graph=True, num_warmup_iterations=5); print('warm-up %.1f s' % (time.time() - t0))
dt = planner.trajopt_solver.config.interpolation_dt
home = np.array([0.0, -0.35, 3.14, -2.27, 0.0, 0.96, 1.57])
q0 = JointState.from_position(torch.tensor(home[None], device='cuda', dtype=torch.float32), joint_names=planner.joint_names)
GRASP_XYZW = np.array([0.773, 0.635, -0.015, 0.019])
rng = np.random.default_rng(2)
rows = []
for i in range(24):
    x, y = rng.uniform(0.25, 0.55), rng.uniform(-0.30, 0.30)
    yaw = rng.choice([0, 45, 90, 135])
    q = (R.from_quat(GRASP_XYZW) * R.from_euler('z', yaw, degrees=True)).as_quat()
    goal = GoalToolPose(tool_frames=planner.tool_frames,
                        position=torch.tensor([[[[[x, y, 0.37]]]]], device='cuda', dtype=torch.float32),
                        quaternion=torch.tensor([[[[[q[3], q[0], q[1], q[2]]]]]], device='cuda', dtype=torch.float32))
    t0 = time.time(); res = planner.plan_pose(goal, q0); torch.cuda.synchronize(); ms = 1000 * (time.time() - t0)
    ok = res is not None and bool(res.success.any())
    row = {'x': x, 'y': y, 'yaw': int(yaw), 'ok': ok, 'ms': ms}
    if ok:
        tr = res.get_interpolated_plan().position.reshape(-1, planner.trajopt_solver.config.action_dim if hasattr(planner.trajopt_solver.config, 'action_dim') else 7)[:, :7].cpu().numpy() if False else res.get_interpolated_plan().position.squeeze().cpu().numpy()[:, :7]
        row.update(duration_s=len(tr) * dt, path_rad=float(np.abs(np.diff(tr, axis=0)).sum(axis=0).max()), max_joint_move=float(np.abs(tr[-1] - tr[0]).max()),
                   peak_speed=float(np.abs(np.diff(tr, axis=0)).max() / dt))
    rows.append(row)
okr = [r for r in rows if r['ok']]
print(f'{len(okr)}/{len(rows)} goals planned; planning time median {np.median([r["ms"] for r in okr]):.0f} ms, p90 {np.percentile([r["ms"] for r in okr], 90):.0f} ms')
print(f'trajectory duration: median {np.median([r["duration_s"] for r in okr]):.2f} s, p90 {np.percentile([r["duration_s"] for r in okr], 90):.2f} s, max {max(r["duration_s"] for r in okr):.2f} s')
print(f'largest single-joint move: median {np.median([r["max_joint_move"] for r in okr]):.2f} rad, p90 {np.percentile([r["max_joint_move"] for r in okr], 90):.2f}; peak joint speed median {np.median([r["peak_speed"] for r in okr]):.2f} rad/s')
json.dump(rows, open('/mnt/ros_workspace/Shashproject/docs/data/curobo_plan_home_to_hover_2026-10-20.json', 'w'), indent=1)
