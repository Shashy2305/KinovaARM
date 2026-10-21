import sys, yaml, numpy as np, torch
from scipy.spatial.transform import Rotation as R
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState
variants = {'vel 0.3 only': (0.3, 1.0, 1.0), 'vel 0.3 + acc 0.172': (0.3, 0.172, 1.0), 'vel 0.3 + acc 0.172 + jerk 0.016': (0.3, 0.172, 0.016),
            'vel 0.3 + acc 0.172 + jerk 0.05': (0.3, 0.172, 0.05), 'vel 0.45 + acc 0.172 + jerk 0.05': (0.45, 0.172, 0.05)}
home = np.array([0.0, -0.35, 3.14, -2.27, 0.0, 0.96, 1.57])
rng = np.random.default_rng(2); goals = []
for i in range(8):
    x, y = rng.uniform(0.25, 0.55), rng.uniform(-0.30, 0.30); yaw = rng.choice([0, 45, 90, 135])
    goals.append((x, y, (R.from_quat([0.773, 0.635, -0.015, 0.019]) * R.from_euler('z', yaw, degrees=True)).as_quat()))
scene = {'cuboid': {'table': {'dims': [2.0, 2.0, 0.05], 'pose': [0.5, 0.0, -0.0375, 1, 0, 0, 0]}}}
for name, (v, a, j) in variants.items():
    d = yaml.safe_load(open('gen3_robotiq_flange.yml')); cs = d['kinematics']['cspace']
    cs['velocity_scale'] = [v] * 7; cs['acceleration_scale'] = [a] * 7; cs['jerk_scale'] = [j] * 7
    yaml.safe_dump(d, open('/tmp/s4/cu_variant.yml', 'w'), default_flow_style=False)
    planner = MotionPlanner(MotionPlannerCfg.create(robot='/tmp/s4/cu_variant.yml', scene_model=scene)); planner.warmup(enable_graph=True, num_warmup_iterations=3)
    dt = planner.trajopt_solver.config.interpolation_dt
    q0 = JointState.from_position(torch.tensor(home[None], device='cuda', dtype=torch.float32), joint_names=planner.joint_names)
    ok, dur = 0, []
    for x, y, q in goals:
        goal = GoalToolPose(tool_frames=planner.tool_frames, position=torch.tensor([[[[[x, y, 0.37]]]]], device='cuda', dtype=torch.float32),
                            quaternion=torch.tensor([[[[[q[3], q[0], q[1], q[2]]]]]], device='cuda', dtype=torch.float32))
        res = planner.plan_pose(goal, q0)
        if res is not None and bool(res.success.any()):
            ok += 1; dur.append(len(res.get_interpolated_plan().position.squeeze()) * dt)
    print(f'{name:36s}: planned {ok}/{len(goals)}' + (f', duration median {np.median(dur):.2f} s, max {max(dur):.2f} s' if dur else ''))
    del planner; torch.cuda.empty_cache()
