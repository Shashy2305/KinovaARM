"""
PLAN-ONLY safety sweep for the grasp-parameter offsets exploration mode would need to randomize
(grasp_z_offset, hover_above_m). Task 9 of the RL-project handoff found these have no established
safe range (unlike exploration.py's 8 motion-timing parameters). This never executes: it uses the
same stub as plan_only_test.py (pretends the arm is at home, applies a temporary table slab + rear
wall to MoveIt's planning scene, plans with the controller's own methods, never moves).

    source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
    python3 -u /mnt/ros_workspace/Shashproject/scripts/dev/grasp_param_safety_sweep.py

For each offset value and a spread of 18 real logged cup/mouse pick positions (from the handoff CSV
export), computes grasp_z and hover_z via the real ws.pick_heights() heuristic and checks whether a
collision-free plan exists for each. An offset value "passes" only if every sampled position plans
successfully; the safe range is the contiguous band of offsets that all pass.
"""
import csv
import functools
import threading
import time
import types

import rclpy
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene, GetPositionFK, GetPositionIK
from pymoveit2 import MoveIt2
from rclpy.node import Node

from thesis_robot import safety_geometry as sg
from thesis_robot import wrist_servo as ws
from thesis_robot.arm_controller_node import (ArmControllerNode as A, BASE_LINK, EE_LINK, GRASP_QUAT_XYZW,
                                               GROUP_NAME, HOME_JOINTS, JOINT_NAMES)

GEOM = {'table_top_z': -0.06, 'x': [-0.15, 0.75], 'y': [-0.80, 0.85], 'rear_wall_x': -0.20}
REAL_GEOM, _ = sg.load_geometry()  # real calibrated table_top_z, used to compute grasp_z/hover_z

POSITIONS_CSV = '/mnt/ros_workspace/handoff_2026-11-05/data/csv/picks.csv'


def load_sample_positions(n_cup=10, n_mouse=8):
    with open(POSITIONS_CSV) as f:
        rows = list(csv.DictReader(f))
    live = [r for r in rows if r['live'] == 'True' and r['label'] in ('cup', 'mouse') and r['x'] and r['y'] and r['z']]

    def spread(label, n):
        rs = sorted((r for r in live if r['label'] == label), key=lambda r: (float(r['x']), float(r['y'])))
        step = max(1, len(rs) // n)
        return [(r['label'], float(r['x']), float(r['y']), float(r['z'])) for r in rs[::step][:n]]

    return spread('cup', n_cup) + spread('mouse', n_mouse)


rclpy.init()
node = Node('grasp_param_safety_sweep')
helper = rclpy.create_node('grasp_param_safety_sweep_moveit')
ex = rclpy.executors.MultiThreadedExecutor(2)
ex.add_node(node)
threading.Thread(target=ex.spin, daemon=True).start()
m = MoveIt2(node=helper, joint_names=JOINT_NAMES, base_link_name=BASE_LINK, end_effector_name=EE_LINK, group_name=GROUP_NAME)
m.max_velocity = 0.2
m.max_acceleration = 0.2
h = types.SimpleNamespace(
    moveit2=m, get_logger=node.get_logger, get_node_names_and_namespaces=node.get_node_names_and_namespaces,
    _apply_scene_client=node.create_client(ApplyPlanningScene, '/apply_planning_scene'),
    _fk_client=node.create_client(GetPositionFK, '/compute_fk'), _ik_client=node.create_client(GetPositionIK, '/compute_ik'))
for name in ('_move_group_count', '_ensure_safety_scene', '_fk_link_positions', '_check_trajectory', '_ik', '_solve_goal_joints', '_plan_joint_goal'):
    setattr(h, name, functools.partial(getattr(A, name), h))
h._current_joint_vector = lambda: list(HOME_JOINTS)
h.TURN_MARGIN_RAD = A.TURN_MARGIN_RAD
time.sleep(3.0)
A._ensure_safety_scene(h, GEOM)
print('table + wall applied to the live scene', flush=True)


def plan_ok(flange_xyz):
    try:
        goal, quat = h._solve_goal_joints(flange_xyz, GRASP_QUAT_XYZW, True)
        traj, why = h._plan_joint_goal(goal, GEOM)
    except Exception as e:
        return False, str(e)
    if traj is None:
        return False, why
    return True, None


def sweep(label_grid, offset_field, positions):
    """offset_field: 'grasp_z_offset' or 'hover_above_m'. label_grid: list of offset values to try."""
    results = []
    for off in label_grid:
        kwargs = {'grasp_offset': off} if offset_field == 'grasp_z_offset' else {'hover_above': off}
        n_pass = n_fail = 0
        fails = []
        for label, ox, oy, oz in positions:
            tip_clear = 0.03
            grasp_z, hover_z, lift_z, low_floor = ws.pick_heights(
                oz, label, REAL_GEOM['table_top_z'], sg.TCP_REACH_M, sg.flange_floor_z(REAL_GEOM), tip_clear,
                sg.TCP_CLEARANCE_M, approach_z=0.0, **kwargs)
            target_z = grasp_z if offset_field == 'grasp_z_offset' else hover_z
            ok, why = plan_ok((ox, oy, target_z))
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                fails.append((label, ox, oy, round(target_z, 3), why))
        results.append((off, n_pass, n_fail, fails))
        print(f'{offset_field}={off:+.3f}: {n_pass}/{n_pass + n_fail} positions plan ok' +
              (f'  -- e.g. {fails[0][:4]} ({fails[0][4]})' if fails else ''), flush=True)
    return results


positions = load_sample_positions()
print(f'sampled {len(positions)} real logged positions '
      f'({sum(1 for p in positions if p[0] == "cup")} cup, {sum(1 for p in positions if p[0] == "mouse")} mouse)', flush=True)

print('\n-- grasp_z_offset sweep (hover_above_m at default 0.12) --', flush=True)
r1 = sweep([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03], 'grasp_z_offset', positions)

print('\n-- hover_above_m sweep (grasp_z_offset at default 0.0) --', flush=True)
r2 = sweep([0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24], 'hover_above_m', positions)

ps = PlanningScene()
ps.is_diff = True
for oid in ('table_slab', 'rear_safety_wall'):
    co = CollisionObject()
    co.id = oid
    co.header.frame_id = BASE_LINK
    co.operation = CollisionObject.REMOVE
    ps.world.collision_objects.append(co)
f = h._apply_scene_client.call_async(ApplyPlanningScene.Request(scene=ps))
t0 = time.time()
while not f.done() and time.time() - t0 < 5:
    time.sleep(0.05)
print('\ntest objects removed:', f.result().success if f.done() else 'TIMEOUT', flush=True)
