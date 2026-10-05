"""
PLAN-ONLY check of arm_controller's real planning code against the LIVE move_group.
It NEVER executes: it pretends the arm is at home, applies a temporary table slab +
rear wall to MoveIt's planning scene, plans a few targets with the controller's own
methods (seeded IK + tool-yaw freedom + Pilz PTP + FK path check), prints how big the
motions are, then removes the temporary objects.

    source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
    python3 -u -s /mnt/ros_workspace/Shashproject/scripts/dev/plan_only_test.py

Needs robot_bringup running (move_group + /compute_ik + /compute_fk). Expected:
six "ok" lines with max joint swing <= ~2.2 rad and min link x >= +0.10, and three
"REFUSED" lines for targets that would put the fingers in/near the table or behind
the robot. Anything else is a regression in the motion/safety code.
"""
import time, types, threading, functools, rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
from moveit_msgs.srv import ApplyPlanningScene, GetPositionFK, GetPositionIK
from moveit_msgs.msg import PlanningScene, CollisionObject
from thesis_robot import safety_geometry as sg
from thesis_robot.arm_controller_node import ArmControllerNode as A, HOME_JOINTS, JOINT_NAMES, BASE_LINK, EE_LINK, GROUP_NAME, GRASP_QUAT_XYZW

GEOM = {'table_top_z': -0.06, 'x': [-0.15, 0.75], 'y': [-0.80, 0.85], 'rear_wall_x': -0.20}
rclpy.init()
node = Node('controller_plan_test')                                # spun by the executor (like arm_controller)
helper = rclpy.create_node('controller_plan_test_moveit')          # private node for pymoveit2 (like arm_controller)
ex = rclpy.executors.MultiThreadedExecutor(2); ex.add_node(node); threading.Thread(target=ex.spin, daemon=True).start()
m = MoveIt2(node=helper, joint_names=JOINT_NAMES, base_link_name=BASE_LINK, end_effector_name=EE_LINK, group_name=GROUP_NAME)
m.max_velocity = 0.2; m.max_acceleration = 0.2
h = types.SimpleNamespace(moveit2=m, get_logger=node.get_logger, get_node_names_and_namespaces=node.get_node_names_and_namespaces,
                          _apply_scene_client=node.create_client(ApplyPlanningScene, '/apply_planning_scene'),
                          _fk_client=node.create_client(GetPositionFK, '/compute_fk'), _ik_client=node.create_client(GetPositionIK, '/compute_ik'))
for name in ('_move_group_count', '_ensure_safety_scene', '_fk_link_positions', '_check_trajectory', '_ik', '_solve_goal_joints', '_plan_joint_goal'):
    setattr(h, name, functools.partial(getattr(A, name), h))
h._current_joint_vector = lambda: list(HOME_JOINTS)      # pretend the arm is at home (planning only)
time.sleep(3.0)
A._ensure_safety_scene(h, GEOM); print('table + wall applied to the live scene', flush=True)

def run(label, pos):
    t = time.time()
    try:
        goal, quat = h._solve_goal_joints(pos, GRASP_QUAT_XYZW, True)
        traj, why = h._plan_joint_goal(goal, GEOM)
    except Exception as e:
        print(f'  {label:36s} REFUSED: {e}', flush=True); return
    if traj is None: print(f'  {label:36s} no safe plan: {why}', flush=True); return
    names = list(traj.joint_names); q = [[p.positions[names.index(j)] for j in JOINT_NAMES] for p in traj.points]
    swing = max(max(abs(q[i][k] - q[0][k]) for i in range(len(q))) for k in range(7))
    minx = minz = 9
    for i in sg.sample_indices(len(q), 25):
        for l, v in h._fk_link_positions(JOINT_NAMES, q[i]).items(): minx = min(minx, v[0]); minz = min(minz, v[2])
    print(f'  {label:36s} ok: {len(q):3d} pts, max joint swing {swing:4.2f} rad, min link x {minx:+.2f}, min link z {minz:+.2f}  ({time.time()-t:.1f}s)', flush=True)

print('-- plan-only from home', flush=True)
for label, pos in [('near-base cup flange z=0.22', (0.16, 0.13, 0.22)), ('mid-table z=0.30', (0.40, 0.0, 0.30)), ('far-right z=0.30', (0.45, 0.30, 0.30))]:
    for _ in range(2): run(label, pos)
print('-- targets that must NOT be allowed', flush=True)
run('fingertips into table (flange z=0.05)', (0.40, 0.0, 0.05))
run('too close to table (flange z=0.15)', (0.40, 0.0, 0.15))
run('behind the robot (x=-0.30)', (-0.30, 0.0, 0.30))

ps = PlanningScene(); ps.is_diff = True
for oid in ('table_slab', 'rear_safety_wall'):
    co = CollisionObject(); co.id = oid; co.header.frame_id = BASE_LINK; co.operation = CollisionObject.REMOVE; ps.world.collision_objects.append(co)
f = h._apply_scene_client.call_async(ApplyPlanningScene.Request(scene=ps)); t0 = time.time()
while not f.done() and time.time() - t0 < 5: time.sleep(0.05)
print('test objects removed:', f.result().success if f.done() else 'TIMEOUT', flush=True)
