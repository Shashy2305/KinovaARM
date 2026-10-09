#!/usr/bin/env python3
"""
Angled-approach feasibility study. READ-ONLY: asks MoveIt's /compute_ik and /compute_fk services; the arm never moves.

For a grid of fingertip targets on the table it asks: can the gripper reach the grasp pose with the tool tilted by
`tilt` degrees from vertical, leading toward azimuth `az` (the direction the fingertips point toward), and the pregrasp
pose `standoff` metres back along the tool axis? A pose counts as feasible when IK finds a solution, MoveIt can follow the straight line from the pregrasp to the grasp (>= 98%), the joints stay
inside the planner's limits, the swing from the seed pose is acceptable, and FK shows both finger pads at least 5 cm
above the table at the grasp AND the pregrasp.

    source scripts/ros_env.sh && python3 scripts/angled_approach_study.py [--tilts 0 15 30 45] [--out file.json]
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetCartesianPath, GetPositionFK, GetPositionIK
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState

sys.path.insert(0, 'src/thesis_robot')
from thesis_robot import angled_approach as aa  # noqa: E402
from thesis_robot import motion_utils as mu  # noqa: E402
from thesis_robot import safety_geometry as sg  # noqa: E402

JOINTS = [f'joint_{i}' for i in range(1, 8)]
GRASP_QUAT_XYZW = [0.773, 0.635, -0.015, 0.019]
PADS = ['left_inner_finger_pad', 'right_inner_finger_pad']


class Study(Node):
    def __init__(self):
        super().__init__('angled_approach_study')
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        self.fk = self.create_client(GetPositionFK, '/compute_fk')
        self.cart = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.js = None
        self.create_subscription(JointState, '/joint_states', lambda m: setattr(self, 'js', dict(zip(m.name, m.position))), 5)

    def spin_until(self, cond, timeout=5.0):
        t = time.time()
        while not cond() and time.time() - t < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)

    def call(self, client, req, timeout=3.0):
        fut = client.call_async(req)
        self.spin_until(fut.done, timeout)
        return fut.result() if fut.done() else None

    def seed_state(self, q=None):
        rs = RobotState()
        rs.joint_state.name = JOINTS
        rs.joint_state.position = list(q if q is not None else [self.js[j] for j in JOINTS])
        return rs

    def solve_ik(self, pos, quat, seed_q, timeout=0.3):
        req = GetPositionIK.Request()
        req.ik_request.group_name = 'manipulator'
        req.ik_request.ik_link_name = 'end_effector_link'
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout.sec, req.ik_request.timeout.nanosec = 0, int(timeout * 1e9)
        req.ik_request.robot_state = self.seed_state(seed_q)
        ps = PoseStamped()
        ps.header.frame_id = 'base_link'
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = [float(v) for v in pos]
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = [float(v) for v in quat]
        req.ik_request.pose_stamped = ps
        res = self.call(self.ik, req, 4.0)
        if res is None or res.error_code.val != 1:
            return None
        sol = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        return [sol[j] for j in JOINTS]

    def cartesian_fraction(self, start_q, pos, quat):
        """Fraction of the straight-line (Cartesian) path from the start joints to the pose that MoveIt can follow
        with the tool orientation held (1.0 = all of it)."""
        req = GetCartesianPath.Request()
        req.header.frame_id = 'base_link'
        req.group_name = 'manipulator'
        req.link_name = 'end_effector_link'
        req.start_state = self.seed_state(start_q)
        ps = PoseStamped()
        ps.header.frame_id = 'base_link'
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = [float(v) for v in pos]
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = [float(v) for v in quat]
        req.waypoints = [ps.pose]
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = False
        res = self.call(self.cart, req, 5.0)
        return 0.0 if res is None else float(res.fraction)

    def pad_heights(self, q):
        req = GetPositionFK.Request()
        req.header.frame_id = 'base_link'
        req.fk_link_names = PADS + ['end_effector_link']
        req.robot_state = self.seed_state(q)
        res = self.call(self.fk, req, 3.0)
        if res is None or res.error_code.val != 1:
            return None
        return [p.pose.position.z for p in res.pose_stamped]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tilts', type=float, nargs='+', default=[0, 15, 30, 45])
    ap.add_argument('--azimuths', type=float, nargs='+', default=[0, 90, 180, 270])
    ap.add_argument('--yaws', type=float, nargs='+', default=[0, 45, 90, 135])
    ap.add_argument('--tip-z', type=float, default=0.03, help='fingertip height above the table at the grasp (m)')
    ap.add_argument('--standoff', type=float, default=0.12)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    rclpy.init()
    n = Study()
    n.spin_until(lambda: n.js is not None and n.ik.service_is_ready() and n.fk.service_is_ready(), 10)
    if n.js is None or not n.ik.service_is_ready():
        print('need /joint_states and move_group (/compute_ik, /compute_fk)')
        sys.exit(2)
    geom, _ = sg.load_geometry()
    table = geom['table_top_z']
    margin = table + sg.LINK_TABLE_CLEARANCE_M
    seed = [n.js[j] for j in JOINTS]
    xs, ys = [0.25, 0.35, 0.45, 0.55], [-0.30, -0.15, 0.0, 0.15, 0.30]
    results = []
    for tilt in args.tilts:
        for az in (args.azimuths if tilt > 0 else [0.0]):
            ok = total = 0
            reasons = {'no_ik': 0, 'swing': 0, 'low_pad': 0, 'limit': 0, 'line': 0}
            for x in xs:
                for y in ys:
                    total += 1
                    good = False
                    for yaw in args.yaws:
                        a = aa.tool_axis(tilt, az)
                        tip = np.array([x, y, table + args.tip_z])
                        quat = aa.tool_quat(GRASP_QUAT_XYZW, tilt, az, yaw)
                        poses = {'grasp': aa.flange_for_tip(tip, a), 'pregrasp': aa.flange_for_tip(tip - args.standoff * a, a)}
                        solved = {}
                        why = None
                        for name, fl in poses.items():
                            sol = n.solve_ik(fl, quat, seed)
                            if sol is None:
                                why = 'no_ik'
                                break
                            sol = mu.unwrap_to_seed(sol, seed)
                            if sol is None:
                                why = 'limit'
                                break
                            if max(mu.joint_deltas(sol, seed)) > mu.MAX_JOINT_DELTA_RAD:
                                why = 'swing'
                                break
                            hs = n.pad_heights(sol)
                            if hs is None or min(hs[:2]) < margin - 0.002:
                                why = 'low_pad'
                                break
                            solved[name] = sol
                        if why is None:
                            frac = n.cartesian_fraction(solved['pregrasp'], poses['grasp'], quat)
                            if frac < 0.98:
                                why = 'line'
                        if why is None:
                            good = True
                            break
                        reasons[why] += 1
                    ok += good
            results.append({'tilt': tilt, 'azimuth': az, 'feasible': ok, 'total': total, 'fail_reasons': reasons})
            print(f'tilt {tilt:4.0f} deg  azimuth {az:4.0f} deg : {ok:2d}/{total} targets reachable   {reasons}', flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'tip_z': args.tip_z, 'standoff': args.standoff, 'results': results}, f, indent=1)
        print('saved', args.out)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
