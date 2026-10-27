#!/usr/bin/env python3
"""
Side-grasp reach study. READ-ONLY: asks MoveIt's /compute_ik, /compute_fk and /compute_cartesian_path; the arm never moves.

For a grid of cup positions and the 8 approach azimuths (grasp_options.AZIMUTHS) it asks, the way the real pick would: starting from the upright hover pose over the object, can the arm
(1) swing to the HIGH pre-pose (tool horizontal, 12 cm above and 12 cm back), (2) go straight down to the PRE pose, (3) go straight in to the GRASP pose, with both finger pads >= 5 cm over
the table at every pose and the joint swing inside the planner's limit? Both wrist orientations (the 180 deg flip) are tried.

    source scripts/ros_env.sh && python3 scripts/side_grasp_study.py [--label cup] [--out file.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'thesis_robot'))
from angled_approach_study import GRASP_QUAT_XYZW, JOINTS, Study  # noqa: E402
from thesis_robot import grasp_options as go  # noqa: E402
from thesis_robot import motion_utils as mu  # noqa: E402
from thesis_robot import safety_geometry as sg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='cup')
    ap.add_argument('--hover-z', type=float, default=0.38, help='flange height of the upright hover pose the swing starts from')
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
    pad_z = go.side_pad_z(args.label, table)
    here = [n.js[j] for j in JOINTS]
    xs, ys = [0.25, 0.35, 0.45, 0.55], [-0.30, -0.15, 0.0, 0.15, 0.30]
    results, table_out = [], {}
    for az in go.AZIMUTHS:
        ok = total = 0
        reasons = {'no_hover': 0, 'no_ik': 0, 'limit': 0, 'swing': 0, 'low_pad': 0, 'line': 0, 'outside': 0}
        for x in xs:
            for y in ys:
                total += 1
                hover = n.solve_ik([x, y, args.hover_z], GRASP_QUAT_XYZW, here)
                if hover is None:
                    reasons['no_hover'] += 1
                    continue
                high, pre, grasp, _a = go.side_poses((x, y), pad_z, az)
                if not all(go.flange_inside(p) for p in (high, pre, grasp)):
                    reasons['outside'] += 1
                    table_out[f'{az}/{x}/{y}'] = 'outside'
                    continue
                good, why = False, None
                for flip in (False, True):
                    quat = go.side_quat(az, flip)
                    sols, why = {}, None
                    for name, fl in (('high', high), ('pre', pre), ('grasp', grasp)):
                        sol = n.solve_ik(fl, quat, hover if name == 'high' else sols.get('high', hover))
                        if sol is None:
                            why = 'no_ik'
                            break
                        sol = mu.unwrap_to_seed(sol, hover)
                        if sol is None:
                            why = 'limit'
                            break
                        if name == 'high' and max(mu.joint_deltas(sol, hover)) > mu.MAX_JOINT_DELTA_RAD:
                            why = 'swing'
                            break
                        hs = n.pad_heights(sol)
                        if hs is None or min(hs[:2]) < margin - 0.002:
                            why = 'low_pad'
                            break
                        sols[name] = sol
                    if why is None:
                        if n.cartesian_fraction(sols['high'], pre, quat) < 0.98 or n.cartesian_fraction(sols['pre'], grasp, quat) < 0.98:
                            why = 'line'
                    if why is None:
                        good = True
                        break
                if good:
                    ok += 1
                    table_out[f'{az}/{x}/{y}'] = 'ok'
                else:
                    reasons[why] += 1
                    table_out[f'{az}/{x}/{y}'] = why
        results.append({'azimuth': az, 'feasible': ok, 'total': total, 'fail_reasons': reasons})
        print(f'azimuth {az:4.0f} deg : {ok:2d}/{total} cup positions reachable   {reasons}', flush=True)
    if args.out:
        json.dump({'label': args.label, 'pad_z': pad_z, 'hover_z': args.hover_z, 'results': results, 'cells': table_out}, open(args.out, 'w'), indent=1)
        print('saved', args.out)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
