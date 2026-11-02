#!/usr/bin/env python3
"""
Where can a cup standing on a coffee machine's tray be taken from the side? READ-ONLY (MoveIt /compute_ik, /compute_fk, /compute_cartesian_path): the arm never moves.

Same test as scripts/side_grasp_study.py, but on a finer grid and with the cup raised by --tray-z (the height of the surface the cup stands on above the table). For every cup position and each of
the 8 approach azimuths it records whether the arm can swing to the high pre-pose, go down, go in and lift, with both pads >= 5 cm over the table.

    source scripts/ros_env.sh; ROS_DOMAIN_ID=42 python3 scripts/coffee_site_study.py --tray-z 0.03 --out docs/data/coffee_site_study_tray3cm.json
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
    ap.add_argument('--tray-z', type=float, default=0.03, help='height of the surface the cup stands on, above the table (m)')
    ap.add_argument('--hover-z', type=float, default=0.38)
    ap.add_argument('--xs', default='0.30,0.35,0.40,0.45,0.50,0.55')
    ap.add_argument('--ys', default='-0.40,-0.30,-0.20,-0.10,0.0,0.10,0.20,0.30,0.40')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rclpy.init()
    n = Study()
    n.spin_until(lambda: n.js is not None and n.ik.service_is_ready() and n.fk.service_is_ready(), 10)
    if n.js is None or not n.ik.service_is_ready():
        print('need /joint_states and move_group (/compute_ik, /compute_fk)')
        sys.exit(2)
    geom, _ = sg.load_geometry()
    table = geom['table_top_z']
    margin = table + sg.LINK_TABLE_CLEARANCE_M
    pad_z = go.side_pad_z(a.label, table) + a.tray_z
    here = [n.js[j] for j in JOINTS]

    def ik(fl, quat, seed, tries=4):
        """The IK solver restarts from random seeds and misses good solutions now and then: ask a few times before saying 'no solution'."""
        for _ in range(tries):
            sol = n.solve_ik(fl, quat, seed)
            if sol is not None:
                return sol
        return None

    xs, ys = [float(v) for v in a.xs.split(',')], [float(v) for v in a.ys.split(',')]
    cells = {}
    for x in xs:
        for y in ys:
            hover = ik([x, y, a.hover_z], GRASP_QUAT_XYZW, here)
            row = {}
            for az in go.AZIMUTHS:
                if hover is None:
                    row[str(int(az))] = 'no_hover'
                    continue
                high, pre, grasp, _ax = go.side_poses((x, y), pad_z, az)
                if not all(go.flange_inside(p) for p in (high, pre, grasp)):
                    row[str(int(az))] = 'outside'
                    continue
                verdict = 'no_ik'
                for flip in (False, True):
                    quat = go.side_quat(az, flip)
                    sols, why = {}, None
                    for name, fl in (('high', high), ('pre', pre), ('grasp', grasp), ('lift', grasp + np.array([0.0, 0.0, 0.10]))):
                        sol = ik(fl, quat, hover if name == 'high' else sols.get('high', hover) if name == 'pre' else sols[prev])
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
                        prev = name
                    # (no straight-line check: the live pick moves down and in with short joint-space steps, because the straight-line planner flips the wrist for a horizontal tool)
                    if why is None:
                        verdict = 'ok'
                        break
                    verdict = why
                row[str(int(az))] = verdict
            cells[f'{x:.2f}/{y:.2f}'] = row
            print(f'x={x:.2f} y={y:+.2f}: ' + ' '.join(f"{az}:{'+' if v == 'ok' else '-'}" for az, v in row.items()), flush=True)
    json.dump({'label': a.label, 'tray_z': a.tray_z, 'pad_z': pad_z, 'xs': xs, 'ys': ys, 'cells': cells}, open(a.out, 'w'), indent=1)
    print('saved', a.out)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
