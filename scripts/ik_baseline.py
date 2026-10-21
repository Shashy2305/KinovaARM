#!/usr/bin/env python3
"""
ik_baseline.py -- what does MoveIt's IK cost today? READ-ONLY (asks /compute_ik; the arm never moves). The baseline for any faster solver (cuRobo).

    source /mnt/ros_workspace/Shashproject/scripts/ros_env.sh
    python3 scripts/ik_baseline.py [--targets 60] [--out docs/data/ik_baseline_moveit.json]

For ~N fingertip targets on the table (x 0.25-0.55, y -0.30..0.30, vertical tool, 4 wrist yaws) it asks the same service with the same timeouts the arm controller uses
(0.15 s, then 0.4 s) from the current joints as the seed, and reports success rate, latency percentiles and the largest single-joint change of each solution.
It also reports what ONE controller move costs: the controller's _solve_goal_joints tries up to 13 wrist yaws x 2 solves.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import rclpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'thesis_robot'))
import angled_approach_study as st  # noqa: E402
from thesis_robot import motion_utils as mu  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', type=int, default=60)
    ap.add_argument('--flange-z', type=float, default=0.25, help='flange height above base_link z=0 for the targets')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    rclpy.init()
    n = st.Study()
    n.spin_until(lambda: n.js is not None and n.ik.service_is_ready(), 10)
    if n.js is None or not n.ik.service_is_ready():
        raise SystemExit('need /joint_states and move_group (/compute_ik)')
    seed = [n.js[j] for j in st.JOINTS]
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(0.25, 0.55, args.targets), rng.uniform(-0.30, 0.30, args.targets)])
    rows = []
    for (x, y) in pts:
        for yaw in (0, 45, 90, 135):
            q = mu.rotate_about_tool_z(st.GRASP_QUAT_XYZW, yaw)
            best = None
            for timeout in (0.15, 0.4):
                t0 = time.time()
                sol = n.solve_ik([x, y, args.flange_z], q, seed, timeout)
                ms = 1000 * (time.time() - t0)
                rows.append({'x': float(x), 'y': float(y), 'yaw': yaw, 'timeout_s': timeout, 'ok': sol is not None, 'ms': ms,
                             'max_joint_change': float(np.max(np.abs(np.array(sol) - np.array(seed)))) if sol else None})
                if sol is not None:
                    break
    for timeout in (0.15, 0.4):
        sel = [r for r in rows if r['timeout_s'] == timeout]
        if not sel:
            continue
        ms = np.array([r['ms'] for r in sel])
        ok = np.mean([r['ok'] for r in sel])
        print(f'timeout {timeout}: {len(sel)} calls, solved {100 * ok:.0f}%, latency median {np.median(ms):.0f} ms, p90 {np.percentile(ms, 90):.0f} ms, p99 {np.percentile(ms, 99):.0f} ms, max {ms.max():.0f} ms')
    cases = {}
    for r in rows:
        cases.setdefault((r['x'], r['y'], r['yaw']), []).append(r)
    solved = sum(any(r['ok'] for r in v) for v in cases.values())
    totals = [sum(r['ms'] for r in v) for v in cases.values()]
    print(f'targets x yaw: {len(cases)}; solved within the controller\'s two tries: {100 * solved / len(cases):.0f}%; time per target (both tries when needed): median {np.median(totals):.0f} ms, p90 {np.percentile(totals, 90):.0f} ms')
    # one controller move with wrist-yaw flexibility tries up to 13 yaw candidates x 2 solves, stopping early only for a good enough solution
    print(f'one controller move with yaw flexibility (13 yaws x up to 2 solves): about {13 * np.median(totals) / 1000:.1f} s of IK at the median')
    jc = [r['max_joint_change'] for r in rows if r['ok']]
    print(f'largest single-joint change of the solutions from the current pose: median {np.median(jc):.2f} rad, p90 {np.percentile(jc, 90):.2f} rad, max {np.max(jc):.2f} rad')
    if args.out:
        json.dump({'date': time.strftime('%Y-%m-%d %H:%M'), 'seed': seed, 'rows': rows}, open(args.out, 'w'))
        print('wrote', args.out)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
