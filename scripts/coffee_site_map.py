#!/usr/bin/env python3
"""
Draws the table from above with where a Keurig K-Compact can stand and where the cup is taken from it (data: scripts/coffee_site_study.py, repeated runs), plus the drop-off point.

    python3 scripts/coffee_site_map.py --runs '/tmp/coffee_runs/run*_tray3cm.json' --out docs/coffee/coffee_site_map.png

View: from ABOVE with the robot at the bottom, x (forward, into the table) UP, y to the LEFT (the robot's left is +y): the way you see it standing behind the robot.
"""
import argparse
import glob
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AZ = [0, 45, 90, 135, 180, 225, 270, 315]
TABLE_X, TABLE_Y = (-0.15, 0.784), (-0.899, 0.85)
PLAN_X, PLAN_Y = (0.07, 0.65), (-0.50, 0.50)
SIDE_ZONE_X, SIDE_ZONE_Y = (0.30, 0.52), (-0.22, 0.22)
# Keurig K-Compact (K35): 8.2 in wide, 13.2 in deep. The cup stands on the tray under the spout: CUP_FROM_FRONT_M is an ASSUMPTION (centre of the cup this far behind the front edge): measure it.
MACH_W, MACH_D = 0.208, 0.335
CUP_FROM_FRONT_M = 0.10


def load(pattern):
    runs = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
    if not runs:
        raise SystemExit(f'no files match {pattern}')
    xs, ys = runs[0]['xs'], runs[0]['ys']
    freq = {}
    for x in xs:
        for y in ys:
            key = f'{x:.2f}/{y:.2f}'
            for az in AZ:
                freq[(x, y, az)] = float(np.mean([1.0 if r['cells'][key][str(az)] == 'ok' else 0.0 for r in runs]))
    return xs, ys, freq, len(runs)


def machine_polygon(cup, front_dir_deg):
    """Corners of the machine footprint when the cup is at `cup` and the machine's FRONT faces front_dir_deg (an outward direction in the x-y plane)."""
    f = np.array([math.cos(math.radians(front_dir_deg)), math.sin(math.radians(front_dir_deg))])    # outward normal of the front face
    s = np.array([-f[1], f[0]])                                                                       # sideways
    front_c = np.array(cup) + f * CUP_FROM_FRONT_M                                                    # centre of the front edge
    back_c = front_c - f * MACH_D
    return [front_c + s * MACH_W / 2, front_c - s * MACH_W / 2, back_c - s * MACH_W / 2, back_c + s * MACH_W / 2]


def draw_table(ax):
    ax.add_patch(mp.Rectangle((TABLE_Y[0], TABLE_X[0]), TABLE_Y[1] - TABLE_Y[0], TABLE_X[1] - TABLE_X[0], fc='#e9d8c4', ec='#8b6b4a', lw=2, zorder=0))
    ax.add_patch(mp.Rectangle((-0.50, PLAN_X[0]), 1.0, PLAN_X[1] - PLAN_X[0], fc='none', ec='#2b6cb0', lw=1.4, ls='--', zorder=3))
    ax.text(0.495, 0.655, 'planner workspace', color='#2b6cb0', fontsize=8, ha='right')
    ax.add_patch(mp.Rectangle((-SIDE_ZONE_Y[1], SIDE_ZONE_X[0]), SIDE_ZONE_Y[1] - SIDE_ZONE_Y[0], SIDE_ZONE_X[1] - SIDE_ZONE_X[0], fc='none', ec='#2f855a', lw=1.4, ls=':', zorder=3))
    ax.text(-0.215, 0.525, 'side-grasp zone', color='#2f855a', fontsize=8, ha='left')
    for r in (0.40, 0.55, 0.66):
        ax.add_patch(mp.Arc((0, 0), 2 * r, 2 * r, theta1=0, theta2=180, ec='#999999', lw=0.8, ls=':', zorder=2))
        ax.text(0.0, r + 0.005, f'{r:.2f} m', fontsize=7, color='#777777', ha='center')
    ax.add_patch(mp.Circle((0, 0), 0.075, fc='#2d3748', ec='k', zorder=5))
    ax.text(0, -0.035, 'robot base', ha='center', va='top', fontsize=8, color='#2d3748')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs3', default='/tmp/coffee_runs/run*_tray3cm.json')
    ap.add_argument('--runs0', default='/tmp/coffee_runs/run*_tray0cm.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cup', default='', help='x,y of the recommended cup spot (default: the best cell)')
    ap.add_argument('--serve', default='', help='x,y of the drop-off point')
    a = ap.parse_args()
    xs, ys, f3, n3 = load(a.runs3)
    _, _, f0, n0 = load(a.runs0)
    # reliability of a spot = the best direction's success rate; both tray heights must work (the tray can be taken out)
    score = {(x, y): max(min(f3[(x, y, az)], f0[(x, y, az)]) for az in AZ) for x in xs for y in ys}
    best_az = {(x, y): max(AZ, key=lambda az: (min(f3[(x, y, az)], f0[(x, y, az)]), f3[(x, y, az)])) for x in xs for y in ys}

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5))
    # --- left: reliability map ---------------------------------------------------------------------------------------------------------------------
    ax = axes[0]
    draw_table(ax)
    for (x, y), v in score.items():
        ax.add_patch(mp.Rectangle((y - 0.05, x - 0.025), 0.10, 0.05, fc=plt.cm.RdYlGn(v), ec='white', lw=0.5, alpha=0.85, zorder=1))
        ax.text(y, x, f'{int(round(100 * v))}', ha='center', va='center', fontsize=7, zorder=4)
    ax.set_title(f'Where a cup can be taken from the SIDE (reliability % of {n3 + n0} IK runs, best of 8 directions)', fontsize=10)

    # --- right: recommendation ---------------------------------------------------------------------------------------------------------------------
    ax = axes[1]
    draw_table(ax)
    cands = sorted(score.items(), key=lambda kv: (-kv[1], abs(kv[0][1]), kv[0][0]))
    if a.cup:
        cx, cy = [float(v) for v in a.cup.split(',')]
        cup = (cx, cy)
        az = best_az.get((min(xs, key=lambda v: abs(v - cx)), min(ys, key=lambda v: abs(v - cy))), 45)
    else:
        (cx, cy), _v = next(((k, v) for k, v in cands if 0.35 <= k[0] <= 0.50 and abs(k[1]) <= 0.30), cands[0])
        cup, az = (cx, cy), best_az[(cx, cy)]
    d = np.array([math.cos(math.radians(az)), math.sin(math.radians(az))])                          # the tool points along d (into the machine)
    front_deg = math.degrees(math.atan2(-d[1], -d[0]))                                              # the machine's front faces the opposite way
    poly = machine_polygon(cup, front_deg)
    ax.add_patch(mp.Polygon([(p[1], p[0]) for p in poly], closed=True, fc='#c53030', ec='#742a2a', alpha=0.85, zorder=6))
    fc = np.mean(poly[:2], axis=0)
    perp = np.array([-math.sin(math.radians(front_deg)), math.cos(math.radians(front_deg))])      # along the front edge, away from the arrow
    lab = fc + perp * 0.12
    ax.annotate('FRONT\n(open side)', xy=(lab[1], lab[0]), ha='center', va='center', color='white', fontsize=8, weight='bold', zorder=8)
    ax.add_patch(mp.Circle((cup[1], cup[0]), 0.045, fc='white', ec='k', zorder=9))
    ax.text(cup[1], cup[0], 'cup', ha='center', va='center', fontsize=7, zorder=10)
    start = np.array(cup) - d * 0.26
    ax.annotate('', xy=(cup[1] - d[1] * 0.05, cup[0] - d[0] * 0.05), xytext=(start[1], start[0]), arrowprops=dict(arrowstyle='-|>', color='#2b6cb0', lw=2.5), zorder=11)
    ax.text(start[1], start[0] + 0.035, f'gripper comes in\n(azimuth {az}°)', color='#2b6cb0', fontsize=8, ha='center', va='bottom', zorder=11)
    if a.serve:
        sx, sy = [float(v) for v in a.serve.split(',')]
        ax.plot([sy], [sx], marker='*', ms=22, color='#d69e2e', mec='k', zorder=9)
        ax.text(sy, sx - 0.085, 'DROP-OFF / serving', ha='center', va='top', fontsize=9, weight='bold', zorder=10)
        ax.add_patch(mp.Circle((sy, sx), 0.06, fc='none', ec='#d69e2e', lw=2, ls='--', zorder=9))
    ax.set_title(f'Recommended layout: machine {MACH_W * 100:.0f} x {MACH_D * 100:.0f} cm, cup at ({cup[0]:.2f}, {cup[1]:+.2f}) m', fontsize=10)

    for ax in axes:
        ax.set_xlim(0.62, -0.62)                                         # +y (the robot's left) on the left
        ax.set_ylim(-0.12, 0.72)
        ax.set_aspect('equal')
        ax.set_xlabel('y (m), the robot\'s LEFT is on the left of the picture')
        ax.set_ylabel('x (m), forward (into the table)')
        ax.grid(True, ls=':', alpha=0.4)
    fig.suptitle('Coffee machine site: view from above, robot at the bottom', fontsize=13, weight='bold')
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print('cup', cup, 'azimuth', az, 'front faces', round(front_deg), 'deg; reliability', round(100 * score[(min(xs, key=lambda v: abs(v - cup[0])), min(ys, key=lambda v: abs(v - cup[1])))]), '%')
    print('top spots:', [(f'{x:.2f},{y:+.2f}', round(100 * v)) for (x, y), v in cands[:12]])


if __name__ == '__main__':
    main()
