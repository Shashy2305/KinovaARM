#!/usr/bin/env python3
"""
A dimensioned plan for placing the coffee machine with a tape measure. Drawn in the SAME orientation as the RS1 camera photo (robot base at the top, the table extending toward you, +y to the RIGHT of the
picture). Distances are from the centre of the robot's base ring. No data needed: pure geometry from the chosen cup spot.

    python3 scripts/coffee_placement_plan.py --out docs/coffee/coffee_placement_plan.png
"""
import argparse
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

MACH_W, MACH_D, CUP_FROM_FRONT = 0.208, 0.335, 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cup', default='0.40,0.20')
    ap.add_argument('--serve', default='0.50,-0.10')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cx, cy = [float(v) for v in a.cup.split(',')]
    sx, sy = [float(v) for v in a.serve.split(',')]
    # the machine front faces -y (toward the centre line): front plane at y = cy - CUP_FROM_FRONT, body toward +y
    y0, y1 = cy - CUP_FROM_FRONT, cy - CUP_FROM_FRONT + MACH_D
    x0, x1 = cx - MACH_W / 2, cx + MACH_W / 2
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.add_patch(mp.Rectangle((-0.62, -0.18), 1.24, 0.98, fc='#e9d8c4', ec='#8b6b4a', lw=2))
    ax.add_patch(mp.Circle((0, 0), 0.075, fc='#2d3748', ec='k', zorder=5))
    ax.text(0, 0, 'BASE\ncentre', color='white', ha='center', va='center', fontsize=7, zorder=6)
    ax.annotate('', xy=(0, 0.66), xytext=(0, 0.075), arrowprops=dict(arrowstyle='-', color='#2b6cb0', lw=1.6, ls='--'))
    ax.text(0.01, 0.62, 'centre line\n(straight out from the base,\nperpendicular to the table edge)', color='#2b6cb0', fontsize=8, va='top')
    ax.add_patch(mp.Rectangle((y0, x0), y1 - y0, x1 - x0, fc='#fc8181', ec='#9b2c2c', lw=2.5, alpha=0.85, zorder=4))
    ax.plot([y0, y0], [x0, x1], color='#ecc94b', lw=7, zorder=6, solid_capstyle='butt')
    ax.text(y0 - 0.012, cx, 'FRONT\n(open side,\ncup goes here)', color='#744210', ha='right', va='center', fontsize=9, weight='bold', zorder=7)
    ax.add_patch(mp.Circle((cy, cx), 0.045, fc='white', ec='k', lw=1.5, zorder=8))
    ax.plot([cy], [cx], marker='+', color='k', ms=12, zorder=9)
    ax.text(cy + 0.06, cx - 0.03, 'cup centre', fontsize=8, zorder=9)
    ax.add_patch(mp.Circle((sy, sx), 0.06, fc='none', ec='#2f855a', lw=3, ls='--', zorder=5))
    ax.plot([sy], [sx], marker='*', ms=20, color='#d69e2e', mec='k', zorder=6)
    ax.text(sy, sx + 0.075, 'DROP-OFF\n(serving)', ha='center', va='top', fontsize=9, weight='bold', zorder=7)

    def dim(p, q, text, off=(0, 0), colour='#1a365d'):
        ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle='<->', color=colour, lw=1.4), zorder=10)
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, color=colour, fontsize=9, weight='bold', ha='center', va='center', zorder=11,
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
    dim((0.0, x0), (y0, x0), f'{(y0) * 100:.0f} cm', (0, -0.025))                       # centre line -> front face, measured along the table edge direction
    dim((y0, x0 - 0.02), (y0 + MACH_D, x0 - 0.02), f'{MACH_D * 100:.1f} cm deep', (0, -0.03))
    dim((y1 + 0.02, x0), (y1 + 0.02, x1), f'{MACH_W * 100:.1f} cm wide', (0.045, 0))
    dim((-0.47, 0.0), (-0.47, x0), f'{x0 * 100:.1f} cm', (0.0, 0.0))                   # base centre -> near side of the machine, measured straight out
    dim((-0.47, x0), (-0.47, x1), f'{(x1 - x0) * 100:.1f}', (0.0, 0.0))
    ax.text(-0.47, x1 + 0.03, f'machine spans\n{x0 * 100:.1f} to {x1 * 100:.1f} cm\nstraight out', ha='center', va='top', fontsize=8, color='#1a365d')
    steps = ('HOW TO MARK IT (tape measure + 4 pieces of tape)\n'
             f'1. From the centre of the base ring, measure straight out {x0 * 100:.1f} cm and {x1 * 100:.1f} cm along the centre line: mark both points.\n'
             f'2. From each point measure {y0 * 100:.0f} cm sideways toward the machine side (right of the picture); these are the two FRONT corners.\n'
             f'3. From each front corner measure {MACH_D * 100:.1f} cm further sideways: the two BACK corners. Tape the 4 corners.\n'
             f'4. Stand the machine inside the corners with its open FRONT facing the centre line. The cup sits {cx * 100:.0f} cm out, {cy * 100:.0f} cm sideways.\n'
             f'5. Drop-off circle centre: {sx * 100:.0f} cm out, {abs(sy) * 100:.0f} cm sideways the OTHER way (left of the picture).')
    fig.text(0.02, 0.01, steps, fontsize=8.5, va='bottom', ha='left', family='monospace')
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(0.80, -0.18)                                              # x increases DOWN the picture (toward the camera), like the photo
    ax.set_aspect('equal')
    ax.set_xlabel('y (m): sideways from the centre line')
    ax.set_ylabel('x (m): straight out from the base')
    ax.grid(True, ls=':', alpha=0.4)
    ax.set_title('Where to put the Keurig K-Compact  (measure from the centre of the robot base)\nSame view as the camera photo: base at the top, you look from the far end of the table; the machine side is on the RIGHT', fontsize=10, weight='bold')
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(a.out, dpi=130)
    print('wrote', a.out, f'corners x {x0:.3f}..{x1:.3f}, y {y0:.3f}..{y1:.3f}')


if __name__ == '__main__':
    main()
