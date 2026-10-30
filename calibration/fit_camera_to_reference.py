#!/usr/bin/env python3
"""
fit_camera_to_reference.py -- correct a static camera's x/y position and rotation about the vertical so that the objects it sees land where a REFERENCE camera puts them. READ-ONLY; writes a candidate.

    python3 calibration/fit_camera_to_reference.py [--pairs FILE] [--camera realsense] [--reference realsense2] [--file ~/.ros/realsense_calibration.yaml] [--out CANDIDATE.yaml]

Input: ~/.ros/calibration_candidates/shared_object_pairs.jsonl from record_shared_objects.py (the same objects as seen by both cameras, in base_link).
The camera's base_link points p are corrected by p' = Rz(theta) p + (tx, ty, 0), the best 2D rigid fit of its points onto the reference's (least squares after trimming pairs that do not belong). Heights are left alone (the
table-levelling fixes them). The camera's pose is pre-multiplied by the same transform: T' = [Rz t] T, so every point it sees moves exactly like that. Needs >= 3 pairs that are NOT in a line; the more places the
objects have stood, the better. Check the result with record_shared_objects.py afterwards and with the tape.
"""
import argparse
import json
import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

PAIRS = os.path.expanduser('~/.ros/calibration_candidates/shared_object_pairs.jsonl')


def load_pairs(path, cam, ref, sets=None):
    out = []
    for line in open(path):
        rec = json.loads(line)
        if rec['a'] != cam or rec['b'] != ref or (sets and rec['set'] not in sets):
            continue
        for p in rec['pairs']:
            out.append((rec['set'], p['label'], np.array(p[cam][:2]), np.array(p[ref][:2])))
    return out


def fit_rigid_2d(A, B):
    """theta, t minimising |Rz(theta) A + t - B|. A, B: (n, 2)."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A - ca).T @ (B - cb)
    theta = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s = np.cos(theta), np.sin(theta)
    Rm = np.array([[c, -s], [s, c]])
    return float(theta), cb - Rm @ ca


def apply(theta, t, A):
    c, s = np.cos(theta), np.sin(theta)
    return A @ np.array([[c, -s], [s, c]]).T + t


def robust_fit(A, B, min_keep=3, max_resid_m=0.04):
    """Fit, drop the worst pair while its residual is > max_resid_m (and > 2.5x the median), refit. Returns theta, t, kept mask."""
    keep = np.ones(len(A), bool)
    while True:
        theta, t = fit_rigid_2d(A[keep], B[keep])
        res = np.linalg.norm(apply(theta, t, A) - B, axis=1)
        worst = int(np.argmax(np.where(keep, res, -1)))
        if keep.sum() > min_keep and res[worst] > max_resid_m and res[worst] > 2.5 * np.median(res[keep]):
            keep[worst] = False
            continue
        return theta, t, keep, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default=PAIRS)
    ap.add_argument('--camera', default='realsense')
    ap.add_argument('--reference', default='realsense2')
    ap.add_argument('--file', default=os.path.expanduser('~/.ros/realsense_calibration.yaml'))
    ap.add_argument('--out', default='')
    ap.add_argument('--sets', nargs='*')
    args = ap.parse_args()
    pairs = load_pairs(args.pairs, args.camera, args.reference, args.sets)
    if len(pairs) < 3:
        print(f'only {len(pairs)} pair(s): need at least 3 (objects at 3+ different places)')
        return
    A, B = np.array([p[2] for p in pairs]), np.array([p[3] for p in pairs])
    spread = np.linalg.svd(A - A.mean(axis=0), compute_uv=False)
    print(f'{len(pairs)} pairs; spread of the positions along their two main directions: {spread[0] ** 0.5 * 100:.0f} / {spread[1] ** 0.5 * 100:.0f} cm' + ('   <-- nearly a line: add objects in other places' if spread[1] < 0.02 else ''))
    theta, t, keep, res = robust_fit(A, B)
    before = np.linalg.norm(A - B, axis=1)
    print(f'{"set":8s} {"label":8s} {"before cm":>10s} {"after cm":>9s}')
    for i, p in enumerate(pairs):
        print(f'{p[0]:8s} {p[1]:8s} {100 * before[i]:10.1f} {100 * res[i]:9.1f}   {"" if keep[i] else "(dropped: does not fit)"}')
    print(f'fit on {int(keep.sum())}/{len(pairs)}: rotate {np.degrees(theta):+.2f} deg about the vertical, shift ({100 * t[0]:+.1f}, {100 * t[1]:+.1f}) cm;  median residual {100 * np.median(res[keep]):.1f} cm (was {100 * np.median(before[keep]):.1f})')
    if args.out:
        d = yaml.safe_load(open(args.file))
        q, tr = d['rotation_quat'], d['translation']
        T_R = Rotation.from_quat([q['x'], q['y'], q['z'], q['w']])
        T_t = np.array([tr['x'], tr['y'], tr['z']])
        Rz = Rotation.from_euler('z', theta)
        new_R = Rz * T_R
        new_t = Rz.apply(T_t) + np.array([t[0], t[1], 0.0])
        d['rotation_quat'] = dict(zip('xyzw', [float(v) for v in new_R.as_quat()]))
        d['translation'] = dict(zip('xyz', [float(v) for v in new_t]))
        d['note'] = (d.get('note', '') + f' Then aligned to {args.reference} with shared objects ({int(keep.sum())} pairs): rotated {np.degrees(theta):+.2f} deg about the vertical, shifted ({100 * t[0]:+.1f}, {100 * t[1]:+.1f}) cm.').strip()
        yaml.safe_dump(d, open(args.out, 'w'))
        print('wrote', args.out)


if __name__ == '__main__':
    main()
