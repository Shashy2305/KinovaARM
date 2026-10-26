#!/usr/bin/env python3
"""
wizard_consensus.py -- combine SEVERAL wrist-anchored board captures of one static camera into one calibration, with outlier rejection. READ-ONLY (writes a candidate file, installs nothing).

    python3 calibration/wizard_consensus.py --camera realsense2 [--since "2026-10-26 11:00"] [--square-mm 38.15] [--out FILE]

The dashboard Calibration tab (multi_camera_calibrate.py) appends the raw pieces of every capture to ~/.ros/calibration_candidates/wizard_captures.jsonl:
the wrist camera's pose in base_link, the board in the wrist camera, the board in the static camera, and the square size it assumed. From them each capture gives  T_base_cam = T_base_wrist
* T_wrist_board * inverse(T_cam_board).  A single capture of a far camera
looking at a small board is noisy (cm), and the 2026-10-22 second run disagreed with the first by up to 1.4 m with no way to tell which was wrong. Here every capture is recomputed with the
REAL square size
(the board translations scale with it, rotations do not), captures far from the others are dropped (> 8 cm from the median position or > 4 deg from the mean orientation, and more than 3x
the typical deviation), and the rest are averaged.
Move the board and the arm between captures: a different board place and a different wrist pose each time, so the errors are independent.
"""
import argparse
import json
import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

LOG = os.path.expanduser('~/.ros/calibration_candidates/wizard_captures.jsonl')
MAX_TRANSLATION_DEV_M = 0.08           # a capture of a far camera is good to a few cm: only drop what is clearly outside that (and outside 3x the typical deviation)
MAX_ROTATION_DEV_DEG = 4.0


def scaled(T, k):
    T = np.array(T, float)
    T[:3, 3] *= k
    return T


def camera_pose(rec, square_mm=None):
    """4x4 base_link <- camera of one logged capture, recomputed with the real square size (default: what the capture assumed)."""
    k = 1.0 if square_mm is None else (square_mm / 1000.0) / rec['square_m']
    T_bw = np.array(rec['T_base_wrist'], float)
    return T_bw @ scaled(rec['T_wrist_board'], k) @ np.linalg.inv(scaled(rec['T_cam_board'], k))


def combine(poses):
    """poses: list of 4x4. Returns {'T': mean pose, 'kept': indices, 'dropped': [(index, translation dev cm, rotation dev deg)],
    'spread_cm': rms translation spread of the kept, 'spread_deg': rms rotation spread}."""
    t = np.array([p[:3, 3] for p in poses])
    R = Rotation.from_matrix([p[:3, :3] for p in poses])
    kept = list(range(len(poses)))
    for _ in range(4):                         # a few rounds: dropping the worst changes the median and the mean
        med_t = np.median(t[kept], axis=0)
        mean_R = R[kept].mean()
        dev_t = np.linalg.norm(t - med_t, axis=1)
        dev_r = np.array([np.degrees((mean_R.inv() * r).magnitude()) for r in R])
        lim_t = max(MAX_TRANSLATION_DEV_M, 3 * float(np.median(dev_t[kept])))
        lim_r = max(MAX_ROTATION_DEV_DEG, 3 * float(np.median(dev_r[kept])))
        new = [i for i in range(len(poses)) if dev_t[i] <= lim_t and dev_r[i] <= lim_r]
        if len(new) < 2 or new == kept:
            kept = new if len(new) >= 2 else kept
            break
        kept = new
    med_t = np.median(t[kept], axis=0)
    mean_R = R[kept].mean()
    T = np.eye(4)
    T[:3, :3] = mean_R.as_matrix()
    T[:3, 3] = np.mean(t[kept], axis=0)
    dropped = [(i, float(100 * np.linalg.norm(t[i] - med_t)), float(np.degrees((mean_R.inv() * R[i]).magnitude()))) for i in range(len(poses)) if i not in kept]
    spread_cm = float(100 * np.sqrt(np.mean(np.sum((t[kept] - T[:3, 3]) ** 2, axis=1))))
    spread_deg = float(np.sqrt(np.mean([np.degrees((mean_R.inv() * R[i]).magnitude()) ** 2 for i in kept])))
    return {'T': T, 'kept': kept, 'dropped': dropped, 'spread_cm': spread_cm, 'spread_deg': spread_deg}


def load(camera, since=None):
    out = []
    try:
        for ln in open(LOG):
            r = json.loads(ln)
            if r.get('camera') == camera and (since is None or r['time'] >= since):
                out.append(r)
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', required=True)
    ap.add_argument('--since', default=None, help='only captures logged at or after this time, "YYYY-MM-DD HH:MM[:SS]"')
    ap.add_argument('--square-mm', type=float, default=None, help='the real board square size (default: what each capture assumed)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    recs = load(a.camera, a.since)
    print(f'{len(recs)} captures of {a.camera}' + (f' since {a.since}' if a.since else ''))
    if len(recs) < 2:
        raise SystemExit('need at least 2 captures (3-5 recommended, with the board and the arm moved between them)')
    poses = [camera_pose(r, a.square_mm) for r in recs]
    for r, p in zip(recs, poses):
        print(f'  {r["time"]}  square {r["square_m"] * 1000:.2f} mm  t = {np.round(p[:3, 3], 3).tolist()}')
    c = combine(poses)
    T = c['T']
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    print(f'\nkept {len(c["kept"])}/{len(recs)}; spread {c["spread_cm"]:.1f} cm, {c["spread_deg"]:.2f} deg')
    for i, dt, dr in c['dropped']:
        print(f'  dropped capture {i + 1} ({recs[i]["time"]}): {dt:.1f} cm / {dr:.1f} deg from the others')
    print(f'combined t = {np.round(T[:3, 3], 3).tolist()}')
    trust = 'GOOD (captures agree)' if len(c['kept']) >= 3 and c['spread_cm'] < 2.0 and c['spread_deg'] < 1.0 else 'WEAK: take more captures from different board places / wrist poses'
    print(f'consistency: {trust}')
    if a.out:
        out = {'parent_frame': 'base_link', 'child_frame': None, 'translation': {k: float(v) for k, v in zip('xyz', T[:3, 3])},
               'rotation_quat': {k: float(v) for k, v in zip('xyzw', q)},
               'note': (f'Combined from {len(c["kept"])} of {len(recs)} wizard captures (calibration/wizard_consensus.py), spread {c["spread_cm"]:.1f} cm / {c["spread_deg"]:.2f} deg. '
                        'NOT verified until it passes check_arm_depth.py.')}
        with open(os.path.expanduser(a.out), 'w') as f:
            yaml.safe_dump(out, f, default_flow_style=False)
        print(f'wrote {a.out} (child_frame left empty: copy it from the camera\'s existing calibration file)')


if __name__ == '__main__':
    main()
