#!/usr/bin/env python3
"""
rescale_wizard_result.py -- correct a wrist-anchored board calibration (multi_camera_calibrate.py / the dashboard Calibration tab) that was computed with the WRONG board square size.

    python3 calibration/rescale_wizard_result.py --file IN.yaml --assumed-mm 35 --true-mm 38.15 --wrist 0.40 0.05 0.50 --out OUT.yaml

Why it is exact up to a small term. The wizard computes  x_cam = p_w + R_w t_wb - R_c t_cb  (wrist camera position p_w; the board seen from the wrist camera t_wb and from the static camera t_cb, both taken
from the corner geometry with the assumed square size). A wrong square size scales EVERY board translation by k = assumed/true, rotations are untouched, so

    x_est = (1 - k) p_w + k x_true        =>        x_true = x_est + (1 - k)/k * (x_est - p_w)

p_w (where the wrist camera was when the board was captured) is not stored, but it only enters multiplied by (1-k)/k: for a 38.15 mm board captured with 35 mm, 9 %, so a 15 cm uncertainty
in p_w is 1.4 cm. Give the wrist position the board was captured from (~ above the board); the printout says what +-15 cm of it would change. The rotation is kept as it is.
"""
import argparse
import os

import numpy as np
import yaml


def rescale(t_est, p_w, assumed_m, true_m):
    k = assumed_m / true_m
    t_est, p_w = np.asarray(t_est, float), np.asarray(p_w, float)
    return t_est + (1.0 - k) / k * (t_est - p_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--assumed-mm', type=float, required=True)
    ap.add_argument('--true-mm', type=float, required=True)
    ap.add_argument('--wrist', type=float, nargs=3, required=True, help='x y z of the wrist camera in base_link when the board was captured')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    with open(os.path.expanduser(a.file)) as f:
        y = yaml.safe_load(f)
    t = [y['translation'][k] for k in 'xyz']
    t2 = rescale(t, a.wrist, a.assumed_mm / 1000, a.true_mm / 1000)
    k = a.assumed_mm / a.true_mm
    print(f'{os.path.basename(a.file)}: t {np.round(t, 3).tolist()} -> {np.round(t2, 3).tolist()}   (k = {k:.4f}, shift {np.linalg.norm(t2 - np.array(t)) * 100:.1f} cm)')
    print(f'  sensitivity: +-15 cm in the assumed wrist position changes the result by {abs(1 - k) / k * 0.15 * 100:.1f} cm')
    y['translation'] = {k_: float(v) for k_, v in zip('xyz', t2)}
    y['note'] = (f'{y.get("note", "")} Rescaled from a {a.assumed_mm:.1f} mm board assumption to {a.true_mm:.2f} mm (calibration/rescale_wizard_result.py), wrist position assumed '
                 f'{list(a.wrist)}. NOT verified until it passes calibration/check_arm_depth.py.').strip()
    with open(os.path.expanduser(a.out), 'w') as f:
        yaml.safe_dump(y, f, default_flow_style=False)


if __name__ == '__main__':
    main()
