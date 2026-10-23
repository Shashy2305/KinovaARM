"""calibration/check_wrist_board.py: with a board that does not move, the board's point in base_link must come out the same from every wrist pose, and the fitted square size must
recover the true one even when the code assumes the wrong one (the 35 mm assumed vs 39 mm printed mistake of 2026-10-22)."""
import os
import sys

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip('rclpy')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'calibration'))
import check_wrist_board as cw  # noqa: E402

TRUE_SQUARE, ASSUMED_SQUARE = 0.039, 0.035
BOARD = np.array([0.45, 0.20, -0.01])                 # board origin in base_link, flat on the table
R_BOARD = Rotation.from_euler('z', 20, degrees=True).as_matrix()


def make_snaps(n, rng, mount_error_m=0.0):
    snaps = []
    for _ in range(n):
        p = np.array([0.45, 0.20, 0.0]) + np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), rng.uniform(0.35, 0.55)])
        look = Rotation.from_euler('xyz', [180 + rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(-30, 30)], degrees=True).as_matrix()   # camera looking down
        R = look
        # board origin in the camera, in TRUE metres; the code only knows the board in units of ASSUMED squares, so what PnP returns is scaled by assumed / true
        t_true = R.T @ (BOARD - p)
        R_cb = R.T @ R_BOARD
        t_pnp = t_true * (ASSUMED_SQUARE / TRUE_SQUARE)
        if mount_error_m:
            err = np.array([mount_error_m, 0, 0])
            p = p + R @ err                              # the camera is not where the TF says it is
        snaps.append((p, R, cv2.Rodrigues(R_cb)[0], t_pnp.reshape(3, 1)))
    return snaps


def test_the_fit_recovers_the_true_square_size_and_a_tiny_scatter():
    r = cw.analyse(make_snaps(8, np.random.default_rng(1)), assumed_square_m=ASSUMED_SQUARE)
    assert r['true_square_mm'] == pytest.approx(39.0, abs=0.05)
    assert r['scatter_mm_fitted'] < 0.5
    assert r['scatter_mm_assumed'] > 8                        # with the wrong size the same point wanders by centimetres
    assert r['orientation_spread_deg'] < 0.1
    assert r['board_origin_base'] == pytest.approx(BOARD.tolist(), abs=1e-3)


def test_a_wrong_camera_mount_shows_up_as_scatter_not_as_a_wrong_size():
    r = cw.analyse(make_snaps(8, np.random.default_rng(2), mount_error_m=0.02), assumed_square_m=ASSUMED_SQUARE)
    assert r['scatter_mm_fitted'] > 5                          # a 2 cm mount error cannot be absorbed by a single scale factor
    clean = cw.analyse(make_snaps(8, np.random.default_rng(2)), assumed_square_m=ASSUMED_SQUARE)
    assert r['scatter_mm_fitted'] > 10 * clean['scatter_mm_fitted']
