"""calibration/wizard_consensus.py: several noisy captures combine into one good pose, a bad capture is dropped, and the real board size is applied exactly."""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'calibration'))
import wizard_consensus as wc  # noqa: E402

CAM_TRUE = np.eye(4)
CAM_TRUE[:3, :3] = Rotation.from_euler('xyz', [-120, 3, 160], degrees=True).as_matrix()
CAM_TRUE[:3, 3] = [0.38, -1.12, 0.15]


def capture(rng, assumed_m=0.035, true_m=0.03815, noise_m=0.0, noise_deg=0.0):
    """one wizard capture: random wrist pose, random board; the board translations are what the corner geometry gives with the ASSUMED square size"""
    T_bw = np.eye(4)
    T_bw[:3, :3] = Rotation.from_euler('xyz', [180 + rng.normal(0, 8), rng.normal(0, 8), rng.normal(0, 20)], degrees=True).as_matrix()
    T_bw[:3, 3] = [0.4 + rng.normal(0, 0.05), 0.05 + rng.normal(0, 0.08), 0.5 + rng.normal(0, 0.05)]
    T_bb = np.eye(4)
    T_bb[:3, :3] = Rotation.from_euler('z', rng.uniform(-30, 30), degrees=True).as_matrix()
    T_bb[:3, 3] = [0.45 + rng.normal(0, 0.05), 0.1 + rng.normal(0, 0.1), -0.01]
    T_wb_true = np.linalg.inv(T_bw) @ T_bb
    T_cb_true = np.linalg.inv(CAM_TRUE) @ T_bb
    T_cb_true[:3, 3] += rng.normal(0, noise_m, 3)                                           # the static camera's view of a small far board is the noisy part
    T_cb_true[:3, :3] = (Rotation.from_rotvec(np.radians(rng.normal(0, noise_deg, 3))) * Rotation.from_matrix(T_cb_true[:3, :3])).as_matrix()
    k = assumed_m / true_m
    s = lambda T: np.vstack([np.hstack([T[:3, :3], T[:3, 3:] * k]), [0, 0, 0, 1]])           # noqa: E731
    return {'time': '2026-10-26 12:00:00', 'camera': 'realsense2', 'square_m': assumed_m,
            'T_base_wrist': T_bw.tolist(), 'T_wrist_board': s(T_wb_true).tolist(), 'T_cam_board': s(T_cb_true).tolist()}


def test_the_real_square_size_undoes_the_scale_mistake_exactly():
    rec = capture(np.random.default_rng(1))
    wrong = wc.camera_pose(rec)
    right = wc.camera_pose(rec, square_mm=38.15)
    assert np.linalg.norm(wrong[:3, 3] - CAM_TRUE[:3, 3]) > 0.05
    assert np.allclose(right, CAM_TRUE, atol=1e-9)


def test_noisy_captures_average_out_and_a_bad_one_is_dropped():
    combined, singles = [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        recs = [capture(rng, noise_m=0.02, noise_deg=1.0) for _ in range(6)]
        recs.append(capture(rng, noise_m=0.5, noise_deg=30.0))                               # one garbage capture (a flipped board pose)
        poses = [wc.camera_pose(r, 38.15) for r in recs]
        c = wc.combine(poses)
        assert [i for i, _, _ in c['dropped']] == [6] and len(c['kept']) == 6
        combined.append(np.linalg.norm(c['T'][:3, 3] - CAM_TRUE[:3, 3]))
        singles.append(np.mean([np.linalg.norm(p[:3, 3] - CAM_TRUE[:3, 3]) for p in poses[:6]]))
    assert np.median(combined) < 0.7 * np.median(singles) and max(combined) < 0.06             # averaging helps; one capture is a few cm off, six are about one or two


def test_clean_captures_keep_everything():
    rng = np.random.default_rng(3)
    c = wc.combine([wc.camera_pose(capture(rng), 38.15) for _ in range(4)])
    assert len(c['kept']) == 4 and c['dropped'] == [] and c['spread_cm'] < 0.1
