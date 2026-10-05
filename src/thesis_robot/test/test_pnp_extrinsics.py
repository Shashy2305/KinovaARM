import os
import sys

import numpy as np
import pytest

pytest.importorskip('cv2')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'calibration'))
import pnp_extrinsics as pe  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])


def true_camera():
    """Camera 1.1 m in front of the base and 0.9 m up, looking at the arm (optical frame:
    x right, y down, z forward)."""
    pos = np.array([1.1, 0.2, 0.9])
    fwd = np.array([0.3, 0.2, 0.2]) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.column_stack([right, down, fwd]), pos


def landmarks():
    # base, shoulder, elbow-ish, flange and the two fingertips of a reaching arm
    return np.array([[0, 0, 0], [0, 0, 0.16], [0.35, 0.1, 0.45], [0.55, 0.25, 0.30],
                     [0.62, 0.30, 0.20], [0.60, 0.22, 0.05], [0.64, 0.38, 0.05]], float)


def pixels(R, t, noise=0.0, seed=0):
    uv, z = pe.project(landmarks(), R, t, K)
    assert (z > 0).all(), 'test camera must see all landmarks'
    return uv + np.random.default_rng(seed).normal(0, noise, uv.shape)


def test_recovers_the_camera_pose_with_realistic_click_noise():
    R, t = true_camera()
    sol = pe.solve(landmarks(), pixels(R, t, noise=3.0), K)
    rot, trans = pe.pose_difference(R, t, sol['R_bc'], sol['t_bc'])
    assert rot < 1.5 and trans < 0.03
    assert sol['rms_px'] < 6


def test_exact_clicks_give_exact_pose():
    R, t = true_camera()
    sol = pe.solve(landmarks(), pixels(R, t), K)
    rot, trans = pe.pose_difference(R, t, sol['R_bc'], sol['t_bc'])
    assert rot < 0.05 and trans < 1e-3 and sol['rms_px'] < 0.1


def test_a_misclick_is_identified():
    R, t = true_camera()
    uv = pixels(R, t, noise=1.0)
    uv[4] += [90, -70]                                    # one wild click
    sol = pe.solve(landmarks(), uv, K)
    assert sol['suspect'] == 4 and any('mis-click' in w for w in sol['warnings'])
    keep = [0, 1, 2, 3, 5, 6]
    clean = pe.solve(landmarks()[keep], uv[keep], K)       # dropping it recovers the pose
    assert pe.pose_difference(R, t, clean['R_bc'], clean['t_bc'])[1] < 0.05


def test_good_clicks_report_no_suspect():
    R, t = true_camera()
    assert pe.solve(landmarks(), pixels(R, t, noise=2.0), K)['suspect'] is None


def test_too_few_or_degenerate_landmarks_are_reported():
    R, t = true_camera()
    assert 'error' in pe.solve(landmarks()[:4], pixels(R, t)[:4], K)
    line = np.array([[0, 0, z] for z in (0, 0.1, 0.2, 0.3, 0.4, 0.5)], float)
    uv, _ = pe.project(line, R, t, K)
    sol = pe.solve(line, uv, K)
    assert 'error' in sol or any('collinear' in w for w in sol['warnings'])


def test_yaml_dict_matches_what_the_tf_broadcaster_reads():
    R, t = true_camera()
    d = pe.to_yaml_dict(pe.solve(landmarks(), pixels(R, t), K), child='cam')
    assert set(d) == {'parent_frame', 'child_frame', 'translation', 'rotation_quat'}
    q = d['rotation_quat']
    assert abs(q['x'] ** 2 + q['y'] ** 2 + q['z'] ** 2 + q['w'] ** 2 - 1) < 1e-9
