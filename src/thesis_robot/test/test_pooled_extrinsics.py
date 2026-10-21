"""calibration/pooled_extrinsics.py: several arm poses solved together, with and without hub landmarks that sit off the joint points (synthetic arm and camera)."""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'calibration'))
import pnp_extrinsics as pe  # noqa: E402
import pooled_extrinsics as px  # noqa: E402

K = np.array([[930.0, 0, 640], [0, 930.0, 360], [0, 0, 1]])
CAM = np.array([0.75, 0.65, 0.95])
TRUE_OFFSETS = {'joint2': np.array([0.0, 0.0, 0.045]), 'elbow': np.array([0.0, 0.0, -0.040]), 'joint6': np.array([0.0, 0.0, 0.035])}


def look_at_R():
    z = np.array([0.3, 0.0, 0.2]) - CAM
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


R_BC = look_at_R()


def make_data(rng, n_poses=7, noise=2.0, with_offsets=True):
    pts, pix, ids, rots = [], [], [], []
    for _ in range(n_poses):
        elbow = np.array([rng.uniform(0.1, 0.35), rng.uniform(-0.3, 0.3), rng.uniform(0.25, 0.6)])
        j6 = np.array([rng.uniform(0.3, 0.55), rng.uniform(-0.3, 0.3), rng.uniform(0.1, 0.45)])
        flange = j6 + np.array([rng.uniform(0.0, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(-0.15, -0.05)])
        tips = [flange + np.array([s * 0.05, 0, -0.2]) for s in (1, -1)]
        entries = [('base_center', np.zeros(3), None), ('joint2', np.array([0, 0, 0.28]), Rotation.random(random_state=int(rng.integers(1e6))).as_matrix()),
                   ('elbow', elbow, Rotation.random(random_state=int(rng.integers(1e6))).as_matrix()),
                   ('joint6', j6, Rotation.random(random_state=int(rng.integers(1e6))).as_matrix()),
                   ('flange', flange, None), ('fingertip_1', tips[0], None), ('fingertip_2', tips[1], None)]
        for name, p, R in entries:
            seen = p + (R @ TRUE_OFFSETS[name] if (with_offsets and name in TRUE_OFFSETS) else 0.0)     # where the person clicks: the cap, not the joint point
            uv, z = pe.project(seen[None], R_BC, CAM, K)
            assert z[0] > 0
            pts.append(p)
            pix.append(uv[0] + rng.normal(0, noise, 2))
            ids.append(name)
            rots.append(R)
    return np.array(pts), np.array(pix), ids, rots


def pose_error(sol):
    cm = 100 * float(np.linalg.norm(sol['t_bc'] - CAM))
    deg = float(np.degrees(Rotation.from_matrix(sol['R_bc'] @ R_BC.T).magnitude()))
    return cm, deg


def test_hub_offsets_make_the_pooled_all_landmark_fit_bad_and_the_other_modes_recover_the_camera():
    rng = np.random.default_rng(11)
    P, uv, ids, rots = make_data(rng)
    a = px.solve_pooled(P, uv, ids, rots, K, 'all')
    s = px.solve_pooled(P, uv, ids, rots, K, 'sharp')
    o = px.solve_pooled(P, uv, ids, rots, K, 'offsets')
    # all three are solved; the plain fit shows the symptom seen live (a high pooled RMS)
    assert a['rms_px'] > 2.0 * s['rms_px'] and a['rms_px'] > 8.0
    cm_a, _ = pose_error(a)
    cm_s, deg_s = pose_error(s)
    cm_o, deg_o = pose_error(o)
    assert cm_s < 3.0 and deg_s < 1.5 and s['rms_px'] < 6.0
    assert cm_o < 3.0 and deg_o < 1.5 and o['rms_px'] < 6.0
    assert cm_a > cm_o                                         # the hub offsets bias the plain fit
    for h, true in TRUE_OFFSETS.items():                       # and the offsets come out near the truth
        assert np.linalg.norm(np.array(o['offsets'][h]) - true) < 0.02, (h, o['offsets'][h])


def test_without_hub_offsets_all_three_modes_agree_and_fit_at_the_noise_level():
    rng = np.random.default_rng(3)
    P, uv, ids, rots = make_data(rng, with_offsets=False)
    for mode in px.MODES:
        r = px.solve_pooled(P, uv, ids, rots, K, mode)
        assert r['rms_px'] < 4.0 and pose_error(r)[0] < 3.0, mode


def test_per_landmark_errors_point_at_the_hubs_in_the_plain_fit():
    rng = np.random.default_rng(5)
    P, uv, ids, rots = make_data(rng)
    a = px.solve_pooled(P, uv, ids, rots, K, 'all')
    worst = max(a['per_landmark_px'], key=a['per_landmark_px'].get)
    assert worst in px.HUBS


def test_sharp_mode_needs_enough_sharp_points():
    r = px.solve_pooled(np.zeros((4, 3)) + np.arange(12).reshape(4, 3) * 0.1, np.zeros((4, 2)) + 100, ['base_center', 'flange', 'fingertip_1', 'joint2'], [None] * 4, K, 'sharp')
    assert 'error' in r
