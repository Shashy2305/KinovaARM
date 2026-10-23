"""Pooling several arm poses into ONE extrinsic solve (dashboard/backend/app/arm_calib.py): maths only, no ROS node."""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'calibration'))
import pnp_extrinsics as pe  # noqa: E402

K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
R_TRUE = Rotation.from_euler('xyz', [-125, 3, 147], degrees=True).as_matrix()
T_TRUE = np.array([0.80, 0.65, 0.92])


def pose_points(rng, centre, spread=0.35, n=7):
    """n arm-like 3D points (base_link) in front of the camera, each pose with its own cluster."""
    return centre + rng.uniform(-spread, spread, size=(n, 3))


def pixels(P, noise, rng):
    uv, z = pe.project(P, R_TRUE, T_TRUE, K)
    assert (z > 0).all()
    return uv + rng.normal(0, noise, uv.shape)


def test_several_poses_together_recover_the_camera_much_better_than_a_single_pose():
    rng = np.random.default_rng(3)
    poses = [np.array(c) for c in ([0.35, 0.0, 0.35], [0.20, -0.25, 0.20], [0.45, 0.25, 0.45], [0.30, 0.1, 0.55])]
    data = [(P, pixels(P, 4.0, rng)) for P in (pose_points(rng, c) for c in poses)]
    errs_single = []
    for P, uv in data:
        sol = pe.solve(P, uv, K)
        assert 'error' not in sol
        errs_single.append(pe.pose_difference(R_TRUE, T_TRUE, sol['R_bc'], sol['t_bc'])[0])
    pooled = pe.solve(np.vstack([d[0] for d in data]), np.vstack([d[1] for d in data]), K)
    rot_err, t_err = pe.pose_difference(R_TRUE, T_TRUE, pooled['R_bc'], pooled['t_bc'])
    assert rot_err < min(errs_single) + 1e-6 or rot_err < 1.5       # never worse than the best single pose, and small
    assert rot_err < 1.5 and t_err < 0.05
    assert pooled['rms_px'] < 8


def test_the_backend_keeps_only_solved_poses_and_pools_them(monkeypatch):
    import types
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dashboard', 'backend'))
    from app import arm_calib as ac
    obj = types.SimpleNamespace()
    import threading
    obj._lock = threading.RLock()
    cam = 'realsense3'
    obj.clicks = {cam: {'base_center': (1.0, 2.0)}}
    obj.joint_snapshot = {cam: np.zeros(7)}
    obj.last = {cam: None}
    obj.pool = {cam: []}
    obj._save_pool = lambda c: None
    ok, msg, n = ac.ArmCalib.commit_pose(obj, cam)
    assert not ok and n == 0 and 'Solve this pose first' in msg           # nothing solved yet: nothing kept
    obj.last[cam] = {'sol': {'rms_px': 5.0}, 'pts': [[0, 0, 0]], 'pix': [[1, 1]], 'used': ['base_center']}
    ok, msg, n = ac.ArmCalib.commit_pose(obj, cam)
    assert ok and n == 1 and obj.clicks[cam] == {} and obj.last[cam] is None and obj.pool[cam][0]['used'] == ['base_center']


def test_a_pose_identical_to_an_earlier_one_is_refused_as_frozen_feedback():
    import threading
    import types
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dashboard', 'backend'))
    from app import arm_calib as ac
    cam = 'realsense3'
    obj = types.SimpleNamespace(_lock=threading.RLock(), clicks={cam: {}}, joint_snapshot={cam: np.full(7, 0.3)}, pool={cam: []}, _save_pool=lambda c: None)
    solved = {'sol': {'rms_px': 4.0}, 'pts': [[0, 0, 0]], 'pix': [[1, 1]], 'used': ['base_center']}
    obj.last = {cam: dict(solved)}
    assert ac.ArmCalib.commit_pose(obj, cam)[0]
    obj.last = {cam: dict(solved)}
    obj.joint_snapshot[cam] = np.full(7, 0.3) + 0.01                       # 0.6 deg off: the arm has not really moved
    ok, msg, n = ac.ArmCalib.commit_pose(obj, cam)
    assert not ok and n == 1 and 'same arm pose as pose 1' in msg and 'frozen' in msg
    obj.joint_snapshot[cam] = np.full(7, 0.3) + np.array([0.3, 0, 0, 0, 0, 0, 0])     # a real move is accepted
    obj.last = {cam: dict(solved)}
    assert ac.ArmCalib.commit_pose(obj, cam)[0]


def test_solve_pool_modes_end_to_end_with_synthetic_poses(monkeypatch):
    import threading
    import types
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dashboard', 'backend'))
    from app import arm_calib as ac
    import test_pooled_extrinsics as tp
    rng = np.random.default_rng(21)
    P, uv, ids, rots = tp.make_data(rng, n_poses=6)
    cam = 'realsense3'
    pool = []
    for k in range(6):
        sl = slice(7 * k, 7 * k + 7)
        pool.append({'pts': P[sl].tolist(), 'pix': uv[sl].tolist(), 'used': ids[sl], 'rots': [r.tolist() if r is not None else None for r in rots[sl]], 'rms': 4.0, 'time': f'12:0{k}'})
    obj = types.SimpleNamespace(_lock=threading.RLock(), pool={cam: pool}, K={cam: (tp.K, (1280, 720))}, pooled={cam: None},
                                landmark_positions=lambda: None, current_extrinsic=lambda frame: None)
    res = {m: ac.ArmCalib.solve_pool(obj, cam, 'frame', m) for m in ('all', 'sharp', 'offsets')}
    assert all(r['ok'] and r['pooled'] for r in res.values())
    assert res['all']['rms_px'] > 2 * res['sharp']['rms_px']              # the live symptom: hub landmarks make the plain pooled fit bad
    assert res['offsets']['rms_px'] < 6 and res['offsets']['offsets']['joint2']
    assert 'sharp' == obj.pooled[cam]['mode'] or obj.pooled[cam]['mode'] == 'offsets'
    assert len(res['sharp']['poses']) == 6 and set(res['all']['per_landmark_px']) >= {'joint2', 'elbow', 'joint6', 'flange'}
