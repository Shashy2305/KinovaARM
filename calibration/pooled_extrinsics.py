"""
Camera extrinsics from the clicked arm landmarks of SEVERAL arm poses solved together (pure numpy/OpenCV/scipy, no ROS).

Why a second solver next to pnp_extrinsics.solve: on 2026-10-21 seven arm poses of RealSense 3 each fitted at 4-10 px, but together they fitted at 18 px (worst 37 px), spread
evenly over all poses. One pose can hide a systematic landmark error by nudging the camera; several poses cannot. The likely culprit is the hub landmarks (shoulder, elbow,
wrist): the person clicks the centre of the visible round cap, the model's joint point is inside the arm, a few centimetres along the joint axis, and at 1.3 m
4-5 cm are 30-40 px. The base centre, the wrist flange and the fingertips are sharp points with no such offset.

Modes:
  'all'      every clicked landmark (what pnp_extrinsics does, with all poses stacked)
  'sharp'    only base centre, flange and fingertips (needs >= 6 points)
  'offsets'  every landmark, plus one unknown 3D offset per hub landmark, expressed in that link's own frame (so it turns with the arm), estimated together
             with the camera pose; a weak prior keeps each offset below about 8 cm and a robust loss limits the damage of a mis-click.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import pnp_extrinsics as pe

SHARP = ('base_center', 'flange', 'fingertip_1', 'fingertip_2')
HUBS = ('joint2', 'elbow', 'joint6')
MODES = ('all', 'sharp', 'offsets')
OFFSET_PRIOR_M = 0.05                    # an offset of this size costs PRIOR_PX pixels
PRIOR_PX = 10.0


def _pack(ids, points, pixels, rots):
    return list(ids), np.asarray(points, float).reshape(-1, 3), np.asarray(pixels, float).reshape(-1, 2), rots


def solve_pooled(points, pixels, ids, rots, K, mode='all'):
    """points: (N,3) base_link positions of the landmarks (from forward kinematics); pixels: (N,2); ids: landmark id per point; rots: (3,3) rotation of the
    landmark's link frame in base_link per point (None where unused). Returns {'R_bc','t_bc','quat_xyzw','rms_px','per_point_px','per_landmark_px','offsets','mode','n'}
    or {'error': ...}."""
    ids, P, uv, rots = _pack(ids, points, pixels, rots)
    K = np.asarray(K, float)
    if mode not in MODES:
        return {'error': f'unknown mode {mode!r}'}
    keep = np.arange(len(ids))
    if mode == 'sharp':
        keep = np.array([i for i, k in enumerate(ids) if k in SHARP])
        if len(keep) < 6:
            return {'error': f'only {len(keep)} sharp landmarks (base centre, flange, fingertips) in the pool: need at least 6; add poses or click them'}
    base = pe.solve(P[keep], uv[keep], K)
    if 'error' in base:
        return base
    R_bc, t_bc = base['R_bc'], base['t_bc']
    offsets = {}
    if mode == 'offsets':
        hub_idx = {h: [i for i, k in enumerate(ids) if k == h] for h in HUBS}
        # start from the camera that fits the sharp points (if there are enough), else all points
        sharp_idx = np.array([i for i, k in enumerate(ids) if k in SHARP])
        if len(sharp_idx) >= 6:
            s0 = pe.solve(P[sharp_idx], uv[sharp_idx], K)
            if 'error' not in s0:
                R_bc, t_bc = s0['R_bc'], s0['t_bc']
        rv0 = Rotation.from_matrix(R_bc.T).as_rotvec()               # camera <- base rotation vector
        t0 = -R_bc.T @ t_bc
        x0 = np.concatenate([rv0, t0, np.zeros(3 * len(HUBS))])

        def model(x):
            rv, tv, off = x[:3], x[3:6], x[6:].reshape(len(HUBS), 3)
            Q = P.copy()
            for h_i, h in enumerate(HUBS):
                for i in hub_idx[h]:
                    if rots[i] is not None:
                        Q[i] = Q[i] + np.asarray(rots[i], float) @ off[h_i]
            proj, _ = cv2.projectPoints(Q, rv, tv, K, None)
            return proj.reshape(-1, 2)

        def resid(x):
            r = (model(x) - uv).reshape(-1)
            prior = x[6:] * (PRIOR_PX / OFFSET_PRIOR_M)
            return np.concatenate([r, prior])

        sol = least_squares(resid, x0, loss='soft_l1', f_scale=8.0, max_nfev=400)
        rv, tv = sol.x[:3], sol.x[3:6]
        R_cb = Rotation.from_rotvec(rv).as_matrix()
        R_bc = R_cb.T
        t_bc = -R_bc @ tv
        off = sol.x[6:].reshape(len(HUBS), 3)
        offsets = {h: [float(v) for v in off[i]] for i, h in enumerate(HUBS)}
        err_all = np.linalg.norm(model(sol.x) - uv, axis=1)
    else:
        proj, _ = pe.project(P, R_bc, t_bc, K)
        err_all = np.linalg.norm(proj - uv, axis=1)
    used = err_all[keep]
    per_landmark = {}
    for k in sorted(set(ids)):
        sel = [i for i, kk in enumerate(ids) if kk == k and i in set(keep.tolist())]
        if sel:
            per_landmark[k] = float(np.sqrt(np.mean(err_all[sel] ** 2)))
    q = Rotation.from_matrix(R_bc).as_quat()
    return {'R_bc': R_bc, 't_bc': t_bc, 'quat_xyzw': [float(v) for v in q], 'rms_px': float(np.sqrt(np.mean(used ** 2))), 'worst_px': float(used.max()),
            'per_point_px': [float(v) for v in err_all], 'per_landmark_px': per_landmark, 'offsets': offsets, 'mode': mode, 'n': int(len(keep)), 'kept': keep.tolist()}
