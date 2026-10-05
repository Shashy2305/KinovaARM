"""
Camera extrinsics from clicked landmarks on the ROBOT ARM (pure numpy/OpenCV, no ROS).

The arm is the ideal calibration target: its pose is known exactly from the
joint angles (forward kinematics), it is rigidly attached to base_link, and every
static camera can see it. Given N >= 5 landmarks with known base_link positions and
the pixel where each appears in a camera image, solve the camera pose with PnP.

Convention (matches how the whole pipeline uses images): the image's frame_id is
treated as an OPTICAL frame (x right, y down, z forward). The result is the
transform base_link -> that frame, i.e. the `translation` / `rotation_quat` that
static_tf_broadcaster.py publishes.
"""
import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

MIN_POINTS = 5


def project(points_base, R_bc, t_bc, K):
    """Pixels of base_link points seen by a camera at pose (R_bc, t_bc) in base_link.
    Returns (N, 2) pixels and (N,) depths (negative = behind the camera)."""
    P = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    cam = (np.asarray(R_bc).T @ (P - np.asarray(t_bc)).T).T
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * cam[:, 1] / safe + K[1, 2]
    return np.column_stack([u, v]), z


def _solve_once(P, uv, K):
    rvec = tvec = None
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP):     # SQPnP asserts on degenerate sets
        try:
            ok, rv, tv = cv2.solvePnP(P, uv, K, None, flags=flag)
        except cv2.error:
            continue
        if ok:
            rvec, tvec = rv, tv
            break
    if rvec is None:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(P, uv, K, None, rvec, tvec)
    R_cb, _ = cv2.Rodrigues(rvec)                      # x_cam = R_cb x_base + t
    R_bc = R_cb.T
    t_bc = -R_bc @ tvec.reshape(3)
    return R_bc, t_bc


def solve(points_base, pixels, K):
    """Returns a dict with R_bc, t_bc, quat_xyzw, per-point reprojection error,
    the index of a suspected mis-click, rms and human-readable warnings; or
    {'error': ...} if it cannot be solved."""
    P = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    uv = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    n = len(P)
    if n < MIN_POINTS:
        return {'error': f'need at least {MIN_POINTS} landmarks, got {n}'}
    sol = _solve_once(P, uv, K)
    if sol is None:
        return {'error': 'PnP failed'}
    R_bc, t_bc = sol
    proj, z = project(P, R_bc, t_bc, K)
    err = np.linalg.norm(proj - uv, axis=1)

    # Mis-click finder: refit without each point in turn; the culprit is the point
    # whose REMOVAL shrinks the remaining points' reprojection RMS the most.
    drop_rms = []
    for i in range(n):
        keep = np.arange(n) != i
        s = _solve_once(P[keep], uv[keep], K) if n - 1 >= 4 else None
        if s is None:
            drop_rms.append(float('nan'))
            continue
        p, _ = project(P[keep], s[0], s[1], K)
        drop_rms.append(float(np.sqrt(np.mean(np.sum((p - uv[keep]) ** 2, axis=1)))))
    rms_all = float(np.sqrt(np.mean(err ** 2)))
    suspect = None
    if not np.all(np.isnan(drop_rms)):
        j = int(np.nanargmin(drop_rms))
        if drop_rms[j] < 0.5 * rms_all and rms_all > 4:
            suspect = j

    warnings = []
    sv = np.linalg.svd(P - P.mean(0), compute_uv=False)
    if sv[2] < 0.05 * sv[0]:
        warnings.append('landmarks are nearly coplanar/collinear: the result is poorly constrained, '
                        'move the arm so the points spread out in 3D')
    if (z <= 0).any():
        warnings.append('some landmarks end up behind the camera: wrong click or wrong camera')
    spread = float(np.ptp(uv[:, 0]) + np.ptp(uv[:, 1]))
    if spread < 150:
        warnings.append('landmarks cover a small part of the image: accuracy will be poor')
    rms = float(np.sqrt(np.mean(err ** 2)))
    if rms > 8:
        warnings.append(f'reprojection RMS {rms:.1f}px is high' + (f': landmark #{suspect + 1} looks like a mis-click' if suspect is not None else ''))
    q = Rotation.from_matrix(R_bc).as_quat()
    return {
        'R_bc': R_bc, 't_bc': t_bc, 'quat_xyzw': [float(v) for v in q],
        'reproj_px': [float(e) for e in err], 'drop_rms_px': drop_rms, 'suspect': suspect,
        'rms_px': rms, 'warnings': warnings,
    }


def pose_difference(R_a, t_a, R_b, t_b):
    """(rotation difference in degrees, translation difference in metres)."""
    ang = Rotation.from_matrix(np.asarray(R_a).T @ np.asarray(R_b)).magnitude()
    return math.degrees(ang), float(np.linalg.norm(np.asarray(t_a) - np.asarray(t_b)))


def to_yaml_dict(sol, parent='base_link', child='camera_optical_frame'):
    t, q = sol['t_bc'], sol['quat_xyzw']
    return {
        'parent_frame': parent, 'child_frame': child,
        'translation': {'x': float(t[0]), 'y': float(t[1]), 'z': float(t[2])},
        'rotation_quat': {'x': q[0], 'y': q[1], 'z': q[2], 'w': q[3]},
    }
