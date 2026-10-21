"""
Camera extrinsics from clicked landmarks on the robot arm (see
calibration/pnp_extrinsics.py for the maths).

The operator clicks known points of the arm (base, elbow hub, wrist flange, the two
fingertips) in a still camera frame; their base_link positions come from the arm's own
forward kinematics (TF). The solve produces a CANDIDATE calibration file only; it never
touches the live ~/.ros/<camera>_calibration.yaml.
"""
import os
import sys
import threading
import time

import numpy as np
import rclpy
import tf2_ros
import yaml
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, JointState

from . import config

sys.path.insert(0, os.path.join(config.REPO_ROOT, 'calibration'))
import pnp_extrinsics as pe  # noqa: E402

CAMERA_INFO = {
    'oakd': '/global_camera/color/camera_info',
    'realsense': '/global_camera/global_camera/color/camera_info',
    'realsense2': '/global_camera_2/global_camera_2/color/camera_info',
    'realsense3': '/global_camera_3/global_camera_3/color/camera_info',
}
CANDIDATE_DIR = os.path.expanduser('~/.ros/calibration_candidates')
PAD_FRAMES = ('left_inner_finger_pad', 'right_inner_finger_pad')
PAD_TIP_OFFSET_M = 0.035          # pad frame origin -> end of the pad along the tool axis
MAX_ARM_DRIFT_RAD = 0.01          # the arm must not move between the first click and the solve

# Click any 5 or more that are visible in the current pose; hubs hidden behind the
# arm can simply be skipped. Joint hub = the centre of the round cap of that joint.
LANDMARKS = [
    {'id': 'base_center', 'frame': None,
     'label': 'Base centre',
     'hint': 'Centre of the bottom of the round base where it meets the black mounting disc (use the centre of the ellipse, not its front edge).'},
    {'id': 'joint2', 'frame': 'half_arm_1_link',
     'label': 'Shoulder hub (joint 2)',
     'hint': 'Centre of the round cap where the upper arm pivots on the shoulder.'},
    {'id': 'elbow', 'frame': 'forearm_link',
     'label': 'Elbow hub (joint 4)',
     'hint': 'Centre of the round cap of the elbow joint.'},
    {'id': 'joint6', 'frame': 'spherical_wrist_2_link',
     'label': 'Wrist hub (joint 6)',
     'hint': 'Centre of the round cap of the wrist bend just before the gripper.'},
    {'id': 'flange', 'frame': 'end_effector_link',
     'label': 'Wrist flange',
     'hint': 'Centre of the round face where the gripper bolts onto the wrist.'},
    {'id': 'fingertip_1', 'frame': None,
     'label': 'Fingertip (either)',
     'hint': 'The very end of one gripper finger. The two fingertips are interchangeable.'},
    {'id': 'fingertip_2', 'frame': None,
     'label': 'Fingertip (the other)',
     'hint': 'The very end of the other gripper finger.'},
]


class ArmCalib(Node):
    def __init__(self):
        super().__init__('arm_calib')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._lock = threading.RLock()
        self.K = {}
        self.clicks = {c: {} for c in CAMERA_INFO}
        self.joint_snapshot = {c: None for c in CAMERA_INFO}
        self.last = {c: None for c in CAMERA_INFO}      # last solve result
        # Poses added to the pool: one arm pose fixes a far camera only weakly (live 2026-10-20: two single-pose solves of RealSense 3
        # at 5.5 and 9.7 px RMS disagreed by 17.7 deg), so the matched (arm position, pixel) pairs of several poses are solved TOGETHER.
        self.pool = {c: [] for c in CAMERA_INFO}
        self.pooled = {c: None for c in CAMERA_INFO}    # last pooled solve
        self._js = None
        for cam, topic in CAMERA_INFO.items():
            self.create_subscription(CameraInfo, topic, lambda m, c=cam: self._on_info(c, m), 1)
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)

    def _on_info(self, cam, msg):
        with self._lock:
            self.K[cam] = (np.array(msg.k).reshape(3, 3), (msg.width, msg.height))

    def _on_js(self, msg):
        with self._lock:
            self._js = dict(zip(msg.name, msg.position))

    # ── arm landmarks from forward kinematics ─────────────────────────
    def _pos(self, frame):
        t = self.tf_buffer.lookup_transform('base_link', frame, rclpy.time.Time())
        tr, q = t.transform.translation, t.transform.rotation
        return np.array([tr.x, tr.y, tr.z]), Rotation.from_quat([q.x, q.y, q.z, q.w])

    def landmark_positions(self):
        """{id: xyz in base_link} or None when the arm's TF is unavailable."""
        try:
            out = {'base_center': np.zeros(3)}
            for lm in LANDMARKS:
                if lm['frame'] and lm['id'] != 'flange':
                    out[lm['id']] = self._pos(lm['frame'])[0]
            p_ee, r_ee = self._pos('end_effector_link')
            out['flange'] = p_ee
            axis = r_ee.apply([0, 0, 1])                           # tool axis in base_link
            tips = [self._pos(f)[0] + PAD_TIP_OFFSET_M * axis for f in PAD_FRAMES]
            out['fingertip_1'], out['fingertip_2'] = tips
            return out
        except Exception:
            return None

    def current_extrinsic(self, frame_id):
        try:
            p, r = self._pos(frame_id)
            return r.as_matrix(), p
        except Exception:
            return None

    def arm_joint_vector(self):
        with self._lock:
            js = dict(self._js or {})
        return np.array([js.get(f'joint_{i}', np.nan) for i in range(1, 8)])

    # ── operations ────────────────────────────────────────────────────
    def state(self, cam, frame_id, image_size):
        pos = self.landmark_positions()
        with self._lock:
            K = self.K.get(cam)
        out = {'camera': cam, 'frame_id': frame_id, 'image': image_size,
               'arm_ok': pos is not None, 'k_ok': K is not None, 'landmarks': [],
               'pool': [{'n': len(p['pts']), 'rms_px': p['rms'], 'time': p['time']} for p in self.pool[cam]]}
        cur = self.current_extrinsic(frame_id) if frame_id else None
        out['current_extrinsic_ok'] = cur is not None
        with self._lock:
            clicked = dict(self.clicks[cam])
        for lm in LANDMARKS:
            e = {k: v for k, v in lm.items() if k != 'frame'}
            e.update(clicked=clicked.get(lm['id']), xyz=None, projected_now=None)
            if pos is not None:
                e['xyz'] = [round(float(v), 4) for v in pos[lm['id']]]
                if K is not None and cur is not None:
                    uv, z = pe.project(pos[lm['id']][None, :], cur[0], cur[1], K[0])
                    if z[0] > 0:
                        e['projected_now'] = [float(uv[0, 0]), float(uv[0, 1])]
            out['landmarks'].append(e)
        return out

    def click(self, cam, landmark, u, v):
        ids = {lm['id'] for lm in LANDMARKS}
        if landmark not in ids:
            return False, f'unknown landmark {landmark!r}'
        joints = self.arm_joint_vector()        # takes self._lock itself: never call it while holding it
        with self._lock:
            if not self.clicks[cam]:
                self.joint_snapshot[cam] = joints
            self.clicks[cam][landmark] = (float(u), float(v))
        return True, 'ok'

    def reset(self, cam):
        with self._lock:
            self.clicks[cam] = {}
            self.joint_snapshot[cam] = None
            self.last[cam] = None

    def solve(self, cam, frame_id):
        with self._lock:
            clicks = dict(self.clicks[cam])
            K = self.K.get(cam)
            snap = self.joint_snapshot[cam]
        if K is None:
            return {'error': 'no camera_info for this camera yet'}
        pos = self.landmark_positions()
        if pos is None:
            return {'error': 'arm pose unavailable (TF)'}
        if snap is not None:
            drift = np.nanmax(np.abs(self.arm_joint_vector() - snap))
            if drift > MAX_ARM_DRIFT_RAD:
                return {'error': f'the arm moved ({drift:.3f} rad) since the first click; Reset and click again'}
        fixed = [lm['id'] for lm in LANDMARKS if not lm['id'].startswith('fingertip')]   # fingertips are handled as an interchangeable pair below
        pts, pix, names = [], [], []
        for i in fixed:
            if i in clicks:
                pts.append(pos[i]); pix.append(clicks[i]); names.append(i)
        tips = [clicks[i] for i in ('fingertip_1', 'fingertip_2') if i in clicks]
        best = None
        if len(tips) == 2:
            # the two fingertips are interchangeable: try both assignments, keep the better fit
            for swap in (False, True):
                t_pts = [pos['fingertip_1'], pos['fingertip_2']]
                t_pix = tips[::-1] if swap else tips
                sol = pe.solve(pts + t_pts, pix + t_pix, K[0])
                if 'error' not in sol and (best is None or sol['rms_px'] < best[0]['rms_px']):
                    best = (sol, names + ['fingertip_1', 'fingertip_2'], pts + t_pts, pix + t_pix)
        else:
            use = list(tips)
            sol = pe.solve(pts + ([pos['fingertip_1']] if use else []), pix + use, K[0])
            if 'error' not in sol:
                best = (sol, names + (['fingertip_1'] if use else []), pts + ([pos['fingertip_1']] if use else []), pix + use)
            else:
                return sol
        if best is None:
            return {'error': 'PnP failed'}
        sol, used, P, uv = best
        proj_all, _ = pe.project(np.array([pos[lm['id']] for lm in LANDMARKS]), sol['R_bc'], sol['t_bc'], K[0])
        cur = self.current_extrinsic(frame_id) if frame_id else None
        diff = pe.pose_difference(cur[0], cur[1], sol['R_bc'], sol['t_bc']) if cur else None
        result = {
            'ok': True, 'landmarks_used': used, 'rms_px': sol['rms_px'],
            'reproj_px': dict(zip(used, sol['reproj_px'])),
            'suspect': used[sol['suspect']] if sol['suspect'] is not None else None,
            'warnings': sol['warnings'],
            'solved_projection': {lm['id']: [float(proj_all[k, 0]), float(proj_all[k, 1])] for k, lm in enumerate(LANDMARKS)},
            'translation': [float(v) for v in sol['t_bc']], 'quat_xyzw': sol['quat_xyzw'],
            'vs_current': {'rotation_deg': diff[0], 'translation_m': diff[1]} if diff else None,
        }
        with self._lock:
            self.last[cam] = {'sol': sol, 'frame_id': frame_id, 'pts': [list(map(float, p)) for p in P],
                              'pix': [list(map(float, u)) for u in uv], 'used': list(used)}
        return result

    # ── several poses solved together ───────────────────────────────────────────────────────────────────────────────
    def commit_pose(self, cam):
        """Keep the points of the pose just solved, then clear the clicks so the next arm pose can be clicked."""
        with self._lock:
            last = self.last.get(cam)
            if not last or 'pts' not in last:
                return False, 'Solve this pose first: its points are only kept after a successful solve', len(self.pool[cam])
            self.pool[cam].append({'pts': last['pts'], 'pix': last['pix'], 'used': last['used'],
                                   'rms': float(last['sol']['rms_px']), 'time': time.strftime('%H:%M:%S')})
            self.clicks[cam] = {}
            self.joint_snapshot[cam] = None
            self.last[cam] = None
            n = len(self.pool[cam])
        return True, f'pose {n} added to the pool; move the arm to a different pose and click again', n

    def clear_pool(self, cam):
        with self._lock:
            self.pool[cam] = []
            self.pooled[cam] = None

    def solve_pool(self, cam, frame_id):
        with self._lock:
            pool = list(self.pool[cam])
            K = self.K.get(cam)
        if K is None:
            return {'error': 'no camera_info for this camera yet'}
        if not pool:
            return {'error': 'the pool is empty: solve a pose, then press Add this pose'}
        P = np.array([p for e in pool for p in e['pts']])
        uv = np.array([u for e in pool for u in e['pix']])
        sol = pe.solve(P, uv, K[0])
        if 'error' in sol:
            return sol
        proj, _ = pe.project(P, sol['R_bc'], sol['t_bc'], K[0])
        err = np.linalg.norm(proj - uv, axis=1)
        poses, k = [], 0
        for e in pool:
            n = len(e['pts'])
            poses.append({'n': n, 'rms_px': float(np.sqrt(np.mean(err[k:k + n] ** 2))), 'single_pose_rms_px': e['rms'], 'time': e['time']})
            k += n
        cur_pos = self.landmark_positions()
        solved_projection = {}
        if cur_pos is not None:
            pa, _ = pe.project(np.array([cur_pos[lm['id']] for lm in LANDMARKS]), sol['R_bc'], sol['t_bc'], K[0])
            solved_projection = {lm['id']: [float(pa[i, 0]), float(pa[i, 1])] for i, lm in enumerate(LANDMARKS)}
        cur = self.current_extrinsic(frame_id) if frame_id else None
        diff = pe.pose_difference(cur[0], cur[1], sol['R_bc'], sol['t_bc']) if cur else None
        result = {
            'ok': True, 'pooled': True, 'rms_px': sol['rms_px'], 'points': int(len(P)), 'poses': poses,
            'worst_px': float(err.max()), 'warnings': sol['warnings'], 'solved_projection': solved_projection,
            'translation': [float(v) for v in sol['t_bc']], 'quat_xyzw': sol['quat_xyzw'],
            'vs_current': {'rotation_deg': diff[0], 'translation_m': diff[1]} if diff else None,
        }
        with self._lock:
            self.pooled[cam] = {'sol': sol, 'frame_id': frame_id, 'poses': len(pool), 'points': int(len(P))}
        return result

    def save_candidate(self, cam, pooled=False):
        with self._lock:
            last = self.pooled.get(cam) if pooled else self.last.get(cam)
        if not last:
            return False, 'solve the pool first' if pooled else 'solve first', None
        os.makedirs(CANDIDATE_DIR, exist_ok=True)
        path = os.path.join(CANDIDATE_DIR, f'{cam}_calibration.candidate.yaml')
        d = pe.to_yaml_dict(last['sol'], 'base_link', last['frame_id'])
        d['note'] = (f'Candidate from clicked arm landmarks (calibration/pnp_extrinsics.py)' + (f', {last["poses"]} arm poses solved together ({last["points"]} points)' if pooled else '') + ', '
                     f'{time.strftime("%Y-%m-%d %H:%M")}, RMS {last["sol"]["rms_px"]:.1f}px. '
                     f'NOT active: copy it over ~/.ros/{cam}_calibration.yaml after checking it.')
        with open(path, 'w') as f:
            yaml.safe_dump(d, f, default_flow_style=False)
        return True, f'saved {path}', path
