"""
Forward kinematics and Jacobian of the arm straight from the URDF text (pure numpy, no ROS), for the dashboard jog.

Why not MoveIt's /compute_fk: the jog asks for a Cartesian step ten times a second and must check the guarded links for every one, so it needs an answer in
microseconds and without a service call that can time out. The URDF comes from /robot_state_publisher's `robot_description` parameter (the model that is running).
Mimic joints (the gripper's linkage) follow their master joint; every non-arm joint that is neither given nor mimicking stays at 0 (fingers open).
"""
import xml.etree.ElementTree as ET

import numpy as np

ARM_JOINTS = [f'joint_{i}' for i in range(1, 8)]


def _vec(text, default):
    return np.array([float(v) for v in text.split()]) if text else np.array(default, float)


def rot_axis(axis, angle):
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n == 0:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rot_rpy(r, p, y):
    return rot_axis([0, 0, 1], y) @ rot_axis([0, 1, 0], p) @ rot_axis([1, 0, 0], r)


def rotvec(R):
    """Rotation vector of a rotation matrix (small or large angles)."""
    c = (np.trace(R) - 1.0) / 2.0
    ang = float(np.arccos(np.clip(c, -1.0, 1.0)))
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if abs(np.pi - ang) < 1e-6:                    # half turn: sin ~ 0, use the symmetric part
        M = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(M), 0))
        i = int(np.argmax(axis))
        axis = M[:, i] / max(axis[i], 1e-12)
        return axis / np.linalg.norm(axis) * ang
    return v / (2.0 * np.sin(ang)) * ang


class Kinematics:
    def __init__(self, urdf_xml, root='base_link'):
        tree = ET.fromstring(urdf_xml)
        self.root = root
        self.joints = {}
        self.parent_joint = {}                      # child link -> joint name
        for j in tree.findall('joint'):
            name = j.get('name')
            origin = j.find('origin')
            xyz = _vec(origin.get('xyz') if origin is not None else None, [0, 0, 0])
            rpy = _vec(origin.get('rpy') if origin is not None else None, [0, 0, 0])
            axis_el = j.find('axis')
            axis = _vec(axis_el.get('xyz') if axis_el is not None else None, [1, 0, 0])
            lim = j.find('limit')
            mim = j.find('mimic')
            kind = j.get('type')
            lower = upper = None
            if kind == 'revolute' and lim is not None:
                lower, upper = float(lim.get('lower')), float(lim.get('upper'))
            T0 = np.eye(4)
            T0[:3, :3] = rot_rpy(*rpy)
            T0[:3, 3] = xyz
            self.joints[name] = {
                'type': kind, 'parent': j.find('parent').get('link'), 'child': j.find('child').get('link'),
                'T0': T0, 'axis': axis, 'lower': lower, 'upper': upper,
                'mimic': (mim.get('joint'), float(mim.get('multiplier', 1.0)), float(mim.get('offset', 0.0))) if mim is not None else None,
            }
            self.parent_joint[j.find('child').get('link')] = name
        missing = [j for j in ARM_JOINTS if j not in self.joints]
        if missing:
            raise ValueError(f'URDF has no joints {missing}')
        self._path_cache = {}

    def limits(self):
        """(lower, upper) arrays for the 7 arm joints; +-inf for continuous joints."""
        lo = [self.joints[j]['lower'] if self.joints[j]['lower'] is not None else -np.inf for j in ARM_JOINTS]
        hi = [self.joints[j]['upper'] if self.joints[j]['upper'] is not None else np.inf for j in ARM_JOINTS]
        return np.array(lo), np.array(hi)

    def _path(self, link):
        if link not in self._path_cache:
            chain, cur = [], link
            while cur != self.root:
                if cur not in self.parent_joint:
                    raise KeyError(f'link {link!r} is not below {self.root!r}')
                jn = self.parent_joint[cur]
                chain.append(jn)
                cur = self.joints[jn]['parent']
            self._path_cache[link] = chain[::-1]
        return self._path_cache[link]

    def _value(self, jn, q):
        j = self.joints[jn]
        if jn in q:
            return q[jn]
        if j['mimic'] is not None:
            master, mult, off = j['mimic']
            return mult * q.get(master, 0.0) + off
        return 0.0

    def link_pose(self, link, q):
        """4x4 pose of `link` in the root frame. q: {joint name: value} (missing joints = 0)."""
        T = np.eye(4)
        for jn in self._path(link):
            j = self.joints[jn]
            T = T @ j['T0']
            if j['type'] in ('revolute', 'continuous'):
                M = np.eye(4)
                M[:3, :3] = rot_axis(j['axis'], self._value(jn, q))
                T = T @ M
        return T

    def pose_for(self, links, qvec, extra=None):
        """{link: 4x4} for an arm configuration (7 values) plus optional extra joint values (gripper)."""
        q = dict(zip(ARM_JOINTS, np.asarray(qvec, float)))
        if extra:
            q.update(extra)
        return {name: self.link_pose(name, q) for name in links}

    def jacobian(self, qvec, link='end_effector_link', eps=1e-6):
        """6x7 geometric Jacobian in the root frame: rows 0-2 linear velocity of the link origin, rows 3-5 angular velocity (rotation-vector rate)."""
        qvec = np.asarray(qvec, float)
        T0 = self.pose_for([link], qvec)[link]
        J = np.zeros((6, len(ARM_JOINTS)))
        for i in range(len(ARM_JOINTS)):
            dq = qvec.copy()
            dq[i] += eps
            T1 = self.pose_for([link], dq)[link]
            J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
            J[3:, i] = rotvec(T1[:3, :3] @ T0[:3, :3].T) / eps
        return J
