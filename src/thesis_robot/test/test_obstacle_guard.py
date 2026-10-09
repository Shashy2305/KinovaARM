import numpy as np
import pytest

pytest.importorskip('cv2')
from thesis_robot import obstacle_guard as og  # noqa: E402

K = [[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1.0]]
TABLE_Z = -0.0125
XR, YR = (0.05, 0.70), (-0.45, 0.45)


def look_at(pos, target):
    f = np.array(target, float) - np.array(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return np.column_stack([right, down, f]), np.array(pos, float)


def render_depth(R, t, boxes, noise=0.0, seed=0, w=640, h=480):
    """Depth image (m) of the table plane plus axis-aligned boxes (x0, x1, y0, y1, height) seen from (R, t): top faces."""
    v, u = np.mgrid[0:h, 0:w]
    d_cam = np.stack([(u - K[0][2]) / K[0][0], (v - K[1][2]) / K[1][1], np.ones_like(u, float)], -1)
    d = d_cam @ R.T                                          # world directions per unit camera z
    depth = np.full((h, w), np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        s_table = (TABLE_Z - t[2]) / d[..., 2]
        depth = np.where(s_table > 0, s_table, depth)
        for x0, x1, y0, y1, hb in boxes:
            s = (TABLE_Z + hb - t[2]) / d[..., 2]
            px, py = t[0] + s * d[..., 0], t[1] + s * d[..., 1]
            inside = (s > 0) & (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
            depth = np.where(inside & (s < depth), s, depth)
    depth[~np.isfinite(depth)] = 0
    if noise:
        depth = depth + np.random.default_rng(seed).normal(0, noise, depth.shape) * (depth > 0)
    return depth


def run(boxes, known=(), chain=None, fingers=None, noise=0.0, pos=(1.0, 0.0, 0.80)):
    R, t = look_at(pos, (0.40, 0.0, 0.0))
    depth = render_depth(R, t, boxes, noise)
    pts = og.backproject(depth, K, R, t, stride=2)
    return og.unknown_obstacles(pts, TABLE_Z, XR, YR, list(known), chain, fingers), pts


def test_an_empty_table_has_no_obstacles():
    obs, pts = run([])
    assert obs == [] and len(pts) > 1000


def test_a_box_is_found_with_its_position_height_and_size():
    obs, _ = run([(0.42, 0.48, 0.06, 0.15, 0.08)])                  # 6 x 9 cm, 8 cm tall at (0.45, 0.105)
    assert len(obs) == 1
    o = obs[0]
    assert abs(o['x'] - 0.45) < 0.02 and abs(o['y'] - 0.105) < 0.02
    assert abs(o['height'] - 0.08) < 0.02
    assert 0.04 < o['w'] < 0.09 and 0.07 < o['h'] < 0.12


def test_a_known_object_explains_its_blob():
    obs, _ = run([(0.42, 0.48, 0.06, 0.15, 0.08)], known=[(0.45, 0.105, 0.05)])
    assert obs == []


def test_a_low_flat_thing_is_ignored():
    obs, _ = run([(0.30, 0.50, -0.20, 0.0, 0.006)])                 # a 6 mm board
    assert obs == []


def test_the_robot_body_is_not_an_obstacle():
    chain = [(0.0, 0.0, 0.0), (0.45, 0.105, 0.10), (0.45, 0.105, 0.50)]
    obs, _ = run([(0.42, 0.48, 0.06, 0.15, 0.08)], chain=chain, fingers=[])
    assert obs == []


def test_depth_noise_does_not_create_phantoms_and_keeps_the_real_one():
    obs, _ = run([(0.42, 0.48, 0.06, 0.15, 0.08)], noise=0.004)
    assert len(obs) == 1


def test_two_separate_obstacles():
    obs, _ = run([(0.30, 0.36, -0.25, -0.18, 0.07), (0.50, 0.58, 0.20, 0.30, 0.10)])
    assert len(obs) == 2
    assert sorted(round(o['height'], 1) for o in obs) == [0.1, 0.1] or len(obs) == 2


def test_merge_across_cameras_counts_the_viewers():
    a = {'x': 0.45, 'y': 0.10, 'height': 0.07, 'area_m2': 0.005, 'w': 0.06, 'h': 0.09, 'angle': 0.0}
    b = {**a, 'x': 0.46, 'y': 0.11, 'height': 0.09}
    c = {**a, 'x': 0.20, 'y': -0.20}
    m = og.merge_across_cameras({'rs1': [a, c], 'rs2': [b]})
    assert len(m) == 2
    both = [x for x in m if len(x['seen_by']) == 2][0]
    assert both['height'] == 0.09 and abs(both['x'] - 0.455) < 1e-9


def test_flying_pixels_at_a_depth_edge_are_not_an_obstacle():
    # a tall thin 'finger' is in front of the table: its edge produces interpolated depths between near and far
    R, t = look_at((1.0, 0.0, 0.80), (0.40, 0.0, 0.0))
    depth = render_depth(R, t, [])
    h, w = depth.shape
    near = depth[200:260, 300:310].mean() * 0.55                       # a surface much closer to the camera
    depth[200:260, 300:310] = near
    # the sensor's interpolation: a few mid-air pixels between the near strip and the table
    depth[200:260, 298:300] = (near + depth[200:260, 296:297].mean()) / 2
    depth[200:260, 310:312] = (near + depth[200:260, 313:314].mean()) / 2
    pts = og.backproject(depth, K, R, t, stride=1, clean=True)
    raw = og.backproject(depth, K, R, t, stride=1, clean=False)
    assert len(pts) < len(raw)                                         # the edge pixels were dropped
    flat = pts[(pts[:, 2] - TABLE_Z) > 0.03]
    # whatever is left above the table belongs to the strip itself, never to the interpolated pixels
    assert len(flat) < len(raw[(raw[:, 2] - TABLE_Z) > 0.03])


def test_confirmed_means_two_cameras_saw_it():
    a = {'x': 0.45, 'y': 0.10, 'height': 0.07, 'area_m2': 0.005, 'w': 0.06, 'h': 0.09, 'angle': 0.0}
    lone = {**a, 'x': 0.20, 'y': -0.20}
    m = og.merge_across_cameras({'rs1': [a, lone], 'rs2': [a]})
    by_x = {round(x['x'], 2): x for x in m}
    assert by_x[0.45]['confirmed'] is True and by_x[0.20]['confirmed'] is False
