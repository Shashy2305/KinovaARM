"""2026-11-05: a cup's handle pulls the detection box's centre toward itself; the round body's centre is where the pads of a top-down grasp must close."""
import math
import time

import numpy as np
import pytest

from thesis_robot import wrist_servo as ws

from test_pick_place_flow import node  # noqa: F401


def circle_with_handle(cx=100, cy=100, r=40, handle=(140, 95, 178, 106)):
    t = np.linspace(0, 2 * math.pi, 90, endpoint=False)
    pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in t]
    x0, y0, x1, y1 = handle
    return pts, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def raster(*polys, size=(240, 240)):
    import cv2
    m = np.zeros(size, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.array(p)).astype(np.int32)], 255)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=len).reshape(-1, 2).tolist()


def test_the_body_centre_ignores_a_handle_the_box_centre_does_not():
    outline = raster(*circle_with_handle())                                              # a round body of radius 40 at (100, 100) with a handle sticking out 38 px to the right
    xs, ys = zip(*outline)
    box_centre = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    assert box_centre[0] - 100 > 8                                                       # the box is pulled toward the handle
    u, v, r = ws.body_centre(outline)
    assert abs(u - 100) < 3 and abs(v - 100) < 3 and abs(r - 40) < 3                     # the thickest part is the body


def test_a_round_object_has_the_same_body_and_box_centre_and_bad_outlines_give_none():
    outline = raster(circle_with_handle(handle=(0, 0, 1, 1))[0])
    u, v, r = ws.body_centre(outline)
    assert abs(u - 100) < 2 and abs(v - 100) < 2
    assert ws.body_centre([(1, 1), (2, 2)]) is None and ws.body_centre([]) is None and ws.body_centre(None) is None
    assert ws.body_centre([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]) is None     # a line: no thickness


def setup_detection(node, monkeypatch, u, v, bu, bv, age=0.0, label='cup'):
    node.dry_run, node.moveit2 = False, object()
    node._wrist_K = np.eye(3)
    det = {'label': label, 'confidence': 0.9, 'u': u, 'v': v}
    if bu is not None:
        det.update({'bu': bu, 'bv': bv})
    node._wrist_dets = (time.monotonic() - age, {'frame_id': 'wrist', 'stamp': None, 'detections': [det]})
    monkeypatch.setattr(node, '_tf_pose', lambda target, source, stamp=None: (np.eye(3), [0.0, 0.0, 0.0]))
    monkeypatch.setattr(node, '_pad_midpoint_xy', lambda default=None: (0.100, 0.100))
    monkeypatch.setattr(ws, 'pixel_to_plane_xy', lambda u_, v_, K, R, t, z: (u_ * 0.001, v_ * 0.001))      # 1 px = 1 mm
    monkeypatch.setattr(ws, 'center_plane_height', lambda label, top: 0.05)


GEOM = {'table_top_z': -0.0125}


def test_the_controller_shifts_the_fingers_from_the_box_centre_to_the_body_centre(node, monkeypatch):
    setup_detection(node, monkeypatch, u=100, v=100, bu=80, bv=100)                      # the body centre is 20 mm to the side of the box centre
    shift = node._body_centre_shift('cup', GEOM)
    assert shift == pytest.approx((-0.020, 0.0), abs=1e-9)


def test_no_shift_for_a_missing_old_tiny_or_implausible_body_centre(node, monkeypatch):
    setup_detection(node, monkeypatch, 100, 100, None, None)
    assert node._body_centre_shift('cup', GEOM) is None                                  # the detector gave none
    setup_detection(node, monkeypatch, 100, 100, 80, 100, age=5.0)
    assert node._body_centre_shift('cup', GEOM) is None                                  # an old frame (the arm has moved)
    setup_detection(node, monkeypatch, 100, 100, 101, 100)
    assert node._body_centre_shift('cup', GEOM) is None                                  # 1 mm: nothing to correct
    setup_detection(node, monkeypatch, 100, 100, 40, 100)
    assert node._body_centre_shift('cup', GEOM) is None                                  # 6 cm: not a handle's pull, something else
    node.dry_run = True
    setup_detection(node, monkeypatch, 100, 100, 80, 100)
    node.dry_run = True
    assert node._body_centre_shift('cup', GEOM) is None                                  # a dry run never looks
    node.dry_run = False
