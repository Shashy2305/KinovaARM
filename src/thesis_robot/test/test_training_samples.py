import json
import os
import time

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
from test_shape_analysis import background, capsule, draw_poly  # noqa: E402
from thesis_robot import training_samples as ts  # noqa: E402


def frame_with_object(cx=240, cy=180, angle=30):
    img = background()
    poly = capsule(cx, cy, 140, 60, angle)
    draw_poly(img, poly)
    return img, poly


def meta_for(label, u, v, poly=None):
    dets, polys = [], []
    if poly is not None:
        dets = [{'label': 'cup', 'confidence': 0.3, 'u': int(poly[:, 0].mean()), 'v': int(poly[:, 1].mean())}]
        polys = [poly.round(1).tolist()]
    return {'label': label, 'u': u, 'v': v, 'detections': dets, 'polygons': polys}


def test_save_commit_and_discard(tmp_path):
    d = str(tmp_path)
    img, poly = frame_with_object()
    ts.save_pending(d, 'a1', img, meta_for('mouse', 240, 180, poly))
    ts.save_pending(d, 'a2', img, meta_for('mouse', 240, 180, poly))
    assert os.path.exists(tmp_path / 'pending' / 'a1.jpg') and os.path.exists(tmp_path / 'pending' / 'a2.json')
    assert ts.commit(d, 'a1') is True and ts.discard(d, 'a2') is True
    assert os.path.exists(tmp_path / 'samples' / 'a1.jpg') and os.path.exists(tmp_path / 'samples' / 'a1.json')
    assert not os.path.exists(tmp_path / 'pending' / 'a1.jpg') and not os.path.exists(tmp_path / 'pending' / 'a2.jpg')
    assert ts.commit(d, 'nope') is False and ts.discard(d, 'nope') is False
    assert [m['id'] for _, m in ts.list_samples(d)] == ['a1']


def test_old_pending_samples_are_purged(tmp_path):
    d = str(tmp_path)
    img, poly = frame_with_object()
    ts.save_pending(d, 'old', img, meta_for('cup', 240, 180, poly))
    assert ts.purge_old_pending(d, max_age_s=3600, now=time.time() + 7200) == 2        # the jpg and the json
    assert ts.purge_old_pending(d, max_age_s=3600) == 0


def test_the_detector_polygon_is_used_when_a_detection_is_near_the_expected_pixel():
    img, poly = frame_with_object()
    p, src = ts.polygon_for(img, meta_for('mouse', 245, 185, poly))
    assert src == 'detector' and len(p) == len(poly)


def test_grabcut_makes_a_mask_when_the_detector_saw_nothing_there():
    img, poly = frame_with_object()
    p, src = ts.polygon_for(img, meta_for('bottle', 240, 180, None))
    assert src == 'auto-mask' and len(p) >= 6
    area = cv2.contourArea(np.array(p, np.float32).reshape(-1, 1, 2))
    true_area = cv2.contourArea(poly.astype(np.float32).reshape(-1, 1, 2))
    assert 0.7 * true_area < area < 1.3 * true_area


def test_a_far_detection_is_not_taken_for_the_object():
    img, poly = frame_with_object(cx=120, cy=100)
    other = capsule(380, 280, 100, 50, 0)
    meta = meta_for('mouse', 120, 100, other)                                           # the only detection is elsewhere
    p, src = ts.polygon_for(img, meta)
    assert src == 'auto-mask'


def test_export_writes_a_yolo_segmentation_dataset(tmp_path):
    d, out = str(tmp_path / 'td'), str(tmp_path / 'ds')
    for i, (label, ang) in enumerate([('mouse', 20), ('mouse', 60), ('bottle', 100), ('cup', 140), ('mouse', 10)]):
        img, poly = frame_with_object(angle=ang)
        ts.save_pending(d, f's{i}', img, meta_for(label, 240, 180, poly if label != 'bottle' else None))
        ts.commit(d, f's{i}')
    img, poly = frame_with_object()
    ts.save_pending(d, 'weird', img, meta_for('teapot', 240, 180, poly))                # not a class we train
    ts.commit(d, 'weird')
    info = ts.export_yolo(d, out)
    assert info['counts'] == {'mouse': 3, 'bottle': 1, 'cup': 1} and info['skipped'] == 1 and info['auto_mask'] == 1
    labels = [os.path.join(r, f) for r, _, fs in os.walk(os.path.join(out, 'labels')) for f in fs]
    assert len(labels) == 5
    line = open(labels[0]).read().split()
    assert int(line[0]) in (ts.CLASS_NAMES.index(c) for c in ('mouse', 'bottle', 'cup'))
    coords = [float(v) for v in line[1:]]
    assert len(coords) % 2 == 0 and len(coords) >= 6 and all(0.0 <= c <= 1.0 for c in coords)
    yaml_text = open(info['path']).read()
    assert 'train: images/train' in yaml_text and '0: bottle' in yaml_text
    assert any(os.listdir(os.path.join(out, 'images', s)) for s in ('train', 'val'))
    assert os.listdir(os.path.join(out, 'images', 'val'))                               # a validation split exists


def test_min_per_class_drops_rare_classes(tmp_path):
    d, out = str(tmp_path / 'td'), str(tmp_path / 'ds')
    for i, label in enumerate(['mouse', 'mouse', 'cup']):
        img, poly = frame_with_object(angle=20 * (i + 1))
        ts.save_pending(d, f's{i}', img, meta_for(label, 240, 180, poly))
        ts.commit(d, f's{i}')
    ts.export_yolo(d, out, min_per_class=2)
    names = [f for r, _, fs in os.walk(os.path.join(out, 'labels')) for f in fs]
    assert len(names) == 2
