"""
Self-labelled training data for the object detector (pure Python + OpenCV, no ROS).

The detectors mislabel objects seen from straight above (a water bottle is "sports ball" or "bowl", a mouse is "cup"), and fixing
that needs training images from exactly that viewpoint, correctly labelled. The robot already KNOWS the label: it was asked to
pick up "the bottle", the scene fusion (two static cameras) agreed, and the pick ended with the object in the gripper.

Flow, per pick attempt:
  save     right before the descent the wrist detector stores the current frame and what the detector saw      -> pending/
  commit   the pick ended with the object held (finger reading in range): keep it                              -> samples/
  discard  the pick failed or was refused: the label may be wrong (a different object, a ghost): delete it
So only picks that succeeded become training data. export_yolo() turns the kept samples into a YOLOv8 segmentation dataset
(polygon labels). Where the detector found nothing at all (the bottle from above!) the mask is made with GrabCut seeded at
the pixel where the object was expected, and the sample is marked `auto-mask` so it can be reviewed.

Directory (default ~/.ros/training_data): pending/<id>.jpg|.json, samples/<id>.jpg|.json
"""
import json
import os
import shutil

import cv2
import numpy as np

DEFAULT_DIR = os.path.expanduser('~/.ros/training_data')
CLASS_NAMES = ['bottle', 'cup', 'bowl', 'mouse', 'cell phone', 'remote', 'book', 'scissors', 'vase']
SEED_BOX_PX = 220                 # GrabCut seed box around the expected pixel when no detection polygon is near it
NEAR_DETECTION_PX = 110           # a detection whose box centre is this close to the expected pixel is the object


def _paths(directory, kind, sample_id):
    return (os.path.join(directory, kind, f'{sample_id}.jpg'), os.path.join(directory, kind, f'{sample_id}.json'))


def save_pending(directory, sample_id, frame_bgr, meta):
    """Store a frame and its metadata as pending. meta: {'label', 'u', 'v', 'detections': [...], 'polygons': [...], ...}."""
    img, js = _paths(directory, 'pending', sample_id)
    os.makedirs(os.path.dirname(img), exist_ok=True)
    cv2.imwrite(img, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(js, 'w') as f:
        json.dump({'id': sample_id, **meta}, f)
    return img


def _move(directory, sample_id, src_kind, dst_kind):
    moved = False
    for a, b in zip(_paths(directory, src_kind, sample_id), _paths(directory, dst_kind, sample_id)):
        if os.path.exists(a):
            os.makedirs(os.path.dirname(b), exist_ok=True)
            shutil.move(a, b)
            moved = True
    return moved


def commit(directory, sample_id):
    """The pick succeeded: keep the pending sample. True if there was one."""
    return _move(directory, sample_id, 'pending', 'samples')


def discard(directory, sample_id):
    """The pick failed: delete the pending sample. True if there was one."""
    removed = False
    for p in _paths(directory, 'pending', sample_id):
        if os.path.exists(p):
            os.remove(p)
            removed = True
    return removed


def purge_old_pending(directory, max_age_s=3600.0, now=None):
    """Pending samples older than max_age_s never got a verdict (a crash): drop them."""
    import time
    now = time.time() if now is None else now
    d = os.path.join(directory, 'pending')
    n = 0
    if os.path.isdir(d):
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if now - os.path.getmtime(p) > max_age_s:
                os.remove(p)
                n += 1
    return n


def list_samples(directory):
    out = []
    d = os.path.join(directory, 'samples')
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if name.endswith('.json'):
            try:
                with open(os.path.join(d, name)) as f:
                    meta = json.load(f)
                out.append((os.path.join(d, name[:-5] + '.jpg'), meta))
            except (OSError, json.JSONDecodeError):
                continue
    return out


# ── mask for a sample ───────────────────────────────────────────────────────────────────────────────────────────────
def polygon_for(frame_bgr, meta):
    """(polygon [(x, y)...], source) for the labelled object: the detector's own mask polygon if a detection's box centre is near
    the expected pixel, else a GrabCut mask seeded at that pixel ('auto-mask'). (None, None) if neither works."""
    u, v = meta.get('u'), meta.get('v')
    if u is None or v is None:
        return None, None
    best = None
    for det, poly in zip(meta.get('detections', []), meta.get('polygons', [])):
        if not poly:
            continue
        d = float(np.hypot(det['u'] - u, det['v'] - v))
        if d <= NEAR_DETECTION_PX and (best is None or d < best[0]):
            best = (d, poly)
    if best is not None:
        return best[1], 'detector'
    h, w = frame_bgr.shape[:2]
    half = SEED_BOX_PX // 2
    box = (max(0, int(u) - half), max(0, int(v) - half), min(w - 1, int(u) + half), min(h - 1, int(v) + half))
    from thesis_robot import shape_analysis as sa
    mask = sa.refine_mask_grabcut(frame_bgr, box, None, pad=6)
    if mask is None:
        return None, None
    cnt = sa.largest_contour(mask)
    if cnt is None or len(cnt) < 8:
        return None, None
    peri = cv2.arcLength(cnt.reshape(-1, 1, 2), True)
    approx = cv2.approxPolyDP(cnt.reshape(-1, 1, 2), 0.004 * peri, True).reshape(-1, 2)
    return approx.tolist(), 'auto-mask'


def export_yolo(directory, out_dir, val_fraction=0.2, min_per_class=0):
    """Write a YOLOv8 segmentation dataset from the kept samples. Returns {'counts': {class: n}, 'auto_mask': n, 'skipped': n, 'path': yaml}."""
    samples = list_samples(directory)
    counts, auto, skipped = {}, 0, 0
    rows = []
    for img_path, meta in samples:
        label = meta.get('label')
        if label not in CLASS_NAMES or not os.path.exists(img_path):
            skipped += 1
            continue
        frame = cv2.imread(img_path)
        poly, source = polygon_for(frame, meta)
        if poly is None or len(poly) < 3:
            skipped += 1
            continue
        rows.append((img_path, frame.shape[1], frame.shape[0], label, poly, source))
        counts[label] = counts.get(label, 0) + 1
        auto += int(source == 'auto-mask')
    rows = [r for r in rows if counts[r[3]] >= min_per_class]
    for split in ('train', 'val'):
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)
    for i, (img_path, w, h, label, poly, source) in enumerate(rows):
        split = 'val' if (i % max(2, int(round(1 / max(val_fraction, 1e-6))))) == 0 else 'train'
        stem = os.path.splitext(os.path.basename(img_path))[0]
        shutil.copy(img_path, os.path.join(out_dir, 'images', split, stem + '.jpg'))
        coords = ' '.join(f'{min(max(x / w, 0.0), 1.0):.5f} {min(max(y / h, 0.0), 1.0):.5f}' for x, y in poly)
        with open(os.path.join(out_dir, 'labels', split, stem + '.txt'), 'w') as f:
            f.write(f'{CLASS_NAMES.index(label)} {coords}\n')
    yaml_path = os.path.join(out_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\nnames:\n')
        for i, n in enumerate(CLASS_NAMES):
            f.write(f'  {i}: {n}\n')
    return {'counts': counts, 'auto_mask': auto, 'skipped': skipped, 'path': yaml_path}
