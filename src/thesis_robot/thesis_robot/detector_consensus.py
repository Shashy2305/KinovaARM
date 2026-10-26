"""
Combining independent opinions about WHAT an object is (pure python + numpy, no ROS): the logic of the shadow-mode second detector (detector_b_node.py), see docs/DETECTOR_SECOND_OPINION.md.

Opinion A is the running detector (YOLOv8), B a different architecture (RT-DETR), C a verifier trained on the robot's own verified picks (CLIP image features, nearest neighbour). Every COCO model
confuses the same look-alikes,
(a cup seen from above is "cup", "vase", "bowl" or "mouse" depending on the frame) so opinions are compared by CONFUSION GROUP, not by exact label: two opinions agree when
they name the same group. Where the opinions do not agree the verdict is 'uncertain', which is the property the system lacks today.
"""
import numpy as np

GROUPS = {
    'cup': 'container', 'mug': 'container', 'vase': 'container', 'bowl': 'container', 'wine glass': 'container',
    'mouse': 'small_flat', 'remote': 'small_flat', 'cell phone': 'small_flat', 'book': 'small_flat',
    'bottle': 'bottle',
    'scissors': 'scissors',
}


def group_of(label):
    if label is None:
        return None
    return GROUPS.get(label, label)             # an unknown label is its own group: it agrees only with itself


def consensus(a=None, b=None, c=None):
    """a, b, c: a label string or None (no opinion). Returns {'verdict', 'group', 'label', 'n', 'votes'}.
    verdict: 'agree' (every opinion present names the same group, at least two), 'majority' (two of three agree, one does not), 'uncertain' (at least two opinions, no two agree),
    'single' (only one opinion), 'none'. label: C's label if C is in the winning group, else A's, else B's."""
    ops = {k: v for k, v in (('a', a), ('b', b), ('c', c)) if v is not None}
    votes = {k: group_of(v) for k, v in ops.items()}
    out = {'verdict': 'none', 'group': None, 'label': None, 'n': len(ops), 'votes': votes}
    if not ops:
        return out
    if len(ops) == 1:
        k, v = next(iter(ops.items()))
        out.update(verdict='single', group=votes[k], label=v)
        return out
    counts = {}
    for g in votes.values():
        counts[g] = counts.get(g, 0) + 1
    best_g, best_n = max(counts.items(), key=lambda kv: kv[1])
    if best_n == len(ops):
        out['verdict'] = 'agree'
    elif best_n >= 2:
        out['verdict'] = 'majority'
    else:
        out['verdict'] = 'uncertain'
        return out
    out['group'] = best_g
    for k in ('c', 'a', 'b'):
        if k in ops and votes[k] == best_g:
            out['label'] = ops[k]
            break
    return out


def project_to_pixel(x, y, z, fx, fy, cx, cy):
    """pixel (u, v) of a camera-frame point; None if it is not in front of the camera."""
    if z is None or z <= 0:
        return None
    return fx * x / z + cx, fy * y / z + cy


def box_containing(point, boxes, min_conf=0.0, max_offset=0.35):
    """The index of the box (x1, y1, x2, y2, conf) that IS the object at this pixel, or None: the pixel must lie inside the box AND near its centre (within max_offset of the box's size along
    each axis), and the SMALLEST such box wins. A big box that merely encloses the pixel (a chair around a mouse on it, the table under everything) is not an opinion about this object."""
    u, v = point
    best, best_area = None, None
    for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
        if conf < min_conf or not (x1 <= u <= x2 and y1 <= v <= y2):
            continue
        w, h = x2 - x1, y2 - y1
        if abs(u - (x1 + x2) / 2) > max_offset * w or abs(v - (y1 + y2) / 2) > max_offset * h:
            continue
        area = w * h
        if best is None or area < best_area:
            best, best_area = i, area
    return best


def near_any(point, points, px=60.0):
    """True if the pixel is within px of any of the pixels (is A's object the same as this B box?)."""
    return any((point[0] - u) ** 2 + (point[1] - v) ** 2 <= px * px for u, v in points)


def knn_label(feat, bank_feats, bank_labels, min_similarity=0.80):
    """Nearest neighbour by cosine similarity among the robot's verified samples. Returns (label or None, similarity). None = the object is not like anything verified (the verifier abstains
    instead of guessing: 'unknown object')."""
    if len(bank_labels) == 0:
        return None, 0.0
    f = np.asarray(feat, float)
    f = f / (np.linalg.norm(f) + 1e-12)
    B = np.asarray(bank_feats, float)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    sims = B @ f
    i = int(np.argmax(sims))
    sim = float(sims[i])
    return (bank_labels[i] if sim >= min_similarity else None), sim
