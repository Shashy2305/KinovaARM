#!/usr/bin/env python3
"""
detector_benchmark.py -- how often is each candidate detector RIGHT about the object the robot actually picked, and does a second opinion help?

Ground truth: the self-labelled wrist frames (~/.ros/training_data/samples): a pick only becomes a sample when the object ended up in the gripper, so
the label is the command's label and (u, v) is the pixel where the object was expected. For each model we run the frame at a low confidence floor, take
the detections whose box contains (u, v) (or whose centre is within 120 px) and read the label of the most confident one.

    source /opt/ros/humble/setup.bash && source /mnt/ros_workspace/venv/bin/activate
    python3 scripts/detector_benchmark.py [--models yolov8m-seg rtdetr-l yolo11m ...] [--imgsz 640] [--out docs/data/detector_benchmark.json]

Weights not present are downloaded by Ultralytics into /mnt/ros_workspace/models/candidates. Read-only: touches no ROS node.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

SAMPLES = os.path.expanduser('~/.ros/training_data/samples')
CAND_DIR = '/mnt/ros_workspace/models/candidates'
SCENE = {'bottle', 'cup', 'bowl', 'mouse', 'cell phone', 'remote', 'book', 'scissors', 'vase'}
NEAR_PX = 120


def load_samples():
    out = []
    for js in sorted(glob.glob(os.path.join(SAMPLES, '*.json'))):
        m = json.load(open(js))
        img = js[:-5] + '.jpg'
        if os.path.exists(img) and m.get('u') is not None and m.get('label') in SCENE:
            out.append((img, m))
    return out


def load_model(name):
    os.chdir(CAND_DIR)
    from ultralytics import RTDETR, YOLO, YOLOWorld
    if name.startswith('rtdetr'):
        return RTDETR(name + '.pt'), None
    if 'world' in name:
        mdl = YOLOWorld(name + '.pt')
        mdl.set_classes(['cup', 'mug', 'computer mouse', 'bottle', 'bowl', 'cell phone', 'remote control', 'book', 'scissors', 'vase'])
        return mdl, {'mug': 'cup', 'computer mouse': 'mouse', 'remote control': 'remote'}
    path = name + '.pt'
    if name == 'yolov8m-seg':
        path = '/mnt/ros_workspace/models/yolov8m-seg.pt'
    return YOLO(path), None


def at_pixel(result, names, u, v, rename):
    """[(label, conf)] of the detections around (u, v), most confident first."""
    out = []
    for b in result.boxes:
        x1, y1, x2, y2 = [float(t) for t in b.xyxy[0]]
        label = names[int(b.cls)]
        label = (rename or {}).get(label, label)
        inside = x1 <= u <= x2 and y1 <= v <= y2
        near = np.hypot((x1 + x2) / 2 - u, (y1 + y2) / 2 - v) < NEAR_PX
        if inside or near:
            out.append((label, float(b.conf)))
    return sorted(out, key=lambda t: -t[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['yolov8m-seg', 'rtdetr-l', 'yolo11m', 'yolo12m', 'yolov10m', 'yolov9c'])
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    samples = load_samples()
    print(f'{len(samples)} labelled wrist frames: {dict(Counter(m["label"] for _, m in samples))}')
    results = {}
    for name in args.models:
        try:
            model, rename = load_model(name)
        except Exception as e:
            print(f'{name}: could not load ({str(e)[:90]})'); continue
        rows, t_ms = [], []
        for img, m in samples:
            frame = cv2.imread(img)
            t0 = time.time()
            r = model(frame, conf=0.05, imgsz=args.imgsz, verbose=False)[0]
            t_ms.append(1000 * (time.time() - t0))
            cands = at_pixel(r, model.names, m['u'], m['v'], rename)
            top = cands[0] if cands else (None, 0.0)
            # a label counts as "seen" if ANY candidate around the pixel carries it (the arm matches by position and class, not only the top one)
            rows.append({'img': os.path.basename(img), 'truth': m['label'], 'top': top[0], 'conf': round(top[1], 2),
                         'any': sorted({c[0] for c in cands}), 'truth_conf': round(max([c[1] for c in cands if c[0] == m['label']] or [0.0]), 2)})
        t_ms = t_ms[2:] or t_ms
        top_ok = sum(r['top'] == r['truth'] for r in rows)
        any_ok = sum(r['truth'] in r['any'] for r in rows)
        miss = sum(r['top'] is None for r in rows)
        per = defaultdict(lambda: [0, 0])
        for r in rows:
            per[r['truth']][1] += 1
            per[r['truth']][0] += r['top'] == r['truth']
        conf = Counter((r['truth'], r['top']) for r in rows if r['top'] != r['truth'])
        results[name] = {'rows': rows, 'top1': top_ok, 'any': any_ok, 'missed': miss, 'ms': round(float(np.median(t_ms)), 1)}
        print(f'\n{name}: top-1 correct {top_ok}/{len(rows)}, truth among the candidates {any_ok}/{len(rows)}, nothing found {miss}, {np.median(t_ms):.0f} ms/frame')
        print('   per object:', {k: f'{a}/{b}' for k, (a, b) in per.items()})
        print('   confusions (truth -> called):', {f'{t}->{c}': n for (t, c), n in conf.most_common(8)})
        del model
    names = list(results)
    if 'yolov8m-seg' in results and len(names) > 1:
        print('\n=== second opinion vs the current detector (yolov8m-seg), top-1 label at the pixel ===')
        base = results['yolov8m-seg']['rows']
        for n in names:
            if n == 'yolov8m-seg':
                continue
            other = results[n]['rows']
            both = sum(a['top'] == a['truth'] and b['top'] == b['truth'] for a, b in zip(base, other))
            only_a = sum(a['top'] == a['truth'] and b['top'] != b['truth'] for a, b in zip(base, other))
            only_b = sum(a['top'] != a['truth'] and b['top'] == b['truth'] for a, b in zip(base, other))
            neither = len(base) - both - only_a - only_b
            agree = [(a, b) for a, b in zip(base, other) if a['top'] is not None and a['top'] == b['top']]
            agree_ok = sum(a['top'] == a['truth'] for a, _ in agree)
            either = sum(a['truth'] in a['any'] or b['truth'] in b['any'] for a, b in zip(base, other))
            print(f'{n:12s} both right {both}, only yolov8 {only_a}, only {n} {only_b}, neither {neither} | they agree on {len(agree)}/{len(base)} frames and are right on {agree_ok} of those'
                  f' | truth in either one\'s candidates {either}/{len(base)}')
    if args.out:
        json.dump(results, open(args.out, 'w'), indent=1)
        print('wrote', args.out)


if __name__ == '__main__':
    main()
