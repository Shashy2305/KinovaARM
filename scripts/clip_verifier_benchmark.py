#!/usr/bin/env python3
"""
clip_verifier_benchmark.py -- can an image-text model (CLIP, trained on web captions, not on COCO boxes) name the object the robot actually picked,
from a crop of the wrist frame? Uses the same 21+ self-labelled wrist frames as detector_benchmark.py. A candidate for the "verifier" in a two-opinion detector.

    source /opt/ros/humble/setup.bash && source /mnt/ros_workspace/venv/bin/activate
    PYTHONPATH=/mnt/ros_workspace/venv_extra python3 scripts/clip_verifier_benchmark.py [--model ViT-B-32 --pretrained laion2b_s34b_b79k]
(open_clip lives in /mnt/ros_workspace/venv_extra so that the production venv is untouched.)
"""
import argparse
import glob
import json
import os
import time
from collections import Counter

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

SAMPLES = os.path.expanduser('~/.ros/training_data/samples')
CLASSES = {  # our label -> prompts (the objects are seen mostly from above at the wrist, from the side by the fixed cameras)
    'cup': ['a top-down photo of a coffee mug with a handle', 'a photo of a mug with a handle', 'a coffee cup with a handle seen from above'],
    'mouse': ['a photo of a computer mouse', 'a top-down photo of a computer mouse'],
    'bottle': ['a photo of a plastic water bottle', 'a top-down photo of a water bottle'],
    'bowl': ['a top-down photo of a round bowl with no handle', 'a photo of a bowl without a handle'],
    'remote': ['a photo of a remote control', 'a photo of a game controller'],
    'cell phone': ['a photo of a cell phone', 'a photo of a smartphone'],
    'book': ['a photo of a book', 'a photo of a cardboard box'],
    'scissors': ['a photo of a pair of scissors'],
    'vase': ['a photo of a vase', 'a photo of a flower vase'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ViT-B-32')
    ap.add_argument('--pretrained', default='laion2b_s34b_b79k')
    ap.add_argument('--crop', type=int, default=280, help='side of the square crop around the expected pixel (px of the 1280x720 wrist frame)')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, pre = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=dev)
    tok = open_clip.get_tokenizer(args.model)
    model.eval()
    labels = list(CLASSES)
    with torch.no_grad():
        feats = []
        for lab in labels:
            t = model.encode_text(tok(CLASSES[lab]).to(dev))
            t = t / t.norm(dim=-1, keepdim=True)
            m = t.mean(0)
            feats.append(m / m.norm())
        T = torch.stack(feats)
    rows = []
    for js in sorted(glob.glob(os.path.join(SAMPLES, '*.json'))):
        m = json.load(open(js))
        img = cv2.imread(js[:-5] + '.jpg')
        if img is None or m.get('label') not in CLASSES or m.get('u') is None:
            continue
        h, w = img.shape[:2]
        half = args.crop // 2
        u, v = int(m['u']), int(m['v'])
        crop = img[max(0, v - half):min(h, v + half), max(0, u - half):min(w, u + half)]
        x = pre(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(dev)
        t0 = time.time()
        with torch.no_grad():
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100.0 * f @ T.T).softmax(dim=-1)[0].cpu().numpy()
        ms = 1000 * (time.time() - t0)
        order = np.argsort(-p)
        rows.append({'truth': m['label'], 'top': labels[order[0]], 'p': round(float(p[order[0]]), 2), 'second': labels[order[1]], 'p_truth': round(float(p[labels.index(m['label'])]), 2), 'ms': ms})
    ok = sum(r['top'] == r['truth'] for r in rows)
    top2 = sum(r['truth'] in (r['top'], r['second']) for r in rows)
    print(f'{args.model}/{args.pretrained}: top-1 {ok}/{len(rows)}, truth in the top 2 {top2}/{len(rows)}, median {np.median([r["ms"] for r in rows[1:]]):.1f} ms per crop')
    per = {}
    for r in rows:
        a = per.setdefault(r['truth'], [0, 0]); a[1] += 1; a[0] += r['top'] == r['truth']
    print('   per object:', {k: f'{a}/{b}' for k, (a, b) in per.items()})
    print('   confusions (truth -> called):', dict(Counter(f"{r['truth']}->{r['top']}" for r in rows if r['top'] != r['truth']).most_common(8)))
    print('   mean confidence when right %.2f, when wrong %.2f' % (np.mean([r['p'] for r in rows if r['top'] == r['truth']] or [0]), np.mean([r['p'] for r in rows if r['top'] != r['truth']] or [0])))
    json.dump(rows, open('/tmp/s4/clip_rows.json', 'w'))


if __name__ == '__main__':
    main()
