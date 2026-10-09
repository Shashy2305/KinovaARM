#!/usr/bin/env python3
"""Turn the self-labelled wrist frames collected during picks into a YOLOv8 segmentation dataset.

    python3 scripts/export_training_dataset.py --out ~/.ros/yolo_dataset [--min-per-class 10]
    # then, once every class you care about has a few dozen samples (and you have looked at the 'auto-mask' ones):
    yolo segment train data=~/.ros/yolo_dataset/data.yaml model=/mnt/ros_workspace/models/yolov8m-seg.pt epochs=60 imgsz=640

Samples live in ~/.ros/training_data/samples (kept only for picks that ended with the object held). Where the detector found nothing
at the object (a bottle from above) the mask comes from GrabCut seeded at the expected pixel and is counted as 'auto-mask'.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'thesis_robot'))
from thesis_robot import training_samples as ts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=ts.DEFAULT_DIR)
ap.add_argument('--out', default=os.path.expanduser('~/.ros/yolo_dataset'))
ap.add_argument('--min-per-class', type=int, default=0)
args = ap.parse_args()
n = len(ts.list_samples(args.src))
print(f'{n} kept samples in {args.src}')
if n == 0:
    sys.exit(0)
info = ts.export_yolo(args.src, args.out, min_per_class=args.min_per_class)
print('samples per class:', info['counts'])
print(f"auto-mask (GrabCut, review these): {info['auto_mask']}   skipped (unknown class / no mask): {info['skipped']}")
print('dataset:', info['path'])
