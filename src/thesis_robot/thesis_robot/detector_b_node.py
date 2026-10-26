#!/usr/bin/env python3
"""
Shadow-mode second opinion for the object detector (docs/DETECTOR_SECOND_OPINION.md). It NEVER changes behaviour: it publishes /detector_b_report and writes a log; nothing subscribes to it.

For every object the running detector (opinion A, YOLOv8) reports on a camera it asks two independent opinions:
  B  RT-DETR-L on the same colour image (a different architecture, a different label prior): the label of the smallest box that contains A's pixel
  C  (wrist camera only) CLIP image features of the crop, nearest neighbour among the robot's own VERIFIED picks (~/.ros/training_data/samples); it abstains when nothing verified looks like the crop
and combines them by confusion group (detector_consensus.py): agree / majority / uncertain. Objects the second detector sees in a class the robot works with but that A did not report are logged
as 'only_b' (candidate misses of A).

Log: ~/.ros/detector_b/log_<date>.jsonl (every non-agreeing object, with a crop in crops/), stats.json (counts per camera/verdict, rewritten every minute). Summary: scripts/detector_b_report.py.
Default OFF (dashboard process 'detector_b'); ~1 GB of GPU memory. open_clip is in /mnt/ros_workspace/venv_extra (PYTHONPATH); without it the node still runs as A-vs-B.
"""
import glob
import json
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from thesis_robot import detector_consensus as dc

CAMERAS = {
    'wrist': {'image': '/camera/color/image_raw', 'info': '/camera/color/camera_info', 'dets': '/wrist_pixel_detections', 'kind': 'pixel', 'clip': True},
    'realsense': {'image': '/global_camera/global_camera/color/image_raw', 'info': '/global_camera/global_camera/color/camera_info', 'dets': '/detections_side', 'kind': 'cam3d'},
    'realsense2': {'image': '/global_camera_2/global_camera_2/color/image_raw', 'info': '/global_camera_2/global_camera_2/color/camera_info', 'dets': '/detections_realsense2', 'kind': 'cam3d'},
    'realsense3': {'image': '/global_camera_3/global_camera_3/color/image_raw', 'info': '/global_camera_3/global_camera_3/color/camera_info', 'dets': '/detections_realsense3', 'kind': 'cam3d'},
}
WORK_CLASSES = {'bottle', 'cup', 'bowl', 'cell phone', 'remote', 'book', 'scissors', 'vase', 'mouse'}
SAMPLES_DIR = os.path.expanduser('~/.ros/training_data/samples')
LOG_DIR = os.path.expanduser('~/.ros/detector_b')
CROP_PX = 280
MAX_CROPS = 400
LOG_MIN_PERIOD_S = 2.0
DEDUPE_S = 30.0


class DetectorB(Node):
    def __init__(self):
        super().__init__('detector_b')
        self.declare_parameter('cameras', ['wrist', 'realsense'])
        self.declare_parameter('rate_hz', 3.0)
        self.declare_parameter('rtdetr_path', '/mnt/ros_workspace/models/candidates/rtdetr-l.pt')
        self.declare_parameter('b_min_conf', 0.25)
        self.declare_parameter('only_b_min_conf', 0.55)
        self.declare_parameter('clip_model', 'ViT-B-32')
        self.declare_parameter('clip_pretrained', 'laion2b_s34b_b79k')
        self.declare_parameter('clip_min_similarity', 0.80)
        self.declare_parameter('device', 'cuda')
        self.cams = [c for c in self.get_parameter('cameras').value if c in CAMERAS]
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self.image = {c: None for c in self.cams}               # (monotonic time, Image msg)
        self.K = {c: None for c in self.cams}
        self.dets = {c: None for c in self.cams}                 # (monotonic time, [(label, conf, u, v)])
        self.stats = {c: {} for c in self.cams}
        self._last_log = {c: 0.0 for c in self.cams}
        self._seen = {}                                        # signature of a logged disagreement -> time: the same object is not logged again for DEDUPE_S
        self._last_stats_write = time.monotonic()
        os.makedirs(os.path.join(LOG_DIR, 'crops'), exist_ok=True)

        from ultralytics import RTDETR
        self.rtdetr = RTDETR(self.get_parameter('rtdetr_path').value)
        self.device = self.get_parameter('device').value
        self.get_logger().info(f'RT-DETR loaded ({self.get_parameter("rtdetr_path").value}); cameras {self.cams}')
        self.clip = None
        self.bank_feats, self.bank_labels, self.bank_n = np.zeros((0, 512)), [], -1
        if any(CAMERAS[c].get('clip') for c in self.cams):
            self._load_clip()
            self._reload_bank()

        for c in self.cams:
            cfg = CAMERAS[c]
            self.create_subscription(Image, cfg['image'], lambda m, c=c: self._on_image(c, m), qos_profile_sensor_data)
            self.create_subscription(CameraInfo, cfg['info'], lambda m, c=c: self._on_info(c, m), 1)
            self.create_subscription(String, cfg['dets'], lambda m, c=c: self._on_dets(c, m), 5)
        self.report_pub = self.create_publisher(String, '/detector_b_report', 10)
        self.create_timer(1.0 / float(self.get_parameter('rate_hz').value), self._tick)
        self.create_timer(300.0, self._reload_bank)

    # ── inputs ────────────────────────────────────────────────────────
    def _on_image(self, cam, msg):
        self.image[cam] = (time.monotonic(), msg)

    def _on_info(self, cam, msg):
        self.K[cam] = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])      # fx, fy, cx, cy

    def _on_dets(self, cam, msg):
        try:
            raw = json.loads(msg.data)
        except ValueError:
            return
        out = []
        for d in raw if isinstance(raw, list) else []:
            label = d.get('label')
            conf = float(d.get('confidence', d.get('conf', 0.0)))
            if CAMERAS[cam]['kind'] == 'pixel':
                if d.get('u') is None:
                    continue
                out.append((label, conf, float(d['u']), float(d['v'])))
            else:
                K = self.K[cam]
                p = dc.project_to_pixel(d.get('cx_3d'), d.get('cy_3d'), d.get('cz_3d'), *K) if K else None
                if p is not None:
                    out.append((label, conf, p[0], p[1]))
        self.dets[cam] = (time.monotonic(), out)

    # ── opinion C: the robot's own verified picks ─────────────────────
    def _load_clip(self):
        try:
            import open_clip
            import torch
            self._torch = torch
            name, pre = self.get_parameter('clip_model').value, self.get_parameter('clip_pretrained').value
            self.clip, _, self.clip_pre = open_clip.create_model_and_transforms(name, pretrained=pre, device=self.device)
            self.clip.eval()
            self.get_logger().info(f'CLIP {name} loaded (opinion C on the wrist camera)')
        except Exception as e:
            self.clip = None
            self.get_logger().warn(f'opinion C disabled (open_clip not importable: {e}); run with PYTHONPATH=/mnt/ros_workspace/venv_extra')

    def _clip_feature(self, crop_bgr):
        from PIL import Image as PILImage
        x = self.clip_pre(PILImage.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            f = self.clip.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
        return f[0].cpu().numpy()

    def _crop(self, img, u, v, half=CROP_PX // 2):
        h, w = img.shape[:2]
        return img[max(0, int(v) - half):min(h, int(v) + half), max(0, int(u) - half):min(w, int(u) + half)]

    def _reload_bank(self):
        if self.clip is None:
            return
        files = sorted(glob.glob(os.path.join(SAMPLES_DIR, '*.json')))
        if len(files) == self.bank_n:
            return
        feats, labels = [], []
        for js in files:
            try:
                m = json.load(open(js))
                img = cv2.imread(js[:-5] + '.jpg')
                if img is None or m.get('u') is None or not m.get('label'):
                    continue
                feats.append(self._clip_feature(self._crop(img, m['u'], m['v'])))
                labels.append(m['label'])
            except Exception as e:
                self.get_logger().warn(f'bank sample {js}: {e}')
        self.bank_feats, self.bank_labels, self.bank_n = (np.array(feats) if feats else np.zeros((0, 512))), labels, len(files)
        self.get_logger().info(f'verified bank: {len(labels)} samples {sorted(set(labels))}')

    # ── the comparison ────────────────────────────────────────────────
    def _tick(self):
        now = time.monotonic()
        for cam in self.cams:
            im, dets = self.image[cam], self.dets[cam]
            if im is None or dets is None or now - im[0] > 0.7 or now - dets[0] > 1.5:
                continue
            try:
                self._compare(cam, im[1], dets[1])
            except Exception as e:                       # a shadow node must never become a problem for anything else
                self.get_logger().warn(f'{cam}: {e}', throttle_duration_sec=10.0)
        if now - self._last_stats_write > 60.0:
            self._write_stats()
            self._last_stats_write = now

    def _compare(self, cam, msg, a_dets):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        res = self.rtdetr(bgr, conf=float(self.get_parameter('b_min_conf').value), verbose=False, device=self.device)[0]
        names = res.names
        boxes = [(*map(float, xy), float(cf)) for xy, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy())]
        b_labels = [names[int(k)] for k in res.boxes.cls.cpu().numpy()]
        objects, used = [], set()
        for label, conf, u, v in a_dets:
            bi = dc.box_containing((u, v), boxes)
            b_label = b_labels[bi] if bi is not None else None
            if bi is not None:
                used.add(bi)
            c_label, c_sim = None, None
            if CAMERAS[cam].get('clip') and self.clip is not None and len(self.bank_labels):
                c_label, c_sim = dc.knn_label(self._clip_feature(self._crop(bgr, u, v)), self.bank_feats, self.bank_labels, float(self.get_parameter('clip_min_similarity').value))
            r = dc.consensus(label, b_label, c_label)
            objects.append({'u': round(u), 'v': round(v), 'a': {'label': label, 'conf': round(conf, 2)},
                            'b': None if bi is None else {'label': b_label, 'conf': round(boxes[bi][4], 2)},
                            'c': None if c_sim is None else {'label': c_label, 'sim': round(c_sim, 3)}, 'verdict': r['verdict'], 'group': r['group'], 'label': r['label']})
        only_b_conf = float(self.get_parameter('only_b_min_conf').value)
        a_points = [(u, v) for _l, _c, u, v in a_dets]
        for i, (box, lab) in enumerate(zip(boxes, b_labels)):
            centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            if i in used or lab not in WORK_CLASSES or box[4] < only_b_conf or dc.near_any(centre, a_points):
                continue                       # a work-class object the second detector sees and the first one did not report
            objects.append({'u': round(centre[0]), 'v': round(centre[1]), 'a': None, 'b': {'label': lab, 'conf': round(box[4], 2)}, 'c': None, 'verdict': 'only_b',
                            'group': dc.group_of(lab), 'label': lab})
        if not objects:
            return
        for o in objects:
            self.stats[cam][o['verdict']] = self.stats[cam].get(o['verdict'], 0) + 1
        self.report_pub.publish(String(data=json.dumps({'camera': cam, 'stamp': time.time(), 'objects': objects})))
        now = time.monotonic()
        bad = []
        for o in objects:
            if o['verdict'] in ('agree', 'single'):
                continue
            sig = (cam, o['verdict'], (o['a'] or {}).get('label'), (o['b'] or {}).get('label'), round(o['u'] / 60), round(o['v'] / 60))
            if now - self._seen.get(sig, -1e9) > DEDUPE_S:
                self._seen[sig] = now
                bad.append(o)
        if len(self._seen) > 500:
            self._seen = {k: t for k, t in self._seen.items() if now - t < DEDUPE_S}
        if bad and now - self._last_log[cam] > LOG_MIN_PERIOD_S:
            self._last_log[cam] = time.monotonic()
            self._log(cam, bgr, bad)

    def _log(self, cam, bgr, objects):
        stamp = time.strftime('%Y%m%d_%H%M%S')
        crop_name = f'{stamp}_{cam}.jpg'
        o = objects[0]
        cv2.imwrite(os.path.join(LOG_DIR, 'crops', crop_name), self._crop(bgr, o['u'], o['v'], half=200), [cv2.IMWRITE_JPEG_QUALITY, 85])
        with open(os.path.join(LOG_DIR, f'log_{time.strftime("%Y%m%d")}.jsonl'), 'a') as f:
            f.write(json.dumps({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'camera': cam, 'crop': crop_name, 'objects': objects}) + '\n')
        crops = sorted(glob.glob(os.path.join(LOG_DIR, 'crops', '*.jpg')))
        for old in crops[:-MAX_CROPS]:
            os.remove(old)

    def _write_stats(self):
        with open(os.path.join(LOG_DIR, 'stats.json'), 'w') as f:
            json.dump({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'stats': self.stats, 'bank': self.bank_n}, f)


def main():
    rclpy.init()
    node = DetectorB()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
