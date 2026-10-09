#!/usr/bin/env python3
"""
Health monitor: one CSV line per minute describing the whole stack, so an overnight stall can be read in the morning.

    source scripts/ros_env.sh && nohup python3 scripts/health_monitor.py > /dev/null 2>&1 &
    column -s, -t ~/.ros/health/health.csv | less -S

Columns: time, load1, mem_avail_gb, swap_used_gb, disk_root_free_gb, disk_ws_free_gb, arm_status, fault, joint_state_publishers,
joint_states_hz (3 s sample), rs1_hz, rs2_hz, wrist_hz, cam_status, scene_objects (fresh), unknown_obstacles (confirmed),
ollama_loaded, flags (camera recalibration flag files present).
It only observes (subscribes, reads files); it commands nothing and costs < 1% CPU (joint_states is sampled for 3 s a minute,
not continuously, because it publishes at ~900 Hz).
"""
import json
import os
import shutil
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from example_interfaces.msg import Bool
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

OUT_DIR = os.path.expanduser('~/.ros/health')
PERIOD_S = 60.0
CSV_HEADER = ('time,load1,mem_avail_gb,swap_used_gb,disk_root_free_gb,disk_ws_free_gb,arm_status,fault,joint_state_publishers,'
              'joint_states_hz,rs1_hz,rs2_hz,wrist_hz,cam_status,scene_objects,unknown_confirmed,ollama_loaded,flags')


class Monitor(Node):
    def __init__(self):
        super().__init__('health_monitor')
        self.count = {'rs1': 0, 'rs2': 0, 'wrist': 0, 'js': 0}
        self.arm = self.cam = self.scene = self.unknown = self.fault = None
        self.create_subscription(Image, '/global_camera/global_camera/color/image_raw', lambda m: self._inc('rs1'), 1)
        self.create_subscription(Image, '/global_camera_2/global_camera_2/color/image_raw', lambda m: self._inc('rs2'), 1)
        self.create_subscription(Image, '/camera/color/image_raw', lambda m: self._inc('wrist'), 1)
        self.create_subscription(String, '/arm_status', lambda m: setattr(self, 'arm', m.data), 5)
        self.create_subscription(String, '/camera_calibration_status', lambda m: setattr(self, 'cam', m.data), 5)
        self.create_subscription(String, '/scene_snapshot', lambda m: setattr(self, 'scene', m.data), 2)
        self.create_subscription(String, '/unknown_obstacles', lambda m: setattr(self, 'unknown', m.data), 2)
        self.create_subscription(Bool, '/fault_controller/internal_fault', lambda m: setattr(self, 'fault', bool(m.data)), 1)
        self._js_sub = None

    def _inc(self, k):
        self.count[k] += 1

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def sample_joint_states(self, seconds=3.0):
        self.count['js'] = 0
        sub = self.create_subscription(JointState, '/joint_states', lambda m: self._inc('js'), 10)
        self.spin_for(seconds)
        self.destroy_subscription(sub)
        return self.count['js'] / seconds


def ollama_loaded():
    try:
        out = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return out[1].split()[0] if len(out) > 1 else 'none'
    except Exception:
        return '?'


def gb(path):
    try:
        return round(shutil.disk_usage(path).free / 1e9, 1)
    except OSError:
        return -1


def meminfo():
    d = {}
    with open('/proc/meminfo') as f:
        for ln in f:
            k, v = ln.split(':')
            d[k] = int(v.split()[0])
    return round(d['MemAvailable'] / 1e6, 1), round((d['SwapTotal'] - d['SwapFree']) / 1e6, 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, 'health.csv')
    new = not os.path.exists(path)
    rclpy.init()
    n = Monitor()
    n.spin_for(5.0)
    with open(path, 'a') as f:
        if new:
            f.write(CSV_HEADER + '\n')
        while True:
            n.count.update({'rs1': 0, 'rs2': 0, 'wrist': 0})
            t0 = time.time()
            js_hz = n.sample_joint_states(3.0)
            n.spin_for(max(0.0, 7.0 - (time.time() - t0)))
            dt = time.time() - t0
            try:
                pubs = n.count_publishers('/joint_states')
            except Exception:
                pubs = -1
            try:
                scene = json.loads(n.scene or '{}')
                fresh = sum(1 for o in scene.values() if isinstance(o, dict) and not o.get('stale'))
            except json.JSONDecodeError:
                fresh = -1
            try:
                unk = sum(1 for o in json.loads(n.unknown or '{}').get('obstacles', []) if o.get('confirmed'))
            except json.JSONDecodeError:
                unk = -1
            try:
                cam = json.loads(n.cam or '{}')
                cam = '/'.join(f'{k[:3]}:{v[:3]}' for k, v in cam.items())
            except json.JSONDecodeError:
                cam = '?'
            flags = ''.join(c for c, name in (('1', 'realsense'), ('2', 'realsense2')) if os.path.exists(os.path.expanduser(f'~/.ros/{name}_needs_recalibration.flag')))
            mem, swap = meminfo()
            row = [time.strftime('%Y-%m-%d %H:%M:%S'), round(os.getloadavg()[0], 1), mem, swap, gb('/'), gb('/mnt/ros_workspace'),
                   (n.arm or '?').replace(',', ';'), '?' if n.fault is None else ('FAULT' if n.fault else 'ok'), pubs, round(js_hz), round(n.count['rs1'] / max(dt, 1e-6), 1),
                   round(n.count['rs2'] / max(dt, 1e-6), 1), round(n.count['wrist'] / max(dt, 1e-6), 1), cam, fresh, unk,
                   ollama_loaded(), flags or '-']
            f.write(','.join(str(v) for v in row) + '\n')
            f.flush()
            n.spin_for(max(0.0, PERIOD_S - (time.time() - t0)))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
